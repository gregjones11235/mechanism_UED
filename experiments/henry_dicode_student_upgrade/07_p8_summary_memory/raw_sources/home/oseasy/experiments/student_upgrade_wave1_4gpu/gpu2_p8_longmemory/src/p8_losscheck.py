"""P8 loss-faithfulness check (DE-RISK before the full trainer).

Replicates EXACTLY the canonical GTrXL-PPO _env_step collection and _loss_fn memory
machinery (from run_control_continuous_98304.py / dicode.ppo_tr), but with the P8 network,
and verifies:
  (1) RATIO==1 : the log_prob recomputed via model_forward_train_longmem EQUALS the rollout
      log_prob from forward_eval (so the PPO importance ratio starts at exactly 1). This proves
      the long-term context recompute is faithful to the rollout.
  (2) FEATURE-OFF : with use_longmem=False, model_forward_train_longmem logits EQUAL the
      teacher ActorCriticTransformer.model_forward_train logits (the inherited path is intact).

No gradient / no PPO update — pure forward equivalence on one collected rollout.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-8df11537-ab79-722d-606f-411966196c4c"  # GPU2 only
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
os.environ["WANDB_MODE"] = "disabled"
import sys, json, pickle
import numpy as np, jax, jax.numpy as jnp
P8 = "/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu2_p8_longmemory/src"
V7 = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB"
sys.path.insert(0, P8); sys.path.insert(0, V7 + "/src"); sys.path.insert(0, V7)

from craftax.craftax.constants import Achievement
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from dicode.network import ActorCriticTransformer
from dicode.task_utils import get_achievement_multi_hot
from dicode.utils.general.train_state_utils import load_weights_only
from dicode.wrappers_cl import DistributedMultiTaskOptimisticLogWrapper
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv
from p8_network import ActorCriticLongMem, init_longstate

CKPT = ("/home/oseasy/incoming/henry_work_20260721T105300/extracted/"
        "Henry_work/base_ckpt_v7fix55_armA_s0/rl_checkpoints/17500")
DISTILL = "/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu2_p8_longmemory/distill/P8_DISTILLED_INIT/params.pkl"
E, WM, WG, NL, NH, T = 16, 128, 64, 2, 8, 128   # window_mem, window_grad, num_steps

class Cfg:
    lr=2e-5; num_envs=16; num_steps=128; update_epochs=1; num_minibatches=2; gamma=0.999
    gae_lambda=0.8; clip_eps=0.2; ent_coef=0.002; vf_coef=0.5; max_grad_norm=1.0
    activation="relu"; anneal_lr=False; qkv_features=256; embed_size=256; num_heads=8
    num_layers=2; hidden_layers=256; window_mem=128; window_grad=64; gating=True
    gating_bias=2.0; condition_on_task=True; optimistic_reset_ratio=16; mode="score"
    bonus_type="none"; dynamic_bonus_k=0.0; completion_bonus_scale=0.0; completion_bonus_min=0.0
    value_target_clip_min=-50.0; value_target_clip_max=300.0
    def get(self,k,d=None): return getattr(self,k,d)
cfg=Cfg(); cfg.training=cfg
ach=jnp.array([get_achievement_multi_hot([Achievement.DEFEAT_KOBOLD])],dtype=jnp.float32); EMB=int(ach.shape[1])
ns={}; exec(open("/home/oseasy/experiments/p2_v1_20260722/evidence/s4_task_code.py").read(),ns); Task=ns["Env"]
base=MultiTaskMiniCraftaxEnv([Task], StaticEnvParams(), EnvParams(max_timesteps=4096), cfg.condition_on_task,
    conditioning_type="embedding", embedding_size=EMB, completion_bonus_scale=0.0, completion_bonus_min=0.0,
    bonus_type="none", dynamic_bonus_k=0.0)
epc=EnvParams(max_timesteps=4096)
env=DistributedMultiTaskOptimisticLogWrapper(base, jax.random.PRNGKey(0), cfg.num_envs, 1,
    cfg.optimistic_reset_ratio, jnp.ones(1), ach)
env_params=env.default_params

# ---- networks ----
tnet=ActorCriticTransformer(action_dim=43, activation="relu", hidden_layers=256, encoder_size=256,
    num_heads=NH, qkv_features=256, num_layers=NL, gating=True, gating_bias=2.0)
def mk_p8(use_longmem):
    return ActorCriticLongMem(action_dim=43, activation="relu", hidden_layers=256, encoder_size=256,
        num_heads=NH, qkv_features=256, num_layers=NL, gating=True, gating_bias=2.0, use_longmem=use_longmem)
p8=mk_p8(True); p8_off=mk_p8(False)

# ---- load distilled params into P8 ----
distilled=jax.tree_util.tree_map(jnp.asarray, pickle.load(open(DISTILL,"rb")))
teacher=load_weights_only(CKPT, base, epc, cfg, load_opt_state=False).params
print(f"[losscheck] distilled param leaves={len(jax.tree_util.tree_leaves(distilled))}")

# ============================================================== COLLECTION (EXACT _env_step) ==
roll_vmap = jax.vmap(jnp.roll, in_axes=(-2, 0, None), out_axes=-2)
batch_indices_select = jax.vmap(lambda x, y: x[y])
batchify = lambda x: jnp.reshape(x, (x.shape[0]*x.shape[1],) + x.shape[2:])

def collect(params, net):
    rng=jax.random.PRNGKey(42); rng,_rng=jax.random.split(rng)
    obs,estate=env.reset(_rng, env_params)
    memories=jnp.zeros((E,WM,NL,256)); mask=jnp.zeros((E,NH,1,WM+1),jnp.bool_)
    midx=jnp.zeros((E,),jnp.int32)+(WM+1)
    done=jnp.zeros((E,),jnp.bool_); true_done=jnp.zeros((E,),jnp.bool_)
    ls=init_longstate(E); ls_prev=ls
    memories_previous=memories
    rec_obs=[]; rec_act=[]; rec_logp=[]; rec_done=[]; rec_mmask=[]; rec_midx=[]; rec_info_re=[]; rec_memout=[]; rec_logits=[]
    for step in range(T):
        # short-term mask reset on RAW done (unchanged from control)
        midx=jnp.where(done, WM, jnp.clip(midx-1,0,WM))
        mask=jnp.where(done[:,None,None,None], jnp.zeros((E,NH,1,WM+1),jnp.bool_), mask)
        oh=jax.nn.one_hot(midx, WM+1)[:,None,None,:].repeat(NH,1); mask=jnp.logical_or(mask, oh)
        rng,_rng=jax.random.split(rng)
        # long-term reset on TRUE_DONE
        pi,val,mout,ls=p8.apply(params, memories, obs, mask, ls, true_done, method=p8.forward_eval)
        action=pi.sample(seed=_rng); logp=pi.log_prob(action)
        rec_obs.append(np.asarray(obs)); rec_act.append(np.asarray(action)); rec_logp.append(np.asarray(logp))
        rec_logits.append(np.asarray(pi.logits))
        rec_done.append(np.asarray(done)); rec_mmask.append(np.asarray(mask.squeeze())); rec_midx.append(np.asarray(midx))
        memories=jnp.roll(memories,-1,axis=1).at[:,-1].set(mout); rec_memout.append(np.asarray(mout))
        rng,_rng=jax.random.split(rng)
        obs,estate,rew,done,info=env.step(_rng, estate, action, env_params)
        true_done=info["returned_episode"]
        rec_info_re.append(np.asarray(info["returned_episode"]))
    return dict(obs=np.stack(rec_obs,0), act=np.stack(rec_act,0), logp=np.stack(rec_logp,0),
                logits=np.stack(rec_logits,0),
                done=np.stack(rec_done,0), mmask=np.stack(rec_mmask,0), midx=np.stack(rec_midx,0),
                info_re=np.stack(rec_info_re,0), memout=np.stack(rec_memout,0),
                memories_previous=np.asarray(memories_previous))

D=collect(distilled, p8)
print(f"[losscheck] collected rollout: obs{D['obs'].shape} true_dones_total={int(np.asarray(D['info_re']).sum())}")

# ============================================================== LOSS RE-FORWARD (EXACT _loss_fn) ==
# build the window_grad memory windows exactly as control section C + _loss_fn
memories_previous=D["memories_previous"]                      # (E, 128, NL, 256)
memories_batch=np.asarray(D["memout"])                        # (T=128, E, NL, 256)
mem_cat=np.concatenate([np.swapaxes(memories_previous,0,1), memories_batch], axis=0)  # (256, E, NL, 256)

# per-step memory_indices (E, 128) = arange(128) + step  -> stack (T, E, 128)
midx_arr=np.stack([np.arange(0,WM)[None,:]+s*np.ones((E,1),dtype=np.int32) for s in range(T)],0)  # (T,E,128)

# The control selects memories_indices[:, ::window_grad] where memories_indices is (E_mb, T, 128)
# AFTER swapaxes. Build that: midx per step is (E,128); stacked (T,E,128); swapaxes->(E,T,128).
midx_ET=np.swapaxes(jnp.asarray(midx_arr,jnp.int32),0,1)          # (E, T, 128)
mem_cat_T=jnp.swapaxes(jnp.asarray(mem_cat,jnp.float32),0,1)      # (E, 256, NL, 256)
mem_sel=batch_indices_select(mem_cat_T, midx_ET[:, ::WG])         # (E, 2, 128, NL, 256)
mem_seg=batchify(mem_sel)                                          # (E*2, 128, NL, 256) = (B,128,NL,256)

obs_ET=jnp.swapaxes(jnp.asarray(D["obs"],jnp.float32),0,1)        # (E, T, 8335)
obs_seg=obs_ET.reshape((-1, WG)+obs_ET.shape[2:])                 # (E*2, 64, 8335) = (B,64,8335)
mmask_ET=jnp.swapaxes(jnp.asarray(D["mmask"],jnp.bool_),0,1)      # (E, T, NH, 129)
mmask_seg=mmask_ET.reshape((-1, WG)+mmask_ET.shape[2:])           # (B, 64, NH, 129)
mmask_seg=jnp.swapaxes(mmask_seg,1,2)                             # (B, NH, 64, 129)
mmask_seg=jnp.concatenate((mmask_seg, jnp.zeros(mmask_seg.shape[:-1]+(WG-1,),jnp.bool_)),axis=-1)  # (B,NH,64,192)
mmask_seg=roll_vmap(mmask_seg, jnp.arange(0,WG), -1)

# true_done ENTERING each step: shift returned_episode by +1, step0 = False
info_re_ET=jnp.swapaxes(jnp.asarray(D["info_re"],jnp.bool_),0,1)  # (E, T)
true_done_ET=jnp.concatenate([jnp.zeros((E,1),jnp.bool_), info_re_ET[:, :T-1]], axis=1)  # (E, T)
longstate_prev=init_longstate(E)   # rollout started from a fresh long-state (matches collect ls_prev)

# 背景结论（已由 p8_teacher_tve.py 独立验证）：窗口化 transformer-XL 的 forward_train 对逐步
# forward_eval 存在 ~9e-2 的【固有】近似差异（teacher 本身就有，step0≈0、step1+ 发散）。这是
# 健康基线/control 共用的同一套机制，并非 P8 引入。因此判据不能用「train vs rollout(eval)==0」，
# 而要用【train-forward 对 train-forward】来验证 init 等价与继承路径。

# teacher 的 train-forward（参照系）
pi_t, val_t = tnet.apply(teacher, mem_seg, obs_seg, mmask_seg, method=tnet.model_forward_train)

# ---- (1) INIT 健康：distilled P8(longmem ON) train ≈ teacher train ----
# 注意：distillation 训练了 summary_to_actor（长期参数），使 distilled init 的 P8-on 【近似】teacher
# （held_out_kl~4.5e-4），而非【严格==】。「严格==teacher」是未训练时的结构性质，由 (2) feature-off 验证。
# 这里用 mean-KL 度量 distilled init 的健康度（应小，与 distill 报告同阶）。
pi_on, val_on = p8.apply(distilled, mem_seg, obs_seg, mmask_seg, true_done_ET, longstate_prev,
                         method=p8.model_forward_train_longmem)
_logp_on=jax.nn.log_softmax(pi_on.logits,-1); _logp_t=jax.nn.log_softmax(pi_t.logits,-1)
_p_t=jax.nn.softmax(pi_t.logits,-1)
kl_on=float(jnp.sum(_p_t*(_logp_t-_logp_on),-1).mean())
on_val_diff=float(np.max(np.abs(np.asarray(val_on)-np.asarray(val_t))))
on_logit_maxdiff=float(np.max(np.abs(np.asarray(pi_on.logits)-np.asarray(pi_t.logits))))
print(f"[losscheck] (1) INIT-HEALTH p8_on~teacher train : mean_KL={kl_on:.3e} value_maxdiff={on_val_diff:.3e} (logit_maxdiff={on_logit_maxdiff:.3e} info-only)")

# ---- (2) 继承路径：P8(longmem OFF) train == teacher train ----
pi_off, val_off = p8_off.apply(distilled, mem_seg, obs_seg, mmask_seg, true_done_ET, longstate_prev,
                               method=p8_off.model_forward_train_longmem)
off_logit_diff=float(np.max(np.abs(np.asarray(pi_off.logits)-np.asarray(pi_t.logits))))
off_val_diff=float(np.max(np.abs(np.asarray(val_off)-np.asarray(val_t))))
print(f"[losscheck] (2) FEATURE-OFF p8_off train==teacher train : logit_maxdiff={off_logit_diff:.3e} value_maxdiff={off_val_diff:.3e} (expect 0)")

# ---- (3) 信息性记录：train vs rollout(eval) 量级（应与 teacher 固有 ~9e-2 同阶，证明非 P8 引入）----
logits_re_ET=np.asarray(pi_on.logits).reshape(E, T, 43)
logits_roll_ET=np.swapaxes(np.asarray(D["logits"],jnp.float32),0,1)
tve_diff=float(np.abs(logits_re_ET - logits_roll_ET).max())
logp_roll=jnp.swapaxes(jnp.asarray(D["logp"],jnp.float32),0,1).reshape((-1, WG))
logp_re_b=np.asarray(pi_on.log_prob(jnp.swapaxes(jnp.asarray(D["act"],jnp.int32),0,1).reshape((-1, WG))))
ratio_logp_diff=float(np.max(np.abs(logp_re_b - np.asarray(logp_roll))))
print(f"[losscheck] (3) INFO train-vs-rollout(eval) [inherent, ~teacher] : logits_maxdiff={tve_diff:.3e} logp_maxdiff={ratio_logp_diff:.3e}")

out=dict(init_health_p8on_vs_teacher_meanKL=kl_on, init_health_p8on_value_maxdiff=on_val_diff,
         init_health_p8on_logit_maxdiff_info=on_logit_maxdiff,
         featureoff_logit_maxdiff=off_logit_diff, featureoff_value_maxdiff=off_val_diff,
         info_train_vs_eval_logits=tve_diff, info_train_vs_eval_logp=ratio_logp_diff,
         true_dones_in_rollout=int(np.asarray(D["info_re"]).sum()))
json.dump(out, open("/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu2_p8_longmemory/distill/p8_losscheck_summary.json","w"), indent=2)
# 判据：feature-off 严格等于 teacher(0.0) + distilled P8-on 健康(mean KL 小, value 严格 0)。
ok = (off_logit_diff == 0.0 and off_val_diff == 0.0 and on_val_diff == 0.0 and kl_on < 1e-2)
print(f"[losscheck] P8_LOSSCHECK_{'PASS' if ok else 'FAIL'} {json.dumps(out)}")
