"""判定：短期 x 的 train-vs-eval 差异是否为原设计固有。
用 teacher(ActorCriticTransformer) 跑 rollout(model_forward_eval) 与重算(model_forward_train)，
逐步对比 logits。若 teacher 也有 ~9e-2 差异 => 固有（P8 沿用同一 loss 机制即公平）。"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-8df11537-ab79-722d-606f-411966196c4c"
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
os.environ["WANDB_MODE"] = "disabled"
import sys
import numpy as np, jax, jax.numpy as jnp
V7 = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB"
sys.path.insert(0, V7 + "/src"); sys.path.insert(0, V7)
from craftax.craftax.constants import Achievement
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from dicode.network import ActorCriticTransformer
from dicode.task_utils import get_achievement_multi_hot
from dicode.utils.general.train_state_utils import load_weights_only
from dicode.wrappers_cl import DistributedMultiTaskOptimisticLogWrapper
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv

E, WM, WG, NL, NH, T = 16, 128, 64, 2, 8, 128
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
CKPT = ("/home/oseasy/incoming/henry_work_20260721T105300/extracted/"
        "Henry_work/base_ckpt_v7fix55_armA_s0/rl_checkpoints/17500")
tnet=ActorCriticTransformer(action_dim=43, activation="relu", hidden_layers=256, encoder_size=256,
    num_heads=NH, qkv_features=256, num_layers=NL, gating=True, gating_bias=2.0)
teacher=load_weights_only(CKPT, base, epc, cfg, load_opt_state=False).params

roll_vmap = jax.vmap(jnp.roll, in_axes=(-2, 0, None), out_axes=-2)
batch_indices_select = jax.vmap(lambda x, y: x[y])
batchify = lambda x: jnp.reshape(x, (x.shape[0]*x.shape[1],) + x.shape[2:])

# ---- rollout via model_forward_eval ----
rng=jax.random.PRNGKey(42); rng,_rng=jax.random.split(rng)
obs,estate=env.reset(_rng, env_params)
mem=jnp.zeros((E,WM,NL,256)); mask=jnp.zeros((E,NH,1,WM+1),jnp.bool_); midx=jnp.zeros((E,),jnp.int32)+(WM+1)
done=jnp.zeros((E,),jnp.bool_)
memories_previous=mem
rec_obs=[]; rec_logits=[]; rec_mmask=[]; rec_memout=[]
for step in range(T):
    midx=jnp.where(done, WM, jnp.clip(midx-1,0,WM))
    mask=jnp.where(done[:,None,None,None], jnp.zeros((E,NH,1,WM+1),jnp.bool_), mask)
    oh=jax.nn.one_hot(midx, WM+1)[:,None,None,:].repeat(NH,1); mask=jnp.logical_or(mask, oh)
    rng,_rng=jax.random.split(rng)
    pi,val,mout=tnet.apply(teacher, mem, obs, mask, method=tnet.model_forward_eval)
    a=pi.sample(seed=_rng)
    rec_obs.append(np.asarray(obs)); rec_logits.append(np.asarray(pi.logits)); rec_mmask.append(np.asarray(mask.squeeze()))
    mem=jnp.roll(mem,-1,axis=1).at[:,-1].set(mout); rec_memout.append(np.asarray(mout))
    rng,_rng=jax.random.split(rng)
    obs,estate,rew,done,info=env.step(_rng, estate, a, env_params)
roll_logits=np.stack(rec_logits,0)   # (T, E, 43)

# ---- memory machinery (EXACT control) ----
mem_cat=np.concatenate([np.swapaxes(np.asarray(memories_previous),0,1), np.stack(rec_memout,0)], axis=0)  # (256,E,NL,256)
midx_arr=np.stack([np.arange(0,WM)[None,:]+s*np.ones((E,1),dtype=np.int32) for s in range(T)],0)  # (T,E,128)
midx_ET=np.swapaxes(midx_arr,0,1)                                  # (E,T,128)
mem_cat_T=np.swapaxes(mem_cat,0,1)                                 # (E,256,NL,256)
mem_seg=batchify(batch_indices_select(mem_cat_T, midx_ET[:, ::WG]))# (B,128,NL,256)
obs_ET=np.swapaxes(np.stack(rec_obs,0),0,1)                        # (E,T,8335)
obs_seg=obs_ET.reshape((-1, WG)+obs_ET.shape[2:])                  # (B,64,8335)
mmask_ET=np.swapaxes(np.stack(rec_mmask,0),0,1)                    # (E,T,NH,129)
mmask_seg=mmask_ET.reshape((-1, WG)+mmask_ET.shape[2:])            # (B,64,NH,129)
mmask_seg=jnp.swapaxes(mmask_seg,1,2)                              # (B,NH,64,129)
mmask_seg=jnp.concatenate((mmask_seg, jnp.zeros(mmask_seg.shape[:-1]+(WG-1,),jnp.bool_)),axis=-1)  # (B,NH,64,192)
mmask_seg=roll_vmap(mmask_seg, jnp.arange(0,WG), -1)

# ---- recompute via model_forward_train ----
pi_re, val_re = tnet.apply(teacher, mem_seg, obs_seg, mmask_seg, method=tnet.model_forward_train)
logits_re_ET=np.asarray(pi_re.logits).reshape(E, T, 43)
logits_roll_ET=np.swapaxes(np.asarray(roll_logits,jnp.float32),0,1)
diff=np.abs(logits_re_ET - logits_roll_ET).max(axis=(0,2))   # (T,)
print(f"[teacher-tve] per-step logits max|train - eval| overall={float(diff.max()):.3e}")
print(f"[teacher-tve] per-step diff [0:8]={np.array2string(diff[:8], precision=5)}")
print(f"[teacher-tve] per-step diff [60:68]={np.array2string(diff[60:68], precision=5)}")
print(f"[teacher-tve] step0={float(diff[0]):.3e}  mean[1:]={float(diff[1:].mean()):.3e}")
