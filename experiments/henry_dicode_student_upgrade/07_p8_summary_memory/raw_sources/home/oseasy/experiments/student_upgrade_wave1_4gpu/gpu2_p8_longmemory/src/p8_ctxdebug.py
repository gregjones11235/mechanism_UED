"""Debug: localize the long-term context mismatch between rollout (forward_eval) and the
train recompute scan. Compares (a) final long-state and (b) per-step context vector."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-8df11537-ab79-722d-606f-411966196c4c"
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
os.environ["WANDB_MODE"] = "disabled"
import sys, pickle
import numpy as np, jax, jax.numpy as jnp
P8 = "/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu2_p8_longmemory/src"
V7 = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB"
sys.path.insert(0, P8); sys.path.insert(0, V7 + "/src"); sys.path.insert(0, V7)
from craftax.craftax.constants import Achievement
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from dicode.task_utils import get_achievement_multi_hot
from dicode.wrappers_cl import DistributedMultiTaskOptimisticLogWrapper
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv
from p8_network import ActorCriticLongMem, init_longstate

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
p8=ActorCriticLongMem(action_dim=43, activation="relu", hidden_layers=256, encoder_size=256,
    num_heads=NH, qkv_features=256, num_layers=NL, gating=True, gating_bias=2.0, use_longmem=True)
DISTILL = "/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu2_p8_longmemory/distill/P8_DISTILLED_INIT/params.pkl"
P=jax.tree_util.tree_map(jnp.asarray, pickle.load(open(DISTILL,"rb")))

# helper applies for encoder + long_context
def enc(o): return p8.apply(P, o, method=lambda m, o: m.transformer.encoder(o))
def ctx_of(ls, q): return p8.apply(P, ls, q, method=lambda m, ls, q: m._long_context(ls, q))

# ---- rollout: capture per-step context + final ls ----
rng=jax.random.PRNGKey(42); rng,_rng=jax.random.split(rng)
obs,estate=env.reset(_rng, env_params)
mem=jnp.zeros((E,WM,NL,256)); mask=jnp.zeros((E,NH,1,WM+1),jnp.bool_); midx=jnp.zeros((E,),jnp.int32)+(WM+1)
done=jnp.zeros((E,),jnp.bool_); true_done=jnp.zeros((E,),jnp.bool_); ls=init_longstate(E)
roll_ctx=[]; roll_obs=[]; roll_td=[]
for step in range(T):
    midx=jnp.where(done, WM, jnp.clip(midx-1,0,WM))
    mask=jnp.where(done[:,None,None,None], jnp.zeros((E,NH,1,WM+1),jnp.bool_), mask)
    oh=jax.nn.one_hot(midx, WM+1)[:,None,None,:].repeat(NH,1); mask=jnp.logical_or(mask, oh)
    rng,_rng=jax.random.split(rng)
    pi,val,mout,ls=p8.apply(P, mem, obs, mask, ls, true_done, method=p8.forward_eval)
    # the context the actor USED at this step = _long_context(ls_new, encoded)
    e=enc(obs); c=ctx_of(ls, e)
    roll_ctx.append(np.asarray(c)); roll_obs.append(np.asarray(obs)); roll_td.append(np.asarray(true_done))
    mem=jnp.roll(mem,-1,axis=1).at[:,-1].set(mout)
    rng,_rng=jax.random.split(rng)
    obs,estate,rew,done,info=env.step(_rng, estate, pi.sample(seed=_rng), env_params)
    true_done=info["returned_episode"]
roll_ctx=np.stack(roll_ctx,0)   # (T, E, 256)
roll_obs=np.stack(roll_obs,0)   # (T, E, 8335)
roll_td=np.stack(roll_td,0)     # (T, E)  -- true_done ENTERING each step
ls_roll_final=jax.tree_util.tree_map(np.asarray, ls)
print(f"[dbg] rollout done; final summaries norm={np.linalg.norm(ls_roll_final['summaries']):.4f} valid_sum={int(ls_roll_final['valid'].sum())}")

# ---- recompute scan (same as model_forward_train_longmem) ----
obs_ET=jnp.swapaxes(jnp.asarray(roll_obs,jnp.float32),0,1)       # (E, T, 8335)
encoded_ET=p8.apply(P, obs_ET, method=lambda m, o: m.transformer.encoder(o))  # (E, T, 256)
encoded_T=jnp.swapaxes(encoded_ET,0,1)                           # (T, E, 256)
td_T=jnp.asarray(roll_td,jnp.bool_)                              # already (T, E)
def ctx_step(ls, inp):
    enc_t, td_t = inp
    ls = p8.apply(P, ls, enc_t, td_t, method=lambda m, ls, e, r: m._long_update(ls, e, r))
    ctx = p8.apply(P, ls, enc_t, method=lambda m, ls, q: m._long_context(ls, q))
    return ls, ctx
ls0=init_longstate(E)
ls_scan_final, ctx_T = jax.lax.scan(ctx_step, ls0, (encoded_T, td_T))   # ctx_T (T, E, 256)
ctx_scan=np.asarray(ctx_T)
ls_scan_final=jax.tree_util.tree_map(np.asarray, ls_scan_final)

# ---- compare per-step context ----
diff=np.abs(ctx_scan - roll_ctx).max(axis=(1,2))   # (T,)
first_bad=int(np.argmax(diff>1e-5)) if (diff>1e-5).any() else -1
print(f"[dbg] per-step context max|diff| overall={float(diff.max()):.3e}  first_bad_step={first_bad}")
print(f"[dbg] first 8 per-step diffs: {np.array2string(diff[:8], precision=4)}")
print(f"[dbg] diffs around step 60-68: {np.array2string(diff[60:68], precision=4)}")
# compare final ls
for k in ls_roll_final:
    d=float(np.abs(ls_scan_final[k]-ls_roll_final[k]).max())
    print(f"[dbg] final ls[{k}] maxdiff={d:.3e}")
# compare encoded
enc_roll=np.stack([np.asarray(enc(jnp.asarray(roll_obs[t],jnp.float32))) for t in range(T)],0)
enc_diff=float(np.abs(np.asarray(encoded_T)-enc_roll).max())
print(f"[dbg] encoded (scan vs per-step rollout) maxdiff={enc_diff:.3e}")
# td match
print(f"[dbg] td_T sum={int(np.asarray(td_T).sum())} roll_td sum={int(np.asarray(roll_td).sum())}")
