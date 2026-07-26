"""P8 behavioural distillation (FROZEN design §3).

Because the long-term path is ZERO-initialised, P8 at init == healthy teacher EXACTLY
(verified p8_foundation_summary.json). This script (a) generates a FIXED teacher trajectory
dataset under the frozen config (seed=42), then (b) runs a BOUNDED distillation of ONLY the
long-memory params so the summary readout becomes informative while logits/value stay close
to the teacher (inherited GTrXL/encoder/head weights are FROZEN). Produces one artifact:
distill/P8_DISTILLED_INIT/ + summary JSON (held-out KL / value error / param count).
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-8df11537-ab79-722d-606f-411966196c4c"  # GPU2 only
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
os.environ["WANDB_MODE"] = "disabled"
import sys, json, pickle, hashlib
import numpy as np, jax, jax.numpy as jnp, optax
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
OUT = "/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu2_p8_longmemory/distill"
E, WM, NL, NH = 16, 128, 2, 8
SEED, T_ROLL, T_WARM = 42, 1280, 256   # 1280 traj steps; first 256 = warmup (not in loss)

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
ts=load_weights_only(CKPT, base, epc, cfg, load_opt_state=False)
teacher=ts.params
tnet=ActorCriticTransformer(action_dim=43, activation="relu", hidden_layers=256, encoder_size=256,
    num_heads=NH, qkv_features=256, num_layers=NL, gating=True, gating_bias=2.0)

def sha(p):
    h=hashlib.sha256()
    for v in jax.tree_util.tree_leaves(p): h.update(np.ascontiguousarray(np.asarray(v)).tobytes())
    return h.hexdigest()
teacher_sha=sha(teacher)
print(f"[distill] teacher params sha={teacher_sha}")

# ---- generate FIXED teacher trajectory (obs, true_done, teacher logits/value) ----
rng=jax.random.PRNGKey(SEED)
rng,_rng=jax.random.split(rng)
obs,estate=env.reset(_rng, env_params)
mem=jnp.zeros((E,WM,NL,256)); mask=jnp.zeros((E,NH,1,WM+1),jnp.bool_); midx=jnp.zeros((E,),jnp.int32)+(WM+1)
done=jnp.zeros((E,),jnp.bool_)
rec_obs=[]; rec_logits=[]; rec_value=[]; rec_reset=[]   # reset = true_done for NEXT step
for t in range(T_ROLL):
    midx=jnp.where(done, WM, jnp.clip(midx-1,0,WM))
    mask=jnp.where(done[:,None,None,None], jnp.zeros((E,NH,1,WM+1),jnp.bool_), mask)
    oh=jax.nn.one_hot(midx, WM+1)[:,None,None,:].repeat(NH,1)
    mask=jnp.logical_or(mask, oh)
    rng,_rng=jax.random.split(rng)
    pi,val,mout=tnet.apply(teacher, mem, obs, mask, method=tnet.model_forward_eval)
    action=pi.sample(seed=_rng)
    rec_obs.append(np.asarray(obs)); rec_logits.append(np.asarray(pi.logits)); rec_value.append(np.asarray(val))
    rec_reset.append(np.asarray(done))   # the reset flag entering THIS step
    mem=jnp.roll(mem,-1,axis=1).at[:,-1].set(mout)
    rng,_rng=jax.random.split(rng)
    obs,estate,rew,done,info=env.step(_rng, estate, action, env_params)
rec_obs=np.stack(rec_obs,1); rec_logits=np.stack(rec_logits,1); rec_value=np.stack(rec_value,1); rec_reset=np.stack(rec_reset,1)
# dataset sha (obs only, deterministic)
dsha=hashlib.sha256(np.ascontiguousarray(rec_obs).tobytes()).hexdigest()
np.savez_compressed(f"{OUT}/teacher_traj.npz", obs=rec_obs, logits=rec_logits, value=rec_value,
                    reset=rec_reset)
print(f"[distill] teacher trajectory: obs{rec_obs.shape} logits{rec_logits.shape} value{rec_value.shape} traj_sha={dsha}")

# ---- P8 with inherited weights; train ONLY long-mem params ----
p8=ActorCriticLongMem(action_dim=43, activation="relu", hidden_layers=256, encoder_size=256,
    num_heads=NH, qkv_features=256, num_layers=NL, gating=True, gating_bias=2.0, use_longmem=True)
ls0=init_longstate(E)
p8p=p8.init(jax.random.PRNGKey(0), mem, jnp.zeros((E,8335)), mask, ls0, jnp.zeros((E,),jnp.bool_), method=p8.forward_eval)
def flat(d):
    o={}
    def rec(x,pre):
        if isinstance(x,dict):
            for k,v in x.items(): rec(v,pre+(str(k),))
        else: o[pre]=x
    rec(d,()); return o
def unflat(fd,tmpl):
    def rec(t,pre):
        if isinstance(t,dict): return {k:rec(v,pre+(str(k),)) for k,v in t.items()}
        return fd[pre]
    return rec(tmpl,())
# overlay teacher inherited leaves
tf=flat(teacher); sf=flat(p8p); newsf=dict(sf)
for path,val in tf.items():
    if path in sf and np.asarray(sf[path]).shape==np.asarray(val).shape: newsf[path]=val
p8i=unflat(newsf,p8p)
# long-mem leaf mask (fresh paths)
fresh_paths=[p for p in sf if p not in tf]
LONGMEM_KEYS=set(id(v) for p in fresh_paths for v in [sf[p]])  # placeholder

# partition params: train only the fresh (long-memory) leaves (string-path keys, consistent with tf)
import jax.tree_util as jtu
n_train = sum(1 for p in flat(p8i) if p not in tf)
n_freeze = sum(1 for p in flat(p8i) if p in tf)
print(f"[distill] training {n_train} long-mem leaves; freezing {n_freeze} inherited")

LR_D=2e-4   # distillation init fit (NOT RL); inherited params frozen so safe
opt=optax.adam(LR_D, eps=1e-5)
opt_state=opt.init(p8i)

@jax.jit
def distill_step(params, opt_state, obs_t, reset_t, mem_c, mask_c, ls_c):
    # one timestep forward of P8 (teacher-forced obs); loss vs teacher logits/value computed in caller
    def loss_fn(params):
        pi, val, mout, ls_new = p8.apply(params, mem_c, obs_t, mask_c, ls_c, reset_t, method=p8.forward_eval)
        return pi, val, mout, ls_new
    return loss_fn(params)

# We do a scan over the recorded trajectory, accumulating KL+MSE, with the long-mem params
# differentiated and inherited frozen via a masked gradient.
def rollout_loss(longmem_delta, base_params, obs_seq, logits_seq, value_seq, reset_seq):
    # base_params already has inherited+current longmem; we add delta to longmem leaves only
    pass

# Truncated BPTT distillation: data is processed in SEG-step segments; gradients flow WITHIN
# each segment but the carry (short-term memory + long-term state + mask + midx) is
# stop_gradient-ed across segment boundaries. The long-term state still PROPAGATES forward
# (so summaries committed early influence later readouts) but we do not backprop through the
# full 1280-step unroll -> memory is bounded to one SEG-step transformer window. This matches
# the scale the real trainer differentiates through (num_steps=128). Inherited params are
# frozen afterwards by masking their gradients to zero.
SEG=128
N_SEG=T_ROLL//SEG
@jax.jit
def update(params, opt_state, obs_seq, logits_seq, value_seq, reset_seq):
    def loss_fn(params):
        # (E,T,...) -> (T,E,...) -> (N_SEG, SEG, E, ...)
        def to_seg(t):
            s=jnp.swapaxes(t,0,1)                       # (T, E, ...)
            return s.reshape((N_SEG, SEG)+s.shape[1:])  # (N_SEG, SEG, E, ...)
        obs_s=to_seg(obs_seq); logits_s=to_seg(logits_seq); value_s=to_seg(value_seq); reset_s=to_seg(reset_seq)
        def scan_step(carry, x):
            mem_c, mask_c, ls_c, midx_c = carry
            obs_t, logits_t, value_t, reset_t = x
            midx_c=jnp.where(reset_t, WM, jnp.clip(midx_c-1,0,WM))
            mask_c=jnp.where(reset_t[:,None,None,None], jnp.zeros((E,NH,1,WM+1),jnp.bool_), mask_c)
            oh=jax.nn.one_hot(midx_c, WM+1)[:,None,None,:].repeat(NH,1)
            mask_c=jnp.logical_or(mask_c, oh)
            pi, val, mout, ls_new = p8.apply(params, mem_c, obs_t, mask_c, ls_c, reset_t, True, method=p8.forward_eval)
            mem_c=jnp.roll(mem_c,-1,axis=1).at[:,-1].set(mout)
            logp=jax.nn.log_softmax(pi.logits, axis=-1)
            tprob=jax.nn.softmax(logits_t, axis=-1)
            kl=jnp.sum(tprob*(jnp.log(tprob+1e-12)-logp), axis=-1).mean()
            vmse=jnp.square(val-value_t).mean()
            return (mem_c, mask_c, ls_new, midx_c), (kl, vmse)
        def seg_step(carry, seg_xs):
            # truncate BPTT across segment boundary (carry forward without gradient)
            carry=jax.tree_util.tree_map(jax.lax.stop_gradient, carry)
            carry, (kls, vmses)=jax.lax.scan(scan_step, carry, seg_xs)
            return carry, (kls, vmses)
        init_carry=(jnp.zeros((E,WM,NL,256)), jnp.zeros((E,NH,1,WM+1),jnp.bool_), init_longstate(E),
                    jnp.zeros((E,),jnp.int32)+(WM+1))
        _, (kls, vmses)=jax.lax.scan(seg_step, init_carry, (obs_s, logits_s, value_s, reset_s))
        # kls/vmses shape (N_SEG, SEG); flatten to time order, drop warmup steps
        kls=kls.reshape(-1)[T_WARM:]; vmses=vmses.reshape(-1)[T_WARM:]
        return kls.mean()+0.5*vmses.mean(), (kls.mean(), vmses.mean())
    (loss,(kl,vmse)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    # mask gradients: zero out inherited leaves (consistent string-path keys)
    gd = flat(grads)
    gd = {p: (g * 0.0 if p in tf else g) for p, g in gd.items()}
    grads = unflat(gd, grads)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, kl, vmse

obs_seq=jnp.asarray(rec_obs, jnp.float32); logits_seq=jnp.asarray(rec_logits, jnp.float32)
value_seq=jnp.asarray(rec_value, jnp.float32); reset_seq=jnp.asarray(rec_reset, jnp.bool_)
# held-out split: use a shifted window as held-out (different seed trajectory)
rng2=jax.random.PRNGKey(SEED+1)
rng2,_r=jax.random.split(rng2); obs2,es2=env.reset(_r, env_params)
mem2=jnp.zeros((E,WM,NL,256)); mask2=jnp.zeros((E,NH,1,WM+1),jnp.bool_); midx2=jnp.zeros((E,),jnp.int32)+(WM+1); done2=jnp.zeros((E,),jnp.bool_)
ho_obs=[];ho_log=[];ho_val=[];ho_reset=[]
for t in range(T_ROLL):
    midx2=jnp.where(done2,WM,jnp.clip(midx2-1,0,WM))
    mask2=jnp.where(done2[:,None,None,None],jnp.zeros((E,NH,1,WM+1),jnp.bool_),mask2)
    oh=jax.nn.one_hot(midx2,WM+1)[:,None,None,:].repeat(NH,1); mask2=jnp.logical_or(mask2,oh)
    rng2,_r=jax.random.split(rng2)
    pi,val,mout=tnet.apply(teacher, mem2, obs2, mask2, method=tnet.model_forward_eval)
    a=pi.sample(seed=_r); ho_obs.append(np.asarray(obs2));ho_log.append(np.asarray(pi.logits));ho_val.append(np.asarray(val));ho_reset.append(np.asarray(done2))
    mem2=jnp.roll(mem2,-1,axis=1).at[:,-1].set(mout)
    rng2,_r=jax.random.split(rng2); obs2,es2,_,done2,_=env.step(_r,es2,a,env_params)
ho_obs=jnp.asarray(np.stack(ho_obs,1),jnp.float32);ho_log=jnp.asarray(np.stack(ho_log,1),jnp.float32)
ho_val=jnp.asarray(np.stack(ho_val,1),jnp.float32);ho_reset=jnp.asarray(np.stack(ho_reset,1),jnp.bool_)

EPOCHS=12
params=p8i
for ep in range(EPOCHS):
    params, opt_state, loss, kl, vmse = update(params, opt_state, obs_seq, logits_seq, value_seq, reset_seq)
    print(f"[distill] epoch {ep:02d} loss={float(loss):.6f} trainKL={float(kl):.6f} trainVMSE={float(vmse):.6f}")

# held-out eval (no grad)
@jax.jit
def eval_heldout(params, obs_seq, logits_seq, value_seq, reset_seq):
    def scan_step(carry, x):
        mem_c, mask_c, ls_c, midx_c = carry
        obs_t, logits_t, value_t, reset_t = x
        midx_c=jnp.where(reset_t, WM, jnp.clip(midx_c-1,0,WM))
        mask_c=jnp.where(reset_t[:,None,None,None], jnp.zeros((E,NH,1,WM+1),jnp.bool_), mask_c)
        oh=jax.nn.one_hot(midx_c, WM+1)[:,None,None,:].repeat(NH,1); mask_c=jnp.logical_or(mask_c, oh)
        pi, val, mout, ls_new = p8.apply(params, mem_c, obs_t, mask_c, ls_c, reset_t, method=p8.forward_eval)
        mem_c=jnp.roll(mem_c,-1,axis=1).at[:,-1].set(mout)
        logp=jax.nn.log_softmax(pi.logits, axis=-1); tprob=jax.nn.softmax(logits_t, axis=-1)
        kl=jnp.sum(tprob*(jnp.log(tprob+1e-12)-logp), axis=-1).mean()
        vmse=jnp.square(val-value_t).mean()
        return (mem_c, mask_c, ls_new, midx_c), (kl, vmse)
    init_carry=(jnp.zeros((E,WM,NL,256)), jnp.zeros((E,NH,1,WM+1),jnp.bool_), init_longstate(E), jnp.zeros((E,),jnp.int32)+(WM+1))
    _, (kls, vmses)=jax.lax.scan(scan_step, init_carry,
        (jnp.swapaxes(obs_seq,0,1), jnp.swapaxes(logits_seq,0,1),
         jnp.swapaxes(value_seq,0,1), jnp.swapaxes(reset_seq,0,1)))
    kls=kls[T_WARM:]; vmses=vmses[T_WARM:]
    return kls.mean(), vmses.mean()
ho_kl, ho_vmse = eval_heldout(params, ho_obs, ho_log, ho_val, ho_reset)
print(f"[distill] HELD-OUT KL={float(ho_kl):.6f} VMSE={float(ho_vmse):.6f}")

# verify inherited leaves unchanged after distillation (frozen)
inh_ok=all(np.array_equal(np.asarray(flat(params)[p]), np.asarray(tf[p])) for p in tf if p in flat(params) and np.asarray(flat(params)[p]).shape==np.asarray(tf[p]).shape)
print(f"[distill] inherited leaves unchanged after distill: {inh_ok}")

# persist artifact
os.makedirs(f"{OUT}/P8_DISTILLED_INIT", exist_ok=True)
with open(f"{OUT}/P8_DISTILLED_INIT/params.pkl","wb") as f:
    pickle.dump(jax.tree_util.tree_map(np.asarray, params), f)
# param count
n_params=sum(np.asarray(v).size for v in jax.tree_util.tree_leaves(params))
n_longmem=sum(np.asarray(flat(params)[p]).size for p in flat(params) if p not in tf)
summary={"teacher_sha":teacher_sha,"traj_sha":dsha,"seed":SEED,"T_ROLL":T_ROLL,"T_WARM":T_WARM,
         "distill_lr":LR_D,"epochs":EPOCHS,"held_out_kl":float(ho_kl),"held_out_vmse":float(ho_vmse),
         "final_train_kl":float(kl),"final_train_vmse":float(vmse),"inherited_unchanged":bool(inh_ok),
         "n_params_total":int(n_params),"n_params_longmem":int(n_longmem),
         "K":64,"N":16,"coverage_steps":1024}
json.dump(summary, open(f"{OUT}/P8_DISTILLED_INIT/summary.json","w"), indent=2)
print("[distill] P8_DISTILLED_INIT_WRITTEN", json.dumps(summary))
