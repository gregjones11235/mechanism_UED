import os
os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-8df11537-ab79-722d-606f-411966196c4c"  # GPU2 only
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
os.environ["WANDB_MODE"] = "disabled"
import sys, json
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
from p8_network import ActorCriticLongMem, init_longstate, K, N, DIM

CKPT = ("/home/oseasy/incoming/henry_work_20260721T105300/extracted/"
        "Henry_work/base_ckpt_v7fix55_armA_s0/rl_checkpoints/17500")
E, WM, NL, NH = 16, 128, 2, 8

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
ts=load_weights_only(CKPT, base, epc, cfg, load_opt_state=False)
teacher=ts.params
T=teacher["params"] if "params" in teacher else teacher
obs_dim, action_dim = 8335, 43

# ---- nets ----
tnet=ActorCriticTransformer(action_dim=action_dim, activation="relu", hidden_layers=256,
    encoder_size=256, num_heads=NH, qkv_features=256, num_layers=NL, gating=True, gating_bias=2.0)
p8=ActorCriticLongMem(action_dim=action_dim, activation="relu", hidden_layers=256, encoder_size=256,
    num_heads=NH, qkv_features=256, num_layers=NL, gating=True, gating_bias=2.0, use_longmem=True)
mem=jnp.zeros((E, WM, NL, 256)); mask=jnp.zeros((E, NH, 1, WM+1), jnp.bool_)
ls0=init_longstate(E); reset0=jnp.zeros((E,), jnp.bool_)
obs0=jnp.zeros((E, obs_dim))
p8p=p8.init(jax.random.PRNGKey(0), mem, obs0, mask, ls0, reset0, method=p8.forward_eval)

# ---- overlay inherited teacher leaves (exact path match) ----
def flat(d):
    out={}
    def rec(x,pre):
        if isinstance(x,dict):
            for k,v in x.items(): rec(v,pre+(str(k),))
        else: out[pre]=x
    rec(d,()); return out
tf=flat(teacher); sf=flat(p8p)
inh=[]; newsf=dict(sf)
for path,val in tf.items():
    if path in sf and np.asarray(sf[path]).shape==np.asarray(val).shape:
        newsf[path]=val; inh.append(path)
def unflat(fd,tmpl):
    def rec(t,pre):
        if isinstance(t,dict): return {k:rec(v,pre+(str(k),)) for k,v in t.items()}
        return fd[pre]
    return rec(tmpl,())
p8i=unflat(newsf,p8p)
fresh=[p for p in sf if p not in tf]
print(f"[p8] inherited {len(inh)} leaves; fresh (long-mem) {len(fresh)}: {['/'.join(p) for p in fresh]}")

# ---- 1. bit-exact teacher equivalence (feature-off via zero-init) ----
rng=np.random.default_rng(42)
obs_r=jnp.asarray(rng.standard_normal((E,obs_dim)).astype("float32"))
pi_t,v_t,_=tnet.apply(teacher, mem, obs_r, mask, method=tnet.model_forward_eval)
pi_p,v_p,mo_p,ls1=p8.apply(p8i, mem, obs_r, mask, ls0, reset0, method=p8.forward_eval)
logit_diff=float(np.max(np.abs(np.asarray(pi_t.logits)-np.asarray(pi_p.logits))))
value_diff=float(np.max(np.abs(np.asarray(v_t)-np.asarray(v_p))))
print(f"[p8] teacher-equiv at init (zero-init long path): logit_maxdiff={logit_diff} value_maxdiff={value_diff}")
assert logit_diff==0.0 and value_diff==0.0, "P8 init != teacher (feature-off not exact)"

# ---- 2. long-term persistence across rollout boundary (no reset) ----
ls=ls0; commits=0
for step in range(2*K+5):                       # cross a 'rollout boundary' (we simply keep stepping)
    _,_,_,ls=p8.apply(p8i, mem, obs_r, mask, ls, jnp.zeros((E,),jnp.bool_), method=p8.forward_eval)
nvalid=int(np.asarray(ls["valid"]).sum(axis=1).mean())
acc=int(np.asarray(ls["accum_count"]).mean())
print(f"[p8] after {2*K+5} steps (no reset): mean valid summaries={nvalid} (expect ~{(2*K+5)//K}), accum_count={acc}")
assert nvalid>=2, "summaries not accumulating/persisting"
assert nvalid<=N, "ring overflow"

# ---- 3. true_done reset clears long-term state & reverts logits to teacher ----
pi_r,v_r,_,ls_reset=p8.apply(p8i, mem, obs_r, mask, ls, jnp.ones((E,),jnp.bool_), method=p8.forward_eval)
valid_after_reset=int(np.asarray(ls_reset["valid"]).sum())
# after a reset, accum has only this step's obs -> no committed summary yet -> valid all False -> context 0 -> logits==teacher
print(f"[p8] after true_done reset: total valid entries={valid_after_reset} (expect 0)")
assert valid_after_reset==0, "true_done did not clear long-term state"
lr_diff=float(np.max(np.abs(np.asarray(pi_t.logits)-np.asarray(pi_r.logits))))
print(f"[p8] logits revert to teacher after reset: maxdiff={lr_diff}")
assert lr_diff==0.0, "post-reset logits != teacher"

# ---- 4. vector-env isolation (no crosstalk) ----
perm=jnp.asarray(rng.permutation(E))
lsA=init_longstate(E); lsB=init_longstate(E)
for _ in range(3):
    _,_,_,lsA=p8.apply(p8i, mem, obs_r, mask, lsA, reset0, method=p8.forward_eval)
    _,_,_,lsB=p8.apply(p8i, mem[perm], obs_r[perm], mask[perm], lsB, reset0[perm], method=p8.forward_eval)
iso=float(np.max(np.abs(np.asarray(lsA["summaries"])[np.asarray(perm)]-np.asarray(lsB["summaries"]))))
print(f"[p8] vector-env isolation (summaries) maxdiff={iso}")
assert iso<1e-6, "vector-env crosstalk"

# ---- 5. feature-off switch also exact ----
p8_off=ActorCriticLongMem(action_dim=action_dim, activation="relu", hidden_layers=256, encoder_size=256,
    num_heads=NH, qkv_features=256, num_layers=NL, gating=True, gating_bias=2.0, use_longmem=False)
offp=p8_off.init(jax.random.PRNGKey(0), mem, obs0, mask, ls0, reset0, method=p8_off.forward_eval)
offi=unflat({**flat(offp), **{k:v for k,v in flat(p8i).items() if k in flat(offp)}}, offp)
# use inherited teacher weights for the off model too
offi=unflat({**flat(offp), **{path:teacher_v for path,teacher_v in flat(teacher).items() if path in flat(offp)}}, offp)
pi_off,_,_,_=p8_off.apply(offi, mem, obs_r, mask, ls, jnp.zeros((E,),jnp.bool_), method=p8_off.forward_eval)
off_diff=float(np.max(np.abs(np.asarray(pi_t.logits)-np.asarray(pi_off.logits))))
print(f"[p8] use_longmem=False path == teacher: maxdiff={off_diff}")
assert off_diff==0.0, "feature-off switch not exact"

out={"inherited_leaves":len(inh),"fresh_leaves":len(fresh),"teacher_equiv_logit_diff":logit_diff,
     "teacher_equiv_value_diff":value_diff,"valid_summaries_after_run":nvalid,
     "valid_after_true_done_reset":valid_after_reset,"env_isolation":iso,"feature_off_diff":off_diff,
     "K":K,"N":N,"coverage_steps":K*N}
json.dump(out, open("/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu2_p8_longmemory/distill/p8_foundation_summary.json","w"), indent=2)
print("[p8] P8_FOUNDATION_PASS", json.dumps(out))
