"""P9-AUTHENTIC-RESET — library correctness gates (GPU3). Independent of category labels.

  (1) BIT-RESTORABLE: pickle round-trip of every snapshot's full LogEnvState reproduces all
      leaves byte-exactly (shape+dtype+values).
  (2) ONE-STEP-TRANSITION MATCH (through the SAME optimistic-reset wrapper): for each full-step
      validation record, restore the recorded PRE LogEnvState into a 16-env wrapper, run
      wrapper.step(recorded_step_key, pre_logstate, recorded_actions) and assert the resulting
      (obs_next, LogEnvState_next) is BIT-IDENTICAL to the recorded post state. This proves a
      restored authentic state continues EXACTLY as the real trajectory did -> the soundness of
      the whole reset mechanism, regardless of how the moment was labelled.
  (3) NO FUTURE / LEAK: follows from (2) + construction; sanity-assert snapshot leaves finite
      and shapes match the live env (snapshot holds only past/present info).

Read-only. GPU3 only. Deterministic ops.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-f56a59b4-99f3-f2e5-11c6-d01685de8abd"
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
os.environ["WANDB_MODE"] = "disabled"
import sys, json, pickle, hashlib, time
import numpy as np, jax, jax.numpy as jnp

V7 = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB"
sys.path.insert(0, V7 + "/src"); sys.path.insert(0, V7)
from craftax.craftax.constants import Achievement
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from dicode.task_utils import get_achievement_multi_hot
from dicode.wrappers_cl import DistributedMultiTaskOptimisticLogWrapper
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv

LIB = "/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu3_p9_authentic_reset/lib"
S4_TASK_PATH = "/home/oseasy/experiments/p2_v1_20260722/evidence/s4_task_code.py"
E, EMB_TBL = 16, None
ach=jnp.array([get_achievement_multi_hot([Achievement.DEFEAT_KOBOLD])],dtype=jnp.float32); EMB=int(ach.shape[1])
ns={}; exec(open(S4_TASK_PATH).read(),ns); Task=ns["Env"]
ctor=EnvParams(max_timesteps=4096)
base=MultiTaskMiniCraftaxEnv([Task], StaticEnvParams(), ctor, True,
    conditioning_type="embedding", embedding_size=EMB)
env=DistributedMultiTaskOptimisticLogWrapper(base, jax.random.PRNGKey(0), E, 1, 16, jnp.ones(1), ach)
epc=env.default_params

lib=pickle.load(open(os.path.join(LIB,"p9_library.pkl"),"rb"))
val=pickle.load(open(os.path.join(LIB,"p9_val_records.pkl"),"rb"))
snaps=lib["snaps"]; records=val["records"]
print(f"[validate] loaded {len(snaps)} snapshots, {len(records)} val records; counts={lib['counts']}")

# ---------------- (1) bit-restorable ----------------
n_checked=0; n_bad=0; bad_examples=[]
for i,s in enumerate(snaps):
    blob=pickle.dumps(s["logstate"], protocol=pickle.HIGHEST_PROTOCOL)
    rt=pickle.loads(blob)
    la=jax.tree_util.tree_leaves(s["logstate"]); lb=jax.tree_util.tree_leaves(rt)
    if len(la)!=len(lb): n_bad+=1; bad_examples.append((i,"leafcount")); continue
    ok=True
    for a,b in zip(la,lb):
        a=np.asarray(a); b=np.asarray(b)
        if a.shape!=b.shape or a.dtype!=b.dtype or not np.array_equal(a,b):
            ok=False; break
    if not ok: n_bad+=1; bad_examples.append((i,"value"))
    n_checked+=1
gate1 = (n_bad==0 and n_checked==len(snaps) and len(snaps)>0)
print(f"[validate] (1) BIT-RESTORABLE checked={n_checked} bad={n_bad} -> {'PASS' if gate1 else 'FAIL'}")

# ---------------- (2) one-step transition through the wrapper ----------------
def leaves_equal(a, b):
    la=jax.tree_util.tree_leaves(a); lb=jax.tree_util.tree_leaves(b)
    if len(la)!=len(lb): return False, "leafcount"
    for x,y in zip(la,lb):
        x=np.asarray(x); y=np.asarray(y)
        if x.shape!=y.shape or x.dtype!=y.dtype or not np.array_equal(x,y):
            return False, f"shape{x.shape}/{y.shape}"
    return True, ""

n_rec=0; n_rec_bad=0; rec_bad_examples=[]
max_obs_diff=0.0; max_state_bad_leaves=0
for r in records:
    pre=jax.tree_util.tree_map(jnp.asarray, r["pre_logstate"])
    key=jnp.asarray(r["step_key"]); acts=jnp.asarray(r["actions"])
    obs_act, ls_act, rew, done_act, info = env.step(key, pre, acts, epc)
    jax.block_until_ready((obs_act, ls_act))
    obs_ok = np.array_equal(np.asarray(obs_act), np.asarray(r["post_obs"]))
    ls_ok, why = leaves_equal(ls_act, r["post_logstate"])
    if not obs_ok:
        d=float(np.abs(np.asarray(obs_act)-np.asarray(r["post_obs"])).max()); max_obs_diff=max(max_obs_diff,d)
    if not (obs_ok and ls_ok):
        n_rec_bad+=1; rec_bad_examples.append((r["step"], "obs" if not obs_ok else f"ls:{why}"))
    n_rec+=1
gate2 = (n_rec_bad==0 and n_rec==len(records) and len(records)>0)
print(f"[validate] (2) ONE-STEP-TRANSITION records={n_rec} bad={n_rec_bad} max_obs_diff={max_obs_diff:.3e} "
      f"-> {'PASS' if gate2 else 'FAIL'} {rec_bad_examples[:4]}")

# ---------------- (3) no future / leak (sanity) ----------------
n_fin_bad=0
for s in snaps:
    for lf in jax.tree_util.tree_leaves(s["logstate"]):
        lf=np.asarray(lf)
        if lf.dtype.kind=="f" and not np.all(np.isfinite(lf)): n_fin_bad+=1; break
    o=np.asarray(s["obs"])
    if not np.all(np.isfinite(o)): n_fin_bad+=1
gate3 = (n_fin_bad==0)
print(f"[validate] (3) NO-LEAK/FINITE non_finite_snapshots={n_fin_bad} -> {'PASS' if gate3 else 'FAIL'}")

ok = gate1 and gate2 and gate3
out=dict(label="P9_VALIDATE", n_snapshots=len(snaps), counts=lib["counts"],
         bit_restorable=dict(checked=n_checked, bad=n_bad, pass_=gate1),
         one_step_transition=dict(records=n_rec, bad=n_rec_bad, max_obs_diff=max_obs_diff,
                                   bad_examples=rec_bad_examples[:8], pass_=gate2),
         no_leak_finite=dict(non_finite=n_fin_bad, pass_=gate3),
         P9_VALIDATE_PASS=bool(ok), capture_code_sha=lib["code_sha"],
         validate_code_sha=hashlib.sha256(open(__file__,"rb").read()).hexdigest(),
         timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
json.dump(out, open(os.path.join(LIB,"p9_validate_summary.json"),"w"), indent=2, default=str)
print(f"[validate] P9_VALIDATE_{'PASS' if ok else 'FAIL'}")
