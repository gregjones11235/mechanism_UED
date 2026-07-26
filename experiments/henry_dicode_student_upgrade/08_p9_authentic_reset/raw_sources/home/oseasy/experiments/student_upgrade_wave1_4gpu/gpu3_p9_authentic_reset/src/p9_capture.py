"""P9-AUTHENTIC-RESET — authentic state-library capture (GPU3, READ-ONLY harvesting).

Rolls out the HEALTHY student (ckpt17500, original ActorCriticTransformer) under the FROZEN
config (seed=42, S4_dark, num_envs=16, stochastic policy) using a collection loop that is
BYTE-IDENTICAL to the canonical control trainer's _env_step (same rng split order, same
short-term memory mask machinery, same optimistic-reset wrapper). At qualifying moments it
snapshots the FULL resumable reached-state and writes a READ-ONLY library.

A snapshot == the carry ENTERING _env_step at some real step t:
    (env_state_pre [full wrapper LogEnvState], obs_pre, memories_carry [pre-roll],
     memories_mask_carry [PRE top-reset], memories_mask_idx_carry [PRE top-reset],
     done_{t-1}, category)
Injecting this carry into the trainer reproduces the real trajectory bit-for-bit from there
(the trainer applies the same single top-reset + forward + env.step).

NO synthetic states, nothing hand-added. Five FROZEN categories. Selection uses only the
instant's sim state. NOTE: floor2_reached / gate_unlocked / mid_clear are TRANSITION-defined;
under S4_dark the floor-2 spawn is pre-cleared (ENTER_GNOMISH_MINES already true,
monsters_killed[2]=8), so these three are structurally (near-)empty and the library is carried
by the genuinely-occurring saw_stair_lost + near_floor3_failed critical-phase states — exactly
the floor2->3 phase the design wants the student to practice. near_floor3_failed additionally
keeps only candidates from episodes that ended WITHOUT ENTER_SEWERS (a post-hoc filter on WHICH
already-captured states survive; the stored snapshot content is purely past/present, so no
future information enters the snapshot).

Also writes a few FULL-STEP validation records (all 16 envs' pre-LogEnvState + the exact key
passed to env.step + actions + the recorded post (obs, LogEnvState)) so p9_validate.py can
prove the one-step-transition gate THROUGH THE SAME WRAPPER.

GPU3 only. Deterministic ops.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-f56a59b4-99f3-f2e5-11c6-d01685de8abd"  # GPU3 only
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
os.environ["WANDB_MODE"] = "disabled"
import sys, json, pickle, hashlib, time
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

OUT = "/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu3_p9_authentic_reset/lib"
TEACHER_CKPT = ("/home/oseasy/incoming/henry_work_20260721T105300/extracted/"
                "Henry_work/base_ckpt_v7fix55_armA_s0/rl_checkpoints/17500")
S4_TASK_PATH = "/home/oseasy/experiments/p2_v1_20260722/evidence/s4_task_code.py"

# ---- frozen config ----
E, WM, NL, NH = 16, 128, 2, 8
CAPTURE_STEPS = 8192
CAT_CAP = 32                      # per-category ring capacity (structural equal-weight, not tuned)
VIEW_RADIUS = 3                   # Chebyshev half-width ~= Craftax 7x7 local view
RECENT_WINDOW = 8                 # "saw stair recently" horizon (steps)
N_VAL_RECORDS = 8                 # full-step validation records to keep
SEED = 42
EGM = int(Achievement.ENTER_GNOMISH_MINES.value)     # 28
SEWERS = int(Achievement.ENTER_SEWERS.value)         # 30

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
ns={}; exec(open(S4_TASK_PATH).read(),ns); Task=ns["Env"]
base=MultiTaskMiniCraftaxEnv([Task], StaticEnvParams(), EnvParams(max_timesteps=4096), cfg.condition_on_task,
    conditioning_type="embedding", embedding_size=EMB, completion_bonus_scale=0.0, completion_bonus_min=0.0,
    bonus_type="none", dynamic_bonus_k=0.0)
epc=EnvParams(max_timesteps=4096)
env=DistributedMultiTaskOptimisticLogWrapper(base, jax.random.PRNGKey(0), cfg.num_envs, 1,
    cfg.optimistic_reset_ratio, jnp.ones(1), ach)
env_params=env.default_params
ACTION_DIM=int(base.action_space(epc).n)

net=ActorCriticTransformer(action_dim=ACTION_DIM, activation="relu", hidden_layers=256, encoder_size=256,
    num_heads=NH, qkv_features=256, num_layers=NL, gating=True, gating_bias=2.0)
teacher=load_weights_only(TEACHER_CKPT, base, epc, cfg, load_opt_state=False).params
print(f"[capture] devices={[str(d) for d in jax.devices()]} obs_dim={base.observation_space(epc).shape[0]} action_dim={ACTION_DIM}")

CATS = ["floor2_reached","mid_clear","gate_unlocked","saw_stair_lost","near_floor3_failed"]
rings = {c: [] for c in CATS}
val_records = []

def snap_slice(logstate, obs, memories, mask, midx, done, e, cat):
    """Per-env snapshot of the carry ENTERING _env_step (all leaves env-sliced, host arrays)."""
    return dict(
        logstate=jax.tree_util.tree_map(lambda x: np.asarray(x)[e], logstate),
        obs=np.asarray(obs)[e],
        memories=np.asarray(memories)[e],
        mask=np.asarray(mask)[e],
        midx=int(np.asarray(midx)[e]),
        done=bool(np.asarray(done)[e]),
        category=cat)

# ---- collection loop init (EXACT control init) ----
rng=jax.random.PRNGKey(SEED); rng,_rng=jax.random.split(rng)
obs,logstate=env.reset(_rng, env_params)
memories=jnp.zeros((E,WM,NL,256)); mask=jnp.zeros((E,NH,1,WM+1),jnp.bool_); midx=jnp.zeros((E,),jnp.int32)+(WM+1)
done=jnp.zeros((E,),jnp.bool_)

# per-env trackers. Transition trackers init from the ACTUAL reset state (no init artifacts):
# under S4_dark the floor-2 spawn already has EGM=True and monsters_killed[2]=8, so the
# transition-defined categories (floor2_reached/gate_unlocked/mid_clear) are structurally empty.
_in0=logstate.env_state
prev_egm=np.asarray(_in0.achievements)[:,EGM].astype(bool).copy()
prev_kills2_ge8=(np.asarray(_in0.monsters_killed)[:,2]>=8).copy()
saw_stair_age=np.full(E,10**6,np.int32)
ever_sewers=np.zeros(E,bool)
pending_near3=[None]*E

t0=time.time()
for step in range(CAPTURE_STEPS):
    # ===================== (0) signals from PRE-step inner state =====================
    inner=logstate.env_state
    pl=np.asarray(inner.player_level)
    kills2=np.asarray(inner.monsters_killed)[:,2]
    egm=np.asarray(inner.achievements)[:,EGM].astype(bool)
    sew=np.asarray(inner.achievements)[:,SEWERS].astype(bool)
    dl=np.asarray(inner.down_ladders); pp=np.asarray(inner.player_position)
    stair_pos=dl[np.arange(E), pl]
    cheb=np.maximum(np.abs(pp-stair_pos)[:,0], np.abs(pp-stair_pos)[:,1])
    stair_vis=cheb<=VIEW_RADIUS
    on_mines=(pl==2)
    sig_floor2 = egm & ~prev_egm
    sig_mid    = on_mines & (kills2>=1) & (kills2<=7)
    sig_gate   = on_mines & (kills2>=8) & ~prev_kills2_ge8
    sig_lost   = on_mines & (~stair_vis) & (saw_stair_age<=RECENT_WINDOW)

    # ===================== (1) snapshot the carry ENTERING this step =====================
    # (logstate, obs, memories[pre-roll], mask[PRE-reset], midx[PRE-reset], done_{t-1})
    for e in range(E):
        if on_mines[e] and kills2[e]>=8 and saw_stair_age[e]<=RECENT_WINDOW and (not ever_sewers[e]):
            pending_near3[e]=snap_slice(logstate, obs, memories, mask, midx, done, e, "near_floor3_failed")
        def put(cat, e=e):
            if len(rings[cat])<CAT_CAP:
                rings[cat].append(snap_slice(logstate, obs, memories, mask, midx, done, e, cat))
        if sig_floor2[e]: put("floor2_reached")
        if sig_mid[e]:    put("mid_clear")
        if sig_gate[e]:   put("gate_unlocked")
        if sig_lost[e]:   put("saw_stair_lost")

    # ===================== (2) full-step validation record (PRE env.step) =====================
    if step < N_VAL_RECORDS:
        val_records.append(dict(step=step, pre_logstate=jax.tree_util.tree_map(np.asarray, logstate),
                                pre_obs=np.asarray(obs)))

    # ===================== (3) top-of-step mask reset (EXACT control) =====================
    midx=jnp.where(done, WM, jnp.clip(midx-1,0,WM))
    mask=jnp.where(done[:,None,None,None], jnp.zeros((E,NH,1,WM+1),jnp.bool_), mask)
    oh=jax.nn.one_hot(midx, WM+1)[:,None,None,:].repeat(NH,1); mask=jnp.logical_or(mask, oh)

    # ===================== (4) forward + action (EXACT) =====================
    rng,_rng=jax.random.split(rng)
    pi,value,mout=net.apply(teacher, memories, obs, mask, method=net.model_forward_eval)
    rng,_rng=jax.random.split(rng)
    action=pi.sample(seed=_rng)
    step_key=np.asarray(_rng)

    # ===================== (5) env.step (EXACT) =====================
    memories=jnp.roll(memories,-1,axis=1).at[:,-1].set(mout)
    obs_next,logstate_next,rew,done_next,info=env.step(_rng, logstate, action, env_params)
    true_done=np.asarray(info["returned_episode"]).astype(bool)

    if step < N_VAL_RECORDS:
        val_records[-1].update(step_key=step_key, actions=np.asarray(action),
            pre_memories=np.asarray(memories), pre_mask=np.asarray(mask), pre_midx=np.asarray(midx),
            post_obs=np.asarray(obs_next), post_logstate=jax.tree_util.tree_map(np.asarray, logstate_next))

    # ===================== (6) near_floor3 commit/discard on episode end =====================
    for e in range(E):
        if true_done[e]:
            if (not ever_sewers[e]) and pending_near3[e] is not None and len(rings["near_floor3_failed"])<CAT_CAP:
                rings["near_floor3_failed"].append(pending_near3[e])
            pending_near3[e]=None; ever_sewers[e]=False; saw_stair_age[e]=10**6

    # advance trackers (prev_* carry the true previous state across optimistic resets via copy)
    ever_sewers |= sew
    prev_egm=egm.copy(); prev_kills2_ge8=(kills2>=8).copy()
    saw_stair_age=np.where(stair_vis, 0, saw_stair_age+1).astype(np.int32)

    obs=obs_next; logstate=logstate_next; done=done_next
    if step%1000==0:
        print(f"[capture] step {step}/{CAPTURE_STEPS} rings="+
              " ".join(f"{c}={len(rings[c])}" for c in CATS), flush=True)

all_snaps=[s for c in CATS for s in rings[c]]
counts={c: len(rings[c]) for c in CATS}
print(f"[capture] DONE in {time.time()-t0:.1f}s  total={len(all_snaps)}  counts={counts}", flush=True)

code_sha=hashlib.sha256(open(__file__,"rb").read()).hexdigest()
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT,"p9_library.pkl"),"wb") as f:
    pickle.dump(dict(snaps=all_snaps, counts=counts, cat_cap=CAT_CAP,
                     view_radius=VIEW_RADIUS, recent_window=RECENT_WINDOW,
                     capture_steps=CAPTURE_STEPS, seed=SEED, code_sha=code_sha,
                     teacher_ckpt=TEACHER_CKPT), f, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join(OUT,"p9_val_records.pkl"),"wb") as f:
    pickle.dump(dict(records=val_records, num_envs=E, code_sha=code_sha), f, protocol=pickle.HIGHEST_PROTOCOL)
summary=dict(label="P9_CAPTURE", total_snapshots=len(all_snaps), counts=counts,
             cat_cap=CAT_CAP, view_radius=VIEW_RADIUS, recent_window=RECENT_WINDOW,
             capture_steps=CAPTURE_STEPS, seed=SEED, n_val_records=len(val_records),
             code_sha=code_sha, teacher_ckpt=TEACHER_CKPT,
             timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
json.dump(summary, open(os.path.join(OUT,"p9_capture_summary.json"),"w"), indent=2)
print("[capture] P9_CAPTURE_OK "+json.dumps(counts), flush=True)
