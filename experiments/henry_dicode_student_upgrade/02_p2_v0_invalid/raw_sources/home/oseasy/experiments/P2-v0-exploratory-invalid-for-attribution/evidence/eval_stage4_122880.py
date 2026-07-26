#!/usr/bin/env python3
"""P2 Stage-4 evaluation @ checkpoint 122880 — 512 episodes, stochastic, fresh world.

Exact same protocol as the 98304 baseline (v7 evaluator):
  SHA256: 4b208ac175e4864d8dae2ddfbd6b22081fc6bf074031a7a9769e30c0295ec3f6

Loads checkpoint 122880, runs 512 zero-shot episodes on the stage-4 scaffold,
records full metrics panel with 95% binomial Wilson CIs for ALL sub-metrics,
and appends to stage4_learning_curve.csv.

Policy mode: STOCHASTIC (pi.sample), seed=42.
"""

import csv
import hashlib
import json
import math
import os
import sys
import time

# ---- bind GPU 0 ----
GPU_UUID = "GPU-e8c08612-c22a-c8a4-6df5-affb2dd1f9a6"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import jax
import jax.numpy as jnp
import numpy as np

# ---- path ----
V7_SRC = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src"
for p in [V7_SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

from craftax.craftax.constants import Achievement
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams

from dicode.network import ActorCriticTransformer
from dicode.task_utils import get_achievement_multi_hot
from dicode.utils.general.train_state_utils import load_weights_only
from dicode.wrappers_cl import DistributedMultiTaskOptimisticLogWrapper
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv

from omegaconf import OmegaConf

# ===========================================================================
# Config (matches P2 network architecture exactly)
# ===========================================================================
cfg = OmegaConf.create({
    "activation": "relu",
    "embed_size": 256,
    "hidden_layers": 256,
    "num_heads": 8,
    "qkv_features": 256,
    "num_layers": 2,
    "gating": True,
    "gating_bias": 2.0,
    "window_mem": 128,
    "window_grad": 64,
    "anneal_lr": False,
    "lr": 2e-4,
    "min_lr": 2e-6,
    "max_grad_norm": 1.0,
    "total_timesteps": 2_005_401_600,
    "num_envs": 1024,
    "num_steps": 128,
    "update_epochs": 4,
    "num_minibatches": 8,
    "max_updates_per_session": 500,
})

# ===========================================================================
# Stage-4 level template (identical to 98304 baseline)
# ===========================================================================
S4_TASK_CODE = """
import jax
from craftax.craftax.constants import Achievement, ItemType
from minicraftax.craftax_state import EnvState, TaskParams
from minicraftax.tasks.base_task import BaseTask
from minicraftax.world_builder import WorldBuilder

class Env(BaseTask):
    def __init__(self, static_params, params):
        super().__init__(static_params, params)
        self.relevant_achievements = [Achievement.DEFEAT_KOBOLD]
        self.completed_achievements = []
        self.label = "DEFEAT_KOBOLD"

    def get_task_params(self) -> TaskParams:
        return TaskParams(needs_depletion_multiplier=0.3)

    def generate_world(self, rng: jax.Array) -> EnvState:
        rng, _rng = jax.random.split(rng)
        builder = WorldBuilder(_rng, self.static_params, self.params)
        builder.set_starting_floor(2)
        builder.set_monsters_killed(2, 8)
        builder.set_player_inventory({'wood': 7, 'stone': 27, 'coal': 3, 'iron': 3,
                                      'sapling': 1, 'pickaxe': 3, 'sword': 3, 'bow': 1,
                                      'arrows': 7, 'torches': 10})
        state = builder.build(rng)
        up = builder.ladders_up[2]
        state = state.replace(item_map=state.item_map.at[2, up[0], up[1]].set(ItemType.NONE.value))
        return state
"""

# ===========================================================================
# Constants (identical to baseline protocol)
# ===========================================================================
WALL = "defeat_kobold"
SPAWN_FLOOR = 2
KOBOLD_FLOOR = 8
NUM_ENVS = 512
NUM_STEPS = 4096
MAX_TIMESTEPS = 4096

CKPT_PATH = "/home/oseasy/experiments/henry_student_p2_amago_20260721/checkpoints/122880"
CKPT_STEP = 122880

# ── Evaluator SHA256 (of the v7 frozen evaluator, not this script) ──
FROZEN_EVAL_PATH = ("/home/oseasy/experiments/mechanism_UED_continuation_20260715/"
                    "workers/gpu1_soft_copeland/_eval_stage4_baseline.py")
with open(FROZEN_EVAL_PATH, "rb") as f:
    FROZEN_EVAL_SHA256 = hashlib.sha256(f.read()).hexdigest()

# ── This script's own SHA256 ──
with open(__file__, "rb") as f:
    ADAPTER_SHA256 = hashlib.sha256(f.read()).hexdigest()

# ── Output: append to existing stage4_learning_curve.csv ──
CSV_PATH = ("/home/oseasy/experiments/mechanism_UED_continuation_20260715/"
            "workers/gpu1_soft_copeland/gpu1_soft_copeland_evidence/"
            "stage4_learning_curve.csv")

# Local report
OUT_DIR = "/home/oseasy/experiments/henry_student_p2_amago_20260721/evidence"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 72)
print("P2 Stage-4 Evaluation @ 122880")
print(f"  checkpoint: {CKPT_PATH} (step {CKPT_STEP})")
print(f"  v7 codebase: {V7_SRC}")
print(f"  GPU UUID: {GPU_UUID}")
print(f"  devices: {[str(d) for d in jax.devices()]}")
print(f"  episodes: {NUM_ENVS}")
print(f"  policy mode: STOCHASTIC (pi.sample)")
print(f"  frozen evaluator SHA256: {FROZEN_EVAL_SHA256}")
print(f"  adapter SHA256: {ADAPTER_SHA256}")
print("=" * 72)

# Verify frozen evaluator SHA256 matches expected
EXPECTED_V7_SHA = "4b208ac175e4864d8dae2ddfbd6b22081fc6bf074031a7a9769e30c0295ec3f6"
assert FROZEN_EVAL_SHA256 == EXPECTED_V7_SHA, (
    f"V7 evaluator SHA256 MISMATCH: expected {EXPECTED_V7_SHA}, got {FROZEN_EVAL_SHA256}"
)
print("  ✓ V7 evaluator SHA256 verified")

# ===========================================================================
# 1. Restore weights
# ===========================================================================
t0 = time.time()

ach = Achievement.DEFEAT_KOBOLD
table = jnp.array([get_achievement_multi_hot([ach])], dtype=jnp.float32)
EMB = int(table.shape[1])
print(f"\n[1/4] Achievement embedding size: {EMB}")

ns: dict = {}
exec(S4_TASK_CODE, ns)
EnvCls = ns["Env"]
ctor_params = EnvParams(max_timesteps=MAX_TIMESTEPS)
base_env = MultiTaskMiniCraftaxEnv(
    [EnvCls], StaticEnvParams(), ctor_params, True,
    conditioning_type="embedding", embedding_size=EMB,
)

train_state = load_weights_only(CKPT_PATH, base_env, ctor_params, cfg, load_opt_state=False)
restore_time = time.time() - t0
param_leaves = len(jax.tree_util.tree_leaves(train_state.params))
print(f"[1/4] Weights restored: {param_leaves} param leaves ({restore_time:.1f}s)")

# ===========================================================================
# 2. Build network + env
# ===========================================================================
network = ActorCriticTransformer(
    action_dim=base_env.action_space(ctor_params).n,
    activation=cfg.activation,
    hidden_layers=cfg.hidden_layers,
    encoder_size=cfg.embed_size,
    num_heads=cfg.num_heads,
    qkv_features=cfg.qkv_features,
    num_layers=cfg.num_layers,
    gating=cfg.gating,
    gating_bias=cfg.gating_bias,
)
ACTION_DIM = base_env.action_space(ctor_params).n
print(f"[2/4] Network: action_dim={ACTION_DIM}, embed={cfg.embed_size}, "
      f"heads={cfg.num_heads}, layers={cfg.num_layers}, hidden={cfg.hidden_layers}")

env = DistributedMultiTaskOptimisticLogWrapper(
    base_env, jax.random.PRNGKey(0), NUM_ENVS, 1, 16, jnp.array([1.0]), table
)

# ===========================================================================
# 3. Evaluation rollout — 512 envs × 4096 steps
# ===========================================================================
print(f"\n[3/4] Running {NUM_ENVS} envs × {NUM_STEPS} steps (stochastic sampling)...")

ach_idx = int(ach.value)

memories = jnp.zeros((NUM_ENVS, cfg.window_mem, cfg.num_layers, cfg.embed_size))
mem_mask = jnp.zeros((NUM_ENVS, cfg.num_heads, 1, cfg.window_mem + 1), dtype=jnp.bool_)
mem_idx = jnp.zeros((NUM_ENVS,), dtype=jnp.int32) + (cfg.window_mem + 1)


def _rollout_step(carry, _):
    (log_state, memories, mem_mask, mem_idx, last_obs, done, finished, ep_len,
     max_floor, seen, info_acc, rng) = carry

    mem_idx = jnp.where(done, cfg.window_mem, jnp.clip(mem_idx - 1, 0, cfg.window_mem))
    mem_mask = jnp.where(done[:, None, None, None], jnp.zeros_like(mem_mask), mem_mask)
    ohot = jax.nn.one_hot(mem_idx, cfg.window_mem + 1)
    ohot = ohot[:, None, None, :].repeat(cfg.num_heads, 1)
    mem_mask = jnp.logical_or(mem_mask, ohot)

    rng, a_rng, s_rng = jax.random.split(rng, 3)
    pi, _, mem_out = network.apply(
        train_state.params, memories, last_obs, mem_mask,
        method=network.model_forward_eval,
    )
    action = pi.sample(seed=a_rng)
    memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(mem_out)

    pre_state = log_state.env_state
    next_obs, next_log_state, reward, next_done, info = env.step(
        s_rng, log_state, action, ctor_params
    )

    active = ~finished
    ep_len = ep_len + active.astype(jnp.int32)
    max_floor = jnp.where(active, jnp.maximum(max_floor, pre_state.player_level), max_floor)
    seen = seen | (pre_state.achievements[:, ach_idx].astype(bool) & active)
    keys = [k for k in info if "Achievements" in k and "kobold" in k.lower()]
    if keys:
        info_acc = info_acc + jnp.asarray(info[keys[0]], jnp.float32) * active.astype(jnp.float32)
    finished = finished | next_done

    return (next_log_state, memories, mem_mask, mem_idx, next_obs, next_done,
            finished, ep_len, max_floor, seen, info_acc, rng), None


rng = jax.random.PRNGKey(42)
rng, reset_rng = jax.random.split(rng)
obsv, log_state = env.reset(reset_rng, ctor_params)

init = (
    log_state, memories, mem_mask, mem_idx, obsv,
    jnp.zeros((NUM_ENVS,), dtype=jnp.bool_),
    jnp.zeros((NUM_ENVS,), dtype=jnp.bool_),
    jnp.zeros((NUM_ENVS,), dtype=jnp.int32),
    jnp.full((NUM_ENVS,), SPAWN_FLOOR, dtype=jnp.int32),
    jnp.zeros((NUM_ENVS,), dtype=jnp.bool_),
    jnp.zeros((NUM_ENVS,), dtype=jnp.float32),
    rng,
)

t0_roll = time.time()
final, _ = jax.lax.scan(_rollout_step, init, None, NUM_STEPS)
(_, _, _, _, _, _, finished, ep_len, max_floor, seen, info_acc, _) = final
roll_time = time.time() - t0_roll

finished_np = np.asarray(finished)
ep_len_np = np.asarray(ep_len)
max_floor_np = np.asarray(max_floor)
seen_np = np.asarray(seen)
info_acc_np = np.asarray(info_acc)

success_np = seen_np | (info_acc_np > 0)
timeout_np = finished_np & (ep_len_np >= NUM_STEPS) & ~success_np
died_np = finished_np & ~success_np & ~timeout_np

N_FINISHED = int(np.sum(finished_np))
N_NOT_FINISHED = NUM_ENVS - N_FINISHED

if N_NOT_FINISHED > 0:
    print(f"  WARNING: {N_NOT_FINISHED}/{NUM_ENVS} envs did NOT finish within {NUM_STEPS} steps")
    timeout_np = timeout_np | ~finished_np

# ---- derived metrics ----
n_success = int(np.sum(success_np))
n_died = int(np.sum(died_np))
n_timeout = int(np.sum(timeout_np))
n_down_stair = int(np.sum(max_floor_np > SPAWN_FLOOR))
n_floor3 = int(np.sum(max_floor_np >= 3))
n_kobold_encounter = int(np.sum(max_floor_np >= KOBOLD_FLOOR))
n_kobold_kill = n_success

sr = n_success / NUM_ENVS
down_stair_rate = n_down_stair / NUM_ENVS
floor3_rate = n_floor3 / NUM_ENVS
kobold_encounter_rate = n_kobold_encounter / NUM_ENVS
kobold_kill_rate = n_kobold_kill / NUM_ENVS
death_rate = n_died / NUM_ENVS
timeout_rate = n_timeout / NUM_ENVS

mean_ep_len = float(np.mean(ep_len_np))
median_ep_len = float(np.median(ep_len_np))


# -----------------------------------------------------------------------
# 95% Wilson score confidence intervals for ALL binomial proportions
# -----------------------------------------------------------------------
def wilson_ci(successes: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return (lo, hi)


sr_ci_lo, sr_ci_hi = wilson_ci(n_success, NUM_ENVS)
ds_ci_lo, ds_ci_hi = wilson_ci(n_down_stair, NUM_ENVS)
f3_ci_lo, f3_ci_hi = wilson_ci(n_floor3, NUM_ENVS)
ke_ci_lo, ke_ci_hi = wilson_ci(n_kobold_encounter, NUM_ENVS)
kk_ci_lo, kk_ci_hi = wilson_ci(n_kobold_kill, NUM_ENVS)
death_ci_lo, death_ci_hi = wilson_ci(n_died, NUM_ENVS)
timeout_ci_lo, timeout_ci_hi = wilson_ci(n_timeout, NUM_ENVS)

# max_floor distribution
floor_counts = {}
for f in range(SPAWN_FLOOR, KOBOLD_FLOOR + 2):
    c = int(np.sum(max_floor_np >= f))
    if c > 0 or f <= KOBOLD_FLOOR:
        floor_counts[f"floor_{f}+"] = c

print(f"\n[3/4] Rollout complete ({roll_time:.1f}s, "
      f"{NUM_ENVS * NUM_STEPS / roll_time:.0f} steps/s)")
print(f"  finished: {N_FINISHED}/{NUM_ENVS}")
print(f"  success (DEFEAT_KOBOLD):  {n_success}/{NUM_ENVS} = {sr*100:.2f}%  "
      f"95% CI [{sr_ci_lo*100:.2f}%, {sr_ci_hi*100:.2f}%]")
print(f"  died:                     {n_died}/{NUM_ENVS} = {death_rate*100:.2f}%")
print(f"  timeout:                  {n_timeout}/{NUM_ENVS} = {timeout_rate*100:.2f}%")
print(f"  down_stair (floor>{SPAWN_FLOOR}): {n_down_stair}/{NUM_ENVS} = {down_stair_rate*100:.2f}%")
print(f"  floor3_reach (>=3):       {n_floor3}/{NUM_ENVS} = {floor3_rate*100:.2f}%")
print(f"  kobold_encounter (>={KOBOLD_FLOOR}):  {n_kobold_encounter}/{NUM_ENVS} = {kobold_encounter_rate*100:.2f}%")
print(f"  kobold_kill (==success):  {n_kobold_kill}/{NUM_ENVS} = {kobold_kill_rate*100:.2f}%")
print(f"  mean ep_len: {mean_ep_len:.1f}  median: {median_ep_len:.1f}")
print(f"  floor progression: {floor_counts}")

# ===========================================================================
# 4. Append to stage4_learning_curve.csv
# ===========================================================================
FIELD_NAMES = [
    "checkpoint_step",
    "num_episodes",
    "policy_mode",
    "evaluation_seed",
    "stage4_DEFEAT_KOBOLD_SR",
    "SR_95CI_lo", "SR_95CI_hi",
    "down_stair_found_rate", "DS_95CI_lo", "DS_95CI_hi",
    "floor3_reach_rate", "floor3_95CI_lo", "floor3_95CI_hi",
    "kobold_encounter_rate", "KE_95CI_lo", "KE_95CI_hi",
    "kobold_kill_rate", "KK_95CI_lo", "KK_95CI_hi",
    "death_rate", "death_95CI_lo", "death_95CI_hi",
    "timeout_rate", "timeout_95CI_lo", "timeout_95CI_hi",
    "mean_episode_length",
    "median_episode_length",
    "max_floor_max",
    "max_floor_median",
    "wall",
    "spawn_floor",
    "kobold_floor",
    "n_success",
    "n_died",
    "n_timeout",
    "n_kobold_encounter",
    "n_total",
    "evaluator_sha256",
    "timestamp_utc",
]

row = {
    "checkpoint_step": CKPT_STEP,
    "num_episodes": NUM_ENVS,
    "policy_mode": "stochastic",
    "evaluation_seed": 42,
    "stage4_DEFEAT_KOBOLD_SR": round(sr, 6),
    "SR_95CI_lo": round(sr_ci_lo, 6),
    "SR_95CI_hi": round(sr_ci_hi, 6),
    "down_stair_found_rate": round(down_stair_rate, 6),
    "DS_95CI_lo": round(ds_ci_lo, 6),
    "DS_95CI_hi": round(ds_ci_hi, 6),
    "floor3_reach_rate": round(floor3_rate, 6),
    "floor3_95CI_lo": round(f3_ci_lo, 6),
    "floor3_95CI_hi": round(f3_ci_hi, 6),
    "kobold_encounter_rate": round(kobold_encounter_rate, 6),
    "KE_95CI_lo": round(ke_ci_lo, 6),
    "KE_95CI_hi": round(ke_ci_hi, 6),
    "kobold_kill_rate": round(kobold_kill_rate, 6),
    "KK_95CI_lo": round(kk_ci_lo, 6),
    "KK_95CI_hi": round(kk_ci_hi, 6),
    "death_rate": round(death_rate, 6),
    "death_95CI_lo": round(death_ci_lo, 6),
    "death_95CI_hi": round(death_ci_hi, 6),
    "timeout_rate": round(timeout_rate, 6),
    "timeout_95CI_lo": round(timeout_ci_lo, 6),
    "timeout_95CI_hi": round(timeout_ci_hi, 6),
    "mean_episode_length": round(mean_ep_len, 1),
    "median_episode_length": round(median_ep_len, 1),
    "max_floor_max": int(np.max(max_floor_np)),
    "max_floor_median": float(np.median(max_floor_np)),
    "wall": WALL,
    "spawn_floor": SPAWN_FLOOR,
    "kobold_floor": KOBOLD_FLOOR,
    "n_success": n_success,
    "n_died": n_died,
    "n_timeout": n_timeout,
    "n_kobold_encounter": n_kobold_encounter,
    "n_total": NUM_ENVS,
    "evaluator_sha256": FROZEN_EVAL_SHA256,  # v7 frozen evaluator SHA
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}

# Read existing CSV header to verify column match
with open(CSV_PATH, "r") as f:
    reader = csv.DictReader(f)
    existing_fields = reader.fieldnames
print(f"\n  Existing CSV columns: {len(existing_fields)}")
print(f"  Our columns:          {len(FIELD_NAMES)}")

# Verify column names match
if existing_fields != FIELD_NAMES:
    print(f"  WARNING: Column mismatch!")
    print(f"    Existing but not ours: {set(existing_fields) - set(FIELD_NAMES)}")
    print(f"    Ours but not existing: {set(FIELD_NAMES) - set(existing_fields)}")

# Append row
with open(CSV_PATH, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELD_NAMES)
    writer.writerow(row)

print(f"\n[4/4] Row appended to: {CSV_PATH}")

# ===========================================================================
# Full JSON report (local evidence)
# ===========================================================================
REPORT_PATH = os.path.join(OUT_DIR, "stage4_eval_122880_report.json")

per_episode_fields = ["ep_len", "max_floor", "success", "died", "timeout"]
ep_details = {
    "ep_len": ep_len_np.tolist(),
    "max_floor": max_floor_np.tolist(),
    "success": success_np.astype(int).tolist(),
    "died": died_np.astype(int).tolist(),
    "timeout": timeout_np.astype(int).tolist(),
}

report = {
    "timestamp_utc": row["timestamp_utc"],
    "frozen_evaluator_sha256": FROZEN_EVAL_SHA256,
    "adapter_sha256": ADAPTER_SHA256,
    "p2_checkpoint": CKPT_PATH,
    "p2_checkpoint_step": CKPT_STEP,
    "v7_codebase": V7_SRC,
    "gpu_uuid_bound": GPU_UUID,
    "gpu_devices_visible": [str(d) for d in jax.devices()],
    "evaluation_config": {
        "num_episodes": NUM_ENVS,
        "num_steps": NUM_STEPS,
        "policy_mode": "stochastic",
        "evaluation_seed": 42,
        "wall": WALL,
        "spawn_floor": SPAWN_FLOOR,
        "kobold_floor": KOBOLD_FLOOR,
    },
    "network_architecture": {
        "embed_size": cfg.embed_size,
        "num_heads": cfg.num_heads,
        "num_layers": cfg.num_layers,
        "hidden_layers": cfg.hidden_layers,
        "qkv_features": cfg.qkv_features,
        "gating": cfg.gating,
        "gating_bias": cfg.gating_bias,
        "window_mem": cfg.window_mem,
        "action_dim": ACTION_DIM,
    },
    "weight_restore": {
        "param_leaves": param_leaves,
        "restore_time_s": round(restore_time, 1),
    },
    "results": {
        "stage4_DEFEAT_KOBOLD_SR": round(sr, 6),
        "SR_95CI_lo": round(sr_ci_lo, 6),
        "SR_95CI_hi": round(sr_ci_hi, 6),
        "down_stair_found_rate": round(down_stair_rate, 6),
        "DS_95CI_lo": round(ds_ci_lo, 6),
        "DS_95CI_hi": round(ds_ci_hi, 6),
        "floor3_reach_rate": round(floor3_rate, 6),
        "floor3_95CI_lo": round(f3_ci_lo, 6),
        "floor3_95CI_hi": round(f3_ci_hi, 6),
        "kobold_encounter_rate": round(kobold_encounter_rate, 6),
        "KE_95CI_lo": round(ke_ci_lo, 6),
        "KE_95CI_hi": round(ke_ci_hi, 6),
        "kobold_kill_rate": round(kobold_kill_rate, 6),
        "KK_95CI_lo": round(kk_ci_lo, 6),
        "KK_95CI_hi": round(kk_ci_hi, 6),
        "death_rate": round(death_rate, 6),
        "death_95CI_lo": round(death_ci_lo, 6),
        "death_95CI_hi": round(death_ci_hi, 6),
        "timeout_rate": round(timeout_rate, 6),
        "timeout_95CI_lo": round(timeout_ci_lo, 6),
        "timeout_95CI_hi": round(timeout_ci_hi, 6),
        "mean_episode_length": round(mean_ep_len, 1),
        "median_episode_length": round(median_ep_len, 1),
        "max_floor_max": int(np.max(max_floor_np)),
        "max_floor_median": float(np.median(max_floor_np)),
        "n_success": n_success,
        "n_died": n_died,
        "n_timeout": n_timeout,
        "n_total": NUM_ENVS,
    },
    "floor_progression": floor_counts,
    "rollout_time_s": round(roll_time, 1),
}

with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2, default=str)

print(f"Full report written to: {REPORT_PATH}")

# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 72)
print("STAGE 4 @ 122880 — FINAL")
print(f"  checkpoint_step:  {CKPT_STEP}")
print(f"  DEFEAT_KOBOLD SR: {sr*100:.2f}%  [{sr_ci_lo*100:.2f}%, {sr_ci_hi*100:.2f}%]  (n={NUM_ENVS})")
print(f"  Baseline (98304): 1.56%  [0.79%, 3.05%]  ← for reference")
print(f"  kobold_encounter: {kobold_encounter_rate*100:.2f}%  [{ke_ci_lo*100:.2f}%, {ke_ci_hi*100:.2f}%]")
print(f"  kobold_kill_rate: {kobold_kill_rate*100:.2f}%")
print(f"  down_stair_found: {down_stair_rate*100:.2f}%")
print(f"  floor3_reach:     {floor3_rate*100:.2f}%")
print(f"  death_rate:       {death_rate*100:.2f}%")
print(f"  timeout_rate:     {timeout_rate*100:.2f}%")
print(f"  mean ep_len:      {mean_ep_len:.1f}")
print(f"  median ep_len:    {median_ep_len:.1f}")
print(f"  policy_mode:      STOCHASTIC (pi.sample)")
print(f"  frozen evaluator: {FROZEN_EVAL_SHA256[:16]}...")
print("=" * 72)
