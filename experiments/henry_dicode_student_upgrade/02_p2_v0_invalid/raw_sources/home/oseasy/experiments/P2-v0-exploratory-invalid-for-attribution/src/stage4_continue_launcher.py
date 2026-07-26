#!/usr/bin/env python3
"""D059 Stage4 Continuation Launcher — Experiment A (AMAGO_STYLE_EXPLORATORY_P2).

Resumes from checkpoints/98304 with FULL state restoration:
  - Model parameters restored via orbax (matching template → extract params)
  - Optimizer RESTARTED with anneal_lr=False (Adam lr=2e-4, eps=1e-5)
    The saved checkpoint uses anneal_lr=True (schedule-based optimizer) which
    is structurally incompatible with a fixed-LR optimizer in optax.  We
    restore params only and create a fresh optimizer.  47 prior gradient
    updates — momentum loss is negligible.  Explicitly documented.
  - Replay buffer (112 trajectories + counters + internal RNG)
  - JAX RNG key
  - Global step counter (continues from 98304)

Runs sessions of exactly 24,576 env steps each (16 envs × 128 rollout × 12
updates).  After every session, saves an atomic checkpoint containing:
  params, optimizer, RNG, global_step, resolved_config, source_hash,
  replay_meta, and Stage4 identity.

Fixed Stage4 configuration:
  - gamma=0.999, lambda=0.8 (unchanged from base checkpoint)
  - anneal_lr=False (NO automatic learning-rate annealing)
  - Floor-2 spawn, 8/8 monster gate pre-cleared, up-ladder REMOVED
  - Winner-median starting kit, DEFEAT_KOBOLD target
  - depletion_multiplier=0.3

Independent output — never overwrites existing checkpoints or D059 outputs.

Usage:
  python src/stage4_continue_launcher.py [--max-sessions N] [--smoke-test]
"""

import hashlib
import json
import os
import pickle
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax.training.train_state import TrainState

# ── Paths ────────────────────────────────────────────────────────────
_HENRY_SRC = ("/home/oseasy/incoming/henry_work_20260721T105300/extracted/"
              "Henry_work/code/dicode_v7fix58_armB/src")
_AMAGO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, _HENRY_SRC)
sys.path.insert(0, _AMAGO_SRC)

from craftax.craftax.constants import Achievement
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from dicode.network import ActorCriticTransformer
from dicode.task_utils import get_achievement_multi_hot
from minicraftax.envs.craftax import CraftaxAugObsTrain
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv
from dicode.wrappers_cl import DistributedMultiTaskOptimisticLogWrapper

from trajectory_replay import Trajectory, TrajectoryReplayBuffer
from hindsight import relabel_sample
from long_context_learner import LongContextLearner
from checkpointing import save_full_checkpoint

# ═══════════════════════════════════════════════════════════════════════
# IMMUTABLE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

EXPECTED_GPU_UUID = "GPU-e8c08612-c22a-c8a4-6df5-affb2dd1f9a6"
SESSION_ENV_STEPS = 24576
NUM_ENVS = 16
ROLLOUT_STEPS = 128
NUM_UPDATES_PER_SESSION = 12  # 16 * 128 * 12 == 24576
assert NUM_ENVS * ROLLOUT_STEPS * NUM_UPDATES_PER_SESSION == SESSION_ENV_STEPS, \
    f"STEP ARITHMETIC FAIL: {NUM_ENVS}*{ROLLOUT_STEPS}*{NUM_UPDATES_PER_SESSION} != {SESSION_ENV_STEPS}"

GAMMA = 0.999
GAE_LAMBDA = 0.8

# ── Resume checkpoint ─────────────────────────────────────────────────
CKPT_PARENT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "checkpoints"))
# CKPT_RESUME_STEP is now dynamic: set via --resume-from or auto-detected
# from the latest numbered step directory under checkpoints/.

# ── Output paths ──────────────────────────────────────────────────────
OUTPUT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
CKPT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints"))
EVIDENCE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evidence"))
S4_TASK_PATH = os.path.join(EVIDENCE_DIR, "s4_task_code.py")

for d in [OUTPUT_ROOT, CKPT_ROOT, EVIDENCE_DIR]:
    os.makedirs(d, exist_ok=True)


def detect_latest_checkpoint_step(ckpt_parent: str) -> int:
    """Scan the checkpoints directory for the highest-numbered step.

    Only considers directories whose names are pure integers and that
    contain a valid 'default/' orbax subdirectory + 'replay_meta.pkl'.
    Skips 0 (the Stage A/B initial template checkpoint) and any
    underscore-prefixed smoke-test directories.
    """
    root = Path(ckpt_parent)
    candidates = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        if not d.name.isdigit():
            continue
        step = int(d.name)
        if step == 0:
            continue  # skip initial template checkpoint
        # Must contain orbax data (default/) AND replay_meta.pkl
        if not (d / "default").is_dir():
            continue
        if not (d / "replay_meta.pkl").exists():
            continue
        candidates.append(step)

    if not candidates:
        raise FileNotFoundError(
            f"No valid checkpoint steps found under {ckpt_parent}. "
            f"Expected at least one numbered directory with default/ and replay_meta.pkl."
        )

    latest = max(candidates)
    print(f"[detect] Latest checkpoint step: {latest}  "
          f"(candidates: {sorted(candidates)})")
    return latest

# ═══════════════════════════════════════════════════════════════════════
# Stage4 Fixed Config (anneal_lr=False — no automatic annealing)
# ═══════════════════════════════════════════════════════════════════════

class Cfg:
    """Stage4 config: gamma=0.999, lambda=0.8, NO annealing."""
    lr = 2e-4
    min_lr = 2e-6
    num_envs = NUM_ENVS
    num_steps = ROLLOUT_STEPS
    update_epochs = 1
    num_minibatches = 2
    gamma = GAMMA
    gae_lambda = GAE_LAMBDA
    clip_eps = 0.2
    ent_coef = 0.002
    vf_coef = 0.5
    max_grad_norm = 1.0
    activation = "relu"
    anneal_lr = False  # ← Stage4: NO automatic annealing
    qkv_features = 256
    embed_size = 256
    num_heads = 8
    num_layers = 2
    hidden_layers = 256
    window_mem = 128
    window_grad = 64
    gating = True
    gating_bias = 2.0
    condition_on_task = True
    optimistic_reset_ratio = 16
    mode = "score"
    bonus_type = "none"
    dynamic_bonus_k = 0.0
    completion_bonus_scale = 0.0
    completion_bonus_min = 0.0
    max_updates_per_session = NUM_UPDATES_PER_SESSION
    total_timesteps = 2_005_401_600
    scoring_window_updates = 4
    value_target_clip_min = -50.0
    value_target_clip_max = 300.0
    guard_session_vloss_max = 1000.0
    guard_session_entropy_min = 0.10
    guard_max_consecutive_reverts = 2
    lr_restart = 0.0
    lr_restart_at = 0
    lr_restart_horizon = 0
    lr_restart_warmup = 50
    sil = False
    sil_pools = []


def _cfg_resolved_dict(cfg: type) -> dict:
    """Extract the resolved config as a plain dict for checkpoint metadata."""
    result = {}
    for k in dir(cfg):
        if k.startswith("__") or k.startswith("_"):
            continue
        v = getattr(cfg, k)
        if callable(v):
            continue
        result[k] = v
    return result


# ═══════════════════════════════════════════════════════════════════════
# Guards
# ═══════════════════════════════════════════════════════════════════════

def verify_gpu():
    """Verify the target GPU is present and JAX can see it."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            text=True
        ).strip().split("\n")
    except Exception:
        print("STOP: cannot query nvidia-smi"); sys.exit(1)

    if EXPECTED_GPU_UUID not in out:
        print(f"STOP: GPU {EXPECTED_GPU_UUID} not found. Available: {out}")
        sys.exit(1)

    devices = jax.devices("gpu")
    if not devices:
        print("STOP: JAX sees no GPU devices (CPU fallback)")
        sys.exit(1)

    # Bind JAX to the specific GPU
    target_idx = out.index(EXPECTED_GPU_UUID)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(target_idx)
    print(f"[guard] GPU OK  |  target={EXPECTED_GPU_UUID}  "
          f"index={target_idx}  JAX devices={len(devices)}")


def guard_output_collision(session_dir: str) -> str:
    """Ensure the session output directory is fresh (timestamped for uniqueness).

    Unlike D059 stage_A/stage_B, this does NOT fail on collision — it
    creates a unique timestamped subdirectory so multiple continuation
    runs never overwrite each other.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    unique_dir = os.path.join(session_dir, f"session_{ts}")
    os.makedirs(unique_dir, exist_ok=False)
    return unique_dir


# ═══════════════════════════════════════════════════════════════════════
# Network initialization (matching load_weights_only internals exactly)
# ═══════════════════════════════════════════════════════════════════════

def init_network_params(
    network: ActorCriticTransformer,
    obs_dim: int,
    cfg: type,
    rng: jax.random.PRNGKey,
) -> dict:
    """Initialize network parameters with the correct shapes.

    MUST match the init call inside load_weights_only in
    dicode/utils/general/train_state_utils.py exactly:
      - batch_size = 2
      - NO method= parameter (uses default apply)
      - memory shape: [B, window_mem, num_layers, embed_size]
      - obs shape: [B, obs_dim]
      - mask shape: [B, num_heads, 1, window_mem + 1]
    """
    rng, _rng = jax.random.split(rng)
    init_obs = jnp.zeros((2, obs_dim))
    init_memory = jnp.zeros((2, cfg.window_mem, cfg.num_layers, cfg.embed_size))
    init_mask = jnp.zeros((2, cfg.num_heads, 1, cfg.window_mem + 1), dtype=jnp.bool_)
    return network.init(_rng, init_memory, init_obs, init_mask)


def build_matching_template(
    network: ActorCriticTransformer,
    obs_dim: int,
    rng: jax.random.PRNGKey,
) -> TrainState:
    """Build a TrainState template matching the SAVED checkpoint structure.

    The checkpoint at 98304 was saved with anneal_lr=True (schedule-based
    Adam).  We must create an identical optimizer structure so orbax can
    deserialize the saved opt_state.  After restore we extract params and
    rebuild with anneal_lr=False.
    """
    # Use a temporary config with anneal_lr=True to match saved structure
    class _MatchCfg:
        lr = 2e-4; min_lr = 2e-6; num_envs = NUM_ENVS; num_steps = ROLLOUT_STEPS
        update_epochs = 1; num_minibatches = 2
        gamma = GAMMA; gae_lambda = GAE_LAMBDA
        clip_eps = 0.2; ent_coef = 0.002; vf_coef = 0.5; max_grad_norm = 1.0
        activation = "relu"; anneal_lr = True  # ← match saved structure
        qkv_features = 256; embed_size = 256; num_heads = 8; num_layers = 2
        hidden_layers = 256; window_mem = 128; window_grad = 64
        gating = True; gating_bias = 2.0
        condition_on_task = True; optimistic_reset_ratio = 16
        mode = "score"; bonus_type = "none"; dynamic_bonus_k = 0.0
        completion_bonus_scale = 0.0; completion_bonus_min = 0.0
        max_updates_per_session = NUM_UPDATES_PER_SESSION
        total_timesteps = 2_005_401_600
        scoring_window_updates = 4
        value_target_clip_min = -50.0; value_target_clip_max = 300.0
        guard_session_vloss_max = 1000.0; guard_session_entropy_min = 0.10
        guard_max_consecutive_reverts = 2
        lr_restart = 0.0; lr_restart_at = 0
        lr_restart_horizon = 0; lr_restart_warmup = 50
        sil = False; sil_pools = []

    mc = _MatchCfg()

    network_params = init_network_params(network, obs_dim, mc, rng)

    # Reconstruct the EXACT optimizer used when the checkpoint was saved
    TOTAL_GLOBAL_UPDATES = (
        (mc.total_timesteps // mc.num_envs // mc.num_steps // mc.max_updates_per_session) + 1
    ) * mc.max_updates_per_session

    def linear_schedule(count):
        u = count // (mc.num_minibatches * mc.update_epochs)
        frac = 1.0 - u / TOTAL_GLOBAL_UPDATES
        frac = jnp.maximum(frac, 0.0)
        lr_val = mc.min_lr + (mc.lr - mc.min_lr) * frac
        if (getattr(mc, "lr_restart", 0.0) or 0.0) > 0.0:
            span = mc.lr_restart_horizon - mc.lr_restart_at
            frac2 = jnp.clip((mc.lr_restart_horizon - u) / span, 0.0, 1.0)
            leg2 = mc.min_lr + (mc.lr_restart - mc.min_lr) * frac2
            warm = jnp.clip((u - mc.lr_restart_at) / mc.lr_restart_warmup, 0.0, 1.0)
            leg2 = mc.min_lr + (leg2 - mc.min_lr) * warm
            lr_val = jnp.where(u >= mc.lr_restart_at, leg2, lr_val)
        return lr_val

    tx = optax.chain(
        optax.clip_by_global_norm(mc.max_grad_norm),
        optax.adam(learning_rate=linear_schedule, eps=1e-5),
    )
    return TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)


def build_stage4_train_state(
    network: ActorCriticTransformer,
    restored_params: dict,
    cfg: type,
) -> TrainState:
    """Build a fresh Stage4 TrainState with restored params and NO annealing."""
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(cfg.lr, eps=1e-5),
    )
    return TrainState.create(apply_fn=network.apply, params=restored_params, tx=tx)


# ═══════════════════════════════════════════════════════════════════════
# Checkpoint resume
# ═══════════════════════════════════════════════════════════════════════

def restore_from_checkpoint(
    ckpt_parent: str,
    ckpt_step: int,
    network: ActorCriticTransformer,
    cfg: type,
    obs_dim: int,
) -> dict:
    """Restore full training state from a D059 checkpoint.

    Strategy:
      1. Try restoring with the Stage4 template (anneal_lr=False) first —
         this works for checkpoints saved by THIS launcher.
      2. If that fails (legacy checkpoints like 98304 use anneal_lr=True),
         fall back to a matching template (anneal_lr=True) for orbax
         deserialization.
      3. Extract params from whichever restore succeeded.
      4. ALWAYS rebuild with Stage4 anneal_lr=False optimizer (documented
         optimizer restart when falling back to legacy template).
      5. Restore replay buffer + RNG + global_step from replay_meta.pkl.

    Returns dict with keys: train_state, replay_buffer, rng, global_step,
    step, manifest, optimizer_restarted.
    """
    print(f"\n[resume] Restoring from: {ckpt_parent}  step={ckpt_step}")

    init_rng = jax.random.PRNGKey(0)
    root = Path(ckpt_parent)
    checkpointer = ocp.PyTreeCheckpointer()
    options = ocp.CheckpointManagerOptions(create=False)
    ckpt_manager = ocp.CheckpointManager(str(root), checkpointer, options=options)

    restored_params = None
    optimizer_restarted = False
    optimizer_note = ""

    # ── Attempt 1: Stage4 template (anneal_lr=False) ──────────────
    # Works for checkpoints saved by this launcher.
    stage4_ts = build_stage4_train_state(
        network, init_network_params(network, obs_dim, cfg, init_rng), cfg)
    try:
        restored_ts = ckpt_manager.restore(ckpt_step, items=stage4_ts)
        restored_params = restored_ts.params
        saved_opt_step = int(jax.tree_util.tree_leaves(restored_ts.opt_state)[0])
        print(f"  Orbax restore OK (Stage4 template, anneal_lr=False)  |"
              f"  param_leaves={len(jax.tree_util.tree_leaves(restored_params))}"
              f"  saved_opt_step={saved_opt_step}")
        # Optimizer structure matches — we can use the restored TrainState
        # directly (no rebuild needed).  But we still call build_stage4_train_state
        # for consistency — it's a no-op in structure terms.
        ts_stage4 = restored_ts
    except (ValueError, TypeError, AssertionError) as e:
        print(f"  Stage4 template restore skipped (structure mismatch, expected"
              f" for legacy checkpoints): {str(e)[:120]}")

        # ── Attempt 2: Legacy matching template (anneal_lr=True) ──
        print("  Building matching template (anneal_lr=True) for legacy ckpt ...")
        match_ts = build_matching_template(network, obs_dim, init_rng)
        try:
            restored_ts = ckpt_manager.restore(ckpt_step, items=match_ts)
            restored_params = restored_ts.params
            saved_opt_step = int(jax.tree_util.tree_leaves(restored_ts.opt_state)[0])
            print(f"  Orbax restore OK (legacy template, anneal_lr=True)  |"
                  f"  param_leaves={len(jax.tree_util.tree_leaves(restored_params))}"
                  f"  saved_opt_step={saved_opt_step}")
            # Rebuild with Stage4 optimizer (optimizer RESTART)
            ts_stage4 = build_stage4_train_state(network, restored_params, cfg)
            optimizer_restarted = True
            optimizer_note = (
                "Optimizer RESTARTED: legacy ckpt uses anneal_lr=True "
                "(schedule-based Adam, incompatible optax structure with "
                "anneal_lr=False).  Params restored; fresh Adam(lr="
                f"{cfg.lr}) created."
            )
        except Exception as e2:
            raise RuntimeError(
                f"Failed to restore checkpoint {ckpt_parent}/{ckpt_step} "
                f"with either template. Stage4 error: {e}; "
                f"Legacy error: {e2}"
            ) from e2

    assert restored_params is not None, "Failed to restore params"

    if not optimizer_restarted:
        # Already matched — still note it in the return
        optimizer_note = (
            f"Optimizer CONTINUED: ckpt already uses anneal_lr=False "
            f"(Stage4-compatible structure).  Optimizer state preserved."
        )
    print(f"  TrainState ready: anneal_lr=False"
          f"{' (optimizer RESTARTED from legacy ckpt)' if optimizer_restarted else ''}")

    # ── Restore replay buffer + RNG + global_step from pickle ─────
    replay_meta_path = os.path.join(ckpt_parent, str(ckpt_step), "replay_meta.pkl")
    with open(replay_meta_path, "rb") as f:
        replay_meta = pickle.load(f)
    replay = TrajectoryReplayBuffer.from_state_dict(replay_meta["replay_state"])
    rng = replay_meta["rng_key"]
    gs = replay_meta["global_step"]

    # ── Load manifest ─────────────────────────────────────────────
    manifest_path = os.path.join(ckpt_parent, str(ckpt_step), "manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

    print(f"  replay_buffer: {len(replay)} trajectories (RESTORED)")
    print(f"  rng:           {rng}")
    print(f"  global_step:   {gs}")

    # Verify continuity
    assert gs == ckpt_step, \
        f"Global step mismatch: expected {ckpt_step}, got {gs}"
    assert replay is not None and len(replay) > 0, \
        "Replay buffer is empty — cannot continue training"

    return {
        "train_state": ts_stage4,
        "replay_buffer": replay,
        "rng": rng,
        "global_step": gs,
        "step": ckpt_step,
        "manifest": manifest,
        "optimizer_restarted": optimizer_restarted,
        "optimizer_note": optimizer_note,
    }


# ═══════════════════════════════════════════════════════════════════════
# Per-session training
# ═══════════════════════════════════════════════════════════════════════

def run_session(
    ts: TrainState,
    replay: TrajectoryReplayBuffer,
    rng: jax.random.PRNGKey,
    global_step: int,
    session_index: int,
    cfg: type,
    network: ActorCriticTransformer,
    learner: LongContextLearner,
    env,
    env_params,
    ach_table: jnp.ndarray,
    out_dir: str,
) -> dict:
    """Run ONE session of 24,576 env steps and return updated state.

    Returns dict with updated: ts, replay, rng, global_step, metrics,
    session_log, crash_info.
    """
    session_t0 = time.time()
    total_env_steps = 0
    crash_info = None

    # Per-env episode buffers
    ep = [{"obs": [], "act": [], "rew": [], "don": [], "val": [], "lp": [],
           "ach": [], "mem_seq": [], "init_mem": None} for _ in range(NUM_ENVS)]

    all_logs = []

    # Counters for this session
    pass_trajs_long = 0
    pass_samples_long = 0
    pass_grad_updates = 0
    pass_relabel_evidence = []

    obsv_j, env_state = None, None

    memories_j = jnp.zeros((NUM_ENVS, cfg.window_mem, cfg.num_layers, cfg.embed_size))
    mem_mask_j = jnp.zeros((NUM_ENVS, cfg.num_heads, 1, cfg.window_mem + 1), dtype=jnp.bool_)
    mem_idx_j = jnp.full((NUM_ENVS,), cfg.window_mem + 1, dtype=jnp.int32)

    # Compiled forward pass  (batch_size=16 — fine, transformer minimum is 2)
    @jax.jit
    def jit_forward(params, mem, obs, mask):
        pi, value, mem_out = network.apply(
            params, mem, obs, mask, method=network.model_forward_eval)
        return pi.logits, value, mem_out

    def env_step_call(env_state, action, step_rng):
        return env.step(step_rng, env_state, action, env_params)

    print(f"\n── Session {session_index} "
          f"(global_step={global_step}) "
          f"target={SESSION_ENV_STEPS} steps ──")

    try:
        # ── Env reset at session start ────────────────────────────
        rng, reset_rng = jax.random.split(rng)
        obsv_j, env_state = env.reset(reset_rng, env_params)

        for up in range(NUM_UPDATES_PER_SESSION):
            # ── Collect ROLLOUT_STEPS transitions ─────────────────
            for st in range(ROLLOUT_STEPS):
                total_env_steps += NUM_ENVS

                mem_idx_j = jnp.clip(mem_idx_j - 1, 0, cfg.window_mem)
                ohot = jax.nn.one_hot(mem_idx_j, cfg.window_mem + 1)
                ohot = ohot[:, None, None, :].repeat(cfg.num_heads, 1)
                mem_mask_j = jnp.logical_or(mem_mask_j, ohot)

                rng, a_rng = jax.random.split(rng)
                logits, value, mem_out = jit_forward(
                    ts.params, memories_j, obsv_j, mem_mask_j)
                logits_np = np.asarray(logits)
                value_np = np.asarray(value)
                mem_out_np = np.asarray(mem_out)

                probs = jax.nn.softmax(jnp.asarray(logits), axis=-1)
                probs_np = np.asarray(probs)
                actions_np = np.array([
                    np.random.choice(probs_np.shape[1], p=p)
                    for p in probs_np])
                logp_np = np.log(
                    probs_np[np.arange(NUM_ENVS), actions_np] + 1e-12)

                memories_j = jnp.roll(memories_j, -1, axis=1).at[:, -1].set(mem_out_np)

                rng, s_rng = jax.random.split(rng)
                obsv_j, env_state, reward_j, done_j, info = env_step_call(
                    env_state, actions_np, s_rng)
                reward_np = np.asarray(reward_j)
                done_np = np.asarray(done_j)

                memories_j = jnp.where(
                    done_np[:, None, None, None],
                    jnp.zeros_like(memories_j), memories_j)
                mem_mask_j = jnp.where(
                    done_np[:, None, None, None],
                    jnp.zeros_like(mem_mask_j), mem_mask_j)
                mem_idx_j = jnp.where(done_np, cfg.window_mem, mem_idx_j)

                # Extract achievements
                ach_data = np.zeros((NUM_ENVS, 67), dtype=np.float32)
                try:
                    est = env_state.env_state
                    if hasattr(est, 'achievements'):
                        ach_data = np.asarray(est.achievements).astype(np.float32)
                except Exception:
                    pass

                for e in range(NUM_ENVS):
                    buf = ep[e]
                    if buf["init_mem"] is None:
                        buf["init_mem"] = np.asarray(memories_j[e]).copy()
                    buf["obs"].append(np.asarray(obsv_j[e]))
                    buf["act"].append(int(actions_np[e]))
                    buf["rew"].append(float(reward_np[e]))
                    buf["don"].append(bool(done_np[e]))
                    buf["val"].append(float(value_np[e]))
                    buf["lp"].append(float(logp_np[e]))
                    buf["ach"].append(ach_data[e].copy())
                    buf["mem_seq"].append(np.asarray(memories_j[e]).copy())

                    if done_np[e]:
                        L = len(buf["obs"])
                        if L > 0 and buf["init_mem"] is not None:
                            traj = Trajectory(
                                observations=np.stack(buf["obs"]),
                                actions=np.array(buf["act"], dtype=np.int32),
                                rewards=np.array(buf["rew"], dtype=np.float32),
                                dones=np.array(buf["don"], dtype=bool),
                                values=np.array(buf["val"], dtype=np.float32),
                                log_probs=np.array(buf["lp"], dtype=np.float32),
                                initial_memory=buf["init_mem"],
                                achievements=np.stack(buf["ach"]),
                                target_achievements=np.asarray(ach_table[0]),
                                memory_sequence=np.stack(buf["mem_seq"]))
                            replay.insert(traj)
                            replay.counters.trajectories_collected += 1
                            if L > 128:
                                pass_trajs_long += 1
                        buf.update({"obs": [], "act": [], "rew": [], "don": [],
                                     "val": [], "lp": [], "ach": [], "mem_seq": [],
                                     "init_mem": None})

            # ── Off-policy update ─────────────────────────────────
            off_metrics = None
            if replay.can_sample():
                try:
                    sample = replay.sample()
                    pass_samples_long += 1

                    ach_any = sample.achievements.max(axis=0)
                    if ach_any.any():
                        old_target = np.argmax(sample.target_achievements)
                        sample = relabel_sample(sample)
                        new_target = np.argmax(sample.target_achievements)
                        replay.counters.relabelled_samples += 1
                        pass_relabel_evidence.append({
                            "from_goal_idx": int(old_target),
                            "to_goal_idx": int(new_target),
                        })

                    ts, metrics = learner.update(ts, sample)
                    replay.counters.gradient_updates += 1
                    off_metrics = metrics

                    if (metrics["grad_norm"] > 1e-12
                            and np.isfinite(metrics["grad_norm"])):
                        pass_grad_updates += 1

                    if not np.isfinite(metrics["total_loss"]):
                        raise RuntimeError(
                            f"NaN/Inf loss: {metrics['total_loss']}")
                    if not np.isfinite(metrics["grad_norm"]):
                        raise RuntimeError(
                            f"NaN/Inf grad_norm: {metrics['grad_norm']}")
                except (ValueError, RuntimeError) as e:
                    msg = str(e)
                    if "Gate" in msg or "128" in msg:
                        pass
                    else:
                        raise

            # ── Log ───────────────────────────────────────────────
            cs = replay.counters.snapshot()
            log = {
                "session": session_index, "update": up,
                "session_env_steps": total_env_steps,
                "global_env_steps": global_step + total_env_steps,
                "elapsed_s": round(time.time() - session_t0, 1),
                "replay_size": len(replay),
                "replay_max_len": replay.longest_trajectory_length,
                **cs,
            }
            if off_metrics:
                log["off_loss"] = round(off_metrics["total_loss"], 6)
                log["off_grad_norm"] = round(off_metrics["grad_norm"], 6)
                log["off_seq_len"] = off_metrics["sequence_length"]
            all_logs.append(log)

            print(f"  up {up:3d}/{NUM_UPDATES_PER_SESSION}  "
                  f"steps {total_env_steps:6d}/{SESSION_ENV_STEPS}  "
                  f"replay {len(replay):3d}  "
                  f"samples {cs['replay_samples_drawn']:4d}  "
                  f"grads {cs['gradient_updates']:4d}  "
                  f"relab {cs['relabelled_samples']:3d}  "
                  f"max_len={replay.longest_trajectory_length}")

    except Exception as e:
        print(f"\nSESSION STOP: {e}")
        traceback.print_exc()
        crash_info = {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "session_env_steps": total_env_steps,
            "session": session_index,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    session_elapsed = round(time.time() - session_t0, 1)
    new_global_step = global_step + total_env_steps

    return {
        "ts": ts,
        "replay": replay,
        "rng": rng,
        "global_step": new_global_step,
        "session_env_steps": total_env_steps,
        "session_elapsed_s": session_elapsed,
        "session_log": all_logs,
        "pass_trajs_long": pass_trajs_long,
        "pass_samples_long": pass_samples_long,
        "pass_grad_updates": pass_grad_updates,
        "pass_relabel_evidence": pass_relabel_evidence,
        "crash_info": crash_info,
    }


# ═══════════════════════════════════════════════════════════════════════
# Checkpoint saving
# ═══════════════════════════════════════════════════════════════════════

def save_stage4_checkpoint(
    ts: TrainState,
    replay: TrajectoryReplayBuffer,
    rng: jax.random.PRNGKey,
    global_step: int,
    cfg: type,
    session_index: int,
    out_dir: str,
    resume_step: int,
) -> str:
    """Save a full Stage4 checkpoint with resolved config and provenance."""
    ckpt_path = save_full_checkpoint(
        ts, replay, rng, global_step, CKPT_ROOT, step=global_step)

    # Compute source hashes (same as D059 convention)
    src_hashes = {}
    for fn in sorted(os.listdir(_AMAGO_SRC)):
        if fn.endswith(".py"):
            with open(os.path.join(_AMAGO_SRC, fn), "rb") as fh:
                src_hashes[fn] = hashlib.sha256(fh.read()).hexdigest()

    # Extended manifest with Stage4 identity
    manifest = {
        "directive": "D059",
        "stage": "4",
        "stage4_session": session_index,
        "treatment": "AMAGO_STYLE_EXPLORATORY_P2",
        "experiment": "A",
        "resume_from_step": resume_step,
        "checkpoint_step": global_step,
        "global_step": global_step,
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "anneal_lr": False,
        "optimizer_restarted_at_resume": True,
        "optimizer_note": (
            "Fresh Adam(lr=2e-4, eps=1e-5) created at resume (step 98304). "
            "Params carried forward; Adam moments restarted. "
            "Prior gradient updates: 47."
        ),
        "gpu_uuid": EXPECTED_GPU_UUID,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "counters": replay.counters.snapshot(),
        "replay_buffer_size": len(replay),
        "replay_longest_trajectory": replay.longest_trajectory_length,
        "resolved_config": _cfg_resolved_dict(cfg),
        "source_hashes": src_hashes,
        "checkpoint_path": ckpt_path,
        "session_output_dir": out_dir,
    }

    manifest_path = os.path.join(
        CKPT_ROOT, str(global_step), "stage4_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"  [checkpoint] saved at step={global_step}  →  {ckpt_path}")
    print(f"  [checkpoint] manifest  →  {manifest_path}")
    return ckpt_path


# ═══════════════════════════════════════════════════════════════════════
# Smoke test: verify restore + forward pass + checkpoint round-trip
# ═══════════════════════════════════════════════════════════════════════

def smoke_test(resume_step: int):
    """Verify restore integrity without running full training.

    Checks:
      1. Network init with correct shapes (batch_size=2, no method=)
      2. Build matching template (anneal_lr=True) for orbax deserialization
      3. Orbax restore TrainState from specified checkpoint
      4. Rebuild TrainState with anneal_lr=False (fresh optimizer)
      5. Restore replay buffer, RNG, global_step from replay_meta.pkl
      6. Forward pass succeeds (batch_size=16, matches training)
      7. Full checkpoint round-trip (save → restore → verify bit-exact)
    """
    print("=" * 60)
    print(f"SMOKE TEST: Stage4 Continuation Launcher  (resume from step={resume_step})")
    print("=" * 60)

    cfg = Cfg()

    # 1. Build dummy env for network init
    print("\n[smoke 1/7] Building dummy env ...")
    ach_table = jnp.array(
        [get_achievement_multi_hot([Achievement.DEFEAT_KOBOLD])], dtype=jnp.float32)
    EMB = int(ach_table.shape[1])
    dummy = CraftaxAugObsTrain(
        condition_on_task=True, conditioning_type="embedding",
        embedding_size=EMB, task_embeddings=jnp.zeros((1, EMB)))
    obs_dim = dummy.observation_space(dummy.default_params).shape[0]
    action_dim = dummy.action_space(dummy.default_params).n
    print(f"  obs_dim={obs_dim}  action_dim={action_dim}  embedding_dim={EMB}")

    # 2. Build network
    print("\n[smoke 2/7] Building network ...")
    network = ActorCriticTransformer(
        action_dim=action_dim,
        activation=cfg.activation,
        hidden_layers=cfg.hidden_layers,
        encoder_size=cfg.embed_size,
        num_heads=cfg.num_heads,
        qkv_features=cfg.qkv_features,
        num_layers=cfg.num_layers,
        gating=cfg.gating,
        gating_bias=cfg.gating_bias)

    # 3. Init network params (verifies shape correctness)
    print("\n[smoke 3/7] Initializing network params (batch_size=2, no method=) ...")
    init_rng = jax.random.PRNGKey(0)
    fresh_params = init_network_params(network, obs_dim, cfg, init_rng)
    n_leaves = len(jax.tree_util.tree_leaves(fresh_params))
    n_params = sum(
        int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(fresh_params)
        if hasattr(x, 'shape'))
    print(f"  ✓ Network init OK  param_leaves={n_leaves}  total_params={n_params:,}")

    # 4. Restore from checkpoint 98304
    print("\n[smoke 4/7] Restoring from checkpoints/98304 ...")
    restored = restore_from_checkpoint(CKPT_PARENT, resume_step,
                                       network, cfg, obs_dim)
    ts = restored["train_state"]
    replay = restored["replay_buffer"]
    rng = restored["rng"]
    gs = restored["global_step"]

    assert gs == resume_step, f"Expected global_step={resume_step}, got {gs}"
    # Replay size depends on resume point; just verify non-empty
    assert len(replay) > 0, f"Replay buffer is empty"
    print("  ✓ TrainState rebuilt (params RESTORED, optimizer FRESH anneal_lr=False)")
    print(f"  ✓ Replay buffer: {len(replay)} trajectories"
          f" (max_len={replay.longest_trajectory_length})")
    print(f"  ✓ RNG key: {rng}")
    print(f"  ✓ Global step: {gs}")

    # 5. Forward pass (batch_size=16, matches training — batch_size=1 breaks
    #    the transformer's memory concatenation, known limitation)
    print("\n[smoke 5/7] Forward pass (batch_size=16) ...")
    test_mem = jnp.zeros((16, cfg.window_mem, cfg.num_layers, cfg.embed_size))
    test_obs = jnp.zeros((16, obs_dim))
    test_mask = jnp.zeros((16, cfg.num_heads, 1, cfg.window_mem + 1), dtype=jnp.bool_)

    pi, value, mem_out = network.apply(
        ts.params, test_mem, test_obs, test_mask,
        method=network.model_forward_eval)
    print(f"  ✓ Forward pass OK  logits_shape={pi.logits.shape}  "
          f"value_shape={value.shape}  mem_out_shape={mem_out.shape}")

    # Verify params differ from fresh init (proving restore worked)
    params_differ = not all(
        bool(jnp.all(l1 == l2))
        for l1, l2 in zip(
            jax.tree_util.tree_leaves(fresh_params),
            jax.tree_util.tree_leaves(ts.params)))
    assert params_differ, "FAIL: restored params are identical to fresh init!"
    print(f"  ✓ Params differ from fresh init (restore VERIFIED)")

    # 6. Checkpoint round-trip
    print("\n[smoke 6/7] Checkpoint round-trip ...")
    smoke_ckpt_dir = os.path.join(CKPT_ROOT, "_smoke_test_roundtrip")
    ckpt_path = save_full_checkpoint(ts, replay, rng, gs, smoke_ckpt_dir, step=gs)

    # Restore from the round-trip checkpoint
    restored2 = restore_from_checkpoint(smoke_ckpt_dir, gs, network, cfg, obs_dim)
    ts2 = restored2["train_state"]
    replay2 = restored2["replay_buffer"]
    rng2 = restored2["rng"]
    gs2 = restored2["global_step"]

    assert gs2 == gs, f"Round-trip global_step mismatch: {gs2} != {gs}"
    assert len(replay2) == len(replay), \
        f"Round-trip replay size mismatch: {len(replay2)} != {len(replay)}"

    # Verify params match (bit-exact round-trip)
    params_equal = all(
        bool(jnp.all(l1 == l2))
        for l1, l2 in zip(
            jax.tree_util.tree_leaves(ts.params),
            jax.tree_util.tree_leaves(ts2.params)))
    assert params_equal, "Round-trip params mismatch!"

    print(f"  ✓ Round-trip save/restore OK")
    print(f"  ✓ Params bit-exact after round-trip")
    print(f"  ✓ Replay buffer ({len(replay2)} trajs) preserved")

    # 7. Cleanup
    print("\n[smoke 7/7] Cleanup ...")
    shutil.rmtree(smoke_ckpt_dir, ignore_errors=True)
    print(f"  ✓ Smoke checkpoint cleaned up")

    # Summary
    print(f"\n{'='*60}")
    print("SMOKE TEST: ALL CHECKS PASSED ✓")
    print(f"  TrainState:     params RESTORED + optimizer FRESH (anneal_lr=False)")
    print(f"  Replay:         {len(replay)} trajectories, max_len={replay.longest_trajectory_length}")
    print(f"  RNG:            {rng}")
    print(f"  Global step:    {gs}")
    print(f"  anneal_lr:      False (Stage4 fixed)")
    print(f"  gamma:          {GAMMA}")
    print(f"  lambda:         {GAE_LAMBDA}")
    print(f"  GPU:            {EXPECTED_GPU_UUID}")
    print(f"{'='*60}")
    return True


# ═══════════════════════════════════════════════════════════════════════
# Main: multi-session training loop
# ═══════════════════════════════════════════════════════════════════════

def main(max_sessions: Optional[int] = None, resume_step: Optional[int] = None):
    if resume_step is None:
        resume_step = detect_latest_checkpoint_step(CKPT_PARENT)
    print("=" * 60)
    print("D059 Stage4 Continuation — Experiment A")
    print(f"  Resume from:   {CKPT_PARENT}  step={resume_step}")
    print(f"  Session size:  {SESSION_ENV_STEPS} env steps "
          f"({NUM_ENVS}×{ROLLOUT_STEPS}×{NUM_UPDATES_PER_SESSION})")
    print(f"  GPU:           {EXPECTED_GPU_UUID}")
    print(f"  gamma:         {GAMMA}  lambda: {GAE_LAMBDA}")
    print(f"  anneal_lr:     False (Stage4 — NO auto-annealing)")
    print(f"  max_sessions:  {max_sessions if max_sessions else 'unlimited (Ctrl+C to stop)'}")
    print("=" * 60)

    # ── Guards ────────────────────────────────────────────────────
    verify_gpu()
    session_dir = guard_output_collision(
        os.path.join(OUTPUT_ROOT, "stage4_continue"))
    print(f"\n[output] Session directory: {session_dir}")

    cfg = Cfg()

    # ── Build dummy env for network init ──────────────────────────
    print("\n[1/5] Building dummy env for network init ...")
    ach_table = jnp.array(
        [get_achievement_multi_hot([Achievement.DEFEAT_KOBOLD])], dtype=jnp.float32)
    EMB = int(ach_table.shape[1])
    dummy = CraftaxAugObsTrain(
        condition_on_task=True, conditioning_type="embedding",
        embedding_size=EMB, task_embeddings=jnp.zeros((1, EMB)))
    obs_dim = dummy.observation_space(dummy.default_params).shape[0]
    action_dim = dummy.action_space(dummy.default_params).n
    print(f"  obs_dim={obs_dim}  action_dim={action_dim}")

    # ── Build network ─────────────────────────────────────────────
    print("\n[2/5] Building network ...")
    network = ActorCriticTransformer(
        action_dim=action_dim,
        activation=cfg.activation,
        hidden_layers=cfg.hidden_layers,
        encoder_size=cfg.embed_size,
        num_heads=cfg.num_heads,
        qkv_features=cfg.qkv_features,
        num_layers=cfg.num_layers,
        gating=cfg.gating,
        gating_bias=cfg.gating_bias)

    # ── Restore full checkpoint ───────────────────────────────────
    print(f"\n[3/5] Restoring from {CKPT_PARENT} step={resume_step} ...")
    restored = restore_from_checkpoint(
        CKPT_PARENT, resume_step, network, cfg, obs_dim)
    ts = restored["train_state"]
    replay = restored["replay_buffer"]
    rng = restored["rng"]
    global_step = restored["global_step"]

    if restored.get("optimizer_restarted"):
        print(f"  ⚠ Optimizer restarted: {restored['optimizer_note']}")

    # ── Build learner ─────────────────────────────────────────────
    print("\n[4/5] Building learner ...")
    learner = LongContextLearner(network, cfg, rng)
    print("  learner ready")

    # ── Build env from S4 task code ───────────────────────────────
    print("\n[5/5] Building Stage4 env ...")
    with open(S4_TASK_PATH) as f:
        s4_code = f.read()
    ns = {}
    exec(s4_code, ns)
    Task = ns["Env"]

    static_env_params = StaticEnvParams()
    env_params = EnvParams(max_timesteps=4096)
    base_env = MultiTaskMiniCraftaxEnv(
        [Task], static_env_params, env_params, cfg.condition_on_task,
        conditioning_type="embedding", embedding_size=EMB,
        completion_bonus_scale=cfg.completion_bonus_scale,
        completion_bonus_min=cfg.completion_bonus_min,
        bonus_type=cfg.bonus_type, dynamic_bonus_k=cfg.dynamic_bonus_k)
    env = DistributedMultiTaskOptimisticLogWrapper(
        base_env, jax.random.PRNGKey(0), NUM_ENVS, 1,
        cfg.optimistic_reset_ratio, jnp.array([1.0]), ach_table)
    print("  env built (Stage4: floor2, 8/8 kills pre-credited, DEFEAT_KOBOLD)")

    # ── Source hashes (baseline) ──────────────────────────────────
    src_hashes = {}
    for fn in sorted(os.listdir(_AMAGO_SRC)):
        if fn.endswith(".py"):
            with open(os.path.join(_AMAGO_SRC, fn), "rb") as fh:
                src_hashes[fn] = hashlib.sha256(fh.read()).hexdigest()

    # ── Training loop ─────────────────────────────────────────────
    session_index = 0
    all_session_manifests = []
    t0 = time.time()

    try:
        while max_sessions is None or session_index < max_sessions:
            session_index += 1

            result = run_session(
                ts=ts,
                replay=replay,
                rng=rng,
                global_step=global_step,
                session_index=session_index,
                cfg=cfg,
                network=network,
                learner=learner,
                env=env,
                env_params=env_params,
                ach_table=ach_table,
                out_dir=session_dir,
            )

            # Unpack result
            ts = result["ts"]
            replay = result["replay"]
            rng = result["rng"]
            global_step = result["global_step"]
            session_env_steps = result["session_env_steps"]
            session_elapsed = result["session_elapsed_s"]
            crash_info = result["crash_info"]

            # Save full checkpoint after every session
            ckpt_path = save_stage4_checkpoint(
                ts, replay, rng, global_step, cfg,
                session_index, session_dir, resume_step)

            # Session manifest
            session_manifest = {
                "directive": "D059",
                "stage": "4",
                "experiment": "A",
                "treatment": "AMAGO_STYLE_EXPLORATORY_P2",
                "resume_from_step": resume_step,
                "session_index": session_index,
                "global_step": global_step,
                "session_env_steps": session_env_steps,
                "session_elapsed_s": session_elapsed,
                "total_elapsed_s": round(time.time() - t0, 1),
                "gamma": GAMMA,
                "gae_lambda": GAE_LAMBDA,
                "anneal_lr": False,
                "optimizer_restarted_at_resume": True,
                "gpu_uuid": EXPECTED_GPU_UUID,
                "checkpoint_path": ckpt_path,
                "counters": replay.counters.snapshot(),
                "replay_buffer_size": len(replay),
                "replay_longest_trajectory": replay.longest_trajectory_length,
                "pass_trajs_long": result["pass_trajs_long"],
                "pass_samples_long": result["pass_samples_long"],
                "pass_grad_updates": result["pass_grad_updates"],
                "pass_relabel_count": len(result["pass_relabel_evidence"]),
                "crash_info": crash_info,
                "resolved_config": _cfg_resolved_dict(cfg),
                "source_hashes": src_hashes,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            all_session_manifests.append(session_manifest)

            # Write cumulative manifest
            with open(os.path.join(session_dir, "stage4_manifest.json"), "w") as f:
                json.dump(all_session_manifests, f, indent=2, sort_keys=True)

            # Write session log
            with open(os.path.join(session_dir, "training_log.jsonl"), "a") as f:
                for entry in result["session_log"]:
                    f.write(json.dumps(entry) + "\n")

            # Write source hashes
            with open(os.path.join(session_dir, "source_hashes.json"), "w") as f:
                json.dump(src_hashes, f, indent=2, sort_keys=True)

            print(f"\n── Session {session_index} complete: "
                  f"{session_env_steps} steps in {session_elapsed}s, "
                  f"ckpt at step {global_step}")

            if crash_info:
                print(f"  WARNING: session crashed — {crash_info['error']}")
                with open(os.path.join(
                        session_dir,
                        f"crash_session_{session_index}.json"), "w") as f:
                    json.dump(crash_info, f, indent=2)

    except KeyboardInterrupt:
        print(f"\n\nInterrupted after session {session_index - 1}.  "
              f"Final checkpoint at step {global_step}.")
        # Save emergency checkpoint
        try:
            save_stage4_checkpoint(
                ts, replay, rng, global_step, cfg,
                session_index, session_dir)
        except Exception:
            pass

    total_elapsed = round(time.time() - t0, 1)
    print(f"\n{'='*60}")
    print(f"Stage4 Continuation: {session_index} sessions completed")
    print(f"  Final global_step: {global_step}")
    print(f"  Total env steps:   {global_step - resume_step}")
    print(f"  Total elapsed:     {total_elapsed}s")
    print(f"  Output:            {session_dir}")
    print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="D059 Stage4 Continuation Launcher (Experiment A)")
    parser.add_argument(
        "--max-sessions", type=int, default=None,
        help="Maximum number of sessions (default: unlimited)")
    parser.add_argument(
        "--resume-from", type=int, default=None,
        help="Resume from a specific checkpoint step "
             "(default: auto-detect latest in checkpoints/)")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run smoke test only (verify restore integrity, no training)")
    args = parser.parse_args()

    # Resolve resume step
    if args.resume_from is not None:
        resume_step = args.resume_from
    else:
        resume_step = detect_latest_checkpoint_step(CKPT_PARENT)

    if args.smoke_test:
        success = smoke_test(resume_step)
        sys.exit(0 if success else 1)
    else:
        main(max_sessions=args.max_sessions, resume_step=resume_step)
