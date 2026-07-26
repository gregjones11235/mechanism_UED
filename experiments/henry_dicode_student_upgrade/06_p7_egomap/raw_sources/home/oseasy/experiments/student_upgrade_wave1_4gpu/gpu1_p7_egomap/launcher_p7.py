#!/usr/bin/env python3
"""P7-EGOMAP-WAVE1 launcher: dual Control/EgoMap training from ckpt17500.

Config: num_envs=16, num_steps=128, lr=2e-5, gamma=0.999, gae_lambda=0.8,
        total_steps=98304 (48 updates), window_mem=128, window_grad=64,
        embed_size=256, num_heads=8, num_layers=2, hidden_layers=256.

Checkpoint nodes: 0/4096/24576/49152/73728/98304 env steps (0/2/12/24/36/48 updates).

Usage:
  CUDA_VISIBLE_DEVICES=1 python launcher_p7.py [--control-only | --egomap-only]
"""
import os, sys, json, time, hashlib, argparse
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp, optax, orbax.checkpoint as ocp
from flax.training.train_state import TrainState

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src")

from minicraftax.tasks.seed_tasks.original import Env as OriginalTask
from craftax.craftax.constants import Achievement
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from minicraftax.envs.craftax import CraftaxAugObsTrain
from dicode.task_utils import get_achievement_multi_hot
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv
from dicode.wrappers_cl import DistributedMultiTaskOptimisticLogWrapper
import ppo_tr_egomap as PE

# ─── Constants ───────────────────────────────────────────────────────────────
GPU1_UUID = "GPU-3c7a2864-755b-7045-b293-6f80e748283f"
CKPT_PARENT = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/base_ckpt_v7fix55_armA_s0/rl_checkpoints"
CKPT_STEP = 17500
OUT_ROOT = "/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu1_p7_egomap"
NUM_ENVS = 16
ROLLOUT_STEPS = 128
TOTAL_STEPS = 98304
UPDATES_PER_STEP = NUM_ENVS * ROLLOUT_STEPS  # 2048
TOTAL_UPDATES = TOTAL_STEPS // UPDATES_PER_STEP  # 48
# Checkpoint nodes in env steps and updates
CKPT_NODES_STEPS = [0, 4096, 24576, 49152, 73728, 98304]
CKPT_NODES_UPDATES = [s // UPDATES_PER_STEP for s in CKPT_NODES_STEPS]  # [0,2,12,24,36,48]
SEED = 42
EMB = 76  # achievement multi-hot dim (8335 - 8217 - 42 = 76)


class Cfg:
    """P7 config: lr=2e-5 (spec), gamma=0.999, gae_lambda=0.8, no annealing."""
    lr = 2e-5
    min_lr = 2e-6
    num_envs = NUM_ENVS
    num_steps = ROLLOUT_STEPS
    update_epochs = 4
    num_minibatches = 8
    gamma = 0.999
    gae_lambda = 0.8
    clip_eps = 0.2
    ent_coef = 0.002
    vf_coef = 0.5
    max_grad_norm = 1.0
    activation = "relu"
    anneal_lr = False
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
    mode = "task"
    bonus_type = "dynamic"
    dynamic_bonus_k = 2.0
    completion_bonus_scale = 2.0
    completion_bonus_min = 20.0
    max_updates_per_session = 100
    total_timesteps = 2_005_401_600
    scoring_window_updates = 40
    value_target_clip_min = -50.0
    value_target_clip_max = 300.0
    sil = False
    sil_pools = []
    debug = False
    use_wandb = False
    # EgoMap (set per-arm)
    egomap_enabled = True
    egomap_map_size = 32
    egomap_num_floors = 9
    egomap_cnn_features = (16, 32)


def verify_gpu():
    """Verify GPU1 is present and idle."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader"],
            text=True
        ).strip().split("\n")
    except Exception:
        print("STOP: cannot query nvidia-smi"); sys.exit(1)
    uuids = [l.split(",")[0].strip() for l in out]
    if GPU1_UUID not in uuids:
        print(f"STOP: GPU1 {GPU1_UUID} not found. Available: {uuids}"); sys.exit(1)
    idx = uuids.index(GPU1_UUID)
    # Note: skip memory check — JAX pre-allocates GPU memory on import,
    # so nvidia-smi will show high usage even when GPU is idle.
    # The CUDA_VISIBLE_DEVICES is set in the shell env before JAX imports.
    print(f"[guard] GPU1 OK | UUID={GPU1_UUID} index={idx}")


def load_base_params(rng):
    """Load ckpt17500 base-subset params via orbax (weights-only)."""
    from dicode.network import ActorCriticTransformer
    base_net = ActorCriticTransformer(
        action_dim=43, activation="relu", hidden_layers=256, encoder_size=256,
        num_heads=8, qkv_features=256, num_layers=2, gating=True, gating_bias=2.0)
    # Init with correct shapes (matching load_weights_only internals)
    rng, _rng = jax.random.split(rng)
    init_obs = jnp.zeros((2, 8335))
    init_memory = jnp.zeros((2, 128, 2, 256))
    init_mask = jnp.zeros((2, 8, 1, 129), dtype=jnp.bool_)
    base_params = base_net.init(_rng, init_memory, init_obs, init_mask)  # full init dict (matches ckpt structure)
    # Restore from ckpt17500 (weights-only: extract params from restored TrainState)
    checkpointer = ocp.PyTreeCheckpointer()
    options = ocp.CheckpointManagerOptions(create=False)
    ckpt_manager = ocp.CheckpointManager(CKPT_PARENT, checkpointer, options=options)
    # Build a template TrainState matching ckpt17500 structure (anneal_lr=True)
    TOTAL_GLOBAL_UPDATES = ((Cfg.total_timesteps // Cfg.num_envs // Cfg.num_steps // Cfg.max_updates_per_session) + 1) * Cfg.max_updates_per_session
    def linear_schedule(count):
        u = count // (Cfg.num_minibatches * Cfg.update_epochs)
        frac = jnp.maximum(1.0 - u / TOTAL_GLOBAL_UPDATES, 0.0)
        return Cfg.min_lr + (2e-4 - Cfg.min_lr) * frac  # base used lr=2e-4
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=linear_schedule, eps=1e-5))
    template = TrainState.create(apply_fn=base_net.apply, params=base_params, tx=tx)
    restored = ckpt_manager.restore(CKPT_STEP, items=template)
    print(f"  [load] ckpt17500 restored | param_leaves={len(jax.tree_util.tree_leaves(restored.params))}")
    return restored.params


def merge_ego_params(base_params, egomap_params):
    """Merge base-subset params from ckpt17500 with zero-init ego_encoder/ego_gate.

    base_params: full init dict {"params": {...}} from restored TrainState.
    egomap_params: flat params dict {..., "ego_encoder": ..., "ego_gate": ...} from init_egomap_network.
    Returns: flat params dict for the EgoMap network.
    """
    merged = dict(base_params["params"])  # extract flat params from restored TrainState
    merged["ego_encoder"] = egomap_params["ego_encoder"]
    merged["ego_gate"] = egomap_params["ego_gate"]
    return merged


def init_egomap_network(rng, egomap_enabled):
    """Init the EgoMap network and return (network, params)."""
    network = PE.ActorCriticTransformerEgoMap(
        action_dim=43, activation="relu", hidden_layers=256, encoder_size=256,
        num_heads=8, qkv_features=256, num_layers=2, gating=True, gating_bias=2.0,
        egomap_channels=PE.egomap_lib.N_MAP_CH, egomap_cnn_features=(16, 32))
    rng, _rng = jax.random.split(rng)
    init_obs = jnp.zeros((2, 8335))
    init_memory = jnp.zeros((2, 128, 2, 256))
    init_mask = jnp.zeros((2, 8, 1, 129), dtype=jnp.bool_)
    init_ego = jnp.zeros((2, 32, 32, PE.egomap_lib.N_MAP_CH))
    full_params = network.init(_rng, init_memory, init_obs, init_mask,
                               ego_features=init_ego, egomap_enabled=True)["params"]
    return network, full_params


def build_train_state(network, params):
    """Build fresh TrainState with lr=2e-5, no annealing.

    ppo_tr_egomap.py calls network.apply(train_state.params, ...) directly,
    so TrainState.params must be the full init dict {"params": {...}}.
    """
    tx = optax.chain(optax.clip_by_global_norm(Cfg.max_grad_norm),
                     optax.adam(Cfg.lr, eps=1e-5))
    return TrainState.create(apply_fn=network.apply, params={"params": params}, tx=tx)


def build_env(rng):
    """Build the Craftax env with OriginalTask and achievement multi-hot embedding."""
    ach_table = jnp.array([get_achievement_multi_hot([Achievement.DEFEAT_KOBOLD])], dtype=jnp.float32)
    # ach_table is (1, 67) — used by wrapper for achievement tracking only.
    # EMB=76 is the obs embedding dim (67 achievements + 9 padding); embedding_size=EMB below.
    task_embeddings = jnp.zeros((1, EMB))  # zeros (proven P2 approach; A/B fair)
    static_env_params = StaticEnvParams()
    env_params = EnvParams(max_timesteps=4096)
    base_env = MultiTaskMiniCraftaxEnv(
        [OriginalTask], static_env_params, env_params, Cfg.condition_on_task,
        conditioning_type="embedding", embedding_size=EMB,
        completion_bonus_scale=Cfg.completion_bonus_scale, completion_bonus_min=Cfg.completion_bonus_min,
        bonus_type=Cfg.bonus_type, dynamic_bonus_k=Cfg.dynamic_bonus_k)
    env = DistributedMultiTaskOptimisticLogWrapper(
        base_env, rng, NUM_ENVS, 1, Cfg.optimistic_reset_ratio, jnp.array([1.0]), ach_table)
    return env, ach_table


def save_carry(carry, step, out_dir):
    """Save the full recurrent carry (egomap_state + memories + env_state + rng) to disk."""
    import pickle
    path = os.path.join(out_dir, f"carry_{step}.pkl")
    with open(path, "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.asarray, carry), f)
    print(f"  [ckpt] carry saved → {path}")
    return path


def save_params(params, step, out_dir):
    """Save params as numpy dict."""
    import pickle
    path = os.path.join(out_dir, f"params_{step}.pkl")
    with open(path, "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.asarray, params), f)
    print(f"  [ckpt] params saved → {path}")
    return path


def train_arm(arm_name, egomap_enabled, rng):
    """Train one arm (Control or EgoMap) from ckpt17500 for 98304 steps."""
    out_dir = os.path.join(OUT_ROOT, "outputs", arm_name)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(OUT_ROOT, "checkpoints", arm_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\n{'='*60}\n  ARM: {arm_name}  (egomap_enabled={egomap_enabled})\n{'='*60}")

    # 1. Load base params + init ego params
    print("[1/5] Loading ckpt17500 base params ...")
    rng, _rng = jax.random.split(rng)
    base_params = load_base_params(_rng)

    print("[2/5] Initializing EgoMap network ...")
    rng, _rng = jax.random.split(rng)
    network, full_params = init_egomap_network(_rng, egomap_enabled)
    # Merge: base-subset from ckpt17500, ego_encoder/ego_gate from fresh init (zeros)
    merged_params = merge_ego_params(base_params, full_params)
    ts = build_train_state(network, merged_params)
    print(f"  param_leaves={len(jax.tree_util.tree_leaves(ts.params))}")

    # 2. Build env
    print("[3/5] Building env ...")
    rng, _rng = jax.random.split(rng)
    env, ach_table = build_env(_rng)

    # 3. Save step-0 checkpoint (initial state)
    print("[4/5] Saving step-0 checkpoint ...")
    save_params(ts.params["params"], 0, ckpt_dir)

    # 4. Segmented training
    print(f"[5/5] Training {TOTAL_UPDATES} updates in segments ...")
    # Pass ach_table (1,67) as the wrapper's embedding arg — matches ckpt17500's
    # obs_dim=8335.  Passing zeros(1,76) would make the wrapper produce obs_dim=8344.
    cfg = Cfg()
    cfg.egomap_enabled = egomap_enabled

    # Segment boundaries (in updates): [0→2, 2→12, 12→24, 24→36, 36→48]
    segments = list(zip(CKPT_NODES_UPDATES[:-1], CKPT_NODES_UPDATES[1:]))
    current_ts = ts
    current_carry = None
    rng_train = jax.random.PRNGKey(SEED)

    for seg_idx, (start_u, end_u) in enumerate(segments):
        seg_updates = end_u - start_u
        seg_steps = seg_updates * UPDATES_PER_STEP
        print(f"\n  Segment {seg_idx+1}/{len(segments)}: updates {start_u}→{end_u} ({seg_steps} env steps)")

        # Build train_fn for this segment
        train_fn = PE.make_train(cfg, [OriginalTask], seg_updates, ach_table, None, start_u)
        train_jit = jax.jit(train_fn)

        t0 = time.time()
        if current_carry is None:
            # Cold start (first segment)
            out = train_jit(rng_train, train_state=current_ts)
        else:
            # Resume from carry
            out = train_jit(current_carry["rng"], train_state=current_ts,
                           current_original_return=0.0, resume_carry=current_carry)
        elapsed = time.time() - t0
        print(f"    done in {elapsed:.1f}s")

        current_ts = out["train_state"]
        current_carry = out["carry"]

        # Save checkpoint at end of segment
        end_step = end_u * UPDATES_PER_STEP
        save_params(current_ts.params["params"], end_step, ckpt_dir)
        save_carry(current_carry, end_step, ckpt_dir)

    print(f"\n  ARM {arm_name} COMPLETE | final_step={TOTAL_STEPS}")
    return current_ts, current_carry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-only", action="store_true")
    parser.add_argument("--egomap-only", action="store_true")
    args = parser.parse_args()

    # ppo_tr_egomap._log_callback unconditionally calls wandb.log();
    # even with WANDB_MODE=disabled, wandb.init() must be called first.
    import wandb
    wandb.init(mode="disabled", project="p7-egomap-wave1")

    print("P7-EGOMAP-WAVE1 Launcher")
    print(f"  total_steps={TOTAL_STEPS}  num_envs={NUM_ENVS}  rollout={ROLLOUT_STEPS}")
    print(f"  lr={Cfg.lr}  gamma={Cfg.gamma}  gae_lambda={Cfg.gae_lambda}")
    print(f"  ckpt_nodes={CKPT_NODES_STEPS}")

    verify_gpu()

    rng = jax.random.PRNGKey(SEED)

    if not args.egomap_only:
        rng, _rng = jax.random.split(rng)
        train_arm("control", egomap_enabled=False, rng=_rng)

    if not args.control_only:
        rng, _rng = jax.random.split(rng)
        train_arm("egomap", egomap_enabled=True, rng=_rng)

    print("\n" + "="*60)
    print("ALL ARMS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
