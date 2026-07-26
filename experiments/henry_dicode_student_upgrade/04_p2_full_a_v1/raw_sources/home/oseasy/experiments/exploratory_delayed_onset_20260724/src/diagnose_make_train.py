#!/usr/bin/env python3
"""DECISIVE DIAGNOSIS: run Henry's OWN run_training_session (the EXACT code path that
produced the frozen ece6fa99 Control anchor in run_control_kl_telemetry.py) for 12 updates
from ckpt17500, and compare the resulting params SHA to ece6fa99.

  - If it REPRODUCES ece6fa99  -> the environment/library is unchanged and the frozen anchor
    is still reproducible; therefore the continuous-retrain launcher's chunked reimplementation
    has a HIDDEN divergence from make_train that must be found and fixed.
  - If it does NOT reproduce    -> the environment/library has DRIFTED since the telemetry run
    and the ece6fa99 anchor is no longer reproducible by ANY code path; the anchor assumption
    of plan-1 must be re-evaluated (this is not a launcher bug).

Setup mirrors run_control_kl_telemetry.py lines 210-257 EXACTLY (cfg + cfg.training=cfg +
ach_table + S4 Task + base_env + load_weights_only(load_opt_state=False)). GPU0 + det-ops.
This script ONLY runs training (no probe, no telemetry patches) and prints the SHA verdict.
"""
import os
GPU_UUID = "GPU-e8c08612-c22a-c8a4-6df5-affb2dd1f9a6"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"
import sys, hashlib

V7_SRC = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src"
if V7_SRC not in sys.path:
    sys.path.insert(0, V7_SRC)

import wandb
wandb.init(mode="disabled")

import numpy as np
import jax
import jax.numpy as jnp

from craftax.craftax.constants import Achievement
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from dicode.task_utils import get_achievement_multi_hot
from dicode.utils.general.train_state_utils import load_weights_only
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv
from dicode.ppo_tr import run_training_session

SESSION175_CKPT = ("/home/oseasy/incoming/henry_work_20260721T105300/extracted/"
                   "Henry_work/base_ckpt_v7fix55_armA_s0/rl_checkpoints/17500")
S4_TASK_PATH = "/home/oseasy/experiments/p2_v1_20260722/evidence/s4_task_code.py"
ECE6FA99 = "ece6fa9962e815123ce947577a93040057bc9df0b1e686dd28424cb2bbdabf55"
SOURCE_SHA = "d4e85af58b7f87d689fadea12eec70c852fa098a09f5ea8907448684b3bf60f5"
NUM_UPDATES = 12
ENTROPY_COLLAPSE_FLOOR = 0.05


def _params_sha(params):
    h = hashlib.sha256()
    for v in jax.tree_util.tree_leaves(params):
        h.update(np.ascontiguousarray(np.asarray(v)).tobytes())
    return h.hexdigest()


# ---- Cfg: VERBATIM run_control_kl_telemetry.py lines 119-170 ----
class Cfg:
    lr = 2e-5
    min_lr = 2e-6
    num_envs = 16
    num_steps = 128
    update_epochs = 1
    num_minibatches = 2
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
    mode = "score"
    bonus_type = "none"
    dynamic_bonus_k = 0.0
    completion_bonus_scale = 0.0
    completion_bonus_min = 0.0
    max_updates_per_session = 12
    total_timesteps = 2_005_401_600
    scoring_window_updates = 4
    value_target_clip_min = -50.0
    value_target_clip_max = 300.0
    guard_session_vloss_max = 1000.0
    guard_session_entropy_min = ENTROPY_COLLAPSE_FLOOR
    guard_max_consecutive_reverts = 2
    lr_restart = 0.0
    lr_restart_at = 0
    lr_restart_horizon = 0
    lr_restart_warmup = 50
    sil = False
    sil_pools = []
    use_wandb = False
    debug = False
    validation = None
    dicode_manager = None

    def get(self, key, default=None):
        return getattr(self, key, default)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_updates", type=int, default=NUM_UPDATES,
                    help="set to 1 to get make_train's 1-update params SHA for diffing")
    a = ap.parse_args()
    num_updates = a.num_updates

    cfg = Cfg()
    cfg.lr = 2e-5
    cfg.max_updates_per_session = num_updates
    cfg.training = cfg                      # run_training_session reads config.training

    devs = jax.local_devices()
    print(f"[diagnose] devices: {devs}  (expect GPU0 only)", flush=True)
    assert len(devs) == 1, f"expected exactly 1 visible device, got {devs}"

    # ---- S4_dark env (VERBATIM telemetry lines 219-235) ----
    ach_table = jnp.array([get_achievement_multi_hot([Achievement.DEFEAT_KOBOLD])],
                          dtype=jnp.float32)
    EMB = int(ach_table.shape[1])
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
    obs_dim = int(base_env.observation_space(env_params).shape[0])
    print(f"[diagnose] S4_dark env: obs_dim={obs_dim} emb={EMB}", flush=True)

    # ---- load ckpt17500 (VERBATIM telemetry lines 241-248) ----
    ts = load_weights_only(SESSION175_CKPT, base_env, env_params, cfg, load_opt_state=False)
    source_sha = _params_sha(ts.params)
    assert source_sha == SOURCE_SHA, \
        f"REFUSED: source sha {source_sha} != expected {SOURCE_SHA}"
    assert int(ts.step) == 0
    print(f"[diagnose] ckpt17500 loaded: source_sha={source_sha} opt_step={int(ts.step)}",
          flush=True)

    # ---- run Henry's OWN training path for 12 updates (EXACT telemetry call) ----
    print(f"[diagnose] running run_training_session(num_updates={num_updates}, rng=PRNGKey(42)) ...",
          flush=True)
    result = run_training_session(
        cfg, jax.random.PRNGKey(42), [Task], num_updates,
        task_embeddings=ach_table, train_state=ts,
        global_update_step=0, current_original_return=0.0)
    trained_ts = result["train_state"]
    jax.block_until_ready(trained_ts.params)
    sha = _params_sha(trained_ts.params)

    print("=" * 78, flush=True)
    print(f"[diagnose] make_train 12-update params_sha = {sha}", flush=True)
    print(f"[diagnose] frozen anchor ece6fa99          = {ECE6FA99}", flush=True)
    print(f"[diagnose] opt_step after 12 updates       = {int(trained_ts.step)}", flush=True)
    finite = bool(np.all([np.all(np.isfinite(np.asarray(v)))
                          for v in jax.tree_util.tree_leaves(trained_ts.params)]))
    print(f"[diagnose] params finite                   = {finite}", flush=True)
    if sha == ECE6FA99:
        print("VERDICT: ANCHOR_REPRODUCED_BY_MAKE_TRAIN — environment unchanged; the "
              "continuous-retrain launcher's chunked reimplementation has a HIDDEN divergence "
              "from make_train that must be found and fixed.", flush=True)
    else:
        print("VERDICT: ANCHOR_NOT_REPRODUCED_BY_MAKE_TRAIN — Henry's OWN code path no longer "
              "reproduces ece6fa99 from ckpt17500; the environment/library has DRIFTED since the "
              "telemetry run. The ece6fa99 anchor is not reproducible by any code path now; this "
              "is NOT a launcher bug — the plan-1 anchor assumption must be re-evaluated.",
              flush=True)
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
