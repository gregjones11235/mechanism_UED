#!/usr/bin/env python3
"""D052 Phase 3: 128-episode evaluation of 3 auction_budgeted cells (lpg_hrl, lpac, clpa).

Frozen evaluator: d052_pilot_via_run_session_eval.py
SHA256: 01f7f92f5f4f4cbe5780cf82fc746cfbbd8c03827adf57dfad028d5904f80ee6

Cells (all auction_budgeted training, 98304 checkpoint):
  - auction_budgeted_x_lpg_hrl  (seed0_1784566018)
  - auction_budgeted_x_lpac     (seed0_1784565782)
  - auction_budgeted_x_clpa     (seed0_1784543391)

Each cell evaluated in an INDEPENDENT subprocess to avoid GPU memory
fragmentation/crashes across JAX allocations.

Output: evaluation_results_phase3/
  - evaluation_results.json
  - summary.json
  - provenance.json

Usage:
  CUDA_VISIBLE_DEVICES=0 python evidence/d052_phase3_eval.py
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

# ── Frozen evaluator identity ──
FROZEN_EVAL_PATH = Path(
    "/home/oseasy/experiments/mechanism_UED_continuation_20260715/"
    "workers/gpu2_budgeted_soft_copeland/evidence/"
    "d052_pilot_via_run_session_eval.py"
)
FROZEN_SHA256_EXPECTED = (
    "01f7f92f5f4f4cbe5780cf82fc746cfbbd8c03827adf57dfad028d5904f80ee6"
)

# Verify frozen evaluator
_frozen_bytes = FROZEN_EVAL_PATH.read_bytes()
_actual_sha = hashlib.sha256(_frozen_bytes).hexdigest()
assert _actual_sha == FROZEN_SHA256_EXPECTED, (
    f"FROZEN EVALUATOR TAMPERED: expected {FROZEN_SHA256_EXPECTED}, got {_actual_sha}"
)
print(f"[phase3] Frozen evaluator verified: {FROZEN_SHA256_EXPECTED[:16]}...")

# ── This script's identity ──
THIS_SCRIPT = Path(__file__).resolve()
PHASE3_SHA256 = hashlib.sha256(THIS_SCRIPT.read_bytes()).hexdigest()

# ── Configuration ──
GPU_UUID = "GPU-e8c08612-c22a-c8a4-6df5-affb2dd1f9a6"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

NUM_ENVS = 128
NUM_STEPS = 8192
EVAL_SEED = 42
STEP = 98304

CKPT_ROOT = os.path.expanduser(
    "~/experiments/mechanism_UED_continuation_20260715/shared_r0/runs/d052_dynamic"
)

CELLS = {
    "auction_budgeted_x_lpg_hrl": {
        "seed_dir": "seed0_1784566018",
        "training": "auction_budgeted",
        "aggregation": "lpg_hrl",
    },
    "auction_budgeted_x_lpac": {
        "seed_dir": "seed0_1784565782",
        "training": "auction_budgeted",
        "aggregation": "lpac",
    },
    "auction_budgeted_x_clpa": {
        "seed_dir": "seed0_1784543391",
        "training": "auction_budgeted",
        "aggregation": "clpa",
    },
}

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(CURRENT_DIR, "evaluation_results_phase3")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Single-cell evaluation script (runs in subprocess) ──
EVAL_CELL_SCRIPT = r"""
import hashlib, json, os, sys, traceback, math
import jax, jax.numpy as jnp
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

_DICODE_V6 = os.path.expanduser(
    "~/experiments/mechanism_UED_continuation_20260715/workers/gpu0_original/dicode_v6")
_DICODE_SRC = os.path.join(_DICODE_V6, "src")
for _p in [_DICODE_SRC, _DICODE_V6]:
    if _p in sys.path: sys.path.remove(_p)
    sys.path.insert(0, _p)

from dicode.craftax_evaluation import main as craftax_evaluate
from dicode.utils.general.train_state_utils import load_weights_only
from dicode.task_utils import get_achievement_multi_hot
from minicraftax.envs.craftax import CraftaxAugObsTrain
from craftax.craftax.constants import Achievement

# ── Parameters passed via environment ──
CELL_NAME = os.environ["D052_CELL_NAME"]
CELL_SEED_DIR = os.environ["D052_CELL_SEED_DIR"]
CELL_TRAINING = os.environ["D052_CELL_TRAINING"]
CELL_AGGREGATION = os.environ["D052_CELL_AGGREGATION"]
CKPT_ROOT = os.environ["D052_CKPT_ROOT"]
STEP = int(os.environ["D052_STEP"])
NUM_ENVS = int(os.environ["D052_NUM_ENVS"])
NUM_STEPS = int(os.environ["D052_NUM_STEPS"])
EVAL_SEED = int(os.environ["D052_EVAL_SEED"])
OUTPUT_DIR = os.environ["D052_OUTPUT_DIR"]
_BASE_OBS_DIM = int(os.environ["D052_BASE_OBS_DIM"])

SHARED_ARCH = {
    "embed_size": 32, "hidden_layers": 64,
    "num_heads": 4, "num_layers": 1,
    "qkv_features": 128, "gating": True,
}

print(f"[sub:{CELL_NAME}] Starting evaluation...")
print(f"[sub:{CELL_NAME}] JAX devices: {jax.devices()}")

# Detect architecture
ckpt_path = Path(CKPT_ROOT) / CELL_NAME / CELL_SEED_DIR / "checkpoints"
import orbax.checkpoint as ocp
handler = ocp.PyTreeCheckpointHandler()
full_state = handler.restore(ckpt_path / str(STEP) / "default")
params = full_state["train_state"]["params"]["params"]
encoder_shape = params["transformer"]["encoder"]["kernel"].shape
input_dim, encoder_size = int(encoder_shape[0]), int(encoder_shape[1])
cond_dim = input_dim - _BASE_OBS_DIM
arch = dict(SHARED_ARCH)
arch["conditioning_dim"] = cond_dim
arch["encoder_input_dim"] = input_dim

print(f"[sub:{CELL_NAME}] cond_dim={cond_dim}, encoder_input_dim={input_dim}")

# Build config
config = OmegaConf.create({
    "seed": EVAL_SEED, "use_wandb": False,
    "training": {
        "env_name": "Craftax-Symbolic-v1",
        "lr": 2e-4, "min_lr": 2e-6,
        "num_envs": 1024, "num_steps": 128,
        "total_timesteps": 2_005_401_600,
        "update_epochs": 4, "num_minibatches": 8,
        "gamma": 0.999, "gae_lambda": 0.8,
        "clip_eps": 0.2, "ent_coef": 0.002, "vf_coef": 0.5,
        "max_grad_norm": 1.0, "activation": "relu",
        "anneal_lr": True, "anneal_method": "linear",
        "qkv_features": arch["qkv_features"],
        "embed_size": arch["embed_size"],
        "num_heads": arch["num_heads"], "num_layers": arch["num_layers"],
        "hidden_layers": arch["hidden_layers"],
        "window_mem": 128, "window_grad": 64,
        "gating": arch["gating"], "gating_bias": 2.0,
        "condition_on_task": True, "conditioning_type": "one_hot",
        "use_optimistic_resets": True, "optimistic_reset_ratio": 16,
        "jit": True, "save_policy": False, "debug": True, "num_repeats": 1,
        "scoring_window_updates": 40,
        "completion_bonus_scale": 0.0, "completion_bonus_min": 0.0,
        "max_updates_per_session": 64,
        "mode": "coop", "bonus_type": "none", "dynamic_bonus_k": 0.0,
    },
    "evaluation": {
        "num_steps": NUM_STEPS, "num_envs": NUM_ENVS,
        "optimistic_reset_ratio": 16,
    },
})

# Load checkpoint
ckpt_full_path = str(ckpt_path / str(STEP))
dummy_env = CraftaxAugObsTrain(
    condition_on_task=True, conditioning_type="one_hot",
    embedding_size=cond_dim, task_embeddings=jnp.zeros((1, cond_dim)),
)
rl_train_state = load_weights_only(
    checkpoint_path=ckpt_full_path, env=dummy_env,
    env_params=dummy_env.default_params, config=config.training,
)
param_leaves = len(jax.tree_util.tree_leaves(rl_train_state.params))
print(f"[sub:{CELL_NAME}] Loaded {param_leaves} param leaves")

# Build eval_embedding with conditioning adapter
eval_env_tmp = CraftaxAugObsTrain()
full_multi_hot = get_achievement_multi_hot(eval_env_tmp.relevant_achievements)
eval_embedding = jnp.tile(
    full_multi_hot[:cond_dim], (NUM_ENVS, 1)
).astype(jnp.float32)

# Run evaluation
rng = jax.random.PRNGKey(EVAL_SEED)
rng, eval_rng = jax.random.split(rng)
metrics_dict = craftax_evaluate(
    config, eval_rng, train_state=rl_train_state,
    eval_embedding=eval_embedding,
)

mr = float(metrics_dict.get("mean_return", float("nan")))
mp = float(metrics_dict.get("mean_performance", float("nan")))
el = float(metrics_dict.get("average_episode_length", float("nan")))
n_finished = int(float(metrics_dict.get("_cooc_total", 0)))

# Per-env data
chain_finished = np.asarray(metrics_dict.get("_chain_finished", []))
chain_first_step = np.asarray(metrics_dict.get("_chain_first_step", []))
cooc_count = np.asarray(metrics_dict.get("_cooc_count", []))
cooc_names = metrics_dict.get("_cooc_names", [])

# Achievement success rates
achievements = {}
if len(cooc_count) > 0 and len(cooc_names) > 0:
    for i, ach_name in enumerate(cooc_names):
        if i < len(cooc_count):
            count_i = float(cooc_count[i])
            sr = count_i / max(n_finished, 1) * 100.0
            achievements[ach_name] = {
                "success_count": int(count_i),
                "success_rate_pct": round(sr, 2),
            }

# NaN check
nan_found = False
for k, v in metrics_dict.items():
    if isinstance(v, (float, np.floating, jnp.ndarray)):
        val = float(np.asarray(v).mean()) if hasattr(v, "shape") and v.shape else float(v)
        if np.isnan(val) or np.isinf(val):
            nan_found = True
            break

result = {
    "cell": CELL_NAME,
    "training": CELL_TRAINING,
    "aggregation": CELL_AGGREGATION,
    "seed_dir": CELL_SEED_DIR,
    "checkpoint_path": ckpt_full_path,
    "cond_dim": cond_dim,
    "param_leaves": param_leaves,
    "metrics": {
        "mean_return": mr,
        "mean_performance": mp,
        "average_episode_length": el,
        "finished_episodes": n_finished,
        "total_episodes": NUM_ENVS,
    },
    "achievements": achievements,
    "nan_free": not nan_found,
    "n_episodes_with_achievements": int(
        sum(1 for env_idx in range(min(NUM_ENVS, len(chain_first_step)))
            if np.any(np.asarray(chain_first_step[env_idx]) >= 0))
    ) if len(chain_first_step) > 0 else 0,
}

# Write per-cell result
cell_out = os.path.join(OUTPUT_DIR, f"result_{CELL_NAME}.json")
with open(cell_out, "w") as f:
    json.dump(result, f, indent=2, default=str)

print(f"[sub:{CELL_NAME}] DONE: return={mr:.4f} perf={mp:.2f}% "
      f"len={el:.1f} finished={n_finished}/{NUM_ENVS} nan={nan_found}")
print(f"[sub:{CELL_NAME}] Result written to: {cell_out}")
"""


def evaluate_cell_in_subprocess(name: str, info: Dict) -> Dict:
    """Run one cell evaluation in an isolated subprocess."""
    print(f"\n{'='*60}")
    print(f"  CELL: {name}")
    print(f"{'='*60}")

    ckpt_path = os.path.join(
        CKPT_ROOT, name, info["seed_dir"], "checkpoints", str(STEP))
    assert os.path.isdir(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    print(f"  Checkpoint: {ckpt_path}")

    # Write eval script to temp file
    tmpf = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir="/tmp")
    tmpf.write(EVAL_CELL_SCRIPT)
    tmpf.close()

    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "0",
        "D052_CELL_NAME": name,
        "D052_CELL_SEED_DIR": info["seed_dir"],
        "D052_CELL_TRAINING": info["training"],
        "D052_CELL_AGGREGATION": info["aggregation"],
        "D052_CKPT_ROOT": CKPT_ROOT,
        "D052_STEP": str(STEP),
        "D052_NUM_ENVS": str(NUM_ENVS),
        "D052_NUM_STEPS": str(NUM_STEPS),
        "D052_EVAL_SEED": str(EVAL_SEED),
        "D052_OUTPUT_DIR": OUTPUT_DIR,
        "D052_BASE_OBS_DIM": "8268",
    })

    try:
        result = subprocess.run(
            [sys.executable, tmpf.name],
            env=env, capture_output=True, text=True, timeout=1800,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"  STDERR:\n{result.stderr}")
            return {"cell": name, "error": result.stderr,
                    "returncode": result.returncode}
    except subprocess.TimeoutExpired:
        return {"cell": name, "error": "TIMEOUT (30 min)"}
    finally:
        os.unlink(tmpf.name)

    # Try to read the result file
    result_path = os.path.join(OUTPUT_DIR, f"result_{name}.json")
    if os.path.exists(result_path):
        with open(result_path) as f:
            return json.load(f)
    return {"cell": name, "error": "No result file produced"}


def main():
    start_time = datetime.now(timezone.utc)
    print("=" * 70)
    print("  D052 PHASE 3 — 128-episode Evaluation (3 auction_budgeted cells)")
    print(f"  Start: {start_time.isoformat()}")
    print(f"  GPU: GPU-e8c08612-c22a-c8a4-6df5-affb2dd1f9a6 (index 0)")
    print(f"  Episodes/cell: {NUM_ENVS}")
    print(f"  Max steps: {NUM_STEPS}")
    print(f"  Eval seed: {EVAL_SEED}")
    print(f"  Frozen evaluator SHA256: {FROZEN_SHA256_EXPECTED}")
    print(f"  Phase3 script SHA256: {PHASE3_SHA256}")
    print(f"  Per-cell ISOLATION: independent subprocess")
    print("=" * 70)

    # Preflight: validate checkpoints
    print("\n[Preflight] Checking checkpoints...")
    for name, info in CELLS.items():
        ckpt_path = os.path.join(
            CKPT_ROOT, name, info["seed_dir"], "checkpoints", str(STEP))
        if os.path.isdir(ckpt_path):
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name}: NOT FOUND at {ckpt_path}")
            info["_skip"] = True

    # Evaluate each cell in independent subprocess
    all_results = []
    for name, info in CELLS.items():
        if info.get("_skip"):
            continue
        result = evaluate_cell_in_subprocess(name, info)
        all_results.append(result)

    end_time = datetime.now(timezone.utc)

    # ── Generate output files ──
    print(f"\n{'='*70}")
    print(f"  Generating phase 3 output files...")
    print(f"{'='*70}")

    # 1. evaluation_results.json
    eval_results = {
        "phase": "3",
        "timestamp": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "num_episodes_per_cell": NUM_ENVS,
        "max_steps": NUM_STEPS,
        "eval_seed": EVAL_SEED,
        "gpu": GPU_UUID,
        "frozen_evaluator_sha256": FROZEN_SHA256_EXPECTED,
        "phase3_script_sha256": PHASE3_SHA256,
        "cells": all_results,
    }
    with open(os.path.join(OUTPUT_DIR, "evaluation_results.json"), "w") as f:
        json.dump(eval_results, f, indent=2, default=str)
    print(f"  [1/3] evaluation_results.json")

    # 2. summary.json
    summary = {
        "phase": "3",
        "timestamp": end_time.isoformat(),
        "comparison": [],
    }
    for r in all_results:
        m = r.get("metrics", {})
        summary["comparison"].append({
            "cell": r["cell"],
            "training": r.get("training", "?"),
            "aggregation": r.get("aggregation", "?"),
            "mean_return": m.get("mean_return", float("nan")),
            "mean_performance": m.get("mean_performance", float("nan")),
            "avg_episode_length": m.get("average_episode_length", float("nan")),
            "finished": f"{m.get('finished_episodes', 0)}/{m.get('total_episodes', NUM_ENVS)}",
            "cond_dim": r.get("cond_dim", "?"),
            "nan_free": r.get("nan_free", False),
            "error": r.get("error"),
        })
    summary["comparison"].sort(
        key=lambda x: x["mean_return"] if not (
            isinstance(x["mean_return"], float) and
            (x["mean_return"] != x["mean_return"])  # NaN check
        ) else -float("inf"),
        reverse=True,
    )
    for i, c in enumerate(summary["comparison"]):
        c["rank"] = i + 1
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  [2/3] summary.json")

    # 3. provenance.json
    provenance = {
        "phase": "3",
        "timestamp": end_time.isoformat(),
        "frozen_evaluator": {
            "path": str(FROZEN_EVAL_PATH),
            "sha256": FROZEN_SHA256_EXPECTED,
        },
        "phase3_script": {
            "path": str(THIS_SCRIPT),
            "sha256": PHASE3_SHA256,
        },
        "config": {
            "num_episodes_per_cell": NUM_ENVS,
            "max_steps": NUM_STEPS,
            "eval_seed": EVAL_SEED,
            "gpu": GPU_UUID,
            "step": STEP,
            "per_cell_isolation": "independent subprocess (avoid JAX memory fragmentation)",
        },
        "checkpoints": {},
    }
    for r in all_results:
        provenance["checkpoints"][r["cell"]] = {
            "path": r.get("checkpoint_path", "?"),
            "training": r.get("training", "?"),
            "aggregation": r.get("aggregation", "?"),
            "cond_dim": r.get("cond_dim", "?"),
            "param_leaves": r.get("param_leaves", "?"),
        }
    with open(os.path.join(OUTPUT_DIR, "provenance.json"), "w") as f:
        json.dump(provenance, f, indent=2, default=str)
    print(f"  [3/3] provenance.json")

    # ── Print summary ──
    print(f"\n{'='*70}")
    print(f"  PHASE 3 SUMMARY")
    print(f"{'='*70}")
    for c in summary["comparison"]:
        err = " [ERROR]" if c.get("error") else ""
        print(f"  {c['rank']}. {c['cell']}: "
              f"return={c['mean_return']:.4f}  "
              f"perf={c['mean_performance']:.2f}%  "
              f"len={c['avg_episode_length']:.1f}  "
              f"finished={c['finished']}{err}")
    print(f"\n  Output: {OUTPUT_DIR}")
    print(f"  Duration: {(end_time - start_time).total_seconds():.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
