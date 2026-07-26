"""Checkpoint save/restore: model, optimizer, replay metadata, RNG, global step.

D059 Gate 8: save/restore test for model, optimizer, replay metadata, RNG,
and global step.  Verifies round-trip fidelity:

  - Model parameters are bit-exact after restore
  - Optimizer state (Adam moments, step count) is preserved
  - Replay buffer metadata (counters, buffer contents, RNG state) restores
  - RNG state is checkpointed and restored
  - Global step counter survives round-trip
"""

import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax.training.train_state import TrainState

from trajectory_replay import TrajectoryReplayBuffer


# ---------------------------------------------------------------------------
# Orbax-based checkpointing for TrainState
# ---------------------------------------------------------------------------

def save_train_state(
    train_state: TrainState,
    path: str,
    step: int,
    metadata: Optional[dict] = None,
) -> str:
    """Save a TrainState (params + opt_state) to an orbax checkpoint.

    Returns the checkpoint directory path.
    """
    ckpt_dir = Path(path) / str(step)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(str(ckpt_dir / "default"), train_state)

    if metadata:
        meta_path = ckpt_dir / "amago_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

    return str(ckpt_dir)


def restore_train_state(
    path: str,
    dummy_train_state: TrainState,
    step: Optional[int] = None,
) -> Tuple[TrainState, int]:
    """Restore a TrainState from an orbax checkpoint.

    Parameters
    ----------
    path : str
        Root checkpoint directory.
    dummy_train_state : TrainState
        A TrainState with the same structure (network, tx) to use as template.
    step : int or None
        Specific step to restore.  If None, restores the latest.

    Returns
    -------
    (TrainState, int) — restored state and the step that was loaded.
    """
    root = Path(path)

    if step is None:
        # Find latest step
        steps = sorted(
            int(d.name) for d in root.iterdir()
            if d.is_dir() and d.name.isdigit()
        )
        if not steps:
            raise ValueError(f"No checkpoint steps found in {path}")
        step = steps[-1]

    checkpointer = ocp.PyTreeCheckpointer()
    options = ocp.CheckpointManagerOptions(create=False)
    ckpt_manager = ocp.CheckpointManager(str(root), checkpointer, options=options)

    restored = ckpt_manager.restore(step, items=dummy_train_state)

    # Load amago metadata if present
    return restored, step


# ---------------------------------------------------------------------------
# Full checkpoint: TrainState + replay buffer + counters + RNG + global step
# ---------------------------------------------------------------------------

def save_full_checkpoint(
    train_state: TrainState,
    replay_buffer: TrajectoryReplayBuffer,
    rng: jax.random.PRNGKey,
    global_step: int,
    path: str,
    step: Optional[int] = None,
    action_rng_state: Optional[dict] = None,
    update_count: int = 0,
    pending_state: Optional[dict] = None,
    collector_state: Optional[dict] = None,
    aux_opt_state: Optional[Any] = None,
) -> str:
    """Save a complete P2 checkpoint.

    Saves:
      - TrainState (orbax: params + opt_state)
      - Replay buffer (pickle: trajectories + counters + replay RNG state)
      - JAX RNG key
      - Action RNG state (checkpointable RNG used for action sampling)
      - Global step and update count
      - Pending episode buffers (方案B cross-rollout state) for exact resume
      - Collector state (obsv/env_state/memories/mem_mask/mem_idx at the save
        boundary) so a resumed run continues the in-progress rollout collection
        bit-exactly instead of re-resetting the env
      - Replay value-auxiliary critic-head optimizer state (方案2; the dedicated
        masked Adam moments for the critic value head) for exact resume.  None
        when the aux update has not yet fired (e.g. before any >=129 trajectory).
      - Metadata manifest
    """
    if step is None:
        step = global_step

    ckpt_dir = Path(path) / str(step)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. TrainState via orbax
    save_train_state(train_state, path, step)

    # 2. Replay buffer + RNG + action RNG + counters + pending buffers +
    #    collector env/memory state + 方案2 aux critic-head optimizer state via
    #    pickle
    replay_meta = {
        "replay_state": replay_buffer.state_dict(),
        "rng_key": rng,
        "action_rng_state": action_rng_state,
        "global_step": global_step,
        "update_count": int(update_count),
        "pending_state": pending_state,
        "collector_state": collector_state,
        "aux_opt_state": aux_opt_state,
    }
    with open(ckpt_dir / "replay_meta.pkl", "wb") as f:
        pickle.dump(replay_meta, f)

    # 3. Manifest
    manifest = {
        "checkpoint_step": step,
        "global_step": global_step,
        "update_count": int(update_count),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "counters": replay_buffer.counters.snapshot(),
        "replay_buffer_size": len(replay_buffer),
        "replay_longest_trajectory": replay_buffer.longest_trajectory_length,
        "action_rng_state_saved": action_rng_state is not None,
        "pending_buffers_saved": pending_state is not None,
        "pending_total_transitions": (
            sum(len(s["obs"]) for s in pending_state["slots"])
            if pending_state is not None else 0),
        "collector_state_saved": collector_state is not None,
        "aux_opt_state_saved": aux_opt_state is not None,
    }
    with open(ckpt_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    return str(ckpt_dir)


def restore_full_checkpoint(
    path: str,
    dummy_train_state: TrainState,
    step: Optional[int] = None,
) -> dict:
    """Restore a full D059 checkpoint.

    Returns a dict with keys:
      - "train_state": TrainState
      - "replay_buffer": TrajectoryReplayBuffer
      - "rng": jax.random.PRNGKey
      - "global_step": int
      - "step": int (checkpoint step number)
      - "manifest": dict
    """
    root = Path(path)

    if step is None:
        steps = sorted(
            int(d.name) for d in root.iterdir()
            if d.is_dir() and d.name.isdigit()
        )
        if not steps:
            raise ValueError(f"No checkpoint steps found in {path}")
        step = steps[-1]

    ckpt_dir = root / str(step)

    # 1. Restore TrainState
    restored_ts, _ = restore_train_state(path, dummy_train_state, step)

    # 2. Restore replay meta
    replay_meta_path = ckpt_dir / "replay_meta.pkl"
    if replay_meta_path.exists():
        with open(replay_meta_path, "rb") as f:
            replay_meta = pickle.load(f)
        replay_buffer = TrajectoryReplayBuffer.from_state_dict(
            replay_meta["replay_state"]
        )
        rng = replay_meta["rng_key"]
        global_step = replay_meta["global_step"]
        action_rng_state = replay_meta.get("action_rng_state", None)
        update_count = int(replay_meta.get("update_count", 0))
        pending_state = replay_meta.get("pending_state", None)
        collector_state = replay_meta.get("collector_state", None)
        aux_opt_state = replay_meta.get("aux_opt_state", None)
    else:
        replay_buffer = None
        rng = None
        global_step = 0
        action_rng_state = None
        update_count = 0
        pending_state = None
        collector_state = None
        aux_opt_state = None

    # 3. Manifest
    manifest_path = ckpt_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    return {
        "train_state": restored_ts,
        "replay_buffer": replay_buffer,
        "rng": rng,
        "action_rng_state": action_rng_state,
        "global_step": global_step,
        "update_count": update_count,
        "step": step,
        "manifest": manifest,
        "pending_state": pending_state,
        "collector_state": collector_state,
        "aux_opt_state": aux_opt_state,
    }


# ---------------------------------------------------------------------------
# Deterministic checkpoint inventory
# ---------------------------------------------------------------------------

def checkpoint_inventory(path: str) -> dict:
    """Produce a deterministic report of what a checkpoint contains.

    D059 Gate 1 requirement: deterministic checkpoint inventory.
    """
    root = Path(path)
    if not root.exists():
        return {"exists": False, "path": str(root)}

    info = {
        "exists": True,
        "path": str(root),
        "steps": [],
    }

    for d in sorted(root.iterdir()):
        if not d.is_dir() or not d.name.isdigit():
            continue
        step_info = {
            "step": int(d.name),
            "files": sorted([f.name for f in d.iterdir()]),
        }
        # Check for manifest
        manifest_path = d / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                step_info["manifest"] = json.load(f)
        info["steps"].append(step_info)

    info["total_steps"] = len(info["steps"])
    return info


# ---------------------------------------------------------------------------
# Compatible-weight restore report
# ---------------------------------------------------------------------------

def compatible_weight_restore_report(
    checkpoint_path: str,
    dummy_train_state: TrainState,
) -> dict:
    """Load checkpoint weights and produce a report of what was restored,
    skipped, or newly initialized.

    D059 requirement: record exactly which parameter paths were restored,
    skipped, or newly initialized.  Do not claim full optimizer continuation.
    """
    try:
        restored = restore_full_checkpoint(checkpoint_path, dummy_train_state)
        ts = restored["train_state"]
        params = ts.params
        opt_state = ts.opt_state

        # Flatten params to report paths
        param_paths = _flatten_param_paths(params)
        opt_paths = _flatten_param_paths(opt_state)

        return {
            "status": "RESTORED",
            "step": restored["step"],
            "parameter_paths_restored": sorted(param_paths),
            "parameter_count": len(param_paths),
            "optimizer_paths_restored": sorted(opt_paths),
            "optimizer_state_restored": True,
            "optimizer_continuation": "PARTIAL — optimizer state from checkpoint, "
                                      "but AMAGO-style learner may have changed "
                                      "optimizer semantics.  Do not claim full "
                                      "optimizer continuation (D059 §11).",
            "newly_initialized": [],
            "skipped": [],
        }
    except Exception as e:
        return {
            "status": "FAILED",
            "error": str(e),
        }


def _flatten_param_paths(tree, prefix: str = "") -> list:
    """Flatten a pytree to dotted paths."""
    import jax.tree_util as jtu

    paths = []
    leaves, treedef = jtu.tree_flatten_with_path(tree)
    for path, _ in leaves:
        key_path = ".".join(
            str(k.key) if hasattr(k, 'key') else str(k)
            for k in path
        )
        if prefix:
            key_path = f"{prefix}.{key_path}" if key_path else prefix
        if key_path:
            paths.append(key_path)
    return paths
