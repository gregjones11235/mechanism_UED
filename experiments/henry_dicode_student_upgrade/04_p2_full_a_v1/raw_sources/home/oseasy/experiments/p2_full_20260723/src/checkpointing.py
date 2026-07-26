"""P2-Full-A full checkpoint save/restore — PURE PICKLE (CPU/GPU portable).

Design decision (frozen): P2-Full-A's OWN checkpoints use pure pickle with all JAX
arrays converted to numpy first. This sidesteps the orbax ocdbt GPU<->CPU
incompatibility (an orbax checkpoint saved on GPU cannot be restored under CPU), so
exact-resume and round-trip conservation tests can run on CPU while training runs on
GPU0. Reading the Henry base ckpt17500 still uses orbax (see compat_init.py, GPU0 only).

Saved per checkpoint step (<path>/<step>/):
  * full_state.pkl  — params, target_params (EMA), opt_state, replay_buffer.state_dict
                      (incl anchors), pending.state_dict (incl anchors), rng_key,
                      action_rng_state, global_step, update_count, collector_state,
                      config (asdict).
  * manifest.json   — human/JSON-readable provenance (step, counters, sizes, config,
                      params content SHA256, timestamp).

All JAX pytrees (params / target_params / opt_state / collector_state) are run through
tree_map(np.asarray) before pickling and tree_map(jnp.asarray) on restore, so a
checkpoint written on GPU0 restores bit-exactly on CPU (and vice versa).
"""
import dataclasses
import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# pytree portability helpers
# ---------------------------------------------------------------------------

def _to_numpy(tree):
    """Recursively convert every leaf of a JAX pytree to a numpy array."""
    return jax.tree_util.tree_map(lambda x: np.asarray(x), tree)


def _to_jnp(tree):
    """Recursively convert every leaf of a (numpy) pytree back to jnp arrays."""
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x), tree)


def params_content_sha256(params) -> str:
    """Deterministic content hash over flattened parameter leaves (shape + bytes),
    in tree-leaf traversal order. Matches the provenance hash used for ckpt17500."""
    h = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(params):
        arr = np.asarray(leaf)
        h.update(str(arr.shape).encode())
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def param_count(params) -> int:
    return int(sum(np.asarray(l).size for l in jax.tree_util.tree_leaves(params)))


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------

def save_full_checkpoint(
    params,
    target_params,
    opt_state,
    replay_buffer,
    rng_key,
    global_step: int,
    path: str,
    step: int = None,
    action_rng_state: dict = None,
    update_count: int = 0,
    pending=None,
    collector_state: dict = None,
    config=None,
    keep: int = 3,
    extra_manifest: dict = None,
) -> str:
    """Write a complete, exactly-resumable P2-Full-A checkpoint. Returns its dir."""
    if step is None:
        step = int(global_step)
    ckpt_dir = Path(path) / str(int(step))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    pending_state = None
    if pending is not None:
        pending_state = pending.state_dict() if hasattr(pending, "state_dict") else pending

    config_dict = None
    if config is not None:
        config_dict = dataclasses.asdict(config) if dataclasses.is_dataclass(config) else dict(config)

    full_state = {
        "params": _to_numpy(params),
        "target_params": _to_numpy(target_params),
        "opt_state": _to_numpy(opt_state),
        "replay_state": replay_buffer.state_dict(),
        "rng_key": np.asarray(rng_key),
        "action_rng_state": action_rng_state,
        "global_step": int(global_step),
        "update_count": int(update_count),
        "pending_state": pending_state,
        "collector_state": (_to_numpy(collector_state)
                            if collector_state is not None else None),
        "config": config_dict,
    }
    # atomic-ish write: dump to tmp then replace
    tmp = ckpt_dir / "full_state.pkl.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(full_state, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(ckpt_dir / "full_state.pkl")

    manifest = {
        "checkpoint_step": int(step),
        "global_step": int(global_step),
        "update_count": int(update_count),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "format": "p2_full_a_pure_pickle_v1",
        "params_sha256": params_content_sha256(params),
        "param_count": param_count(params),
        "counters": replay_buffer.counters.snapshot(),
        "replay_buffer_size": len(replay_buffer),
        "replay_longest_trajectory": replay_buffer.longest_trajectory_length,
        "replay_hash_digest": replay_buffer.hash_digest(),
        "action_rng_state_saved": action_rng_state is not None,
        "pending_buffers_saved": pending_state is not None,
        "pending_total_transitions": (
            sum(len(s["obs"]) for s in pending_state["slots"])
            if pending_state is not None else 0),
        "pending_total_anchors": (
            sum(len(s["anchor_mem"]) for s in pending_state["slots"])
            if pending_state is not None else 0),
        "collector_state_saved": collector_state is not None,
        "config": config_dict,
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    with open(ckpt_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, default=str)
        f.write("\n")

    if keep and keep > 0:
        strip_old_checkpoints(path, keep=keep)
    return str(ckpt_dir)


# ---------------------------------------------------------------------------
# restore
# ---------------------------------------------------------------------------

def detect_latest_step(path: str):
    root = Path(path)
    if not root.exists():
        return None
    steps = [int(d.name) for d in root.iterdir()
             if d.is_dir() and d.name.isdigit()
             and (d / "full_state.pkl").exists()]
    return max(steps) if steps else None


def restore_full_checkpoint(path: str, step: int = None) -> dict:
    """Restore a P2-Full-A checkpoint. Returns a dict with jnp pytrees + objects."""
    from replay_buffer import ReplayBuffer
    from pending_episodes import PendingEpisodeBuffers

    if step is None:
        step = detect_latest_step(path)
        if step is None:
            raise ValueError(f"No P2-Full-A checkpoint steps found in {path}")
    ckpt_dir = Path(path) / str(int(step))
    pkl = ckpt_dir / "full_state.pkl"
    if not pkl.exists():
        raise ValueError(f"Missing full_state.pkl at {pkl}")
    with open(pkl, "rb") as f:
        st = pickle.load(f)

    replay_buffer = ReplayBuffer.from_state_dict(st["replay_state"])
    pending = (PendingEpisodeBuffers.from_state_dict(st["pending_state"])
               if st.get("pending_state") is not None else None)

    manifest = {}
    mp = ckpt_dir / "manifest.json"
    if mp.exists():
        with open(mp) as f:
            manifest = json.load(f)

    return {
        "params": _to_jnp(st["params"]),
        "target_params": _to_jnp(st["target_params"]),
        "opt_state": _to_jnp(st["opt_state"]),
        "replay_buffer": replay_buffer,
        "pending": pending,
        "rng_key": jnp.asarray(st["rng_key"]),
        "action_rng_state": st.get("action_rng_state"),
        "global_step": int(st["global_step"]),
        "update_count": int(st.get("update_count", 0)),
        "collector_state": (_to_jnp(st["collector_state"])
                            if st.get("collector_state") is not None else None),
        "config": st.get("config"),
        "step": int(step),
        "manifest": manifest,
    }


# ---------------------------------------------------------------------------
# inventory / strip
# ---------------------------------------------------------------------------

def checkpoint_inventory(path: str) -> dict:
    """Deterministic report of every checkpoint step present under `path`."""
    root = Path(path)
    if not root.exists():
        return {"exists": False, "path": str(root)}
    info = {"exists": True, "path": str(root), "steps": []}
    for d in sorted(root.iterdir(), key=lambda p: p.name):
        if not d.is_dir() or not d.name.isdigit():
            continue
        entry = {"step": int(d.name),
                 "files": sorted(f.name for f in d.iterdir())}
        mp = d / "manifest.json"
        if mp.exists():
            with open(mp) as f:
                entry["manifest"] = json.load(f)
        info["steps"].append(entry)
    info["total_steps"] = len(info["steps"])
    info["latest_step"] = (info["steps"][-1]["step"] if info["steps"] else None)
    return info


def strip_old_checkpoints(path: str, keep: int = 3) -> list:
    """Keep only the newest `keep` checkpoint steps; remove older ones. Returns removed."""
    root = Path(path)
    if not root.exists():
        return []
    steps = sorted(int(d.name) for d in root.iterdir()
                   if d.is_dir() and d.name.isdigit())
    removed = []
    for s in steps[:-keep] if keep > 0 else steps:
        d = root / str(s)
        for f in d.iterdir():
            if f.is_file():
                f.unlink()
        try:
            d.rmdir()
        except OSError:
            pass
        removed.append(s)
    return removed
