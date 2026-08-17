"""Recurrent-state capture / validation / state-start alignment (G3).

SlowGRU key-alias canonicalization (P0): the TrainingBackend and the
StudentAdapter name the fast-window GTrXL memory differently, which would
otherwise break cross-surface restore / hashing:

  TrainingBackend (slowgru training)      StudentAdapter (read-only probe)
  ----------------------------------      --------------------------------
  mem_mask                                memories_mask
  mem_idx                                 memories_mask_idx

``canonicalize_memory`` normalizes both spellings to the TrainingBackend keys
so that capture, hashing and validation are alias-robust.  Longstate keys
(``longstate.h/buf/count``) are identical on both surfaces and left untouched.
"""
from __future__ import annotations

from typing import Any, Mapping

import jax
import numpy as np

from .hashing import hash_pytree

SLOWGRU_LONGSTATE_KEYS = ("longstate.h", "longstate.buf", "longstate.count")

# adapter spelling -> training-backend (canonical) spelling
SLOWGRU_KEY_ALIASES = {
    "memories_mask": "mem_mask",
    "memories_mask_idx": "mem_idx",
}


def canonicalize_memory(memory: Mapping[str, Any], *,
                        architecture_family: str = "slice") -> dict:
    """Return memory with SlowGRU fast-window keys normalized to canonical form.

    Non-SlowGRU families are returned as-is (numpy leaves).  For SlowGRU, both
    ``mem_mask``/``mem_idx`` (backend) and ``memories_mask``/``memories_mask_idx``
    (adapter) are mapped to ``mem_mask``/``mem_idx``.
    """
    mem = {k: np.asarray(v) for k, v in dict(memory).items()}
    if architecture_family.lower() != "slowgru":
        return mem
    return {SLOWGRU_KEY_ALIASES.get(k, k): v for k, v in mem.items()}


class RecurrentStateError(RuntimeError):
    """RECURRENT_STATE_MISALIGNED (fail closed)."""


def memory_batch_size(memory: Mapping[str, Any]) -> int:
    sizes = set()
    for leaf in jax.tree_util.tree_leaves(dict(memory)):
        arr = np.asarray(leaf)
        if arr.ndim >= 1:
            sizes.add(int(arr.shape[0]))
    if not sizes:
        raise RecurrentStateError("memory has no batched leaves")
    if len(sizes) != 1:
        raise RecurrentStateError(f"memory batch dims disagree: {sorted(sizes)}")
    return sizes.pop()


def capture_memory(memory: Mapping[str, Any], *,
                   architecture_family: str = "slice") -> dict:
    return canonicalize_memory(memory, architecture_family=architecture_family)


def memory_hash(memory: Mapping[str, Any], *,
                architecture_family: str = "slice") -> str:
    return hash_pytree(canonicalize_memory(memory,
                                           architecture_family=architecture_family))


def stack_memories(memories: list) -> dict:
    keys = list(memories[0].keys())
    return {k: np.stack([np.asarray(m[k]) for m in memories], axis=0) for k in keys}


def validate_memory(memory: Mapping[str, Any], batch_size: int, *,
                    architecture_family: str = "slice",
                    allow_memory_reset_experiment: bool = False) -> dict:
    reasons = []
    mem = canonicalize_memory(memory, architecture_family=architecture_family)
    try:
        actual = memory_batch_size(mem)
        if actual != batch_size:
            reasons.append(f"memory batch {actual} != env batch {batch_size}")
    except RecurrentStateError as exc:
        reasons.append(str(exc))
    for key, leaf in mem.items():
        arr = np.asarray(leaf)
        if arr.dtype.kind == "f" and arr.size and not np.isfinite(arr).all():
            reasons.append(f"non-finite memory leaf {key}")
    if architecture_family.lower() == "slowgru":
        missing = [k for k in SLOWGRU_LONGSTATE_KEYS if k not in mem]
        if missing:
            reasons.append(f"slowgru persistent longstate missing {missing}")
        zeroed = [k for k in SLOWGRU_LONGSTATE_KEYS
                  if k in mem and not np.asarray(mem[k]).any()]
        if zeroed and not allow_memory_reset_experiment:
            reasons.append(
                f"slowgru longstate all-zero {zeroed}: EnvState from episode "
                "mid-point with reset memory is forbidden unless this is an "
                "explicit memory-reset intervention "
                "(allow_memory_reset_experiment=True)")
        # fast-window GTrXL memory must be present on the canonical keys too
        for key in ("memories", "mem_mask", "mem_idx"):
            if key not in mem:
                reasons.append(f"slowgru fast-window memory missing {key}")
    return {"ok": not reasons, "reasons": reasons}


def assert_state_start_alignment(memory: Mapping[str, Any], batch_size: int, *,
                                 architecture_family: str = "slice",
                                 allow_memory_reset_experiment: bool = False) -> None:
    check = validate_memory(memory, batch_size,
                            architecture_family=architecture_family,
                            allow_memory_reset_experiment=allow_memory_reset_experiment)
    if not check["ok"]:
        raise RecurrentStateError(
            f"RECURRENT_STATE_MISALIGNED: {check['reasons']}")