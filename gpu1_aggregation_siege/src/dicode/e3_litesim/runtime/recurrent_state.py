"""Recurrent-state capture / validation / state-start alignment (G3)."""
from __future__ import annotations

from typing import Any, Mapping

import jax
import numpy as np

from .hashing import hash_pytree

SLOWGRU_LONGSTATE_KEYS = ("longstate.h", "longstate.buf", "longstate.count")


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


def capture_memory(memory: Mapping[str, Any]) -> dict:
    return {key: np.asarray(leaf) for key, leaf in dict(memory).items()}


def memory_hash(memory: Mapping[str, Any]) -> str:
    return hash_pytree({k: np.asarray(v) for k, v in dict(memory).items()})


def stack_memories(memories: list) -> dict:
    keys = list(memories[0].keys())
    return {k: np.stack([np.asarray(m[k]) for m in memories], axis=0) for k in keys}


def validate_memory(memory: Mapping[str, Any], batch_size: int, *,
                    architecture_family: str = "slice",
                    allow_memory_reset_experiment: bool = False) -> dict:
    reasons = []
    mem = dict(memory)
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
    if architecture_family == "slowgru":
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