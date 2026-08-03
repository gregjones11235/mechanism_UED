"""Explicit memory-restore modes; zero memory is never implicit."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class MemoryRestoreMode(str, Enum):
    ZERO_MEMORY = "ZERO_MEMORY"
    SAVED_POLICY_MEMORY = "SAVED_POLICY_MEMORY"
    HISTORY_BURN_IN = "HISTORY_BURN_IN"


@dataclass(frozen=True)
class MemoryRestoreRequest:
    mode: MemoryRestoreMode
    policy_architecture_id: str
    checkpoint_id: str | None = None
    memory_tree_structure: Any = None
    batch_size: int | None = None
    dtype: str | None = None
    history_length: int | None = None
    source_episode: str | None = None
    source_timestep: int | None = None
    stop_gradient: bool | None = None


@dataclass(frozen=True)
class MemoryCompatibilityReport:
    compatible: bool
    reasons: tuple[str, ...]
    warning_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class MemoryRestoreResult:
    mode: MemoryRestoreMode
    memory: Any
    report: MemoryCompatibilityReport


def validate_memory_request(request: MemoryRestoreRequest, *, checkpoint_id: str | None = None,
                            architecture_id: str | None = None, memory_tree_structure: Any = None) -> MemoryCompatibilityReport:
    reasons: list[str] = []
    warnings: list[str] = []
    if not request.policy_architecture_id:
        reasons.append("missing policy_architecture_id")
    if request.mode is MemoryRestoreMode.ZERO_MEMORY:
        warnings.append("ZERO_MEMORY_EXPLICIT_DIAGNOSTIC_ONLY")
    elif request.mode is MemoryRestoreMode.SAVED_POLICY_MEMORY:
        if not request.checkpoint_id:
            reasons.append("missing checkpoint_id")
        if checkpoint_id is not None and request.checkpoint_id != checkpoint_id:
            reasons.append("checkpoint mismatch")
        if architecture_id is not None and request.policy_architecture_id != architecture_id:
            reasons.append("architecture mismatch")
        if request.memory_tree_structure is None or (memory_tree_structure is not None and request.memory_tree_structure != memory_tree_structure):
            reasons.append("memory tree mismatch or missing")
    elif request.mode is MemoryRestoreMode.HISTORY_BURN_IN:
        if request.history_length is None or request.history_length <= 0:
            reasons.append("invalid history_length")
        if not request.source_episode:
            reasons.append("missing history reference")
        if request.stop_gradient is None:
            reasons.append("stop_gradient must be explicit")
    return MemoryCompatibilityReport(not reasons, tuple(reasons), tuple(warnings))
