"""Deterministic feasibility classification for actual-N evidence (P0-5).

Turns measured feasibility evidence into one of six frontier classes using
versioned, threshold-explicit rules.  Everything is deterministic: the same
inputs always produce the same class, the thresholds travel with the result,
and every decision step is recorded as a reason code.

The class values are shared with ``llm_contracts.FRONTIER_CLASSES`` (checked
at import time); the LLM diagnostician may *name* a class, but the official
classification used by the evidence selector always comes from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from .errors import InvalidEvidenceError
from .llm_contracts import FRONTIER_CLASSES
from .memory_modes import MemoryCompatibilityReport
from .search_statistics import BranchOutcome, FeasibilityEstimate

CLASSIFICATION_VERSION = "frontier-classify/v1"


class FrontierClass(str, Enum):
    TOO_EASY = "TOO_EASY"
    LEARNABLE_FRONTIER = "LEARNABLE_FRONTIER"
    TOO_HARD = "TOO_HARD"
    UNCERTAIN = "UNCERTAIN"
    INVALID = "INVALID"
    MEMORY_MISMATCH_SUSPECTED = "MEMORY_MISMATCH_SUSPECTED"


if {c.value for c in FrontierClass} != set(FRONTIER_CLASSES):
    raise InvalidEvidenceError(
        "FrontierClass values drifted from llm_contracts.FRONTIER_CLASSES")


@dataclass(frozen=True)
class FrontierClassification:
    """One deterministic classification verdict (thresholds travel with it)."""

    state_id: str
    frontier_class: FrontierClass
    reason_codes: tuple[str, ...]
    thresholds: Mapping[str, float]
    failure_distribution: Mapping[str, int]
    terminal_distribution: Mapping[str, int]
    classification_version: str = CLASSIFICATION_VERSION


def _count(values: Sequence[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(value)
        out[key] = out.get(key, 0) + 1
    return out


def classify_frontier(estimate: FeasibilityEstimate, *,
                      outcomes: Sequence[BranchOutcome] = (),
                      memory_reports: Sequence[MemoryCompatibilityReport] = (),
                      memory_mode_estimates: Mapping[str, FeasibilityEstimate] | None = None,
                      min_branches_for_certainty: int = 8,
                      easy_success_threshold: float = 0.9,
                      hard_success_threshold: float = 0.05,
                      uncertainty_threshold: float = 0.5,
                      memory_mode_divergence: float = 0.25) -> FrontierClassification:
    """Deterministic, versioned classification of measured feasibility.

    Rule order (first match wins):
      1. no actual branches                 -> INVALID
      2. memory incompatibility/divergence  -> MEMORY_MISMATCH_SUSPECTED
      3. too few branches / wide CI         -> UNCERTAIN
      4. success rate >= easy threshold     -> TOO_EASY
      5. success rate <= hard threshold     -> TOO_HARD
      6. otherwise                          -> LEARNABLE_FRONTIER
    """
    if not (0.0 <= hard_success_threshold < easy_success_threshold <= 1.0):
        raise InvalidEvidenceError(
            "classification thresholds must satisfy 0 <= hard < easy <= 1")
    if int(min_branches_for_certainty) <= 0:
        raise InvalidEvidenceError("min_branches_for_certainty must be > 0")
    if not (0.0 <= uncertainty_threshold <= 1.0):
        raise InvalidEvidenceError("uncertainty_threshold must be in [0, 1]")

    state_id = estimate.state_id
    rows = list(outcomes)
    for row in rows:
        if row.state_id != state_id:
            raise InvalidEvidenceError(
                f"outcome {row.branch_id} belongs to state {row.state_id!r}, not "
                f"{state_id!r} (never mix states)")

    thresholds = {
        "min_branches_for_certainty": float(min_branches_for_certainty),
        "easy_success_threshold": float(easy_success_threshold),
        "hard_success_threshold": float(hard_success_threshold),
        "uncertainty_threshold": float(uncertainty_threshold),
        "memory_mode_divergence": float(memory_mode_divergence),
    }
    failure_distribution = _count([r.failure_category for r in rows
                                   if r.failure_category is not None])
    terminal_distribution = _count([r.terminal_event for r in rows
                                    if r.terminal_event is not None])

    def verdict(frontier_class: FrontierClass, reasons: Sequence[str]) -> FrontierClassification:
        return FrontierClassification(
            state_id=state_id,
            frontier_class=frontier_class,
            reason_codes=tuple(reasons),
            thresholds=thresholds,
            failure_distribution=failure_distribution,
            terminal_distribution=terminal_distribution,
        )

    if estimate.total_actual_branches == 0:
        return verdict(FrontierClass.INVALID, ("NO_ACTUAL_BRANCHES",))

    memory_reasons: list[str] = []
    for report in memory_reports:
        if not report.compatible:
            memory_reasons.append("MEMORY_REPORT_INCOMPATIBLE")
            break
    if memory_mode_estimates:
        measured = {mode: est for mode, est in memory_mode_estimates.items()
                    if est.total_actual_branches > 0}
        rates = sorted(est.success_rate for est in measured.values())
        if len(rates) >= 2 and (rates[-1] - rates[0]) > memory_mode_divergence:
            memory_reasons.append("MEMORY_MODE_DIVERGENCE")
    if memory_reasons:
        return verdict(FrontierClass.MEMORY_MISMATCH_SUSPECTED, memory_reasons)

    if estimate.total_actual_branches < min_branches_for_certainty:
        return verdict(FrontierClass.UNCERTAIN, ("INSUFFICIENT_ACTUAL_N",))
    if estimate.uncertainty > uncertainty_threshold:
        return verdict(FrontierClass.UNCERTAIN, ("HIGH_UNCERTAINTY",))

    if estimate.success_rate >= easy_success_threshold:
        return verdict(FrontierClass.TOO_EASY, ("SUCCESS_RATE_ABOVE_EASY_THRESHOLD",))
    if estimate.success_rate <= hard_success_threshold:
        return verdict(FrontierClass.TOO_HARD, ("SUCCESS_RATE_BELOW_HARD_THRESHOLD",))
    return verdict(FrontierClass.LEARNABLE_FRONTIER, ("SUCCESS_RATE_IN_FRONTIER_BAND",))
