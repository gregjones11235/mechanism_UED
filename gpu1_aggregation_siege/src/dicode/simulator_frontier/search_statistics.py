"""Actual-N branch outcome statistics; no extrapolated N in success_rate."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class BranchOutcome:
    branch_id: str
    state_id: str
    search_source: str
    rng_seed: int
    horizon: int
    transitions_used: int
    success: bool
    progress: float
    terminal_event: str | None
    failure_category: str | None
    memory_mode: str
    outcome_hash: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.branch_id or not self.state_id or self.horizon < 0 or self.transitions_used < 0:
            raise ValueError("invalid branch outcome identity or budget")
        if not math.isfinite(float(self.progress)):
            raise ValueError("progress must be finite")


@dataclass(frozen=True)
class FeasibilityEstimate:
    state_id: str
    total_actual_branches: int
    actual_branches_by_source: Mapping[str, int]
    successes: int
    success_rate: float
    confidence_interval: tuple[float, float]
    mean_progress: float
    max_progress: float
    transition_cost: int
    uncertainty: float
    estimate_version: str = "actual-n-wilson-v1"
    budget_curve: tuple[Mapping[str, Any], ...] = ()


def _wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - margin), min(1.0, centre + margin)


def estimate_feasibility(outcomes: Sequence[BranchOutcome], *, state_id: str | None = None) -> FeasibilityEstimate:
    rows = list(outcomes)
    if state_id is not None:
        rows = [r for r in rows if r.state_id == state_id]
    if not rows:
        sid = state_id or ""
        return FeasibilityEstimate(sid, 0, {}, 0, 0.0, (0.0, 1.0), 0.0, 0.0, 0, 1.0)
    seen = set()
    for row in rows:
        if row.branch_id in seen:
            raise ValueError(f"duplicate branch_id: {row.branch_id}")
        seen.add(row.branch_id)
    sid_set = {r.state_id for r in rows}
    if len(sid_set) != 1:
        raise ValueError("estimate_feasibility requires one state_id")
    successes = sum(bool(r.success) for r in rows)
    n = len(rows)
    ci = _wilson(successes, n)
    by_source: dict[str, int] = {}
    for row in rows:
        by_source[row.search_source] = by_source.get(row.search_source, 0) + 1
    costs = sum(r.transitions_used for r in rows)
    return FeasibilityEstimate(rows[0].state_id, n, by_source, successes, successes / n, ci,
                               sum(r.progress for r in rows) / n, max(r.progress for r in rows), costs,
                               ci[1] - ci[0])
