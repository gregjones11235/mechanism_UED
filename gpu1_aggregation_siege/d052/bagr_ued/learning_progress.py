"""Learning progress scoring (task sections 1 / 12).

LP = improvement of the Student on an environment between two review windows
(failure-score reduction), clamped to [0, 1]. With a single window (this
round's dry run) LP is honestly 0.0 with has_history=false — no fabricated
progress. Deterministic.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

from pydantic import Field

from d052.schemas.common import CanonicalModel, validate_finite


class LearningProgressScore(CanonicalModel):
    environment_id: str = Field(min_length=1)
    learning_progress: float = Field(ge=0.0, le=1.0)
    has_history: bool
    windows_seen: int = Field(ge=0)


def compute_learning_progress(
        failure_history_by_env: Dict[str, Sequence[float]]
        ) -> List[LearningProgressScore]:
    """history: per-environment failure scores in window order (oldest first)."""
    out: List[LearningProgressScore] = []
    for eid in sorted(failure_history_by_env):
        hist = list(failure_history_by_env[eid])
        for v in hist:
            validate_finite(v, f"failure_score[{eid}]")
        if len(hist) < 2:
            out.append(LearningProgressScore(
                environment_id=eid, learning_progress=0.0,
                has_history=False, windows_seen=len(hist)))
            continue
        delta = float(hist[-2]) - float(hist[-1])   # improvement = reduction
        out.append(LearningProgressScore(
            environment_id=eid,
            learning_progress=round(max(0.0, min(1.0, delta)), 6),
            has_history=True, windows_seen=len(hist)))
    return out
