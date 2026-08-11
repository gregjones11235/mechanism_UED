"""Behavioral gap scoring (task section 12, NEW dimension).

    behavioral_gap = max(0, student_behavior_failure_score
                            - reference_behavior_failure_score)

Worse (more severe anomalous behavior) => larger gap. The Student's behavior
failure score is derived DETERMINISTICALLY from the extractor's anomalies
(severity-weighted recurrence density per environment); the Reference's score
comes from the same deterministic extraction over Reference evidence admitted
through the SAME boundary (Reference trajectories are evidence, never
demonstrations). Kept strictly separate from front_regret / global_regret —
the three are NOT fused at the bottom layer.
"""
from __future__ import annotations

from typing import Dict, List

from pydantic import Field, model_validator

from d052.bagr_ued.event_extractor import AnomalyCandidate
from d052.schemas.common import CanonicalModel, validate_finite


def behavior_failure_score(anomalies: List[AnomalyCandidate]) -> float:
    """Severity-weighted anomaly load, bounded to [0, 1]."""
    if not anomalies:
        return 0.0
    load = sum(a.severity * min(a.recurrence, 10) / 10.0 for a in anomalies)
    return round(min(1.0, load / max(1, len({a.episode_id for a in anomalies}))), 6)


class BehavioralGapScore(CanonicalModel):
    environment_id: str = Field(min_length=1)
    student_behavior_failure_score: float = Field(ge=0.0, le=1.0)
    reference_behavior_failure_score: float = Field(ge=0.0, le=1.0)
    behavioral_gap: float = Field(ge=0.0, le=1.0)
    per_pattern_gap: Dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _gap_definition(self) -> "BehavioralGapScore":
        for name in ("student_behavior_failure_score",
                     "reference_behavior_failure_score", "behavioral_gap"):
            validate_finite(getattr(self, name), name)
        expected = max(0.0, round(self.student_behavior_failure_score
                                  - self.reference_behavior_failure_score, 6))
        if abs(self.behavioral_gap - expected) > 1e-9:
            raise ValueError(
                f"BEHAVIORAL_GAP_DEFINITION_VIOLATED: gap={self.behavioral_gap} "
                f"!= max(0, student - reference)={expected}")
        return self


def compute_behavioral_gaps(
        student_anomalies_by_env: Dict[str, List[AnomalyCandidate]],
        reference_failure_score_by_env: Dict[str, float]) -> List[BehavioralGapScore]:
    scores: List[BehavioralGapScore] = []
    for eid in sorted(set(student_anomalies_by_env) |
                      set(reference_failure_score_by_env)):
        anomalies = student_anomalies_by_env.get(eid, [])
        student = behavior_failure_score(anomalies)
        reference = float(reference_failure_score_by_env.get(eid, 0.0))
        per_pattern: Dict[str, List[float]] = {}
        for a in anomalies:
            per_pattern.setdefault(a.behavior_pattern, []).append(
                a.severity * min(a.recurrence, 10) / 10.0)
        scores.append(BehavioralGapScore(
            environment_id=eid,
            student_behavior_failure_score=student,
            reference_behavior_failure_score=round(min(1.0, reference), 6),
            behavioral_gap=round(max(0.0, student - reference), 6),
            per_pattern_gap={p: round(min(1.0, sum(v)), 6)
                             for p, v in sorted(per_pattern.items())}))
    return scores
