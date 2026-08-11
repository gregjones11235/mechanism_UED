"""Regret scorers: front_regret + global_regret (task sections 1 / 12 / 13).

Two SEPARATE scorers, two SEPARATE outputs — front_regret (the Tier3 FRONT
bottleneck signal) and global_regret (GLOBAL training scope) are NEVER fused
in this module. Fusion, if ever, happens only inside Soft Copeland under
explicit versioned weights.

This round is a dry run: scores are computed from MOCK rollout evidence
(RegretEvidence), labeled as such, and must be validated by REAL rollout
evidence before any real curriculum decision (final environment value is
never LLM judgment alone).

Boundary: both the Student evidence and the Reference evidence enter through
the same admissible-source boundary; Reference trajectories are evidence for
a regret BASELINE, never demonstrations (TrajectorySupervisionGuard applies
to anything emitted downstream).
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, List

from pydantic import Field, model_validator

from d052.bagr_ued import constants as C
from d052.bagr_ued.trajectory_evidence import EvidenceSource
from d052.schemas.common import CanonicalModel, validate_finite


class ScenarioScope(str, Enum):
    FRONT = "front"      # Tier3 FRONT bottleneck scenarios
    GLOBAL = "global"    # GLOBAL scope scenarios


class RegretEvidence(CanonicalModel):
    """Mock rollout evidence for one environment in one scope (dry run)."""

    environment_id: str = Field(min_length=1)
    scope: ScenarioScope
    student_success_rate: float = Field(ge=0.0, le=1.0)
    reference_success_rate: float = Field(ge=0.0, le=1.0)
    severity_weight: float = Field(gt=0.0, le=1.0)
    source: EvidenceSource = EvidenceSource.SYNTHETIC_TEST_TRACE
    mock: bool = True    # this round: always mock evidence

    @model_validator(mode="after")
    def _admissible(self) -> "RegretEvidence":
        validate_finite(self.student_success_rate, "student_success_rate")
        validate_finite(self.reference_success_rate, "reference_success_rate")
        if not self.source.admissible:
            raise ValueError(
                f"FORBIDDEN_REGRET_SOURCE: {self.source.value} (formal "
                f"evaluation evidence may not drive regret)")
        return self


class RegretScores(CanonicalModel):
    """front and global regret kept as DISTINCT fields (never merged here)."""

    front_regret: float = Field(ge=0.0)
    global_regret: float = Field(ge=0.0)
    per_environment: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    evidence_scope_counts: Dict[str, int] = Field(default_factory=dict)
    fused: bool = False  # structural assertion: these are NOT one number


class _RegretScorerBase:
    scope: ScenarioScope

    def _aggregate(self, evidences: List[RegretEvidence]) -> Dict[str, float]:
        per_env: Dict[str, List[float]] = {}
        for e in evidences:
            if e.scope is not self.scope:
                continue
            gap = max(0.0, e.reference_success_rate - e.student_success_rate)
            per_env.setdefault(e.environment_id, []).append(
                gap * e.severity_weight)
        return {eid: round(sum(v) / len(v), 6) for eid, v in per_env.items()}


class FrontRegretScorer(_RegretScorerBase):
    scope = ScenarioScope.FRONT

    def score(self, evidences: List[RegretEvidence]) -> Dict[str, float]:
        return self._aggregate(evidences)


class GlobalRegretScorer(_RegretScorerBase):
    scope = ScenarioScope.GLOBAL

    def score(self, evidences: List[RegretEvidence]) -> Dict[str, float]:
        return self._aggregate(evidences)


def combined_regret_scores(evidences: List[RegretEvidence]) -> RegretScores:
    """Convenience composition that STILL keeps the two values separate."""
    front = FrontRegretScorer().score(evidences)
    glob = GlobalRegretScorer().score(evidences)
    envs = sorted(set(front) | set(glob))
    per_env = {eid: {"front_regret": front.get(eid, 0.0),
                     "global_regret": glob.get(eid, 0.0)} for eid in envs}
    front_mean = round(sum(front.values()) / len(front), 6) if front else 0.0
    glob_mean = round(sum(glob.values()) / len(glob), 6) if glob else 0.0
    counts = {"front": sum(1 for e in evidences if e.scope is ScenarioScope.FRONT),
              "global": sum(1 for e in evidences if e.scope is ScenarioScope.GLOBAL)}
    return RegretScores(front_regret=front_mean, global_regret=glob_mean,
                        per_environment=per_env, evidence_scope_counts=counts)


class MockReferenceEvidenceAdapter:
    """Mock Reference statistics (NOT a demonstration).

    Provides a Reference behavior baseline for regret computation. It must
    never emit an action sequence — the TrajectorySupervisionGuard rejects
    such keys in anything this package outputs.
    """

    def reference_stats(self, environment_ids: List[str], *,
                        scope: ScenarioScope,
                        success_rate: float = 0.85) -> List[RegretEvidence]:
        return [RegretEvidence(
            environment_id=eid, scope=scope,
            student_success_rate=success_rate,   # placeholder; real student
            reference_success_rate=success_rate,  # stats come from the caller
            severity_weight=1.0) for eid in environment_ids]


GLOBAL_SCOPE_REQUIRED = C.GLOBAL_SIGNAL_REQUIRED
