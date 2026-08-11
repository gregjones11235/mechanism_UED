"""Learnability scoring (task sections 1 / 12).

Mock, deterministic learnability proxy: an environment is most learnable when
the Student's success rate sits in the middle of its range (neither solved
nor hopeless). This round this is a LABELED MOCK signal; a real learnability
estimate requires rollout evidence across Student snapshots.
"""
from __future__ import annotations

from typing import Dict, List

from pydantic import Field

from d052.schemas.common import CanonicalModel, validate_finite


class LearnabilityScore(CanonicalModel):
    environment_id: str = Field(min_length=1)
    learnability: float = Field(ge=0.0, le=1.0)
    basis: str = "mock: 1 - 2*|success_rate - 0.5| (dry run; requires real " \
                 "rollout evidence for production)"


def learnability_from_success_rate(success_rate: float) -> float:
    validate_finite(success_rate, "success_rate")
    if not 0.0 <= success_rate <= 1.0:
        raise ValueError(f"SUCCESS_RATE_RANGE: {success_rate}")
    return round(1.0 - 2.0 * abs(success_rate - 0.5), 6)


def compute_learnability(success_rate_by_env: Dict[str, float]
                         ) -> List[LearnabilityScore]:
    return [LearnabilityScore(
        environment_id=eid,
        learnability=learnability_from_success_rate(rate))
        for eid, rate in sorted(success_rate_by_env.items())]
