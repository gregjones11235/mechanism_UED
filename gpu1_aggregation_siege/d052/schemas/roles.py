"""Role protocol schemas.

Four roles exist (Tutor / Critic / Explorer / Modeler). The three SCORING roles
(tutor/critic/explorer) emit a per-candidate ``RoleJudgment``; the Modeler runs
once per session over the student state and emits a ``ModelerJudgment`` (defined
in d052/profiling, Commit 4), not per-candidate scores.

``RoleJudgment`` enforces the role-specific headline score keys (so a judgment
missing the signal a selector consumes is a schema error, not a silent 0):
  tutor    -> progression_score
  critic   -> critic_penalty AND critic_reject (bool)
  explorer -> novelty_score
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import Field, model_validator

from d052.schemas.common import CanonicalModel, validate_finite


class RoleName(str, Enum):
    TUTOR = "tutor"
    CRITIC = "critic"
    EXPLORER = "explorer"
    MODELER = "modeler"


class ScoringRole(str, Enum):
    """Roles that emit a per-candidate numeric judgment."""

    TUTOR = "tutor"
    CRITIC = "critic"
    EXPLORER = "explorer"


#: required headline score key per scoring role
HEADLINE_SCORE_KEY: Dict[ScoringRole, str] = {
    ScoringRole.TUTOR: "progression_score",
    ScoringRole.CRITIC: "critic_penalty",
    ScoringRole.EXPLORER: "novelty_score",
}


class RoleJudgment(CanonicalModel):
    """One role's judgment of one candidate (strict-JSON LLM output contract)."""

    role: ScoringRole
    candidate_id: str = Field(min_length=1)
    #: role headline + any auxiliary numeric scores; all values finite
    scores: Dict[str, float]
    #: required for critic (the hard-veto bit); forbidden-absent for critic
    critic_reject: Optional[bool] = None
    #: free-text rationale (audit trail; never parsed for scoring)
    rationale: str = ""
    #: optional provenance of the judgment (provider/exact model/prompt version)
    provider: Optional[str] = None
    exact_model_id: Optional[str] = None
    prompt_version: Optional[str] = None

    @model_validator(mode="after")
    def _validate_scores(self) -> "RoleJudgment":
        if not self.scores:
            raise ValueError("EMPTY_SCORES: scores must not be empty")
        for k, v in self.scores.items():
            validate_finite(v, f"scores.{k}")
        headline = HEADLINE_SCORE_KEY[self.role]
        if headline not in self.scores:
            raise ValueError(
                f"MISSING_HEADLINE_SCORE: role={self.role.value} requires "
                f"scores['{headline}']")
        if self.role is ScoringRole.CRITIC and self.critic_reject is None:
            raise ValueError(
                "MISSING_CRITIC_REJECT: critic judgments must set critic_reject")
        if self.role is not ScoringRole.CRITIC and self.critic_reject is not None:
            raise ValueError(
                "UNEXPECTED_CRITIC_REJECT: only critic sets critic_reject")
        return self

    @property
    def headline_score(self) -> float:
        return float(self.scores[HEADLINE_SCORE_KEY[self.role]])


class NormalizedEntry(CanonicalModel):
    """One candidate's per-role raw + rank_percentile_v1 normalized score."""

    candidate_id: str = Field(min_length=1)
    raw: float
    normalized: float = Field(ge=0.0, le=1.0)
    rank: int = Field(ge=0)
    #: 0-based group index for deterministic tie handling (same raw -> same group)
    tie_group: int = Field(ge=0)

    @model_validator(mode="after")
    def _finite(self) -> "NormalizedEntry":
        validate_finite(self.raw, "raw")
        validate_finite(self.normalized, "normalized")
        return self


class NormalizedRoleScores(CanonicalModel):
    """Per-role normalized score column (output of rank_percentile_v1).

    ``normalized`` is strictly in [0,1]; raw is preserved; ties are deterministic
    (same raw -> same tie_group/rank). No reverse-weighting from outcomes.
    """

    role: ScoringRole
    normalization: str = "rank_percentile_v1"
    entries: List[NormalizedEntry]

    @model_validator(mode="after")
    def _validate_entries(self) -> "NormalizedRoleScores":
        ids = [e.candidate_id for e in self.entries]
        if len(set(ids)) != len(ids):
            raise ValueError("DUPLICATE_CANDIDATE: candidate_id repeats in column")
        if self.normalization != "rank_percentile_v1":
            raise ValueError(
                f"UNKNOWN_NORMALIZATION: expected rank_percentile_v1, "
                f"got {self.normalization!r}")
        return self
