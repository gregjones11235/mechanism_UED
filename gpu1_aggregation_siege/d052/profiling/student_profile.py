"""Deterministic Student profile from held-out success rates.

Reuses the semantics of gpu1_aggregation_siege/src/dicode/siege/student_profile.py
(StudentProfileLog tier engine; REUSE_AS_IS): tiers from held-out SR thresholds
0.1 / 0.5 / 0.8 / 0.95, is_proficient / is_mastered, per-depth-tier mastery.

Deterministic and conservative: an achievement ABSENT from the held-out map is
treated as SR 0.0 (NOT mastered), matching the source's SAFE default
(missing_is_mastered=False) so the profile never waves through an unmeasured skill.

Tier labels are derived HERE (deterministic machine-facts) and are NEVER handed to
the LLM Modeler / selector (source mandate) -- the Modeler consumes the SR series,
not these tier labels.
"""
from __future__ import annotations

from typing import Dict, List, Mapping

from pydantic import Field, model_validator

from d052.achievements import REGISTRY
from d052.schemas.common import CanonicalModel, validate_finite

# held-out SR thresholds (source: student_profile.py)
TIER_THRESHOLDS = (0.1, 0.5, 0.8, 0.95)
PROFICIENT_SR = 0.8
MASTERED_SR = 0.95


def mastery_tier(sr: float) -> int:
    """0..4 from held-out SR (tier 4 == mastered at >= 0.95)."""
    sr = float(sr)
    tier = 0
    for t in TIER_THRESHOLDS:
        if sr >= t:
            tier += 1
        else:
            break
    return tier


def is_proficient(sr: float) -> bool:
    return float(sr) >= PROFICIENT_SR


def is_mastered(sr: float) -> bool:
    return float(sr) >= MASTERED_SR


class StudentProfile(CanonicalModel):
    """Deterministic snapshot of student mastery over the 67 achievements."""

    per_achievement_sr: Dict[str, float] = Field(default_factory=dict)
    per_depth_tier_mastery: Dict[int, float] = Field(default_factory=dict)
    overall_mastery: float = 0.0
    mastered_count: int = 0
    proficient_count: int = 0
    measured_count: int = 0
    evidence_source: str = "held_out_evaluation"

    @model_validator(mode="after")
    def _bounds(self) -> "StudentProfile":
        validate_finite(self.overall_mastery, "overall_mastery")
        if not (0.0 <= self.overall_mastery <= 1.0):
            raise ValueError("overall_mastery out of [0,1]")
        for t, m in self.per_depth_tier_mastery.items():
            validate_finite(m, f"per_depth_tier_mastery.{t}")
            if not (0.0 <= m <= 1.0):
                raise ValueError(f"tier {t} mastery out of [0,1]")
        return self


def build_student_profile(per_achievement_sr: Mapping[str, float]) -> StudentProfile:
    """Build a deterministic StudentProfile.

    Keys must be canonical achievement names (aliases resolved; unknown -> error).
    Missing achievements count as SR 0.0 over the FULL 67 (conservative).
    """
    resolved: Dict[str, float] = {}
    for name, sr in per_achievement_sr.items():
        canon = REGISTRY.resolve(name)  # unknown -> AchievementError
        validate_finite(sr, f"sr[{name}]")
        if not (0.0 <= float(sr) <= 1.0):
            raise ValueError(f"sr[{name}] out of [0,1]: {sr}")
        resolved[canon] = float(sr)

    # full-universe view (missing -> 0.0, conservative)
    all_sr = {name: resolved.get(name, 0.0) for name in REGISTRY.names}

    # per depth tier (1..4) mean mastery
    tier_acc: Dict[int, List[float]] = {}
    for name, sr in all_sr.items():
        tier_acc.setdefault(REGISTRY.depth_tier(name), []).append(sr)
    per_tier = {t: (sum(v) / len(v) if v else 0.0) for t, v in tier_acc.items()}

    overall = sum(all_sr.values()) / len(all_sr)
    return StudentProfile(
        per_achievement_sr=resolved,
        per_depth_tier_mastery=per_tier,
        overall_mastery=overall,
        mastered_count=sum(1 for v in all_sr.values() if is_mastered(v)),
        proficient_count=sum(1 for v in all_sr.values() if is_proficient(v)),
        measured_count=len(resolved),
    )
