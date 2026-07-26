"""StudentProfile -> Modeler context channel + modeler-alignment bonus (Phase 2.5).

Task §StudentProfile 到 Modeler 上下文的传递. The Modeler is the FOURTH role and the
ONLY non-scoring, conditioning role: it runs once per session over the student
state and steers selection via a deterministic ``modeler_bonus`` (consumed by the
S2_FOUR_ROLE_MODELER selector). This is the channel the B/C counterfactual ablates:
arm B is modeler-OFF (bonus 0 everywhere); arm C is modeler-ON (bonus derived here).

FIREWALL (structural, source mandate): the Modeler is handed FACTS, not verdicts.
The StudentProfile's mastery TIER labels (per_depth_tier_mastery / mastery_tier)
are a deterministic downstream derivation that must NEVER reach the LLM Modeler or
the selector. So this channel passes ONLY the held-out success-rate (SR) series and
explicitly STRIPS every tier label; ``assert_modeler_firewall`` fails closed if any
tier token survives into the serialized context.

Determinism: the context_hash binds the SR series + the modeler judgment's
canonical fields + the bonus weight, and EXCLUDES tier labels by construction. The
modeler_bonus is a pure function of (candidate targets, siege_foci, SR series,
weight) -- no randomness, no network.

HONESTY NOTE: the modeler JUDGMENT consumed here is supplied by the caller. In the
offline Phase-2.5 harness it is a clearly-labeled deterministic FIXTURE (the real
Modeler CC judgment cache is pending); this module does not fabricate a judgment.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import Field, model_validator

from d052.achievements import REGISTRY
from d052.counterfactual._hash import canonical_json, sha256_hex
from d052.profiling.modeler import ModelerJudgment
from d052.profiling.student_profile import StudentProfile
from d052.schemas.candidate import Candidate
from d052.schemas.common import CanonicalModel, validate_finite, validate_sha256_hex

#: Frozen canonical weight on the modeler-alignment bonus. Part of the arm's
#: deterministic identity (bound into the manifest); NOT a tunable a run may hide.
MODELER_BONUS_WEIGHT = 0.07

#: substrings that must NEVER appear in a Modeler context (the tier-label firewall)
_FORBIDDEN_TIER_TOKENS = ("tier", "mastery_tier", "per_depth_tier", "depth_tier")


class ModelerFirewallError(Exception):
    TIER_LABEL_LEAK = "TIER_LABEL_LEAK"
    EMPTY_SR_SERIES = "EMPTY_SR_SERIES"

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"[{code}] {message}")


def student_profile_hash(profile: StudentProfile) -> str:
    """Deterministic hash of the StudentProfile's SR SERIES ONLY.

    Deliberately computed over per_achievement_sr (the machine facts) and NOT over
    per_depth_tier_mastery / counts-derived tiers: the channel identity must not
    depend on tier labels, so a tier-label change can never alter what the Modeler
    sees. Binds overall_mastery + measured_count for traceability (non-tier).
    """
    payload = {
        "per_achievement_sr": {k: float(v)
                               for k, v in sorted(profile.per_achievement_sr.items())},
        "overall_mastery": float(profile.overall_mastery),
        "measured_count": profile.measured_count,
        "evidence_source": profile.evidence_source,
    }
    return sha256_hex(payload)


class ModelerContext(CanonicalModel):
    """The firewalled context handed to the Modeler + the derived bonus rule.

    Carries the SR series (latest_sr), the modeler judgment's canonical fields,
    and the bonus weight. Carries NO tier labels (enforced). context_hash binds
    exactly these fields.
    """

    student_profile_hash: str
    latest_sr: Dict[str, float] = Field(default_factory=dict)
    num_snapshots: int = Field(ge=0)
    student_state: str = Field(min_length=1)
    recommendation: str = Field(min_length=1)
    evidence_check: str = Field(min_length=1)
    siege_foci: List[str] = Field(default_factory=list)
    modeler_bonus_weight: float = MODELER_BONUS_WEIGHT
    context_hash: str
    #: records that tier labels were stripped (audit attestation)
    firewall_attestation: Dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _verify(self) -> "ModelerContext":
        validate_sha256_hex(self.student_profile_hash, "student_profile_hash")
        validate_sha256_hex(self.context_hash, "context_hash")
        validate_finite(self.modeler_bonus_weight, "modeler_bonus_weight")
        if not (0.0 <= self.modeler_bonus_weight <= 1.0):
            raise ValueError("modeler_bonus_weight out of [0,1]")
        for name in self.latest_sr:
            REGISTRY.resolve(name)         # unknown -> AchievementError
        for name in self.siege_foci:
            REGISTRY.resolve(name)
        if self.context_hash != _context_hash_payload_hash(self):
            raise ValueError("MODELER_CONTEXT_HASH_MISMATCH")
        assert_modeler_firewall(self)      # tier-label firewall (fail closed)
        return self


def _context_hash_payload(ctx: "ModelerContext") -> dict:
    """The exact payload hashed into context_hash (tier labels EXCLUDED)."""
    return {
        "student_profile_hash": ctx.student_profile_hash,
        "latest_sr": {k: float(v) for k, v in sorted(ctx.latest_sr.items())},
        "num_snapshots": ctx.num_snapshots,
        "student_state": ctx.student_state,
        "recommendation": ctx.recommendation,
        "evidence_check": ctx.evidence_check,
        "siege_foci": sorted(ctx.siege_foci),
        "modeler_bonus_weight": float(ctx.modeler_bonus_weight),
    }


def _context_hash_payload_hash(ctx: "ModelerContext") -> str:
    return sha256_hex(_context_hash_payload(ctx))


def assert_modeler_firewall(ctx: ModelerContext) -> None:
    """Fail closed if ANY tier label leaks into the serialized Modeler context."""
    serialized = canonical_json(_context_hash_payload(ctx)).lower()
    for tok in _FORBIDDEN_TIER_TOKENS:
        if tok in serialized:
            raise ModelerFirewallError(
                ModelerFirewallError.TIER_LABEL_LEAK,
                f"tier label token {tok!r} leaked into the Modeler context; the "
                f"Modeler consumes the SR series (facts), never mastery tiers")
    if not ctx.firewall_attestation.get("tier_labels_stripped"):
        raise ModelerFirewallError(
            ModelerFirewallError.TIER_LABEL_LEAK,
            "ModelerContext missing the tier_labels_stripped attestation")


def build_modeler_context(profile: StudentProfile, judgment: ModelerJudgment,
                          *, num_snapshots: int = 1,
                          weight: float = MODELER_BONUS_WEIGHT) -> ModelerContext:
    """Build the firewalled Modeler context from a StudentProfile + judgment.

    The SR series comes from the profile's per_achievement_sr (machine facts); the
    profile's per_depth_tier_mastery is DELIBERATELY NOT carried (tier firewall).
    The judgment supplies student_state / recommendation / siege_foci / evidence.
    """
    if not profile.per_achievement_sr:
        raise ModelerFirewallError(
            ModelerFirewallError.EMPTY_SR_SERIES,
            "cannot build a Modeler context from an empty SR series")
    latest_sr = {k: float(v) for k, v in profile.per_achievement_sr.items()}
    sp_hash = student_profile_hash(profile)

    ctx = ModelerContext.model_construct(
        protocol_version="canonical_v2",
        student_profile_hash=sp_hash,
        latest_sr=latest_sr,
        num_snapshots=num_snapshots,
        student_state=judgment.student_state.value,
        recommendation=judgment.recommendation.value,
        evidence_check=judgment.evidence_check.value,
        siege_foci=list(judgment.siege_foci),
        modeler_bonus_weight=float(weight),
        context_hash="",
        firewall_attestation={
            "tier_labels_stripped": True,
            "excluded_profile_fields": ["per_depth_tier_mastery", "mastered_count",
                                        "proficient_count"],
            "channel": "student_profile_sr_series_only",
        },
    )
    object.__setattr__(ctx, "context_hash", _context_hash_payload_hash(ctx))
    # run full validation (recomputes + verifies hash, enforces firewall)
    return ModelerContext.model_validate(ctx.model_dump())


def modeler_bonus_for(candidate: Candidate, ctx: Optional[ModelerContext],
                      *, modeler_enabled: bool) -> float:
    """Deterministic modeler-alignment bonus in [0,1] for one candidate.

    modeler-OFF (arm B) or no context -> 0.0 for every candidate (the ablation
    baseline). modeler-ON (arm C) -> weight * focus_alignment, where
    focus_alignment = max over the candidate's canonical targets that appear in the
    Modeler's siege_foci of (1 - SR(target)); a candidate the Modeler did not flag
    gets 0. Lower mastery (SR) on a flagged skill => larger bonus, steering
    selection toward the student's weakest modeler-flagged skills.
    """
    if not modeler_enabled or ctx is None:
        return 0.0
    foci = set(ctx.siege_foci)
    align = 0.0
    for name in candidate.canonical_target_names:
        if name in foci:
            sr = float(ctx.latest_sr.get(name, 0.0))   # missing -> 0.0 (conservative)
            align = max(align, 1.0 - sr)
    bonus = float(ctx.modeler_bonus_weight) * align
    return min(1.0, max(0.0, bonus))
