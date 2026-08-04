"""Evidence-owned deterministic selector: the OFFICIAL final authority (P0-5).

Production selection is a pure, deterministic function of the measured
evidence — feasibility class, Student/Reference success and progress,
Student–Reference gap, actual_N, uncertainty, transition cost, memory
compatibility, bucket diversity, global retention and anchor coverage.

LLM outputs are advisory only: the typed planner proposes, this selector
decides.  There is NO priority_score input surface anywhere on the official
path; ``invocation_gate.deterministic_select`` (LLM-score ranking) is kept
strictly as an ablation/consultative surface and never decides a production
window.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .errors import InvalidEvidenceError
from .feasibility_classifier import FrontierClass
from .llm_contracts import PlannerOutput

SELECTOR_VERSION = "evidence-select/v1"

# Memory statuses that may enter a production window.  ZERO_MEMORY ablation
# status is deliberately absent: zero memory can never back a production run.
PRODUCTION_MEMORY_STATUSES = frozenset({
    "COMPATIBLE",
    "SAVED_POLICY_MEMORY_VERIFIED",
    "HISTORY_BURN_IN_VERIFIED",
})

# Deterministic mixed-start weights for the one-update window: the share of
# restored-frontier starts vs standard-reset starts, by frontier class.
FRONTIER_START_WEIGHTS = {
    FrontierClass.LEARNABLE_FRONTIER: 0.75,
    FrontierClass.TOO_EASY: 0.25,
    FrontierClass.TOO_HARD: 0.25,
}


@dataclass(frozen=True)
class SelectionEvidence:
    """The full evidence vector the official selector consumes."""

    state_id: str
    feasibility_class: FrontierClass
    student_success_rate: float
    student_mean_progress: float
    reference_success_rate: float
    reference_mean_progress: float
    student_reference_gap: float
    actual_n: int
    uncertainty: float
    transition_cost: int
    memory_compatibility_status: str
    bucket_diversity: int
    global_retention_ok: bool
    anchor_coverage_ok: bool
    evidence_hash: str

    def __post_init__(self) -> None:
        if not str(self.state_id).strip():
            raise InvalidEvidenceError("SelectionEvidence.state_id is empty")
        if not isinstance(self.feasibility_class, FrontierClass):
            raise InvalidEvidenceError(
                f"feasibility_class must be a FrontierClass, got {self.feasibility_class!r}")
        for name in ("student_success_rate", "student_mean_progress",
                     "reference_success_rate", "reference_mean_progress"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise InvalidEvidenceError(f"{name} must be a finite number in [0, 1]")
        expected_gap = float(self.reference_success_rate) - float(self.student_success_rate)
        if abs(float(self.student_reference_gap) - expected_gap) > 1e-9:
            raise InvalidEvidenceError(
                "student_reference_gap must equal reference_success_rate - "
                "student_success_rate (never self-reported)")
        if not isinstance(self.actual_n, int) or isinstance(self.actual_n, bool) \
                or self.actual_n < 0:
            raise InvalidEvidenceError("actual_n must be an int >= 0")
        if isinstance(self.uncertainty, bool) or not isinstance(self.uncertainty, (int, float)) \
                or not math.isfinite(float(self.uncertainty)) or float(self.uncertainty) < 0.0:
            raise InvalidEvidenceError("uncertainty must be a finite number >= 0")
        if not isinstance(self.transition_cost, int) or isinstance(self.transition_cost, bool) \
                or self.transition_cost < 0:
            raise InvalidEvidenceError("transition_cost must be an int >= 0")
        if not str(self.memory_compatibility_status).strip():
            raise InvalidEvidenceError("memory_compatibility_status is empty")
        if not isinstance(self.bucket_diversity, int) or isinstance(self.bucket_diversity, bool) \
                or self.bucket_diversity < 0:
            raise InvalidEvidenceError("bucket_diversity must be an int >= 0")
        if not str(self.evidence_hash).strip():
            raise InvalidEvidenceError("SelectionEvidence.evidence_hash is empty")


@dataclass(frozen=True)
class EvidenceSelectionResult:
    """The selector's binding decision (reproducible via selection_hash)."""

    accepted: bool
    plan_id: str
    frontier_start_weight: float
    reason_codes: tuple[str, ...]
    selection_hash: str
    selector_version: str = SELECTOR_VERSION


def _selection_hash(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def evidence_based_select(plan: PlannerOutput, *,
                          evidence: SelectionEvidence) -> EvidenceSelectionResult:
    """OFFICIAL final selection authority: pure deterministic evidence rules.

    Rejects (never executes) when evidence is INVALID, when memory mismatch
    is suspected, or when the evidence is UNCERTAIN (UNCERTAIN may only
    request more evidence).  Accepts TOO_EASY / TOO_HARD / LEARNABLE_FRONTIER
    with class-dependent mixed-start weights.  Missing actual-N evidence,
    unbound anchors, a violated retention contract or a non-production
    memory status are hard rejections.  The decision hash binds evidence +
    plan so the choice is reproducible.
    """
    if not isinstance(plan, PlannerOutput):
        raise InvalidEvidenceError(
            "evidence_based_select only accepts a typed PlannerOutput "
            "(arbitrary mappings — and any priority_score surface — are refused)")
    if not isinstance(evidence, SelectionEvidence):
        raise InvalidEvidenceError("evidence_based_select requires SelectionEvidence")

    reasons: list[str] = []
    frontier_class = evidence.feasibility_class

    if frontier_class is FrontierClass.INVALID:
        reasons.append("INVALID_EVIDENCE")
    if frontier_class is FrontierClass.MEMORY_MISMATCH_SUSPECTED:
        reasons.append("MEMORY_MISMATCH_SUSPECTED")
    if frontier_class is FrontierClass.UNCERTAIN:
        reasons.append("REQUIRES_MORE_EVIDENCE")
    if evidence.actual_n <= 0:
        reasons.append("NO_ACTUAL_N_EVIDENCE")
    if not evidence.anchor_coverage_ok:
        reasons.append("ANCHORS_UNBOUND")
    if not evidence.global_retention_ok:
        reasons.append("RETENTION_CONTRACT_VIOLATED")
    if evidence.memory_compatibility_status not in PRODUCTION_MEMORY_STATUSES:
        reasons.append("MEMORY_STATUS_NOT_PRODUCTION")
    if plan.memory_mode == "ZERO_MEMORY":
        reasons.append("ZERO_MEMORY_NOT_A_PRODUCTION_MODE")

    rejected = bool(reasons)
    if rejected:
        accepted = False
        frontier_start_weight = 0.0
    else:
        accepted = True
        frontier_start_weight = FRONTIER_START_WEIGHTS[frontier_class]
        reasons.append(f"ACCEPTED_{frontier_class.value}")
        if plan.actual_n > evidence.actual_n:
            # The planner may not spend more actual-N than evidence measured.
            accepted = False
            frontier_start_weight = 0.0
            reasons.append("PLANNER_ACTUAL_N_EXCEEDS_EVIDENCE")

    payload = {
        "selector_version": SELECTOR_VERSION,
        "evidence": asdict(evidence),
        "plan_id": plan.plan_id,
        "plan_hash": plan.plan_hash,
        "accepted": accepted,
        "frontier_start_weight": frontier_start_weight,
        "reason_codes": tuple(reasons),
    }
    return EvidenceSelectionResult(
        accepted=accepted,
        plan_id=plan.plan_id,
        frontier_start_weight=frontier_start_weight,
        reason_codes=tuple(reasons),
        selection_hash=_selection_hash(payload),
    )
