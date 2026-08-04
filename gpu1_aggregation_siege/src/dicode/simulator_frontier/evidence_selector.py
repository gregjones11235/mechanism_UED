"""Evidence-owned deterministic selector: the OFFICIAL final authority (P0-5).

Production selection is a pure, deterministic function of the measured
evidence — feasibility class, Student/Reference success and progress,
Student–Reference gap, actual_N (per source), uncertainty, transition cost,
memory compatibility, bucket diversity, global retention and anchor coverage.

CC4 follow-up (P0-7) — simulator-owned selection authority:

- ``SelectionEvidence`` is MINT-ONLY.  Its ``evidence_hash`` has no
  constructor parameter; the hash is recomputed inside the dataclass from
  the measured fields themselves, and the official minter
  (``mint_selection_evidence_from_outcomes``) derives every rate, progress,
  per-source count and cost directly from the attested ``BranchOutcome``
  rows.  Self-reported evidence is therefore structurally impossible, and
  ``verify_selection_evidence`` re-derives the hash to reject tampering and
  refuses plain Mappings outright.
- The decision rules actually consume gap, progress, transition cost and
  bucket diversity (previously ignored inputs).
- TOO_EASY and TOO_HARD states are NEVER admitted to a training window:
  only LEARNABLE_FRONTIER carries a start weight.

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
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Mapping, Sequence

from .branch_search_runner import (
    SEARCH_SOURCE_REFERENCE_POLICY,
    SEARCH_SOURCE_STUDENT_DETERMINISTIC,
    SEARCH_SOURCE_STUDENT_STOCHASTIC,
)
from .errors import InvalidEvidenceError
from .feasibility_classifier import FrontierClass
from .llm_contracts import PlannerOutput
from .search_statistics import BranchOutcome, estimate_feasibility

SELECTOR_VERSION = "evidence-select/v2"
SELECTION_EVIDENCE_SCHEMA = "simulator_frontier.selection-evidence/v1"

# Memory statuses that may enter a production window.  ZERO_MEMORY ablation
# status is deliberately absent: zero memory can never back a production run.
PRODUCTION_MEMORY_STATUSES = frozenset({
    "COMPATIBLE",
    "SAVED_POLICY_MEMORY_VERIFIED",
    "HISTORY_BURN_IN_VERIFIED",
})

# Deterministic mixed-start weights for the one-update window: the share of
# restored-frontier starts vs standard-reset starts, by frontier class.
# CC4 follow-up (P0-7): TOO_EASY / TOO_HARD deliberately carry NO weight —
# they are never training targets, so they can never be admitted.
FRONTIER_START_WEIGHTS = {
    FrontierClass.LEARNABLE_FRONTIER: 0.75,
}

# Minimum number of distinct buckets the archive must already cover before a
# mixed-start frontier window may be admitted (diversity is binding input).
MIN_BUCKET_DIVERSITY = 2

# If the Student already beats the Reference by at least this success-rate
# margin (and Reference evidence exists), the state is no longer a frontier
# target for the Student.
STUDENT_DOMINANCE_GAP_THRESHOLD = 0.5

_STUDENT_SOURCES = frozenset({
    SEARCH_SOURCE_STUDENT_DETERMINISTIC,
    SEARCH_SOURCE_STUDENT_STOCHASTIC,
})


def _evidence_hash(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _evidence_payload(evidence: "SelectionEvidence") -> dict:
    """Canonical payload over every measured field (hash field excluded).

    Built field-by-field instead of via ``asdict`` so it works inside
    ``__post_init__`` before the derived hash exists.
    """
    return {f.name: getattr(evidence, f.name)
            for f in fields(evidence) if f.name != "evidence_hash"}


@dataclass(frozen=True)
class SelectionEvidence:
    """The full evidence vector the official selector consumes.

    MINT-ONLY: ``evidence_hash`` has no constructor parameter — it is
    recomputed from the measured fields in ``__post_init__``, so a caller
    can never supply (or forge) the signature of its own evidence.
    """

    state_id: str
    feasibility_class: FrontierClass
    student_success_rate: float
    student_mean_progress: float
    reference_success_rate: float
    reference_mean_progress: float
    student_reference_gap: float
    actual_n: int
    student_actual_n: int
    reference_actual_n: int
    uncertainty: float
    transition_cost: int
    memory_compatibility_status: str
    bucket_diversity: int
    global_retention_ok: bool
    anchor_coverage_ok: bool
    evidence_schema: str = SELECTION_EVIDENCE_SCHEMA
    evidence_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not str(self.state_id).strip():
            raise InvalidEvidenceError("SelectionEvidence.state_id is empty")
        if not isinstance(self.feasibility_class, FrontierClass):
            raise InvalidEvidenceError(
                f"feasibility_class must be a FrontierClass, got {self.feasibility_class!r}")
        if self.evidence_schema != SELECTION_EVIDENCE_SCHEMA:
            raise InvalidEvidenceError(
                f"evidence_schema must be {SELECTION_EVIDENCE_SCHEMA!r} (never self-issued)")
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
        for name in ("actual_n", "student_actual_n", "reference_actual_n",
                     "transition_cost", "bucket_diversity"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise InvalidEvidenceError(f"{name} must be an int >= 0")
        if self.actual_n != self.student_actual_n + self.reference_actual_n:
            raise InvalidEvidenceError(
                "actual_n must equal student_actual_n + reference_actual_n "
                "(per-source counts are binding, never re-declared)")
        if isinstance(self.uncertainty, bool) or not isinstance(self.uncertainty, (int, float)) \
                or not math.isfinite(float(self.uncertainty)) or float(self.uncertainty) < 0.0:
            raise InvalidEvidenceError("uncertainty must be a finite number >= 0")
        if not str(self.memory_compatibility_status).strip():
            raise InvalidEvidenceError("memory_compatibility_status is empty")
        # The hash is DERIVED, never accepted: recompute it from exactly the
        # measured fields above (evidence_hash itself is excluded).
        object.__setattr__(self, "evidence_hash", _evidence_hash(_evidence_payload(self)))


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


def verify_selection_evidence(evidence: Any) -> None:
    """Re-derive the evidence hash and reject any tampering or fake surface.

    Plain Mappings are refused outright (a dict is not evidence), foreign
    types are refused, and any field mutation after minting breaks the
    recomputed hash.
    """
    if isinstance(evidence, Mapping):
        raise InvalidEvidenceError(
            "plain mappings are not SelectionEvidence (evidence is mint-only; "
            "a hand-built mapping can never back a production selection)")
    if not isinstance(evidence, SelectionEvidence):
        raise InvalidEvidenceError(
            f"selection evidence must be a minted SelectionEvidence, got {type(evidence).__name__}")
    if evidence.evidence_schema != SELECTION_EVIDENCE_SCHEMA:
        raise InvalidEvidenceError("selection evidence schema mismatch")
    recomputed = _evidence_hash(_evidence_payload(evidence))
    if recomputed != evidence.evidence_hash:
        raise InvalidEvidenceError(
            "selection evidence hash mismatch (evidence was modified after "
            "minting; tampering is rejected fail-closed)")


def mint_selection_evidence_from_outcomes(
        *, state_id: str,
        frontier_class: FrontierClass,
        outcomes: Sequence[BranchOutcome],
        retention_ok: bool,
        anchor_coverage_ok: bool,
        bucket_diversity: int) -> SelectionEvidence:
    """Mint the selection evidence straight from attested branch outcomes.

    Every measured quantity — per-source rates, mean progress, per-source
    actual-N counts, gap, uncertainty and the total transition cost — is
    recomputed here from the ``BranchOutcome`` rows; none of it can be
    supplied by the caller.  Mixed memory statuses across branches, unknown
    search sources, or an empty Student branch set raise (fail closed).
    """
    rows = list(outcomes)
    if not rows:
        raise InvalidEvidenceError(
            "selection evidence requires at least one attested branch outcome")
    student_rows = [o for o in rows if o.search_source in _STUDENT_SOURCES]
    reference_rows = [o for o in rows
                      if o.search_source == SEARCH_SOURCE_REFERENCE_POLICY]
    if len(student_rows) + len(reference_rows) != len(rows):
        unknown = sorted({o.search_source for o in rows}
                         - _STUDENT_SOURCES - {SEARCH_SOURCE_REFERENCE_POLICY})
        raise InvalidEvidenceError(
            f"unknown search source(s) {unknown!r} in branch outcomes "
            "(never fold unclassified evidence into selection)")
    if not student_rows:
        raise InvalidEvidenceError(
            "selection evidence requires attested Student branches "
            "(the training frontier is never classified from Reference-only runs)")
    statuses = {str(o.memory_compatibility_status) for o in rows}
    if len(statuses) != 1:
        raise InvalidEvidenceError(
            f"branch outcomes carry mixed memory statuses {sorted(statuses)!r}; "
            "selection evidence requires one resolved status (fail closed)")

    def _rate(group: list[BranchOutcome]) -> float:
        return sum(bool(o.success) for o in group) / len(group) if group else 0.0

    def _mean_progress(group: list[BranchOutcome]) -> float:
        return sum(float(o.progress) for o in group) / len(group) if group else 0.0

    student_rate = _rate(student_rows)
    reference_rate = _rate(reference_rows)
    # Uncertainty comes from the Student Wilson estimate (the frontier class
    # is also Student-derived); the transition cost is the measured spend of
    # ALL executed branches.
    student_estimate = estimate_feasibility(student_rows)
    return SelectionEvidence(
        state_id=state_id,
        feasibility_class=frontier_class,
        student_success_rate=student_rate,
        student_mean_progress=_mean_progress(student_rows),
        reference_success_rate=reference_rate,
        reference_mean_progress=_mean_progress(reference_rows),
        student_reference_gap=reference_rate - student_rate,
        actual_n=len(rows),
        student_actual_n=len(student_rows),
        reference_actual_n=len(reference_rows),
        uncertainty=float(student_estimate.uncertainty),
        transition_cost=int(sum(o.transitions_used for o in rows)),
        memory_compatibility_status=statuses.pop(),
        bucket_diversity=int(bucket_diversity),
        global_retention_ok=bool(retention_ok),
        anchor_coverage_ok=bool(anchor_coverage_ok),
    )


def evidence_based_select(plan: PlannerOutput, *,
                          evidence: SelectionEvidence) -> EvidenceSelectionResult:
    """OFFICIAL final selection authority: pure deterministic evidence rules.

    Rejects (never executes) when evidence is INVALID, when memory mismatch
    is suspected, when the evidence is UNCERTAIN (UNCERTAIN may only request
    more evidence), or when the state is TOO_EASY / TOO_HARD (they are never
    training targets).  Only LEARNABLE_FRONTIER may be admitted, and only
    when every measured input survives the deterministic rules:

    - gap: with Reference evidence present, a Student dominance margin of at
      least ``STUDENT_DOMINANCE_GAP_THRESHOLD`` means the state is no longer
      a frontier target;
    - progress: the Student must show at least one measured sign of progress
      (a success or positive mean progress);
    - cost: the measured transition spend may not exceed the plan's declared
      transition budget (``plan.horizon * plan.actual_n``);
    - diversity: the archive must already cover at least
      ``MIN_BUCKET_DIVERSITY`` buckets.

    Missing actual-N evidence, unbound anchors, a violated retention contract
    or a non-production memory status are hard rejections.  The decision hash
    binds evidence + plan so the choice is reproducible.
    """
    if not isinstance(plan, PlannerOutput):
        raise InvalidEvidenceError(
            "evidence_based_select only accepts a typed PlannerOutput "
            "(arbitrary mappings — and any priority_score surface — are refused)")
    verify_selection_evidence(evidence)

    reasons: list[str] = []
    frontier_class = evidence.feasibility_class

    if frontier_class is FrontierClass.INVALID:
        reasons.append("INVALID_EVIDENCE")
    if frontier_class is FrontierClass.MEMORY_MISMATCH_SUSPECTED:
        reasons.append("MEMORY_MISMATCH_SUSPECTED")
    if frontier_class is FrontierClass.UNCERTAIN:
        reasons.append("REQUIRES_MORE_EVIDENCE")
    # CC4 follow-up (P0-7): TOO_EASY / TOO_HARD are never training targets.
    if frontier_class is FrontierClass.TOO_EASY:
        reasons.append("TOO_EASY_NOT_TRAINING_TARGET")
    if frontier_class is FrontierClass.TOO_HARD:
        reasons.append("TOO_HARD_NOT_TRAINING_TARGET")
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
    # Gap is binding only when Reference evidence actually exists; a
    # Reference-less run yields reference_rate 0.0, which is absence of
    # evidence, not a measured gap.
    if evidence.reference_actual_n > 0 and evidence.student_reference_gap \
            <= -STUDENT_DOMINANCE_GAP_THRESHOLD:
        reasons.append("STUDENT_DOMINATES_REFERENCE_NOT_FRONTIER")
    # Progress: an admitted state must show measured Student progress.
    if evidence.student_actual_n > 0 and evidence.student_success_rate == 0.0 \
            and evidence.student_mean_progress <= 0.0:
        reasons.append("NO_STUDENT_PROGRESS_EVIDENCE")
    # Cost: measured spend may not exceed the plan's declared budget.
    transition_budget = int(plan.horizon) * int(plan.actual_n)
    if evidence.transition_cost > transition_budget:
        reasons.append("TRANSITION_BUDGET_EXCEEDED")
    # Diversity: mixed-start training needs a multi-bucket frontier.
    if evidence.bucket_diversity < MIN_BUCKET_DIVERSITY:
        reasons.append("BUCKET_DIVERSITY_INSUFFICIENT")

    rejected = bool(reasons)
    if rejected:
        accepted = False
        frontier_start_weight = 0.0
    else:
        # Every non-LEARNABLE class has already contributed a rejection
        # reason above, so acceptance implies LEARNABLE_FRONTIER; the weight
        # lookup stays fail-closed regardless.
        frontier_start_weight = FRONTIER_START_WEIGHTS.get(frontier_class)
        if frontier_start_weight is None:
            raise InvalidEvidenceError(
                f"frontier class {frontier_class!r} carries no start weight "
                "(only LEARNABLE_FRONTIER is ever admissible)")
        accepted = True
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
