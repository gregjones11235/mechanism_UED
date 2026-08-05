"""Dynamic long-run launch gate + training-budget semantics (P0-17).

Before this contract existed, the long-run entrypoint returned a HARDCODED
blocker list: it could never reflect reality in either direction — resolved
dependencies stayed "blocked" on paper, and nothing mechanical would have
added a blocker if a new dependency had appeared.

The gate is now a pure, fail-closed function of ACTUAL evidence:

* the real E3 preflight result (carried over verbatim — never re-derived
  from memory; a missing preflight evaluation is itself a blocker);
* the Reference designation state (a PENDING/empty candidate id blocks);
* the shared anchor manifest reference state;
* a minted, signed ``LongRunBudgetDecision`` — the experiment director's
  explicit training-budget semantics, one of exactly two values:

  - ``TOTAL_FROM_COMMON_INITIALIZATION`` — the frozen step budget is the
    TOTAL training budget measured from the common initialization;
  - ``ADDITIONAL_FROM_PRETRAINED_CHECKPOINT`` — the frozen step budget is
    ADDITIONAL training on top of the pretrained checkpoint the Student
    starts from.

  No decision (or a self-signed one) yields
  ``BLOCKED_WAITING_DIRECTOR_BUDGET_DECISION`` — the budget is never
  assumed, never defaulted;
* the external audit approval flag.

``evaluate_launch_blockers`` therefore shrinks mechanically as production
dependencies bind, and grows mechanically if any of them drifts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields
from typing import Any, Mapping, Sequence

from .errors import InvalidEvidenceError

LONGRUN_GATE_VERSION = "e3-longrun-gate/v2"

BUDGET_DECISION_SCHEMA = "simulator_frontier.longrun-budget-decision/v1"

TRAINING_BUDGET_TOTAL_FROM_COMMON_INITIALIZATION = (
    "TOTAL_FROM_COMMON_INITIALIZATION")
TRAINING_BUDGET_ADDITIONAL_FROM_PRETRAINED_CHECKPOINT = (
    "ADDITIONAL_FROM_PRETRAINED_CHECKPOINT")
TRAINING_BUDGET_SEMANTICS = (
    TRAINING_BUDGET_TOTAL_FROM_COMMON_INITIALIZATION,
    TRAINING_BUDGET_ADDITIONAL_FROM_PRETRAINED_CHECKPOINT,
)

BLOCKED_WAITING_DIRECTOR_BUDGET_DECISION = (
    "BLOCKED_WAITING_DIRECTOR_BUDGET_DECISION")
BLOCKED_E3_PREFLIGHT_NOT_EVALUATED = "BLOCKED_E3_PREFLIGHT_NOT_EVALUATED"
BLOCKED_REFERENCE_IDENTITY_PENDING_CONTROLLER_DESIGNATION = (
    "BLOCKED_REFERENCE_IDENTITY_PENDING_CONTROLLER_DESIGNATION")
BLOCKED_AUDIT_APPROVAL_NOT_GRANTED = "BLOCKED_AUDIT_APPROVAL_NOT_GRANTED"
BLOCKED_SHARED_ANCHOR_MANIFEST_PENDING = "BLOCKED_SHARED_ANCHOR_MANIFEST"

_PENDING_MARKER = "PENDING"
_SYNTHETIC_SIGNATURE_PREFIX = "SYNTHETIC_SIGNATURE_"


@dataclass(frozen=True)
class LongRunBudgetDecision:
    """One immutable, director-signed training-budget decision (mint-only).

    ``decision_hash`` is NOT a constructor argument: it is computed in
    ``__post_init__`` from the decision fields only.
    """

    decision_id: str
    director_id: str
    signature_ref: str
    budget_semantics: str
    total_env_steps: int
    decision_hash: str = field(init=False)
    decision_schema: str = BUDGET_DECISION_SCHEMA

    def __post_init__(self) -> None:
        for label, value in (("decision_id", self.decision_id),
                             ("director_id", self.director_id),
                             ("signature_ref", self.signature_ref)):
            if not str(value).strip():
                raise InvalidEvidenceError(
                    f"LongRunBudgetDecision.{label} is empty — a budget decision "
                    "is never anonymous and never unsigned")
        if self.budget_semantics not in TRAINING_BUDGET_SEMANTICS:
            raise InvalidEvidenceError(
                f"LongRunBudgetDecision.budget_semantics must be one of "
                f"{list(TRAINING_BUDGET_SEMANTICS)}, got {self.budget_semantics!r}")
        if isinstance(self.total_env_steps, bool) \
                or not isinstance(self.total_env_steps, int) \
                or self.total_env_steps <= 0:
            raise InvalidEvidenceError(
                f"LongRunBudgetDecision.total_env_steps must be a positive int, "
                f"got {self.total_env_steps!r}")
        payload = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name != "decision_hash"
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, default=str)
        object.__setattr__(
            self, "decision_hash",
            hashlib.sha256(blob.encode("utf-8")).hexdigest())


def mint_longrun_budget_decision(*, decision_id: Any, director_id: Any,
                                 signature_ref: Any, budget_semantics: Any,
                                 total_env_steps: Any) -> LongRunBudgetDecision:
    """Mint the immutable budget decision (fail closed on any gap)."""
    if str(signature_ref).startswith(_SYNTHETIC_SIGNATURE_PREFIX):
        raise InvalidEvidenceError(
            f"budget decision signature_ref {signature_ref!r} is synthetic — a "
            "self-signed budget decision is never production evidence")
    return LongRunBudgetDecision(
        decision_id=str(decision_id),
        director_id=str(director_id),
        signature_ref=str(signature_ref),
        budget_semantics=str(budget_semantics),
        total_env_steps=int(total_env_steps),
    )


def verify_longrun_budget_decision(decision: Any) -> None:
    """Recompute the decision hash; reject mappings, foreign types, tamper."""
    if isinstance(decision, Mapping):
        raise InvalidEvidenceError(
            "verify_longrun_budget_decision requires a minted "
            "LongRunBudgetDecision, not a mapping")
    if not isinstance(decision, LongRunBudgetDecision):
        raise InvalidEvidenceError(
            f"verify_longrun_budget_decision requires a minted "
            f"LongRunBudgetDecision, got {type(decision).__name__}")
    payload = {
        f.name: getattr(decision, f.name)
        for f in fields(decision)
        if f.name != "decision_hash"
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)
    expected = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    if expected != decision.decision_hash:
        raise InvalidEvidenceError(
            "decision_hash mismatch: the LongRunBudgetDecision was tampered "
            "with or self-reported (fail closed)")


def budget_decision_from_payload(payload: Any) -> LongRunBudgetDecision:
    """Rebuild + verify the decision from its JSON payload (fail closed)."""
    if not isinstance(payload, Mapping):
        raise InvalidEvidenceError("budget decision payload must be a JSON object")
    if payload.get("schema") != BUDGET_DECISION_SCHEMA:
        raise InvalidEvidenceError(
            f"budget decision schema must be {BUDGET_DECISION_SCHEMA!r}, got "
            f"{payload.get('schema')!r}")
    if str(payload.get("signature_ref", "")).startswith(_SYNTHETIC_SIGNATURE_PREFIX):
        raise InvalidEvidenceError(
            "budget decision payload signature_ref is synthetic — a self-signed "
            "decision is never production evidence (fail closed)")
    raw_steps = payload.get("total_env_steps", 0)
    if isinstance(raw_steps, bool) or not isinstance(raw_steps, int) \
            or raw_steps <= 0:
        raise InvalidEvidenceError(
            f"budget decision total_env_steps must be a positive int, got "
            f"{raw_steps!r}")
    decision = LongRunBudgetDecision(
        decision_id=str(payload.get("decision_id", "")),
        director_id=str(payload.get("director_id", "")),
        signature_ref=str(payload.get("signature_ref", "")),
        budget_semantics=str(payload.get("budget_semantics", "")),
        total_env_steps=raw_steps,
    )
    if str(payload.get("decision_hash", "")) != decision.decision_hash:
        raise InvalidEvidenceError(
            "budget decision payload hash mismatch: the payload does not "
            "recompute to its declared decision_hash (fail closed)")
    verify_longrun_budget_decision(decision)
    return decision


def evaluate_launch_blockers(*, preflight_blockers: Sequence[str] | None,
                             reference_candidate_id: str,
                             anchor_manifest_ref: str,
                             budget_decision: LongRunBudgetDecision | None,
                             audit_approved: bool) -> tuple[str, ...]:
    """The DYNAMIC launch gate: every unresolved dependency blocks, and the
    list mechanically reflects what is actually bound right now."""
    blockers: list[str] = []
    if preflight_blockers is None:
        blockers.append(BLOCKED_E3_PREFLIGHT_NOT_EVALUATED)
    else:
        blockers.extend(str(b) for b in preflight_blockers)
    reference = str(reference_candidate_id).strip()
    if not reference or _PENDING_MARKER in reference.upper():
        blockers.append(BLOCKED_REFERENCE_IDENTITY_PENDING_CONTROLLER_DESIGNATION)
    manifest_ref = str(anchor_manifest_ref).strip()
    if not manifest_ref or _PENDING_MARKER in manifest_ref.upper():
        blockers.append(BLOCKED_SHARED_ANCHOR_MANIFEST_PENDING)
    if budget_decision is None:
        blockers.append(BLOCKED_WAITING_DIRECTOR_BUDGET_DECISION)
    else:
        try:
            verify_longrun_budget_decision(budget_decision)
        except InvalidEvidenceError:
            blockers.append(BLOCKED_WAITING_DIRECTOR_BUDGET_DECISION)
        else:
            if str(budget_decision.signature_ref).startswith(
                    _SYNTHETIC_SIGNATURE_PREFIX):
                blockers.append(BLOCKED_WAITING_DIRECTOR_BUDGET_DECISION)
    if not bool(audit_approved):
        blockers.append(BLOCKED_AUDIT_APPROVAL_NOT_GRANTED)
    deduped: list[str] = []
    for blocker in blockers:
        if blocker and blocker not in deduped:
            deduped.append(blocker)
    return tuple(deduped)
