"""InvocationGate: the official 0-or-2 LLM-call rule (roadmap R7, contract only).

Official rule (总控 roadmap, supersedes the older "fixed 0 / optional <=1"):

  * NO significant change  -> STRICTLY 0 LLM calls; the previous plan MUST be
    reused via an explicit ``reuse_plan_ref`` (never implied).
  * Revision required      -> EXACTLY 2 LLM calls, in the fixed order
    Frontier Evidence Diagnostician -> Curriculum & Search Planner.
  * Any attempts == 1 state is a contract violation and raises.

Evidence boundary: both LLMs may read ONLY aggregate evidence (Feasibility
Statistics aggregates + archive summaries).  Successful actions, routes,
waypoints, logits, hidden states and expert trajectories are FORBIDDEN from
entering the LLM context, the curriculum or the Student (SearchActionLeakage
semantics, extended here).  Selection authority belongs to the deterministic
selector: LLM outputs are advisory candidates only.

This module is CONTRACT ONLY: all tests use ``FakeLLMClient``; no real API is
called.  Enabling a real two-LLM run requires explicit 总控 authorization, so
``REAL_TWO_LLM_CALL_EXECUTED`` stays false for this round.

The TYPED production two-LLM path lives in ``llm_contracts.py`` (strict
DiagnosticianOutput / PlannerOutput schemas with hash recomputation, and a
production entry that fails closed rather than falling back to fake clients).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from .errors import InvalidEvidenceError, ProvenanceViolationError
from .provenance import FormalDataLeakageGuard, SearchActionLeakageGuard
from .search_statistics import FeasibilityEstimate

# Honest round status: contract + fake-client tests only, zero real API calls.
TWO_LLM_GATE_CONTRACT_ONLY = True  # == CONTRACT_AND_FAKE_CLIENT_TEST_READY
REAL_TWO_LLM_CALL_EXECUTED = False


class InvocationReason(str, Enum):
    NO_SIGNIFICANT_CHANGE = "NO_SIGNIFICANT_CHANGE"
    REVISION_REQUIRED = "REVISION_REQUIRED"


LLM_ROLE_SEQUENCE = ("frontier_evidence_diagnostician", "curriculum_search_planner")

# Extension of SearchActionLeakageGuard.FORBIDDEN_KEYS for the LLM boundary:
# exact-match key names only (aggregate keys like "action_count" stay legal).
INVOCATION_EXTRA_FORBIDDEN_KEYS = frozenset({
    "successful_actions", "successful_action", "action", "actions",
    "teacher_action", "expert_action",
})
INVOCATION_FORBIDDEN_KEYS = frozenset(SearchActionLeakageGuard.FORBIDDEN_KEYS) | \
    INVOCATION_EXTRA_FORBIDDEN_KEYS


class InvocationContractError(InvalidEvidenceError):
    """The 0-or-2 invocation contract was violated (fail closed)."""


@dataclass(frozen=True)
class InvocationDecision:
    reason: InvocationReason
    llm_calls: int
    reuse_plan_ref: str | None
    planned_roles: tuple[str, ...]

    def __post_init__(self) -> None:
        # THE core invariant: exactly 0 or exactly 2, never 1.
        if self.llm_calls not in (0, 2):
            raise InvocationContractError(
                f"LLM call count must be 0 or 2 (official rule), got {self.llm_calls}; "
                "a single LLM call is never a valid InvocationGate outcome")
        if self.llm_calls == 0:
            if not self.reuse_plan_ref:
                raise InvocationContractError(
                    "NO_SIGNIFICANT_CHANGE requires an explicit reuse_plan_ref "
                    "(reuse is never implied)")
            if self.planned_roles:
                raise InvocationContractError("0-call decision cannot schedule LLM roles")
        else:
            if self.reuse_plan_ref:
                raise InvocationContractError(
                    "REVISION_REQUIRED must not carry a reuse_plan_ref")
            if tuple(self.planned_roles) != LLM_ROLE_SEQUENCE:
                raise InvocationContractError(
                    f"2-call decision must schedule exactly {LLM_ROLE_SEQUENCE} in order")


def decide_invocation(reason: InvocationReason | str, *,
                      reuse_plan_ref: str | None = None) -> InvocationDecision:
    """Fail-closed factory enforcing the official 0-or-2 rule."""
    reason = InvocationReason(reason)
    if reason is InvocationReason.NO_SIGNIFICANT_CHANGE:
        if not reuse_plan_ref:
            raise InvocationContractError(
                "NO_SIGNIFICANT_CHANGE -> 0 LLM calls and reuse_plan_ref is REQUIRED")
        return InvocationDecision(reason, 0, reuse_plan_ref, ())
    return InvocationDecision(reason, 2, None, LLM_ROLE_SEQUENCE)


# ---------------------------------------------------------------------------
# aggregate evidence boundary
# ---------------------------------------------------------------------------

_FEASIBILITY_AGGREGATE_FIELDS = (
    "state_id", "total_actual_branches", "actual_branches_by_source", "successes",
    "success_rate", "confidence_interval", "mean_progress", "max_progress",
    "transition_cost", "uncertainty", "estimate_version",
)


def _walk_forbidden(node: Any, forbidden: frozenset[str], path: str = "") -> None:
    if isinstance(node, Mapping):
        for key, child in node.items():
            name = str(key).lower()
            if name in forbidden:
                raise ProvenanceViolationError(
                    f"forbidden action-guidance field at {path}.{key} "
                    "(aggregate evidence only — actions/routes/logits never enter "
                    "the LLM context, the curriculum or the Student)")
            _walk_forbidden(child, forbidden, f"{path}.{key}" if path else str(key))
    elif isinstance(node, (list, tuple)):
        for i, child in enumerate(node):
            _walk_forbidden(child, forbidden, f"{path}[{i}]")


def build_aggregate_evidence(feasibility: FeasibilityEstimate,
                             archive_summary: Mapping[str, Any],
                             *, data_source: str) -> dict[str, Any]:
    """Assemble the ONLY kind of evidence the two LLMs may read.

    Fail closed on three layers:
      1. formal-bank provenance can never feed the gate (FormalDataLeakageGuard
         with consumer "curriculum": the gate's output drives curriculum planning);
      2. the base SearchActionLeakageGuard field walk;
      3. the extended invocation field walk (successful_actions/action/...).
    """
    FormalDataLeakageGuard.assert_allowed(data_source, "curriculum")
    evidence: dict[str, Any] = {
        "feasibility": {k: getattr(feasibility, k) for k in _FEASIBILITY_AGGREGATE_FIELDS},
        "archive_summary": dict(archive_summary),
        "data_source": str(data_source),
    }
    SearchActionLeakageGuard.validate_aggregate(evidence)
    _walk_forbidden(evidence, INVOCATION_FORBIDDEN_KEYS)
    return evidence


# ---------------------------------------------------------------------------
# fake client (contract tests only; names must carry fake_client/contract)
# ---------------------------------------------------------------------------

class FakeLLMClient:
    """CONTRACT-ONLY stand-in for one LLM role.  Never touches a real API.

    Deterministic: returns the canned payload and records every call so tests
    can assert the exact 0-or-2 call pattern and the fixed role order.
    """

    def __init__(self, role: str, canned_payload: Mapping[str, Any] | None = None,
                 *, raise_on_call: Exception | None = None):
        if role not in LLM_ROLE_SEQUENCE:
            raise InvocationContractError(f"unknown LLM role {role!r}")
        self.role = role
        self.canned_payload = dict(canned_payload or {})
        self.raise_on_call = raise_on_call
        self.calls: list[Mapping[str, Any]] = []

    def complete(self, evidence: Mapping[str, Any]) -> dict[str, Any]:
        self.calls.append({"role": self.role, "evidence": dict(evidence)})
        if self.raise_on_call is not None:
            raise self.raise_on_call
        return dict(self.canned_payload)


def assert_never_exactly_one_call(call_counts: Sequence[int]) -> None:
    """Guardrail used by drivers/tests: attempts == 1 is a contract violation."""
    for count in call_counts:
        if int(count) == 1:
            raise InvocationContractError(
                "attempts == 1 is forbidden by the official 0-or-2 rule")


def run_two_llm_gate(decision: InvocationDecision,
                     evidence: Mapping[str, Any],
                     *,
                     diagnostician_client: Any,
                     planner_client: Any) -> dict[str, Any]:
    """Execute the decision's call pattern against the (fake) clients.

    0 calls: clients are never invoked (a client that recorded a call is a
    contract breach the caller's tests should catch).
    2 calls: Diagnostician first, then Planner with the Diagnostician's
    aggregate summary attached; the FIXED order is enforced; every LLM output
    is re-validated against the forbidden-field boundary before it may be
    handed to the deterministic selector.
    """
    if not isinstance(decision, InvocationDecision):
        raise InvocationContractError("run_two_llm_gate requires an InvocationDecision")
    _walk_forbidden(evidence, INVOCATION_FORBIDDEN_KEYS)

    if decision.llm_calls == 0:
        return {"llm_calls": 0, "reuse_plan_ref": decision.reuse_plan_ref,
                "candidates": (), "role_order": ()}

    role_order: list[str] = []
    diagnosis = diagnostician_client.complete(dict(evidence))
    role_order.append(LLM_ROLE_SEQUENCE[0])
    if not isinstance(diagnosis, Mapping):
        raise InvocationContractError("diagnostician output must be a mapping")
    _walk_forbidden(diagnosis, INVOCATION_FORBIDDEN_KEYS, "diagnostician")

    planner_input = dict(evidence)
    planner_input["diagnostician_summary"] = dict(diagnosis)
    plan = planner_client.complete(planner_input)
    role_order.append(LLM_ROLE_SEQUENCE[1])
    if not isinstance(plan, Mapping):
        raise InvocationContractError("planner output must be a mapping")
    _walk_forbidden(plan, INVOCATION_FORBIDDEN_KEYS, "planner")

    if tuple(role_order) != LLM_ROLE_SEQUENCE:
        raise InvocationContractError(f"role order violated: {role_order}")
    return {"llm_calls": 2, "reuse_plan_ref": None,
            "candidates": (dict(diagnosis), dict(plan)),
            "role_order": tuple(role_order)}


# ---------------------------------------------------------------------------
# deterministic selector (final selection authority; reproducible hash)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SelectionResult:
    chosen_plan_id: str
    selection_hash: str
    rejected: tuple[str, ...]
    reason_codes: tuple[str, ...]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)


def deterministic_select(candidates: Sequence[Mapping[str, Any]], *,
                         evidence_hash: str) -> SelectionResult:
    """Deterministically choose among advisory LLM candidates.

    A candidate is VALID only if it carries plan_id/curriculum_ref and a
    finite priority_score AND contains no forbidden action-guidance field —
    LLM outputs that smuggle actions are rejected outright (the "LLM output
    vetoed" path).  Winner = max priority_score, ties broken by smallest
    plan_id; the selection hash binds evidence + winner + rejections so the
    decision is reproducible.  Zero valid candidates -> fail closed.
    """
    if not evidence_hash:
        raise InvalidEvidenceError("deterministic_select requires evidence_hash")
    valid: list[dict[str, Any]] = []
    rejected: list[str] = []
    reasons: list[str] = []
    for i, cand in enumerate(candidates):
        label = str(cand.get("plan_id", f"candidate[{i}]")) if isinstance(cand, Mapping) else f"candidate[{i}]"
        if not isinstance(cand, Mapping):
            rejected.append(label)
            reasons.append("NOT_A_MAPPING")
            continue
        try:
            _walk_forbidden(cand, INVOCATION_FORBIDDEN_KEYS)
        except ProvenanceViolationError as exc:
            rejected.append(label)
            reasons.append("FORBIDDEN_ACTION_GUIDANCE_FIELD")
            _ = exc
            continue
        plan_id = cand.get("plan_id")
        curriculum_ref = cand.get("curriculum_ref")
        score = cand.get("priority_score")
        if not plan_id or not curriculum_ref or not isinstance(score, (int, float)) \
                or isinstance(score, bool) or not _finite(score):
            rejected.append(label)
            reasons.append("MISSING_OR_INVALID_REQUIRED_FIELDS")
            continue
        valid.append({"plan_id": str(plan_id), "curriculum_ref": str(curriculum_ref),
                      "priority_score": float(score)})
    if not valid:
        raise InvalidEvidenceError(
            f"deterministic selector rejected ALL candidates "
            f"(reasons={reasons}); fail closed — no silent fallback plan")
    valid.sort(key=lambda c: (-c["priority_score"], c["plan_id"]))
    winner = valid[0]
    payload = {"evidence_hash": evidence_hash, "chosen": winner,
               "rejected": sorted(rejected), "reason_codes": reasons}
    return SelectionResult(
        chosen_plan_id=winner["plan_id"],
        selection_hash=hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest(),
        rejected=tuple(sorted(rejected)),
        reason_codes=tuple(reasons))


def _finite(x: float) -> bool:
    import math
    return math.isfinite(float(x))


def evidence_hash_of(evidence: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(dict(evidence)).encode("utf-8")).hexdigest()
