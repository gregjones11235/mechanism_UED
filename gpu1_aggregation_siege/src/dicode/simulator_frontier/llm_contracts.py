"""Typed production contracts for the fixed two-LLM path (P0-4).

Official rule (unchanged): 0 LLM calls with an explicit ``reuse_plan_ref``,
or EXACTLY 2 calls in the fixed order Frontier Evidence Diagnostician ->
Curriculum & Search Planner.  This module upgrades the contract layer from
"accept any Mapping" to STRICT typed schemas:

- ``DiagnosticianOutput`` — exactly 10 fields;
- ``PlannerOutput`` — exactly 15 fields and NO ``priority_score`` (final
  ranking authority belongs to the deterministic evidence selector, never to
  an LLM score).

Validators reject unknown fields (``UNKNOWN_FIELD``), missing fields
(``MISSING_FIELD``), forbidden action-guidance fields, wrong types/domains,
and recompute the ``diagnosis_hash`` / ``plan_hash`` so an output is only
accepted if its hash genuinely binds the aggregate evidence.  Arbitrary
Mappings are never accepted.

Production path (``run_two_llm_production``): real authorized clients only.
When no authorized client factory is bound the path raises
``ProductionBlockedError("REAL_TWO_LLM_BLOCKED_NO_AUTHORIZED_CLIENT")`` —
it NEVER falls back to a fake client and never claims a real run.  This
module does not import ``FakeLLMClient`` at all.

Honest round status: no real authorized double-LLM run has executed;
``REAL_TWO_LLM_CALL_EXECUTED`` stays false.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from .errors import ProductionBlockedError
from .invocation_gate import (
    INVOCATION_FORBIDDEN_KEYS,
    InvocationContractError,
    InvocationDecision,
    InvocationReason,
    LLM_ROLE_SEQUENCE,
    decide_invocation,
    evidence_hash_of,
)
from .memory_modes import MemoryRestoreMode
from .provenance import SearchActionLeakageGuard

REAL_TWO_LLM_BLOCKED_NO_AUTHORIZED_CLIENT = "REAL_TWO_LLM_BLOCKED_NO_AUTHORIZED_CLIENT"

DIAGNOSTICIAN_OUTPUT_SCHEMA = "simulator_frontier.llm.diagnostician/v1"
PLANNER_OUTPUT_SCHEMA = "simulator_frontier.llm.planner/v1"

# Canonical frontier classes (shared with feasibility_classifier; the values
# must stay identical — the classifier enum is the authority).
FRONTIER_CLASSES = (
    "TOO_EASY",
    "LEARNABLE_FRONTIER",
    "TOO_HARD",
    "UNCERTAIN",
    "INVALID",
    "MEMORY_MISMATCH_SUSPECTED",
)

# Same three real search sources as branch_search_runner (kept jax-free here;
# the window preflight re-asserts equality with the runner's tuple).
PLANNER_SEARCH_SOURCES = (
    "STUDENT_DETERMINISTIC",
    "STUDENT_STOCHASTIC",
    "REFERENCE_POLICY",
)

DIAGNOSTICIAN_FIELDS = (
    "state_id", "bucket_id", "frontier_class", "confidence", "dominant_failure",
    "memory_mismatch_suspected", "search_budget_sufficient", "evidence_ids",
    "recommended_evidence_action", "diagnosis_hash",
)

PLANNER_FIELDS = (
    "plan_id", "based_on_diagnosis_hash", "bucket_modifications",
    "start_distribution", "taskparam_ranges", "seed_distribution",
    "stochasticity_distribution", "search_source", "actual_n", "horizon",
    "memory_mode", "anchor_ratio", "retention_constraints", "reason",
    "plan_hash",
)


class LLMContractError(InvocationContractError):
    """A typed two-LLM contract was violated (fail closed)."""


def _walk_forbidden(node: Any, path: str = "") -> None:
    if isinstance(node, Mapping):
        for key, child in node.items():
            name = str(key).lower()
            if name in INVOCATION_FORBIDDEN_KEYS:
                raise LLMContractError(
                    f"FORBIDDEN_ACTION_GUIDANCE_FIELD at {path}.{key}: aggregate evidence "
                    "only — actions/routes/logits/hidden states never enter or leave the "
                    "two-LLM path")
            _walk_forbidden(child, f"{path}.{key}" if path else str(key))
    elif isinstance(node, (list, tuple)):
        for i, child in enumerate(node):
            _walk_forbidden(child, f"{path}[{i}]")


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DiagnosticianOutput:
    """Typed Frontier Evidence Diagnostician output (exactly 10 fields)."""

    state_id: str
    bucket_id: str
    frontier_class: str
    confidence: float
    dominant_failure: str
    memory_mismatch_suspected: bool
    search_budget_sufficient: bool
    evidence_ids: tuple[str, ...]
    recommended_evidence_action: str
    diagnosis_hash: str


@dataclass(frozen=True)
class PlannerOutput:
    """Typed Curriculum & Search Planner output (exactly 15 fields).

    Deliberately carries NO priority_score: the deterministic evidence
    selector owns final ranking authority.
    """

    plan_id: str
    based_on_diagnosis_hash: str
    bucket_modifications: Mapping[str, Any]
    start_distribution: Mapping[str, Any]
    taskparam_ranges: Mapping[str, Any]
    seed_distribution: Mapping[str, Any]
    stochasticity_distribution: Mapping[str, Any]
    search_source: str
    actual_n: int
    horizon: int
    memory_mode: str
    anchor_ratio: float
    retention_constraints: tuple[str, ...]
    reason: str
    plan_hash: str


def compute_diagnostician_hash(output_fields: Mapping[str, Any], *, evidence_hash: str) -> str:
    """Canonical diagnosis hash binding every field except the hash itself."""
    payload = {
        "schema": DIAGNOSTICIAN_OUTPUT_SCHEMA,
        "evidence_hash": evidence_hash,
        "output": {k: v for k, v in dict(output_fields).items() if k != "diagnosis_hash"},
    }
    return _canonical_sha256(payload)


def compute_planner_hash(output_fields: Mapping[str, Any], *, evidence_hash: str) -> str:
    """Canonical plan hash binding every field except the hash itself."""
    payload = {
        "schema": PLANNER_OUTPUT_SCHEMA,
        "evidence_hash": evidence_hash,
        "output": {k: v for k, v in dict(output_fields).items() if k != "plan_hash"},
    }
    return _canonical_sha256(payload)


def _require_nonempty_str(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LLMContractError(f"INVALID_FIELD {name}: expected a non-empty string, got {value!r}")
    return value


def _require_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise LLMContractError(f"INVALID_FIELD {name}: expected bool, got {type(value).__name__}")
    return value


def _require_positive_int(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise LLMContractError(f"INVALID_FIELD {name}: expected int > 0, got {value!r}")
    return value


def _exact_key_set(raw: Mapping[str, Any], expected: Sequence[str], role: str) -> None:
    keys = set(raw)
    missing = sorted(set(expected) - keys)
    unknown = sorted(keys - set(expected))
    if missing:
        raise LLMContractError(f"MISSING_FIELD in {role} output: {missing}")
    if unknown:
        raise LLMContractError(f"UNKNOWN_FIELD in {role} output: {unknown} (strict schema; "
                               "arbitrary mappings are never accepted)")


def validate_diagnostician_output(raw: Any, *, evidence_hash: str,
                                  expected_state_id: str) -> DiagnosticianOutput:
    """Strictly validate a diagnostician payload; never accepts arbitrary Mappings."""
    if not evidence_hash:
        raise LLMContractError("evidence_hash is required to validate diagnostician output")
    if not isinstance(raw, Mapping):
        raise LLMContractError(f"NOT_A_MAPPING: diagnostician output must be a mapping, "
                               f"got {type(raw).__name__}")
    SearchActionLeakageGuard.validate_aggregate(dict(raw))
    _walk_forbidden(raw, "diagnostician")
    _exact_key_set(raw, DIAGNOSTICIAN_FIELDS, "diagnostician")

    state_id = _require_nonempty_str("state_id", raw["state_id"])
    if state_id != expected_state_id:
        raise LLMContractError(
            f"STATE_ID_MISMATCH: diagnostician output references {state_id!r}, "
            f"expected {expected_state_id!r}")
    bucket_id = _require_nonempty_str("bucket_id", raw["bucket_id"])
    frontier_class = raw["frontier_class"]
    if frontier_class not in FRONTIER_CLASSES:
        raise LLMContractError(
            f"INVALID_FIELD frontier_class: {frontier_class!r} not in {FRONTIER_CLASSES}")
    confidence = raw["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) \
            or not math.isfinite(float(confidence)) or not 0.0 <= float(confidence) <= 1.0:
        raise LLMContractError(f"INVALID_FIELD confidence: expected finite float in [0, 1], "
                               f"got {confidence!r}")
    dominant_failure = _require_nonempty_str("dominant_failure", raw["dominant_failure"])
    memory_mismatch_suspected = _require_bool("memory_mismatch_suspected",
                                              raw["memory_mismatch_suspected"])
    search_budget_sufficient = _require_bool("search_budget_sufficient",
                                             raw["search_budget_sufficient"])
    evidence_ids_raw = raw["evidence_ids"]
    if isinstance(evidence_ids_raw, Mapping) or isinstance(evidence_ids_raw, str) \
            or not isinstance(evidence_ids_raw, Sequence) or len(evidence_ids_raw) == 0:
        raise LLMContractError("INVALID_FIELD evidence_ids: expected a non-empty sequence "
                               "of evidence ids")
    evidence_ids = tuple(_require_nonempty_str(f"evidence_ids[{i}]", v)
                         for i, v in enumerate(evidence_ids_raw))
    recommended_evidence_action = _require_nonempty_str("recommended_evidence_action",
                                                        raw["recommended_evidence_action"])
    diagnosis_hash = _require_nonempty_str("diagnosis_hash", raw["diagnosis_hash"])

    expected_hash = compute_diagnostician_hash(dict(raw), evidence_hash=evidence_hash)
    if diagnosis_hash != expected_hash:
        raise LLMContractError(
            "DIAGNOSIS_HASH_MISMATCH: diagnosis_hash does not recompute from the output "
            "content + evidence hash (reject rather than trust)")
    return DiagnosticianOutput(
        state_id=state_id,
        bucket_id=bucket_id,
        frontier_class=str(frontier_class),
        confidence=float(confidence),
        dominant_failure=dominant_failure,
        memory_mismatch_suspected=memory_mismatch_suspected,
        search_budget_sufficient=search_budget_sufficient,
        evidence_ids=evidence_ids,
        recommended_evidence_action=recommended_evidence_action,
        diagnosis_hash=diagnosis_hash,
    )


def validate_planner_output(raw: Any, *, evidence_hash: str,
                            diagnosis: DiagnosticianOutput) -> PlannerOutput:
    """Strictly validate a planner payload; binds it to the typed diagnosis."""
    if not evidence_hash:
        raise LLMContractError("evidence_hash is required to validate planner output")
    if not isinstance(raw, Mapping):
        raise LLMContractError(f"NOT_A_MAPPING: planner output must be a mapping, "
                               f"got {type(raw).__name__}")
    SearchActionLeakageGuard.validate_aggregate(dict(raw))
    _walk_forbidden(raw, "planner")
    _exact_key_set(raw, PLANNER_FIELDS, "planner")

    plan_id = _require_nonempty_str("plan_id", raw["plan_id"])
    diagnosis_hash_recomputed = compute_diagnostician_hash(asdict(diagnosis),
                                                           evidence_hash=evidence_hash)
    if diagnosis.diagnosis_hash != diagnosis_hash_recomputed:
        raise LLMContractError("DIAGNOSIS_BINDING_BROKEN: the supplied diagnosis does not "
                               "recompute to its own diagnosis_hash")
    based_on = raw["based_on_diagnosis_hash"]
    if based_on != diagnosis_hash_recomputed:
        raise LLMContractError(
            "PLANNER_NOT_BOUND_TO_DIAGNOSIS: based_on_diagnosis_hash does not equal the "
            "recomputed diagnosis hash")

    distribution_fields = {}
    for name in ("bucket_modifications", "start_distribution", "taskparam_ranges",
                 "seed_distribution", "stochasticity_distribution"):
        value = raw[name]
        if not isinstance(value, Mapping):
            raise LLMContractError(f"INVALID_FIELD {name}: expected a mapping, "
                                   f"got {type(value).__name__}")
        distribution_fields[name] = dict(value)

    search_source = raw["search_source"]
    if search_source not in PLANNER_SEARCH_SOURCES:
        raise LLMContractError(
            f"INVALID_FIELD search_source: {search_source!r} not in {PLANNER_SEARCH_SOURCES}")
    actual_n = _require_positive_int("actual_n", raw["actual_n"])
    horizon = _require_positive_int("horizon", raw["horizon"])
    try:
        memory_mode = MemoryRestoreMode(str(raw["memory_mode"]))
    except ValueError as exc:
        raise LLMContractError(f"INVALID_FIELD memory_mode: {raw['memory_mode']!r}") from exc
    if memory_mode is MemoryRestoreMode.ZERO_MEMORY:
        raise LLMContractError(
            "ZERO_MEMORY_NOT_A_PRODUCTION_MODE: the planner may only propose "
            "SAVED_POLICY_MEMORY or HISTORY_BURN_IN for production windows")
    anchor_ratio = raw["anchor_ratio"]
    if isinstance(anchor_ratio, bool) or not isinstance(anchor_ratio, (int, float)) \
            or not math.isfinite(float(anchor_ratio)) or not 0.0 < float(anchor_ratio) <= 1.0:
        raise LLMContractError("INVALID_FIELD anchor_ratio: expected finite float in (0, 1] "
                               "(retention contract: anchor_ratio > 0)")
    retention_raw = raw["retention_constraints"]
    if isinstance(retention_raw, Mapping) or isinstance(retention_raw, str) \
            or not isinstance(retention_raw, Sequence):
        raise LLMContractError("INVALID_FIELD retention_constraints: expected a sequence "
                               "of constraint strings")
    retention_constraints = tuple(_require_nonempty_str(f"retention_constraints[{i}]", v)
                                  for i, v in enumerate(retention_raw))
    reason = _require_nonempty_str("reason", raw["reason"])
    plan_hash = _require_nonempty_str("plan_hash", raw["plan_hash"])

    expected_hash = compute_planner_hash(dict(raw), evidence_hash=evidence_hash)
    if plan_hash != expected_hash:
        raise LLMContractError(
            "PLAN_HASH_MISMATCH: plan_hash does not recompute from the output content + "
            "evidence hash (reject rather than trust)")
    return PlannerOutput(
        plan_id=plan_id,
        based_on_diagnosis_hash=str(based_on),
        bucket_modifications=distribution_fields["bucket_modifications"],
        start_distribution=distribution_fields["start_distribution"],
        taskparam_ranges=distribution_fields["taskparam_ranges"],
        seed_distribution=distribution_fields["seed_distribution"],
        stochasticity_distribution=distribution_fields["stochasticity_distribution"],
        search_source=str(search_source),
        actual_n=actual_n,
        horizon=horizon,
        memory_mode=memory_mode.value,
        anchor_ratio=float(anchor_ratio),
        retention_constraints=retention_constraints,
        reason=reason,
        plan_hash=plan_hash,
    )


def assert_planner_output_bound(plan: Any, *, evidence_hash: str) -> None:
    """Verify a FULL typed plan genuinely binds the evidence hash it claims.

    CC4 follow-up (P0-9): a reused plan is never trusted by reference — its
    ``plan_hash`` is recomputed from the typed plan's own fields and must
    equal the stored hash for the supplied evidence hash.  Any field
    mutation, or a plan bound to different evidence, raises.
    """
    if isinstance(plan, Mapping):
        raise LLMContractError(
            "plain mappings are not a PlannerOutput (a reused plan must be "
            "fully typed; hand-built mappings are refused)")
    if not isinstance(plan, PlannerOutput):
        raise LLMContractError(
            f"reused plan must be a typed PlannerOutput, got {type(plan).__name__}")
    if not evidence_hash:
        raise LLMContractError(
            "assert_planner_output_bound requires the evidence hash the plan claims")
    recomputed = compute_planner_hash(asdict(plan), evidence_hash=evidence_hash)
    if recomputed != plan.plan_hash:
        raise LLMContractError(
            "PREVIOUS_PLAN_HASH_MISMATCH: the typed previous plan does not "
            "recompute to its own plan_hash for the claimed evidence "
            "(reuse refused; fail closed)")


def derive_invocation_from_evidence(
        *, current_evidence_hash: str,
        previous_plan: Any,
        previous_evidence_hash: str,
        requested_n: int,
        horizon: int,
        memory_mode: str) -> tuple[InvocationDecision, tuple[str, ...]]:
    """CC4 follow-up (P0-9): derive the 0-or-2 decision from EVIDENCE CHANGE.

    The decision no longer depends on whether somebody supplied a previous
    plan reference: it is computed by deterministic rules comparing the
    current aggregate evidence against what the previous typed plan was
    bound to.  Revision (2 calls) is required whenever:

    - there is no previous typed plan at all (``NO_PREVIOUS_PLAN``);
    - the previous plan's bound evidence hash is missing
      (``NO_PREVIOUS_EVIDENCE_HASH``);
    - the aggregate evidence changed (``EVIDENCE_HASH_CHANGED``);
    - the plan's search budget / memory mode no longer matches the window's
      configured budget / memory mode (``PLAN_ACTUAL_N_STALE``,
      ``PLAN_HORIZON_STALE``, ``PLAN_MEMORY_MODE_STALE``).

    Only when NONE of those rules fire is NO_SIGNIFICANT_CHANGE (0 calls,
    reuse of the exact typed plan) returned.  The previous plan is
    mechanically re-verified (``assert_planner_output_bound``) before reuse
    is even considered; a stale or tampered plan always forces a revision.
    """
    _require_sha256("current_evidence_hash", current_evidence_hash)
    reasons: list[str] = []
    if previous_plan is None:
        reasons.append("NO_PREVIOUS_PLAN")
    else:
        if isinstance(previous_plan, Mapping):
            raise LLMContractError(
                "previous plan must be a typed PlannerOutput, not a mapping "
                "(an untyped reference can never suppress the two LLM calls)")
        if not isinstance(previous_plan, PlannerOutput):
            raise LLMContractError(
                f"previous plan must be a typed PlannerOutput, got "
                f"{type(previous_plan).__name__}")
        if not str(previous_evidence_hash).strip():
            reasons.append("NO_PREVIOUS_EVIDENCE_HASH")
        else:
            # Full typed verification: raises on ANY tampering or stale bind.
            assert_planner_output_bound(previous_plan,
                                        evidence_hash=previous_evidence_hash)
            if str(previous_evidence_hash) != str(current_evidence_hash):
                reasons.append("EVIDENCE_HASH_CHANGED")
            if int(previous_plan.actual_n) != int(requested_n):
                reasons.append("PLAN_ACTUAL_N_STALE")
            if int(previous_plan.horizon) != int(horizon):
                reasons.append("PLAN_HORIZON_STALE")
            if str(previous_plan.memory_mode) != str(memory_mode):
                reasons.append("PLAN_MEMORY_MODE_STALE")
    if reasons:
        return decide_invocation(InvocationReason.REVISION_REQUIRED), tuple(reasons)
    return decide_invocation(InvocationReason.NO_SIGNIFICANT_CHANGE,
                             reuse_plan_ref=previous_plan.plan_id), ()


def run_typed_two_llm_gate(decision: InvocationDecision, evidence: Mapping[str, Any], *,
                           diagnostician_client: Any, planner_client: Any,
                           expected_state_id: str) -> dict[str, Any]:
    """Execute the 0-or-2 decision with TYPED outputs (fixed role order).

    0 calls: no client is touched; the explicit ``reuse_plan_ref`` is echoed.
    2 calls: Diagnostician first, then Planner whose input additionally carries
    the typed diagnostician summary; both outputs must pass the strict
    validators (exact schema, forbidden-field sweep, hash recomputation).
    """
    if not isinstance(decision, InvocationDecision):
        raise LLMContractError("run_typed_two_llm_gate requires an InvocationDecision")
    _walk_forbidden(evidence, "evidence")
    evidence_hash = evidence_hash_of(dict(evidence))

    if decision.llm_calls == 0:
        return {"llm_calls": 0, "reuse_plan_ref": decision.reuse_plan_ref,
                "diagnostician": None, "planner": None, "role_order": (),
                "evidence_hash": evidence_hash}

    diagnosis_raw = diagnostician_client.complete(dict(evidence))
    diagnosis = validate_diagnostician_output(diagnosis_raw, evidence_hash=evidence_hash,
                                              expected_state_id=expected_state_id)
    planner_input = dict(evidence)
    planner_input["diagnostician_summary"] = asdict(diagnosis)
    plan_raw = planner_client.complete(planner_input)
    plan = validate_planner_output(plan_raw, evidence_hash=evidence_hash, diagnosis=diagnosis)
    return {"llm_calls": 2, "reuse_plan_ref": None,
            "diagnostician": diagnosis, "planner": plan,
            "role_order": LLM_ROLE_SEQUENCE, "evidence_hash": evidence_hash}


TWO_LLM_AUTHORIZATION_SCHEMA = "simulator_frontier.two-llm.authorization/v1"
TWO_LLM_CALL_CEILING = 2  # exactly two LOGICAL calls; never more, never fewer

# Fixed chain seed for the call journal (deterministic, version-labelled).
JOURNAL_CHAIN_SEED = hashlib.sha256(
    b"simulator_frontier.two-llm.journal/v1|chain-seed").hexdigest()


def _require_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64 \
            or any(c not in "0123456789abcdef" for c in value):
        raise LLMContractError(f"INVALID_HASH {name}: expected a 64-hex sha256, got {value!r}")
    return value


@dataclass(frozen=True)
class TwoLLMAuthorization:
    """MINT-ONLY authorization for the exact two logical LLM calls.

    CC4 follow-up (P0-8): the roles, the call ceiling and the hash are all
    derived inside the minter — a caller can never authorize more calls,
    different roles, or itself.  Only the controller-side authorizer issues
    these, and the production runtime verifies the hash before any call.
    """

    authorization_id: str
    authorizer_id: str
    authorization_schema: str = TWO_LLM_AUTHORIZATION_SCHEMA
    roles: tuple[str, ...] = field(init=False)
    max_logical_calls: int = field(init=False)
    authorization_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not str(self.authorization_id).strip():
            raise LLMContractError("TwoLLMAuthorization.authorization_id is empty")
        if not str(self.authorizer_id).strip():
            raise LLMContractError(
                "TwoLLMAuthorization.authorizer_id is empty (an anonymous "
                "authorization can never be accepted)")
        if self.authorization_schema != TWO_LLM_AUTHORIZATION_SCHEMA:
            raise LLMContractError("TwoLLMAuthorization schema mismatch")
        object.__setattr__(self, "roles", tuple(LLM_ROLE_SEQUENCE))
        object.__setattr__(self, "max_logical_calls", TWO_LLM_CALL_CEILING)
        payload = {
            "schema": TWO_LLM_AUTHORIZATION_SCHEMA,
            "authorization_id": self.authorization_id,
            "authorizer_id": self.authorizer_id,
            "roles": list(LLM_ROLE_SEQUENCE),
            "max_logical_calls": TWO_LLM_CALL_CEILING,
        }
        object.__setattr__(self, "authorization_hash", _canonical_sha256(payload))


def mint_two_llm_authorization(*, authorization_id: str,
                               authorizer_id: str) -> TwoLLMAuthorization:
    """The ONLY way to obtain a production two-LLM authorization (mint-only)."""
    return TwoLLMAuthorization(authorization_id=authorization_id,
                               authorizer_id=authorizer_id)


def verify_two_llm_authorization(authorization: Any) -> None:
    """Reject mappings, foreign types, tampering and self-issued authorizations."""
    if isinstance(authorization, Mapping):
        raise LLMContractError(
            "plain mappings are not a TwoLLMAuthorization (authorization is "
            "mint-only; a hand-built mapping can never authorize LLM calls)")
    if not isinstance(authorization, TwoLLMAuthorization):
        raise LLMContractError(
            f"authorization must be a minted TwoLLMAuthorization, got "
            f"{type(authorization).__name__}")
    if authorization.authorization_schema != TWO_LLM_AUTHORIZATION_SCHEMA:
        raise LLMContractError("TwoLLMAuthorization schema mismatch")
    if tuple(authorization.roles) != tuple(LLM_ROLE_SEQUENCE):
        raise LLMContractError(
            "TwoLLMAuthorization roles do not equal the fixed LLM role sequence")
    if int(authorization.max_logical_calls) != TWO_LLM_CALL_CEILING:
        raise LLMContractError(
            "TwoLLMAuthorization call ceiling is not exactly the two logical calls")
    payload = {
        "schema": TWO_LLM_AUTHORIZATION_SCHEMA,
        "authorization_id": authorization.authorization_id,
        "authorizer_id": authorization.authorizer_id,
        "roles": list(LLM_ROLE_SEQUENCE),
        "max_logical_calls": TWO_LLM_CALL_CEILING,
    }
    if _canonical_sha256(payload) != authorization.authorization_hash:
        raise LLMContractError(
            "TwoLLMAuthorization hash mismatch (authorization was modified "
            "after minting; tampering is rejected fail-closed)")


@dataclass(frozen=True)
class CallJournalEntry:
    """One journaled logical LLM call (hash-chained to its predecessor)."""

    sequence: int
    role: str
    input_hash: str
    output_hash: str
    prev_hash: str
    entry_hash: str


class CallJournal:
    """Tamper-evident journal of the exact two logical LLM calls.

    Entries are hash-chained (each entry binds its predecessor), the role
    order is enforced against ``LLM_ROLE_SEQUENCE``, and the ceiling of
    ``TWO_LLM_CALL_CEILING`` calls is absolute: a third call raises.
    """

    def __init__(self) -> None:
        self._entries: list[CallJournalEntry] = []
        self._prev_hash: str = JOURNAL_CHAIN_SEED

    @property
    def entries(self) -> tuple[CallJournalEntry, ...]:
        return tuple(self._entries)

    @property
    def journal_hash(self) -> str:
        """The current chain head (the seed for an empty journal)."""
        return self._prev_hash

    def record(self, role: str, *, input_hash: str, output_hash: str) -> CallJournalEntry:
        if len(self._entries) >= TWO_LLM_CALL_CEILING:
            raise LLMContractError(
                "TWO_LLM_CALL_CEILING_EXCEEDED: the production window allows "
                f"exactly {TWO_LLM_CALL_CEILING} logical LLM calls; a further "
                "call is refused")
        expected_role = LLM_ROLE_SEQUENCE[len(self._entries)]
        if role != expected_role:
            raise LLMContractError(
                f"TWO_LLM_ROLE_ORDER_VIOLATED: call {len(self._entries)} must be "
                f"role {expected_role!r}, got {role!r} (fixed role order is binding)")
        input_hash = _require_sha256("input_hash", input_hash)
        output_hash = _require_sha256("output_hash", output_hash)
        payload = {
            "schema": "simulator_frontier.two-llm.journal-entry/v1",
            "sequence": len(self._entries),
            "role": role,
            "input_hash": input_hash,
            "output_hash": output_hash,
            "prev_hash": self._prev_hash,
        }
        entry_hash = _canonical_sha256(payload)
        entry = CallJournalEntry(
            sequence=len(self._entries), role=role, input_hash=input_hash,
            output_hash=output_hash, prev_hash=self._prev_hash,
            entry_hash=entry_hash)
        self._entries.append(entry)
        self._prev_hash = entry_hash
        return entry

    def verify(self) -> None:
        """Recompute the whole chain from the seed; any tampering raises."""
        prev = JOURNAL_CHAIN_SEED
        for index, entry in enumerate(self._entries):
            if entry.sequence != index or entry.prev_hash != prev \
                    or entry.role != LLM_ROLE_SEQUENCE[index]:
                raise LLMContractError(
                    f"TWO_LLM_JOURNAL_CHAIN_BROKEN at entry {index} "
                    "(sequence, role order or prev-hash mismatch)")
            payload = {
                "schema": "simulator_frontier.two-llm.journal-entry/v1",
                "sequence": entry.sequence,
                "role": entry.role,
                "input_hash": entry.input_hash,
                "output_hash": entry.output_hash,
                "prev_hash": entry.prev_hash,
            }
            recomputed = _canonical_sha256(payload)
            if recomputed != entry.entry_hash:
                raise LLMContractError(
                    f"TWO_LLM_JOURNAL_CHAIN_BROKEN at entry {index} "
                    "(entry hash does not recompute)")
            prev = entry.entry_hash


class _JournaledClient:
    """Wraps a real authorized client so every call is journaled."""

    def __init__(self, inner: Any, role: str, journal: CallJournal) -> None:
        self._inner = inner
        self._role = role
        self._journal = journal

    def complete(self, payload: Mapping[str, Any]) -> Any:
        input_hash = _canonical_sha256({"payload": dict(payload)})
        output = self._inner.complete(payload)
        if not isinstance(output, Mapping):
            raise LLMContractError(
                f"TWO_LLM_OUTPUT_NOT_A_MAPPING: role {self._role!r} returned "
                f"{type(output).__name__}; only typed mapping outputs are journaled")
        output_hash = _canonical_sha256(dict(output))
        self._journal.record(self._role, input_hash=input_hash,
                             output_hash=output_hash)
        return output


@dataclass(frozen=True)
class AuthorizedTwoLLMRuntime:
    """The ONLY production surface for the 0-or-2 LLM decision (CC4 P0-8).

    Combines a mint-only ``TwoLLMAuthorization`` with the injected client
    factory and a tamper-evident ``CallJournal``: exactly two logical calls,
    fixed role order, every call hash-journaled.  Plain client factories are
    no longer accepted by the production path.
    """

    authorization: TwoLLMAuthorization
    client_factory: Any

    def __post_init__(self) -> None:
        verify_two_llm_authorization(self.authorization)
        if self.client_factory is None or not callable(self.client_factory):
            raise ProductionBlockedError(
                f"{REAL_TWO_LLM_BLOCKED_NO_AUTHORIZED_CLIENT}: an authorized two-LLM "
                "runtime requires a controller-injected client factory (never a fake, "
                "never self-built)")

    def execute(self, decision: InvocationDecision, evidence: Mapping[str, Any], *,
                expected_state_id: str) -> dict[str, Any]:
        if not isinstance(decision, InvocationDecision):
            raise LLMContractError(
                "AuthorizedTwoLLMRuntime.execute requires an InvocationDecision")
        _walk_forbidden(evidence, "evidence")
        evidence_hash = evidence_hash_of(dict(evidence))
        journal = CallJournal()
        if decision.llm_calls == 0:
            # No client is touched: the journal stays empty but its chain
            # seed is still bound into the returned audit record.
            return {"llm_calls": 0, "reuse_plan_ref": decision.reuse_plan_ref,
                    "diagnostician": None, "planner": None, "role_order": (),
                    "evidence_hash": evidence_hash,
                    "authorization_id": self.authorization.authorization_id,
                    "journal": {"entries": (), "journal_hash": journal.journal_hash}}

        clients = self.client_factory(LLM_ROLE_SEQUENCE)
        if not isinstance(clients, Mapping):
            raise ProductionBlockedError(
                f"{REAL_TWO_LLM_BLOCKED_NO_AUTHORIZED_CLIENT}: client factory must "
                f"return a role->client mapping, got {type(clients).__name__}")
        for role in LLM_ROLE_SEQUENCE:
            client = clients.get(role)
            if client is None or not callable(getattr(client, "complete", None)):
                raise ProductionBlockedError(
                    f"{REAL_TWO_LLM_BLOCKED_NO_AUTHORIZED_CLIENT}: no authorized "
                    f"client for role {role!r} (fail closed; never substitute a fake)")
        diagnostician = _JournaledClient(clients[LLM_ROLE_SEQUENCE[0]],
                                         LLM_ROLE_SEQUENCE[0], journal)
        planner = _JournaledClient(clients[LLM_ROLE_SEQUENCE[1]],
                                   LLM_ROLE_SEQUENCE[1], journal)
        result = run_typed_two_llm_gate(
            decision, evidence,
            diagnostician_client=diagnostician,
            planner_client=planner,
            expected_state_id=expected_state_id)
        # Defense in depth: the gate makes exactly two logical calls, and the
        # journal must have recorded both, in order, with an intact chain.
        journal.verify()
        if len(journal.entries) != TWO_LLM_CALL_CEILING:
            raise LLMContractError(
                f"TWO_LLM_JOURNAL_INCOMPLETE: expected exactly "
                f"{TWO_LLM_CALL_CEILING} journaled calls, got {len(journal.entries)}")
        result["authorization_id"] = self.authorization.authorization_id
        result["journal"] = {
            "entries": tuple(asdict(entry) for entry in journal.entries),
            "journal_hash": journal.journal_hash,
        }
        return result


def run_two_llm_production(decision: InvocationDecision, evidence: Mapping[str, Any], *,
                           runtime: Any, expected_state_id: str) -> dict[str, Any]:
    """PRODUCTION two-LLM path: an AuthorizedTwoLLMRuntime only (never faked).

    CC4 follow-up (P0-8): a bare client factory is no longer accepted — the
    production path requires a runtime carrying a mint-only authorization
    and a tamper-evident call journal.  When the runtime is absent this path
    raises ``ProductionBlockedError(REAL_TWO_LLM_BLOCKED_NO_AUTHORIZED_CLIENT)``;
    it NEVER falls back to ``FakeLLMClient`` and never reports a fake run as
    real.
    """
    if not isinstance(runtime, AuthorizedTwoLLMRuntime):
        raise ProductionBlockedError(
            f"{REAL_TWO_LLM_BLOCKED_NO_AUTHORIZED_CLIENT}: the production two-LLM "
            "path requires an AuthorizedTwoLLMRuntime (mint-only authorization + "
            "call journal); plain client factories and fakes are refused")
    return runtime.execute(decision, evidence, expected_state_id=expected_state_id)
