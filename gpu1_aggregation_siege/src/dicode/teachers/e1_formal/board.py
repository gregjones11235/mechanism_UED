"""Stage 2b: the six-role Review Board (fixed order, all-or-nothing).

Every TRIGGERED window runs ALL six roles in the fixed order

    student_modeler -> behavior_auditor -> causal_failure_analyst
    -> intervention_tutor -> explorer -> critic

SEQUENTIAL COOPERATION (round-3 P0-1): role k's prompt binds
(a) the board context — window identity, session, trigger code,
evidence hash and the pinned Student candidate id — and (b) every
SUCCESSFULLY PARSED structured output of roles 0..k-1, rendered into
the user prompt AND hash-bound into the prompt envelope. Roles never
see a bare evidence-only prompt, and no later role can be replayed
against an earlier role's envelope: the replay key changes with the
whole upstream chain.

There is NO two-role subset and NO conditional reviewer path. The
critic always runs and is never conditioned away. All six calls are
recorded by the accounting ledger even when the window's outputs are
later voided; a void window yields ``INCOMPLETE_REVIEW_WINDOW`` (any
role failed) or ``ALL_FAMILIES_VETOED`` (critic vetoed every proposed
intervention family) and the curriculum REUSEs the previous tasks.

Discipline per role output:
1. the reply comes from the (replay) LLM client; a cache miss is a
   HARD FAIL that propagates (no fallback, no void-and-continue);
2. the raw text passes the F1-F7 content guards BEFORE parsing;
3. the JSON block is extracted and validated fail-closed against the
   per-role contract (roles 3/4 reuse the committed
   ``parse_diagnosis`` / ``parse_intervention_plan``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ..static_llm.guards import raise_if_forbidden
from ..static_llm.schemas import parse_diagnosis, parse_intervention_plan
from .canonical import canonical_json, canonical_sha256
from .evidence import EvidenceSnapshot, render_evidence_for_prompt
from .json_parse import extract_json_block
from .llm_client import make_replay_key
from .manifest import BOARD_PROMPT_VERSION, BOARD_ROLE_ORDER, ROLE_OUTPUT_SCHEMA_VERSION
from .schemas import E1Code, E1SchemaError
from .student_contract import (
    ALLOWED_STUDENT_CANDIDATE_IDS,
    PERSISTENT_STUDENT_CANDIDATE_ID,
)

WINDOW_STATUS_COMPLETE = "COMPLETE"
WINDOW_STATUS_VOID = "VOID"

#: The ONLY veto reasons the critic verdict consumer accepts.
VALID_VETO_REASONS = frozenset(
    {
        "GUARD_VETO",
        "RETENTION_VETO",
        "CRITIC_VETO_CONTRACT_VIOLATION",
        "CRITIC_VETO_UNSUPPORTED_CLAIM",
        "CRITIC_VETO_EVIDENCE_MISMATCH",
    }
)

MAX_EXPLORER_AXES = 8


class BoardError(E1SchemaError):
    """Fail-closed review-board violation; ``code`` is greppable."""


class _BCode:
    REPLY_BAD_SHAPE = "BOARD_REPLY_BAD_SHAPE"
    OUTPUT_BAD_TYPE = "BOARD_OUTPUT_BAD_TYPE"
    OUTPUT_MISSING_FIELD = "BOARD_OUTPUT_MISSING_FIELD"
    OUTPUT_EMPTY_FIELD = "BOARD_OUTPUT_EMPTY_FIELD"
    OUTPUT_UNKNOWN_FIELD = "BOARD_OUTPUT_UNKNOWN_FIELD"
    OUTPUT_OUT_OF_RANGE = "BOARD_OUTPUT_OUT_OF_RANGE"
    DUPLICATE_ID = "BOARD_OUTPUT_DUPLICATE_ID"
    TOO_MANY_ITEMS = "BOARD_OUTPUT_TOO_MANY_ITEMS"
    CHAIN_HASH_MISMATCH = "BOARD_CHAIN_HASH_MISMATCH"
    WINDOW_HASH_MISMATCH = "WINDOW_HASH_MISMATCH"
    RECORD_BAD_TYPE = "WINDOW_RECORD_BAD_TYPE"
    RECORD_UNKNOWN_FIELD = "WINDOW_RECORD_UNKNOWN_FIELD"
    RECORD_MISSING_FIELD = "WINDOW_RECORD_MISSING_FIELD"


ROLE_SYSTEM_PROMPTS: Dict[str, str] = {
    "student_modeler": (
        "You are the StudentModeler of the E1 review board. Model the "
        "Student's current capabilities strictly from the training-window "
        "evidence. Never invent evaluation results."
    ),
    "behavior_auditor": (
        "You are the BehaviorAuditor of the E1 review board. Audit the "
        "Student's observed behavior failures strictly from the "
        "training-window evidence. Never invent evaluation results."
    ),
    "causal_failure_analyst": (
        "You are the CausalFailureAnalyst of the E1 review board. Produce "
        "causal hypotheses about the observed failures as a Diagnosis "
        "object. Never invent evaluation results."
    ),
    "intervention_tutor": (
        "You are the InterventionTutor of the E1 review board. Propose "
        "environment intervention families as an InterventionPlan object. "
        "Never invent evaluation results."
    ),
    "explorer": (
        "You are the Explorer of the E1 review board. Propose exploration "
        "axes beyond the current curriculum. Never invent evaluation "
        "results."
    ),
    "critic": (
        "You are the Critic/Skeptic of the E1 review board. You always "
        "run and your verdict is never conditioned away. Veto proposed "
        "intervention families ONLY with an allowed veto reason."
    ),
}

_ROLE_TASK_LINES: Dict[str, str] = {
    "student_modeler": "Produce the Student capability model for this window.",
    "behavior_auditor": "List the behavior-failure findings for this window.",
    "causal_failure_analyst": "Produce the causal Diagnosis for this window.",
    "intervention_tutor": "Produce the InterventionPlan for this window.",
    "explorer": "Propose exploration axes for this window.",
    "critic": "Veto unacceptable intervention families (or veto none).",
}


@dataclass(frozen=True)
class ReviewWindow:
    """Immutable, hash-identified record of one review window."""

    window_id: str
    session_idx: int
    trigger_code: str
    evidence_hash: str
    status: str  # COMPLETE | VOID
    void_code: str  # "" when COMPLETE
    role_results: Tuple[Tuple[str, Any], ...]  # (role, parsed JSON or None)
    surviving_families: Tuple[str, ...]
    ignored_vetoes: Tuple[Dict[str, str], ...]
    window_hash: str


@dataclass(frozen=True)
class BoardContext:
    """Identity every role prompt is bound to (round-3 P0-1).

    ``student_candidate_id`` is the pinned strong-Student identity of
    this pipeline; every role models/audits/intervenes with respect to
    THAT student, never an anonymous one.
    """

    window_id: str
    session_idx: int
    trigger_code: str
    evidence_hash: str
    student_candidate_id: str


@dataclass(frozen=True)
class UpstreamOutput:
    """One successfully-parsed earlier role output in the chain."""

    role: str
    output: Any  # canonical-encodable parsed JSON object
    output_hash: str  # == role_output_hash(role, output)


def make_board_context(
    *, window_id: str, session_idx: int, trigger_code: str,
    evidence_hash: str, student_candidate_id: str = PERSISTENT_STUDENT_CANDIDATE_ID,
) -> BoardContext:
    """Build the board context for the window's Student identity.

    CC2-Student: the parameter defaults to the Persistent candidate
    ONLY as a lower-level backward-compat default for the replay-key
    builders; the E1 flow binds the DIRECTOR-SELECTED Student via the
    SelectedStudentMount continuity (never a silent re-selection).
    """
    if student_candidate_id not in ALLOWED_STUDENT_CANDIDATE_IDS:
        raise BoardError(
            _BCode.UNKNOWN_FIELD,
            f"board_context: student_candidate_id "
            f"{student_candidate_id!r} is not in the allowed set "
            f"{sorted(ALLOWED_STUDENT_CANDIDATE_IDS)}",
        )
    return BoardContext(
        window_id=window_id,
        session_idx=session_idx,
        trigger_code=trigger_code,
        evidence_hash=evidence_hash,
        student_candidate_id=student_candidate_id,
    )


def role_output_hash(role: str, parsed: Any) -> str:
    """Canonical hash of one role's parsed structured output."""
    return canonical_sha256({"role": role, "output": parsed})


# ---------------------------------------------------------------------------
# Prompt construction (deterministic; context + upstream chain bound)
# ---------------------------------------------------------------------------
def _render_upstream(upstream: Sequence[UpstreamOutput]) -> str:
    if not upstream:
        return "(none — you are the first role in the sequence)"
    lines = []
    for entry in upstream:
        lines.append(
            f"[role={entry.role} output_hash={entry.output_hash}]\n"
            f"{canonical_json(entry.output)}"
        )
    return "\n".join(lines)


def build_role_prompt(
    role: str,
    evidence: EvidenceSnapshot,
    *,
    context: BoardContext,
    upstream: Sequence[UpstreamOutput] = (),
) -> Tuple[str, str]:
    """Return (system_prompt, user_prompt) for a board role.

    The user prompt carries TASK / WINDOW / STUDENT / EVIDENCE /
    UPSTREAM_ROLE_OUTPUTS sections: every later role literally reads
    the structured outputs of all earlier roles that parsed
    successfully, and the envelope hash binds the same chain.
    """
    if role not in ROLE_SYSTEM_PROMPTS:
        raise BoardError(
            _BCode.OUTPUT_UNKNOWN_FIELD,
            f"no prompt defined for role {role!r} (allowed: "
            f"{list(BOARD_ROLE_ORDER)})",
        )
    if not isinstance(context, BoardContext):
        raise BoardError(
            _BCode.OUTPUT_BAD_TYPE,
            f"build_role_prompt requires a BoardContext, got "
            f"{type(context).__name__}",
        )
    upstream = tuple(upstream)
    for i, entry in enumerate(upstream):
        if not isinstance(entry, UpstreamOutput):
            raise BoardError(
                _BCode.OUTPUT_BAD_TYPE,
                f"upstream[{i}] must be an UpstreamOutput, got "
                f"{type(entry).__name__}",
            )
        if entry.output_hash != role_output_hash(entry.role, entry.output):
            raise BoardError(
                _BCode.CHAIN_HASH_MISMATCH,
                f"upstream[{i}] ({entry.role}) output_hash does not match "
                f"its output (chain corruption)",
            )
    user_prompt = (
        f"TASK: {_ROLE_TASK_LINES[role]}\n"
        f"WINDOW: window_id={context.window_id} "
        f"session_idx={context.session_idx} "
        f"trigger_code={context.trigger_code}\n"
        f"STUDENT: student_candidate_id={context.student_candidate_id}\n"
        f"EVIDENCE:\n{render_evidence_for_prompt(evidence)}\n"
        f"UPSTREAM_ROLE_OUTPUTS (parsed outputs of earlier roles in the "
        f"fixed sequence; build on them, do not contradict them without "
        f"stating why):\n{_render_upstream(upstream)}\n"
        "Respond with exactly one JSON object matching your role output "
        "contract."
    )
    return ROLE_SYSTEM_PROMPTS[role], user_prompt


def build_prompt_envelope_hash(
    role: str,
    evidence: EvidenceSnapshot,
    *,
    context: BoardContext,
    upstream: Sequence[UpstreamOutput] = (),
) -> str:
    """Canonical hash of the full prompt envelope for a board role.

    Binds the prompts AND the board identity (window/session/trigger/
    evidence hash as ``window_identity``, Student candidate id) AND the
    hash chain of every upstream role output — so the envelope (and the
    replay key derived from it) changes with the sequential chain.
    """
    system_prompt, user_prompt = build_role_prompt(
        role, evidence, context=context, upstream=upstream
    )
    upstream = tuple(upstream)
    return canonical_sha256(
        {
            "role": role,
            "prompt_version": BOARD_PROMPT_VERSION,
            "schema_version": ROLE_OUTPUT_SCHEMA_VERSION,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "base_evidence_hash": evidence.evidence_hash,
            "window_identity": [
                context.window_id,
                context.session_idx,
                context.trigger_code,
                context.evidence_hash,
            ],
            "student_candidate_id": context.student_candidate_id,
            "upstream_outputs": [
                [entry.role, entry.output_hash] for entry in upstream
            ],
        }
    )


# ---------------------------------------------------------------------------
# Per-role output contracts (fail-closed)
# ---------------------------------------------------------------------------
def _require_mapping(obj: Any, ctx: str) -> Mapping[str, Any]:
    if not isinstance(obj, Mapping):
        raise BoardError(
            _BCode.OUTPUT_BAD_TYPE,
            f"{ctx}: expected a JSON object, got {type(obj).__name__}",
        )
    return obj


def _require_str(obj: Mapping[str, Any], key: str, ctx: str) -> str:
    if key not in obj:
        raise BoardError(
            _BCode.OUTPUT_MISSING_FIELD, f"{ctx}: missing field {key!r}"
        )
    value = obj[key]
    if not isinstance(value, str) or not value.strip():
        raise BoardError(
            _BCode.OUTPUT_EMPTY_FIELD,
            f"{ctx}: field {key!r} must be a non-empty string, got {value!r}",
        )
    return value.strip()


def _reject_unknown(obj: Mapping[str, Any], allowed: Sequence[str], ctx: str) -> None:
    unknown = sorted(k for k in obj if k not in set(allowed))
    if unknown:
        raise BoardError(
            _BCode.OUTPUT_UNKNOWN_FIELD, f"{ctx}: unknown field(s) {unknown}"
        )


def _check_unique(ids: Sequence[str], ctx: str) -> None:
    seen = set()
    for i in ids:
        if i in seen:
            raise BoardError(_BCode.DUPLICATE_ID, f"{ctx}: duplicate id {i!r}")
        seen.add(i)


def _parse_student_modeler(obj: Any, ctx: str) -> Mapping[str, Any]:
    obj = _require_mapping(obj, ctx)
    _reject_unknown(obj, ("model_summary", "capability_profile"), ctx)
    _require_str(obj, "model_summary", ctx)
    if "capability_profile" not in obj:
        raise BoardError(
            _BCode.OUTPUT_MISSING_FIELD, f"{ctx}: missing field 'capability_profile'"
        )
    profile = obj["capability_profile"]
    if not isinstance(profile, (list, tuple)):
        raise BoardError(
            _BCode.OUTPUT_BAD_TYPE, f"{ctx}: capability_profile must be a list"
        )
    for i, entry in enumerate(profile):
        entry = _require_mapping(entry, f"{ctx}.capability_profile[{i}]")
        _reject_unknown(entry, ("skill_id", "success_rate"), f"{ctx}.capability_profile[{i}]")
        _require_str(entry, "skill_id", f"{ctx}.capability_profile[{i}]")
        if "success_rate" not in entry:
            raise BoardError(
                _BCode.OUTPUT_MISSING_FIELD,
                f"{ctx}.capability_profile[{i}]: missing field 'success_rate'",
            )
        rate = entry["success_rate"]
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not (
            0.0 <= float(rate) <= 1.0
        ):
            raise BoardError(
                _BCode.OUTPUT_OUT_OF_RANGE,
                f"{ctx}.capability_profile[{i}]: success_rate {rate!r} not in [0, 1]",
            )
    return obj


def _parse_behavior_auditor(obj: Any, ctx: str) -> Mapping[str, Any]:
    obj = _require_mapping(obj, ctx)
    _reject_unknown(obj, ("findings",), ctx)
    if "findings" not in obj:
        raise BoardError(_BCode.OUTPUT_MISSING_FIELD, f"{ctx}: missing field 'findings'")
    findings = obj["findings"]
    if not isinstance(findings, (list, tuple)):
        raise BoardError(_BCode.OUTPUT_BAD_TYPE, f"{ctx}: findings must be a list")
    if len(findings) == 0:
        raise BoardError(
            _BCode.OUTPUT_EMPTY_FIELD, f"{ctx}: at least one finding is required"
        )
    ids = []
    for i, f in enumerate(findings):
        f = _require_mapping(f, f"{ctx}.findings[{i}]")
        _reject_unknown(f, ("finding_id", "description"), f"{ctx}.findings[{i}]")
        ids.append(_require_str(f, "finding_id", f"{ctx}.findings[{i}]"))
        _require_str(f, "description", f"{ctx}.findings[{i}]")
    _check_unique(ids, ctx)
    return obj


def _parse_explorer(obj: Any, ctx: str) -> Mapping[str, Any]:
    obj = _require_mapping(obj, ctx)
    _reject_unknown(obj, ("exploration_rationale", "candidate_axes"), ctx)
    _require_str(obj, "exploration_rationale", ctx)
    if "candidate_axes" not in obj:
        raise BoardError(
            _BCode.OUTPUT_MISSING_FIELD, f"{ctx}: missing field 'candidate_axes'"
        )
    axes = obj["candidate_axes"]
    if not isinstance(axes, (list, tuple)):
        raise BoardError(
            _BCode.OUTPUT_BAD_TYPE, f"{ctx}: candidate_axes must be a list"
        )
    if len(axes) == 0:
        raise BoardError(
            _BCode.OUTPUT_EMPTY_FIELD, f"{ctx}: at least one exploration axis is required"
        )
    if len(axes) > MAX_EXPLORER_AXES:
        raise BoardError(
            _BCode.TOO_MANY_ITEMS,
            f"{ctx}: {len(axes)} axes > {MAX_EXPLORER_AXES}",
        )
    cleaned = []
    for i, a in enumerate(axes):
        if not isinstance(a, str) or not a.strip():
            raise BoardError(
                _BCode.OUTPUT_EMPTY_FIELD,
                f"{ctx}: candidate_axes[{i}] must be a non-empty string",
            )
        cleaned.append(a.strip())
    _check_unique(cleaned, f"{ctx}.candidate_axes")
    return obj


def _parse_critic(obj: Any, ctx: str) -> Mapping[str, Any]:
    obj = _require_mapping(obj, ctx)
    _reject_unknown(obj, ("vetoes", "notes"), ctx)
    if "vetoes" not in obj:
        raise BoardError(_BCode.OUTPUT_MISSING_FIELD, f"{ctx}: missing field 'vetoes'")
    vetoes = obj["vetoes"]
    if not isinstance(vetoes, (list, tuple)):
        raise BoardError(_BCode.OUTPUT_BAD_TYPE, f"{ctx}: vetoes must be a list")
    for i, v in enumerate(vetoes):
        v = _require_mapping(v, f"{ctx}.vetoes[{i}]")
        _reject_unknown(v, ("family_id", "reason"), f"{ctx}.vetoes[{i}]")
        _require_str(v, "family_id", f"{ctx}.vetoes[{i}]")
        _require_str(v, "reason", f"{ctx}.vetoes[{i}]")
    if "notes" in obj and not isinstance(obj["notes"], str):
        raise BoardError(
            _BCode.OUTPUT_BAD_TYPE, f"{ctx}: notes must be a string"
        )
    return obj


def parse_role_output(role: str, obj: Any, ctx: str) -> Any:
    """Validate one parsed role output fail-closed; returns the object."""
    if role == "student_modeler":
        return _parse_student_modeler(obj, ctx)
    if role == "behavior_auditor":
        return _parse_behavior_auditor(obj, ctx)
    if role == "causal_failure_analyst":
        parse_diagnosis(obj)  # committed static_llm contract (validate)
        return _require_mapping(obj, ctx)  # keep the canonical mapping
    if role == "intervention_tutor":
        parse_intervention_plan(obj)  # committed static_llm contract (validate)
        return _require_mapping(obj, ctx)  # keep the canonical mapping
    if role == "explorer":
        return _parse_explorer(obj, ctx)
    if role == "critic":
        return _parse_critic(obj, ctx)
    raise BoardError(
        _BCode.OUTPUT_UNKNOWN_FIELD, f"no output contract for role {role!r}"
    )


def _require_reply_content(reply: Any, ctx: str) -> str:
    if (
        not isinstance(reply, (list, tuple))
        or len(reply) != 1
        or not isinstance(reply[0], Mapping)
        or not isinstance(reply[0].get("content"), str)
    ):
        raise BoardError(
            _BCode.REPLY_BAD_SHAPE,
            f"{ctx}: replay reply must be [{'content': str}], got {reply!r}",
        )
    return reply[0]["content"]


def _parse_role_content(role: str, content: str, ctx: str) -> Any:
    # Content guards BEFORE parsing; then fail-closed contract validation.
    raise_if_forbidden(content, ctx)
    obj = extract_json_block(content, ctx)
    raise_if_forbidden(obj, ctx)
    return parse_role_output(role, obj, ctx)


# ---------------------------------------------------------------------------
# Critic verdict (fixed reason set; invented reasons ignored + recorded)
# ---------------------------------------------------------------------------
def apply_critic_verdict(
    family_ids: Sequence[str], critic_output: Mapping[str, Any], ctx: str
) -> Tuple[Tuple[str, ...], Tuple[Dict[str, str], ...]]:
    """Return (surviving_families, ignored_vetoes).

    Only the fixed ``VALID_VETO_REASONS`` set is accepted; veto entries
    with invented reasons or unknown family targets are ignored and
    recorded (never acted upon). Family order is preserved.
    """
    known = set(family_ids)
    vetoed = set()
    ignored: List[Dict[str, str]] = []
    for i, v in enumerate(critic_output.get("vetoes", [])):
        v = _require_mapping(v, f"{ctx}.vetoes[{i}]")
        family_id = _require_str(v, "family_id", f"{ctx}.vetoes[{i}]")
        reason = _require_str(v, "reason", f"{ctx}.vetoes[{i}]")
        if reason not in VALID_VETO_REASONS:
            ignored.append(
                {
                    "family_id": family_id,
                    "reason": reason,
                    "why_ignored": "UNKNOWN_VETO_REASON",
                }
            )
            continue
        if family_id not in known:
            ignored.append(
                {
                    "family_id": family_id,
                    "reason": reason,
                    "why_ignored": "UNKNOWN_FAMILY_TARGET",
                }
            )
            continue
        vetoed.add(family_id)
    surviving = tuple(fid for fid in family_ids if fid not in vetoed)
    return surviving, tuple(ignored)


# ---------------------------------------------------------------------------
# Window hash + persistence (JSONL with reload-time hash re-verification)
# ---------------------------------------------------------------------------
def _window_payload(
    window_id: str,
    session_idx: int,
    trigger_code: str,
    evidence_hash: str,
    status: str,
    void_code: str,
    role_results: Sequence[Tuple[str, Any]],
    surviving_families: Sequence[str],
    ignored_vetoes: Sequence[Mapping[str, str]],
) -> Dict[str, Any]:
    return {
        "window_id": window_id,
        "session_idx": session_idx,
        "trigger_code": trigger_code,
        "evidence_hash": evidence_hash,
        "status": status,
        "void_code": void_code,
        "role_results": [[role, obj] for role, obj in role_results],
        "surviving_families": list(surviving_families),
        "ignored_vetoes": [dict(v) for v in ignored_vetoes],
    }


def _compute_window_hash(payload: Mapping[str, Any]) -> str:
    return canonical_sha256(dict(payload))


def window_to_record(window: ReviewWindow) -> Dict[str, Any]:
    """JSONL-ready record including the window hash."""
    payload = _window_payload(
        window.window_id,
        window.session_idx,
        window.trigger_code,
        window.evidence_hash,
        window.status,
        window.void_code,
        window.role_results,
        window.surviving_families,
        window.ignored_vetoes,
    )
    payload["window_hash"] = window.window_hash
    return payload


_RECORD_FIELDS = frozenset(
    {
        "window_id",
        "session_idx",
        "trigger_code",
        "evidence_hash",
        "status",
        "void_code",
        "role_results",
        "surviving_families",
        "ignored_vetoes",
        "window_hash",
    }
)


def verify_window_record(record: Mapping[str, Any], ctx: str) -> ReviewWindow:
    """Rebuild a ReviewWindow from a record, re-verifying its hash."""
    if not isinstance(record, Mapping):
        raise BoardError(
            _BCode.RECORD_BAD_TYPE,
            f"{ctx}: window record must be a mapping, got {type(record).__name__}",
        )
    unknown = sorted(k for k in record if k not in _RECORD_FIELDS)
    if unknown:
        raise BoardError(
            _BCode.RECORD_UNKNOWN_FIELD, f"{ctx}: unknown record field(s) {unknown}"
        )
    for key in _RECORD_FIELDS:
        if key not in record:
            raise BoardError(
                _BCode.RECORD_MISSING_FIELD, f"{ctx}: record missing field {key!r}"
            )
    role_results_raw = record["role_results"]
    if not isinstance(role_results_raw, (list, tuple)):
        raise BoardError(
            _BCode.RECORD_BAD_TYPE, f"{ctx}: role_results must be a list"
        )
    role_results = []
    for i, pair in enumerate(role_results_raw):
        if (
            not isinstance(pair, (list, tuple))
            or len(pair) != 2
            or not isinstance(pair[0], str)
        ):
            raise BoardError(
                _BCode.RECORD_BAD_TYPE,
                f"{ctx}: role_results[{i}] must be [role, obj]",
            )
        role_results.append((pair[0], pair[1]))
    payload = _window_payload(
        record["window_id"],
        record["session_idx"],
        record["trigger_code"],
        record["evidence_hash"],
        record["status"],
        record["void_code"],
        role_results,
        tuple(record["surviving_families"]),
        tuple(dict(v) for v in record["ignored_vetoes"]),
    )
    recomputed = _compute_window_hash(payload)
    if recomputed != record["window_hash"]:
        raise BoardError(
            _BCode.WINDOW_HASH_MISMATCH,
            f"{ctx}: recorded window_hash {record['window_hash']!r} != "
            f"recomputed {recomputed!r}",
        )
    return ReviewWindow(
        window_id=record["window_id"],
        session_idx=record["session_idx"],
        trigger_code=record["trigger_code"],
        evidence_hash=record["evidence_hash"],
        status=record["status"],
        void_code=record["void_code"],
        role_results=tuple(role_results),
        surviving_families=tuple(record["surviving_families"]),
        ignored_vetoes=tuple(dict(v) for v in record["ignored_vetoes"]),
        window_hash=record["window_hash"],
    )


# ---------------------------------------------------------------------------
# Board execution
# ---------------------------------------------------------------------------
def run_review_board(
    llm: Any,
    *,
    window_id: str,
    session_idx: int,
    trigger_code: str,
    evidence: EvidenceSnapshot,
    ledger: Any,
) -> ReviewWindow:
    """Run the full six-role board; every call is accounted.

    SEQUENTIAL COOPERATION (round-3 P0-1): the board context (window
    identity + pinned Student candidate id + evidence hash) is built
    once; each role k receives the parsed structured outputs of roles
    0..k-1 THAT PARSED SUCCESSFULLY, rendered into its user prompt and
    hash-bound into its envelope (hence into its replay key). A failed
    earlier role therefore changes every later envelope — the chain is
    honest, never patched over.

    All six roles ALWAYS run once the window is open; any per-role
    failure voids the whole window (``INCOMPLETE_REVIEW_WINDOW``) but
    does not skip the remaining roles. A replay cache miss is a HARD
    FAIL and propagates (no fallback).
    """
    board_context = make_board_context(
        window_id=window_id,
        session_idx=session_idx,
        trigger_code=trigger_code,
        evidence_hash=evidence.evidence_hash,
    )
    ledger.record_window_open(window_id)
    role_results: List[Tuple[str, Any]] = []
    failures: List[Dict[str, str]] = []
    parsed: Dict[str, Any] = {}
    upstream: List[UpstreamOutput] = []

    for role in BOARD_ROLE_ORDER:
        ledger.record_board_call(window_id, role)
        system_prompt, user_prompt = build_role_prompt(
            role, evidence, context=board_context, upstream=upstream
        )
        envelope_hash = build_prompt_envelope_hash(
            role, evidence, context=board_context, upstream=upstream
        )
        cache_key = make_replay_key(
            role=role,
            evidence_hash=evidence.evidence_hash,
            prompt_envelope_hash=envelope_hash,
            prompt_version=BOARD_PROMPT_VERSION,
            schema_version=ROLE_OUTPUT_SCHEMA_VERSION,
        )
        # HARD FAIL (replay miss / malformed reply) propagates on purpose.
        reply = llm.query(
            system_prompt, [user_prompt], cache_key=cache_key, role=role
        )
        content = _require_reply_content(
            reply, f"window {window_id} role {role}"
        )
        try:
            obj = _parse_role_content(
                role, content, f"window {window_id} role {role}"
            )
        except Exception as e:  # fail closed: any violation voids the role
            failures.append(
                {
                    "role": role,
                    "code": getattr(e, "code", "BOARD_ROLE_FAILURE"),
                    "detail": str(e),
                }
            )
            role_results.append((role, None))
        else:
            parsed[role] = obj
            role_results.append((role, obj))
            upstream.append(
                UpstreamOutput(
                    role=role,
                    output=obj,
                    output_hash=role_output_hash(role, obj),
                )
            )

    if failures:
        status, void_code = WINDOW_STATUS_VOID, E1Code.INCOMPLETE_REVIEW_WINDOW
        surviving: Tuple[str, ...] = ()
        ignored: Tuple[Dict[str, str], ...] = ()
    else:
        status, void_code = WINDOW_STATUS_COMPLETE, ""
        plan = parsed["intervention_tutor"]
        family_ids = tuple(f["family_id"] for f in plan["families"])
        surviving, ignored = apply_critic_verdict(
            family_ids, parsed["critic"], f"window {window_id} critic"
        )
        if not surviving:
            status, void_code = WINDOW_STATUS_VOID, E1Code.ALL_FAMILIES_VETOED

    payload = _window_payload(
        window_id,
        session_idx,
        trigger_code,
        evidence.evidence_hash,
        status,
        void_code,
        role_results,
        surviving,
        ignored,
    )
    return ReviewWindow(
        window_id=window_id,
        session_idx=session_idx,
        trigger_code=trigger_code,
        evidence_hash=evidence.evidence_hash,
        status=status,
        void_code=void_code,
        role_results=tuple(role_results),
        surviving_families=surviving,
        ignored_vetoes=ignored,
        window_hash=_compute_window_hash(payload),
    )
