"""The six-role Review Board (C6) — every review window, all six roles.

Replaces the abolished Diagnostician+Designer+conditional-Reviewer 2/3-call
pattern: EVERY board window runs the complete sequence

    StudentModeler -> BehaviorAuditor -> CausalFailureAnalyst ->
    InterventionTutor -> Explorer -> Critic/Skeptic

unconditionally (``C.BOARD_CALLS_PER_WINDOW == 6``), in all three comparison
modes. The board reads ONLY:

* the window's behavior-failure evidence (``BoardContext``), and
* the ``FeedbackView`` — the sole surface exposing probe feedback. In the
  double-window state machine the view holds the frozen records of EXACTLY
  window k-1 (CC3 C9 gate: the lag is exactly one window; older records are
  stale and current/future records do not exist yet from the board's point
  of view — all fail closed as STALE_FEEDBACK_ID); the static mode hands
  over ``NullFeedbackView`` (structurally empty); the shuffled mode hands
  over a frozen anonymized permuted view (C9).

The assembled ``BoardOutput`` carries the four board deliverables — verdicts
(with explicit feedback_id / prediction-signature citations), new hypotheses,
AxisDirectives, per-family proposals — and fails closed on any citation or
consistency violation. The output is stamped ENGINEERING_SCAFFOLD.

P0-0 (CC3 follow-up audit): the board's honesty check is a MODE-AWARE USAGE
DELTA verification, not the old unconditional ``assert_no_real`` (which
hard-blocked the legitimate real path): a usage snapshot is taken before the
first role call, and after the six roles the delta must equal EXACTLY

* mock backend   : real Δ=0, replay Δ=0, mock Δ=6
* replay backend : real Δ=0, mock Δ=0, replay Δ=6
* real backend   : real Δ=6, mock Δ=0, replay Δ=0

Any role failure propagates (the window is never marked board-complete), and
any mixed / short / long delta raises ``BoardUsageDeltaMismatch``. The
end-of-run honesty check in the controller does NOT replace this role-local
check — both stay in force.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

from pydantic import Field, model_validator

from d052.bagr_ued.hashing import canonical_sha256
from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued import (
    behavior_auditor,
    causal_failure_analyst,
    critic_skeptic,
    explorer,
    intervention_tutor,
    student_modeler,
)
from d052.feedback_llm_ued.axis_directive import (
    AxisDirective,
    assert_directive_batch_legal,
)
from d052.feedback_llm_ued.behavior_auditor import BehaviorAuditOutput
from d052.feedback_llm_ued.behavior_failure import BoardContext
from d052.feedback_llm_ued.causal_failure_analyst import (
    BoardHypothesisVerdict,
    CausalAnalysisOutput,
    NewHypothesisProposal,
)
from d052.feedback_llm_ued.critic_skeptic import CriticOutput
from d052.feedback_llm_ued.explorer import ExplorerOutput
from d052.feedback_llm_ued.feedback_contracts import FeedbackRoleEnvelope
from d052.feedback_llm_ued.intervention_tutor import (
    BOARD_FAMILY_DECISIONS,
    FamilyProposal,
    InterventionOutput,
)
from d052.feedback_llm_ued.student_modeler import StudentModelOutput
from d052.schemas.common import CanonicalModel

#: fixed role -> module dispatch (order == C.BOARD_ROLES)
BOARD_ROLE_MODULES = {
    C.ROLE_STUDENT_MODELER: student_modeler,
    C.ROLE_BEHAVIOR_AUDITOR: behavior_auditor,
    C.ROLE_CAUSAL_FAILURE_ANALYST: causal_failure_analyst,
    C.ROLE_INTERVENTION_TUTOR: intervention_tutor,
    C.ROLE_EXPLORER: explorer,
    C.ROLE_CRITIC_SKEPTIC: critic_skeptic,
}

_HYPOTHESIS_PROMPT_KEYS = ("hypothesis_id", "target_behavior",
                           "environment_family", "confidence",
                           "predicted_signature", "status", "source_window")


class BoardUsageDeltaMismatch(RuntimeError):
    """The six-role board's usage delta did not match the backend kind's
    contract exactly — the window is NOT board-complete (fail closed)."""


#: one legitimate six-role board run adds EXACTLY
#: ``BOARD_CALLS_PER_WINDOW`` completions, ALL of the backend's own kind and
#: zero of the other two kinds — the same contract in every execution mode.
BOARD_USAGE_DELTA_BY_KIND = {
    C.BACKEND_KIND_MOCK: dict(real_calls=0, replay_calls=0,
                              mock_calls=C.BOARD_CALLS_PER_WINDOW),
    C.BACKEND_KIND_REPLAY: dict(real_calls=0,
                                replay_calls=C.BOARD_CALLS_PER_WINDOW,
                                mock_calls=0),
    C.BACKEND_KIND_REAL: dict(real_calls=C.BOARD_CALLS_PER_WINDOW,
                              replay_calls=0, mock_calls=0),
}


def expected_board_usage_delta(backend_kind: str) -> Dict[str, int]:
    """The required (real_calls, replay_calls, mock_calls) delta of one
    complete board run for the given backend kind. Unknown kinds fail
    closed."""
    try:
        return dict(BOARD_USAGE_DELTA_BY_KIND[backend_kind])
    except KeyError:
        raise ValueError(
            f"UNKNOWN_BACKEND_KIND: {backend_kind!r}") from None


def verify_board_usage_delta(backend, *, before) -> Dict[str, int]:
    """Role-local, mode-aware usage reconciliation (P0-0, CC3 follow-up
    audit).

    Replaces the old unconditional ``backend.usage.assert_no_real()``: that
    check (a) hard-blocked a legitimate REAL six-role board on window 1 and
    (b) only inspected cumulative state, never the window's own delta. One
    board run must add exactly ``BOARD_USAGE_DELTA_BY_KIND[backend.kind]`` —
    a silently mixed-in call of another kind (NO_SILENT_FALLBACK), a missing
    call or an extra call all refuse the board. Returns the observed delta.
    """
    expected = expected_board_usage_delta(backend.kind)
    usage = backend.usage
    observed = dict(real_calls=usage.real_calls - before.real_calls,
                    replay_calls=usage.replay_calls - before.replay_calls,
                    mock_calls=usage.mock_calls - before.mock_calls)
    if observed != expected:
        raise BoardUsageDeltaMismatch(
            f"BOARD_USAGE_DELTA_MISMATCH: backend kind {backend.kind!r} "
            f"observed board usage delta {observed} != required {expected} "
            f"— a six-role board must add exactly "
            f"{C.BOARD_CALLS_PER_WINDOW} completions, all of the backend's "
            "own kind; mixed / missing / extra calls fail closed and the "
            "window is NOT board-complete")
    return observed


def normalize_hypothesis_inputs(hypotheses: Sequence[object]) -> List[dict]:
    """Reduce HypothesisRecords (or dicts) to the prompt-visible slice.

    Deterministic order (sorted by hypothesis_id); unknown shapes fail
    closed instead of silently dropping fields.
    """
    out: List[dict] = []
    for hyp in hypotheses:
        dump = hyp.model_dump() if hasattr(hyp, "model_dump") else dict(hyp)  # type: ignore[arg-type]
        if not dump.get("hypothesis_id"):
            raise ValueError(
                f"HYPOTHESIS_INPUT_MISSING_ID: {sorted(dump.keys())}")
        out.append({k: dump.get(k) for k in _HYPOTHESIS_PROMPT_KEYS
                    if dump.get(k) is not None})
    return sorted(out, key=lambda h: h["hypothesis_id"])


#: P0-1 (CC3 follow-up audit): prompt schema version of the SEQUENTIAL
#: six-role chain — the base context every role starts from, plus the
#: structured outputs of every upstream role, bound by canonical hashes.
SEQUENTIAL_PROMPT_SCHEMA_VERSION = "feedback_llm_ued.sequential_board.v1"


class SequentialChainBroken(RuntimeError):
    """The sequential six-role context chain failed closed: a role was asked
    to run without the exact structured outputs of ALL its upstream roles."""


def make_upstream_entry(role: str, parsed_dump: Mapping[str, object]
                        ) -> Dict[str, object]:
    """One upstream role's STRUCTURED chain entry: name + parsed output +
    canonical output hash. Natural-language-only concatenation is forbidden
    — downstream prompts bind exactly this entry."""
    if role not in C.BOARD_ROLES:
        raise ValueError(f"UNKNOWN_BOARD_ROLE: {role!r}")
    if not isinstance(parsed_dump, Mapping) or not parsed_dump:
        raise SequentialChainBroken(
            f"UPSTREAM_OUTPUT_MISSING: role {role!r} produced an empty "
            "parsed output; the sequential chain cannot continue")
    dump = dict(parsed_dump)
    return dict(role=role, output=dump,
                output_hash=canonical_sha256(dump))


def sequential_role_context(base_context: Dict[str, object], *, role: str,
                            upstream_entries: Sequence[Mapping[str, object]]
                            ) -> Dict[str, object]:
    """One board role's CHAINED context (P0-1): the shared base context plus
    the structured outputs of every upstream role in the fixed board order —

        base                                -> StudentModeler
        base + StudentModeler               -> BehaviorAuditor
        base + SM + BA                      -> CausalFailureAnalyst
        base + first three                  -> InterventionTutor
        base + first four                   -> Explorer
        base + all five                     -> Critic/Skeptic

    Fail closed: UPSTREAM_CHAIN_MISMATCH (wrong count — e.g. the critic
    missing ANY of the five upstream outputs), UPSTREAM_ROLE_MISMATCH (wrong
    role name / order), UPSTREAM_OUTPUT_MISSING (an entry without a parsed
    output and hash). The chain is STRUCTURED — every upstream entry keeps
    its role name, parsed output and canonical output hash; nothing is
    flattened into prose.
    """
    if role not in C.BOARD_ROLES:
        raise ValueError(f"UNKNOWN_BOARD_ROLE: {role!r}")
    position = C.BOARD_ROLES.index(role)
    required = list(C.BOARD_ROLES[:position])
    if len(upstream_entries) != len(required):
        raise SequentialChainBroken(
            f"UPSTREAM_CHAIN_MISMATCH: role {role!r} requires EXACTLY the "
            f"{len(required)} upstream output(s) {required}; got "
            f"{len(upstream_entries)} — every role reads ALL its upstream "
            "roles and the critic must truly read all five")
    for expected_role, entry in zip(required, upstream_entries):
        entry_role = (entry.get("role")
                      if isinstance(entry, Mapping) else None)
        if entry_role != expected_role:
            raise SequentialChainBroken(
                f"UPSTREAM_ROLE_MISMATCH: role {role!r} expected upstream "
                f"{expected_role!r} at this chain position, got "
                f"{entry_role!r}")
        if not entry.get("output") or not entry.get("output_hash"):
            raise SequentialChainBroken(
                f"UPSTREAM_OUTPUT_MISSING: upstream role {expected_role!r} "
                "carries no parsed output / output hash — the chain is "
                "broken and no downstream prompt may be built")
    return dict(base_context,
                prompt_schema_version=SEQUENTIAL_PROMPT_SCHEMA_VERSION,
                upstream_roles=list(required),
                upstream_outputs=[dict(e) for e in upstream_entries])


def build_board_prompt_context(*, window: int, mode: str,
                               board_context: BoardContext,
                               view,
                               hypotheses: Sequence[dict]) -> Dict[str, object]:
    """The shared BASE context of the board (P0-1): window / mode /
    behavior evidence / feedback view / hypotheses. Each role's chained
    context adds the structured outputs of its upstream roles — see
    :func:`sequential_role_context`."""
    if mode not in C.FEEDBACK_MODES:
        raise ValueError(f"UNKNOWN_MODE: {mode!r}")
    # double-window discipline at the evidence layer (CC3 C9 gate): every
    # behavior item a window-k board reads must come from EXACTLY window
    # k-1's probes (window 0 reads no evidence at all)
    for item in board_context.behavior_evidence:
        if item.window != window - 1:
            raise ValueError(
                f"BOARD_CONTEXT_WINDOW_MISMATCH: evidence item "
                f"{item.feedback_id!r} is from window {item.window}; a "
                f"window-{window} board may only read evidence from "
                f"EXACTLY window {window - 1}")
    if board_context.window != max(0, window - 1):
        raise ValueError(
            f"BOARD_CONTEXT_WINDOW_MISMATCH: board context window "
            f"{board_context.window} != evidence window "
            f"{max(0, window - 1)} for board window {window} (CC3 C9 "
            f"gate: the lag is exactly one window)")
    return dict(window=window,
                mode=mode,
                board_context=board_context.model_dump(),
                feedback=view.to_prompt_payload(),
                feedback_view_label=view.label,
                hypotheses=list(hypotheses))


def validate_citations(verdicts: Sequence[BoardHypothesisVerdict],
                       visible_by_id: Mapping[str, dict],
                       window: int,
                       known_hypothesis_ids: frozenset) -> None:
    """Double-window citation discipline (fail closed).

    * every verdict must target a hypothesis the ledger actually holds;
    * every cited feedback id must be VISIBLE in the window's FeedbackView;
    * cited feedback must come from EXACTLY window k-1 for a window-k board
      (CC3 C9 gate: older, current and future records all fail closed as
      STALE_FEEDBACK_ID);
    * a verdict may not cite the same feedback id twice;
    * a hypothesis may receive at most ONE verdict per board;
    * with an empty view there is nothing to cite, so no verdicts.
    """
    seen_hypotheses: set = set()
    for v in verdicts:
        if v.hypothesis_id in seen_hypotheses:
            raise ValueError(
                f"DUPLICATE_HYPOTHESIS_VERDICT: {v.hypothesis_id!r} is "
                f"verdicted more than once in window {window}")
        seen_hypotheses.add(v.hypothesis_id)
        if known_hypothesis_ids is not None and \
                v.hypothesis_id not in known_hypothesis_ids:
            raise ValueError(
                f"UNKNOWN_HYPOTHESIS_ID: verdict targets {v.hypothesis_id!r} "
                f"which the ledger does not hold")
        if not v.cited_feedback_ids:
            continue                            # STALE verdict: nothing cited
        seen_citations: set = set()
        for fid in v.cited_feedback_ids:
            if fid in seen_citations:
                raise ValueError(
                    f"DUPLICATE_FEEDBACK_CITATION: {fid!r} cited twice by "
                    f"verdict for {v.hypothesis_id!r}")
            seen_citations.add(fid)
            record = visible_by_id.get(fid)
            if record is None:
                raise ValueError(
                    f"UNKNOWN_FEEDBACK_ID: {fid!r} is not visible in this "
                    f"window's FeedbackView (window {window})")
            if int(record.get("window", -1)) != window - 1:
                raise ValueError(
                    f"STALE_FEEDBACK_ID: {fid!r} comes from window "
                    f"{record.get('window')} — a window-{window} board may "
                    f"only cite feedback from EXACTLY window {window - 1} "
                    f"(older/current/future records fail closed; CC3 C9 "
                    f"gate)")
    if verdicts and any(v.cited_feedback_ids for v in verdicts) and \
            not visible_by_id:
        raise ValueError(
            "NO_FEEDBACK_TO_CITE: verdicts cite feedback but the "
            "FeedbackView is empty")


class BoardOutput(CanonicalModel):
    """Everything one review window's six roles produced (audit-grade)."""

    window: int = Field(ge=0)
    mode: str
    board_call_count: int = Field(ge=0)
    student_model: StudentModelOutput
    behavior_audit: BehaviorAuditOutput
    causal_analysis: CausalAnalysisOutput
    intervention: InterventionOutput
    explorer_output: ExplorerOutput
    critic: CriticOutput
    envelopes: List[FeedbackRoleEnvelope] = Field(default_factory=list)
    # -- the four board deliverables (must equal the role outputs) --------
    verdicts: List[BoardHypothesisVerdict] = Field(default_factory=list)
    new_hypotheses: List[NewHypothesisProposal] = Field(default_factory=list)
    directives: List[AxisDirective] = Field(default_factory=list)
    family_proposals: List[FamilyProposal] = Field(default_factory=list)
    request_control: bool = False
    #: honesty stamp — mock roles only this round; any other value is refused
    evidence_status: str = C.ENGINEERING_SCAFFOLD
    board_hash: str = ""

    @model_validator(mode="after")
    def _validate_and_hash(self) -> "BoardOutput":
        if self.mode not in C.FEEDBACK_MODES:
            raise ValueError(f"UNKNOWN_MODE: {self.mode!r}")
        if self.board_call_count != C.BOARD_CALLS_PER_WINDOW:
            raise ValueError(
                f"BOARD_CALL_COUNT_MISMATCH: window {self.window} ran "
                f"{self.board_call_count} board call(s); the formal board "
                f"is ALWAYS {C.BOARD_CALLS_PER_WINDOW} roles")
        if self.evidence_status != C.ENGINEERING_SCAFFOLD:
            raise ValueError(
                f"ILLEGAL_EVIDENCE_STATUS: {self.evidence_status!r} — this "
                f"round's board is mock-rule based and must be stamped "
                f"{C.ENGINEERING_SCAFFOLD}")
        roles = [e.role for e in self.envelopes]
        if roles != list(C.BOARD_ROLES):
            raise ValueError(
                f"BOARD_ROLE_SEQUENCE_MISMATCH: {roles} != "
                f"{list(C.BOARD_ROLES)}")
        for e in self.envelopes:
            if e.window != self.window:
                raise ValueError(
                    f"ENVELOPE_WINDOW_MISMATCH: role {e.role!r} envelope is "
                    f"window {e.window}, board is window {self.window}")
        # deliverables must equal the role outputs that produced them
        pairs = (
            ([v.model_dump() for v in self.verdicts],
             [v.model_dump() for v in
              self.causal_analysis.hypothesis_verdicts]),
            ([h.model_dump() for h in self.new_hypotheses],
             [h.model_dump() for h in self.causal_analysis.new_hypotheses]),
            ([d.model_dump() for d in self.directives],
             [d.model_dump() for d in self.explorer_output.directives]),
            ([p.model_dump() for p in self.family_proposals],
             [p.model_dump() for p in
              self.intervention.family_proposals]),
        )
        for derived, source in pairs:
            if derived != source:
                raise ValueError(
                    "BOARD_OUTPUT_INCONSISTENT: top-level deliverables "
                    "diverge from the role outputs that produced them")
        # directive batch law (unique ids, <= 1 treatment per window/family/
        # axis) + every directive belongs to THIS board window
        assert_directive_batch_legal(self.directives)
        for d in self.directives:
            if d.source_window != self.window:
                raise ValueError(
                    f"DIRECTIVE_WINDOW_MISMATCH: {d.directive_id!r} is "
                    f"source_window {d.source_window}, board is window "
                    f"{self.window}")
        for p in self.family_proposals:
            if p.decision not in BOARD_FAMILY_DECISIONS:
                raise ValueError(
                    f"ILLEGAL_BOARD_DECISION: {p.decision!r}")
        if not self.board_hash:
            payload = self.model_dump()
            payload.pop("board_hash", None)
            object.__setattr__(self, "board_hash", canonical_sha256(payload))
        return self

    def rehash(self) -> str:
        payload = self.model_dump()
        payload.pop("board_hash", None)
        return canonical_sha256(payload)


def run_review_board(*, window: int, mode: str, board_context: BoardContext,
                     view, hypotheses: Sequence[object], backend,
                     sequence_start: int,
                     student_identity_hash: str = "",
                     reference_identity_hash: str = "",
                     previous_plan_hash: str = "") -> BoardOutput:
    """Run the complete six-role board and assemble the audited output.

    Exactly ``C.BOARD_CALLS_PER_WINDOW`` backend calls are made, one per
    role, in the fixed order — and since P0-1 (CC3 follow-up audit) with a
    truly SEQUENTIAL context: each role's prompt embeds the structured
    outputs of ALL its upstream roles (bound by canonical hashes, never
    prose concatenation), with the critic reading all five. Citation
    discipline and the mode-aware usage delta are enforced here, fail
    closed.

    The three trailing identity hashes are bound into every role's
    ``context_binding`` when the caller holds them (production path); they
    default to empty strings (= unbound, never derived silently).
    """
    if mode not in C.FEEDBACK_MODES:
        raise ValueError(f"UNKNOWN_MODE: {mode!r}")
    hyp_dicts = normalize_hypothesis_inputs(hypotheses)
    base_context = build_board_prompt_context(
        window=window, mode=mode, board_context=board_context, view=view,
        hypotheses=hyp_dicts)

    #: P0-1: the structured hashes every role envelope binds (window /
    #: sequence / backend / model / prompt schema version ride on the
    #: envelope's own fields)
    view_payload = view.to_prompt_payload()
    binding_base = dict(
        window=window,
        feedback_view_hash=canonical_sha256(view_payload),
        behavior_evidence_hash=canonical_sha256(
            [item.model_dump() for item in board_context.behavior_evidence]),
        student_identity_hash=student_identity_hash,
        reference_identity_hash=reference_identity_hash,
        hypothesis_ledger_hash=canonical_sha256(hyp_dicts),
        previous_plan_hash=previous_plan_hash,
        prompt_schema_version=SEQUENTIAL_PROMPT_SCHEMA_VERSION,
        backend_id=backend.backend_id,
        model_id=backend.model_id)

    #: P0-0: usage snapshot BEFORE the first role call — the delta taken
    #: below is this board's own, never cumulative state
    usage_before = backend.usage.snapshot()

    envelopes: List[FeedbackRoleEnvelope] = []
    upstream: List[Dict[str, object]] = []
    for i, role in enumerate(C.BOARD_ROLES):
        #: a role failure (backend refusal, transport exhaustion, parse or
        #: schema violation) propagates immediately: the window is never
        #: marked board-complete (no board output exists to freeze). The
        #: sequential chain fails closed just as hard: a role never runs
        #: without the exact structured outputs of ALL its upstream roles.
        role_context = sequential_role_context(
            base_context, role=role, upstream_entries=upstream)
        context_binding = dict(
            binding_base,
            sequence=sequence_start + i,
            upstream_roles=[str(e["role"]) for e in upstream],
            upstream_output_hashes={str(e["role"]): str(e["output_hash"])
                                    for e in upstream})
        module = BOARD_ROLE_MODULES[role]
        envelope = module.run(role_context, backend, window,
                              sequence_start + i,
                              context_binding=context_binding)
        envelopes.append(envelope)
        upstream.append(make_upstream_entry(role, envelope.parsed_json))
    if len(envelopes) != C.BOARD_CALLS_PER_WINDOW:
        raise RuntimeError(
            f"BOARD_CALL_COUNT_MISMATCH: {len(envelopes)} != "
            f"{C.BOARD_CALLS_PER_WINDOW}")

    #: P0-0: role-local, mode-aware usage reconciliation — the old
    #: unconditional ``assert_no_real()`` is abolished; a real six-role board
    #: adds exactly 6 real calls (and zero mock/replay), a mock board exactly
    #: 6 mock calls, a replay board exactly 6 replay calls. Any deviation
    #: refuses the board here (the controller's end-of-run honesty check
    #: remains in force and does NOT replace this role-local check).
    verify_board_usage_delta(backend, before=usage_before)

    parsed = {e.role: e.parsed_json for e in envelopes}
    student_model = student_modeler.OUTPUT_MODEL(
        **parsed[C.ROLE_STUDENT_MODELER])
    behavior_audit = behavior_auditor.OUTPUT_MODEL(
        **parsed[C.ROLE_BEHAVIOR_AUDITOR])
    causal = causal_failure_analyst.OUTPUT_MODEL(
        **parsed[C.ROLE_CAUSAL_FAILURE_ANALYST])
    intervention = intervention_tutor.OUTPUT_MODEL(
        **parsed[C.ROLE_INTERVENTION_TUTOR])
    explorer_out = explorer.OUTPUT_MODEL(**parsed[C.ROLE_EXPLORER])
    critic = critic_skeptic.OUTPUT_MODEL(**parsed[C.ROLE_CRITIC_SKEPTIC])

    # double-window citation discipline — BEFORE anything is assembled
    visible = {p["feedback_id"]: p for p in view_payload}
    known_ids = frozenset(h["hypothesis_id"] for h in hyp_dicts)
    validate_citations(causal.hypothesis_verdicts, visible, window, known_ids)

    request_control = critic.request_control or any(
        p.decision == C.DECISION_REQUEST_CONTROL
        for p in intervention.family_proposals)

    output = BoardOutput(
        window=window, mode=mode,
        board_call_count=len(envelopes),
        student_model=student_model,
        behavior_audit=behavior_audit,
        causal_analysis=causal,
        intervention=intervention,
        explorer_output=explorer_out,
        critic=critic,
        envelopes=envelopes,
        verdicts=list(causal.hypothesis_verdicts),
        new_hypotheses=list(causal.new_hypotheses),
        directives=list(explorer_out.directives),
        family_proposals=list(intervention.family_proposals),
        request_control=request_control)
    return output
