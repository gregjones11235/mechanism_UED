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
consistency violation. Honesty: mock backends only this round; the output is
stamped ENGINEERING_SCAFFOLD and real LLM usage is asserted to be zero.
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


def build_board_prompt_context(*, window: int, mode: str,
                               board_context: BoardContext,
                               view,
                               hypotheses: Sequence[dict]) -> Dict[str, object]:
    """The ONE context every board role reads (shared, role outputs never
    feed back into it — the critic stays independent)."""
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
                     sequence_start: int) -> BoardOutput:
    """Run the complete six-role board and assemble the audited output.

    Exactly ``C.BOARD_CALLS_PER_WINDOW`` backend calls are made, one per
    role, in the fixed order. Citation discipline and honesty are enforced
    here, fail closed.
    """
    if mode not in C.FEEDBACK_MODES:
        raise ValueError(f"UNKNOWN_MODE: {mode!r}")
    hyp_dicts = normalize_hypothesis_inputs(hypotheses)
    context = build_board_prompt_context(
        window=window, mode=mode, board_context=board_context, view=view,
        hypotheses=hyp_dicts)

    envelopes: List[FeedbackRoleEnvelope] = []
    for i, role in enumerate(C.BOARD_ROLES):
        module = BOARD_ROLE_MODULES[role]
        envelopes.append(module.run(context, backend, window,
                                    sequence_start + i))
    if len(envelopes) != C.BOARD_CALLS_PER_WINDOW:
        raise RuntimeError(
            f"BOARD_CALL_COUNT_MISMATCH: {len(envelopes)} != "
            f"{C.BOARD_CALLS_PER_WINDOW}")

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
    visible = {p["feedback_id"]: p for p in view.to_prompt_payload()}
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

    # honesty: this round's board never makes a real LLM call
    backend.usage.assert_no_real()
    return output
