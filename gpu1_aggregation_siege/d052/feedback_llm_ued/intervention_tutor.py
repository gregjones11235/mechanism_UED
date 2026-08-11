"""Board role 4/6: InterventionTutor (six-role Review Board, C6).

Turns the analyst's verdicts into per-family curriculum proposals:
RETAIN / MUTATE / RETIRE / REQUEST_CONTROL — environment-FAMILY level only,
never action / reward / policy knobs.

Honesty invariant (same shape as the legacy DesignerOutput, re-checked by
the DeterministicReconciler downstream): a proposal that cites feedback is
feedback-driven; a proposal with NO citation may only be EXPLORATION
(MUTATE, is_exploration=True). A REQUEST_CONTROL proposal must cite the
evidence that triggered it.

C10 RETIRE lifecycle: families the board context lists under
``retired_families`` / ``families_in_cooldown`` are skipped ENTIRELY — no
proposal of any decision targets them, and in particular a STALE verdict
can never re-open a retired family as uncited exploration (resurrection is
structurally impossible; the Reconciler re-checks fail closed).

ENGINEERING_SCAFFOLD: deterministic mock rule; no real LLM call this round.
"""
from __future__ import annotations

import json
from typing import List, Optional

from pydantic import Field, model_validator

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.real_call_journal import (
    OUTPUT_SCHEMA_FAILED,
    OUTPUT_SCHEMA_PARSED,
    journal_role_schema_outcome,
)
from d052.feedback_llm_ued.feedback_contracts import (
    CONTEXT_CLOSE,
    CONTEXT_OPEN,
    FeedbackRoleEnvelope,
)
from d052.schemas.common import CanonicalModel

ROLE = C.ROLE_INTERVENTION_TUTOR
PROMPT_VERSION = f"{C.ROLE_PROMPT_VERSION}.{ROLE}"

#: the board's per-family decision vocabulary (director-approved). Budget
#: reshaping (EXPAND_BUDGET / REDUCE_BUDGET) stays a Reconciler concern.
BOARD_FAMILY_DECISIONS = frozenset({
    C.DECISION_RETAIN,
    C.DECISION_MUTATE,
    C.DECISION_RETIRE,
    C.DECISION_REQUEST_CONTROL,
})

PROMPT_TEMPLATE = f"""\
You are the InterventionTutor role of the six-role Review Board of the
simulator-grounded feedback-adaptive LLM-UED loop. From the causal analyst's
verdicts and new hypotheses, propose per-family curriculum actions:
RETAIN / MUTATE / RETIRE / REQUEST_CONTROL. Every feedback-driven proposal
must cite the feedback ids and hypothesis ids it is based on; uncited
proposals may only be exploration. Environment-family level only: no
action / reward / policy knobs.
Cold-start rule (the controller enforces it): a family with NO hypothesis
and NO feedback evidence receives its first controlled measurement as an
EXPLORATION MUTATE (is_exploration=true) — NEVER REQUEST_CONTROL.
REQUEST_CONTROL is reserved for a family whose CITED evidence genuinely
escalated (cite the triggering feedback/hypothesis ids). A family with no
evidence at all must not propose REQUEST_CONTROL.
Prompt version: {{prompt_version}}
{CONTEXT_OPEN}
{{context_json}}
{CONTEXT_CLOSE}
Respond with a single JSON object matching the InterventionOutput schema.
"""


class FamilyProposal(CanonicalModel):
    """One environment family's proposed curriculum action."""

    environment_family: str = Field(min_length=1)
    decision: str = Field(min_length=1)
    based_on_feedback_ids: List[str] = Field(default_factory=list)
    based_on_hypothesis_ids: List[str] = Field(default_factory=list)
    reason: str = ""
    is_exploration: bool = False

    @model_validator(mode="after")
    def _honesty(self) -> "FamilyProposal":
        if self.environment_family not in C.ENVIRONMENT_FAMILIES:
            raise ValueError(
                f"UNKNOWN_ENVIRONMENT_FAMILY: {self.environment_family!r}")
        if self.decision not in BOARD_FAMILY_DECISIONS:
            raise ValueError(f"ILLEGAL_BOARD_DECISION: {self.decision!r}")
        cited = bool(self.based_on_feedback_ids
                     or self.based_on_hypothesis_ids)
        if not cited:
            if not self.is_exploration:
                raise ValueError(
                    "EXPLORATION_LABEL_REQUIRED: uncited proposal for "
                    f"{self.environment_family!r} must be exploration")
            if self.decision != C.DECISION_MUTATE:
                raise ValueError(
                    "EXPLORATION_DECISION_ONLY: uncited proposal for "
                    f"{self.environment_family!r} may only MUTATE")
        else:
            if self.is_exploration:
                raise ValueError(
                    "MASQUERADE_FORBIDDEN: cited proposal for "
                    f"{self.environment_family!r} may not be exploration")
            if self.decision == C.DECISION_REQUEST_CONTROL and \
                    not self.based_on_feedback_ids:
                raise ValueError(
                    "REQUEST_CONTROL_REQUIRES_EVIDENCE: REQUEST_CONTROL for "
                    f"{self.environment_family!r} must cite feedback ids")
        return self


class InterventionOutput(CanonicalModel):
    window: int = Field(ge=0)
    family_proposals: List[FamilyProposal] = Field(default_factory=list)
    rationale: str = ""


def build_prompt(context: dict) -> str:
    return PROMPT_TEMPLATE.format(
        prompt_version=PROMPT_VERSION,
        context_json=json.dumps(context, sort_keys=True, ensure_ascii=False,
                                default=str))


def parse(raw: str) -> InterventionOutput:
    return InterventionOutput.model_validate_json(raw)


#: output class exposed for the board assembler (single source of truth)
OUTPUT_MODEL = InterventionOutput


def _decision_for(verdict: str) -> str:
    if verdict == C.HYPOTHESIS_SUPPORTED:
        return C.DECISION_RETAIN
    if verdict == C.HYPOTHESIS_REFUTED:
        return C.DECISION_RETIRE
    return C.DECISION_MUTATE        # INCONCLUSIVE / STALE -> sharpen/re-probe


def mock_rule(context: dict) -> dict:
    """Deterministically map visible-feedback verdicts to family proposals.

    The tutor re-derives verdicts from the SAME visible feedback the analyst
    sees (roles share one context; no role reads another role's output), so
    the two stay consistent by construction.

    Proposal sources, in deterministic order:

    1. one proposal per hypothesis-bearing family (first hypothesis by id):
       SUPPORTED -> RETAIN, REFUTED -> RETIRE, INCONCLUSIVE -> MUTATE (all
       cited), STALE -> uncited exploration MUTATE;
    2. one uncited exploration MUTATE per family that carries NO ledger
       hypothesis at all — the loop must never die of an empty budget just
       because every seeded line of inquiry was retired (the Reconciler
       still applies the exploration cap; these are proposals, rules
       dispose).

    C10: families listed under ``retired_families`` /
    ``families_in_cooldown`` in the board context are skipped in BOTH
    sources — a retired family receives no proposal of any decision until
    the controller reopens it (a STALE verdict cannot resurrect it).
    """
    window = int(context.get("window", 0))
    hypotheses = context.get("hypotheses", [])
    feedback = context.get("feedback", [])
    board_context = context.get("board_context", {})
    blocked = set(board_context.get("retired_families", [])) | \
        set(board_context.get("families_in_cooldown", []))

    by_hyp: dict = {}
    for fb in feedback:
        for hid in fb.get("distinguishes_hypothesis_ids", []):
            by_hyp.setdefault(hid, []).append(fb)
    hyp_family = {h["hypothesis_id"]: h.get("environment_family", "")
                  for h in hypotheses}

    proposals: List[dict] = []
    seen_families: set = set()
    for hid in sorted(hyp_family):
        fam = hyp_family[hid]
        if fam in seen_families:
            continue
        if fam in blocked:
            # retired / in cooldown: no proposal of any decision — STALE
            # exploration in particular must not resurrect the family
            seen_families.add(fam)
            continue
        recs = by_hyp.get(hid, [])
        agree = sum(1 for r in recs
                    if r.get("expected_observed_match")
                    == C.MATCH_DIRECTION_AGREE)
        opposite = sum(1 for r in recs
                       if r.get("expected_observed_match")
                       == C.MATCH_DIRECTION_OPPOSITE)
        if not recs:
            verdict = C.HYPOTHESIS_STALE
        elif agree > opposite:
            verdict = C.HYPOTHESIS_SUPPORTED
        elif opposite > agree:
            verdict = C.HYPOTHESIS_REFUTED
        else:
            verdict = C.HYPOTHESIS_INCONCLUSIVE
        if verdict == C.HYPOTHESIS_STALE:
            proposals.append(dict(
                environment_family=fam, decision=C.DECISION_MUTATE,
                based_on_feedback_ids=[], based_on_hypothesis_ids=[],
                reason=f"stale hypothesis {hid}: re-probe as exploration",
                is_exploration=True))
        else:
            proposals.append(dict(
                environment_family=fam, decision=_decision_for(verdict),
                based_on_feedback_ids=sorted({r["feedback_id"]
                                              for r in recs}),
                based_on_hypothesis_ids=[hid],
                reason=f"{verdict} by visible probe feedback for "
                       f"hypothesis {hid}",
                is_exploration=False))
        seen_families.add(fam)

    # families with no hypothesis at all: bounded exploration so the dynamic
    # budget is never empty when all seeded lines of inquiry were retired
    for fam in C.ENVIRONMENT_FAMILIES:
        if fam in seen_families or fam in blocked:
            continue
        proposals.append(dict(
            environment_family=fam, decision=C.DECISION_MUTATE,
            based_on_feedback_ids=[], based_on_hypothesis_ids=[],
            reason=f"no hypothesis yet for {fam}: first controlled "
                   f"measurement as exploration",
            is_exploration=True))
        seen_families.add(fam)

    rationale = (f"window {window}: {len(proposals)} family proposal(s) "
                 f"from {len(feedback)} visible feedback record(s); "
                 f"skipped retired/cooldown families: {sorted(blocked)}")
    return dict(window=window, family_proposals=proposals,
                rationale=rationale)


def run(context: dict, backend, window: int, sequence: int,
        context_binding: Optional[dict] = None) -> FeedbackRoleEnvelope:
    prompt = build_prompt(context)
    raw = backend.complete(ROLE, prompt)
    try:
        parsed = parse(raw)
    except Exception:
        journal_role_schema_outcome(
            backend, role=ROLE, prompt=prompt, status=OUTPUT_SCHEMA_FAILED,
            window=window, sequence=sequence)
        raise
    journal_role_schema_outcome(
        backend, role=ROLE, prompt=prompt, status=OUTPUT_SCHEMA_PARSED,
        window=window, sequence=sequence)
    return FeedbackRoleEnvelope.make(
        role=ROLE, prompt_version=PROMPT_VERSION, backend_id=backend.backend_id,
        model_id=backend.model_id, window=window, sequence=sequence,
        prompt=prompt, raw_response=raw, parsed_dump=parsed.model_dump(),
        context_binding=context_binding)
