"""Role: AdversarialReviewer (task §3 CONDITIONAL third LLM call).

The default loop spends 2 LLM calls per triggered window (Diagnostician +
Designer). A third call is made ONLY when at least one of the seven risk
triggers fires. This module owns (a) the deterministic evaluation of those
triggers and (b) the reviewer role that stress-tests the designer's plan.

The reviewer is adversarial: it looks for over-confident verdicts, budget
contradictions, and exploration that is not actually bounded. Its output is a
set of concerns + forced corrections that the DeterministicReconciler applies
BEFORE the plan is finalized, so the reviewer can veto but never silently
rewrite.
"""
from __future__ import annotations

import json
from typing import Dict, List

from pydantic import Field, model_validator

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.feedback_contracts import (
    CONTEXT_CLOSE,
    CONTEXT_OPEN,
    FeedbackRoleEnvelope,
)
from d052.schemas.common import CanonicalModel

ROLE = C.ROLE_ADVERSARIAL_REVIEWER
PROMPT_VERSION = f"{C.ROLE_PROMPT_VERSION}.{ROLE}"

PROMPT_TEMPLATE = f"""\
You are the AdversarialReviewer role of the simulator-grounded feedback-adaptive
LLM-UED loop. You are invoked because at least one risk trigger fired. Critique
the designer's proposed plan: flag over-confident verdicts, contradictory
family budgets, and unbounded exploration. Output concrete concerns and forced
corrections. Environment-level only.
Prompt version: {{prompt_version}}
{CONTEXT_OPEN}
{{context_json}}
{CONTEXT_CLOSE}
Respond with a single JSON object matching the ReviewerOutput schema.
"""


def evaluate_risk_triggers(context: dict) -> List[str]:
    """Return the sorted list of fired risk triggers (empty -> no reviewer)."""
    fired: List[str] = []
    overall = float(context.get("overall_confidence", 1.0))
    if overall < C.REVIEWER_CONFIDENCE_FLOOR:
        fired.append(C.RISK_LOW_CONFIDENCE)

    # conflicting interventions: same family gets contradictory decisions
    decisions: Dict[str, set] = {}
    for a in context.get("allocations", []):
        decisions.setdefault(a.get("environment_family"), set()).add(
            a.get("decision"))
    for fam, decs in decisions.items():
        if {C.DECISION_RETAIN, C.DECISION_RETIRE} <= decs or \
                {C.DECISION_EXPAND_BUDGET, C.DECISION_RETIRE} <= decs:
            fired.append(C.RISK_CONFLICTING_INTERVENTIONS)
            break

    if context.get("global_risk", "LOW") == "HIGH":
        fired.append(C.RISK_GLOBAL_RISK_HIGH)
    if int(context.get("windows_without_improvement", 0)) >= 2:
        fired.append(C.RISK_NO_IMPROVEMENT_TWO_WINDOWS)
    if int(context.get("opposite_probe_count", 0)) > 0:
        fired.append(C.RISK_PROBE_OPPOSITE_DIRECTION)
    if float(context.get("reject_rate", 0.0)) > C.REVIEWER_REJECT_RATE_FLOOR:
        fired.append(C.RISK_HIGH_REJECT_RATE)
    if bool(context.get("preparing_formal_run", False)):
        fired.append(C.RISK_BEFORE_FORMAL_CANDIDATE_RUN)
    # de-dup + stable order
    return sorted(set(fired))


class ReviewerOutput(CanonicalModel):
    window: int = Field(ge=0)
    triggered_by: List[str] = Field(default_factory=list)
    concerns: List[str] = Field(default_factory=list)
    #: families the reviewer forces RETIRE on (over-confidence guard)
    forced_retire_families: List[str] = Field(default_factory=list)
    approve: bool = True

    @model_validator(mode="after")
    def _triggers_legal(self) -> "ReviewerOutput":
        for t in self.triggered_by:
            if t not in C.REVIEWER_RISK_TRIGGERS:
                raise ValueError(f"ILLEGAL_RISK_TRIGGER: {t!r}")
        return self


def build_prompt(context: dict) -> str:
    return PROMPT_TEMPLATE.format(
        prompt_version=PROMPT_VERSION,
        context_json=json.dumps(context, sort_keys=True, ensure_ascii=False,
                                default=str))


def parse(raw: str) -> ReviewerOutput:
    return ReviewerOutput.model_validate_json(raw)


def mock_rule(context: dict) -> dict:
    """Deterministic adversarial pass over the proposed plan."""
    window = int(context.get("window", 0))
    triggered = context.get("triggered_by", [])
    verdicts = context.get("verdicts", [])
    allocs = context.get("allocations", [])
    concerns: List[str] = []
    forced_retire: List[str] = []

    # over-confidence guard: a SUPPORTED verdict at very high confidence that
    # was earned from a single agree record is fragile -> force a downgrade by
    # retiring nothing but flagging it; a REFUTED family still budgeted RETAIN
    # is a contradiction -> force retire.
    verdict_by_id = {v["hypothesis_id"]: v for v in verdicts}
    hyp_family = {h["hypothesis_id"]: h["environment_family"]
                  for h in context.get("hypotheses", [])}
    for v in verdicts:
        if v["verdict"] == C.HYPOTHESIS_SUPPORTED and v["agree_count"] == 1 \
                and v["new_confidence"] >= 0.75:
            concerns.append(
                f"over-confident SUPPORT for {v['hypothesis_id']} from a "
                f"single agree record")

    # family-level contradiction: RETAIN and RETIRE both present
    decisions: Dict[str, set] = {}
    for a in allocs:
        decisions.setdefault(a.get("environment_family"), set()).add(
            a.get("decision"))
    for fam, decs in sorted(decisions.items()):
        if {C.DECISION_RETAIN, C.DECISION_RETIRE} <= decs:
            concerns.append(f"contradictory budget for family {fam}: "
                            f"RETAIN and RETIRE both proposed")
            forced_retire.append(fam)

    # exploration bound check
    n_exploration = sum(1 for a in allocs if a.get("is_exploration"))
    if n_exploration > C.MAX_EXPLORATION_PROPOSALS:
        concerns.append(f"exploration not bounded: {n_exploration} > "
                        f"{C.MAX_EXPLORATION_PROPOSALS}")

    if C.RISK_HIGH_REJECT_RATE in triggered:
        concerns.append("high candidate reject rate suggests the proposal "
                        "distribution is mis-tuned")
    if C.RISK_NO_IMPROVEMENT_TWO_WINDOWS in triggered:
        concerns.append("no measured improvement for two windows; consider "
                        "widening mutation axes")

    approve = not forced_retire and n_exploration <= C.MAX_EXPLORATION_PROPOSALS
    return dict(window=window, triggered_by=list(triggered),
                concerns=concerns, forced_retire_families=sorted(set(forced_retire)),
                approve=approve)


def run(context: dict, backend, window: int, sequence: int) -> FeedbackRoleEnvelope:
    prompt = build_prompt(context)
    raw = backend.complete(ROLE, prompt)
    parsed = parse(raw)
    return FeedbackRoleEnvelope.make(
        role=ROLE, prompt_version=PROMPT_VERSION, backend_id=backend.backend_id,
        model_id=backend.model_id, window=window, sequence=sequence,
        prompt=prompt, raw_response=raw, parsed_dump=parsed.model_dump())
