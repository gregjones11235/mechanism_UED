"""Board role 6/6: Critic/Skeptic (six-role Review Board, C6).

Independently re-derives skepticism from the SAME raw board context the other
five roles read (roles never consume each other's outputs, so the critic
cannot be talked into agreement). It flags: severe Student-vs-Reference gaps,
expected-vs-observed contradictions, ungraded feedback (comparator skipped —
a honesty objection), and wide confidence intervals; it escalates global risk
and may raise request_control.

C11: ``request_control`` now HALTS the whole loop (HumanDecisionArtifact, no
execution batch, loop stops). The mock therefore escalates only where
autonomous continuation is indefensible:

* honesty violation — feedback the comparator never graded; or
* HIGH risk computed from THIN evidence (pooled CI half-width >= WIDE_CI):
  severe AND uncertain at the same time.

Severe-but-PRECISE evidence stays HIGH risk WITHOUT halting the loop — that
is exactly what the RETIRE / MUTATE curriculum actions are for. Risk grading
itself is unchanged.

ENGINEERING_SCAFFOLD: deterministic mock rule; no real LLM call this round.
"""
from __future__ import annotations

import json
from typing import List

from pydantic import Field, model_validator

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.behavior_failure import SEVERITY_HIGH
from d052.feedback_llm_ued.feedback_contracts import (
    CONTEXT_CLOSE,
    CONTEXT_OPEN,
    FeedbackRoleEnvelope,
)
from d052.feedback_llm_ued.simulator_feedback_store import MATCH_UNGRADED
from d052.schemas.common import CanonicalModel

ROLE = C.ROLE_CRITIC_SKEPTIC
PROMPT_VERSION = f"{C.ROLE_PROMPT_VERSION}.{ROLE}"

GLOBAL_RISKS = ("LOW", "MEDIUM", "HIGH")

#: escalation thresholds (documented round choice)
HIGH_SEVERITY_FLOOR = 3          # >= 3 high-severity gaps  -> HIGH risk
OPPOSITE_MATCH_FLOOR = 2         # >= 2 opposite predictions -> HIGH risk
WIDE_CI = 0.5                    # pooled CI half-width this wide -> concern

PROMPT_TEMPLATE = f"""\
You are the Critic/Skeptic role of the six-role Review Board of the
simulator-grounded feedback-adaptive LLM-UED loop. Independently of the other
five roles, re-examine the raw board context: severe Student-vs-Reference
gaps, expected-vs-observed contradictions, feedback records never graded by
the comparator, and wide confidence intervals. List objections, set the
global risk (LOW / MEDIUM / HIGH), and say whether the board's conclusions
should be endorsed or escalated to human control. Trust nothing the other
roles claim — recompute from the evidence.
Prompt version: {{prompt_version}}
{CONTEXT_OPEN}
{{context_json}}
{CONTEXT_CLOSE}
Respond with a single JSON object matching the CriticOutput schema.
"""


class CriticOutput(CanonicalModel):
    window: int = Field(ge=0)
    objections: List[str] = Field(default_factory=list)
    global_risk: str = "LOW"
    request_control: bool = False
    endorsed: bool = True
    honesty_check_passed: bool = True
    critique_summary: str = ""

    @model_validator(mode="after")
    def _risk_legal(self) -> "CriticOutput":
        if self.global_risk not in GLOBAL_RISKS:
            raise ValueError(f"ILLEGAL_GLOBAL_RISK: {self.global_risk!r}")
        return self


def build_prompt(context: dict) -> str:
    return PROMPT_TEMPLATE.format(
        prompt_version=PROMPT_VERSION,
        context_json=json.dumps(context, sort_keys=True, ensure_ascii=False,
                                default=str))


def parse(raw: str) -> CriticOutput:
    return CriticOutput.model_validate_json(raw)


#: output class exposed for the board assembler (single source of truth)
OUTPUT_MODEL = CriticOutput


def mock_rule(context: dict) -> dict:
    """Deterministically recompute skepticism from the raw board context."""
    window = int(context.get("window", 0))
    bc = context.get("board_context", {})
    evidence = bc.get("behavior_evidence", [])
    feedback = context.get("feedback", [])
    ci = float(bc.get("student_success_rate_ci", 1.0))

    high_sev = sum(1 for e in evidence
                   if e.get("severity") == SEVERITY_HIGH)
    opposite = sum(1 for f in feedback
                   if f.get("expected_observed_match")
                   == C.MATCH_DIRECTION_OPPOSITE)
    ungraded = [f.get("feedback_id", "") for f in feedback
                if f.get("expected_observed_match") == MATCH_UNGRADED]

    objections: List[str] = []
    if high_sev:
        objections.append(f"{high_sev} high-severity Student-vs-Reference "
                          f"gap(s) in window evidence")
    if opposite:
        objections.append(f"{opposite} probe record(s) observed the OPPOSITE "
                          f"of their predicted signature")
    if ungraded:
        objections.append(
            f"feedback never graded by the comparator: {sorted(ungraded)}")
    if ci >= WIDE_CI:
        objections.append(f"pooled Student success rate CI half-width "
                          f"{ci:.3f} >= {WIDE_CI}: evidence is thin")

    if high_sev >= HIGH_SEVERITY_FLOOR or opposite >= OPPOSITE_MATCH_FLOOR:
        risk = "HIGH"
    elif high_sev >= 1 or opposite >= 1 or ci >= WIDE_CI or ungraded:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    # C11: escalation halts the loop, so it is decoupled from risk grading.
    # Escalate only on an honesty violation (ungraded feedback) or on HIGH
    # risk built from THIN evidence (wide CI). Severe-but-precise evidence
    # stays HIGH risk without stopping the autonomous loop.
    escalate = bool(ungraded) or (risk == "HIGH" and ci >= WIDE_CI)

    return dict(window=window,
                objections=objections,
                global_risk=risk,
                request_control=escalate,
                endorsed=(risk != "HIGH"),
                honesty_check_passed=(not ungraded),
                critique_summary=(
                    f"window {window}: risk={risk} from {high_sev} "
                    f"high-severity gap(s), {opposite} opposite match(es), "
                    f"{len(ungraded)} ungraded record(s)"))


def run(context: dict, backend, window: int, sequence: int
        ) -> FeedbackRoleEnvelope:
    prompt = build_prompt(context)
    raw = backend.complete(ROLE, prompt)
    parsed = parse(raw)
    return FeedbackRoleEnvelope.make(
        role=ROLE, prompt_version=PROMPT_VERSION, backend_id=backend.backend_id,
        model_id=backend.model_id, window=window, sequence=sequence,
        prompt=prompt, raw_response=raw, parsed_dump=parsed.model_dump())
