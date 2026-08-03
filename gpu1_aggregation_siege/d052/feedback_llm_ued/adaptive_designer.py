"""Role: AdaptiveEnvironmentDesigner (task §3 role 2 of the default 2 calls).

Consumes the FeedbackDiagnostician's verdicts and turns them into the NEXT
curriculum plan's family-level modifications — the plan_{k+1} that closes the
plan_k -> probe -> feedback -> diagnosis -> plan_{k+1} loop. This is an
ENVIRONMENT-LEVEL role only: it moves slot budgets across environment families
via RETAIN / MUTATE / RETIRE / EXPAND_BUDGET / REDUCE_BUDGET / REQUEST_CONTROL.
It never touches action, reward, or policy knobs.

Honesty invariant (mirrors PlanRevisionRecord): a modification that cites at
least one feedback id is feedback-driven; a modification with NO feedback id
may only be EXPLORATION (MUTATE / EXPAND_BUDGET, is_exploration=True). The
DeterministicReconciler re-checks this, so a masquerade cannot slip through
even if a (real) LLM tried to emit one.
"""
from __future__ import annotations

import json
from typing import Dict, List

from pydantic import Field, model_validator

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.feedback_contracts import (
    CONTEXT_CLOSE,
    CONTEXT_OPEN,
    FamilyAllocation,
    FeedbackRoleEnvelope,
)
from d052.schemas.common import CanonicalModel

ROLE = C.ROLE_ADAPTIVE_ENVIRONMENT_DESIGNER
PROMPT_VERSION = f"{C.ROLE_PROMPT_VERSION}.{ROLE}"

PROMPT_TEMPLATE = f"""\
You are the AdaptiveEnvironmentDesigner role of the simulator-grounded
feedback-adaptive LLM-UED loop. From the FeedbackDiagnostician's verdicts and
the hypothesis ledger, propose the next curriculum plan as environment-FAMILY
slot modifications (RETAIN / MUTATE / RETIRE / EXPAND_BUDGET / REDUCE_BUDGET /
REQUEST_CONTROL). Every feedback-driven modification must cite the feedback
ids it is based on. Environment-level only: no action / reward / policy knobs.
Prompt version: {{prompt_version}}
{CONTEXT_OPEN}
{{context_json}}
{CONTEXT_CLOSE}
Respond with a single JSON object matching the DesignerOutput schema.
"""


class DesignerOutput(CanonicalModel):
    window: int = Field(ge=0)
    allocations: List[FamilyAllocation] = Field(default_factory=list)
    rationale: str = Field(default="")
    request_control: bool = False

    @model_validator(mode="after")
    def _honesty(self) -> "DesignerOutput":
        for a in self.allocations:
            if not a.based_on_feedback_ids:
                if not a.is_exploration:
                    raise ValueError(
                        f"EXPLORATION_LABEL_REQUIRED: uncited allocation for "
                        f"{a.environment_family!r} must be exploration")
                if a.decision not in C.EXPLORATION_DECISIONS:
                    raise ValueError(
                        f"EXPLORATION_DECISION_ONLY: uncited allocation for "
                        f"{a.environment_family!r} may only use "
                        f"{sorted(C.EXPLORATION_DECISIONS)}")
            elif a.is_exploration:
                raise ValueError(
                    f"MASQUERADE_FORBIDDEN: cited allocation for "
                    f"{a.environment_family!r} may not be exploration")
        return self


def build_prompt(context: dict) -> str:
    return PROMPT_TEMPLATE.format(
        prompt_version=PROMPT_VERSION,
        context_json=json.dumps(context, sort_keys=True, ensure_ascii=False,
                                default=str))


def parse(raw: str) -> DesignerOutput:
    return DesignerOutput.model_validate_json(raw)


def _decision_for(verdict: str) -> str:
    if verdict == C.HYPOTHESIS_SUPPORTED:
        return C.DECISION_RETAIN
    if verdict == C.HYPOTHESIS_REFUTED:
        return C.DECISION_RETIRE
    if verdict == C.HYPOTHESIS_INCONCLUSIVE:
        return C.DECISION_MUTATE
    return C.DECISION_MUTATE                 # STALE -> re-probe via mutation


def mock_rule(context: dict) -> dict:
    """Deterministically turn verdicts into family-level slot modifications.

    SUPPORTED   -> RETAIN the family (keep budget);
    REFUTED     -> RETIRE the family (free budget);
    INCONCLUSIVE-> MUTATE the family (sharpen so it discriminates);
    STALE       -> MUTATE as EXPLORATION (no feedback cited this window).

    Families with no hypothesis at all are proposed as bounded EXPLORATION.
    A HIGH global risk from the diagnostician adds a REQUEST_CONTROL flag.
    """
    window = int(context.get("window", 0))
    verdicts = context.get("verdicts", [])
    hypotheses = context.get("hypotheses", [])
    hyp_family = {h["hypothesis_id"]: h["environment_family"] for h in hypotheses}
    budget_hint = context.get("budget", C.DYNAMIC_UED_SLOTS)
    global_risk = context.get("global_risk", "LOW")

    allocs: List[dict] = []
    seen_families: set = set()
    # feedback-driven modifications, one per hypothesis verdict
    for v in sorted(verdicts, key=lambda x: x["hypothesis_id"]):
        fam = hyp_family.get(v["hypothesis_id"])
        if fam is None or fam in seen_families:
            continue
        decision = _decision_for(v["verdict"])
        if v["verdict"] == C.HYPOTHESIS_STALE:
            # no feedback cited it -> must be exploration, honest label
            allocs.append(dict(environment_family=fam, decision=C.DECISION_MUTATE,
                               slots=1, based_on_feedback_ids=[],
                               reason=f"stale hypothesis {v['hypothesis_id']}: "
                                      f"re-probe as exploration",
                               is_exploration=True))
        else:
            slots = 0 if decision == C.DECISION_RETIRE else 2
            allocs.append(dict(environment_family=fam, decision=decision,
                               slots=slots,
                               based_on_feedback_ids=list(v["feedback_ids"]),
                               reason=f"{v['verdict']} by probe feedback for "
                                      f"hypothesis {v['hypothesis_id']}",
                               is_exploration=False))
        seen_families.add(fam)

    # bounded exploration over families with no hypothesis yet
    exploration_left = C.MAX_EXPLORATION_PROPOSALS
    for fam in C.ENVIRONMENT_FAMILIES:
        if exploration_left <= 0:
            break
        if fam in seen_families:
            continue
        allocs.append(dict(environment_family=fam, decision=C.DECISION_MUTATE,
                           slots=1, based_on_feedback_ids=[],
                           reason="no hypothesis yet: explore this family",
                           is_exploration=True))
        seen_families.add(fam)
        exploration_left -= 1

    rationale = (f"window {window}: {len(allocs)} family allocation(s) derived "
                 f"from {len(verdicts)} verdict(s); budget target {budget_hint}; "
                 f"global_risk={global_risk}")
    return dict(window=window, allocations=allocs, rationale=rationale,
                request_control=(global_risk == "HIGH"))


def run(context: dict, backend, window: int, sequence: int) -> FeedbackRoleEnvelope:
    prompt = build_prompt(context)
    raw = backend.complete(ROLE, prompt)
    parsed = parse(raw)
    return FeedbackRoleEnvelope.make(
        role=ROLE, prompt_version=PROMPT_VERSION, backend_id=backend.backend_id,
        model_id=backend.model_id, window=window, sequence=sequence,
        prompt=prompt, raw_response=raw, parsed_dump=parsed.model_dump())
