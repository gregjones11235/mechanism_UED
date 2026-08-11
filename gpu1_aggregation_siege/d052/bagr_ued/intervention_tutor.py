"""Role: InterventionTutor (task section 7).

Converts supported causal hypotheses into ENVIRONMENT-INDUCTION directions:
legal TaskParams mutation axes, controlled variables, counterfactual group
structure (control + single-axis groups), and expected GLOBAL effects.

Forbidden outputs (validation + board-level TrajectorySupervisionGuard):
  * "flee when seeing a monster", "walk left", "don't sleep" — any direct
    action instruction
  * fixed action sequences
  * reward / penalty modifications (not an axis; not even representable here)

For the required unsafe_rest synthetic test the generated interventions
collectively cover: threat-distance grading, safe-rest-area availability,
rest-need pressure, threat count, and view occlusion.
"""
from __future__ import annotations

import json
from typing import List

from pydantic import Field, field_validator, model_validator

from d052.bagr_ued import constants as C
from d052.bagr_ued.causal_failure_analyst import CATEGORY_AXES
from d052.bagr_ued.review_contracts import CONTEXT_CLOSE, CONTEXT_OPEN, RoleEnvelope
from d052.schemas.common import CanonicalModel

ROLE = C.ROLE_INTERVENTION_TUTOR
PROMPT_VERSION = f"{C.ROLE_PROMPT_VERSION}.{ROLE}"

PROMPT_TEMPLATE = f"""\
You are the InterventionTutor role of the BA-BAGR-UED review board.
Convert the causal hypotheses into environment-induction interventions:
legal TaskParams MUTATION AXES only, explicit controlled variables, a
counterfactual group structure containing a control group, and expected
GLOBAL effects. You may NEVER prescribe Student actions, action sequences,
or reward/penalty changes.
Prompt version: {{prompt_version}}
{CONTEXT_OPEN}
{{context_json}}
{CONTEXT_CLOSE}
Respond with a single JSON object matching the InterventionTutorOutput schema.
"""


class InterventionHypothesis(CanonicalModel):
    intervention_id: str = Field(min_length=1)
    target_hypothesis_ids: List[str] = Field(min_length=1)
    mutation_axes: List[str] = Field(min_length=1)
    controlled_variables: List[str] = Field(default_factory=list)
    counterfactual_groups: List[str] = Field(default_factory=list)
    expected_behavior_change: str = Field(min_length=1)
    expected_global_effect: str = Field(min_length=1)
    ued_justification: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("mutation_axes", "controlled_variables")
    @classmethod
    def _legal_axes(cls, v: List[str]) -> List[str]:
        for a in v:
            if a not in C.MUTATION_AXES:
                raise ValueError(f"ILLEGAL_MUTATION_AXIS: {a!r} is not in the "
                                 f"legal mutation-axis vocabulary")
        return v

    @model_validator(mode="after")
    def _control_group(self) -> "InterventionHypothesis":
        if self.counterfactual_groups and \
                "control" not in self.counterfactual_groups:
            raise ValueError(
                "MISSING_CONTROL_GROUP: counterfactual_groups must contain a "
                "'control' group")
        overlap = set(self.mutation_axes) & set(self.controlled_variables)
        if overlap:
            raise ValueError(
                f"AXIS_BOTH_MUTATED_AND_CONTROLLED: {sorted(overlap)}")
        return self


class InterventionTutorOutput(CanonicalModel):
    intervention_hypotheses: List[InterventionHypothesis] = Field(
        default_factory=list)


def build_prompt(context: dict) -> str:
    return PROMPT_TEMPLATE.format(
        prompt_version=PROMPT_VERSION,
        context_json=json.dumps(context, sort_keys=True, ensure_ascii=False,
                                default=str))


def parse(raw: str) -> InterventionTutorOutput:
    return InterventionTutorOutput.model_validate_json(raw)


def mock_rule(context: dict) -> dict:
    """Group hypotheses by cause category -> one intervention per category."""
    hypotheses = context.get("causal_hypotheses", [])
    by_cat: dict = {}
    for h in hypotheses:
        axes = CATEGORY_AXES.get(h["cause_category"], [])
        if not axes:
            continue  # e.g. implementation_or_adapter_bug: no env intervention
        by_cat.setdefault(h["cause_category"], []).append(h)

    all_axes = set(C.MUTATION_AXES)
    interventions = []
    for i, (cat, hs) in enumerate(sorted(by_cat.items())):
        axes = sorted(set(a for h in hs for a in CATEGORY_AXES[h["cause_category"]]))
        ids = sorted({h["hypothesis_id"] for h in hs})
        patterns = sorted({h["finding_id"] for h in hs})
        controlled = sorted(all_axes - set(axes))
        groups = ["control"] + [f"single_axis:{a}" for a in axes]
        if len(axes) >= 2:
            groups.append(f"factorial:{axes[0]}x{axes[1]}")
        conf = round(max(0.1, sum(h["confidence"] for h in hs) / len(hs) * 0.9), 4)
        interventions.append(dict(
            intervention_id=f"itv:{cat}:{i:02d}",
            target_hypothesis_ids=ids,
            mutation_axes=axes,
            controlled_variables=controlled,
            counterfactual_groups=groups,
            expected_behavior_change=(
                f"Occurrence rate of behaviors behind findings "
                f"{patterns} should respond monotonically to graded "
                f"axes {axes}, with the control group holding all other "
                f"axes constant."),
            expected_global_effect=(
                "Measured as movement in GLOBAL regret and the behavioral "
                "gap across global scenarios — not a floor-specific metric."),
            ued_justification=(
                f"Discriminates cause category {cat!r} from its competing "
                f"categories via a single-axis counterfactual contrast; "
                f"induces environment structure, never Student actions."),
            confidence=conf,
        ))
    return dict(intervention_hypotheses=interventions)


def run(context: dict, backend, sequence: int) -> RoleEnvelope:
    prompt = build_prompt(context)
    raw = backend.complete(ROLE, prompt)
    parsed = parse(raw)
    return RoleEnvelope.make(
        role=ROLE, prompt_version=PROMPT_VERSION, backend_id=backend.backend_id,
        model_id=backend.model_id, sequence=sequence, prompt=prompt,
        raw_response=raw, parsed_dump=parsed.model_dump())
