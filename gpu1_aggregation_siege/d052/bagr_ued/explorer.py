"""Role: Explorer (task section 8).

Proposes environment FAMILIES DIFFERENT from the Tutor's main-hypothesis
interventions (resource pressure, day/night-rest need, visibility,
multi-threat interference, long-term memory requirement, global task
conflict...). Every proposal states its novelty, its difference from existing
proposals, a testable prediction, its global UED value, and potential side
effects. The mock rule guarantees family-level disjointness from the axes the
Tutor already mutates.
"""
from __future__ import annotations

import json
from typing import List

from pydantic import Field, field_validator

from d052.bagr_ued import constants as C
from d052.bagr_ued.review_contracts import CONTEXT_CLOSE, CONTEXT_OPEN, RoleEnvelope
from d052.schemas.common import CanonicalModel

ROLE = C.ROLE_EXPLORER
PROMPT_VERSION = f"{C.ROLE_PROMPT_VERSION}.{ROLE}"

PROMPT_TEMPLATE = f"""\
You are the Explorer role of the BA-BAGR-UED review board.
Propose environment families DIFFERENT from the Tutor's intervention axes.
For each: novelty statement, difference from existing proposals, a testable
prediction, global UED value, and potential side effects. No Student action
advice; no Tier3-only framing.
Prompt version: {{prompt_version}}
{CONTEXT_OPEN}
{{context_json}}
{CONTEXT_CLOSE}
Respond with a single JSON object matching the ExplorerOutput schema.
"""

#: family -> the mutation axes that would make it "already covered" by Tutor
FAMILY_PRIMARY_AXES = {
    "threat_distance_family": {"threat_distance_grading", "threat_count"},
    "resource_pressure_family": {"resource_pressure"},
    "day_night_rest_need_family": {"day_night_rest_need", "rest_need_pressure"},
    "visibility_family": {"visibility", "view_occlusion"},
    "multi_threat_interference_family": {"multi_threat_interference"},
    "long_term_memory_family": {"long_term_memory_requirement"},
    "global_task_conflict_family": {"global_task_conflict"},
}

FAMILY_PREDICTION = {
    "threat_distance_family": "Behavior rates should track the threat-"
        "distance grade even when rest/resource structure is fixed.",
    "resource_pressure_family": "Failure behaviors should scale with resource "
        "pressure even when threat structure is fixed.",
    "day_night_rest_need_family": "Unsafe-rest behaviors should couple to the "
        "day/night cycle only under high rest-need coupling.",
    "visibility_family": "Unprepared-approach behaviors should increase with "
        "occlusion when threat density is fixed.",
    "multi_threat_interference_family": "Combat-freeze behaviors should "
        "increase with the number of interfering threats.",
    "long_term_memory_family": "Progress-regression behaviors should increase "
        "with the required memory horizon.",
    "global_task_conflict_family": "Resource-neglect and planning failures "
        "should increase with inter-objective conflict.",
}


class AlternativeFamilyProposal(CanonicalModel):
    proposal_id: str = Field(min_length=1)
    environment_family: str = Field(min_length=1)
    novelty_statement: str = Field(min_length=1)
    difference_from_existing: str = Field(min_length=1)
    testable_prediction: str = Field(min_length=1)
    global_ued_value: str = Field(min_length=1)
    potential_side_effects: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("environment_family")
    @classmethod
    def _family(cls, v: str) -> str:
        if v not in C.ENVIRONMENT_FAMILIES:
            raise ValueError(f"UNKNOWN_ENVIRONMENT_FAMILY: {v!r}")
        return v


class ExplorerOutput(CanonicalModel):
    alternative_environment_proposals: List[AlternativeFamilyProposal] = \
        Field(default_factory=list)


def build_prompt(context: dict) -> str:
    return PROMPT_TEMPLATE.format(
        prompt_version=PROMPT_VERSION,
        context_json=json.dumps(context, sort_keys=True, ensure_ascii=False,
                                default=str))


def parse(raw: str) -> ExplorerOutput:
    return ExplorerOutput.model_validate_json(raw)


def mock_rule(context: dict) -> dict:
    """Propose families disjoint from the Tutor's mutated axes."""
    tutor_axes = set()
    for itv in context.get("intervention_hypotheses", []):
        tutor_axes.update(itv.get("mutation_axes", []))
    covered = {fam for fam, axes in FAMILY_PRIMARY_AXES.items()
               if axes & tutor_axes}
    fresh = [fam for fam in C.ENVIRONMENT_FAMILIES if fam not in covered]
    if not fresh:  # everything covered -> still propose the least-overlapping
        fresh = list(C.ENVIRONMENT_FAMILIES)
    proposals = []
    for i, fam in enumerate(fresh[:4]):
        proposals.append(dict(
            proposal_id=f"exp:{fam}:{i:02d}",
            environment_family=fam,
            novelty_statement=(f"{fam} is not among the families implied by "
                               f"the Tutor's mutated axes {sorted(tutor_axes)}; "
                               f"it probes an orthogonal global structure."),
            difference_from_existing=(f"Differs from Tutor interventions "
                                      f"(axes {sorted(tutor_axes)}) and from "
                                      f"other Explorer families by its primary "
                                      f"structure {sorted(FAMILY_PRIMARY_AXES[fam])}."),
            testable_prediction=FAMILY_PREDICTION[fam],
            global_ued_value=("Extends global-regret coverage beyond the "
                              "Tier3 threat axis; supports GLOBAL, not "
                              "Tier3-only, curriculum pressure."),
            potential_side_effects=("May raise episode difficulty variance; "
                                    "must be balanced by the control group in "
                                    "the counterfactual design."),
            confidence=0.5,
        ))
    return dict(alternative_environment_proposals=proposals)


def run(context: dict, backend, sequence: int) -> RoleEnvelope:
    prompt = build_prompt(context)
    raw = backend.complete(ROLE, prompt)
    parsed = parse(raw)
    return RoleEnvelope.make(
        role=ROLE, prompt_version=PROMPT_VERSION, backend_id=backend.backend_id,
        model_id=backend.model_id, sequence=sequence, prompt=prompt,
        raw_response=raw, parsed_dump=parsed.model_dump())
