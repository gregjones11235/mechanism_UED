"""Role: CausalFailureAnalyst (task section 6).

For every behavior finding, emits MULTIPLE COMPETING causal hypotheses drawn
from the closed cause-category vocabulary. Each hypothesis carries supporting
AND contradicting evidence, alternative explanations, a testable prediction,
and the counterfactual environment variables required to test it.

Forbidden by validation (fail-closed):
  * a finding explained by a single hypothesis  -> SINGLE_CAUSE_FORBIDDEN
  * a cause category outside the vocabulary     -> UNKNOWN_CAUSE_CATEGORY
  * counterfactual variables outside the legal mutation-axis vocabulary
                                                -> ILLEGAL_COUNTERFACTUAL_VARIABLE
  * correlation asserted as proven causation (statements are hedged:
    "consistent with", "may", and always list contradicting evidence)
  * direct Student actions/paths (TrajectorySupervisionGuard scans this
    role's output at the board level)
"""
from __future__ import annotations

import json
from typing import List

from pydantic import Field, field_validator, model_validator

from d052.bagr_ued import constants as C
from d052.bagr_ued.review_contracts import CONTEXT_CLOSE, CONTEXT_OPEN, RoleEnvelope
from d052.schemas.common import CanonicalModel

ROLE = C.ROLE_CAUSAL_FAILURE_ANALYST
PROMPT_VERSION = f"{C.ROLE_PROMPT_VERSION}.{ROLE}"

PROMPT_TEMPLATE = f"""\
You are the CausalFailureAnalyst role of the BA-BAGR-UED review board.
For EACH behavior finding, propose MULTIPLE COMPETING causal hypotheses from
the closed cause-category vocabulary. Every hypothesis needs supporting AND
contradicting evidence, alternative explanations, a falsifiable prediction,
and the counterfactual variables that would test it. Never assert correlation
as proven causation; never prescribe Student actions.
Prompt version: {{prompt_version}}
{CONTEXT_OPEN}
{{context_json}}
{CONTEXT_CLOSE}
Respond with a single JSON object matching the CausalAnalystOutput schema.
"""

#: behavior pattern -> competing cause categories (>=2 each; order = prior rank)
PATTERN_CAUSE_ROTATION = {
    "unsafe_rest_near_hostile": [
        "value_or_risk_misestimation", "perception_or_observability",
        "memory_or_context_retention", "environment_ambiguity"],
    "repeated_no_effect": [
        "action_semantics_confusion", "implementation_or_adapter_bug",
        "exploration_noise"],
    "oscillation_loop": [
        "exploration_noise", "memory_or_context_retention",
        "value_or_risk_misestimation"],
    "combat_freeze": [
        "action_semantics_confusion", "value_or_risk_misestimation",
        "perception_or_observability"],
    "resource_neglect": [
        "resource_planning_failure", "memory_or_context_retention",
        "perception_or_observability"],
    "unprepared_threat_approach": [
        "resource_planning_failure", "value_or_risk_misestimation",
        "perception_or_observability", "distribution_shift"],
    "progress_regression": [
        "memory_or_context_retention", "resource_planning_failure",
        "distribution_shift", "value_or_risk_misestimation"],
    "premature_terminal": [
        "value_or_risk_misestimation", "perception_or_observability", "unknown"],
}

CATEGORY_STATEMENT = {
    "perception_or_observability": "The behavior is consistent with the "
        "relevant threat/safety signal being poorly discriminable from the "
        "Student's observation.",
    "memory_or_context_retention": "The behavior is consistent with the "
        "Student failing to retain the recent threat/need context across "
        "steps.",
    "value_or_risk_misestimation": "The behavior is consistent with the "
        "Student over-weighting the immediate need value relative to survival "
        "risk.",
    "resource_planning_failure": "The behavior is consistent with a failure "
        "to plan resource acquisition ahead of need.",
    "exploration_noise": "The behavior may be incidental exploration noise "
        "rather than a systematic policy defect.",
    "action_semantics_confusion": "The behavior is consistent with a "
        "misassociation between an action and its effect.",
    "distribution_shift": "The behavior is consistent with the situation "
        "lying outside the Student's training distribution.",
    "environment_ambiguity": "The behavior may be explained by the "
        "environment itself lacking a discriminable safety signal (not a "
        "Student defect).",
    "implementation_or_adapter_bug": "The behavior may be an artifact of an "
        "observation/action adapter bug; this requires code inspection, not "
        "an environment intervention.",
    "unknown": "The current evidence does not narrow the cause class.",
}

CATEGORY_PREDICTION = {
    "perception_or_observability": "If observability (occlusion/visibility) "
        "is graded while threat structure is held constant, the occurrence "
        "rate should vary monotonically with observability.",
    "memory_or_context_retention": "If the long-term memory requirement is "
        "increased while local cues are held constant, recurrence should "
        "increase.",
    "value_or_risk_misestimation": "If threat-distance grading is varied, "
        "the behavior should concentrate at the short-distance grades.",
    "resource_planning_failure": "If need pressure is varied while threat "
        "structure is held constant, the behavior should scale with pressure.",
    "exploration_noise": "If the same situation is replayed with different "
        "exploration seeds, the occurrence rate should fluctuate rather than "
        "persist.",
    "action_semantics_confusion": "If a safe variant of the relevant action "
        "context is made available, the behavior rate should drop.",
    "distribution_shift": "If the situation is moved toward the training "
        "distribution, the behavior should diminish.",
    "environment_ambiguity": "If an explicit safety signal is added to the "
        "environment, the behavior should diminish even with an unchanged "
        "Student.",
    "implementation_or_adapter_bug": "No environment intervention applies; "
        "the prediction is that adapter code inspection reproduces the "
        "anomaly deterministically.",
    "unknown": "No targeted prediction; the counterfactual design is "
        "exploratory.",
}

#: cause category -> legal counterfactual variables (mutation axes ONLY)
CATEGORY_AXES = {
    "perception_or_observability": ["view_occlusion", "visibility"],
    "memory_or_context_retention": ["long_term_memory_requirement"],
    "value_or_risk_misestimation": ["threat_distance_grading", "threat_count"],
    "resource_planning_failure": ["resource_pressure", "rest_need_pressure"],
    "exploration_noise": ["threat_distance_grading"],
    "action_semantics_confusion": ["safe_rest_area_availability"],
    "distribution_shift": ["multi_threat_interference", "global_task_conflict"],
    "environment_ambiguity": ["safe_rest_area_availability",
                              "rest_need_pressure", "view_occlusion"],
    "implementation_or_adapter_bug": [],
    "unknown": [],
}


class CausalHypothesis(CanonicalModel):
    hypothesis_id: str = Field(min_length=1)
    finding_id: str = Field(min_length=1)
    cause_category: str = Field(min_length=1)
    causal_statement: str = Field(min_length=1)
    supporting_evidence: List[str] = Field(default_factory=list)
    contradicting_evidence: List[str] = Field(default_factory=list)
    alternative_explanations: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    testable_prediction: str = Field(min_length=1)
    required_counterfactual_variables: List[str] = Field(default_factory=list)

    @field_validator("cause_category")
    @classmethod
    def _category(cls, v: str) -> str:
        if v not in C.CAUSE_CATEGORIES:
            raise ValueError(f"UNKNOWN_CAUSE_CATEGORY: {v!r} not in the closed "
                             f"cause-category vocabulary")
        return v

    @field_validator("required_counterfactual_variables")
    @classmethod
    def _axes(cls, v: List[str]) -> List[str]:
        for a in v:
            if a not in C.MUTATION_AXES:
                raise ValueError(f"ILLEGAL_COUNTERFACTUAL_VARIABLE: {a!r} is "
                                 f"not a legal mutation axis")
        return v


class CausalAnalystOutput(CanonicalModel):
    causal_hypotheses: List[CausalHypothesis] = Field(default_factory=list)

    @model_validator(mode="after")
    def _multi_cause(self) -> "CausalAnalystOutput":
        by_finding: dict = {}
        for h in self.causal_hypotheses:
            by_finding.setdefault(h.finding_id, set()).add(h.cause_category)
        for fid, cats in by_finding.items():
            if len(cats) < 2:
                raise ValueError(
                    f"SINGLE_CAUSE_FORBIDDEN: finding {fid} has fewer than 2 "
                    f"competing cause categories ({sorted(cats)}); single-"
                    f"evidence single-cause attribution is forbidden")
        return self


def build_prompt(context: dict) -> str:
    return PROMPT_TEMPLATE.format(
        prompt_version=PROMPT_VERSION,
        context_json=json.dumps(context, sort_keys=True, ensure_ascii=False,
                                default=str))


def parse(raw: str) -> CausalAnalystOutput:
    return CausalAnalystOutput.model_validate_json(raw)


def mock_rule(context: dict) -> dict:
    """Deterministically generate competing hypotheses per finding."""
    hypotheses = []
    findings = context.get("behavior_findings", [])
    for f in sorted(findings, key=lambda x: x["finding_id"]):
        cats = PATTERN_CAUSE_ROTATION.get(
            f["behavior_pattern"],
            ["unknown", "implementation_or_adapter_bug"])
        n = len(cats)
        for i, cat in enumerate(cats):
            conf = round(max(0.15, 0.6 - 0.12 * i), 4)
            hypotheses.append(dict(
                hypothesis_id=f"hyp:{f['finding_id']}:{cat}",
                finding_id=f["finding_id"],
                cause_category=cat,
                causal_statement=CATEGORY_STATEMENT[cat],
                supporting_evidence=[
                    f"pattern={f['behavior_pattern']}",
                    f"severity={f['severity']}",
                    f"recurrence={f['recurrence']}",
                ] + list(f.get("supporting_fields", []))[:2],
                contradicting_evidence=(
                    list(f.get("counter_evidence", []))[:2]
                    or ["no recorded counter-evidence for this category"]),
                alternative_explanations=[c for c in cats if c != cat],
                confidence=conf,
                testable_prediction=CATEGORY_PREDICTION[cat],
                required_counterfactual_variables=list(CATEGORY_AXES[cat]),
            ))
        assert n >= 2, "SINGLE_CAUSE_RULE_BROKEN"
    return dict(causal_hypotheses=hypotheses)


def run(context: dict, backend, sequence: int) -> RoleEnvelope:
    prompt = build_prompt(context)
    raw = backend.complete(ROLE, prompt)
    parsed = parse(raw)
    return RoleEnvelope.make(
        role=ROLE, prompt_version=PROMPT_VERSION, backend_id=backend.backend_id,
        model_id=backend.model_id, sequence=sequence, prompt=prompt,
        raw_response=raw, parsed_dump=parsed.model_dump())
