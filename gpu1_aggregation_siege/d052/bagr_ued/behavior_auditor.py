"""Role: BehaviorAuditor (task section 5).

Answers ONLY: what unreasonable/inefficient behaviors occurred, where the
evidence is, whether they recur, their severity, and whether they may be
incidental exploration. The output schema (extra=forbid) STRUCTURALLY cannot
carry a failure-cause conclusion, an environment proposal, or action advice —
those belong to later roles.
"""
from __future__ import annotations

import json
from typing import List

from pydantic import Field, model_validator

from d052.bagr_ued import constants as C
from d052.bagr_ued.review_contracts import CONTEXT_CLOSE, CONTEXT_OPEN, RoleEnvelope
from d052.schemas.common import CanonicalModel

ROLE = C.ROLE_BEHAVIOR_AUDITOR
PROMPT_VERSION = f"{C.ROLE_PROMPT_VERSION}.{ROLE}"

PROMPT_TEMPLATE = f"""\
You are the BehaviorAuditor role of the BA-BAGR-UED review board.
From the deterministic anomaly candidates below, report ONLY behavior
findings: pattern, severity, recurrence, evidence, counter-evidence, and
whether recurrence suggests incidental exploration. Do NOT infer failure
causes, do NOT propose environments, do NOT advise actions.
You MAY additionally report OUT-OF-TAXONOMY observations as PROVISIONAL
anomaly hypotheses (provisional=true, requires_deterministic_validation=true,
with evidence clip ids, the observed pattern, a confidence, and an
alternative explanation). Provisional hypotheses are evidence bookkeeping
ONLY — they never enter the selector or the batch plan without later
deterministic validation and real rollout evidence.
Prompt version: {{prompt_version}}
{CONTEXT_OPEN}
{{context_json}}
{CONTEXT_CLOSE}
Respond with a single JSON object matching the BehaviorAuditorOutput schema.
"""


class BehaviorFinding(CanonicalModel):
    finding_id: str = Field(min_length=1)
    behavior_pattern: str = Field(min_length=1)
    severity: float = Field(ge=0.0, le=1.0)
    recurrence: int = Field(ge=1)
    evidence_span_ids: List[str] = Field(default_factory=list)
    supporting_fields: List[str] = Field(default_factory=list)
    counter_evidence: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class ProvisionalAnomalyHypothesis(CanonicalModel):
    """CC3 fix2 (§13): an OUT-OF-TAXONOMY behavior observation, PROVISIONAL.

    The board may notice behavior the fixed detector taxonomy does not cover
    and record it here — but ONLY as a provisional, evidence-bound hypothesis
    that REQUIRES deterministic validation before anything downstream may act
    on it. A provisional hypothesis NEVER enters the selector, the Soft
    Copeland ranking, the budget plan, or the archive: it must pass schema
    validation -> Critic/Skeptic scrutiny -> Reconciler handling -> an
    environment-level counterfactual proposal -> later REAL rollout evidence.
    This round exercises the contract with mock/synthetic evidence only.
    """

    hypothesis_id: str = Field(min_length=1)
    #: MUST be true — a non-provisional "new anomaly" from the LLM path is a
    #: contract violation (refused by the validator below)
    provisional: bool = True
    taxonomy_status: str = Field(
        pattern=r"^out_of_taxonomy$", default="out_of_taxonomy")
    evidence_clip_ids: List[str] = Field(default_factory=list)
    observed_pattern: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    alternative_explanation: str = Field(min_length=1)
    #: MUST be true — provisional findings require deterministic validation
    requires_deterministic_validation: bool = True
    #: hard downstream contract, recorded so no consumer can claim ignorance
    selector_or_batch_entry_forbidden: bool = True

    @model_validator(mode="after")
    def _provisional_contract(self) -> "ProvisionalAnomalyHypothesis":
        if not self.provisional:
            raise ValueError(
                "PROVISIONAL_HYPOTHESIS_MUST_BE_PROVISIONAL: an LLM-proposed "
                "new anomaly can only enter as provisional=true")
        if not self.requires_deterministic_validation:
            raise ValueError(
                "PROVISIONAL_REQUIRES_DETERMINISTIC_VALIDATION: provisional "
                "hypotheses must require deterministic validation")
        if not self.selector_or_batch_entry_forbidden:
            raise ValueError(
                "PROVISIONAL_SELECTOR_ENTRY_FORBIDDEN: provisional hypotheses "
                "may never enter the selector or batch plan directly")
        return self


class BehaviorAuditorOutput(CanonicalModel):
    behavior_findings: List[BehaviorFinding] = Field(default_factory=list)
    #: CC3 fix2 (§13): provisional out-of-taxonomy observations. Recorded for
    #: the audit trail; NOT consumed by the reconciler's acceptance rules and
    #: NEVER by the selector/budget/archive chain.
    provisional_anomaly_hypotheses: List[ProvisionalAnomalyHypothesis] = \
        Field(default_factory=list)


def build_prompt(context: dict) -> str:
    return PROMPT_TEMPLATE.format(
        prompt_version=PROMPT_VERSION,
        context_json=json.dumps(context, sort_keys=True, ensure_ascii=False,
                                default=str))


def parse(raw: str) -> BehaviorAuditorOutput:
    return BehaviorAuditorOutput.model_validate_json(raw)


def mock_rule(context: dict) -> dict:
    """One finding per deterministic anomaly, bound to covering clips.

    CC3 fix2 (§13): additionally emits ONE deterministic out-of-taxonomy
    PROVISIONAL hypothesis when the symbolic clip evidence shows a pattern
    the fixed detector taxonomy does not cover (repeated no-effect action
    semantics with flat progress bands). It is bound to evidence clip ids,
    carries an alternative explanation, and is contractually forbidden from
    entering the selector or batch plan.
    """
    clips = context.get("clips", [])
    clip_by_anomaly: dict = {}
    for c in clips:
        for aid in c.get("reason_anomaly_ids", []):
            clip_by_anomaly.setdefault(aid, []).append(c["clip_id"])
    findings = []
    for a in context.get("anomalies", []):
        sev = float(a["severity"])
        rec = int(a["recurrence"])
        confidence = round(max(0.0, min(1.0, 0.5 + 0.1 * rec + 0.3 * sev)), 4)
        findings.append(dict(
            finding_id=f"finding:{a['anomaly_id']}",
            behavior_pattern=a["behavior_pattern"],
            severity=sev,
            recurrence=rec,
            evidence_span_ids=sorted(set(clip_by_anomaly.get(a["anomaly_id"], []))),
            supporting_fields=[
                f"{e['event_type']}@step{e['step_index']}:{json.dumps(e['fields'], sort_keys=True)}"
                for e in a.get("supporting_events", [])],
            counter_evidence=list(a.get("counter_evidence", [])),
            confidence=confidence,
        ))
    findings.sort(key=lambda f: f["finding_id"])

    provisionals = []
    no_effect_aids = {a["anomaly_id"] for a in context.get("anomalies", [])
                      if a.get("behavior_pattern") == "repeated_no_effect"}
    no_effect_clip_ids = sorted({
        c["clip_id"] for c in clips
        if no_effect_aids & set(c.get("reason_anomaly_ids", []))})
    if no_effect_clip_ids:
        provisionals.append(dict(
            hypothesis_id="provisional:out_of_taxonomy:stall_without_"
                          "negative_event",
            provisional=True,
            taxonomy_status="out_of_taxonomy",
            evidence_clip_ids=no_effect_clip_ids,
            observed_pattern="repeated action semantics with flat progress "
                             "delta bands and NO negative env event — a "
                             "stall pattern the detector taxonomy only "
                             "covers as repeated_no_effect",
            confidence=0.35,
            alternative_explanation="incidental exploration noise or an "
                                    "adapter discretization artifact; must "
                                    "be ruled out by deterministic "
                                    "validation before any downstream use",
            requires_deterministic_validation=True,
            selector_or_batch_entry_forbidden=True))
    return dict(behavior_findings=findings,
                provisional_anomaly_hypotheses=provisionals)


def run(context: dict, backend, sequence: int) -> RoleEnvelope:
    prompt = build_prompt(context)
    raw = backend.complete(ROLE, prompt)
    parsed = parse(raw)
    return RoleEnvelope.make(
        role=ROLE, prompt_version=PROMPT_VERSION, backend_id=backend.backend_id,
        model_id=backend.model_id, sequence=sequence, prompt=prompt,
        raw_response=raw, parsed_dump=parsed.model_dump())
