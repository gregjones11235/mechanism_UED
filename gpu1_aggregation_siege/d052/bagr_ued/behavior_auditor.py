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

from pydantic import Field

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


class BehaviorAuditorOutput(CanonicalModel):
    behavior_findings: List[BehaviorFinding] = Field(default_factory=list)


def build_prompt(context: dict) -> str:
    return PROMPT_TEMPLATE.format(
        prompt_version=PROMPT_VERSION,
        context_json=json.dumps(context, sort_keys=True, ensure_ascii=False,
                                default=str))


def parse(raw: str) -> BehaviorAuditorOutput:
    return BehaviorAuditorOutput.model_validate_json(raw)


def mock_rule(context: dict) -> dict:
    """One finding per deterministic anomaly, bound to covering clips."""
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
    return dict(behavior_findings=findings)


def run(context: dict, backend, sequence: int) -> RoleEnvelope:
    prompt = build_prompt(context)
    raw = backend.complete(ROLE, prompt)
    parsed = parse(raw)
    return RoleEnvelope.make(
        role=ROLE, prompt_version=PROMPT_VERSION, backend_id=backend.backend_id,
        model_id=backend.model_id, sequence=sequence, prompt=prompt,
        raw_response=raw, parsed_dump=parsed.model_dump())
