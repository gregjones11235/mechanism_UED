"""Board role 2/6: BehaviorAuditor (six-role Review Board, C6).

Audits the window's behavior-failure evidence record by record: which probes
showed a Student-vs-Reference gap, how severe, in which families. Strictly
DESCRIPTIVE — the audit neither explains causes (CausalFailureAnalyst) nor
prescribes environments (InterventionTutor / Explorer).

ENGINEERING_SCAFFOLD: deterministic mock rule; no real LLM call this round.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from pydantic import Field, model_validator

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.behavior_failure import SEVERITIES
from d052.feedback_llm_ued.feedback_contracts import (
    CONTEXT_CLOSE,
    CONTEXT_OPEN,
    FeedbackRoleEnvelope,
)
from d052.schemas.common import CanonicalModel

ROLE = C.ROLE_BEHAVIOR_AUDITOR
PROMPT_VERSION = f"{C.ROLE_PROMPT_VERSION}.{ROLE}"

PROMPT_TEMPLATE = f"""\
You are the BehaviorAuditor role of the six-role Review Board of the
simulator-grounded feedback-adaptive LLM-UED loop. Audit each behavior-failure
evidence item: confirm its Student-vs-Reference gap and severity, count the
severity classes, and name the worst family. Descriptive audit only — no
causal claims, no environment advice.
Prompt version: {{prompt_version}}
{CONTEXT_OPEN}
{{context_json}}
{CONTEXT_CLOSE}
Respond with a single JSON object matching the BehaviorAuditOutput schema.
"""


class AuditFinding(CanonicalModel):
    feedback_id: str = Field(min_length=1)
    environment_family: str = Field(min_length=1)
    severity: str
    reference_gap: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _severity_legal(self) -> "AuditFinding":
        if self.severity not in SEVERITIES:
            raise ValueError(f"ILLEGAL_SEVERITY: {self.severity!r}")
        return self


class BehaviorAuditOutput(CanonicalModel):
    window: int = Field(ge=0)
    findings: List[AuditFinding] = Field(default_factory=list)
    severity_counts: Dict[str, int] = Field(default_factory=dict)
    worst_family: str = ""
    audit_summary: str = ""


def build_prompt(context: dict) -> str:
    return PROMPT_TEMPLATE.format(
        prompt_version=PROMPT_VERSION,
        context_json=json.dumps(context, sort_keys=True, ensure_ascii=False,
                                default=str))


def parse(raw: str) -> BehaviorAuditOutput:
    return BehaviorAuditOutput.model_validate_json(raw)


#: output class exposed for the board assembler (single source of truth)
OUTPUT_MODEL = BehaviorAuditOutput


def mock_rule(context: dict) -> dict:
    """Deterministically re-state the evidence as audited findings."""
    window = int(context.get("window", 0))
    bc = context.get("board_context", {})
    evidence = bc.get("behavior_evidence", [])

    findings: List[dict] = []
    counts: Dict[str, int] = {}
    fam_gap: Dict[str, float] = {}
    for e in sorted(evidence, key=lambda x: x.get("feedback_id", "")):
        sev = e.get("severity", "none")
        gap = float(e.get("reference_gap", 0.0))
        fam = e.get("environment_family", "")
        findings.append(dict(feedback_id=e.get("feedback_id", ""),
                             environment_family=fam,
                             severity=sev,
                             reference_gap=gap))
        counts[sev] = counts.get(sev, 0) + 1
        fam_gap[fam] = max(fam_gap.get(fam, 0.0), gap)
    worst = ""
    if fam_gap:
        worst = sorted(fam_gap.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    summary = (f"window {window}: audited {len(findings)} evidence item(s); "
               + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
               if counts else f"window {window}: no behavior evidence to audit")
    return dict(window=window, findings=findings, severity_counts=counts,
                worst_family=worst, audit_summary=summary)


def run(context: dict, backend, window: int, sequence: int,
        context_binding: Optional[dict] = None) -> FeedbackRoleEnvelope:
    prompt = build_prompt(context)
    raw = backend.complete(ROLE, prompt)
    parsed = parse(raw)
    return FeedbackRoleEnvelope.make(
        role=ROLE, prompt_version=PROMPT_VERSION, backend_id=backend.backend_id,
        model_id=backend.model_id, window=window, sequence=sequence,
        prompt=prompt, raw_response=raw, parsed_dump=parsed.model_dump(),
        context_binding=context_binding)
