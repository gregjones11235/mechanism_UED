"""Role: StudentModeler (task section 1).

Reads the evidence bundle + extracted anomalies and produces a descriptive
capability snapshot of the CURRENT Student: recurring difficulties and
readiness for counterfactual testing. DESCRIPTIVE ONLY — no causes, no
proposals, no advice (the schema structurally cannot carry them).
"""
from __future__ import annotations

import json

from pydantic import Field

from d052.bagr_ued import constants as C
from d052.bagr_ued.review_contracts import CONTEXT_CLOSE, CONTEXT_OPEN, RoleEnvelope
from d052.schemas.common import CanonicalModel

ROLE = C.ROLE_STUDENT_MODELER
PROMPT_VERSION = f"{C.ROLE_PROMPT_VERSION}.{ROLE}"

PROMPT_TEMPLATE = f"""\
You are the StudentModeler role of the BA-BAGR-UED review board.
Describe the CURRENT Student's recurring difficulties from deterministic
trajectory evidence. Be descriptive: no failure causes, no environment
proposals, no action advice.
Prompt version: {{prompt_version}}
{CONTEXT_OPEN}
{{context_json}}
{CONTEXT_CLOSE}
Respond with a single JSON object matching the StudentModelSnapshot schema.
"""


class StudentModelSnapshot(CanonicalModel):
    student_capability_summary: str = Field(min_length=1)
    recurring_difficulties: list = Field(default_factory=list)
    evidence_clip_ids: list = Field(default_factory=list)
    readiness_for_counterfactual_tests: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)


def build_prompt(context: dict) -> str:
    return PROMPT_TEMPLATE.format(
        prompt_version=PROMPT_VERSION,
        context_json=json.dumps(context, sort_keys=True, ensure_ascii=False,
                                default=str))


def parse(raw: str) -> StudentModelSnapshot:
    return StudentModelSnapshot.model_validate_json(raw)


def mock_rule(context: dict) -> dict:
    """Deterministic derivation of the snapshot from evidence context."""
    anomalies = context.get("anomalies", [])
    patterns = sorted({a["behavior_pattern"] for a in anomalies})
    episodes = sorted({a["episode_id"] for a in anomalies})
    by_pattern = {}
    for a in anomalies:
        e = by_pattern.setdefault(a["behavior_pattern"], dict(sev=0.0, rec=0))
        e["sev"] = max(e["sev"], float(a["severity"]))
        e["rec"] = max(e["rec"], int(a["recurrence"]))
    difficulties = [f"{p}: severity up to {e['sev']:.2f}, recurrence up to "
                    f"{e['rec']} (global scope, not Tier3-only)"
                    for p, e in sorted(by_pattern.items())]
    summary = (f"Current Student shows {len(patterns)} recurring behavior "
               f"pattern group(s) across {len(episodes)} episode(s) of "
               f"generative-training evidence; patterns: "
               f"{', '.join(patterns) if patterns else 'none detected'}.")
    readiness = round(min(0.9, 0.5 + 0.05 * len(anomalies)), 4)
    confidence = round(min(1.0, 0.4 + 0.1 * len(anomalies)), 4)
    return dict(
        student_capability_summary=summary,
        recurring_difficulties=difficulties,
        evidence_clip_ids=sorted({c["clip_id"] for c in context.get("clips", [])}),
        readiness_for_counterfactual_tests=readiness,
        confidence=confidence,
    )


def run(context: dict, backend, sequence: int) -> RoleEnvelope:
    prompt = build_prompt(context)
    raw = backend.complete(ROLE, prompt)
    parsed = parse(raw)
    return RoleEnvelope.make(
        role=ROLE, prompt_version=PROMPT_VERSION, backend_id=backend.backend_id,
        model_id=backend.model_id, sequence=sequence, prompt=prompt,
        raw_response=raw, parsed_dump=parsed.model_dump())
