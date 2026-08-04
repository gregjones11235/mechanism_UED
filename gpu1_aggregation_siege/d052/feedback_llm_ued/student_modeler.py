"""Board role 1/6: StudentModeler (six-role Review Board, C6).

DESCRIPTIVE model of the Student's current capability, derived from the
pooled behavior evidence (episode-level success rate + CI) — no environment
knobs, no advice. The model the board reasons about in every downstream role
starts here.

ENGINEERING_SCAFFOLD: the mock rule is a deterministic function of the board
context; no real LLM call happens this round.
"""
from __future__ import annotations

import json
from typing import List, Optional

from pydantic import Field, model_validator

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.feedback_contracts import (
    CONTEXT_CLOSE,
    CONTEXT_OPEN,
    FeedbackRoleEnvelope,
)
from d052.schemas.common import CanonicalModel

ROLE = C.ROLE_STUDENT_MODELER
PROMPT_VERSION = f"{C.ROLE_PROMPT_VERSION}.{ROLE}"

PROMPT_TEMPLATE = f"""\
You are the StudentModeler role of the six-role Review Board of the
simulator-grounded feedback-adaptive LLM-UED loop. From the pooled Student
behavior evidence (episode-level success rate with confidence interval) build
a DESCRIPTIVE model of the Student's current capability: estimated success
rate, the weakest environment family, and the strongest failure signals.
No environment knobs, no advice — description only.
Prompt version: {{prompt_version}}
{CONTEXT_OPEN}
{{context_json}}
{CONTEXT_CLOSE}
Respond with a single JSON object matching the StudentModelOutput schema.
"""


class StudentModelOutput(CanonicalModel):
    window: int = Field(ge=0)
    modeled_success_rate: float = Field(ge=0.0, le=1.0)
    modeled_success_rate_ci: float = Field(ge=0.0)
    weakest_family: str = ""
    weakness_signals: List[str] = Field(default_factory=list)
    summary: str = ""


def build_prompt(context: dict) -> str:
    return PROMPT_TEMPLATE.format(
        prompt_version=PROMPT_VERSION,
        context_json=json.dumps(context, sort_keys=True, ensure_ascii=False,
                                default=str))


def parse(raw: str) -> StudentModelOutput:
    return StudentModelOutput.model_validate_json(raw)


#: output class exposed for the board assembler (single source of truth)
OUTPUT_MODEL = StudentModelOutput


def mock_rule(context: dict) -> dict:
    """Deterministically summarize the pooled behavior evidence."""
    window = int(context.get("window", 0))
    bc = context.get("board_context", {})
    evidence = bc.get("behavior_evidence", [])
    pooled_sr = float(bc.get("pooled_student_success_rate", 0.0))
    ci = float(bc.get("student_success_rate_ci", 1.0))

    weakest = ""
    signals: List[str] = []
    if evidence:
        # worst signals first (deterministic tie-break on feedback_id)
        ranked = sorted(evidence,
                        key=lambda e: (-float(e.get("reference_gap", 0.0)),
                                       e.get("feedback_id", "")))
        worst = ranked[0]
        weakest = worst.get("environment_family", "")
        for e in ranked[:C.MAX_WEAKNESSES]:
            signals.append(f"{e.get('feedback_id', '')}:"
                           f"{e.get('severity', '')}")
    summary = (f"window {window}: pooled Student success rate "
               f"{pooled_sr:.3f} +/- {ci:.3f} over "
               f"{int(bc.get('pooled_episodes', 0))} episode(s); "
               f"weakest family: {weakest or 'n/a'}")
    return dict(window=window,
                modeled_success_rate=pooled_sr,
                modeled_success_rate_ci=ci,
                weakest_family=weakest,
                weakness_signals=signals,
                summary=summary)


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
