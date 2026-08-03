"""Role: FeedbackDiagnostician (task §3 role 1 of the default 2 calls).

This is the step that makes the loop FEEDBACK-DRIVEN rather than
generate-then-accept/reject. The Diagnostician reads the HypothesisLedger
together with the SimulatorFeedbackStore records produced by REAL candidate
probes this window (their expected-vs-observed match direction) and turns
them into per-hypothesis verdicts: SUPPORTED / REFUTED / INCONCLUSIVE / STALE,
with an updated confidence. The verdict is DERIVED from the feedback content,
so shuffling which candidate a feedback record is bound to (the
``shuffled_feedback`` mode) changes the verdicts — which is exactly what §5's
normal-vs-shuffled comparison measures.

DESCRIPTIVE ONLY: no environment knobs, no action/reward/policy advice. The
schema structurally cannot carry them; environment design is the
AdaptiveEnvironmentDesigner's job.
"""
from __future__ import annotations

import json
from typing import Dict, List

from pydantic import Field, model_validator

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.feedback_contracts import (
    CONTEXT_CLOSE,
    CONTEXT_OPEN,
    FeedbackRoleEnvelope,
)
from d052.schemas.common import CanonicalModel

ROLE = C.ROLE_FEEDBACK_DIAGNOSTICIAN
PROMPT_VERSION = f"{C.ROLE_PROMPT_VERSION}.{ROLE}"

PROMPT_TEMPLATE = f"""\
You are the FeedbackDiagnostician role of the simulator-grounded
feedback-adaptive LLM-UED loop. Read the current hypothesis ledger and the
simulator probe feedback for this window. For EACH hypothesis, compare its
predicted metric signature against the observed probe outcomes and emit a
verdict (SUPPORTED / REFUTED / INCONCLUSIVE / STALE) plus an updated
confidence. Be descriptive: no environment knobs, no action advice.
Prompt version: {{prompt_version}}
{CONTEXT_OPEN}
{{context_json}}
{CONTEXT_CLOSE}
Respond with a single JSON object matching the DiagnosisOutput schema.
"""


class HypothesisVerdict(CanonicalModel):
    hypothesis_id: str = Field(min_length=1)
    verdict: str = Field(min_length=1)
    new_confidence: float = Field(ge=0.0, le=1.0)
    agree_count: int = Field(ge=0)
    opposite_count: int = Field(ge=0)
    neutral_count: int = Field(ge=0)
    feedback_ids: List[str] = Field(default_factory=list)
    reason: str = Field(default="")

    @model_validator(mode="after")
    def _verdict_legal(self) -> "HypothesisVerdict":
        if self.verdict not in C.HYPOTHESIS_STATUSES:
            raise ValueError(f"ILLEGAL_VERDICT: {self.verdict!r}")
        return self


class DiagnosisOutput(CanonicalModel):
    window: int = Field(ge=0)
    hypothesis_verdicts: List[HypothesisVerdict] = Field(default_factory=list)
    weakness_summary: str = Field(default="")
    overall_confidence: float = Field(ge=0.0, le=1.0)
    global_risk: str = "LOW"

    @model_validator(mode="after")
    def _risk_legal(self) -> "DiagnosisOutput":
        if self.global_risk not in ("LOW", "MEDIUM", "HIGH"):
            raise ValueError(f"ILLEGAL_GLOBAL_RISK: {self.global_risk!r}")
        return self


def build_prompt(context: dict) -> str:
    return PROMPT_TEMPLATE.format(
        prompt_version=PROMPT_VERSION,
        context_json=json.dumps(context, sort_keys=True, ensure_ascii=False,
                                default=str))


def parse(raw: str) -> DiagnosisOutput:
    return DiagnosisOutput.model_validate_json(raw)


def _verdict_for(agree: int, opposite: int, has_any: bool) -> str:
    if not has_any:
        return C.HYPOTHESIS_STALE            # nothing probed it this window
    if agree > opposite:
        return C.HYPOTHESIS_SUPPORTED
    if opposite > agree:
        return C.HYPOTHESIS_REFUTED
    return C.HYPOTHESIS_INCONCLUSIVE


def mock_rule(context: dict) -> dict:
    """Deterministically derive per-hypothesis verdicts from probe feedback.

    The derivation is purely a function of (a) which feedback records list the
    hypothesis in ``distinguishes_hypothesis_ids`` and (b) each record's
    ``expected_observed_match``. Therefore re-binding feedback to different
    candidates (shuffled mode) changes the verdicts.
    """
    hypotheses = context.get("hypotheses", [])
    feedback = context.get("feedback", [])
    max_diagnose = C.MAX_DIAGNOSED_HYPOTHESES_PER_WINDOW

    # index feedback by the hypotheses they distinguish
    by_hyp: Dict[str, List[dict]] = {}
    for fb in feedback:
        for hid in fb.get("distinguishes_hypothesis_ids", []):
            by_hyp.setdefault(hid, []).append(fb)

    verdicts: List[dict] = []
    confidences: List[float] = []
    for hyp in sorted(hypotheses, key=lambda h: h["hypothesis_id"])[:max_diagnose]:
        hid = hyp["hypothesis_id"]
        recs = by_hyp.get(hid, [])
        agree = sum(1 for r in recs
                    if r.get("expected_observed_match") == C.MATCH_DIRECTION_AGREE)
        opposite = sum(1 for r in recs
                       if r.get("expected_observed_match") == C.MATCH_DIRECTION_OPPOSITE)
        neutral = len(recs) - agree - opposite
        base = float(hyp.get("confidence", 0.5))
        verdict = _verdict_for(agree, opposite, has_any=bool(recs))
        # confidence drifts with evidence; bounded away from 0/1
        delta = 0.10 * agree - 0.15 * opposite
        new_conf = min(0.95, max(0.05, round(base + delta, 4)))
        if verdict == C.HYPOTHESIS_STALE:
            new_conf = round(base, 4)          # no evidence -> no update
        reason = (f"{agree} agree / {opposite} opposite / {neutral} neutral "
                  f"probe feedback record(s) distinguish this hypothesis")
        verdicts.append(dict(
            hypothesis_id=hid, verdict=verdict, new_confidence=new_conf,
            agree_count=agree, opposite_count=opposite, neutral_count=neutral,
            feedback_ids=sorted({r["feedback_id"] for r in recs}),
            reason=reason))
        confidences.append(new_conf)

    overall = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
    refuted = [v for v in verdicts if v["verdict"] == C.HYPOTHESIS_REFUTED]
    weakness = (f"{len(refuted)} hypothesis(es) refuted by probe feedback: "
                + ", ".join(v["hypothesis_id"] for v in refuted)
                if refuted else "no hypothesis refuted this window")
    # risk escalates with refutations + low confidence (feeds the reviewer gate)
    if refuted and overall < C.REVIEWER_CONFIDENCE_FLOOR:
        risk = "HIGH"
    elif refuted or overall < C.REVIEWER_CONFIDENCE_FLOOR:
        risk = "MEDIUM"
    else:
        risk = "LOW"
    return dict(
        window=int(context.get("window", 0)),
        hypothesis_verdicts=verdicts,
        weakness_summary=weakness,
        overall_confidence=overall,
        global_risk=risk)


def run(context: dict, backend, window: int, sequence: int) -> FeedbackRoleEnvelope:
    prompt = build_prompt(context)
    raw = backend.complete(ROLE, prompt)
    parsed = parse(raw)
    return FeedbackRoleEnvelope.make(
        role=ROLE, prompt_version=PROMPT_VERSION, backend_id=backend.backend_id,
        model_id=backend.model_id, window=window, sequence=sequence,
        prompt=prompt, raw_response=raw, parsed_dump=parsed.model_dump())
