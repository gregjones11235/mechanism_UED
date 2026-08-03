"""Board role 3/6: CausalFailureAnalyst (six-role Review Board, C6).

Turns the audited behavior failures into causal claims: per-hypothesis
VERDICTS on the frozen (<= k-1) probe feedback — each verdict explicitly
cites the feedback_ids and the prediction_signature it grades — plus NEW
hypotheses (PENDING with predicted signatures) for families the ledger does
not yet cover.

The citation discipline lives in the data contract here: a verdict carries
``cited_feedback_ids`` + ``cited_prediction_signature``; review_board.py
enforces that every cited feedback id is visible in the window's
FeedbackView and strictly EARLIER than the board's own window
(double-window state machine).

ENGINEERING_SCAFFOLD: deterministic mock rule; no real LLM call this round.
"""
from __future__ import annotations

import json
import math
from typing import Dict, List

from pydantic import Field, model_validator

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.feedback_contracts import (
    CONTEXT_CLOSE,
    CONTEXT_OPEN,
    FeedbackRoleEnvelope,
)
from d052.schemas.common import CanonicalModel

ROLE = C.ROLE_CAUSAL_FAILURE_ANALYST
PROMPT_VERSION = f"{C.ROLE_PROMPT_VERSION}.{ROLE}"

PROMPT_TEMPLATE = f"""\
You are the CausalFailureAnalyst role of the six-role Review Board of the
simulator-grounded feedback-adaptive LLM-UED loop. From the visible frozen
probe feedback (window <= k-1) and the hypothesis ledger: (1) emit a verdict
(SUPPORTED / REFUTED / INCONCLUSIVE / STALE) for each existing hypothesis,
explicitly citing the feedback ids and the predicted signature the verdict
grades; (2) propose NEW hypotheses (PENDING, with predicted signatures) for
families the ledger does not cover yet. Cite only feedback you can see.
Prompt version: {{prompt_version}}
{CONTEXT_OPEN}
{{context_json}}
{CONTEXT_CLOSE}
Respond with a single JSON object matching the CausalAnalysisOutput schema.
"""


class BoardHypothesisVerdict(CanonicalModel):
    """One hypothesis verdict with explicit feedback citations."""

    hypothesis_id: str = Field(min_length=1)
    verdict: str = Field(min_length=1)
    new_confidence: float = Field(ge=0.0, le=1.0)
    cited_feedback_ids: List[str] = Field(default_factory=list)
    #: the predicted signature this verdict grades (copied from the
    #: hypothesis so the citation is self-contained and auditable)
    cited_prediction_signature: Dict[str, float] = Field(default_factory=dict)
    agree_count: int = Field(default=0, ge=0)
    opposite_count: int = Field(default=0, ge=0)
    neutral_count: int = Field(default=0, ge=0)
    reason: str = ""

    @model_validator(mode="after")
    def _verdict_legal(self) -> "BoardHypothesisVerdict":
        if self.verdict not in C.HYPOTHESIS_STATUSES:
            raise ValueError(f"ILLEGAL_VERDICT: {self.verdict!r}")
        for key, value in self.cited_prediction_signature.items():
            if not isinstance(value, (int, float)) or \
                    not math.isfinite(float(value)):
                raise ValueError(
                    f"NON_FINITE_PREDICTION_CITATION: {key}={value!r}")
        return self


class NewHypothesisProposal(CanonicalModel):
    """A new PENDING hypothesis with its predicted probe signature."""

    hypothesis_id: str = Field(min_length=1)
    target_behavior: str = Field(min_length=1)
    environment_family: str = Field(min_length=1)
    predicted_signature: Dict[str, float] = Field(default_factory=dict)
    initial_confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate(self) -> "NewHypothesisProposal":
        if self.environment_family not in C.ENVIRONMENT_FAMILIES:
            raise ValueError(
                f"UNKNOWN_ENVIRONMENT_FAMILY: {self.environment_family!r}")
        if not self.predicted_signature:
            raise ValueError(
                "EMPTY_PREDICTED_SIGNATURE: a new hypothesis must predict "
                "the probe signature that would test it")
        for key, value in self.predicted_signature.items():
            if not isinstance(value, (int, float)) or \
                    not math.isfinite(float(value)):
                raise ValueError(f"NON_FINITE_EXPECTATION: {key}={value!r}")
        return self


class CausalAnalysisOutput(CanonicalModel):
    window: int = Field(ge=0)
    hypothesis_verdicts: List[BoardHypothesisVerdict] = Field(
        default_factory=list)
    new_hypotheses: List[NewHypothesisProposal] = Field(default_factory=list)
    analysis_summary: str = ""


def build_prompt(context: dict) -> str:
    return PROMPT_TEMPLATE.format(
        prompt_version=PROMPT_VERSION,
        context_json=json.dumps(context, sort_keys=True, ensure_ascii=False,
                                default=str))


def parse(raw: str) -> CausalAnalysisOutput:
    return CausalAnalysisOutput.model_validate_json(raw)


#: output class exposed for the board assembler (single source of truth)
OUTPUT_MODEL = CausalAnalysisOutput


def _verdict_for(agree: int, opposite: int, has_any: bool) -> str:
    if not has_any:
        return C.HYPOTHESIS_STALE            # nothing visible probed it
    if agree > opposite:
        return C.HYPOTHESIS_SUPPORTED
    if opposite > agree:
        return C.HYPOTHESIS_REFUTED
    return C.HYPOTHESIS_INCONCLUSIVE


def mock_rule(context: dict) -> dict:
    """Deterministically grade visible feedback against ledger hypotheses.

    The derivation is purely a function of (a) which VISIBLE feedback records
    list the hypothesis in ``distinguishes_hypothesis_ids`` and (b) each
    record's ``expected_observed_match`` — so re-binding feedback (shuffled
    mode) or hiding it (static mode) changes the verdicts.
    """
    window = int(context.get("window", 0))
    hypotheses = context.get("hypotheses", [])
    feedback = context.get("feedback", [])

    by_hyp: Dict[str, List[dict]] = {}
    for fb in feedback:
        for hid in fb.get("distinguishes_hypothesis_ids", []):
            by_hyp.setdefault(hid, []).append(fb)

    verdicts: List[dict] = []
    covered_families: set = set()
    for hyp in sorted(hypotheses, key=lambda h: h["hypothesis_id"]):
        hid = hyp["hypothesis_id"]
        covered_families.add(hyp.get("environment_family", ""))
        recs = by_hyp.get(hid, [])
        agree = sum(1 for r in recs
                    if r.get("expected_observed_match")
                    == C.MATCH_DIRECTION_AGREE)
        opposite = sum(1 for r in recs
                       if r.get("expected_observed_match")
                       == C.MATCH_DIRECTION_OPPOSITE)
        neutral = len(recs) - agree - opposite
        base = float(hyp.get("confidence", 0.5))
        verdict = _verdict_for(agree, opposite, has_any=bool(recs))
        delta = 0.10 * agree - 0.15 * opposite
        new_conf = min(0.95, max(0.05, round(base + delta, 4)))
        if verdict == C.HYPOTHESIS_STALE:
            new_conf = round(base, 4)          # no evidence -> no update
        verdicts.append(dict(
            hypothesis_id=hid, verdict=verdict, new_confidence=new_conf,
            cited_feedback_ids=sorted({r["feedback_id"] for r in recs}),
            cited_prediction_signature=dict(
                hyp.get("predicted_signature", {})),
            agree_count=agree, opposite_count=opposite,
            neutral_count=neutral,
            reason=(f"{agree} agree / {opposite} opposite / {neutral} "
                    f"neutral visible probe record(s) distinguish this "
                    f"hypothesis")))

    # new hypotheses for families with visible feedback but no ledger entry
    new_hyps: List[dict] = []
    fam_records: Dict[str, List[dict]] = {}
    for fb in feedback:
        fam_records.setdefault(fb.get("environment_family", ""),
                               []).append(fb)
    for fam in sorted(fam_records):
        if fam in covered_families or fam not in C.ENVIRONMENT_FAMILIES:
            continue
        if len(new_hyps) >= C.MAX_EXPLORATION_PROPOSALS:
            break
        recs = fam_records[fam]
        mean_sr = sum(float(r.get("student_success_rate", 0.0))
                      for r in recs) / len(recs)
        new_hyps.append(dict(
            hypothesis_id=f"hyp-w{window:02d}-{fam}-new",
            target_behavior=(f"Student success behavior in {fam} "
                             f"(observed pooled success rate "
                             f"{mean_sr:.3f})"),
            environment_family=fam,
            predicted_signature={
                "student_success_rate": round(min(0.95, max(0.05, mean_sr)),
                                              4)},
            initial_confidence=0.5))

    summary = (f"window {window}: {len(verdicts)} verdict(s) on visible "
               f"feedback, {len(new_hyps)} new hypothesis proposal(s)")
    return dict(window=window, hypothesis_verdicts=verdicts,
                new_hypotheses=new_hyps, analysis_summary=summary)


def run(context: dict, backend, window: int, sequence: int
        ) -> FeedbackRoleEnvelope:
    prompt = build_prompt(context)
    raw = backend.complete(ROLE, prompt)
    parsed = parse(raw)
    return FeedbackRoleEnvelope.make(
        role=ROLE, prompt_version=PROMPT_VERSION, backend_id=backend.backend_id,
        model_id=backend.model_id, window=window, sequence=sequence,
        prompt=prompt, raw_response=raw, parsed_dump=parsed.model_dump())
