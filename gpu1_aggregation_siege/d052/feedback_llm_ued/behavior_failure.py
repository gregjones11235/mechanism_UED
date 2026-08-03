"""Student behavior-failure evidence (board input layer, C4).

Turns graded probe feedback records into the coarse failure signals the
six-role Review Board reads: return shortfall vs the Reference, behavior
activation gap, front-progress gap, and early-stop evidence. Everything is
EPISODE-LEVEL (no action-guidance carriers) and every extraction is
deterministic.

Double-window rule: evidence extracted from window-k probes is consumed by
the board at window k+1 — this module only extracts and structures; it never
revises anything.

Honesty: ``ProbeMetrics`` carries no episode lengths, so early-stop evidence
is explicitly ``early_stop_measured=False`` on the record path; the real
seam (which does measure episode lengths) attaches a measured rate through
``with_early_stop``.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import Field, model_validator

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.simulator_feedback_store import (
    MATCH_STATES,
    SimulatorFeedbackRecord,
)
from d052.feedback_llm_ued.uncertainty import (
    Z_95,
    ci_halfwidth,
    episodes_from_transitions,
)
from d052.schemas.common import CanonicalModel

SEVERITY_NONE = "none"
SEVERITY_LOW = "low"
SEVERITY_MEDIUM = "medium"
SEVERITY_HIGH = "high"
SEVERITIES = frozenset({SEVERITY_NONE, SEVERITY_LOW, SEVERITY_MEDIUM,
                        SEVERITY_HIGH})

#: reference-gap thresholds for the severity ladder (documented round choice)
REFERENCE_GAP_LOW = 0.05
REFERENCE_GAP_MEDIUM = 0.15
REFERENCE_GAP_HIGH = 0.30


def severity_for(reference_gap: float) -> str:
    if reference_gap < REFERENCE_GAP_LOW:
        return SEVERITY_NONE
    if reference_gap < REFERENCE_GAP_MEDIUM:
        return SEVERITY_LOW
    if reference_gap < REFERENCE_GAP_HIGH:
        return SEVERITY_MEDIUM
    return SEVERITY_HIGH


class BehaviorFailureEvidence(CanonicalModel):
    """Coarse Student-failure signals for ONE probed candidate."""

    feedback_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    window: int = Field(ge=0)
    environment_family: str = Field(min_length=1)
    student_success_rate: float = Field(ge=0.0, le=1.0)
    reference_success_rate: float = Field(ge=0.0, le=1.0)
    return_shortfall: float = Field(ge=0.0)
    behavior_activation_gap: float = Field(ge=0.0)
    front_progress_gap: float = Field(ge=0.0)
    #: worst of the three gaps — the single number the severity ladder uses
    reference_gap: float = Field(ge=0.0)
    severity: str
    #: fraction of Student episodes that terminated early; measured only when
    #: episode lengths exist (real seam). ProbeMetrics path: measured=False.
    early_stop_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    early_stop_measured: bool = False
    expected_observed_match: str

    @model_validator(mode="after")
    def _validate(self) -> "BehaviorFailureEvidence":
        if self.severity not in SEVERITIES:
            raise ValueError(f"ILLEGAL_SEVERITY: {self.severity!r}")
        if self.expected_observed_match not in MATCH_STATES:
            raise ValueError(
                f"ILLEGAL_MATCH_STATE: {self.expected_observed_match!r}")
        return self

    @classmethod
    def from_record(cls,
                    record: SimulatorFeedbackRecord
                    ) -> "BehaviorFailureEvidence":
        metrics = record.stage2_metrics or record.stage1_metrics
        if metrics is None:
            raise ValueError(
                f"NO_PROBE_METRICS: feedback {record.feedback_id!r} carries "
                "no stage metrics; it cannot become behavior evidence")
        shortfall = round(max(0.0, metrics.reference_success_rate
                              - metrics.student_success_rate), 6)
        activation_gap = round(max(0.0, metrics.reference_behavior_activation
                                   - metrics.student_behavior_activation), 6)
        progress_gap = round(max(0.0, metrics.reference_mean_progress
                                 - metrics.student_front_progress), 6)
        gap = round(max(shortfall, activation_gap, progress_gap), 6)
        return cls(feedback_id=record.feedback_id,
                   candidate_id=record.candidate_id,
                   window=record.window,
                   environment_family=record.environment_family,
                   student_success_rate=metrics.student_success_rate,
                   reference_success_rate=metrics.reference_success_rate,
                   return_shortfall=shortfall,
                   behavior_activation_gap=activation_gap,
                   front_progress_gap=progress_gap,
                   reference_gap=gap,
                   severity=severity_for(gap),
                   expected_observed_match=record.expected_observed_match)

    def with_early_stop(self, early_stop_rate: float
                        ) -> "BehaviorFailureEvidence":
        """Return a copy carrying a MEASURED early-stop rate (real seam)."""
        if not isinstance(early_stop_rate, (int, float)) or \
                not 0.0 <= float(early_stop_rate) <= 1.0:
            raise ValueError(f"ILLEGAL_EARLY_STOP_RATE: {early_stop_rate!r}")
        payload = self.model_dump()
        payload["early_stop_rate"] = round(float(early_stop_rate), 6)
        payload["early_stop_measured"] = True
        return BehaviorFailureEvidence(**payload)


def extract_window_evidence(store, window: int) -> List[BehaviorFailureEvidence]:
    """Deterministic evidence list for one window's graded feedback records."""
    records = sorted(store.for_window(window), key=lambda r: r.feedback_id)
    return [BehaviorFailureEvidence.from_record(r) for r in records]


#: label for a BoardContext whose feedback view has not been bound yet (the
#: view abstraction arrives with the six-role board; the label is enforced
#: then, never silently).
FEEDBACK_VIEW_UNBOUND = "UNBOUND"


class BoardContext(CanonicalModel):
    """Everything the six-role board reads about window k's PROBE EVIDENCE
    (assembled at window k+1 — the double-window state machine's phase A).
    """

    window: int = Field(ge=0)
    mode: str
    behavior_evidence: List[BehaviorFailureEvidence] = Field(
        default_factory=list)
    pooled_episodes: int = Field(default=0, ge=0)
    pooled_student_success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    #: CI half-width on the pooled Student success rate; 1.0 (maximal
    #: uncertainty) when no episodes were pooled — window 0's board reads an
    #: empty k-1 evidence set, which is legal.
    student_success_rate_ci: float = Field(default=1.0, ge=0.0)
    feedback_view_label: str = FEEDBACK_VIEW_UNBOUND

    @model_validator(mode="after")
    def _validate(self) -> "BoardContext":
        if self.mode not in C.FEEDBACK_MODES:
            raise ValueError(f"UNKNOWN_MODE: {self.mode!r}")
        return self


def assemble_board_context(store, *, window: int, mode: str,
                           feedback_view_label: str = FEEDBACK_VIEW_UNBOUND,
                           z: float = Z_95) -> BoardContext:
    """Phase-A assembly: window-k evidence + pooled-episode uncertainty."""
    evidence = extract_window_evidence(store, window)
    pooled = 0
    for record in store.for_window(window):
        for metrics in (record.stage1_metrics, record.stage2_metrics):
            if metrics is not None:
                pooled += episodes_from_transitions(
                    metrics.simulator_transitions, C.ROLLOUT_LENGTH)
    if evidence:
        mean_sr = round(sum(e.student_success_rate for e in evidence)
                        / len(evidence), 6)
    else:
        mean_sr = 0.0
    ci = 1.0 if pooled == 0 else ci_halfwidth(mean_sr, pooled, z)
    return BoardContext(window=window, mode=mode,
                        behavior_evidence=evidence,
                        pooled_episodes=pooled,
                        pooled_student_success_rate=mean_sr,
                        student_success_rate_ci=round(ci, 6),
                        feedback_view_label=feedback_view_label)
