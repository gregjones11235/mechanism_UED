"""FeedbackView — the ONLY surface through which the Review Board touches
probe feedback (structural basis of the comparison-mode isolation).

The board never receives a store. It receives a view:

* ``NormalFeedbackView``   — read-only frozen snapshot of explicitly injected
                             records (the controller selects window <= k-1);
* ``NullFeedbackView``     — STRUCTURAL blocking: the type holds no store, no
                             records and no ids — feedback is unreachable by
                             construction, not by prompt omission (static
                             mode);
* ``PermutedFeedbackView`` — frozen recomputable permutation with anonymized
                             identities (shuffled mode; arrives with C9).

Every view exposes the same prompt payload shape, so the six roles are
identical across modes — only what they can see differs.
"""
from __future__ import annotations

from typing import Dict, List, Protocol, Sequence, runtime_checkable

from d052.feedback_llm_ued.simulator_feedback_store import (
    SimulatorFeedbackRecord,
)

VIEW_LABEL_NORMAL = "normal"
VIEW_LABEL_NULL = "null"
VIEW_LABEL_PERMUTED = "permuted"


@runtime_checkable
class FeedbackView(Protocol):
    label: str
    window_scope: int

    def records(self) -> List[SimulatorFeedbackRecord]:
        ...

    def to_prompt_payload(self) -> List[dict]:
        ...


def record_payload(record: SimulatorFeedbackRecord) -> Dict[str, object]:
    """The coarse, episode-level slice of one record the board may see.

    Environment-level configuration (mutation axes / axis values / held axes)
    is included — it is TaskParams-level information, never an action-
    guidance carrier. Probe metrics are reduced to the two episode-level
    success rates.
    """
    metrics = record.stage2_metrics or record.stage1_metrics
    return dict(
        feedback_id=record.feedback_id,
        candidate_id=record.candidate_id,
        window=record.window,
        environment_family=record.environment_family,
        mutation_axes=list(record.mutation_axes),
        axis_values=dict(record.axis_values),
        held_constant_axes=dict(record.held_constant_axes),
        distinguishes_hypothesis_ids=list(record.distinguishes_hypothesis_ids),
        expected_observed_match=record.expected_observed_match,
        expected_signature=dict(record.expected_signature),
        student_success_rate=(metrics.student_success_rate
                              if metrics is not None else 0.0),
        reference_success_rate=(metrics.reference_success_rate
                                if metrics is not None else 0.0))


class NormalFeedbackView:
    """Read-only frozen snapshot of explicitly injected feedback records."""

    label = VIEW_LABEL_NORMAL

    def __init__(self, records: Sequence[SimulatorFeedbackRecord], *,
                 window_scope: int) -> None:
        if window_scope < 0:
            raise ValueError(f"ILLEGAL_VIEW_WINDOW_SCOPE: {window_scope}")
        self.window_scope = window_scope
        #: sorted + tuple: immutable snapshot, deterministic order
        self._records = tuple(sorted(records, key=lambda r: r.feedback_id))

    @classmethod
    def from_store(cls, store, *, max_window: int) -> "NormalFeedbackView":
        records = [r for r in store.all() if r.window <= max_window]
        return cls(records, window_scope=max_window)

    def records(self) -> List[SimulatorFeedbackRecord]:
        return list(self._records)

    def to_prompt_payload(self) -> List[dict]:
        return [record_payload(r) for r in self._records]


class NullFeedbackView:
    """STRUCTURAL no-feedback view (static-no-feedback mode).

    Holds no store reference, no record list, no id index — there is nothing
    inside this object that could leak feedback, so isolation does not depend
    on prompt construction discipline. (C9 asserts the board context built
    from this view carries a zero feedback payload.)
    """

    label = VIEW_LABEL_NULL
    window_scope = -1

    def records(self) -> List[SimulatorFeedbackRecord]:
        return []

    def to_prompt_payload(self) -> List[dict]:
        return []
