"""SimulatorFeedbackStore — canonical store of per-candidate probe feedback.

One record per probed candidate environment: which plan produced it, which
axes it changed/held constant, what the Student and Reference probes measured
(episode-level aggregates ONLY), how expensive the probe was (simulator
transitions), and how observation compared to prediction (the
expected-vs-observed match, graded by the ExpectedObservedComparator and
stored here with hash binding).

Hard invariants (fail-closed, never silent coercion):

* Reference statistics live in a dict whose keys MUST be a subset of
  ``REFERENCE_ALLOWED_FIELDS``; any forbidden action-guidance carrier raises
  through ``ReferenceOutputGuard``.
* Formal-evaluation sources may never enter this store
  (``FormalSourceIsolationGuard``).
* Comparator verdicts are written only through ``bind_match`` and re-stamp
  ``record_hash``, so an audit can prove which feedback record was graded by
  which comparison.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import Field, model_validator

from d052.bagr_ued.hashing import canonical_sha256
from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.feedback_contracts import ProbeMetrics
from d052.feedback_llm_ued.formal_isolation import (
    FormalSourceIsolationGuard,
    ReferenceOutputGuard,
)
from d052.schemas.common import CanonicalModel, is_sha256_hex

#: a feedback record whose expected-vs-observed match has not been graded yet
MATCH_UNGRADED = "ungraded"
MATCH_STATES = frozenset({
    C.MATCH_DIRECTION_AGREE,
    C.MATCH_DIRECTION_OPPOSITE,
    C.MATCH_DIRECTION_NEUTRAL,
    MATCH_UNGRADED,
})

_ISOLATION = FormalSourceIsolationGuard()
_REFERENCE_GUARD = ReferenceOutputGuard()


class SimulatorFeedbackRecord(CanonicalModel):
    """Everything the loop may ever cite as feedback about ONE candidate."""

    feedback_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    candidate_hash: str = Field(min_length=1)
    source_plan_id: str = Field(min_length=1)
    window: int = Field(ge=0)
    environment_family: str = Field(min_length=1)
    #: changed axes (the mutation this candidate applies)
    mutation_axes: List[str] = Field(default_factory=list)
    axis_values: Dict[str, str] = Field(default_factory=dict)
    held_constant_axes: Dict[str, str] = Field(default_factory=dict)
    distinguishes_hypothesis_ids: List[str] = Field(default_factory=list)
    #: Stage-1 fast probe + Stage-2 full probe results (episode-level only)
    stage1_metrics: Optional[ProbeMetrics] = None
    stage2_metrics: Optional[ProbeMetrics] = None
    #: raw Reference statistics — keys MUST be in REFERENCE_ALLOWED_FIELDS
    reference_stats: Dict[str, float] = Field(default_factory=dict)
    #: the predicted signature this candidate was probed against (copied from
    #: the hypothesis it distinguishes; input to the comparator)
    expected_signature: Dict[str, float] = Field(default_factory=dict)
    expected_observed_match: str = MATCH_UNGRADED
    match_detail: Dict[str, object] = Field(default_factory=dict)
    provenance: Dict[str, object] = Field(default_factory=dict)
    record_hash: str = ""

    @model_validator(mode="after")
    def _validate_and_hash(self) -> "SimulatorFeedbackRecord":
        if self.environment_family not in C.ENVIRONMENT_FAMILIES:
            raise ValueError(
                f"UNKNOWN_ENVIRONMENT_FAMILY: {self.environment_family!r}")
        for axis in self.mutation_axes:
            if axis not in C.MUTATION_AXES:
                raise ValueError(f"ILLEGAL_FEEDBACK_AXIS: {axis!r}")
        if not is_sha256_hex(self.candidate_hash):
            raise ValueError(
                f"CANDIDATE_HASH_NOT_SHA256: {self.candidate_hash!r}")
        unknown = set(self.reference_stats) - C.REFERENCE_ALLOWED_FIELDS
        if unknown:
            raise ValueError(
                f"REFERENCE_FIELD_FORBIDDEN: {sorted(unknown)} — allowed "
                f"Reference fields are {sorted(C.REFERENCE_ALLOWED_FIELDS)}")
        _REFERENCE_GUARD.assert_clean(
            self.reference_stats,
            label=f"feedback:{self.feedback_id}.reference_stats")
        _REFERENCE_GUARD.assert_clean(
            self.provenance, label=f"feedback:{self.feedback_id}.provenance")
        _ISOLATION.assert_record_clean(
            self.model_dump(), label=f"feedback:{self.feedback_id}")
        if self.expected_observed_match not in MATCH_STATES:
            raise ValueError(
                f"ILLEGAL_MATCH_STATE: {self.expected_observed_match!r}")
        if not self.record_hash:
            payload = self.model_dump()
            payload.pop("record_hash", None)
            object.__setattr__(self, "record_hash", canonical_sha256(payload))
        return self

    def rehash(self) -> str:
        payload = self.model_dump()
        payload.pop("record_hash", None)
        return canonical_sha256(payload)

    @property
    def simulator_transitions(self) -> int:
        """Total probe cost of this candidate across stages."""
        total = 0
        if self.stage1_metrics is not None:
            total += self.stage1_metrics.simulator_transitions
        if self.stage2_metrics is not None:
            total += self.stage2_metrics.simulator_transitions
        return total


class SimulatorFeedbackStore:
    """Indexed, replayable store of SimulatorFeedbackRecords.

    The store is the ONLY writer of the expected-vs-observed match: the
    comparator proposes a direction and the store applies it through
    ``bind_match`` so the hash binding cannot be bypassed.
    """

    def __init__(self) -> None:
        self._by_feedback_id: Dict[str, SimulatorFeedbackRecord] = {}
        self._by_candidate_id: Dict[str, List[str]] = {}
        self._order: List[str] = []

    # -- construction -------------------------------------------------------
    def add(self, record: SimulatorFeedbackRecord) -> SimulatorFeedbackRecord:
        if record.feedback_id in self._by_feedback_id:
            raise ValueError(f"DUPLICATE_FEEDBACK_ID: {record.feedback_id!r}")
        self._by_feedback_id[record.feedback_id] = record
        self._by_candidate_id.setdefault(
            record.candidate_id, []).append(record.feedback_id)
        self._order.append(record.feedback_id)
        return record

    # -- comparator binding ---------------------------------------------------
    def bind_match(self, feedback_id: str, *, direction: str,
                   detail: Optional[Dict[str, object]] = None
                   ) -> SimulatorFeedbackRecord:
        if direction not in (C.MATCH_DIRECTION_AGREE,
                             C.MATCH_DIRECTION_OPPOSITE,
                             C.MATCH_DIRECTION_NEUTRAL):
            raise ValueError(f"ILLEGAL_MATCH_DIRECTION: {direction!r}")
        rec = self._require(feedback_id)
        object.__setattr__(rec, "expected_observed_match", direction)
        object.__setattr__(rec, "match_detail", dict(detail or {}))
        object.__setattr__(rec, "record_hash", rec.rehash())
        return rec

    # -- queries --------------------------------------------------------------
    def get(self, feedback_id: str) -> SimulatorFeedbackRecord:
        return self._require(feedback_id)

    def for_candidate(self, candidate_id: str) -> List[SimulatorFeedbackRecord]:
        return [self._by_feedback_id[f]
                for f in self._by_candidate_id.get(candidate_id, [])]

    def for_plan(self, plan_id: str) -> List[SimulatorFeedbackRecord]:
        return [self._by_feedback_id[f] for f in self._order
                if self._by_feedback_id[f].source_plan_id == plan_id]

    def for_window(self, window: int) -> List[SimulatorFeedbackRecord]:
        return [self._by_feedback_id[f] for f in self._order
                if self._by_feedback_id[f].window == window]

    def graded(self, direction: str) -> List[SimulatorFeedbackRecord]:
        return [self._by_feedback_id[f] for f in self._order
                if self._by_feedback_id[f].expected_observed_match == direction]

    def ids(self) -> List[str]:
        return list(self._order)

    def all(self) -> List[SimulatorFeedbackRecord]:
        return [self._by_feedback_id[f] for f in self._order]

    def dump(self) -> List[dict]:
        return [self._by_feedback_id[f].model_dump() for f in self._order]

    def _require(self, feedback_id: str) -> SimulatorFeedbackRecord:
        if feedback_id not in self._by_feedback_id:
            raise KeyError(f"UNKNOWN_FEEDBACK_ID: {feedback_id!r}")
        return self._by_feedback_id[feedback_id]
