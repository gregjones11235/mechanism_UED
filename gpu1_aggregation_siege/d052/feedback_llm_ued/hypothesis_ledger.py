"""HypothesisLedger — the durable, hash-bound ledger of behavior hypotheses.

A *hypothesis* is a falsifiable claim about the Student's behavior that an
environment family is meant to probe, e.g. "the Student under-weights rest
need when a hostile band is near". Each hypothesis carries a PREDICTED
signature (what the probe metrics should look like if the claim holds) so the
ExpectedObservedComparator can score agreement, and the FeedbackDiagnostician
can flip its status.

Every record is a ``CanonicalModel`` (extra=forbid) with a content hash; every
status change appends an immutable revision-history entry binding (window,
old->new status, feedback ids, reason) so the ledger is replayable and a later
audit can prove WHICH feedback moved WHICH hypothesis.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import Field, model_validator

from d052.bagr_ued.hashing import canonical_sha256, verify_content_hash
from d052.feedback_llm_ued import constants as C
from d052.schemas.common import CanonicalModel


class HypothesisRecord(CanonicalModel):
    hypothesis_id: str = Field(min_length=1)
    source_window: int = Field(ge=0)
    target_behavior: str = Field(min_length=1)
    evidence_ids: List[str] = Field(default_factory=list)
    predicted_signature: Dict[str, float] = Field(default_factory=dict)
    environment_family: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    status: str = C.HYPOTHESIS_PENDING
    supporting_feedback_ids: List[str] = Field(default_factory=list)
    contradicting_feedback_ids: List[str] = Field(default_factory=list)
    revision_history: List[Dict[str, object]] = Field(default_factory=list)
    record_hash: str = ""

    @model_validator(mode="after")
    def _validate_and_hash(self) -> "HypothesisRecord":
        if self.status not in C.HYPOTHESIS_STATUSES:
            raise ValueError(f"ILLEGAL_HYPOTHESIS_STATUS: {self.status!r}")
        if self.environment_family not in C.ENVIRONMENT_FAMILIES:
            raise ValueError(
                f"UNKNOWN_ENVIRONMENT_FAMILY: {self.environment_family!r}")
        # C14: an externally carried record_hash is recomputed and compared
        # verbatim (CONTENT_HASH_MISMATCH fails closed)
        computed = verify_content_hash(self.model_dump(),
                                       hash_field="record_hash",
                                       carried=self.record_hash,
                                       kind="HypothesisRecord")
        object.__setattr__(self, "record_hash", computed)
        return self

    def rehash(self) -> str:
        payload = self.model_dump()
        payload.pop("record_hash", None)
        return canonical_sha256(payload)


class HypothesisLedger:
    """Indexed, replayable store of HypothesisRecords.

    The ledger is the ONLY writer of hypothesis status; the diagnostician
    proposes verdicts and the ledger applies them through ``apply_verdict`` so
    the revision history + hash binding cannot be bypassed.
    """

    def __init__(self) -> None:
        self._by_id: Dict[str, HypothesisRecord] = {}
        self._order: List[str] = []

    # -- construction -------------------------------------------------------
    def register(self, record: HypothesisRecord) -> HypothesisRecord:
        if record.hypothesis_id in self._by_id:
            raise ValueError(
                f"DUPLICATE_HYPOTHESIS_ID: {record.hypothesis_id!r}")
        self._by_id[record.hypothesis_id] = record
        self._order.append(record.hypothesis_id)
        return record

    # -- feedback binding ---------------------------------------------------
    def bind_feedback(self, hypothesis_id: str, feedback_id: str, *,
                      agrees: bool) -> None:
        rec = self._require(hypothesis_id)
        bucket = (rec.supporting_feedback_ids if agrees
                  else rec.contradicting_feedback_ids)
        if feedback_id not in bucket:
            bucket.append(feedback_id)
        object.__setattr__(rec, "record_hash", rec.rehash())

    # -- verdicts -----------------------------------------------------------
    def apply_verdict(self, hypothesis_id: str, *, status: str, window: int,
                      reason: str, feedback_ids: Optional[List[str]] = None,
                      confidence: Optional[float] = None) -> HypothesisRecord:
        rec = self._require(hypothesis_id)
        if status not in C.HYPOTHESIS_STATUSES:
            raise ValueError(f"ILLEGAL_HYPOTHESIS_STATUS: {status!r}")
        previous = rec.status
        entry = dict(window=window, previous_status=previous, new_status=status,
                     feedback_ids=list(feedback_ids or []), reason=reason,
                     previous_record_hash=rec.record_hash)
        object.__setattr__(rec, "status", status)
        if confidence is not None:
            if not (0.0 <= confidence <= 1.0):
                raise ValueError(f"CONFIDENCE_OUT_OF_RANGE: {confidence!r}")
            object.__setattr__(rec, "confidence", confidence)
        object.__setattr__(rec, "revision_history",
                           rec.revision_history + [entry])
        object.__setattr__(rec, "record_hash", rec.rehash())
        return rec

    def mark_stale(self, hypothesis_id: str, *, window: int,
                   reason: str) -> HypothesisRecord:
        return self.apply_verdict(hypothesis_id, status=C.HYPOTHESIS_STALE,
                                  window=window, reason=reason)

    # -- queries ------------------------------------------------------------
    def get(self, hypothesis_id: str) -> HypothesisRecord:
        return self._require(hypothesis_id)

    def all(self) -> List[HypothesisRecord]:
        return [self._by_id[h] for h in self._order]

    def by_status(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {s: [] for s in
                                     sorted(C.HYPOTHESIS_STATUSES)}
        for h in self._order:
            out[self._by_id[h].status].append(h)
        return out

    def ids(self) -> List[str]:
        return list(self._order)

    def dump(self) -> List[dict]:
        return [self._by_id[h].model_dump() for h in self._order]

    def _require(self, hypothesis_id: str) -> HypothesisRecord:
        if hypothesis_id not in self._by_id:
            raise KeyError(f"UNKNOWN_HYPOTHESIS_ID: {hypothesis_id!r}")
        return self._by_id[hypothesis_id]
