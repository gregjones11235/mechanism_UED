"""CausalEvidence aggregation: UNKNOWN is a first-class outcome."""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import List

from ..runtime.hashing import hash_payload


@dataclass
class CauseRecord:
    name: str
    cause_family: str
    delta_success: float
    delta_progress: float
    support: dict

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CausalEvidence:
    records: List[CauseRecord]
    cause: str
    confidence: float
    confounders: List[str]
    unknown: bool
    evidence_hash: str

    def to_dict(self) -> dict:
        return {
            "records": [r.to_dict() for r in self.records],
            "cause": self.cause, "confidence": self.confidence,
            "confounders": self.confounders, "unknown": self.unknown,
            "evidence_hash": self.evidence_hash,
        }


def aggregate_causal_evidence(records: List[CauseRecord],
                              threshold: float = 0.15) -> CausalEvidence:
    ranked = sorted(records, key=lambda r: -abs(r.delta_success))
    confounders = [r.name for r in records
                   if abs(r.delta_success) >= threshold]
    if ranked and abs(ranked[0].delta_success) >= threshold:
        cause = ranked[0].cause_family
        confidence = min(1.0, abs(ranked[0].delta_success))
        unknown = False
    else:
        cause = "UNKNOWN"
        confidence = 0.0
        unknown = True
    body = [r.to_dict() for r in records]
    return CausalEvidence(records=records, cause=cause,
                          confidence=float(confidence),
                          confounders=confounders, unknown=unknown,
                          evidence_hash=hash_payload(body))