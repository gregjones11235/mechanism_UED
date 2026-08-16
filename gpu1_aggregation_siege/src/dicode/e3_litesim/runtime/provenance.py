"""Hash-chained provenance records for litesim states and artifacts."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping

from .hashing import hash_payload

GENESIS_HASH = "0" * 64


@dataclass(frozen=True)
class ProvenanceRecord:
    kind: str
    payload: Mapping[str, Any]
    prev_hash: str = GENESIS_HASH
    record_hash: str = ""

    def finalized(self) -> "ProvenanceRecord":
        body = {"kind": self.kind, "prev_hash": self.prev_hash, "payload": dict(self.payload)}
        return ProvenanceRecord(self.kind, dict(self.payload), self.prev_hash,
                                hash_payload(body))

    def to_dict(self) -> dict:
        return asdict(self)