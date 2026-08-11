"""The REAL auxiliary compute ledger.

Records every paid / real computation (LLM call, probe episode, update,
checkpoint write) with enough identity to prove idempotency. The ledger
is append-only inside one run; its digest binds the smoke evidence.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, List


class ProductionComputeLedger:
    """Append-only compute ledger (registry asset)."""

    def __init__(self):
        self._entries: List[Dict[str, Any]] = []
        self.object_identity_hash = hashlib.sha256(
            b"shared_runtime.production_compute_ledger.v1"
        ).hexdigest()
        self.registry_identity = self.object_identity_hash

    def record(self, *, kind: str, identity: str, **fields: Any) -> None:
        if not kind or not identity:
            raise ValueError(
                "compute ledger entries require kind + identity")
        entry = {
            "seq": len(self._entries),
            "kind": kind,
            "identity": identity,
            "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                             time.gmtime()),
        }
        entry.update(fields)
        self._entries.append(entry)

    def entries(self) -> List[Dict[str, Any]]:
        return list(self._entries)

    def counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for entry in self._entries:
            counts[entry["kind"]] = counts.get(entry["kind"], 0) + 1
        return counts

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self._entries, sort_keys=True,
                       separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()
