"""Bounded, provenance-aware archive for encoded simulator states."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .archive_schema import FrontierArchiveEntry
from .errors import SchemaMismatchError
from .state_codec import EncodedState, StateCodec


class FrontierArchive:
    def __init__(self, capacity: int = 1024, per_bucket_quota: int = 32, codec: StateCodec | None = None):
        if capacity <= 0 or per_bucket_quota <= 0:
            raise ValueError("capacity and per_bucket_quota must be positive")
        self.capacity = int(capacity)
        self.per_bucket_quota = int(per_bucket_quota)
        self.codec = codec or StateCodec()
        self._entries: dict[str, FrontierArchiveEntry] = {}
        self._states: dict[str, EncodedState] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, entry: FrontierArchiveEntry, encoded_state: EncodedState) -> bool:
        if entry.state_id in self._entries:
            return False
        if len(self) >= self.capacity:
            return False
        if sum(e.bucket() == entry.bucket() for e in self._entries.values()) >= self.per_bucket_quota:
            return False
        if encoded_state.payload_hash != entry.state_hash:
            raise SchemaMismatchError("archive state_hash does not match encoded payload")
        self._entries[entry.state_id] = entry
        self._states[entry.state_id] = encoded_state
        return True

    def get(self, state_id: str) -> tuple[FrontierArchiveEntry, EncodedState]:
        return self._entries[state_id], self._states[state_id]

    def list(self) -> list[FrontierArchiveEntry]:
        return list(self._entries.values())

    def dedup(self) -> int:
        seen: set[str] = set()
        removed = 0
        for state_id in list(self._entries):
            entry = self._entries[state_id]
            key = entry.state_hash
            if key in seen:
                self._entries.pop(state_id); self._states.pop(state_id); removed += 1
            else:
                seen.add(key)
        return removed

    def validate(self) -> list[str]:
        errors = []
        for state_id, entry in self._entries.items():
            if state_id != entry.state_id:
                errors.append(f"entry key mismatch: {state_id}")
            state = self._states.get(state_id)
            if state is None or state.payload_hash != entry.state_hash:
                errors.append(f"state hash mismatch: {state_id}")
            if not entry.provenance_hash:
                errors.append(f"missing provenance hash: {state_id}")
        return errors

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": "frontier-archive/v1", "capacity": self.capacity,
                   "per_bucket_quota": self.per_bucket_quota,
                   "entries": [asdict(e) for e in self._entries.values()],
                   "states": {k: asdict(v) for k, v in self._states.items()}}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
        payload["archive_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
        path.write_text(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, codec: StateCodec | None = None) -> "FrontierArchive":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != "frontier-archive/v1":
            raise SchemaMismatchError("unsupported archive schema")
        archive = cls(int(payload["capacity"]), int(payload["per_bucket_quota"]), codec)
        for raw in payload.get("entries", []):
            entry = FrontierArchiveEntry(**raw)
            state = EncodedState(**payload["states"][entry.state_id])
            if not archive.add(entry, state):
                raise SchemaMismatchError("archive contains duplicate or over-quota entry")
        errors = archive.validate()
        if errors:
            raise SchemaMismatchError("; ".join(errors))
        return archive
