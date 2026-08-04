"""Bounded, provenance-aware archive for encoded simulator states.

``add()`` / ``save()`` / ``load()`` are the frozen v1 contract surface (their
behaviour and signatures never change — regression-pinned).

Production surface (P0-1, additive only):

- ``add_production_entry`` — the ONE production write entry.  Runs the full
  ``archive_guards.verify_production_entry`` guard chain INTERNALLY (binding,
  checkpoint/params hash, discovery provenance, formal-leakage, memory
  compatibility, codec payload re-hash, entry provenance recompute) before
  delegating dup/capacity/quota to the frozen ``add()``.  Never relies on the
  caller having run any guard first; any violation raises and nothing is
  written.  CC4 audit closure: this signature carries NO caller-supplied
  ``registry=`` and NO ``allow_synthetic_fixture=`` — the registry is read
  inside the guard chain from the controller injection slot only.
- ``add_test_fixture_entry`` — strictly separated TEST-ONLY write surface for
  contract tests: TEST_ONLY registry + SYNTHETIC_FIXTURE provenance +
  ``TEST_ONLY_``-prefixed capture reason; never returns a production
  attestation and never loadable by ``load_production``.
- ``save_production`` — v2 layout (entry_order + counts + archive_hash over
  everything except the hash itself), written atomically
  tmp → flush → fsync → ``os.replace``.  Refuses any entry whose discovery
  provenance is not TRAINING_DISCOVERY (fixture entries can never be
  persisted in the production layout).
- ``load_production`` — recomputes the archive hash, enforces 1:1
  entry ↔ encoded-state pairing, re-decodes every payload (hash recomputation),
  rechecks binding + entry provenance hashes + TRAINING_DISCOVERY provenance,
  and re-validates the capacity / quota invariants.  Any mismatch raises.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .archive_guards import (
    compute_entry_provenance_hash,
    verify_production_entry,
    verify_test_fixture_entry,
)
from .archive_schema import FrontierArchiveEntry
from .discovery_provenance import DiscoveryProvenance
from .errors import ArchiveWriteGuardError, SchemaMismatchError
from .memory_modes import MemoryRestoreMode
from .state_codec import EncodedState, StateCodec
from .student_binding import assert_entry_bound

PRODUCTION_SCHEMA_VERSION = "frontier-archive/v2"


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically: tmp → flush → fsync → os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


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
        payload = {"schema_version": "frontier-archive/v1", "capacity": self.capacity,
                   "per_bucket_quota": self.per_bucket_quota,
                   "entries": [asdict(e) for e in self._entries.values()],
                   "states": {k: asdict(v) for k, v in self._states.items()}}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
        payload["archive_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
        _atomic_write_text(path, json.dumps(payload, sort_keys=True, indent=2,
                                            ensure_ascii=False, default=str))

    # ------------------------------------------------------------------
    # Production surface (P0-1).  Additive only: the v1 methods above keep
    # their exact behaviour and signatures.
    # ------------------------------------------------------------------

    def add_production_entry(self, entry: FrontierArchiveEntry,
                             encoded_state: EncodedState, *,
                             capture_provenance: Any,
                             student_identity: Any,
                             expected_parameter_hash: str,
                             memory_request: Any,
                             expected_checkpoint_id: str | None = None
                             ) -> tuple[bool, FrontierArchiveEntry]:
        """The ONE production write entry (fail closed).

        Runs the complete ``archive_guards.verify_production_entry`` chain
        internally — binding, identity cross-binding, checkpoint/params hash,
        discovery provenance (registry resolved ONLY from the controller
        injection slot inside the guard chain), formal-leakage sweep, memory
        compatibility, codec payload re-hash and entry provenance recompute —
        then delegates dup/capacity/quota to the frozen ``add()``.

        CC4 audit closure (P0-1): this signature carries NO caller-supplied
        ``registry=`` and NO ``allow_synthetic_fixture=`` surface — those
        parameters were the registry-injection bypass and are removed.

        Any guard violation raises and nothing is written.  Returns
        ``(added, finalized_entry)`` where ``added`` is False only for the
        frozen capacity/dup/quota outcomes of ``add()``.
        """
        finalized = verify_production_entry(
            entry, encoded_state,
            capture_provenance=capture_provenance,
            student_identity=student_identity,
            expected_parameter_hash=expected_parameter_hash,
            memory_request=memory_request,
            codec=self.codec,
            expected_checkpoint_id=expected_checkpoint_id)
        return self.add(finalized, encoded_state), finalized

    def add_test_fixture_entry(self, entry: FrontierArchiveEntry,
                               encoded_state: EncodedState, *,
                               capture_provenance: Any,
                               registry: Any,
                               student_identity: Any,
                               expected_parameter_hash: str,
                               memory_request: Any,
                               expected_checkpoint_id: str | None = None
                               ) -> tuple[bool, FrontierArchiveEntry]:
        """Strictly separated TEST-ONLY write entry (contract tests only).

        Requirements enforced by ``archive_guards.verify_test_fixture_entry``:
        the registry must be ``usage=TEST_ONLY`` (a caller-supplied PRODUCTION
        registry is rejected, as is the injected production registry object);
        entry + capture provenance must be SYNTHETIC_FIXTURE; the entry
        ``capture_reason`` must carry the ``TEST_ONLY_`` prefix.

        This path NEVER returns a production attestation: entries added here
        carry SYNTHETIC_FIXTURE discovery provenance, which ``save_production``
        refuses to persist and ``load_production`` refuses to load — so
        fixture entries can never enter the production persistence layout nor
        be imported by ``one_window_pipeline``.
        """
        finalized = verify_test_fixture_entry(
            entry, encoded_state,
            capture_provenance=capture_provenance,
            registry=registry,
            student_identity=student_identity,
            expected_parameter_hash=expected_parameter_hash,
            memory_request=memory_request,
            codec=self.codec,
            expected_checkpoint_id=expected_checkpoint_id)
        return self.add(finalized, encoded_state), finalized

    def save_production(self, path: str | Path) -> None:
        """Persist the archive atomically in the v2 production layout.

        Refuses (raises ``ArchiveWriteGuardError``) to write unless every
        entry is fully bound, carries TRAINING_DISCOVERY discovery provenance
        (test fixture entries can never be persisted in the production
        layout), carries a self-consistent recomputed provenance hash, and
        its encoded state payload hash matches — the production layout never
        persists partially-bound or fixture-labelled content.

        Layout: schema_version=frontier-archive/v2, capacity, per_bucket_quota,
        entry_order, entry_count, state_count, entries, states, and an
        ``archive_hash`` computed over everything except the hash itself.
        The write goes tmp → flush → fsync → ``os.replace``.
        """
        for state_id, entry in self._entries.items():
            assert_entry_bound(entry)
            if entry.discovery_provenance != DiscoveryProvenance.TRAINING_DISCOVERY.value:
                raise ArchiveWriteGuardError(
                    f"production save refused: entry {state_id} discovery_provenance is "
                    f"{entry.discovery_provenance!r}; the production persistence layout "
                    f"accepts ONLY {DiscoveryProvenance.TRAINING_DISCOVERY.value} entries "
                    "(test fixture entries are never production content; fail closed)")
            expected = compute_entry_provenance_hash(entry)
            if entry.provenance_hash != expected:
                raise ArchiveWriteGuardError(
                    f"production save refused: entry {state_id} provenance hash "
                    f"{str(entry.provenance_hash)[:16]}… does not match recomputed "
                    f"{expected[:16]}… (fail closed)")
            state = self._states.get(state_id)
            if state is None or state.payload_hash != entry.state_hash:
                raise ArchiveWriteGuardError(
                    f"production save refused: entry {state_id} state payload hash "
                    "does not match entry.state_hash (fail closed)")
        payload = {"schema_version": PRODUCTION_SCHEMA_VERSION,
                   "capacity": self.capacity,
                   "per_bucket_quota": self.per_bucket_quota,
                   "entry_order": [e.state_id for e in self._entries.values()],
                   "entry_count": len(self._entries),
                   "state_count": len(self._states),
                   "entries": [asdict(e) for e in self._entries.values()],
                   "states": {k: asdict(v) for k, v in self._states.items()}}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, default=str)
        payload["archive_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        _atomic_write_text(Path(path), json.dumps(payload, sort_keys=True, indent=2,
                                                  ensure_ascii=False, default=str))

    @classmethod
    def load_production(cls, path: str | Path, codec: StateCodec | None = None) -> "FrontierArchive":
        """Load a v2 production archive WITH full re-verification.

        Re-verification (each mismatch raises, nothing is trusted from disk):
        archive_hash recomputation; 1:1 entry_order ↔ entries ↔ states pairing;
        per-entry ``codec.decode`` (payload hash recomputation); entry ↔ state
        hash equality; binding presence; entry provenance hash recomputation;
        memory mode validity + mode-conditional bundle presence; capacity and
        quota invariants via the frozen ``add()``.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != PRODUCTION_SCHEMA_VERSION:
            raise SchemaMismatchError(
                f"production load requires {PRODUCTION_SCHEMA_VERSION}, "
                f"got {payload.get('schema_version')!r}")
        stored_hash = payload.pop("archive_hash", None)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, default=str)
        recomputed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if stored_hash != recomputed:
            raise SchemaMismatchError(
                "production archive hash mismatch (content tampered or corrupted)")
        entries_raw = payload.get("entries", [])
        states_raw = payload.get("states", {})
        order = payload.get("entry_order", [])
        if len(entries_raw) != int(payload.get("entry_count", -1)):
            raise SchemaMismatchError("production archive entry_count mismatch")
        if len(states_raw) != int(payload.get("state_count", -1)):
            raise SchemaMismatchError("production archive state_count mismatch")
        if [raw.get("state_id") for raw in entries_raw] != list(order):
            raise SchemaMismatchError("production archive entry_order does not match entries")
        if {raw.get("state_id") for raw in entries_raw} != set(states_raw):
            raise SchemaMismatchError("production archive entries and states are not 1:1")

        archive = cls(int(payload["capacity"]), int(payload["per_bucket_quota"]), codec)
        for raw in entries_raw:
            entry = FrontierArchiveEntry(**raw)
            state = EncodedState(**states_raw[entry.state_id])
            # Codec decode recomputes the payload hash (tamper-sensitive).
            bundle = archive.codec.decode(state)
            if state.payload_hash != entry.state_hash:
                raise SchemaMismatchError(
                    f"production load: state payload hash != entry.state_hash ({entry.state_id})")
            assert_entry_bound(entry)
            if entry.discovery_provenance != DiscoveryProvenance.TRAINING_DISCOVERY.value:
                raise SchemaMismatchError(
                    f"production load: entry {entry.state_id} discovery_provenance is "
                    f"{entry.discovery_provenance!r}; only "
                    f"{DiscoveryProvenance.TRAINING_DISCOVERY.value} entries may exist in "
                    "the production persistence layout (fixture entries rejected)")
            expected_prov = compute_entry_provenance_hash(entry)
            if entry.provenance_hash != expected_prov:
                raise SchemaMismatchError(
                    f"production load: entry provenance hash mismatch ({entry.state_id})")
            try:
                mode = MemoryRestoreMode(str(entry.memory_mode))
            except ValueError as exc:
                raise SchemaMismatchError(
                    f"production load: unknown memory mode {entry.memory_mode!r} "
                    f"({entry.state_id})") from exc
            if mode is MemoryRestoreMode.SAVED_POLICY_MEMORY and bundle.policy_memory is None:
                raise SchemaMismatchError(
                    f"production load: SAVED_POLICY_MEMORY entry {entry.state_id} "
                    "lacks its policy memory in the bundle")
            if mode is MemoryRestoreMode.HISTORY_BURN_IN and bundle.history_reference is None:
                raise SchemaMismatchError(
                    f"production load: HISTORY_BURN_IN entry {entry.state_id} "
                    "lacks its history reference in the bundle")
            if not archive.add(entry, state):
                raise SchemaMismatchError(
                    f"production load: duplicate or over-quota entry ({entry.state_id})")
        errors = archive.validate()
        if errors:
            raise SchemaMismatchError("; ".join(errors))
        return archive

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
