"""Production guard chain for the ONE production Archive write entry (P0-1).

``FrontierArchive.add_production_entry`` is the sole production write path and
runs this ENTIRE chain internally — it never relies on callers having invoked
any guard first.  Every guard is fail closed: any empty key identity, hash
mismatch, unregistered capture provenance, leakage hit or memory
incompatibility raises and nothing is written.

Guard order (each step raises on violation):

  1. StudentBindingGuard        — all five binding fields bound (no empties,
                                  no placeholders: ``assert_entry_bound``);
  2. identity cross-binding     — entry hashes must equal the supplied Student
                                  identity hashes exactly;
  3. checkpoint/params hash     — checkpoint id non-empty (+ optional expected
                                  id match), params hash 64-hex and equal to
                                  the expected recomputed value;
  4. DiscoveryProvenanceGuard   — entry + capture provenance are both
                                  TRAINING_DISCOVERY; the capture provenance
                                  must validate against the supplied registry
                                  (production registries only via the
                                  controller injection slot — enforced by
                                  ``validate_capture_provenance``);
  5. FormalDataLeakageGuard     — formal sources can never feed the archive;
                                  the achievement snapshot is swept for
                                  action-guidance keys;
  6. MemoryCompatibilityGuard   — entry memory mode equals the restore
                                  request mode; ``check_bound_entry_memory_request``
                                  must report compatible;
  7. StateCodec hash            — ``codec.decode`` recomputes the payload hash
                                  (tamper-sensitive), payload hash must equal
                                  ``entry.state_hash``, and the mode-conditional
                                  bundle fields (policy_memory /
                                  history_reference) must actually be present;
  8. entry provenance recompute — ``provenance_hash`` is recomputed from the
                                  entry's own content (set if empty, rejected
                                  on mismatch).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, replace
from typing import Any, Mapping

from .archive_schema import FrontierArchiveEntry
from .discovery_provenance import (
    BLOCKED_WAITING_FROZEN_FORMAL_ASSET_REGISTRY,
    DiscoveryProvenance,
    DiscoveryProvenanceRegistry,
    validate_capture_provenance,
)
from .errors import ArchiveWriteGuardError, ProductionBlockedError, SchemaMismatchError
from .memory_modes import MemoryRestoreMode, MemoryRestoreRequest
from .provenance import DataSource, FormalDataLeakageGuard, SearchActionLeakageGuard
from .state_codec import SCHEMA_VERSION, EncodedState, StateCodec
from .student_binding import assert_entry_bound, check_bound_entry_memory_request, identity_mapping_hash_fields

# Production write-path contract is implemented and enforced by this module.
ARCHIVE_PRODUCTION_WRITE_READY = True

ENTRY_PROVENANCE_SCHEMA = "frontier-archive.entry-provenance/v1"
SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


def compute_entry_provenance_hash(entry: FrontierArchiveEntry) -> str:
    """Canonical self-content hash of an entry (its own provenance excluded).

    Shared by the write guard and ``FrontierArchive.load_production`` so the
    semantics can never drift between the two paths.
    """
    payload = {
        "schema": ENTRY_PROVENANCE_SCHEMA,
        "entry": {k: v for k, v in asdict(entry).items() if k != "provenance_hash"},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def finalize_entry_provenance(entry: FrontierArchiveEntry) -> FrontierArchiveEntry:
    """Set the recomputed provenance hash if empty; reject any mismatch."""
    expected = compute_entry_provenance_hash(entry)
    if not str(entry.provenance_hash).strip():
        return replace(entry, provenance_hash=expected)
    if entry.provenance_hash != expected:
        raise ArchiveWriteGuardError(
            f"entry provenance hash mismatch: entry carries {str(entry.provenance_hash)[:16]}… "
            f"but its content recomputes to {expected[:16]}… (fail closed)")
    return entry


def verify_production_entry(entry: FrontierArchiveEntry,
                            encoded_state: EncodedState,
                            *,
                            capture_provenance: Any,
                            registry: DiscoveryProvenanceRegistry | None,
                            student_identity: Mapping[str, Any],
                            expected_parameter_hash: str,
                            memory_request: MemoryRestoreRequest,
                            codec: StateCodec | None = None,
                            allow_synthetic_fixture: bool = False,
                            expected_checkpoint_id: str | None = None) -> FrontierArchiveEntry:
    """Run the full production guard chain; return the provenance-finalized entry.

    Raises (never writes) on any violation.  ``registry=None`` is blocked
    outright: a production capture without the controller-injected registry is
    never admitted (``BLOCKED_WAITING_FROZEN_FORMAL_ASSET_REGISTRY``).
    """
    # 1. StudentBindingGuard: every binding field present and non-placeholder.
    assert_entry_bound(entry)

    # 2. Identity cross-binding against the supplied Student identity.
    fields = identity_mapping_hash_fields(student_identity)
    if not SHA256_HEX_RE.match(str(entry.source_student_identity_hash)):
        raise ArchiveWriteGuardError("source_student_identity_hash is not a sha256 hex digest")
    if entry.source_student_identity_hash != fields["identity_hash"]:
        raise ArchiveWriteGuardError(
            "entry source_student_identity_hash does not equal the bound Student identity hash")
    if entry.source_memory_spec_hash != fields["memory_spec_hash"]:
        raise ArchiveWriteGuardError(
            "entry source_memory_spec_hash does not equal the bound Student memory spec hash")

    # 3. Checkpoint + params hash verification.
    if not str(entry.source_checkpoint_id).strip():
        raise ArchiveWriteGuardError("source_checkpoint_id is empty (never guess)")
    if expected_checkpoint_id is not None and entry.source_checkpoint_id != expected_checkpoint_id:
        raise ArchiveWriteGuardError(
            f"checkpoint id mismatch: entry carries {entry.source_checkpoint_id!r}, "
            f"expected {expected_checkpoint_id!r}")
    if not SHA256_HEX_RE.match(str(entry.source_parameter_hash)):
        raise ArchiveWriteGuardError("source_parameter_hash is not a sha256 hex digest")
    if entry.source_parameter_hash != str(expected_parameter_hash):
        raise ArchiveWriteGuardError(
            "params hash mismatch: the entry's source_parameter_hash must equal the "
            "expected recomputed checkpoint params hash")
    if entry.source_parameter_hash != fields["params_sha256"]:
        raise ArchiveWriteGuardError(
            "entry source_parameter_hash does not equal the bound Student params_sha256")

    # 4. DiscoveryProvenanceGuard: TRAINING_DISCOVERY only, registry-validated.
    if registry is None:
        raise ProductionBlockedError(
            f"{BLOCKED_WAITING_FROZEN_FORMAL_ASSET_REGISTRY}: production archive writes "
            "require the controller-injected frozen formal asset registry")
    if entry.discovery_provenance != DiscoveryProvenance.TRAINING_DISCOVERY.value:
        raise ArchiveWriteGuardError(
            f"production capture provenance must be {DiscoveryProvenance.TRAINING_DISCOVERY.value}, "
            f"got {entry.discovery_provenance!r}")
    try:
        capture_enum = DiscoveryProvenance(capture_provenance.provenance)
    except (AttributeError, ValueError) as exc:
        raise ArchiveWriteGuardError(
            f"capture provenance carries an unknown provenance value: {exc}") from exc
    if capture_enum is not DiscoveryProvenance.TRAINING_DISCOVERY:
        raise ArchiveWriteGuardError(
            "capture provenance must be TRAINING_DISCOVERY for production capture")
    validate_capture_provenance(capture_provenance, registry=registry,
                                allow_synthetic_fixture=allow_synthetic_fixture)

    # 5. FormalDataLeakageGuard + action-guidance sweep over the snapshot.
    FormalDataLeakageGuard.assert_allowed(DataSource.TRAINING_FRONTIER_CAPTURE, "FrontierArchive")
    SearchActionLeakageGuard.validate_aggregate(
        {"achievement_snapshot": dict(entry.achievement_snapshot)})

    # 6. MemoryCompatibilityGuard: mode agreement + compatibility report.
    try:
        entry_mode = MemoryRestoreMode(str(entry.memory_mode))
        request_mode = MemoryRestoreMode(memory_request.mode)
    except ValueError as exc:
        raise ArchiveWriteGuardError(f"unknown memory restore mode: {exc}") from exc
    if entry_mode is not request_mode:
        raise ArchiveWriteGuardError(
            f"memory mode mismatch: entry carries {entry_mode.value}, "
            f"restore request carries {request_mode.value}")
    report = check_bound_entry_memory_request(entry, memory_request)
    if not report.compatible:
        raise ArchiveWriteGuardError(
            f"memory compatibility guard rejected the restore request: {tuple(report.reasons)}")

    # 7. StateCodec hash verification (payload recomputation) + presence checks.
    codec = codec or StateCodec()
    if encoded_state.schema_version != SCHEMA_VERSION:
        raise SchemaMismatchError(
            f"encoded state schema {encoded_state.schema_version!r} != {SCHEMA_VERSION!r}")
    bundle = codec.decode(encoded_state)  # recomputes payload hash; raises on tamper
    if encoded_state.payload_hash != entry.state_hash:
        raise SchemaMismatchError("archive state_hash does not match encoded payload hash")
    if entry_mode is MemoryRestoreMode.SAVED_POLICY_MEMORY and bundle.policy_memory is None:
        raise ArchiveWriteGuardError(
            "SAVED_POLICY_MEMORY entry requires the captured policy memory inside the "
            "state bundle (empty reference is never accepted)")
    if entry_mode is MemoryRestoreMode.HISTORY_BURN_IN and bundle.history_reference is None:
        raise ArchiveWriteGuardError(
            "HISTORY_BURN_IN entry requires a history/burn-in reference inside the "
            "state bundle (empty reference is never accepted)")

    # 8. Entry provenance recompute (set if empty, reject on mismatch).
    return finalize_entry_provenance(entry)
