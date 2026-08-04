"""Production guard chain for the ONE production Archive write entry (P0-1).

``FrontierArchive.add_production_entry`` is the sole production write path and
runs this ENTIRE chain internally — it never relies on callers having invoked
any guard first.  Every guard is fail closed: any empty key identity, hash
mismatch, unregistered capture provenance, leakage hit or memory
incompatibility raises and nothing is written.

Audit closure (CC4 follow-up, P0-1): the production signature carries NO
caller-supplied ``registry=`` and NO ``allow_synthetic_fixture=`` surface any
more.  The registry is read EXACTLY ONCE, inside this module, from the
controller injection slot (``discovery_provenance.production_registry()``),
and the capture provenance is validated ONLY through
``validate_capture_provenance_production``.  A TEST_ONLY registry can never
back a production write; an un-injected slot fails closed with
``BLOCKED_WAITING_FROZEN_FORMAL_ASSET_REGISTRY``; synthetic capture
provenance is rejected on the production path by construction.

Tests that need a synthetic registry use the strictly separated
``verify_test_fixture_entry`` / ``FrontierArchive.add_test_fixture_entry``
surface: TEST_ONLY naming (registry usage TEST_ONLY + SYNTHETIC_FIXTURE
provenance + ``TEST_ONLY_``-prefixed capture reason), never a production
attestation, and never loadable by the production persistence path
(``save_production`` / ``load_production`` refuse any entry whose discovery
provenance is not TRAINING_DISCOVERY), so fixture entries can never be
imported by ``one_window_pipeline``.

Guard order (each step raises on violation):

  1. StudentBindingGuard        — all five binding fields bound (no empties,
                                  no placeholders: ``assert_entry_bound``);
  2. identity cross-binding     — entry hashes must equal the supplied Student
                                  identity hashes exactly;
  3. checkpoint/params hash     — checkpoint id non-empty (+ optional expected
                                  id match), params hash 64-hex and equal to
                                  the expected recomputed value;
  4. DiscoveryProvenanceGuard   — entry + capture provenance must both equal
                                  the path's required class
                                  (TRAINING_DISCOVERY on production,
                                  SYNTHETIC_FIXTURE on the labelled test
                                  fixture path); the capture provenance must
                                  validate against a registry that the
                                  production path resolves ONLY from the
                                  controller injection slot;
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
    REGISTRY_USAGE_PRODUCTION,
    REGISTRY_USAGE_TEST_ONLY,
    DiscoveryProvenance,
    DiscoveryProvenanceRegistry,
    production_registry,
    production_registry_bound,
    validate_capture_provenance,
)
from .errors import (
    ArchiveWriteGuardError,
    ProductionBlockedError,
    ProvenanceViolationError,
    SchemaMismatchError,
)
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


def _verify_entry_chain(entry: FrontierArchiveEntry,
                        encoded_state: EncodedState,
                        *,
                        capture_provenance: Any,
                        registry: DiscoveryProvenanceRegistry,
                        required_provenance: DiscoveryProvenance,
                        allow_synthetic_fixture: bool,
                        student_identity: Mapping[str, Any],
                        expected_parameter_hash: str,
                        memory_request: MemoryRestoreRequest,
                        codec: StateCodec | None = None,
                        expected_checkpoint_id: str | None = None) -> FrontierArchiveEntry:
    """Shared guard chain (steps 1–8) for both entry paths.

    The registry object is ALWAYS resolved by the caller before entering the
    chain (production: the controller injection slot, single read; test
    fixture: an explicitly TEST_ONLY registry).  No caller-supplied registry
    ever reaches the production path.
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

    # 4. DiscoveryProvenanceGuard: path-specific provenance class only,
    #    validated against the path-resolved registry.
    if entry.discovery_provenance != required_provenance.value:
        raise ArchiveWriteGuardError(
            f"capture entry discovery_provenance must be {required_provenance.value}, "
            f"got {entry.discovery_provenance!r}")
    try:
        capture_enum = DiscoveryProvenance(capture_provenance.provenance)
    except (AttributeError, ValueError) as exc:
        raise ArchiveWriteGuardError(
            f"capture provenance carries an unknown provenance value: {exc}") from exc
    if capture_enum is not required_provenance:
        raise ArchiveWriteGuardError(
            f"capture provenance must be {required_provenance.value} for this entry path, "
            f"got {capture_enum.value}")
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


# Fixture entries must be labelled TEST_ONLY by their capture reason so that a
# synthetic entry can never masquerade as a real capture in any report.
TEST_FIXTURE_CAPTURE_REASON_PREFIX = "TEST_ONLY_"


def verify_production_entry(entry: FrontierArchiveEntry,
                            encoded_state: EncodedState,
                            *,
                            capture_provenance: Any,
                            student_identity: Mapping[str, Any],
                            expected_parameter_hash: str,
                            memory_request: MemoryRestoreRequest,
                            codec: StateCodec | None = None,
                            expected_checkpoint_id: str | None = None) -> FrontierArchiveEntry:
    """Run the full PRODUCTION guard chain; return the provenance-finalized entry.

    The registry is resolved EXACTLY ONCE from the controller injection slot
    (``discovery_provenance.production_registry()``) — this signature carries
    NO ``registry=`` and NO ``allow_synthetic_fixture=`` parameter any more
    (P0-1 bypass closure).  Raises (never writes) on any violation:

    * un-injected slot → ``ProductionBlockedError``
      (``BLOCKED_WAITING_FROZEN_FORMAL_ASSET_REGISTRY``);
    * a TEST_ONLY registry in the production slot → rejected (the injection
      slot itself only accepts PRODUCTION registries; this is defence in
      depth against a slot mutated out of band);
    * synthetic capture provenance → rejected by
      ``validate_capture_provenance`` with ``allow_synthetic_fixture=False``.

    The single ``production_registry()`` read also closes clear/re-inject
    races fail closed: if the slot is rotated mid-call, the inner
    registry-identity check of ``validate_capture_provenance`` rejects the
    stale reference instead of silently trusting it.
    """
    try:
        registry = production_registry()
    except ProvenanceViolationError as exc:
        raise ProductionBlockedError(
            f"{BLOCKED_WAITING_FROZEN_FORMAL_ASSET_REGISTRY}: production archive writes "
            f"require the controller-injected frozen formal asset registry ({exc})") from exc
    if registry.usage != REGISTRY_USAGE_PRODUCTION:
        raise ProductionBlockedError(
            "production archive writes reject a TEST_ONLY registry in the production "
            f"slot (usage={registry.usage!r}); only usage={REGISTRY_USAGE_PRODUCTION} "
            "registries injected via inject_frozen_formal_asset_registry may back "
            "production captures (fail closed)")
    return _verify_entry_chain(
        entry, encoded_state,
        capture_provenance=capture_provenance,
        registry=registry,
        required_provenance=DiscoveryProvenance.TRAINING_DISCOVERY,
        allow_synthetic_fixture=False,
        student_identity=student_identity,
        expected_parameter_hash=expected_parameter_hash,
        memory_request=memory_request,
        codec=codec,
        expected_checkpoint_id=expected_checkpoint_id)


def verify_test_fixture_entry(entry: FrontierArchiveEntry,
                              encoded_state: EncodedState,
                              *,
                              capture_provenance: Any,
                              registry: DiscoveryProvenanceRegistry,
                              student_identity: Mapping[str, Any],
                              expected_parameter_hash: str,
                              memory_request: MemoryRestoreRequest,
                              codec: StateCodec | None = None,
                              expected_checkpoint_id: str | None = None) -> FrontierArchiveEntry:
    """Strictly separated TEST-FIXTURE guard chain (contract tests only).

    Mechanical separation from the production path:

    * the registry is caller-supplied but MUST be ``usage=TEST_ONLY`` — a
      caller-supplied PRODUCTION registry is rejected here (PRODUCTION
      registries are admissible ONLY through the injection slot /
      ``verify_production_entry``); the currently injected production
      registry object is rejected too;
    * entry + capture provenance MUST be SYNTHETIC_FIXTURE
      (TRAINING_DISCOVERY is rejected — a fixture can never pose as a real
      capture);
    * the entry ``capture_reason`` MUST carry the ``TEST_ONLY_`` prefix
      (TEST_ONLY naming discipline);
    * the result is never a production attestation: ``save_production`` /
      ``load_production`` refuse any entry whose discovery provenance is not
      TRAINING_DISCOVERY, so fixture entries can never enter the production
      persistence layout nor be imported by ``one_window_pipeline``.
    """
    if not isinstance(registry, DiscoveryProvenanceRegistry):
        raise ArchiveWriteGuardError(
            "verify_test_fixture_entry requires an explicit DiscoveryProvenanceRegistry, "
            f"got {type(registry).__name__}")
    if registry.usage != REGISTRY_USAGE_TEST_ONLY:
        raise ArchiveWriteGuardError(
            "test fixture entries accept ONLY usage=TEST_ONLY registries; a "
            f"usage={registry.usage!r} registry can never be used outside the "
            "production injection slot (fail closed)")
    if production_registry_bound() and registry is production_registry():
        raise ArchiveWriteGuardError(
            "the injected production registry can never be reused as a test fixture "
            "registry (fail closed)")
    if not str(entry.capture_reason).startswith(TEST_FIXTURE_CAPTURE_REASON_PREFIX):
        raise ArchiveWriteGuardError(
            "test fixture entries must carry TEST_ONLY naming: capture_reason must start "
            f"with {TEST_FIXTURE_CAPTURE_REASON_PREFIX!r}, got {entry.capture_reason!r}")
    return _verify_entry_chain(
        entry, encoded_state,
        capture_provenance=capture_provenance,
        registry=registry,
        required_provenance=DiscoveryProvenance.SYNTHETIC_FIXTURE,
        allow_synthetic_fixture=True,
        student_identity=student_identity,
        expected_parameter_hash=expected_parameter_hash,
        memory_request=memory_request,
        codec=codec,
        expected_checkpoint_id=expected_checkpoint_id)
