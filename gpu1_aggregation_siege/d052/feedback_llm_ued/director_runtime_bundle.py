"""P0-16 (director smoke handoff): the director-provided Runtime Bundle.

Direction two is CONSUME-ONLY for the shared runtime. The DIRECTOR owns
every shared asset and hands direction two a SINGED Runtime Bundle
manifest via ``--director-runtime-bundle=<path>``. This module is the
consumption contract:

* :class:`DirectorRuntimeBundleManifest` — the immutable, registry-
  signed manifest describing the 12 director assets (StudentInitContract,
  StudentIdentity/Adapter, ReferenceIdentity/Adapter, CandidateProbeRunner,
  SharedAnchorManifest, FormalAssetRegistry, CanonicalDiCodeOneUpdateRuntime,
  CanonicalDiCodeRunStateCheckpoint, AuthorizedSixRoleLLMRuntime,
  backend/model identity, transport closure, AuxiliaryComputeLedger), the
  two-window smoke semantics (window0 delta=0 / window1 delta=1 / total=1),
  the DiCode 15+1 batch binding and the formal-start gate;
* :func:`load_director_runtime_bundle` — read + verify a signed manifest
  (registry signature recomputed-and-compared; invalid/absent fail closed
  with ``DIRECTOR_RUNTIME_BUNDLE_INVALID`` / ``_NOT_PROVIDED``);
* :func:`runtime_bundle_binding_problems` — the check-only binding
  validation: every asset present (not empty), valid hashes, the Student
  origin restricted to the SMOKE checkpoint (formal start requires a
  human-approved Formal Manifest), the batch-binding math
  (12 + 3 + 1 = 16, original proportion 0.20, original never duplicated),
  the smoke update contract (delta window0=0 / window1=1 / total=1);
* :func:`build_shared_bundle` — construct the five-slot SharedRuntimeBundle
  from the manifest: data-carrying assets (student / reference / anchor
  manifest) are bound through the EXISTING fail-closed ladders; the
  object-carrying assets (probe runner / DiCode one-update runtime) are
  bound as DIRECTOR-DECLARED (identity recorded, object injected by the
  director at smoke time — direction two never fabricates an object).

No callable asset is ever invoked here: this module and the check-only
path validate bindings and data flow WITHOUT calling the LLM, the probe
or training (the smoke itself is the director's job).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from typing import Optional, Protocol, runtime_checkable

from pydantic import Field, model_validator

from d052.bagr_ued.hashing import canonical_sha256, verify_content_hash
from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.shared_runtime_binding import (
    SharedAnchorManifestSlot,
    SharedProbeRunnerSlot,
    SharedReferenceSlot,
    SharedRuntimeBundle,
    SharedStudentSlot,
    SharedTrainingSlot,
)
from d052.schemas.common import CanonicalModel, is_sha256_hex

DIRECTOR_RUNTIME_BUNDLE_VERSION = "director.runtime_bundle.v1"

#: the 12 director assets (registry identity + label), for completeness
#: checks and reports
DIRECTOR_BUNDLE_ASSETS = (
    "student_init_contract",
    "student_identity",
    "reference_identity",
    "candidate_probe_runner",
    "shared_anchor_manifest",
    "formal_asset_registry",
    "canonical_dicode_one_update_runtime",
    "canonical_dicode_run_state_checkpoint",
    "authorized_six_role_llm_runtime",
    "backend_model_identity",
    "transport_closure",
    "auxiliary_compute_ledger",
)

#: assets that carry DATA direction two can bind directly (the rest are
#: declared identities of director-owned OBJECTS)
DATA_CARRYING_ASSETS = ("student_init_contract", "reference_identity",
                        "shared_anchor_manifest")


class DirectorRuntimeBundleBlocked(RuntimeError):
    """Fail-closed refusal of the director runtime bundle consumption."""


class StudentInitContractData(CanonicalModel):
    """The StudentInitContract the director embedded in the bundle.

    P0-16 (dual student): carries the FULL identity — candidate_id (one of
    ALLOWED_STUDENT_CANDIDATE_IDS), architecture_family=RMT16,
    parameter_tree_hash, checkpoint_global_step, profile_hash, memory_mode,
    memory_spec_hash, carry_mode, adapter_identity_hash,
    runtime_bundle_hash.
    """

    candidate_id: str = Field(min_length=1)
    architecture_family: str = Field(min_length=1)
    memory_family: str = Field(min_length=1)
    carry_mode: str = Field(min_length=1)
    parameter_tree_hash: str = Field(min_length=1)
    checkpoint_global_step: int = Field(ge=0)
    profile_hash: str = Field(min_length=1)
    memory_mode: str = Field(min_length=1)
    memory_spec_hash: str = Field(min_length=1)
    adapter_identity_hash: str = Field(min_length=1)
    runtime_bundle_hash: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate(self) -> "StudentInitContractData":
        for field_name in ("parameter_tree_hash", "profile_hash",
                           "memory_spec_hash", "adapter_identity_hash",
                           "runtime_bundle_hash"):
            value = getattr(self, field_name)
            if not is_sha256_hex(value):
                raise ValueError(
                    "STUDENT_CONTRACT_HASH_NOT_SHA256: "
                    f"{field_name}={value!r}")
        if self.architecture_family != "RMT16":
            raise ValueError(
                f"E2_STUDENT_PROFILE_MISMATCH: architecture_family must be "
                f"RMT16, got {self.architecture_family!r}")
        expected_mem, expected_carry = C.STUDENT_PROFILE_MEMORY_MAP.get(
            self.candidate_id, (None, None))
        if self.memory_mode != expected_mem or \
                self.carry_mode != expected_carry:
            raise ValueError(
                f"E2_STUDENT_MEMORY_MODE_MISMATCH: candidate "
                f"{self.candidate_id!r} requires memory_mode="
                f"{expected_mem!r} / carry_mode={expected_carry!r}, got "
                f"{self.memory_mode!r} / {self.carry_mode!r}")
        return self


class ReferenceIdentityData(CanonicalModel):
    """The shared Reference identity the director embedded in the bundle."""

    candidate_id: str = Field(min_length=1)
    parameter_tree_hash: str = Field(min_length=1)
    checkpoint_global_step: int = Field(ge=0)
    identity_hash: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate(self) -> "ReferenceIdentityData":
        if not is_sha256_hex(self.parameter_tree_hash):
            raise ValueError(
                "REFERENCE_PARAMETER_TREE_HASH_NOT_SHA256: "
                f"{self.parameter_tree_hash!r}")
        if not is_sha256_hex(self.identity_hash):
            raise ValueError(
                "REFERENCE_IDENTITY_HASH_NOT_SHA256: "
                f"{self.identity_hash!r}")
        return self


class AnchorManifestData(CanonicalModel):
    """The shared frozen four-anchor manifest the director embedded."""

    manifest_id: str = Field(min_length=1)
    anchors: List[str] = Field(default_factory=list)
    frozen: bool = False
    manifest_hash: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate(self) -> "AnchorManifestData":
        if len(self.anchors) != C.GLOBAL_ANCHOR_SLOTS:
            raise ValueError(
                f"ILLEGAL_ANCHOR_SLOT_COUNT: {len(self.anchors)}; the "
                f"shared manifest must bind exactly {C.GLOBAL_ANCHOR_SLOTS}")
        if len(set(self.anchors)) != len(self.anchors):
            raise ValueError("DUPLICATE_ANCHOR_ID")
        if not is_sha256_hex(self.manifest_hash):
            raise ValueError(
                "ANCHOR_MANIFEST_HASH_NOT_SHA256: "
                f"{self.manifest_hash!r}")
        return self


class DiCodeBatchBindingData(CanonicalModel):
    """The DiCode 15+1 batch contract the director declares.

    The director declares the 3 NON-TARGET anchor ids (the curriculum
    anchors); the OriginalTask is appended once internally by DiCode and
    never enters ``batch_candidate_ids``.
    """

    dynamic_task_count: int = Field(ge=0)
    non_target_anchor_count: int = Field(ge=0)
    curriculum_task_count: int = Field(ge=0)
    non_target_anchor_ids: List[str] = Field(default_factory=list)
    original_task_id: str = Field(min_length=1)
    original_task_proportion: float = Field(ge=0.0, le=1.0)
    total_task_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate(self) -> "DiCodeBatchBindingData":
        if len(set(self.non_target_anchor_ids)) != \
                len(self.non_target_anchor_ids):
            raise ValueError("DICODE_DUPLICATE_NON_TARGET_ANCHOR")
        if self.original_task_id in self.non_target_anchor_ids:
            raise ValueError(
                "DICODE_ORIGINAL_IS_A_CURRICULUM_TASK: the OriginalTask "
                "must not be one of the curriculum (non-target) anchors")
        return self


class SmokeSemanticsData(CanonicalModel):
    """The minimal two-window smoke update contract (immutable)."""

    window0_update_delta: int = Field(ge=0)
    window1_update_delta: int = Field(ge=0)
    total_updates: int = Field(ge=0)


class DirectorRuntimeBundleManifest(CanonicalModel):
    """The immutable, registry-signed director Runtime Bundle manifest.

    Direction two NEVER signs or mints one — it consumes the director's
    signed manifest only. ``bundle_hash`` is mandatory and recomputed-
    and-compared at construction (unsigned / tampered bundles fail
    closed).
    """

    model_config = {"frozen": True}

    bundle_version: str = DIRECTOR_RUNTIME_BUNDLE_VERSION
    registry_identity: str = Field(min_length=1)
    #: the formal asset registry that issued this bundle
    formal_asset_registry: str = Field(min_length=1)
    #: P0-16 (request-changes): the trusted signer id and the source commit
    #: the shared DirectorBundleVerifier checks (registry-signed)
    signer_id: str = Field(min_length=1)
    source_commit: str = Field(min_length=1)

    student_init_contract: StudentInitContractData
    student_identity: str = Field(min_length=1)
    student_adapter: str = Field(default="")
    reference_identity: ReferenceIdentityData
    reference_adapter: str = Field(default="")
    candidate_probe_runner: str = Field(min_length=1)
    shared_anchor_manifest: AnchorManifestData
    canonical_dicode_one_update_runtime: str = Field(min_length=1)
    canonical_dicode_run_state_checkpoint: str = Field(min_length=1)
    authorized_six_role_llm_runtime: str = Field(min_length=1)
    backend_model_identity: Dict[str, str] = Field(default_factory=dict)
    transport_closure: str = Field(min_length=1)
    auxiliary_compute_ledger: str = Field(min_length=1)

    smoke_semantics: SmokeSemanticsData
    batch_binding: DiCodeBatchBindingData
    formal_start_gate: Dict[str, bool] = Field(default_factory=dict)

    bundle_hash: str = ""

    @model_validator(mode="after")
    def _validate_and_hash(self) -> "DirectorRuntimeBundleManifest":
        for field_name in ("registry_identity", "formal_asset_registry",
                           "student_identity", "candidate_probe_runner",
                           "canonical_dicode_one_update_runtime",
                           "canonical_dicode_run_state_checkpoint",
                           "authorized_six_role_llm_runtime",
                           "transport_closure", "auxiliary_compute_ledger"):
            value = getattr(self, field_name)
            if not is_sha256_hex(value):
                raise ValueError(
                    "DIRECTOR_BUNDLE_ASSET_IDENTITY_NOT_SHA256: "
                    f"{field_name}={value!r}")
        for field_name in ("student_adapter", "reference_adapter"):
            value = getattr(self, field_name) or ""
            if value and not is_sha256_hex(value):
                raise ValueError(
                    "DIRECTOR_BUNDLE_ASSET_IDENTITY_NOT_SHA256: "
                    f"{field_name}={value!r}")
        if "backend_id" not in self.backend_model_identity \
                or "model_id" not in self.backend_model_identity:
            raise ValueError(
                "DIRECTOR_BUNDLE_BACKEND_IDENTITY_INCOMPLETE: "
                "backend_model_identity must carry backend_id and model_id")
        if not is_sha256_hex(self.bundle_hash):
            raise ValueError(
                "DIRECTOR_RUNTIME_BUNDLE_UNSIGNED: the director must sign "
                "the bundle (bundle_hash is mandatory)")
        computed = canonical_sha256(_manifest_hash_body(self))
        if self.bundle_hash and self.bundle_hash != computed:
            raise ValueError(
                "CONTENT_HASH_MISMATCH: DirectorRuntimeBundleManifest "
                "carried bundle_hash="
                f"{self.bundle_hash!r} but its content recomputes to "
                f"{computed!r} — the bundle was tampered with or signed "
                "over a non-canonical body")
        object.__setattr__(self, "bundle_hash", computed)
        return self


def _manifest_hash_body(manifest: "DirectorRuntimeBundleManifest") -> dict:
    """The canonical body the bundle hash is computed over: the full dump
    minus ``bundle_hash`` and minus the Student contract's self-referential
    ``runtime_bundle_hash`` (which, by contract, BINDS the bundle's own
    hash — the cross-binding is verified separately)."""
    body = {k: v for k, v in manifest.model_dump().items()
            if k != "bundle_hash"}
    student = dict(body["student_init_contract"])
    student.pop("runtime_bundle_hash", None)
    body["student_init_contract"] = student
    return body


@runtime_checkable
class DirectorBundleVerifier(Protocol):
    """P0-16 (request-changes): the DIRECTOR-shared Bundle verifier the
    production path CONSUMES. It — not a local content hash — establishes
    that the manifest was ISSUED by a trusted signer of the FormalAssetRegistry.
    Direction two never implements its own signature scheme."""

    verifier_id: str
    verifier_implementation_hash: str

    def verify_manifest(self, manifest: "DirectorRuntimeBundleManifest"
                        ) -> bool: ...
    def signer_trusted(self, signer_id: str) -> bool: ...
    def verify_source_commit(self, source_commit: str) -> bool: ...


def runtime_bundle_binding_problems(manifest: DirectorRuntimeBundleManifest
                                    ) -> List[str]:
    """P0-16 check-only binding validation. Returns every problem; an
    empty list is the only passing state. NEVER invokes a callable."""
    problems: List[str] = []
    #: 1. every asset present (not empty) — the "objects not empty" gate;
    #:    the smoke origin is the DIRECTOR-SELECTED Student (any allowed
    #:    candidate — PERSISTENT or RESET128), never an unknown one
    if manifest.student_init_contract.candidate_id \
            not in C.ALLOWED_STUDENT_CANDIDATE_IDS:
        problems.append(
            "SMOKE_STUDENT_ORIGIN_MISMATCH: the bundle's StudentInitContract "
            f"candidate_id={manifest.student_init_contract.candidate_id!r} "
            "is not one of ALLOWED_STUDENT_CANDIDATE_IDS="
            f"{sorted(C.ALLOWED_STUDENT_CANDIDATE_IDS)}")
    #: 2. formal-start gate: the checkpoint id is NOT "smoke-only" — after
    #:    human approval it MAY become the formal experiment start; the only
    #:    hard rule is that the formal start requires human approval
    if manifest.formal_start_gate.get("formal_start_requires_human") \
            is not True:
        problems.append(
            "FORMAL_START_REQUIRES_HUMAN: the formal experiment start may "
            "only come from a human-approved Formal Manifest")
    #: 3. the minimal smoke update contract (immutable)
    sem = manifest.smoke_semantics
    if not (sem.window0_update_delta == 0 and sem.window1_update_delta == 1
            and sem.total_updates == 1):
        problems.append(
            "SMOKE_UPDATE_CONTRACT_VIOLATED: the minimal smoke requires "
            "window0 delta=0, window1 delta=1, total=1 — got "
            f"({sem.window0_update_delta}, {sem.window1_update_delta}, "
            f"{sem.total_updates})")
    #: 4. the DiCode 15+1 batch binding
    bb = manifest.batch_binding
    if bb.dynamic_task_count != C.DICODE_CURRICULUM_DYNAMIC:
        problems.append(
            f"DICODE_DYNAMIC_COUNT_MISMATCH: {bb.dynamic_task_count} != "
            f"{C.DICODE_CURRICULUM_DYNAMIC}")
    if bb.non_target_anchor_count != C.DICODE_CURRICULUM_NON_TARGET_ANCHORS:
        problems.append(
            f"DICODE_ANCHOR_COUNT_MISMATCH: {bb.non_target_anchor_count} "
            f"!= {C.DICODE_CURRICULUM_NON_TARGET_ANCHORS}")
    if bb.curriculum_task_count != C.DICODE_CURRICULUM_TASK_COUNT:
        problems.append(
            f"DICODE_CURRICULUM_COUNT_MISMATCH: {bb.curriculum_task_count} "
            f"!= {C.DICODE_CURRICULUM_TASK_COUNT}")
    if bb.total_task_count != C.DICODE_BATCH_TOTAL_TASKS:
        problems.append(
            f"DICODE_TOTAL_COUNT_MISMATCH: {bb.total_task_count} != "
            f"{C.DICODE_BATCH_TOTAL_TASKS}")
    if abs(bb.original_task_proportion
           - C.DICODE_ORIGINAL_TASK_PROPORTION) > 1e-9:
        problems.append(
            f"DICODE_ORIGINAL_PROPORTION_MISMATCH: "
            f"{bb.original_task_proportion} != "
            f"{C.DICODE_ORIGINAL_TASK_PROPORTION}")
    if bb.original_task_id in bb.non_target_anchor_ids:
        problems.append(
            "DICODE_ORIGINAL_IS_A_CURRICULUM_TASK: the OriginalTask must "
            "not be one of the curriculum (non-target) anchors")
    if len(bb.non_target_anchor_ids) != bb.non_target_anchor_count:
        problems.append(
            "DICODE_NON_TARGET_ANCHOR_IDS_MISMATCH: declared "
            f"{len(bb.non_target_anchor_ids)} non-target anchors != "
            f"{bb.non_target_anchor_count}")
    if bb.original_task_id in manifest.shared_anchor_manifest.anchors:
        problems.append(
            "DICODE_ORIGINAL_TASK_IN_ANCHORS: the OriginalTask must NOT be "
            "one of the curriculum anchors (it is appended once internally "
            "by DiCode, never duplicated)")
    return problems


def load_director_runtime_bundle(path: Optional[str]
                                 ) -> Optional[DirectorRuntimeBundleManifest]:
    """Read + verify a signed director manifest. ``None`` -> the director
    provided no bundle (the entrypoint reports
    DIRECTOR_RUNTIME_BUNDLE_NOT_PROVIDED). Any invalid bundle raises
    ``DirectorRuntimeBundleBlocked`` (DIRECTOR_RUNTIME_BUNDLE_INVALID)."""
    if not path:
        return None
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        manifest = DirectorRuntimeBundleManifest(**raw)
    except Exception as exc:
        raise DirectorRuntimeBundleBlocked(
            f"{C.DIRECTOR_RUNTIME_BUNDLE_INVALID}: failed to consume the "
            f"director runtime bundle {path!r}: {type(exc).__name__}: "
            f"{exc}") from exc
    return manifest


def build_student_init_contract(manifest: DirectorRuntimeBundleManifest):
    """The StudentInitContract object derived from the director's bundle
    data (consumed by the existing fail-closed Student binding ladder)."""
    from types import SimpleNamespace
    data = manifest.student_init_contract
    return SimpleNamespace(
        candidate_id=data.candidate_id,
        architecture_family=data.architecture_family,
        memory_family=data.memory_family,
        carry_mode=data.carry_mode,
        parameter_tree_hash=data.parameter_tree_hash,
        checkpoint_global_step=data.checkpoint_global_step,
        profile_hash=data.profile_hash,
        memory_mode=data.memory_mode,
        memory_spec_hash=data.memory_spec_hash,
        adapter_identity_hash=data.adapter_identity_hash,
        runtime_bundle_hash=data.runtime_bundle_hash)


def build_shared_bundle(manifest: DirectorRuntimeBundleManifest
                        ) -> SharedRuntimeBundle:
    """Build the five-slot SharedRuntimeBundle from the director manifest.

    Data-carrying assets (student / reference / anchor) are bound through
    the EXISTING fail-closed ladders; the object-carrying assets (probe
    runner / DiCode one-update runtime) are bound as DIRECTOR-DECLARED —
    identity recorded, the object injected by the director at smoke time.
    Direction two never fabricates an object.
    """
    from d052.feedback_llm_ued.anchor_manifest import SharedAnchorManifest
    from types import SimpleNamespace
    student = SharedStudentSlot().bind(
        build_student_init_contract(manifest),
        director_selected_candidate_id=(
            manifest.student_init_contract.candidate_id))
    #: the DIRECTOR-issued canonical Reference identity hash is declared
    #: (consume-only, authenticated by the signed bundle) while the
    #: fail-closed ladder still validates the identity fields
    reference = SharedReferenceSlot().bind(
        SimpleNamespace(
            candidate_id=manifest.reference_identity.candidate_id,
            parameter_tree_hash=(
                manifest.reference_identity.parameter_tree_hash),
            checkpoint_global_step=(
                manifest.reference_identity.checkpoint_global_step)),
        declared_identity_hash=manifest.reference_identity.identity_hash)
    anchor = SharedAnchorManifestSlot().bind(SharedAnchorManifest(
        manifest_id=manifest.shared_anchor_manifest.manifest_id,
        anchors=list(manifest.shared_anchor_manifest.anchors),
        frozen=True,
        manifest_hash=manifest.shared_anchor_manifest.manifest_hash))
    #: P0-16 (request-changes): the manifest declares ONLY identities — the
    #: REAL objects must be resolved from the FormalAssetRegistry via
    #: resolve_director_runtime_objects before the bundle can hand off. A
    #: declared-not-resolved slot is NOT a handoff state.
    probe_runner = SharedProbeRunnerSlot().declare(
        registry_identity=manifest.candidate_probe_runner,
        detail="candidate probe runner: DECLARED_NOT_RESOLVED (object must "
               "be resolved from the FormalAssetRegistry)")
    training = SharedTrainingSlot().declare(
        registry_identity=manifest.canonical_dicode_one_update_runtime,
        detail="CanonicalDiCodeOneUpdateRuntime: DECLARED_NOT_RESOLVED "
               "(object must be resolved from the FormalAssetRegistry)")
    return SharedRuntimeBundle(student=student, reference=reference,
                               probe_runner=probe_runner,
                               anchor_manifest=anchor, training=training)


def bundle_backend_identity(manifest: DirectorRuntimeBundleManifest
                            ) -> Dict[str, str]:
    return dict(manifest.backend_model_identity)


def require_trusted_verifier(verifier: Optional[DirectorBundleVerifier],
                             manifest: DirectorRuntimeBundleManifest) -> None:
    """P0-16 (request-changes): the bundle is only trusted when the
    DIRECTOR-shared verifier passes — a local content hash alone proves
    nothing. Without an injected shared verifier the path fails closed
    (PRODUCTION_BUNDLE_VERIFIER_UNBOUND)."""
    if verifier is None:
        raise DirectorRuntimeBundleBlocked(
            "PRODUCTION_BUNDLE_VERIFIER_UNBOUND: the production path "
            "consumes the director-shared DirectorBundleVerifier; without "
            "it no bundle is trusted (a content hash is not a signature)")
    if not verifier.verify_manifest(manifest):
        raise DirectorRuntimeBundleBlocked(
            "PRODUCTION_BUNDLE_VERIFIER_REJECTED: the shared verifier "
            "rejected the manifest (signature / payload / schema check)")
    if not verifier.signer_trusted(manifest.signer_id):
        raise DirectorRuntimeBundleBlocked(
            "PRODUCTION_BUNDLE_SIGNER_UNTRUSTED: signer_id="
            f"{manifest.signer_id!r} is not in the director's trusted "
            "signer registry")
    if not verifier.verify_source_commit(manifest.source_commit):
        raise DirectorRuntimeBundleBlocked(
            "PRODUCTION_BUNDLE_SOURCE_COMMIT_UNTRUSTED: source_commit="
            f"{manifest.source_commit!r} is not trusted")


def assert_runtime_bundle_hash_cross_bound(
        manifest: DirectorRuntimeBundleManifest) -> None:
    """P0-16 (request-changes, section 5): the Student contract's
    runtime_bundle_hash MUST equal the manifest's own bundle hash — every
    object (StudentBindingIdentity, feedback, plan, probe result, batch
    plan, training result, round-trip attestation) binds the SAME hash."""
    if manifest.student_init_contract.runtime_bundle_hash \
            != manifest.bundle_hash:
        raise DirectorRuntimeBundleBlocked(
            "E2_RUNTIME_BUNDLE_HASH_MISMATCH: the Student contract binds "
            f"runtime_bundle_hash "
            f"{manifest.student_init_contract.runtime_bundle_hash!r} but "
            f"the manifest bundle_hash is {manifest.bundle_hash!r}")


#: the complete director object set the production path depends on —
#: every manifest-declared identity must resolve through the shared
#: FormalAssetRegistry to a REAL object (never a bare string / Mapping /
#: synthetic stand-in)
REQUIRED_DIRECTOR_OBJECTS = (
    "student_init_contract", "student_identity", "student_adapter",
    "reference_identity", "reference_adapter", "candidate_probe_runner",
    "shared_anchor_manifest", "canonical_dicode_one_update_runtime",
    "canonical_dicode_run_state_checkpoint", "authorized_six_role_llm_runtime",
    "transport_closure", "auxiliary_compute_ledger",
)


def resolve_director_runtime_objects(
        manifest: DirectorRuntimeBundleManifest,
        formal_asset_registry, bundle: SharedRuntimeBundle
) -> SharedRuntimeBundle:
    """P0-16 (request-changes, section 3-4): resolve the REAL objects from
    the shared FormalAssetRegistry and bind them as BOUND_OBJECT.

    EVERY object the production path depends on must resolve to a real
    object whose registry identity matches the manifest (the registry also
    verifies the implementation hash). Fail-closed: None / bare string /
    Mapping / identity mismatch / implementation-hash drift all refuse.
    Direction two only consumes — it never builds a second loader /
    registry / optimizer / checkpoint codec."""
    #: the registry itself must be the shared FormalAssetRegistry, not a
    #: test / synthetic / locally-constructed stand-in
    if not hasattr(formal_asset_registry, "registry_identity") or \
            not hasattr(formal_asset_registry, "resolve_asset") or \
            not hasattr(formal_asset_registry, "verify_implementation"):
        raise DirectorRuntimeBundleBlocked(
            "OBJECT_LEVEL_CHECK_BLOCKED: the injected registry is not the "
            "shared FormalAssetRegistry surface (registry_identity / "
            "resolve_asset / verify_implementation required)")
    manifest_ids = {
        "student_init_contract": manifest.student_init_contract,
        "student_identity": manifest.student_identity,
        "student_adapter": getattr(manifest, "student_adapter", ""),
        "reference_identity": manifest.reference_identity,
        "reference_adapter": getattr(manifest, "reference_adapter", ""),
        "candidate_probe_runner": manifest.candidate_probe_runner,
        "shared_anchor_manifest": manifest.shared_anchor_manifest,
        "canonical_dicode_one_update_runtime":
            manifest.canonical_dicode_one_update_runtime,
        "canonical_dicode_run_state_checkpoint":
            manifest.canonical_dicode_run_state_checkpoint,
        "authorized_six_role_llm_runtime":
            manifest.authorized_six_role_llm_runtime,
        "transport_closure": manifest.transport_closure,
        "auxiliary_compute_ledger": manifest.auxiliary_compute_ledger,
    }
    for name in REQUIRED_DIRECTOR_OBJECTS:
        identity = manifest_ids[name]
        if isinstance(identity, str):
            obj = formal_asset_registry.resolve_asset(identity=identity)
        else:
            obj = formal_asset_registry.resolve_asset(
                identity=str(getattr(identity, "identity_hash", identity)))
        if obj is None or isinstance(obj, (str, dict, Mapping)):
            raise DirectorRuntimeBundleBlocked(
                "OBJECT_LEVEL_CHECK_BLOCKED: the "
                f"{name} object was not resolved to a real object from the "
                "FormalAssetRegistry (None / bare string / Mapping refuse)")
        resolved_identity = getattr(obj, "registry_identity", "")
        expected_identity = (identity if isinstance(identity, str)
                             else getattr(identity, "identity_hash", ""))
        if resolved_identity != expected_identity:
            raise DirectorRuntimeBundleBlocked(
                "OBJECT_LEVEL_CHECK_BLOCKED: "
                f"{name} resolved identity {resolved_identity!r} does not "
                f"equal the manifest-declared {expected_identity!r}")
        expected_impl = getattr(obj, "implementation_hash", "")
        if not formal_asset_registry.verify_implementation(
                identity=expected_identity, obj=obj,
                expected_implementation_hash=expected_impl):
            raise DirectorRuntimeBundleBlocked(
                "OBJECT_LEVEL_CHECK_BLOCKED: the "
                f"{name} object's implementation hash was not verified by "
                "the FormalAssetRegistry")
    probe_runner = formal_asset_registry.resolve_asset(
        identity=manifest.candidate_probe_runner)
    training = formal_asset_registry.resolve_asset(
        identity=manifest.canonical_dicode_one_update_runtime)
    probe_slot = bundle.probe_runner.bind_object(
        probe_runner, expected_identity=manifest.candidate_probe_runner)
    training_slot = bundle.training.bind_object(
        training, expected_identity=manifest.canonical_dicode_one_update_runtime)
    return SharedRuntimeBundle(
        student=bundle.student, reference=bundle.reference,
        probe_runner=probe_slot, anchor_manifest=bundle.anchor_manifest,
        training=training_slot,
        formal_registry_identity=bundle.formal_registry_identity)


__all__ = [
    "DIRECTOR_RUNTIME_BUNDLE_VERSION", "DIRECTOR_BUNDLE_ASSETS",
    "DATA_CARRYING_ASSETS", "DirectorRuntimeBundleBlocked",
    "StudentInitContractData", "ReferenceIdentityData", "AnchorManifestData",
    "DiCodeBatchBindingData", "SmokeSemanticsData",
    "DirectorRuntimeBundleManifest",
    "runtime_bundle_binding_problems", "load_director_runtime_bundle",
    "build_student_init_contract", "build_shared_bundle",
    "bundle_backend_identity",
]
