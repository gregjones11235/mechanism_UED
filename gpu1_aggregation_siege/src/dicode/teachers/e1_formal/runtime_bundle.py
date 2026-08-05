"""CC2 follow-up P0-1/P0-3: the signed E1 runtime bundle.

The one-window production pipeline consumes EVERY shared runtime
object through ONE signed carrier — the ``E1RuntimeBundle`` — instead
of string contract names or caller-shaped mappings:

* a PRODUCTION bundle's ``object_ref``s can only come from the
  supervisor-signed ``E1ProductionRuntimeBundle`` surface; the signer
  whitelist (``AUTHORIZED_BUNDLE_SIGNERS``) is supervisor-owned and
  EMPTY this round, so no production bundle can verify yet (honest
  ``RUNTIME_BUNDLE_SIGNER_UNAUTHORIZED``);
* a TEST_ONLY bundle carries conspicuously-marked SYNTHETIC objects
  (``TEST_ONLY`` / ``SYNTHETIC`` / ``NOT_REAL_EXECUTION``) and is ONLY
  admissible on explicitly test-only flows: any production surface
  refuses it with ``RUNTIME_BUNDLE_TEST_ONLY_REJECTED``. It exists to
  prove the code path and identity binding, never to mint real
  evidence;
* a bundle is hash-bound: ``bundle_hash`` is the canonical sha256 of
  its identity fields + per-contract capability descriptors, and every
  consumer re-checks it (tamper => fail closed).

Discipline: this module NEVER constructs a shared object on its own;
it only carries, parses and verifies. String placeholders are refused
everywhere a real object is required.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from .canonical import canonical_sha256
from .schemas import E1SchemaError
from .student_contract import (
    ALLOWED_STUDENT_CANDIDATE_IDS,
    STUDENT_CANDIDATE_BY_PROFILE,
    STUDENT_CARRY_MODE_BY_CANDIDATE,
    STUDENT_MEMORY_MODE_BY_CANDIDATE,
    STUDENT_PROFILE_BY_CANDIDATE,
)

#: bundle modes — the ONLY two values ever admitted
BUNDLE_MODE_PRODUCTION = "PRODUCTION"
BUNDLE_MODE_TEST_ONLY = "TEST_ONLY"

#: the synthetic signer every TEST_ONLY bundle must carry (greppable)
SYNTHETIC_TEST_ONLY_SIGNER = "SYNTHETIC_TEST_ONLY_SIGNER"

#: supervisor-owned production signer whitelist — EMPTY this round.
#: A production bundle is verifiable ONLY when its signer is here.
AUTHORIZED_BUNDLE_SIGNERS: Tuple[str, ...] = ()

#: the nine runtime capability contracts EVERY bundle must carry as
#: REAL objects (never strings, never None)
RUNTIME_CAPABILITY_CONTRACTS = (
    "student_identity",
    "reference_identity",
    "student_adapter",
    "reference_adapter",
    "anchor_manifest",
    "formal_asset_registry",
    "probe_runner",
    "training",
    "full_state_checkpoint",
)

# fail-closed codes (greppable)
RUNTIME_BUNDLE_BAD_TYPE = "RUNTIME_BUNDLE_BAD_TYPE"
RUNTIME_BUNDLE_MISSING_FIELD = "RUNTIME_BUNDLE_MISSING_FIELD"
RUNTIME_BUNDLE_UNKNOWN_FIELD = "RUNTIME_BUNDLE_UNKNOWN_FIELD"
RUNTIME_BUNDLE_HASH_MISMATCH = "RUNTIME_BUNDLE_HASH_MISMATCH"
RUNTIME_BUNDLE_SIGNER_UNAUTHORIZED = "RUNTIME_BUNDLE_SIGNER_UNAUTHORIZED"
RUNTIME_BUNDLE_TEST_ONLY_REJECTED = "RUNTIME_BUNDLE_TEST_ONLY_REJECTED"
RUNTIME_BUNDLE_STRING_PLACEHOLDER = "RUNTIME_BUNDLE_STRING_PLACEHOLDER"
RUNTIME_BUNDLE_UNBOUND = "RUNTIME_BUNDLE_UNBOUND"
RUNTIME_BUNDLE_GRANT_MISMATCH = "RUNTIME_BUNDLE_GRANT_MISMATCH"
#: student-selection schema codes
RUNTIME_BUNDLE_STUDENT_SELECTION_MISSING = (
    "RUNTIME_BUNDLE_STUDENT_SELECTION_MISSING"
)
RUNTIME_BUNDLE_STUDENT_SELECTION_BAD = (
    "RUNTIME_BUNDLE_STUDENT_SELECTION_BAD"
)
RUNTIME_BUNDLE_STUDENT_IDENTITY_MISMATCH = (
    "RUNTIME_BUNDLE_STUDENT_IDENTITY_MISMATCH"
)


@dataclass(frozen=True)
class StudentSelectionDescriptor:
    """The director-issued Student selection (immutable, hash-bound).

    CC2-Student repair: the Runtime Bundle EXPLICITLY carries the
    selected Student; the selection is never read from a nonexistent
    ``bundle.student`` attribute. The profile / memory mode / carry
    mode come from the EXPLICIT frozen mapping (never guessed).
    """

    selected_candidate_id: str
    profile_id: str
    architecture_family: str
    memory_mode: str
    memory_spec_hash: str
    carry_mode: str
    checkpoint_path: str
    checkpoint_file_sha256: str
    params_sha256: str
    adapter_identity_hash: str
    adapter_implementation_hash: str
    driver_source_path: str
    driver_source_sha256: str
    source_commit: str
    descriptor_hash: str


#: the exact manifest field set for the student selection block
_STUDENT_SELECTION_FIELDS = frozenset(
    {
        "selected_candidate_id",
        "profile_id",
        "architecture_family",
        "memory_mode",
        "memory_spec_hash",
        "carry_mode",
        "checkpoint_path",
        "checkpoint_file_sha256",
        "params_sha256",
        "adapter_identity_hash",
        "adapter_implementation_hash",
        "driver_source_path",
        "driver_source_sha256",
        "source_commit",
    }
)

_STUDENT_HASH_FIELDS = (
    "memory_spec_hash",
    "checkpoint_file_sha256",
    "params_sha256",
    "adapter_identity_hash",
    "adapter_implementation_hash",
    "driver_source_sha256",
)


def compute_student_selection_hash(descriptor: Any) -> str:
    """The canonical identity of one StudentSelectionDescriptor."""
    return canonical_sha256(
        {
            "selected_candidate_id": descriptor.selected_candidate_id,
            "profile_id": descriptor.profile_id,
            "architecture_family": descriptor.architecture_family,
            "memory_mode": descriptor.memory_mode,
            "memory_spec_hash": descriptor.memory_spec_hash,
            "carry_mode": descriptor.carry_mode,
            "checkpoint_path": descriptor.checkpoint_path,
            "checkpoint_file_sha256": descriptor.checkpoint_file_sha256,
            "params_sha256": descriptor.params_sha256,
            "adapter_identity_hash": descriptor.adapter_identity_hash,
            "adapter_implementation_hash": (
                descriptor.adapter_implementation_hash
            ),
            "driver_source_path": descriptor.driver_source_path,
            "driver_source_sha256": descriptor.driver_source_sha256,
            "source_commit": descriptor.source_commit,
        }
    )


_HEX_CHARS = frozenset("0123456789abcdef")


def _require_sha64(value: Any, name: str, ctx: str) -> str:
    """STRICT 64-lowercase-hex validation (§九)."""
    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeBundleError(
            RUNTIME_BUNDLE_STUDENT_SELECTION_BAD,
            f"{ctx}: {name} must be a 64-hex hash, got {value!r}",
        )
    if any(c not in _HEX_CHARS for c in value):
        raise RuntimeBundleError(
            RUNTIME_BUNDLE_STUDENT_SELECTION_BAD,
            f"{ctx}: {name} must be LOWERCASE hexadecimal "
            f"(0123456789abcdef), got {value!r}",
        )
    return value


def _require_non_empty_str(value: Any, name: str, ctx: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeBundleError(
            RUNTIME_BUNDLE_STUDENT_SELECTION_BAD,
            f"{ctx}: student_selection.{name} must be a non-empty str, "
            f"got {value!r}",
        )
    return value.strip()


def parse_student_selection(
    mapping: Any, ctx: str
) -> StudentSelectionDescriptor:
    """Parse + verify the student_selection block fail-closed."""
    if not isinstance(mapping, Mapping):
        raise RuntimeBundleError(
            RUNTIME_BUNDLE_STUDENT_SELECTION_MISSING,
            f"{ctx}: the runtime bundle manifest carries no "
            "student_selection block; a director-issued bundle MUST "
            "select exactly one allowed Student (never defaulted)",
        )
    unknown = sorted(k for k in mapping if k not in _STUDENT_SELECTION_FIELDS)
    if unknown:
        raise RuntimeBundleError(
            RUNTIME_BUNDLE_UNKNOWN_FIELD,
            f"{ctx}: unknown student_selection field(s) {unknown}",
        )
    selected = _require_non_empty_str(
        mapping.get("selected_candidate_id"),
        "selected_candidate_id",
        ctx,
    )
    if selected not in ALLOWED_STUDENT_CANDIDATE_IDS:
        raise RuntimeBundleError(
            RUNTIME_BUNDLE_STUDENT_SELECTION_BAD,
            f"{ctx}: selected_candidate_id {selected!r} is not in the "
            f"director-frozen allowed set "
            f"{sorted(ALLOWED_STUDENT_CANDIDATE_IDS)}",
        )
    expected_profile = STUDENT_PROFILE_BY_CANDIDATE[selected]
    profile_id = _require_non_empty_str(
        mapping.get("profile_id"), "profile_id", ctx
    )
    if profile_id != expected_profile:
        raise RuntimeBundleError(
            RUNTIME_BUNDLE_STUDENT_IDENTITY_MISMATCH,
            f"{ctx}: profile_id {profile_id!r} != the frozen profile "
            f"{expected_profile!r} for {selected!r}",
        )
    architecture_family = _require_non_empty_str(
        mapping.get("architecture_family"), "architecture_family", ctx
    )
    if architecture_family.upper() != "RMT16":
        raise RuntimeBundleError(
            RUNTIME_BUNDLE_STUDENT_SELECTION_BAD,
            f"{ctx}: architecture_family must be RMT16, got "
            f"{architecture_family!r}",
        )
    expected_memory = STUDENT_MEMORY_MODE_BY_CANDIDATE[selected]
    memory_mode = _require_non_empty_str(
        mapping.get("memory_mode"), "memory_mode", ctx
    )
    if memory_mode != expected_memory:
        raise RuntimeBundleError(
            RUNTIME_BUNDLE_STUDENT_IDENTITY_MISMATCH,
            f"{ctx}: memory_mode {memory_mode!r} != the frozen mode "
            f"{expected_memory!r} for {selected!r}",
        )
    expected_carry = STUDENT_CARRY_MODE_BY_CANDIDATE[selected]
    carry_mode = _require_non_empty_str(
        mapping.get("carry_mode"), "carry_mode", ctx
    )
    if carry_mode != expected_carry:
        raise RuntimeBundleError(
            RUNTIME_BUNDLE_STUDENT_IDENTITY_MISMATCH,
            f"{ctx}: carry_mode {carry_mode!r} != the frozen mode "
            f"{expected_carry!r} for {selected!r}",
        )
    hashes = {
        name: _require_sha64(mapping.get(name), name, ctx)
        for name in _STUDENT_HASH_FIELDS
    }
    descriptor = StudentSelectionDescriptor(
        selected_candidate_id=selected,
        profile_id=profile_id,
        architecture_family=_require_non_empty_str(
            mapping.get("architecture_family"),
            "architecture_family",
            ctx,
        ),
        memory_mode=memory_mode,
        memory_spec_hash=hashes["memory_spec_hash"],
        carry_mode=carry_mode,
        checkpoint_path=_require_non_empty_str(
            mapping.get("checkpoint_path"), "checkpoint_path", ctx
        ),
        checkpoint_file_sha256=hashes["checkpoint_file_sha256"],
        params_sha256=hashes["params_sha256"],
        adapter_identity_hash=hashes["adapter_identity_hash"],
        adapter_implementation_hash=hashes["adapter_implementation_hash"],
        driver_source_path=_require_non_empty_str(
            mapping.get("driver_source_path"),
            "driver_source_path",
            ctx,
        ),
        driver_source_sha256=hashes["driver_source_sha256"],
        source_commit=_require_non_empty_str(
            mapping.get("source_commit"), "source_commit", ctx
        ),
        descriptor_hash="",
    )
    descriptor = StudentSelectionDescriptor(
        **{
            **{name: getattr(descriptor, name) for name in (
                "selected_candidate_id",
                "profile_id",
                "architecture_family",
                "memory_mode",
                "memory_spec_hash",
                "carry_mode",
                "checkpoint_path",
                "checkpoint_file_sha256",
                "params_sha256",
                "adapter_identity_hash",
                "adapter_implementation_hash",
                "driver_source_path",
                "driver_source_sha256",
                "source_commit",
            )},
            "descriptor_hash": compute_student_selection_hash(descriptor),
        }
    )
    return descriptor


def _synthetic_test_only_student_selection() -> StudentSelectionDescriptor:
    """TEST_ONLY / SYNTHETIC default selection for TEST_ONLY bundles
    (conspicuously marked; never admissible on a production path)."""
    # deterministic per-process synthetic selection (explicitly TEST_ONLY)
    candidate = sorted(ALLOWED_STUDENT_CANDIDATE_IDS)[0]
    block = {
        "selected_candidate_id": candidate,
        "profile_id": STUDENT_PROFILE_BY_CANDIDATE[candidate],
        "architecture_family": "rmt16",
        "memory_mode": STUDENT_MEMORY_MODE_BY_CANDIDATE[candidate],
        "memory_spec_hash": "d0" * 32,
        "carry_mode": STUDENT_CARRY_MODE_BY_CANDIDATE[candidate],
        "checkpoint_path": "TEST_ONLY_SYNTHETIC_CHECKPOINT_PATH",
        "checkpoint_file_sha256": "d1" * 32,
        "params_sha256": "d2" * 32,
        "adapter_identity_hash": "d3" * 32,
        "adapter_implementation_hash": "d4" * 32,
        "driver_source_path": "TEST_ONLY_SYNTHETIC_DRIVER_PATH",
        "driver_source_sha256": "d5" * 32,
        "source_commit": "TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
    }
    descriptor = parse_student_selection(
        block, "runtime_bundle.test_only.student_selection"
    )
    return descriptor


class RuntimeBundleError(E1SchemaError):
    """Fail-closed bundle violation; ``code`` is greppable."""


def _require_non_empty_str(value: Any, name: str, ctx: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeBundleError(
            RUNTIME_BUNDLE_BAD_TYPE,
            f"{ctx}: {name} must be a non-empty str, got {value!r}",
        )
    return value.strip()


@dataclass(frozen=True)
class E1RuntimeBundle:
    """The signed carrier of every shared runtime object E1 consumes.

    ``capabilities`` maps each contract name to its REAL object; the
    bundle is useless (fail-closed) if any of them is missing, None or
    a mere string. ``object_identity_hashes`` binds each object's
    canonical identity hash at signing time — consumers re-derive and
    compare before trusting anything.
    """

    bundle_id: str
    mode: str  # PRODUCTION | TEST_ONLY
    source_commit: str
    signer_id: str
    authorization_grant_hash: str
    capabilities: Tuple[Tuple[str, Any], ...]  # RUNTIME_CAPABILITY_CONTRACTS
    object_identity_hashes: Tuple[Tuple[str, str], ...]
    bundle_hash: str
    #: CC2-Repair-3 (§一): director-signature + registry identity fields
    #: (empty on TEST_ONLY; the PRODUCTION trust lives in the injected
    #: DirectorBundleVerifier, never a static signer tuple)
    signature_ref: str
    registry_identity: str
    registry_hash: str
    #: CC2-Student repair: the EXPLICIT director-issued Student
    #: selection (never read from a nonexistent bundle.student)
    student_selection: StudentSelectionDescriptor

    def capability(self, contract: str) -> Any:
        for name, obj in self.capabilities:
            if name == contract:
                return obj
        raise RuntimeBundleError(
            RUNTIME_BUNDLE_UNBOUND,
            f"runtime bundle {self.bundle_id!r} carries no capability "
            f"{contract!r}",
        )

    def object_identity_hash(self, contract: str) -> str:
        for name, digest in self.object_identity_hashes:
            if name == contract:
                return digest
        return ""

    # convenience aliases consumed by the driver / seam ---------------
    @property
    def student_identity(self) -> Any:
        return self.capability("student_identity")

    @property
    def reference_identity(self) -> Any:
        return self.capability("reference_identity")

    @property
    def student_adapter(self) -> Any:
        return self.capability("student_adapter")

    @property
    def reference_adapter(self) -> Any:
        return self.capability("reference_adapter")

    @property
    def anchor_manifest(self) -> Any:
        return self.capability("anchor_manifest")

    @property
    def formal_asset_registry(self) -> Any:
        return self.capability("formal_asset_registry")

    @property
    def probe_runner(self) -> Any:
        return self.capability("probe_runner")

    @property
    def training(self) -> Any:
        return self.capability("training")

    @property
    def full_state_checkpoint(self) -> Any:
        return self.capability("full_state_checkpoint")

    @property
    def student_selection_mapping(self) -> Dict[str, str]:
        """The manifest-serializable student_selection block (no
        descriptor_hash — the hash is derived, never stored in the
        manifest)."""
        descriptor = self.student_selection
        return {
            "selected_candidate_id": descriptor.selected_candidate_id,
            "profile_id": descriptor.profile_id,
            "architecture_family": descriptor.architecture_family,
            "memory_mode": descriptor.memory_mode,
            "memory_spec_hash": descriptor.memory_spec_hash,
            "carry_mode": descriptor.carry_mode,
            "checkpoint_path": descriptor.checkpoint_path,
            "checkpoint_file_sha256": descriptor.checkpoint_file_sha256,
            "params_sha256": descriptor.params_sha256,
            "adapter_identity_hash": descriptor.adapter_identity_hash,
            "adapter_implementation_hash": (
                descriptor.adapter_implementation_hash
            ),
            "driver_source_path": descriptor.driver_source_path,
            "driver_source_sha256": descriptor.driver_source_sha256,
            "source_commit": descriptor.source_commit,
        }


def _capability_mapping(
    capabilities: Mapping[str, Any], ctx: str
) -> Dict[str, Any]:
    """Validate the capability set: EXACTLY the nine contracts, each a
    real object (never None, never a string placeholder)."""
    if not isinstance(capabilities, Mapping):
        raise RuntimeBundleError(
            RUNTIME_BUNDLE_BAD_TYPE,
            f"{ctx}: capabilities must be a mapping, got "
            f"{type(capabilities).__name__}",
        )
    unknown = sorted(
        k for k in capabilities if k not in RUNTIME_CAPABILITY_CONTRACTS
    )
    if unknown:
        raise RuntimeBundleError(
            RUNTIME_BUNDLE_UNKNOWN_FIELD,
            f"{ctx}: unknown capability contract(s) {unknown}",
        )
    resolved: Dict[str, Any] = {}
    for contract in RUNTIME_CAPABILITY_CONTRACTS:
        if contract not in capabilities:
            raise RuntimeBundleError(
                RUNTIME_BUNDLE_MISSING_FIELD,
                f"{ctx}: capability {contract!r} is missing from the "
                "bundle (no string placeholders, no silent gaps)",
            )
        obj = capabilities[contract]
        if obj is None:
            raise RuntimeBundleError(
                RUNTIME_BUNDLE_UNBOUND,
                f"{ctx}: capability {contract!r} is None — a bundle "
                "carries REAL objects only",
            )
        if isinstance(obj, str):
            raise RuntimeBundleError(
                RUNTIME_BUNDLE_STRING_PLACEHOLDER,
                f"{ctx}: capability {contract!r} is a bare string "
                f"({obj!r}); a string contract name is NOT an object "
                "resolution",
            )
        resolved[contract] = obj
    return resolved


def compute_bundle_hash(
    *,
    bundle_id: str,
    mode: str,
    source_commit: str,
    signer_id: str,
    authorization_grant_hash: str,
    object_identity_hashes: Mapping[str, str],
    student_selection_hash: str = "",
    signature_ref: str = "",
    registry_identity: str = "",
    registry_hash: str = "",
) -> str:
    """The canonical identity of one bundle (tamper-evident).

    CC2-Student repair: ``student_selection_hash`` is part of the
    bundle identity. CC2-Repair-3: the director signature reference and
    the registry identity/hash are also part of the identity.
    """
    return canonical_sha256(
        {
            "bundle_id": bundle_id,
            "mode": mode,
            "source_commit": source_commit,
            "signer_id": signer_id,
            "authorization_grant_hash": authorization_grant_hash,
            "object_identity_hashes": [
                [contract, object_identity_hashes[contract]]
                for contract in RUNTIME_CAPABILITY_CONTRACTS
            ],
            "student_selection_hash": student_selection_hash,
            "signature_ref": signature_ref,
            "registry_identity": registry_identity,
            "registry_hash": registry_hash,
        }
    )


def object_identity_hash(obj: Any) -> str:
    """Canonical identity hash of one capability object.

    The object's own ``identity_hash`` / ``object_identity_hash``
    surface wins when present (the shared runtime signs its own
    identities); otherwise the canonical hash over its ``__dict__``
    (frozen dataclasses) or its mapping content. Never a guess.
    """
    for attr in ("object_identity_hash", "identity_hash"):
        value = getattr(obj, attr, None)
        if isinstance(value, str) and len(value) == 64:
            return value
    if isinstance(obj, Mapping):
        return canonical_sha256(dict(obj))
    if hasattr(obj, "__dict__"):
        return canonical_sha256(
            {"type": type(obj).__name__, "state": dict(vars(obj))}
        )
    return canonical_sha256({"type": type(obj).__name__, "repr": repr(obj)})


def _assemble(
    *,
    bundle_id: str,
    mode: str,
    source_commit: str,
    signer_id: str,
    authorization_grant_hash: str,
    capabilities: Mapping[str, Any],
    student_selection: Any,
    signature_ref: str = "",
    registry_identity: str = "",
    registry_hash: str = "",
    ctx: str,
) -> E1RuntimeBundle:
    resolved = _capability_mapping(capabilities, ctx)
    if isinstance(student_selection, StudentSelectionDescriptor):
        descriptor = student_selection
    else:
        descriptor = parse_student_selection(student_selection, ctx)
    identity_hashes = {
        contract: object_identity_hash(resolved[contract])
        for contract in RUNTIME_CAPABILITY_CONTRACTS
    }
    bundle_hash = compute_bundle_hash(
        bundle_id=bundle_id,
        mode=mode,
        source_commit=source_commit,
        signer_id=signer_id,
        authorization_grant_hash=authorization_grant_hash,
        object_identity_hashes=identity_hashes,
        student_selection_hash=descriptor.descriptor_hash,
        signature_ref=signature_ref,
        registry_identity=registry_identity,
        registry_hash=registry_hash,
    )
    return E1RuntimeBundle(
        bundle_id=bundle_id,
        mode=mode,
        source_commit=source_commit,
        signer_id=signer_id,
        authorization_grant_hash=authorization_grant_hash,
        capabilities=tuple(
            (contract, resolved[contract])
            for contract in RUNTIME_CAPABILITY_CONTRACTS
        ),
        object_identity_hashes=tuple(
            (contract, identity_hashes[contract])
            for contract in RUNTIME_CAPABILITY_CONTRACTS
        ),
        bundle_hash=bundle_hash,
        signature_ref=signature_ref,
        registry_identity=registry_identity,
        registry_hash=registry_hash,
        student_selection=descriptor,
    )


def build_test_only_runtime_bundle(
    *,
    source_commit: str,
    capabilities: Mapping[str, Any],
    bundle_id: str = "e1-test-only-runtime-bundle",
    authorization_grant_hash: str = "",
    student_selection: Any = None,
) -> E1RuntimeBundle:
    """Assemble the TEST_ONLY bundle from conspicuously-marked
    SYNTHETIC capability objects.

    TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION / NOT_SCIENTIFIC_EVIDENCE:
    this bundle proves code path and identity binding ONLY. It never
    authorizes a real LLM / EnvCoder / probe / update, never flips a
    REAL_* flag, and every production surface refuses it. Absent
    ``student_selection`` => the conspicuously-marked TEST_ONLY
    synthetic selection (never admissible on a production path).
    """
    ctx = "runtime_bundle.test_only"
    return _assemble(
        bundle_id=_require_non_empty_str(bundle_id, "bundle_id", ctx),
        mode=BUNDLE_MODE_TEST_ONLY,
        source_commit=_require_non_empty_str(
            source_commit, "source_commit", ctx
        ),
        signer_id=SYNTHETIC_TEST_ONLY_SIGNER,
        authorization_grant_hash=authorization_grant_hash,
        capabilities=capabilities,
        student_selection=(
            _synthetic_test_only_student_selection()
            if student_selection is None
            else student_selection
        ),
        ctx=ctx,
    )


_MANIFEST_FIELDS = frozenset(
    {
        "bundle_id",
        "mode",
        "source_commit",
        "signer_id",
        "authorization_grant_hash",
        "object_identity_hashes",
        "bundle_hash",
        "student_selection",
        "signature_ref",
        "registry_identity",
        "registry_hash",
    }
)


def load_verified_runtime_bundle(mapping: Any, ctx: str) -> E1RuntimeBundle:
    """Parse + verify a signed bundle MANIFEST fail-closed.

    A manifest is the SIGNED DESCRIPTION of a bundle (identity fields
    + per-contract object identity hashes + the bundle hash); the real
    objects themselves arrive from the shared runtime registry, never
    from the manifest. Verification order: shape -> hash -> signer.
    This function returns the manifest-level bundle record with EMPTY
    capability objects (``capabilities=()``); object binding is the
    seam's job (``shared_runtime_seam.resolve_contract_from_bundle``),
    which re-checks every identity hash against the real object.
    """
    if not isinstance(mapping, Mapping):
        raise RuntimeBundleError(
            RUNTIME_BUNDLE_BAD_TYPE,
            f"{ctx}: bundle manifest must be a mapping, got "
            f"{type(mapping).__name__}",
        )
    unknown = sorted(k for k in mapping if k not in _MANIFEST_FIELDS)
    if unknown:
        raise RuntimeBundleError(
            RUNTIME_BUNDLE_UNKNOWN_FIELD,
            f"{ctx}: unknown bundle manifest field(s) {unknown}",
        )
    bundle_id = _require_non_empty_str(
        mapping.get("bundle_id"), "bundle_id", ctx
    )
    mode = mapping.get("mode")
    if mode not in (BUNDLE_MODE_PRODUCTION, BUNDLE_MODE_TEST_ONLY):
        raise RuntimeBundleError(
            RUNTIME_BUNDLE_BAD_TYPE,
            f"{ctx}: mode must be one of {[BUNDLE_MODE_PRODUCTION, BUNDLE_MODE_TEST_ONLY]}, got {mode!r}",
        )
    source_commit = _require_non_empty_str(
        mapping.get("source_commit"), "source_commit", ctx
    )
    signer_id = _require_non_empty_str(
        mapping.get("signer_id"), "signer_id", ctx
    )
    grant_hash = mapping.get("authorization_grant_hash")
    if not isinstance(grant_hash, str):
        raise RuntimeBundleError(
            RUNTIME_BUNDLE_BAD_TYPE,
            f"{ctx}: authorization_grant_hash must be a str, got "
            f"{grant_hash!r}",
        )
    raw_hashes = mapping.get("object_identity_hashes")
    if not isinstance(raw_hashes, Mapping):
        raise RuntimeBundleError(
            RUNTIME_BUNDLE_MISSING_FIELD,
            f"{ctx}: object_identity_hashes must be a mapping over the "
            f"nine contracts {list(RUNTIME_CAPABILITY_CONTRACTS)}",
        )
    identity_hashes: Dict[str, str] = {}
    for contract in RUNTIME_CAPABILITY_CONTRACTS:
        digest = raw_hashes.get(contract)
        if not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeBundleError(
                RUNTIME_BUNDLE_MISSING_FIELD,
                f"{ctx}: object_identity_hashes[{contract!r}] must be a "
                f"64-hex identity hash, got {digest!r}",
            )
        identity_hashes[contract] = digest
    # ---- director signature + registry identity (CC2-Repair-3) --------
    signature_ref = mapping.get("signature_ref") or ""
    registry_identity = mapping.get("registry_identity") or ""
    registry_hash = mapping.get("registry_hash") or ""
    if not isinstance(signature_ref, str) or not isinstance(
        registry_identity, str
    ) or not isinstance(registry_hash, str):
        raise RuntimeBundleError(
            RUNTIME_BUNDLE_BAD_TYPE,
            f"{ctx}: signature_ref / registry_identity / registry_hash "
            "must be str",
        )
    if mode == BUNDLE_MODE_PRODUCTION and not signature_ref.strip():
        raise RuntimeBundleError(
            RUNTIME_BUNDLE_MISSING_FIELD,
            f"{ctx}: a PRODUCTION bundle must carry a signature_ref",
        )
    # ---- student_selection (REQUIRED; parsed + verified fail-closed) --
    descriptor = parse_student_selection(
        mapping.get("student_selection"), f"{ctx}.student_selection"
    )
    declared_hash = _require_non_empty_str(
        mapping.get("bundle_hash"), "bundle_hash", ctx
    )
    recomputed = compute_bundle_hash(
        bundle_id=bundle_id,
        mode=mode,
        source_commit=source_commit,
        signer_id=signer_id,
        authorization_grant_hash=grant_hash,
        object_identity_hashes=identity_hashes,
        student_selection_hash=descriptor.descriptor_hash,
        signature_ref=signature_ref,
        registry_identity=registry_identity,
        registry_hash=registry_hash,
    )
    if recomputed != declared_hash:
        raise RuntimeBundleError(
            RUNTIME_BUNDLE_HASH_MISMATCH,
            f"{ctx}: manifest bundle_hash {declared_hash} != recomputed "
            f"{recomputed} (tampered or stale manifest)",
        )
    # ---- signer gate (BEFORE anything else may trust the bundle) ----
    if mode == BUNDLE_MODE_PRODUCTION:
        if signer_id not in AUTHORIZED_BUNDLE_SIGNERS:
            raise RuntimeBundleError(
                RUNTIME_BUNDLE_SIGNER_UNAUTHORIZED,
                f"{ctx}: production bundle signer {signer_id!r} is not "
                "on the supervisor-owned whitelist (EMPTY this round); "
                "no production runtime bundle can verify yet",
            )
    else:  # TEST_ONLY manifest
        if signer_id != SYNTHETIC_TEST_ONLY_SIGNER:
            raise RuntimeBundleError(
                RUNTIME_BUNDLE_SIGNER_UNAUTHORIZED,
                f"{ctx}: TEST_ONLY bundles must be signed by "
                f"{SYNTHETIC_TEST_ONLY_SIGNER!r}, got {signer_id!r}",
            )
    return E1RuntimeBundle(
        bundle_id=bundle_id,
        mode=mode,
        source_commit=source_commit,
        signer_id=signer_id,
        authorization_grant_hash=grant_hash,
        capabilities=(),  # objects bind at the seam, never in the manifest
        object_identity_hashes=tuple(
            (contract, identity_hashes[contract])
            for contract in RUNTIME_CAPABILITY_CONTRACTS
        ),
        bundle_hash=declared_hash,
        signature_ref=signature_ref,
        registry_identity=registry_identity,
        registry_hash=registry_hash,
        student_selection=descriptor,
    )


def require_bundle_admissible_for_production(
    bundle: E1RuntimeBundle, ctx: str
) -> None:
    """Production surfaces refuse TEST_ONLY bundles outright.

    A TEST_ONLY attestation/object can NEVER pose as production
    evidence; the mode check is mechanical, not a comment.
    """
    if not isinstance(bundle, E1RuntimeBundle):
        raise RuntimeBundleError(
            RUNTIME_BUNDLE_BAD_TYPE,
            f"{ctx}: expected an E1RuntimeBundle, got "
            f"{type(bundle).__name__}",
        )
    if bundle.mode == BUNDLE_MODE_TEST_ONLY:
        raise RuntimeBundleError(
            RUNTIME_BUNDLE_TEST_ONLY_REJECTED,
            f"{ctx}: bundle {bundle.bundle_id!r} is TEST_ONLY "
            f"(signer {bundle.signer_id!r}); TEST_ONLY objects never "
            "enter a production path, never flip a REAL_* flag and "
            "never grant readiness",
        )
    if bundle.signer_id not in AUTHORIZED_BUNDLE_SIGNERS:
        raise RuntimeBundleError(
            RUNTIME_BUNDLE_SIGNER_UNAUTHORIZED,
            f"{ctx}: bundle signer {bundle.signer_id!r} is not on the "
            "supervisor-owned production whitelist",
        )


# ---------------------------------------------------------------------------
# CC2-Repair-3 (§一): director-injected bundle verifier (production trust)
# ---------------------------------------------------------------------------
from typing import Protocol, runtime_checkable  # noqa: E402


@runtime_checkable
class DirectorBundleVerifier(Protocol):
    """The director's trusted bundle verifier (real object only).

    The static EMPTY ``AUTHORIZED_BUNDLE_SIGNERS`` is NEVER the final
    production gate; trust lives in this injected verifier.
    """

    verifier_id: str
    verifier_identity_hash: str
    trusted_signer_registry_hash: str

    def verify_bundle(
        self,
        *,
        signer_id: str,
        payload_hash: str,
        signature_ref: str,
        source_commit: str,
        registry_identity: str,
    ) -> bool: ...


E1_PRODUCTION_BUNDLE_VERIFIER_UNBOUND = (
    "E1_PRODUCTION_BUNDLE_VERIFIER_UNBOUND"
)
E1_PRODUCTION_BUNDLE_SIGNATURE_REJECTED = (
    "E1_PRODUCTION_BUNDLE_SIGNATURE_REJECTED"
)
E1_PRODUCTION_BUNDLE_VERIFIER_BAD_TYPE = (
    "E1_PRODUCTION_BUNDLE_VERIFIER_BAD_TYPE"
)


class ProductionBundleVerificationError(RuntimeBundleError):
    """Fail-closed production bundle verification violation."""


def require_production_bundle_verifier(
    verifier: Any, ctx: str
) -> DirectorBundleVerifier:
    if verifier is None:
        raise ProductionBundleVerificationError(
            E1_PRODUCTION_BUNDLE_VERIFIER_UNBOUND,
            f"{ctx}: no director-injected DirectorBundleVerifier; "
            "E1_PRODUCTION_BUNDLE_VERIFIER_UNBOUND — the production "
            "bundle cannot be trusted",
        )
    if isinstance(verifier, str) or isinstance(verifier, Mapping):
        raise ProductionBundleVerificationError(
            E1_PRODUCTION_BUNDLE_VERIFIER_BAD_TYPE,
            f"{ctx}: the verifier must be a real object, never a "
            "string / Mapping",
        )
    if not isinstance(verifier, DirectorBundleVerifier):
        raise ProductionBundleVerificationError(
            E1_PRODUCTION_BUNDLE_VERIFIER_BAD_TYPE,
            f"{ctx}: the injected verifier does not implement the "
            "DirectorBundleVerifier Protocol",
        )
    if getattr(verifier, "test_only", False):
        raise ProductionBundleVerificationError(
            E1_PRODUCTION_BUNDLE_VERIFIER_BAD_TYPE,
            f"{ctx}: a TEST_ONLY verifier never enters a production "
            "verification",
        )
    return verifier


def verify_production_runtime_bundle(
    bundle: E1RuntimeBundle, verifier: Any, ctx: str
) -> None:
    """Verify a PRODUCTION bundle through the director-injected
    verifier (strictly True required) BEFORE any object resolution."""
    if bundle.mode != BUNDLE_MODE_PRODUCTION:
        raise ProductionBundleVerificationError(
            RUNTIME_BUNDLE_TEST_ONLY_REJECTED,
            f"{ctx}: only PRODUCTION bundles go through the production "
            "verifier",
        )
    require_production_bundle_verifier(verifier, ctx)
    if not bundle.signature_ref.strip():
        raise ProductionBundleVerificationError(
            RUNTIME_BUNDLE_MISSING_FIELD,
            f"{ctx}: a PRODUCTION bundle must carry a signature_ref",
        )
    result = verifier.verify_bundle(
        signer_id=bundle.signer_id,
        payload_hash=bundle.bundle_hash,
        signature_ref=bundle.signature_ref,
        source_commit=bundle.source_commit,
        registry_identity=bundle.registry_identity,
    )
    if result is not True:
        raise ProductionBundleVerificationError(
            E1_PRODUCTION_BUNDLE_SIGNATURE_REJECTED,
            f"{ctx}: the director verifier did not strictly return True "
            f"(got {result!r}); E1_PRODUCTION_BUNDLE_SIGNATURE_REJECTED",
        )
