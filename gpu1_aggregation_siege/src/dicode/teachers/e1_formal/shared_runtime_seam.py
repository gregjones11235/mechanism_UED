"""Shared-runtime seam (resolution ONLY, never invention).

CC2 follow-up P0-3: the seam no longer answers only WHETHER a shared
contract exists — its resolution now CARRIES the bound object::

    SharedRuntimeResolution[T](
        contract, bound, code, detail,
        object_ref,               # the REAL object (None when unbound)
        object_identity_hash,     # canonical identity of object_ref
        registry_bundle_hash,     # the signed bundle it came from
        provider_module_hash,     # provider identity (never a guess)
        capability_descriptor,    # canonical capability description
        attestation_hash,         # canonical hash of the whole record
    )

Resolution rules (all fail-closed, all mechanical):

* an ABSENT object resolves ``bound=False`` with its honest
  ``BLOCKED_WAITING_SHARED_RUNTIME_<CONTRACT>`` code — NEVER a string
  placeholder, NEVER a constructed substitute;
* a PRODUCTION ``object_ref`` can only come from the supervisor-signed
  ``E1RuntimeBundle`` (``resolve_contract_from_bundle``): identity
  hashes are re-derived from the real object and compared against the
  bundle's signed declaration BEFORE anything may consume it; a
  mismatch, a TEST_ONLY bundle, or an unsigned signer fails closed;
* TEST_ONLY bundles are admissible ONLY through the explicitly-marked
  ``allow_test_only=True`` surface (the TEST_ONLY closed loop), never
  through the default production path.

Legacy surface preserved unchanged: the eight canonical contracts,
their BLOCKED codes, the per-contract resolvers and
``resolve_all_shared_runtime()`` (this round: every contract unbound).

Discipline (hard constraints):
* the seam ONLY resolves and verifies; it NEVER constructs, mints,
  loads or disguises a shared object; a fixture identity can never
  pose as a real one through this surface;
* NO second checkpoint loader / Student registry / Reference registry /
  anchor manifest / formal asset registry / probe minting is created
  here or anywhere in E1.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Optional, Tuple, TypeVar

from .canonical import canonical_sha256
from .runtime_bundle import (
    BUNDLE_MODE_TEST_ONLY,
    E1RuntimeBundle,
    RUNTIME_CAPABILITY_CONTRACTS,
    RuntimeBundleError,
    object_identity_hash as bundle_object_identity_hash,
    require_bundle_admissible_for_production,
)
from .schemas import E1SchemaError

T = TypeVar("T")

#: base block code (the generic status) and per-contract codes
BLOCKED_WAITING_SHARED_RUNTIME = "BLOCKED_WAITING_SHARED_RUNTIME"
BLOCKED_WAITING_SHARED_RUNTIME_STUDENT_IDENTITY = (
    "BLOCKED_WAITING_SHARED_RUNTIME_STUDENT_IDENTITY"
)
BLOCKED_WAITING_SHARED_RUNTIME_STUDENT_ADAPTER = (
    "BLOCKED_WAITING_SHARED_RUNTIME_STUDENT_ADAPTER"
)
BLOCKED_WAITING_SHARED_RUNTIME_REFERENCE_IDENTITY = (
    "BLOCKED_WAITING_SHARED_RUNTIME_REFERENCE_IDENTITY"
)
BLOCKED_WAITING_SHARED_RUNTIME_REFERENCE_ADAPTER = (
    "BLOCKED_WAITING_SHARED_RUNTIME_REFERENCE_ADAPTER"
)
BLOCKED_WAITING_SHARED_RUNTIME_ANCHOR_MANIFEST = (
    "BLOCKED_WAITING_SHARED_RUNTIME_ANCHOR_MANIFEST"
)
BLOCKED_WAITING_SHARED_RUNTIME_FORMAL_ASSET_REGISTRY = (
    "BLOCKED_WAITING_SHARED_RUNTIME_FORMAL_ASSET_REGISTRY"
)
BLOCKED_WAITING_SHARED_RUNTIME_CANDIDATE_PROBE_RESULT = (
    "BLOCKED_WAITING_SHARED_RUNTIME_CANDIDATE_PROBE_RESULT"
)
BLOCKED_WAITING_SHARED_RUNTIME_FULL_STATE_CHECKPOINT = (
    "BLOCKED_WAITING_SHARED_RUNTIME_FULL_STATE_CHECKPOINT"
)
#: CC2 follow-up: the shared TRAINING runtime (exactly-one-update
#: surface) — resolved via the signed bundle surface only; the legacy
#: canonical eight stay untouched.
BLOCKED_WAITING_SHARED_RUNTIME_TRAINING_RUNTIME = (
    "BLOCKED_WAITING_SHARED_RUNTIME_TRAINING_RUNTIME"
)

#: the eight canonical shared contracts (legacy order — pinned)
SHARED_CONTRACTS = (
    "StudentIdentity",
    "StudentAdapter",
    "ReferenceIdentity",
    "ReferenceAdapter",
    "AnchorManifest",
    "FormalAssetRegistry",
    "CandidateProbeResult",
    "FullStateCheckpoint",
)

#: the future home of the shared runtime (CC4). The module does NOT
#: exist in this worktree; resolvers attempt the lazy import and
#: report honestly. E1 never vendors a substitute.
SHARED_RUNTIME_MODULE = "dicode.shared_runtime"

#: contract -> attribute name on the future shared runtime module
_CONTRACT_ATTRIBUTES = {
    "StudentIdentity": "student_identity",
    "StudentAdapter": "student_adapter",
    "ReferenceIdentity": "reference_identity",
    "ReferenceAdapter": "reference_adapter",
    "AnchorManifest": "anchor_manifest",
    "FormalAssetRegistry": "formal_asset_registry",
    "CandidateProbeResult": "candidate_probe_result",
    "FullStateCheckpoint": "full_state_checkpoint",
    "TrainingRuntime": "training_runtime",
}

_CONTRACT_CODES = {
    "StudentIdentity": BLOCKED_WAITING_SHARED_RUNTIME_STUDENT_IDENTITY,
    "StudentAdapter": BLOCKED_WAITING_SHARED_RUNTIME_STUDENT_ADAPTER,
    "ReferenceIdentity": BLOCKED_WAITING_SHARED_RUNTIME_REFERENCE_IDENTITY,
    "ReferenceAdapter": BLOCKED_WAITING_SHARED_RUNTIME_REFERENCE_ADAPTER,
    "AnchorManifest": BLOCKED_WAITING_SHARED_RUNTIME_ANCHOR_MANIFEST,
    "FormalAssetRegistry": BLOCKED_WAITING_SHARED_RUNTIME_FORMAL_ASSET_REGISTRY,
    "CandidateProbeResult": BLOCKED_WAITING_SHARED_RUNTIME_CANDIDATE_PROBE_RESULT,
    "FullStateCheckpoint": BLOCKED_WAITING_SHARED_RUNTIME_FULL_STATE_CHECKPOINT,
    "TrainingRuntime": BLOCKED_WAITING_SHARED_RUNTIME_TRAINING_RUNTIME,
}

#: CC2 follow-up P0-3: the bundle's nine capability contracts (snake
#: case) mapped onto seam contracts. ``probe_runner`` consumes the
#: CandidateProbeResult contract; ``training`` is the new shared
#: TrainingRuntime contract (bundle surface only).
BUNDLE_SEAM_CONTRACTS = {
    "student_identity": "StudentIdentity",
    "student_adapter": "StudentAdapter",
    "reference_identity": "ReferenceIdentity",
    "reference_adapter": "ReferenceAdapter",
    "anchor_manifest": "AnchorManifest",
    "formal_asset_registry": "FormalAssetRegistry",
    "probe_runner": "CandidateProbeResult",
    "training": "TrainingRuntime",
    "full_state_checkpoint": "FullStateCheckpoint",
}

# fail-closed seam codes (greppable)
SEAM_BAD_TYPE = "SEAM_BAD_TYPE"
SEAM_UNBOUND = "SEAM_UNBOUND"
SEAM_STRING_PLACEHOLDER = "SEAM_STRING_PLACEHOLDER"
SEAM_UNKNOWN_CONTRACT = "SEAM_UNKNOWN_CONTRACT"
SEAM_IDENTITY_MISMATCH = "SEAM_IDENTITY_MISMATCH"


class SeamError(E1SchemaError):
    """Fail-closed seam violation; ``code`` is greppable."""


@dataclass(frozen=True)
class SharedRuntimeResolution(Generic[T]):
    """One contract's resolution state — INCLUDING the bound object.

    ``object_ref`` is the REAL shared object when ``bound``; it is
    ``None`` whenever the contract is unbound (never a string
    placeholder, never a summary dict). ``object_identity_hash`` is
    the canonical identity of the bound object, re-derived and
    compared against the signed bundle declaration before any
    consumer may trust it. ``attestation_hash`` binds the whole
    record (tamper-evident).
    """

    contract: str
    bound: bool
    code: str  # "" when bound, else BLOCKED_WAITING_SHARED_RUNTIME_<CONTRACT>
    detail: str
    object_ref: Optional[T] = None
    object_identity_hash: str = ""
    registry_bundle_hash: str = ""
    provider_module_hash: str = ""
    capability_descriptor: str = ""
    attestation_hash: str = ""

    @property
    def contract_name(self) -> str:
        return self.contract

    def require_bound(self, ctx: str) -> "SharedRuntimeResolution[T]":
        if not self.bound or self.object_ref is None:
            raise SeamError(
                self.code or SEAM_UNBOUND,
                f"{ctx}: shared contract {self.contract!r} is unbound; "
                "the seam never substitutes a placeholder object",
            )
        return self


#: backward-compatible alias (pre-CC2 consumers keep their name)
SeamResolution = SharedRuntimeResolution


def _attestation_hash(
    *,
    contract: str,
    bound: bool,
    code: str,
    detail: str,
    object_identity_hash: str,
    registry_bundle_hash: str,
    provider_module_hash: str,
    capability_descriptor: str,
) -> str:
    return canonical_sha256(
        {
            "contract": contract,
            "bound": bound,
            "code": code,
            "detail": detail,
            "object_identity_hash": object_identity_hash,
            "registry_bundle_hash": registry_bundle_hash,
            "provider_module_hash": provider_module_hash,
            "capability_descriptor": capability_descriptor,
        }
    )


def _verify_object_surface(contract: str, obj: Any, ctx: str) -> None:
    """Mechanical sanity of one bound object (this round's checks).

    The shared runtime's full schema lands with CC4; until then the
    seam mechanically refuses every shape that could fake an object:
    None, bare strings, booleans and bare numbers. A real object is a
    mapping or a stateful object with a canonical identity.
    """
    if obj is None:
        raise SeamError(
            SEAM_UNBOUND,
            f"{ctx}: shared contract {contract!r} resolved to None — "
            "an absent object is bound=False, never a null object_ref",
        )
    if isinstance(obj, str):
        raise SeamError(
            SEAM_STRING_PLACEHOLDER,
            f"{ctx}: shared contract {contract!r} resolved to the bare "
            f"string {obj!r}; a contract name is NOT an object",
        )
    if isinstance(obj, (bool, int, float)):
        raise SeamError(
            SEAM_BAD_TYPE,
            f"{ctx}: shared contract {contract!r} resolved to a bare "
            f"number {obj!r}; that is not a shared runtime object",
        )
    if not hasattr(obj, "__dict__") and not isinstance(obj, dict):
        raise SeamError(
            SEAM_BAD_TYPE,
            f"{ctx}: shared contract {contract!r} resolved to "
            f"{type(obj).__name__} with no canonical state; the seam "
            "cannot derive an identity hash for it",
        )


def _resolve(contract: str) -> SharedRuntimeResolution:
    code = _CONTRACT_CODES[contract]
    attribute = _CONTRACT_ATTRIBUTES[contract]
    try:
        module = importlib.import_module(SHARED_RUNTIME_MODULE)
    except ImportError:
        return SharedRuntimeResolution(
            contract=contract,
            bound=False,
            code=code,
            detail=(
                f"shared runtime module {SHARED_RUNTIME_MODULE!r} is not "
                "importable in this worktree; waiting for the CC4 shared "
                "runtime (E1 never constructs a substitute)"
            ),
        )
    obj = getattr(module, attribute, None)
    if obj is None:
        return SharedRuntimeResolution(
            contract=contract,
            bound=False,
            code=code,
            detail=(
                f"shared runtime module {SHARED_RUNTIME_MODULE!r} exists "
                f"but provides no {attribute!r}; waiting for the CC4 "
                "shared runtime contract"
            ),
        )
    ctx = f"shared_runtime_seam.{contract}"
    try:
        _verify_object_surface(contract, obj, ctx)
    except SeamError as e:
        # a malformed shared surface is reported honestly, never bound
        return SharedRuntimeResolution(
            contract=contract,
            bound=False,
            code=e.code,
            detail=str(e),
        )
    digest = bundle_object_identity_hash(obj)
    provider_module_hash = canonical_sha256(
        {"provider_module": SHARED_RUNTIME_MODULE, "attribute": attribute}
    )
    capability_descriptor = canonical_sha256(
        {"contract": contract, "object_type": type(obj).__name__}
    )
    return SharedRuntimeResolution(
        contract=contract,
        bound=True,
        code="",
        detail=(
            f"{SHARED_RUNTIME_MODULE}.{attribute} is present; the real "
            "shared object is obtained from the shared runtime itself"
        ),
        object_ref=obj,
        object_identity_hash=digest,
        provider_module_hash=provider_module_hash,
        capability_descriptor=capability_descriptor,
        attestation_hash=_attestation_hash(
            contract=contract,
            bound=True,
            code="",
            detail="",
            object_identity_hash=digest,
            registry_bundle_hash="",
            provider_module_hash=provider_module_hash,
            capability_descriptor=capability_descriptor,
        ),
    )


def resolve_student_identity() -> SharedRuntimeResolution:
    return _resolve("StudentIdentity")


def resolve_student_adapter() -> SharedRuntimeResolution:
    return _resolve("StudentAdapter")


def resolve_reference_identity() -> SharedRuntimeResolution:
    return _resolve("ReferenceIdentity")


def resolve_reference_adapter() -> SharedRuntimeResolution:
    return _resolve("ReferenceAdapter")


def resolve_anchor_manifest() -> SharedRuntimeResolution:
    return _resolve("AnchorManifest")


def resolve_formal_asset_registry() -> SharedRuntimeResolution:
    return _resolve("FormalAssetRegistry")


def resolve_candidate_probe_result() -> SharedRuntimeResolution:
    return _resolve("CandidateProbeResult")


def resolve_full_state_checkpoint() -> SharedRuntimeResolution:
    return _resolve("FullStateCheckpoint")


_RESOLVERS: Tuple[Tuple[str, Callable[[], SharedRuntimeResolution]], ...] = (
    ("StudentIdentity", resolve_student_identity),
    ("StudentAdapter", resolve_student_adapter),
    ("ReferenceIdentity", resolve_reference_identity),
    ("ReferenceAdapter", resolve_reference_adapter),
    ("AnchorManifest", resolve_anchor_manifest),
    ("FormalAssetRegistry", resolve_formal_asset_registry),
    ("CandidateProbeResult", resolve_candidate_probe_result),
    ("FullStateCheckpoint", resolve_full_state_checkpoint),
)


def resolve_all_shared_runtime() -> Dict[str, SharedRuntimeResolution]:
    """Resolve the EIGHT canonical contracts (order preserved).

    Pure resolution: no construction, no minting, no checkpoint
    loading and no I/O beyond the lazy import attempt.
    """
    return {contract: resolver() for contract, resolver in _RESOLVERS}


# ---------------------------------------------------------------------------
# CC2 follow-up P0-3: bundle-bound resolution (objects + identity checks)
# ---------------------------------------------------------------------------
def resolve_contract_from_bundle(
    bundle: Any,
    bundle_contract: str,
    ctx: str,
    *,
    allow_test_only: bool = False,
) -> SharedRuntimeResolution:
    """Bind ONE shared contract's REAL object from a signed bundle.

    Mechanical verification, in order:

    1. ``bundle`` is an ``E1RuntimeBundle`` and ``bundle_contract`` is
       one of the nine capability contracts;
    2. PRODUCTION admissibility (signer whitelist + mode) — a TEST_ONLY
       bundle is refused unless the caller explicitly opens the
       conspicuously-marked ``allow_test_only`` surface;
    3. the capability object exists, is real (never None / string /
       bare number) and its RE-DERIVED canonical identity hash equals
       the bundle's signed declaration (tamper => fail closed).

    Returns the bound ``SharedRuntimeResolution`` carrying
    ``object_ref`` — the ONLY production path to a shared object.
    """
    if not isinstance(bundle, E1RuntimeBundle):
        raise SeamError(
            SEAM_BAD_TYPE,
            f"{ctx}: expected an E1RuntimeBundle, got "
            f"{type(bundle).__name__}; shared objects arrive ONLY via "
            "the signed bundle",
        )
    if bundle_contract not in RUNTIME_CAPABILITY_CONTRACTS:
        raise SeamError(
            SEAM_UNKNOWN_CONTRACT,
            f"{ctx}: {bundle_contract!r} is not a runtime capability "
            f"contract (known: {list(RUNTIME_CAPABILITY_CONTRACTS)})",
        )
    seam_contract = BUNDLE_SEAM_CONTRACTS[bundle_contract]
    if not allow_test_only:
        # propagates RUNTIME_BUNDLE_TEST_ONLY_REJECTED /
        # RUNTIME_BUNDLE_SIGNER_UNAUTHORIZED fail-closed
        require_bundle_admissible_for_production(
            bundle, f"{ctx}.{bundle_contract}"
        )
    elif bundle.mode != BUNDLE_MODE_TEST_ONLY:
        # the test-only surface must never smuggle a production bundle
        # into a TEST_ONLY flow (and vice versa) — mode discipline
        raise SeamError(
            SEAM_BAD_TYPE,
            f"{ctx}: allow_test_only surface received a "
            f"{bundle.mode!r} bundle; the two surfaces never mix",
        )
    obj = bundle.capability(bundle_contract)  # unbound => bundle error
    _verify_object_surface(seam_contract, obj, f"{ctx}.{bundle_contract}")
    declared = bundle.object_identity_hash(bundle_contract)
    derived = bundle_object_identity_hash(obj)
    if declared != derived:
        raise SeamError(
            SEAM_IDENTITY_MISMATCH,
            f"{ctx}: bundle {bundle.bundle_id!r} declares identity "
            f"{declared!r} for {bundle_contract!r} but the bound object "
            f"re-derives {derived!r}; refusing a tampered binding",
        )
    provider_module_hash = canonical_sha256(
        {
            "provider": "E1RuntimeBundle",
            "bundle_id": bundle.bundle_id,
            "mode": bundle.mode,
            "signer_id": bundle.signer_id,
            "source_commit": bundle.source_commit,
        }
    )
    capability_descriptor = canonical_sha256(
        {
            "contract": seam_contract,
            "bundle_contract": bundle_contract,
            "object_type": type(obj).__name__,
        }
    )
    detail = (
        f"{bundle_contract!r} bound from signed bundle "
        f"{bundle.bundle_id!r} (mode {bundle.mode}, signer "
        f"{bundle.signer_id}); identity hash re-derived and matched"
    )
    return SharedRuntimeResolution(
        contract=seam_contract,
        bound=True,
        code="",
        detail=detail,
        object_ref=obj,
        object_identity_hash=derived,
        registry_bundle_hash=bundle.bundle_hash,
        provider_module_hash=provider_module_hash,
        capability_descriptor=capability_descriptor,
        attestation_hash=_attestation_hash(
            contract=seam_contract,
            bound=True,
            code="",
            detail=detail,
            object_identity_hash=derived,
            registry_bundle_hash=bundle.bundle_hash,
            provider_module_hash=provider_module_hash,
            capability_descriptor=capability_descriptor,
        ),
    )


def resolve_all_from_bundle(
    bundle: Any,
    ctx: str,
    *,
    allow_test_only: bool = False,
) -> Dict[str, SharedRuntimeResolution]:
    """Bind ALL nine capability contracts from one signed bundle.

    Keyed by the bundle's snake-case capability contract names. ANY
    failure propagates fail-closed (no partial silent bindings).
    """
    return {
        bundle_contract: resolve_contract_from_bundle(
            bundle,
            bundle_contract,
            ctx,
            allow_test_only=allow_test_only,
        )
        for bundle_contract in RUNTIME_CAPABILITY_CONTRACTS
    }


def unbound_resolution(
    bundle_contract: str, ctx: str, detail: str
) -> SharedRuntimeResolution:
    """The honest UNBOUND record for one bundle contract (never a
    placeholder object; ``object_ref`` stays None)."""
    if bundle_contract not in BUNDLE_SEAM_CONTRACTS:
        raise SeamError(
            SEAM_UNKNOWN_CONTRACT,
            f"{ctx}: {bundle_contract!r} is not a bundle capability "
            "contract",
        )
    seam_contract = BUNDLE_SEAM_CONTRACTS[bundle_contract]
    code = _CONTRACT_CODES[seam_contract]
    return SharedRuntimeResolution(
        contract=seam_contract,
        bound=False,
        code=code,
        detail=detail,
    )


__all__ = [
    "BLOCKED_WAITING_SHARED_RUNTIME",
    "BLOCKED_WAITING_SHARED_RUNTIME_STUDENT_IDENTITY",
    "BLOCKED_WAITING_SHARED_RUNTIME_STUDENT_ADAPTER",
    "BLOCKED_WAITING_SHARED_RUNTIME_REFERENCE_IDENTITY",
    "BLOCKED_WAITING_SHARED_RUNTIME_REFERENCE_ADAPTER",
    "BLOCKED_WAITING_SHARED_RUNTIME_ANCHOR_MANIFEST",
    "BLOCKED_WAITING_SHARED_RUNTIME_FORMAL_ASSET_REGISTRY",
    "BLOCKED_WAITING_SHARED_RUNTIME_CANDIDATE_PROBE_RESULT",
    "BLOCKED_WAITING_SHARED_RUNTIME_FULL_STATE_CHECKPOINT",
    "BLOCKED_WAITING_SHARED_RUNTIME_TRAINING_RUNTIME",
    "SHARED_CONTRACTS",
    "SHARED_RUNTIME_MODULE",
    "BUNDLE_SEAM_CONTRACTS",
    "SEAM_BAD_TYPE",
    "SEAM_UNBOUND",
    "SEAM_STRING_PLACEHOLDER",
    "SEAM_UNKNOWN_CONTRACT",
    "SEAM_IDENTITY_MISMATCH",
    "SeamError",
    "SharedRuntimeResolution",
    "SeamResolution",
    "resolve_student_identity",
    "resolve_student_adapter",
    "resolve_reference_identity",
    "resolve_reference_adapter",
    "resolve_anchor_manifest",
    "resolve_formal_asset_registry",
    "resolve_candidate_probe_result",
    "resolve_full_state_checkpoint",
    "resolve_all_shared_runtime",
    "resolve_contract_from_bundle",
    "resolve_all_from_bundle",
    "unbound_resolution",
]
