"""CC2-Repair-2: resolve the REAL shared runtime objects (strict).

The runtime bundle MANIFEST declares an identity for EVERY required
runtime object. The real objects are resolved ONLY from the director's
injected ``FormalAssetRegistry`` (a real object implementing the
Protocol — never a dict / JSON mapping / string path / None / a local
un-issued registry).

Strict identity rule (§四): every REQUIRED object must have a DECLARED
identity in the manifest; missing declaration => fail closed
(``OBJ_RESOLUTION_DECLARED_IDENTITY_MISSING``), and a mismatch =>
``OBJ_RESOLUTION_IDENTITY_MISMATCH``. There is NO "declared empty =>
skip verification" path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

from .runtime_bundle import (
    E1RuntimeBundle,
    RUNTIME_CAPABILITY_CONTRACTS,
    object_identity_hash,
)

#: the unified REQUIRED runtime object set (§四, 方案A)
REQUIRED_RUNTIME_OBJECTS = (
    "student_init_contract",
    "student_identity",
    "student_adapter",
    "reference_identity",
    "reference_adapter",
    "anchor_manifest",
    "formal_asset_registry",
    "probe_runner",
    "canonical_dicode_one_update_runtime",
    "canonical_dicode_runstate_checkpoint",
    "authorized_six_role_llm_runtime",
    "auxiliary_compute_ledger",
)

#: objects NOT among the nine bundle capabilities (they need a
#: declaration in the manifest's ``declared_runtime_objects`` block)
EXTRA_RUNTIME_CONTRACTS = tuple(
    c
    for c in REQUIRED_RUNTIME_OBJECTS
    if c not in RUNTIME_CAPABILITY_CONTRACTS
)


@runtime_checkable
class FormalAssetRegistry(Protocol):
    """The director's shared asset registry (real object only)."""

    registry_identity: str
    registry_hash: str

    def resolve_asset(
        self, *, contract: str, expected_identity: str
    ) -> object: ...

    def verify_implementation(
        self,
        *,
        contract: str,
        obj: object,
        expected_implementation_hash: str,
    ) -> bool: ...


# fail-closed codes (greppable)
OBJ_RESOLUTION_BAD_TYPE = "OBJ_RESOLUTION_BAD_TYPE"
OBJ_RESOLUTION_MISSING = "OBJ_RESOLUTION_MISSING"
OBJ_RESOLUTION_STRING_PLACEHOLDER = "OBJ_RESOLUTION_STRING_PLACEHOLDER"
OBJ_RESOLUTION_MAPPING_IMPERSONATION = "OBJ_RESOLUTION_MAPPING_IMPERSONATION"
OBJ_RESOLUTION_NONE_AS_BOUND = "OBJ_RESOLUTION_NONE_AS_BOUND"
OBJ_RESOLUTION_IDENTITY_MISMATCH = "OBJ_RESOLUTION_IDENTITY_MISMATCH"
OBJ_RESOLUTION_DECLARED_IDENTITY_MISSING = (
    "OBJ_RESOLUTION_DECLARED_IDENTITY_MISSING"
)
OBJ_RESOLUTION_REGISTRY_UNBOUND = "FORMAL_ASSET_REGISTRY_UNBOUND"
OBJ_RESOLUTION_REGISTRY_NOT_PROTOCOL = "OBJ_RESOLUTION_REGISTRY_NOT_PROTOCOL"
OBJ_RESOLUTION_TEST_ONLY_REGISTRY = "OBJ_RESOLUTION_TEST_ONLY_REGISTRY"


class RuntimeObjectResolutionError(Exception):
    """Fail-closed resolution violation; ``code`` is greppable."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"[{code}] {message}")


def require_real_registry(formal_asset_registry: Any, ctx: str) -> Any:
    """The registry must be a REAL object implementing the Protocol."""
    if formal_asset_registry is None:
        raise RuntimeObjectResolutionError(
            OBJ_RESOLUTION_REGISTRY_UNBOUND,
            f"{ctx}: FORMAL_ASSET_REGISTRY_UNBOUND — no director-"
            "injected shared FormalAssetRegistry; the object-level "
            "check honestly BLOCKS",
        )
    if isinstance(formal_asset_registry, str):
        raise RuntimeObjectResolutionError(
            OBJ_RESOLUTION_STRING_PLACEHOLDER,
            f"{ctx}: the registry must be a real object, never a "
            "string path / JSON file",
        )
    if isinstance(formal_asset_registry, Mapping):
        raise RuntimeObjectResolutionError(
            OBJ_RESOLUTION_MAPPING_IMPERSONATION,
            f"{ctx}: a plain Mapping / JSON Mapping is never a "
            "FormalAssetRegistry (JSON cannot carry real Python "
            "runtime objects)",
        )
    if not isinstance(formal_asset_registry, FormalAssetRegistry):
        raise RuntimeObjectResolutionError(
            OBJ_RESOLUTION_REGISTRY_NOT_PROTOCOL,
            f"{ctx}: the injected registry does not implement the "
            "FormalAssetRegistry Protocol (registry_identity / "
            "registry_hash / resolve_asset / verify_implementation)",
        )
    return formal_asset_registry


def require_authorized_registry(
    formal_asset_registry: Any, ctx: str
) -> Any:
    """Production rejects synthetic / local un-issued registries."""
    if getattr(formal_asset_registry, "test_only", False):
        raise RuntimeObjectResolutionError(
            OBJ_RESOLUTION_TEST_ONLY_REGISTRY,
            f"{ctx}: a TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION / "
            "NOT_OBJECT_LEVEL_CHECK registry never passes an object-"
            "level check",
        )
    return formal_asset_registry


def _declared_identity(
    bundle: E1RuntimeBundle, contract: str, ctx: str
) -> str:
    """The manifest-declared identity for one REQUIRED object (strict)."""
    if contract in RUNTIME_CAPABILITY_CONTRACTS:
        declared = bundle.object_identity_hash(contract)
    else:
        extra = getattr(bundle, "declared_runtime_objects", None) or {}
        declared = extra.get(contract, "")
    if not isinstance(declared, str) or len(declared) != 64:
        raise RuntimeObjectResolutionError(
            OBJ_RESOLUTION_DECLARED_IDENTITY_MISSING,
            f"{ctx}: required object {contract!r} has NO declared "
            "identity in the manifest; a missing declaration fails "
            "closed (never skipped)",
        )
    return declared


@dataclass(frozen=True)
class RuntimeObjectResolution:
    contract: str
    bound: bool
    code: str
    object: Any
    identity_hash: str


def resolve_e1_runtime_objects(
    verified_manifest: Any,
    formal_asset_registry: Any,
    ctx: str,
) -> dict:
    """Resolve every REQUIRED object from the REAL registry, verifying
    each declared identity + implementation hash fail-closed."""
    if not isinstance(verified_manifest, E1RuntimeBundle):
        raise RuntimeObjectResolutionError(
            OBJ_RESOLUTION_BAD_TYPE,
            f"{ctx}: verified_manifest must be an E1RuntimeBundle, got "
            f"{type(verified_manifest).__name__}",
        )
    require_real_registry(formal_asset_registry, ctx)
    require_authorized_registry(formal_asset_registry, ctx)
    resolved = {}
    missing = []
    for contract in REQUIRED_RUNTIME_OBJECTS:
        try:
            declared = _declared_identity(
                verified_manifest, contract, ctx
            )
        except RuntimeObjectResolutionError as e:
            resolved[contract] = RuntimeObjectResolution(
                contract=contract, bound=False, code=e.code,
                object=None, identity_hash="")
            missing.append(contract)
            continue
        try:
            obj = formal_asset_registry.resolve_asset(
                contract=contract, expected_identity=declared
            )
        except Exception as e:
            resolved[contract] = RuntimeObjectResolution(
                contract=contract, bound=False,
                code=getattr(e, "code", OBJ_RESOLUTION_MISSING),
                object=None, identity_hash="")
            missing.append(contract)
            continue
        if obj is None:
            resolved[contract] = RuntimeObjectResolution(
                contract=contract, bound=False,
                code=OBJ_RESOLUTION_NONE_AS_BOUND,
                object=None, identity_hash="")
            missing.append(contract)
            continue
        if isinstance(obj, str):
            resolved[contract] = RuntimeObjectResolution(
                contract=contract, bound=False,
                code=OBJ_RESOLUTION_STRING_PLACEHOLDER,
                object=None, identity_hash="")
            missing.append(contract)
            continue
        if isinstance(obj, Mapping):
            resolved[contract] = RuntimeObjectResolution(
                contract=contract, bound=False,
                code=OBJ_RESOLUTION_MAPPING_IMPERSONATION,
                object=None, identity_hash="")
            missing.append(contract)
            continue
        actual = object_identity_hash(obj)
        if actual != declared:
            resolved[contract] = RuntimeObjectResolution(
                contract=contract, bound=False,
                code=OBJ_RESOLUTION_IDENTITY_MISMATCH,
                object=None, identity_hash=actual)
            missing.append(contract)
            continue
        resolved[contract] = RuntimeObjectResolution(
            contract=contract, bound=True, code="",
            object=obj, identity_hash=actual)
    return {
        "resolutions": resolved,
        "all_bound": not missing,
        "missing": sorted(missing),
        "ctx": ctx,
    }
