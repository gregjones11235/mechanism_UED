"""CC2-Student repair: resolve the REAL shared runtime objects.

The runtime bundle MANIFEST only describes object identities. The real
objects are resolved by the director's shared FormalAssetRegistry::

    resolution = resolve_e1_runtime_objects(
        verified_manifest, formal_asset_registry, ctx)

Every resolved object must satisfy::

    actual_object_identity_hash(obj) == manifest.object_identity_hashes[contract]

Forbidden:
* a string object name impersonating a real object;
* a plain Mapping impersonating an issued object;
* None marked as BOUND;
* E1 constructing a second loader / checkpoint codec / trainer.

Returns a per-contract report {contract: {bound, code, object}} plus an
all_bound flag. This round the shared registry is absent, so every
object resolves honestly UNBOUND.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .runtime_bundle import (
    E1RuntimeBundle,
    RUNTIME_CAPABILITY_CONTRACTS,
    object_identity_hash,
)

#: contracts resolved from the shared FormalAssetRegistry in addition
#: to the nine bundle capabilities
EXTRA_RUNTIME_CONTRACTS = (
    "student_init_contract",
    "canonical_dicode_one_update_runtime",
    "canonical_dicode_runstate_checkpoint",
)

# fail-closed codes (greppable)
OBJ_RESOLUTION_BAD_TYPE = "OBJ_RESOLUTION_BAD_TYPE"
OBJ_RESOLUTION_MISSING = "OBJ_RESOLUTION_MISSING"
OBJ_RESOLUTION_STRING_PLACEHOLDER = "OBJ_RESOLUTION_STRING_PLACEHOLDER"
OBJ_RESOLUTION_MAPPING_IMPERSONATION = "OBJ_RESOLUTION_MAPPING_IMPERSONATION"
OBJ_RESOLUTION_NONE_AS_BOUND = "OBJ_RESOLUTION_NONE_AS_BOUND"
OBJ_RESOLUTION_IDENTITY_MISMATCH = "OBJ_RESOLUTION_IDENTITY_MISMATCH"


class RuntimeObjectResolutionError(Exception):
    """Fail-closed resolution violation; ``code`` is greppable."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"[{code}] {message}")


def _require_real_object(obj: Any, contract: str, ctx: str) -> Any:
    if obj is None:
        raise RuntimeObjectResolutionError(
            OBJ_RESOLUTION_NONE_AS_BOUND,
            f"{ctx}: {contract!r} is None — None is never marked BOUND",
        )
    if isinstance(obj, str):
        raise RuntimeObjectResolutionError(
            OBJ_RESOLUTION_STRING_PLACEHOLDER,
            f"{ctx}: {contract!r} resolved to the bare string {obj!r}; "
            "a string object name never stands in for a real object",
        )
    if isinstance(obj, Mapping):
        raise RuntimeObjectResolutionError(
            OBJ_RESOLUTION_MAPPING_IMPERSONATION,
            f"{ctx}: {contract!r} resolved to a plain Mapping; an "
            "issued shared object is never a dict",
        )
    return obj


@dataclass(frozen=True)
class RuntimeObjectResolution:
    """One contract's resolution state (real object or honest unbound)."""

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
    """Resolve every required runtime object from the shared registry,
    verifying each identity hash against the manifest fail-closed.

    ``formal_asset_registry`` is the director's shared registry (a
    Mapping contract -> object); absent/None => every contract honestly
    UNBOUND (this round).
    """
    if not isinstance(verified_manifest, E1RuntimeBundle):
        raise RuntimeObjectResolutionError(
            OBJ_RESOLUTION_BAD_TYPE,
            f"{ctx}: verified_manifest must be an E1RuntimeBundle, got "
            f"{type(verified_manifest).__name__}",
        )
    if formal_asset_registry is None:
        formal_asset_registry = {}
    if not isinstance(formal_asset_registry, Mapping):
        raise RuntimeObjectResolutionError(
            OBJ_RESOLUTION_BAD_TYPE,
            f"{ctx}: formal_asset_registry must be a Mapping (the "
            "director's shared registry), got "
            f"{type(formal_asset_registry).__name__}",
        )
    all_contracts = list(RUNTIME_CAPABILITY_CONTRACTS) + list(
        EXTRA_RUNTIME_CONTRACTS
    )
    resolved = {}
    missing = []
    for contract in all_contracts:
        if contract not in formal_asset_registry:
            resolved[contract] = RuntimeObjectResolution(
                contract=contract,
                bound=False,
                code=OBJ_RESOLUTION_MISSING,
                object=None,
                identity_hash="",
            )
            missing.append(contract)
            continue
        obj = formal_asset_registry[contract]
        try:
            obj = _require_real_object(obj, contract, ctx)
        except RuntimeObjectResolutionError as e:
            resolved[contract] = RuntimeObjectResolution(
                contract=contract,
                bound=False,
                code=e.code,
                object=None,
                identity_hash="",
            )
            missing.append(contract)
            continue
        actual = object_identity_hash(obj)
        declared = verified_manifest.object_identity_hash(contract)
        if declared and actual != declared:
            resolved[contract] = RuntimeObjectResolution(
                contract=contract,
                bound=False,
                code=OBJ_RESOLUTION_IDENTITY_MISMATCH,
                object=None,
                identity_hash=actual,
            )
            missing.append(contract)
            continue
        resolved[contract] = RuntimeObjectResolution(
            contract=contract,
            bound=True,
            code="",
            object=obj,
            identity_hash=actual,
        )
    return {
        "resolutions": resolved,
        "all_bound": not missing,
        "missing": sorted(missing),
        "ctx": ctx,
    }
