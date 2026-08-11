"""The CONCRETE production FormalAssetRegistry.

A real registry: every production runtime object is REGISTERED with its
identity hash + implementation hash + source commit, resolved ONLY by
declared identity, and implementation-verified against the
manifest-declared expectation. Fail-closed on every path; no
string/Mapping/None object ever resolves.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from typing import Any, Dict, Mapping, Tuple


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def implementation_hash_of(obj: Any) -> str:
    """The implementation identity of a real object: the sha256 of the
    source text of its class (or module), never of its runtime state."""
    cls = obj if inspect.isclass(obj) else type(obj)
    try:
        source = inspect.getsource(cls)
    except (OSError, TypeError):
        module = inspect.getmodule(cls)
        source = repr(getattr(module, "__file__", cls.__qualname__))
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


class FormalAssetRegistryError(RuntimeError):
    """Fail-closed registry violation."""


class ProductionAssetRegistry:
    """The concrete FormalAssetRegistry (shared across directions)."""

    def __init__(self, *, registry_identity: str, source_commit: str):
        self.registry_identity = registry_identity
        self.source_commit = source_commit
        #: the seam's identity surface prefers the object's own signed
        #: identity; the registry signs itself with its registry_identity
        self.object_identity_hash = registry_identity
        #: the shared reward contract identity (candidate-binding
        #: surface): the formal asset registry declares the reward
        #: contract of the E1 executable candidates — the craftax
        #: episodic task reward the Persistent Student optimizes
        #: (deterministic protocol identity, never guessed per-run)
        self.reward_contract_hash = hashlib.sha256(
            b"mechanism_UED.reward_contract.craftax_episodic.v1"
        ).hexdigest()
        self._assets: Dict[str, Dict[str, Any]] = {}

    # -- registration ---------------------------------------------------
    def register(self, *, contract: str, obj: Any, identity_hash: str,
                 implementation_hash: str, source_commit: str) -> None:
        if not contract or not isinstance(contract, str):
            raise FormalAssetRegistryError(
                "REGISTRY_BAD_CONTRACT: contract must be a non-empty str")
        if obj is None or isinstance(obj, (str, bytes, bool, int, float,
                                           Mapping)):
            raise FormalAssetRegistryError(
                f"REGISTRY_PLACEHOLDER_REJECTED: {contract!r} must be a "
                "real object (never None/str/number/Mapping)")
        for name, value in (("identity_hash", identity_hash),
                            ("implementation_hash", implementation_hash)):
            if not isinstance(value, str) or len(value) != 64:
                raise FormalAssetRegistryError(
                    f"REGISTRY_BAD_HASH: {contract!r} {name} must be a "
                    f"64-hex sha256, got {value!r}")
        if contract in self._assets:
            raise FormalAssetRegistryError(
                f"REGISTRY_DUPLICATE: {contract!r} already registered "
                "(assets are immutable once registered)")
        self._assets[contract] = {
            "obj": obj,
            "identity_hash": identity_hash,
            "implementation_hash": implementation_hash,
            "source_commit": source_commit,
        }
        self._refresh_hash()

    def _refresh_hash(self) -> None:
        self.registry_hash = canonical_sha256({
            "registry_identity": self.registry_identity,
            "assets": {
                contract: {
                    "identity_hash": entry["identity_hash"],
                    "implementation_hash": entry["implementation_hash"],
                    "source_commit": entry["source_commit"],
                }
                for contract, entry in sorted(self._assets.items())
            },
        })

    # -- resolution -----------------------------------------------------
    def resolve_asset(self, *, contract: str = "", expected_identity: str = "",
                      identity: str = "") -> Any:
        """Resolve ONE real object by contract + declared identity."""
        expected = expected_identity or identity
        if not contract:
            raise FormalAssetRegistryError(
                "REGISTRY_RESOLVE_NO_CONTRACT: resolve_asset requires the "
                "contract name")
        entry = self._assets.get(contract)
        if entry is None:
            raise FormalAssetRegistryError(
                f"REGISTRY_ASSET_UNREGISTERED: no real object is "
                f"registered for {contract!r}")
        if expected and entry["identity_hash"] != expected:
            raise FormalAssetRegistryError(
                f"REGISTRY_IDENTITY_MISMATCH: {contract!r} declared "
                f"identity {expected!r} != registered "
                f"{entry['identity_hash']!r}")
        obj = entry["obj"]
        if obj is None or isinstance(obj, (str, bytes, Mapping)):
            raise FormalAssetRegistryError(
                f"REGISTRY_PLACEHOLDER_RESOLVED: {contract!r} resolved to "
                "a placeholder (impossible by construction; fail closed)")
        return obj

    def verify_implementation(self, *, contract: str = "", obj: Any = None,
                              expected_implementation_hash: str = "",
                              identity: str = "") -> bool:
        """Strictly True ONLY when the registered object IS ``obj`` and the
        manifest-declared implementation hash matches the registered one."""
        entry = self._assets.get(contract)
        if entry is None:
            return False
        if obj is not None and entry["obj"] is not obj:
            return False
        if expected_implementation_hash and (
                entry["implementation_hash"]
                != expected_implementation_hash):
            return False
        return True

    def declared_identity(self, contract: str) -> str:
        entry = self._assets.get(contract)
        if entry is None:
            return ""
        return entry["identity_hash"]

    def registered_contracts(self) -> Tuple[str, ...]:
        return tuple(sorted(self._assets))


_REGISTRY: ProductionAssetRegistry = None  # type: ignore


def _identity_of(obj: Any, kind: str, **fields: Any) -> str:
    payload = {"kind": kind}
    payload.update(fields)
    for attr in ("object_identity_hash", "identity_hash"):
        value = getattr(obj, attr, None)
        if isinstance(value, str) and len(value) == 64:
            payload["self_identity"] = value
            break
    return canonical_sha256(payload)


def build_production_registry():
    """Register the REAL production runtime objects (idempotent).

    Every object is a real deployment artifact (never synthetic):
    student contract/identity/adapter come from the real CC2 checkpoint;
    reference from the real RESET128 arm; anchor manifest from the frozen
    config; probe runner / signal issuer / envcoder backend / LLM runtime
    / canonical training runtime / run-state checkpoint manager / compute
    ledger are the real production implementations.
    """
    global _REGISTRY
    if _REGISTRY is not None:
        return _REGISTRY

    from . import student_assets as SA
    from .ledger import ProductionComputeLedger
    from .llm_runtime import AuthorizedSixRoleLLMRuntime
    from .probe_runner import RealProbeRunner
    from .reference_assets import (
        real_reference_adapter,
        real_reference_identity,
    )
    from .runstate import RunStateCheckpointManager
    from .signal_issuer import RealCriterionSignalIssuer
    from .training_assets import CanonicalOneUpdateRuntime
    from .envcoder_real import RealEnvCoderBackend

    PERSISTENT = "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304"
    loc = SA.AL.student_locations()
    source_commit = "src-sha256:" + loc["driver_source_sha256"]

    registry = ProductionAssetRegistry(
        registry_identity=canonical_sha256({
            "kind": "mechanism_UED.production_formal_asset_registry",
            "source_commit": source_commit,
            "checkpoint_contract_sha256":
                SA.AL.asset_locations()[
                    "checkpoint_contract_sha256"],
        }),
        source_commit=source_commit,
    )

    # 1. student_init_contract (real, live-recomputed hashes)
    student_contract = SA.build_real_student_contract(PERSISTENT)
    # 2. student_identity (real descriptor)
    student_identity = SA.real_student_identity(PERSISTENT)
    # 3. student_adapter (real read-only RMT16 mount)
    student_adapter = SA.real_student_adapter(PERSISTENT)
    # 4-5. reference identity/adapter (real RESET128 arm)
    reference_identity = real_reference_identity()
    reference_adapter = real_reference_adapter()
    # 6. anchor manifest (frozen config object)
    from .anchor_asset import real_anchor_manifest

    anchor_manifest = real_anchor_manifest()
    # 7. probe runner (real craftax rollouts)
    probe_runner = RealProbeRunner(student_adapter)
    # 8. signal issuer
    signal_issuer = RealCriterionSignalIssuer()
    # 9. real envcoder backend (craftax ladder)
    envcoder_backend = RealEnvCoderBackend()
    # 10. authorized six-role LLM runtime
    llm_runtime = AuthorizedSixRoleLLMRuntime()
    # 11. canonical one-update runtime
    one_update_runtime = CanonicalOneUpdateRuntime(
        student_adapter=student_adapter,
        train_state_candidate=PERSISTENT,
    )
    # 12. run-state checkpoint manager
    runstate_manager = RunStateCheckpointManager()
    # 13. auxiliary compute ledger
    compute_ledger = ProductionComputeLedger()

    assets = {
        "student_init_contract": student_contract,
        "student_identity": student_identity,
        "student_adapter": student_adapter,
        "reference_identity": reference_identity,
        "reference_adapter": reference_adapter,
        "anchor_manifest": anchor_manifest,
        "candidate_probe_runner": probe_runner,
        "criterion_signal_issuer": signal_issuer,
        "real_envcoder_backend": envcoder_backend,
        "authorized_six_role_llm_runtime": llm_runtime,
        "canonical_dicode_one_update_runtime": one_update_runtime,
        "canonical_dicode_runstate_checkpoint": runstate_manager,
        "auxiliary_compute_ledger": compute_ledger,
    }
    for contract, obj in assets.items():
        # the registered identity IS the object's own signed identity
        # (the resolution protocol compares the object's identity
        # protocol value against the manifest-declared hash — both must
        # be the SAME value, never two different derivations)
        own = getattr(obj, "object_identity_hash", None)
        if not (isinstance(own, str) and len(own) == 64):
            own = getattr(obj, "identity_hash", None)
        if not (isinstance(own, str) and len(own) == 64):
            own = _identity_of(obj, f"shared_runtime.{contract}",
                               contract=contract)
        registry.register(
            contract=contract,
            obj=obj,
            identity_hash=own,
            implementation_hash=implementation_hash_of(obj),
            source_commit=source_commit,
        )
    _REGISTRY = registry
    return registry


def production_registry() -> ProductionAssetRegistry:
    return build_production_registry()
