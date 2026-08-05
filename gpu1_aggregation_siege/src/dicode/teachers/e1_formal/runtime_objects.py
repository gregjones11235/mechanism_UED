"""CC2-Repair-4 (§一/§二): the unified Runtime Object Manifest schema.

The Runtime Bundle Manifest must declare EVERY production runtime
object through ONE schema — no dual-name sets, no alias fallbacks::

    runtime_objects: {
        <contract>: {
            identity_hash, implementation_hash, source_commit,
            registry_identity
        }
    }

Every declaration participates in the bundle hash. Rules:

* the exact key set MUST equal ``REQUIRED_RUNTIME_OBJECTS`` — a missing
  object is rejected, an unknown object is rejected;
* identity/implementation hashes are strict 64-lowercase-hex;
* ``source_commit`` is either a 40-lowercase-hex git SHA or an
  explicit ``src-sha256:<64hex>`` form;
* ``registry_identity`` must be a non-empty str;
* the descriptor hash participates in the bundle hash.

The FormalAssetRegistry itself is verified from the manifest's
top-level registry_identity/registry_hash — it never resolves itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from .canonical import canonical_sha256
from .schemas import E1SchemaError

#: the SINGLE unified production runtime object set (no dual names)
REQUIRED_RUNTIME_OBJECTS = (
    "student_init_contract",
    "student_identity",
    "student_adapter",
    "reference_identity",
    "reference_adapter",
    "anchor_manifest",
    "candidate_probe_runner",
    "criterion_signal_issuer",
    "real_envcoder_backend",
    "authorized_six_role_llm_runtime",
    "canonical_dicode_one_update_runtime",
    "canonical_dicode_runstate_checkpoint",
    "auxiliary_compute_ledger",
)

#: required methods per contract (ABI; checked at resolution time)
REQUIRED_METHODS: Mapping[str, Tuple[str, ...]] = {
    "student_init_contract": (),
    "student_identity": (),
    "student_adapter": ("load_read_only",),
    "reference_identity": (),
    "reference_adapter": (),
    "anchor_manifest": (),
    "candidate_probe_runner": ("run_probes",),
    "criterion_signal_issuer": ("issue_signals",),
    "real_envcoder_backend": ("validate",),
    "authorized_six_role_llm_runtime": ("make_client",),
    "canonical_dicode_one_update_runtime": ("execute_one_update",),
    "canonical_dicode_runstate_checkpoint": ("save", "restore"),
    "auxiliary_compute_ledger": ("record",),
}

#: capability-contract ALIASES kept for compatibility with the legacy
#: nine-capability surface (the canonical names above are authoritative;
#: the aliases map ONLY into the canonical set, never a second set)
CAPABILITY_ALIAS_TO_CANONICAL: Mapping[str, str] = {
    "student_identity": "student_identity",
    "reference_identity": "reference_identity",
    "student_adapter": "student_adapter",
    "reference_adapter": "reference_adapter",
    "anchor_manifest": "anchor_manifest",
    "formal_asset_registry": "formal_asset_registry",
    "probe_runner": "candidate_probe_runner",
    "training": "canonical_dicode_one_update_runtime",
    "full_state_checkpoint": "canonical_dicode_runstate_checkpoint",
}

# fail-closed codes (greppable)
RUNTIME_OBJECTS_BAD_TYPE = "RUNTIME_OBJECTS_BAD_TYPE"
RUNTIME_OBJECTS_MISSING = "RUNTIME_OBJECTS_MISSING"
RUNTIME_OBJECTS_UNKNOWN = "RUNTIME_OBJECTS_UNKNOWN"
RUNTIME_OBJECT_HASH_BAD = "RUNTIME_OBJECT_HASH_BAD"
RUNTIME_OBJECT_SOURCE_COMMIT_BAD = "RUNTIME_OBJECT_SOURCE_COMMIT_BAD"

_HEX = frozenset("0123456789abcdef")


class RuntimeObjectError(E1SchemaError):
    """Fail-closed runtime-object violation; ``code`` is greppable."""


@dataclass(frozen=True)
class RuntimeObjectDescriptor:
    """One declared production runtime object (immutable, hash-bound)."""

    contract: str
    identity_hash: str
    implementation_hash: str
    source_commit: str
    registry_identity: str
    descriptor_hash: str


def _require_sha64_lower(value: Any, name: str, ctx: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeObjectError(
            RUNTIME_OBJECT_HASH_BAD,
            f"{ctx}: {name} must be a 64-hex hash, got {value!r}",
        )
    if any(c not in _HEX for c in value):
        raise RuntimeObjectError(
            RUNTIME_OBJECT_HASH_BAD,
            f"{ctx}: {name} must be LOWERCASE hexadecimal, got {value!r}",
        )
    return value


def _require_source_commit(value: Any, ctx: str) -> str:
    """A source commit is a 40-lowercase git SHA or src-sha256:<64hex>."""
    if not isinstance(value, str) or not value.strip():
        raise RuntimeObjectError(
            RUNTIME_OBJECT_SOURCE_COMMIT_BAD,
            f"{ctx}: source_commit must be a non-empty str, got {value!r}",
        )
    value = value.strip()
    if value.startswith("src-sha256:"):
        digest = value[len("src-sha256:"):]
        if len(digest) != 64 or any(c not in _HEX for c in digest):
            raise RuntimeObjectError(
                RUNTIME_OBJECT_SOURCE_COMMIT_BAD,
                f"{ctx}: src-sha256:<64hex> form is invalid: {value!r}",
            )
        return value
    if len(value) == 40 and all(c in _HEX for c in value):
        return value
    raise RuntimeObjectError(
        RUNTIME_OBJECT_SOURCE_COMMIT_BAD,
        f"{ctx}: source_commit must be a 40-lowercase git SHA or "
        f"src-sha256:<64hex>, got {value!r}",
    )


def compute_descriptor_hash(
    *,
    contract: str,
    identity_hash: str,
    implementation_hash: str,
    source_commit: str,
    registry_identity: str,
) -> str:
    return canonical_sha256(
        {
            "contract": contract,
            "identity_hash": identity_hash,
            "implementation_hash": implementation_hash,
            "source_commit": source_commit,
            "registry_identity": registry_identity,
        }
    )


def parse_runtime_objects(
    mapping: Any, ctx: str
) -> Dict[str, RuntimeObjectDescriptor]:
    """Parse the runtime_objects block fail-closed (exact key set)."""
    if not isinstance(mapping, Mapping):
        raise RuntimeObjectError(
            RUNTIME_OBJECTS_BAD_TYPE,
            f"{ctx}: runtime_objects must be a mapping, got "
            f"{type(mapping).__name__}",
        )
    missing = sorted(c for c in REQUIRED_RUNTIME_OBJECTS if c not in mapping)
    if missing:
        raise RuntimeObjectError(
            RUNTIME_OBJECTS_MISSING,
            f"{ctx}: missing runtime object declaration(s) {missing}",
        )
    unknown = sorted(k for k in mapping if k not in REQUIRED_RUNTIME_OBJECTS)
    if unknown:
        raise RuntimeObjectError(
            RUNTIME_OBJECTS_UNKNOWN,
            f"{ctx}: unknown runtime object declaration(s) {unknown}",
        )
    descriptors: Dict[str, RuntimeObjectDescriptor] = {}
    for contract in REQUIRED_RUNTIME_OBJECTS:
        block = mapping[contract]
        if not isinstance(block, Mapping):
            raise RuntimeObjectError(
                RUNTIME_OBJECTS_BAD_TYPE,
                f"{ctx}: runtime_objects[{contract!r}] must be a mapping",
            )
        identity_hash = _require_sha64_lower(
            block.get("identity_hash"),
            f"runtime_objects[{contract}].identity_hash",
            ctx,
        )
        implementation_hash = _require_sha64_lower(
            block.get("implementation_hash"),
            f"runtime_objects[{contract}].implementation_hash",
            ctx,
        )
        source_commit = _require_source_commit(
            block.get("source_commit"),
            f"{ctx}.runtime_objects[{contract}]",
        )
        registry_identity = block.get("registry_identity")
        if not isinstance(registry_identity, str) or not registry_identity.strip():
            raise RuntimeObjectError(
                RUNTIME_OBJECTS_BAD_TYPE,
                f"{ctx}: runtime_objects[{contract}].registry_identity "
                "must be a non-empty str",
            )
        descriptors[contract] = RuntimeObjectDescriptor(
            contract=contract,
            identity_hash=identity_hash,
            implementation_hash=implementation_hash,
            source_commit=source_commit,
            registry_identity=registry_identity.strip(),
            descriptor_hash=compute_descriptor_hash(
                contract=contract,
                identity_hash=identity_hash,
                implementation_hash=implementation_hash,
                source_commit=source_commit,
                registry_identity=registry_identity.strip(),
            ),
        )
    return descriptors


def compute_runtime_objects_hash(
    descriptors: Mapping[str, RuntimeObjectDescriptor]
) -> str:
    """The canonical hash of all declarations (part of bundle_hash)."""
    return canonical_sha256(
        {
            contract: descriptors[contract].descriptor_hash
            for contract in sorted(descriptors)
        }
    )
