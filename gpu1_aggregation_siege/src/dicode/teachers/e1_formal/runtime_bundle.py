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
) -> str:
    """The canonical identity of one bundle (tamper-evident)."""
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
    ctx: str,
) -> E1RuntimeBundle:
    resolved = _capability_mapping(capabilities, ctx)
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
    )


def build_test_only_runtime_bundle(
    *,
    source_commit: str,
    capabilities: Mapping[str, Any],
    bundle_id: str = "e1-test-only-runtime-bundle",
    authorization_grant_hash: str = "",
) -> E1RuntimeBundle:
    """Assemble the TEST_ONLY bundle from conspicuously-marked
    SYNTHETIC capability objects.

    TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION / NOT_SCIENTIFIC_EVIDENCE:
    this bundle proves code path and identity binding ONLY. It never
    authorizes a real LLM / EnvCoder / probe / update, never flips a
    REAL_* flag, and every production surface refuses it.
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
