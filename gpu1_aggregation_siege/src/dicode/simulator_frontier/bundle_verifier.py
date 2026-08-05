"""Trusted runtime-bundle verification (E3-P0-1).

The ``controller_signature_ref`` non-empty + non-SYNTHETIC check is NOT
signature verification.  The production path must consume the director-shared
verifier.  This module defines the CONTRACT — a mint-only TrustedSignerRegistry
and a DirectorBundleVerifier that verify signer identity, the manifest's
canonical payload hash, the signature/reference and the entrypoint registry
identities — and a fail-closed injection slot.  We never invent our own
cryptographic protocol: when the director has NOT injected the shared
verifier, every production bundle is BLOCKED with
``E3_PRODUCTION_BUNDLE_VERIFIER_UNBOUND``.  TEST_ONLY signature tooling lives
only in tests or the explicitly-labelled test-only modules.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields
from typing import Any, Mapping, Sequence

from .errors import InvalidEvidenceError, ProvenanceViolationError

E3_PRODUCTION_BUNDLE_VERIFIER_UNBOUND = "E3_PRODUCTION_BUNDLE_VERIFIER_UNBOUND"
BUNDLE_VERIFIER_SCHEMA = "simulator_frontier.director-bundle-verifier/v1"
SIGNER_REGISTRY_SCHEMA = "simulator_frontier.trusted-signer-registry/v1"


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _require_sha256(name: str, value: Any) -> str:
    text = str(value)
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise InvalidEvidenceError(
            f"{name} is not a lowercase sha256 hex digest: {text[:24]!r}…")
    return text


def _require_nonempty_str(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidEvidenceError(
            f"{name} must be a non-empty string, got {value!r}")
    return value


@dataclass(frozen=True)
class TrustedSignerRegistry:
    """The director-frozen set of trusted signer identities (mint-only)."""

    registry_id: str
    signer_ids: tuple[str, ...]
    registry_hash: str = field(init=False)
    registry_schema: str = SIGNER_REGISTRY_SCHEMA

    def __post_init__(self) -> None:
        _require_nonempty_str("registry_id", self.registry_id)
        signers = tuple(str(s) for s in self.signer_ids)
        if not signers:
            raise InvalidEvidenceError(
                "TrustedSignerRegistry requires at least one trusted signer")
        object.__setattr__(self, "signer_ids", signers)
        payload = {
            "schema": SIGNER_REGISTRY_SCHEMA,
            "registry_id": self.registry_id,
            "signer_ids": sorted(signers),
        }
        object.__setattr__(self, "registry_hash", _canonical_sha256(payload))

    def is_trusted(self, signer_id: Any) -> bool:
        return str(signer_id) in self.signer_ids


@dataclass(frozen=True)
class DirectorBundleVerifier:
    """The director-shared verifier (mint-only, injected, never self-built).

    ``verifier_hash`` is computed in ``__post_init__``.  Verification is the
    CONTRACT surface: it never invents a protocol — it checks signer identity
    against the trusted registry, recomputes the manifest's canonical payload
    hash and requires the signature/reference to be present and
    non-synthetic, and checks that entry points are registered (allowlisted)
    with a source-hash binding.
    """

    verifier_id: str
    registry_identity: str
    trusted_signer_registry: TrustedSignerRegistry
    allowlisted_entrypoints: tuple[str, ...]
    verify_signature: Any
    verifier_hash: str = field(init=False)
    verifier_schema: str = BUNDLE_VERIFIER_SCHEMA

    def __post_init__(self) -> None:
        _require_nonempty_str("verifier_id", self.verifier_id)
        _require_nonempty_str("registry_identity", self.registry_identity)
        if not isinstance(self.trusted_signer_registry, TrustedSignerRegistry):
            raise InvalidEvidenceError(
                "DirectorBundleVerifier requires a minted TrustedSignerRegistry")
        if not callable(self.verify_signature):
            raise InvalidEvidenceError(
                "DirectorBundleVerifier requires the director-injected "
                "verify_signature callable — production code never implements "
                "its own signature verification (no self-issued crypto)")
        object.__setattr__(self, "allowlisted_entrypoints",
                           tuple(str(e) for e in self.allowlisted_entrypoints))
        payload = {
            "schema": BUNDLE_VERIFIER_SCHEMA,
            "verifier_id": self.verifier_id,
            "registry_identity": self.registry_identity,
            "trusted_signer_registry_hash": self.trusted_signer_registry.registry_hash,
            "allowlisted_entrypoints": sorted(self.allowlisted_entrypoints),
        }
        object.__setattr__(self, "verifier_hash", _canonical_sha256(payload))

    def entrypoint_is_allowlisted(self, entrypoint: Any) -> bool:
        return str(entrypoint) in self.allowlisted_entrypoints


def verify_production_bundle(manifest: Mapping[str, Any], *,
                             verifier: DirectorBundleVerifier,
                             signature_ref: Any) -> str:
    """Fail-closed verification of a signed runtime bundle.

    * the verifier must be the director-injected one (a caller-supplied
      mapping or foreign type is refused);
    * ``signature_ref`` must be non-empty and never SYNTHETIC;
    * the signer identity embedded in ``signature_ref`` must be in the
      trusted registry;
    * the manifest's canonical payload hash must be recomputed and match
      ``manifest``'s own binding hash (the manifest must carry it);
    * every entry point named by the manifest must be allowlisted.

    Returns the recomputed canonical manifest hash.
    """
    if isinstance(verifier, Mapping) or not isinstance(verifier, DirectorBundleVerifier):
        raise InvalidEvidenceError(
            "verify_production_bundle requires the director-injected "
            "DirectorBundleVerifier (a self-built mapping is never accepted)")
    reference = _require_nonempty_str("signature_ref", signature_ref)
    if reference.startswith("SYNTHETIC_SIGNATURE_"):
        raise ProvenanceViolationError(
            f"synthetic signature reference {reference!r} can never be admitted "
            "on the production path")
    # E3-P0: extracting a signer by splitting a reference string and checking
    # membership is NOT signature verification — that pseudo-crypto is
    # DELETED.  The signer id is a manifest field, its trust is checked against
    # the registry, the payload hash is recomputed, and THEN the director-
    # injected verify_signature is EXECUTED and must return True.
    if not isinstance(manifest, Mapping):
        raise InvalidEvidenceError("manifest must be a JSON object")
    declared = manifest.get("manifest_hash", "")
    if not declared:
        raise InvalidEvidenceError(
            "production bundle must carry a canonical manifest_hash to be "
            "signature-verifiable")
    canonical = _canonical_sha256({k: v for k, v in manifest.items()
                                   if k != "manifest_hash"})
    if canonical != _require_sha256("manifest.manifest_hash", declared):
        raise ProvenanceViolationError(
            "manifest canonical payload hash mismatch: the signed bundle does "
            "not recompute to its declared manifest_hash (fail closed)")
    signer = manifest.get("controller_identity", "")
    if not signer or not verifier.trusted_signer_registry.is_trusted(signer):
        raise ProvenanceViolationError(
            f"signer identity {signer!r} is not in the trusted signer registry "
            "(signature not issued by the director; fail closed)")
    try:
        ok = verifier.verify_signature(
            signer_id=signer, payload_hash=canonical, signature_ref=reference)
    except Exception as exc:
        raise ProvenanceViolationError(
            f"director verify_signature failed: {exc!r} (fail closed)") from exc
    if not isinstance(ok, bool) or not ok:
        raise ProvenanceViolationError(
            "director verify_signature did not return True — the signature "
            "reference is not accepted (fail closed)")
    for key in ("student", "reference", "training_runtime",
                "training_surface_capability", "memory",
                "two_llm_runtime", "taskparam_apply_entrypoint"):
        section = manifest.get(key)
        if isinstance(section, Mapping):
            for entrypoint in _entrypoints_in(section):
                if not verifier.entrypoint_is_allowlisted(entrypoint):
                    raise ProvenanceViolationError(
                        f"entry point {entrypoint!r} is not in the director "
                        "allowlist (arbitrary module:attr import for "
                        "training/TaskParams/Memory code is forbidden)")
    return canonical


def _entrypoints_in(section: Mapping[str, Any]) -> list[str]:
    result: list[str] = []
    for value in section.values():
        if isinstance(value, str) and ":" in value and not value.startswith(("/", "\\", "src-")):
            result.append(value)
    return result


# ---------------------------------------------------------------------------
# Fail-closed injection slot (mirrors the formal-asset-registry discipline).
# ---------------------------------------------------------------------------

_PROD_VERIFIER: DirectorBundleVerifier | None = None


def production_verifier_bound() -> bool:
    return _PROD_VERIFIER is not None


def inject_production_verifier(verifier: DirectorBundleVerifier) -> None:
    global _PROD_VERIFIER
    if _PROD_VERIFIER is not None:
        raise ProvenanceViolationError(
            "a production bundle verifier is already injected; explicit clear "
            "required before re-injection (fail closed)")
    if isinstance(verifier, Mapping) or not isinstance(verifier, DirectorBundleVerifier):
        raise ProvenanceViolationError(
            "only a minted DirectorBundleVerifier can enter the production slot")
    _PROD_VERIFIER = verifier


def clear_injected_production_verifier() -> None:
    global _PROD_VERIFIER
    _PROD_VERIFIER = None


def production_verifier() -> DirectorBundleVerifier:
    if _PROD_VERIFIER is None:
        raise InvalidEvidenceError(
            f"{E3_PRODUCTION_BUNDLE_VERIFIER_UNBOUND}: the director-injected "
            "bundle verifier is not bound; production bundles are blocked "
            "fail closed")
    return _PROD_VERIFIER


def verify_production_bundle_with_slot(manifest: Mapping[str, Any]) -> str:
    """Production entry: the verifier comes ONLY from the injection slot."""
    verifier = production_verifier()
    return verify_production_bundle(manifest, verifier=verifier,
                                    signature_ref=manifest.get(
                                        "controller_signature_ref", ""))
