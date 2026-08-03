"""TRAINING_DISCOVERY capture provenance with formal-data isolation (condition 2).

Frontier-collection rollouts must carry TRAINING_DISCOVERY provenance and must
stay structurally isolated from the frozen formal evaluation banks/worlds:
formal bank/world identities can never enter capture requests, the
FrontierArchive, the InvocationGate, an LLM prompt, or the selector.

Hardening after the controller's PASS_WITH_BLOCKER review (2026-08-04):
marker-string denylists alone are bypassable (a formal bank passed under a
neutral alias or as a bare SHA/ID sailed through).  Isolation is therefore
TWO-LAYER and registry-bound:

  1. ALLOWLIST BINDING (primary): every discovery input (bank ref, world set
     id, world set hash) must resolve to a record in an explicit
     ``DiscoveryProvenanceRegistry``.  Unregistered strings raise — no
     guessing, no "closest match", no default registry.
  2. FORBIDDEN IDENTITY SWEEP (defence in depth): the controller-injected
     frozen set of formal asset identities (canonical id AND sha256) is
     swept case-insensitively over every textual field of the capture,
     including nested notes; case variants, bare SHAs and embedded mentions
     all raise.

Honest status (controller rule 5): the REAL frozen formal asset identity set
has not been injected by the controller this round, so real isolation is NOT
proven.  Until it is bound: ``DISCOVERY_FORMAL_PROVENANCE_ISOLATED`` stays
False and the status is ``BLOCKED_WAITING_FROZEN_FORMAL_ASSET_REGISTRY``;
only ``DISCOVERY_PROVENANCE_CONTRACT_READY`` may be claimed.  Synthetic
registries used in tests are fixtures, never the real manifest.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

from .errors import ProvenanceViolationError
from .provenance import DataSource, FormalDataLeakageGuard

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PLACEHOLDER_RE = re.compile(r"^(PENDING.*|UNKNOWN|TODO|NONE|N/?A)$", re.IGNORECASE)


class DiscoveryProvenance(str, Enum):
    """Legal provenance classes for frontier capture requests."""

    TRAINING_DISCOVERY = "TRAINING_DISCOVERY"
    # Explicitly-labelled synthetic fixtures only (contract tests).  Never a
    # legal source for real capture: validate_capture_provenance rejects it
    # unless the caller opts in with allow_synthetic_fixture=True.
    SYNTHETIC_FIXTURE = "SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT"


class AssetKind(str, Enum):
    BANK = "bank"
    WORLD_SET = "world_set"


# Legacy marker denylist kept ONLY as a tertiary defence layer.  It is not
# sufficient on its own (see module docstring) — registry binding is primary.
_FORMAL_BANK_MARKERS = ("FORMAL_FRONT", "FORMAL_BACK", "FORMAL_FULL")
_FORMAL_WORLD_MARKERS = ("FORMAL_WORLD", "FORMAL_EVAL_WORLD")

# ---------------------------------------------------------------------------
# Honest round status (controller rule 5): contract ready, real isolation
# blocked until the controller injects the frozen formal asset identity set.
# ---------------------------------------------------------------------------
DISCOVERY_PROVENANCE_CONTRACT_READY = True
FROZEN_FORMAL_ASSET_REGISTRY_BOUND = False
DISCOVERY_FORMAL_PROVENANCE_ISOLATED = False
BLOCKED_WAITING_FROZEN_FORMAL_ASSET_REGISTRY = "BLOCKED_WAITING_FROZEN_FORMAL_ASSET_REGISTRY"


def registry_status() -> dict:
    return {
        "bound": False,
        "status": BLOCKED_WAITING_FROZEN_FORMAL_ASSET_REGISTRY,
        "contract_ready": DISCOVERY_PROVENANCE_CONTRACT_READY,
        "real_isolation_proven": False,
        "note": ("the frozen forbidden formal asset identity set must be injected "
                 "by the controller; synthetic registries in tests are fixtures, "
                 "never the real manifest"),
    }


@dataclass(frozen=True)
class FormalAssetIdentity:
    """One forbidden formal evaluation asset identity (controller-injected).

    Both the canonical id and the sha256 are required so that neither a
    neutral alias nor a bare SHA can smuggle the asset into discovery.
    """

    asset_kind: AssetKind
    canonical_id: str
    sha256: str

    def __post_init__(self) -> None:
        if isinstance(self.asset_kind, str):
            object.__setattr__(self, "asset_kind", AssetKind(self.asset_kind))
        if not isinstance(self.asset_kind, AssetKind):
            raise ProvenanceViolationError(
                f"FormalAssetIdentity.asset_kind must be AssetKind, got {self.asset_kind!r}")
        if not self.canonical_id or _PLACEHOLDER_RE.match(self.canonical_id.strip()):
            raise ProvenanceViolationError(
                f"FormalAssetIdentity.canonical_id must be non-empty and not a placeholder, "
                f"got {self.canonical_id!r}")
        if not _SHA256_RE.match(self.sha256):
            raise ProvenanceViolationError(
                f"FormalAssetIdentity.sha256 must be a 64-hex sha256, got {self.sha256!r}")


@dataclass(frozen=True)
class DiscoveryAssetRecord:
    """One registered TRAINING_DISCOVERY asset (allowlist entry)."""

    asset_id: str
    asset_kind: AssetKind
    world_set_hash: str = ""
    content_sha256: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.asset_kind, str):
            object.__setattr__(self, "asset_kind", AssetKind(self.asset_kind))
        if not isinstance(self.asset_kind, AssetKind):
            raise ProvenanceViolationError(
                f"DiscoveryAssetRecord.asset_kind must be AssetKind, got {self.asset_kind!r}")
        if not self.asset_id or _PLACEHOLDER_RE.match(self.asset_id.strip()):
            raise ProvenanceViolationError(
                f"DiscoveryAssetRecord.asset_id must be non-empty and not a placeholder, "
                f"got {self.asset_id!r}")
        if self.asset_kind is AssetKind.WORLD_SET:
            if not _SHA256_RE.match(self.world_set_hash):
                raise ProvenanceViolationError(
                    "WORLD_SET discovery records require a 64-hex world_set_hash "
                    f"(fail closed), got {self.world_set_hash!r}")
        elif self.world_set_hash:
            raise ProvenanceViolationError(
                "BANK discovery records must not carry a world_set_hash (kind discipline)")
        if self.content_sha256 and not _SHA256_RE.match(self.content_sha256):
            raise ProvenanceViolationError(
                f"DiscoveryAssetRecord.content_sha256 must be 64-hex when present, "
                f"got {self.content_sha256!r}")


@dataclass(frozen=True)
class DiscoveryProvenanceRegistry:
    """Controlled, verifiable binding surface for discovery captures.

    Must be injected by the caller (ultimately: the controller-signed frozen
    manifest).  A missing/invalid registry fails closed — never guessed.
    """

    registry_id: str
    controller_signature_ref: str
    frozen: bool
    forbidden_formal_identities: tuple[FormalAssetIdentity, ...]
    allowed_discovery_assets: tuple[DiscoveryAssetRecord, ...]
    registry_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "forbidden_formal_identities",
                           tuple(self.forbidden_formal_identities))
        object.__setattr__(self, "allowed_discovery_assets",
                           tuple(self.allowed_discovery_assets))


def registry_hash_of(registry_id: str, controller_signature_ref: str,
                     forbidden: tuple[FormalAssetIdentity, ...],
                     allowed: tuple[DiscoveryAssetRecord, ...]) -> str:
    forbidden_rows = sorted(
        (ident.asset_kind.value, ident.canonical_id, ident.sha256) for ident in forbidden)
    allowed_rows = sorted(
        (rec.asset_id, rec.asset_kind.value, rec.world_set_hash, rec.content_sha256)
        for rec in allowed)
    payload = {
        "registry_id": registry_id,
        "controller_signature_ref": controller_signature_ref,
        "forbidden_formal_identities": forbidden_rows,
        "allowed_discovery_assets": allowed_rows,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _forbidden_hit(text: str, forbidden: tuple[FormalAssetIdentity, ...]) -> FormalAssetIdentity | None:
    """Case-insensitive containment sweep of canonical ids AND sha256s."""
    low = str(text).casefold()
    for ident in forbidden:
        if ident.canonical_id.casefold() in low or ident.sha256.casefold() in low:
            return ident
    return None


def validate_discovery_registry(registry: DiscoveryProvenanceRegistry) -> None:
    """Fail-closed validation of the registry itself (never trust, always check)."""
    if not isinstance(registry, DiscoveryProvenanceRegistry):
        raise ProvenanceViolationError(
            f"expected DiscoveryProvenanceRegistry, got {type(registry).__name__}")
    if not registry.registry_id or _PLACEHOLDER_RE.match(registry.registry_id.strip()):
        raise ProvenanceViolationError(
            f"registry_id must be non-empty and not a placeholder, got {registry.registry_id!r}")
    if not registry.controller_signature_ref:
        raise ProvenanceViolationError(
            "discovery registry requires a controller signature reference "
            "(BLOCKED_WAITING_FROZEN_FORMAL_ASSET_REGISTRY until injected)")
    if not registry.frozen:
        raise ProvenanceViolationError("discovery registry must be frozen")
    forbidden = registry.forbidden_formal_identities
    allowed = registry.allowed_discovery_assets
    if not forbidden:
        raise ProvenanceViolationError(
            "discovery registry requires a non-empty forbidden formal asset identity set")
    kinds = {ident.asset_kind for ident in forbidden}
    if AssetKind.BANK not in kinds or AssetKind.WORLD_SET not in kinds:
        raise ProvenanceViolationError(
            "forbidden identity set must cover at least one bank AND one world_set identity "
            "(canonical id + sha256 each)")
    if not allowed:
        raise ProvenanceViolationError(
            "discovery registry requires a non-empty TRAINING_DISCOVERY allowlist")

    seen: set[str] = set()
    for rec in allowed:
        key = rec.asset_id.casefold()
        if key in seen:
            raise ProvenanceViolationError(f"duplicate discovery asset_id {rec.asset_id!r}")
        seen.add(key)

    forbidden_ids = {ident.canonical_id.casefold() for ident in forbidden}
    forbidden_shas = {ident.sha256.casefold() for ident in forbidden}
    for rec in allowed:
        if rec.asset_id.casefold() in forbidden_ids:
            raise ProvenanceViolationError(
                f"discovery asset {rec.asset_id!r} collides with a forbidden formal identity")
        for value in (rec.world_set_hash, rec.content_sha256):
            if value and value.casefold() in forbidden_shas:
                raise ProvenanceViolationError(
                    f"discovery asset {rec.asset_id!r} carries a forbidden formal sha256")
        if _forbidden_hit(rec.asset_id, forbidden) is not None:
            raise ProvenanceViolationError(
                f"discovery asset {rec.asset_id!r} embeds a forbidden formal identity")

    expected = registry_hash_of(registry.registry_id, registry.controller_signature_ref,
                                forbidden, allowed)
    if registry.registry_hash != expected:
        raise ProvenanceViolationError(
            f"discovery registry hash mismatch: got {registry.registry_hash!r}, "
            f"expected {expected!r}")


@dataclass(frozen=True)
class CaptureProvenance:
    """Strongly-typed provenance attached to every frontier capture request."""

    provenance: DiscoveryProvenance
    rollout_protocol_id: str
    world_set_hash: str
    bank_refs: tuple[str, ...] = ()
    world_set_id: str = ""
    notes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.provenance, str):
            object.__setattr__(self, "provenance", DiscoveryProvenance(self.provenance))
        object.__setattr__(self, "bank_refs", tuple(self.bank_refs))


def _contains_formal_marker(text: str) -> str | None:
    upper = text.upper()
    for marker in _FORMAL_BANK_MARKERS + _FORMAL_WORLD_MARKERS:
        if marker in upper:
            return marker
    return None


def validate_capture_provenance(cap: CaptureProvenance, *,
                                registry: DiscoveryProvenanceRegistry | None,
                                allow_synthetic_fixture: bool = False) -> None:
    """Fail-closed, registry-bound validation of capture provenance.

    The registry is MANDATORY: ``registry=None`` (or any missing/invalid
    registry) raises instead of guessing — real isolation requires the
    controller-injected frozen formal asset identity set.

    Raises ProvenanceViolationError when:
    - the registry is missing or invalid (fail closed);
    - provenance is not TRAINING_DISCOVERY (synthetic only via explicit opt-in);
    - rollout protocol id is missing, or world_set_hash is missing / not a
      64-hex hash / not registered in the discovery allowlist;
    - any bank ref does not resolve to a registered discovery BANK record
      (closes the neutral-alias bypass);
    - a world_set_id does not resolve to a registered WORLD_SET record;
    - any field (refs, world id, protocol id, notes keys/values — including
      nested/embedded text) matches a forbidden formal identity by canonical
      id or sha256, case-insensitively;
    - a legacy FORMAL_* marker appears anywhere (tertiary layer);
    - the formal leakage guard forbids the combination for frontier consumers.
    """
    if registry is None:
        raise ProvenanceViolationError(
            "capture provenance validation requires a DiscoveryProvenanceRegistry; "
            "none provided -> fail closed, never guess "
            f"({BLOCKED_WAITING_FROZEN_FORMAL_ASSET_REGISTRY})")
    validate_discovery_registry(registry)
    forbidden = registry.forbidden_formal_identities

    if cap.provenance is DiscoveryProvenance.SYNTHETIC_FIXTURE:
        if not allow_synthetic_fixture:
            raise ProvenanceViolationError(
                "SYNTHETIC_FIXTURE provenance is only legal in explicitly labelled contract tests")
    elif cap.provenance is not DiscoveryProvenance.TRAINING_DISCOVERY:
        raise ProvenanceViolationError(
            f"illegal capture provenance: {cap.provenance!r} (TRAINING_DISCOVERY required)")

    if not cap.rollout_protocol_id:
        raise ProvenanceViolationError("capture provenance requires rollout_protocol_id")
    hit = _forbidden_hit(cap.rollout_protocol_id, forbidden)
    if hit is not None:
        raise ProvenanceViolationError(
            f"rollout_protocol_id embeds forbidden formal identity {hit.canonical_id!r}")

    # --- world_set_hash: must be a registered discovery world set (not just non-empty)
    if not cap.world_set_hash:
        raise ProvenanceViolationError("capture provenance requires world_set_hash")
    if not _SHA256_RE.match(cap.world_set_hash):
        raise ProvenanceViolationError(
            f"world_set_hash must be a 64-hex sha256, got {cap.world_set_hash!r}")
    registered_world_hashes = {
        rec.world_set_hash.casefold() for rec in registry.allowed_discovery_assets
        if rec.asset_kind is AssetKind.WORLD_SET}
    if cap.world_set_hash.casefold() not in registered_world_hashes:
        raise ProvenanceViolationError(
            f"world_set_hash {cap.world_set_hash!r} is not registered in the discovery "
            "allowlist (fail closed; unregistered world sets cannot enter capture)")

    # --- bank refs: allowlist resolution (primary) + forbidden sweep (secondary)
    allowed_bank_ids = {
        rec.asset_id.casefold() for rec in registry.allowed_discovery_assets
        if rec.asset_kind is AssetKind.BANK}
    for ref in cap.bank_refs:
        if not str(ref):
            raise ProvenanceViolationError("bank_refs entries must be non-empty")
        if str(ref).casefold() not in allowed_bank_ids:
            raise ProvenanceViolationError(
                f"bank ref {ref!r} is not a registered discovery asset (allowlist binding "
                "is mandatory; unregistered refs — including formal banks under neutral "
                "aliases — are rejected fail closed)")
        hit = _forbidden_hit(str(ref), forbidden)
        if hit is not None:
            raise ProvenanceViolationError(
                f"bank ref {ref!r} matches forbidden formal identity {hit.canonical_id!r}")
        marker = _contains_formal_marker(str(ref))
        if marker is not None:
            raise ProvenanceViolationError(
                f"formal bank identifier {marker!r} cannot enter capture provenance")
        for formal in FormalDataLeakageGuard.FORBIDDEN:
            if str(ref).upper() == formal.value:
                FormalDataLeakageGuard.assert_allowed(formal, consumer="FrontierArchive")

    # --- world_set_id: optional, but must resolve to a registered world set
    if cap.world_set_id:
        allowed_world_ids = {
            rec.asset_id.casefold() for rec in registry.allowed_discovery_assets
            if rec.asset_kind is AssetKind.WORLD_SET}
        if cap.world_set_id.casefold() not in allowed_world_ids:
            raise ProvenanceViolationError(
                f"world_set_id {cap.world_set_id!r} is not a registered discovery world set "
                "(fail closed)")
        hit = _forbidden_hit(cap.world_set_id, forbidden)
        if hit is not None:
            raise ProvenanceViolationError(
                f"world_set_id {cap.world_set_id!r} matches forbidden formal identity "
                f"{hit.canonical_id!r}")
        marker = _contains_formal_marker(cap.world_set_id)
        if marker is not None:
            raise ProvenanceViolationError(
                f"formal world identifier {marker!r} cannot enter capture provenance")

    # --- notes: forbidden identity sweep over keys and values (incl. embedded text)
    for key, value in cap.notes.items():
        for text in (str(key), str(value)):
            hit = _forbidden_hit(text, forbidden)
            if hit is not None:
                raise ProvenanceViolationError(
                    f"capture provenance notes embed forbidden formal identity "
                    f"{hit.canonical_id!r} (in {str(key)!r})")
            marker = _contains_formal_marker(text)
            if marker is not None:
                raise ProvenanceViolationError(
                    f"formal identifier {marker!r} cannot enter capture provenance notes")


def discovery_source_for(provenance: DiscoveryProvenance) -> DataSource:
    """Map capture provenance onto the shared DataSource taxonomy."""
    if provenance is DiscoveryProvenance.TRAINING_DISCOVERY:
        return DataSource.TRAINING_FRONTIER_CAPTURE
    return DataSource.SYNTHETIC_TEST


def assert_not_formal(source: DataSource | str, consumer: str) -> None:
    """Convenience bridge: reuse FormalDataLeakageGuard for arbitrary consumers."""
    FormalDataLeakageGuard.assert_allowed(DataSource(source), consumer)
