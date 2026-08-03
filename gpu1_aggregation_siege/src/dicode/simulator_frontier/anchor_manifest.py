"""Shared frozen anchor-manifest schema + fail-closed binding (condition 5).

The four standard-reset anchors are a THREE-DIRECTION SHARED frozen manifest.
CC4's responsibility is ONLY the shared schema and the binding interface:
the scientific content (which distributions/worlds/seeds the anchors use)
MUST come from the 总控 manifest and is never self-invented here.

Round status (no 总控 manifest received yet):
    SHARED_ANCHOR_MANIFEST_BOUND = False
    status                        = BLOCKED_SHARED_ANCHOR_MANIFEST

All tests use synthetic fixtures explicitly labeled
``SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT`` — they exercise the schema and
binding rules, they do NOT select anchor science.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .errors import InvalidEvidenceError
from .provenance import ProvenanceViolationError

ANCHOR_SLOT_COUNT = 4
DYNAMIC_DISTRIBUTION_COUNT = 12
STANDARD_RESET_PROTOCOL = "STANDARD_RESET"

# This round's honest status (no 总控 manifest received).
SHARED_ANCHOR_MANIFEST_BOUND = False
BLOCKED_SHARED_ANCHOR_MANIFEST = "BLOCKED_SHARED_ANCHOR_MANIFEST"


@dataclass(frozen=True)
class AnchorDefinition:
    """One anchor slot.  References ONLY — no scientific constants live here."""

    anchor_id: str
    world_set_ref: str
    reset_protocol: str
    seed_policy_ref: str

    def __post_init__(self) -> None:
        for name in ("anchor_id", "world_set_ref", "reset_protocol", "seed_policy_ref"):
            if not getattr(self, name):
                raise InvalidEvidenceError(f"AnchorDefinition.{name} is required")


@dataclass(frozen=True)
class AnchorManifest:
    """The shared frozen manifest as issued by 总控."""

    manifest_id: str
    controller_signature_ref: str
    frozen: bool
    anchors: tuple[AnchorDefinition, ...]
    manifest_hash: str


def manifest_hash_of(manifest_id: str, controller_signature_ref: str,
                     anchors: tuple[AnchorDefinition, ...]) -> str:
    """Canonical frozen hash binding id + signature ref + anchor slots."""
    payload = {"manifest_id": manifest_id,
               "controller_signature_ref": controller_signature_ref,
               "anchors": [asdict(a) for a in anchors]}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_anchor_manifest(manifest: AnchorManifest) -> None:
    """Fail closed on ANY structural violation (never warn-and-continue)."""
    if not isinstance(manifest, AnchorManifest):
        raise InvalidEvidenceError("manifest must be an AnchorManifest")
    if not manifest.manifest_id:
        raise InvalidEvidenceError("manifest_id is required")
    if not manifest.controller_signature_ref:
        raise InvalidEvidenceError(
            "FAIL CLOSED: anchor manifest without 总控 signature reference is "
            "never accepted (anchors are a shared three-direction contract)")
    if not manifest.frozen:
        raise InvalidEvidenceError("anchor manifest must be frozen before binding")
    if len(manifest.anchors) != ANCHOR_SLOT_COUNT:
        raise InvalidEvidenceError(
            f"anchor manifest must define exactly {ANCHOR_SLOT_COUNT} anchor slots, "
            f"got {len(manifest.anchors)}")
    ids = [a.anchor_id for a in manifest.anchors]
    if len(set(ids)) != len(ids):
        raise InvalidEvidenceError(f"duplicate anchor_id values: {ids}")
    for anchor in manifest.anchors:
        if anchor.reset_protocol != STANDARD_RESET_PROTOCOL:
            raise InvalidEvidenceError(
                f"anchor {anchor.anchor_id}: reset_protocol must be "
                f"{STANDARD_RESET_PROTOCOL}, got {anchor.reset_protocol!r}")
    expected = manifest_hash_of(manifest.manifest_id,
                                manifest.controller_signature_ref, manifest.anchors)
    if manifest.manifest_hash != expected:
        raise InvalidEvidenceError(
            f"manifest_hash mismatch: manifest carries {manifest.manifest_hash[:16]}… "
            f"but its content hashes to {expected[:16]}… (fail closed)")


@dataclass(frozen=True)
class RetentionContract:
    """The 12 dynamic + 4 anchor retention shape the manifest binds into."""

    dynamic_distribution_count: int
    anchor_slot_count: int
    anchor_ratio: float
    formal_banks_in_online_curriculum: bool = False

    def validate(self) -> None:
        if int(self.dynamic_distribution_count) != DYNAMIC_DISTRIBUTION_COUNT:
            raise InvalidEvidenceError(
                f"retention contract requires {DYNAMIC_DISTRIBUTION_COUNT} dynamic "
                f"frontier distributions, got {self.dynamic_distribution_count}")
        if int(self.anchor_slot_count) != ANCHOR_SLOT_COUNT:
            raise InvalidEvidenceError(
                f"retention contract requires {ANCHOR_SLOT_COUNT} anchor slots, "
                f"got {self.anchor_slot_count}")
        if not (0.0 < float(self.anchor_ratio) <= 1.0):
            raise InvalidEvidenceError(
                "anchor ratio must be strictly > 0 (anchors are mandatory, "
                f"never 0), got {self.anchor_ratio}")
        if self.formal_banks_in_online_curriculum:
            raise ProvenanceViolationError(
                "formal evaluation banks must never enter the online curriculum")


def bind_anchor_manifest(manifest: AnchorManifest,
                         retention: RetentionContract) -> Mapping[str, Any]:
    """Fail-closed binding of the 总控 manifest into the retention contract.

    Both sides are validated; any violation raises.  Returns the bound record
    (manifest id + frozen hash + retention shape) for reports/contracts.
    """
    validate_anchor_manifest(manifest)
    retention.validate()
    return {
        "bound": True,
        "manifest_id": manifest.manifest_id,
        "manifest_hash": manifest.manifest_hash,
        "controller_signature_ref": manifest.controller_signature_ref,
        "anchor_ids": tuple(a.anchor_id for a in manifest.anchors),
        "dynamic_distribution_count": int(retention.dynamic_distribution_count),
        "anchor_ratio": float(retention.anchor_ratio),
        "status": "SHARED_ANCHOR_MANIFEST_BOUND",
    }


def unbound_status() -> Mapping[str, Any]:
    """Honest status while no 总控 manifest has been received."""
    return {
        "bound": False,
        "status": BLOCKED_SHARED_ANCHOR_MANIFEST,
        "reason": ("the shared frozen anchor manifest must be issued by 总控; "
                   "CC4 provides the schema/binding interface only and never "
                   "self-selects anchor science"),
        "schema_ready": True,
    }
