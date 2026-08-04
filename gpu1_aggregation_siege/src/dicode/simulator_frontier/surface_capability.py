"""Signed training-surface capability descriptors (CC4 follow-up, P0-15).

Before this contract existed, the preflight INFERRED the training surface by
calling ``save_full_state`` / ``restore_full_state`` with an empty path and
reading the exception type: NotImplementedError meant "absent", any other
failure meant "present".  That probe is spoofable — any adapter can raise a
generic error from a junk path and be certified as training-capable without
implementing anything.  Exception inference is therefore deleted from the
production preflight.

Capability is now evidence, not inference: a ``TrainingSurfaceCapability``
descriptor is MINTED (never supplied as a mapping, never self-signed into
production) and names exactly which adapter it describes
(``adapter_identity_hash``), whether ``save_full_state`` /
``restore_full_state`` are capable, WHO verified it (``verifier_id``) and
the controller signature reference (``signature_ref``).  The preflight only
accepts a descriptor that verifies (mint-only ``capability_hash``), is bound
to the MOUNTED adapter's identity, and carries a non-synthetic signature —
otherwise the surface stays blocked.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields
from typing import Any, Mapping

from .errors import InvalidEvidenceError

SURFACE_CAPABILITY_VERSION = "training-surface-capability/v1"

BLOCKED_NO_SIGNED_TRAINING_SURFACE_CAPABILITY = (
    "BLOCKED_NO_SIGNED_TRAINING_SURFACE_CAPABILITY")


def _require_sha256_field(label: str, digest: Any) -> str:
    text = str(digest)
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise InvalidEvidenceError(
            f"{label} is not a lowercase sha256 hex digest: {text[:24]!r}…")
    return text


@dataclass(frozen=True)
class TrainingSurfaceCapability:
    """One immutable, signed capability descriptor (mint-only).

    ``capability_hash`` is NOT a constructor argument: it is computed in
    ``__post_init__`` from the descriptor fields only, so no caller can
    supply (self-report) the hash.
    """

    descriptor_id: str
    adapter_identity_hash: str
    save_full_state_capable: bool
    restore_full_state_capable: bool
    verifier_id: str
    signature_ref: str
    capability_hash: str = field(init=False)
    capability_version: str = SURFACE_CAPABILITY_VERSION

    def __post_init__(self) -> None:
        for label, value in (("descriptor_id", self.descriptor_id),
                             ("verifier_id", self.verifier_id),
                             ("signature_ref", self.signature_ref)):
            if not str(value).strip():
                raise InvalidEvidenceError(
                    f"TrainingSurfaceCapability.{label} is empty — a capability "
                    "descriptor is never anonymous, never unverified and never "
                    "unsigned")
        _require_sha256_field(
            "TrainingSurfaceCapability.adapter_identity_hash",
            self.adapter_identity_hash)
        for label, flag in (("save_full_state_capable",
                             self.save_full_state_capable),
                            ("restore_full_state_capable",
                             self.restore_full_state_capable)):
            if not isinstance(flag, bool):
                raise InvalidEvidenceError(
                    f"TrainingSurfaceCapability.{label} must be a genuine bool, "
                    f"got {flag!r}")
        payload = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name != "capability_hash"
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, default=str)
        object.__setattr__(
            self, "capability_hash",
            hashlib.sha256(blob.encode("utf-8")).hexdigest())


def mint_training_surface_capability(*, descriptor_id: Any,
                                     adapter_identity_hash: Any,
                                     save_full_state_capable: Any,
                                     restore_full_state_capable: Any,
                                     verifier_id: Any,
                                     signature_ref: Any
                                     ) -> TrainingSurfaceCapability:
    """Mint the immutable capability descriptor (fail closed on any gap)."""
    _require_sha256_field("adapter_identity_hash", adapter_identity_hash)
    for label, flag in (("save_full_state_capable", save_full_state_capable),
                        ("restore_full_state_capable",
                         restore_full_state_capable)):
        if not isinstance(flag, bool):
            raise InvalidEvidenceError(
                f"{label} must be a genuine bool, got {flag!r}")
    return TrainingSurfaceCapability(
        descriptor_id=str(descriptor_id),
        adapter_identity_hash=str(adapter_identity_hash),
        save_full_state_capable=bool(save_full_state_capable),
        restore_full_state_capable=bool(restore_full_state_capable),
        verifier_id=str(verifier_id),
        signature_ref=str(signature_ref),
    )


def verify_training_surface_capability(capability: Any) -> None:
    """Recompute the descriptor hash; reject mappings, foreign types, tamper."""
    if isinstance(capability, Mapping):
        raise InvalidEvidenceError(
            "verify_training_surface_capability requires a minted "
            "TrainingSurfaceCapability, not a mapping")
    if not isinstance(capability, TrainingSurfaceCapability):
        raise InvalidEvidenceError(
            f"verify_training_surface_capability requires a minted "
            f"TrainingSurfaceCapability, got {type(capability).__name__}")
    _require_sha256_field(
        "TrainingSurfaceCapability.adapter_identity_hash",
        capability.adapter_identity_hash)
    payload = {
        f.name: getattr(capability, f.name)
        for f in fields(capability)
        if f.name != "capability_hash"
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)
    expected = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    if expected != capability.capability_hash:
        raise InvalidEvidenceError(
            "capability_hash mismatch: the TrainingSurfaceCapability was "
            "tampered with or self-reported (fail closed)")
