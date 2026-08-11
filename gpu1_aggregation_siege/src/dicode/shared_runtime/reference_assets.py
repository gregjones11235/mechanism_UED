"""The REAL Reference assets (frozen RESET128 RMT16 arm).

The Reference is the second real CC2 arm (RESET128_RMT16_ORIGINAL_VTRACE_
98304): a real checkpoint with a verified file sha256, mounted read-only
through the same RMT16 adapter family. E1 never guesses a Reference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

from . import asset_locations as AL
from . import student_assets as SA

REFERENCE_CANDIDATE_ID = "RESET128_RMT16_ORIGINAL_VTRACE_98304"


@dataclass(frozen=True)
class ReferenceIdentityDescriptor:
    """The real Reference identity (immutable, hash-bound)."""

    candidate_id: str
    architecture_family: str
    memory_mode: str
    params_sha256: str
    checkpoint_file_sha256: str
    profile_hash: str
    source_commit: str
    object_identity_hash: str


_REFERENCE_CACHE: Dict[str, Any] = {}


def real_reference_adapter():
    """The REAL read-only Reference adapter (RESET128 arm)."""
    if "adapter" not in _REFERENCE_CACHE:
        _REFERENCE_CACHE["adapter"] = SA.real_student_adapter(
            REFERENCE_CANDIDATE_ID)
    return _REFERENCE_CACHE["adapter"]


def real_reference_identity() -> ReferenceIdentityDescriptor:
    adapter = real_reference_adapter()
    loc = AL.student_locations()
    profile_hash = AL.file_sha256(
        AL.resolve_repo_relative(loc["reset128_profile"]))
    identity = SA._canonical_sha256({
        "kind": "shared_runtime.reference_identity",
        "candidate_id": REFERENCE_CANDIDATE_ID,
        "architecture_family": "RMT16",
        "params_sha256": adapter.params_sha256,
        "checkpoint_file_sha256": adapter.checkpoint_file_sha256,
        "profile_hash": profile_hash,
    })
    if "identity" not in _REFERENCE_CACHE:
        _REFERENCE_CACHE["identity"] = ReferenceIdentityDescriptor(
            candidate_id=REFERENCE_CANDIDATE_ID,
            architecture_family="RMT16",
            memory_mode="RESET128",
            params_sha256=adapter.params_sha256,
            checkpoint_file_sha256=adapter.checkpoint_file_sha256,
            profile_hash=profile_hash,
            source_commit="src-sha256:" + loc["driver_source_sha256"],
            object_identity_hash=identity,
        )
    return _REFERENCE_CACHE["identity"]
