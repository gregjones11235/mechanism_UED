"""Student identity: the minimal, hash-bound description of a high-capability Student.

Identity is fail-closed: any missing/placeholder field raises instead of
guessing.  The identity hash is the canonical binding key used by the
simulator_frontier student_binding module and the Frontier Archive.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, fields
from typing import Any, Mapping


_PLACEHOLDER = re.compile(r"^(PENDING.*|UNKNOWN|TODO|NONE|N/?A)$", re.IGNORECASE)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class StudentIdentityError(ValueError):
    """Raised when an identity is missing fields or carries placeholders."""


@dataclass(frozen=True)
class StudentIdentity:
    """Minimal identity fields (§19).  All are mandatory; nothing is inferred."""

    candidate_id: str
    architecture_family: str  # e.g. "RMT16", "GTrXL128", "SLOWGRU"
    checkpoint_format: str  # one of checkpoint_codec.FORMAT_*
    global_step: int
    total_env_steps: int
    params_sha256: str  # 64-hex params tree hash recorded at mount time
    source_commit: str
    observation_shape: tuple[int, ...]
    action_count: int
    memory_spec_hash: str = ""  # MemorySpec.spec_hash(); "" only pre-binding
    extras: Mapping[str, Any] = field(default_factory=dict)

    def identity_hash(self) -> str:
        payload = {
            "candidate_id": self.candidate_id,
            "architecture_family": self.architecture_family,
            "checkpoint_format": self.checkpoint_format,
            "global_step": int(self.global_step),
            "total_env_steps": int(self.total_env_steps),
            "params_sha256": self.params_sha256,
            "source_commit": self.source_commit,
            "observation_shape": [int(x) for x in self.observation_shape],
            "action_count": int(self.action_count),
            "memory_spec_hash": self.memory_spec_hash,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _require_text(name: str, value: Any) -> str:
    if value is None or not isinstance(value, str) or not value.strip():
        raise StudentIdentityError(f"identity field {name} is missing or empty")
    if _PLACEHOLDER.match(value.strip()):
        raise StudentIdentityError(f"identity field {name} carries placeholder {value!r}; never guess")
    return value.strip()


def validate_identity(identity: StudentIdentity) -> StudentIdentity:
    """Strictly validate an identity in place (returns it); raises on any gap."""
    if not isinstance(identity, StudentIdentity):
        raise StudentIdentityError(f"expected StudentIdentity, got {type(identity).__name__}")
    _require_text("candidate_id", identity.candidate_id)
    _require_text("architecture_family", identity.architecture_family)
    _require_text("checkpoint_format", identity.checkpoint_format)
    _require_text("source_commit", identity.source_commit)
    if not _HEX64.match(identity.params_sha256 or ""):
        raise StudentIdentityError(
            f"params_sha256 must be a 64-char hex sha256, got {identity.params_sha256!r}")
    if int(identity.global_step) < 0 or int(identity.total_env_steps) < 0:
        raise StudentIdentityError("global_step/total_env_steps must be >= 0")
    shape = identity.observation_shape
    if not isinstance(shape, tuple) or not shape or any(int(x) <= 0 for x in shape):
        raise StudentIdentityError(f"observation_shape must be a non-empty positive tuple, got {shape!r}")
    if int(identity.action_count) <= 0:
        raise StudentIdentityError("action_count must be > 0")
    if identity.memory_spec_hash:
        if not _HEX64.match(identity.memory_spec_hash):
            raise StudentIdentityError("memory_spec_hash must be empty or a 64-char hex sha256")
    return identity


def identity_to_mapping(identity: StudentIdentity) -> dict:
    """Mapping form used by simulator_frontier.student_binding hash-field checks."""
    validate_identity(identity)
    return {
        "identity_hash": identity.identity_hash(),
        "params_sha256": identity.params_sha256,
        "memory_spec_hash": identity.memory_spec_hash or "",
        "candidate_id": identity.candidate_id,
    }


def identity_field_names() -> tuple[str, ...]:
    return tuple(f.name for f in fields(StudentIdentity))
