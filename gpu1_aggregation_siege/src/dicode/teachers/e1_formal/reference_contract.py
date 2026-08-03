"""G1: ReferenceIdentityContract — configurable, fail-closed, never guessed.

The Reference policy used for regret / candidate evaluation MUST be
frozen by the supervisor (or CC4) before any evaluation seam may run.
This module only VALIDATES the frozen identity; it performs no file
I/O, no checkpoint loading, no resolution of any kind — checkpoint_ref
is an opaque string consumed only by the CC4 adapter.

Frozen identity categories (supervisor gate G1):
  1. candidate_id
  2. checkpoint_ref        — checkpoint path OR external reference
  3. file_sha256 / params_sha256
  4. network architecture  — family / version / config hash
  5. memory semantics      — description + hash
  6. global_step (+ optional total_env_steps)
  7. source_commit
  8. seed + episode/reset protocol id + hash
plus ``frozen_manifest_hash`` binding the supervisor's frozen manifest.

Rules:
* every identity field is REQUIRED and has NO default;
* placeholder/wildcard values are rejected as GUESSED (we never guess
  a Reference);
* unless the block carries ``frozen: true`` (literal bool), the whole
  evaluation seam is blocked with REFERENCE_CONTRACT_UNFROZEN.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from ..static_llm.schemas import SchemaError, assert_admissible_provenance
from .canonical import sha256_hex

REFERENCE_CONTRACT_SCHEMA_VERSION = "e1_formal.reference_identity_contract.v1"


class ReferenceContractError(SchemaError):
    """Fail-closed Reference contract violation; ``code`` is greppable."""


class _RC:
    BAD_TYPE = "REFERENCE_CONTRACT_BAD_TYPE"
    MISSING_FIELD = "REFERENCE_CONTRACT_MISSING_FIELD"
    EMPTY_FIELD = "REFERENCE_CONTRACT_EMPTY_FIELD"
    UNKNOWN_FIELD = "REFERENCE_CONTRACT_UNKNOWN_FIELD"
    GUESSED_FORBIDDEN = "REFERENCE_CONTRACT_GUESSED_FORBIDDEN"
    UNFROZEN = "REFERENCE_CONTRACT_UNFROZEN"
    MANIFEST_HASH_MISMATCH = "REFERENCE_CONTRACT_MANIFEST_HASH_MISMATCH"
    BAD_HASH = "REFERENCE_CONTRACT_BAD_HASH"
    BAD_STEP = "REFERENCE_CONTRACT_BAD_STEP"


_HEX_DIGITS = frozenset("0123456789abcdef")

#: required non-empty string fields (no defaults, no guessing)
_STRING_FIELDS = (
    "candidate_id",
    "checkpoint_ref",
    "memory_semantics",
    "architecture_family",
    "architecture_version",
    "episode_reset_protocol_id",
    "source_commit",
)

#: required lowercase sha256-hex fields
_HASH_FIELDS = (
    "file_sha256",
    "params_sha256",
    "architecture_config_hash",
    "memory_semantics_hash",
    "episode_reset_protocol_hash",
    "frozen_manifest_hash",
)

_REQUIRED_FIELDS = _STRING_FIELDS + _HASH_FIELDS + ("global_step", "seed")
_OPTIONAL_FIELDS = ("total_env_steps", "provenance")
_CONTROL_FIELDS = ("frozen", "schema_version")
_ALL_FIELDS = frozenset(_REQUIRED_FIELDS + _OPTIONAL_FIELDS + _CONTROL_FIELDS)

#: placeholder / wildcard values => the Reference would be GUESSED.
_PLACEHOLDER_VALUES = frozenset(
    {
        "",
        "todo",
        "tbd",
        "latest",
        "auto",
        "none",
        "null",
        "unknown",
        "placeholder",
        "fixme",
        "unset",
        "default",
        "?",
        "-",
        "xxx",
        "dummy",
        "example",
    }
)


@dataclass(frozen=True)
class ReferenceIdentityContract:
    """Frozen identity of the Reference policy (identity only, no I/O)."""

    candidate_id: str
    checkpoint_ref: str
    file_sha256: str
    params_sha256: str
    architecture_family: str
    architecture_version: str
    architecture_config_hash: str
    memory_semantics: str
    memory_semantics_hash: str
    global_step: int
    source_commit: str
    seed: int
    episode_reset_protocol_id: str
    episode_reset_protocol_hash: str
    frozen_manifest_hash: str
    total_env_steps: Optional[int] = None
    provenance: Optional[str] = None
    schema_version: str = REFERENCE_CONTRACT_SCHEMA_VERSION


def _fail(code: str, message: str) -> ReferenceContractError:
    return ReferenceContractError(code, message)


def _is_sha256_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in _HEX_DIGITS for c in value)
    )


def _is_placeholder(value: str) -> bool:
    stripped = value.strip()
    if stripped.lower() in _PLACEHOLDER_VALUES:
        return True
    if stripped.startswith("<") and stripped.endswith(">"):
        return True
    if stripped.startswith("${"):
        return True
    return False


def consume_reference_identity_contract(
    mapping: Any, context: str
) -> ReferenceIdentityContract:
    """Parse the frozen Reference identity block fail-closed.

    Raises:
        ReferenceContractError: with a greppable code on ANY violation,
            including REFERENCE_CONTRACT_UNFROZEN when the block is not
            explicitly frozen.
        SchemaError: on inadmissible provenance (FORMAL_* rejected).
    """
    if not isinstance(mapping, Mapping):
        raise _fail(
            _RC.BAD_TYPE,
            f"{context}: reference_contract must be a mapping, got "
            f"{type(mapping).__name__}",
        )

    for key in mapping:
        if key not in _ALL_FIELDS:
            raise _fail(
                _RC.UNKNOWN_FIELD,
                f"{context}: unknown reference_contract field {key!r} "
                "(fail-closed)",
            )

    frozen = mapping.get("frozen")
    if frozen is not True:
        raise _fail(
            _RC.UNFROZEN,
            f"{context}: reference_contract is not frozen "
            f"(frozen={frozen!r}); the evaluation seam stays blocked; "
            "E1 never guesses a Reference",
        )

    values = {}
    for name in _STRING_FIELDS:
        if name not in mapping:
            raise _fail(_RC.MISSING_FIELD, f"{context}: missing {name!r}")
        value = mapping[name]
        if not isinstance(value, str):
            raise _fail(
                _RC.BAD_TYPE,
                f"{context}: field {name!r} must be str, got "
                f"{type(value).__name__}",
            )
        if not value.strip():
            raise _fail(_RC.EMPTY_FIELD, f"{context}: field {name!r} is empty")
        if _is_placeholder(value):
            raise _fail(
                _RC.GUESSED_FORBIDDEN,
                f"{context}: field {name!r} carries placeholder/wildcard "
                f"value {value!r}; the Reference must be frozen by the "
                "supervisor, never guessed",
            )
        values[name] = value.strip()

    for name in _HASH_FIELDS:
        if name not in mapping:
            raise _fail(_RC.MISSING_FIELD, f"{context}: missing {name!r}")
        value = mapping[name]
        if not _is_sha256_hex(value):
            raise _fail(
                _RC.BAD_HASH,
                f"{context}: field {name!r} must be lowercase sha256 hex "
                f"(64 chars), got {value!r}",
            )
        values[name] = value

    for name in ("global_step", "seed"):
        if name not in mapping:
            raise _fail(_RC.MISSING_FIELD, f"{context}: missing {name!r}")
        value = mapping[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise _fail(
                _RC.BAD_TYPE,
                f"{context}: field {name!r} must be int, got "
                f"{type(value).__name__}",
            )
        if value < 0:
            raise _fail(
                _RC.BAD_STEP,
                f"{context}: field {name!r} must be >= 0, got {value}",
            )
        values[name] = value

    total_env_steps = None
    if "total_env_steps" in mapping:
        value = mapping["total_env_steps"]
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise _fail(
                    _RC.BAD_STEP,
                    f"{context}: total_env_steps must be int >= 0, got "
                    f"{value!r}",
                )
            total_env_steps = value

    provenance = None
    if "provenance" in mapping and mapping["provenance"] is not None:
        provenance = assert_admissible_provenance(
            mapping["provenance"], context
        ).value

    return ReferenceIdentityContract(
        total_env_steps=total_env_steps, provenance=provenance, **values
    )


def verify_reference_manifest_bytes(
    contract: ReferenceIdentityContract, manifest_bytes: bytes, context: str
) -> None:
    """Assert the frozen manifest bytes match the contracted hash."""
    if not isinstance(manifest_bytes, (bytes, bytearray)):
        raise _fail(
            _RC.BAD_TYPE,
            f"{context}: manifest bytes required, got "
            f"{type(manifest_bytes).__name__}",
        )
    digest = sha256_hex(bytes(manifest_bytes))
    if digest != contract.frozen_manifest_hash:
        raise _fail(
            _RC.MANIFEST_HASH_MISMATCH,
            f"{context}: manifest sha256 {digest} != contracted "
            f"{contract.frozen_manifest_hash}",
        )
