"""CC4 shared StudentInitContract — identity-only thin consumer (C2 / D14).

This module is the THIN CONSUMER side of the single shared CC4
StudentAdapter / StudentInitContract. It is deliberately:

* NOT a loader — there is no checkpoint-path field anywhere in this
  module, so file I/O toward a checkpoint is inexpressible here by
  construction; resolution/loading happens ONLY inside the CC4 adapter.
* NOT a registry — there is no lookup table, no discovery, no second
  student mechanism. Exactly one pinned candidate id is consumed.
* Pure standard library (plus the committed sibling ``schemas`` module)
  — no jax / flax / optax / orbax / pickle / torch imports, enforced by
  an AST test.

All identity values must be supplied verbatim by the frozen CC4
contract mapping; NOTHING is defaulted, guessed, or derived. Every
violation fails closed through a greppable code defined in THIS file
(the committed ``schemas.py`` is not modified).

Fail-closed codes::

    STUDENT_CONTRACT_BAD_TYPE        mapping/field has the wrong type
    STUDENT_CONTRACT_MISSING_FIELD   required field absent
    STUDENT_CONTRACT_EMPTY_FIELD     string field empty/blank
    STUDENT_CONTRACT_UNKNOWN_FIELD   unexpected key (fail-closed, no
                                     silent forward-compat coercion)
    STUDENT_CONTRACT_BAD_STEP        integer counter negative
    STUDENT_CONTRACT_BAD_HASH        hash field is not lowercase sha256 hex
    STUDENT_ID_MISMATCH              candidate_id != pinned candidate
    (provenance violations reuse ``SchemaError`` codes via
     ``assert_admissible_provenance``: FORMAL_* rejected fail-closed)
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping, Optional

from .schemas import SchemaError, assert_admissible_provenance

#: The ONE strong-Student candidate this teacher may consume (supervisor
#: directive 2026-08-03). Hard-pinned; never configurable, never guessed.
PINNED_STUDENT_CANDIDATE_ID = "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304"

CONTRACT_SCHEMA_VERSION = "static_llm_ued.student_init_contract.v1"


class StudentContractError(SchemaError):
    """Fail-closed contract violation; ``code`` is greppable."""


class _Code:
    BAD_TYPE = "STUDENT_CONTRACT_BAD_TYPE"
    MISSING_FIELD = "STUDENT_CONTRACT_MISSING_FIELD"
    EMPTY_FIELD = "STUDENT_CONTRACT_EMPTY_FIELD"
    UNKNOWN_FIELD = "STUDENT_CONTRACT_UNKNOWN_FIELD"
    BAD_STEP = "STUDENT_CONTRACT_BAD_STEP"
    BAD_HASH = "STUDENT_CONTRACT_BAD_HASH"
    ID_MISMATCH = "STUDENT_ID_MISMATCH"


#: Required string fields (non-empty after strip).
_STRING_FIELDS = (
    "candidate_id",
    "architecture_family",
    "architecture_version",
    "checkpoint_format",
    "source_commit",
    "adapter_id",
    "adapter_version",
)

#: Required non-negative integer counters.
_STEP_FIELDS = ("checkpoint_global_step", "total_env_steps")

#: Required lowercase sha256-hex tree hashes.
_HASH_FIELDS = ("parameter_tree_hash", "optimizer_tree_hash")

_REQUIRED_FIELDS = _STRING_FIELDS + _STEP_FIELDS + _HASH_FIELDS

#: Optional field(s); tolerated absence, validated when present.
_OPTIONAL_FIELDS = ("provenance",)

_ALL_FIELDS = frozenset(_REQUIRED_FIELDS + _OPTIONAL_FIELDS)

_HEX_DIGITS = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class StudentInitContract:
    """Identity-only snapshot of the pinned strong Student.

    IDENTITY ONLY: every field is a name, a version string, a counter,
    or a content hash. There is NO path field of any kind — checkpoint
    location is the CC4 adapter's business and cannot be expressed here.
    """

    candidate_id: str
    architecture_family: str
    architecture_version: str
    checkpoint_format: str
    checkpoint_global_step: int
    total_env_steps: int
    source_commit: str
    parameter_tree_hash: str
    optimizer_tree_hash: str
    adapter_id: str
    adapter_version: str
    provenance: Optional[str] = None
    schema_version: str = CONTRACT_SCHEMA_VERSION


def _fail(code: str, message: str) -> "StudentContractError":
    return StudentContractError(code, message)


def _is_sha256_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in _HEX_DIGITS for c in value)
    )


def consume_student_init_contract(
    mapping: Any, context: str
) -> StudentInitContract:
    """Parse the frozen CC4 contract mapping fail-closed (identity only).

    Args:
        mapping: the CC4-provided contract mapping. Must be a Mapping
            with exactly the known keys (extra keys fail closed).
        context: human-readable call site (error messages / audit).

    Raises:
        StudentContractError: with a greppable code on ANY violation.
        SchemaError: on inadmissible provenance (FORMAL_* rejected).
    """
    if not isinstance(mapping, Mapping):
        raise _fail(
            _Code.BAD_TYPE,
            f"{context}: contract must be a mapping, got "
            f"{type(mapping).__name__}",
        )

    known = _ALL_FIELDS
    for key in mapping:
        if key not in known:
            raise _fail(
                _Code.UNKNOWN_FIELD,
                f"{context}: unknown contract field {key!r} "
                "(fail-closed; no silent forward-compat coercion)",
            )

    values = {}
    for name in _STRING_FIELDS:
        if name not in mapping:
            raise _fail(_Code.MISSING_FIELD, f"{context}: missing {name!r}")
        value = mapping[name]
        if not isinstance(value, str):
            raise _fail(
                _Code.BAD_TYPE,
                f"{context}: field {name!r} must be str, got "
                f"{type(value).__name__}",
            )
        if not value.strip():
            raise _fail(
                _Code.EMPTY_FIELD, f"{context}: field {name!r} is empty"
            )
        values[name] = value.strip()

    for name in _STEP_FIELDS:
        if name not in mapping:
            raise _fail(_Code.MISSING_FIELD, f"{context}: missing {name!r}")
        value = mapping[name]
        # bool is an int subclass: reject explicitly (no coercion).
        if isinstance(value, bool) or not isinstance(value, int):
            raise _fail(
                _Code.BAD_TYPE,
                f"{context}: field {name!r} must be int, got "
                f"{type(value).__name__}",
            )
        if value < 0:
            raise _fail(
                _Code.BAD_STEP,
                f"{context}: field {name!r} must be >= 0, got {value}",
            )
        values[name] = value

    for name in _HASH_FIELDS:
        if name not in mapping:
            raise _fail(_Code.MISSING_FIELD, f"{context}: missing {name!r}")
        value = mapping[name]
        if not _is_sha256_hex(value):
            raise _fail(
                _Code.BAD_HASH,
                f"{context}: field {name!r} must be lowercase sha256 hex "
                f"(64 chars), got {value!r}",
            )
        values[name] = value

    provenance = None
    if "provenance" in mapping:
        raw = mapping["provenance"]
        if raw is not None:
            # Raises SchemaError fail-closed for FORMAL_*/unknown/missing.
            provenance = assert_admissible_provenance(raw, context).value

    return StudentInitContract(provenance=provenance, **values)


def assert_pinned_candidate(
    contract: StudentInitContract,
    pinned_id: str = PINNED_STUDENT_CANDIDATE_ID,
) -> None:
    """Assert the contract refers to the ONE pinned strong Student.

    Raises:
        StudentContractError: STUDENT_ID_MISMATCH otherwise.
    """
    if contract.candidate_id != pinned_id:
        raise _fail(
            _Code.ID_MISMATCH,
            f"contract candidate_id {contract.candidate_id!r} != pinned "
            f"{pinned_id!r}; no other student identity is admissible",
        )


def contract_field_names() -> tuple:
    """Field names of the contract (identity-only audit helper)."""
    return tuple(f.name for f in fields(StudentInitContract))
