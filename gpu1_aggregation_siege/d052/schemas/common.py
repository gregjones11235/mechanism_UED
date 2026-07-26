"""Shared schema base + validators for all canonical_v2 D052 data contracts.

Every schema:
  * forbids extra/unknown fields (extra="forbid") -> NO_SILENT_SCHEMA_COERCION;
  * carries protocol_version pinned to "canonical_v2" (legacy records cannot be
    expressed in these types at all);
  * uses fail-closed validators that surface the registry/gate error CODE inside
    the pydantic ValidationError message (so a teammate can grep the code).

pydantic v2 (2.12.5 in-env). No numpy at this layer.
"""
from __future__ import annotations

import math
import re
from typing import Any

from pydantic import BaseModel, ConfigDict

from d052.achievements import REGISTRY, AchievementError
from d052.legacy.canonical_constants import CANONICAL_PROTOCOL_VERSION

#: sha256 hex digest (lowercase, 64 chars).
SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


def is_sha256_hex(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA256_HEX_RE.match(value))


def validate_sha256_hex(value: Any, field_name: str) -> str:
    if not is_sha256_hex(value):
        raise ValueError(
            f"INVALID_HASH: {field_name} must be a lowercase 64-char sha256 hex "
            f"digest, got {value!r}")
    return value


def validate_finite(value: Any, field_name: str) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"NON_FINITE: {field_name} must be finite, got {value!r}")
    return v


def resolve_canonical_name(value: Any) -> str:
    """Resolve an achievement name via the audited alias allow-list, or raise.

    Raises ValueError (-> pydantic ValidationError) carrying the registry error
    CODE so the failure is greppable. Unknown / empty / wrong-type all fail closed.
    """
    try:
        return REGISTRY.resolve(value)
    except AchievementError as e:
        raise ValueError(f"{e.code}: {e}") from e


class CanonicalModel(BaseModel):
    """Base for all canonical_v2 D052 schemas."""

    model_config = ConfigDict(
        extra="forbid",          # NO unknown fields
        validate_assignment=True,
        str_strip_whitespace=False,  # do NOT silently trim (coercion)
    )

    #: Self-identifies the record as canonical_v2. Legacy cannot be expressed here.
    protocol_version: str = CANONICAL_PROTOCOL_VERSION
