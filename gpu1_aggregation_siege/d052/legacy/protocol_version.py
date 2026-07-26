"""protocol_version gate + legacy/canonical_v2 compatibility layer.

This is the FIRST thing every D052 entry point must call. It enforces the hard
versioning discipline of D052_CANONICAL_IN_PLACE_REFACTOR:

  1. protocol_version is REQUIRED. A config that OMITS the field FAILS
     (MISSING_PROTOCOL_VERSION). It is NEVER silently defaulted during parsing.
     ("canonical_v2 is the default" is an AUTHORING convention for new configs,
     not a parse-time fallback.)
  2. Only two values are legal: "legacy" and "canonical_v2". Anything else
     (including wrong case) -> UNKNOWN_PROTOCOL_VERSION. We do NOT case-fold or
     otherwise coerce, honoring NO_SILENT_SCHEMA_COERCION.
  3. "legacy" requires an EXPLICIT opt-in (allow_legacy_d052=True, surfaced as the
     CLI flag --allow-legacy-d052). Without it -> LEGACY_NOT_AUTHORIZED.
  4. legacy configs are NEVER auto-upgraded to canonical_v2 and NEVER coerced.
     Loading a legacy config returns a context flagged legacy with a warning and
     no fixed canonical config attached.
  5. Under legacy, training is NEVER permitted (this phase runs zero training):
     assert_training_permitted() raises LEGACY_TRAINING_FORBIDDEN.

Stdlib only (no pydantic / numpy / yaml) so the gate is testable in isolation and
importable before the heavier schema layer is available.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional

from d052.legacy.canonical_constants import (
    CANONICAL_PROTOCOL_VERSION,
    LEGACY_PROTOCOL_VERSION,
    LEGAL_PROTOCOL_VERSIONS,
    canonical_v2_config,
)

#: Distinct process exit code for protocol-gate failures (CLI maps to this).
PROTOCOL_GATE_EXIT_CODE = 2

#: Config key that carries the protocol version.
PROTOCOL_VERSION_KEY = "protocol_version"

# Sentinel distinguishing "key absent" from "key present with value None".
_MISSING = object()


class D052ProtocolError(Exception):
    """Hard protocol-gate failure. Carries a stable machine-readable ``code``.

    The CLI catches this and exits with ``PROTOCOL_GATE_EXIT_CODE`` (nonzero).
    """

    #: stable error codes (do not rename without bumping the protocol)
    MISSING_PROTOCOL_VERSION = "MISSING_PROTOCOL_VERSION"
    UNKNOWN_PROTOCOL_VERSION = "UNKNOWN_PROTOCOL_VERSION"
    LEGACY_NOT_AUTHORIZED = "LEGACY_NOT_AUTHORIZED"
    LEGACY_TRAINING_FORBIDDEN = "LEGACY_TRAINING_FORBIDDEN"
    INVALID_CONFIG_TYPE = "INVALID_CONFIG_TYPE"

    exit_code = PROTOCOL_GATE_EXIT_CODE

    def __init__(self, code: str, message: str, *, source: str = "<config>",
                 offending_value: Any = _MISSING) -> None:
        self.code = code
        self.source = source
        self.offending_value = (
            None if offending_value is _MISSING else offending_value)
        full = f"[{code}] {message} (source={source})"
        if offending_value is not _MISSING:
            full += f" (offending_value={offending_value!r})"
        super().__init__(full)


class ProtocolVersion(Enum):
    """The two legal protocol versions."""

    LEGACY = LEGACY_PROTOCOL_VERSION
    CANONICAL_V2 = CANONICAL_PROTOCOL_VERSION

    @property
    def is_canonical(self) -> bool:
        return self is ProtocolVersion.CANONICAL_V2

    @property
    def is_legacy(self) -> bool:
        return self is ProtocolVersion.LEGACY


@dataclass(frozen=True)
class ProtocolContext:
    """Resolved, immutable protocol decision for one config/run.

    ``fixed_config`` is populated ONLY for canonical_v2 (a copy of the frozen
    canonical constants). It is None for legacy -- legacy is never upgraded.
    """

    version: ProtocolVersion
    source: str
    allow_legacy: bool
    fixed_config: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def is_canonical(self) -> bool:
        return self.version.is_canonical

    @property
    def is_legacy(self) -> bool:
        return self.version.is_legacy

    def assert_canonical(self, *, purpose: str = "operation") -> None:
        """Raise unless this context is canonical_v2."""
        if not self.is_canonical:
            raise D052ProtocolError(
                D052ProtocolError.UNKNOWN_PROTOCOL_VERSION,
                f"{purpose} requires protocol_version == "
                f"'{CANONICAL_PROTOCOL_VERSION}'; got '{self.version.value}'",
                source=self.source,
                offending_value=self.version.value,
            )


def _warn_and_record(message: str, sink: List[str]) -> None:
    sink.append(message)
    warnings.warn(message, stacklevel=3)


def resolve_protocol_version(
    raw_value: Any,
    *,
    allow_legacy_d052: bool = False,
    source: str = "<config>",
    warning_sink: Optional[List[str]] = None,
) -> ProtocolVersion:
    """Resolve a raw protocol_version value to a ProtocolVersion, or raise.

    Args:
        raw_value: the value of the config's protocol_version field. ``None`` or
            the internal MISSING sentinel both count as "missing".
        allow_legacy_d052: explicit opt-in for legacy (the --allow-legacy-d052
            CLI flag). Required for legacy; irrelevant for canonical_v2.
        source: human-readable origin of the config (for error messages/audit).
        warning_sink: optional list to receive warning strings (for the audit
            trail); warnings are ALSO emitted via ``warnings.warn``.

    Raises:
        D052ProtocolError: with a stable ``code`` on any violation.
    """
    sink = warning_sink if warning_sink is not None else []

    # 1. Missing -> hard fail. Never defaulted at parse time.
    if raw_value is None or raw_value is _MISSING:
        raise D052ProtocolError(
            D052ProtocolError.MISSING_PROTOCOL_VERSION,
            f"'{PROTOCOL_VERSION_KEY}' is required and was not provided; "
            f"refusing to guess a protocol version",
            source=source,
        )

    # 2. Type guard (NO_SILENT_SCHEMA_COERCION): must be a str, exactly.
    if not isinstance(raw_value, str):
        raise D052ProtocolError(
            D052ProtocolError.UNKNOWN_PROTOCOL_VERSION,
            f"'{PROTOCOL_VERSION_KEY}' must be a string, got "
            f"{type(raw_value).__name__}; will not coerce",
            source=source,
            offending_value=raw_value,
        )

    value = raw_value.strip()

    # 3. Exact canonical-string match. No case-folding (that would be coercion).
    if value not in LEGAL_PROTOCOL_VERSIONS:
        raise D052ProtocolError(
            D052ProtocolError.UNKNOWN_PROTOCOL_VERSION,
            f"'{PROTOCOL_VERSION_KEY}' must be one of "
            f"{sorted(LEGAL_PROTOCOL_VERSIONS)}; got {value!r} "
            f"(no auto-correction / no case-folding)",
            source=source,
            offending_value=raw_value,
        )

    # 4. legacy requires explicit opt-in.
    if value == LEGACY_PROTOCOL_VERSION:
        if not allow_legacy_d052:
            raise D052ProtocolError(
                D052ProtocolError.LEGACY_NOT_AUTHORIZED,
                "protocol_version == 'legacy' requires the explicit "
                "--allow-legacy-d052 opt-in; legacy runs are read-only and "
                "never train",
                source=source,
                offending_value=raw_value,
            )
        _warn_and_record(
            f"D052 LEGACY protocol explicitly enabled for {source}. Legacy is "
            f"deprecated, read-only, and will NOT be upgraded to "
            f"'{CANONICAL_PROTOCOL_VERSION}'. No training is permitted.",
            sink,
        )
        return ProtocolVersion.LEGACY

    # 5. canonical_v2.
    return ProtocolVersion.CANONICAL_V2


def load_protocol_context(
    config: Mapping[str, Any],
    *,
    allow_legacy_d052: bool = False,
    source: str = "<config>",
) -> ProtocolContext:
    """Parse a config mapping into an immutable ProtocolContext, or raise.

    This is the primary entry point for callers. It extracts
    ``protocol_version`` (failing if absent), resolves it, and -- for
    canonical_v2 -- attaches a fresh copy of the frozen canonical fixed config.
    Legacy contexts get ``fixed_config=None`` and are NEVER upgraded.
    """
    if not isinstance(config, Mapping):
        raise D052ProtocolError(
            D052ProtocolError.INVALID_CONFIG_TYPE,
            f"config must be a mapping, got {type(config).__name__}",
            source=source,
        )

    warning_sink: List[str] = []
    raw = config.get(PROTOCOL_VERSION_KEY, _MISSING)
    version = resolve_protocol_version(
        raw,
        allow_legacy_d052=allow_legacy_d052,
        source=source,
        warning_sink=warning_sink,
    )

    fixed_config = canonical_v2_config() if version.is_canonical else None
    return ProtocolContext(
        version=version,
        source=source,
        allow_legacy=allow_legacy_d052,
        fixed_config=fixed_config,
        warnings=warning_sink,
    )


def assert_training_permitted(context: ProtocolContext) -> None:
    """Version-level training gate.

    Raises LEGACY_TRAINING_FORBIDDEN for legacy contexts. (canonical_v2 training
    is additionally gated by the per-cell authorization layer in d052/cells/;
    that stronger gate is enforced there, not here.)
    """
    if context.is_legacy:
        raise D052ProtocolError(
            D052ProtocolError.LEGACY_TRAINING_FORBIDDEN,
            "training is NEVER permitted under protocol_version == 'legacy'; "
            "this phase performs zero long training runs",
            source=context.source,
        )
