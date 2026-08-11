"""E1 REAL_* flags (design D7): default False, no path can force True.

The three REAL_* flags are declared in the teacher config AND pinned in
the frozen manifest; at init they must agree exactly, otherwise
``FLAG_MANIFEST_MISMATCH`` fails closed. Values are booleans only — no
coercion, no defaults other than the all-False constructor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .schemas import E1Code, E1SchemaError

FLAG_FIELDS = (
    "real_envcoder_used",
    "real_student_reference_eval",
    "real_training_update_executed",
)


@dataclass(frozen=True)
class E1Flags:
    """All three REAL_* flags default to False; nothing sets them True."""

    real_envcoder_used: bool = False
    real_student_reference_eval: bool = False
    real_training_update_executed: bool = False


def parse_flags(mapping: Any, context: str) -> E1Flags:
    """Parse the flags block fail-closed (all fields required, bool only)."""
    if not isinstance(mapping, Mapping):
        raise E1SchemaError(
            E1Code.FLAGS_BAD_TYPE,
            f"{context}: flags block must be a mapping, got "
            f"{type(mapping).__name__}",
        )
    for key in mapping:
        if key not in FLAG_FIELDS:
            raise E1SchemaError(
                E1Code.FLAGS_UNKNOWN_FIELD,
                f"{context}: unknown flag {key!r} (fail-closed)",
            )
    values = {}
    for name in FLAG_FIELDS:
        if name not in mapping:
            raise E1SchemaError(
                E1Code.FLAGS_MISSING_FIELD,
                f"{context}: flag {name!r} missing (must be declared "
                "explicitly)",
            )
        value = mapping[name]
        if not isinstance(value, bool):
            raise E1SchemaError(
                E1Code.FLAGS_BAD_TYPE,
                f"{context}: flag {name!r} must be bool, got "
                f"{type(value).__name__} (no coercion)",
            )
        values[name] = value
    return E1Flags(**values)


def assert_flags_match_manifest(
    flags: E1Flags, manifest: Mapping[str, Any], context: str
) -> None:
    """Assert config flags agree with the frozen manifest (fail-closed).

    The manifest carries the flags under ``manifest["flags"]``.
    """
    manifest_flags = manifest.get("flags")
    if not isinstance(manifest_flags, Mapping):
        raise E1SchemaError(
            E1Code.FLAG_MANIFEST_MISMATCH,
            f"{context}: frozen manifest has no flags block",
        )
    for name in FLAG_FIELDS:
        config_value = getattr(flags, name)
        manifest_value = manifest_flags.get(name)
        if manifest_value != config_value or not isinstance(manifest_value, bool):
            raise E1SchemaError(
                E1Code.FLAG_MANIFEST_MISMATCH,
                f"{context}: flag {name!r} config={config_value!r} != "
                f"manifest={manifest_value!r}",
            )
