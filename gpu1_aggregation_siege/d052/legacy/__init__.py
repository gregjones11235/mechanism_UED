"""Legacy compatibility + protocol_version gate.

Everything in this subpackage is stdlib-only (no pydantic / numpy / yaml) so the
protocol gate and the canonical constants are importable and testable in any
environment, independent of the heavier schema layer in d052/schemas/.
"""

from d052.legacy.canonical_constants import (
    CANONICAL_PROTOCOL_VERSION,
    CANONICAL_V2_FIXED_CONFIG,
    LEGACY_PROTOCOL_VERSION,
    canonical_v2_config,
)
from d052.legacy.protocol_version import (
    D052ProtocolError,
    ProtocolContext,
    ProtocolVersion,
    assert_training_permitted,
    load_protocol_context,
    resolve_protocol_version,
)

__all__ = [
    "CANONICAL_PROTOCOL_VERSION",
    "CANONICAL_V2_FIXED_CONFIG",
    "LEGACY_PROTOCOL_VERSION",
    "canonical_v2_config",
    "D052ProtocolError",
    "ProtocolContext",
    "ProtocolVersion",
    "assert_training_permitted",
    "load_protocol_context",
    "resolve_protocol_version",
]
