"""TRAINING_DISCOVERY capture provenance with formal-data isolation (condition 2).

Frontier-collection rollouts must carry TRAINING_DISCOVERY provenance and must
stay structurally isolated from the frozen formal evaluation banks/worlds:
formal bank ids and formal world-set ids can never enter capture requests,
the FrontierArchive, the InvocationGate, an LLM prompt, or the selector.

This module is contract-level isolation: it validates capture provenance
fail-closed.  No real collection run happens this round.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

from .errors import ProvenanceViolationError
from .provenance import DataSource, FormalDataLeakageGuard


class DiscoveryProvenance(str, Enum):
    """Legal provenance classes for frontier capture requests."""

    TRAINING_DISCOVERY = "TRAINING_DISCOVERY"
    # Explicitly-labelled synthetic fixtures only (contract tests).  Never a
    # legal source for real capture: validate_capture_provenance rejects it
    # unless the caller opts in with allow_synthetic_fixture=True.
    SYNTHETIC_FIXTURE = "SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT"


# Formal identifiers that must never appear in capture provenance.
_FORMAL_BANK_MARKERS = ("FORMAL_FRONT", "FORMAL_BACK", "FORMAL_FULL")
_FORMAL_WORLD_MARKERS = ("FORMAL_WORLD", "FORMAL_EVAL_WORLD")


@dataclass(frozen=True)
class CaptureProvenance:
    """Strongly-typed provenance attached to every frontier capture request."""

    provenance: DiscoveryProvenance
    rollout_protocol_id: str
    world_set_hash: str
    bank_refs: tuple[str, ...] = ()
    world_set_id: str = ""
    notes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.provenance, str):
            object.__setattr__(self, "provenance", DiscoveryProvenance(self.provenance))
        object.__setattr__(self, "bank_refs", tuple(self.bank_refs))


def _contains_formal_marker(text: str) -> str | None:
    upper = text.upper()
    for marker in _FORMAL_BANK_MARKERS + _FORMAL_WORLD_MARKERS:
        if marker in upper:
            return marker
    return None


def validate_capture_provenance(cap: CaptureProvenance, *, allow_synthetic_fixture: bool = False) -> None:
    """Fail-closed validation of capture provenance.

    Raises ProvenanceViolationError when:
    - provenance is not TRAINING_DISCOVERY (synthetic only via explicit opt-in);
    - rollout protocol id or world set hash is missing;
    - any bank ref / world set id / note carries a FORMAL bank or formal world
      identifier (structural isolation from frozen evaluation assets);
    - the formal leakage guard forbids the combination for frontier consumers.
    """
    if cap.provenance is DiscoveryProvenance.SYNTHETIC_FIXTURE:
        if not allow_synthetic_fixture:
            raise ProvenanceViolationError(
                "SYNTHETIC_FIXTURE provenance is only legal in explicitly labelled contract tests")
    elif cap.provenance is not DiscoveryProvenance.TRAINING_DISCOVERY:
        raise ProvenanceViolationError(
            f"illegal capture provenance: {cap.provenance!r} (TRAINING_DISCOVERY required)")

    if not cap.rollout_protocol_id:
        raise ProvenanceViolationError("capture provenance requires rollout_protocol_id")
    if not cap.world_set_hash:
        raise ProvenanceViolationError("capture provenance requires world_set_hash")

    for ref in cap.bank_refs:
        marker = _contains_formal_marker(str(ref))
        if marker is not None:
            raise ProvenanceViolationError(f"formal bank identifier {marker!r} cannot enter capture provenance")
        # Cross-check against the shared leakage guard: formal sources may not
        # feed any frontier consumer.
        for formal in FormalDataLeakageGuard.FORBIDDEN:
            if str(ref).upper() == formal.value:
                FormalDataLeakageGuard.assert_allowed(formal, consumer="FrontierArchive")

    if cap.world_set_id:
        marker = _contains_formal_marker(cap.world_set_id)
        if marker is not None:
            raise ProvenanceViolationError(f"formal world identifier {marker!r} cannot enter capture provenance")

    for key, value in cap.notes.items():
        marker = _contains_formal_marker(str(key)) or _contains_formal_marker(str(value))
        if marker is not None:
            raise ProvenanceViolationError(f"formal identifier {marker!r} cannot enter capture provenance notes")


def discovery_source_for(provenance: DiscoveryProvenance) -> DataSource:
    """Map capture provenance onto the shared DataSource taxonomy."""
    if provenance is DiscoveryProvenance.TRAINING_DISCOVERY:
        return DataSource.TRAINING_FRONTIER_CAPTURE
    return DataSource.SYNTHETIC_TEST


def assert_not_formal(source: DataSource | str, consumer: str) -> None:
    """Convenience bridge: reuse FormalDataLeakageGuard for arbitrary consumers."""
    FormalDataLeakageGuard.assert_allowed(DataSource(source), consumer)
