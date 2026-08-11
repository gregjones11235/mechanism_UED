"""Stage 1: admissible Student behavior-failure evidence snapshot.

Builds the evidence that feeds the six-role Review Board. Rules:

* ONLY ``TRAINING`` / ``NORMAL_TRAINING_FEEDBACK`` evidence may enter
  (LLM-role admissible tier, enforced fail-closed at build time);
  FORMAL_* and CANDIDATE_EVALUATION items are rejected HERE, before
  any prompt is ever assembled;
* the item schema has NO tier/verdict field — upstream profiling
  verdicts (e.g. StudentProfileLog tiers) cannot enter by construction
  ("hands facts, not verdicts"); raw facts only: success-rate series,
  window metrics, behavioral fingerprint summaries;
* extraction from jax-side structures (archive, siege profile logs)
  happens in the gen_manager EDGE layer; this module consumes plain
  JSON-shaped facts and stays standard-library only;
* ``evolve_tasks`` resume-path metrics are ignored entirely by the
  teacher; anything offered here carries provenance and is re-verified
  (the teacher never trusts the call site).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

from .canonical import canonical_sha256
from .schemas import E1Code, E1SchemaError, assert_llm_role_admissible
from ..static_llm.guards import raise_if_forbidden

MAX_EVIDENCE_ITEMS = 64


class EvidenceError(E1SchemaError):
    """Fail-closed evidence violation; ``code`` is greppable."""


class _EvCode:
    EMPTY = "EVIDENCE_EMPTY"
    TOO_MANY = "EVIDENCE_TOO_MANY_ITEMS"
    BAD_TYPE = "EVIDENCE_BAD_TYPE"
    MISSING_FIELD = "EVIDENCE_MISSING_FIELD"
    EMPTY_FIELD = "EVIDENCE_EMPTY_FIELD"
    UNKNOWN_FIELD = "EVIDENCE_UNKNOWN_FIELD"
    UNSUPPORTED_FACTS = "EVIDENCE_UNSUPPORTED_FACTS"


_REQUIRED_FIELDS = ("source", "session_idx", "provenance", "facts")
_ALL_FIELDS = frozenset(_REQUIRED_FIELDS)

_KNOWN_SOURCES = frozenset(
    {
        "archive.performance_history",
        "training_window.session_metrics",
        "behavior_fingerprint.summary",
    }
)


@dataclass(frozen=True)
class EvidenceItem:
    """One admissible evidence record (raw facts only, no verdicts)."""

    source: str
    session_idx: int
    provenance: str
    facts_sha: str  # canonical sha256 of the facts mapping (identity)


@dataclass(frozen=True)
class EvidenceSnapshot:
    """Immutable, hash-identified evidence bundle for one review window."""

    items: Tuple[EvidenceItem, ...]
    evidence_hash: str


def _fail(code: str, message: str) -> EvidenceError:
    return EvidenceError(code, message)


def _validate_facts(facts: Any, context: str) -> str:
    """Validate JSON-shaped facts and return their canonical sha256."""
    if not isinstance(facts, Mapping):
        raise _fail(
            _EvCode.UNSUPPORTED_FACTS,
            f"{context}: facts must be a mapping, got {type(facts).__name__}",
        )
    try:
        return canonical_sha256(dict(facts))
    except E1SchemaError as e:
        raise _fail(
            _EvCode.UNSUPPORTED_FACTS,
            f"{context}: facts are not canonical-encodable: {e}",
        ) from e


def build_evidence_snapshot(
    raw_items: Sequence[Mapping[str, Any]], context: str
) -> EvidenceSnapshot:
    """Build the snapshot fail-closed from raw evidence mappings.

    Each raw item must carry exactly {source, session_idx, provenance,
    facts}. Provenance is re-verified against the LLM-role admissible
    tier; facts pass the deterministic content guards (F1-F7) so a
    poisoned training-window record cannot smuggle forbidden content
    toward a prompt.
    """
    if not isinstance(raw_items, (list, tuple)):
        raise _fail(
            _EvCode.BAD_TYPE,
            f"{context}: evidence items must be a sequence, got "
            f"{type(raw_items).__name__}",
        )
    if len(raw_items) == 0:
        raise _fail(_EvCode.EMPTY, f"{context}: evidence snapshot is empty")
    if len(raw_items) > MAX_EVIDENCE_ITEMS:
        raise _fail(
            _EvCode.TOO_MANY,
            f"{context}: {len(raw_items)} evidence items > "
            f"{MAX_EVIDENCE_ITEMS}",
        )

    items = []
    for i, raw in enumerate(raw_items):
        item_ctx = f"{context}.items[{i}]"
        if not isinstance(raw, Mapping):
            raise _fail(
                _EvCode.BAD_TYPE,
                f"{item_ctx}: item must be a mapping, got "
                f"{type(raw).__name__}",
            )
        for key in raw:
            if key not in _ALL_FIELDS:
                raise _fail(
                    _EvCode.UNKNOWN_FIELD,
                    f"{item_ctx}: unknown evidence field {key!r} "
                    "(fail-closed; no tier/verdict fields admissible)",
                )
        for name in _REQUIRED_FIELDS:
            if name not in raw:
                raise _fail(
                    _EvCode.MISSING_FIELD,
                    f"{item_ctx}: missing field {name!r}",
                )

        source = raw["source"]
        if not isinstance(source, str) or not source.strip():
            raise _fail(
                _EvCode.EMPTY_FIELD, f"{item_ctx}: source must be non-empty str"
            )
        if source not in _KNOWN_SOURCES:
            raise _fail(
                _EvCode.UNKNOWN_FIELD,
                f"{item_ctx}: unknown evidence source {source!r} "
                f"(allowed: {sorted(_KNOWN_SOURCES)})",
            )

        session_idx = raw["session_idx"]
        if isinstance(session_idx, bool) or not isinstance(session_idx, int):
            raise _fail(
                _EvCode.BAD_TYPE,
                f"{item_ctx}: session_idx must be int, got "
                f"{type(session_idx).__name__}",
            )
        if session_idx < 0:
            raise _fail(
                _EvCode.BAD_TYPE,
                f"{item_ctx}: session_idx must be >= 0, got {session_idx}",
            )

        # Re-verify provenance HERE (never trust the call site). Fails
        # closed for FORMAL_* / CANDIDATE_EVALUATION / unknown, and is
        # re-raised as the evidence-layer error type with the SAME code.
        try:
            provenance = assert_llm_role_admissible(raw["provenance"], item_ctx)
        except E1SchemaError as e:
            raise _fail(
                e.code,
                f"{item_ctx}: evidence provenance rejected ({e.code})",
            ) from e

        facts = raw["facts"]
        facts_sha = _validate_facts(facts, item_ctx)
        # Content guards: poisoned records fail closed before any prompt.
        raise_if_forbidden(dict(facts), f"evidence[{i}]")

        items.append(
            EvidenceItem(
                source=source,
                session_idx=session_idx,
                provenance=provenance,
                facts_sha=facts_sha,
            )
        )

    snapshot_payload = [
        {
            "source": it.source,
            "session_idx": it.session_idx,
            "provenance": it.provenance,
            "facts_sha": it.facts_sha,
        }
        for it in items
    ]
    return EvidenceSnapshot(
        items=tuple(items), evidence_hash=canonical_sha256(snapshot_payload)
    )


def render_evidence_for_prompt(snapshot: EvidenceSnapshot) -> str:
    """Deterministic prompt rendering of the snapshot (facts only).

    The rendering lists sources, sessions, provenance labels and facts
    hashes; the full facts are re-validated by the guards before the
    board consumes them. No tier/verdict vocabulary exists in the item
    schema, so none can leak here.
    """
    lines = ["EVIDENCE_SNAPSHOT hash=" + snapshot.evidence_hash]
    for i, item in enumerate(snapshot.items):
        lines.append(
            f"[{i}] source={item.source} session={item.session_idx} "
            f"provenance={item.provenance} facts_sha256={item.facts_sha}"
        )
    return "\n".join(lines)
