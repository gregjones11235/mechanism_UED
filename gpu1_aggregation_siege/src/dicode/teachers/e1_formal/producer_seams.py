"""CC2 follow-up P0-14: failure-pattern and curriculum-drift producers.

The ``new_failure_pattern`` and ``curriculum_drift`` gate signals
previously had NO data producer and were computed False with
``SIGNAL_NO_PRODUCER``. This module is the producer seam: two explicit
producer states, the signed data contracts, a producer registry, and
the gate-signal derivation that consumes ONLY signed objects.

States (greppable, consumed by the readiness/gate surfaces):

* ``INVOCATION_THRESHOLDS_UNFROZEN`` — a threshold-driven derivation
  was attempted while the supervisor-frozen thresholds were absent
  (NO fabricated defaults, ever);
* ``FAILURE_PATTERN_PRODUCER_UNBOUND`` — no failure-pattern producer
  is registered yet;
* ``CURRICULUM_DRIFT_PRODUCER_UNBOUND`` — no curriculum-drift producer
  is registered yet.

This round: the registry is EMPTY, so both producers stay honestly
UNBOUND; the contracts + derivation are exercised by the TEST_ONLY
contract tests only. A producer, once registered, mints the signed
data object; the gate signal derivation never trusts a caller-built
mapping.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

from .canonical import canonical_sha256
from .schemas import E1SchemaError

#: gate states (greppable)
INVOCATION_THRESHOLDS_UNFROZEN = "INVOCATION_THRESHOLDS_UNFROZEN"
FAILURE_PATTERN_PRODUCER_UNBOUND = "FAILURE_PATTERN_PRODUCER_UNBOUND"
CURRICULUM_DRIFT_PRODUCER_UNBOUND = "CURRICULUM_DRIFT_PRODUCER_UNBOUND"

#: version pins
FAILURE_PATTERN_DETECTOR_VERSION = "e1-failure-pattern-detector-v1"
CURRICULUM_DRIFT_METRIC_VERSION = "e1-curriculum-drift-metric-v1"

# fail-closed codes (greppable)
PRODUCER_BAD_TYPE = "PRODUCER_BAD_TYPE"
PRODUCER_UNBOUND = "PRODUCER_UNBOUND"
PRODUCER_SIGNAL_BAD = "PRODUCER_SIGNAL_BAD"
PRODUCER_THRESHOLD_UNFROZEN = "PRODUCER_THRESHOLD_UNFROZEN"


class ProducerSeamError(E1SchemaError):
    """Fail-closed producer-seam violation; ``code`` is greppable."""


# ---------------------------------------------------------------------------
# signed data contracts
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FailurePatternFingerprint:
    """One failure-pattern fingerprint (immutable, hash-bound)."""

    behavior_clip_ids: Tuple[str, ...]
    behavior_clip_hash: str
    detector_version: str
    failure_family: str
    novelty: float
    student_checkpoint_hash: str
    window_hash: str
    provenance_hash: str


@dataclass(frozen=True)
class CurriculumCompositionHistory:
    """The batch-composition history record (immutable, hash-bound)."""

    prior_batch_hashes: Tuple[str, ...]
    family_composition: Tuple[Tuple[str, int], ...]
    bucket_composition: Tuple[Tuple[str, int], ...]
    anchor_share: float
    drift_metric: float
    student_checkpoint_progression: Tuple[str, ...]
    provenance_hash: str


def _fingerprint_hash(fingerprint: FailurePatternFingerprint) -> str:
    return canonical_sha256(
        {
            "detector_version": fingerprint.detector_version,
            "behavior_clip_ids": list(fingerprint.behavior_clip_ids),
            "behavior_clip_hash": fingerprint.behavior_clip_hash,
            "failure_family": fingerprint.failure_family,
            "novelty": fingerprint.novelty,
            "student_checkpoint_hash": (
                fingerprint.student_checkpoint_hash
            ),
            "window_hash": fingerprint.window_hash,
        }
    )


def _history_hash(history: CurriculumCompositionHistory) -> str:
    return canonical_sha256(
        {
            "metric_version": CURRICULUM_DRIFT_METRIC_VERSION,
            "prior_batch_hashes": list(history.prior_batch_hashes),
            "family_composition": [
                list(pair) for pair in history.family_composition
            ],
            "bucket_composition": [
                list(pair) for pair in history.bucket_composition
            ],
            "anchor_share": history.anchor_share,
            "drift_metric": history.drift_metric,
            "student_checkpoint_progression": list(
                history.student_checkpoint_progression
            ),
        }
    )


# ---------------------------------------------------------------------------
# producer registry (EMPTY this round -> UNBOUND states)
# ---------------------------------------------------------------------------
_FAILURE_PATTERN_PRODUCER: Optional[Callable[..., Any]] = None
_CURRICULUM_DRIFT_PRODUCER: Optional[Callable[..., Any]] = None


def register_failure_pattern_producer(producer: Any) -> None:
    """Register the failure-pattern producer (a callable minting
    FailurePatternFingerprint objects; never a string/name)."""
    global _FAILURE_PATTERN_PRODUCER
    if isinstance(producer, str) or not callable(producer):
        raise ProducerSeamError(
            PRODUCER_BAD_TYPE,
            "producer_seams: failure-pattern producer must be a "
            "callable, never a string contract name",
        )
    _FAILURE_PATTERN_PRODUCER = producer


def register_curriculum_drift_producer(producer: Any) -> None:
    """Register the curriculum-drift producer (a callable minting
    CurriculumCompositionHistory objects; never a string/name)."""
    global _CURRICULUM_DRIFT_PRODUCER
    if isinstance(producer, str) or not callable(producer):
        raise ProducerSeamError(
            PRODUCER_BAD_TYPE,
            "producer_seams: curriculum-drift producer must be a "
            "callable, never a string contract name",
        )
    _CURRICULUM_DRIFT_PRODUCER = producer


def resolve_failure_pattern_producer() -> Optional[Callable[..., Any]]:
    return _FAILURE_PATTERN_PRODUCER


def resolve_curriculum_drift_producer() -> Optional[Callable[..., Any]]:
    return _CURRICULUM_DRIFT_PRODUCER


def producer_states() -> dict:
    """The honest gate states for the two producer seams (this round:
    both UNBOUND)."""
    return {
        "new_failure_pattern": (
            "BOUND"
            if _FAILURE_PATTERN_PRODUCER is not None
            else FAILURE_PATTERN_PRODUCER_UNBOUND
        ),
        "curriculum_drift": (
            "BOUND"
            if _CURRICULUM_DRIFT_PRODUCER is not None
            else CURRICULUM_DRIFT_PRODUCER_UNBOUND
        ),
    }


# ---------------------------------------------------------------------------
# gate-signal derivation: consumes SIGNED objects only
# ---------------------------------------------------------------------------
def derive_failure_pattern_signal(
    fingerprint: Any, *, novelty_threshold: Any
) -> dict:
    """Derive the ``new_failure_pattern`` signal from a SIGNED
    fingerprint object (never a caller-built mapping).

    ``novelty_threshold`` is supervisor-frozen; absent => the
    INVOCATION_THRESHOLDS_UNFROZEN state (no fabricated default).
    """
    ctx = "producer_seams.failure_pattern"
    if not isinstance(fingerprint, FailurePatternFingerprint):
        raise ProducerSeamError(
            PRODUCER_SIGNAL_BAD,
            f"{ctx}: signals consume FailurePatternFingerprint objects "
            f"only, got {type(fingerprint).__name__}",
        )
    if novelty_threshold is None:
        raise ProducerSeamError(
            PRODUCER_THRESHOLD_UNFROZEN,
            f"{ctx}: novelty_threshold is not supervisor-frozen; the "
            "signal stays in the INVOCATION_THRESHOLDS_UNFROZEN state "
            "(no fabricated default)",
        )
    if isinstance(novelty_threshold, bool) or not isinstance(
        novelty_threshold, (int, float)
    ):
        raise ProducerSeamError(
            PRODUCER_THRESHOLD_UNFROZEN,
            f"{ctx}: novelty_threshold must be a number, got "
            f"{novelty_threshold!r}",
        )
    return {
        "field": "new_failure_pattern",
        "triggered": bool(fingerprint.novelty >= float(novelty_threshold)),
        "fingerprint_hash": _fingerprint_hash(fingerprint),
        "detector_version": fingerprint.detector_version,
        "threshold": float(novelty_threshold),
    }


def derive_curriculum_drift_signal(
    history: Any, *, drift_threshold: Any
) -> dict:
    """Derive the ``curriculum_drift`` signal from a SIGNED history
    object (never a caller-built mapping). ``drift_threshold`` is
    supervisor-frozen; absent => INVOCATION_THRESHOLDS_UNFROZEN."""
    ctx = "producer_seams.curriculum_drift"
    if not isinstance(history, CurriculumCompositionHistory):
        raise ProducerSeamError(
            PRODUCER_SIGNAL_BAD,
            f"{ctx}: signals consume CurriculumCompositionHistory "
            f"objects only, got {type(history).__name__}",
        )
    if drift_threshold is None:
        raise ProducerSeamError(
            PRODUCER_THRESHOLD_UNFROZEN,
            f"{ctx}: drift_threshold is not supervisor-frozen; the "
            "signal stays in the INVOCATION_THRESHOLDS_UNFROZEN state "
            "(no fabricated default)",
        )
    if isinstance(drift_threshold, bool) or not isinstance(
        drift_threshold, (int, float)
    ):
        raise ProducerSeamError(
            PRODUCER_THRESHOLD_UNFROZEN,
            f"{ctx}: drift_threshold must be a number, got "
            f"{drift_threshold!r}",
        )
    return {
        "field": "curriculum_drift",
        "triggered": bool(history.drift_metric >= float(drift_threshold)),
        "history_hash": _history_hash(history),
        "metric_version": CURRICULUM_DRIFT_METRIC_VERSION,
        "threshold": float(drift_threshold),
    }


def mint_failure_pattern_fingerprint(
    *,
    window_hash: str,
    student_checkpoint_hash: str,
    **inputs: Any,
) -> FailurePatternFingerprint:
    """Mint ONE fingerprint through the registered producer.

    Fail-closed: no producer is registered => FAILURE_PATTERN_
    PRODUCER_UNBOUND. The producer itself mints the signed object.
    """
    producer = _FAILURE_PATTERN_PRODUCER
    if producer is None:
        raise ProducerSeamError(
            FAILURE_PATTERN_PRODUCER_UNBOUND,
            "producer_seams: no failure-pattern producer is registered "
            "in this worktree; the signal stays UNBOUND",
        )
    result = producer(
        window_hash=window_hash,
        student_checkpoint_hash=student_checkpoint_hash,
        **inputs,
    )
    if not isinstance(result, FailurePatternFingerprint):
        raise ProducerSeamError(
            PRODUCER_SIGNAL_BAD,
            f"producer_seams: the registered failure-pattern producer "
            f"returned {type(result).__name__}, not a "
            "FailurePatternFingerprint",
        )
    return result


def mint_curriculum_composition_history(
    *, prior_batch_hashes: Tuple[str, ...], **inputs: Any
) -> CurriculumCompositionHistory:
    """Mint ONE history record through the registered producer.

    Fail-closed: no producer is registered => CURRICULUM_DRIFT_
    PRODUCER_UNBOUND. The producer itself mints the signed object.
    """
    producer = _CURRICULUM_DRIFT_PRODUCER
    if producer is None:
        raise ProducerSeamError(
            CURRICULUM_DRIFT_PRODUCER_UNBOUND,
            "producer_seams: no curriculum-drift producer is registered "
            "in this worktree; the signal stays UNBOUND",
        )
    result = producer(
        prior_batch_hashes=prior_batch_hashes, **inputs
    )
    if not isinstance(result, CurriculumCompositionHistory):
        raise ProducerSeamError(
            PRODUCER_SIGNAL_BAD,
            f"producer_seams: the registered curriculum-drift producer "
            f"returned {type(result).__name__}, not a "
            "CurriculumCompositionHistory",
        )
    return result
