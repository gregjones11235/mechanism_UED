"""CC2 follow-up P0-9: attested criterion selection.

The selector's outcome is bound hash-wise to EVERYTHING it consumed,
so the GenManager certification can verify the whole selection
mechanically — no trust in summaries::

    execute_criterion_selection(
        window, candidates, probe_results, signed_signals, ...
    ) -> (SelectionOutcome, SelectionAttestation)

The ``SelectionAttestation`` binds: the selected ids, the ORDERED
candidate pool hash, the probe pool hash, the signed-signals pool
hash, the selector source identity, the criterion constants, the
resolved weights, the family cap, the seed, the selected-set hash and
the selector's own selection hash — plus the window identity. Any
tamper changes ``attestation_hash`` and certification fails closed.

Signal discipline: ONLY ``SignedCriterionSignals`` enter selection —
each is re-verified against its candidate + probe result, the signer
gate runs BEFORE scoring, and TEST_ONLY / production signals never
mix. Caller-shaped signal mappings have no path in.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

from . import signed_signals as SS
from .canonical import canonical_sha256
from .criterion_selector import (
    CRITERIA,
    CRITERION_SELECTOR_NAME,
    LOWER_IS_BETTER,
    NORMALIZATION_NAME,
    resolve_weights,
    select_criterion_batch,
)
from .executable_candidates import ExecutableCandidate
from .probe_result_binding import CandidateProbeResult
from .schemas import E1SchemaError
from .selector import SelectionOutcome

#: attestation binding version
SELECTION_ATTESTATION_VERSION = "e1-selection-attestation-v1"

#: selector source identity (mechanical; never a free-form claim)
SELECTOR_SOURCE_IDENTITY = {
    "selector": CRITERION_SELECTOR_NAME,
    "normalization": NORMALIZATION_NAME,
    "module": "dicode.teachers.e1_formal.criterion_selector",
}

# fail-closed codes (greppable)
SELECTION_BAD_TYPE = "SELECTION_BAD_TYPE"
SELECTION_POOL_EMPTY = "SELECTION_POOL_EMPTY"
SELECTION_POOL_MISMATCH = "SELECTION_POOL_MISMATCH"
SELECTION_PROBE_MISSING = "SELECTION_PROBE_MISSING"
SELECTION_SIGNAL_ORPHAN = "SELECTION_SIGNAL_ORPHAN"
SELECTION_SIGNAL_SIGNER_UNAUTHORIZED = (
    "SELECTION_SIGNAL_SIGNER_UNAUTHORIZED"
)
SELECTION_TEST_ONLY_MIXED = "SELECTION_TEST_ONLY_MIXED"
SELECTION_ATTESTATION_TAMPERED = "SELECTION_ATTESTATION_TAMPERED"
SELECTION_WINDOW_MISMATCH = "SELECTION_WINDOW_MISMATCH"
SELECTION_BAD_COUNT = "SELECTION_BAD_COUNT"
SELECTION_FAMILY_CAP_VIOLATED = "SELECTION_FAMILY_CAP_VIOLATED"
SELECTION_ANCHOR_SELECTED = "SELECTION_ANCHOR_SELECTED"
SELECTION_UNKNOWN_CANDIDATE = "SELECTION_UNKNOWN_CANDIDATE"
SELECTION_PROBE_BINDING_MISMATCH = "SELECTION_PROBE_BINDING_MISMATCH"


class SelectionAttestationError(E1SchemaError):
    """Fail-closed selection-attestation violation; ``code`` is
    greppable."""


@dataclass(frozen=True)
class SelectionAttestation:
    """The hash-bound record of ONE criterion selection (immutable)."""

    window_id: str
    window_hash: str
    selected_ids: Tuple[str, ...]
    candidate_pool_hash: str
    probe_pool_hash: str
    signals_pool_hash: str
    selector_source_hash: str
    constants_hash: str
    weights_hash: str
    family_cap: int
    seed: int
    k: int
    selected_set_hash: str
    selection_hash: str  # the selector's own outcome hash
    attestation_hash: str


def compute_pool_hashes(
    candidates: Sequence[Any],
    probe_results: Sequence[Any],
    signed_signals: Sequence[Any],
) -> Tuple[str, str, str]:
    """Ordered pool hashes (order matters; pools are never sorted
    silently)."""
    candidate_pool_hash = canonical_sha256(
        [candidate.candidate_hash for candidate in candidates]
    )
    probe_pool_hash = canonical_sha256(
        [probe.attestation_hash for probe in probe_results]
    )
    signals_pool_hash = canonical_sha256(
        [signal.signal_hash for signal in signed_signals]
    )
    return candidate_pool_hash, probe_pool_hash, signals_pool_hash


def _attestation_payload(
    *,
    window_id: str,
    window_hash: str,
    selected_ids: Tuple[str, ...],
    candidate_pool_hash: str,
    probe_pool_hash: str,
    signals_pool_hash: str,
    selector_source_hash: str,
    constants_hash: str,
    weights_hash: str,
    family_cap: int,
    seed: int,
    k: int,
    selected_set_hash: str,
    selection_hash: str,
) -> dict:
    return {
        "attestation_version": SELECTION_ATTESTATION_VERSION,
        "window_id": window_id,
        "window_hash": window_hash,
        "selected_ids": list(selected_ids),
        "candidate_pool_hash": candidate_pool_hash,
        "probe_pool_hash": probe_pool_hash,
        "signals_pool_hash": signals_pool_hash,
        "selector_source_hash": selector_source_hash,
        "constants_hash": constants_hash,
        "weights_hash": weights_hash,
        "family_cap": family_cap,
        "seed": seed,
        "k": k,
        "selected_set_hash": selected_set_hash,
        "selection_hash": selection_hash,
    }


def verify_selection_attestation(
    attestation: Any,
    *,
    candidates: Sequence[Any],
    probe_results: Sequence[Any],
    signed_signals: Sequence[Any],
    window_hash: str,
    ctx: str,
) -> None:
    """Re-derive the attestation against the CURRENT pools (tamper-
    evident; any drift fails closed)."""
    if not isinstance(attestation, SelectionAttestation):
        raise SelectionAttestationError(
            SELECTION_BAD_TYPE,
            f"{ctx}: expected a SelectionAttestation, got "
            f"{type(attestation).__name__}",
        )
    if attestation.window_hash != window_hash:
        raise SelectionAttestationError(
            SELECTION_WINDOW_MISMATCH,
            f"{ctx}: attestation binds window "
            f"{attestation.window_hash!r} but the current window is "
            f"{window_hash!r}",
        )
    pool_hashes = compute_pool_hashes(
        candidates, probe_results, signed_signals
    )
    if pool_hashes[0] != attestation.candidate_pool_hash:
        raise SelectionAttestationError(
            SELECTION_POOL_MISMATCH,
            f"{ctx}: candidate pool hash drift "
            f"({attestation.candidate_pool_hash!r} != {pool_hashes[0]!r})",
        )
    if pool_hashes[1] != attestation.probe_pool_hash:
        raise SelectionAttestationError(
            SELECTION_POOL_MISMATCH,
            f"{ctx}: probe pool hash drift "
            f"({attestation.probe_pool_hash!r} != {pool_hashes[1]!r})",
        )
    if pool_hashes[2] != attestation.signals_pool_hash:
        raise SelectionAttestationError(
            SELECTION_POOL_MISMATCH,
            f"{ctx}: signals pool hash drift "
            f"({attestation.signals_pool_hash!r} != {pool_hashes[2]!r})",
        )
    payload = _attestation_payload(
        window_id=attestation.window_id,
        window_hash=attestation.window_hash,
        selected_ids=attestation.selected_ids,
        candidate_pool_hash=attestation.candidate_pool_hash,
        probe_pool_hash=attestation.probe_pool_hash,
        signals_pool_hash=attestation.signals_pool_hash,
        selector_source_hash=attestation.selector_source_hash,
        constants_hash=attestation.constants_hash,
        weights_hash=attestation.weights_hash,
        family_cap=attestation.family_cap,
        seed=attestation.seed,
        k=attestation.k,
        selected_set_hash=attestation.selected_set_hash,
        selection_hash=attestation.selection_hash,
    )
    recomputed = canonical_sha256(payload)
    if recomputed != attestation.attestation_hash:
        raise SelectionAttestationError(
            SELECTION_ATTESTATION_TAMPERED,
            f"{ctx}: attestation_hash {attestation.attestation_hash!r} "
            f"!= recomputed {recomputed!r} (tampered selection)",
        )


def execute_criterion_selection(
    *,
    window_id: str,
    window_hash: str,
    candidates: Sequence[Any],
    probe_results: Sequence[Any],
    signed_signals: Sequence[Any],
    k: int,
    seed: int,
    critic_policy: str,
    family_cap: int,
    weights: Optional[Mapping[str, Any]] = None,
    allow_test_only: bool = False,
) -> Tuple[SelectionOutcome, SelectionAttestation]:
    """Run the criterion-wise selection under FULL binding.

    Signals are ONLY ``SignedCriterionSignals``; each is re-verified
    against its candidate and probe result BEFORE scoring. The outcome
    and the attestation are returned together — the attestation is
    what the GenManager certification consumes.
    """
    ctx = "selection_attestation.execute"
    if not isinstance(candidates, (tuple, list)) or len(candidates) == 0:
        raise SelectionAttestationError(
            SELECTION_POOL_EMPTY,
            f"{ctx}: the candidate pool is empty",
        )
    if (
        not isinstance(probe_results, (tuple, list))
        or len(probe_results) == 0
    ):
        raise SelectionAttestationError(
            SELECTION_POOL_EMPTY,
            f"{ctx}: the probe pool is empty",
        )
    if (
        not isinstance(signed_signals, (tuple, list))
        or len(signed_signals) == 0
    ):
        raise SelectionAttestationError(
            SELECTION_POOL_EMPTY,
            f"{ctx}: the signed-signals pool is empty",
        )
    candidates_by_hash = {}
    for candidate in candidates:
        if not isinstance(candidate, ExecutableCandidate):
            raise SelectionAttestationError(
                SELECTION_BAD_TYPE,
                f"{ctx}: candidates must be ExecutableCandidate "
                f"objects, got {type(candidate).__name__}",
            )
        candidates_by_hash[candidate.candidate_hash] = candidate
    probes_by_candidate = {}
    for probe in probe_results:
        if not isinstance(probe, CandidateProbeResult):
            raise SelectionAttestationError(
                SELECTION_BAD_TYPE,
                f"{ctx}: probe results must be CandidateProbeResult "
                f"objects, got {type(probe).__name__}",
            )
        probes_by_candidate[probe.candidate_hash] = probe
    # ---- signal gating BEFORE any scoring ----------------------------
    selector_signals = []
    for signal in signed_signals:
        if not isinstance(signal, SS.SignedCriterionSignals):
            raise SelectionAttestationError(
                SELECTION_BAD_TYPE,
                f"{ctx}: signals must be SignedCriterionSignals "
                f"objects (caller-shaped mappings never enter), got "
                f"{type(signal).__name__}",
            )
        if allow_test_only:
            if not signal.test_only:
                raise SelectionAttestationError(
                    SELECTION_TEST_ONLY_MIXED,
                    f"{ctx}: the TEST_ONLY selection surface received a "
                    "PRODUCTION signal; the two surfaces never mix",
                )
            if signal.signer_id != SS.SYNTHETIC_TEST_ONLY_SIGNAL_SIGNER:
                raise SelectionAttestationError(
                    SELECTION_SIGNAL_SIGNER_UNAUTHORIZED,
                    f"{ctx}: TEST_ONLY signal signer {signal.signer_id!r} "
                    "is not the synthetic signer",
                )
        else:
            if signal.test_only:
                raise SelectionAttestationError(
                    SELECTION_TEST_ONLY_MIXED,
                    f"{ctx}: TEST_ONLY signals never enter a production "
                    "selection",
                )
            if signal.signer_id not in SS.AUTHORIZED_SIGNAL_SIGNERS:
                raise SelectionAttestationError(
                    SELECTION_SIGNAL_SIGNER_UNAUTHORIZED,
                    f"{ctx}: signal signer {signal.signer_id!r} is not "
                    "on the supervisor-owned whitelist (EMPTY this "
                    "round)",
                )
        candidate = candidates_by_hash.get(signal.candidate_hash)
        if candidate is None:
            raise SelectionAttestationError(
                SELECTION_SIGNAL_ORPHAN,
                f"{ctx}: signal for candidate {signal.candidate_hash!r} "
                "has no candidate in the pool",
            )
        probe = probes_by_candidate.get(signal.candidate_hash)
        if probe is None:
            raise SelectionAttestationError(
                SELECTION_PROBE_MISSING,
                f"{ctx}: candidate {signal.candidate_hash!r} has "
                "signals but no probe result",
            )
        SS.verify_signed_criterion_signals(
            signal, candidate=candidate, probe_result=probe, ctx=ctx
        )
        selector_signals.append(signal.to_criterion_signals())
    covered = {signal.candidate_hash for signal in signed_signals}
    if covered != set(candidates_by_hash):
        raise SelectionAttestationError(
            SELECTION_POOL_MISMATCH,
            f"{ctx}: signed signals cover {len(covered)} candidate(s) "
            f"but the pool has {len(candidates_by_hash)}; every "
            "candidate needs its signed eight-criterion evidence",
        )
    # ---- weights + selection ------------------------------------------
    resolved_weights, weights_source = resolve_weights(weights)
    outcome = select_criterion_batch(
        selector_signals,
        k=k,
        seed=seed,
        critic_policy=critic_policy,
        family_cap=family_cap,
        weights=resolved_weights,
    )
    # ---- attestation ----------------------------------------------------
    candidate_pool_hash, probe_pool_hash, signals_pool_hash = (
        compute_pool_hashes(candidates, probe_results, signed_signals)
    )
    selector_source_hash = canonical_sha256(SELECTOR_SOURCE_IDENTITY)
    constants_hash = canonical_sha256(
        {"criteria": list(CRITERIA), "lower_is_better": list(LOWER_IS_BETTER)}
    )
    weights_hash = canonical_sha256(
        {
            "weights_source": weights_source,
            "weights": [
                [criterion, str(resolved_weights[criterion])]
                for criterion in CRITERIA
            ],
        }
    )
    selected_set_hash = canonical_sha256(sorted(outcome.selected_ids))
    payload = _attestation_payload(
        window_id=window_id,
        window_hash=window_hash,
        selected_ids=outcome.selected_ids,
        candidate_pool_hash=candidate_pool_hash,
        probe_pool_hash=probe_pool_hash,
        signals_pool_hash=signals_pool_hash,
        selector_source_hash=selector_source_hash,
        constants_hash=constants_hash,
        weights_hash=weights_hash,
        family_cap=family_cap,
        seed=seed,
        k=k,
        selected_set_hash=selected_set_hash,
        selection_hash=outcome.selection_hash,
    )
    attestation = SelectionAttestation(
        window_id=window_id,
        window_hash=window_hash,
        selected_ids=outcome.selected_ids,
        candidate_pool_hash=candidate_pool_hash,
        probe_pool_hash=probe_pool_hash,
        signals_pool_hash=signals_pool_hash,
        selector_source_hash=selector_source_hash,
        constants_hash=constants_hash,
        weights_hash=weights_hash,
        family_cap=family_cap,
        seed=seed,
        k=k,
        selected_set_hash=selected_set_hash,
        selection_hash=outcome.selection_hash,
        attestation_hash=canonical_sha256(payload),
    )
    return outcome, attestation
