"""CC2 follow-up P0-8: signed criterion signals derived from probes.

The criterion-wise selector must never trust caller-shaped signals.
``SignedCriterionSignals`` is the immutable, hash-bound record of the
eight criterion values for ONE candidate, and the ONLY minting path
is ``derive_criterion_signals_from_probe_result`` — which derives
every criterion from a REAL registry-signed probe result plus
explicit evidence records, fail-closed on any missing source:

* front_regret / global_regret / behavioral_gap / learnability /
  learning_progress — the corresponding fields of the probe's
  aggregate metrics (the metrics mapping must RE-HASH to the probe
  result's signed ``aggregate_metrics_hash`` — tamper fails closed);
* diversity — the diversity evidence record (axis_count /
  pool_axis_max), hashed into ``diversity_evidence_hash``;
* global_retention — the retention evidence record in [0, 1], hashed
  into ``retention_evidence_hash`` (no archive prior, no heuristic
  substitute — absent evidence blocks derivation);
* simulator_cost — the cost evidence ``episodes``, which must EQUAL
  the probe result's completed episodes, hashed into
  ``cost_evidence_hash``.

Signer discipline: production minting requires a signer on the
supervisor-owned whitelist (EMPTY this round); TEST_ONLY minting
requires the synthetic signer + the explicit ``test_only`` flag.
Consumers re-derive ``signal_hash`` before trusting anything.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from .canonical import canonical_sha256
from .criterion_selector import CRITERIA, CriterionSignals
from .executable_candidates import ExecutableCandidate
from .probe_result_binding import CandidateProbeResult
from .schemas import E1SchemaError, PROVENANCE_CANDIDATE_EVALUATION

#: derivation version (frozen per mechanism revision)
SIGNAL_DERIVATION_VERSION = "e1-criterion-derivation-v1"

#: synthetic TEST_ONLY signer (greppable)
SYNTHETIC_TEST_ONLY_SIGNAL_SIGNER = "SYNTHETIC_TEST_ONLY_SIGNAL_SIGNER"

#: supervisor-owned production signer whitelist — EMPTY this round
AUTHORIZED_SIGNAL_SIGNERS: Tuple[str, ...] = ()

#: the five criteria sourced from the probe's aggregate metrics
_METRIC_CRITERIA = (
    "front_regret",
    "global_regret",
    "behavioral_gap",
    "learnability",
    "learning_progress",
)

# fail-closed codes (greppable)
SIGNALS_BAD_TYPE = "SIGNALS_BAD_TYPE"
SIGNALS_MISSING_SOURCE = "SIGNALS_MISSING_SOURCE"
SIGNALS_OUT_OF_RANGE = "SIGNALS_OUT_OF_RANGE"
SIGNALS_HASH_MISMATCH = "SIGNALS_HASH_MISMATCH"
SIGNALS_PROBE_MISMATCH = "SIGNALS_PROBE_MISMATCH"
SIGNALS_CANDIDATE_MISMATCH = "SIGNALS_CANDIDATE_MISMATCH"
SIGNALS_SIGNER_UNAUTHORIZED = "SIGNALS_SIGNER_UNAUTHORIZED"
SIGNALS_TEST_ONLY_REJECTED = "SIGNALS_TEST_ONLY_REJECTED"
SIGNALS_COST_MISMATCH = "SIGNALS_COST_MISMATCH"


class SignedSignalsError(E1SchemaError):
    """Fail-closed signed-signals violation; ``code`` is greppable."""


@dataclass(frozen=True)
class SignedCriterionSignals:
    """One candidate's signed eight-criterion evidence (immutable)."""

    candidate_id: str
    candidate_hash: str
    family_id: str
    probe_result_id: str
    probe_result_hash: str
    student_identity_hash: str
    student_checkpoint_hash: str
    reference_identity_hash: str
    reference_checkpoint_hash: str
    values: Tuple[Tuple[str, float], ...]  # the eight CRITERIA
    derivation_version: str
    input_hashes: Tuple[Tuple[str, str], ...]
    retention_evidence_hash: str
    diversity_evidence_hash: str
    cost_evidence_hash: str
    signer_id: str
    verifier_hash: str
    signal_hash: str
    test_only: bool

    def values_dict(self) -> Dict[str, float]:
        return dict(self.values)

    def to_criterion_signals(self) -> CriterionSignals:
        """The selector-facing view (real-probe-backed by minting)."""
        return CriterionSignals(
            candidate_id=self.candidate_id,
            family_id=self.family_id,
            values=self.values,
            provenance=PROVENANCE_CANDIDATE_EVALUATION,
            has_real_probe=True,
        )


def compute_signal_hash(payload: Mapping[str, Any]) -> str:
    return canonical_sha256(dict(payload))


def _signal_payload(
    *,
    candidate: ExecutableCandidate,
    probe_result: CandidateProbeResult,
    values: Tuple[Tuple[str, float], ...],
    input_hashes: Tuple[Tuple[str, str], ...],
    retention_evidence_hash: str,
    diversity_evidence_hash: str,
    cost_evidence_hash: str,
    signer_id: str,
    verifier_hash: str,
    test_only: bool,
) -> Dict[str, Any]:
    return {
        "derivation_version": SIGNAL_DERIVATION_VERSION,
        "candidate_id": candidate.candidate_id,
        "candidate_hash": candidate.candidate_hash,
        "family_id": candidate.family_id,
        "probe_result_id": probe_result.result_id,
        "probe_result_hash": probe_result.attestation_hash,
        "student_identity_hash": probe_result.student_identity_hash,
        "student_checkpoint_hash": probe_result.student_checkpoint_hash,
        "reference_identity_hash": (
            probe_result.reference_identity_hash
        ),
        "reference_checkpoint_hash": (
            probe_result.reference_checkpoint_hash
        ),
        "values": [[name, value] for name, value in values],
        "input_hashes": [list(pair) for pair in input_hashes],
        "retention_evidence_hash": retention_evidence_hash,
        "diversity_evidence_hash": diversity_evidence_hash,
        "cost_evidence_hash": cost_evidence_hash,
        "signer_id": signer_id,
        "verifier_hash": verifier_hash,
        "test_only": test_only,
    }


def _require_finite(name: str, value: Any, ctx: str) -> float:
    import math

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SignedSignalsError(
            SIGNALS_BAD_TYPE,
            f"{ctx}: {name} must be a number, got {value!r}",
        )
    value = float(value)
    if not math.isfinite(value):
        raise SignedSignalsError(
            SIGNALS_BAD_TYPE,
            f"{ctx}: {name} must be finite, got {value!r}",
        )
    return value


def derive_criterion_signals_from_probe_result(
    *,
    probe_result: Any,
    candidate: Any,
    aggregate_metrics: Mapping[str, Any],
    retention_evidence: Mapping[str, Any],
    diversity_evidence: Mapping[str, Any],
    cost_evidence: Mapping[str, Any],
    signer_id: str,
    test_only: bool = False,
) -> SignedCriterionSignals:
    """The ONLY minting path for criterion signals (fail-closed).

    Every criterion needs its REAL source; anything missing raises
    ``SIGNALS_MISSING_SOURCE`` — no prior, no heuristic, no LLM score
    ever substitutes. The aggregate-metrics mapping must re-hash to
    the probe result's signed hash (tamper => fail closed).
    """
    ctx = "signed_signals.derive"
    if not isinstance(probe_result, CandidateProbeResult):
        raise SignedSignalsError(
            SIGNALS_BAD_TYPE,
            f"{ctx}: probe_result must be a CandidateProbeResult, got "
            f"{type(probe_result).__name__}",
        )
    if not isinstance(candidate, ExecutableCandidate):
        raise SignedSignalsError(
            SIGNALS_BAD_TYPE,
            f"{ctx}: candidate must be an ExecutableCandidate, got "
            f"{type(candidate).__name__}",
        )
    if probe_result.candidate_hash != candidate.candidate_hash:
        raise SignedSignalsError(
            SIGNALS_PROBE_MISMATCH,
            f"{ctx}: probe result binds candidate "
            f"{probe_result.candidate_hash!r} but the candidate is "
            f"{candidate.candidate_hash!r}",
        )
    # ---- signer gate (BEFORE any value is trusted) -------------------
    if not isinstance(signer_id, str) or not signer_id.strip():
        raise SignedSignalsError(
            SIGNALS_BAD_TYPE,
            f"{ctx}: signer_id must be a non-empty str, got "
            f"{signer_id!r}",
        )
    if test_only:
        if signer_id != SYNTHETIC_TEST_ONLY_SIGNAL_SIGNER:
            raise SignedSignalsError(
                SIGNALS_TEST_ONLY_REJECTED,
                f"{ctx}: TEST_ONLY signals must be signed by "
                f"{SYNTHETIC_TEST_ONLY_SIGNAL_SIGNER!r}, got "
                f"{signer_id!r}",
            )
    elif signer_id not in AUTHORIZED_SIGNAL_SIGNERS:
        raise SignedSignalsError(
            SIGNALS_SIGNER_UNAUTHORIZED,
            f"{ctx}: signer {signer_id!r} is not on the supervisor-"
            "owned signal whitelist (EMPTY this round)",
        )
    # ---- metric criteria: re-hash the aggregate metrics --------------
    if not isinstance(aggregate_metrics, Mapping):
        raise SignedSignalsError(
            SIGNALS_BAD_TYPE,
            f"{ctx}: aggregate_metrics must be a mapping, got "
            f"{type(aggregate_metrics).__name__}",
        )
    if (
        canonical_sha256(dict(aggregate_metrics))
        != probe_result.aggregate_metrics_hash
    ):
        raise SignedSignalsError(
            SIGNALS_HASH_MISMATCH,
            f"{ctx}: aggregate_metrics do not re-hash to the probe "
            "result's signed aggregate_metrics_hash (tampered input)",
        )
    values: Dict[str, float] = {}
    for criterion in _METRIC_CRITERIA:
        if criterion not in aggregate_metrics:
            raise SignedSignalsError(
                SIGNALS_MISSING_SOURCE,
                f"{ctx}: criterion {criterion!r} has no real source in "
                "the probe's aggregate metrics; derivation fails "
                "closed (never fabricated)",
            )
        values[criterion] = _require_finite(
            f"aggregate_metrics[{criterion}]",
            aggregate_metrics[criterion],
            ctx,
        )
    # ---- retention evidence -------------------------------------------
    if not isinstance(retention_evidence, Mapping):
        raise SignedSignalsError(
            SIGNALS_BAD_TYPE,
            f"{ctx}: retention_evidence must be a mapping, got "
            f"{type(retention_evidence).__name__}",
        )
    if "global_retention" not in retention_evidence:
        raise SignedSignalsError(
            SIGNALS_MISSING_SOURCE,
            f"{ctx}: criterion 'global_retention' has no real evidence "
            "record (no archive prior, no heuristic substitute)",
        )
    retention = _require_finite(
        "retention_evidence[global_retention]",
        retention_evidence["global_retention"],
        ctx,
    )
    if retention < 0.0 or retention > 1.0:
        raise SignedSignalsError(
            SIGNALS_OUT_OF_RANGE,
            f"{ctx}: global_retention outside [0, 1]: {retention}",
        )
    values["global_retention"] = retention
    retention_evidence_hash = canonical_sha256(dict(retention_evidence))
    # ---- diversity evidence --------------------------------------------
    if not isinstance(diversity_evidence, Mapping):
        raise SignedSignalsError(
            SIGNALS_BAD_TYPE,
            f"{ctx}: diversity_evidence must be a mapping, got "
            f"{type(diversity_evidence).__name__}",
        )
    if (
        "axis_count" not in diversity_evidence
        or "pool_axis_max" not in diversity_evidence
    ):
        raise SignedSignalsError(
            SIGNALS_MISSING_SOURCE,
            f"{ctx}: criterion 'diversity' needs axis_count and "
            "pool_axis_max evidence",
        )
    axis_count = _require_finite(
        "diversity_evidence[axis_count]",
        diversity_evidence["axis_count"],
        ctx,
    )
    pool_axis_max = _require_finite(
        "diversity_evidence[pool_axis_max]",
        diversity_evidence["pool_axis_max"],
        ctx,
    )
    if axis_count < 0 or pool_axis_max <= 0:
        raise SignedSignalsError(
            SIGNALS_OUT_OF_RANGE,
            f"{ctx}: diversity evidence requires axis_count >= 0 and "
            "pool_axis_max > 0",
        )
    values["diversity"] = min(1.0, axis_count / pool_axis_max)
    diversity_evidence_hash = canonical_sha256(dict(diversity_evidence))
    # ---- cost evidence ---------------------------------------------------
    if not isinstance(cost_evidence, Mapping):
        raise SignedSignalsError(
            SIGNALS_BAD_TYPE,
            f"{ctx}: cost_evidence must be a mapping, got "
            f"{type(cost_evidence).__name__}",
        )
    if "episodes" not in cost_evidence:
        raise SignedSignalsError(
            SIGNALS_MISSING_SOURCE,
            f"{ctx}: criterion 'simulator_cost' needs the episodes "
            "evidence",
        )
    episodes = _require_finite(
        "cost_evidence[episodes]", cost_evidence["episodes"], ctx
    )
    if episodes < 0:
        raise SignedSignalsError(
            SIGNALS_OUT_OF_RANGE,
            f"{ctx}: episodes must be >= 0, got {episodes}",
        )
    if episodes != float(probe_result.episodes_completed):
        raise SignedSignalsError(
            SIGNALS_COST_MISMATCH,
            f"{ctx}: cost evidence episodes {episodes} != probe "
            f"result completed episodes {probe_result.episodes_completed}",
        )
    values["simulator_cost"] = episodes
    cost_evidence_hash = canonical_sha256(dict(cost_evidence))
    # ---- assemble + sign -------------------------------------------------
    if set(values) != set(CRITERIA):
        raise SignedSignalsError(
            SIGNALS_MISSING_SOURCE,
            f"{ctx}: derived values {sorted(values)} != the eight "
            f"CRITERIA {list(CRITERIA)}",
        )
    frozen_values = tuple(
        (criterion, values[criterion]) for criterion in CRITERIA
    )
    input_hashes = (
        ("aggregate_metrics_hash", probe_result.aggregate_metrics_hash),
        ("uncertainty_ci_hash", probe_result.uncertainty_ci_hash),
        (
            "terminal_event_aggregates_hash",
            probe_result.terminal_event_aggregates_hash,
        ),
    )
    verifier_hash = canonical_sha256(
        {
            "verifier": SIGNAL_DERIVATION_VERSION,
            "probe_binding": "e1-probe-result-binding-v1",
        }
    )
    payload = _signal_payload(
        candidate=candidate,
        probe_result=probe_result,
        values=frozen_values,
        input_hashes=input_hashes,
        retention_evidence_hash=retention_evidence_hash,
        diversity_evidence_hash=diversity_evidence_hash,
        cost_evidence_hash=cost_evidence_hash,
        signer_id=signer_id,
        verifier_hash=verifier_hash,
        test_only=test_only,
    )
    return SignedCriterionSignals(
        candidate_id=candidate.candidate_id,
        candidate_hash=candidate.candidate_hash,
        family_id=candidate.family_id,
        probe_result_id=probe_result.result_id,
        probe_result_hash=probe_result.attestation_hash,
        student_identity_hash=probe_result.student_identity_hash,
        student_checkpoint_hash=probe_result.student_checkpoint_hash,
        reference_identity_hash=probe_result.reference_identity_hash,
        reference_checkpoint_hash=(
            probe_result.reference_checkpoint_hash
        ),
        values=frozen_values,
        derivation_version=SIGNAL_DERIVATION_VERSION,
        input_hashes=input_hashes,
        retention_evidence_hash=retention_evidence_hash,
        diversity_evidence_hash=diversity_evidence_hash,
        cost_evidence_hash=cost_evidence_hash,
        signer_id=signer_id,
        verifier_hash=verifier_hash,
        signal_hash=compute_signal_hash(payload),
        test_only=test_only,
    )


def verify_signed_criterion_signals(
    signed: Any,
    *,
    candidate: Any,
    probe_result: Any,
    ctx: str = "signed_signals.verify",
) -> None:
    """Re-derive the signal hash + every binding fail-closed.

    Consumers (selection attestation, GenManager certification) call
    this before trusting ANY signed signal.
    """
    if not isinstance(signed, SignedCriterionSignals):
        raise SignedSignalsError(
            SIGNALS_BAD_TYPE,
            f"{ctx}: expected SignedCriterionSignals, got "
            f"{type(signed).__name__}",
        )
    if signed.candidate_hash != candidate.candidate_hash:
        raise SignedSignalsError(
            SIGNALS_CANDIDATE_MISMATCH,
            f"{ctx}: signals bind candidate {signed.candidate_hash!r} "
            f"but the candidate is {candidate.candidate_hash!r}",
        )
    if signed.probe_result_hash != probe_result.attestation_hash:
        raise SignedSignalsError(
            SIGNALS_PROBE_MISMATCH,
            f"{ctx}: signals bind probe result "
            f"{signed.probe_result_hash!r} but the probe result is "
            f"{probe_result.attestation_hash!r}",
        )
    payload = _signal_payload(
        candidate=candidate,
        probe_result=probe_result,
        values=signed.values,
        input_hashes=signed.input_hashes,
        retention_evidence_hash=signed.retention_evidence_hash,
        diversity_evidence_hash=signed.diversity_evidence_hash,
        cost_evidence_hash=signed.cost_evidence_hash,
        signer_id=signed.signer_id,
        verifier_hash=signed.verifier_hash,
        test_only=signed.test_only,
    )
    recomputed = compute_signal_hash(payload)
    if recomputed != signed.signal_hash:
        raise SignedSignalsError(
            SIGNALS_HASH_MISMATCH,
            f"{ctx}: signal_hash {signed.signal_hash!r} != recomputed "
            f"{recomputed!r} (tampered signals)",
        )
    if set(dict(signed.values)) != set(CRITERIA):
        raise SignedSignalsError(
            SIGNALS_MISSING_SOURCE,
            f"{ctx}: signals carry {sorted(dict(signed.values))}, not "
            f"the eight CRITERIA {list(CRITERIA)}",
        )
