"""CC2 follow-up P0-4: registry-signed CandidateProbeResult intake.

E1 NEVER mints probe results — it CONSUMES results issued and signed
by the shared probe runner registry. One result is the immutable
record of probing ONE executable candidate with ONE Student and ONE
Reference under ONE frozen seed bank + reset protocol::

    issue_candidate_probe_result(...)      (registry side / TEST_ONLY)
    consume_registry_signed_probe_results(...)   (E1 side, fail-closed)

Consumption refuses, mechanically and in this order:

* an EMPTY pool;
* a result whose recomputed attestation hash != declared hash
  (tamper) or with an empty signer (UNSIGNED);
* a TEST_ONLY signer on the production surface, and any signer not on
  the supervisor-owned registry whitelist (EMPTY this round — honest
  PROBE_SIGNER_UNAUTHORIZED);
* a result bound to the wrong candidate / executable artifact /
  Student identity / Reference identity / checkpoint / seed bank /
  reset protocol;
* a STALE result (candidate not in the current window's pool);
* DUPLICATE results (same result id, or two results for one
  candidate);
* PARTIAL episodes (completed + failed != requested, or zero
  completed episodes);
* a mock/replay runner disguised as a registry runner.

Discipline: consumption never repairs, completes or re-signs a
rejected result — the pool is refused whole.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from .canonical import canonical_sha256
from .executable_candidates import ExecutableCandidate
from .schemas import E1SchemaError

#: registry signer identities
SYNTHETIC_TEST_ONLY_PROBE_SIGNER = "SYNTHETIC_TEST_ONLY_PROBE_SIGNER"

#: supervisor-owned registry signer whitelist — EMPTY this round (no
#: real probe runner exists; nothing may be consumed on the
#: production path yet)
AUTHORIZED_PROBE_RESULT_SIGNERS: Tuple[str, ...] = ()

#: probe result binding version
PROBE_RESULT_BINDING_VERSION = "e1-probe-result-binding-v1"

# fail-closed codes (greppable)
PROBE_BAD_TYPE = "PROBE_BAD_TYPE"
PROBE_POOL_EMPTY = "PROBE_POOL_EMPTY"
PROBE_UNSIGNED = "PROBE_UNSIGNED"
PROBE_HASH_MISMATCH = "PROBE_HASH_MISMATCH"
PROBE_SIGNER_UNAUTHORIZED = "PROBE_SIGNER_UNAUTHORIZED"
PROBE_TEST_ONLY_SIGNER_REJECTED = "PROBE_TEST_ONLY_SIGNER_REJECTED"
PROBE_CANDIDATE_MISMATCH = "PROBE_CANDIDATE_MISMATCH"
PROBE_ARTIFACT_MISMATCH = "PROBE_ARTIFACT_MISMATCH"
PROBE_STUDENT_MISMATCH = "PROBE_STUDENT_MISMATCH"
PROBE_REFERENCE_MISMATCH = "PROBE_REFERENCE_MISMATCH"
PROBE_CHECKPOINT_MISMATCH = "PROBE_CHECKPOINT_MISMATCH"
PROBE_SEED_BANK_MISMATCH = "PROBE_SEED_BANK_MISMATCH"
PROBE_RESET_PROTOCOL_MISMATCH = "PROBE_RESET_PROTOCOL_MISMATCH"
PROBE_STALE = "PROBE_STALE"
PROBE_DUPLICATE = "PROBE_DUPLICATE"
PROBE_PARTIAL_EPISODES = "PROBE_PARTIAL_EPISODES"
PROBE_MOCK_RUNNER_DISGUISED = "PROBE_MOCK_RUNNER_DISGUISED"


class ProbeResultError(E1SchemaError):
    """Fail-closed probe-result violation; ``code`` is greppable."""


def _require_sha64(value: Any, name: str, ctx: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ProbeResultError(
            PROBE_BAD_TYPE,
            f"{ctx}: {name} must be a 64-hex hash, got {value!r}",
        )
    return value


def _require_non_empty_str(value: Any, name: str, ctx: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProbeResultError(
            PROBE_BAD_TYPE,
            f"{ctx}: {name} must be a non-empty str, got {value!r}",
        )
    return value.strip()


def _require_count(value: Any, name: str, ctx: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProbeResultError(
            PROBE_BAD_TYPE,
            f"{ctx}: {name} must be a non-negative int, got {value!r}",
        )
    return value


@dataclass(frozen=True)
class CandidateProbeResult:
    """One registry-signed probe of one executable candidate
    (immutable)."""

    result_id: str
    candidate_id: str
    candidate_hash: str
    executable_artifact_id: str
    executable_artifact_hash: str
    student_identity_hash: str
    student_checkpoint_hash: str
    reference_identity_hash: str
    reference_checkpoint_hash: str
    runner_registry_id: str
    runner_registry_hash: str
    seed_bank_hash: str
    reset_protocol_id: str
    reset_protocol_hash: str
    episodes_requested: int
    episodes_completed: int
    episodes_failed: int
    simulator_transitions: int
    aggregate_metrics_hash: str
    uncertainty_ci_hash: str
    terminal_event_aggregates_hash: str
    provenance_hash: str
    signer_id: str
    attestation_hash: str
    test_only: bool
    #: the plaintext aggregate_metrics carried for downstream signal
    #: derivation (NOT part of attestation_hash/provenance_hash — those
    #: bind only aggregate_metrics_hash; the plaintext is an audit/
    #: consumption convenience so the signal issuer can derive
    #: criterion signals without re-rolling)
    aggregate_metrics: Optional[Mapping[str, Any]] = None


def compute_probe_attestation_hash(
    *,
    candidate_id: str,
    candidate_hash: str,
    executable_artifact_id: str,
    executable_artifact_hash: str,
    student_identity_hash: str,
    student_checkpoint_hash: str,
    reference_identity_hash: str,
    reference_checkpoint_hash: str,
    runner_registry_id: str,
    runner_registry_hash: str,
    seed_bank_hash: str,
    reset_protocol_id: str,
    reset_protocol_hash: str,
    episodes_requested: int,
    episodes_completed: int,
    episodes_failed: int,
    simulator_transitions: int,
    aggregate_metrics_hash: str,
    uncertainty_ci_hash: str,
    terminal_event_aggregates_hash: str,
    provenance_hash: str,
    signer_id: str,
    test_only: bool,
) -> str:
    """The canonical attestation over the WHOLE result (tamper-
    evident; recomputed on consumption)."""
    return canonical_sha256(
        {
            "binding_version": PROBE_RESULT_BINDING_VERSION,
            "candidate_id": candidate_id,
            "candidate_hash": candidate_hash,
            "executable_artifact_id": executable_artifact_id,
            "executable_artifact_hash": executable_artifact_hash,
            "student_identity_hash": student_identity_hash,
            "student_checkpoint_hash": student_checkpoint_hash,
            "reference_identity_hash": reference_identity_hash,
            "reference_checkpoint_hash": reference_checkpoint_hash,
            "runner_registry_id": runner_registry_id,
            "runner_registry_hash": runner_registry_hash,
            "seed_bank_hash": seed_bank_hash,
            "reset_protocol_id": reset_protocol_id,
            "reset_protocol_hash": reset_protocol_hash,
            "episodes_requested": episodes_requested,
            "episodes_completed": episodes_completed,
            "episodes_failed": episodes_failed,
            "simulator_transitions": simulator_transitions,
            "aggregate_metrics_hash": aggregate_metrics_hash,
            "uncertainty_ci_hash": uncertainty_ci_hash,
            "terminal_event_aggregates_hash": (
                terminal_event_aggregates_hash
            ),
            "provenance_hash": provenance_hash,
            "signer_id": signer_id,
            "test_only": test_only,
        }
    )


def issue_candidate_probe_result(
    *,
    candidate: Any,
    student_identity_hash: str,
    student_checkpoint_hash: str,
    reference_identity_hash: str,
    reference_checkpoint_hash: str,
    runner_registry_id: str,
    runner_registry_hash: str,
    seed_bank_hash: str,
    reset_protocol_id: str,
    reset_protocol_hash: str,
    episodes_requested: int,
    episodes_completed: int,
    episodes_failed: int,
    simulator_transitions: int,
    aggregate_metrics: Mapping[str, Any],
    uncertainty_ci: Mapping[str, Any],
    terminal_event_aggregates: Mapping[str, Any],
    signer_id: str,
    test_only: bool = False,
) -> CandidateProbeResult:
    """Issue ONE probe result bound fail-closed to its candidate.

    The registry side (or the TEST_ONLY closed loop) issues; E1 only
    consumes. Every identity hash is validated 64-hex; the candidate
    link is validated against the real ExecutableCandidate object.
    """
    ctx = "probe_result.issue"
    if not isinstance(candidate, ExecutableCandidate):
        raise ProbeResultError(
            PROBE_BAD_TYPE,
            f"{ctx}: candidate must be an ExecutableCandidate, got "
            f"{type(candidate).__name__}",
        )
    student_identity_hash = _require_sha64(
        student_identity_hash, "student_identity_hash", ctx
    )
    student_checkpoint_hash = _require_sha64(
        student_checkpoint_hash, "student_checkpoint_hash", ctx
    )
    reference_identity_hash = _require_sha64(
        reference_identity_hash, "reference_identity_hash", ctx
    )
    reference_checkpoint_hash = _require_sha64(
        reference_checkpoint_hash, "reference_checkpoint_hash", ctx
    )
    runner_registry_id = _require_non_empty_str(
        runner_registry_id, "runner_registry_id", ctx
    )
    runner_registry_hash = _require_sha64(
        runner_registry_hash, "runner_registry_hash", ctx
    )
    seed_bank_hash = _require_sha64(seed_bank_hash, "seed_bank_hash", ctx)
    reset_protocol_id = _require_non_empty_str(
        reset_protocol_id, "reset_protocol_id", ctx
    )
    reset_protocol_hash = _require_sha64(
        reset_protocol_hash, "reset_protocol_hash", ctx
    )
    episodes_requested = _require_count(
        episodes_requested, "episodes_requested", ctx
    )
    episodes_completed = _require_count(
        episodes_completed, "episodes_completed", ctx
    )
    episodes_failed = _require_count(
        episodes_failed, "episodes_failed", ctx
    )
    simulator_transitions = _require_count(
        simulator_transitions, "simulator_transitions", ctx
    )
    if episodes_completed + episodes_failed != episodes_requested:
        raise ProbeResultError(
            PROBE_PARTIAL_EPISODES,
            f"{ctx}: episodes_completed ({episodes_completed}) + "
            f"episodes_failed ({episodes_failed}) != episodes_requested "
            f"({episodes_requested}); a partial probe is never issued",
        )
    if episodes_completed == 0:
        raise ProbeResultError(
            PROBE_PARTIAL_EPISODES,
            f"{ctx}: zero completed episodes; a probe with no completed "
            "episodes proves nothing",
        )
    signer_id = _require_non_empty_str(signer_id, "signer_id", ctx)
    if not isinstance(aggregate_metrics, Mapping):
        raise ProbeResultError(
            PROBE_BAD_TYPE,
            f"{ctx}: aggregate_metrics must be a mapping, got "
            f"{type(aggregate_metrics).__name__}",
        )
    if not isinstance(uncertainty_ci, Mapping):
        raise ProbeResultError(
            PROBE_BAD_TYPE,
            f"{ctx}: uncertainty_ci must be a mapping, got "
            f"{type(uncertainty_ci).__name__}",
        )
    if not isinstance(terminal_event_aggregates, Mapping):
        raise ProbeResultError(
            PROBE_BAD_TYPE,
            f"{ctx}: terminal_event_aggregates must be a mapping, got "
            f"{type(terminal_event_aggregates).__name__}",
        )
    aggregate_metrics_hash = canonical_sha256(dict(aggregate_metrics))
    uncertainty_ci_hash = canonical_sha256(dict(uncertainty_ci))
    terminal_event_aggregates_hash = canonical_sha256(
        dict(terminal_event_aggregates)
    )
    provenance_hash = canonical_sha256(
        {
            "binding_version": PROBE_RESULT_BINDING_VERSION,
            "kind": "candidate_probe_result",
            "candidate_hash": candidate.candidate_hash,
            "runner_registry_hash": runner_registry_hash,
        }
    )
    attestation_hash = compute_probe_attestation_hash(
        candidate_id=candidate.candidate_id,
        candidate_hash=candidate.candidate_hash,
        executable_artifact_id=candidate.executable_artifact_id,
        executable_artifact_hash=candidate.executable_artifact_hash,
        student_identity_hash=student_identity_hash,
        student_checkpoint_hash=student_checkpoint_hash,
        reference_identity_hash=reference_identity_hash,
        reference_checkpoint_hash=reference_checkpoint_hash,
        runner_registry_id=runner_registry_id,
        runner_registry_hash=runner_registry_hash,
        seed_bank_hash=seed_bank_hash,
        reset_protocol_id=reset_protocol_id,
        reset_protocol_hash=reset_protocol_hash,
        episodes_requested=episodes_requested,
        episodes_completed=episodes_completed,
        episodes_failed=episodes_failed,
        simulator_transitions=simulator_transitions,
        aggregate_metrics_hash=aggregate_metrics_hash,
        uncertainty_ci_hash=uncertainty_ci_hash,
        terminal_event_aggregates_hash=terminal_event_aggregates_hash,
        provenance_hash=provenance_hash,
        signer_id=signer_id,
        test_only=test_only,
    )
    return CandidateProbeResult(
        result_id=f"{attestation_hash}::probe",
        candidate_id=candidate.candidate_id,
        candidate_hash=candidate.candidate_hash,
        executable_artifact_id=candidate.executable_artifact_id,
        executable_artifact_hash=candidate.executable_artifact_hash,
        student_identity_hash=student_identity_hash,
        student_checkpoint_hash=student_checkpoint_hash,
        reference_identity_hash=reference_identity_hash,
        reference_checkpoint_hash=reference_checkpoint_hash,
        runner_registry_id=runner_registry_id,
        runner_registry_hash=runner_registry_hash,
        seed_bank_hash=seed_bank_hash,
        reset_protocol_id=reset_protocol_id,
        reset_protocol_hash=reset_protocol_hash,
        episodes_requested=episodes_requested,
        episodes_completed=episodes_completed,
        episodes_failed=episodes_failed,
        simulator_transitions=simulator_transitions,
        aggregate_metrics_hash=aggregate_metrics_hash,
        uncertainty_ci_hash=uncertainty_ci_hash,
        terminal_event_aggregates_hash=terminal_event_aggregates_hash,
        provenance_hash=provenance_hash,
        signer_id=signer_id,
        attestation_hash=attestation_hash,
        test_only=test_only,
        aggregate_metrics=dict(aggregate_metrics),
    )


def _verify_one(
    result: Any,
    *,
    candidates_by_hash: Dict[str, ExecutableCandidate],
    student_identity_hash: str,
    student_checkpoint_hash: str,
    reference_identity_hash: str,
    reference_checkpoint_hash: str,
    seed_bank_hash: str,
    reset_protocol_hash: str,
    runner_registry_hash: str,
    allow_test_only: bool,
    ctx: str,
) -> None:
    if not isinstance(result, CandidateProbeResult):
        raise ProbeResultError(
            PROBE_BAD_TYPE,
            f"{ctx}: pool entries must be CandidateProbeResult "
            f"objects, got {type(result).__name__}",
        )
    # ---- tamper check: recompute the whole attestation --------------
    recomputed = compute_probe_attestation_hash(
        candidate_id=result.candidate_id,
        candidate_hash=result.candidate_hash,
        executable_artifact_id=result.executable_artifact_id,
        executable_artifact_hash=result.executable_artifact_hash,
        student_identity_hash=result.student_identity_hash,
        student_checkpoint_hash=result.student_checkpoint_hash,
        reference_identity_hash=result.reference_identity_hash,
        reference_checkpoint_hash=result.reference_checkpoint_hash,
        runner_registry_id=result.runner_registry_id,
        runner_registry_hash=result.runner_registry_hash,
        seed_bank_hash=result.seed_bank_hash,
        reset_protocol_id=result.reset_protocol_id,
        reset_protocol_hash=result.reset_protocol_hash,
        episodes_requested=result.episodes_requested,
        episodes_completed=result.episodes_completed,
        episodes_failed=result.episodes_failed,
        simulator_transitions=result.simulator_transitions,
        aggregate_metrics_hash=result.aggregate_metrics_hash,
        uncertainty_ci_hash=result.uncertainty_ci_hash,
        terminal_event_aggregates_hash=(
            result.terminal_event_aggregates_hash
        ),
        provenance_hash=result.provenance_hash,
        signer_id=result.signer_id,
        test_only=result.test_only,
    )
    if recomputed != result.attestation_hash:
        raise ProbeResultError(
            PROBE_HASH_MISMATCH,
            f"{ctx}: result {result.result_id!r} attestation "
            f"{result.attestation_hash!r} != recomputed {recomputed!r} "
            "(tampered result)",
        )
    # ---- signer gate -------------------------------------------------
    if not result.signer_id.strip():
        raise ProbeResultError(
            PROBE_UNSIGNED,
            f"{ctx}: result {result.result_id!r} carries an empty "
            "signer; unsigned results are never consumed",
        )
    if result.test_only:
        if not allow_test_only:
            raise ProbeResultError(
                PROBE_TEST_ONLY_SIGNER_REJECTED,
                f"{ctx}: result {result.result_id!r} is TEST_ONLY "
                f"(signer {result.signer_id!r}); TEST_ONLY probe "
                "results never enter a production window",
            )
        if result.signer_id != SYNTHETIC_TEST_ONLY_PROBE_SIGNER:
            raise ProbeResultError(
                PROBE_TEST_ONLY_SIGNER_REJECTED,
                f"{ctx}: TEST_ONLY results must be signed by "
                f"{SYNTHETIC_TEST_ONLY_PROBE_SIGNER!r}, got "
                f"{result.signer_id!r}",
            )
    else:
        if result.signer_id not in AUTHORIZED_PROBE_RESULT_SIGNERS:
            raise ProbeResultError(
                PROBE_SIGNER_UNAUTHORIZED,
                f"{ctx}: signer {result.signer_id!r} is not on the "
                "supervisor-owned probe registry whitelist (EMPTY "
                "this round)",
            )
    # ---- candidate / artifact binding --------------------------------
    candidate = candidates_by_hash.get(result.candidate_hash)
    if candidate is None:
        raise ProbeResultError(
            PROBE_STALE,
            f"{ctx}: result {result.result_id!r} binds candidate "
            f"{result.candidate_hash!r} which is NOT in the current "
            "window's pool (stale result)",
        )
    if result.candidate_id != candidate.candidate_id:
        raise ProbeResultError(
            PROBE_CANDIDATE_MISMATCH,
            f"{ctx}: result candidate_id {result.candidate_id!r} != "
            f"pool candidate {candidate.candidate_id!r}",
        )
    if (
        result.executable_artifact_id
        != candidate.executable_artifact_id
        or result.executable_artifact_hash
        != candidate.executable_artifact_hash
    ):
        raise ProbeResultError(
            PROBE_ARTIFACT_MISMATCH,
            f"{ctx}: result executable artifact "
            f"{result.executable_artifact_id!r}/"
            f"{result.executable_artifact_hash!r} != candidate "
            f"{candidate.executable_artifact_id!r}/"
            f"{candidate.executable_artifact_hash!r}",
        )
    # ---- Student / Reference / checkpoint binding --------------------
    if result.student_identity_hash != student_identity_hash:
        raise ProbeResultError(
            PROBE_STUDENT_MISMATCH,
            f"{ctx}: result Student identity "
            f"{result.student_identity_hash!r} != window Student "
            f"{student_identity_hash!r}",
        )
    if result.student_checkpoint_hash != student_checkpoint_hash:
        raise ProbeResultError(
            PROBE_CHECKPOINT_MISMATCH,
            f"{ctx}: result Student checkpoint "
            f"{result.student_checkpoint_hash!r} != window checkpoint "
            f"{student_checkpoint_hash!r}",
        )
    if result.reference_identity_hash != reference_identity_hash:
        raise ProbeResultError(
            PROBE_REFERENCE_MISMATCH,
            f"{ctx}: result Reference identity "
            f"{result.reference_identity_hash!r} != window Reference "
            f"{reference_identity_hash!r}",
        )
    if result.reference_checkpoint_hash != reference_checkpoint_hash:
        raise ProbeResultError(
            PROBE_CHECKPOINT_MISMATCH,
            f"{ctx}: result Reference checkpoint "
            f"{result.reference_checkpoint_hash!r} != window Reference "
            f"checkpoint {reference_checkpoint_hash!r}",
        )
    # ---- seed bank / reset protocol / runner registry ----------------
    if result.seed_bank_hash != seed_bank_hash:
        raise ProbeResultError(
            PROBE_SEED_BANK_MISMATCH,
            f"{ctx}: result seed bank {result.seed_bank_hash!r} != "
            f"window seed bank {seed_bank_hash!r}",
        )
    if result.reset_protocol_hash != reset_protocol_hash:
        raise ProbeResultError(
            PROBE_RESET_PROTOCOL_MISMATCH,
            f"{ctx}: result reset protocol "
            f"{result.reset_protocol_hash!r} != window reset protocol "
            f"{reset_protocol_hash!r}",
        )
    if result.runner_registry_hash != runner_registry_hash:
        raise ProbeResultError(
            PROBE_CANDIDATE_MISMATCH,
            f"{ctx}: result runner registry "
            f"{result.runner_registry_hash!r} != bundle-bound probe "
            f"runner registry {runner_registry_hash!r}",
        )
    marker = result.runner_registry_id.upper()
    if "MOCK" in marker or "REPLAY" in marker:
        raise ProbeResultError(
            PROBE_MOCK_RUNNER_DISGUISED,
            f"{ctx}: runner registry id {result.runner_registry_id!r} "
            "is a mock/replay identity; a mock runner never poses as "
            "a registry probe runner",
        )
    # ---- episode accounting -------------------------------------------
    if (
        result.episodes_completed + result.episodes_failed
        != result.episodes_requested
        or result.episodes_completed == 0
    ):
        raise ProbeResultError(
            PROBE_PARTIAL_EPISODES,
            f"{ctx}: result {result.result_id!r} completed "
            f"{result.episodes_completed}/{result.episodes_requested} "
            "episodes (partial probes are never consumed)",
        )


def consume_registry_signed_probe_results(
    pool: Any,
    *,
    candidates: Any,
    student_identity_hash: str,
    student_checkpoint_hash: str,
    reference_identity_hash: str,
    reference_checkpoint_hash: str,
    seed_bank_hash: str,
    reset_protocol_hash: str,
    runner_registry_hash: str,
    ctx: str,
    allow_test_only: bool = False,
) -> Tuple[CandidateProbeResult, ...]:
    """Consume the window's WHOLE probe pool fail-closed.

    ANY violation refuses the entire pool (no partial acceptance, no
    repair). The pool must cover DISTINCT candidates; duplicates fail
    closed. Returns the verified pool in input order.
    """
    if not isinstance(pool, (tuple, list)) or len(pool) == 0:
        raise ProbeResultError(
            PROBE_POOL_EMPTY,
            f"{ctx}: the probe pool is empty or not a sequence; a "
            "window without probe results selects NOTHING",
        )
    if not isinstance(candidates, (tuple, list)) or len(candidates) == 0:
        raise ProbeResultError(
            PROBE_POOL_EMPTY,
            f"{ctx}: the candidate pool is empty; probe results must "
            "bind real executable candidates",
        )
    candidates_by_hash = {}
    for candidate in candidates:
        if not isinstance(candidate, ExecutableCandidate):
            raise ProbeResultError(
                PROBE_BAD_TYPE,
                f"{ctx}: candidates must be ExecutableCandidate "
                f"objects, got {type(candidate).__name__}",
            )
        candidates_by_hash[candidate.candidate_hash] = candidate
    seen_result_ids = set()
    seen_candidate_hashes = set()
    for result in pool:
        _verify_one(
            result,
            candidates_by_hash=candidates_by_hash,
            student_identity_hash=student_identity_hash,
            student_checkpoint_hash=student_checkpoint_hash,
            reference_identity_hash=reference_identity_hash,
            reference_checkpoint_hash=reference_checkpoint_hash,
            seed_bank_hash=seed_bank_hash,
            reset_protocol_hash=reset_protocol_hash,
            runner_registry_hash=runner_registry_hash,
            allow_test_only=allow_test_only,
            ctx=ctx,
        )
        if result.result_id in seen_result_ids:
            raise ProbeResultError(
                PROBE_DUPLICATE,
                f"{ctx}: duplicate result id {result.result_id!r}",
            )
        if result.candidate_hash in seen_candidate_hashes:
            raise ProbeResultError(
                PROBE_DUPLICATE,
                f"{ctx}: candidate {result.candidate_hash!r} carries "
                "two probe results in one window",
            )
        seen_result_ids.add(result.result_id)
        seen_candidate_hashes.add(result.candidate_hash)
    return tuple(pool)
