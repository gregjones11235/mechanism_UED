"""CC2 follow-up P0-11: exactly-one optimizer update attestation.

The shared OriginalTrainingRuntime issues the update; E1 consumes the
immutable ``OptimizerUpdateAttestation`` — never a dict self-report,
never a caller-supplied ``update_count``::

    runtime  = authorize_original_training_runtime(mode=..., ...)
    record   = training_surface.execute_exactly_one_update(...)  # real
    attested = attest_exactly_one_update(runtime, record, ...)
    verify_optimizer_update_attestation(attested, runtime, ...)

The attestation binds the Student identity, the input/output
checkpoint hashes, the params / optimizer-state / RNG hashes before
and after, the global-env / update / optimizer step counters, the
rollout batch hash, the 12+4 verified batch hash, the transitions
consumed, ``update_count``, the loss/optimizer identities, and the
gradient finiteness report. Mechanical invariants verified on
consumption:

* optimizer_step_after == optimizer_step_before + 1 — an update that
  does not advance the optimizer step is NOT an update;
* update_count == 1 (EXACTLY one update per window);
* output params hash != input params hash (real parameter change);
* transitions_consumed > 0 (an update over zero transitions never
  attests).

This round performs NO real training: PRODUCTION authorization is
impossible (empty whitelist); the TEST_ONLY contract exercises the
attestation surface with a conspicuously-marked synthetic record.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .canonical import canonical_sha256
from .schemas import E1SchemaError

#: authorization modes
TRAINING_RUNTIME_MODE_PRODUCTION = "PRODUCTION"
TRAINING_RUNTIME_MODE_TEST_ONLY = "TEST_ONLY"

#: synthetic TEST_ONLY identities (greppable)
SYNTHETIC_TEST_ONLY_TRAINING_SIGNER = "SYNTHETIC_TEST_ONLY_TRAINING_SIGNER"
SYNTHETIC_TEST_ONLY_TRAINING_RUNTIME = "SYNTHETIC_TEST_ONLY_TRAINING_RUNTIME"

#: supervisor-owned production whitelist — EMPTY this round
AUTHORIZED_TRAINING_RUNTIMES: tuple = ()

#: attestation version
UPDATE_ATTESTATION_VERSION = "e1-optimizer-update-attestation-v1"

# fail-closed codes (greppable)
UPDATE_BAD_TYPE = "UPDATE_BAD_TYPE"
UPDATE_RUNTIME_UNAUTHORIZED = "UPDATE_RUNTIME_UNAUTHORIZED"
UPDATE_RUNTIME_FORBIDDEN = "UPDATE_RUNTIME_FORBIDDEN"
UPDATE_TEST_ONLY_REJECTED = "UPDATE_TEST_ONLY_REJECTED"
UPDATE_HASH_MISMATCH = "UPDATE_HASH_MISMATCH"
UPDATE_STUDENT_MISMATCH = "UPDATE_STUDENT_MISMATCH"
UPDATE_STEP_ADVANCE = "UPDATE_STEP_ADVANCE"
UPDATE_COUNT = "UPDATE_COUNT"
UPDATE_NO_PARAM_CHANGE = "UPDATE_NO_PARAM_CHANGE"
UPDATE_ZERO_TRANSITIONS = "UPDATE_ZERO_TRANSITIONS"
UPDATE_BATCH_MISMATCH = "UPDATE_BATCH_MISMATCH"


class UpdateAttestationError(E1SchemaError):
    """Fail-closed update-attestation violation; ``code`` is
    greppable."""


def _require_sha64(value: Any, name: str, ctx: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise UpdateAttestationError(
            UPDATE_BAD_TYPE,
            f"{ctx}: {name} must be a 64-hex hash, got {value!r}",
        )
    return value


def _require_count(value: Any, name: str, ctx: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise UpdateAttestationError(
            UPDATE_BAD_TYPE,
            f"{ctx}: {name} must be a non-negative int, got {value!r}",
        )
    return value


@dataclass(frozen=True)
class OriginalTrainingRuntime:
    """The shared OriginalTrainingRuntime identity (immutable)."""

    mode: str
    run_id: str
    student_identity_hash: str
    network_identity_hash: str
    loss_identity_hash: str
    optimizer_identity_hash: str
    rollout_schema_hash: str
    transition_accounting_version: str
    reward_identity_hash: str
    source_commit: str
    runtime_hash: str


def authorize_original_training_runtime(
    *,
    mode: str,
    run_id: str,
    student_identity_hash: str,
    network_identity_hash: str,
    loss_identity_hash: str,
    optimizer_identity_hash: str,
    rollout_schema_hash: str,
    transition_accounting_version: str,
    reward_identity_hash: str,
    source_commit: str,
) -> OriginalTrainingRuntime:
    """Authorize the training runtime fail-closed on every field.

    PRODUCTION this round is impossible (the whitelist is empty); the
    TEST_ONLY contract uses the synthetic runtime id + signer.
    """
    ctx = "update_attestation.authorize"
    if mode not in (
        TRAINING_RUNTIME_MODE_PRODUCTION,
        TRAINING_RUNTIME_MODE_TEST_ONLY,
    ):
        raise UpdateAttestationError(
            UPDATE_BAD_TYPE,
            f"{ctx}: mode must be one of "
            f"{[TRAINING_RUNTIME_MODE_PRODUCTION, TRAINING_RUNTIME_MODE_TEST_ONLY]}, got {mode!r}",
        )
    if mode == TRAINING_RUNTIME_MODE_PRODUCTION:
        if run_id in ("replay", "mock", "SYNTHETIC_TEST_ONLY_TRAINING_RUNTIME"):
            raise UpdateAttestationError(
                UPDATE_RUNTIME_FORBIDDEN,
                f"{ctx}: runtime {run_id!r} is a replay/mock/synthetic "
                "identity; it may never serve the production training "
                "surface",
            )
        if run_id not in AUTHORIZED_TRAINING_RUNTIMES:
            raise UpdateAttestationError(
                UPDATE_RUNTIME_UNAUTHORIZED,
                f"{ctx}: training runtime {run_id!r} is not on the "
                "supervisor-owned whitelist (EMPTY this round); no real "
                "optimizer update is authorized",
            )
    else:
        if run_id != SYNTHETIC_TEST_ONLY_TRAINING_RUNTIME:
            raise UpdateAttestationError(
                UPDATE_RUNTIME_FORBIDDEN,
                f"{ctx}: TEST_ONLY runtimes must use "
                f"{SYNTHETIC_TEST_ONLY_TRAINING_RUNTIME!r}, got "
                f"{run_id!r}",
            )
    fields = dict(
        mode=mode,
        run_id=run_id,
        student_identity_hash=_require_sha64(
            student_identity_hash, "student_identity_hash", ctx
        ),
        network_identity_hash=_require_sha64(
            network_identity_hash, "network_identity_hash", ctx
        ),
        loss_identity_hash=_require_sha64(
            loss_identity_hash, "loss_identity_hash", ctx
        ),
        optimizer_identity_hash=_require_sha64(
            optimizer_identity_hash, "optimizer_identity_hash", ctx
        ),
        rollout_schema_hash=_require_sha64(
            rollout_schema_hash, "rollout_schema_hash", ctx
        ),
        transition_accounting_version=(
            transition_accounting_version
        ),
        reward_identity_hash=_require_sha64(
            reward_identity_hash, "reward_identity_hash", ctx
        ),
        source_commit=source_commit,
    )
    runtime_hash = canonical_sha256(
        {
            "attestation_version": UPDATE_ATTESTATION_VERSION,
            **fields,
        }
    )
    return OriginalTrainingRuntime(
        mode=mode,
        run_id=run_id,
        student_identity_hash=fields["student_identity_hash"],
        network_identity_hash=fields["network_identity_hash"],
        loss_identity_hash=fields["loss_identity_hash"],
        optimizer_identity_hash=fields["optimizer_identity_hash"],
        rollout_schema_hash=fields["rollout_schema_hash"],
        transition_accounting_version=transition_accounting_version,
        reward_identity_hash=fields["reward_identity_hash"],
        source_commit=source_commit,
        runtime_hash=runtime_hash,
    )


@dataclass(frozen=True)
class UpdateExecutionRecord:
    """The structured record a real training surface returns (never a
    caller-supplied dict)."""

    run_id: str
    input_checkpoint_hash: str
    output_checkpoint_hash: str
    params_hash_before: str
    params_hash_after: str
    optimizer_state_hash_before: str
    optimizer_state_hash_after: str
    rng_hash_before: str
    rng_hash_after: str
    global_env_steps_before: int
    global_env_steps_after: int
    update_step_before: int
    update_step_after: int
    optimizer_step_before: int
    optimizer_step_after: int
    rollout_batch_hash: str
    transitions_consumed: int
    update_count: int
    loss_identity_hash: str
    optimizer_identity_hash: str
    gradient_finite: bool
    record_hash: str


def compute_update_record_hash(
    *,
    run_id: str,
    input_checkpoint_hash: str,
    output_checkpoint_hash: str,
    params_hash_before: str,
    params_hash_after: str,
    optimizer_state_hash_before: str,
    optimizer_state_hash_after: str,
    rng_hash_before: str,
    rng_hash_after: str,
    global_env_steps_before: int,
    global_env_steps_after: int,
    update_step_before: int,
    update_step_after: int,
    optimizer_step_before: int,
    optimizer_step_after: int,
    rollout_batch_hash: str,
    transitions_consumed: int,
    update_count: int,
    loss_identity_hash: str,
    optimizer_identity_hash: str,
    gradient_finite: bool,
) -> str:
    return canonical_sha256(
        {
            "attestation_version": UPDATE_ATTESTATION_VERSION,
            "run_id": run_id,
            "input_checkpoint_hash": input_checkpoint_hash,
            "output_checkpoint_hash": output_checkpoint_hash,
            "params_hash_before": params_hash_before,
            "params_hash_after": params_hash_after,
            "optimizer_state_hash_before": optimizer_state_hash_before,
            "optimizer_state_hash_after": optimizer_state_hash_after,
            "rng_hash_before": rng_hash_before,
            "rng_hash_after": rng_hash_after,
            "global_env_steps_before": global_env_steps_before,
            "global_env_steps_after": global_env_steps_after,
            "update_step_before": update_step_before,
            "update_step_after": update_step_after,
            "optimizer_step_before": optimizer_step_before,
            "optimizer_step_after": optimizer_step_after,
            "rollout_batch_hash": rollout_batch_hash,
            "transitions_consumed": transitions_consumed,
            "update_count": update_count,
            "loss_identity_hash": loss_identity_hash,
            "optimizer_identity_hash": optimizer_identity_hash,
            "gradient_finite": gradient_finite,
        }
    )


def verify_update_execution_record(record: Any, ctx: str) -> None:
    """Mechanical invariants of ONE real update (P0-11)."""
    if not isinstance(record, UpdateExecutionRecord):
        raise UpdateAttestationError(
            UPDATE_BAD_TYPE,
            f"{ctx}: expected an UpdateExecutionRecord, got "
            f"{type(record).__name__}",
        )
    recomputed = compute_update_record_hash(
        run_id=record.run_id,
        input_checkpoint_hash=record.input_checkpoint_hash,
        output_checkpoint_hash=record.output_checkpoint_hash,
        params_hash_before=record.params_hash_before,
        params_hash_after=record.params_hash_after,
        optimizer_state_hash_before=record.optimizer_state_hash_before,
        optimizer_state_hash_after=record.optimizer_state_hash_after,
        rng_hash_before=record.rng_hash_before,
        rng_hash_after=record.rng_hash_after,
        global_env_steps_before=record.global_env_steps_before,
        global_env_steps_after=record.global_env_steps_after,
        update_step_before=record.update_step_before,
        update_step_after=record.update_step_after,
        optimizer_step_before=record.optimizer_step_before,
        optimizer_step_after=record.optimizer_step_after,
        rollout_batch_hash=record.rollout_batch_hash,
        transitions_consumed=record.transitions_consumed,
        update_count=record.update_count,
        loss_identity_hash=record.loss_identity_hash,
        optimizer_identity_hash=record.optimizer_identity_hash,
        gradient_finite=record.gradient_finite,
    )
    if recomputed != record.record_hash:
        raise UpdateAttestationError(
            UPDATE_HASH_MISMATCH,
            f"{ctx}: record_hash {record.record_hash!r} != recomputed "
            f"{recomputed!r} (tampered record)",
        )
    if record.update_count != 1:
        raise UpdateAttestationError(
            UPDATE_COUNT,
            f"{ctx}: update_count is {record.update_count}; EXACTLY ONE "
            "optimizer update per window is required",
        )
    if record.optimizer_step_after != record.optimizer_step_before + 1:
        raise UpdateAttestationError(
            UPDATE_STEP_ADVANCE,
            f"{ctx}: optimizer_step_after {record.optimizer_step_after} "
            f"!= before+1 ({record.optimizer_step_before} + 1); an "
            "update that does not advance the optimizer step is NOT an "
            "update",
        )
    if record.params_hash_after == record.params_hash_before:
        raise UpdateAttestationError(
            UPDATE_NO_PARAM_CHANGE,
            f"{ctx}: output params hash equals the input — a real "
            "update changes parameters",
        )
    if record.transitions_consumed <= 0:
        raise UpdateAttestationError(
            UPDATE_ZERO_TRANSITIONS,
            f"{ctx}: an update over zero transitions never attests",
        )
    if not isinstance(record.gradient_finite, bool):
        raise UpdateAttestationError(
            UPDATE_BAD_TYPE,
            f"{ctx}: gradient_finite must be bool, got "
            f"{record.gradient_finite!r}",
        )


@dataclass(frozen=True)
class OptimizerUpdateAttestation:
    """The consumed, hash-bound attestation of exactly one update."""

    run_id: str
    student_identity_hash: str
    input_checkpoint_hash: str
    output_checkpoint_hash: str
    params_hash_before: str
    params_hash_after: str
    optimizer_state_hash_before: str
    optimizer_state_hash_after: str
    rng_hash_before: str
    rng_hash_after: str
    global_env_steps_before: int
    global_env_steps_after: int
    update_step_before: int
    update_step_after: int
    optimizer_step_before: int
    optimizer_step_after: int
    rollout_batch_hash: str
    verified_batch_hash: str  # the certified 12+4 batch hash
    transitions_consumed: int
    update_count: int
    loss_identity_hash: str
    optimizer_identity_hash: str
    gradient_finite: bool
    signer_id: str
    verifier_hash: str
    attestation_hash: str
    test_only: bool


def attest_exactly_one_update(
    runtime: Any,
    record: Any,
    *,
    verified_batch_hash: str,
    signer_id: str,
    test_only: bool = False,
    ctx: str = "update_attestation.attest",
) -> OptimizerUpdateAttestation:
    """Wrap ONE structured update record into the consumed attestation.

    Verifies the record's mechanical invariants first (exactly one
    update, optimizer step advance, parameter change, nonzero
    transitions), binds the runtime's Student identity + batch hash,
    then signs. ``signer_id`` is gated (synthetic for TEST_ONLY;
    supervisor whitelist for production).
    """
    if not isinstance(runtime, OriginalTrainingRuntime):
        raise UpdateAttestationError(
            UPDATE_BAD_TYPE,
            f"{ctx}: runtime must be an OriginalTrainingRuntime, got "
            f"{type(runtime).__name__}",
        )
    verify_update_execution_record(record, ctx)
    if test_only:
        if signer_id != SYNTHETIC_TEST_ONLY_TRAINING_SIGNER:
            raise UpdateAttestationError(
                UPDATE_TEST_ONLY_REJECTED,
                f"{ctx}: TEST_ONLY attestations must be signed by "
                f"{SYNTHETIC_TEST_ONLY_TRAINING_SIGNER!r}, got "
                f"{signer_id!r}",
            )
        if runtime.mode != TRAINING_RUNTIME_MODE_TEST_ONLY:
            raise UpdateAttestationError(
                UPDATE_TEST_ONLY_REJECTED,
                f"{ctx}: a TEST_ONLY attestation requires a TEST_ONLY "
                "runtime",
            )
    else:
        if runtime.run_id not in AUTHORIZED_TRAINING_RUNTIMES:
            raise UpdateAttestationError(
                UPDATE_RUNTIME_UNAUTHORIZED,
                f"{ctx}: training runtime {runtime.run_id!r} is not "
                "authorized for production",
            )
    if record.run_id != runtime.run_id:
        raise UpdateAttestationError(
            UPDATE_HASH_MISMATCH,
            f"{ctx}: record run_id {record.run_id!r} != runtime "
            f"{runtime.run_id!r}",
        )
    verified_batch_hash = _require_sha64(
        verified_batch_hash, "verified_batch_hash", ctx
    )
    verifier_hash = canonical_sha256(
        {"verifier": UPDATE_ATTESTATION_VERSION, "runtime": runtime.run_id}
    )
    attestation_hash = canonical_sha256(
        {
            "record_hash": record.record_hash,
            "student_identity_hash": runtime.student_identity_hash,
            "verified_batch_hash": verified_batch_hash,
            "signer_id": signer_id,
            "verifier_hash": verifier_hash,
            "test_only": test_only,
        }
    )
    return OptimizerUpdateAttestation(
        run_id=record.run_id,
        student_identity_hash=runtime.student_identity_hash,
        input_checkpoint_hash=record.input_checkpoint_hash,
        output_checkpoint_hash=record.output_checkpoint_hash,
        params_hash_before=record.params_hash_before,
        params_hash_after=record.params_hash_after,
        optimizer_state_hash_before=record.optimizer_state_hash_before,
        optimizer_state_hash_after=record.optimizer_state_hash_after,
        rng_hash_before=record.rng_hash_before,
        rng_hash_after=record.rng_hash_after,
        global_env_steps_before=record.global_env_steps_before,
        global_env_steps_after=record.global_env_steps_after,
        update_step_before=record.update_step_before,
        update_step_after=record.update_step_after,
        optimizer_step_before=record.optimizer_step_before,
        optimizer_step_after=record.optimizer_step_after,
        rollout_batch_hash=record.rollout_batch_hash,
        verified_batch_hash=verified_batch_hash,
        transitions_consumed=record.transitions_consumed,
        update_count=record.update_count,
        loss_identity_hash=record.loss_identity_hash,
        optimizer_identity_hash=record.optimizer_identity_hash,
        gradient_finite=record.gradient_finite,
        signer_id=signer_id,
        verifier_hash=verifier_hash,
        attestation_hash=attestation_hash,
        test_only=test_only,
    )


def verify_optimizer_update_attestation(
    attested: Any,
    *,
    runtime: Any,
    ctx: str = "update_attestation.verify",
) -> None:
    """Re-derive the attestation + record fail-closed."""
    if not isinstance(attested, OptimizerUpdateAttestation):
        raise UpdateAttestationError(
            UPDATE_BAD_TYPE,
            f"{ctx}: expected an OptimizerUpdateAttestation, got "
            f"{type(attested).__name__}",
        )
    if not isinstance(runtime, OriginalTrainingRuntime):
        raise UpdateAttestationError(
            UPDATE_BAD_TYPE,
            f"{ctx}: runtime must be an OriginalTrainingRuntime, got "
            f"{type(runtime).__name__}",
        )
    if attested.run_id != runtime.run_id:
        raise UpdateAttestationError(
            UPDATE_HASH_MISMATCH,
            f"{ctx}: attested run {attested.run_id!r} != runtime "
            f"{runtime.run_id!r}",
        )
    if attested.student_identity_hash != runtime.student_identity_hash:
        raise UpdateAttestationError(
            UPDATE_STUDENT_MISMATCH,
            f"{ctx}: attested Student "
            f"{attested.student_identity_hash!r} != runtime Student "
            f"{runtime.student_identity_hash!r}",
        )
    recomputed_record = compute_update_record_hash(
        run_id=attested.run_id,
        input_checkpoint_hash=attested.input_checkpoint_hash,
        output_checkpoint_hash=attested.output_checkpoint_hash,
        params_hash_before=attested.params_hash_before,
        params_hash_after=attested.params_hash_after,
        optimizer_state_hash_before=attested.optimizer_state_hash_before,
        optimizer_state_hash_after=attested.optimizer_state_hash_after,
        rng_hash_before=attested.rng_hash_before,
        rng_hash_after=attested.rng_hash_after,
        global_env_steps_before=attested.global_env_steps_before,
        global_env_steps_after=attested.global_env_steps_after,
        update_step_before=attested.update_step_before,
        update_step_after=attested.update_step_after,
        optimizer_step_before=attested.optimizer_step_before,
        optimizer_step_after=attested.optimizer_step_after,
        rollout_batch_hash=attested.rollout_batch_hash,
        transitions_consumed=attested.transitions_consumed,
        update_count=attested.update_count,
        loss_identity_hash=attested.loss_identity_hash,
        optimizer_identity_hash=attested.optimizer_identity_hash,
        gradient_finite=attested.gradient_finite,
    )
    recomputed_attestation = canonical_sha256(
        {
            "record_hash": recomputed_record,
            "student_identity_hash": attested.student_identity_hash,
            "verified_batch_hash": attested.verified_batch_hash,
            "signer_id": attested.signer_id,
            "verifier_hash": attested.verifier_hash,
            "test_only": attested.test_only,
        }
    )
    if recomputed_attestation != attested.attestation_hash:
        raise UpdateAttestationError(
            UPDATE_HASH_MISMATCH,
            f"{ctx}: attestation_hash {attested.attestation_hash!r} != "
            f"recomputed {recomputed_attestation!r} (tampered)",
        )
    # re-run the mechanical invariants over the attested fields
    record = UpdateExecutionRecord(
        run_id=attested.run_id,
        input_checkpoint_hash=attested.input_checkpoint_hash,
        output_checkpoint_hash=attested.output_checkpoint_hash,
        params_hash_before=attested.params_hash_before,
        params_hash_after=attested.params_hash_after,
        optimizer_state_hash_before=attested.optimizer_state_hash_before,
        optimizer_state_hash_after=attested.optimizer_state_hash_after,
        rng_hash_before=attested.rng_hash_before,
        rng_hash_after=attested.rng_hash_after,
        global_env_steps_before=attested.global_env_steps_before,
        global_env_steps_after=attested.global_env_steps_after,
        update_step_before=attested.update_step_before,
        update_step_after=attested.update_step_after,
        optimizer_step_before=attested.optimizer_step_before,
        optimizer_step_after=attested.optimizer_step_after,
        rollout_batch_hash=attested.rollout_batch_hash,
        transitions_consumed=attested.transitions_consumed,
        update_count=attested.update_count,
        loss_identity_hash=attested.loss_identity_hash,
        optimizer_identity_hash=attested.optimizer_identity_hash,
        gradient_finite=attested.gradient_finite,
        record_hash=recomputed_record,
    )
    verify_update_execution_record(record, ctx)
