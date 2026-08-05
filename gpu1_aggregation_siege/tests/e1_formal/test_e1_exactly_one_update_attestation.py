"""CC2 follow-up P0-11 tests: exactly-one optimizer update attestation.

TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION / NOT_SCIENTIFIC_EVIDENCE:
no real training runs here; the production training whitelist is
EMPTY, so production attestation must fail closed. The TEST_ONLY
contract exercises the attestation surface with a conspicuously-
marked synthetic record.

Covered negative matrix:
* production runtime unauthorized           -> UPDATE_RUNTIME_UNAUTHORIZED
* replay/mock runtime forbidden            -> UPDATE_RUNTIME_FORBIDDEN
* update_count != 1                        -> UPDATE_COUNT
* optimizer step no advance                -> UPDATE_STEP_ADVANCE
* output params == input params            -> UPDATE_NO_PARAM_CHANGE
* zero transitions                         -> UPDATE_ZERO_TRANSITIONS
* record / attestation tamper              -> UPDATE_HASH_MISMATCH
* Student mismatch                         -> UPDATE_STUDENT_MISMATCH
* TEST_ONLY signer on production surface   -> UPDATE_TEST_ONLY_REJECTED
"""
from dataclasses import replace

import pytest

from dicode.teachers.e1_formal import update_attestation as UA

_RUN_ID = UA.SYNTHETIC_TEST_ONLY_TRAINING_RUNTIME
_STUDENT = "11" * 32
_INPUT = "12" * 32
_OUTPUT = "13" * 32
_BATCH = "21" * 32
_ROLLOUT = "22" * 32
_SIGNER = UA.SYNTHETIC_TEST_ONLY_TRAINING_SIGNER


def _runtime(**overrides):
    kwargs = dict(
        mode=UA.TRAINING_RUNTIME_MODE_TEST_ONLY,
        run_id=_RUN_ID,
        student_identity_hash=_STUDENT,
        network_identity_hash="31" * 32,
        loss_identity_hash="32" * 32,
        optimizer_identity_hash="33" * 32,
        rollout_schema_hash="34" * 32,
        transition_accounting_version="e1-transition-accounting-v1",
        reward_identity_hash="35" * 32,
        source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
    )
    kwargs.update(overrides)
    return UA.authorize_original_training_runtime(**kwargs)


def _record(**overrides):
    kwargs = dict(
        run_id=_RUN_ID,
        input_checkpoint_hash=_INPUT,
        output_checkpoint_hash=_OUTPUT,
        params_hash_before="41" * 32,
        params_hash_after="42" * 32,
        optimizer_state_hash_before="43" * 32,
        optimizer_state_hash_after="44" * 32,
        rng_hash_before="45" * 32,
        rng_hash_after="46" * 32,
        global_env_steps_before=4096,
        global_env_steps_after=8192,
        update_step_before=7,
        update_step_after=8,
        optimizer_step_before=42,
        optimizer_step_after=43,
        rollout_batch_hash=_ROLLOUT,
        transitions_consumed=2048,
        update_count=1,
        loss_identity_hash="32" * 32,
        optimizer_identity_hash="33" * 32,
        gradient_finite=True,
    )
    kwargs.update(overrides)
    record_hash = UA.compute_update_record_hash(**kwargs)
    return UA.UpdateExecutionRecord(record_hash=record_hash, **kwargs)


def _attest(**overrides):
    kwargs = dict(
        runtime=_runtime(),
        record=_record(),
        verified_batch_hash=_BATCH,
        signer_id=_SIGNER,
        test_only=True,
        ctx="test",
    )
    kwargs.update(overrides)
    return UA.attest_exactly_one_update(**kwargs)


class TestAuthorization:
    def test_production_whitelist_is_empty_this_round(self):
        assert UA.AUTHORIZED_TRAINING_RUNTIMES == ()

    def test_production_runtime_unauthorized(self):
        with pytest.raises(UA.UpdateAttestationError) as excinfo:
            _runtime(
                mode=UA.TRAINING_RUNTIME_MODE_PRODUCTION,
                run_id="real-training-runtime",
            )
        assert excinfo.value.code == UA.UPDATE_RUNTIME_UNAUTHORIZED

    def test_replay_and_mock_forbidden(self):
        for run_id in ("replay", "mock"):
            with pytest.raises(UA.UpdateAttestationError) as excinfo:
                _runtime(
                    mode=UA.TRAINING_RUNTIME_MODE_PRODUCTION,
                    run_id=run_id,
                )
            assert excinfo.value.code == UA.UPDATE_RUNTIME_FORBIDDEN

    def test_test_only_runtime_assembles(self):
        runtime = _runtime()
        assert runtime.run_id == _RUN_ID
        assert len(runtime.runtime_hash) == 64
        assert runtime.student_identity_hash == _STUDENT

    def test_wrong_test_only_runtime_refused(self):
        with pytest.raises(UA.UpdateAttestationError) as excinfo:
            _runtime(run_id="other-test-only-runtime")
        assert excinfo.value.code == UA.UPDATE_RUNTIME_FORBIDDEN


class TestExecutionRecordInvariants:
    def test_valid_record_passes(self):
        UA.verify_update_execution_record(_record(), "test")

    def test_update_count_must_be_one(self):
        with pytest.raises(UA.UpdateAttestationError) as excinfo:
            UA.verify_update_execution_record(
                _record(update_count=2), "test"
            )
        assert excinfo.value.code == UA.UPDATE_COUNT

    def test_optimizer_step_must_advance_by_one(self):
        with pytest.raises(UA.UpdateAttestationError) as excinfo:
            UA.verify_update_execution_record(
                _record(
                    optimizer_step_before=42, optimizer_step_after=42
                ),
                "test",
            )
        assert excinfo.value.code == UA.UPDATE_STEP_ADVANCE

    def test_no_parameter_change_refused(self):
        with pytest.raises(UA.UpdateAttestationError) as excinfo:
            UA.verify_update_execution_record(
                _record(params_hash_after="41" * 32), "test"
            )
        assert excinfo.value.code == UA.UPDATE_NO_PARAM_CHANGE

    def test_zero_transitions_refused(self):
        with pytest.raises(UA.UpdateAttestationError) as excinfo:
            UA.verify_update_execution_record(
                _record(transitions_consumed=0), "test"
            )
        assert excinfo.value.code == UA.UPDATE_ZERO_TRANSITIONS

    def test_record_tamper_detected(self):
        tampered = replace(_record(), update_count=3)
        with pytest.raises(UA.UpdateAttestationError) as excinfo:
            UA.verify_update_execution_record(tampered, "test")
        assert excinfo.value.code == UA.UPDATE_HASH_MISMATCH


class TestAttestation:
    def test_test_only_attestation_binds_everything(self):
        attested = _attest()
        assert attested.run_id == _RUN_ID
        assert attested.student_identity_hash == _STUDENT
        assert attested.input_checkpoint_hash == _INPUT
        assert attested.output_checkpoint_hash == _OUTPUT
        assert attested.optimizer_step_after == (
            attested.optimizer_step_before + 1
        )
        assert attested.update_count == 1
        assert attested.verified_batch_hash == _BATCH
        assert attested.test_only is True
        assert len(attested.attestation_hash) == 64

    def test_verification_passes_untampered(self):
        runtime = _runtime()
        attested = _attest(runtime=runtime)
        UA.verify_optimizer_update_attestation(
            attested, runtime=runtime
        )

    def test_attestation_tamper_detected(self):
        runtime = _runtime()
        attested = _attest(runtime=runtime)
        tampered = replace(attested, update_count=2)
        with pytest.raises(UA.UpdateAttestationError) as excinfo:
            UA.verify_optimizer_update_attestation(
                tampered, runtime=runtime
            )
        assert excinfo.value.code == UA.UPDATE_HASH_MISMATCH

    def test_student_mismatch_detected(self):
        runtime = _runtime()
        other_runtime = _runtime(student_identity_hash="ff" * 32)
        attested = _attest(runtime=runtime)
        with pytest.raises(UA.UpdateAttestationError) as excinfo:
            UA.verify_optimizer_update_attestation(
                attested, runtime=other_runtime
            )
        assert excinfo.value.code == UA.UPDATE_STUDENT_MISMATCH

    def test_test_only_signer_refused_on_production_path(self):
        with pytest.raises(UA.UpdateAttestationError) as excinfo:
            _attest(test_only=False)
        # production surface: the synthetic runtime is not authorized
        assert excinfo.value.code == UA.UPDATE_RUNTIME_UNAUTHORIZED

    def test_wrong_test_only_signer_refused(self):
        with pytest.raises(UA.UpdateAttestationError) as excinfo:
            _attest(signer_id="attacker-train-signer")
        assert excinfo.value.code == UA.UPDATE_TEST_ONLY_REJECTED
