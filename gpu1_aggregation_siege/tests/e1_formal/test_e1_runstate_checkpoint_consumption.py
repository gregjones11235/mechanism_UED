"""CC2-Director tests: CanonicalDiCodeRunStateCheckpoint consumption.

TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION / NOT_SCIENTIFIC_EVIDENCE.

方向一 consumes the shared full run-state checkpoint; params-only or
plain JSON is NEVER full-state.

Covered negative matrix:
* non-64-hex checkpoint fields              -> PLAN_BAD_TYPE
* non-negative step fields                  -> PLAN_BAD_TYPE
* runtime-bundle drift                      -> PLAN_BINDING_MISMATCH
* params drift vs update output             -> PLAN_BINDING_MISMATCH
* step drift vs update output               -> PLAN_BINDING_MISMATCH
* round-trip identity vs checkpoint drift   -> PLAN_BINDING_MISMATCH
* plain dict is not a checkpoint            -> PLAN_BAD_TYPE
"""
import pytest

from dicode.teachers.e1_formal import dicode_protocol as DP
from dicode.teachers.e1_formal import roundtrip_attestation as RA
from dicode.teachers.e1_formal import update_attestation as UA

_RUNTIME_BUNDLE_HASH = "0a" * 32


def _checkpoint(**overrides):
    kwargs = dict(
        params_hash="42" * 32,
        optimizer_state_hash="44" * 32,
        optimizer_step=43,
        global_update_step=8,
        global_env_steps=6144,
        rng_hash="46" * 32,
        session_index=1,
        gen_manager_archive_hash="51" * 32,
        e1_ledger_hash="52" * 32,
        pending_worker_policy_hash="53" * 32,
        config_hash="54" * 32,
        runtime_bundle_hash=_RUNTIME_BUNDLE_HASH,
        ctx="test",
    )
    kwargs.update(overrides)
    return DP.build_canonical_runstate_checkpoint(**kwargs)


def _update_attestation(**overrides):
    runtime = UA.authorize_original_training_runtime(
        mode=UA.TRAINING_RUNTIME_MODE_TEST_ONLY,
        run_id=UA.SYNTHETIC_TEST_ONLY_TRAINING_RUNTIME,
        student_identity_hash="11" * 32,
        network_identity_hash="31" * 32,
        loss_identity_hash="32" * 32,
        optimizer_identity_hash="33" * 32,
        rollout_schema_hash="34" * 32,
        transition_accounting_version="e1-transition-accounting-v1",
        reward_identity_hash="35" * 32,
        source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
    )
    record_kwargs = dict(
        run_id=UA.SYNTHETIC_TEST_ONLY_TRAINING_RUNTIME,
        input_checkpoint_hash="12" * 32,
        output_checkpoint_hash="13" * 32,
        params_hash_before="41" * 32,
        params_hash_after="42" * 32,
        optimizer_state_hash_before="43" * 32,
        optimizer_state_hash_after="44" * 32,
        rng_hash_before="45" * 32,
        rng_hash_after="46" * 32,
        global_env_steps_before=4096,
        global_env_steps_after=6144,
        update_step_before=7,
        update_step_after=8,
        optimizer_step_before=42,
        optimizer_step_after=43,
        rollout_batch_hash="22" * 32,
        transitions_consumed=2048,
        update_count=1,
        loss_identity_hash="32" * 32,
        optimizer_identity_hash="33" * 32,
        gradient_finite=True,
    )
    record_kwargs.update(overrides)
    record_hash = UA.compute_update_record_hash(**record_kwargs)
    record = UA.UpdateExecutionRecord(
        record_hash=record_hash, **record_kwargs
    )
    return UA.attest_exactly_one_update(
        runtime,
        record,
        verified_batch_hash="bb" * 32,
        signer_id=UA.SYNTHETIC_TEST_ONLY_TRAINING_SIGNER,
        test_only=True,
        ctx="test",
    )


def _roundtrip_identity(checkpoint):
    return RA.build_full_state_checkpoint_identity(
        params_hash=checkpoint.params_hash,
        optimizer_state_hash=checkpoint.optimizer_state_hash,
        global_env_steps=checkpoint.global_env_steps,
        update_step=checkpoint.global_update_step,
        optimizer_step=checkpoint.optimizer_step,
        training_rng_hash="45" * 32,
        env_rng_hash="46" * 32,
        env_state_hash="55" * 32,
        wrapper_state_hash="56" * 32,
        prev_action_reward_hash="57" * 32,
        policy_memory_history_hash="58" * 32,
        student_identity_hash="11" * 32,
        anchor_manifest_hash="aa" * 32,
        formal_asset_registry_hash="ab" * 32,
        window_hash="e" * 64,
        selection_hash="h" * 64,
        verified_batch_hash="bb" * 32,
        source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
    )


class TestBuildCheckpoint:
    def test_builds_the_full_run_state(self):
        checkpoint = _checkpoint()
        assert checkpoint.params_hash == "42" * 32
        assert checkpoint.optimizer_step == 43
        assert checkpoint.global_update_step == 8
        assert checkpoint.global_env_steps == 6144
        assert checkpoint.session_index == 1
        assert len(checkpoint.checkpoint_hash) == 64

    def test_bad_hash_field_refused(self):
        with pytest.raises(DP.DiCodePlanError) as excinfo:
            _checkpoint(params_hash="short")
        assert excinfo.value.code == DP.PLAN_BAD_TYPE

    def test_bad_step_field_refused(self):
        with pytest.raises(DP.DiCodePlanError) as excinfo:
            _checkpoint(optimizer_step=-1)
        assert excinfo.value.code == DP.PLAN_BAD_TYPE

    def test_plain_dict_is_never_full_state(self):
        with pytest.raises(DP.DiCodePlanError) as excinfo:
            DP.verify_runstate_checkpoint_binds_update(
                {"params_hash": "42" * 32},
                update_attestation=_update_attestation(),
                runtime_bundle_hash=_RUNTIME_BUNDLE_HASH,
                ctx="test",
            )
        assert excinfo.value.code == DP.PLAN_BAD_TYPE


class TestBindUpdate:
    def test_checkpoint_binds_the_update_output(self):
        update_att = _update_attestation()
        checkpoint = _checkpoint()
        DP.verify_runstate_checkpoint_binds_update(
            checkpoint,
            update_attestation=update_att,
            runtime_bundle_hash=_RUNTIME_BUNDLE_HASH,
            ctx="test",
        )

    def test_runtime_bundle_drift_refused(self):
        update_att = _update_attestation()
        with pytest.raises(DP.DiCodePlanError) as excinfo:
            DP.verify_runstate_checkpoint_binds_update(
                _checkpoint(),
                update_attestation=update_att,
                runtime_bundle_hash="ff" * 32,
                ctx="test",
            )
        assert excinfo.value.code == DP.PLAN_BINDING_MISMATCH

    def test_params_drift_refused(self):
        update_att = _update_attestation()
        with pytest.raises(DP.DiCodePlanError) as excinfo:
            DP.verify_runstate_checkpoint_binds_update(
                _checkpoint(params_hash="00" * 32),
                update_attestation=update_att,
                runtime_bundle_hash=_RUNTIME_BUNDLE_HASH,
                ctx="test",
            )
        assert excinfo.value.code == DP.PLAN_BINDING_MISMATCH

    def test_global_env_steps_drift_refused(self):
        update_att = _update_attestation()
        with pytest.raises(DP.DiCodePlanError) as excinfo:
            DP.verify_runstate_checkpoint_binds_update(
                _checkpoint(global_env_steps=1),
                update_attestation=update_att,
                runtime_bundle_hash=_RUNTIME_BUNDLE_HASH,
                ctx="test",
            )
        assert excinfo.value.code == DP.PLAN_BINDING_MISMATCH


class TestRoundtripIdentity:
    def test_identity_matches_checkpoint(self):
        checkpoint = _checkpoint()
        identity = _roundtrip_identity(checkpoint)
        DP.assert_roundtrip_identity_matches_checkpoint(
            identity, checkpoint, "test"
        )

    def test_identity_drift_refused(self):
        from dataclasses import replace

        checkpoint = _checkpoint()
        drifted = replace(
            _roundtrip_identity(checkpoint), params_hash="00" * 32
        )
        with pytest.raises(DP.DiCodePlanError) as excinfo:
            DP.assert_roundtrip_identity_matches_checkpoint(
                drifted, checkpoint, "test"
            )
        assert excinfo.value.code == DP.PLAN_BINDING_MISMATCH

    def test_plain_json_identity_refused(self):
        checkpoint = _checkpoint()
        with pytest.raises(DP.DiCodePlanError) as excinfo:
            DP.assert_roundtrip_identity_matches_checkpoint(
                {"params_hash": checkpoint.params_hash},
                checkpoint,
                "test",
            )
        assert excinfo.value.code == DP.PLAN_BAD_TYPE
