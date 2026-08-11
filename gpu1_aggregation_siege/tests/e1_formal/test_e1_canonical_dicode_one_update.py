"""CC2-Director tests: the canonical DiCode one-update runtime.

TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION / NOT_SCIENTIFIC_EVIDENCE.

The ONLY training timeline is the DiCode timeline
(config.training.total_timesteps, global_env_steps,
global_update_step, session index); the update's before/after counts
must come from that timeline. A fixed local longrun horizon is never
a substitute.

Covered negative matrix:
* non-positive total_timesteps              -> PLAN_COUNT
* bad timeline field types                  -> PLAN_BAD_TYPE
* non-OriginalTrainingRuntime               -> PLAN_BAD_TYPE
* update_count != 1                         -> UPDATE_COUNT
* timeline drift                            -> PLAN_BINDING_MISMATCH
* plan tamper                               -> PLAN_HASH_MISMATCH
* wrong TEST_ONLY signer                    -> UPDATE_TEST_ONLY_REJECTED
"""
from types import SimpleNamespace

import pytest

from dicode.teachers.e1_formal import dicode_protocol as DP
from dicode.teachers.e1_formal import one_window_driver as DRV
from dicode.teachers.e1_formal import runtime_bundle as RB
from dicode.teachers.e1_formal import update_attestation as UA
from dicode.teachers.e1_formal.selection_attestation import (
    SelectionAttestation,
)

_TOTAL_TIMESTEPS = 2_005_401_600
_ANCHOR_MANIFEST_HASH = "aa" * 32


def _attestation():
    return SelectionAttestation(
        window_id="e1-w000001",
        window_hash="e" * 64,
        selected_ids=tuple(f"cand-{i:03d}" for i in range(12)),
        candidate_pool_hash="a" * 64,
        probe_pool_hash="b" * 64,
        signals_pool_hash="c" * 64,
        selector_source_hash="d" * 64,
        constants_hash="e2" * 32,
        weights_hash="f" * 64,
        family_cap=6,
        seed=7,
        k=12,
        selected_set_hash="g" * 64,
        selection_hash="h" * 64,
        attestation_hash="i" * 64,
    )


def _training_runtime():
    return UA.authorize_original_training_runtime(
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


def _one_update_runtime(**overrides):
    kwargs = dict(
        training_runtime=_training_runtime(),
        total_timesteps=_TOTAL_TIMESTEPS,
        session_idx=1,
        global_env_steps=4096,
        global_update_step=7,
        ctx="test",
    )
    kwargs.update(overrides)
    return DP.authorize_canonical_dicode_one_update_runtime(**kwargs)


def _record(runtime, **overrides):
    kwargs = dict(
        run_id=UA.SYNTHETIC_TEST_ONLY_TRAINING_RUNTIME,
        input_checkpoint_hash="12" * 32,
        output_checkpoint_hash="13" * 32,
        params_hash_before="41" * 32,
        params_hash_after="42" * 32,
        optimizer_state_hash_before="43" * 32,
        optimizer_state_hash_after="44" * 32,
        rng_hash_before="45" * 32,
        rng_hash_after="46" * 32,
        global_env_steps_before=runtime.global_env_steps,
        global_env_steps_after=runtime.global_env_steps + 2048,
        update_step_before=runtime.global_update_step,
        update_step_after=runtime.global_update_step + 1,
        optimizer_step_before=42,
        optimizer_step_after=43,
        rollout_batch_hash="22" * 32,
        transitions_consumed=2048,
        update_count=1,
        loss_identity_hash="32" * 32,
        optimizer_identity_hash="33" * 32,
        gradient_finite=True,
    )
    kwargs.update(overrides)
    record_hash = UA.compute_update_record_hash(**kwargs)
    return UA.UpdateExecutionRecord(record_hash=record_hash, **kwargs)


def _plan():
    return DP.build_canonical_dicode_training_batch_plan(
        selection_attestation=_attestation(),
        anchor_manifest_hash=_ANCHOR_MANIFEST_HASH,
        ctx="test",
    )


class TestAuthorizeRuntime:
    def test_authorizes_the_dicode_timeline(self):
        runtime = _one_update_runtime()
        assert runtime.total_timesteps == _TOTAL_TIMESTEPS
        assert runtime.global_update_step == 7
        assert runtime.global_env_steps == 4096
        assert len(runtime.runtime_hash) == 64

    def test_non_positive_total_timesteps_refused(self):
        with pytest.raises(DP.DiCodePlanError) as excinfo:
            _one_update_runtime(total_timesteps=0)
        assert excinfo.value.code == DP.PLAN_COUNT
        with pytest.raises(DP.DiCodePlanError) as excinfo:
            _one_update_runtime(total_timesteps=-5)
        assert excinfo.value.code == DP.PLAN_BAD_TYPE

    def test_bad_timeline_field_type_refused(self):
        with pytest.raises(DP.DiCodePlanError) as excinfo:
            _one_update_runtime(session_idx=True)
        assert excinfo.value.code == DP.PLAN_BAD_TYPE

    def test_non_training_runtime_refused(self):
        with pytest.raises(DP.DiCodePlanError) as excinfo:
            DP.authorize_canonical_dicode_one_update_runtime(
                training_runtime=SimpleNamespace(),
                total_timesteps=_TOTAL_TIMESTEPS,
                session_idx=1,
                global_env_steps=4096,
                global_update_step=7,
                ctx="test",
            )
        assert excinfo.value.code == DP.PLAN_BAD_TYPE


class TestExecuteOneUpdate:
    def _run(self, **overrides):
        plan = _plan()
        runtime = _one_update_runtime()
        record = _record(runtime)
        kwargs = dict(
            plan=plan,
            selection_attestation=_attestation(),
            one_update_runtime=runtime,
            update_record=record,
            anchor_manifest_hash=_ANCHOR_MANIFEST_HASH,
            signer_id=UA.SYNTHETIC_TEST_ONLY_TRAINING_SIGNER,
            test_only=True,
        )
        kwargs.update(overrides)
        return DRV.execute_canonical_dicode_one_update(**kwargs)

    def test_happy_path_counts_come_from_dicode(self):
        update_att = self._run()
        assert update_att.update_count == 1
        assert update_att.verified_batch_hash == _plan().plan_hash
        # the update output counts came from the DiCode timeline
        assert update_att.update_step_after == 8
        assert update_att.global_env_steps_after == 6144

    def test_update_count_must_be_one(self):
        # a timeline-consistent record with update_count=2 still fails
        # the EXACTLY-ONE invariant (the counts come from DiCode; an
        # update count other than 1 is never attested)
        runtime = _one_update_runtime()
        with pytest.raises(UA.UpdateAttestationError) as excinfo:
            self._run(
                update_record=_record(
                    runtime,
                    update_count=2,
                    update_step_before=runtime.global_update_step,
                    update_step_after=runtime.global_update_step + 2,
                )
            )
        assert excinfo.value.code == UA.UPDATE_COUNT

    def test_timeline_drift_refused(self):
        with pytest.raises(DP.DiCodePlanError) as excinfo:
            self._run(
                update_record=_record(
                    _one_update_runtime(), global_env_steps_after=1
                )
            )
        assert excinfo.value.code == DP.PLAN_BINDING_MISMATCH

    def test_plan_tamper_refused(self):
        from dataclasses import replace

        with pytest.raises(DP.DiCodePlanError) as excinfo:
            self._run(plan=replace(_plan(), plan_hash="f" * 64))
        assert excinfo.value.code == DP.PLAN_HASH_MISMATCH

    def test_wrong_test_only_signer_refused(self):
        with pytest.raises(UA.UpdateAttestationError) as excinfo:
            self._run(signer_id="attacker-train-signer")
        assert excinfo.value.code == UA.UPDATE_TEST_ONLY_REJECTED

    def test_production_surface_refused(self):
        with pytest.raises(UA.UpdateAttestationError) as excinfo:
            self._run(test_only=False)
        assert excinfo.value.code == UA.UPDATE_RUNTIME_UNAUTHORIZED
