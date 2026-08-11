"""CC2-Director tests: the post-selection driver stages.

TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION / NOT_SCIENTIFIC_EVIDENCE.

Stage 6..9 of the one-window chain: SelectionAttestation ->
CanonicalDiCodeTrainingBatchPlan (15+1) -> canonical DiCode one update
-> full run-state round trip -> signed smoke attestation.

Covered negative matrix:
* non-attestation input                    -> PLAN_BAD_TYPE
* DiCode timeline drift                    -> PLAN_BINDING_MISMATCH
* update_count != 1                        -> UPDATE_COUNT
* checkpoint runtime-bundle drift          -> PLAN_BINDING_MISMATCH
* round-trip identity vs checkpoint drift  -> PLAN_BINDING_MISMATCH
* smoke attestation binds the whole chain  -> verifies
"""
from types import SimpleNamespace

import pytest

from dicode.teachers.e1_formal import dicode_protocol as DP
from dicode.teachers.e1_formal import one_window_driver as DRV
from dicode.teachers.e1_formal import roundtrip_attestation as RA
from dicode.teachers.e1_formal import runtime_bundle as RB
from dicode.teachers.e1_formal import smoke_attestation as SM
from dicode.teachers.e1_formal import update_attestation as UA
from dicode.teachers.e1_formal.canonical import canonical_sha256
from dicode.teachers.e1_formal.selection_attestation import (
    SelectionAttestation,
)

_ANCHOR_MANIFEST_HASH = "aa" * 32
_FORMAL_REGISTRY_HASH = "ab" * 32
_TOTAL_TIMESTEPS = 2_005_401_600


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


def _bundle():
    return RB.build_test_only_runtime_bundle(
        source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
        capabilities={
            contract: SimpleNamespace(
                kind=contract, identity_id=f"test-only-{contract}"
            )
            for contract in RB.RUNTIME_CAPABILITY_CONTRACTS
        },
    )


def _plan(attestation=None):
    return DP.build_canonical_dicode_training_batch_plan(
        selection_attestation=attestation or _attestation(),
        anchor_manifest_hash=_ANCHOR_MANIFEST_HASH,
        ctx="test",
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


def _one_update_runtime(training_runtime=None):
    return DP.authorize_canonical_dicode_one_update_runtime(
        training_runtime=training_runtime or _training_runtime(),
        total_timesteps=_TOTAL_TIMESTEPS,
        session_idx=1,
        global_env_steps=4096,
        global_update_step=7,
        ctx="test",
    )


def _record(one_update_runtime=None, **overrides):
    runtime = one_update_runtime or _one_update_runtime()
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


def _update_attestation(plan, runtime=None, record=None):
    runtime = runtime or _one_update_runtime()
    record = record or _record(runtime)
    return UA.attest_exactly_one_update(
        runtime.training_runtime,
        record,
        verified_batch_hash=plan.plan_hash,
        signer_id=UA.SYNTHETIC_TEST_ONLY_TRAINING_SIGNER,
        test_only=True,
        ctx="test",
    )


def _checkpoint(bundle, plan, update_att):
    return DP.build_canonical_runstate_checkpoint(
        params_hash=update_att.params_hash_after,
        optimizer_state_hash=update_att.optimizer_state_hash_after,
        optimizer_step=update_att.optimizer_step_after,
        global_update_step=update_att.update_step_after,
        global_env_steps=update_att.global_env_steps_after,
        rng_hash=update_att.rng_hash_after,
        session_index=1,
        gen_manager_archive_hash="51" * 32,
        e1_ledger_hash="52" * 32,
        pending_worker_policy_hash="53" * 32,
        config_hash="54" * 32,
        runtime_bundle_hash=bundle.bundle_hash,
        ctx="test",
    )


def _roundtrip(checkpoint, bundle, plan, update_att):
    identity = RA.build_full_state_checkpoint_identity(
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
        student_identity_hash=update_att.student_identity_hash,
        anchor_manifest_hash=_ANCHOR_MANIFEST_HASH,
        formal_asset_registry_hash=_FORMAL_REGISTRY_HASH,
        window_hash="e" * 64,
        selection_hash="h" * 64,
        verified_batch_hash=plan.plan_hash,
        source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
    )
    restored = "61" * 32
    attestation = RA.attest_full_state_round_trip(
        identity,
        restored_state_hash=restored,
        leaf_comparison_hash=restored,
        next_policy_step_hash="62" * 32,
        fresh_process_restored=True,
        replay_identical=True,
        signer_id=RA.SYNTHETIC_TEST_ONLY_ROUNDTRIP_SIGNER,
        test_only=True,
        ctx="test",
    )
    return identity, attestation


class TestBatchCertification:
    def test_builds_the_15_plus_1_plan(self):
        attestation = _attestation()
        plan = DRV.execute_real_batch_certification(
            selection_attestation=attestation,
            anchor_manifest_hash=_ANCHOR_MANIFEST_HASH,
        )
        assert len(plan.dynamic_task_ids) == 12
        assert len(plan.non_target_anchor_ids) == 3
        assert len(plan.curriculum_task_ids) == 15
        assert plan.selection_attestation_hash == (
            attestation.attestation_hash
        )
        assert plan.target_probability == 0.20

    def test_non_attestation_refused(self):
        with pytest.raises(DP.DiCodePlanError) as excinfo:
            DRV.execute_real_batch_certification(
                selection_attestation={"selected_ids": ()},
                anchor_manifest_hash=_ANCHOR_MANIFEST_HASH,
            )
        assert excinfo.value.code == DP.PLAN_BAD_TYPE


class TestCanonicalDicodeOneUpdate:
    def test_attests_exactly_one_update_bound_to_the_plan(self):
        attestation = _attestation()
        plan = _plan(attestation)
        runtime = _one_update_runtime()
        record = _record(runtime)
        update_att = DRV.execute_canonical_dicode_one_update(
            plan=plan,
            selection_attestation=attestation,
            one_update_runtime=runtime,
            update_record=record,
            anchor_manifest_hash=_ANCHOR_MANIFEST_HASH,
            signer_id=UA.SYNTHETIC_TEST_ONLY_TRAINING_SIGNER,
            test_only=True,
        )
        assert update_att.update_count == 1
        assert update_att.optimizer_step_after == (
            update_att.optimizer_step_before + 1
        )
        assert update_att.verified_batch_hash == plan.plan_hash
        UA.verify_optimizer_update_attestation(
            update_att, runtime=runtime.training_runtime
        )

    def test_timeline_drift_refused(self):
        attestation = _attestation()
        plan = _plan(attestation)
        runtime = _one_update_runtime()
        record = _record(
            runtime, global_env_steps_after=999999  # off-timeline
        )
        with pytest.raises(DP.DiCodePlanError) as excinfo:
            DRV.execute_canonical_dicode_one_update(
                plan=plan,
                selection_attestation=attestation,
                one_update_runtime=runtime,
                update_record=record,
                anchor_manifest_hash=_ANCHOR_MANIFEST_HASH,
                signer_id=UA.SYNTHETIC_TEST_ONLY_TRAINING_SIGNER,
                test_only=True,
            )
        assert excinfo.value.code == DP.PLAN_BINDING_MISMATCH

    def test_update_count_must_be_one(self):
        attestation = _attestation()
        plan = _plan(attestation)
        runtime = _one_update_runtime()
        # a timeline-consistent record with update_count=2 still fails
        # the EXACTLY-ONE invariant (counts come from DiCode)
        record = _record(
            runtime,
            update_count=2,
            update_step_after=runtime.global_update_step + 2,
        )
        with pytest.raises(UA.UpdateAttestationError) as excinfo:
            DRV.execute_canonical_dicode_one_update(
                plan=plan,
                selection_attestation=attestation,
                one_update_runtime=runtime,
                update_record=record,
                anchor_manifest_hash=_ANCHOR_MANIFEST_HASH,
                signer_id=UA.SYNTHETIC_TEST_ONLY_TRAINING_SIGNER,
                test_only=True,
            )
        assert excinfo.value.code == UA.UPDATE_COUNT

    def test_plan_tamper_refused(self):
        from dataclasses import replace

        attestation = _attestation()
        plan = replace(_plan(attestation), plan_hash="f" * 64)
        runtime = _one_update_runtime()
        with pytest.raises(DP.DiCodePlanError) as excinfo:
            DRV.execute_canonical_dicode_one_update(
                plan=plan,
                selection_attestation=attestation,
                one_update_runtime=runtime,
                update_record=_record(runtime),
                anchor_manifest_hash=_ANCHOR_MANIFEST_HASH,
                signer_id=UA.SYNTHETIC_TEST_ONLY_TRAINING_SIGNER,
                test_only=True,
            )
        assert excinfo.value.code == DP.PLAN_HASH_MISMATCH


class TestRunstateRoundtrip:
    def test_consumes_the_full_runstate_round_trip(self):
        plan = _plan()
        runtime = _one_update_runtime()
        update_att = _update_attestation(plan, runtime)
        bundle = _bundle()
        checkpoint = _checkpoint(bundle, plan, update_att)
        roundtrip = _roundtrip(checkpoint, bundle, plan, update_att)
        attestation = DRV.consume_full_runstate_roundtrip(
            checkpoint=checkpoint,
            update_attestation=update_att,
            runtime_bundle_hash=bundle.bundle_hash,
            roundtrip_evidence=roundtrip,
        )
        RA.verify_full_state_round_trip(attestation, roundtrip[0])

    def test_runtime_bundle_drift_refused(self):
        plan = _plan()
        runtime = _one_update_runtime()
        update_att = _update_attestation(plan, runtime)
        bundle = _bundle()
        checkpoint = _checkpoint(bundle, plan, update_att)
        roundtrip = _roundtrip(checkpoint, bundle, plan, update_att)
        with pytest.raises(DP.DiCodePlanError) as excinfo:
            DRV.consume_full_runstate_roundtrip(
                checkpoint=checkpoint,
                update_attestation=update_att,
                runtime_bundle_hash="ff" * 32,  # wrong bundle
                roundtrip_evidence=roundtrip,
            )
        assert excinfo.value.code == DP.PLAN_BINDING_MISMATCH

    def test_identity_vs_checkpoint_drift_refused(self):
        from dataclasses import replace

        plan = _plan()
        runtime = _one_update_runtime()
        update_att = _update_attestation(plan, runtime)
        bundle = _bundle()
        checkpoint = _checkpoint(bundle, plan, update_att)
        identity, _attestation_obj = _roundtrip(
            checkpoint, bundle, plan, update_att
        )
        # forged identity with a different params hash
        drifted_identity = replace(
            identity, params_hash="00" * 32
        )
        roundtrip = (drifted_identity, _attestation_obj)
        with pytest.raises(DP.DiCodePlanError) as excinfo:
            DRV.consume_full_runstate_roundtrip(
                checkpoint=checkpoint,
                update_attestation=update_att,
                runtime_bundle_hash=bundle.bundle_hash,
                roundtrip_evidence=roundtrip,
            )
        assert excinfo.value.code == DP.PLAN_BINDING_MISMATCH


class TestSmokeAttestation:
    def test_binds_the_whole_chain(self):
        attestation = _attestation()
        plan = _plan(attestation)
        runtime = _one_update_runtime()
        update_att = _update_attestation(plan, runtime)
        bundle = _bundle()
        checkpoint = _checkpoint(bundle, plan, update_att)
        roundtrip = _roundtrip(checkpoint, bundle, plan, update_att)
        roundtrip_att = DRV.consume_full_runstate_roundtrip(
            checkpoint=checkpoint,
            update_attestation=update_att,
            runtime_bundle_hash=bundle.bundle_hash,
            roundtrip_evidence=roundtrip,
        )
        window_result = SimpleNamespace(window_result_hash="e" * 64)
        materials = SimpleNamespace(materials_hash="ab" * 32)
        probes = (
            SimpleNamespace(attestation_hash="b" * 64),
            SimpleNamespace(attestation_hash="bb" * 64),
        )
        expected = {
            "run_id": "TEST_ONLY_SYNTHETIC_RUN",
            "branch": "test-branch",
            "git_sha": "ab" * 32,
            "runtime_bundle_hash": bundle.bundle_hash,
            "student_identity_hash": bundle.object_identity_hash(
                "student_identity"
            ),
            "student_checkpoint_hash": "12" * 32,
            "reference_identity_hash": bundle.object_identity_hash(
                "reference_identity"
            ),
            "reference_checkpoint_hash": "22" * 32,
            "board_journal_hash": "e" * 64,
            "envcoder_artifact_pool_hash": "ab" * 32,
            "probe_pool_hash": canonical_sha256(
                ["b" * 64, "bb" * 64]
            ),
            "selection_attestation_hash": attestation.attestation_hash,
            "verified_batch_hash": plan.plan_hash,
            "update_attestation_hash": update_att.attestation_hash,
            "roundtrip_attestation_hash": (
                roundtrip_att.attestation_hash
            ),
            "formal_asset_registry_hash": _FORMAL_REGISTRY_HASH,
            "anchor_manifest_hash": _ANCHOR_MANIFEST_HASH,
        }
        smoke = DRV.build_e1_smoke_attestation(
            run_id="TEST_ONLY_SYNTHETIC_RUN",
            branch="test-branch",
            git_sha="ab" * 32,
            window_result=window_result,
            candidate_materials=materials,
            probe_pool=probes,
            plan=plan,
            update_attestation=update_att,
            roundtrip_attestation=roundtrip_att,
            runtime=bundle,
            student_checkpoint_identity="12" * 32,
            reference_checkpoint_identity="22" * 32,
            formal_asset_registry_hash=_FORMAL_REGISTRY_HASH,
            anchor_manifest_hash=_ANCHOR_MANIFEST_HASH,
            signer_id=SM.SYNTHETIC_TEST_ONLY_SMOKE_SIGNER,
            test_only=True,
        )
        SM.verify_e1_real_smoke_attestation(
            smoke, expected=expected
        )
        assert smoke.test_only is True
        assert smoke.status == SM.SMOKE_STATUS_EXECUTED
