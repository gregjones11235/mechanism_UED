"""CC2-Director tests: the director runtime injection entrypoint.

TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION / NOT_SCIENTIFIC_EVIDENCE.

The single-update entrypoint must NO LONGER fixed-return
E1_PRODUCTION_PIPELINE_UNAUTHORIZED: the director's signed runtime
bundle is the injection channel, and when the external gates clear
the entry reaches the REAL one-window pipeline call surface. This
round only --check-only and tests run; no real Smoke.

Covered negative matrix:
* no fixed E1_PRODUCTION_PIPELINE_UNAUTHORIZED in the source
* missing director bundle                   -> BLOCKED (REQUIRED)
* full run with a TEST_ONLY bundle          -> BLOCKED (TEST_ONLY)
* check-only with TEST_ONLY bundle          -> CHECK_ONLY_OK +
  fifteen_plus_one_batch_ready, never EXECUTED
* the pipeline call surface runs a TEST_ONLY chain end-to-end and
  produces a Smoke handoff report with every REAL flag false
"""
import json
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
import e1_production_runtime as RT  # noqa: E402
import run_e1_real_one_update as ENT  # noqa: E402

from dicode.teachers.e1_formal import accounting as ACCT
from dicode.teachers.e1_formal import board as B
from dicode.teachers.e1_formal import dicode_protocol as DP
from dicode.teachers.e1_formal import envcoder as EC
from dicode.teachers.e1_formal import evidence as E
from dicode.teachers.e1_formal import gen_manager as GM
from dicode.teachers.e1_formal import llm_client as LC
from dicode.teachers.e1_formal import manifest as M
from dicode.teachers.e1_formal import roundtrip_attestation as RA
from dicode.teachers.e1_formal import runtime_bundle as RB
from dicode.teachers.e1_formal import selector as SEL
from dicode.teachers.e1_formal import task_specs as TS
from dicode.teachers.e1_formal import update_attestation as UA
from dicode.teachers.e1_formal.canonical import canonical_sha256
from test_e1_testonly_closed_loop import (  # noqa: E402
    ARCHIVE_SNAPSHOT,
    _add_envcoder_entries,
    _build_board_store,
    _probe,
    _signal,
    _test_only_bundle,
    _REFERENCE_CHECKPOINT,
    _STUDENT_CHECKPOINT,
    _STUDENT_IDENTITY,
)
from test_task_specs import _family  # noqa: E402

SCRIPT_PATH = os.path.join(SCRIPTS_DIR, "run_e1_real_one_update.py")


def _committed_files():
    import yaml

    with open(
        os.path.join(REPO_ROOT, "conf", "teacher", "e1_formal.yaml"),
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(handle)
    with open(
        os.path.join(REPO_ROOT, "configs", "e1_formal_ued.yaml"),
        "r",
        encoding="utf-8",
    ) as handle:
        frozen = yaml.safe_load(handle)
    with open(
        os.path.join(
            REPO_ROOT,
            "configs",
            "e1_formal_ued_anchor_manifest.DRAFT.json",
        ),
        "r",
        encoding="utf-8",
    ) as handle:
        draft = json.load(handle)
    return config, frozen, draft


def _test_only_manifest_path(tmp_path):
    """A verifiable TEST_ONLY director-bundle manifest on disk."""
    bundle = _test_only_bundle()
    manifest = {
        "bundle_id": bundle.bundle_id,
        "mode": bundle.mode,
        "source_commit": bundle.source_commit,
        "signer_id": bundle.signer_id,
        "authorization_grant_hash": bundle.authorization_grant_hash,
        "object_identity_hashes": dict(bundle.object_identity_hashes),
        "student_selection": bundle.student_selection_mapping,
        "bundle_hash": bundle.bundle_hash,
    }
    path = tmp_path / "test_only_director_bundle.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return str(path)


def _read(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


class TestEntrypoint:
    def test_no_fixed_pipeline_unauthorized_gate_in_source(self):
        with open(SCRIPT_PATH, "r", encoding="utf-8") as handle:
            source = handle.read()
        assert "E1_PRODUCTION_PIPELINE_UNAUTHORIZED" not in source

    def test_missing_director_bundle_blocks(self, tmp_path):
        report_path = str(tmp_path / "blocked.json")
        rc = ENT.main(["--report-out", report_path])
        assert rc == 2
        report = _read(report_path)
        assert report["status"] == "BLOCKED"
        assert report["real_one_update_executed"] is False
        codes = [b["code"] for b in report["blockers"]]
        assert ENT.E1_DIRECTOR_RUNTIME_BUNDLE_REQUIRED in codes

    def test_full_run_with_test_only_bundle_refused(self, tmp_path):
        bundle_path = _test_only_manifest_path(tmp_path)
        report_path = str(tmp_path / "full_test_only.json")
        rc = ENT.main(
            [
                "--director-runtime-bundle",
                bundle_path,
                "--report-out",
                report_path,
            ]
        )
        assert rc == 2
        report = _read(report_path)
        assert report["status"] == "BLOCKED"
        assert report["real_one_update_executed"] is False
        codes = [b["code"] for b in report["blockers"]]
        assert RB.RUNTIME_BUNDLE_TEST_ONLY_REJECTED in codes

    def test_check_only_verifies_15_plus_1_and_never_executes(self, tmp_path):
        bundle_path = _test_only_manifest_path(tmp_path)
        report_path = str(tmp_path / "check_only.json")
        rc = ENT.main(
            [
                "--check-only",
                "--director-runtime-bundle",
                bundle_path,
                "--report-out",
                report_path,
            ]
        )
        assert rc == 0
        report = _read(report_path)
        assert report["status"] == ENT.E1_TEST_ONLY_CONTRACT_OK
        assert report["executed"] is False  # NEVER executes
        checks = report["checks"]
        assert checks["bundle_manifest_verified"] is True
        assert checks["capability_contracts_declared"] is True
        assert checks["driver_dataflow_constructible"] is True
        assert checks["fifteen_plus_one_batch_ready"] is True
        # all shared contracts stay honestly unbound on this host
        assert all(
            bound is False
            for bound in checks["shared_runtime_objects_bound"].values()
        )


class TestPipelineCallSurface:
    def _chain(self):
        config, frozen, draft = _committed_files()
        bundle = _test_only_bundle()
        probe_teacher = GM.E1FormalGenManager(
            config,
            frozen_manifest=frozen,
            anchor_manifest_mapping=draft,
            archive_snapshot=ARCHIVE_SNAPSHOT,
        )
        raw_items = probe_teacher.collect_evidence_raw_items()
        evidence = E.build_evidence_snapshot(raw_items, "handoff")
        families = [_family(f"fam_{i}") for i in range(6)]
        store = _build_board_store(evidence, families)
        specs = TS.compile_task_specs(
            B.run_review_board(
                LC.ReplayLLMClient(store, "test"),
                window_id="e1-w000001",
                session_idx=1,
                trigger_code="FIRST_WINDOW",
                evidence=evidence,
                ledger=ACCT.LLMCallLedger(),
            )
        ).specs
        _add_envcoder_entries(
            store,
            specs,
            tuple(config["teacher"]["envcoder"]["seed_examples"]),
        )
        teacher = GM.E1FormalGenManager(
            config,
            frozen_manifest=frozen,
            anchor_manifest_mapping=draft,
            replay_store=store,
            archive_snapshot=ARCHIVE_SNAPSHOT,
        )
        return teacher, bundle

    def _probe_issuer(self, bundle):
        def issuer(candidates, bundle):
            return tuple(
                _probe(candidate, bundle, index)
                for index, candidate in enumerate(candidates)
            )

        return issuer

    def _signal_issuer(self):
        def issuer(candidates, probe_pool):
            return tuple(
                _signal(candidate, probe, index)
                for index, (candidate, probe) in enumerate(
                    zip(candidates, probe_pool)
                )
            )

        return issuer

    def _update_chain(self, bundle):
        training_runtime = UA.authorize_original_training_runtime(
            mode=UA.TRAINING_RUNTIME_MODE_TEST_ONLY,
            run_id=UA.SYNTHETIC_TEST_ONLY_TRAINING_RUNTIME,
            student_identity_hash=_STUDENT_IDENTITY,
            network_identity_hash="31" * 32,
            loss_identity_hash="32" * 32,
            optimizer_identity_hash="33" * 32,
            rollout_schema_hash="34" * 32,
            transition_accounting_version="e1-transition-accounting-v1",
            reward_identity_hash="35" * 32,
            source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
        )
        one_update_runtime = DP.authorize_canonical_dicode_one_update_runtime(
            training_runtime=training_runtime,
            total_timesteps=2_005_401_600,
            session_idx=1,
            global_env_steps=4096,
            global_update_step=7,
            ctx="test",
        )
        record = UA.UpdateExecutionRecord(
            run_id=UA.SYNTHETIC_TEST_ONLY_TRAINING_RUNTIME,
            input_checkpoint_hash=_STUDENT_CHECKPOINT,
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
            record_hash="",
        )
        record = UA.UpdateExecutionRecord(
            record_hash=UA.compute_update_record_hash(
                run_id=record.run_id,
                input_checkpoint_hash=record.input_checkpoint_hash,
                output_checkpoint_hash=record.output_checkpoint_hash,
                params_hash_before=record.params_hash_before,
                params_hash_after=record.params_hash_after,
                optimizer_state_hash_before=(
                    record.optimizer_state_hash_before
                ),
                optimizer_state_hash_after=(
                    record.optimizer_state_hash_after
                ),
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
            ),
            **{name: getattr(record, name)
               for name in (
                   "run_id",
                   "input_checkpoint_hash",
                   "output_checkpoint_hash",
                   "params_hash_before",
                   "params_hash_after",
                   "optimizer_state_hash_before",
                   "optimizer_state_hash_after",
                   "rng_hash_before",
                   "rng_hash_after",
                   "global_env_steps_before",
                   "global_env_steps_after",
                   "update_step_before",
                   "update_step_after",
                   "optimizer_step_before",
                   "optimizer_step_after",
                   "rollout_batch_hash",
                   "transitions_consumed",
                   "update_count",
                   "loss_identity_hash",
                   "optimizer_identity_hash",
                   "gradient_finite",
               )}
        )
        checkpoint = DP.build_canonical_runstate_checkpoint(
            params_hash=record.params_hash_after,
            optimizer_state_hash=record.optimizer_state_hash_after,
            optimizer_step=record.optimizer_step_after,
            global_update_step=record.update_step_after,
            global_env_steps=record.global_env_steps_after,
            rng_hash=record.rng_hash_after,
            session_index=1,
            gen_manager_archive_hash="51" * 32,
            e1_ledger_hash="52" * 32,
            pending_worker_policy_hash="53" * 32,
            config_hash="54" * 32,
            runtime_bundle_hash=bundle.bundle_hash,
            ctx="test",
        )
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
            student_identity_hash=_STUDENT_IDENTITY,
            anchor_manifest_hash="aa" * 32,
            formal_asset_registry_hash="ab" * 32,
            window_hash="e" * 64,
            selection_hash="h" * 64,
            verified_batch_hash="bb" * 32,
            source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
        )
        restored = "61" * 32
        roundtrip_att = RA.attest_full_state_round_trip(
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
        return one_update_runtime, record, checkpoint, (
            identity,
            roundtrip_att,
        )

    def test_pipeline_runs_the_full_chain_test_only(self):
        teacher, bundle = self._chain()
        one_update_runtime, record, checkpoint, roundtrip = (
            self._update_chain(bundle)
        )
        report = ENT.run_director_one_window_pipeline(
            teacher=teacher,
            bundle=bundle,
            run_id="TEST_ONLY_SYNTHETIC_RUN",
            branch="test-branch",
            git_sha="ab" * 32,
            probe_issuer=self._probe_issuer(bundle),
            signal_issuer=self._signal_issuer(),
            one_update_runtime=one_update_runtime,
            student_checkpoint_identity=_STUDENT_CHECKPOINT,
            reference_checkpoint_identity=_REFERENCE_CHECKPOINT,
            anchor_manifest_hash="aa" * 32,
            formal_asset_registry_hash="ab" * 32,
            update_record=record,
            checkpoint=checkpoint,
            roundtrip_evidence=roundtrip,
            k=12,
            seed=7,
            critic_policy=SEL.CRITIC_HARD_VETO,
            family_cap=6,
            smoke_signer_id=(
                "SYNTHETIC_TEST_ONLY_SMOKE_SIGNER"
            ),
            update_signer_id=(
                UA.SYNTHETIC_TEST_ONLY_TRAINING_SIGNER
            ),
            roundtrip_signer_id=(
                RA.SYNTHETIC_TEST_ONLY_ROUNDTRIP_SIGNER
            ),
            allow_test_only=True,
        )
        assert report["status"] == "TEST_ONLY_SMOKE_HANDOFF"
        assert report["executed"] is True
        assert report["selected_count"] == 12
        # 15 curriculum + 1 target
        assert len(report["curriculum_task_ids"]) == 15
        assert report["target_task_id"] == "original_craftax"
        assert report["target_probability"] == 0.20
        assert len(report["smoke_attestation_hash"]) == 64
        # NO REAL flag flips; this is code-path evidence only
        assert report["real_flags"]["REAL_LLM_EXECUTED"] is False
        assert (
            report["real_flags"]["REAL_OPTIMIZER_UPDATE_EXECUTED"]
            is False
        )
        assert report["real_flags"]["FORMAL_EXPERIMENT_AUTHORIZED"] is False
