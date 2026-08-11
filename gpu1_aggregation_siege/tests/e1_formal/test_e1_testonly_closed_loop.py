"""CC2 follow-up P0-18: the TEST_ONLY complete dataflow closed loop.

TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION / NOT_SCIENTIFIC_EVIDENCE:
the WHOLE loop runs on conspicuously-marked synthetic objects and
replay stores. It proves the production code PATH is connected (the
one-window driver, the signed bundles, the registry-signed probe
intake, the signed signals, the attested selection, the GenManager
certification, the exactly-one update attestation, the full-state
round-trip attestation) WITHOUT any real LLM / EnvCoder / probe /
training / Craftax. NO REAL_* flag flips; the final pipeline result is
TEST_ONLY.

Closed loop (as specified):
  sequential six-role Fixture Board (replay, COMPLETE)
  -> 6 Templates / 12 Variants (compile_task_specs on the real window)
  -> 12 ExecutableCandidates (TEST_ONLY bundle-bound surfaces)
  -> 12 signed Synthetic CandidateProbeResults (consumed fail-closed)
  -> 12 SignedCriterionSignals (derived from the probes)
  -> criterion-wise Soft Copeland selects 12 (STATUS_OK)
  -> same GenManager certifies the 12+4 (mechanical P0-9 checks pass;
     the committed teacher's REAL gates then honestly refuse promotion
     with GEN_MANAGER_PROMOTION_BLOCKED)
  -> Synthetic OriginalTrainingRuntime issues exactly-one update
  -> Synthetic FullStateCheckpoint verifies full-state restore
  -> TEST_ONLY pipeline result (all REAL_* flags still false).
"""
import json
import os

import pytest

from dicode.teachers.e1_formal import accounting as ACCT
from dicode.teachers.e1_formal import board as B
from dicode.teachers.e1_formal import envcoder as EC
from dicode.teachers.e1_formal import evidence as E
from dicode.teachers.e1_formal import gen_manager as GM
from dicode.teachers.e1_formal import llm_client as LC
from dicode.teachers.e1_formal import manifest as M
from dicode.teachers.e1_formal import one_window_driver as DRV
from dicode.teachers.e1_formal import probe_result_binding as PRB
from dicode.teachers.e1_formal import roundtrip_attestation as RA
from dicode.teachers.e1_formal import runtime_bundle as RB
from dicode.teachers.e1_formal import selector as SEL
from dicode.teachers.e1_formal import signed_signals as SS
from dicode.teachers.e1_formal import task_specs as TS
from dicode.teachers.e1_formal import update_attestation as UA
from dicode.teachers.e1_formal.canonical import canonical_sha256
from test_task_specs import _family  # noqa: E402

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

_STUDENT_IDENTITY = "11" * 32
_STUDENT_CHECKPOINT = "12" * 32
_REFERENCE_IDENTITY = "21" * 32
_REFERENCE_CHECKPOINT = "22" * 32
_SEED_BANK = "77" * 32
_RESET_PROTOCOL = "88" * 32
_ABI = "a1" * 32
_REWARD = "b2" * 32
_RUNNER_ID = "test-only-probe-runner-registry"
_STUDENT_STEPS = 4096

ARCHIVE_SNAPSHOT = {
    "tasks": [
        {
            "task_id": "task_a",
            "provenance": "TRAINING",
            "performance_history": [
                {"session_idx": 3, "success_rate": 0.4}
            ],
        }
    ]
}


# ---------------------------------------------------------------------------
# replay stores (TEST_ONLY fixture content — NOT_REAL_EXECUTION)
# ---------------------------------------------------------------------------
def _role_payloads(families):
    return {
        "student_modeler": {
            "model_summary": "TEST_ONLY model summary",
            "capability_profile": [
                {"skill_id": "get_wood", "success_rate": 0.75},
                {"skill_id": "build_bridge", "success_rate": 0.1},
            ],
        },
        "behavior_auditor": {
            "findings": [
                {
                    "finding_id": "bf1",
                    "description": "TEST_ONLY repeated gap failure",
                }
            ]
        },
        "causal_failure_analyst": {
            "weaknesses": [
                {
                    "weakness_id": "w1",
                    "name": "gap crossing",
                    "evidence_refs": ["bf1"],
                    "priority": 1,
                }
            ],
            "hypotheses": [
                {
                    "hypothesis_id": "h1",
                    "weakness_id": "w1",
                    "statement": "TEST_ONLY undertrained placement",
                }
            ],
            "reuse_previous_direction": False,
            "overall_confidence": 0.7,
        },
        "intervention_tutor": {"families": families, "explorations": []},
        "explorer": {
            "exploration_rationale": "TEST_ONLY terrain variety",
            "candidate_axes": ["terrain_roughness"],
        },
        "critic": {"vetoes": [], "notes": "TEST_ONLY nothing to veto"},
    }


def _build_board_store(evidence, families):
    payloads = _role_payloads(families)
    store = {}
    context = B.make_board_context(
        window_id="e1-w000001",
        session_idx=1,
        trigger_code="FIRST_WINDOW",
        evidence_hash=evidence.evidence_hash,
    )
    upstream = []
    for role in M.BOARD_ROLE_ORDER:
        envelope = B.build_prompt_envelope_hash(
            role, evidence, context=context, upstream=upstream
        )
        key = LC.make_replay_key(
            role=role,
            evidence_hash=evidence.evidence_hash,
            prompt_envelope_hash=envelope,
            prompt_version=M.BOARD_PROMPT_VERSION,
            schema_version=M.ROLE_OUTPUT_SCHEMA_VERSION,
        )
        content = json.dumps(payloads[role])
        store[key] = content
        parsed = B._parse_role_content(role, content, f"store-build {role}")
        upstream.append(
            B.UpstreamOutput(
                role=role,
                output=parsed,
                output_hash=B.role_output_hash(role, parsed),
            )
        )
    return store


def _envcoder_entry(spec, seeds):
    envelope = EC.build_envcoder_envelope_hash(spec, seed_examples=seeds)
    key = LC.make_replay_key(
        role=M.ENVCODER_ROLE,
        evidence_hash=spec.template_hash,
        prompt_envelope_hash=envelope,
        prompt_version=M.ENVCODER_PROMPT_VERSION,
        schema_version=M.ENVCODER_OUTPUT_SCHEMA_VERSION,
    )
    return {
        key: json.dumps(
            {
                "artifact_id": spec.template_artifact_id,
                "env_code": "def make_env():\n    return 'env'",
            }
        )
    }


def _add_envcoder_entries(store, specs, seeds):
    seen = set()
    for spec in specs:
        if spec.template_hash in seen:
            continue
        seen.add(spec.template_hash)
        store.update(_envcoder_entry(spec, seeds))


# ---------------------------------------------------------------------------
# the TEST_ONLY bundle (surfaces bound to the capability objects)
# ---------------------------------------------------------------------------
class _Capability:
    """TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION capability object."""

    def __init__(self, kind, **surfaces):
        self.kind = kind
        self.identity_id = f"test-only-{kind}"
        for name, value in surfaces.items():
            setattr(self, name, value)


def _test_only_bundle():
    capabilities = {
        "student_adapter": _Capability(
            "student_adapter", observation_action_abi_hash=_ABI
        ),
        "formal_asset_registry": _Capability(
            "formal_asset_registry", reward_contract_hash=_REWARD
        ),
        "probe_runner": _Capability(
            "probe_runner",
            seed_bank_hash=_SEED_BANK,
            reset_protocol_hash=_RESET_PROTOCOL,
        ),
    }
    for contract in RB.RUNTIME_CAPABILITY_CONTRACTS:
        if contract not in capabilities:
            capabilities[contract] = _Capability(contract)
    return RB.build_test_only_runtime_bundle(
        source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
        capabilities=capabilities,
    )


# ---------------------------------------------------------------------------
# signed objects
# ---------------------------------------------------------------------------
def _probe(candidate, bundle, index):
    return PRB.issue_candidate_probe_result(
        candidate=candidate,
        student_identity_hash=bundle.object_identity_hash(
            "student_identity"
        ),
        student_checkpoint_hash=_STUDENT_CHECKPOINT,
        reference_identity_hash=bundle.object_identity_hash(
            "reference_identity"
        ),
        reference_checkpoint_hash=_REFERENCE_CHECKPOINT,
        runner_registry_id=_RUNNER_ID,
        runner_registry_hash=bundle.object_identity_hash("probe_runner"),
        seed_bank_hash=_SEED_BANK,
        reset_protocol_id="test-only-reset-protocol-v1",
        reset_protocol_hash=_RESET_PROTOCOL,
        episodes_requested=3,
        episodes_completed=3,
        episodes_failed=0,
        simulator_transitions=384,
        aggregate_metrics=_metrics(index),
        uncertainty_ci={"ci95": [0.4, 0.6]},
        terminal_event_aggregates={"terminal_events": 3},
        signer_id=PRB.SYNTHETIC_TEST_ONLY_PROBE_SIGNER,
        test_only=True,
    )


def _metrics(index):
    base = min(0.9, 0.1 * (index % 6 + 1))
    return {
        "front_regret": base,
        "global_regret": min(0.9, base + 0.05),
        "behavioral_gap": min(0.9, base + 0.02),
        "learnability": min(0.9, base + 0.03),
        "learning_progress": min(0.9, base + 0.01),
    }


def _signal(candidate, probe, index):
    return SS.derive_criterion_signals_from_probe_result(
        probe_result=probe,
        candidate=candidate,
        aggregate_metrics=_metrics(index),
        retention_evidence={"global_retention": 0.75},
        diversity_evidence={"axis_count": 2, "pool_axis_max": 4},
        cost_evidence={"episodes": 3},
        signer_id=SS.SYNTHETIC_TEST_ONLY_SIGNAL_SIGNER,
        test_only=True,
    )


# ---------------------------------------------------------------------------
# the closed loop
# ---------------------------------------------------------------------------
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


class TestTestOnlyClosedLoop:
    def test_complete_test_only_dataflow(self):
        config, frozen, draft = _committed_files()
        bundle = _test_only_bundle()

        # 0) evidence from the archive snapshot (probe teacher only)
        probe_teacher = GM.E1FormalGenManager(
            config,
            frozen_manifest=frozen,
            anchor_manifest_mapping=draft,
            archive_snapshot=ARCHIVE_SNAPSHOT,
        )
        raw_items = probe_teacher.collect_evidence_raw_items()
        assert len(raw_items) >= 1
        evidence = E.build_evidence_snapshot(raw_items, "closed-loop")

        # 1) six families -> six templates -> twelve variants
        families = [_family(f"fam_{i}") for i in range(6)]
        board_store = _build_board_store(evidence, families)
        specs = TS.compile_task_specs(
            B.run_review_board(
                LC.ReplayLLMClient(board_store, "test"),
                window_id="e1-w000001",
                session_idx=1,
                trigger_code="FIRST_WINDOW",
                evidence=evidence,
                ledger=ACCT.LLMCallLedger(),
            )
        ).specs
        assert len(specs) == 12
        assert len({s.template_hash for s in specs}) == 6

        seeds = tuple(config["teacher"]["envcoder"]["seed_examples"])
        _add_envcoder_entries(board_store, specs, seeds)

        # 2) the REAL teacher carries the full replay store + archive
        teacher = GM.E1FormalGenManager(
            config,
            frozen_manifest=frozen,
            anchor_manifest_mapping=draft,
            replay_store=board_store,
            archive_snapshot=ARCHIVE_SNAPSHOT,
        )

        # 3) window stage: the six-role fixture board completes
        window_result = DRV.execute_real_review_window(teacher, bundle)
        assert window_result.window.status == B.WINDOW_STATUS_COMPLETE
        assert window_result.window.window_id == "e1-w000001"

        # 4) envcoder + compile stage: 6 templates -> materials
        materials = DRV.execute_real_envcoder_and_compile(
            teacher, window_result, bundle
        )
        assert len(materials.compile_result.specs) == 12

        # 5) executable candidate binding: 12 candidates, markers set
        candidates = DRV.execute_real_candidate_binding(
            teacher,
            window_result,
            materials,
            bundle,
            allow_test_only=True,
        )
        assert len(candidates) == 12
        for candidate in candidates:
            assert candidate.execution_marker == (
                EC_module_marker()
            )
            assert candidate.variant_params_executed is False

        # 6) twelve signed synthetic probe results (consumed)
        probes = tuple(
            _probe(candidate, bundle, index)
            for index, candidate in enumerate(candidates)
        )
        probe_pool = DRV.execute_real_candidate_probes(
            teacher,
            candidates,
            bundle,
            probe_results=probes,
            student_checkpoint_identity=_STUDENT_CHECKPOINT,
            reference_checkpoint_identity=_REFERENCE_CHECKPOINT,
            window_result=window_result,
            allow_test_only=True,
        )
        assert len(probe_pool) == 12

        # 7) twelve signed criterion signals
        signals = tuple(
            _signal(candidate, probe, index)
            for index, (candidate, probe) in enumerate(
                zip(candidates, probe_pool)
            )
        )
        assert len(signals) == 12

        # 8) criterion-wise Soft Copeland selects 12 (STATUS_OK)
        outcome, attestation = DRV.execute_real_criterion_selection(
            teacher,
            window_result,
            candidates,
            probe_pool,
            signals,
            bundle,
            k=12,
            seed=7,
            critic_policy=SEL.CRITIC_HARD_VETO,
            family_cap=6,
            allow_test_only=True,
        )
        assert outcome.status == SEL.STATUS_OK
        assert len(outcome.selected_ids) == 12

        # the 12+4 verified batch hash: the attested selected set + the
        # four frozen shared anchors (TEST_ONLY layout)
        verified_batch_hash = canonical_sha256(
            {
                "selected_set_hash": attestation.selected_set_hash,
                "anchors": sorted(
                    ("anchor_task_1", "anchor_task_2", "anchor_task_3",
                     "anchor_original_craftax")
                ),
            }
        )

        # 9) the SAME GenManager certifies the 12+4: every P0-9
        #    mechanical check passes; the committed teacher's REAL
        #    gates then honestly refuse promotion — the loop reaches
        #    the promotion gate, never a certification failure
        with pytest.raises(GM.GenManagerError) as excinfo:
            teacher.certify_and_build_training_batch(
                selection_attestation=attestation,
                candidate_pool=candidates,
                probe_pool=probe_pool,
                signals_pool=signals,
                window_hash=window_result.window.window_hash,
                student_checkpoint_hash=_STUDENT_CHECKPOINT,
                reference_checkpoint_hash=_REFERENCE_CHECKPOINT,
            )
        assert excinfo.value.code == GM.GEN_MANAGER_PROMOTION_BLOCKED

        # 10) Synthetic OriginalTrainingRuntime issues exactly-one update
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
            global_env_steps_before=_STUDENT_STEPS,
            global_env_steps_after=2 * _STUDENT_STEPS,
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
            record_hash="",  # filled below
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
        update_attestation = UA.attest_exactly_one_update(
            training_runtime,
            record,
            verified_batch_hash=verified_batch_hash,
            signer_id=UA.SYNTHETIC_TEST_ONLY_TRAINING_SIGNER,
            test_only=True,
            ctx="closed-loop",
        )
        UA.verify_optimizer_update_attestation(
            update_attestation, runtime=training_runtime
        )
        assert update_attestation.optimizer_step_after == 43

        # 11) Synthetic FullStateCheckpoint verifies full-state restore
        identity = RA.build_full_state_checkpoint_identity(
            params_hash=record.params_hash_after,
            optimizer_state_hash=record.optimizer_state_hash_after,
            global_env_steps=2 * _STUDENT_STEPS,
            update_step=8,
            optimizer_step=43,
            training_rng_hash="45" * 32,
            env_rng_hash="46" * 32,
            env_state_hash="51" * 32,
            wrapper_state_hash="52" * 32,
            prev_action_reward_hash="53" * 32,
            policy_memory_history_hash="54" * 32,
            student_identity_hash=_STUDENT_IDENTITY,
            anchor_manifest_hash="55" * 32,
            formal_asset_registry_hash="56" * 32,
            window_hash=window_result.window.window_hash,
            selection_hash=attestation.selection_hash,
            verified_batch_hash=verified_batch_hash,
            source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
        )
        restored = "61" * 32
        roundtrip = RA.attest_full_state_round_trip(
            identity,
            restored_state_hash=restored,
            leaf_comparison_hash=restored,
            next_policy_step_hash="62" * 32,
            fresh_process_restored=True,
            replay_identical=True,
            signer_id=RA.SYNTHETIC_TEST_ONLY_ROUNDTRIP_SIGNER,
            test_only=True,
            ctx="closed-loop",
        )
        RA.verify_full_state_round_trip(roundtrip, identity)

        # 12) TEST_ONLY pipeline result: the path is connected, but NO
        #     REAL_* flag flips and no real execution ever happened
        report = teacher.status_report()
        assert report["flags"]["real_envcoder_used"] is False
        assert report["flags"]["real_student_reference_eval"] is False
        assert report["flags"]["real_training_update_executed"] is False
        assert teacher.cycles_run == 1


def EC_module_marker():
    from dicode.teachers.e1_formal import executable_candidates as EX

    return EX.VARIANT_PARAMETER_NOT_EXECUTED
