"""C10 tests: E1FormalGenManager duck surface + honest degradation.

Every LLM answer used here comes from a replay store or an explicitly
labeled fake client — no real LLM, no real evaluation, no training.
"""
import ast
import json
import os

import pytest

from dicode.teachers.e1_formal import anchor_manifest as AM
from dicode.teachers.e1_formal import board as B
from dicode.teachers.e1_formal import envcoder as EC
from dicode.teachers.e1_formal import gen_manager as GM
from dicode.teachers.e1_formal import layout as L
from dicode.teachers.e1_formal import llm_client as LC
from dicode.teachers.e1_formal import manifest as M
from dicode.teachers.e1_formal import selector as S
from dicode.teachers.e1_formal.accounting import LLMCallLedger
from dicode.teachers.e1_formal.evidence import build_evidence_snapshot
from dicode.teachers.e1_formal.task_specs import compile_task_specs

from test_board import _build_store  # fixture reuse
from test_task_specs import _family  # REGISTRY-valid intervention families

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

FLAGS = {
    "real_envcoder_used": False,
    "real_student_reference_eval": False,
    "real_training_update_executed": False,
}

SEEDS = [
    {"task_id": "task_1", "description": "collect coal near spawn"},
    {"task_id": "task_2", "description": "cross a narrow gap"},
    {"task_id": "task_3", "description": "defeat the orc soldier"},
]

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


def _frozen_manifest():
    return {
        "schema_version": "e1_formal.frozen_manifest.v1",
        "flags": dict(FLAGS),
        "copeland": {
            "protocol_version": S.COPELAND_PROTOCOL_VERSION,
            "source_sha256": S.COPELAND_SOURCE_SHA256,
            "constants_sha256": S.COPELAND_CONSTANTS_SHA256,
            "base_sha256": S.COPELAND_BASE_SHA256,
        },
        "replay": {
            "provider": "replay",
            "model_id": GM.REPLAY_MODEL_ID,
            "record": "disabled",
        },
        # round-3 P0-3: the gate-signal regime version is frozen; the
        # threshold VALUES stay supervisor-owned (null in the teacher
        # config => INVOCATION_THRESHOLD_MISSING, honestly blocked)
        "invocation": {"threshold_version": "e1-gate-thresholds-v1"},
        "anchors": {"task_ids": list(L.ANCHOR_TASK_IDS)},
        "strong_student": {
            "candidate_id": "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304"
        },
    }


def _draft_manifest_mapping():
    """The COMMITTED draft artifact (not an in-memory copy)."""
    path = os.path.join(
        REPO_ROOT, "configs", "e1_formal_ued_anchor_manifest.DRAFT.json"
    )
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _teacher_config():
    return {
        "teacher": {
            "teacher_type": "e1_formal",
            "flags": dict(FLAGS),
            "reference_contract": {
                "frozen": False,
                "candidate_id": None,
                "checkpoint_ref": None,
                "file_sha256": None,
                "params_sha256": None,
                "architecture_family": None,
                "architecture_version": None,
                "architecture_config_hash": None,
                "memory_semantics": None,
                "memory_semantics_hash": None,
                "global_step": None,
                "source_commit": None,
                "seed": None,
                "episode_reset_protocol_id": None,
                "episode_reset_protocol_hash": None,
                "frozen_manifest_hash": None,
            },
            "learnability": {
                "tau_saturated": None,
                "tau_reachable": None,
                "tau_unreachable": None,
                "delta_min": None,
                "min_episodes": None,
                "ci_level": None,
            },
            # round-3 P0-3: mirrors conf/teacher/e1_formal.yaml — the
            # version is pinned, all six threshold values stay null
            # (unfrozen) => gate signals computed False with
            # INVOCATION_THRESHOLD_MISSING reasons
            "invocation": {
                "threshold_version": "e1-gate-thresholds-v1",
                "thresholds": {
                    "capability_shift_delta": None,
                    "stagnation_max_delta": None,
                    "stagnation_min_sessions": None,
                    "forgetting_regression_drop": None,
                    "intervention_exhaustion_max_reuses": None,
                    "exploration_slot_period": None,
                },
            },
            "selection": {
                "critic_policy": "hard_veto",
                "k": 12,
                "seed": 20260803,
            },
            "anchors": {
                "task_ids": list(L.ANCHOR_TASK_IDS),
                "manifest_path":
                    "configs/e1_formal_ued_anchor_manifest.DRAFT.json",
            },
            "envcoder": {"seed_examples": [dict(s) for s in SEEDS]},
            "replay": {
                "provider": "replay",
                "model_id": GM.REPLAY_MODEL_ID,
                "record": "disabled",
            },
        }
    }


def _manager(**kwargs):
    return GM.E1FormalGenManager(
        _teacher_config(),
        frozen_manifest=_frozen_manifest(),
        anchor_manifest_mapping=_draft_manifest_mapping(),
        **kwargs,
    )


class _GarbageLLM:
    """Answers syntactically valid but contract-invalid JSON-less text."""

    def query(self, system_prompt, user_prompts, *, cache_key, role):
        return [{"content": "not a json object"}]

    def record(self, *args, **kwargs):  # pragma: no cover - never used
        raise RuntimeError("record disabled")


def _evidence_from_archive():
    from dicode.teachers.e1_formal.archive_view import consume_archive_snapshot

    view = consume_archive_snapshot(ARCHIVE_SNAPSHOT, "test")
    return build_evidence_snapshot(view.evidence_items(), "test")


def _board_store(evidence, families=None):
    """Replay store for a COMPLETE window whose families target
    REGISTRY-valid achievements (collect_coal).

    Round-3 P0-1/P0-3: the manager's FIRST evolve opens window
    ``e1-w000001`` at session_idx=1 via FIRST_WINDOW — the sequential
    board envelopes bind exactly that identity, so the store must too.
    """
    if families is None:
        families = [_family("fam_a"), _family("fam_b")]
    return _build_store(
        evidence,
        overrides={
            "intervention_tutor": {
                "families": families,
                "explorations": [],
            }
        },
        window_id="e1-w000001",
        session_idx=1,
        trigger_code="FIRST_WINDOW",
    )


def _envcoder_entry(spec, seeds, env_code="def make_env():\n    return 'env'"):
    # round-3 P0-2: the EnvCoder is TEMPLATE-keyed (one call per unique
    # template; the replay key's evidence hash is the template hash)
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
            {"artifact_id": spec.template_artifact_id, "env_code": env_code}
        )
    }


def _add_template_entries(store, specs, seeds, env_code_by_template=None):
    """One EnvCoder replay entry per UNIQUE template in ``specs``."""
    seen = set()
    for spec in specs:
        if spec.template_hash in seen:
            continue
        seen.add(spec.template_hash)
        code = (
            env_code_by_template(spec)
            if env_code_by_template is not None
            else "def make_env():\n    return 'env'"
        )
        store.update(_envcoder_entry(spec, seeds, env_code=code))


def _specs_via_board(evidence, store):
    """Run the SAME deterministic board the manager will run."""
    window = B.run_review_board(
        LC.ReplayLLMClient(store, "test"),
        window_id="e1-w000001",
        session_idx=1,
        trigger_code="FIRST_WINDOW",
        evidence=evidence,
        ledger=LLMCallLedger(),
    )
    assert window.status == B.WINDOW_STATUS_COMPLETE
    return compile_task_specs(window).specs


class TestInit:
    def test_happy_init_records_all_honest_block_codes(self):
        manager = _manager()
        blocked = manager.current_blocked_codes()
        assert "REFERENCE_CONTRACT_UNFROZEN" in blocked
        assert "LEARNABILITY_THRESHOLD_MISSING" in blocked
        assert AM.BLOCKED_SHARED_ANCHOR_MANIFEST in blocked
        assert "SELECTION_BLOCKED_NO_REAL_EVIDENCE" in blocked
        # round-3 P0-3: unfrozen invocation thresholds degrade window
        # INVOCATION (signals computed False + explicit reason) but are
        # NOT a training-gate blocker — the C13 gate's evidence chain
        # is the verified dual-probe/anchor-manifest snapshot
        assert "INVOCATION_THRESHOLD_MISSING" not in blocked
        assert manager.reference_contract is None
        assert manager.thresholds is None
        assert manager.anchor_manifest.status == AM.STATUS_DRAFT_UNFROZEN
        assert manager.invocation_thresholds_present is False
        report = manager.status_report()
        assert report["invocation_thresholds_present"] is False
        assert report["invocation_degradation"] == "INVOCATION_THRESHOLD_MISSING"

    def test_wrong_teacher_type_fails_closed(self):
        config = _teacher_config()
        config["teacher"]["teacher_type"] = "static_llm"
        with pytest.raises(GM.GenManagerError) as excinfo:
            GM.E1FormalGenManager(
                config,
                frozen_manifest=_frozen_manifest(),
                anchor_manifest_mapping=_draft_manifest_mapping(),
            )
        assert excinfo.value.code == GM.GEN_MANAGER_BAD_TEACHER_TYPE

    def test_flag_manifest_mismatch_fails_closed(self):
        manifest = _frozen_manifest()
        manifest["flags"]["real_envcoder_used"] = True
        with pytest.raises(Exception) as excinfo:
            GM.E1FormalGenManager(
                _teacher_config(),
                frozen_manifest=manifest,
                anchor_manifest_mapping=_draft_manifest_mapping(),
            )
        assert excinfo.value.code == "FLAG_MANIFEST_MISMATCH"

    def test_copeland_pin_mismatch_fails_closed(self):
        manifest = _frozen_manifest()
        manifest["copeland"]["source_sha256"] = "0" * 64
        with pytest.raises(GM.GenManagerError) as excinfo:
            GM.E1FormalGenManager(
                _teacher_config(),
                frozen_manifest=manifest,
                anchor_manifest_mapping=_draft_manifest_mapping(),
            )
        assert excinfo.value.code == GM.GEN_MANAGER_MANIFEST_MISMATCH

    def test_replay_model_mismatch_fails_closed(self):
        manifest = _frozen_manifest()
        manifest["replay"]["model_id"] = "some-paid-model"
        with pytest.raises(GM.GenManagerError) as excinfo:
            GM.E1FormalGenManager(
                _teacher_config(),
                frozen_manifest=manifest,
                anchor_manifest_mapping=_draft_manifest_mapping(),
            )
        assert excinfo.value.code == GM.GEN_MANAGER_MANIFEST_MISMATCH

    def test_strong_student_mismatch_fails_closed(self):
        manifest = _frozen_manifest()
        manifest["strong_student"]["candidate_id"] = "SOMETHING_ELSE"
        with pytest.raises(GM.GenManagerError) as excinfo:
            GM.E1FormalGenManager(
                _teacher_config(),
                frozen_manifest=manifest,
                anchor_manifest_mapping=_draft_manifest_mapping(),
            )
        assert excinfo.value.code == GM.GEN_MANAGER_MANIFEST_MISMATCH

    @pytest.mark.parametrize("k", [0, 11, 13, True])
    def test_k_must_equal_the_12_dynamic_slots(self, k):
        config = _teacher_config()
        config["teacher"]["selection"]["k"] = k
        with pytest.raises(GM.GenManagerError) as excinfo:
            GM.E1FormalGenManager(
                config,
                frozen_manifest=_frozen_manifest(),
                anchor_manifest_mapping=_draft_manifest_mapping(),
            )
        assert excinfo.value.code == GM.GEN_MANAGER_OUT_OF_RANGE

    def test_bad_critic_policy_fails_closed(self):
        config = _teacher_config()
        config["teacher"]["selection"]["critic_policy"] = "score_highest"
        with pytest.raises(GM.GenManagerError):
            GM.E1FormalGenManager(
                config,
                frozen_manifest=_frozen_manifest(),
                anchor_manifest_mapping=_draft_manifest_mapping(),
            )

    def test_wrong_anchor_ids_fail_closed(self):
        config = _teacher_config()
        config["teacher"]["anchors"]["task_ids"] = ["task_1", "task_2"]
        with pytest.raises(GM.GenManagerError) as excinfo:
            GM.E1FormalGenManager(
                config,
                frozen_manifest=_frozen_manifest(),
                anchor_manifest_mapping=_draft_manifest_mapping(),
            )
        assert excinfo.value.code == GM.GEN_MANAGER_OUT_OF_RANGE

    def test_record_mode_must_be_disabled(self):
        config = _teacher_config()
        config["teacher"]["replay"]["record"] = "enabled"
        with pytest.raises(GM.GenManagerError) as excinfo:
            GM.E1FormalGenManager(
                config,
                frozen_manifest=_frozen_manifest(),
                anchor_manifest_mapping=_draft_manifest_mapping(),
            )
        assert excinfo.value.code == GM.GEN_MANAGER_OUT_OF_RANGE

    def test_seed_example_unknown_field_fails_closed(self):
        config = _teacher_config()
        config["teacher"]["envcoder"]["seed_examples"][0]["code"] = "x=1"
        with pytest.raises(GM.GenManagerError) as excinfo:
            GM.E1FormalGenManager(
                config,
                frozen_manifest=_frozen_manifest(),
                anchor_manifest_mapping=_draft_manifest_mapping(),
            )
        assert excinfo.value.code == GM.GEN_MANAGER_UNKNOWN_FIELD

    def test_missing_block_fails_closed(self):
        config = _teacher_config()
        del config["teacher"]["learnability"]
        with pytest.raises(GM.GenManagerError) as excinfo:
            GM.E1FormalGenManager(
                config,
                frozen_manifest=_frozen_manifest(),
                anchor_manifest_mapping=_draft_manifest_mapping(),
            )
        assert excinfo.value.code == GM.GEN_MANAGER_MISSING_FIELD

    def test_frozen_reference_contract_is_consumed(self):
        from test_reference_contract import _block

        config = _teacher_config()
        config["teacher"]["reference_contract"] = _block()
        manager = GM.E1FormalGenManager(
            config,
            frozen_manifest=_frozen_manifest(),
            anchor_manifest_mapping=_draft_manifest_mapping(),
        )
        assert manager.reference_contract is not None
        assert "REFERENCE_CONTRACT_UNFROZEN" not in manager.current_blocked_codes()


class TestEvolveDuck:
    def test_no_evidence_zero_llm_calls_and_exactly_12_reuse_stubs(self):
        manager = _manager()
        workers = manager.evolve_tasks(
            {"poisoned_metrics": {"mastered": 99}}, {"profile": "ignored"}
        )
        assert len(workers) == L.NUM_DYNAMIC_SLOTS == 12
        assert all(w["compiled"] is False for w in workers)
        assert all(w["e1_status"]["reuse"] is True for w in workers)
        counts = manager.ledger.counts()
        assert counts["G1"] == 0
        assert counts["board_calls"] == 0
        assert counts["K1"] == 0
        assert counts["T1"] == 0
        assert counts["F1"] == 0
        assert counts["N1"] == 0
        blocked = workers[0]["e1_status"]["blocked_codes"]
        assert GM.GEN_MANAGER_NO_ADMISSIBLE_EVIDENCE in blocked
        assert "SELECTION_BLOCKED_NO_REAL_EVIDENCE" in blocked

    def test_evolve_arguments_are_entirely_ignored(self):
        # identical output with or without poisoned evolve-side metrics
        poisoned = _manager()
        clean = _manager()
        w_poisoned = poisoned.evolve_tasks(
            {"reward_hack": True, "lp": 0.25}, {"tier": "gold"}
        )
        w_clean = clean.evolve_tasks()
        assert w_poisoned == w_clean
        blob = json.dumps(w_poisoned, sort_keys=True)
        assert "reward_hack" not in blob
        assert "gold" not in blob

    def test_void_window_yields_reuse_batch_with_six_board_calls(self):
        manager = _manager(
            llm_client=_GarbageLLM(), archive_snapshot=ARCHIVE_SNAPSHOT
        )
        workers = manager.evolve_tasks()
        assert len(workers) == 12
        assert all(w["e1_status"]["reuse"] for w in workers)
        counts = manager.ledger.reconcile()
        assert counts["G1"] == 1
        assert counts["board_calls"] == 6  # all six roles ran
        assert counts["K1"] == 0
        assert counts["N1"] == 6
        note = workers[0]["e1_status"]["note"]
        assert "INCOMPLETE_REVIEW_WINDOW" in note

    def test_full_window_produces_compiled_workers_and_exact_12(self):
        # round-3 P0-2: 6 families x 2 variants = exactly the 12
        # dynamic slots, ALL real compiled artifacts — zero stubs
        families = [_family(f"fam_{i}") for i in range(6)]
        evidence = _evidence_from_archive()
        store = _board_store(evidence, families=families)
        specs = _specs_via_board(evidence, dict(store))
        assert len(specs) == 12
        _add_template_entries(store, specs, SEEDS)
        manager = _manager(
            replay_store=store, archive_snapshot=ARCHIVE_SNAPSHOT
        )
        workers = manager.evolve_tasks({"ignored": 1}, {"ignored": 2})
        assert len(workers) == 12
        assert all(w["e1_status"]["reuse"] is False for w in workers)
        for worker in workers:
            assert worker["compiled"] is True
            assert worker["code"].startswith("def make_env")
            assert worker["e1_status"]["compiled"] is True
            assert worker["e1_status"]["window_id"] == "e1-w000001"
            assert worker["e1_status"]["template_artifact_id"].endswith("::tpl")
        counts = manager.ledger.reconcile()
        assert counts["G1"] == 1
        assert counts["board_calls"] == 6
        assert counts["K1"] == 6  # one EnvCoder call per unique template
        assert counts["T1"] == 0
        assert counts["F1"] == 0
        assert counts["N1"] == 6 + 6

    def test_fewer_than_12_compiled_refuses_the_whole_window(self):
        # round-3 P0-2: 2 families x 2 variants = 4 compiled artifacts
        # < 12 => INSUFFICIENT_DYNAMIC_ARTIFACTS; the whole window is
        # refused and the batch is the honest 12-entry reuse batch —
        # never stub/placeholder padding of the missing slots
        evidence = _evidence_from_archive()
        store = _board_store(evidence)  # default: fam_a + fam_b
        specs = _specs_via_board(evidence, dict(store))
        assert len(specs) == 4
        _add_template_entries(store, specs, SEEDS)
        manager = _manager(
            replay_store=store, archive_snapshot=ARCHIVE_SNAPSHOT
        )
        workers = manager.evolve_tasks()
        assert len(workers) == L.NUM_DYNAMIC_SLOTS == 12
        assert all(w["e1_status"]["reuse"] is True for w in workers)
        blocked = workers[0]["e1_status"]["blocked_codes"]
        assert "INSUFFICIENT_DYNAMIC_ARTIFACTS" in blocked
        assert "produced 4 compiled" in workers[0]["e1_status"]["note"]
        counts = manager.ledger.reconcile()
        assert counts["K1"] == 2  # the honest template calls still count
        assert counts["board_calls"] == 6

    def test_double_run_equality(self):
        families = [_family(f"fam_{i}") for i in range(6)]
        evidence = _evidence_from_archive()
        store = _board_store(evidence, families=families)
        _add_template_entries(
            store, _specs_via_board(evidence, dict(store)), SEEDS
        )
        first = _manager(
            replay_store=store, archive_snapshot=ARCHIVE_SNAPSHOT
        ).evolve_tasks()
        second = _manager(
            replay_store=store, archive_snapshot=ARCHIVE_SNAPSHOT
        ).evolve_tasks()
        assert first == second

    def test_one_broken_template_among_six_refuses_the_window(self):
        # round-3 P0-2: one template's env-code has a syntax error =>
        # 10 compiled < 12 => the WHOLE window is refused (the broken
        # template's 2 variants are never padded with stubs)
        families = [_family(f"fam_{i}") for i in range(6)]
        evidence = _evidence_from_archive()
        store = _board_store(evidence, families=families)
        specs = _specs_via_board(evidence, dict(store))

        def code_for(spec):
            if spec.family_id == "fam_0":
                return "def broken(:"
            return "def make_env():\n    return 'env'"

        _add_template_entries(store, specs, SEEDS, env_code_by_template=code_for)
        manager = _manager(
            replay_store=store, archive_snapshot=ARCHIVE_SNAPSHOT
        )
        workers = manager.evolve_tasks()
        assert len(workers) == 12
        assert all(w["e1_status"]["reuse"] is True for w in workers)
        blocked = workers[0]["e1_status"]["blocked_codes"]
        assert "INSUFFICIENT_DYNAMIC_ARTIFACTS" in blocked
        assert "produced 10 compiled" in workers[0]["e1_status"]["note"]

    def test_envcoder_parse_failure_yields_failed_worker_no_repair(self):
        # round-3 P0-2: with 8 families (the board contract maximum)
        # one template's artifact_id mismatch kills only its 2
        # variants; 14 compiled >= 12, so the pool is returned with
        # honest failed workers and still NO repair call (F1=0)
        families = [_family(f"fam_{i}") for i in range(8)]
        evidence = _evidence_from_archive()
        store = _board_store(evidence, families=families)
        specs = _specs_via_board(evidence, dict(store))
        assert len(specs) == 16
        broken = next(s for s in specs if s.family_id == "fam_0")
        envelope = EC.build_envcoder_envelope_hash(
            broken, seed_examples=SEEDS
        )
        key = LC.make_replay_key(
            role=M.ENVCODER_ROLE,
            evidence_hash=broken.template_hash,
            prompt_envelope_hash=envelope,
            prompt_version=M.ENVCODER_PROMPT_VERSION,
            schema_version=M.ENVCODER_OUTPUT_SCHEMA_VERSION,
        )
        store[key] = json.dumps(
            {"artifact_id": "WRONG::id", "env_code": "x = 1"}
        )
        _add_template_entries(
            store,
            [s for s in specs if s.template_hash != broken.template_hash],
            SEEDS,
        )
        manager = _manager(
            replay_store=store, archive_snapshot=ARCHIVE_SNAPSHOT
        )
        workers = manager.evolve_tasks()
        assert len(workers) == 16
        failed = [w for w in workers if not w["compiled"]]
        assert len(failed) == 2  # both variants of the broken template
        for worker in failed:
            assert worker["e1_status"]["reuse"] is False
            assert worker["e1_status"]["envcoder_failed_code"] != ""
        counts = manager.ledger.reconcile()
        assert counts["K1"] == 8
        assert counts["F1"] == 0  # NO repair loop this round

    def test_replay_miss_is_a_hard_fail(self):
        manager = _manager(archive_snapshot=ARCHIVE_SNAPSHOT)  # empty store
        with pytest.raises(RuntimeError) as excinfo:
            manager.evolve_tasks()
        assert "HARD FAIL: replay cache miss" in str(excinfo.value)


class TestCheckCompilation:
    def test_valid_code_compiles(self):
        ok, note = _manager().env_generator.check_compilation(
            "def make_env():\n    return 'env'"
        )
        assert ok is True
        assert note == ""

    def test_syntax_error_reported_honestly(self):
        ok, note = _manager().env_generator.check_compilation("def broken(:")
        assert ok is False
        assert note.startswith("SYNTAX_ERROR")

    def test_forbidden_content_is_guard_rejected(self):
        ok, note = _manager().env_generator.check_compilation(
            "def make_env():\n    self.total_reward += 1\n    return 'env'"
        )
        assert ok is False
        assert note != ""
        assert "SYNTAX_ERROR" not in note  # guard, not syntax

    def test_non_str_and_empty_fail_closed(self):
        ok, note = _manager().env_generator.check_compilation(123)
        assert ok is False
        assert GM.GEN_MANAGER_BAD_TYPE in note
        ok, note = _manager().env_generator.check_compilation("   ")
        assert ok is False


class TestObserveSessionFeedback:
    def test_normal_training_feedback_is_stored_and_becomes_evidence(self):
        manager = _manager(llm_client=_GarbageLLM())
        manager.observe_session_feedback(
            4, {"provenance": "NORMAL_TRAINING_FEEDBACK", "success_rate": 0.3}
        )
        manager.evolve_tasks()  # feedback opens a review cycle attempt
        assert manager._cycles_run == 1

    def test_training_provenance_also_admissible(self):
        manager = _manager()
        manager.observe_session_feedback(
            4, {"provenance": "TRAINING", "success_rate": 0.3}
        )
        assert len(manager._pending_feedback) == 1

    @pytest.mark.parametrize(
        "provenance",
        ["FORMAL_FRONT", "FORMAL_EVALUATION", "CANDIDATE_EVALUATION", "junk"],
    )
    def test_inadmissible_provenance_rejected_before_storage(self, provenance):
        manager = _manager()
        with pytest.raises(Exception) as excinfo:
            manager.observe_session_feedback(
                4, {"provenance": provenance, "success_rate": 0.3}
            )
        assert getattr(excinfo.value, "code", "") != ""
        assert manager._pending_feedback == []

    def test_missing_provenance_fails_closed(self):
        manager = _manager()
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.observe_session_feedback(4, {"success_rate": 0.3})
        assert excinfo.value.code == GM.GEN_MANAGER_MISSING_FIELD

    def test_empty_facts_fail_closed(self):
        manager = _manager()
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.observe_session_feedback(
                4, {"provenance": "TRAINING"}
            )
        assert excinfo.value.code == GM.GEN_MANAGER_FEEDBACK_BAD_FACTS

    def test_bad_session_idx_fails_closed(self):
        manager = _manager()
        with pytest.raises(GM.GenManagerError):
            manager.observe_session_feedback(
                True, {"provenance": "TRAINING", "success_rate": 0.3}
            )
        with pytest.raises(GM.GenManagerError):
            manager.observe_session_feedback(
                -1, {"provenance": "TRAINING", "success_rate": 0.3}
            )

    def test_forbidden_fact_content_rejected_before_storage(self):
        manager = _manager()
        with pytest.raises(Exception):
            manager.observe_session_feedback(
                4,
                {
                    "provenance": "TRAINING",
                    "note": "expert waypoint_list goes here",
                },
            )
        assert manager._pending_feedback == []


class TestBatchAndLayout:
    def test_blocked_batch_permits_zero_training_not_anchors_only(self):
        # C13: while hard gates block, the batch trains NOTHING — the
        # old anchors-only batch was a sneak path and is gone. Full
        # gate matrix in test_training_gate.py.
        batch = _manager().build_training_batch()
        assert batch["task_ids"] == []
        assert batch["training_permitted"] is False
        assert batch["provenance"] == "BLOCKED"
        assert batch["layout"] is None
        assert batch["reuse_only"] is True
        assert batch["dynamic_promoted"] == 0
        assert batch["reuse_evidence"] is None
        assert "SELECTION_BLOCKED_NO_REAL_EVIDENCE" in batch["blocked_codes"]
        assert AM.BLOCKED_SHARED_ANCHOR_MANIFEST in batch["blocked_codes"]

    def test_promotion_while_blocked_fails_closed(self):
        # C13: real selection is impossible while gates block; a
        # "promoted" batch must never become trainable here.
        dynamic = [f"dyn_{i:02d}" for i in range(12)]
        with pytest.raises(GM.GenManagerError) as excinfo:
            _manager().build_training_batch(dynamic)
        assert excinfo.value.code == GM.GEN_MANAGER_PROMOTION_BLOCKED

    @pytest.mark.parametrize("count", [1, 11, 13])
    def test_wrong_promoted_count_fails_closed(self, count):
        with pytest.raises(GM.GenManagerError) as excinfo:
            _manager().build_training_batch(
                [f"dyn_{i}" for i in range(count)]
            )
        assert excinfo.value.code == GM.GEN_MANAGER_BAD_DYNAMIC_SET

    def test_build_training_layout_hook_semantics(self):
        manager = _manager()
        assert manager.build_training_layout([]) is None
        assert manager.build_training_layout(None) is None
        layout_map = manager.build_training_layout(
            [f"dyn_{i:02d}" for i in range(12)]
        )
        assert len(layout_map) == 16
        with pytest.raises(L.LayoutError):
            manager.build_training_layout(["only", "five", "ids", "x", "y"])

    def test_anchor_task_ids_property(self):
        assert _manager().anchor_task_ids == L.ANCHOR_TASK_IDS

    def test_select_context_tasks_is_honestly_empty(self):
        manager = _manager()
        assert manager.select_context_tasks() == []
        assert manager.select_context_tasks({"any": 1}, 7) == []


class TestStatusReport:
    def test_status_report_is_honest(self):
        report = _manager().status_report()
        assert report["teacher_type"] == "e1_formal"
        flags = report["flags"]
        assert flags["real_envcoder_used"] is False
        assert flags["real_student_reference_eval"] is False
        assert flags["real_training_update_executed"] is False
        assert report["reference_contract_frozen"] is False
        assert report["anchor_manifest_status"] == AM.STATUS_DRAFT_UNFROZEN
        assert report["learnability_thresholds_present"] is False
        assert report["copeland"]["protocol_version"] == "canonical_v2"
        assert (
            report["copeland"]["source_sha256"] == S.COPELAND_SOURCE_SHA256
        )
        assert "SELECTION_BLOCKED_NO_REAL_EVIDENCE" in report["blocked_codes"]
        assert len(report["disclaimers"]) == 4
        assert "craftax" in report["envcoder_check_scope"]


class TestDuckSurface:
    def test_legacy_surface_exists(self):
        manager = _manager()
        assert isinstance(manager.session_idx, int)
        assert hasattr(manager, "archive")
        assert hasattr(manager.archive, "graph")
        assert hasattr(manager.archive, "save_graph")
        assert hasattr(manager.archive, "_lock")
        assert callable(manager.evolve_tasks)
        assert callable(manager.select_context_tasks)
        assert callable(manager.observe_session_feedback)
        assert callable(manager.build_training_batch)
        assert callable(manager.build_training_layout)


class TestNoHeavyImports:
    """The edge stays importable without jax/craftax (pure stdlib +
    teacher modules)."""

    MODULES = (
        "gen_manager.py",
        "archive_view.py",
        "training_gate.py",
        "eval_adapter.py",
    )

    def test_no_jax_or_craftax_imports(self):
        for relpath in self.MODULES:
            path = os.path.join(
                os.path.dirname(__file__), "..", "..", "src", "dicode",
                "teachers", "e1_formal", relpath,
            )
            with open(path, "r", encoding="utf-8") as handle:
                tree = ast.parse(handle.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert not alias.name.startswith(("jax", "craftax")), relpath
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    assert not module.startswith(("jax", "craftax")), relpath
