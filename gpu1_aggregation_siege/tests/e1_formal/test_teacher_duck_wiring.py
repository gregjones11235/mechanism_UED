"""C11 tests: the E1 teacher's duck surface consumed by the legacy
training loop (setup.py / run_dicode.py / training.py /
evolution_efficient.py hooks).

Everything here runs offline: the teacher degrades honestly (anchors +
REUSE only), worker dicts carry the legacy key aliases, the artifact
registry records compiled artifacts without ever promoting them via
legacy activation, and the archive view answers ``get_task_codes``
with the honest empty mapping.
"""
import pytest

from dicode.teachers.e1_formal import layout
from dicode.teachers.e1_formal.archive_view import (
    ARCHIVE_VIEW_BAD_TYPE,
    ArchiveViewError,
)
from dicode.teachers.e1_formal.gen_manager import (
    GEN_MANAGER_BAD_TYPE,
    GEN_MANAGER_MISSING_FIELD,
    GenManagerError,
)
from test_gen_manager_duck import _manager  # committed-config fixture


class TestWorkerDictLegacyAliases:
    def test_reuse_stubs_carry_both_key_spellings(self):
        manager = _manager()
        workers = manager.evolve_tasks()
        assert len(workers) == 12
        for worker in workers:
            assert worker["generated_task_id"] == worker["task_id"]
            assert worker["code_string"] == worker["code"]
            assert worker["code"] is None  # REUSE stubs carry no code

    def test_all_workers_carry_the_legacy_reasoning_key(self):
        manager = _manager()
        for worker in manager.evolve_tasks():
            assert "reasoning" in worker


class TestConsumeWorkerResults:
    def _compiled(self, task_id="e1-w000001::fam_a::v1"):
        return {
            "task_id": task_id,
            "generated_task_id": task_id,
            "compiled": True,
            "code": "class Env:\n    pass\n",
            "code_string": "class Env:\n    pass\n",
            "reasoning": "",
            "e1_status": {
                "reuse": False,
                "artifact_id": f"{task_id}::a1",
                "spec_hash": "s" * 64,
                "window_id": "e1-w000001",
                "compiled": True,
                "compile_note": "",
            },
        }

    def test_compiled_artifact_recorded_and_legacy_activation_bypassed(self):
        manager = _manager()
        new_ids, compiled_count = manager.consume_worker_results(
            [self._compiled(), {"task_id": "stub", "compiled": False}]
        )
        assert new_ids == []  # E1 NEVER promotes via legacy activation
        assert compiled_count == 1
        registry = manager.artifact_registry
        assert set(registry) == {"e1-w000001::fam_a::v1"}
        record = registry["e1-w000001::fam_a::v1"]
        assert record["code"] == "class Env:\n    pass\n"
        assert record["window_id"] == "e1-w000001"
        assert record["spec_hash"] == "s" * 64

    def test_registry_copy_is_read_only(self):
        manager = _manager()
        manager.consume_worker_results([self._compiled()])
        copy = manager.artifact_registry
        copy["e1-w000001::fam_a::v1"]["code"] = "HACKED"
        copy["intruder"] = {"code": "x"}
        fresh = manager.artifact_registry
        assert fresh["e1-w000001::fam_a::v1"]["code"] == "class Env:\n    pass\n"
        assert "intruder" not in fresh

    @pytest.mark.parametrize(
        "bad",
        [
            None,
            "not a list",
            {"task_id": "a"},
            42,
        ],
    )
    def test_non_list_input_fails_closed(self, bad):
        manager = _manager()
        with pytest.raises(GenManagerError) as excinfo:
            manager.consume_worker_results(bad)
        assert excinfo.value.code == GEN_MANAGER_BAD_TYPE

    def test_missing_task_id_fails_closed(self):
        manager = _manager()
        with pytest.raises(GenManagerError) as excinfo:
            manager.consume_worker_results([{"compiled": False}])
        assert excinfo.value.code == GEN_MANAGER_MISSING_FIELD

    def test_non_bool_compiled_fails_closed(self):
        manager = _manager()
        with pytest.raises(GenManagerError) as excinfo:
            manager.consume_worker_results(
                [{"task_id": "a", "compiled": "yes"}]
            )
        assert excinfo.value.code == GEN_MANAGER_BAD_TYPE

    def test_compiled_without_code_fails_closed(self):
        manager = _manager()
        with pytest.raises(GenManagerError) as excinfo:
            manager.consume_worker_results(
                [{"task_id": "a", "compiled": True, "code": None}]
            )
        assert excinfo.value.code == GEN_MANAGER_MISSING_FIELD

    def test_legacy_shaped_dict_without_task_id_fails_closed(self):
        # run_dicode-shaped dicts lacking the E1 key are rejected, not
        # guessed from generated_task_id
        manager = _manager()
        with pytest.raises(GenManagerError) as excinfo:
            manager.consume_worker_results(
                [{"generated_task_id": "a", "compiled": False}]
            )
        assert excinfo.value.code == GEN_MANAGER_MISSING_FIELD


class TestArchiveGetTaskCodes:
    def test_honest_empty_mapping(self):
        manager = _manager()
        assert manager.archive.get_task_codes(["task_1", "dyn_a"]) == {}

    def test_empty_request_is_empty_answer(self):
        manager = _manager()
        assert manager.archive.get_task_codes([]) == {}

    def test_non_sequence_fails_closed(self):
        manager = _manager()
        with pytest.raises(ArchiveViewError) as excinfo:
            manager.archive.get_task_codes("task_1")
        assert excinfo.value.code == ARCHIVE_VIEW_BAD_TYPE

    def test_bad_id_fails_closed(self):
        manager = _manager()
        with pytest.raises(ArchiveViewError) as excinfo:
            manager.archive.get_task_codes(["task_1", 5])
        assert excinfo.value.code == ARCHIVE_VIEW_BAD_TYPE


class TestSessionFeedbackHookPayload:
    def test_exact_run_dicode_payload_is_admissible(self):
        manager = _manager()
        manager.observe_session_feedback(
            1,
            {
                "provenance": "NORMAL_TRAINING_FEEDBACK",
                "num_updates_in_session": 10,
                "num_tasks_trained": 16,
            },
        )
        assert len(manager._pending_feedback) == 1
        stored = manager._pending_feedback[0]
        assert stored["provenance"] == "NORMAL_TRAINING_FEEDBACK"
        assert stored["facts"]["num_updates_in_session"] == 10

    def test_payload_without_provenance_rejected(self):
        manager = _manager()
        with pytest.raises(GenManagerError):
            manager.observe_session_feedback(
                1, {"num_updates_in_session": 10}
            )

    def test_candidate_evaluation_provenance_rejected_at_llm_layer(self):
        # selector-admissible only; never LLM-role admissible
        manager = _manager()
        with pytest.raises(Exception) as excinfo:
            manager.observe_session_feedback(
                1,
                {
                    "provenance": "CANDIDATE_EVALUATION",
                    "num_updates_in_session": 10,
                },
            )
        assert getattr(excinfo.value, "code", "") != ""


class TestLayoutHookContract:
    def test_empty_dynamic_set_returns_none_legacy_path(self):
        manager = _manager()
        assert manager.build_training_layout([]) is None
        assert manager.build_training_layout(None) is None

    def test_twelve_dynamic_ids_return_pinned_layout(self):
        manager = _manager()
        dynamic = [f"dyn_{i:02d}" for i in range(12)]
        result = manager.build_training_layout(dynamic)
        assert list(result) == dynamic + list(layout.ANCHOR_TASK_IDS)
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-15)

    def test_wrong_count_still_fails_closed(self):
        manager = _manager()
        with pytest.raises(layout.LayoutError):
            manager.build_training_layout(["a", "b", "c"])


class TestContextSelectionHook:
    def test_always_empty_this_round(self):
        manager = _manager()
        assert manager.select_context_tasks("config", 14) == []
        assert manager.select_context_tasks() == []
