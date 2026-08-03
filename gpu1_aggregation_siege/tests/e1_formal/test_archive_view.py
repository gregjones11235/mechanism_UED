"""C10 tests: provenance-admissible archive view (feeds evidence.py)."""
import pytest

from dicode.teachers.e1_formal import archive_view as AV


def _task(task_id="task_a", provenance="TRAINING", history=((3, 0.4),)):
    return {
        "task_id": task_id,
        "provenance": provenance,
        "performance_history": [
            {"session_idx": s, "success_rate": r} for s, r in history
        ],
    }


def _snapshot(tasks):
    return {"tasks": list(tasks)}


class TestConsume:
    def test_happy_path(self):
        view = AV.consume_archive_snapshot(
            _snapshot([_task(), _task("task_b", "NORMAL_TRAINING_FEEDBACK")]),
            "test",
        )
        assert view.task_ids == ("task_a", "task_b")
        assert view.tasks[0].provenance == "TRAINING"
        assert view.tasks[0].history == ((3, 0.4),)

    def test_empty_tasks_is_honest_initial_state(self):
        view = AV.consume_archive_snapshot(_snapshot([]), "test")
        assert view.tasks == ()
        assert view.evidence_items() == []
        assert AV.empty_archive_view().tasks == ()

    def test_empty_history_task_emits_no_evidence(self):
        view = AV.consume_archive_snapshot(
            _snapshot([_task(history=())]), "test"
        )
        assert view.evidence_items() == []

    @pytest.mark.parametrize(
        "provenance", ["TRAINING", "NORMAL_TRAINING_FEEDBACK"]
    )
    def test_llm_role_admissible_provenances_accepted(self, provenance):
        view = AV.consume_archive_snapshot(
            _snapshot([_task(provenance=provenance)]), "test"
        )
        assert view.tasks[0].provenance == provenance

    @pytest.mark.parametrize(
        "provenance",
        ["FORMAL_FRONT", "FORMAL_EVALUATION", "CANDIDATE_EVALUATION", "junk"],
    )
    def test_inadmissible_provenances_rejected(self, provenance):
        with pytest.raises(Exception) as excinfo:
            AV.consume_archive_snapshot(
                _snapshot([_task(provenance=provenance)]), "test"
            )
        assert getattr(excinfo.value, "code", "") != ""

    def test_unknown_snapshot_field_fails_closed(self):
        with pytest.raises(AV.ArchiveViewError) as excinfo:
            AV.consume_archive_snapshot(
                {"tasks": [], "tier": "gold"}, "test"
            )
        assert excinfo.value.code == AV.ARCHIVE_VIEW_UNKNOWN_FIELD

    def test_missing_tasks_fails_closed(self):
        with pytest.raises(AV.ArchiveViewError) as excinfo:
            AV.consume_archive_snapshot({}, "test")
        assert excinfo.value.code == AV.ARCHIVE_VIEW_MISSING_FIELD

    def test_unknown_task_field_fails_closed(self):
        task = _task()
        task["learnability_prior_lp"] = 0.25
        with pytest.raises(AV.ArchiveViewError) as excinfo:
            AV.consume_archive_snapshot(_snapshot([task]), "test")
        assert excinfo.value.code == AV.ARCHIVE_VIEW_UNKNOWN_FIELD

    def test_missing_task_field_fails_closed(self):
        task = _task()
        del task["performance_history"]
        with pytest.raises(AV.ArchiveViewError) as excinfo:
            AV.consume_archive_snapshot(_snapshot([task]), "test")
        assert excinfo.value.code == AV.ARCHIVE_VIEW_MISSING_FIELD

    def test_duplicate_task_id_fails_closed(self):
        with pytest.raises(AV.ArchiveViewError) as excinfo:
            AV.consume_archive_snapshot(
                _snapshot([_task(), _task()]), "test"
            )
        assert excinfo.value.code == AV.ARCHIVE_VIEW_OUT_OF_RANGE

    @pytest.mark.parametrize(
        "field,value",
        [
            ("session_idx", True),
            ("session_idx", 2.0),
            ("session_idx", -1),
            ("success_rate", True),
            ("success_rate", "0.4"),
            ("success_rate", 1.5),
            ("success_rate", -0.1),
        ],
    )
    def test_bad_history_entries_fail_closed(self, field, value):
        task = _task()
        task["performance_history"] = [
            {"session_idx": 3, "success_rate": 0.4}
        ]
        task["performance_history"][0][field] = value
        with pytest.raises(AV.ArchiveViewError):
            AV.consume_archive_snapshot(_snapshot([task]), "test")

    def test_unknown_history_field_fails_closed(self):
        task = _task()
        task["performance_history"] = [
            {"session_idx": 3, "success_rate": 0.4, "verdict": "good"}
        ]
        with pytest.raises(AV.ArchiveViewError) as excinfo:
            AV.consume_archive_snapshot(_snapshot([task]), "test")
        assert excinfo.value.code == AV.ARCHIVE_VIEW_UNKNOWN_FIELD

    def test_non_mapping_inputs_fail_closed(self):
        with pytest.raises(AV.ArchiveViewError) as excinfo:
            AV.consume_archive_snapshot([], "test")
        assert excinfo.value.code == AV.ARCHIVE_VIEW_BAD_TYPE


class TestEvidenceEmission:
    def test_items_carry_known_source_and_latest_session(self):
        view = AV.consume_archive_snapshot(
            _snapshot([_task(history=((1, 0.2), (5, 0.6), (2, 0.3)))]),
            "test",
        )
        items = view.evidence_items()
        assert len(items) == 1
        item = items[0]
        assert item["source"] == AV.SOURCE_PERFORMANCE_HISTORY
        assert item["session_idx"] == 5  # latest session in the history
        assert item["provenance"] == "TRAINING"
        assert item["facts"]["task_id"] == "task_a"
        assert item["facts"]["history"] == [[1, 0.2], [5, 0.6], [2, 0.3]]

    def test_items_are_consumed_by_evidence_builder(self):
        # cross-module contract: emitted items pass evidence.py
        from dicode.teachers.e1_formal.evidence import build_evidence_snapshot

        view = AV.consume_archive_snapshot(_snapshot([_task()]), "test")
        snapshot = build_evidence_snapshot(view.evidence_items(), "test")
        assert len(snapshot.items) == 1
        assert len(snapshot.evidence_hash) == 64


class TestDuckSurface:
    def test_graph_is_read_only_snapshot(self):
        view = AV.consume_archive_snapshot(_snapshot([_task()]), "test")
        graph = view.graph
        assert set(graph) == {"task_a"}
        assert graph["task_a"]["provenance"] == "TRAINING"
        assert graph["task_a"]["performance_history"] == [
            {"session_idx": 3, "success_rate": 0.4}
        ]
        # mutating the returned dict must not corrupt the view
        graph["task_a"]["provenance"] = "HACKED"
        assert view.graph["task_a"]["provenance"] == "TRAINING"

    def test_save_graph_is_an_honest_noop(self):
        view = AV.consume_archive_snapshot(_snapshot([_task()]), "test")
        assert view.save_graph() is None
        assert view.task_ids == ("task_a",)

    def test_lock_is_a_usable_context_manager(self):
        view = AV.consume_archive_snapshot(_snapshot([_task()]), "test")
        with view._lock:
            assert view.task_ids == ("task_a",)
