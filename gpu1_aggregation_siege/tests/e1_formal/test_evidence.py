"""C4 tests: admissible evidence snapshot + provenance isolation.

Simulates the resume-path injection attack (formal-eval metrics shaped
like training metrics) and asserts the evidence layer rejects it
fail-closed BEFORE any prompt exists. Offline only.
"""
import pytest

from dicode.teachers.e1_formal import evidence as E


def _item(source="training_window.session_metrics", provenance="NORMAL_TRAINING_FEEDBACK", **facts_overrides):
    facts = {"success_rate": 0.4, "skill_get_wood": 0.75}
    facts.update(facts_overrides)
    return {
        "source": source,
        "session_idx": 3,
        "provenance": provenance,
        "facts": facts,
    }


class TestBuildSnapshot:
    def test_valid_snapshot_builds(self):
        snap = E.build_evidence_snapshot(
            [
                _item(),
                _item(
                    source="archive.performance_history",
                    provenance="TRAINING",
                    success_rate_series=[0.1, 0.2, 0.4],
                ),
            ],
            "unit-test",
        )
        assert len(snap.items) == 2
        assert len(snap.evidence_hash) == 64

    def test_build_is_deterministic_double_run(self):
        items = [_item(), _item(source="archive.performance_history", provenance="TRAINING")]
        assert E.build_evidence_snapshot(items, "t") == E.build_evidence_snapshot(
            items, "t"
        )

    def test_empty_snapshot_rejected(self):
        with pytest.raises(E.EvidenceError) as excinfo:
            E.build_evidence_snapshot([], "t")
        assert excinfo.value.code == E._EvCode.EMPTY

    def test_too_many_items_rejected(self):
        items = [_item() for _ in range(E.MAX_EVIDENCE_ITEMS + 1)]
        with pytest.raises(E.EvidenceError) as excinfo:
            E.build_evidence_snapshot(items, "t")
        assert excinfo.value.code == E._EvCode.TOO_MANY

    def test_unknown_field_rejected_no_tier_field_admissible(self):
        item = _item()
        item["tier"] = "gold"
        with pytest.raises(E.EvidenceError) as excinfo:
            E.build_evidence_snapshot([item], "t")
        assert excinfo.value.code == E._EvCode.UNKNOWN_FIELD

    def test_unknown_source_rejected(self):
        with pytest.raises(E.EvidenceError) as excinfo:
            E.build_evidence_snapshot([_item(source="formal_eval.output")], "t")
        assert excinfo.value.code == E._EvCode.UNKNOWN_FIELD


class TestProvenanceIsolation:
    @pytest.mark.parametrize("label", sorted(["FORMAL_FRONT", "FORMAL_BACK", "FORMAL_FULL"]))
    def test_formal_evidence_rejected_at_build_time(self, label):
        with pytest.raises(E.EvidenceError) as excinfo:
            E.build_evidence_snapshot([_item(provenance=label)], "t")
        assert excinfo.value.code == "FORMAL_PROVENANCE_REJECTED"

    def test_candidate_evaluation_rejected_from_evidence(self):
        # selector-side provenance must NEVER enter LLM-bound evidence
        with pytest.raises(E.EvidenceError) as excinfo:
            E.build_evidence_snapshot(
                [_item(provenance="CANDIDATE_EVALUATION")], "t"
            )
        assert excinfo.value.code == "LLM_PROVENANCE_VIOLATION"

    def test_simulated_resume_path_injection_rejected(self):
        # On the resume path, formal-eval output is shaped exactly like
        # training metrics; provenance is re-verified here, not trusted.
        formal_shaped = _item(
            provenance="FORMAL_FULL",
            success_rate=0.93,
            achievement_srs={"get_wood": 1.0},
        )
        with pytest.raises(E.EvidenceError) as excinfo:
            E.build_evidence_snapshot([formal_shaped], "resume-path")
        assert excinfo.value.code == "FORMAL_PROVENANCE_REJECTED"

    def test_missing_provenance_fails_closed(self):
        item = _item()
        del item["provenance"]
        with pytest.raises(E.EvidenceError) as excinfo:
            E.build_evidence_snapshot([item], "t")
        assert excinfo.value.code == "EVIDENCE_MISSING_FIELD"

    def test_poisoned_facts_rejected_by_content_guards(self):
        with pytest.raises(Exception) as excinfo:
            E.build_evidence_snapshot(
                [_item(note="Step 2: press forward")], "t"
            )
        assert getattr(excinfo.value, "code", "") == "ACTION_SEQUENCE_DETECTED"


class TestRendering:
    def test_render_is_deterministic_and_fact_only(self):
        snap = E.build_evidence_snapshot([_item()], "t")
        text = E.render_evidence_for_prompt(snap)
        assert text == E.render_evidence_for_prompt(snap)
        assert "EVIDENCE_SNAPSHOT hash=" in text
        assert "provenance=NORMAL_TRAINING_FEEDBACK" in text
        # no verdict vocabulary anywhere in the rendering
        for banned in ("tier", "verdict", "gold", "bronze", "silver"):
            assert banned not in text.lower()
