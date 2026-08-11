"""Guard tests: both guards fail CLOSED, allowed evidence passes (section 15)."""
import pytest

from d052.bagr_ued import constants as C
from d052.bagr_ued.formal_evaluation_leakage_guard import (
    FormalEvaluationLeakageGuard,
    FormalLeakageViolation,
)
from d052.bagr_ued.trajectory_evidence import (
    EpisodeEvidence,
    EvidenceSource,
    MockSymbolicAdapter,
    TrajectoryEvidenceError,
)
from d052.bagr_ued.synthetic_traces import TEST_VOCABULARY
from d052.bagr_ued.trajectory_supervision_guard import (
    GuardViolation,
    TrajectorySupervisionGuard,
)


class TestTrajectorySupervisionGuard:
    def setup_method(self):
        self.g = TrajectorySupervisionGuard()

    @pytest.mark.parametrize("key", sorted(C.FORBIDDEN_SUPERVISION_KEYS))
    def test_every_forbidden_key_fails_closed(self, key):
        report = self.g.scan({"a": {"b": {key: [1, 2]}}})
        assert not report["passed"]
        assert report["findings"][0]["code"] == \
            GuardViolation.TRAJECTORY_SUPERVISION_KEY_FORBIDDEN
        with pytest.raises(GuardViolation):
            self.g.assert_clean({key: []})

    @pytest.mark.parametrize("text", [
        "不要睡觉，附近有怪物",
        "请远离怪物",
        "Student 应该攻击敌人",
        "you should flee from the hostile",
        "don't sleep here",
        "move away immediately",
        "attack the monster now",
        "walk left to safety",
    ])
    def test_direct_action_advice_fails_closed(self, text):
        report = self.g.scan({"note": text})
        assert not report["passed"]
        assert report["findings"][0]["code"] == \
            GuardViolation.DIRECT_ACTION_ADVICE_FORBIDDEN

    def test_clean_payload_passes(self):
        clean = {"intervention": {"mutation_axes": ["threat_distance_grading"],
                                 "expected_behavior_change":
                                     "rest frequency should respond to graded "
                                     "threat distance at the population level"}}
        report = self.g.assert_clean(clean)
        assert report["passed"]


class TestFormalEvaluationLeakageGuard:
    def setup_method(self):
        self.g = FormalEvaluationLeakageGuard()

    @pytest.mark.parametrize("source", sorted(C.FORBIDDEN_EVIDENCE_SOURCES))
    def test_every_forbidden_source_fails_closed(self, source):
        with pytest.raises(FormalLeakageViolation) as ei:
            self.g.assert_admissible_source(source)
        assert ei.value.code == \
            FormalLeakageViolation.FORMAL_EVALUATION_LEAKAGE
        report = self.g.scan({"data_source": source})
        assert not report["passed"]

    @pytest.mark.parametrize("source", sorted(C.ALLOWED_EVIDENCE_SOURCES))
    def test_allowed_sources_pass(self, source):
        self.g.assert_admissible_source(source)
        assert self.g.scan({"data_source": source})["passed"]

    def test_unknown_source_fails_closed(self):
        with pytest.raises(FormalLeakageViolation) as ei:
            self.g.assert_admissible_source("SOME_RANDOM_SOURCE")
        assert ei.value.code == FormalLeakageViolation.SOURCE_NOT_DECLARED

    def test_frozen_payload_key_fails_closed(self):
        report = self.g.scan({"bundle": {"front_bank_states": "<blob>"}})
        assert not report["passed"]
        assert report["findings"][0]["code"] == \
            FormalLeakageViolation.FORBIDDEN_PROVENANCE_KEY

    def test_free_text_formal_front_mention_fails_closed(self):
        report = self.g.scan(["trajectory loaded from FORMAL_FRONT bank"])
        assert not report["passed"]

    def test_episode_schema_backstop_rejects_formal_source(self):
        with pytest.raises(FormalLeakageViolation):
            EpisodeEvidence(episode_id="x", source=EvidenceSource.FORMAL_FRONT)


class TestSymbolicAdapterBoundary:
    def setup_method(self):
        self.a = MockSymbolicAdapter(TEST_VOCABULARY)

    def test_raw_int_without_vocabulary_fails_closed(self):
        with pytest.raises(TrajectoryEvidenceError) as ei:
            self.a.resolve_action(12345)
        assert ei.value.code == \
            TrajectoryEvidenceError.RAW_ACTION_INT_UNRESOLVED

    def test_raw_int_with_vocabulary_resolves(self):
        name, classes = self.a.resolve_action(5)
        assert name == "REST" and "rest_class" in classes

    def test_leaf_index_state_key_fails_closed(self):
        with pytest.raises(TrajectoryEvidenceError) as ei:
            self.a.summarize_state({"state[42]": 1})
        assert ei.value.code == \
            TrajectoryEvidenceError.RAW_STATE_LEAF_INDEX_FORBIDDEN
