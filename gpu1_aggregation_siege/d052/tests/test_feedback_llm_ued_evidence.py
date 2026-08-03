"""C4: behavior-failure evidence + uncertainty CI (board input layer)."""
import pytest
from pydantic import ValidationError

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.behavior_failure import (
    FEEDBACK_VIEW_UNBOUND,
    REFERENCE_GAP_HIGH,
    REFERENCE_GAP_LOW,
    REFERENCE_GAP_MEDIUM,
    SEVERITY_HIGH,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
    SEVERITY_NONE,
    BehaviorFailureEvidence,
    BoardContext,
    assemble_board_context,
    extract_window_evidence,
    severity_for,
)
from d052.feedback_llm_ued.simulator_feedback_store import (
    MATCH_UNGRADED,
    SimulatorFeedbackRecord,
    SimulatorFeedbackStore,
)
from d052.feedback_llm_ued.synthetic_feedback import (
    synthetic_candidate,
    synthetic_feedback_record,
)
from d052.feedback_llm_ued.uncertainty import (
    UncertainRate,
    ci_halfwidth,
    episodes_from_transitions,
    rate_with_ci,
)


def make_record(i, *, window=0, student_sr=0.4, **kw):
    cand = synthetic_candidate(candidate_id=f"c-ev-{i}",
                               family=C.ENVIRONMENT_FAMILIES[i % 7])
    return synthetic_feedback_record(
        feedback_id=f"fb-ev-w{window}-{i}", candidate=cand, plan_id="plan-ev",
        window=window, student_success_rate=student_sr,
        expected_signature={"student_success_rate": 0.47}, **kw)


class TestUncertainty:
    def test_halfwidth_formula_and_monotonicity(self):
        assert ci_halfwidth(0.5, 100) == pytest.approx(1.96 * 0.05)
        assert ci_halfwidth(0.5, 10) > ci_halfwidth(0.5, 100) \
            > ci_halfwidth(0.5, 1000)
        assert ci_halfwidth(0.0, 8) == 0.0       # degenerate rate: no spread
        assert ci_halfwidth(1.0, 8) == 0.0

    def test_fail_closed_inputs(self):
        with pytest.raises(ValueError, match="ILLEGAL_EPISODE_COUNT"):
            ci_halfwidth(0.5, 0)
        with pytest.raises(ValueError, match="ILLEGAL_EPISODE_COUNT"):
            ci_halfwidth(0.5, True)              # bool is not an episode count
        with pytest.raises(ValueError, match="ILLEGAL_RATE"):
            ci_halfwidth(1.5, 10)
        with pytest.raises(ValueError, match="ILLEGAL_Z"):
            ci_halfwidth(0.5, 10, z=0.0)

    def test_uncertain_rate_clamps_and_overlaps(self):
        high = UncertainRate(estimate=0.98, episodes=8, ci=0.05)
        low = UncertainRate(estimate=0.01, episodes=8, ci=0.05)
        assert high.hi == 1.0 and low.lo == 0.0
        assert not high.overlaps(low)
        mid = UncertainRate(estimate=0.5, episodes=8, ci=0.6)
        assert mid.overlaps(high) and mid.overlaps(low)

    def test_rate_with_ci_and_episode_floor(self):
        r = rate_with_ci(3, 10)
        assert r.estimate == pytest.approx(0.3)
        assert r.episodes == 10 and r.ci == ci_halfwidth(0.3, 10)
        with pytest.raises(ValueError, match="ILLEGAL_SUCCESS_COUNT"):
            rate_with_ci(11, 10)
        assert episodes_from_transitions(768, C.ROLLOUT_LENGTH) == 6
        assert episodes_from_transitions(100, C.ROLLOUT_LENGTH) == 0  # floor
        with pytest.raises(ValueError, match="ILLEGAL_TRANSITIONS"):
            episodes_from_transitions(-1, C.ROLLOUT_LENGTH)


class TestSeverityLadder:
    def test_ladder_thresholds(self):
        assert severity_for(0.0) == SEVERITY_NONE
        assert severity_for(REFERENCE_GAP_LOW - 0.001) == SEVERITY_NONE
        assert severity_for(REFERENCE_GAP_LOW) == SEVERITY_LOW
        assert severity_for(REFERENCE_GAP_MEDIUM) == SEVERITY_MEDIUM
        assert severity_for(REFERENCE_GAP_HIGH) == SEVERITY_HIGH
        assert severity_for(0.95) == SEVERITY_HIGH


class TestBehaviorFailureEvidence:
    def test_from_record_computes_gaps_from_stage_metrics(self):
        rec = make_record(0, student_sr=0.4)     # ref sr 0.9 by factory default
        ev = BehaviorFailureEvidence.from_record(rec)
        assert ev.feedback_id == rec.feedback_id
        assert ev.return_shortfall == pytest.approx(0.5)
        assert ev.behavior_activation_gap == pytest.approx(0.2)
        assert ev.front_progress_gap == pytest.approx(0.81 - 0.4)
        assert ev.reference_gap == pytest.approx(0.5)
        assert ev.severity == SEVERITY_HIGH
        assert ev.early_stop_measured is False        # honest: no lengths
        assert ev.early_stop_rate == 0.0
        assert ev.expected_observed_match == MATCH_UNGRADED

    @pytest.mark.parametrize("student_sr,activation,front,severity", [
        (0.88, 0.96, 0.8, SEVERITY_NONE),     # gaps 0.02 / 0.04 / 0.01
        (0.88, 0.90, 0.8, SEVERITY_LOW),      # activation gap 0.10
        (0.88, 0.70, 0.8, SEVERITY_MEDIUM),   # activation gap 0.20
    ])
    def test_severity_from_synthetic_records(self, student_sr, activation,
                                             front, severity):
        rec = make_record(1, student_sr=student_sr,
                          behavior_activation=activation, front_progress=front)
        assert BehaviorFailureEvidence.from_record(rec).severity == severity

    def test_from_record_without_metrics_fails_closed(self):
        cand = synthetic_candidate(candidate_id="c-ev-empty",
                                   family=C.ENVIRONMENT_FAMILIES[0])
        rec = SimulatorFeedbackRecord(
            feedback_id="fb-empty", candidate_id=cand.candidate_id,
            candidate_hash=cand.candidate_hash, source_plan_id="p",
            window=0, environment_family=cand.environment_family)
        with pytest.raises(ValueError, match="NO_PROBE_METRICS"):
            BehaviorFailureEvidence.from_record(rec)

    def test_with_early_stop_attaches_measured_rate(self):
        ev = BehaviorFailureEvidence.from_record(make_record(2))
        measured = ev.with_early_stop(0.25)
        assert measured.early_stop_measured is True
        assert measured.early_stop_rate == 0.25
        assert measured.feedback_id == ev.feedback_id
        assert measured.reference_gap == ev.reference_gap
        with pytest.raises(ValueError, match="ILLEGAL_EARLY_STOP_RATE"):
            ev.with_early_stop(1.5)

    def test_illegal_fields_rejected(self):
        ev = BehaviorFailureEvidence.from_record(make_record(3))
        payload = ev.model_dump()
        payload["severity"] = "catastrophic"
        with pytest.raises(ValidationError, match="ILLEGAL_SEVERITY"):
            BehaviorFailureEvidence(**payload)
        payload = ev.model_dump()
        payload["expected_observed_match"] = "maybe"
        with pytest.raises(ValidationError, match="ILLEGAL_MATCH_STATE"):
            BehaviorFailureEvidence(**payload)


class TestWindowExtractionAndBoardContext:
    def _store(self):
        store = SimulatorFeedbackStore()
        for i in range(3):
            store.add(make_record(i, window=0, student_sr=0.30 + 0.1 * i))
        store.add(make_record(9, window=1, student_sr=0.6))
        return store

    def test_extraction_is_window_scoped_and_deterministic(self):
        store = self._store()
        ev0 = extract_window_evidence(store, 0)
        assert [e.feedback_id for e in ev0] == \
            ["fb-ev-w0-0", "fb-ev-w0-1", "fb-ev-w0-2"]
        assert extract_window_evidence(store, 5) == []
        assert [e.model_dump() for e in extract_window_evidence(store, 0)] \
            == [e.model_dump() for e in ev0]

    def test_assemble_board_context_pools_episodes_and_ci(self):
        store = self._store()
        ctx = assemble_board_context(store, window=0,
                                     mode=C.MODE_NORMAL_FEEDBACK)
        assert ctx.window == 0
        assert len(ctx.behavior_evidence) == 3
        # each synthetic record: stage-2 transitions = ROLLOUT*(4+2) -> 6 eps
        assert ctx.pooled_episodes == 18
        expected_sr = round((0.3 + 0.4 + 0.5) / 3, 6)
        assert ctx.pooled_student_success_rate == expected_sr
        assert ctx.student_success_rate_ci == round(
            ci_halfwidth(expected_sr, 18), 6)
        assert ctx.feedback_view_label == FEEDBACK_VIEW_UNBOUND

    def test_empty_evidence_is_maximally_uncertain(self):
        ctx = assemble_board_context(SimulatorFeedbackStore(), window=0,
                                     mode=C.MODE_STATIC_LLM)
        assert ctx.behavior_evidence == []
        assert ctx.pooled_episodes == 0
        assert ctx.pooled_student_success_rate == 0.0
        assert ctx.student_success_rate_ci == 1.0

    def test_board_context_rejects_unknown_mode(self):
        with pytest.raises(ValidationError, match="UNKNOWN_MODE"):
            BoardContext(window=0, mode="self_training")

    def test_assembly_is_deterministic(self):
        store = self._store()
        a = assemble_board_context(store, window=0, mode=C.MODE_NORMAL_FEEDBACK)
        b = assemble_board_context(store, window=0, mode=C.MODE_NORMAL_FEEDBACK)
        assert a.model_dump() == b.model_dump()
