"""The three LLM roles (deterministic mock) + mock backend honesty."""
import json

import pytest

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued import (
    adaptive_designer,
    adversarial_reviewer,
    feedback_diagnostician,
)
from d052.feedback_llm_ued.feedback_contracts import FamilyAllocation
from d052.feedback_llm_ued.llm_backend import DeterministicMockFeedbackBackend

FAM0 = C.ENVIRONMENT_FAMILIES[0]
FAM1 = C.ENVIRONMENT_FAMILIES[1]
FAM2 = C.ENVIRONMENT_FAMILIES[2]
FAM3 = C.ENVIRONMENT_FAMILIES[3]


def _hyp_ctx(hid, family, confidence=0.5):
    return dict(hypothesis_id=hid, environment_family=family,
                target_behavior="t", confidence=confidence,
                status=C.HYPOTHESIS_PENDING)


def _fb(fid, hids, match):
    return dict(feedback_id=fid, candidate_id="c-" + fid,
                distinguishes_hypothesis_ids=list(hids),
                expected_observed_match=match)


class TestDiagnostician:
    def test_verdicts_derived_from_feedback_binding(self):
        ctx = dict(window=1, hypotheses=[
            _hyp_ctx("hyp-00", FAM0), _hyp_ctx("hyp-01", FAM1),
            _hyp_ctx("hyp-02", FAM2), _hyp_ctx("hyp-03", FAM3)],
            feedback=[
                _fb("fb-1", ["hyp-00"], C.MATCH_DIRECTION_AGREE),
                _fb("fb-2", ["hyp-01"], C.MATCH_DIRECTION_OPPOSITE),
                _fb("fb-3", ["hyp-02"], C.MATCH_DIRECTION_AGREE),
                _fb("fb-4", ["hyp-02"], C.MATCH_DIRECTION_OPPOSITE)])
        out = feedback_diagnostician.mock_rule(ctx)
        by_id = {v["hypothesis_id"]: v for v in out["hypothesis_verdicts"]}
        assert by_id["hyp-00"]["verdict"] == C.HYPOTHESIS_SUPPORTED
        assert by_id["hyp-00"]["new_confidence"] == pytest.approx(0.60)
        assert by_id["hyp-01"]["verdict"] == C.HYPOTHESIS_REFUTED
        assert by_id["hyp-01"]["new_confidence"] == pytest.approx(0.35)
        assert by_id["hyp-02"]["verdict"] == C.HYPOTHESIS_INCONCLUSIVE
        assert by_id["hyp-02"]["new_confidence"] == pytest.approx(0.45)
        assert by_id["hyp-03"]["verdict"] == C.HYPOTHESIS_STALE
        assert by_id["hyp-03"]["new_confidence"] == pytest.approx(0.50)
        assert by_id["hyp-00"]["feedback_ids"] == ["fb-1"]

    def test_risk_escalation(self):
        # HIGH: refuted AND overall < 0.55
        ctx = dict(window=1, hypotheses=[_hyp_ctx("hyp-00", FAM0)],
                   feedback=[_fb("fb-1", ["hyp-00"],
                                 C.MATCH_DIRECTION_OPPOSITE)])
        out = feedback_diagnostician.mock_rule(ctx)
        assert out["global_risk"] == "HIGH"
        # MEDIUM: refuted but overall >= 0.55
        ctx = dict(window=1, hypotheses=[_hyp_ctx("hyp-00", FAM0),
                                         _hyp_ctx("hyp-01", FAM1)],
                   feedback=[_fb("fb-1", ["hyp-00"],
                                 C.MATCH_DIRECTION_OPPOSITE),
                             _fb("fb-2", ["hyp-01"],
                                 C.MATCH_DIRECTION_AGREE),
                             _fb("fb-3", ["hyp-01"],
                                 C.MATCH_DIRECTION_AGREE),
                             _fb("fb-4", ["hyp-01"],
                                 C.MATCH_DIRECTION_AGREE)])
        out = feedback_diagnostician.mock_rule(ctx)
        assert out["global_risk"] == "MEDIUM"
        # LOW: no refutation, confidence fine
        ctx = dict(window=1, hypotheses=[_hyp_ctx("hyp-00", FAM0)],
                   feedback=[_fb("fb-1", ["hyp-00"],
                                 C.MATCH_DIRECTION_AGREE)])
        out = feedback_diagnostician.mock_rule(ctx)
        assert out["global_risk"] == "LOW"

    def test_diagnosis_bounded_and_legal(self):
        ctx = dict(window=1, hypotheses=[], feedback=[])
        out = feedback_diagnostician.mock_rule(ctx)
        assert out["hypothesis_verdicts"] == []
        assert out["overall_confidence"] == 0.0
        with pytest.raises(ValueError, match="ILLEGAL_VERDICT"):
            feedback_diagnostician.HypothesisVerdict(
                hypothesis_id="h", verdict="CERTAIN", new_confidence=0.5,
                agree_count=0, opposite_count=0, neutral_count=0)
        with pytest.raises(ValueError, match="ILLEGAL_GLOBAL_RISK"):
            feedback_diagnostician.DiagnosisOutput(
                window=0, overall_confidence=0.5, global_risk="EXTREME")

    def test_confidence_clamped(self):
        ctx = dict(window=1, hypotheses=[_hyp_ctx("hyp-00", FAM0, 0.95)],
                   feedback=[_fb("fb-1", ["hyp-00"],
                                 C.MATCH_DIRECTION_AGREE)])
        out = feedback_diagnostician.mock_rule(ctx)
        assert out["hypothesis_verdicts"][0]["new_confidence"] == 0.95
        assert out["hypothesis_verdicts"][0]["new_confidence"] <= 0.95


class TestDesigner:
    def _verdict(self, hid, verdict, fids=("fb-1",), conf=0.6, agree=1,
                 opposite=0):
        return dict(hypothesis_id=hid, verdict=verdict, new_confidence=conf,
                    agree_count=agree, opposite_count=opposite,
                    neutral_count=0, feedback_ids=list(fids), reason="r")

    def test_verdict_to_decision_mapping(self):
        ctx = dict(window=2, verdicts=[
            self._verdict("hyp-00", C.HYPOTHESIS_SUPPORTED, ("fb-1",)),
            self._verdict("hyp-01", C.HYPOTHESIS_REFUTED, ("fb-2",),
                          conf=0.3, agree=0, opposite=1),
            self._verdict("hyp-02", C.HYPOTHESIS_STALE, (), conf=0.5,
                          agree=0)],
            hypotheses=[dict(hypothesis_id="hyp-00", environment_family=FAM0),
                        dict(hypothesis_id="hyp-01", environment_family=FAM1),
                        dict(hypothesis_id="hyp-02", environment_family=FAM2)],
            budget=C.DYNAMIC_UED_SLOTS, global_risk="LOW")
        out = adaptive_designer.mock_rule(ctx)
        by_fam = {a["environment_family"]: a for a in out["allocations"]}
        assert by_fam[FAM0]["decision"] == C.DECISION_RETAIN
        assert by_fam[FAM0]["based_on_feedback_ids"] == ["fb-1"]
        assert by_fam[FAM0]["is_exploration"] is False
        assert by_fam[FAM1]["decision"] == C.DECISION_RETIRE
        assert by_fam[FAM1]["slots"] == 0
        assert by_fam[FAM2]["decision"] == C.DECISION_MUTATE
        assert by_fam[FAM2]["is_exploration"] is True     # STALE = no feedback
        assert by_fam[FAM2]["based_on_feedback_ids"] == []
        # bounded exploration over hypothesis-less families
        explorers = [a for a in out["allocations"] if a["is_exploration"]]
        assert len(explorers) <= C.MAX_EXPLORATION_PROPOSALS + 1  # +STALE one
        assert out["request_control"] is False

    def test_request_control_on_high_risk(self):
        ctx = dict(window=2, verdicts=[], hypotheses=[],
                   budget=C.DYNAMIC_UED_SLOTS, global_risk="HIGH")
        out = adaptive_designer.mock_rule(ctx)
        assert out["request_control"] is True

    def test_output_honesty_validators(self):
        def fn(**over):
            base = dict(environment_family=FAM0, slots=2,
                        decision=C.DECISION_RETAIN, reason="r")
            base.update(over)
            return adaptive_designer.DesignerOutput(
                window=1, allocations=[FamilyAllocation(**base)])
        with pytest.raises(ValueError, match="EXPLORATION_LABEL_REQUIRED"):
            fn()                                          # uncited RETAIN
        with pytest.raises(ValueError, match="EXPLORATION_DECISION_ONLY"):
            fn(decision=C.DECISION_RETIRE, is_exploration=True, slots=0)
        with pytest.raises(ValueError, match="MASQUERADE_FORBIDDEN"):
            fn(based_on_feedback_ids=["fb-1"], is_exploration=True)

    def test_exploration_never_cites_feedback(self):
        ctx = dict(window=2, verdicts=[], hypotheses=[],
                   budget=C.DYNAMIC_UED_SLOTS, global_risk="LOW")
        out = adaptive_designer.mock_rule(ctx)
        for a in out["allocations"]:
            if a["is_exploration"]:
                assert a["based_on_feedback_ids"] == []
                assert a["decision"] in C.EXPLORATION_DECISIONS


class TestReviewer:
    def test_each_risk_trigger(self):
        cases = [
            (dict(overall_confidence=0.4), C.RISK_LOW_CONFIDENCE),
            (dict(allocations=[
                dict(environment_family=FAM0, decision=C.DECISION_RETAIN),
                dict(environment_family=FAM0, decision=C.DECISION_RETIRE)]),
             C.RISK_CONFLICTING_INTERVENTIONS),
            (dict(global_risk="HIGH"), C.RISK_GLOBAL_RISK_HIGH),
            (dict(windows_without_improvement=2),
             C.RISK_NO_IMPROVEMENT_TWO_WINDOWS),
            (dict(opposite_probe_count=1), C.RISK_PROBE_OPPOSITE_DIRECTION),
            (dict(reject_rate=0.35), C.RISK_HIGH_REJECT_RATE),
            (dict(preparing_formal_run=True),
             C.RISK_BEFORE_FORMAL_CANDIDATE_RUN),
        ]
        for ctx, expected in cases:
            fired = adversarial_reviewer.evaluate_risk_triggers(ctx)
            assert expected in fired, ctx
        assert adversarial_reviewer.evaluate_risk_triggers(
            dict(overall_confidence=0.9, reject_rate=0.1,
                 opposite_probe_count=0)) == []

    def test_reviewer_output_illegal_trigger(self):
        with pytest.raises(ValueError, match="ILLEGAL_RISK_TRIGGER"):
            adversarial_reviewer.ReviewerOutput(window=1,
                                                triggered_by=["vibes"])

    def test_overconfidence_concern(self):
        ctx = dict(window=1, triggered_by=[C.RISK_LOW_CONFIDENCE],
                   verdicts=[dict(hypothesis_id="h", verdict="SUPPORTED",
                                  new_confidence=0.8, agree_count=1,
                                  opposite_count=0, neutral_count=0,
                                  feedback_ids=["fb-1"], reason="r")],
                   hypotheses=[], allocations=[])
        out = adversarial_reviewer.mock_rule(ctx)
        assert any("over-confident" in c for c in out["concerns"])

    def test_family_contradiction_forces_retire(self):
        ctx = dict(window=1, triggered_by=[C.RISK_CONFLICTING_INTERVENTIONS],
                   verdicts=[], hypotheses=[],
                   allocations=[
                       dict(environment_family=FAM0,
                            decision=C.DECISION_RETAIN, is_exploration=False),
                       dict(environment_family=FAM0,
                            decision=C.DECISION_RETIRE, is_exploration=False)])
        out = adversarial_reviewer.mock_rule(ctx)
        assert FAM0 in out["forced_retire_families"]
        assert out["approve"] is False

    def test_unbounded_exploration_flagged(self):
        ctx = dict(window=1, triggered_by=[], verdicts=[], hypotheses=[],
                   allocations=[dict(environment_family=f,
                                     decision=C.DECISION_MUTATE,
                                     is_exploration=True)
                                for f in C.ENVIRONMENT_FAMILIES[:3]])
        out = adversarial_reviewer.mock_rule(ctx)
        assert any("not bounded" in c for c in out["concerns"])
        assert out["approve"] is False


class TestMockBackend:
    def test_unknown_role_raises(self):
        backend = DeterministicMockFeedbackBackend()
        with pytest.raises(KeyError, match="UNKNOWN_ROLE_FOR_MOCK_BACKEND"):
            backend.complete("no_such_role", "prompt")

    def test_real_calls_stay_zero(self):
        backend = DeterministicMockFeedbackBackend()
        ctx = dict(window=1, hypotheses=[_hyp_ctx("hyp-00", FAM0)],
                   feedback=[_fb("fb-1", ["hyp-00"],
                                 C.MATCH_DIRECTION_AGREE)])
        prompt = feedback_diagnostician.build_prompt(ctx)
        raw = backend.complete(C.ROLE_FEEDBACK_DIAGNOSTICIAN, prompt)
        parsed = json.loads(raw)
        assert parsed["window"] == 1
        assert backend.real_calls == 0
        assert backend.mock_calls == 1
        backend.assert_no_real_calls()

    def test_round_trip_envelope_is_hash_bound_and_deterministic(self):
        backend = DeterministicMockFeedbackBackend()
        ctx = dict(window=3, hypotheses=[_hyp_ctx("hyp-00", FAM0)],
                   feedback=[_fb("fb-1", ["hyp-00"],
                                 C.MATCH_DIRECTION_AGREE)])
        env1 = feedback_diagnostician.run(ctx, backend, window=3, sequence=0)
        env2 = feedback_diagnostician.run(ctx, backend, window=3, sequence=0)
        assert env1.role == C.ROLE_FEEDBACK_DIAGNOSTICIAN
        assert len(env1.request_hash) == 64 and len(env1.response_hash) == 64
        assert env1.raw_response == env2.raw_response
        assert env1.request_hash == env2.request_hash
        assert env1.parsed_json["window"] == 3
        assert env1.prompt_version == feedback_diagnostician.PROMPT_VERSION
        assert env1.backend_id == C.MOCK_BACKEND_ID
        assert env1.model_id == C.MOCK_MODEL_ID

    def test_all_three_roles_dispatch(self):
        backend = DeterministicMockFeedbackBackend()
        diag_ctx = dict(window=1, hypotheses=[], feedback=[])
        design_ctx = dict(window=1, verdicts=[], hypotheses=[],
                          budget=C.DYNAMIC_UED_SLOTS, global_risk="LOW")
        review_ctx = dict(window=1, triggered_by=[], verdicts=[],
                          hypotheses=[], allocations=[])
        feedback_diagnostician.run(diag_ctx, backend, window=1, sequence=0)
        adaptive_designer.run(design_ctx, backend, window=1, sequence=1)
        adversarial_reviewer.run(review_ctx, backend, window=1, sequence=2)
        assert backend.mock_calls == 3
        assert backend.real_calls == 0
