"""Canonical contracts + honesty rules of the feedback-adaptive loop."""
import pytest

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.feedback_contracts import (
    CandidateEnvironment,
    CurriculumPlan,
    FamilyAllocation,
    ProbeMetrics,
    build_role_prompt,
    extract_context,
    plan_signature_hash,
)
from d052.feedback_llm_ued.plan_revision import (
    FEEDBACK_DRIVEN_LABEL,
    PlanModification,
    PlanRevisionRecord,
    assert_feedback_ids_known,
    budget_changes,
)

FAM = C.ENVIRONMENT_FAMILIES[0]
FAM2 = C.ENVIRONMENT_FAMILIES[1]
AX = C.MUTATION_AXES[0]


def _candidate(**over):
    base = dict(candidate_id="cand-1", environment_family=FAM,
                axis_values={AX: "high"}, variant_id="v1",
                variant_kind="perturb", mutation_axes=[AX],
                provenance={"source": C.SOURCE_CANDIDATE_PROBE})
    base.update(over)
    return CandidateEnvironment(**base)


class TestCandidateEnvironment:
    def test_hash_computed_and_stable(self):
        a, b = _candidate(), _candidate()
        assert len(a.candidate_hash) == 64
        assert a.candidate_hash == b.candidate_hash

    def test_axis_change_changes_hash(self):
        a = _candidate()
        b = _candidate(axis_values={AX: "low"})
        assert a.candidate_hash != b.candidate_hash

    def test_unknown_family_rejected(self):
        with pytest.raises(ValueError, match="UNKNOWN_ENVIRONMENT_FAMILY"):
            _candidate(environment_family="not_a_family")

    def test_illegal_axis_rejected(self):
        with pytest.raises(ValueError, match="ILLEGAL_CANDIDATE_AXIS"):
            _candidate(mutation_axes=["reward_scale"])

    def test_real_adapter_status_honest(self):
        assert _candidate().real_adapter_status == \
            C.REAL_SIMULATOR_PROBE_STATUS


class TestProbeMetrics:
    def test_formal_source_rejected(self):
        with pytest.raises(ValueError, match="PROBE_SOURCE_NOT_ALLOWED"):
            ProbeMetrics(stage="fast", student_success_rate=0.5,
                         student_behavior_activation=0.5,
                         student_front_progress=0.5,
                         reference_success_rate=0.9,
                         reference_mean_progress=0.8,
                         reference_behavior_activation=0.9,
                         global_retention=0.9, regret=0.1, learnability=0.5,
                         simulator_transitions=128,
                         probe_source=C.SOURCE_FORMAL_FRONT)

    def test_probe_source_ok(self):
        m = ProbeMetrics(stage="fast", student_success_rate=0.5,
                         student_behavior_activation=0.5,
                         student_front_progress=0.5,
                         reference_success_rate=0.9,
                         reference_mean_progress=0.8,
                         reference_behavior_activation=0.9,
                         global_retention=0.9, regret=0.1, learnability=0.5,
                         simulator_transitions=128)
        assert m.probe_source == C.SOURCE_CANDIDATE_PROBE


class TestPlanAndRevision:
    def test_plan_signature_order_independent(self):
        a1 = FamilyAllocation(environment_family=FAM, slots=3,
                              decision=C.DECISION_RETAIN,
                              based_on_feedback_ids=["fb-1"], reason="r")
        a2 = FamilyAllocation(environment_family=FAM2, slots=2,
                              decision=C.DECISION_MUTATE,
                              based_on_feedback_ids=["fb-1"], reason="r")
        p1 = CurriculumPlan(plan_id="p1", window=1,
                            mode=C.MODE_NORMAL_FEEDBACK, allocations=[a1, a2])
        p2 = CurriculumPlan(plan_id="p1", window=1,
                            mode=C.MODE_NORMAL_FEEDBACK, allocations=[a2, a1])
        assert plan_signature_hash(p1) == plan_signature_hash(p2)

    def test_revision_label_forced_exploration(self):
        mod = PlanModification(environment_family=FAM,
                               decision=C.DECISION_MUTATE, reason="explore",
                               is_exploration=True, new_slots=2)
        with pytest.raises(ValueError, match="REVISION_LABEL_FORCED"):
            PlanRevisionRecord(revision_id="rev-1", window=0,
                               mode=C.MODE_STATIC_LLM, new_plan_id="p",
                               modifications=[mod],
                               label=FEEDBACK_DRIVEN_LABEL)

    def test_uncited_non_exploration_decision_rejected(self):
        with pytest.raises(ValueError, match="EXPLORATION_LABEL_REQUIRED"):
            PlanModification(environment_family=FAM,
                             decision=C.DECISION_RETAIN, reason="r",
                             new_slots=2)

    def test_uncited_retire_rejected(self):
        with pytest.raises(ValueError, match="EXPLORATION_DECISION_ONLY"):
            PlanModification(environment_family=FAM,
                             decision=C.DECISION_RETIRE, reason="r",
                             is_exploration=True, new_slots=0)

    def test_masquerade_forbidden(self):
        with pytest.raises(ValueError, match="MASQUERADE_FORBIDDEN"):
            PlanModification(environment_family=FAM,
                             decision=C.DECISION_RETAIN, reason="r",
                             based_on_feedback_ids=["fb-1"],
                             is_exploration=True, new_slots=2)

    def test_feedback_driven_revision_and_citation_check(self):
        mod = PlanModification(environment_family=FAM,
                               decision=C.DECISION_RETAIN, reason="probe ok",
                               based_on_feedback_ids=["fb-1"], old_slots=2,
                               new_slots=3)
        rev = PlanRevisionRecord(revision_id="rev-1", window=1,
                                 mode=C.MODE_NORMAL_FEEDBACK,
                                 previous_plan_id="p0", new_plan_id="p1",
                                 based_on_feedback_ids=["fb-1"],
                                 modifications=[mod],
                                 label=FEEDBACK_DRIVEN_LABEL)
        assert_feedback_ids_known(rev, {"fb-1"})
        with pytest.raises(ValueError, match="UNKNOWN_FEEDBACK_ID"):
            assert_feedback_ids_known(rev, set())
        changes = budget_changes(rev)
        assert changes[0]["delta"] == 1

    def test_record_level_ids_must_equal_union(self):
        mod = PlanModification(environment_family=FAM,
                               decision=C.DECISION_RETAIN, reason="r",
                               based_on_feedback_ids=["fb-1"], new_slots=2)
        with pytest.raises(ValueError, match="FEEDBACK_ID_MISMATCH"):
            PlanRevisionRecord(revision_id="rev-1", window=1,
                               mode=C.MODE_NORMAL_FEEDBACK, new_plan_id="p",
                               based_on_feedback_ids=["fb-2"],
                               modifications=[mod],
                               label=FEEDBACK_DRIVEN_LABEL)


class TestPromptContextRoundTrip:
    def test_round_trip(self):
        ctx = {"window": 1, "hypotheses": [{"id": "h"}]}
        prompt = build_role_prompt("instructions", ctx)
        assert extract_context(prompt) == ctx

    def test_missing_block_raises(self):
        with pytest.raises(ValueError, match="MISSING_CONTEXT_BLOCK"):
            extract_context("no markers here")
