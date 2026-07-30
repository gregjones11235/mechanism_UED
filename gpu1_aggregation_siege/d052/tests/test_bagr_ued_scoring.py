"""Scoring / Soft Copeland / budget tests (sections 12-14)."""
import random

import pytest

from d052.bagr_ued import constants as C
from d052.bagr_ued.behavioral_gap import BehavioralGapScore
from d052.bagr_ued.budget_allocator import BudgetAllocator
from d052.bagr_ued.regret_scorer import (
    RegretEvidence,
    ScenarioScope,
    combined_regret_scores,
)
from d052.bagr_ued.soft_copeland import (
    EnvironmentScoreBundle,
    soft_copeland_rank,
)
from d052.bagr_ued.trajectory_evidence import EvidenceSource


def test_three_scores_never_fused():
    ev = [RegretEvidence(environment_id="e1", scope=ScenarioScope.FRONT,
                         student_success_rate=0.2, reference_success_rate=0.9,
                         severity_weight=1.0),
          RegretEvidence(environment_id="e1", scope=ScenarioScope.GLOBAL,
                         student_success_rate=0.4, reference_success_rate=0.9,
                         severity_weight=1.0)]
    scores = combined_regret_scores(ev)
    assert scores.front_regret == pytest.approx(0.7)
    assert scores.global_regret == pytest.approx(0.5)
    assert scores.fused is False
    assert scores.per_environment["e1"]["front_regret"] != \
        scores.per_environment["e1"]["global_regret"]


def test_regret_rejects_formal_source():
    with pytest.raises(ValueError, match="FORBIDDEN_REGRET_SOURCE"):
        RegretEvidence(environment_id="e", scope=ScenarioScope.FRONT,
                       student_success_rate=0.1, reference_success_rate=0.9,
                       severity_weight=1.0,
                       source=EvidenceSource.FORMAL_FRONT)


def test_behavioral_gap_definition_enforced():
    g = BehavioralGapScore(environment_id="e",
                           student_behavior_failure_score=0.6,
                           reference_behavior_failure_score=0.1,
                           behavioral_gap=0.5)
    assert g.behavioral_gap == 0.5
    # negative gaps clamp to 0
    g0 = BehavioralGapScore(environment_id="e",
                            student_behavior_failure_score=0.05,
                            reference_behavior_failure_score=0.1,
                            behavioral_gap=0.0)
    assert g0.behavioral_gap == 0.0
    with pytest.raises(ValueError, match="BEHAVIORAL_GAP_DEFINITION"):
        BehavioralGapScore(environment_id="e",
                           student_behavior_failure_score=0.6,
                           reference_behavior_failure_score=0.1,
                           behavioral_gap=0.99)


def _bundle(i, penalty=0.0):
    # unit-interval signals stay strictly inside [0, 1] for any i
    return EnvironmentScoreBundle(
        environment_id=f"env{i}", front_regret=0.2 + 0.1 * i,
        global_regret=0.3 + 0.05 * i, behavioral_gap=0.05 * i,
        learning_progress=min(0.95, 0.05 + 0.05 * i), learnability=0.5,
        diversity=min(0.9, 0.05 + 0.05 * i),
        global_retention=0.8, critic_penalty=penalty, alpha_front=0.5)


def test_soft_copeland_deterministic_and_order_invariant():
    bundles = [_bundle(i) for i in range(6)]
    r1 = soft_copeland_rank(bundles)
    shuffled = bundles[:]
    random.Random(7).shuffle(shuffled)
    r2 = soft_copeland_rank(shuffled)
    assert r1.ranking_hash == r2.ranking_hash
    assert [e.environment_id for e in r1.entries] == \
        [e.environment_id for e in r2.entries]


def test_soft_copeland_receives_all_eight_inputs():
    r = soft_copeland_rank([_bundle(i) for i in range(3)])
    comp = r.entries[0].components
    assert {"alpha_front_regret", "one_minus_alpha_global_regret",
            "behavioral_gap", "learning_progress", "learnability",
            "diversity", "global_retention",
            "critic_penalty_subtracted"} <= set(comp)


def test_critic_penalty_lowers_strength():
    clean = soft_copeland_rank([_bundle(1), _bundle(2)])
    penalized = soft_copeland_rank([_bundle(1, penalty=0.9), _bundle(2)])
    s_clean = {e.environment_id: e.strength for e in clean.entries}
    s_pen = {e.environment_id: e.strength for e in penalized.entries}
    assert s_pen["env1"] < s_clean["env1"]


def test_budget_12_ued_plus_4_anchors():
    ranking = soft_copeland_rank([_bundle(i) for i in range(16)])
    plan = BudgetAllocator().allocate(ranking)
    assert plan.status == "OK"
    assert len(plan.ued_slots) == C.UED_ACTIVE_SLOTS == 12
    assert plan.anchor_slots == list(C.GLOBAL_CANONICAL_ANCHOR_IDS)
    assert len(plan.anchor_slots) == 4
    assert not (set(plan.ued_slots) & set(plan.anchor_slots))


def test_budget_honest_shortfall_no_backfill():
    ranking = soft_copeland_rank([_bundle(i) for i in range(5)])
    plan = BudgetAllocator().allocate(ranking)
    assert plan.status == "INSUFFICIENT"
    assert len(plan.ued_slots) == 5
    assert "NO backfill" in plan.shortfall_note
    assert len(plan.anchor_slots) == 4   # anchors always reserved


def test_batch_plan_arithmetic():
    from d052.bagr_ued.batch_planner import BatchPlanner
    plan = BatchPlanner().plan(total_updates=8)
    assert plan.transitions_per_update == C.NUM_ENVS * C.ROLLOUT_LENGTH == 2048
    assert plan.review_interval_transitions == 8192
    assert [w.after_update for w in plan.review_windows] == [4, 8]
    assert plan.review_windows[0].cumulative_transitions == 8192
