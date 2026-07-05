"""Tests for the student ability-gate (2026-07-02, v1_experiment.md §10.9).

Two halves, one shared definition of the reachable ceiling:
  - craftax_achievements.tier_mastery / reachable_ceiling / tier_overreach_factor  (pure helpers)
  - ambition.ambition_gain(..., reachable_ceiling=)  (bid-side soft discount)

The gate anchors ambition to the student's ability: gap on tiers strictly deeper than the reachable
ceiling is geometrically discounted, so ambitious can no longer out-bid feasible on unreachable tiers.
"""

import pytest

from auction.ambition import ambition_gain
from auction.craftax_achievements import (
    DEPTH_TIERS,
    MASTERY_THRESHOLD_DEFAULT,
    reachable_ceiling,
    tier_mastery,
    tier_overreach_factor,
)
from auction.proposal import Proposal


def _p(achs):
    return Proposal("p", "ambitious", "task_0", "d", "r", frozenset(achs))


# --- a representative achievement per tier (all present in DEPTH_TIERS) --------------------------
_T1 = "collect_wood"        # tier 1
_T2 = "make_iron_sword"     # tier 2
_T3 = "defeat_orc_mage"     # tier 3
_T4 = "defeat_necromancer"  # tier 4


def _gap_all(v: float) -> dict:
    """target_gap = v for every achievement in the universe (uniform student)."""
    return {a: v for tier in DEPTH_TIERS.values() for a in tier}


# --- tier_mastery ------------------------------------------------------------------------------

def test_tier_mastery_uniform():
    # gap 0.4 everywhere => mastery 0.6 everywhere.
    m = tier_mastery(_gap_all(0.4))
    for tier in DEPTH_TIERS:
        assert m[tier] == pytest.approx(0.6)


def test_tier_mastery_missing_default_is_unmastered():
    # SAFE default (missing_is_mastered=False): empty gap => every achievement UNMEASURED =>
    # gap 1.0 => mastery 0.0. An absent tier reads as NOT mastered (conservative gate).
    m = tier_mastery({})
    for tier in DEPTH_TIERS:
        assert m[tier] == pytest.approx(0.0)


def test_tier_mastery_missing_lenient_is_mastered():
    # Legacy/lenient: empty gap => gap 0 => mastery 1.0. Only for guaranteed-complete profiles.
    m = tier_mastery({}, missing_is_mastered=True)
    for tier in DEPTH_TIERS:
        assert m[tier] == pytest.approx(1.0)


# --- reachable_ceiling -------------------------------------------------------------------------

def test_ceiling_empty_profile_is_conservative():
    # SAFE default: no measurements => nothing mastered => stay at tier 1 (don't wave anything up).
    assert reachable_ceiling({}) == 1


def test_ceiling_all_mastered_is_deepest():
    # gap 0 explicitly for EVERY achievement (mastery 1.0) => may aim at the deepest tier.
    assert reachable_ceiling(_gap_all(0.0)) == max(DEPTH_TIERS)


def test_ceiling_lenient_empty_is_deepest():
    # Lenient semantics on empty profile reproduces the old "all mastered" reading.
    assert reachable_ceiling({}, missing_is_mastered=True) == max(DEPTH_TIERS)


def test_ceiling_stuck_at_tier1():
    # tier-1 not consolidated (mastery 0.5 < 0.60 default) => ceiling is tier 1.
    assert reachable_ceiling(_gap_all(0.5)) == 1


def test_ceiling_frontier_is_first_unmastered_tier():
    # tier1 & tier2 mastered (>=0.70), tier3 & tier4 not => frontier (ceiling) = tier 3.
    gap = {}
    for a in DEPTH_TIERS[1]:
        gap[a] = 0.1   # mastery 0.9
    for a in DEPTH_TIERS[2]:
        gap[a] = 0.2   # mastery 0.8
    for a in DEPTH_TIERS[3]:
        gap[a] = 0.9   # mastery 0.1
    for a in DEPTH_TIERS[4]:
        gap[a] = 0.9
    assert reachable_ceiling(gap) == 3


def test_ceiling_threshold_configurable():
    # mastery 0.65 everywhere: with default 0.70 -> stuck at tier1; with 0.60 -> reaches deepest.
    gap = _gap_all(0.35)  # mastery 0.65
    assert reachable_ceiling(gap, threshold=0.70) == 1
    assert reachable_ceiling(gap, threshold=0.60) == max(DEPTH_TIERS)


def test_default_threshold_constant():
    assert MASTERY_THRESHOLD_DEFAULT == 0.60


# --- tier_overreach_factor ---------------------------------------------------------------------

def test_overreach_within_reach_is_one():
    assert tier_overreach_factor(2, 3) == 1.0   # tier 2 <= ceiling 3
    assert tier_overreach_factor(3, 3) == 1.0   # at the ceiling


def test_overreach_geometric_decay():
    assert tier_overreach_factor(4, 3, decay=0.3) == pytest.approx(0.3)   # 1 tier over
    assert tier_overreach_factor(4, 2, decay=0.3) == pytest.approx(0.09)  # 2 tiers over
    assert tier_overreach_factor(4, 1, decay=0.5) == pytest.approx(0.125) # 3 tiers over, decay .5


# --- ambition_gain with the gate ----------------------------------------------------------------

def test_ambition_no_ceiling_is_legacy():
    # No ceiling => unchanged behaviour: gap*depth. defeat_orc_mage is tier3 (depth 3), gap 1.0.
    p = _p([_T3])
    assert ambition_gain(p, {_T3: 1.0}) == pytest.approx(3.0)
    assert ambition_gain(p, {_T3: 1.0}, reachable_ceiling=None) == pytest.approx(3.0)


def test_ambition_gate_no_penalty_within_reach():
    # ceiling 3, achievement tier3 => reach factor 1.0 => same as legacy.
    p = _p([_T3])
    assert ambition_gain(p, {_T3: 1.0}, reachable_ceiling=3) == pytest.approx(3.0)


def test_ambition_gate_discounts_overreach():
    # ceiling 2, achievement tier3 (1 over) => 3.0 * 0.3 = 0.9.
    p = _p([_T3])
    assert ambition_gain(p, {_T3: 1.0}, reachable_ceiling=2, overreach_decay=0.3) == pytest.approx(0.9)
    # tier4 (2 over) => depth 4 * 1.0 * 0.3**2 = 0.36.
    p4 = _p([_T4])
    assert ambition_gain(p4, {_T4: 1.0}, reachable_ceiling=2, overreach_decay=0.3) == pytest.approx(0.36)


def test_ambition_gate_lets_feasible_win_over_ambitious():
    # THE core §10.9 fix: with the student stuck at tier2, an ambitious tier-3 level should NOT
    # out-score a feasible tier-2 level on the ambition term once the gate is on.
    ceiling = 2
    ambitious = _p([_T3])           # deep, tier3
    feasible = _p([_T2])            # learnable, tier2
    gap = {_T2: 1.0, _T3: 1.0}      # student fails both equally
    # WITHOUT gate: ambitious (depth3=3.0) beats feasible (depth2=2.0).
    assert ambition_gain(ambitious, gap) > ambition_gain(feasible, gap)
    # WITH gate at ceiling 2: ambitious 3.0*0.3=0.9 < feasible 2.0*1.0=2.0. Feasible wins.
    a_gated = ambition_gain(ambitious, gap, reachable_ceiling=ceiling)
    f_gated = ambition_gain(feasible, gap, reachable_ceiling=ceiling)
    assert f_gated > a_gated
    assert a_gated == pytest.approx(0.9)
    assert f_gated == pytest.approx(2.0)


def test_ambition_gate_still_modular_nonnegative():
    # Gate only scales down non-negative terms => still non-negative (modularity precondition).
    p = _p([_T1, _T2, _T3, _T4])
    v = ambition_gain(p, _gap_all(0.5), reachable_ceiling=1)
    assert v >= 0.0
