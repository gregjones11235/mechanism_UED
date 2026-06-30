"""Tests for GreedyTopKSelector: correctness, assembled-bid submodularity, and (1-1/e) bound."""

import itertools
import math
import random

import pytest

from auction.coverage import coverage
from auction.mock_proposals import random_pool, random_proposal
from auction.proposal import Proposal
from auction.selectors import (
    GreedyTopKSelector,
    SelectionContext,
    assembled_marginal_bid,
)


def _p(pid, achs, proposer="mock"):
    return Proposal(pid, proposer, "task_0", "d", "r", frozenset(achs))


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

def test_picks_complementary_not_redundant():
    # Greedy on Coverage should pick complementary proposals, not three copies of the best one.
    a = _p("a", {"defeat_archer"})                 # weight 8
    a_dup = _p("a2", {"defeat_archer"})            # same coverage -> redundant
    b = _p("b", {"defeat_necromancer"})            # weight 8, disjoint
    sel = GreedyTopKSelector().select([a, a_dup, b], k=2, context=SelectionContext())
    ids = {p.proposal_id for p in sel}
    assert ids == {"a", "b"} or ids == {"a2", "b"}  # never {a, a2}


def test_k_bounds():
    pool = [_p("a", {"collect_wood"}), _p("b", {"collect_stone"})]
    assert GreedyTopKSelector().select(pool, 0, SelectionContext()) == []
    assert len(GreedyTopKSelector().select(pool, 5, SelectionContext())) == 2  # clamped to pool
    assert GreedyTopKSelector().select([], 3, SelectionContext()) == []


def test_already_covered_changes_choice():
    a = _p("a", {"defeat_archer"})
    b = _p("b", {"collect_wood", "collect_stone"})  # weight 1+1=2 vs a's 8
    ctx_fresh = SelectionContext()
    # fresh: a (8) beats b (2)
    assert GreedyTopKSelector().select([a, b], 1, ctx_fresh)[0].proposal_id == "a"
    # if archive already covers defeat_archer, a gives 0 new -> b wins
    ctx = SelectionContext(already_covered=frozenset({"defeat_archer"}))
    assert GreedyTopKSelector().select([a, b], 1, ctx)[0].proposal_id == "b"


# ---------------------------------------------------------------------------
# Assembled-bid submodularity (the discipline: WHOLE bid, not just Coverage)
# ---------------------------------------------------------------------------

def test_assembled_bid_is_submodular():
    rng = random.Random(7)
    for _ in range(1000):
        pool = random_pool(rng, rng.randint(2, 8))
        # random modular signals
        ratings = {
            f"r{i}": {p.proposal_id: rng.random() for p in pool} for i in range(3)
        }
        from auction.craftax_achievements import ALL_ACHIEVEMENTS
        gap = {a: rng.random() for a in ALL_ACHIEVEMENTS}
        ctx = SelectionContext(cross_ratings=ratings, target_gap=gap)
        cut = rng.randint(0, len(pool))
        S, T = pool[:cut], pool
        x = random_proposal(rng, "x")
        gS = assembled_marginal_bid(x, S, ctx)
        gT = assembled_marginal_bid(x, T, ctx)
        assert gS >= gT - 1e-9, f"assembled bid not submodular: {gS} < {gT}"


# ---------------------------------------------------------------------------
# (1 - 1/e) guarantee: greedy >= 0.632 * brute-force-optimal k-subset coverage
# ---------------------------------------------------------------------------

def _brute_force_best_coverage(pool, k, ctx):
    best = 0.0
    for combo in itertools.combinations(pool, k):
        c = coverage(combo, ctx.weights, already_covered=ctx.already_covered)
        best = max(best, c)
    return best


def test_greedy_meets_one_minus_one_over_e_bound():
    rng = random.Random(11)
    ratio = 1.0 - 1.0 / math.e  # ~0.632
    ctx = SelectionContext()  # pure Coverage so brute force is well-defined
    for _ in range(300):
        n = rng.randint(3, 8)
        k = rng.randint(1, n)
        pool = random_pool(rng, n)
        greedy = GreedyTopKSelector().select(pool, k, ctx)
        greedy_cov = coverage(greedy, ctx.weights)
        opt_cov = _brute_force_best_coverage(pool, k, ctx)
        if opt_cov > 0:
            assert greedy_cov >= ratio * opt_cov - 1e-9, (
                f"(1-1/e) violated: greedy={greedy_cov} < {ratio}*opt={ratio*opt_cov}"
            )


def test_greedy_often_optimal_in_practice():
    # In practice greedy on coverage is usually exactly optimal; sanity that it's not far off.
    rng = random.Random(12)
    ctx = SelectionContext()
    hits = 0
    trials = 100
    for _ in range(trials):
        n = rng.randint(3, 6)
        k = rng.randint(1, n)
        pool = random_pool(rng, n)
        greedy = coverage(GreedyTopKSelector().select(pool, k, ctx), ctx.weights)
        opt = _brute_force_best_coverage(pool, k, ctx)
        if abs(greedy - opt) < 1e-9:
            hits += 1
    assert hits >= trials * 0.7  # greedy is exactly optimal in the large majority of cases
