"""Tests for weighted Coverage, including the SUBMODULARITY theorem as an executable assertion.

This is the machine-verified instance of v1_experiment.md §7.5 point 1: Coverage is monotone
and submodular, which is the precondition for the (1-1/e) greedy guarantee backing v1 top-k.
"""

import random

import pytest

from auction.coverage import (
    DEFAULT_WEIGHTS,
    coverage,
    coverage_gain,
    covered_set,
)
from auction.craftax_achievements import ALL_ACHIEVEMENTS
from auction.mock_proposals import random_pool, random_proposal
from auction.proposal import Proposal


def _p(pid, achs):
    return Proposal(pid, "mock", "task_0", "d", "r", frozenset(achs))


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

def test_coverage_is_union_weight():
    a = _p("a", {"collect_wood"})          # tier 1 -> weight 1
    b = _p("b", {"defeat_archer"})          # tier 4 -> weight 8
    assert coverage([a]) == 1.0
    assert coverage([b]) == 8.0
    assert coverage([a, b]) == 9.0


def test_overlap_not_double_counted():
    a = _p("a", {"collect_wood", "collect_stone"})
    b = _p("b", {"collect_stone", "defeat_archer"})  # collect_stone overlaps
    # union = {collect_wood(1), collect_stone(1), defeat_archer(8)} = 10
    assert coverage([a, b]) == 10.0


def test_already_covered_discounts():
    a = _p("a", {"collect_wood", "defeat_archer"})
    # archive already has defeat_archer -> only collect_wood credited
    assert coverage([a], already_covered=frozenset({"defeat_archer"})) == 1.0


def test_empty():
    assert coverage([]) == 0.0
    assert covered_set([]) == frozenset()


# ---------------------------------------------------------------------------
# MONOTONICITY:  S subseteq T  =>  Coverage(S) <= Coverage(T)
# ---------------------------------------------------------------------------

def test_monotonicity_random():
    rng = random.Random(0)
    for _ in range(200):
        pool = random_pool(rng, rng.randint(1, 8))
        k = rng.randint(0, len(pool))
        S = pool[:k]
        T = pool  # S subseteq T
        assert coverage(S) <= coverage(T) + 1e-9


# ---------------------------------------------------------------------------
# SUBMODULARITY (the theorem):  for S subseteq T and x not in T,
#     gain(x | S) >= gain(x | T)
# ---------------------------------------------------------------------------

def test_submodularity_random_weighted():
    rng = random.Random(1)
    for _ in range(2000):
        pool = random_pool(rng, rng.randint(2, 10))
        # split into S subseteq T
        cut = rng.randint(0, len(pool))
        T = pool
        S = pool[:cut]  # S subseteq T (prefix)
        x = random_proposal(rng, "x")
        gS = coverage_gain(x, S)
        gT = coverage_gain(x, T)
        assert gS >= gT - 1e-9, f"submodularity violated: gain|S={gS} < gain|T={gT}"


def test_submodularity_holds_for_arbitrary_nonneg_weights():
    # Submodularity must hold for ANY non-negative weight vector, not just the default.
    rng = random.Random(2)
    for _ in range(500):
        weights = {a: rng.random() * 10 for a in ALL_ACHIEVEMENTS}
        pool = random_pool(rng, rng.randint(2, 8))
        cut = rng.randint(0, len(pool))
        S, T = pool[:cut], pool
        x = random_proposal(rng, "x")
        assert coverage_gain(x, S, weights) >= coverage_gain(x, T, weights) - 1e-9


def test_gain_consistent_with_coverage_diff():
    rng = random.Random(3)
    for _ in range(200):
        S = random_pool(rng, rng.randint(0, 5))
        x = random_proposal(rng, "x")
        expected = coverage(S + [x]) - coverage(S)
        assert abs(coverage_gain(x, S) - expected) < 1e-9


# ---------------------------------------------------------------------------
# REVERSE CHECK: prove the test can actually CATCH a non-submodular function.
# A SUPERMODULAR coverage (pairwise bonus for co-covering) must violate the inequality,
# so this guards against a vacuous test that would pass on anything.
# ---------------------------------------------------------------------------

def _supermodular_coverage(proposals, bonus=5.0):
    """Deliberately NON-submodular: adds a bonus that grows with set size (increasing returns)."""
    covered = covered_set(proposals)
    base = sum(DEFAULT_WEIGHTS[a] for a in covered)
    # synergy term: rewards having MORE proposals together -> increasing marginal returns
    n = sum(1 for _ in proposals)
    return base + bonus * n * n  # n^2 is strictly supermodular in count


def _super_gain(x, selected, bonus=5.0):
    return _supermodular_coverage(list(selected) + [x], bonus) - _supermodular_coverage(
        list(selected), bonus
    )


def test_reverse_supermodular_is_caught():
    """The supermodular function MUST violate gain(x|S) >= gain(x|T); confirm we detect it."""
    rng = random.Random(4)
    violated = False
    for _ in range(100):
        pool = random_pool(rng, 6)
        S, T = pool[:1], pool  # S subseteq T, T strictly larger
        x = random_proposal(rng, "x")
        if _super_gain(x, S) < _super_gain(x, T) - 1e-9:
            violated = True
            break
    assert violated, "reverse check failed: supermodular fn did not violate submodularity — test is vacuous"
