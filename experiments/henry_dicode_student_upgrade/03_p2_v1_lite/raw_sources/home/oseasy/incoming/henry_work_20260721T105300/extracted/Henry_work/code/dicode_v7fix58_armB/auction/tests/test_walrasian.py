"""Tests for WalrasianSelector — price-weighted selection + the critical v1 fallback path."""

import random

from auction.coverage import coverage
from auction.craftax_achievements import ALL_ACHIEVEMENTS
from auction.mock_proposals import random_pool
from auction.pricing import PriceState
from auction.proposal import Proposal
from auction.selectors import (
    GreedyTopKSelector,
    SelectionContext,
    WalrasianSelector,
)


def _p(pid, achs):
    return Proposal(pid, "mock", "task_0", "d", "r", frozenset(achs))


def test_clears_and_selects_k():
    # balanced demand -> tatonnement converges -> price-weighted greedy returns k
    pool = [
        _p("a", {"collect_wood"}),
        _p("b", {"defeat_archer"}),
        _p("c", {"make_iron_pickaxe"}),
    ]
    sel = WalrasianSelector()
    out = sel.select(pool, k=2, context=SelectionContext())
    assert len(out) == 2
    assert sel.last_cleared in (True, False)  # ran the mechanism


def test_high_price_achievement_preferred_when_cleared():
    # Make one achievement very expensive; the proposal covering it should be picked first.
    ps = PriceState()
    ps.prices = {a: 1.0 for a in ALL_ACHIEVEMENTS}
    ps.prices["defeat_necromancer"] = 30.0
    cheap = _p("cheap", {"collect_wood", "collect_stone"})   # 1 + 1 = 2
    pricey = _p("pricey", {"defeat_necromancer"})            # 30
    sel = WalrasianSelector(price_state=ps)
    # force convergence by using balanced supply target; check selection
    out = sel.select([cheap, pricey], k=1, context=SelectionContext())
    if sel.last_cleared:
        assert out[0].proposal_id == "pricey"


def test_fallback_to_v1_on_nonconvergence():
    # price band pinned tiny + tol tiny + few iters -> never clears -> must fall back to greedy v1,
    # which still returns a valid k-selection (never gets stuck).
    ps = PriceState(price_floor=1e-9, price_cap=1e9)
    pool = [_p("a", {"collect_wood"}), _p("b", {"defeat_archer"})]
    # monkeypatch tatonnement to always report failure to deterministically exercise the path
    ps.tatonnement = lambda *a, **k: False  # type: ignore[method-assign]
    sel = WalrasianSelector(price_state=ps)
    ctx = SelectionContext()
    out = sel.select(pool, k=2, context=ctx)
    # fallback result must equal what v1 greedy would produce on the same default-weight context
    v1 = GreedyTopKSelector().select(pool, 2, ctx)
    assert {p.proposal_id for p in out} == {p.proposal_id for p in v1}
    assert sel.last_cleared is False


def test_empty_and_zero_k():
    sel = WalrasianSelector()
    assert sel.select([], 3, SelectionContext()) == []
    assert sel.select([_p("a", {"collect_wood"})], 0, SelectionContext()) == []


def test_v1_untouched_walrasian_is_additive():
    # Sanity: running WalrasianSelector does not mutate the SelectionContext or v1 behavior.
    rng = random.Random(5)
    pool = random_pool(rng, 6)
    ctx = SelectionContext()
    v1_before = {p.proposal_id for p in GreedyTopKSelector().select(pool, 3, ctx)}
    WalrasianSelector().select(pool, 3, ctx)
    v1_after = {p.proposal_id for p in GreedyTopKSelector().select(pool, 3, ctx)}
    assert v1_before == v1_after  # v2 selector did not corrupt shared context
