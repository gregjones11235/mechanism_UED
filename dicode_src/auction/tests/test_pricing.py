"""Tests for PriceState (shadow prices, tatonnement, calibration)."""

import pytest

from auction.craftax_achievements import ALL_ACHIEVEMENTS, depth_of
from auction.pricing import PriceState


def test_default_prices_follow_depth():
    ps = PriceState()
    # deeper achievement should start more expensive
    assert ps.prices["defeat_archer"] > ps.prices["collect_wood"]  # depth 4 > depth 1
    assert all(p >= ps.price_floor for p in ps.prices.values())
    assert len(ps.prices) == 67


def test_as_weights_nonnegative():
    ps = PriceState()
    w = ps.as_weights()
    assert all(v >= 0 for v in w.values())  # required for Coverage submodularity


def test_tatonnement_raises_price_on_excess_demand():
    ps = PriceState()
    a = "collect_wood"
    p0 = ps.prices[a]
    # demand far above supply target -> price should rise
    demand = {x: 0.0 for x in ALL_ACHIEVEMENTS}
    demand[a] = 5.0
    ps.tatonnement(demand, supply_target=1.0, max_iters=5)
    assert ps.prices[a] > p0


def test_tatonnement_lowers_price_on_excess_supply():
    ps = PriceState()
    a = "defeat_archer"
    p0 = ps.prices[a]
    demand = {x: 1.0 for x in ALL_ACHIEVEMENTS}  # exactly target everywhere...
    demand[a] = 0.0  # ...except a is oversupplied (demand 0 < target 1)
    ps.tatonnement(demand, supply_target=1.0, max_iters=5)
    assert ps.prices[a] < p0


def test_tatonnement_converges_at_target():
    ps = PriceState()
    demand = {x: 1.0 for x in ALL_ACHIEVEMENTS}  # demand == supply_target everywhere
    cleared = ps.tatonnement(demand, supply_target=1.0, tol=1e-3)
    assert cleared is True  # zero excess -> immediate convergence


def test_tatonnement_can_report_nonconvergence():
    ps = PriceState(price_cap=1e9, price_floor=1e-9)
    # wildly unbalanced demand with tiny tolerance and few iters -> should not converge
    demand = {x: 0.0 for x in ALL_ACHIEVEMENTS}
    demand["collect_wood"] = 100.0
    cleared = ps.tatonnement(demand, supply_target=1.0, tol=1e-6, max_iters=2)
    assert cleared is False


def test_prices_clamped_to_band():
    ps = PriceState(price_floor=0.5, price_cap=4.0)
    demand = {x: 100.0 for x in ALL_ACHIEVEMENTS}  # blow prices up
    ps.tatonnement(demand, supply_target=1.0, max_iters=50)
    assert all(0.5 <= p <= 4.0 for p in ps.prices.values())


def test_calibrate_raises_price_on_realized_gain():
    ps = PriceState()
    a = "make_iron_pickaxe"
    p0 = ps.prices[a]
    ps.calibrate({a: 0.5})   # injecting levels covering a really helped the student
    assert ps.prices[a] > p0


def test_calibrate_lowers_price_on_no_gain():
    ps = PriceState()
    a = "collect_wood"
    p0 = ps.prices[a]
    ps.calibrate({a: -0.4})  # covering a produced no learning -> cheaper
    assert ps.prices[a] < p0


def test_unknown_achievement_rejected():
    ps = PriceState()
    with pytest.raises(ValueError):
        ps.tatonnement({"bogus": 1.0})
    with pytest.raises(ValueError):
        ps.calibrate({"bogus": 0.1})
