"""Tests for the constrained contest sub-module (price-capped all-pay reward gradient)."""

import pytest

from auction.contest import (
    ContestConfig,
    price_capped_contest_rewards,
    rank_rewards,
)


def test_rank_rewards_geometric_decay():
    cfg = ContestConfig(rank_decay=0.5, base_reward=1.0)
    r = rank_rewards(3, cfg)
    assert r == pytest.approx([1.0, 0.5, 0.25])


def test_flat_schedule_is_contest_off():
    # rank_decay = 1.0 => flat => the clean "contest off" ablation baseline.
    r = rank_rewards(4, ContestConfig(rank_decay=1.0, base_reward=2.0))
    assert r == pytest.approx([2.0, 2.0, 2.0, 2.0])


def test_steeper_decay_raises_top_slot_marginal():
    flat = rank_rewards(3, ContestConfig(rank_decay=1.0))
    steep = rank_rewards(3, ContestConfig(rank_decay=0.5))
    # steeper schedule makes rank gaps larger => more incentive to rank high
    gap_flat = flat[0] - flat[1]
    gap_steep = steep[0] - steep[1]
    assert gap_steep > gap_flat


def test_config_validation():
    with pytest.raises(ValueError):
        ContestConfig(rank_decay=0.0)
    with pytest.raises(ValueError):
        ContestConfig(rank_decay=1.5)
    with pytest.raises(ValueError):
        ContestConfig(base_reward=-1.0)


def test_price_cap_dampens_but_never_amplifies():
    # THE anti-bluff bound: reward[i] <= rank_reward[i] for every slot, always.
    cfg = ContestConfig(rank_decay=0.8, base_reward=10.0)
    prices = {"collect_wood": 1.0, "defeat_archer": 8.0}
    ranked = [
        frozenset({"defeat_archer"}),     # cap 8
        frozenset({"collect_wood"}),      # cap 1
        frozenset(),                      # cap 0 (empty / bluffed proposal)
    ]
    rewards = price_capped_contest_rewards(ranked, prices, cfg)
    schedule = rank_rewards(3, cfg)
    for got, sched in zip(rewards, schedule):
        assert got <= sched + 1e-9  # never amplified beyond the contest schedule


def test_empty_proposal_gets_zero_reward():
    # A bluffed proposal covering nothing priced is capped to 0 regardless of its rank.
    cfg = ContestConfig(rank_decay=0.9, base_reward=10.0)
    ranked = [frozenset()]  # top slot but empty
    rewards = price_capped_contest_rewards(ranked, {"collect_wood": 1.0}, cfg)
    assert rewards == [0.0]


def test_high_price_achievement_keeps_full_contest_reward():
    # When the price cap exceeds the schedule, the contest reward is unaffected (price not binding).
    cfg = ContestConfig(rank_decay=0.5, base_reward=2.0)  # top slot reward 2.0
    prices = {"defeat_archer": 100.0}                      # cap 100 >> 2.0
    ranked = [frozenset({"defeat_archer"})]
    rewards = price_capped_contest_rewards(ranked, prices, cfg)
    assert rewards[0] == pytest.approx(2.0)  # full schedule reward, price not binding
