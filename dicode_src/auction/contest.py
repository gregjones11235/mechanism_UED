"""Constrained contest sub-module for v2 (§7.7) — all-pay/Tullock effort incentives, price-capped.

WHY contest at all: in this setting every Proposer pays the generation cost (LLM call + reasoning)
whether or not it wins — that is the defining structure of an ALL-PAY auction. Contest theory then
tells us how the reward gradient (1st-place vs k-th-place treatment) shapes equilibrium EFFORT
(mapped to proposal quality/ambition). We use it ONLY to set the reward gradient for selected
slots, NOT to replace Walrasian selection.

WHY constrained (the §7.7 risk fix): a steep winner-take-all gradient maximizes effort but also
maximizes the incentive to BLUFF (over-claim coverage to grab the top slot), and lets one
overconfident/dishonest FM dominate. So we CAP each contest reward by the achievement's Walrasian
shadow price: the contest can sharpen incentives WITHIN what the price layer already deems valuable,
but cannot inflate a proposal's reward beyond the objective price ceiling. This grafts contest's
"drive effort" benefit while the price layer's objective anchor caps the bluff incentive.

This module is PURE incentive accounting (no LLM, no selection mutation). It is consumed downstream
when shaping per-slot rewards / feedback to Proposers; selection itself stays with WalrasianSelector.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ContestConfig:
    """Reward-gradient knobs.

    rank_decay: geometric decay of reward by rank (1.0 = flat/no contest; <1.0 = steeper, more
        winner-take-all). Steeper => more effort but more bluff incentive (capped below).
    base_reward: reward scale for the top slot before rank decay and price cap.
    """

    rank_decay: float = 0.7
    base_reward: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 < self.rank_decay <= 1.0):
            raise ValueError(f"rank_decay must be in (0,1], got {self.rank_decay}")
        if self.base_reward < 0.0:
            raise ValueError("base_reward must be >= 0")


def rank_rewards(num_slots: int, config: ContestConfig) -> list[float]:
    """Geometric reward-by-rank schedule: reward[r] = base * rank_decay**r for r = 0..num_slots-1.

    rank_decay == 1.0 reproduces a flat schedule (effectively "no contest"), which is the clean
    ablation baseline (contest off). The all-pay equilibrium-effort intuition: steeper schedule
    (smaller rank_decay) raises the marginal value of being ranked higher => more effort.
    """
    if num_slots < 0:
        raise ValueError("num_slots must be >= 0")
    return [config.base_reward * (config.rank_decay ** r) for r in range(num_slots)]


def price_capped_contest_rewards(
    ranked_proposal_achievements: Sequence[frozenset[str]],
    prices: Mapping[str, float],
    config: ContestConfig,
) -> list[float]:
    """Per-slot contest reward, CAPPED by the max shadow price of the proposal's achievements.

    Args:
        ranked_proposal_achievements: the selected proposals' achievement sets, IN RANK ORDER
            (rank 0 = top slot from selection).
        prices: per-achievement shadow prices (the objective ceiling, from PriceState).
        config: reward-gradient knobs.

    Returns:
        reward[i] = min(rank_reward[i], price_cap_i), where price_cap_i = max price among
        proposal i's achievements (0 if it covers nothing priced). The cap is what prevents a
        steep contest gradient from rewarding a bluffed/empty proposal beyond its objective value.

    Property guaranteed (tested): reward[i] <= rank_reward[i] always (contest can only be DAMPENED
    by the price cap, never amplified beyond the objective price) — this is the anti-bluff bound.
    """
    schedule = rank_rewards(len(ranked_proposal_achievements), config)
    out: list[float] = []
    for i, achs in enumerate(ranked_proposal_achievements):
        cap = max((prices.get(a, 0.0) for a in achs), default=0.0)
        out.append(min(schedule[i], cap))
    return out
