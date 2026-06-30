"""Shadow prices for v2 — the "price guides production" mechanism (v1_experiment.md §7.6/§7.7).

Each achievement carries a non-negative shadow price. Scarce achievements (deep, no-one-covers)
get HIGH price → worth producing; saturated achievements (everyone covers, student already aces)
get LOW price → stop producing. These prices feed back to Proposers as the next round's generation
context AND are passed as the Coverage `weights` (zero rework: v2 reuses v1's Coverage with prices
as weights).

Two update paths:
  (1) tatonnement (within-round, demand-driven): raise price where demand < supply target is unmet,
      lower where over-supplied. Bounded iteration; if it does not converge we report that so the
      selector can FALL BACK to v1 (§7.7: "Walrasian may not clear → retreat to objective top-k").
  (2) post-hoc calibration (cross-round, ground-truth-anchored): after levels are injected and the
      student trains, the REAL achievement success-rate improvement (wandb achievement_srs, §2.1)
      pulls prices toward truth. This is the objective anchor that makes self-reported bids unable
      to drift from reality long-term (§7.7 anti-manipulation).

Non-negativity is enforced (prices used as Coverage weights must be >= 0 to keep submodularity).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from .craftax_achievements import ALL_ACHIEVEMENTS, depth_of


def _default_prices() -> dict[str, float]:
    # Initial prices = depth-tier prior (same shape as v1 DEFAULT_WEIGHTS): deeper == scarcer.
    tier_price = {1: 1.0, 2: 2.0, 3: 4.0, 4: 8.0}
    return {a: float(tier_price[depth_of(a)]) for a in ALL_ACHIEVEMENTS}


@dataclass
class PriceState:
    """Mutable per-achievement shadow prices, persisted across generation cycles.

    prices: {achievement: price >= 0}. Used directly as Coverage weights in v2.
    price_floor / price_cap: keep prices in a sane band (avoid 0 or runaway).
    """

    prices: dict[str, float] = field(default_factory=_default_prices)
    price_floor: float = 0.1
    price_cap: float = 32.0

    def __post_init__(self) -> None:
        bad = set(self.prices) - ALL_ACHIEVEMENTS
        if bad:
            raise ValueError(f"prices reference unknown achievements: {sorted(bad)}")
        # ensure every achievement has a price (missing -> floor) and clamp
        for a in ALL_ACHIEVEMENTS:
            self.prices.setdefault(a, self.price_floor)
        self._clamp_all()

    def _clamp_all(self) -> None:
        for a, p in self.prices.items():
            self.prices[a] = min(self.price_cap, max(self.price_floor, p))

    def as_weights(self) -> dict[str, float]:
        """Return prices as a Coverage-weight mapping (non-negative by construction)."""
        return dict(self.prices)

    # --- (1) within-round tatonnement -------------------------------------------------
    def tatonnement(
        self,
        demand: Mapping[str, float],
        supply_target: float = 1.0,
        *,
        step: float = 0.5,
        max_iters: int = 50,
        tol: float = 1e-3,
    ) -> bool:
        """Adjust prices toward market clearing given per-achievement ``demand``.

        Excess demand (demand > supply_target) raises the price; excess supply lowers it
        (multiplicative tatonnement). Returns True if it converged (max |excess| < tol),
        False otherwise — caller uses False to trigger v1 fallback (§7.7).

        ``demand[a]`` = how many selected/proposed levels cover achievement a this round
        (a proxy for market demand). ``supply_target`` = the desired coverage multiplicity.
        """
        bad = set(demand) - ALL_ACHIEVEMENTS
        if bad:
            raise ValueError(f"demand references unknown achievements: {sorted(bad)}")
        for _ in range(max_iters):
            max_excess = 0.0
            for a in ALL_ACHIEVEMENTS:
                d = float(demand.get(a, 0.0))
                excess = d - supply_target  # >0: under-priced (too wanted); <0: over-supplied
                max_excess = max(max_excess, abs(excess))
                # multiplicative update keeps prices positive
                self.prices[a] *= (1.0 + step * excess / max(supply_target, 1e-6))
            self._clamp_all()
            if max_excess < tol:
                return True
        return False

    # --- (2) cross-round post-hoc calibration -----------------------------------------
    def calibrate(
        self,
        realized_sr_gain: Mapping[str, float],
        *,
        lr: float = 0.3,
    ) -> None:
        """Pull prices toward ground truth using REALIZED achievement SR improvement.

        If injecting levels that covered achievement a actually moved the student's SR on a
        a lot (high realized gain), a was a GOOD investment → its price should rise (still scarce
        and learnable). If covering a produced no SR gain (already mastered or unlearnable),
        lower its price. This is the objective anchor (§7.7) that self-reports cannot fake.

        realized_sr_gain[a] in [-1, 1] roughly (delta success rate). Missing -> no update.
        """
        bad = set(realized_sr_gain) - ALL_ACHIEVEMENTS
        if bad:
            raise ValueError(f"realized_sr_gain references unknown achievements: {sorted(bad)}")
        for a, g in realized_sr_gain.items():
            # price *= (1 + lr * g): positive realized gain bumps price up, negative pulls down.
            self.prices[a] *= (1.0 + lr * float(g))
        self._clamp_all()
