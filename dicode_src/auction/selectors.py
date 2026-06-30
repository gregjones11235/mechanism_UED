"""Selectors — turn a candidate pool into k chosen proposals. v1 = greedy submodular top-k.

Unified interface so v1 / v2 / fallback are the same signature, different implementation;
plugging into DiCode's evolve_mastered is then a one-line selector swap (§7.4).

    select(proposals, k, context) -> list[Proposal]

The bid combined here is:

    bid_marginal(x | selected) = w_cov * coverage_gain(x | selected)      # SUBMODULAR (set-dependent)
                               + w_end * endorsement[x]                   # MODULAR (per-proposal)
                               + w_amb * ambition_gain(x)                 # MODULAR (per-proposal)

DISCIPLINE (v1_experiment.md §7.5): every bid term is submodular or modular, NEVER supermodular.
submodular + non-negative modular = submodular, so the WHOLE marginal bid is submodular and greedy
top-k inherits the Nemhauser (1 - 1/e) guarantee. Greedy picks, at each step, the candidate with
the largest *marginal* bid given what's already selected (Coverage's marginal shrinks as the
selected set grows; the modular terms are constant) — this is exactly the standard submodular
greedy. The submodularity of the assembled bid is asserted in tests, not just claimed here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from .ambition import ambition_gain
from .coverage import DEFAULT_WEIGHTS, coverage_gain
from .craftax_achievements import ALL_ACHIEVEMENTS
from .endorsement import CrossRatings, endorsement_scores
from .pricing import PriceState
from .proposal import Proposal


@dataclass(frozen=True)
class SelectionContext:
    """Everything a selector needs beyond the pool itself. All optional → graceful degradation.

    weights:        per-achievement coverage weights (v2 passes Walrasian prices here).
    already_covered: achievements the archive already covers (only credit NEW coverage).
    cross_ratings:  Proposer cross-rating matrix for Endorsement (None → endorsement term = 0).
    target_gap:     {achievement: 1 - student_SR_on_target} for AmbitionGain (None → term = 0).
    w_cov/w_end/w_amb: bid term weights (dev default 1/1/1; sensitivity ablation later).
    """

    weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    already_covered: frozenset[str] = frozenset()
    cross_ratings: CrossRatings | None = None
    target_gap: Mapping[str, float] | None = None
    w_cov: float = 1.0
    w_end: float = 1.0
    w_amb: float = 1.0


class Selector:
    """Abstract selector. Subclasses implement select()."""

    def select(self, proposals: list[Proposal], k: int, context: SelectionContext) -> list[Proposal]:
        raise NotImplementedError


class GreedyTopKSelector(Selector):
    """v1: greedy submodular maximization of the assembled bid. (1-1/e)-optimal on Coverage.

    Modular terms (endorsement, ambition) are precomputed once per proposal; the submodular
    Coverage term's marginal is recomputed against the growing selected set each step.
    """

    def select(self, proposals: list[Proposal], k: int, context: SelectionContext) -> list[Proposal]:
        if k <= 0 or not proposals:
            return []
        k = min(k, len(proposals))

        # Precompute the modular (per-proposal, set-independent) part of the bid.
        endorse = (
            endorsement_scores(proposals, context.cross_ratings)
            if context.cross_ratings is not None
            else {p.proposal_id: 0.0 for p in proposals}
        )
        modular = {}
        for p in proposals:
            amb = (
                ambition_gain(p, context.target_gap)
                if context.target_gap is not None
                else 0.0
            )
            modular[p.proposal_id] = context.w_end * endorse[p.proposal_id] + context.w_amb * amb

        selected: list[Proposal] = []
        remaining = list(proposals)
        while remaining and len(selected) < k:
            best, best_bid = None, None
            for cand in remaining:
                cov = coverage_gain(
                    cand, selected, context.weights, already_covered=context.already_covered
                )
                bid = context.w_cov * cov + modular[cand.proposal_id]
                if best_bid is None or bid > best_bid:
                    best, best_bid = cand, bid
            assert best is not None
            selected.append(best)
            remaining.remove(best)
        return selected


class WalrasianSelector(Selector):
    """v2 main trunk: shadow prices guide selection (and feed back to production).

    Mechanism (§7.7):
      1. Run tatonnement on the current pool's coverage demand to clear shadow prices.
      2. If it CONVERGES: select greedily with PRICES as the Coverage weights — scarce/deep
         achievements (high price) are preferred, saturated ones (low price) deprioritized.
         This is the same submodular greedy as v1, just with price-weighted Coverage, so the
         (1-1/e) guarantee still holds (non-negative prices => weighted coverage submodular).
      3. If it does NOT converge (no market clearing): FALL BACK to v1 GreedyTopKSelector on the
         default objective weights. v1 always runs => v2 can never get stuck (§7.7 safety net).

    The price feedback to Proposers (next round's generation context) and post-hoc calibration
    live in PriceState; this selector only consumes prices for selection and reports clearing.
    """

    def __init__(self, price_state: PriceState | None = None, *, supply_target: float = 1.0):
        self.price_state = price_state if price_state is not None else PriceState()
        self.supply_target = supply_target
        self.last_cleared: bool | None = None  # introspection: did the last select() clear?

    def _coverage_demand(self, proposals: list[Proposal]) -> dict[str, float]:
        """Demand per achievement = how many proposals in the pool cover it."""
        demand = {a: 0.0 for a in ALL_ACHIEVEMENTS}
        for p in proposals:
            for a in p.achievements:
                demand[a] += 1.0
        return demand

    def select(self, proposals: list[Proposal], k: int, context: SelectionContext) -> list[Proposal]:
        if k <= 0 or not proposals:
            self.last_cleared = None
            return []

        demand = self._coverage_demand(proposals)
        cleared = self.price_state.tatonnement(demand, self.supply_target)
        self.last_cleared = cleared

        if not cleared:
            # §7.7 fallback: market did not clear -> retreat to v1 objective top-k.
            return GreedyTopKSelector().select(proposals, k, context)

        # Cleared: select with prices as Coverage weights (everything else from context unchanged).
        priced_ctx = SelectionContext(
            weights=self.price_state.as_weights(),
            already_covered=context.already_covered,
            cross_ratings=context.cross_ratings,
            target_gap=context.target_gap,
            w_cov=context.w_cov,
            w_end=context.w_end,
            w_amb=context.w_amb,
        )
        return GreedyTopKSelector().select(proposals, k, priced_ctx)


def assembled_marginal_bid(
    candidate: Proposal,
    selected: list[Proposal],
    context: SelectionContext,
) -> float:
    """The full marginal bid of adding ``candidate`` to ``selected`` — exposed for testing
    submodularity of the ASSEMBLED bid (not just Coverage)."""
    cov = coverage_gain(
        candidate, selected, context.weights, already_covered=context.already_covered
    )
    endorse = (
        endorsement_scores([candidate] + selected, context.cross_ratings).get(
            candidate.proposal_id, 0.0
        )
        if context.cross_ratings is not None
        else 0.0
    )
    amb = ambition_gain(candidate, context.target_gap) if context.target_gap is not None else 0.0
    return context.w_cov * cov + context.w_end * endorse + context.w_amb * amb
