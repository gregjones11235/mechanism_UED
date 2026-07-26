"""Auction (raw) + Budgeted Auction selectors.

Re-implements the documented auction semantics self-contained (legacy auction.py
blob ec351728 is not runtime-importable here). Each candidate's bid = its
aggregate utility = mean of all normalized role_scores (0 if none). AUCTION_RAW
ranks by bid and takes top-k. AUCTION_BUDGETED ranks by bid and admits greedily
highest-bid-first under a cumulative-cost cap (shortfall -> INSUFFICIENT, no
backfill). Critic policy is applied via the shared machinery.
"""
from __future__ import annotations

from d052.selectors.base import (
    SelectorSignals,
    select_budgeted,
    select_unbudgeted,
)
from d052.schemas.selector import SelectionResult, SelectorConfig


def _bid(sig) -> float:
    """Utility bid = mean of all normalized role_scores (0 if none)."""
    vals = list(sig.role_scores.values())
    return sum(float(v) for v in vals) / len(vals) if vals else 0.0


def select_auction_raw(config: SelectorConfig,
                       signals: SelectorSignals) -> SelectionResult:
    return select_unbudgeted(config, signals, _bid)


def select_auction_budgeted(config: SelectorConfig,
                            signals: SelectorSignals) -> SelectionResult:
    return select_budgeted(config, signals, _bid)
