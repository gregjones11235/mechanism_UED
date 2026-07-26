"""Unified selector interface (S0/S1/S2 baseline ladder, Soft/Budgeted Copeland,
Auction raw/budgeted) with shared config+result, deterministic bit-identical
replay, and explicit critic-policy consumption (hard_veto / soft_penalty /
score_only).

Single entry point: :func:`select`. Legacy aggregation.py / auction.py semantics
are re-implemented self-contained (see base.py docstring); exact numeric parity
with legacy is NOT claimed.
"""
from d052.selectors.auction import select_auction_budgeted, select_auction_raw
from d052.selectors.base import (
    CandidateSignals,
    SelectorError,
    SelectorSignals,
    mean_role_scores,
    select_budgeted,
    select_unbudgeted,
)
from d052.selectors.baseline import (
    select_s0_baseline,
    select_s1_three_role,
    select_s2_four_role,
)
from d052.selectors.copeland import (
    copeland_scores,
    select_budgeted_soft_copeland,
    select_soft_copeland,
)
from d052.selectors.interface import select

__all__ = [
    "CandidateSignals",
    "SelectorError",
    "SelectorSignals",
    "copeland_scores",
    "mean_role_scores",
    "select",
    "select_auction_budgeted",
    "select_auction_raw",
    "select_budgeted",
    "select_budgeted_soft_copeland",
    "select_s0_baseline",
    "select_s1_three_role",
    "select_s2_four_role",
    "select_soft_copeland",
    "select_unbudgeted",
]
