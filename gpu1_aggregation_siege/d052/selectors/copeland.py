"""Soft Copeland + Budgeted Soft Copeland selectors.

Re-implements the documented Copeland semantics self-contained (legacy
aggregation.py blob 92a7e8b6 is not runtime-importable here). Composite =
pairwise Copeland score over each candidate's aggregated normalized role
strength: for every unordered pair, the stronger candidate scores +1, the weaker
0, a tie +0.5 each. The full-pairwise sum is order-independent, hence
deterministic. Critic policy is applied on top via the shared machinery
(hard_veto pre-filters, soft_penalty subtracts the normalized critic penalty).
Budgeted variant admits greedily by Copeland score under a cumulative-cost cap.
"""
from __future__ import annotations

from typing import Dict

from d052.selectors.base import (
    SelectorSignals,
    select_budgeted,
    select_unbudgeted,
)
from d052.schemas.selector import SelectionResult, SelectorConfig


def _strength(sig) -> float:
    """Aggregate normalized role strength = mean of all role_scores (0 if none)."""
    vals = list(sig.role_scores.values())
    return sum(float(v) for v in vals) / len(vals) if vals else 0.0


def copeland_scores(signals: SelectorSignals) -> Dict[str, float]:
    """Deterministic full-pairwise Copeland score per candidate_id."""
    by = signals.by_id()
    ids = sorted(by)  # deterministic iteration; result is order-independent anyway
    score = {cid: 0.0 for cid in ids}
    for i, a in enumerate(ids):
        sa = _strength(by[a])
        for b in ids[i + 1:]:
            sb = _strength(by[b])
            if sa > sb:
                score[a] += 1.0
            elif sa < sb:
                score[b] += 1.0
            else:
                score[a] += 0.5
                score[b] += 0.5
    return score


def select_soft_copeland(config: SelectorConfig,
                         signals: SelectorSignals) -> SelectionResult:
    cs = copeland_scores(signals)
    return select_unbudgeted(config, signals, lambda sig: cs[sig.candidate_id])


def select_budgeted_soft_copeland(config: SelectorConfig,
                                  signals: SelectorSignals) -> SelectionResult:
    cs = copeland_scores(signals)
    return select_budgeted(config, signals, lambda sig: cs[sig.candidate_id])
