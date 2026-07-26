"""The single unified selector entry point.

``select(config, pool, signals)`` is the ONLY way to run a selection. It:
  1. enforces the shared-frozen-pool invariant -- ``signals.pool_hash`` MUST equal
     ``pool.pool_hash`` and the signals MUST cover exactly the pool's candidates
     (one judgment set per candidate, no silent subset);
  2. dispatches to the configured selector family;
  3. returns one audit-grade, replayable ``SelectionResult``.

All seven selector types share this one contract, so their outputs are directly
comparable and bit-for-bit replayable from (config, pool, signals).
"""
from __future__ import annotations

from d052.schemas.candidate import CandidatePool
from d052.schemas.selector import SelectionResult, SelectorConfig, SelectorType
from d052.selectors.auction import select_auction_budgeted, select_auction_raw
from d052.selectors.base import SelectorError, SelectorSignals
from d052.selectors.baseline import (
    select_s0_baseline,
    select_s1_three_role,
    select_s2_four_role,
)
from d052.selectors.copeland import (
    select_budgeted_soft_copeland,
    select_soft_copeland,
)

_DISPATCH = {
    SelectorType.S0_CANONICAL_BASELINE: select_s0_baseline,
    SelectorType.S1_THREE_ROLE: select_s1_three_role,
    SelectorType.S2_FOUR_ROLE_MODELER: select_s2_four_role,
    SelectorType.SOFT_COPELAND: select_soft_copeland,
    SelectorType.BUDGETED_SOFT_COPELAND: select_budgeted_soft_copeland,
    SelectorType.AUCTION_RAW: select_auction_raw,
    SelectorType.AUCTION_BUDGETED: select_auction_budgeted,
}


def select(config: SelectorConfig, pool: CandidatePool,
           signals: SelectorSignals) -> SelectionResult:
    """Run one deterministic selection over the shared frozen pool."""
    # 1. shared-frozen-pool invariant (hard fail; never silently reconcile)
    if signals.pool_hash != pool.pool_hash:
        raise SelectorError(
            SelectorError.POOL_MISMATCH,
            f"signals pool_hash {signals.pool_hash} != pool.pool_hash "
            f"{pool.pool_hash}; all selectors must consume ONE shared frozen pool")
    pool_ids = {c.task_id for c in pool.candidates}
    signal_ids = {c.candidate_id for c in signals.candidates}
    if signal_ids != pool_ids:
        missing = sorted(pool_ids - signal_ids)
        extra = sorted(signal_ids - pool_ids)
        raise SelectorError(
            SelectorError.SIGNAL_POOL_MISMATCH,
            f"signals must cover exactly the pool candidates; "
            f"missing={missing} extra={extra}")
    # 2. dispatch
    fn = _DISPATCH.get(config.selector)
    if fn is None:
        raise SelectorError(
            SelectorError.UNKNOWN_SELECTOR,
            f"no dispatcher for selector {config.selector!r}")
    return fn(config, signals)
