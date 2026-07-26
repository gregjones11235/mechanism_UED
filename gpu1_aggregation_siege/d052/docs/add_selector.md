# How to add a selector

All seven selectors share ONE config (`SelectorConfig`), ONE result
(`SelectionResult`), and ONE entry point (`d052.selectors.select`). To add a
selector family you supply a deterministic scoring function and register it; the
shared machinery handles critic policy, eligibility, the no-backfill top-k, the
budget cap, and the result hash.

## Selector types today

`S0_CANONICAL_BASELINE` (no-LLM content order) · `S1_THREE_ROLE` ·
`S2_FOUR_ROLE_MODELER` · `SOFT_COPELAND` · `BUDGETED_SOFT_COPELAND` ·
`AUCTION_RAW` · `AUCTION_BUDGETED`.

## Steps

1. **Enum** — add the type to `SelectorType` in `d052/schemas/selector.py`. If it
   is budgeted, add it to `_BUDGETED` so `SelectorConfig` requires `budget`
   (and forbids `budget` on non-budgeted selectors).
2. **Config rules** — encode any role requirements in `SelectorConfig._validate_config`
   (S0 forbids roles; S1/S2 require roles).
3. **Scoring function** — write a module (e.g. `d052/selectors/myfamily.py`) that
   returns a family-specific BASE composite per candidate. Reuse:
   - `select_unbudgeted(config, signals, base_fn)` for top-k selectors, or
   - `select_budgeted(config, signals, base_fn)` for greedy-highest-first under a
     cumulative-cost cap (uses `CandidateSignals.cost`).
   `base_fn(sig: CandidateSignals) -> float` must be deterministic and depend only
   on the signals (e.g. `mean_role_scores`, `copeland_scores`).
4. **Critic policy** — do NOT handle the critic yourself; the shared
   `composite_score` applies `hard_veto` (pre-filter) / `soft_penalty` (subtract
   normalized penalty) / `score_only` (ignore) on top of your base composite.
5. **Register** — add the dispatch entry in `d052/selectors/interface.py:_DISPATCH`
   and export from `d052/selectors/__init__.py`.
6. **Tests** — extend GATE 8 (`test_selectors_critic.py`) and GATE 9
   (`test_selectors_determinism.py`): add the new type to `_all_configs()` so it is
   covered by bit-identical-replay-under-shuffle and the unified-interface check;
   add a family-specific semantics test (as `copeland` has).

## Determinism contract (mandatory)

- Identical inputs in ANY candidate order → identical `selected_ids` +
  `selection_hash`. The shared funnel sorts by `(composite DESC, candidate_id ASC)`.
- The `selection_hash` binds `(selector, critic_policy, k, seed, selected_ids)`;
  `SelectionResult` re-verifies it on construction.
- On shortfall, status is `INSUFFICIENT_ELIGIBLE_CANDIDATES` with an honest
  `shortfall_note` and **NO backfill / NO k-reduction / NO re-LLM**.
- Pure-python (numpy-free) to keep replay bit-identical across environments.

## Shared-frozen-pool invariant

`select(config, pool, signals)` hard-fails if `signals.pool_hash != pool.pool_hash`
or if the signals do not cover exactly the pool's candidates. Every selector
consumes ONE shared frozen pool; there is no per-selector pool.
