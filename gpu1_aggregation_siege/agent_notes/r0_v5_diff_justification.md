# R0 v5 Diff Justification (GPU1 Item 2)

## Semantic changes in production_dispatcher.py

### Change 1: make_test_defaults() factory (ADDED)
**What:** Extracted the 6 inline mock classes (L, NV, FG, FA, FGM, FC) from
`_dispatch_original()`'s else-branch into a module-level `make_test_defaults(pool)`.
**Why:** The factory pattern makes the test-only nature explicit in the name and
docstring. Previously the mock construction was hidden inside a production method.
**Risk:** None. Same code, same behavior, moved to a different scope.

### Change 2: _dispatch_original() requires non-None gen_manager, config
**What:** Removed `=None` defaults; the method now raises TypeError if either is None.
**Why:** Original production mode must use the real DiCode selector
(`sample_tasks_for_training`) with real GenManager + config. Silently falling back
to test mocks in production would produce selections disconnected from the actual
training state (priority scores, session indices, staleness).
**Risk:** Breaks callers that previously relied on the silent fallback. Those callers
must now use `make_test_defaults()` explicitly. This is the intended fail-closed behavior.

### Change 3: dispatch() validates injection for "original" mechanism
**What:** `dispatch(mechanism, gen_manager=None, config=None)` now checks both are
non-None when mechanism=="original", before delegating to `_dispatch_original`.
**Why:** Two-layer defense — the public API validates before the private method.
**Risk:** Same as Change 2.

### Change 4: Whitespace normalization
**What:** Spaces → tabs throughout the file (matching the project's tab-indented style).
**Why:** The original file mixed 4-space and tab indentation. Normalized to tabs.
**Risk:** Inflates the diff but changes no logic. All semantic sections are preserved.

## Semantic changes in test_r0_production_dispatcher.py

### Change 5: Real reset_env/step_env calls
**What:** Replaced `task.generate_world(rng_key)` with actual `env.reset_env(rng, params, task_id)`
and `env.step_env(rng, state, action, params)` calls.
**Why:** The v4 test only called `generate_world` directly, which bypasses:
- The `lax.switch(task_id, ...)` dispatch
- TaskParams stacking and clamping
- Achievement pre-population
- The full step function (mob spawning, player intrinsics, plant updates, etc.)
The `step_env` path exercises 17 game-logic subroutines that `generate_world` skips.
**Risk:** Adds ~30s to test runtime (world generation for each env). Acceptable.

### Change 6: Causal divergence test
**What:** 5-step deterministic rollout for task_id=0 and task_id=1 with fixed RNG seed 42.
Records pytree-leaf hashes at reset and after each step. Fails if no divergence found.
**Why:** Proves that different compiled candidate task IDs produce causally different
environment behavior — different worlds, different mob spawns, different transitions.
This is the essential property that makes the aggregation meaningful: if all tasks
produced identical behavior, selection would be vacuous.
**Risk:** None. Deterministic RNG + fixed action sequence = reproducible.

### Change 7: make_train signature validation
**What:** Inspects the real `make_train(config, task_classes, num_training_updates, ...)`
signature and verifies all 5 adapter outputs are structurally compatible.
**Why:** The v4 test instantiated envs but never checked that the compiled task_classes
could actually be passed to the PPO training entry point. This validates the interface
contract without executing training (CPU-only policy).
**Risk:** None. Pure introspection, no side effects.

### Change 8: Uses make_test_defaults() factory
**What:** Original dispatch now uses `make_test_defaults(d.pool)` instead of relying
on the (now-removed) silent fallback.
**Why:** Tests the injection path explicitly rather than the now-nonexistent default path.
**Risk:** None.

## Summary
All changes are fail-closed: Original mode now refuses to operate without explicit
injection, test mocks are labeled as such, and real env API calls replace the
insufficient generate_world-only tests. No production code path was removed — only
the silent fallback was eliminated.
