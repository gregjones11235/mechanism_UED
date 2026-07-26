# Global Evaluator Diff — CANONICAL_EVALUATOR_V1 remediation

Anchor: `eval_phase2_unified.py` (SHA `224514026aefd273...`, bit-identity verified == local file).
All fixes below are realized as NEW canonical files under this remediation dir; **no existing evaluator or
old result is overwritten or silently changed** (per discipline).

## The 7 mandated fixes

### FIX-E1 — P7 argmax dead code
- **Finding:** `gpu1_p7_egomap/eval_p7.py` uses `pi.mode` (ARGMAX) and contains `build_eval_fn` that is
  never called (TypeError/NameError on invocation); its `compare_arms` (McNemar/bootstrap) is therefore
  unreachable. The functional P7 evaluator `eval_p7_egomap_paired_256.py` is STOCHASTIC (`pi.sample`).
- **Risk:** a dead argmax path could be mis-cited as the P7 protocol, contradicting the stochastic main path.
- **Fix:** QUARANTINE `eval_p7.py`; CANONICAL_EVALUATOR_V1 forbids reporting argmax unless it is the actual
  executed mode. The canonical reference evaluator prints + records `action_mode` (assertions A1/A2) so a dead
  branch can never silently define the mode.

### FIX-E2 — G6 / gate argmax-stochastic inconsistency
- **Finding:** the anchor itself uses `jnp.argmax(logits)` at `:277` — but ONLY as a memory-off diagnostic probe
  (do on/off networks pick different argmax actions?), NOT as the policy. `policy_mode="stochastic"` at `:200`.
- **Risk:** the argmax diagnostic could be mis-read as the evaluation policy.
- **Fix:** RELABEL all such argmax usages as `memory_off_diagnostic_probe`; CANONICAL_EVALUATOR_V1 separates
  `action_mode` (policy) from any diagnostic probe and requires the probe to be named as such in output.

### FIX-E3 — W512 multi-evaluator caliber drift
- **Finding:** W512 arms1-4 use `eval_a_side_unified.py` (SHA dcf7fe20) while arms5-6 use
  `eval_w512_p2replay.py` (SHA f76bb53c). Two files, "functionally equivalent" claimed but NOT byte-identical.
- **Risk:** cross-arm W512 comparisons carry an evaluator-file difference; violates "single canonical protocol".
- **Fix:** MERGE to a single W512 evaluator instance under CANONICAL_EVALUATOR_V1 (one SHA, one world_set_hash),
  re-run all 6 arms through it (future, authorized eval). Until then, W512 cross-file deltas are flagged
  `EVALUATOR_FILE_DRIFT`.

### FIX-E4 — checkpoint partial restore / silent fallback
- **Finding:** `eval_p8_migration.py` uses orbax `load_weights_only` with a nested→flat→raw **silent fallback
  chain**. Silent partial restore can mask missing/extra params.
- **Fix:** CANONICAL_EVALUATOR_V1 mandates HARD-FAIL on any leaf-set mismatch (missing OR extra), GATE4. Compat
  fallback FORBIDDEN unless an explicit SHA-recorded migration map is supplied + audited.

### FIX-E5 — memory-off ablation truly only disables memory
- **Finding:** memory-off must zero the long-state input while keeping the short-term policy path identical;
  verified by the argmax-logit divergence probe (`:277`) which should show off≠on ONLY where memory matters.
- **Fix:** CANONICAL_EVALUATOR_V1 records the memory-off probe result explicitly and requires GATE5/GATE6
  (isolation + done-reset) so "memory-off" is provably only the memory channel.

### FIX-E6 — batched-env state cross-contamination
- **Risk:** in the distributed/batched wrapper, per-world memory rows could leak across the batch.
- **Fix:** GATE5 cross-contamination probe (perturb world i's memory, assert worlds j≠i unchanged). Required
  before any per-world paired statistic is trusted.

### FIX-E7 — episode-done memory reset
- **Risk:** memory not zeroed at episode boundary leaks previous-episode state into the next world's episode.
- **Fix:** GATE6 asserts memory == zero at the start of each episode after a done. Anchor resets short mem on
  done and long state on true_done; canonical evaluator records the reset hook + asserts it.

## action_mode handling (global)
- Only two legal values: `stochastic`, `argmax`.
- MUST be printed at startup (A1) and written to output (A2).
- A behavioral test (A3, `tools/action_mode_consistency_test.py`) confirms sampling matches declaration.
- NEVER inferred from a dead branch or default value.

## seed/world discipline
- seed42 (Phase2/P8/P9/W512) and seed100000 (P7/LC) are DIFFERENT world sets — never pooled.
- No paired comparison without identical `world_set_hash` AND evaluator SHA (GATE11).
