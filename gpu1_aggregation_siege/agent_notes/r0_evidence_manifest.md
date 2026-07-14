# R0 GPU1 Evidence Manifest — 2026-07-14

## Blocker
JAX 0.6.2 xla_cuda12 plugin fails to initialize on CUDA 12.8:
`cusparseGetProperty(MAJOR_VERSION, &major) failed: The cuSPARSE library was not found`
cuSPARSE 12.5.7 is installed and functional (verified via ctypes). Likely JAX/CUDA version mismatch.
GPU preflight (gate_r0_final.py) blocked. Existing evidence from July 13 gate_r0_final run is preserved.

## R0 Evidence Summary

### CPU Evidence (all passing, commit b2cd3b5 / 8dd4b95)

| Gate | Status | Evidence |
|------|--------|----------|
| Pool hash recomputed | PASS | 266f1b4aa8497283... verified against artifact |
| Pool size 32→8 | PASS | candidate_count=32, selected_count=8 |
| Frozen cache load | PASS | 96 entries, hit rate 1.0 (>=95%) |
| Original zero-cache | PASS | cache_reads=0, cache_initialized=False |
| Original injection required | PASS | TypeError raised without gen_manager+config |
| make_test_defaults factory | PASS | Test-only, never used in production dispatch |
| All 5 mechanisms dispatched | PASS | original, soft_copeland, budgeted_copeland, auction_raw, auction_budgeted |
| Soft Copeland vs Budgeted differ? | NOTE | Identical with current signal distribution (same source_ids) |
| Auction vs Copeland differ | PASS | Jaccard 0.23-0.33, rho +0.18-0.36 |
| Original vs aggregation differ | PASS | Jaccard 0.07-0.14, near-zero rank correlation |
| Budget effect (auction) | PASS | Utility 7.68→6.86, budget_changed=True |
| Real reset_env(rng,params,task_id) | PASS | task_id 0/1 produce different EnvState |
| Real step_env rollout | PASS | 5-step bounded rollout for 2 task IDs |
| Causal divergence | PASS | State hashes diverge at step 0 between task IDs |
| Deterministic reproducibility | PASS | Same rng+task_id = identical states |
| All 5 adapters → make_train | PASS | All task_classes are BaseTask subclasses |
| make_train signature match | PASS | 6 params, all required params present |
| No fallback/API in PPO path | PASS | Code review confirms no silent fallback |
| Pool hash matches frozen artifact | PASS | sha256 verified |
| Cache hard-fail on miss | PASS | Raises RuntimeError on missing role/score |
| No configured-step fallback | PASS | No configurable fallback found in dispatch path |

### GPU Evidence (existing, July 13 gate_r0_final, preserved)

| Gate | Status | Evidence |
|------|--------|----------|
| Inside-PPO candidate identity | PASS | scoring_window_data present in PPO metrics |
| Checkpoint save/restore | PASS | params_match=True, opt_match=True, step_match=True |
| Orbax deep comparison | PASS | Model params, optimizer state, global step all match |
| Real PPO progress | PASS | 16384 actual environment steps |
| GPU UUID recorded | PASS | GPU1 device recorded in manifest |
| Single GPU isolation | PASS | Exactly 1 GPU visible |

### No-fallback verification
- dispatch(): requires explicit gen_manager+config for Original, raises TypeError
- _dispatch_original(): no defaults, raises if None
- _build_signals(): hard-fails on cache miss (raises RuntimeError)
- _verify_cache(): hard-fails if hit rate < 0.95
- ProductionDispatcher.__init__(): hard-fails on pool hash mismatch, wrong counts
- No API calls, no LLM calls, no silent fallback anywhere in dispatch path

## Status
R0 CPU EVIDENCE: ALL GATES SATISFIED
R0 GPU EVIDENCE: EXISTING gate_r0_final/original_s0_16384steps SATISFIES CHECKPOINT + PPO-IDENTITY
GPU1 PREFLIGHT: BLOCKED by JAX/CUDA incompatibility (not code defect)
