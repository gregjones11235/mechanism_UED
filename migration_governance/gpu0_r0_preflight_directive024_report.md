# GPU0 R0 Preflight Report — Directive 024

**Date:** 2026-07-14
**Commit:** `46f632a`
**GPU:** GPU0 (UUID: GPU-83d39a25-90a3-b18c-4235-1e624434bdfe)
**Status:** **ALL GATES PASS**

## T1 LPG-HRL

| Gate | Result |
|---|---|
| JAX GPU backend | cuda:0 (1 device) |
| Environment steps | **16,384** |
| Real PPO progress | ✓ (2 updates × 256 envs × 32 steps) |
| Treatment parameters in TrainState | **10 leaves** |
| Nonzero treatment loss | 0.000609 |
| Nonzero treatment gradient | 0.001290 |
| Checkpoint save | ✓ (step 16384) |
| Checkpoint restore + deep compare | ✓ (all leaves match) |
| Elapsed time | 143.0s |
| Output | `/root/experiments/dicode_runs/dspro/r0_preflight/t1_lpg_hrl_1784017027/` |
| Manifest | `/root/experiments/dicode_runs/dspro/r0_preflight/t1_lpg_hrl_1784017027/manifest.json` |

## T2 TSER-PPO

| Gate | Result |
|---|---|
| JAX GPU backend | cuda:0 (1 device) |
| Environment steps | **16,384** |
| Real PPO progress | ✓ (2 updates × 256 envs × 32 steps) |
| Treatment parameters in TrainState | **6 leaves** |
| Nonzero treatment loss | ✓ (MSE occupancy loss) |
| Nonzero treatment gradient | 0.000256 |
| Checkpoint save | ✓ (step 16384) |
| Checkpoint restore + deep compare | ✓ (all leaves match) |
| Elapsed time | 141.4s |
| Output | `/root/experiments/dicode_runs/dspro/r0_preflight/t2_tser_ppo_1784017804/` |
| Manifest | `/root/experiments/dicode_runs/dspro/r0_preflight/t2_tser_ppo_1784017804/manifest.json` |

## Failures Preserved

- `t1_lpg_hrl_seed0_16384/` — First attempt: wrong `max_updates_per_session=1` (8192 steps)
- `t1_lpg_hrl_1784016499/` — Second attempt: output collision
- `t1_lpg_hrl_1784016767/` — Third attempt: deep-compare tolerance
- `t2_tser_ppo_1784017225/` — T2 first attempt: same deep-compare bug

## Status

**READY_FOR_POST_R0_QUEUE** — Both T1 and T2 preflights pass. Per Directive 022/023/025, proceed to bounded screening pilots (50k, 100k) then formal paper-aligned horizons when all gates are met and approval manifests are written.
