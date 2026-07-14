# GPU0 R0 Evidence Pack — Complete

**Date:** 2026-07-14
**Commit:** `b1647b66529558be8d37f10494badd7bcd264330`
**Worktree:** `/root/experiments/dicode-dspro-r0`
**Branch:** `exp/r0-aggregation-audit`

## CPU Tests: 49/49 PASS

| Suite | Tests | Result |
|---|---|---|
| T1 LPG-HRL embed sizes (regression) | 11 | PASS |
| T1 LPG-HRL real module | 9 | PASS |
| T2 TSER-PPO real module | 6 | PASS |
| T3 LPAC real module | 18 | PASS |
| Production integration | 5 | PASS |
| **Total** | **49** | **PASS** |

## GPU Preflights: ALL THREE TREATMENTS PASS

### T3 LPAC (b1647b6) — ADDED

| Gate | Result |
|---|---|
| Runtime commit | `b1647b66529558be8d37f10494badd7bcd264330` |
| JAX GPU | cuda:0 (1 device) |
| Physical UUID | `GPU-83d39a25-90a3-b18c-4235-1e624434bdfe` |
| Seed | 0 |
| Environment steps | 16,384 (control + treatment) |
| LPAC active | **true** (delta=0.005) |
| Entropy off | 0.0100 |
| Entropy on (adapted) | 0.0150 |
| Checkpoint save/restore | step 16384 verified |
| Runtime | 123.9s |
| Output | `/root/experiments/dicode_runs/dspro/r0_preflight/t3_lpac_1784026221/` |
| Manifest | `t3_lpac_1784026221/manifest.json` |

### T1 LPG-HRL (b1647b6)

| Gate | Result |
|---|---|
| Runtime commit | `b1647b66529558be8d37f10494badd7bcd264330` |
| JAX GPU | cuda:0 (1 device) |
| Physical UUID | `GPU-83d39a25-90a3-b18c-4235-1e624434bdfe` |
| Seed | 0 |
| Environment steps | 16,384 |
| Treatment leaves | 10 |
| Loss | 0.000609 (nonzero) |
| Gradient norm | 0.00129 (nonzero) |
| **Optimizer-applied parameter change** | **true** |
| Checkpoint save/restore | 10 leaves verified |
| Runtime | 144.3s |
| Output | `/root/experiments/dicode_runs/dspro/r0_preflight/t1_lpg_hrl_1784023395/` |
| Manifest | `t1_lpg_hrl_1784023395/manifest.json` |

### T2 TSER-PPO (b1647b6)

| Gate | Result |
|---|---|
| Runtime commit | `b1647b66529558be8d37f10494badd7bcd264330` |
| JAX GPU | cuda:0 (1 device) |
| Physical UUID | `GPU-83d39a25-90a3-b18c-4235-1e624434bdfe` |
| Seed | 0 |
| Environment steps | 16,384 |
| Treatment leaves | 6 |
| Loss | 0.09704 (nonzero) |
| Gradient norm | 0.000256 (nonzero) |
| Checkpoint save/restore | 6 leaves verified |
| Runtime | 143.2s |
| Output | `/root/experiments/dicode_runs/dspro/r0_preflight/t2_tser_ppo_1784025323/` |
| Manifest | `t2_tser_ppo_1784025323/manifest.json` |

## Bug Fixes Applied

| Fix | Commit | Tests |
|---|---|---|
| OptionTerminationGate one_hot depth | b1647b6 | 11/11 regression |
| Runtime commit binding | launcher | Verified in both manifests |
| Optimizer-applied param change | launcher | T1: true |

## Launchers Ready

| Launcher | Status |
|---|---|
| `t1_lpg_hrl_16384.py` | Reviewed, runtime commit, params-changed proof |
| `t2_tser_ppo_16384.py` | Reviewed, runtime commit |
| `t1_lpg_hrl_50k_seed0.py` | Prepared, runtime commit, NOT started |

## Prior Failures Preserved

| Run | Failure |
|---|---|
| `t1_lpg_hrl_seed0_16384` | Wrong updates_per_session (8192 steps) |
| `t1_lpg_hrl_1784016499` | Output collision |
| `t1_lpg_hrl_1784016767` | Deep-compare tolerance |
| `t1_lpg_hrl_1784017027` | Before OptionTerminationGate fix |
| `t2_tser_ppo_1784017225` | Same deep-compare bug |
| `t2_tser_ppo_1784017804` | Before runtime-commit fix |
| `t2_tser_ppo_1784025110` | Hardcoded commit mismatch |

## Frozen Reproduction

- GPU0 DeepSeek-V4-Pro reproduction: **untouched**
- Output: `/root/experiments/dicode_runs/dspro/dspro-gpu0-g3-seed0-20260712T041233Z/`

## Status

**GLOBAL R0 EVIDENCE COMPLETE.** 50k screening launchers prepared but not started.
