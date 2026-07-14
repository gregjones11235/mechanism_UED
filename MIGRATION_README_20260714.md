# Dual-GPU DiCode Migration Snapshot — 2026-07-14

This snapshot is prepared for continuing the aggregation and transferable
training-mechanism program on a replacement server. It contains program source,
tests, configurations, dependency locks and tracked documentation. It does not
contain credentials, writable caches, experiment outputs or checkpoints.

## Frozen source snapshots

### GPU0 training mechanisms

- Local directory: `gpu0_training_mechanisms/`
- Server branch: `exp/r0-aggregation-audit`
- Server commit: `b1647b66529558be8d37f10494badd7bcd264330`
- Purpose: LPG-HRL and TSER-PPO engineering, transfer-interface tests and
  bounded screening launchers.
- Important fix in this commit: `OptionTerminationGate` uses configurable
  option count instead of a hard-coded one-hot depth of 128.

### GPU1 aggregation and SIEGE

- Local directory: `gpu1_aggregation_siege/`
- Server branch: `exp/siege-aggregation-sota`
- Server commit: `69f26a38a3200951f125e4d01c92ceb5e79fc5cc`
- Purpose: Original, Soft Copeland, Budgeted Soft Copeland, raw/budgeted
  Auction, SIEGE integration, R0 production dispatch and R1 comparisons.

The two directories are independent source snapshots. Do not combine their
writable output, checkpoint, Hydra, cache or temporary directories.

## Dependency reconstruction

Both snapshots carry identical dependency specifications:

| File | SHA256 |
|---|---|
| `requirements.txt` | `082A544C15DE4B2F48679A17336447BE67FCD20DC94755C30621DB6F85A8DBED` |
| `pyproject.toml` | `B18C59C221E1C413A0AF9DBD4B444160F9DC30866F18CB4DCF3488030CCACB6A` |
| `uv.lock` | `4C0B7F14DE4550278711BBFFEFC419DAEBAD2D98AE0FE7F147382878272C88F6` |

Preferred reconstruction, separately in each worktree:

```bash
uv sync --frozen
```

If `uv` is unavailable, use an isolated Python 3.10 environment and install
from `requirements.txt`, recording any deviation. Do not silently upgrade JAX,
CUDA plugins or other dependencies.

The source server required the system cuSPARSE library to be loaded explicitly:

```bash
export LD_PRELOAD=/usr/local/cuda-12.8/lib64/libcusparse.so.12
```

Verify the actual path on the replacement server before use. Every experiment
must fail closed unless JAX sees exactly one assigned GPU.

## Required non-Git migration data

Checkpoints and run outputs are intentionally excluded from ordinary Git.
Copy them separately with hashes from the old server before decommissioning it:

```text
/root/experiments/dicode_runs/dspro/r0_preflight/
/root/experiments/dicode_runs/siege_aggregation/gate_r0_final/
/root/experiments/dicode_orchestration/manifests/
/root/experiments/dicode_orchestration/reports/
```

Also preserve the completed GPU0 DeepSeek-V4-Pro reproduction directory as
immutable evidence. Never resume, overwrite or repurpose that frozen run.

For each copied artifact record relative path, byte size and SHA256. Never copy
API keys, `.env` files, credentials, writable shared caches or provider tokens.

## Included launchers and bounded evidence

The following output-root launchers were copied explicitly because they are not
tracked by the GPU0 source branch:

| SHA256 | File |
|---|---|
| `7C06805568178D46D55B834A8B2DF8EB1BEF11871DB329BC7E87141B006F3F4B` | `migration_launchers/gpu0/t1_lpg_hrl_16384.py` |
| `97EC36A8BCF07177E60209B985CDBAA26FB957B566342ED0E8FE741ACBC0AE59` | `migration_launchers/gpu0/t1_lpg_hrl_50k_seed0.py` |
| `B11F85CFF6CCAFFDFE5AE4BFFDD89DE6F810C20DAE71710242014AF936CC4B07` | `migration_launchers/gpu0/t2_tser_ppo_16384.py` |

Bounded runtime evidence included for resume validation:

| SHA256 | File |
|---|---|
| `437FFA75645A4D8030A012E058B07C9D8DD402C7D48F9223EAABE5320AEDE403` | `migration_evidence/gpu0/t1_lpg_hrl_b1647b6_16384_manifest.json` |
| `87C105ACFB09C21506628A5A3299A24563AD858847B6435E915CB0279F6B9A26` | `migration_evidence/gpu1/auction_raw_s0_16384_runtime_manifest.json` |

These small manifests do not replace the full checkpoints. Re-run restore tests
after copying the checkpoint directories separately.

## Resume order

1. Recreate isolated Python environments from the locked dependencies.
2. Bind each worktree to one physical GPU UUID and verify one-device JAX.
3. Restore immutable caches and frozen candidate pools read-only; verify hashes.
4. Restore checkpoints into new unique output roots and run fresh-process
   fail-closed restore checks.
5. Re-run R0 production integration before performance training.
6. Complete matched R1 aggregation comparisons.
7. Screen LPG-HRL, TSER-PPO and LPAC under Original selection before combining
   aggregation and training mechanisms.
8. Start paper-scale runs only from a complete binding manifest.

## Snapshot caveats

- GPU0 preflight launchers live under the server output root and are not part of
  the tracked source commit. Preserve them through the non-Git artifact copy.
- A GPU0 T1 manifest previously hard-coded the old commit `46f632a`; do not use
  that record as final binding evidence. Require the runtime commit to match
  `git rev-parse HEAD`.
- Old invalid E1/E3 outputs and stale `COMPLETE` controller states are not R0
  evidence.
- Diagnostics, smoke tests and single-seed pilots are not SOTA evidence.
