# Temporary R0 Repair Edit Authorization

- Authorized by: user in the active Codex thread
- Effective date: 2026-07-14 (Asia/Shanghai)
- Purpose: accelerate completion of the R0 production-integration gate so valid experiments can start.
- Expiry: immediately when an independent R0 review returns `PASS`.

## Temporarily authorized scope

Codex may make the smallest necessary source, test, or configuration edits only in these isolated remote repair worktrees:

- `/root/experiments/dicode-dspro-r0`
- `/root/experiments/dicode-siege-aggregation`

Every edit must address a concrete synchronized R0 blocker, preserve prior evidence, and be followed by diff inspection and proportionate tests.

## Required R0 completion evidence

- Rejected candidates cannot re-enter the active pool.
- All five selector mechanisms use the same frozen 32-candidate pool, immutable judgement cache, and select exactly 8 tasks.
- Candidate specifications causally change executable Craftax `reset_env` / `step_env` behavior consumed by PPO.
- The production adapter reaches the real `make_train` path, and inside-PPO traces preserve candidate identity.
- Checkpoints restore the real model, optimizer, and global step fail-closed.
- Original selector has zero LLM-cache influence.
- Enhanced treatments coherently reuse the pre-existing immutable cache with hit rate at least 95 percent.
- No experimental API, configured-step substitute, silent fallback, output collision, or dependency drift occurs.
- GPU0 mechanism tests demonstrate nonzero treatment gradients, changed parameter leaves after optimizer application, and restorable checkpoints.

## Still prohibited

- Any modification, resumption, overwrite, or deletion of the completed GPU0 DeepSeek-V4-Pro reproduction or its outputs.
- Editing base mirrors, unrelated worktrees, shared caches, credentials, or existing experiment outputs.
- Dependency changes, secret exposure, Git push/reset/rebase/merge, or deletion of failure evidence.
- Starting long or formal training before R0 passes and the standing long-run gates are bound to a valid manifest.

## Automatic reversion

On independent R0 `PASS`, direct file-edit authority ends automatically. The director returns to supervision/control-only operation and changes the monitoring interval from 10 minutes to 30 minutes.
