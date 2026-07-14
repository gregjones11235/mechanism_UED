# Directive 025: Final Dual-GPU Autonomous Handoff

Date: 2026-07-14 (Asia/Shanghai)
Authority: explicit user request for the final Codex control cycle and continuing CC autonomy
Supersedes: any workflow assumption that CC must wait for a fresh Codex or user approval between legal stages

## Operating authority

After this directive is delivered, `cc_gpu0` and `cc_gpu1` own continuous execution of the approved Directive 022/023 research program. They must not return to an empty prompt or wait for Codex while valid work remains.

Within their isolated worktrees and output roots, CC may autonomously:

- inspect and modify experiment source, tests, launchers and configurations;
- run CPU tests, real GPU preflights, bounded pilots and matched experiments;
- create unique output directories, manifests, checkpoints, reports and local commits;
- diagnose and repair engineering failures, preserving the original failure evidence;
- stop or restart only the exact child experiment process that CC itself launched when it is invalid, failed or superseded;
- reorder pending legal work for information gain while preserving matched controls;
- promote treatments through the standing paper-aligned authorization when every recorded gate is satisfied;
- continue beyond five hours and across Codex inactivity without requesting routine approval.

CC must perform its own fail-closed review at every transition and record the evidence. No extra Codex or user response is required for work already within Directives 022/023 and the standing authorization.

## Immediate GPU0 queue

1. Preserve the existing invalid-step T1 attempt as failed engineering evidence. Do not delete or overwrite any further failed output.
2. Run the corrected T1 LPG-HRL GPU0 preflight at exactly 16,384 environment steps, seed 0, physical UUID `GPU-83d39a25-90a3-b18c-4235-1e624434bdfe`, with unique output.
3. Require real PPO progress, nonzero treatment gradients, changed optimizer-applied parameter leaves, complete model/optimizer/global-step checkpoint, and a fresh-process fail-closed restore.
4. If T1 passes, run the matched T2 TSER-PPO preflight with the same geometry and evidence rules.
5. If a safety classifier is temporarily unavailable, do not stop work: preserve the prepared command, retry at bounded intervals, execute CPU-side validation/reporting meanwhile, and launch immediately when the approved command becomes available.
6. Never touch the completed DeepSeek-V4-Pro reproduction or its outputs.

## Immediate GPU1 queue

1. Complete the pending CPU regression test and the remaining R0 production integration tasks.
2. Reduce or explicitly justify the broad dispatcher/test diff; review every semantic change.
3. Prove real reset/step causal divergence for multiple compiled candidates, real `make_train` adapter compatibility, inside-PPO candidate identity, checkpoint restore, frozen 32-to-8 pool/cache equality, Original zero-cache influence and enhanced hit rate at least 95 percent.
4. After a recorded R0 PASS, proceed directly to R1 aggregation comparisons and Directive 023 training-mechanism screening without waiting for Codex.

## Post-R0 dual-GPU schedule

- Use both physical cards concurrently only as separate single-GPU jobs in separate worktrees and output roots.
- Split matched R1 aggregation comparisons and T1/T2/T3 training-mechanism screening across the two cards.
- Run Original/PPO matched controls and preserve identical seeds, evaluation worlds, PPO geometry and checkpoint schedules.
- Promote through 50k, 100k and paper-aligned formal horizons only on recorded preregistered evidence and a complete binding manifest.
- The standing authorization permits formal paper-scale runs without another user interaction once all gates are met.

## Permanent non-negotiable boundaries

Full autonomy does not authorize:

- modifying, resuming, deleting or overwriting the frozen GPU0 DeepSeek-V4-Pro reproduction;
- exposing credentials, calling experimental APIs inside PPO, or using silent/configured-step fallback;
- changing dependencies or shared caches;
- Git push, reset, rebase or merge;
- hiding/deleting failures, overwriting outputs, favorable-seed repetition or tuning on final worlds;
- calling diagnostics, unmatched pilots or single-seed results performance improvement or SOTA;
- using both GPUs for one distributed job without a separately recorded protocol.

If a task is blocked, continue the highest-value legal CPU or alternate-card task and record the blocker. Do not idle merely because Codex is unavailable.
