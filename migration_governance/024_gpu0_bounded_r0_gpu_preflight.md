# Directive 024: GPU0 Bounded R0 GPU Preflight

Date: 2026-07-14 (Asia/Shanghai)
Authority: explicit user request to start GPU0 while GPU1 completes R0

## Scope

Run one bounded, real PPO engineering preflight on physical GPU0 from the clean reviewed commit `46f632a` in `/root/experiments/dicode-dspro-r0`.

This is not a performance experiment and must not be described as Tier 3/Tier 4 improvement, a training-mechanism comparison, or SOTA evidence.

## Required configuration

- Physical device UUID: `GPU-83d39a25-90a3-b18c-4235-1e624434bdfe`.
- `CUDA_VISIBLE_DEVICES=0`; JAX must report a GPU backend and exactly one visible device.
- Commit: exactly `46f632a`; clean worktree before launch.
- Seed: `0`.
- Selector: unmodified Original DiCode; aggregation disabled and zero LLM-cache influence.
- No API calls and no dependency changes.
- Maximum horizon: `16,384` environment steps total.
- Run treatments sequentially, never concurrently: T1 LPG-HRL first, then T2 TSER-PPO only if T1 passes.
- Unique output, log, manifest and checkpoint paths; never reuse an existing run directory.

## Preflight gates

Before each treatment, record branch, commit, command, resolved config, physical UUID, JAX backend/device count, seed, exact horizon and output path.

For each treatment require:

- real PPO/global-step progress reaches the exact bounded horizon;
- nonzero treatment gradient norm and parameters changed through optimizer application;
- no NaN/Inf, traceback, fallback or output collision;
- checkpoint contains model, optimizer, treatment parameters and global step;
- fresh-process checkpoint restore reproduces the recorded global step and treatment parameter tree;
- original frozen GPU0 DeepSeek-V4-Pro reproduction remains untouched.

Stop immediately at the first failure. Preserve all failure evidence. Do not automatically repair or extend the horizon under this directive.

## Completion

Write a timestamped manifest and report under `/root/experiments/dicode_orchestration`. Do not start a 50k or longer pilot until global R0 independently passes or a new explicit directive changes that gate.
