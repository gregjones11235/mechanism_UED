# Standing User Authorization: Dual-GPU DiCode-Paper Long Runs

Date: 2026-07-14 (Asia/Shanghai)
Authority: explicit user delegation in the Codex thread
Status: ACTIVE until explicitly revoked

## Delegation

The user delegates to the independent Codex experiment director the decision to approve long-running GPU0 and GPU1 experiments in the Directive 022/023 research program. Additional per-run user approval is not required when every condition in this document is satisfied.

This standing file supplies the explicit user authorization required by `AGENTS.md`; it does not waive any scientific-validity, isolation, secret, dependency, cache, API-loop, output or evidence gate.

## Paper-aligned formal standard

Formal long runs intended for DiCode paper comparison use:

- approximately 2 billion environment steps per seed, with the resolved launcher-compatible horizon recorded exactly (the existing reproduction used `2,005,401,600`);
- five preregistered seeds, default `0,1,2,3,4`;
- a fixed held-out evaluation set of 1,024 worlds;
- the same evaluation frequency, checkpoint schedule, environment configuration and policy architecture across matched treatments unless the architecture itself is the declared independent variable;
- per-achievement Tier 3/Tier 4 success, mean return, long-chain completion, sample efficiency, confidence intervals, compute/API cost and failures;
- the unmodified DiCode/PPO control and all causal cells needed to attribute aggregation, training and interaction effects.

If an exact paper setting cannot be recovered or reproduced, the director must record the discrepancy before launch and label the result non-paper-comparable.

## Director approval gates

Before creating a long-run approval, the director must independently establish:

1. R0 candidate-to-environment-to-PPO data-plane PASS with real rollout and checkpoint restore evidence.
2. The treatment completed valid engineering tests and bounded screening runs without fallback, collision, NaN/Inf or scientific-invalidity failure.
3. The treatment has a preregistered causal hypothesis and at least one valid reason for promotion based on Tier 3/Tier 4 success, first attainment, prerequisite-frontier progress, long-chain completion or sample efficiency.
4. The matched control, treatment, seed set, exact horizon, physical GPU assignment, commit, dirty-state policy, environment, candidate pool/cache hashes, model identities, output paths, evaluation worlds and stop conditions are bound in a manifest.
5. The completed GPU0 DeepSeek-V4-Pro reproduction remains frozen and untouched; all new GPU0 work uses a new isolated worktree and output root.
6. No API is reachable in PPO, enhanced cache hit rate is at least 95 percent, no silent fallback exists, dependencies are unchanged and secrets are absent.
7. Disk, checkpoint restoration, GPU/JAX backend and unique-output preflight pass immediately before launch.

## Scope of delegated decisions

The director may:

- promote finalists from <=100,000-step screening to intermediate or formal paper-scale runs;
- schedule matched runs sequentially across GPU0 and GPU1;
- reorder not-yet-started treatments based on recorded expected information gain;
- stop or withhold a run for concrete validity, safety, resource or preregistered futility reasons;
- activate CLPA under Directive 023 after the primary-candidate failure rule is met;
- issue local/server orchestration approval flags consistent with the reviewed manifest.

The director may not:

- repeat favorable seeds, discard unfavorable valid seeds or tune on final reporting worlds;
- add unregistered treatments directly to the formal matrix;
- overwrite or conceal failures;
- change metrics, horizons or stopping rules after seeing outcomes;
- use a nonmatched protocol to claim SOTA;
- modify dependencies, expose secrets, use both GPUs for one distributed job without separate review, or interfere with the frozen reproduction.

## Efficient approval policy

Approval should normally be issued once per consolidated matrix rather than once per seed. A matrix manifest may cover all five seeds and both physical GPUs when commands and output paths are enumerated and fail closed. The director reports each 30-minute monitoring result to the user, who may revoke or alter this delegation at any time.
