# Directive 023: Transferable Training Mechanism Selection and Adaptive Schedule

Date: 2026-07-14 (Asia/Shanghai)
Parent directive: 022
Authority: user-selected training-mechanism program and 30-minute autonomous monitoring authorization

## Primary objective

The scientific objective is reliable improvement of held-out Tier 3 and Tier 4 achievement success in Craftax/DiCode. Aggregate return, selector diagnostics, entropy, training loss and curriculum diversity are supporting measurements, not substitutes for Tier 3/Tier 4 performance.

The program order remains:

1. prove the candidate-to-environment-to-PPO data plane;
2. validate aggregation mechanisms independently;
3. screen transferable training mechanisms independently under Original DiCode selection;
4. combine only validated aggregation and training finalists;
5. attempt a published-protocol-comparable SOTA result.

## Primary training candidates

### T1: LPG-HRL

Learned Prerequisite-Graph Hierarchical RL with PPO:

- infer or update an event/achievement prerequisite graph from auditable trajectory evidence;
- use a high-level goal/option policy over graph nodes;
- use goal-conditioned low-level policies with explicit option termination;
- separate manually supplied graph priors from learned edges;
- test transfer to held-out event combinations and changed prerequisite chains.

### T2: TSER-PPO

Transferable Successor-Event Representation with PPO:

- predict discounted future event/achievement occupancy;
- add goal-reachability and prerequisite-progress auxiliary losses;
- keep auxiliary metrics separate from student performance;
- test whether event representations improve held-out long-chain sample efficiency and transfer.

### T3: LPAC

Learning-Progress Adaptive Controller:

- consume normalized held-out progress, stagnation, forgetting and uncertainty signals;
- initially control only entropy coefficient and curriculum sampling temperature;
- compare against fixed settings and preregistered linear annealing;
- never use Tier labels as selector inputs;
- prevent held-out overfitting by separating controller feedback worlds from final reporting worlds.

## High-risk fallback

### T4: CLPA

Counterfactual Learning-Progress Attribution may begin only if all three primary candidates are engineering-valid but fail the screening rule below. It estimates the marginal held-out learning contribution of training tasks. Its causal assumptions, temporal confounders, estimator and negative controls must be written before GPU training.

T4 is not permission to discard or hide negative T1-T3 results.

## Screening sequence

### S0: engineering and transfer-interface tests

For every candidate require:

- real environment behavior reaches PPO rollouts;
- no metadata-only treatment;
- real optimizer/model/global-step checkpoint save and restore;
- deterministic manifests, unique outputs and physical GPU UUID;
- no dependency drift, API in PPO, silent fallback or overwritten evidence;
- an explicit transfer interface based on events, graphs or normalized learning dynamics rather than Craftax-only task names.

### S1: bounded seed-0 pilot

Run each engineering-valid primary candidate and the unmodified PPO control for at most 50,000 environment steps with identical settings. Use Original DiCode selection so aggregation is not a confound.

### S2: bounded confirmation

Promote a candidate to an additional total horizon of at most 100,000 environment steps only when S1 is valid and shows at least one preregistered signal:

- nonzero held-out Tier 3/Tier 4 success absent in the matched control;
- earlier first attainment of a Tier 3/Tier 4 achievement;
- greater held-out prerequisite-frontier depth without increased forgetting;
- materially improved long-chain completion probability under the same evaluation budget.

Supporting signals may justify continued diagnosis but cannot establish performance improvement alone.

### S3: primary failure and T4 activation

T4 may replace the lowest-priority queued work only when T1-T3 have each completed valid S1 and eligible S2 runs, and none provides a credible Tier 3/Tier 4 or frontier advantage over the matched control. Record the activation decision, all negative results and the exact estimator design before implementation.

## Dual-GPU schedule

- GPU0: new isolated work only. Initially owns LPG-HRL and TSER-PPO engineering/pilots. Never touch the frozen DeepSeek-V4-Pro reproduction.
- GPU1: complete R0 first, then aggregation controls and LPAC engineering/pilots in its isolated worktree.
- Use one single-GPU job per physical card. Do not use distributed two-GPU training unless separately reviewed.
- Cross-run at least one selected treatment/control pair on the opposite card before interpreting small timing or performance differences.
- Keep evaluation worlds, seeds, PPO geometry and checkpoint schedule identical across a comparison.

## Adaptive director authority

The director may reorder CPU tests, engineering preflights, valid <=100,000-step pilots and independent reviews based on new evidence. Every reordering must be recorded with timestamp, evidence, expected information gain and protected comparisons.

The director must not:

- choose or repeat seeds based on favorable results;
- stop a valid treatment early while allowing its control to continue;
- change final metrics after seeing results;
- weaken gates, hide failures or overwrite artifacts;
- call a diagnostic, single seed or unmatched protocol SOTA;
- start a run above 100,000 environment steps unless the standing user delegation is satisfied and the director has written a treatment-specific approval manifest binding the reviewed commit, treatment, seeds, horizon, evaluation protocol, outputs and stop conditions.

When a long-run matrix becomes justified, the director has delegated authority to approve it without another user interaction. Write one consolidated approval manifest covering mechanisms, seeds, horizons, commits, model IDs, cache/pool hashes, outputs and stop conditions. The default formal standard is the DiCode paper protocol recorded in the standing authorization; any scientific deviation must be explicitly labelled and cannot support a paper-comparable SOTA claim.

## Required reporting

Every 30-minute cycle should collect, when available:

- CC phase and current task;
- branch, commit and diff;
- exact process, physical GPU UUID and utilization;
- treatment, seed, horizon, actual global/environment step and checkpoint;
- held-out Tier 3/Tier 4 per-achievement success;
- prerequisite-frontier depth and first-attainment step;
- forgetting, long-chain completion, return and sample efficiency;
- pool/cache hashes and model identity for aggregation work;
- errors, NaN/Inf, fallback, collisions and validity status.

Notify the user after every monitoring cycle with an evidence-based concise summary. If no valid GPU process is active, state that explicitly rather than claiming formal training.

## SOTA rule

Before any SOTA claim, identify the exact current published benchmark and reproduce its evaluation protocol using primary sources. Freeze the comparison plan before final seeds. Report confidence intervals, compute/API cost and all failed or stopped runs. The central claim must be Tier 3/Tier 4 improvement, with aggregate return secondary.
