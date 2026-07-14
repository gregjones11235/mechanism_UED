# Directive 022: Dual-GPU Aggregation-to-Training-to-SOTA Roadmap

Date: 2026-07-14 (Asia/Shanghai)
Authority: explicit user research-direction authorization in the Codex thread
Scope: future GPU0 and GPU1 work after the completed GPU0 DeepSeek-V4-Pro reproduction

## Research objective

The program now has three ordered objectives:

1. Establish whether competitive curriculum aggregation improves over the unmodified DiCode selector.
2. Establish whether a validated aggregation mechanism combines beneficially with a validated long-horizon training mechanism for Tier 3 and Tier 4 achievements.
3. Only after both causal questions are answered, run a controlled SOTA attempt.

Do not merge these questions prematurely. Diagnostics, single-seed pilots and mechanism-only results are not SOTA evidence.

## Resource transition and isolation

- The completed GPU0 DeepSeek-V4-Pro reproduction is frozen evidence. Never overwrite, resume in-place, delete or repurpose its run directory, logs, cache or checkpoint.
- GPU0 may now be used for new aggregation-program experiments under a new isolated worktree, branch, output root and manifests.
- GPU1 retains its own isolated SIEGE/aggregation worktree and output root.
- Every process must record the physical GPU UUID and fail closed unless it sees exactly its assigned GPU.
- Never share writable checkpoint, output, Hydra or temporary directories across cards.
- Frozen candidate pools and immutable judgment caches may be shared read-only only when their hashes are recorded and identical across comparisons.
- Do not run concurrent jobs whose CPU, RAM, disk-I/O or API usage invalidates timing or throughput comparisons.

## Gate R0: repair and data-plane validity

Before either card begins new performance training, independently prove on commit `8ff0d96` or a reviewed descendant:

- chain-rejected candidates never re-enter the admitted pool;
- candidate specifications causally change executable environment behavior consumed by PPO;
- automated negative and integration tests execute in the real `dicode310` environment;
- real model, optimizer and global-step state save and restore fail closed;
- the same pre-existing frozen immutable cache is loaded read-only, with coherent enhanced-run hit rate at least 95 percent;
- no configured-step fallback, silent LLM/rule fallback or API call inside PPO exists;
- Original DiCode has aggregation disabled and zero LLM-cache influence on selection.

The stale `COMPLETE` state and old invalid E1/E3 outputs are not evidence for this gate.

## Stage R1: aggregation mechanism question first

Use the same executable 32-candidate frozen pool, select 8, and the same immutable four-role evidence for:

1. unmodified Original DiCode selector;
2. Soft Copeland;
3. Budgeted Soft Copeland;
4. raw Auction;
5. budgeted Auction;
6. SIEGE only after its upstream provenance and actual selector-to-PPO dispatch are proven.

First run mechanism diagnostics without interpreting them as student performance. Then run matched short training pilots only after R0 passes. Required measurements include selection overlap, rank correlation, pairwise consistency, candidate coverage, diversity, budget utilization/sensitivity, cache hit rate, Tier 3/Tier 4 achievement rates, return distribution and sample efficiency.

Choose at most two aggregation finalists using preregistered criteria. Do not tune a mechanism on the same seeds used for the final comparison.

## Stage R2: long-horizon training mechanisms independent of aggregation

Evaluate training mechanisms with the unmodified Original selector first so aggregation is not a confound. SIEGE is one candidate, not the only permitted approach. Prioritize a small, causally distinct set:

1. achievement-graph hierarchical PPO with subgoal-conditioned options and explicit option termination;
2. frontier/archive exploration with checkpointed state restoration or Go-Explore-style return-to-frontier, only if environment-state restoration is deterministic and auditable;
3. reverse or prerequisite curriculum starting from verified states immediately before Tier 3/Tier 4 bottlenecks, with evaluation always from the normal initial-state distribution;
4. achievement-conditioned auxiliary objectives or successor-feature style credit assignment for sparse long-horizon rewards;
5. SIEGE-conditioned training as the existing comparator.

Each candidate must state its causal hypothesis, changed training component, fixed components, checkpoint semantics and failure conditions. Begin with CPU/unit/integration tests, then small GPU pilots. Reject mechanisms that merely relabel tasks, alter reporting tiers, leak future state, change evaluation starts, or fail to affect PPO-consumed trajectories.

Choose at most two training finalists based on held-out Tier 3/Tier 4 success, sample efficiency, stability and compute cost, not aggregate return alone.

## Stage R3: aggregation plus training mechanism

Run a compact factorial study rather than an uncontrolled sweep:

- Original selector plus PPO baseline;
- best aggregation plus PPO baseline;
- Original selector plus best training mechanism;
- best aggregation plus best training mechanism.

If a second finalist is justified, extend only after the core 2-by-2 study is complete. Use identical seeds, evaluation worlds, environment settings, candidate universe, cache snapshot, training horizon and checkpoint schedule. Test interaction effects explicitly; do not attribute a combined gain solely to aggregation or solely to training.

## Stage R4: SOTA attempt

Only after R1-R3 pass:

- freeze code, configs, model IDs, candidate pools, caches, evaluation worlds and analysis plan;
- run the unmodified DiCode/PPO control and strongest published-comparable baseline alongside the proposed method;
- use multiple seeds and report confidence intervals, per-achievement Tier 3/Tier 4 results, aggregate return, sample efficiency, compute/API cost and failures;
- keep tuning seeds separate from final reporting seeds;
- use the standing explicit delegation in `orchestration/manifests/DUAL_GPU_DICODE_PAPER_LONG_RUN_STANDING_AUTHORIZATION_20260714.md`; the director may approve runs above 100,000 environment steps only after the delegated evidence gates pass and a treatment-specific approval manifest is written;
- call a result SOTA only when the metric, evaluation protocol and comparison conditions match the cited benchmark.

## Dual-GPU scheduling

- During R0 and R1, GPU1 owns aggregation correctness and matched aggregation pilots. GPU0 owns independent reproduction of the R0 integration test and, after R0 passes, the Original-selector training controls.
- During R2, GPU0 evaluates non-SIEGE long-horizon mechanisms while GPU1 evaluates SIEGE and the aggregation finalists.
- During R3, assign one matched cell to each GPU at a time; swap or repeat a subset across physical cards to detect hardware/launcher confounds.
- No job may use both GPUs unless a separate distributed-training protocol is reviewed. The default is one isolated single-GPU job per card.

## Immediate next action

1. Independently run the missing real-environment tests for `8ff0d96`; do not launch performance training until they pass.
2. Produce a manifest for the frozen candidate pool/cache and a matched R1 command matrix.
3. Produce short design notes and executable negative tests for the five R2 candidates; select no more than three for initial pilots based on implementation validity and causal distinctness.
4. For any experiment above 100,000 environment steps, the director independently reviews the evidence and writes a treatment-specific approval manifest under the standing user delegation. No additional per-run user message is required unless the proposed run falls outside the standing scope.

This directive supersedes the old assumption that GPU0 must remain permanently observation-only. It does not authorize modifying or overwriting the completed GPU0 reproduction, and it does not by itself approve a long run.
