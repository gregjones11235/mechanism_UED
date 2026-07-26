# D052 Canonical Legacy Reuse Map

- Baseline commit: `a2726e3ea75feff2b475b1e3408c30ef3f9acd7a`
- Branch: `henry/d052-canonical-refactor`
- Freeze timestamp (UTC): `20260726T060626Z`
- Anchor evidence: `audit_outputs/d052_legacy_source_freeze_20260726T060626Z/source_inventory.json` + `SHA256SUMS` (48 files, `sha256sum -c` verified)

This map classifies every D052-relevant legacy module so teammates know exactly what to reuse, wrap, refactor, or avoid. It is the first deliverable of the canonical refactor (task §六). Dispositions:

- **REUSE_AS_IS** — Import/call unchanged; may be wrapped by an adapter but its behavior is canonical.
- **REUSE_WITH_ADAPTER** — Reuse the logic but wrap/strip (e.g. /root paths, prompt/manifest reconciliation) before canonical use.
- **REFACTOR** — Substantially rewrite (silent fallbacks, dup paths, hardcoded identity); keep intent, change structure.
- **DEPRECATE** — Superseded / duplicated; do not use in canonical_v2, keep for reference only.
- **DO_NOT_USE** — Voided legacy experiment code (raw_sources); reference/mine only, never a dependency.

## Summary counts

| Disposition | Count |
|---|---|
| REUSE_AS_IS | 49 |
| REUSE_WITH_ADAPTER | 20 |
| REFACTOR | 6 |
| DEPRECATE | 8 |
| DO_NOT_USE | 2 |

## Frozen source files — REUSE_AS_IS (25)

| Path | Blob SHA1 | Subsystem/role | Rationale |
|---|---|---|---|
| `dicode_src/auction/craftax_achievements.py` | `5bb881a60a36` | canonical_67_single_source | Hand-maintained, ZERO external deps; guarded assert NUM_ACHIEVEMENTS==67, max value==66, len(ALL_ACHIEVEMENTS)==67; _ACHIEVEMENTS_ORDERED collect_wood=0..defeat_archer=66; verified vs craftax main+v1.4.5. Identical blob in dicode_v6. |
| `dicode_src/src/dicode/task_utils.py` | `1df5caf33b27` | conditioning_path | EMBEDDING_SIZE=67; get_achievement_multi_hot sets embedding[ach.value]=1.0; proves canonical_id==goal_vector_index==enum .value. |
| `dicode_src/src/minicraftax/envs/multitask.py` | `8c38bc5cb76e` | conditioning_path | obs = symbolic_obs + task_vector; obs_dim = parent.shape[0] + task_vector_size; task_vector_size=embedding_size=67. Identical blob in v6. |
| `dicode_v6/auction/ambition.py` | `00b87c1f8051` | gate | Ability gate; tests 85725fe9. |
| `dicode_v6/auction/completed_gate.py` | `93e65e4b4535` | gate | _parse_names keep-only-legal filter; completed-achievement gate. |
| `dicode_v6/auction/craftax_achievements.py` | `5bb881a60a36` | canonical_67_single_source_dup | Byte-identical to dicode_src copy (same blob 5bb881a6). |
| `dicode_v6/auction/level_meta.py` | `9cea40bb5eee` | candidate_schema | <level_meta> JSON {type:DEPTH|BREADTH|CONSOLIDATE, drill_target, siege_wall}; tolerant parser; de-facto difficulty tier (no difficulty_tier field exists anywhere). |
| `dicode_v6/auction/mock_proposals.py` | `0b30b5a28b4d` | candidate_generation_fixture | Offline synthetic Proposal factory; samples only legal names; test fixture. |
| `dicode_v6/auction/modeler.py` | `07c8cee428a5` | modeler | 726 LOC single GLM STUDENT MODELER; runs once/session before proposers; classifies NORMAL_EARLY/RISING/STALLED/NOISY/FORGETTING/MASTERED; recommends DEPTH/BREADTH/CONSOLIDATE; _build_state_evidence hands facts not verdicts; evidence_check supported/contradicted/no_evidence. |
| `dicode_v6/auction/proposal.py` | `9896d4baeafd` | candidate_schema | Canonical candidate contract: Proposal(frozen) proposal_id/proposer_id/parent_task_id/docstring/reasoning/achievements:frozenset/skill_tags/self_report; __post_init__ raises on achievements - ALL_ACHIEVEMENTS (hard backstop). Identical in dicode_src + archive. |
| `dicode_v6/auction/student_profile_log.py` | `edab25aabb05` | student_profile | SAME class name different API: append-only snapshot time series, latest()/recent(k)/forgetting_candidates() with COMBAT-family split; the profile the Modeler consumes. Must reconcile with siege/student_profile. |
| `dicode_v6/conf/config.yaml` | `5a29ea393e7a` | config_entry | Root defaults (6 groups), chdir, wandb globals, seed:42. |
| `dicode_v6/conf/gen_manager/auction_c_v6siege.yaml` | `e587e26ac89b` | config_entry | Current method: v6 + siege:true; cost-saving providers. |
| `dicode_v6/conf/training/default.yaml` | `43a879a0940e` | config_entry | PPO+transformer; total_timesteps:2_005_401_600 (long-run budget). |
| `dicode_v6/experiments/training/run_dicode.py` | `8dbfed995bbc` | run_entry | Canonical entry; @hydra.main(version_base=1.2, config_path=../../conf/, config_name=config); zero argparse flags (all Hydra dot-overrides). |
| `dicode_v6/src/dicode/dreaming/gen_manager.py` | `0e1509e522a5` | candidate_generation | 2911 LOC live method head: persona loading, coop proposer loop, _query_and_parse_responses retry, _siege_validate_and_reroll, EnvGenerator reflection loop (infinite re-queue until compile). |
| `dicode_v6/src/minicraftax/craftax_state.py` | `550a9c07d6ba` | candidate_schema | TaskParams = the task_params field: 12 knobs (spawn/health/damage multipliers, melee_trigger_distance, monsters_killed_to_clear_level, ...). |
| `dicode_v6/src/minicraftax/envs/multitask.py` | `8c38bc5cb76e` | conditioning_path_dup | Byte-identical to dicode_src multitask (same blob 8c38bc5c). |
| `dicode_v6/src/minicraftax/tasks/base_task.py` | `16c5024faccc` | candidate_schema | Runtime contract: relevant_achievements/completed_achievements/label; success = all relevant done. |
| `gpu1_aggregation_siege/conf/llm/providers.yaml` | `7897ca6e019f` | config_entry | New-style provider registry (schema_version:1, economy/balanced/strong). |
| `gpu1_aggregation_siege/conf/llm/roles/curriculum_roles.yaml` | `dd0be62ec916` | config_entry | Role registry + output schemas (role_judgment_v2). |
| `gpu1_aggregation_siege/src/dicode/mechanisms/auction.py` | `ec3517288f35` | selector_path | 298 LOC; run_auction_selection auction_type raw/budgeted; build_role_utilities_from_signals. |
| `gpu1_aggregation_siege/src/dicode/mechanisms/llm_costs.py` | `4e64ee91368c` | cost_recording | 98 LOC; LLMCostTracker budget caps, per-provider/per-role token+USD -> llm_cost_log.jsonl. Minor: can_call() hardcodes $0.0006/1k. |
| `gpu1_aggregation_siege/src/dicode/siege/siege_notebook.py` | `e249c08816f1` | student_profile | Orchestrator: profile->chain_order->held_out->focus_quota->rehearsal; emits only binary mastery flags + evidence_source held_out_evaluation. |
| `gpu1_aggregation_siege/src/dicode/siege/student_profile.py` | `26e493fff087` | student_profile | StudentProfileLog tier engine: tier 0-4 from held-out SR (0.1/0.5/0.8/0.95), is_mastered/is_proficient, get_forgetting_risk; docstring mandates tiers NEVER passed to LLM/selector. Byte-identical dup in gpu0 -> dedup. |

## Frozen source files — REUSE_WITH_ADAPTER (11)

| Path | Blob SHA1 | Subsystem/role | Rationale |
|---|---|---|---|
| `dicode_src/src/dicode/evaluation/online_evaluation.py` | `cab56b77b00a` | eval_mapping | :229 Achievement[skill_name_raw.upper()].value; exact, case-insensitive, no alias. v6 copy carries hardcoded author-home @hydra config_path (unusable as entry; run_session_evaluation is the real API). |
| `dicode_v6/src/dicode/dreaming/auction_integration.py` | `9a6b382955f7` | candidate_generation | DiCode<->auction adapter; parse_relevant_achievements does SILENT keep-only-legal drop of unknown names (stage-1 pre-filter); profile_to_target_gap omits unknown skills silently -> canonical_v2 must error, not drop. |
| `gpu1_aggregation_siege/conf/aggregation/default.yaml` | `8c2e02148f4b` | config_entry | Modes A1-A6, blend weights; hardcoded /root cache path -> strip. |
| `gpu1_aggregation_siege/scripts/run_data_plane_preflight_v3.py` | `68b0e25f16b3` | gate | 13-stage hard-gate preflight pipeline; argparse --mechanism {original,soft_copeland,budgeted_copeland,auction_raw,auction_budgeted}; 'Chain-rejected candidates NEVER enter pool'. |
| `gpu1_aggregation_siege/scripts/run_gate_r0_final.py` | `6ca70cf121d4` | gate | argparse --mechanism --steps (16384); R0 production gate. |
| `gpu1_aggregation_siege/src/dicode/mechanisms/aggregation.py` | `92a7e8b6c74d` | selector_path | 775 LOC; modes raw_weighted/robust_weighted/soft_copeland/budgeted_soft_copeland/budgeted_retention_trigger/entropy_regularized; entry select_tasks_with_aggregation; hardcoded /root diag paths to strip. |
| `gpu1_aggregation_siege/src/dicode/mechanisms/llm_providers.py` | `35f34216bc72` | role_protocol | 222 LOC; urllib client; single attempt NO retry despite manifest retry_count:3; static ROLE_PROVIDER_MAP. |
| `gpu1_aggregation_siege/src/dicode/mechanisms/llm_roles.py` | `d6f3ff850c11` | role_protocol | 257 LOC; strict-JSON judge prompts tutor/critic/explorer; prompt templates hardcode qwen-turbo/deepseek-chat conflicting with manifest -> reconcile. |
| `gpu1_aggregation_siege/src/dicode/selection.py` | `8040b3202416` | selector_path | 310 LOC; PLR + aggregation hook. Identical blob in gpu0. |
| `gpu1_aggregation_siege/src/dicode/siege/aggregation_integration.py` | `57b0857bcf52` | candidate_pool | 305 LOC; build_siege_candidate_pool target 32; chain_completeness_gate ('not a weighted preference'). |
| `gpu1_aggregation_siege/tests/test_v3_hard_gates.py` | `efa584d95ec2` | gate_test | Tests the 13-stage gate pipeline; only place literal target_achievements ctor arg appears (test-only envs). |

## Frozen source files — REFACTOR (4)

| Path | Blob SHA1 | Subsystem/role | Rationale |
|---|---|---|---|
| `dicode_v6/src/dicode/evaluation/online_evaluation.py` | `c82d8415b851` | eval_mapping | Same eval mapping but @hydra.main config_path hardcoded to original author's home -> unusable as CLI entry; refactor entry, keep mapping. |
| `gpu1_aggregation_siege/src/dicode/mechanisms/immutable_cache.py` | `9baf861c0bcb` | judgment_cache | 404 LOC; v2 cache, >=95% hit gate, mutation detection; key = sha256(task_code)[:16]_student_stage_id_role_provider_exact_model_prompt_version_schema_version. Dup at src/dicode/immutable_cache.py. |
| `gpu1_aggregation_siege/src/dicode/mechanisms/model_manifest.py` | `97451e80abcd` | role_protocol | 413 LOC; pinned manifest v2.1 aggregation-v2-gpu1; compute_manifest_hash; validate_manifest. Duplicate at src/dicode/model_manifest.py (same blob) -> dedup. |
| `gpu1_aggregation_siege/src/dicode/siege/production_dispatcher.py` | `4b380586809e` | candidate_pool | 332 LOC; ALL_MECHANISMS=[original,soft_copeland,budgeted_copeland,auction_raw,auction_budgeted]; compile_candidate. SILENT default-goal fallback (all_achs[:2]) + loose CandidateEnv schema + /root paths -> fail-closed + align to Proposal/TaskParams. |

## Frozen source files — DEPRECATE (6)

| Path | Blob SHA1 | Subsystem/role | Rationale |
|---|---|---|---|
| `dicode_src/auction/modeler.py` | `3d942434d702` | modeler_old | 372 LOC older v5-only modeler. |
| `dicode_src/auction/student_profile_log.py` | `fbbdc74c2537` | student_profile_old | Older, no combat-family split. |
| `gpu1_aggregation_siege/conf/training/default.yaml` | `da9253649b6a` | config_entry_deprecated_conditioning | L59 conditioning_type:'one_hot' -> the 32-slot one-hot conditioning that must NOT be the canonical training interface (canonical=achievement_multi_hot, dim 67). |
| `gpu1_aggregation_siege/src/dicode/immutable_cache.py` | `9baf861c0bcb` | duplicate_of_mechanisms_immutable_cache | Byte-identical dup (blob 9baf861c); dedup. |
| `gpu1_aggregation_siege/src/dicode/mechanisms/llm_cache.py` | `30d7b64ef13c` | judgment_cache_v1 | 185 LOC; v1 cache, fragile summary-hash key (self-described); superseded by immutable_cache v2. |
| `gpu1_aggregation_siege/src/dicode/model_manifest.py` | `97451e80abcd` | duplicate_of_mechanisms_model_manifest | Byte-identical dup (blob 97451e80); dedup to one import path. |

## Frozen source files — DO_NOT_USE (2)

| Path | Blob SHA1 | Subsystem/role | Rationale |
|---|---|---|---|
| `experiments/henry_dicode_student_upgrade/01_d052/raw_sources/home/oseasy/experiments/d052_unified_eval_20260722/evaluator/obsdim_probe.py` | `0789ade345b4` | legacy_d052_experiment_entry | Carries the voided D052 candidate schema; candidate_hash=sha256(f'{id}:{sorted(names)}')[:16]. Reference/mine only; raw_sources is a vendored server snapshot, NOT a dependency. The 'onehot-32 obs8300' conditioning that fails 67-dim goal-embedding lives here. |
| `experiments/henry_dicode_student_upgrade/08_p9_authentic_reset/raw_sources/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu3_p9_authentic_reset/src/run_p9_authentic_98304.py` | `09cd4afefae2` | legacy_d052_obsdim_assert | assert obs_dim==8335 and EMB==67 — in-repo confirmation of the canonical obs_dim. Reference only (raw_sources vendored snapshot). |

## Extended catalog — REUSE_AS_IS (24)

| Path | Blob SHA1 | Subsystem/role | Rationale |
|---|---|---|---|
| `dicode_v6/auction/behavior_fingerprint_log.py` | `42caa5c0b2d3` | profiling | Deterministic behav_hint evidence log grounding the modeler. |
| `dicode_v6/auction/cooccurrence_log.py` | `7b4f6f3bb88d` | profiling | Deterministic cooc_hint evidence log grounding the modeler. |
| `dicode_v6/auction/pricing.py` | `d4847b6e272a` | selectors.auction | Walrasian shadow prices per achievement (curriculum economics), NOT llm_costs. |
| `dicode_v6/conf/dicode_manager/default.yaml` | `35576b7ad7ab` | config | Orchestrator: max_updates_per_session, training_sample_size_n:16, active_task_capacity:100. |
| `dicode_v6/conf/evaluation/default.yaml` | `fdc6bcb3776b` | config | Eval 8192x1024 envs. |
| `dicode_v6/conf/gen_manager/auction_c_v6.yaml` | `e515cab35055` | config | v6 cost-saving providers. |
| `dicode_v6/conf/gen_manager/default.yaml` | `4b846f77d869` | config | Baseline; all auction/coop/siege switches default OFF; 4 LLM roles. |
| `dicode_v6/conf/gen_manager/llm/dashscope.yaml` | `4b2ea722754c` | config | Per-provider model/base_url/think/sampling (representative of 11 providers). |
| `dicode_v6/src/dicode/dreaming/prompts/cl_/evolve_mastered_r.py` | `ba795811d225` | generation.prompts | Reflection-loop template (PREVIOUS_RESPONSE/ERROR/TASK_DESC). |
| `dicode_v6/src/dicode/dreaming/prompts/cl_/minicraftax_coder.py` | `dbdf6f0df29e` | generation.prompts | Env-codegen prompt; FM sets self.relevant_achievements, returns TaskParams(...). |
| `dicode_v6/src/dicode/dreaming/prompts/dicode/ablation.py` | `bd1b37ab4deb` | generation.prompts | Ablation proposer (random bottleneck, no parent metrics). |
| `dicode_v6/src/dicode/dreaming/prompts/dicode/constants.py` | `00317b07f715` | generation.prompts | Knowledge-base blocks injected via system_prompt.format(CONSTANTS=,MOBS=,...). |
| `dicode_v6/src/dicode/dreaming/prompts/dicode/evolve.py` | `0fbe0c3507ab` | generation.prompts | Baseline single-designer system prompt; blob-identical across packages. |
| `dicode_v6/src/dicode/dreaming/prompts/dicode/persona_ambitious.py` | `47e221b74512` | generation.prompts | PROPOSER-AMBITIOUS (deep prerequisite chains). |
| `dicode_v6/src/dicode/dreaming/prompts/dicode/persona_breadth.py` | `4939bec8c649` | generation.prompts | PROPOSER-BREADTH (+ ARCHIVE_FAMILY_COVERAGE slot). |
| `dicode_v6/src/dicode/dreaming/prompts/dicode/persona_feasible.py` | `61e456579022` | generation.prompts | PROPOSER-FEASIBLE (ability edge p~0.5). |
| `gpu1_aggregation_siege/conf/llm/experiments/role_rotation_a.yaml` | `37836d4576fe` | config | Role->provider assignment arm A (between-experiment rotation, not per-round). |
| `gpu1_aggregation_siege/conf/llm/experiments/role_rotation_b.yaml` | `00347c7043b6` | config | Role->provider assignment arm B. |
| `gpu1_aggregation_siege/src/dicode/dreaming/gen_manager.py` | `382b9065f7c3` | generation.gen_manager | Pre-auction DiCode baseline (control arm); identical to gpu0 (blob 382b9065). |
| `gpu1_aggregation_siege/src/dicode/siege/focus_quota.py` | `bbbe803a67bc` | profiling | Min chain-task quota gate. |
| `gpu1_aggregation_siege/src/dicode/siege/held_out.py` | `0ff8f38a0ed6` | profiling | Held-out success-rate recorder. |
| `gpu1_aggregation_siege/src/dicode/siege/rehearsal.py` | `e9031916d5d5` | profiling | Forgetting rehearsal. |
| `gpu1_aggregation_siege/tests/test_data_plane_integrity.py` | `739a4a5b924d` | integrity | Data-plane integrity tests. |
| `gpu1_aggregation_siege/tests/test_siege_components.py` | `b277f586b25d` | integrity | Siege component tests. |

## Extended catalog — REUSE_WITH_ADAPTER (9)

| Path | Blob SHA1 | Subsystem/role | Rationale |
|---|---|---|---|
| `dicode_src/src/dicode/dreaming/gen_manager.py` | `2b57b790d81d` | generation.gen_manager | 2264 LOC; same as v6 minus siege/level_meta/validator reroll (older). |
| `dicode_v6/src/dicode/dreaming/prompts/cl_/gen_env.py` | `5f629060c17b` | generation.prompts | Legacy GenEnv-style env-gen prompt. |
| `dicode_v6/src/dicode/dreaming/prompts/dicode/persona_ambitious_coop.py` | `1d8863938ac1` | generation.prompts | v5-debate coop persona (MODELER_GUIDANCE/PEER_ALREADY_MADE/REFERENCE_LEVEL/MY_TURN_ORDER); v6 canonical, dicode_src copy older. |
| `dicode_v6siege.sh` | `670d89730c9d` | launch | SLURM v6 SIEGE method run (gen_manager=auction_c_v6siege, siege:true). |
| `gpu1_aggregation_siege/conf/config.yaml` | `fd9cd7a08d1d` | config | Root + extra group aggregation:default. |
| `gpu1_aggregation_siege/conf/gen_manager/default.yaml` | `4fcddc1a1892` | config | Stripped baseline (no auction switches). |
| `gpu1_aggregation_siege/scripts/run_aggregation_sweep.sh` | `bcbf0cb9eaeb` | launch | Mode x trigger sweep; timestamped log dirs; strip /root paths + dedup vs gpu0. |
| `gpu1_aggregation_siege/src/dicode/mechanisms/diagnostics.py` | `cd591081c575` | integrity | Diagnostics writer; hardcoded /root diag paths -> make path-configurable. |
| `gpu1_aggregation_siege/src/dicode/siege/chain_order.py` | `7d9c31a05ea5` | profiling | Prerequisite chains, mastered-link & break-link detection. |

## Extended catalog — REFACTOR (2)

| Path | Blob SHA1 | Subsystem/role | Rationale |
|---|---|---|---|
| `dicode_v6siege_style.sh` | `800311cb090f` | launch | Copy-paste of siege launcher with different run identity -> parametrize. |
| `migration_launchers/gpu0/t1_lpg_hrl_16384.py` | `7bc3473851ce` | launch | One-shot R0 launcher: hardcoded /root sys.path + ad-hoc config objects -> config-driven preflight. |

## Extended catalog — DEPRECATE (2)

| Path | Blob SHA1 | Subsystem/role | Rationale |
|---|---|---|---|
| `dicode_v6/dicode_v6.sh` | `b69babfa11c4` | launch | Older in-package copy, stale endpoints. |
| `gpu0_training_mechanisms/src/dicode/siege/student_profile.py` | `26e493fff087` | profiling | Byte-identical dup of gpu1 student_profile.py (blob 26e493f) -> dedup. |

## Key structural findings

1. **Unified-package home = `gpu1_aggregation_siege`** (strict superset of gpu0: carries `mechanisms/auction.py`, `siege/production_dispatcher.py`, `scripts/`, `tests/`). The canonical `d052/` framework lives at `gpu1_aggregation_siege/d052/` (top-level, separate from `src/dicode/` to avoid the four-package `import dicode` collision).
2. **Canonical-67 single source CONFIRMED**: `dicode_src/auction/craftax_achievements.py` (blob `5bb881a6`), zero-dep, `assert ==67`. canonical_id == goal_vector_index == enum value (task_utils.py).
3. **obs_dim 8335 = base 8268 + 67 multi-hot CONFIRMED**; the legacy `one_hot`/`obs_dim 8300`/`32-slot` interface is NOT in current HEAD (`gpu1/conf/training/default.yaml` L59 `conditioning_type:'one_hot'` is the only conditioning to deprecate) and is banned from canonical_v2.
4. **Silent fallbacks to eliminate**: `production_dispatcher.compile_candidate` default goal `all_achs[:2]`; `auction_integration.parse_relevant_achievements` silent keep-only-legal drop. canonical_v2 turns both into hard errors.
5. **Dedups**: `model_manifest.py` / `immutable_cache.py` each duplicated at two import paths (byte-identical); two `StudentProfileLog` classes (`siege/student_profile.py` tier engine vs `auction/student_profile_log.py` snapshot history) need reconciliation.
6. **No cell scheme exists** in live code (only voided d052 dir-name cells) — the cell registry is built fresh in `d052/cells/`.

