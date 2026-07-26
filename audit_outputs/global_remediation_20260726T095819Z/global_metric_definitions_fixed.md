# Global Metric Definitions (FIXED) — CANONICAL_EVALUATOR_V1

Supersedes the audit-era metric definitions for all NEW reports. Old reports unchanged (read-only).

## Outcome metrics (unchanged, from Phase2 anchor)
- success = seen_target OR (info_accuracy>0), ever-set; denominator = num worlds (256).
- SR = n_success / N (pp = SR*100). floor3_reach = (max_floor>=3). died/timeout/not_finished partition (sum==N).
- conditional_kill = n_success / n_floor3.

## Achievement / tier metrics (FIXED)
- Single source of truth: `craftax.craftax.constants.ACHIEVEMENT_REWARD_MAP` (craftax==1.4.5), 67 achievements, ids 0-66.
- Official tiers (reward weight): BASIC=1 (25), INTERMEDIATE=3 (18), ADVANCED=5 (15), VERY_ADVANCED=8 (9). See `official_achievement_tiers.json`.
- New reports MUST emit: official_tier_name, official_reward_weight, achievement list, mapping source SHA.
- The design-layer `ACHIEVEMENT_DEPTH` is NOT official. It is retained ONLY as `CUSTOM_DEPTH_TIER` and MUST NOT be mixed into official-tier tables or used as 'tier3'.
- 'tier3' in any baseline cross-comparison MUST mean official ADVANCED (reward 5, 15 items). Other uses are invalid.
- New vs old tier results MUST NOT be placed in the same table.

## Baseline identity (FIXED)
- Always cite a `baseline_id` (TEACHER17500_BASELINE | CONTROL24576_BASELINE). Never bare "Baseline".
- Any percentage-point delta MUST verify identical: checkpoint, evaluator SHA, world_set_hash, success definition, denominator, action_mode. Mismatch => PAIRED_COMPARISON_NOT_ALLOWED.

## World identity (FIXED)
- world_set_hash MUST be recorded per run; paired comparison requires identical world_set_hash AND evaluator SHA (GATE11).
- seed42 (Phase2/P8/P9/W512) and seed100000 (P7/LC) are distinct world sets — never pooled.
- This env is JAX-less: the canonical world manifest is RECIPE-ONLY (world_recipe_hash=3377049f3e983bfe...); materialized world_set_hash requires a JAX/Craftax host (GATE2/3 NOT_VERIFIED here).

## Statistics (unchanged)
- Primary: paired McNemar (discordant counts, same 256 worlds) + paired bootstrap 95% CI (fixed seed).
- Signal: p<0.05 AND bootstrap CI not crossing 0. Wilson + Clopper-Pearson per arm. Collapsed-regime comparisons INVALID regardless of p.
