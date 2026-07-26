# World manifest evidence tier (CC4 premerge hardening -- five)

- UTC: `2026-07-26T13:26:29Z`
- **GLOBAL_WORLD_RECIPE = PASS** ; **GLOBAL_WORLD_SET_HASH = BLOCKED_SOURCE_UNVERIFIED**
- deliverable tier: **WORLD_RECIPE_MANIFEST + WORLD_INDEX_MANIFEST (spec only)** -- this is NOT a materialized world set.

## Tier table

| Tier | Present | Evidence |
|---|---|---|
| WORLD_RECIPE_MANIFEST | YES | frozen recipe (fold_in(PRNGKey(wrapper_seed=0), world_index); condition_on_task; optimistic_reset_ratio=16; mode=score; bonus_type=none; max_timesteps=4096; num_worlds=256) |
| WORLD_INDEX_MANIFEST | YES | world_index_order=0..255 ascending; world_count_expected=256; world_count_actual=**null** |
| MATERIALIZED_WORLD_SERIALIZATION | **NO** | world_params_materialized=**false**; per_world_params='NOT_GENERATED (JAX absent)' (both sets) |
| MATERIALIZED_WORLD_HASH | **NO** | world_hashes.json status=NOT_GENERATED; per_world_hashes={} (empty); world_set_hash=**null** |

## Recipe hashes (NOT materialized world hashes)
- seed42 world_recipe_hash: `3377049f3e983bfeeacbb219e8c3be0e1d72966d4b1fe2ca72444caadbd1575e`
- seed100000 world_recipe_hash: `0470f9cc7acc1152edaeeda0b688540682b9efcbcc6a48292551f3600521eff9`
- These hash the FROZEN RECIPE SPEC. They are NOT per-world materialized hashes and must NOT be reported as world_set_hash.

## Manifest file integrity (read-only re-check vs SHA256SUMS)
- `world_manifest.json`: MATCH (live `d77f879a8dbafd9e`)
- `world_hashes.json`: MATCH (live `467d079a7f731ce6`)
- `world_generation_provenance.json`: MATCH (live `303b38c4a8232d8b`)
- `world_generation.log`: MATCH (live `374fe5669beef5a7`)

## Git tracking
- world_set_v1 tracked files in current branch: **0** => travels via evidence tar, NOT the git branch.

## used_by (recipe manifest) vs data world (section four) -- carried divergence
- recipe manifest: seed42 used_by = ['Phase2', 'P8', 'P9', 'W512'] ; seed100000 used_by = ['P7', 'LC']
- section-four DATA finding: W512 bakeoff = seed100000, P7 = seed100000, P8 = seed42, P9 = seed42.
- CONFLICT: manifest assigns W512 to seed42 (evaluator docstring seed=42) but the W512 per-world array carries seed100000. This documented divergence is carried forward, NOT resolved here, and must NOT be silently pooled.

## Forbidden claims
- do NOT call the recipe/index manifest a real materialized world
- do NOT present world_recipe_hash as world_set_hash
- do NOT fabricate or placeholder per-world hashes
- do NOT pool seed42 and seed100000 world sets

## Still required (on a JAX+craftax host)
- run materialization on a JAX+craftax==1.4.5 host (env source + SHA verified)
- generate per-world stable serialization for both seed42 and seed100000
- compute per-world SHA256 and total world_set_hash
- run generation TWICE independently and verify agreement (count, order, per-world hash, total hash)
