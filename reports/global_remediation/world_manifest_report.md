> GLOBAL_EVALUATION_REMEDIATION (CC4) · repo mechanism_UED @ Henry-branch · HEAD=UNVERIFIED (git access BLOCKED)
> Environment: JAX/Craftax ABSENT, experiment server UNREACHABLE, GPU idle (not used), no CC2/CC3 processes.
> Discipline: read-only w.r.t. CC2/CC3 & old audit; NEW_TRAINING_RUNS=0; MISSING/UNVERIFIED never relabeled FAIL.

# World Manifest Report

Two DISTINCT frozen world sets (never pooled): seed42 (Phase2/P8/P9/W512) and seed100000 (P7/LC).
Recipe: `jax.random.fold_in(PRNGKey(wrapper_seed), world_index)`, full RNG inputs, canonical JSON.

- world_recipe_hash (seed42): `3377049f3e983bfeeacbb219e8c3be0e1d72966d4b1fe2ca72444caadbd1575e`
- world_params_materialized: False (needs JAX host)
- world_set_hash: None → GATE3 = PARTIAL (BLOCKED_ON_JAX, NOT FAIL)

tools/build_world_manifest.py runs recipe-only by default; `--materialize` computes the true world_set_hash
on a JAX host. Until then every reported number carries world_recipe_hash and a `world_set_hash=REQUIRED` flag.
