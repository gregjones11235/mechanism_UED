# World-set materialization runbook (CC4 premerge hardening -- six)

- UTC: `2026-07-26T13:30:17Z`
- **Status: NOT_EXECUTED_THIS_ROUND** (prepared only).
- Reason: JAX/jaxlib/craftax/jaxnav/flax/optax/chex ABSENT on this host; a formal run requires a JAX + craftax==1.4.5 host. A formal run here would **FAIL CLOSED** (verified via --dry-run).

## Script
- Path: `tools/global_evaluation/materialize_world_set_twice.py`
- SHA256: `6631f5ddc8be42a93136919ebd0f135665b168309685c172675a6ac5689289de`
- Placement rationale: placed under `tools/global_evaluation/` (the directive's allowed alternative) rather than the frozen `audit_outputs/global_remediation_20260726T095819Z/tools/`, to avoid changing that frozen directory's SHA inventory semantics. Recorded here per directive.
- Compiles: YES ; dry-run demonstrated: YES (verdict here = VALIDATIONS_INCOMPLETE => formal would fail closed).

## Frozen recipe reference
- `audit_outputs/global_world_set_v1/world_manifest.json` (fold_in(PRNGKey(wrapper_seed=0), world_index); condition_on_task; optimistic_reset_ratio=16; mode=score; bonus_type=none; max_timesteps=4096; num_worlds=256; order 0..255; sets seed42 / seed100000).

## The 17 requirements -> implementation

| # | Requirement | Implementation |
|---|---|---|
| 1 | only generate the world set | main() only builds per-world identity + hashes; no other artifact |
| 2 | do NOT load any checkpoint | no checkpoint path argument; materialize_world_set never opens a checkpoint |
| 3 | do NOT train | no training loop / optimizer step anywhere in the script |
| 4 | do NOT formally evaluate | no evaluator invocation; no success metric computed |
| 5 | frozen seed + fold_in rule | jax.random.fold_in(PRNGKey(wrapper_seed=0), world_index); ALLOWED_SEEDS={seed42,seed100000} |
| 6 | fixed world order | for world_index in range(256) ascending; ordered_hashes preserves order |
| 7 | stable serialization per world | canonical sort_keys JSON descriptor + folded-key bytes |
| 8 | SHA256 per world | per_world[str(world_index)] = sha256(blob) |
| 9 | ordered world_set_hash | world_set_hash = sha256(concat of ascending per-world hashes) |
| 10 | two independent processes | --orchestrate spawns this script twice via subprocess (--single-run) into run_A/run_B |
| 11 | compare per-world hashes | require(a['per_world_hashes'] == b['per_world_hashes']) |
| 12 | compare total hash | require(a['world_set_hash'] == b['world_set_hash']) |
| 13 | record JAX version | ident['jax_version'] / jaxlib_version via probe_version |
| 14 | record Craftax version | ident['craftax_version']; must equal 1.4.5 |
| 15 | record environment source SHA | ident['env_source_sha256'] = sha256(--env-source); required |
| 16 | record generation script SHA | ident['generation_script_sha256'] = sha256(this file) |
| 17 | fail closed on missing version/identity | assert_formal_identity raises FailClosed -> exit 2 |

## Preconditions for a real run
- JAX + jaxlib importable (version recorded)
- craftax==1.4.5 importable (version recorded; mismatch => fail closed)
- environment wrapper source (DistributedMultiTaskOptimisticLogWrapper) available; --env-source points to it (SHA recorded)
- generation script SHA recorded automatically
- no checkpoint, no training, no formal evaluation invoked

## Commands on a JAX host
```
# 0. dry-run first (no hash generated)
python tools/global_evaluation/materialize_world_set_twice.py --seed seed42 --out audit_outputs/world_materialization/seed42 --env-source <path/to/wrapper.py> --dry-run
python tools/global_evaluation/materialize_world_set_twice.py --seed seed100000 --out audit_outputs/world_materialization/seed100000 --env-source <path/to/wrapper.py> --dry-run
# 1. formal two-independent-run materialization (seed42 then seed100000; never pooled)
python tools/global_evaluation/materialize_world_set_twice.py --seed seed42 --out audit_outputs/world_materialization/seed42 --env-source <path/to/wrapper.py> --orchestrate
python tools/global_evaluation/materialize_world_set_twice.py --seed seed100000 --out audit_outputs/world_materialization/seed100000 --env-source <path/to/wrapper.py> --orchestrate
# 2. verify world_set_agreement.json shows per_world_hash_agreement=true and world_set_hash_agreement=true for BOTH seeds
```

## Expected outputs
- `audit_outputs/world_materialization/seed42/run_A/world_hashes.json`
- `audit_outputs/world_materialization/seed42/run_B/world_hashes.json`
- `audit_outputs/world_materialization/seed42/world_set_agreement.json`
- `audit_outputs/world_materialization/seed100000/run_A/world_hashes.json`
- `audit_outputs/world_materialization/seed100000/run_B/world_hashes.json`
- `audit_outputs/world_materialization/seed100000/world_set_agreement.json`

## Recorded in each world_hashes.json
- JAX version; jaxlib version; craftax version (==1.4.5); environment source SHA; generation script SHA; frozen recipe; 256 per-world hashes; ordered world_set_hash.

## Global labels
- GLOBAL_WORLD_RECIPE = PASS (already).
- GLOBAL_WORLD_SET_HASH = BLOCKED_SOURCE_UNVERIFIED now; moves to PASS ONLY after two-independent-run agreement for BOTH seed42 and seed100000.

## Discipline
- NOT executed this round
- no checkpoint loaded, no training, no formal evaluation
- no fabricated/placeholder hashes; fail closed instead
- seed42 and seed100000 generated separately and never pooled/paired
- BLOCKED_SOURCE_UNVERIFIED != FAIL
- does NOT modify the 54 frozen files; does NOT rewrite any SHA256SUMS
