#!/usr/bin/env python
"""CANONICAL_EVALUATOR_V1 world-manifest builder.

Builds canonical_worlds_256_seed<SEED>.json binding the FULL world-generation recipe so that an identical
manifest is bit-reproducible under an identical (env_version, world_generator_sha, evaluator_sha).

TWO modes:
  --recipe-only   (default in a JAX-less env): emit the deterministic INPUTS (world_index, base_seed,
                  fold_in rule, full RNG-input derivation, generator/evaluator SHA, env_version). The
                  world_set_hash is left null and a world_recipe_hash (over the recipe) is recorded.
                  world_params_materialized=False. This is HONEST: worlds are not materialized here.
  --materialize   (requires JAX+Craftax): actually run env.reset for all 256 worlds, record per-world
                  task/params, and compute world_set_hash = sha256(canonical_json(materialized worlds)).
                  GATE2 (reproducible) and GATE3 (order-sensitivity) become checkable.

This file is a reference tool. In this audit env JAX is ABSENT, so only --recipe-only is runnable; the
materialized manifest + world_set_hash are produced on a JAX/Craftax host (future authorized run).
"""
import argparse, hashlib, json, os, sys

NUM_WORLDS = 256

def fold_in_rule(key, i):
    """Documented RNG fold rule (matches jax.random.fold_in semantics)."""
    return f"fold_in(base_key, world_index={i})"

def build_recipe(base_seed, wrapper_key_seed, generator_sha, evaluator_sha, env_version):
    worlds = []
    for i in range(NUM_WORLDS):
        worlds.append({
            "world_index": i,
            "base_seed": base_seed,
            "wrapper_prng_seed": wrapper_key_seed,
            "fold_in": fold_in_rule("PRNGKey(%d)" % wrapper_key_seed, i),
            "rng_input": "reset_rng = split(PRNGKey(base_seed)); obsv,log = env.reset(reset_rng, ctor) [world %d]" % i,
            "world_params": None,   # populated only in --materialize
            "task_params": None,     # populated only in --materialize
        })
    recipe = {
        "protocol": "CANONICAL_EVALUATOR_V1",
        "num_worlds": NUM_WORLDS,
        "base_seed": base_seed,
        "wrapper_prng_seed": wrapper_key_seed,
        "fold_in_rule": "jax.random.fold_in(PRNGKey(wrapper_prng_seed), world_index)",
        "world_generator_sha256": generator_sha,
        "evaluator_sha256": evaluator_sha,
        "env_version": env_version,
        "paired_by": "world_index (identical index across arms => paired)",
        "worlds": worlds,
    }
    return recipe

def canonical_json_bytes(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wrapper-key-seed", type=int, default=0)
    ap.add_argument("--generator-sha", default="UNVERIFIED")
    ap.add_argument("--evaluator-sha", default="224514026aefd273a6647e055fc2e1a434760dc5f4b6b9acd0624bbba57035a1")
    ap.add_argument("--env-version", default="craftax==1.4.5(EXPECTED; verify on host)")
    ap.add_argument("--materialize", action="store_true",
                    help="requires JAX+Craftax; materializes worlds and computes world_set_hash")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    recipe = build_recipe(a.seed, a.wrapper_key_seed, a.generator_sha, a.evaluator_sha, a.env_version)
    recipe["world_recipe_hash"] = hashlib.sha256(canonical_json_bytes(recipe)).hexdigest()

    if a.materialize:
        try:
            import jax, craftax  # noqa
        except Exception as e:
            print("MATERIALIZE BLOCKED: JAX/Craftax absent (%s). Emitting recipe-only." % e)
            a.materialize = False
    if a.materialize:
        # Real materialization would run the wrapper here and fill world_params/task_params, then:
        # recipe["world_set_hash"] = sha256(canonical_json(materialized))
        raise SystemExit("materialization path must be completed on a JAX/Craftax host; not implemented in audit env")
    else:
        recipe["world_set_hash"] = None
        recipe["world_params_materialized"] = False
        recipe["status"] = "RECIPE_ONLY (JAX/Craftax absent in this env; world_set_hash requires materialization on host)"
        recipe["gate2_reproducible"] = "NOT_VERIFIED (requires materialization)"
        recipe["gate3_order_sensitive"] = "NOT_VERIFIED (requires materialization)"

    out = a.out or ("canonical_worlds_%d_seed%d.json" % (NUM_WORLDS, a.seed))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(recipe, f, indent=2, ensure_ascii=False)
    sh = hashlib.sha256(open(out, "rb").read()).hexdigest()
    with open(out + ".sha256", "w", encoding="utf-8") as f:
        f.write(sh + "  " + os.path.basename(out) + "\n")
    print("wrote", out, "recipe_hash", recipe["world_recipe_hash"][:16], "world_set_hash", recipe["world_set_hash"])

if __name__ == "__main__":
    main()
