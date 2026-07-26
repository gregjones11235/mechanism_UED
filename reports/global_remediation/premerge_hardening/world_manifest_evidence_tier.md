# World-manifest evidence tier (round-6 revision -- eleven)

- UTC: `2026-07-26T15:10:12Z`
- **OLD_WORLD_KEY_PROTOTYPE = INVALID_FOR_WORLD_SET_HASH**
- **CRAFTAX_WORLD_MATERIALIZER_CODE = IMPLEMENTED** ; **RUNTIME_VALIDATION = NOT_RUN**
- **GLOBAL_WORLD_RECIPE = PASS** ; **GLOBAL_WORLD_SET_HASH = BLOCKED_SOURCE_UNVERIFIED**
- deliverable tier = WORLD_SET_MATERIALIZER_IMPLEMENTED_STATIC (NOT materialized; NOT ready for world_set_hash) ; is_materialized_world = **False**

## What changed in round 6
- located the REAL canonical world-generation path (FOUND) with line+SHA anchors
- implemented an actual stable serializer + actual-reset materializer (static; fails closed without JAX/craftax)
- corrected the seed semantics (evaluation_seed enters the real split chain; old prototype did NOT bind it)
- added the materialization gate that REJECTS key-only output
- downgraded every over-claim: code done does NOT upgrade the world hash

- the former materialize_world_set_twice.py hashed ONLY a fold_in PRNG key + a recipe descriptor; renamed (git mv) to world_key_manifest_prototype.py with a DEPRECATED_INVALID_FOR_WORLD_SET_HASH banner; entry point fails closed (exit 2); history preserved.
- only a frozen recipe/index manifest + a now-deprecated key prototype existed; NO materialized world serialization or world_set_hash. That remains true: GLOBAL_WORLD_SET_HASH stays BLOCKED_SOURCE_UNVERIFIED.
