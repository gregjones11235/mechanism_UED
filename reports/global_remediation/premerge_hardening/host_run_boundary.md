# Host run boundary (round-6 -- twelve)

- UTC: `2026-07-26T15:10:12Z`
- host: jax=False, craftax=False (audit host has neither)

## Only allowed on this host
- python -m compileall (PASS)
- static source review / line+SHA anchoring (world_generation_path_audit)
- pure-Python serializer unit tests -- self-test PASS (12 checks)
- serializer mock tests (negative tests use numpy stand-ins for arrays)
- import / source analysis + static anchor-check vs real canonical sources (PASS, 12 anchors)
- dry-run / fail-closed confirmation of a formal run (single-run/orchestrate exit 2)

## Forbidden on this host
- deriving MATERIALIZER_RUNTIME_PASS from a mock/fake world
- emitting ANY world_set_hash
- asserting GATE18 (seed enters real RNG) as PASS
- asserting TWO_PROCESS_REAL_WORLD_AGREEMENT as anything but NOT_RUN

## Formal run fails closed (proof)
- single-run exit code = **2** ; no output dir created = True
- message: `FAIL CLOSED: real Craftax world materialization requires JAX AND craftax (jax=False, craftax=False)`

## Correct labels on this host
- MATERIALIZER_STATIC_TESTS = **PASS**
- MATERIALIZER_REAL_CRAFTAX_RUN = **NOT_RUN**
- CRAFTAX_WORLD_MATERIALIZER_REAL_RUN = **NOT_RUN**
- EVALUATION_SEED_REAL_RNG_BINDING = **PARTIAL_ENVIRONMENT_BLOCKED**
- TWO_PROCESS_REAL_WORLD_AGREEMENT = **NOT_RUN**
- GLOBAL_WORLD_SET_HASH = **BLOCKED_SOURCE_UNVERIFIED**
