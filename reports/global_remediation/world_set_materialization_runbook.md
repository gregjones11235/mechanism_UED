# World-set materialization runbook (round-6 revision -- eleven)

- UTC: `2026-07-26T15:10:12Z`
- status: **IMPLEMENTED_STATIC; REAL_RUN_NOT_EXECUTED; GLOBAL_WORLD_SET_HASH=BLOCKED_SOURCE_UNVERIFIED**
- script: `tools/global_evaluation/materialize_craftax_world_set_twice.py` ; deprecated: `tools/global_evaluation/world_key_manifest_prototype.py (DO NOT USE; fails closed exit 2)`
- canonical reset path: `split(split(split(PRNGKey(evaluation_seed))[1])[1],256)[world_index] then reset_env split (multitask:129) + generate_world split (s4:39) -- PURE split, NO fold_in`
- serializer schema: `mechanism_UED.craftax_materialized_world/v1 (full 53-field initial EnvState snapshot; arrays bind dtype+shape+C-order bytes; sorted keys; no pickle)`

## Revised preconditions for a REAL run
- 1. evaluator + world-builder SOURCE IDENTITY confirmed: eval_phase2_unified.py sha256 224514026aefd273...; wrapper byte-identical 2ded41d8...; task canonical 45fdd17c... (NOT the P2-v0 invalid df7cde78...); env multitask.py c8f2d5c3...
- 2. the ACTUAL Craftax reset path is the canonical split chain above (verified line-by-line); NO fold_in; materializer reproduces the whole 256-way batch (env.reset(reset_rng) once, then index [i])
- 3. serializer schema = mechanism_UED.craftax_materialized_world/v1 serializes the COMPLETE initial EnvState (53 fields); no result-affecting initial field dropped
- 4. seed-semantics tests: label-only change -> hash unchanged (PASS); numeric seed change -> real RNG change (must be re-asserted on the JAX host; BLOCKED_ENVIRONMENT here); GATE18 must reach PASS only on a real host
- 5. two INDEPENDENT processes agree (do_orchestrate): count/index order/per-world/total/source SHA/versions/numeric seed all equal; any diff -> fail closed (TWO_PROCESS_REAL_WORLD_AGREEMENT=NOT_RUN here)
- 6. negative tests (10) FAIL=0 (PASS=8 BLOCKED=2 here); the 2 BLOCKED must convert to PASS only on a real host, never faked
- 7. JAX + craftax==1.4.5 host; CC4_S4_TASK_PATH set to the canonical s4_task_code.py (sha prefix 45fdd17c)
- 8. ALL OUTPUT REVIEWED BY 总控 before any world_set_hash is accepted into evidence

## Over-claim corrections
- WORLD_SET_MATERIALIZER_READY (round-5 implicit) -> WRONG; was only WORLD_KEY_MANIFEST_PROTOTYPE
- key hash == world hash -> WRONG; explicitly forbidden
- evaluation_seed bound to world gen -> was NOT (old prototype put seed only in descriptor text); NOW bound via PRNGKey(evaluation_seed) split chain
- code implemented == world hash available -> WRONG; GLOBAL_WORLD_SET_HASH stays BLOCKED_SOURCE_UNVERIFIED until a real authorized run + 总控 review

## Host run boundary (this host jax=False craftax=False)
- allowed: compileall; static source review; pure-Python serializer unit tests (self-test PASS); serializer mock tests; import/source analysis; dry-run; static anchor-check PASS; confirm a formal run FAILS CLOSED
- forbidden: deriving MATERIALIZER_RUNTIME_PASS from mock/fake worlds; emitting any world_set_hash
- correct labels: {"MATERIALIZER_STATIC_TESTS": "PASS", "MATERIALIZER_REAL_CRAFTAX_RUN": "NOT_RUN", "GLOBAL_WORLD_SET_HASH": "BLOCKED_SOURCE_UNVERIFIED"}

## Commands on a JAX host
```
python materialize_craftax_world_set_twice.py --self-test
```
```
python materialize_craftax_world_set_twice.py --anchor-check --eval-source <eval> --wrapper-source <wrappers_cl.py> --task-source <s4_task_code.py>
```
```
CC4_S4_TASK_PATH=<canonical s4> python materialize_craftax_world_set_twice.py --orchestrate --seed seed42 --out <dir> --eval-source <eval> --wrapper-source <wrappers_cl.py> --task-source <s4> --env-source <multitask.py>
```
```
python world_materializer_negative_tests.py --eval-source ... (expect NEG02/NEG09 to convert appropriately on a real host)
```
