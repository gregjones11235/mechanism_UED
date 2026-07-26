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


---

## Round 7 (V3) addendum — GLOBAL_WORLD_MATERIALIZER_RUNTIME_IDENTITY_HARDENING_V3

状态：IMPLEMENTED_STATIC；REAL_RUN_NOT_EXECUTED；GLOBAL_WORLD_SET_HASH=BLOCKED_SOURCE_UNVERIFIED（未变）。

**真实 run 现输出四个文件**：`world_hashes.json`（per-world hash + seed-free per-world payload hash + total +
identity_class/protocol_id + world_field_manifests_sha256）、`world_field_manifests.json`、
`world_field_schema_summary.json`、`runtime_source_identity.json`（imported vs requested 路径/realpath/SHA、
identity match、evaluator anchor 状态、task exec 身份、sys.path 前几项、模块版本；**不含**任何 token/secret）。

**两 run 比较现在包含**：world_set_hash、per_world_hashes、per_world_state_payload_hashes、
world_field_manifests_sha256、source_shas、versions、numeric_evaluation_seed、identity_class、protocol_id、
runtime_source_identity —— 任一不同即 fail closed。

**JAX 主机命令（V3）**：
```
python materialize_craftax_world_set_twice.py --self-test     # 36 checks PASS
python materialize_craftax_world_set_twice.py --anchor-check --eval-source <eval> --wrapper-source <wrappers_cl.py> --task-source <s4>
CC4_S4_TASK_PATH=<canonical s4> python materialize_craftax_world_set_twice.py --orchestrate --seed seed42     --out <dir> --eval-source <eval> --wrapper-source <wrappers_cl.py> --task-source <s4> --env-source <multitask.py>
# seed100000 为 PARAMETERIZED variant，单独跑，绝不作为 canonical 精确世界集：
CC4_S4_TASK_PATH=<canonical s4> python materialize_craftax_world_set_twice.py --orchestrate --seed seed100000     --out <dir2> --eval-source <eval> --wrapper-source <wrappers_cl.py> --task-source <s4> --env-source <multitask.py>
python world_materializer_negative_tests.py --eval-source <eval> --wrapper-source <wrappers_cl.py>     --task-source <s4> --env-source <multitask.py>   # NEG02 在真实主机转 PASS；NEG09 严格 builder 仍 BLOCKED
```

**过度声明纠正（V3）**：NEG02 因"装了环境"而 PASS → 已删除；per_world_hash/world_set_hash 差异证明真实 seed RNG
效应 → 错（含 seed-tagged header），须用 seed-free state_payload_hash；所有 source SHA 都"被执行" → 错
（evaluator 是协议锚点，未被执行）；seed100000 == canonical 精确世界集 → 错（PARAMETERIZED variant，独立 P7
evaluator）；GATE19 严格共享 builder 经弱锚点 PASS → 错（严格 builder 保持 BLOCKED，静态锚点是**独立** PASS / NEG11）。

### V3 冻结标签

- CC4_RUNTIME_SOURCE_IDENTITY_CODE = PASS
- CC4_RUNTIME_SOURCE_IDENTITY_REAL_RUN = NOT_RUN
- EXECUTED_WRAPPER_SOURCE_BINDING = PASS_STATIC
- EXECUTED_ENV_SOURCE_BINDING = PASS_STATIC
- EXECUTED_TASK_SOURCE_BINDING = PASS_STATIC
- EVALUATOR_SOURCE_ROLE = STATIC_PROTOCOL_ANCHOR_NOT_EXECUTED
- SEED42_IDENTITY_CLASS = CANONICAL_EVALUATOR_EXACT_WORLD_SET
- SEED100000_IDENTITY_CLASS = PARAMETERIZED_WORLD_GENERATION_PROTOCOL_VARIANT
- EVALUATION_SEED_STATIC_RNG_BINDING = PASS
- EVALUATION_SEED_REAL_WORLD_PAYLOAD_EFFECT = BLOCKED_ENVIRONMENT
- NEG02_FALSE_PASS_REMOVED = PASS
- WORLD_STATE_PAYLOAD_HASH = IMPLEMENTED
- WORLD_FIELD_MANIFEST_CODE = IMPLEMENTED
- WORLD_FIELD_MANIFEST_REAL_OUTPUT = NOT_RUN
- MATERIALIZER_EVALUATOR_SHARED_BUILDER = BLOCKED_EVALUATOR_INLINE_READ_ONLY
- STATIC_ANCHOR_EQUIVALENCE = PASS
- GLOBAL_WORLD_SET_HASH = BLOCKED_SOURCE_UNVERIFIED (unchanged)
