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


---

## Round 7 (V3) addendum — GLOBAL_WORLD_MATERIALIZER_RUNTIME_IDENTITY_HARDENING_V3

本机仍 `jax=False`、`craftax=False`；正式 run 仍 **fail closed（exit 2）**，不产生任何输出。

**V3 只能在 JAX+craftax==1.4.5 主机执行**：
- 运行时执行源码身份绑定（真实 import wrapper/env）；
- NEG02 / GATE21 的**真实** seed42-vs-seed100000 world payload 比较（`materialize_all_world_states(42)` 与
  `(100000)`，比较 seed-free `state_payload_hash`；≥1 world 不同方可 PASS，256/256 相同则 FAIL）；
- 输出 `world_field_manifests.json` / `world_field_schema_summary.json` / `runtime_source_identity.json`；
- 将 GATE20 升级为 PASS_REAL_RUNTIME、GATE23 升级为 PASS_REAL_OUTPUT。

**V3 本机仍可做**：compileall、静态 anchor-check、纯 Python 序列化/payload-hash/manifest/seed-identity
单测（self-test 36 项 PASS）、在真实磁盘 wrapper/env 文件上的绑定逻辑定向测试（NEG12/13 PASS）、正式 run 的
fail-closed 验证。

**V3 本机禁止**：把 NEG02/GATE21 标 PASS（装了环境≠跑了测试）；把 GATE20 标 PASS_REAL_RUNTIME 或 GATE23 标
PASS_REAL_OUTPUT；把 TWO_PROCESS_REAL_WORLD_AGREEMENT 标为非 NOT_RUN；产生任何 world_set_hash；把 seed100000
声称为 canonical 精确世界集。

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
