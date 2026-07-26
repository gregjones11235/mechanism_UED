# Integration & scientific-claims contract (round-6 revision -- eleven)

- UTC: `2026-07-26T15:10:12Z`

## Round-6 new/revised files
- `tools/global_evaluation/world_key_manifest_prototype.py (git mv from materialize_world_set_twice.py; DEPRECATED; fails closed)`
- `tools/global_evaluation/materialize_craftax_world_set_twice.py (actual serializer + materializer; static)`
- `tools/global_evaluation/world_materializer_negative_tests.py (10 negative tests)`
- `reports/global_remediation/premerge_hardening/world_generation_path_audit.{md,json}`
- `reports/global_remediation/premerge_hardening/host_run_boundary.{md,json}`
- `reports/global_remediation/premerge_hardening/{world_manifest_evidence_tier,integration_contract,pure_logic_gate_report}.{md,json} (revised)`
- `reports/global_remediation/world_set_materialization_runbook.{md,json} (revised)`

## World-set status
- OLD_WORLD_KEY_PROTOTYPE = **DEPRECATED_INVALID_FOR_WORLD_SET_HASH**
- WORLD_GENERATION_SOURCE_PATH = **FOUND**
- CRAFTAX_WORLD_MATERIALIZER_CODE = **IMPLEMENTED**
- CRAFTAX_WORLD_MATERIALIZER_REAL_RUN = **NOT_RUN**
- GLOBAL_WORLD_RECIPE = **PASS**
- GLOBAL_WORLD_SET_HASH = **BLOCKED_SOURCE_UNVERIFIED**
- statement = **actual materializer code IMPLEMENTED + statically tested, but NO real world materialized on this host; world_set_hash NOT upgraded.**

## ALLOWED claims (round 6)
- the canonical world-generation path is LOCATED and line/SHA-anchored (FOUND)
- an actual stable serializer + actual-reset materializer is IMPLEMENTED and statically tested (self-test/anchor/negative FAIL=0)
- the materialization gate REJECTS key-only output (GATE17 PASS)
- the deprecated key prototype is sealed (fails closed) with history preserved

## FORBIDDEN claims (round 6)
- the world set has been really materialized (NOT_RUN)
- a world_set_hash exists / is verified (BLOCKED_SOURCE_UNVERIFIED)
- evaluation_seed binding to real worlds is fully verified on hardware (PARTIAL_ENVIRONMENT_BLOCKED)
- materializer and evaluator share a literal builder function (strict BLOCKED)
- checkpoints re-evaluated / training reproduced / Exact Resume bit-exact / matched Replay run (all still NOT_RUN)

- CC2_FILES_TOUCHED=False ; CC3_FILES_TOUCHED=False ; PUSH_PERFORMED=False


---

## Round 7 (V3) addendum — GLOBAL_WORLD_MATERIALIZER_RUNTIME_IDENTITY_HARDENING_V3

**本轮范围（仅四项）**：①绑定"实际执行的 Python 模块"与"记录的源码 SHA"；②修复 NEG02 虚假 PASS；
③拆分 seed42 / seed100000 协议身份；④持久化 world field manifest。

**本轮允许的科学声明**：物化器**代码**已绑定执行源码身份（realpath+SHA，不一致 fail closed）；canonical
evaluator 是静态协议锚点而非执行源码；seed42=CANONICAL_EVALUATOR_EXACT_WORLD_SET、seed100000=PARAMETERIZED
variant（含独立 P7 evaluator）；seed-free world payload hash 已实现且是 seed-effect 的正确载体；field manifest
已实现并在真实 run 持久化、`assert_materialized` 强制要求；NEG02 不再因"装了环境"而 PASS。

**本轮禁止的科学声明**：声称运行时绑定已真实执行（本机 REAL_RUNTIME_NOT_RUN）；声称已测得真实 seed 世界差异
（BLOCKED_ENVIRONMENT）；声称 seed100000 是 canonical 精确世界集；在真实授权 run 前声称 GATE20=PASS_REAL_RUNTIME
或 GATE23=PASS_REAL_OUTPUT；产生/声称任何 GLOBAL_WORLD_SET_HASH（保持 BLOCKED_SOURCE_UNVERIFIED）；声称严格共享
builder 存在（BLOCKED_EVALUATOR_INLINE_READ_ONLY）。

**新增门禁 GATE20-23**（详见 pure_logic_gate_report）；负向测试 V3：PASS=20 / BLOCKED=3 / FAIL=0。
CC2_FILES_TOUCHED=false；CC3_FILES_TOUCHED=false；PUSH_PERFORMED=false。54 冻结文件 0 改动；SHA256SUMS 未重写；
.gitattributes 作用域未扩大。

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
