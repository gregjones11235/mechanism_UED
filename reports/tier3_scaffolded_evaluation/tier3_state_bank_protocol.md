# Tier3 State Bank 协议 (state bank protocol)

- 任务: CC4_TIER3_SCAFFOLDED_EVALUATION_ENVIRONMENT_V1
- 分支: `henry/tier3-scaffolded-evaluation`(基线 `7443aec`)
- 机读文件: `schemas/tier3_state_manifest_v1.json`(合成协议演练样例,明确标 `SYNTHETIC_TEST_ONLY / NOT_MATERIALIZED`)
- 模块: `tier3_state_bank_materializer.py`(双进程 + 顺序敏感哈希)、`tier3_state_serializer.py`(加法式复用 V3 序列化器)
- 状态: **PASS**(`tier3_state_bank_materializer.py --self-test` exit 0;双进程一致;NEG05/06/07/24/26 全检出)

## 1. 核心科学约束

1. **每场景一条公共 bank**:同一 scenario(front_l2 / back_l2)只有**一条** state bank,被所有 arm(Base/Replay/Persistent/Reset128/未来 D052)**逐字节共享**。禁止按 arm 分库(NEG07)。
2. **选择与结果无关**:bank 的 seed 表是纯位置化确定性序列 `seed_base + offset + i*stride`,只依赖 `(scenario, n, seed_base, stride)`,**不依赖任何 Student/checkpoint/params/结果**(NEG26)。PROCESS_B 会从声明参数重新生成 seed 表并比对,任何按结果筛选都会破坏可复现性 → fail-closed。
3. **自有哈希标签,永不冒充全局**:bank 哈希标签为 `FRONT_SCAFFOLD_STATE_BANK_HASH` / `BACK_SCAFFOLD_STATE_BANK_HASH`,**永远不是** `GLOBAL_WORLD_SET_HASH`(后者属 seed42 canonical world materializer,CC4 V3)。把 scaffold 哈希标成全局 → fail-closed(NEG24)。

## 2. 双进程协议(加法式复用 V3 orchestration)

- **PROCESS_A(materialize)**:按固定 seed 表铸造 N 个起点 → 逐个序列化(seed-free payload hash)→ 逐个用冻结边界谓词复核 → 写出**有序** manifest。
- **PROCESS_B(verify)**:独立重载 manifest → 逐个重算哈希并 `verify_payload_hash`(NEG05 完整性,字节只做哈希比对,绝不改字节再反序列化)→ 重跑边界谓词 → 重算 bank 哈希;任何不一致 fail-closed。
- **compare_two_processes**:PROCESS_A 跑两次独立比对(schema/scenario/hash_label/state_count/seeds/state_bank_hash/source_shas/entries),任一字段不一致 fail-closed(复用 V3 `compare_two_runs` 模式)。

## 3. State bank 哈希(顺序敏感)

`state_bank_hash = SHA256( lenprefixed(schema) ‖ lenprefixed(hash_label) ‖ lenprefixed(world_builder_sha) ‖ lenprefixed(canonical_task_sha) ‖ lenprefixed("state_count=N") ‖ 有序[ (index u64, lenprefixed(payload_hash)) ] )`。

**顺序进入哈希**:重排状态 → 哈希改变(NEG06),因此“换顺序冒充同库”会被 PROCESS_B 识破。

## 4. 序列化器加法式复用(不重新实现 V3)

`tier3_state_serializer.py` 强制 `import materialize_craftax_world_set_twice as v3mat`(导入失败即 fail-closed),真实 EnvState 的 seed-free payload 哈希直接委托 `v3mat.serialize_world_payload / state_payload_hash`;`verify_source_identity` 直接委托 V3(NEG02)。纯 Python 的 normalized 视图另有 canonical-JSON 哈希(仅作协议自检载体,**绝不**充当真实物化世界哈希)。

## 5. 本机物化状态(诚实标注)

| 项 | 值 |
|---|---|
| 真实 EnvState 铸造 | **BLOCKED_ENVIRONMENT**(无 JAX/craftax) |
| FRONT_SCAFFOLD_STATE_BANK_HASH | **NOT_MATERIALIZED** |
| BACK_SCAFFOLD_STATE_BANK_HASH | **NOT_MATERIALIZED** |
| 自检所用状态 | `SYNTHETIC_TEST_ONLY`(manifest 明确记录) |
| 双进程协议机制 | PASS(在合成态上演练;顺序/篡改/分库/冒充全局/按结果筛选均被检出) |

合成态自检哈希**不可**被解读为真实 bank;manifest 同时记录 `states_are: SYNTHETIC_TEST_ONLY` 与 `hash_status: NOT_MATERIALIZED`。真实 bank 须待 JAX+craftax==1.4.5 主机执行 `materialize_start` 后产生。

## 6. 自检

`python tools/tier3_scaffolded_evaluation/tier3_state_bank_materializer.py --self-test`
→ `TIER3_STATE_BANK_MATERIALIZER_SELF_TEST_PASS (scenarios=2, two_process=agree, hash_status=NOT_MATERIALIZED, env=BLOCKED_ENVIRONMENT)`,exit 0。
