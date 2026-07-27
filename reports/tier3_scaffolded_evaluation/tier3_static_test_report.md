# Tier3 静态/合成测试报告 (static & synthetic test report)

- 任务: CC4_TIER3_SCAFFOLDED_EVALUATION_ENVIRONMENT_V1
- 分支: `henry/tier3-scaffolded-evaluation`(基线 `7443aec`)
- 证据层级: 至多 `IMPLEMENTED_STATIC` / `TESTED_SYNTHETIC`(真实物化/rollout 均 BLOCKED/NOT_RUN)

## §十八 三层测试

### 第 1 层 — 纯 Python(无 JAX / 无 craftax):PASS
纯数据结构 / schema / 谓词 / 指标 / 分类 / 证书逻辑,在本机全部可跑。
- 聚合自检: `TIER3_AGGREGATE_SELF_TEST_PASS`(modules=11, negative_tests FAIL=0, exit 0)。
- 负向测试: `TIER3_NEGATIVE_TESTS_PASS`(FAIL=0; implemented=26/26, pending_commit3=0)。

### 第 2 层 — 合成 EnvState 协议演练:PASS
用显式标记 `SYNTHETIC_TEST_ONLY` 的 normalized 视图演练 builder/serializer/materializer/evaluator/taxonomy/certificate 全链路;合成哈希**不冒充**真实 bank(manifest `hash_status=NOT_MATERIALIZED`)。
- 双进程: `TIER3_STATE_BANK_MATERIALIZER_SELF_TEST_PASS`(two_process=agree)。

### 第 3 层 — 真实 JAX-Craftax CPU 自检:BLOCKED_ENVIRONMENT
本机 `JAX_AVAILABLE=False`、`CRAFTAX_AVAILABLE=False`(`importlib.find_spec` 均 False)。真实 EnvState 物化、真实 scaffold 铸造、真实 reset/step、真实 Student rollout 均**未运行**。
- `REAL_CRAFTAX_SCAFFOLD_TEST=BLOCKED_ENVIRONMENT`
- `REAL_STUDENT_EVALUATION=NOT_RUN`
- 若在 craftax==1.4.5 主机运行,自检路径已就绪(guard 于 `ser.have_jax_craftax()`),可达 `TESTED_REAL_ENV_RESET`;但本轮不做此声明。

## 自检横幅逐条(本机实测)

| # | 横幅 | 结果 |
|---|---|---|
| 1 | TIER3_SOURCE_AUDIT_SELF_TEST_PASS | in-repo MATCH=12, envstate_fields=53, mobs_fields=5 |
| 2 | TIER3_PREDICATES_SELF_TEST_PASS | checks=24 |
| 3 | TIER3_BOUNDARY_SCHEMA_SELF_TEST_PASS | events=10 |
| 4 | TIER3_STATE_SERIALIZER_SELF_TEST_PASS | v3_reuse=live, env=BLOCKED_ENVIRONMENT |
| 5 | TIER3_SCAFFOLD_BUILDER_SELF_TEST_PASS | scenarios=2, legality=9 flags |
| 6 | TIER3_STATE_BANK_MATERIALIZER_SELF_TEST_PASS | two_process=agree, hash_status=NOT_MATERIALIZED |
| 7 | TIER3_CHECKPOINT_ADAPTER_SELF_TEST_PASS | NEG21/22/23 live |
| 8 | TIER3_METRICS_SELF_TEST_PASS | 冻结指标条件比率 |
| 9 | TIER3_FAILURE_TAXONOMY_SELF_TEST_PASS | NEG20 fail-closed |
| 10 | TIER3_EVALUATOR_SELF_TEST_PASS | NEG19/NEG23 live, rollout=BLOCKED_ENVIRONMENT |
| 11 | TIER3_EVALUATION_CERTIFICATE_SELF_TEST_PASS | NEG24/NEG25 live |
| 12 | TIER3_NEGATIVE_TESTS_PASS | FAIL=0; 26/26 |
| 13 | TIER3_AGGREGATE_SELF_TEST_PASS | modules=11, exit 0 |

## §二十三 最终测试 battery 命令(逐字、退出码捕获)

```
cd D:/cc4tmp
python -m compileall tools/tier3_scaffolded_evaluation ; echo EXIT=$?
python tools/tier3_scaffolded_evaluation/tier3_self_test.py ; echo EXIT=$?
python tools/tier3_scaffolded_evaluation/tier3_negative_tests.py ; echo EXIT=$?
python tools/tier3_scaffolded_evaluation/tier3_boundary_schema.py --self-test ; echo EXIT=$?
python tools/tier3_scaffolded_evaluation/tier3_state_bank_materializer.py --self-test ; echo EXIT=$?
python tools/tier3_scaffolded_evaluation/tier3_evaluation_certificate.py --self-test ; echo EXIT=$?
```

退出码以实际运行为准;任何非 0 即阻断提交,绝不修改测试使其通过以掩盖缺陷。
