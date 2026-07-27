# Tier3 分段支架评测环境 V1 — 最终报告

- 任务: CC4_TIER3_SCAFFOLDED_EVALUATION_ENVIRONMENT_V1(§零–§二十五)
- 模块 owner: TIER3_EVALUATION / TIER3_SCAFFOLD_BUILDER / TIER3_BOUNDARY_SCHEMA / TIER3_STATE_BANK / TIER3_METRIC_SCHEMA / TIER3_EVALUATION_CERTIFICATE = **CC4**
- 分支: `henry/tier3-scaffolded-evaluation`,基线 `7443aec`(不叠在已审分支上,不改写已审历史)
- 研究目标: `PRIMARY_RESEARCH_TARGET=BREAK_TIER3_FRONT_HALF_DARK_MONSTER_CORRIDOR`
- 架构: `EVALUATION_ARCHITECTURE=THREE_LEVEL_DECOMPOSED_EVALUATION`

## 一、三场景(V1)

| 场景 | 起点 | 去除 | 保留 | 主指标 |
|---|---|---|---|---|
| FULL_END_TO_END | canonical S4 DEFEAT_KOBOLD,无 scaffold | 无 | 全程 | `DEFEAT_KOBOLD_SR` |
| TIER3_FRONT_HALF_SCAFFOLDED_L2 | 源码可证前段入口 | 上游资源准备/入城 | 黑暗走廊+导航+多怪+生存+找出口 | `P_CORRIDOR_EXIT_REACHED_GIVEN_VALID_START` + dense `NORMALIZED_CORRIDOR_PROGRESS` |
| TIER3_BACK_HALF_SCAFFOLDED_L2 | 源码可证后段/boss 前边界 | 走廊瓶颈 | 找 boss+接战+战斗+生存+DEFEAT_KOBOLD | `P_DEFEAT_KOBOLD_GIVEN_VALID_BACK_START` |

FRONT_L1 / BACK_L1:**预留不实现**(本轮)。

## 二、模块清单(3 次提交)

- Commit 1 `a4075f8`(边界语义冻结): source_audit / event_predicates / boundary_schema + schema + 2 报告 + 3 配置。
- Commit 2 `9ee137a`(规范状态 bank): state_serializer / scaffold_builder / state_bank_materializer + negative_tests + 2 schema + 2 报告。
- Commit 3(解耦评测器与证书): checkpoint_adapter / evaluator / metrics / failure_taxonomy / evaluation_certificate / self_test + negative_tests 补全 + 3 schema + 7 报告。

## 三、冻结语义

- `FRONT_FLOOR=2`,`BACK_FLOOR=3`,`CORRIDOR_EXIT_FLOOR=3`。
- `corridor_exit_reached := player_level≥3`;`boss_area_reached := player_level==3`(否决 `boss_progress>0`,那是 Necromancer 机制)。
- `defeat_kobold := achievements[Achievement.DEFEAT_KOBOLD.value]`(符号化,craftax==1.4.5 运行时绑定)。
- FRONT 进度 = GRAPH_DISTANCE(评测器私有 BFS traversability),范围 `[0,1]`,非通关替代(`is_success_substitute=False`)。

## 四、合法性与不破坏

- scaffold 仅经 WorldBuilder(唯一合法机制,源自 combat.py "ADDED SCAFFOLDING");9 项 legality flag 全 True;`FORBIDDEN_RESULT_BLINDNESS_KEYS` 守卫禁 arm/checkpoint/结果选择。
- 加法式复用 V3(`materialize_craftax_world_set_twice`),不重新实现;import 失败即 fail-closed。
- scaffold bank 哈希**永不**冒充 `GLOBAL_WORLD_SET_HASH`(NEG24);scaffold 结果**永不**声称 full 通关(NEG25)。
- 未触碰 CC2 / CC3 / Henry-branch / 冻结 54 文件 / 原始 SHA256SUMS;未 push/merge/rebase/amend/force push/reset --hard/git clean。

## 五、本轮禁止事项(逐条遵守)

禁止 Student 训练;禁止 4096 smoke;禁止 24576 screening;禁止 98304;禁止多 seed 性能实验;禁止正式 evaluation run;禁止性能比较;禁止新 LLM 调用;禁止 D052 candidate pool;禁止写 `FRONT_SCAFFOLD_EVALUATION=PASS` / `TIER3_FRONT_HALF_BREAKTHROUGH`(本轮无 Student 性能数据)。均**未违反**。

## 六、测试

- 负向测试 NEG01–NEG26 全部实现,`FAIL=0`。
- 聚合自检 `TIER3_AGGREGATE_SELF_TEST_PASS`(modules=11, exit 0)。
- 详见 `tier3_static_test_report.md`(三层:纯 Python PASS / 合成 EnvState PASS / 真实 JAX-Craftax BLOCKED_ENVIRONMENT)。

## 七、诚实冻结标签

- 至多 `IMPLEMENTED_STATIC` / `TESTED_SYNTHETIC`。
- `REAL_CRAFTAX_SCAFFOLD_TEST=BLOCKED_ENVIRONMENT`;`REAL_STUDENT_EVALUATION=NOT_RUN`;
  `GLOBAL_WORLD_SET_HASH=BLOCKED_SOURCE_UNVERIFIED`;`FRONT/BACK_SCAFFOLD_STATE_BANK_HASH=NOT_MATERIALIZED`;
  `NEW_TRAINING_RUNS=0`;`FORMAL_EVALUATION_RUNS=0`;
  `CC2/CC3_FILES_TOUCHED=false`;`HENRY_BRANCH_TOUCHED=false`;`PUSH_PERFORMED=false`。
- 机器可读版: `tier3_v1_labels.json`。

## 八、结论

V1 分段支架评测环境已**静态落地并通过合成自检**;真实环境物化与 Student 评测因本机无 JAX/craftax 而 BLOCKED/NOT_RUN。任何通关/突破/SOTA/Persistent>Reset128/Replay 增益声明在本轮均**不成立**。scaffold 结果仅供机制诊断,`scaffolded_results_can_replace_full_task=false`。
