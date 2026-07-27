# Tier3 指标语义 (metric semantics)

- 任务: CC4_TIER3_SCAFFOLDED_EVALUATION_ENVIRONMENT_V1;模块 `tier3_metrics.py`
- 状态: **PASS**(`tier3_metrics.py --self-test` exit 0)

## 1. 冻结指标(跨 arm / 跨场景完全一致)

| 场景 | 主指标 | 定义(条件概率) | dense |
|---|---|---|---|
| FULL | `DEFEAT_KOBOLD_SR` | P(defeat_kobold \| valid_full_start) | 无 |
| FRONT_L2 | `P_CORRIDOR_EXIT_REACHED_GIVEN_VALID_START` | P(corridor_exit_reached \| valid_front_start) | `NORMALIZED_CORRIDOR_PROGRESS` |
| BACK_L2 | `P_DEFEAT_KOBOLD_GIVEN_VALID_BACK_START` | P(defeat_kobold \| valid_back_start) | 无 |

- 全部为**条件于 valid_start** 的比率;`valid_starts==0` 时值为 `None`(未定义),**绝不**伪造为 0/1。
- 无效起点(valid_start=False)从分母剔除(NEG19 保证每条 episode 带 valid_start 标志)。

## 2. Dense 进度 NORMALIZED_CORRIDOR_PROGRESS(仅 FRONT)

- 方法 GRAPH_DISTANCE:`progress = clip(1 - d(current,exit)/max(d(start,exit),1), 0, 1)`,BFS 最短路径,评测器私有 traversability。
- 范围 `[0,1]`(越界 fail-closed,NEG17 在谓词层与指标层双重守卫)。
- **单调性不保证**(死胡同/往复使 d_t 增大);暂态死胡同记 0.0。
- `is_success_substitute=False`:dense 进度**不是**通关替代。

## 3. 科学边界

- `scaffolded_results_can_replace_full_task=false`:FRONT/BACK 指标仅用于**机制诊断**,永远不得替代 FULL 的 DEFEAT_KOBOLD_SR。
- 本轮无 Student 性能数据 → 任何主指标值都只能来自合成自检,**不得**作为真实性能声明。
