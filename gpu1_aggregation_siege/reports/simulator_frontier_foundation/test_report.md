# Simulator-Centric Frontier-UED · Foundation 测试报告

## 基线

- branch：`henry/simulator-frontier-foundation-codex`
- base：`Henry-branch` @ `9eca2de`
- worktree：`C:\Users\Lenovo\Desktop\dicode-codex-director\mechanism_UED_sim_foundation`
- GPU：未使用
- 真实 Craftax：未运行
- 真实 branch search：未运行
- checkpoint：未修改
- LLM/API：未调用

## 本轮新增代码

目标包：`src/dicode/simulator_frontier/`

- typed GoalSpec 与 fail-closed evaluator；
- terminal/reset/returned state 分离的 TerminalEventAdapter；
- pickle-free deterministic StateCodec；
- bounded FrontierArchive 与 provenance/hash 校验；
- Actual-N BranchOutcome/FeasibilityEstimate 与 Wilson 区间；
- 显式 memory restore modes；
- formal-data 与 search-action leakage guards。

## 命令与结果

1. `python -m compileall -q src/dicode/simulator_frontier`：PASS。
2. `pytest -q gpu1_aggregation_siege/tests/simulator_frontier --basetemp <workspace-temp>`：**14 passed**。
3. `pytest -q tests/test_checkpoint_fixtures.py tests/test_data_plane_integrity.py tests/test_siege_components.py`：**1 passed**（该目录中其余测试由测试函数返回值驱动，pytest 给出既有 warning）。
4. 受环境/外部产物限制未完成的现有测试集合：
   - `test_r0_production_calls.py`、`test_r0_production_dispatcher.py`：缺少 `/root/experiments/dicode_runs/siege_aggregation/frozen_pool_artifact.json`；
   - `test_v3_hard_gates.py`：本机缺少 `craftax`，且其固定缓存路径没有 96 条产物。

这些失败是基线外部产物/依赖缺失，不是本轮 foundation 模块的失败；没有用 skip/xfail 掩盖。
