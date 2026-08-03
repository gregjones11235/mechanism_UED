# CC1 Handoff · Simulator-Centric Frontier-UED Foundation

## Repository state

- branch：`henry/simulator-frontier-foundation-codex`
- base branch：`Henry-branch`
- base SHA：`9eca2de`
- worktree：`C:\Users\Lenovo\Desktop\dicode-codex-director\mechanism_UED_sim_foundation`
- team ZIP SHA256：`226363969f50fe42b35bd3cdb03d6a0e7cba16be44c2d1463ffc375b0e907e62`

## Foundation interfaces

- `dicode.simulator_frontier.goals`：显式 `AchievementGoal`、`GateProgressGoal`、`StateFactsGoal`、`TerminalEventGoal`、`CompositeGoal`；禁止按 Python 类型猜测字符串语义。
- `terminal_events.py`：保留 `terminal_state`、`returned_state`、`reset_state` 和事件 provenance；done+autoreset 缺 terminal evidence 时 fail closed。
- `state_codec.py`：确定性、带 shape/dtype/hash/version 校验的 pickle-free `StateBundle` 编解码。
- `frontier_archive.py` / `archive_schema.py`：有界容量、分桶配额、重复状态去重、状态 hash 与 provenance hash 校验、JSON 持久化。
- `search_statistics.py`：只用 `N_actual` 计算 success rate、Wilson interval、实际成本；禁止把 best-of-N 外推写入主 success rate。
- `memory_modes.py`：`ZERO_MEMORY`、`SAVED_POLICY_MEMORY`、`HISTORY_BURN_IN` 显式区分；缺 checkpoint/tree/history 时不兼容。
- `provenance.py`：拒绝 FORMAL_* 流入 frontier/branch/curriculum/student optimizer，并拒绝动作序列/路线/logits/hidden states 等行为泄漏字段。

## 未完成且不得误报为完成

- 真实 Craftax EnvState restore/dynamics 对照；
- 真实 saved-policy-memory/history burn-in；
- 真实 branch rollout executor；
- feasibility-based frontier selector；
- mixed standard-reset/frontier-start batch；
- archive state → PPO/V-trace update；
- fresh-process checkpoint round-trip；
- 12 dynamic frontier distributions + 4 anchors；
- 正式 reset smoke evaluation、长程训练和性能结论。

## 推荐 CC1 顺序

1. 用真实 Craftax 环境做 `StateCodec` restore/dynamics 对照；
2. 实现并验证 saved-policy-memory/history burn-in；
3. 接入真实 branch rollout executor，并记录 `N_actual`；
4. 接 feasibility selector；
5. 做 mixed batch；
6. 经独立门禁后才允许 archive→PPO/V-trace；
7. 做 fresh-process checkpoint round-trip；
8. 最后才考虑 12+4 分布和 reset smoke。

## 监控命令

```powershell
$env:PYTHONPATH='gpu1_aggregation_siege/src'
pytest -q gpu1_aggregation_siege/tests/simulator_frontier
python -m compileall -q gpu1_aggregation_siege/src/dicode/simulator_frontier
git status --short --branch
```
