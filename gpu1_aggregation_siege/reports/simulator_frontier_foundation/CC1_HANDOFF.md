# CC1 Handoff · Simulator-Centric Frontier-UED Foundation

## 如何在不修改训练机制核心代码的情况下挂载当前高能力 RMT16 Student 并启动一次真实更新

> 本节为交接指引（接口级）：本轮**未接通**任何真实训练入口，零训练更新。真实启动前必须先走完 R4c 联合门禁与总控授权。

### 命令模板（如实标注：`<actual_training_entry>` 本轮未接通）

```bash
PYTHONPATH=gpu1_aggregation_siege/src PYTHONIOENCODING=utf-8 JAX_PLATFORMS=cpu \
python <actual_training_entry> \
  teacher=simulator_frontier \
  student.profile=rmt16_persistent_98304 \
  student.checkpoint_path=<CHECKPOINT> \
  student.expected_params_sha256=<SHA256> \
  resume.mode=full_state \
  training.total_env_steps=<BOUNDED_SMOKE> \
  simulator_frontier.enabled=true \
  simulator_frontier.search_distillation=false \
  formal_evaluation.enabled=false \
  output_root=<NEW_UNIQUE_ROOT>
```

- `<actual_training_entry>` = `dicode_src/experiments/training/run_dicode.py`（worktree 内存在；本目录树另有 `gpu1_aggregation_siege/experiments/training/run_dicode.py`）。**本轮均未接通 student mount，如实标注。**
- `<CHECKPOINT>` = PERSISTENT_RMT16 CC2 pkl 的实际路径（运行时引用，不入库）；`<SHA256>` = `aa6ba44040a0742bd709ebe6299acde6242e4faf748159dd0765ee2428addd0d`（profile 已绑定，override 必须一致，失配 fail-closed）。
- `student.*` 参数解析只走 `dicode.student_adapters.registry.resolve_runtime_overrides`（纯 argv key=value），不引 hydra、不猜默认值。

### 首跑探针（先于任何真实更新）

```bash
python -m dicode.simulator_frontier.probes.student_compatibility \
  student.profile=rmt16_persistent_98304 \
  student.checkpoint_path=<CHECKPOINT> \
  student.expected_params_sha256=<SHA256> --steps=8
# exit 0 = 20/20 PASS；4 = FAIL；5 = BLOCKED（ADAPTER_PENDING / ARTIFACT_HANDOFF_REQUIRED）
```

### 尚缺清单（真实更新前必须补齐）

- **adapter**：GTrXL128/TEACHER/SLOWGRU 只读 adapter（CONTROL_CONTINUOUS_98304 为匹配对照，最先需要）；
- **artifact**：BASE_GTRXL_ORIGINAL / SLOWGRU 的 server→local handoff；TEACHER canonical pkl；
- **manifest**：总控签名的共享冻结 anchor manifest（4 锚点科学内容；未达 → `BLOCKED_SHARED_ANCHOR_MANIFEST`）；
- **R4c 联合还原器**：optimizer/train_rng/policy_memory/history 组件 restorer（当前 CC2 pkl 中 ABSENT_IN_CHECKPOINT → 需要携带全状态的 checkpoint 或 Phase 2 还原实现）；
- **训练入口接线**：run_dicode → registry 解析 → adapter 挂载（本轮未改训练机制代码）。

### 一次真实更新的 hook 落点与确认方法

- **唯一注入点**：`gpu1_aggregation_siege/src/dicode/ppo_tr.py` 的 `make_train`/`make_eval` 网络构造处（本轮未改）；不得在其他位置旁路构造网络。
- **optimizer 未重置**：resume 后比对 optimizer state 树 hash 与 checkpoint 记录（ABSENT_IN_CHECKPOINT 时必须显式声明并阻断「伪装续训」）。
- **memory 未错误清零**：按 `memory_modes.MemoryRestoreRequest` 的显式模式（ZERO_MEMORY/SAVED_POLICY_MEMORY/HISTORY_BURN_IN）恢复，并经 `adapter.validate_memory` 门禁；persistent carry 行为以 128 步段边界探针确认。
- **checkpoint 可恢复**：更新后 `save_full_state` 产物必须能在 fresh process 中重新通过身份门禁与 params sha 复算。

### R4c 联合门禁放行流程

1. 单一 fresh process 调用 `run_combined_restore(CombinedRestoreRequest(...), restorers=..., cross_checkers=...)`；
2. 9 组件（params/optimizer/global_step/train_rng/env_state/env_rng/wrapper_state/policy_memory/history）全部 RESTORED + `policy_step_next_replay` 交叉核验通过；
3. **env-only PASS ∧ ckpt-only PASS 不放行**（`evaluate_verdict` 机械强制）；
4. 全绿才允许 R5 actual-N 搜索以其为起点；否则 `COMBINED_FRESH_PROCESS_RESTORE=false` 并升级。

### R5–R9 放行前置条件

R5/R6：R4c 放行 + branch rollout executor 就位 + N_actual 实测（禁 best-of-N 外推）。
R7：总控显式授权真实 API（本轮 `REAL_TWO_LLM_CALL_EXECUTED=false`；契约与 fake-client 测试已就位）。
R8：总控共享 anchor manifest 到达并绑定（`SHARED_ANCHOR_MANIFEST_BOUND=false` 前不得自拟锚点）。
R9：R5–R8 全绿 + 同 Student 更新合同（training_integration_contract.json SC5）+ 有界 smoke 预算获批。

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
