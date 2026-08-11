# Arm-C 同分闭环裁决（P0-15，只读调查）

分支 `henry/ba-bagr-ued-review-board-v2`。本文件是 P0-15 的正式裁决记录。
全程只读：冻结证据（judgments、`ranking_C.json`、`expected_behavior.json`、
`selector_config.json`、`aggregation_original.py`）零改动；本结论由
`d052/reconciliation/arm_c_tie_resolution.py::classify_arm_c_tie` 复算并
由 `test_arm_c_tie_resolution.py`（9 项）锁定（含冻结证据字节不变性测试）。

## 1. 事实（实测）

* **精确同分**：Arm-C 最终 Soft-Copeland 分数中
  `d052_r3_0007` 与 `d052_r3_0028` 位对位完全相同
  `0.81423508564678215`，排名 8/9，恰好横跨 k=8 选择切线。
* **裸 argsort**：`d052/reconciliation/replay.py:77` 用
  `np.argsort(-scores)[:8]`（默认 quicksort，非稳定）。本 worktree numpy
  下 quicksort 选中 `0028`，复算选择哈希 = `868a57268d66b90b`（= 冻结
  `C_selection_hash`）；而 `kind='stable'/'mergesort'/'heapsort'` 全部选中
  `0007`。
* **rank 列 vs 选择集矛盾**：冻结 `ranking_C.json` 给 `0007` rank 8、
  `0028` rank 9（稳定次序：同分按 task_id 升序）；冻结
  `expected_behavior.json` 的 `C_selected8` 却含 `0028` 不含 `0007`。
  二者对 k=8 边界候选的判断相反。
* **声明证伪**：`selector_config.json` 声明同分由
  "np.argsort(-scores)[:8] ... ties broken by argsort index order" 打破；
  实测 quicksort 的同分次序并非 "index order"（稳定排序取相反候选），
  且该次序是 numpy 实现相关的、跨环境不可复现。

## 2. 分类

**FROZEN_EVIDENCE_INTERNALLY_INCONSISTENT。**

不存在任何单一确定性 tie-break 规则能同时复现两份冻结产物：
按 rank 列（稳定次序）应选 `0007`；按历史 quicksort 选择集应选 `0028`。
因此：

* `REPLAY_CONTRACT_FIXED` 不成立（裸 quicksort 不是跨环境确定性契约）；
* 该同分问题正式标记为 `LEGACY_REPLAY_BLOCKED_NON_PRODUCTION`：
  不进入方向二生产路径、不回填、不改写、不重新裁决任何冻结证据；
* 复现 `868a57268d66b90b` 仅是"本 numpy 版本 + 本输入"的实现细节，
  不是可审计的确定性证据。

## 3. 处置（全部只读 + 正式阻断）

1. 新增只读分类器 `classify_arm_c_tie()`：纯函数复算分数、检测边界同分、
   比对 rank 列与选择集、输出分类与全部证据字段；
2. 测试锁定：精确同分、quicksort 复现冻结哈希、稳定排序取相反候选、
   rank 列矛盾、分类结果、冻结证据字节不变、历史 replay 锚点仍全绿
   （`ALL_ANCHORS_PASS=True`、`C_selection_hash=868a57268d66b90b`）；
3. 冻结 Replay 合同不再扩张；未来若重启 Replay，必须先以显式稳定
   tie-break（如 `kind='stable'` + task_id）取代裸 argsort，并重新取证。

## 4. 对生产路径的影响

无。`LEGACY_REPLAY_BLOCKED_NON_PRODUCTION` 是记录性阻断：两窗口/长跑
入口不依赖该冻结 Replay；所有 REAL_* 旗标恒 False；本调查不改任何
REAL_* 标志、不启动任何真实执行。
