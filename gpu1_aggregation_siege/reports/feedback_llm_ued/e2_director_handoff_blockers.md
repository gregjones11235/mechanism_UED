# E2 双 Student 总监交接阻断清单（DIRECTOR_SMOKE_HANDOFF_READY 之下限）

分支 `henry/ba-bagr-ued-review-board-v2`。本轮的消费契约已完成——两个允许
Student（PERSISTENT / RESET128）都能进入两窗口 check-only，两窗口强制同一
Student，feedback 按 Student 身份绑定，只读挂载与训练就绪分离。最终状态上限
`E2_DUAL_STUDENT_CONSUMER_READY`。以下为总监执行 Smoke 前仍需满足的外部条件。

## 1. 两窗口入口当前阻断（`run_e2_real_two_window.py --check-only`）

* **无总监 Runtime Bundle**：全部共享资产缺席（BLOCKED_WAITING_SHARED_RUNTIME），
  Student 契约未注入（STUDENT_INIT_CONTRACT_NOT_INJECTED），无真实 transport
  （REAL_MODE_BLOCKED_NO_LLM_BACKEND）—— 默认空态阻断不变；
* **有有效 Bundle**：空 Bundle 阻断消失；仅剩本地 jax/craftax 不可导入
  （LOCAL_RUNTIME_MODULE_MISSING × 2）；
* **无效/被篡改 Bundle**：DIRECTOR_RUNTIME_BUNDLE_INVALID，退出 1；
* **CLI Student 与 Bundle 签发身份冲突**：E2_STUDENT_CANDIDATE_CLI_OVERRIDE_FORBIDDEN。

## 2. 总监解除阻断的唯一路径

1. 提供签名的总监 Runtime Bundle（含 Student profile / checkpoint 路径与哈希 /
   RMT16 Adapter / Reference / Probe Runner / CanonicalDiCodeOneUpdateRuntime /
   CanonicalDiCodeRunStateCheckpoint / 后端与模型身份 / transport / AuxiliaryLedger）；
2. 显式选择允许的 Student（PERSISTENT 或 RESET128）—— 无默认；
3. 本地 jax/craftax 可导入；
4. （真实 Smoke）注入 transport closure 与训练 Runtime 对象；
5. （正式实验）FORMAL_EXPERIMENT_AUTHORIZED 按流程批准（当前 false）。

## 3. 本方向已建成并测试的消费契约

* 允许集合 + 无默认候选（E2_STUDENT_NO_DIRECTOR_SELECTION /
  E2_STUDENT_UNKNOWN_CANDIDATE）；
* StudentInitContract/Identity 全字段 + 合法 memory 映射
  （E2_STUDENT_PROFILE_MISMATCH / E2_STUDENT_MEMORY_MODE_MISMATCH /
  E2_STUDENT_ADAPTER_IDENTITY_MISMATCH）；
* 两窗口同一 Student（E2_TWO_WINDOW_STUDENT_CONTINUITY_VIOLATION 停止）；
* feedback 全 Student 身份盖章 + 跨 Student 拒绝；
* 只读挂载（STUDENT_READ_ONLY_MOUNT_READY=true）≠ 训练就绪
  （STUDENT_TRAINING_RUNTIME_READY=false，无对象即不可训练）；
* round-trip 信任根：只消费总监 DirectorVerifiedRunStateRoundTrip，
  普通 Mapping / 本地自签一律拒绝。

## 4. 未运行 / 未验证（诚实标注）

| 项 | 状态 |
|---|---|
| 真实两窗口 Smoke（任一 Student） | 未启动 —— 缺 Bundle 对象 + 本地 jax/craftax |
| 真实训练更新 | 未执行 —— 缺 CanonicalDiCodeOneUpdateRuntime 对象 |
| 正式实验 | 未启动 —— FORMAL_EXPERIMENT_AUTHORIZED=false |
| 冻结 D052 历史证据 | 未修改（只读） |
| 全部 REAL_* 旗标 | 恒 False |

满足阻断解除路径后，本方向即达 `DIRECTOR_SMOKE_HANDOFF_READY`，交由总监核验并执行 Smoke。
