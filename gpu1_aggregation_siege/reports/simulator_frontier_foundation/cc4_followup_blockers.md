# CC4 follow-up blockers（修正版，动态可评估）

阶段：**REAL_PATH_CONTRACT_READY**（生产合同已实现并被 18 个专用测试文件钉住）；
执行：**BLOCKED_WAITING_SHARED_RUNTIME** —— 本轮未启动任何真实 window / LLM / 训练，
`REAL_ACTUAL_N_EXECUTED / REAL_TWO_LLM_EXECUTED / REAL_ONE_UPDATE_EXECUTED /
REAL_MIXED_START_UPDATE / CHECKPOINT_RELOAD / FORMAL_EVALUATION_EXECUTED` 全部为 false。

状态声明（修正）：不再使用 `ONE_REAL_FRONTIER_WINDOW_CODE_READY` 或
`ONE_REAL_FRONTIER_WINDOW_READY_FOR_EXECUTION`。本轮交付的是**修正后的生产合同 +
专用测试**，不是一次可执行的真实窗口。

## 阻断清单与解除条件

| # | 阻断 | 含义 | 解除条件（总控/审核方动作） |
|---|------|------|-----------------------------|
| 1 | `BLOCKED_TRAINING_SURFACE_PENDING_R9` | RMT16 训练面（save/restore_full_state）为只读挂起；`_probe_training_surface` 异常推断探针已删除 | R9 训练面落地，并由总控签发 `TrainingSurfaceCapability` 描述符 |
| 2 | `BLOCKED_NO_SIGNED_TRAINING_SURFACE_CAPABILITY` | 预检只接受**签名、可验证、绑定挂载适配器身份**的能力描述符；自签/伪造/Mapping 一律拒绝 | 总控签发非合成签名的能力描述符（`signature_ref` 不以 `SYNTHETIC_SIGNATURE_` 开头） |
| 3 | `BLOCKED_WAITING_CONTROLLER_SIGNED_REGISTRY_BUNDLE` | 联合 fresh-process restore 只认总控签名 `ProductionRegistryBundle` | 总控签发 RegistryBundle |
| 4 | `BLOCKED_SHARED_ANCHOR_MANIFEST` | 12+4 锚点组合硬阻断，绝不自拟锚点科学 | 总控签发冻结共享锚点 manifest |
| 5 | `BLOCKED_WAITING_FROZEN_FORMAL_ASSET_REGISTRY` | 正式资产 registry 只从注入槽读取（PRODUCTION usage） | 总控通过 `inject_frozen_formal_asset_registry` 注入冻结 registry |
| 6 | `SAVED_POLICY_MEMORY_BLOCKED_NO_MEMORY_ARTIFACT` | 正式主记忆模式需要真实 artifact（sha256/身份/规格三重绑定） | 提供 artifact + 经授权的 loader |
| 7 | `REAL_TWO_LLM_BLOCKED_NO_AUTHORIZED_CLIENT` | 生产双 LLM 路径要求 `AuthorizedTwoLLMRuntime`（mint-only 授权 + 链式日志）；绝不回退 fake | 总控授权真实 LLM 客户端并铸造授权 |
| 8 | `BLOCKED_NO_BOUND_ORIGINAL_TRAINING_RUNTIME` | 原 loss/update 必须以 `OriginalTrainingRuntime` 绑定到达（源码哈希 + 重算哈希），拒绝裸 callable | 从共享训练运行时提供可绑定的原 loss/update |
| 9 | `BLOCKED_OPTIMIZER_STEP_BASELINE_UNMEASURED` | 优化步进基线必须由 `loaded_state.global_step` 机械给出 | 挂载含 `global_step` 的完整 checkpoint |
| 10 | `BLOCKED_NO_INJECTED_TASKPARAM_APPLY_FN` | 分布 taskparams 必须经注入的 `taskparam_apply_fn` 执行 | 注入经过审计的 taskparam 应用面 |
| 11 | `BLOCKED_REFERENCE_IDENTITY_PENDING_CONTROLLER_DESIGNATION` | Reference 身份由总控指定，绝不自拟 | 总控指定 Reference 候选身份 |
| 12 | `BLOCKED_WAITING_DIRECTOR_BUDGET_DECISION` | 训练预算语义（TOTAL_FROM_COMMON_INITIALIZATION / ADDITIONAL_FROM_PRETRAINED_CHECKPOINT）由总控签署决定 | 总控铸造 `LongRunBudgetDecision`（非合成签名，total=98304） |
| 13 | `BLOCKED_E3_PREFLIGHT_NOT_EVALUATED` | longrun 门需要真实预检报告（`--preflight-report`） | 先跑 `run_e3_runtime_bundle.py --check-only` 并把报告喂给 longrun |
| 14 | `BLOCKED_AUDIT_APPROVAL_NOT_GRANTED` | 外部审核批准是启动前提 | 审核通过后以 `--audit-approved=true` 显式授予 |

## 本轮已关闭的内部问题（对应 19 个原子提交）

- 生产 archive 写路径只认注入槽，`TEST_ONLY` registry 永不能进入生产槽（C1）。
- VerifiedRestoreContext 只由进程证据铸造；`compute_context_hash` 改为字段级哈希，
  修复了 mint 路径在控制器 bundle 到达时会 `AttributeError` 的潜在缺陷（C2 + C18）。
- 捕获绑定 policy_memory/history_reference，`source_timestep` 记录实际执行步数（C3）。
- actual-N 只从验证过的恢复状态出发，逐分支重恢复，绝不报告部分运行（C4）。
- Reference 分支只消费 Reference 记忆面；身份/checkpoint/记忆三重绑定（C5）。
- 分来源可行性：Student 与 Reference 永不混合成单一成功率（C6）。
- 证据选择器为正式最终权威，SelectionEvidence mint-only（C7）。
- 双 LLM 生产路径：mint-only 授权 + 哈希链日志 + 上限 2；裸 factory 与 fake 拒绝（C8）。
- 0-or-2 决策由证据变化推导，过期/篡改的先前计划强制修订（C9）。
- 12 分布由类型化 Planner 输出确定性编译，绝不接受手造列表（C10）。
- 分布字段真正执行：seed/stochasticity/taskparams 均被消费；逐 episode 全新记忆（C11）。
- 原训练运行时绑定：loss/update 以可验证绑定到达，拒绝裸 callable（C12）。
- 优化更新以机械铸造的 attestation 证明：`before+1`、批量摘要、结构有限性（C13）。
- 全状态 round-trip 证据 + 重放等价：params+step 双向保存 + action/logits/value/memory 等价（C14）。
- 预检不再用异常推断探针；训练面能力改为签名描述符证据（C15）。
- 签名 runtime-bundle 入口成为唯一资产注入通道（`--runtime-bundle`，`--check-only`）（C16）。
- longrun 阻断列表动态求值：真实预检报告、Reference/锚点状态、预算决定、审核批准（C17）。
- 18 个专用测试文件 + 1 个生产修复（C18）。

## 下一步（单一最高依赖）

总控签发**签名 runtime bundle**（含共享锚点 manifest、冻结正式资产 registry、
saved-policy-memory artifact、训练面能力描述符、原训练运行时入口），随后
`run_e3_runtime_bundle.py --check-only` 可跑出真实预检并喂给动态 longrun 门。
