# CC3 后续轮：内部 P0 修复完成后的真实路径阻断清单

分支 `henry/ba-bagr-ued-review-board-v2`。本轮（16 个原子提交，`615e619…<报告提交>`）
逐项修复外部审计的 18 项 P0 发现。全部内部 P0 修复已完成并被 §19 新增测试覆盖；
**真实执行仍被外部共享资产阻断**——这是本轮唯一可诚实到达的状态。

## 1. 两窗口入口实测阻断清单（`scripts/run_e2_real_two_window.py --check-only`）

退出码 `1`，共 **10 项**阻断（本轮逐提交复检，恒定）：

| 阻断码 | 数量 | 剩余原因 |
|---|---|---|
| `STUDENT_INIT_CONTRACT_NOT_INJECTED` | 1 | 共享 StudentInitContract 需生产 launcher 显式注入（consume-only，方向二不自建 loader） |
| `REAL_MODE_BLOCKED_NO_LLM_BACKEND` | 1 | 本 worktree 无真实 LLM transport 闭包；回退 Mock 自称真实被禁（NO_SILENT_FALLBACK） |
| `BLOCKED_WAITING_SHARED_RUNTIME` | 5 | 五个共享资产（student/reference/probe_runner/anchor_manifest/training）全部缺席 |
| `LOCAL_RUNTIME_MODULE_MISSING` | 2 | 本地 Python 无 jax / craftax，无法导入共享模拟器运行时 |
| `REAL_BACKEND_IDENTITY_UNDECLARED` | 1 | 真实后端/模型身份需显式声明（审计字段，绝不静默推导） |

## 2. 各 P0 修复对应的接缝门（全部已建成 + 已测试）

| P0 | 接缝 | 修复要点 | §19 测试模块 |
|---|---|---|---|
| P0-0 | 六角色 usage | 板面调用计数按执行模式感知 | test_feedback_llm_ued_real_board_usage.py |
| P0-1 | 六角色顺序链 | 每个角色只消费其上游的结构化输出，链断裂 fail closed | test_feedback_llm_ued_sequential_board.py |
| P0-2 | 可执行工件入 Probe | 候选以绑定工件哈希进入探针；无工件 family 拒绝 | test_feedback_llm_ued_real_envcoder_binding.py |
| P0-3 | 工件验证隔离 | EnvCoder 产物验证在子进程沙箱内执行 | test_feedback_llm_ued_envcoder_sandbox.py |
| P0-4 | 逐指令严格绑定 | AxisDirective → 工件内容逐字段绑定，篡改拒绝 | test_feedback_llm_ued_*（P0-4 系） |
| P0-5 | 全量调用日志 | 每真实调用 transport + PARSED/SCHEMA_FAILED 双条目，哈希链持久化 | test_feedback_llm_ued_real_call_journal.py（56） |
| P0-6 | EnvCoder 序列记账 | 修复链按 n_calls 精确消费；缺失计数 fail closed；授权随快照持久/恢复 | test_feedback_llm_ued_envcoder_sequence.py |
| P0-7 | 共享运行时身份 | bindings_hash 折真实资产身份而非状态串；注册表签发；缺席保持阻断 | test_feedback_llm_ued_shared_runtime_identity.py（51） |
| P0-8 | 签名不可变 ProbeResult | 只消费 registry 签名的不可变结果；记账恒等式；CI 只数有效 episodes | test_feedback_llm_ued_signed_probe_result.py（57） |
| P0-9 | Provenance 绑定闭环 | 未知/错窗/错 plan/重复/缺证据/已入库一律拒绝；禁 `except KeyError: continue` | test_feedback_llm_ued_signed_probe_result.py |
| P0-10 | 恰好一次更新 | 两窗口 smoke Δ=[0,1]，总数校验 fail closed | test_feedback_llm_ued_two_window_update_count.py（17） |
| P0-11 | 全状态 round-trip | 只认 director-verifier 的 FullStateRoundTripResult；无证明即 CHECKPOINT_ROUND_TRIP_PASS=false | test_feedback_llm_ued_full_state_round_trip.py（37） |
| P0-12 | 同形屏蔽对照 | no-feedback = MaskedFeedbackView（同条目数/同字段集/受控 NULL/MASK），非空视图 | test_feedback_llm_ued_masked_feedback_control.py（15） |
| P0-13 | 运行时计算台账 | 每模式 RealComputeLedger；截断运行 = COMPUTE_MATCH_EXECUTION_INCOMPLETE | test_feedback_llm_ued_runtime_compute_ledger.py（11） |
| P0-14 | 训练预算语义 | 默认 BLOCKED_WAITING_DIRECTOR_BUDGET_DECISION；未决禁止启动长跑 | test_feedback_llm_ued_runtime_compute_ledger.py |
| P0-15 | Arm-C 同分 | 只读分类 FROZEN_EVIDENCE_INTERNALLY_INCONSISTENT；正式 LEGACY_REPLAY_BLOCKED_NON_PRODUCTION | test_arm_c_tie_resolution.py（9） |
| §19 收尾 | 授权 + 路径阻断 | 授权阶梯、servicable 门、入口阻断枚举 | test_feedback_llm_ued_real_authorization.py（16）、test_feedback_llm_ued_real_path_blockers.py（11） |

## 3. 阻断是否可被"补丁"绕过？——否

每个真实接缝都是**消费端 fail-closed 契约**：

* 真实授权要求真实后端（`REAL_RUN_BACKEND_NOT_REAL`）、真实 runner
  （`REAL_RUN_PROBE_RUNNER_NOT_REAL`）、真实 StudentInitContract
  （`STUDENT_INIT_CONTRACT_MISSING`）；
* 真实运行结束必须有真实调用（`REAL_LLM_USAGE_MISSING`）且零 mock 调用
  （`MOCK_CALLS_IN_REAL_RUN`）；
* 共享槽只接受注册表签发身份（P0-7），探针只消费签名不可变结果（P0-8），
  训练只认 director-verifier 证明（P0-11）；
* 两窗口 smoke 只允许恰好一次更新（P0-10），未决预算语义禁止长跑（P0-14）。

## 4. 剩余阻断的唯一解除路径

1. 共享运行时资产由其 owner 显式注入（五个槽全部 BOUND）；
2. 真实 LLM transport 闭包显式注入；
3. 后端/模型身份显式声明；
4. （长跑）`E2_PILOT_AUTHORIZED` 按流程授权 + 预算语义由 director 裁决；
5. 本地 jax/craftax 可导入。

在满足之前，入口恒定打印本清单并退出 1——不启动、不注入替身、不翻旗标。
