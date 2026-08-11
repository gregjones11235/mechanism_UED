# 生产就绪状态（production_readiness）— 方向二 TWO_REAL_WINDOWS_READY_FOR_AUDIT

本文件冻结方向二（Simulator-Grounded Feedback-Adaptive Six-LLM UED）在本轮
结束时的生产就绪状态。诚实性规则全程适用：未运行的内容一律标注未运行；
不因为有 preflight/门禁结构就声称真实执行完成。

## 1. 目标状态与本轮范围

| 项 | 值 |
|---|---|
| 分支 | `henry/ba-bagr-ued-review-board-v2` |
| 本轮起点 HEAD | `74b8f5fd6698562c1b107c305ce10a33b518b8ee` |
| 目标状态 | TWO_REAL_WINDOWS_READY_FOR_AUDIT：生产路径代码建成、缺资产显式阻断、冻结本文件、推送远端、等待外部审核 |
| 本轮是否启动真实两窗口 | **否**（本地无真实 LLM transport、无共享运行时资产） |
| 本轮是否启动长跑 | **否**（`E2_PILOT_AUTHORIZED=false`，入口拒绝启动） |
| 测试 | 426 passed 维持不变；本轮**零新增测试、零测试扩建**（指令要求停止测试扩张） |

## 2. 授权与旗标冻结（不可手工翻动）

* `constants.py` 全部 REAL_* 能力旗标恒为 False：
  `REAL_LLM_CALLS_AUTHORIZED / REAL_SIMULATOR_PROBE_AUTHORIZED /
  TRAINING_AUTHORIZED / FORMAL_EVALUATION_AUTHORIZED /
  REAL_CHECKPOINT_LOADED / REAL_SIMULATOR_PROBE / REAL_ENVCODER_USED /
  REAL_TRAINING_UPDATE_EXECUTED / SOTA_INTEGRATION_READY`。
  `_assert_authorization_posture()` 在控制器构造时对
  `NEVER_TRUE_REAL_CAPABILITY_FLAGS` 全表复检，任一为 True 即拒绝构造。
* 生产授权唯一通道 = 运行时授权对象 `RealRuntimeAuthorization`
  （`runtime_authorization.py`）：四个布尔授权默认全 False，只能显式构造；
  授权严格分层（real_training ⇒ real_probe ⇒ real_envcoder ⇒
  real_llm_backend），不一致的授权集在构造时拒绝
  （INCONSISTENT_RUNTIME_GRANTS）；且仅在 `EXECUTION_MODE_REAL` 下生效。
* `REAL_MODE_BLOCKED_NO_LLM_BACKEND` 语义：请求 REAL 但真实 LLM transport
  缺席时，`assert_real_mode_servicable()` 直接阻断——**禁止回退 Mock 后自称
  真实**（NO_SILENT_FALLBACK）。控制器侧另有反向检查：真实授权的运行必须
  有真实调用（real_calls>0）且零 mock 调用（MOCK_CALLS_IN_REAL_RUN /
  REAL_LLM_USAGE_MISSING）；真实授权下 mock/replay 后端构造即被拒
  （REAL_RUN_BACKEND_NOT_REAL），非 real_simulator=True 的 runner 同理
  （REAL_RUN_PROBE_RUNNER_NOT_REAL）。

## 3. 共享运行时资产状态（consume-only，全部缺席）

方向二只消费、不实现共享基础设施（无 loader、无 registry、无 codec）。
`shared_runtime_binding.py` 的五个槽位本轮全部 EMPTY，状态
`BLOCKED_WAITING_SHARED_RUNTIME`：

| 槽位 | 资产 | 状态 |
|---|---|---|
| student | 共享 StudentAdapter（唯一 StudentIdentity，必须 = `PERSISTENT_RMT16_ORIGINAL_VTRACE_98304`） | BLOCKED_WAITING_SHARED_RUNTIME |
| reference | 共享 ReferenceAdapter（唯一 ReferenceIdentity + 输出泄漏守卫） | BLOCKED_WAITING_SHARED_RUNTIME |
| probe_runner | 共享 CandidateProbeRunner（真实 reset/rollout/transition 记账） | BLOCKED_WAITING_SHARED_RUNTIME |
| anchor_manifest | 跨方向共享冻结四锚点 manifest | BLOCKED_WAITING_SHARED_RUNTIME |
| training | 共享全状态 checkpoint + optimizer 表面 | BLOCKED_WAITING_SHARED_RUNTIME |

`resolve_shared_runtime()` 缺任一槽即整体阻断；生产入口不得用本地替身降级。
已核实事实：共享 StudentAdapter 仅存在于 mechanism_UED_sim_foundation
worktree；ReferenceAdapter、共享 CandidateProbeRunner、冻结 AnchorManifest、
正式资产注册表、签名全状态 checkpoint 在本 worktree 全部缺席。本地
Python 环境亦无 jax/craftax。

## 4. 两入口阻断码清单（本轮实测）

### 4.1 `scripts/run_e2_real_two_window.py`（P0-4 + P0-6）

`--check-only` 与直接运行同果：打印完整阻断清单、退出码 1。实测 10 项：

| 阻断码 | 数量 | 说明 |
|---|---|---|
| STUDENT_INIT_CONTRACT_NOT_INJECTED | 1 | 共享 StudentInitContract 未注入（consume-only） |
| REAL_MODE_BLOCKED_NO_LLM_BACKEND | 1 | 无真实 LLM transport 闭包 |
| BLOCKED_WAITING_SHARED_RUNTIME | 5 | 五个共享资产逐一列出 |
| LOCAL_RUNTIME_MODULE_MISSING | 2 | jax / craftax 本地不可导入 |
| REAL_BACKEND_IDENTITY_UNDECLARED | 1 | 真实后端/模型身份未显式声明（审计字段） |

资产齐全路径（本轮不可达）：Window k = 六角色 → hypothesis + prediction
signature → AxisDirective → 真实 EnvCoder（唯一模板
`ENVCODER_UNIQUE_TEMPLATE_V1`、compile/import/reset/step 四环验证、修复上限
`ENVCODER_MAX_REPAIR_ATTEMPTS=2`、超限 REAL_ENVCODER_REPAIR_BUDGET_EXHAUSTED、
无符号回退）→ 真实 Student/Reference Probe → 冻结 feedback_k（同窗应用仍
SAME_WINDOW_REVISION_FORBIDDEN）；Window k+1 = 六角色只读 feedback_k（恰好
滞后一窗）→ 每 active hypothesis 恰好一个 verdict → plan_{k+1} → 真实候选
Probe → criterion-wise 选择 → 12 dynamic + 4 anchors → **恰好一次** optimizer
update（REAL_TRAINING_STEP_COUNT_MISMATCH 强制）→ checkpoint save/load
round-trip（前后哈希必须变化、reload 必须成功）。每条生产反馈记录绑定
`RealProbeProvenance`：源窗/源 plan/假设 ids/候选哈希/变更轴/恒定轴/预测
签名/观测残差/CI 样本数/Student+Reference 身份哈希/checkpoint 哈希/seed
bank/transitions/runner_id；未知、错窗、错 plan、错 family、错身份全部
fail closed。**生产路径禁止用 candidate hash 生成指标**：符号 runner 与
preflight 接缝的 runner_id 是否决名单（FORBIDDEN_PRODUCTION_RUNNER_IDS），
不作为数据来源。

### 4.2 P0-6 REQUEST_CONTROL 复验（零代码改动，只读确认）

控制器既有 phase-B halt 已满足全部要求：board 请求人工控制后立即停止——
不调 EnvCoder、不生成 Probe、不训练、不推进窗口、写入
HumanDecisionArtifact（绑定该窗 board_hash，tutor 引用经 view 解析为真实
store id）；停止窗 phase 停留在 BOARD，后续任何 verdict/plan 改动同样触发
SAME_WINDOW_REVISION_FORBIDDEN；`evaluate_final_batch` 对停止的循环恒
final=false。入口只消费该 halt 的公共摘要面（request_control_stopped /
stopped_window / human_decision_artifact），无任何旁路续跑路径。

### 4.3 `scripts/run_e2_longrun.py`（P0-5）

三配置映射（不 fork 新模式）：normal_feedback → MODE_NORMAL_FEEDBACK；
no_feedback_control → MODE_STATIC_LLM（NullFeedbackView 结构性屏蔽，非
新代码路径）；shuffled_feedback → MODE_SHUFFLED_FEEDBACK（store 恒诚实，
隔离发生在 view 层：冻结可复算置换 + 匿名 id + 身份侧信道屏蔽）。

compute-match 契约（实测通过，篡改即 COMPUTE_MATCH_BROKEN）：三模式同 6
board 调用/窗、同 EnvCoder 预算（1 次唯一模板调用/窗 + 修复上限 2）、同
64→24→12+4 漏斗、同各级 student/reference episodes（2+1 / 8+4）、同 seed
schedule、同 4 锚点、同 checkpoint cadence；探针开销 = 61440 transitions/窗
（三模式逐字节相同，计为 UED 开销）；`total_env_steps` 计 Student 训练
环境步数，三模式均须恰为 `TOTAL_ENV_STEPS_LONG_RUN=98304`（= 强 Student
基线 PERSISTENT_RMT16_ORIGINAL_VTRACE_98304 的训练预算；8 窗 × 12288 步，
每窗恰好一次 optimizer update）。实测启动结果：`E2_PILOT_AUTHORIZED=false`
拒绝启动、退出码 1——**本轮不启动，与指令一致**。

## 5. 本轮提交与生产文件清单

| # | 提交 | 内容 |
|---|---|---|
| 1 | `2a7d71d` feat(e2): add real backend and envcoder execution | runtime_authorization.py、real_call_journal.py（哈希链、重复成功调用拒绝、retry 上限）、real_env_coder.py、llm_backend.py（RealBackendAdapter + journal 钩子）、execution_mode.py（运行时授权通道）、constants.py（生产就绪常量块） |
| 2 | `2ef0eff` feat(e2): connect real student-reference probe | shared_runtime_binding.py（五槽 consume-only）、real_probe_feedback.py（真实 Probe 适配 + RealProbeProvenance + fail-closed 绑定 + reference_identity_hash 生产路径填充） |
| 3 | `981d5df` feat(e2): add two-window real execution entrypoint | controller.py / student_binding.py 最小接线（全部可选 kwargs，默认路径逐字节不变）、real_probe_feedback.py 探针证据轨迹、shared_runtime_binding.py manifest 对象保留、scripts/run_e2_real_two_window.py |
| 4 | `e5d1471` feat(e2): add compute-matched longrun launchers | scripts/run_e2_longrun.py |
| 5 | 本提交 docs(e2): freeze production readiness state | 本文件 |

生产路径文件（全部新增或仅追加可选参数；冻结模块零改动）：
`d052/feedback_llm_ued/{runtime_authorization,real_call_journal,
real_env_coder,shared_runtime_binding,real_probe_feedback}.py`、
`scripts/run_e2_{real_two_window,longrun}.py`、既有
`llm_backend.py / execution_mode.py / controller.py / student_binding.py /
constants.py` 的最小追加。frozen replay bundle、symbolic metrics、
deterministic mock backend、fixture Student、test-only anchor **均未被**
生产模块作为数据来源；`d052/reconciliation/replay.py` 与全部冻结历史产物
零改动。

## 6. LEGACY_REPLAY_BLOCKED_NON_PRODUCTION（Arm-C 同分问题记录）

按指令：该问题不进入正式生产路径；不修改任何冻结历史证据；仅记录。

* 事实一：`d052_r3_0028` 与 `d052_r3_0007` 在 Arm-C 评分上精确同分
  `0.81423508564678215`，排名 8/9，恰好横跨 k=8 选择切线。
* 事实二：`d052/reconciliation/replay.py:77` 使用裸
  `np.argsort(-scores)[:8]`，无稳定 tie-break；同分对的入选与否依赖
  argsort 的实现相关次序。
* 事实三：冻结选择哈希 `C_selection_hash=868a57268d66b90b` 对应的选择集
  包含 `0028`；而 `ranking_C.json` 的 rank 列与该选择集自相矛盾；
  `selector_config` 中 "argsort index order" 的声明被上述事实证伪。
* 处置：状态标记 **LEGACY_REPLAY_BLOCKED_NON_PRODUCTION** —— 冻结历史
  Replay 合同不得继续扩张，该同分问题不作为生产路径输入，不回填、不改写、
  不重新裁决任何冻结证据。未来若重启 Replay，必须先以显式稳定 tie-break
  合同取代裸 argsort，并对该同分对重新取证。

## 7. 未运行 / 未验证清单（诚实标注）

| 项 | 状态 |
|---|---|
| 真实两窗口执行（REAL_LLM/真实 EnvCoder/真实 Probe） | 未运行——本地无真实资产，入口阻断（§4.1） |
| 恰好一次 optimizer update + checkpoint reload | 未运行——依赖共享训练合同，缺席 |
| 长跑三配置 | 未启动——E2_PILOT_AUTHORIZED=false |
| 生产路径新模块的行为测试 | 本轮不新增测试（指令）；生产路径经静态 fail-closed 冒烟验证（导入、阻断清单、授权分层、compute-match 篡改检出），标注为冒烟级证据 |
| mock/符号路径 | 426 passed 全绿（本轮逐提交复检） |

## 8. 审核前约束

完成代码并推送远程后停止；审核通过前不启动完整长跑、不翻动任何 REAL_*
旗标、不注入任何"临时"共享资产替身。解除阻断的唯一合法路径是：共享运行
时资产由其 owner 显式注入 + 真实 LLM transport 闭包显式注入 + 后端/模型
身份显式声明 + （长跑）`E2_PILOT_AUTHORIZED` 按流程授权。
