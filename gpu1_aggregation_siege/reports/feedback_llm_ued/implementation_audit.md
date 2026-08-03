# SIMULATOR-GROUNDED FEEDBACK-ADAPTIVE LLM-UED — 实现审计（implementation_audit）

- 分支：`henry/ba-bagr-ued-review-board-v2`
- worktree：`C:/Users/Lenovo/Desktop/dicode-codex-director/mechanism_UED_bagr_ued_fix1_worktree`
  （只在此 worktree 施工；未切换/未覆盖 CC1、CC2 分支，未整体合并 Mason 分支）
- 代码根：`gpu1_aggregation_siege/d052/feedback_llm_ued/`（34 个模块，7768 行）
- 测试：`gpu1_aggregation_siege/d052/tests/test_feedback_llm_ued_*.py`
  （16 个文件，366 用例）
- 本文件只陈述**已验证**的事实；未运行/未实现内容一律显式标注。

## 1. 提交序列（C1–C16，全部原子提交、显式路径 add、无 amend/force/rebase）

| # | 主题 | SHA |
|---|---|---|
| C1 | Backend 抽象 + 旗标骨架 | `91098c5` |
| C2 | CC4 StudentAdapter 薄绑定 + 反馈身份 | `0e26453` |
| C3 | 真实 Craftax Probe 接口（fail-closed 接缝） | `ff1f9ef` |
| C4 | 行为失败证据 + 不确定性 CI（board 输入层） | `48b7c10` |
| C5 | AxisDirective 受控规格（board→EnvCoder 合同） | `89b2985` |
| C6 | 六角色 Review Board + FeedbackView 抽象 | `98b9bd5` |
| C7 | 独立 EnvCoder + compile/reset/step 门禁 | `03c9279` |
| C8 | Controller 双窗口状态机重写 + 负测 | `c943a1d` |
| C9 | 三模式对照隔离（结构性 Null + 冻结置换） | `6008a0c` |
| C10 | RETIRE 生命周期（cooldown + 受控重开） | `a062e98` |
| C11 | REQUEST_CONTROL 阻断 + HumanDecisionArtifact | `dd6c3b0` |
| C12+C13 | 共享 Soft Copeland 选择层 + anchor manifest 接缝 | `1974119` |
| C14 | 内容哈希重算（CONTENT_HASH_MISMATCH） | `5a7b876` |
| C15 | 持久化 + 跨窗恢复等价 | `a5ed224` |
| C16 | 报告收尾 + 旗标翻转（本次提交） | 见 git log |

基线演进：554→749→889→920 passed；6 个既有环境性失败名单全程未变。

## 2. 模块清单（全部已实现、全部有测试覆盖；行数为 wc -l 实测）

### 2.1 核心闭环
| 模块 | 行数 | 职责 |
|---|---|---|
| `controller.py` | 1046 | 双窗口状态机（A–E phase 机）、三模式、verdict 应用校验、RETIRE/REQUEST_CONTROL、summary 对比、C15 续跑支持 |
| `review_board.py` | 326 | 六角色编排（固定顺序）、BoardOutput=verdict+新假设+AxisDirective+逐族提案+global_risk |
| `deterministic_reconciler.py` | 337 | 提案→FamilyAllocation 收口：诚实性重标、悬空引用、退休优先、预算上限、锚位预留 |
| `hypothesis_ledger.py` | 142 | 假设账本：唯一状态写入者，revision_history 哈希链 |
| `simulator_feedback_store.py` | 217 | 反馈库：reference 白名单、formal 源拒绝、bind_match 重哈希、dump |
| `plan_revision.py` | 143 | PlanRevisionRecord：引用并集强制 label、budget_changes 视图 |
| `expected_observed.py` | 151 | 逐指标 agree/neutral/opposite 评级 + MAJORITY 总判 |
| `persistence.py` | 292 | C15：快照/原子写/重算校验/逐假设链校验/恢复续跑 |

### 2.2 六角色（mock_rule，ENGINEERING_SCAFFOLD）
`student_modeler.py`(108) / `behavior_auditor.py`(116) /
`causal_failure_analyst.py`(223) / `intervention_tutor.py`(242) /
`explorer.py`(232) / `critic_skeptic.py`(162)；board 输入层：
`behavior_failure.py`(215) + `uncertainty.py`(73)。

### 2.3 EnvCoder 与门禁
`env_coder.py`(172)：SpecEnvCoder 确定性符号生成 + LLM 接缝 Blocked；
`env_coder_gate.py`(141)：compile/reset/step 三级 fail-closed。

### 2.4 探针与选择
| 模块 | 行数 | 职责 |
|---|---|---|
| `simulator_probe.py` | 391 | 符号 runner + 64→64→24→12+4 漏斗 + stage-2 Copeland 选择 + 转移核算 |
| `real_simulator_probe.py` | 389 | ExecutableCandidate/RealTaskParamsAdapter/ProbeExecutionContext/Craftax 真实接缝（未授权 Blocked） |
| `multi_criterion_selection.py` | 174 | 八准则分离保存 + 封装共享 soft_copeland_rank（不分叉）+ 族多样性贪心 |
| `anchor_manifest.py` | 117 | AnchorManifestSource：共享冻结 manifest 只读绑定，缺失→BLOCKED |
| `environment_generator.py` | 149 | 计划→恰好 64 候选，假设区分绑定 |
| `axis_directive.py` | 174 | 受控轴移动规格（treatment/control、held constants、期望签名） |
| `skill_preflight_core.py` | 208 | Mason JAX-free 纯核心选择性复用（route/prereq，来源注释，非分支合并） |

### 2.5 合同、后端与隔离
`feedback_contracts.py`(272)：Candidate/ProbeMetrics/CurriculumPlan/Envelope；
`feedback_view.py`(250)：Normal/Null/Permuted 三视图；
`llm_backend.py`(278)：UsageStats + Mock/Replay/Real 适配器；
`execution_mode.py`(173)：ExecutionMode + FeedbackLaunchGate；
`student_binding.py`(175)：固定身份 + CC4 fail-closed + 训练 no-op 记账；
`human_decision.py`(89)；`formal_isolation.py`(99)；
`constants.py`(384)；`synthetic_feedback.py`(75)；`__init__.py`(33)。

## 3. 关键审计结论

### 3.1 双窗口时序（REQUEST_CHANGES 修订核心）
- 窗口 k 六角色只能读 ≤k−1 的冻结反馈；feedback_k 产生后本窗只能原子
  写入并冻结；修订只能由窗口 k+1 的完整六角色显式引用 feedback_k 产生。
  状态机 phase 标记 + 负测证明：same-window apply verdict/改计划 →
  `SAME_WINDOW_REVISION_FORBIDDEN`；引用同窗 feedback → `FUTURE_FEEDBACK_ID`；
  缺失 → `UNKNOWN_FEEDBACK_ID`；重复 → `DUPLICATE_FEEDBACK_CITATION`。
- **NEXT_WINDOW_REVISION_ONLY=True、SAME_WINDOW_REVISION_REJECTED=True**（有负测）。

### 3.2 对照隔离（结构性而非提示词级）
- static：`NullFeedbackView` 类型级不持有 store 引用；测试断言 board 上下文
  零反馈载荷。**STATIC_FEEDBACK_STRUCTURALLY_HIDDEN=True**。
- shuffled：冻结可复算置换（仅由 (mode, 窗口, SEED_SCHEDULE_HASH)+记录集
  派生），匿名化 id 屏蔽身份侧信道；负测证明上下文不可还原真实配对。
  **SHUFFLE_PERMUTATION_FROZEN=True**。
- compute-matched：三模式每窗 7 次 LLM 族调用、61440 transitions、同 seed。

### 3.3 Soft Copeland 与 anchors（诚实性）
- 八准则分离保存；排序**复用** `bagr_ued.soft_copeland.soft_copeland_rank`
  （同一 canonical 协议，ranking_hash 逐字节一致有测试），**未分叉**。
- 共享冻结 anchor manifest 本 worktree **不存在** →
  `BLOCKED_SHARED_ANCHOR_MANIFEST` fail-closed；锚位为脚手架占位
  （`SCAFFOLD_PLACEHOLDER_NOT_SHARED`），**SHARED_ANCHOR_MANIFEST_BOUND=False**。

### 3.4 哈希重算与持久化
- C14：七类对象外部携带哈希一律重算逐字比较（CONTENT_HASH_MISMATCH），
  Envelope 入档 prompt 并重算 prompt/request/response 三哈希；tamper 负测覆盖。
- C15：窗口边界全冻结态快照 + snapshot_hash；tmp+os.replace 原子写；
  load 重算 + 逐假设 revision 链校验；冻结点快照→恢复→续跑与不间断运行
  summary **逐字节一致**（normal/shuffled），fresh subprocess 哈希一致，
  篡改 → `HASH_CHAIN_BROKEN`；REQUEST_CONTROL 停环恢复后保持停止。

### 3.5 Student 与训练
- 固定 `PERSISTENT_RMT16_ORIGINAL_VTRACE_98304`；CC4 共享 StudentAdapter
  缺失 → `STUDENT_ADAPTER_MISSING` fail-closed，未另建 loader/registry/codec；
  **REAL_CHECKPOINT_LOADED=False**。训练接缝 no-op 记账
  （SKIPPED_UNAUTHORIZED），**REAL_TRAINING_UPDATE_EXECUTED=False**。

## 4. 验证矩阵（证据）

| 主张 | 证据 |
|---|---|
| 366 个方向二测试全绿 | `test_report.md` §2（逐文件计数） |
| 全量 d052 套件 920 通过 / 6 既有环境性失败 | `test_report.md` §3 |
| 闭环数值（三模式 6 窗，新架构） | `final_implementation_report.md` §5（smoke 真实输出） |
| 双窗口时序负测 | `test_feedback_llm_ued_controller.py`（57 用例含同窗禁止/未来/重复/缺失引用） |
| 隔离负测 | `test_feedback_llm_ued_view_isolation.py`（18 用例） |
| 跨窗恢复等价 + fresh-process + 篡改矩阵 | `test_feedback_llm_ued_persistence.py`（31 用例） |
| 示例 JSON 非伪造 | 三个 example JSON 由真实 6 窗 normal 运行直接导出（本次重新生成） |
| 确定性 | 两次独立运行 summary 逐字节一致（controller/persistence 测试双重锁定） |

## 5. 已知限制与剩余阻塞（如实）

1. **探针是符号的**：本地无 JAX/Craftax，`REAL_SIMULATOR_PROBE=False`、
   `REAL_SIMULATOR_PROBE_STATUS=BLOCKED_NO_LOCAL_CRAFTAX`；真实接缝存在但
   未授权即 Blocked。**REAL_ENVCODER_USED=False**（EnvCoder 为符号生成 +
   LLM 接缝 Blocked）。
2. **LLM 是确定性 mock**：`real_calls=0` 有运行时断言；全部角色逻辑为
   ENGINEERING_SCAFFOLD 级证据。
3. **无训练步**：`TRAINING_AUTHORIZED=False`，无 optimizer step/checkpoint。
4. **CC4 StudentAdapter 缺失**：REAL_CHECKPOINT_LOADED=False。
5. **共享冻结 anchor manifest 缺失**：SHARED_ANCHOR_MANIFEST_BOUND=False。
6. **push 网络受阻**（任务 #87）：GitHub 推送统一在本地 Windows 仓库完成，
   网络恢复前不重试。
