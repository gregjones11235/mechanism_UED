# SIMULATOR-GROUNDED FEEDBACK-ADAPTIVE LLM-UED — 实现审计（implementation_audit）

- 分支：`henry/ba-bagr-ued-review-board-v2`
- worktree：`C:/Users/Lenovo/Desktop/dicode-codex-director/mechanism_UED_bagr_ued_fix1_worktree`
  （只在此 worktree 施工；未切换/未覆盖 CC1、CC2 分支，未整体合并 Mason 分支）
- 代码根：`gpu1_aggregation_siege/d052/feedback_llm_ued/`（34 个模块）
- 测试：`gpu1_aggregation_siege/d052/tests/test_feedback_llm_ued_*.py`
  （17 个文件，388 用例，含 C9 门禁定向测试 18 用例：CC3 旁路/滞后 10 +
  CC4 再识别负测/字节奇偶 6 + CC4 static 独立性 2）
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
| C16 | 报告收尾 + 旗标翻转 | `794507a` |
| CC3-A | C9 门禁：两个隔离旗标置 False + posture 同步（修复前姿态） | `ec4935f` |
| CC3-B | C9 门禁修复：BoardContext 只经 FeedbackView + 恰好 k−1 滞后 + 定向旁路/滞后测试 + 旗标复 True + 报告重定基线 | `921edad` |
| CC4-C | C9 门禁第二轮：两个隔离旗标再置 False + posture 同步（数值侧信道加固前姿态） | `77cea72` |
| CC4-D | C9 门禁第二轮加固（两处 director 发现）：置换视图数值指纹移除（两层一致的族级窗口聚合）+ family-grain 预测签名移除 + **static phase-A 存储读取修复**（冻结空退休生命周期 + 退休查询对 static fail-closed）+ 再识别负测 3 + 全 prompt 字节奇偶测试 3 + static 独立性负测 2 + 旗标复 True + 报告重定基线（本次提交） | 见 git log |

基线演进：554→749→889→920→934→942 passed；6 个既有环境性失败名单全程未变。

## 2. 模块清单（全部已实现、全部有测试覆盖；行数为 wc -l 实测）

### 2.1 核心闭环
| 模块 | 行数 | 职责 |
|---|---|---|
| `controller.py` | 1087 | 双窗口状态机（A–E phase 机）、三模式、verdict 应用校验（恰好 k−1，STALE_FEEDBACK_ID）、RETIRE/REQUEST_CONTROL、summary 对比、C15 续跑支持；CC4：static phase A 使用冻结空退休生命周期，读取 store 的退休查询对 static fail-closed（STATIC_MODE_HAS_NO_RETIREMENT_LIFECYCLE） |
| `review_board.py` | 339 | 六角色编排（固定顺序）、BoardOutput=verdict+新假设+AxisDirective+逐族提案+global_risk；prompt 上下文双重窗口校验 |
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
`behavior_failure.py`(255，BoardContext 只经视图装配、原始 store 被拒绝) +
`uncertainty.py`(73)。

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
`feedback_view.py`(414)：Normal/Null/Permuted 三视图（CC3 门禁：视图构造即
校验恰好同窗记录 + 每视图自带 behavior_evidence 层；CC4 门禁：
`family_level_metrics` 族级窗口聚合为两层唯一发布数值，置换视图的
payload/证据层一致匿名化——id/轴/候选掩码、精确率与缺口由族级聚合重建、
family-grain 预测签名置空）；
`llm_backend.py`(278)：UsageStats + Mock/Replay/Real 适配器；
`execution_mode.py`(173)：ExecutionMode + FeedbackLaunchGate；
`student_binding.py`(175)：固定身份 + CC4 fail-closed + 训练 no-op 记账；
`human_decision.py`(89)；`formal_isolation.py`(99)；
`constants.py`(407)；`synthetic_feedback.py`(75)；`__init__.py`(33)。

## 3. 关键审计结论

### 3.1 双窗口时序（REQUEST_CHANGES 修订核心 + CC3 C9 门禁收紧）
- 窗口 k 六角色只能读**恰好 k−1** 的冻结反馈（CC3 门禁：旧语义 ≤k−1 收紧
  为恰好一窗，更旧/当前/未来记录一律 fail-closed）；feedback_k 产生后本窗
  只能原子写入并冻结；修订只能由窗口 k+1 的完整六角色显式引用 feedback_k
  产生。四层防线：视图构造（`_assert_exact_window`）、controller 视图选择
  （只选 k−1 记录）、controller 引用校验、board 引用校验 + prompt 上下文
  双重窗口检查。状态机 phase 标记 + 负测证明：same-window apply verdict/
  改计划 → `SAME_WINDOW_REVISION_FORBIDDEN`；引用旧/同窗/未来 feedback →
  `STALE_FEEDBACK_ID`；缺失 → `UNKNOWN_FEEDBACK_ID`；重复 →
  `DUPLICATE_FEEDBACK_CITATION`。
- **NEXT_WINDOW_REVISION_ONLY=True（CC3 收紧语义）、
  SAME_WINDOW_REVISION_REJECTED=True**（有负测）。

### 3.2 对照隔离（结构性而非提示词级；CC3/CC4 C9 门禁重验）
- **CC3 门禁历史（如实）**：总控 CC3 审核发现旧实现中 BoardContext 装配
  直接读取原始 SimulatorFeedbackStore，向 static 上下文泄漏证据、向 shuffled
  上下文泄漏证据层身份；两旗标先置 False（Commit A `ec4935f`），修复 +
  定向旁路/滞后测试（`test_feedback_llm_ued_c9_gate.py`，10 用例）全绿后
  复 True（Commit B）。
- **CC4 门禁第二轮历史（如实）**：总控审核发现两处残余泄漏，两旗标再置
  False（Commit C `77cea72`）：(1) shuffled 置换视图的精确探针率/证据缺口
  是候选哈希的确定性指纹（数值侧信道），且 family-grain 预测签名同样关联
  身份；(2) **static 泄漏**——`_run_window` phase A 对 static 仍调用
  `_retirement_state`，其 reopen 门 `_reopen_eligible` 读取原始
  SimulatorFeedbackStore，retired/cooldown 字段进入 static BoardContext；
  仅靠 NullFeedbackView 不足以使 static 上下文反馈无关（此前为空只是
  巧合）。修复（Commit D）：两层一致只发布族级窗口聚合
  （`family_level_metrics`，缺口与 severity 由聚合重建）+ 预测签名置空；
  static phase A 改用**冻结空退休生命周期**（不再调用读 store 的查询），
  `_retirement_state` 对 static 直接 fail-closed
  （STATIC_MODE_HAS_NO_RETIREMENT_LIFECYCLE）。新增负测 8（再识别 3 +
  字节奇偶 3 + static 独立性 2）全绿后复 True。
- BoardContext 只经 FeedbackView 装配：`assemble_board_context` 对原始
  store 直接抛 `BOARD_CONTEXT_STORE_FORBIDDEN`，视图窗口≠证据窗抛
  `BOARD_CONTEXT_WINDOW_MISMATCH`。
- static：`NullFeedbackView` 类型级不持有 store 引用；门禁测试断言满 store
  下 board 上下文零载荷（证据/SR/CI/候选 id/历史全空，序列化扫描无真实 id）；
  CC4 追加：phase A 的 retired/cooldown 字段来自冻结空生命周期（结构性，
  非巧合为空），负测证明仅反馈记录不同的两个 store 下 static 上下文与全部
  六个 board prompt 逐字节一致。
  **STATIC_FEEDBACK_STRUCTURALLY_HIDDEN=True（CC4 门禁后重验）**。
- shuffled：冻结可复算置换（仅由 (mode, 窗口, SEED_SCHEDULE_HASH)+记录集
  派生），匿名化 id 在 prompt 层**与证据层**均屏蔽身份侧信道（证据匿名 id
  与 payload 逐位一致、candidate_id=MASKED_IDENTITY）；CC4 追加：所有
  身份关联数值只发布族级窗口聚合（两层同值）、family-grain 预测签名移除，
  store 联接对手不可收窄到单例、序列化上下文无任何精确逐记录指标；负测
  证明上下文不可还原真实配对、resolve_citation 为唯一还原路径。
  **SHUFFLE_PERMUTATION_FROZEN=True（CC4 门禁后重验）**。
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
| 388 个方向二测试全绿 | `test_report.md` §2（逐文件计数） |
| 全量 d052 套件 942 通过 / 6 既有环境性失败 | `test_report.md` §3 |
| 闭环数值（三模式 6 窗，CC4 门禁后重定基线） | `final_implementation_report.md` §4（smoke 真实输出；static 泄漏修复后原样重跑逐字节一致） |
| 双窗口时序负测（恰好 k−1） | `test_feedback_llm_ued_controller.py`（58 用例含同窗禁止/旧/未来/重复/缺失引用） |
| CC3 C9 门禁旁路/滞后定向测试 | `test_feedback_llm_ued_c9_gate.py`（10 用例：static 满 store 零载荷、shuffled 证据层无身份、混合窗口/旧/当前/未来 fail-closed、端到端逐引用恰好滞后一窗） |
| CC4 C9 门禁再识别/字节奇偶/static 独立性 | `test_feedback_llm_ued_c9_gate.py`（新增 8 用例：两层数值=公开族级聚合、store 联接不可收窄单例、序列化上下文无精确逐记录指标、独立运行全 prompt 字节一致×2+重组一致、外来 store 污染下 static 上下文与六 prompt 逐字节一致、退休查询对 static fail-closed） |
| 隔离负测 | `test_feedback_llm_ued_view_isolation.py`（18 用例）+ `test_feedback_llm_ued_evidence.py`（19 用例含 BOARD_CONTEXT_STORE_FORBIDDEN） |
| 跨窗恢复等价 + fresh-process + 篡改矩阵 | `test_feedback_llm_ued_persistence.py`（31 用例） |
| 示例 JSON 非伪造 | 三个 example JSON 由真实 6 窗 normal 运行直接导出；CC4 门禁后重跑验证数据载荷逐字节不变（normal 路径未受影响），仅描述更新 |
| 确定性 | 两次独立运行 summary 逐字节一致（controller/persistence 测试双重锁定）+ 两次独立运行全 prompt 上下文字节一致（CC4 奇偶测试） |

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
