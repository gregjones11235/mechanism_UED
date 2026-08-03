# 反馈闭环合同（feedback_loop_contract）— 六角色 Review Board + 双窗口状态机版

本文件是 `d052/feedback_llm_ued` 的规范性合同：数据怎么流、谁可以写什么、
什么时候允许调用 LLM、以及每一处 fail-closed 的触发条件。实现与测试必须
与本文件一致；不一致以实现为准修改本文件并在审计中说明。

本版对应 C1–C16 架构（六角色 Review Board + 独立 EnvCoder + 双窗口状态机 +
三模式结构性隔离 + 共享 Soft Copeland + anchor manifest 接缝 + 哈希重算 +
持久化/跨窗恢复）。旧两角色/条件调用门版本已随 C8 废除。

## 1. 双窗口状态机（窗口时序规范）

```
窗口 k（k≥0；窗 0 的 k−1 反馈视图为空，board 仍完整跑 6 次）
 ├─ A. EVIDENCE：行为失败证据（窗 k−1 probe 提取）+ FeedbackView(k−1)
 │     （仅 k−1 及更早的已冻结反馈；static=结构性 NullFeedbackView；
 │      shuffled=冻结可复算置换视图）
 ├─ B. BOARD：完整六角色 Review Board（6 次 LLM 族调用，无条件）：
 │     StudentModeler→BehaviorAuditor→CausalFailureAnalyst→InterventionTutor
 │     →Explorer→Critic/Skeptic；输出：对 k−1 反馈的 verdict（显式引用
 │     feedback_id/hypothesis_id/prediction_signature）+ 新假设（PENDING+
 │     预测签名）+ 受控环境规格（AxisDirective）+ 逐族提案 + global_risk
 ├─ C. REVISION：verdict 应用（仅引用 ≤k−1 的反馈）→ Ledger（哈希链）；
 │     Reconciler→plan_k；八准则 + 共享 Soft Copeland→12 dynamic；
 │     +4 anchors（manifest 绑定）→ 执行批（训练接缝 no-op 记账）
 ├─ D. PROBING：EnvCoder（第 7 次 LLM 族调用）→ compile/reset/step 门禁 →
 │     漏斗 Probe（64→64→24→12+4）→ Expected-vs-Observed 评级 → feedback_k
 └─ E. FROZEN：原子写入 FeedbackStore/Ledger 并冻结。此后窗口 k 禁止任何
       verdict/计划变更；feedback_k 只能由窗口 k+1 的完整六角色引用修订。
窗口 k+1 从 A 重启，消费冻结的 feedback_k —— 修订永远滞后反馈一个窗口。
```

phase 机常量：`PHASE_EVIDENCE / PHASE_BOARD / PHASE_REVISION / PHASE_PROBING
/ PHASE_FROZEN`；状态迁移非法即 fail-closed。

### 1.1 同窗修订禁止（硬规则）

- feedback_k 产生（phase≥PROBING）后，窗口 k 内任何 `apply_board_verdicts`
  或 `revise_plan` → `SAME_WINDOW_REVISION_FORBIDDEN`；
- 引用同窗生成的 feedback_id → `FUTURE_FEEDBACK_ID`；
- 引用不存在于 store 的 feedback_id → `UNKNOWN_FEEDBACK_ID`；
- 同一 verdict 内重复引用 → `DUPLICATE_FEEDBACK_CITATION`；
- P0-6 绑定守卫：verdict 只能作用于其 feedback 的 `distinguishes` 假设，
  window/plan/family 必须匹配，每窗每假设至多 1 个 verdict。
均配负测（`test_feedback_llm_ued_controller.py`）。

## 2. 六角色与调用预算

- 每窗固定 **7 次 LLM 族调用**：6 次 board（顺序固定）+ 1 次 EnvCoder。
  三模式逐窗相等（compute-matched）；旧条件调用门/REUSED 已废除。
- 角色只通过 `FeedbackRoleEnvelope.make` 留痕：prompt 入档，
  `prompt_sha256`、`request_hash=sha256(role,prompt_version,prompt)`、
  `response_hash=sha256(raw)` 全部在构造时重算比对（C14）。
- Critic/Skeptic 裁决规则（mock）：`HIGH_SEVERITY_FLOOR=3`、
  `OPPOSITE_MATCH_FLOOR=2`、`WIDE_CI=0.5`；
  escalate = 存在未评级反馈 或（risk==HIGH 且 ci≥WIDE_CI）；
  endorsed =（risk != HIGH）。
- `REQUEST_CONTROL`（Critic 升级或 Tutor 引用反馈的提案）→ 本窗在 phase B
  后停环：不生成执行批次、不探针、不冻结；`HumanDecisionArtifact` 入
  summary；`LaunchGate.evaluate_final_batch` final=False。

## 3. 三模式对照隔离（结构性）

| 模式 | 反馈视图 | 隔离方式 |
|---|---|---|
| `static_llm` | `NullFeedbackView` | **类型级**不持有 SimulatorFeedbackStore 引用，board 上下文结构性零反馈载荷（非提示词省略） |
| `normal_feedback` | `NormalFeedbackView` | 只读冻结快照，仅 ≤k−1 |
| `shuffled_feedback` | `PermutedFeedbackView` | **冻结可复算置换**：仅由 (mode, 窗口, SEED_SCHEDULE_HASH)+记录集派生，无运行时随机；匿名化 id 呈现，board 上下文不可还原真实 candidate↔feedback 配对 |

三模式保持相同六角色/EnvCoder/probe/训练接缝/seed/预算（每窗 7 次调用、
61440 transitions）；差异只能归因于反馈的使用方式。

## 4. 探针预算与漏斗（不变量）

- 漏斗：64 raw → L1 静态合法性+哈希去重 → 64 fast-probe → L2 留 24 →
  24 full-probe → Copeland stage-2 选 12 + 4 anchors = FINAL_BATCH 16。
- 每窗探针成本恒 **61440 transitions** = 64×3×128 + 24×12×128；
  超预算/非法 stage 即 raise（不静默截断）。
- Reference 只给 episode 级粗统计（`REFERENCE_ALLOWED_FIELDS`），8 类动作
  指导载体在 runner 边界/store 构造/provenance 三处扫描，命中即 raise。

## 5. 多指标选择层（C12，消费共享 Soft Copeland）

- `multi_criterion_selection.stage2_criteria`：8 个准则（front_regret、
  global_regret、behavioral_gap、learning_progress、learnability、
  diversity、global_retention、critic_penalty）逐候选分离保存，clamp 到
  schema 合法区间（regret 在 ProbeMetrics 中仅 ge=0，可 >1，此处
  min(1,·) 收口并注记退化情形）。
- 排序**直接调用** `d052.bagr_ued.soft_copeland.soft_copeland_rank`
  （不分叉、不改写）：同一 canonical 协议，ranking_hash 逐字节一致有测试
  证明；critic_penalty 越低越好；constant 维度→0.5 中性并 provenance 标注。
- 族多样性贪心：eff = copeland_score − STAGE2_FAMILY_PENALTY(0.10)×同族已选数，
  按 (−score, candidate_id) 确定性取 12。
- 旧手写 `_full_score` 已废除。

## 6. Anchors（C13，共享 manifest 接缝）

- 4 个 standard-reset anchors 的正式来源是跨方向共享**冻结 manifest**，
  经 `anchor_manifest.AnchorManifestSource` 显式注入：校验恰好
  GLOBAL_ANCHOR_SLOTS=4 个、无重复、frozen、`manifest_hash` 重算一致。
- 本 worktree 无该 manifest → `AnchorManifestBlocked(
  BLOCKED_SHARED_ANCHOR_MANIFEST)` fail-closed；锚位回退到本地脚手架占位
  `GLOBAL_CANONICAL_ANCHOR_IDS`，绑定标签显式为
  `SCAFFOLD_PLACEHOLDER_NOT_SHARED`；`SHARED_ANCHOR_MANIFEST_BOUND=False`。
  三模式预算不变。

## 7. RETIRE 生命周期（C10）

- `RETIRE_COOLDOWN_WINDOWS=3`：退休后 3 个窗口内该族不得再入计划
  （`FAMILY_IN_COOLDOWN` fail-closed）；cooldown 过后仍需满足重开条件：
  `human_reopen_families` 显式授权，或全部区分性证据晚于退休窗。
- STALE 不得复活 retired 族；retire 决策必须引用反馈（退休是判决）。

## 8. 哈希绑定与重算（C14）

| 对象 | 哈希字段 | 规则 |
|---|---|---|
| CandidateEnvironment / HypothesisRecord / SimulatorFeedbackRecord / CurriculumPlan / PlanRevisionRecord / AxisDirective | candidate_hash / record_hash / record_hash / plan_hash / record_hash / directive_hash | 外部携带的哈希一律**重算并逐字比较**，不一致 `CONTENT_HASH_MISMATCH`；为空则计算填充 |
| FeedbackRoleEnvelope | prompt_sha256 / request_hash / response_hash | prompt 入档，三哈希全部由存储内容重算比对 |

哈希一律复用 `d052.bagr_ued.hashing`（canonical_sha256 / text_sha256 /
verify_content_hash，单一事实源）。

## 9. 持久化与跨窗恢复（C15）

- `persistence.snapshot_controller` 捕获窗口边界全部冻结态（phase 图、
  mode、ledger/store dumps、revisions、plans（按 id 与按窗）、
  _window_feedback、_sequence、_retired_at、human_reopen_families、
  anchor 绑定、runner/backend 计数、训练日志、envelope 审计元数据、
  board 哈希、已完成 WindowRecord、HumanDecisionArtifact）+
  可重算 `snapshot_hash`。
- `save_controller`：tmp + `os.replace` 原子写；`load_controller`：
  重算顶层哈希 + 逐记录 C14 重算 + 逐假设 revision 链校验
  （previous_status 衔接、窗口单调、previous_record_hash 合法、终态==链尾），
  任何篡改 → `SnapshotCorrupted(HASH_CHAIN_BROKEN)` fail-closed。
- 等价性（测试证明）：冻结点快照→恢复→续跑 与不间断运行 summary
  逐字节一致（normal 与 shuffled）；fresh subprocess 恢复哈希一致；
  REQUEST_CONTROL 停环恢复后保持停止、零新增调用。

## 10. Student 与训练接缝（C2）

- Student 身份固定 `PERSISTENT_RMT16_ORIGINAL_VTRACE_98304`；只消费 CC4
  共享 StudentAdapter（显式注入），缺失 → `STUDENT_ADAPTER_MISSING`
  fail-closed，不另建 loader/registry/codec。本 worktree 无 CC4 adapter →
  符号绑定 + `REAL_CHECKPOINT_LOADED=False`。
- `TRAINING_AUTHORIZED=False`：训练接缝只做 no-op 记账
  （TrainingStepRecord status=SKIPPED_UNAUTHORIZED），无 optimizer step。

## 11. 诚实性硬规则（修订记录）

1. 无 `feedback_id` 引用的修改只能标 `EXPLORATION`（`EXPLORATION_LABEL_REQUIRED`），
   且决策 ∈ EXPLORATION_DECISIONS；
2. 引用了反馈却标探索 = `MASQUERADE_FORBIDDEN`；
3. 记录级 label 由引用并集强制（`REVISION_LABEL_FORCED`）：空→EXPLORATION，
   非空→FEEDBACK_DRIVEN；
4. 记录级 ids ≠ 修改引用并集 → `FEEDBACK_ID_MISMATCH`；
5. 悬空引用 → `UNKNOWN_FEEDBACK_ID`（reconciler 与 revision 双重 fail-closed）；
6. RETIRE 必须引用反馈（`RETIRE_REQUIRES_FEEDBACK`）。

## 12. 授权姿态（本轮）

`TRAINING_AUTHORIZED = FORMAL_EVALUATION_AUTHORIZED = REAL_LLM_CALLS_AUTHORIZED
= REAL_SIMULATOR_PROBE_AUTHORIZED = False`；全部 REAL_* 能力旗标与
`SOTA_INTEGRATION_READY`、`SHARED_ANCHOR_MANIFEST_BOUND` 恒 False。
controller 构造时复验 `NEVER_TRUE_REAL_CAPABILITY_FLAGS`，任一为 True 即
`AUTHORIZATION_POSTURE_VIOLATED`。后端只允许 mock/replay
（`EXECUTION_MODE_MOCK_DRY_RUN`），真实后端适配器构造即 Blocked。
正式评估源进入环路任何组件即 raise（`formal_isolation`）。
