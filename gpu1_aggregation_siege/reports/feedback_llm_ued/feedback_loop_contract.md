# 反馈闭环合同（feedback_loop_contract）

本文件是 `d052/feedback_llm_ued` 的规范性合同：数据怎么流、谁可以写什么、
什么时候允许调用 LLM、以及每一处 fail-closed 的触发条件。实现与测试必须
与本文件一致；不一致以实现为准修改本文件并在审计中说明。

## 1. 单窗口闭环（§1）

```
plan_k
  └─> EnvironmentCandidateGenerator.generate_candidates        （恰好 64 个）
        └─> run_staged_funnel（真实探针预算，见 §4）
              ├─ L1 静态合法性 + 哈希去重
              ├─ L2 快速探针 64 个（Student 2ep + Reference 1ep）→ route → 留 24
              └─ L3 完整探针 24 个（Student 8ep + Reference 4ep）→ 组合分 → 选 12
                    + 4 个全局锚位 = 最终 batch 16
  └─> 每候选 1 条 SimulatorFeedbackRecord（含 expected_signature）
        └─> ExpectedObservedComparator.grade_record
              观测 vs 预测 → agree / opposite / neutral（写回 store，重哈希）
  └─> FeedbackInvocationGate.evaluate_gate（8 条 must-invoke）
        ├─ 无触发：复用 plan_k 与其诊断（REUSED，0 次 LLM，不跑探针）
        └─ 有触发：
              ├─ FeedbackDiagnostician（LLM 调用 1）→ 逐假设 verdict + 置信度
              ├─ HypothesisLedger.apply_verdict（唯一状态写入者，哈希链）
              ├─ AdaptiveEnvironmentDesigner（LLM 调用 2）→ 族级 slot 修改
              ├─ 7 条风险触发器判定 → 至多 1 次 AdversarialReviewer（LLM 调用 3，条件性）
              └─ DeterministicReconciler.reconcile → plan_{k+1} + PlanRevisionRecord
```

窗口 0 是确定性 bootstrap（种子假设 + 初始计划 + 探针），不调用 LLM。

## 2. 哈希绑定与唯一写入者

| 数据 | 唯一写入者 | 绑定方式 |
|---|---|---|
| 假设状态/置信度 | `HypothesisLedger.apply_verdict` | 每次写入追加 revision_history（含 `previous_record_hash`）并重算 `record_hash` |
| expected-vs-observed 结论 | `SimulatorFeedbackStore.bind_match` | 重算 `record_hash`；`ungraded` 是合法初始态 |
| plan_k→plan_{k+1} 审计 | `PlanRevisionRecord` | 记录级 `based_on_feedback_ids` == 各修改引用并集，label 由并集强制 |
| LLM 请求/响应 | `FeedbackRoleEnvelope.make` | `request_hash=sha256(role,prompt_version,prompt)`、`response_hash=sha256(raw)` |
| 计划指纹 | `plan_signature_hash` | 对 allocations/retired/explored 排序后哈希（与顺序无关） |

哈希一律复用 `d052.bagr_ued.hashing.canonical_sha256/text_sha256`（单一事实源）。

## 3. LLM 预算（§3 精简）

- 默认每触发窗口 **2 次**：Diagnostician + Designer。
- 条件第 **3** 次 AdversarialReviewer，当且仅当 ≥1 条触发器命中：
  1. `overall_confidence < 0.55`
  2. 同族冲突干预（RETAIN×RETIRE 或 EXPAND_BUDGET×RETIRE）
  3. 诊断 `global_risk == HIGH`
  4. 连续 2 窗无改进
  5. 上一窗存在 opposite 探针（观测与预测相反）
  6. 候选拒绝率 > 30%
  7. 准备正式 run 之前
- 无任何 must-invoke 门条件命中 → **0 次** LLM，复用先前诊断与计划，
  生成器继续在既有干预邻域内展开。
- static_llm 模式恒 0 次（基线永不读反馈）。

## 4. 探针预算（§4）

- 漏斗：64 raw → L1 → 64 fast-probed → ~24 survivors → 24 full-probed →
  12 dynamic + 4 frozen anchors = 16。
- Episode 预算硬校验：fast 最多 Student 2 / Reference 1；full Student ∈ [4,8]、
  Reference ∈ [2,4]；超预算 raise（不静默截断）。
- 每窗真实模拟转移 = 64×(2+1)×128 + 24×(8+4)×128 = **61440 transitions**。
- Reference 只能给出 episode 级粗统计（`REFERENCE_ALLOWED_FIELDS` 5 字段）；
  动作序列/轨迹/路点/隐状态/logits 等 8 类载体在 runner 边界、store 构造、
  provenance 三处被 `ReferenceOutputGuard` 扫描，命中即 raise。

## 5. 三模式语义（§5）

| 模式 | 初始计划 | 读反馈 | 绑定 |
|---|---|---|---|
| `static_llm` | bootstrap | 否（永不） | 正常 |
| `normal_feedback` | bootstrap | 是 | 正常：候选↔观测一一对应 |
| `shuffled_feedback` | bootstrap | 是 | 打乱：观测载荷按候选 id 排序后循环位移 n//2；候选身份、假设绑定、expected_signature **不变** |

三模式共享同一确定性 bootstrap 与假设种子 → 计划差异只能归因于反馈使用。
必须报告：revision_rate、retain/mutate/retire 分布、feedback citation coverage、
supported-hypothesis retention、refuted-hypothesis retirement、
每有用环境的模拟转移数、normal vs shuffled 计划差异。

## 6. 诚实性硬规则（§2）

1. 无 `feedback_id` 引用的修改只能标 `EXPLORATION`，且决策 ∈ {MUTATE, EXPAND_BUDGET}。
2. 引用了反馈却标探索 = `MASQUERADE_FORBIDDEN`。
3. `RETIRE` 必须引用反馈（退休是判决，不是探索）。
4. 记录级 label 由引用并集强制；并集为空 → `EXPLORATION`，非空 → `FEEDBACK_DRIVEN`。
5. 记录级 ids ≠ 修改引用并集 → `FEEDBACK_ID_MISMATCH`。
6. 引用不存在于 store 的 feedback_id → `UNKNOWN_FEEDBACK_ID`（reconciler 与
   revision 双重 fail-closed）。

## 7. Reconciler 收口规则（按应用顺序）

1. 提案解析为 `FamilyAllocation`（畸形 = 硬错误）；
2. 悬空反馈引用 → `UNKNOWN_FEEDBACK_ID`；
3. 诚实性重标（见 §6）；
4. `REQUEST_CONTROL` → 记录、零预算、上报人类；
5. `RETIRE` 移出动态预算；同族既退休又重新提议 → 退休优先；
6. 同族重复：按 (决策优先级, 顺序) 取先；
7. 干预上限 `MAX_INTERVENTIONS=8`；
8. 探索上限 `MAX_EXPLORATION_PROPOSALS=2` 个、合计 ≤ `EXPLORATION_SLOT_CAP=2` slots；
9. 12 个动态 slot 填充：探索先预留（不被核心需求饿死），核心按
   RETAIN > EXPAND_BUDGET > MUTATE > REDUCE_BUDGET 贪心，余量补给最高优先级
   核心分配；结果为空 → `INSUFFICIENT_DYNAMIC_ALLOCATION`；
10. 4 个全局锚位恒预留（各 1 slot，动态预算之外）。

## 8. 正式评估隔离（§6）

- 环路合法 source：`GENERATIVE_TRAINING_ENV / CANDIDATE_PROBE /
  SYNTHETIC_TEST_TRACE`；正式域 `FORMAL_FRONT/BACK/FULL` 在任何环路组件
  （ledger、LLM 角色、generator、selector、optimizer）出现即 raise。
- `FORMAL_EVALUATION_AUTHORIZED=False` 本轮恒定；正式 run 不属于本闭环。

## 9. 授权姿态（本轮）

`TRAINING_AUTHORIZED = FORMAL_EVALUATION_AUTHORIZED = REAL_LLM_CALLS_AUTHORIZED
= REAL_SIMULATOR_PROBE_AUTHORIZED = False`。controller 构造时复验；任一为 True
即 `AUTHORIZATION_POSTURE_VIOLATED`。真实 Craftax 接缝
`CraftaxPreflightProbeRunner` 在旗标为 False 或 jax/craftax 不可 import 时
构造即 `ProbeRunnerBlocked`。
