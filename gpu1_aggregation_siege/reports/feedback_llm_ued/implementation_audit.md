# SIMULATOR-GROUNDED FEEDBACK-ADAPTIVE LLM-UED — 实现审计（implementation_audit）

- 分支：`henry/ba-bagr-ued-review-board-v2`
- 基线 HEAD：`8cdd88d8d43969762a172013a1643e9974004d1a`（提交前工作树仅含本方向新增文件）
- 代码根：`gpu1_aggregation_siege/d052/feedback_llm_ued/`（19 个模块，3638 行）
- 测试：`gpu1_aggregation_siege/d052/tests/test_feedback_llm_ued_*.py`（6 个文件，109 用例，1423 行）
- 本文件只陈述**已验证**的事实；未运行/未实现内容一律显式标注。

## 1. 仓库状态审计（开始时）

| 项 | 状态 |
|---|---|
| 当前分支 | `henry/ba-bagr-ued-review-board-v2`（未切换、未覆盖 CC1/CC2 分支） |
| 工作树 | 干净，仅 `d052/feedback_llm_ued/` 为新增未跟踪目录 |
| 先前资产 | `d052/bagr_ued/`（六角色板 v2 干跑，554 测试基线绿）可复用其 `hashing.canonical_sha256/text_sha256` |
| Mason 分支 | **未合并**。仅选择性复用其 JAX-free 纯核心（见 §3.2） |
| 授权常量 | `TRAINING_AUTHORIZED / FORMAL_EVALUATION_AUTHORIZED / REAL_LLM_CALLS_AUTHORIZED / REAL_SIMULATOR_PROBE_AUTHORIZED` 全部 `False` |
| 本地 JAX/Craftax | 不存在（`/d/Anaconda/python` 3.12.4 无法 import）→ `REAL_SIMULATOR_PROBE_STATUS="BLOCKED_NO_LOCAL_CRAFTAX"` |

## 2. 模块清单（全部已实现，全部有测试覆盖）

| 模块 | 任务条款 | 行数 | 实现状态 |
|---|---|---|---|
| `constants.py` | 全局常量+授权姿态 | 276 | 完成（4 个授权旗标硬编码 False，controller 启动时复验） |
| `feedback_contracts.py` | §2 合同 | 229 | 完成（CandidateEnvironment/ProbeMetrics/FamilyAllocation/CurriculumPlan/FeedbackRoleEnvelope，全部 CanonicalModel + 内容哈希） |
| `hypothesis_ledger.py` | §2.1 HypothesisLedger | 139 | 完成（PENDING→SUPPORTED/REFUTED/INCONCLUSIVE/STALE，修订历史哈希链） |
| `simulator_feedback_store.py` | §2.2 SimulatorFeedbackStore | 193 | 完成（reference_stats 白名单、formal 源拒绝、bind_match 重哈希） |
| `plan_revision.py` | §2.3 PlanRevisionRecord | 140 | 完成（无引用只能 EXPLORATION 的硬校验、记录级 label 强制、引用并集一致性） |
| `expected_observed.py` | §2.4 ExpectedObservedComparator | 151 | 完成（逐指标 agree/neutral/opposite，总体 MAJORITY，grade_record 绑定） |
| `feedback_diagnostician.py` | §2.5/§3 角色1 | 173 | 完成（mock_rule 从反馈绑定派生 verdict；真实 LLM 未授权） |
| `adaptive_designer.py` | §2.6/§3 角色2 | 168 | 完成（verdict→RETAIN/RETIRE/MUTATE，探索限额，HIGH 风险→REQUEST_CONTROL） |
| `adversarial_reviewer.py` | §3 条件第 3 调用 | 165 | 完成（7 条风险触发器判定 + 对抗审查 mock_rule） |
| `feedback_invocation_gate.py` | §3 调用门 | 93 | 完成（8 条 must-invoke 条件，全不触发则复用计划） |
| `deterministic_reconciler.py` | §2.8 DeterministicReconciler | 299 | 完成（LLM 提议、规则收口：悬空引用/伪装/预算/上限/锚位全部 fail-closed 并留日志） |
| `simulator_probe.py` | §4 探针漏斗 | 378 | 完成（符号 runner + Craftax 真实接缝 BLOCKED + 64→24→12+4 漏斗） |
| `environment_generator.py` | §4 候选展开 | 103 | 完成（计划→恰好 64 个候选，最大余数分配，假设区分绑定） |
| `skill_preflight_core.py` | §4 Mason 纯核心复用 | 208 | 完成（route/PreflightResult/prereq，JAX-free，来源注释，非分支合并） |
| `formal_isolation.py` | §6 隔离 | 99 | 完成（FormalSourceIsolationGuard + ReferenceOutputGuard，raise 不吞） |
| `llm_backend.py` | §3 mock 后端 | 73 | 完成（real_calls=0，UNKNOWN_ROLE 拒绝，assert_no_real_calls） |
| `controller.py` | §1/§5 闭环 | 645 | 完成（三模式 + 对比指标 + 授权姿态复验） |
| `synthetic_feedback.py` | 测试脚手架 | 73 | 完成（SYNTHETIC_TEST_TRACE 源，仅供单测） |
| `__init__.py` | 版本 | 33 | 完成（`FEEDBACK_LLM_UED_VERSION="d052.feedback_llm_ued.v1"`） |

## 3. 关键审计结论

### 3.1 与 CC2 静态 LLM-UED 的区别（§1 要求）
- CC2：LLM 生成→接受/拒绝，不运行探针、不读真实结果。
- 本方向：每个触发窗口**真实运行**两级探针（64 个快速 + 24 个完整探针），
  观测指标与假设预测签名逐指标比对，比对方向绑定到反馈记录并进入
  Diagnostician 的上下文；Designer 的 verdict→决策映射改变下一轮计划。
  `test_feedback_llm_ued_controller.py` 证明：打乱候选↔反馈绑定后计划改变
  （6 窗中 5 窗计划不同，`feedback_binding_matters=True`），
  static 基线 0 次 LLM 调用、计划恒定。

### 3.2 与 CC1 的区别（§1 要求）
- 无 EnvState restore、无 frontier archive、无 state 中途分支；
- 候选是**标准重置**的环境级 TaskParams（mock 命名空间字段白名单，
  真实 TaskParams 适配器 BLOCKED_EXTERNAL_DEPENDENCY，字段名不猜测）；
- LLM = 课程领导者（Diagnostician+Designer），simulator = 验证者/反馈源。
- Mason 复用仅限 JAX-free 纯核心（`route`/`PreflightResult`/prereq ready），
  以来源注释引入，**不是分支合并**；真实 Craftax 探针集中在唯一接缝
  `CraftaxPreflightProbeRunner`，本地构造即 `ProbeRunnerBlocked` fail-closed。

### 3.3 诚实性不变量（§2 要求，全部为硬校验器）
- 无 `feedback_id` 引用的方案只能标 `EXPLORATION`（`EXPLORATION_LABEL_REQUIRED`）；
- 探索决策只能是 `MUTATE/EXPAND_BUDGET`（`EXPLORATION_DECISION_ONLY`）；
- 引用了反馈却标探索 = `MASQUERADE_FORBIDDEN`；
- 记录级 label 由引用并集强制（`REVISION_LABEL_FORCED`），
  记录级 ids 必须等于各修改引用并集（`FEEDBACK_ID_MISMATCH`）；
- `RETIRE` 必须引用反馈（`RETIRE_REQUIRES_FEEDBACK`）；
- 悬空引用 → `UNKNOWN_FEEDBACK_ID`（reconciler + revision 双重检查）。

### 3.4 正式评估隔离（§6）
- 独立 source 枚举：`GENERATIVE_TRAINING_ENV / CANDIDATE_PROBE /
  SYNTHETIC_TEST_TRACE` 为合法环路源；`FORMAL_FRONT/BACK/FULL` 在任何
  ProbeMetrics、SimulatorFeedbackRecord 构造时即 raise（fail-closed，测试覆盖）。
- Reference 输出仅允许 5 个 episode 级粗统计字段；8 类动作指导载体
  （action_sequence/trajectory/waypoints/hidden_state/logits 等）在
  runner 边界、store 构造、provenance 三处扫描，发现即 raise。

## 4. 验证矩阵（证据）

| 主张 | 证据 |
|---|---|
| 109 个新测试全绿 | `test_report.md` §2 命令与输出 |
| 全量 d052 套件 663 通过 / 6 既有环境性失败 | `test_report.md` §3 |
| 闭环数值（三模式 6 窗） | `final_implementation_report.md` §5（smoke 真实输出） |
| 示例 JSON 非伪造 | 三个 example JSON 由真实 controller 运行直接导出（生成脚本不入仓库） |
| 确定性 | 两次独立 normal 运行 summary JSON 逐字节一致（测试 `TestDeterminism`） |

## 5. 已知限制（如实）

1. **探针是符号的**：本地无 JAX/Craftax，`DeterministicSymbolicProbeRunner`
   由候选哈希确定性推导指标（`real_simulator=False` 全程携带）。
   真实 Craftax 接缝存在但 BLOCKED，需在训练主机上由 director 决策后启用。
2. **LLM 是确定性 mock**：`real_calls=0`，规则从 prompt 内嵌上下文派生；
   接入真实后端只需替换 `LLMBackend` Protocol 实现。
3. **无训练步**：`TRAINING_AUTHORIZED=False`，本轮无 optimizer step、
   无 checkpoint、无正式评估。
