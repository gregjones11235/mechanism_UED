# E1 Formal — Behavior-Aware Regret-Guided LLM-UED：工程实施记录

> **INDEPENDENT_AUDIT_REQUIRED = true**
>
> 本文档仅证明**工程计划对齐**（`E1_FORMAL_PLAN_ALIGNED` 的含义），
> 不证明真实闭环。真实闭环所需的冻结件（Reference 身份、anchor
> manifest 冻结、真实 probe）未到位前，相应标志保持 false/BLOCKED。

- 分支：`henry/static-llm-ued-v1`（worktree `mechanism_UED_static_llm_ued_worktree`）
- 基线：`Henry-branch @ 9eca2de914068a33e500e2ad90d50f48e6e4e632`
- 提交链（全部路径限定 + Co-Authored-By trailer，未 push）：
  `edf10cb` schemas → `7c6bc88` C1 guards → `100488e` C2 StudentInitContract →
  `248285e` C3 核心+G1 → `3e18aa9` C4 evidence → `4ae56fa` C5 replay/manifest/accounting →
  `7f56341` C6 board → `f254a61` C7 TaskSpec → `0578559` C8 EnvCoder →
  `4f3e333` C9 metrics/anchors/selector/parity → `8e65663` C10 GenManager+配置 →
  `1fa41ab` C11 集成布线+评价 seam → `0c8f2af` C12 报告 →
  （本提交）C13 fail-closed 训练门禁修复（总控 REQUEST_CHANGES）。

## 一、九阶段管线 → 代码位置

| # | 阶段 | 实现 | 本轮状态 |
|---|---|---|---|
| 1 | Student 行为失败证据 | `teachers/e1_formal/evidence.py` + `archive_view.py`（仅 TRAINING / NORMAL_TRAINING_FEEDBACK；FORMAL_* 构建期拒；tier 永不入 prompt） | 实现+测试 |
| 2 | 完整六角色 Review Board | `board.py`：固定顺序 student_modeler→behavior_auditor→causal_failure_analyst→intervention_tutor→explorer→critic；缺一即 INCOMPLETE_REVIEW_WINDOW→REUSE；无 2 角色/条件路径 | 实现+测试 |
| 3 | 因果假设+干预+Canonical TaskSpec | `task_specs.py`（spec_hash、window_hash 绑定、REGISTRY 校验、≤10/窗） | 实现+测试 |
| 4 | 独立 LLM EnvCoder | `envcoder.py`（prompt 白名单、变体轮换、按唯一 artifact 计 K1、无 repair：F1≡0） | 实现+测试（replay） |
| 5 | 编译门禁 | `gen_manager._E1EnvGenerator.check_compilation`（guards + stdlib syntax；**绝不回灌 LLM**） | 复用+测试 |
| 6 | Student/Reference 真实评价 | `evaluation/candidate_evaluation.py`（G1 门优先；本轮诚实阻断） | seam 实现+测试 |
| 7 | Regret/Gap/Learnability/Retention | `metrics.py`（G2 三态+Wilson CI；LP 仅先验字段；retention 仅 G3 冻结后可用） | 公式+fixture 测试；真实证据诚实缺省 |
| 8 | 确定性 Soft Copeland | `selector.py`（自包含 stdlib 复刻；pin canonical_v2 + 三个源码 SHA；retention 停用无替代） | 实现+parity 门禁 |
| 9 | 12 dynamic + 4 anchors → 训练 | `layout.py`（β=1/4、s=2/5 精确有理；original 恒最后）+ `gen_manager.build_training_batch` | 实现+测试；本轮不真实训练 |

## 二、降级链（D5，逐级诚实，任何一级都不伪造）

```
REFERENCE_CONTRACT_UNFROZEN
  => EVAL_SEAM_SKIPPED_NO_STUDENT_ADAPTER
  => LEARNABILITY_UNAVAILABLE
  => SELECTION_BLOCKED_NO_REAL_EVIDENCE
  => batch 零可训练任务（training_permitted=False，task_ids=[]）
  => 训练门禁拒绝 run_session_training：零 PPO 更新、零 step 前进
```

**C13 修正（总控 REQUEST_CHANGES）**：C11/C12 版本在阻断时返回
"4 anchors + reuse_only" 且 run_dicode 仍会调 run_session_training——
那是 anchors-only 偷跑路径，已删除。现在：

1. 阻断 batch 的 `task_ids` 为空、`training_permitted=False`；
2. `run_dicode.py` 的 E1 钩子分支先过
   `training_gate.enforce_training_gate`（严格 `is True`），未许可即
   抛 `RuntimeError` 显式中止——零更新、零 step、绝不回退 legacy 采样；
3. 门禁模块独立复核：许可 batch 必须恰为 12 dynamic + 4 frozen
   anchors（canonical 顺序）+ 覆盖 16 任务的 pinned layout；anchors-only、
   乱序、重复、legacy 分布一律拒绝；
4. REUSE 仅当存在上一窗口**完整已验证**快照
   （`record_verified_batch`：G1 冻结+G3 冻结+阈值冻结+
   provenance=CANDIDATE_EVALUATION+12 个动态任务逐条绑定本教师
   registry 的 spec_hash/code_sha256+manifest sha 相等），否则
   `TRAINING_BLOCKED_NO_VERIFIED_BATCH`/相应阻断码。

集成 smoke（`tests/e1_formal/test_integration_smoke.py`）断言链上每个码
**如实出现**，且 batch 中没有任何伪造动态任务或伪造数值；
`tests/e1_formal/test_training_gate.py` 提供完整正负矩阵
（DRAFT manifest/缺 dual-probe/空/伪快照 ⇒ 零训练；合法 12+4 ⇒ 训练一次）。

## 三、集成布线（C11；默认路径字节不变义务）

| 位置 | 钩子 | 守护方式 |
|---|---|---|
| `setup.py::_resolve_teacher` | 教师注入 | 无 teacher 组 ⇒ 原 `GenManager(config)` 逐字；e1_formal 惰性导入；static_llm/未知 ⇒ NotImplementedError |
| `training.py::_resolve_session_task_distribution` | 12+4 pinned 布局 | `build_training_layout` getattr 鸭钩；仅在覆盖会话且和恰为 1 时采用，绝不重归一化；legacy 函数一字未动 |
| `evolution_efficient.py::dispatch_evolution_worker` | `select_context_tasks` | getattr 鸭钩；E1 本轮答 [] ⇒ 不派发（诚实：无可采纳上下文任务） |
| `run_dicode.py` | `consume_worker_results` / `build_training_batch` / `observe_session_feedback` | getattr 鸭钩；legacy 键与采样逐字保留；feedback 仅在真实训练指标非空时回喂；**C13：batch 钩子分支先过 `enforce_training_gate`，未许可⇒ RuntimeError，run_session_training 绝不执行** |
| `evaluation/__init__.py` | +1 行导出 seam | 原导出行不动 |

字节不变证据：AST 守护断言（`test_wiring_sources.py`）+ 纯 python mirror
对 legacy 公式 n=0..32 逐位相等 + jnp 公式 float32 舍入内相等
（`test_distribution_byte_identity.py`）+ 全环境 importorskip 运行时等价
（真实 `_calculate_task_distribution` vs mirror；无钩子 fake-GenManager 等价）。

## 四、本轮明确不做（硬停止清单摘要）

真实 Student 训练/真实 Reference 评价/真实 LLM/付费 API/第二套
loader/registry/checkpoint 加载/GPU 占用/任何 push、merge、rebase、
reset、clean；正式评测数据永不进入教师/选择器/archive 优先级。

## 五、已知诚实限制（记录在案）

1. E1 教师本轮**不能**端到端运行 legacy 训练循环：seed 训练需要
   networkx 图/seed 任务，真实会话需要 craftax/checkpoint/CC4
   adapter —— 均不具备。布线为未来轮次准备；与
   `REAL_TRAINING_UPDATE_EXECUTED=false` 一致。
2. `check_compilation` 本轮仅 stdlib 语法 + guards（craftax 不在审计
   venv）；import/reset/step 语义未验证 —— `status_report` 明示。
3. replay store 为空：任何真实开窗尝试都会 HARD FAIL（设计如此，
   防止静默编造 LLM 应答）。
