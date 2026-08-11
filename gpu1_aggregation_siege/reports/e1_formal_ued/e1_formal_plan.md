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
  `b5536d3` C13 fail-closed 训练门禁修复（总控 REQUEST_CHANGES）→
  `e623130` C14 REUSE 证据收紧（record_verified_batch 结构化 dual-probe
  绑定；总控 CC2/E1）→
  `2a6122f` C15 REUSE 认证绑定与全量复核（adapter 铸造 dual-probe
  attestation；调用方串单独永不充分；_snapshot_still_valid 每次复用前
  重验全部绑定；总控 CC2/E1）→
  `1f2598d`（C15-RC 铸造绑定内部 adapter registry + 不可变结果对象；
  总控 REQUEST_CHANGES：删除 mapping 铸造缝；假 adapter/直接铸造/
  未知结果/越域证据一律 fail-closed；1 条 adapter 签发正路径）→
  **round-3**（目标 PRODUCTION_PATH_READY_FOR_AUDIT；停止测试扩张，
  不新增任何测试文件）：
  `62eb560` C1 fix(e1) 顺序 board+12 槽真实可达+真实 InvocationGate
  （P0-1/2/3）→
  `9aeb489` C2 feat(e1) 真实 EnvCoder 执行+有限修复+统一 candidate
  probe+criterion-wise selector+共享接缝（P0-4/5）→
  `9931ecd` C3 feat(e1) 单更新真实门+长跑只准备入口（P0-6）→
  （本提交）C4 docs(e1) 生产就绪与阻断记录（readiness 由脚本从
  实际状态计算，布尔绝不手写）。

## 一、九阶段管线 → 代码位置

| # | 阶段 | 实现 | 本轮状态 |
|---|---|---|---|
| 1 | Student 行为失败证据 | `teachers/e1_formal/evidence.py` + `archive_view.py`（仅 TRAINING / NORMAL_TRAINING_FEEDBACK；FORMAL_* 构建期拒；tier 永不入 prompt） | 实现+测试 |
| 2 | 完整六角色 Review Board | `board.py`：固定顺序 student_modeler→behavior_auditor→causal_failure_analyst→intervention_tutor→explorer→critic；缺一即 INCOMPLETE_REVIEW_WINDOW→REUSE；无 2 角色/条件路径 | 实现+测试 |
| 3 | 因果假设+干预+Canonical TaskSpec | `task_specs.py`（spec_hash、window_hash 绑定、REGISTRY 校验；**round-3**：≤10 唯一模板/窗 `MAX_WINDOW_TEMPLATES=10`，6 模板×2 变体=12 specs，spec 池上限 20；`derive_variant_params` 确定性无 LLM） | 实现+测试 |
| 4 | 独立 LLM EnvCoder | `envcoder.py`（prompt 白名单、**round-3 模板键控：每唯一模板恰 1 次调用、K1 按唯一模板计**；有限修复环 `run_envcoder_with_repair`：F1=真实有界修复计数 ≤2/模板，耗尽⇒`ENVCODER_REPAIR_EXHAUSTED`；round-2"F1≡0"表述作废） | 实现+测试（replay；real 后端未授权恒 fail-closed） |
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
   **C14 收紧**：provenance 串单独永不认证 REUSE——另需
   artifact_id 与内部 registry 对账、结构化 dual-probe 块（pinned
   强 Student + Student/Reference probe id/sha256）、Reference 身份
   哈希、窗口哈希（与 registry 逐条相等）、候选集哈希；晋升路径
   （`build_training_batch` 携带 `dual_probe`）同构收紧。详见
   gate_closure_v2.md「C14 REUSE 证据收紧」。
   **C15 认证绑定**：dual-probe 串/哈希还须匹配 adapter 铸造的
   attestation（调用方串单独永不充分），且 `_snapshot_still_valid`
   在每次 REUSE 前重验全部绑定（registry 逐条、窗口/候选集/身份/
   manifest 哈希、attestation、门禁态）——窗口过期、身份/协议/
   manifest 变更、候选集换序、篡改存储快照、直接私有旁路一律
   失效⇒零训练。详见 gate_closure_v2.md「C15 REUSE 认证绑定与
   全量复核」。
   **C15-RC 收紧（REQUEST_CHANGES）**：首版 mapping 铸造缝
   （`record_dual_probe_attestation`）已删除——任何调用方不得自行
   铸造。铸造绑定内部 `eval_adapter.CandidateEvalAdapterRegistry`：
   adapter 经 fail-closed 注册（pinned 能力串），签发只在 registry
   内部（关键字标量、无 mapping 入参）且只认已注册 adapter；教师
   只消费 registry 签发的不可变 `DualProbeResult`（字段全合法的
   映射也拒），绑定 pinned Student、当前 Reference 候选/
   checkpoint/reset 协议、窗口与候选集作用域；假 adapter、直接
   构造/变形结果（未知结果）、越域证据一律 fail-closed。详见
   gate_closure_v2.md「C15-RC 修订」。

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
   （round-3 起 EnvCoder 验证面升级为分层后端：replay=SYNTAX+
   GUARDS+STRUCTURE，real 全阶段未授权恒 fail-closed，见 §六。）
3. replay store 为空：任何真实开窗尝试都会 HARD FAIL（设计如此，
   防止静默编造 LLM 应答）。

## 六、round-3：PRODUCTION_PATH_READY_FOR_AUDIT（本轮）

总控指令：目标=生产路径可审核；完成代码并推送，然后停止；审核通过
前不启动完整长跑；**停止测试扩张**（本轮不新增任何测试文件，现有
pin 仅在被有意改变行为处做等价或更强的最小更新）。

1. **P0-1 顺序协作**：`BoardContext`/`UpstreamOutput`；角色 k 的
   prompt/envelope 绑定 window 身份四元组 + pinned Student 身份 +
   前序解析成功角色的 `role_output_hash`（渲染+哈希双轨）；
   `BOARD_PROMPT_VERSION=e1-board-prompt-v2`。
2. **P0-2 12 槽真实可达**：`MAX_WINDOW_TEMPLATES=10`（只数唯一
   family 模板）；`TaskTemplate` + `derive_variant_params`（确定性、
   无 LLM）；6 模板×2 变体=12 specs；EnvCoder 每唯一模板恰 1 次
   调用（K1 按唯一模板）；**stub 补槽删除**——编译 artifact ≥12⇒
   全池，1–11⇒整窗拒绝（`INSUFFICIENT_DYNAMIC_ARTIFACTS`⇒REUSE
   批次；`_reuse_stub` 仅存为不可训练 REUSE 标记，永不入训练批次）。
3. **P0-3 真实 InvocationGate**：`gate_signals.py` 从真实
   TRAINING_WINDOW 证据计算八信号（含 prev_window_hash/
   threshold_version 绑定；无生产者⇒计算为 False+原因码）；
   `GateState.signals_binding_hash` 必填。
4. **P0-4 真实 EnvCoder + 有限修复**：`envcoder_backends.py` 八阶段；
   修复环 `run_envcoder_with_repair`（RepairRecord 哈希链；F1=真实
   有界修复计数，`teacher.envcoder.max_repairs` 硬上限 2；耗尽⇒
   `ENVCODER_REPAIR_EXHAUSTED`）；backend 键：replay（生产默认，
   SYNTAX+GUARDS+STRUCTURE）/mock（显式授权消融专用）/real（未授权
   恒 `ENVCODER_BACKEND_BLOCKED`，绝不静默降级）。
5. **P0-5 真实双 Probe + criterion-wise Selector**：
   `criterion_selector.py`（八准则准则内 pairwise Copeland + 权重
   聚合 + family_cap + 取 12 不足无回填；"先均值后 Copeland"结构上
   不存在）；`shared_runtime_seam.py` 八共享合同（只解析、不构造、
   不铸造、不伪装）；统一入口 `evaluate_candidate`（fail-closed
   参数校验⇒共享合同解析⇒未绑定即 BLOCKED_WAITING_SHARED_RUNTIME；
   绑定后真实路径留给未来轮次，绝不 stub）。
6. **P0-6 两个唯一入口**（`scripts/`）：
   - `run_e1_real_one_update.py`：唯一单更新真实门（real reset/step
     → real 六角色 → real candidate probe → criterion-wise 选 12 →
     12+4 batch → `run_session_training` 恰一次 optimizer update →
     checkpoint save/load 回验（仅共享 FullStateCheckpoint 合同）→
     NaN/Inf 全叶检查）。任一缺失⇒BLOCKED+exit≠0，
     `REAL_ONE_UPDATE_EXECUTED=false`。
   - `run_e1_longrun.py`：只准备、不启动（冻结清单：
     total_env_steps=98304、Student/Reference 身份、seed、anchor
     manifest、实时 git SHA、config/checkpoint hash、输出目录；
     任一未冻结⇒拒绝）。
   - 共用 `e1_production_runtime.py`：资产/门禁解析 + 诚实 JSON
     汇报；生产入口不 import tests、不依赖 fixture、不默认
     mock/replay；共享资产缺失⇒明确阻断码退出。
7. **就绪态记录**：`scripts/e1_formal_readiness.py` 从实际状态计算
   `reports/e1_formal_ued/real_smoke_readiness.json`；阻断清单见
   `reports/e1_formal_ued/current_blockers.md`。

**本轮实跑（审计 venv，JAX_PLATFORMS=cpu）**：单更新入口 BLOCKED
（12 条阻断码，exit 2）；长跑入口 REFUSED（exit 2）；套件
993 passed / 5 skipped / 0 failed（基线保持，零回归）。
三个 REAL_* 标志保持 false；审核通过前不启动长跑。
