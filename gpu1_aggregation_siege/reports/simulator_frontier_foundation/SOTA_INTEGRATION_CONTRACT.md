# SOTA Integration Contract · 方向三 Simulator-Centric Frontier-UED（R1–R9 完整路线）

> 权威来源：总控完整路线 R1–R9 + 独立审核 PASS_WITH_CONDITIONS 五条件（全部有效）。
> 本文是接口/契约文档：本轮零真实 API、零 actual-N 长作业、零训练更新；契约测试 ≠ 真实闭环，forward smoke ≠ 性能评估。

## 1. 完整路线（含 R4a/b/c 拆分）

| 段 | 内容 | 本轮状态 |
|---|---|---|
| R1 | 高能力 Student 标准 reset rollout（`TRAINING_DISCOVERY` provenance；正式评估协议/bank/worlds 结构性隔离） | 契约就位（discovery_provenance，已加固为注册表绑定），未运行 |
| R2 | 关键状态捕获（确定性捕获准则 + CaptureProvenance 校验） | 契约就位，未运行 |
| R3 | Frontier Archive 保存 EnvState + RNG + wrapper state + memory/history（StateBundle 全字段 + Student 身份绑定 + discovery_provenance） | StateBundle/Archive 加性字段就位 |
| R4a | env 侧 restore validation（逐叶全等 + dynamics parity） | **PASS（Stage 1）**：`reports/simulator_frontier_cc1/phase1/` |
| R4b | checkpoint 侧 restore validation（params/身份门禁 + fresh-process 复算） | **PASS（Stage 3）**：`student_compatibility_PERSISTENT_RMT16_ORIGINAL_VTRACE_98304.json` |
| R4c | 联合 fresh-process restore 门禁：同一 fresh process 联合恢复 params+optimizer+step+RNG / EnvState / env RNG / wrapper state / memory/history 并交叉核验 | **契约就位，未执行** → `COMBINED_FRESH_PROCESS_RESTORE=false` |
| R5 | 真实 actual-N 多分支搜索（以联合还原态为起点；N_actual 实测；禁 best-of-N 外推） | 未执行（`ACTUAL_N_READY=false`） |
| R6 | Feasibility Statistics（success rate(N_actual)/progress/Wilson CI/cost/failure category） | foundation 接口在位，未实测 |
| R7 | InvocationGate：0 或完整 2 LLM → 确定性 Guard/Reconciler/Selector | 契约 + fake-client 测试就位（`invocation_gate.py`） |
| R8 | 12 dynamic frontier distributions + 4 standard-reset anchors | schema/绑定就位；科学内容待总控 manifest |
| R9 | Mixed-start PPO/V-trace → 同一高能力 Student 更新 → 下一窗口回 R1 | 接口文档级，未执行 |

## 2. 审核五条件并入（逐条强制）

1. **R4 联合 fresh-process 门禁**：`combined_restore_contract.py`——9 组件（params/optimizer/global_step/train_rng/env_state/env_rng/wrapper_state/policy_memory/history）+ 交叉核验（policy_step_next_replay）；**env-only PASS ∧ ckpt-only PASS ≠ 联合证明**，`evaluate_verdict` 机械强制该区分；本轮 `COMBINED_FRESH_PROCESS_RESTORE=false`。
2. **TRAINING_DISCOVERY 隔离**：`discovery_provenance.py` + `CaptureProvenance` fail-closed 校验；FORMAL_FRONT/BACK/FULL bank 标识与 formal world 标识进入采集请求即 raise；与 `FormalDataLeakageGuard` 消费者集合联动。**2026-08-04 按总控 PASS_WITH_BLOCKER 加固**：隔离改为两层——① 所有 discovery 输入（bank ref / world set id / world set hash）必须解析到显式 `DiscoveryProvenanceRegistry` 的 allowlist 记录（中性别名/未注册串一律 fail-closed）；② 总控注入的冻结正式资产身份集（canonical id + sha256）对全部文本字段做大小写不敏感清扫（含嵌套 notes）。缺 registry / registry 无效 → raise，绝不猜测。**2026-08-04 CC4/E3 生产入口强制**：生产采集路径机械地只认总控注入的 registry——`validate_capture_provenance_production` 只读取 `inject_frozen_formal_asset_registry` 注入槽（单次注入、永不接受合成 fixture）；registry 带 `usage` 分级（`TEST_ONLY` fail-closed 默认 / `PRODUCTION`）且 usage 参与 registry hash；TEST_ONLY/合成注册表永不能进入生产槽，绕过注入槽直接使用的 PRODUCTION registry 被 `validate_capture_provenance` 拒绝；未注入时生产入口 fail-closed 于 `BLOCKED_WAITING_FROZEN_FORMAL_ASSET_REGISTRY`。**诚实状态**：真实冻结正式资产身份集本轮未获总控注入 → `DISCOVERY_FORMAL_PROVENANCE_ISOLATED=false`，只可声称 `DISCOVERY_PROVENANCE_CONTRACT_READY=true` + 状态 `BLOCKED_WAITING_FROZEN_FORMAL_ASSET_REGISTRY`；测试中的注册表均为合成 fixture，非真实隔离证明。
3. **TWO_LLM_GATE 语义限定**：`TWO_LLM_GATE_CONTRACT_ONLY=true` **仅** = CONTRACT_AND_FAKE_CLIENT_TEST_READY；`REAL_TWO_LLM_CALL_EXECUTED=false`（本轮恒 false）；所有测试命名带 `fake_client`/`contract`。
4. **FOUNDATION_READY 表述边界**：只表示 A–D 基础门禁（见 `sota_launch_gate.json`）；蕴含规则钉死：`ACTUAL_N_READY=false ∨ REAL_TRAINING_UPDATE_EXECUTED=false ⇒ SOTA_INTEGRATION_READY=false`；`FOUNDATION_SCOPE_HONEST` 自检表述合规。
5. **共享 anchor manifest**：四锚点为三方向共享冻结 manifest；CC4 只提供 schema/绑定接口（`anchor_manifest.py`），绝不自拟科学内容；未获总控 manifest → `SHARED_ANCHOR_MANIFEST_BOUND=false` + `BLOCKED_SHARED_ANCHOR_MANIFEST`。

## 3. 首轮冻结面（不可被 Student 重写）

- `FrontierArchive` / `archive_schema`：容量、分桶配额、去重、hash 校验语义冻结；只允许尾部加性字段。
- `GoalSpec`（goals.py 五类）、`TerminalEventAdapter`（terminal_events.py）、Branch Statistics（search_statistics.py：只认 N_actual）、Curriculum Selector（确定性 selector）：**不得按任何 Student 重写**。
- `provenance.py` 双守卫消费者集合冻结：{"FrontierArchive","BranchOutcome","FeasibilityEstimate","curriculum","Student optimizer"}。
- StudentAdapter 协议方法集冻结（§十八 11 方法）；新 Student 只能新增 adapter 实现，不能改协议。

## 4. RetentionContract（R8）

- **12 dynamic frontier distributions + 4 standard-reset anchors**；`anchor_ratio > 0` 强制（绑定接口 fail-closed 校验）。
- anchors 科学内容（worlds/分布/seeds）**只能来自总控签名的共享冻结 manifest**；`validate_anchor_manifest` 校验：缺签名引用/未冻结/槽位≠4/id 重复/非 STANDARD_RESET/hash 失配 → raise。
- **能力回退触发 gate**：anchor 回归超阈值即触发 gate，阻止课程继续漂移（接口在 RetentionContract，阈值由总控 manifest 给定）。
- **正式银行永不入在线课程**：`formal_banks_in_online_curriculum=True` → `ProvenanceViolationError`。

## 5. LLM 调用口径（覆盖旧「固定 0 / 可选最多 1」）

**官方规则：0 或恰好 2。**

- 无显著变化 → **严格 0 次**，复用上一份计划（必须携带 `reuse_plan_ref`）。
- 需要修订 → **必须完整调用两个 LLM**，顺序钉死：`frontier_evidence_diagnostician` → `curriculum_search_planner`；planner 输入只追加 diagnostician 聚合摘要。
- 任何 attempts==1 状态直接 raise（`assert_never_exactly_one_call` 双重守卫）。
- 两个 LLM 只读**聚合证据**（Feasibility Statistics + archive 摘要）；禁止成功动作/路线/waypoint/logits/hidden states/专家轨迹进入证据或 LLM 输出（扩展禁词表 + 输出复检）。
- **选择权归确定性 selector**：LLM 输出只是候选；`deterministic_select` 按 priority_score 选择（同分取最小 plan_id），夹带禁词字段的候选被一票否决（FORBIDDEN_ACTION_GUIDANCE_FIELD）。
- 真实 API 启用需总控显式授权；本轮全部 fake-client 测试 → `REAL_TWO_LLM_CALL_EXECUTED=false`。

## 6. TRAINING_DISCOVERY 隔离条款

- Frontier 采集用 standard-reset rollout 一律携带 `DiscoveryProvenance.TRAINING_DISCOVERY`（唯一合法采集来源枚举）。
- 与冻结 formal evaluation bank/worlds **结构性隔离**：正式数据不得进入 Archive/Gate/LLM/selector（`FormalDataLeakageGuard` 在 frontier/curriculum/optimizer 消费点强制）。
- Archive entry 携带 `discovery_provenance` 加性字段；空值绑定即 raise。
- 隔离验证采用**注册表绑定 + 冻结身份清扫**两层（见第 2 节条件 2 加固说明）：bank/world 引用必须命中 discovery allowlist；forbidden formal identity（canonical id + SHA，大小写不敏感、含嵌套文本）命中即 raise。
- 本轮无真实采集运行，且**真实冻结正式资产身份集未获总控注入** → 只可声称 `DISCOVERY_PROVENANCE_CONTRACT_READY=true`（契约 + 绕过封闭负例测试级）；`DISCOVERY_FORMAL_PROVENANCE_ISOLATED=false`，状态 `BLOCKED_WAITING_FROZEN_FORMAL_ASSET_REGISTRY`；待总控注入真实 registry 并绑定后方可升级（生产入口强制机制见第 2 节条件 2 的 CC4/E3 说明）。

## 7. 本轮边界声明

- 零付费 API、零 actual-N 长作业、零训练更新（总控常设约束）。
- 只改 `mechanism_UED_sim_foundation` worktree；路径限定提交、不 push。
- `FOUNDATION_READY=true` **仅指 A–D 基础门禁**，不得表述为「方向三可训练」；`SOTA_INTEGRATION_READY=false`（蕴含规则下本轮恒 false）。
- 门禁值与证据路径见 `sota_launch_gate.json`、`student_adapter_matrix.json`、`student_compatibility_report.md`。
