# 五项科学门禁关闭状态（v2 修订版对应总控 REQUEST_CHANGES）

> **INDEPENDENT_AUDIT_REQUIRED = true**。
> "机制关闭" = 代码+fail-closed 测试绿；"证据关闭" = 真实数据到位。
> 本轮 G1/G2/G3 只达到机制关闭，证据一律诚实缺省（false/BLOCKED），
> 未伪造任何 probe、retention 或 Reference 身份。

## G1 ReferenceIdentityContract —— 机制关闭；身份值待总控冻结

- 实现：`src/dicode/teachers/e1_formal/reference_contract.py`
  （frozen dataclass；candidate_id、checkpoint_ref、file_sha256、
  params_sha256、network_architecture、memory_semantics、global_step、
  total_env_steps、source_commit、seed、episode_reset_protocol、
  frozen_manifest_hash、provenance；**全部无默认值**）。
- fail-closed 码：`REFERENCE_CONTRACT_MISSING_FIELD` /
  `_BAD_TYPE` / `_GUESSED_FORBIDDEN`（TODO/latest/auto/空/占位⇒拒）/
  `_UNFROZEN` / `_MANIFEST_HASH_MISMATCH`。
- 配置 seam：`conf/teacher/e1_formal.yaml::reference_contract`
  （默认 `frozen: false`）；评价 seam 第一道门即本契约
  （`evaluation/candidate_evaluation.py`，门序固定：输入校验→G1→
  adapter/state→config knob；阻断结果不带 provenance 章）。
- 证据：`tests/e1_formal/test_reference_contract.py`（逐字段缺失/
  坏型/占位拒绝；未冻结⇒seam 阻断链）。
- **待总控**：冻结的 Reference 身份清单（8 类字段值 + manifest hash）。
  冻结前 `REAL_STUDENT_REFERENCE_EVAL=false` 保持。

## G2 Probe-based Learnability —— 机制关闭；真实 probe 待 CC4

- 实现：`src/dicode/teachers/e1_formal/metrics.py`：
  `classify_learnability` 三态 LEARNABLE / SATURATED /
  BOTH_UNREACHABLE + INSUFFICIENT_EVIDENCE（episode 数/CI 宽不达标
  ⇒无裁决）；Wilson CI；阈值全部来自冻结配置块，缺失即
  `LEARNABILITY_THRESHOLD_MISSING`（无硬编码默认）。
- archive LP 仅作独立先验字段 `learnability_prior_lp` 记录，
  **永不进入排序替代真实证据**；v1 的"无历史→0.25"已删除
  （grep 审计 + `test_learnability.py` 断言 0.25 不在 metrics.py）。
- 本轮无 probe ⇒ `LEARNABILITY_UNAVAILABLE` ⇒
  `SELECTION_BLOCKED_NO_REAL_EVIDENCE`：动态候选不晋升，
  batch `training_permitted=False`、`task_ids=[]` —— **零训练更新**
  （C13：不再有 anchors-only 偷跑；集成 smoke 断言阻断码出现、
  notes 无伪造数值）。
- 证据：`tests/e1_formal/test_learnability.py`（labeled FIXTURE
  数据明示为 fixture，绝不称真实）。
- **待 CC4**：真实双 probe rollout（依赖 G1 冻结 + adapter）。

## G3 Anchor Retention —— 机制关闭；manifest 冻结待总控

- 实现：`src/dicode/teachers/e1_formal/anchor_manifest.py`：
  `SharedAnchorManifest`（每 anchor：anchor_id、source_task_id、
  task_params_hash、seed_protocol、code_hash、reset_protocol、
  frozen_by、frozen_at；整体 manifest_sha256）。
- 草案：`configs/e1_formal_ued_anchor_manifest.DRAFT.json`，
  `status=DRAFT_UNFROZEN`，frozen_by 空；
  DRAFT manifest sha256 =
  `5b81204102e3843fa7d33ce7c14f9258345ffb21759aafa39df7c0e8bce9a1e4`。
- retention 评测仅在 manifest frozen 时可调用；冻结前一律
  `BLOCKED_SHARED_ANCHOR_MANIFEST`（另有
  `ANCHOR_MANIFEST_HASH_MISMATCH` / `_NOT_FROZEN`）。
  selector 的 retention 硬过滤/软罚**整体停用且无替代指标**；
  v1 的"成就数 retention"已删除（grep 审计）。
- 12+4 batch 结构保留：anchor 按注册原样进 batch
  （`[task_1,task_2,task_3,original_craftax]`，original 恒最后），
  教师永不修改 anchor。
- 证据：`tests/e1_formal/test_anchor_manifest.py`（DRAFT⇒必阻断；
  hash 篡改⇒mismatch 码）。
- **待总控**：跨方向共享 anchor manifest 冻结（4 anchor 身份/
  TaskParams/seed/hash + 签署 hash）。

## G4 Soft Copeland Parity —— 机制关闭（对本分支 d052 canonical）

- 事实：本 worktree **不含** CC3 的 `d052/bagr_ued/`；本分支
  canonical 实现 = `d052/selectors/copeland.py`（协议 `canonical_v2`）。
- pin（SHA256，与 `reports/d052_canonical_artifacts_SHA256SUMS` 一致）：
  - copeland.py `80a60829537c87bafcc17aef7715cd37f6fdad0027cc16f27832744f11f6d613`
  - canonical_constants.py `32c7a1c9dd28fc0388d213591061cd7eb5e1a1944fc68ee1ab448c1eec822bf2`
  - base.py `c9d0858548176e50a5ce561258ac0863fb8908b9b789c9293116702ad2ede108`
- E1 selector 为自包含 stdlib 复刻；运行时**不** import d052；
  parity 测试侧只读 import。pin SHA 与分支实际不符⇒
  `COPELAND_SOURCE_SHA_MISMATCH` 硬失败（不得擅自改 pin 绕过）。
- 门禁：`tests/e1_formal/test_copeland_parity.py`（**无 skip**）：
  ≥6 候选 fixture（含平局、veto、输入乱序置换），断言逐候选分数
  向量/全成对矩阵/最终排序/canonical 结果 hash **四项完全相等**
  + 两侧顺序无关性。不等⇒修 E1 侧直至相等，绝不改 d052/放宽断言。
- **待总控（可选扩展）**：CC3 bagr_ued soft_copeland 源或 SHA；
  若 CC3 与 d052 canonical 不同源，请指明权威源（第三实现比对位
  已设计）。

## G5 LLM Accounting —— 关闭

- 实现：`src/dicode/teachers/e1_formal/accounting.py`
  （`LLMCallLedger`，JSONL 持久化 `e1_state/llm_accounting.jsonl`；
  kind∈{BOARD,ENVCODER,REPAIR}）。
- 公式：**N1 = 6·G1 + T1 + K1 + F1**（G1=实际触发的 review window
  数；**T1≡0**，E1 无 TaskGenerator，显式常量；K1=按**唯一模板**
  计（round-3 模板键控：每唯一 family 模板恰 1 次 EnvCoder 调用，
  变体共享模板调用，template_hash 去重）；F1=repair 独立计数，
  round-3 起为**真实有界修复调用计数**——
  `envcoder.run_envcoder_with_repair` 每次修复调用经
  `record_repair_call` 入账，受 `teacher.envcoder.max_repairs`
  （硬上限 `MAX_ENVCODER_REPAIRS=2`）约束；F1 永不与 K1 混合。
  round-2 的"F1≡0/单程出码"表述已作废）。
- "每窗 7 次/第 7 次"表述已从全部源码删除（grep 审计 +
  `test_llm_accounting.py` 模块文档断言）。
- 证据：`tests/e1_formal/test_llm_accounting.py`：REUSE 窗⇒0 调用；
  1 窗+10 唯一模板×2 变体⇒board=6、K1=10（变体共享模板调用，
  绝不为 20）、T1=0、F1=0；重复模板不重复计数；
  `LLM_ACCOUNTING_MISMATCH` 核对失败码。修复耗尽正负 pin：
  `test_gen_manager_duck.py`（坏模板重试 2 次仍失败⇒K1 照计、
  F1=2、`ENVCODER_REPAIR_EXHAUSTED`、整窗诚实拒绝）。
- 无待办。

## 降级链（D5）总览——五门禁未全开时的诚实行为

```
G1 REFERENCE_CONTRACT_UNFROZEN
  => EVAL_SEAM_SKIPPED_NO_STUDENT_ADAPTER
  => G2 LEARNABILITY_UNAVAILABLE
  => SELECTION_BLOCKED_NO_REAL_EVIDENCE
  => batch 零可训练任务（training_permitted=False）
  => enforce_training_gate 拒绝 => 零 PPO 更新、零 global/env step 前进
     （G3 retention 亦 BLOCKED_SHARED_ANCHOR_MANIFEST）
```

集成 smoke 断言链上每个码如实出现；任何一级都不以 archive/
启发式数值伪造真实证据。

## C13 训练门禁（总控 REQUEST_CHANGES 修复）——关闭

**缺陷（已修）**：C11/C12 版本中 `build_training_batch()` 阻断时返回
4 个硬编码 anchors 且 `reuse_only=True`，而 `run_dicode.py` 忽略
`blocked_codes`/`reuse_only` 仍调用 `run_session_training()`——
硬门禁阻塞下仍产生训练更新；且"REUSE"未携带上一窗口已验证的
12 dynamic IDs。

**修复（双层 fail-closed）**：

1. 教师侧（`gen_manager.build_training_batch`）：任何适用硬门禁阻塞
   ⇒ `task_ids=[]`、`training_permitted=False`、携带全部阻断码；
   绝不产出 anchors-only 可训练批次。
2. 门禁侧（新 `training_gate.enforce_training_gate`，纯 stdlib）：
   `training_permitted` 必须严格为字面 `True`；许可批次独立复核为
   恰 12 dynamic + 4 frozen shared anchors（canonical 顺序）且 pinned
   layout 覆盖全部 16 任务——anchors-only/乱序/重复/legacy 分布
   均拒绝。`run_dicode.py` 的 E1 钩子分支先过门禁，未许可即抛
   `RuntimeError`（零更新、零 step、不回退 legacy 采样）。
3. 合法 REUSE = 上一窗口**完整已验证**的 12 dynamic + 4 frozen
   shared anchors，经 `record_verified_batch` 认证并携带
   来源/窗口/hash 证据：`window_id`、provenance 必须恰为
   `CANDIDATE_EVALUATION`（真实双 probe 路径）、12 个动态任务逐条
   绑定本教师 registry 的 `spec_hash`/`code_sha256`、anchor manifest
   sha256 与当前冻结 manifest 相等、Reference candidate_id 与冻结
   契约相等；**C14 起另需全部结构化证据（见下节）**。认证前置：
   G1 冻结 + G3 冻结 + 阈值冻结；任一不满足 ⇒
   `GEN_MANAGER_SNAPSHOT_BLOCKED`。无合法快照 ⇒
   `TRAINING_BLOCKED_NO_VERIFIED_BATCH`（或相应阻断码）⇒ 跳过训练。
- fail-closed 码：`TRAINING_GATE_BLOCKED`、`TRAINING_GATE_BAD_BATCH`、
  `TRAINING_BLOCKED_NO_VERIFIED_BATCH`、`GEN_MANAGER_PROMOTION_BLOCKED`、
  `GEN_MANAGER_SNAPSHOT_{BAD_TYPE,MISSING_FIELD,MISMATCH,BLOCKED}`。
- 证据：`tests/e1_formal/test_training_gate.py`（正负矩阵：DRAFT
  manifest / 缺 dual-probe / 空快照 / 伪快照（篡改 sha、幽灵任务、
  错 provenance、错窗、错锚点）⇒ 全部零训练；FIXTURE 合法 12+4 ⇒
  恰训练一次）；`test_wiring_sources.py` AST 断言 run_dicode 中
  门禁先于 `run_session_training`、无模块级 E1 导入、legacy 采样不变。
- 诚实声明：本轮正例测试使用明示 FIXTURE 的冻结契约/manifest/快照，
  仅证明机制；真实 Reference/anchor 冻结件与真实 probe 到位前，
  生产路径永远落在阻断侧（零训练）。

## C14 REUSE 证据收紧（record_verified_batch）——关闭

**动机（总控 CC2/E1 指令）**：C13 的 `record_verified_batch` 只把
`artifact_id` 校验为非空串（未与内部 registry 对账），且 REUSE 认证
以 provenance 串为中心。收紧后：**provenance 串单独永远不足以认证
REUSE**，必须携带全部结构化证据，逐字段 fail-closed。

新增/收紧的认证要求（`gen_manager.record_verified_batch`，违反即抛
`GEN_MANAGER_SNAPSHOT_{BAD_TYPE,MISSING_FIELD,MISMATCH}`）：

1. **artifact_id 对账**：每个动态任务的 `artifact_id` 必须等于本教师
   内部 registry 记录的 `artifact_id`（引用其它任务的真实 id 亦拒）。
2. **结构化 dual-probe 块** `dual_probe`（全字段必需、未知字段拒）：
   `student_candidate_id` 必须恰为 pinned 强 Student
   `PERSISTENT_RMT16_ORIGINAL_VTRACE_98304`；`student_probe_id` /
   `reference_probe_id` 非空串；`student_probe_hash` /
   `reference_probe_hash` 必为 64 位小写 sha256 hex。
3. **Reference 身份哈希** `reference_identity_hash`：新增
   `reference_contract.reference_identity_sha256`（对冻结契约全部必需
   身份字段+schema_version 的 canonical sha256），快照必须与**当前**
   冻结契约的身份哈希相等；`_snapshot_still_valid` 亦复核此绑定
   （重新冻结身份即使旧 REUSE 失效）。
4. **窗口哈希** `window_hash`：64 位 sha256 hex，且必须等于 12 个动态
   任务在 registry 中逐条记录的 `window_hash`（registry 无窗口哈希的
   artifact 永不可认证）。`consume_worker_results` 因此新增记录
   `window_hash`（附加字段，legacy 行为不变）。
5. **候选集哈希** `candidate_set_hash`：对按序认证的 12 个动态任务 id
   的 canonical sha256；换序/增删/换 id 均拒。
6. **晋升路径同步收紧**：`build_training_batch(promoted_dynamic_ids,
   dual_probe=…)` —— 12 个晋升 id 必须随附经同一校验的 `dual_probe`
   块；`_certify_dynamic_window` 现存储与 `record_verified_batch`
   同构的完整结构化证据（窗口哈希来自 registry、身份哈希来自当前
   契约、候选集哈希按序计算）；registry 无窗口哈希的窗口永不晋升。

- 证据：`tests/e1_formal/test_training_gate.py` 新增 64 条 C14 正负
  测试（`TestC14EvidenceBindingBypassAttempts` /
  `TestC14PromotionBypassAttempts`）：伪 artifact_id（含跨任务真实
  id 调换）、dual_probe 缺失/非映射/未知字段/逐字段缺失/错 Student/
  空 probe id/坏哈希、窗口哈希缺失/坏型/与 registry 不符/registry 无
  窗口哈希、身份哈希缺失/坏型/错值、候选集哈希缺失/坏型/换序/换 id、
  **仅持正确 provenance 而无结构化证据 ⇒ 拒且零训练**；正例：完整
  结构化证据 ⇒ 认证 ⇒ REUSE 恰训练一次，晋升携带同构证据。
  `test_reference_contract.py::TestIdentityHash`（确定性/格式/任一
  身份字段变更⇒哈希变更）6 条。全套 1003 passed / 5 skipped。
- 诚实声明：正例使用的 probe id/哈希、窗口哈希为明示 FIXTURE；真实
  CC4 双 probe 记录到位前，生产路径永远落在阻断侧。
  `b5536d3` 的阻断零训练行为保持不变；REAL_* 标志保持 false。

## C15 REUSE 认证绑定与全量复核——关闭

**指令（总控 CC2/E1）**：调用方自带的 probe 串/哈希单独永不充分
——dual-probe 证据必须由 adapter 铸造（attestation），且每次 REUSE
前 `_snapshot_still_valid` 必须重验全部绑定。

新增/收紧（违反即 fail-closed，码仍为
`GEN_MANAGER_SNAPSHOT_{BAD_TYPE,MISSING_FIELD,MISMATCH,BLOCKED}`）：

1. **adapter 铸造缝** `record_dual_probe_attestation`：真实 probe
   证据只可能来自 CC4 评价 seam（共享 StudentAdapter + 冻结
   Reference），本消费端铸造其记录——Reference 契约必须已冻结
   （未冻结 ⇒ SNAPSHOT_BLOCKED）；字段集固定（未知字段拒）；
   `adapter_id`（铸造 adapter 身份）非空；`student_candidate_id`
   必为 pinned 强 Student；`reference_candidate_id` 必等于**当前**
   冻结契约的候选 id（在旧 Reference 下铸造的 attestation 于重新
   冻结后永不再认证）；Student/Reference probe id 非空且互异、
   probe 哈希为 sha256-hex 且互异（同 id/同哈希=调换或退化 probe
   对 ⇒ MISMATCH）；完全重复的 attestation 只铸造一次；
   `probe_attestations` 属性只读供审计。
2. **调用方串单独永不充分**：`record_verified_batch` 与晋升路径
   （`_certify_dynamic_window` 为认证咽喉点——直接私有调用同样
   fail-closed）现要求 `dual_probe` 块与一条已铸造 attestation
   匹配且绑定当前 Reference 候选（`_require_attested_dual_probe`）；
   格式合法但未铸造、或 Student/Reference 角色互换的 probe ⇒
   MISMATCH。
3. **REUSE 全量复核**：`_snapshot_still_valid` 在每次复用前重验
   全部绑定——门禁 blocker、provenance 恰等、canonical anchors、
   当前 manifest sha、当前 Reference 候选 id 与身份哈希（重冻结
   任一契约字段含 episode reset 协议即失效）、dual-probe 结构与
   attestation 绑定、12 个动态任务逐条对**当前** registry 复核
   （artifact_id/spec_hash/code sha256/window_id/window_hash；
   registry 记录被新窗口覆盖=窗口过期 ⇒ 失效）、候选集哈希按序
   重算；从不抛异常，任何不一致 ⇒ False ⇒ 零训练
   （`TRAINING_BLOCKED_NO_VERIFIED_BATCH`）。

- 证据：`tests/e1_formal/test_training_gate.py` 新增 66 条 C15 测试
  （4 类）：铸造缝 fail-closed 矩阵（阻断教师不可铸造/非映射/未知
  字段/逐字段缺失/空 adapter_id/错 Student/错 Reference/空 probe
  id/同 id/坏哈希/同哈希）；认证期调用方串不足（未铸造串、
  Student↔Reference probe id+哈希互换、Reference 重冻结后旧
  attestation 失效、晋升路径未铸造 probe）；逐次 REUSE 复核
  （整窗/单条记录重消费⇒窗口过期、registry 记录删除⇒未知 id、
  artifact 变更、Reference 身份变更、reset 协议变更、manifest
  变更、候选集换序、存储快照伪造哈希×3、存储快照 probe 互换、
  存储快照 Student 变更）；直接私有旁路（直接翻转
  `_real_selection_completed`、直接安放伪造快照——含与 registry
  完全一致但 probe 未铸造的"完美伪造"、篡改已存副本、直接调用
  `_certify_dynamic_window` 传未铸造/畸形 probe）；以及 1 条
  adapter 铸造正路径（铸造⇒认证⇒REUSE 恰训练一次，且下一次复用
  仍过全量复核）。全套 1069 passed / 5 skipped。
- 诚实声明：正路径的 attestation/probe id/哈希为明示 FIXTURE
  （adapter id `cc4-student-adapter-fixture-v1` 为测试占位）；真实
  CC4 双 probe 记录到位前，生产路径永远落在阻断侧。`b5536d3` 的
  阻断零训练行为保持不变；REAL_* 标志保持 false。

### C15-RC 修订（总控 REQUEST_CHANGES）：铸造绑定内部 registry +
不可变结果对象——关闭

**指令**：C15 首版的 `record_dual_probe_attestation` 是公开调用方
方法——任何调用方可传入任意非空 `adapter_id` + 合法格式的 id/哈希
先铸一条假 attestation，再凭它认证 REUSE。无 adapter 能力/注册表/
真实评价结果查询 ⇒ 「先铸假、再认证」攻击面成立。要求：铸造绑定
内部候选评价 adapter registry/结果对象与不可变 Student/Reference/
checkpoint/window/protocol 证据；即便字段全合法也拒绝调用方形状的
映射；补直接铸造/假 adapter/未知结果负例与 1 条 adapter 签发正路径。

已实施（**上述 mapping 铸造缝已删除**；违反即 fail-closed）：

1. **新模块 `e1_formal/eval_adapter.py`（纯 stdlib）**：
   - `CandidateEvalAdapterRegistry`：adapter 仅经 fail-closed 注册
     进入（字段集固定、id/version 非空非占位、adapter_hash
     sha256-hex、能力串必恰为 pinned
     `candidate_evaluation_dual_probe_v1`；同 id 异 spec 冲突拒、
     同 spec 重注册幂等）。
   - 签发只在 registry 内部（`issue_dual_probe_result`，**仅关键字
     标量参数，铸造链上无任何 mapping 入参**），且只认**已注册**
     adapter——假/未知 adapter ⇒ `EVAL_ADAPTER_UNKNOWN`，对象根本
     不会被创建。
   - `DualProbeResult`：frozen dataclass，14 字段完整证据链
     （Student/Reference 候选 id、**Student/Reference checkpoint
     哈希**、probe id+sha256、窗口 id/哈希、按序候选集哈希、episode
     reset 协议 id+哈希）；构造即 fail-closed 校验（全部 str 非空、
     占位值拒、7 个哈希字段 sha256-hex、probe id/哈希互异）。
2. **教师消费端 `consume_candidate_eval_result`**（替换旧铸造法）：
   只接受 `DualProbeResult` 实例——调用方形状映射（**字段全合法也
   拒**）/None/list/str/int ⇒ SNAPSHOT_BAD_TYPE「NEVER accepted」；
   必须为本教师 registry **签发过**的成员（直接构造或
   `dataclasses.replace` 变形 ⇒ 未知结果 ⇒ SNAPSHOT_MISMATCH）；
   Reference 契约必须当前冻结（SNAPSHOT_BLOCKED）；绑定当前契约：
   pinned Student、Reference 候选 id、Reference checkpoint
   （params_sha256）、reset 协议 id+哈希（任一不符 ⇒ MISMATCH）；
   结果携带的窗口 id/哈希与候选集哈希成为后续 REUSE 认证的**作用域**。
3. **作用域化认证匹配**：`_dual_probe_attested` /
   `_require_attested_dual_probe` 现要求已消费记录同时匹配窗口 id、
   窗口哈希、候选集哈希、5 个 probe 字段与**当前** Reference 候选；
   `record_verified_batch` 的 attestation 检查移到候选集哈希核算
   之后；`_certify_dynamic_window` 先核 registry 证据与窗口哈希、
   后验 probe 作用域；`_snapshot_still_valid` 逐次 REUSE 以快照自身
   窗口/候选集哈希复核作用域。
4. **负例矩阵**：注册 fail-closed（非映射/未知字段/逐字段缺失/空或
   占位 id/坏哈希/错能力/冲突重注册）；签发 fail-closed（**假/未知
   adapter**、未注册任何 adapter、未知字段、14 字段逐一缺失、7 哈希
   字段坏值、空 id 字段、同 probe id/哈希、占位窗口 id、不可变性、
   变形副本非签发成员、重复签发去重）；**直接铸造拒绝**（字段全合法
   的映射、真实结果的映射副本、None/list/str/int、**直接构造的
   合法 DualProbeResult（即便引用已注册 adapter id）**、签发结果的
   replace 变形、阻断教师消费）；**证据链绑定**（错 Reference
   checkpoint 哈希、错 reset 协议 id/哈希、非 pinned Student、非当前
   Reference 候选 ⇒ 消费即拒；为**他窗口/他候选集**签发的结果 ⇒
   认证即拒）；1 条 adapter 签发正路径（注册⇒签发⇒消费⇒认证⇒
   REUSE 恰训练一次，下一次复用仍过全量复核）。晋升正例补窗口 B
   作用域签发（窗口 A 的结果不能覆盖窗口 B 的候选集）。
- 证据：`tests/e1_formal/test_training_gate.py` C15 类重写为 6 类
  共 121 条（registry 注册/签发 fail-closed 矩阵 80、直接铸造/
  伪造/证据链/作用域 18、调用方串不足 4、逐次 REUSE 复核 13、
  直接私有旁路 5、adapter 签发正路径 1；C13/C14 各类 111 条沿用）。
  全套 **1124 passed / 5 skipped**。
- 诚实声明：正路径 adapter/checkpoint/probe 值均为明示 FIXTURE；
  Student checkpoint 哈希本轮仅作格式校验的不可变证据，真实 Student
  checkpoint 核验待 CC4；`issue_dual_probe_result(**mapping)` 在
  Python 层面仍可被解包调用，但每个值都经 fail-closed 类型/格式校验
  且仅已注册 adapter 可签发、教师只认 registry 成员实例。真实 CC4
  双 probe 到位前生产路径恒阻断；REAL_* 标志保持 false。

## 待总控冻结项清单

1. **Reference 身份**（G1 身份值 + manifest hash）；
2. **跨方向共享 anchor manifest 冻结**（G3：4 anchor 身份/
   TaskParams/seed/hash + 签署 hash）；
3. **（可选）CC3 bagr_ued copeland 源/SHA**（G4 扩展比对）；
4. **round-3 新增**：真实 LLM provider 授权白名单（本轮为空；
   六角色 board 仅在显式授权下可用真实 provider，envcoder/probe
   永不回退）；共享运行时 `dicode.shared_runtime`（CC4：八合同
   StudentIdentity/StudentAdapter/ReferenceIdentity/ReferenceAdapter/
   AnchorManifest/FormalAssetRegistry/CandidateProbeResult/
   FullStateCheckpoint）；本机 craftax 运行时（真实 EnvCoder
   后端 IMPORT/INSTANTIATE/RESET/STEP/TERMINAL_AUTORESET 全阶段）。

## round-3 章节：PRODUCTION_PATH_READY_FOR_AUDIT

> 推翻 round-2 表述的四处更正已在对应段落就地完成：
> "≤10 spec/窗"→ **≤10 唯一模板/窗**（`MAX_WINDOW_TEMPLATES=10`；
> 6 模板×2 变体=12 specs，spec 池上限 20）；"F1≡0"→ **F1=真实
> 有界修复计数（≤2/模板）**；"stub 补槽"→ **整窗拒绝
> （`INSUFFICIENT_DYNAMIC_ARTIFACTS`，无跨窗拼接、无 stub、
> `_reuse_stub` 仅存于不可训练 REUSE 批次标记）**；
> "10 spec×2 变体⇒K1=20"→ **K1=10（唯一模板）**。

**C1 `fix(e1): close sequential-board and 12-slot blockers`**
（`62eb560`）：
- P0-1：`BoardContext`/`UpstreamOutput`；`build_role_prompt` /
  `build_prompt_envelope_hash` 绑定 window 身份四元组 + pinned
  Student 身份 + 上游角色输出（prompt 渲染 + role_output_hash 哈希
  双轨）；`BOARD_PROMPT_VERSION=e1-board-prompt-v2`。
- P0-2：模板键控 EnvCoder（`TaskTemplate`，每唯一模板恰 1 次调用，
  K1 按唯一模板）；`derive_variant_params` 确定性无 LLM；删除 stub
  补槽——编译 artifact ≥12 ⇒ 全池；1–11 ⇒ 整窗拒绝
  （`INSUFFICIENT_DYNAMIC_ARTIFACTS`⇒REUSE 批次，ledger 保留诚实
  调用，无跨窗拼接，不训练）。
- P0-3：`gate_signals.py` 八信号从真实 TRAINING_WINDOW 证据计算
  （prev_window_hash/context_hash/evidence_ids/threshold_version
  绑定；无数据生产者⇒计算为 False+SIGNAL_NO_PRODUCER；阈值未冻结
  ⇒False+INVOCATION_THRESHOLD_MISSING，仅窗级降级，不阻塞 C13
  训练门）；`GateState.signals_binding_hash` 必填。

**C2 `feat(e1): wire real envcoder and candidate probe`**
（`9aeb489`）：
- P0-4：`envcoder_backends.py` 八阶段
  SYNTAX→GUARDS→STRUCTURE→IMPORT→INSTANTIATE→RESET→STEP→
  TERMINAL_AUTORESET；`replay`=前三阶段（stdlib-AST 入口面检查，
  诚实标注不执行 craftax）、`mock` 仅显式授权消融、`real` 未授权
  恒 fail-closed（`ENVCODER_BACKEND_BLOCKED`，绝不静默降级）；
  修复环 `run_envcoder_with_repair`（RepairRecord 哈希链、修复
  replay miss=HARD FAIL、耗尽⇒`ENVCODER_REPAIR_EXHAUSTED`）。
- P0-5：`criterion_selector.py` 八准则 criterion-wise Soft Copeland
  （准则内 pairwise，结构上不存在"先均值后 Copeland"；
  `rank_percentile_v1`；权重缺省等权 1/8 且 Fraction 和=1 校验；
  family_cap 必需；不足无回填；无真实 probe 证据⇒
  `SELECTION_BLOCKED_NO_REAL_EVIDENCE`）；`shared_runtime_seam.py`
  八合同只解析、不构造、不铸造、不伪装；统一入口
  `evaluate_candidate`（fail-closed 参数校验⇒共享合同解析⇒未绑定
  即 `BLOCKED_WAITING_SHARED_RUNTIME`，绑定后真实路径为未来轮次、
  绝不 stub）。

**C3 `feat(e1): add one-update and longrun entrypoints`**
（`9931ecd`）：
- `scripts/e1_production_runtime.py`：共用生产运行时解析（八接缝
  合同、真实 EnvCoder 后端授权、G1 冻结、G3 冻结、实时 git SHA、
  诚实 JSON 汇报）；不 import tests/fixtures、不默认 mock/replay、
  不付费调用、门禁未过不训练。
- `scripts/run_e1_real_one_update.py`：唯一单更新真实门——
  real reset/step（RealBackendAdapter 全阶段）→ real 六角色
  （真实 LLM；仅显式授权标志下 board 步可用真实 Replay，
  envcoder/probe 永不）→ real candidate probe（统一
  `evaluate_candidate`）→ criterion-wise 选 12 → 12+4 batch →
  `run_session_training`（`max_updates_per_session=1`，恰一次
  optimizer update）→ checkpoint save/load 回验（仅经共享
  FullStateCheckpoint 合同，鸭型 fail-closed，绝不第二套 loader）
  → NaN/Inf 全 params 叶检查。任一资产/合同缺失⇒对应 BLOCKED 码、
  exit≠0、`REAL_ONE_UPDATE_EXECUTED=false`。
- `scripts/run_e1_longrun.py`：只准备、不启动——冻结清单
  （total_env_steps=98304、pinned Student 身份、冻结 Reference
  身份、seed、冻结 anchor manifest、实时 git SHA、config hash、
  checkpoint hash、输出目录）；任一字段未冻结⇒拒绝（exit≠0）；
  `--launch` 额外要求全部生产门禁，本轮恒拒绝。

**C4 `docs(e1): record production readiness and blockers`**
（本提交）：
- `scripts/e1_formal_readiness.py` 从实际代码/配置/接缝状态计算
  `reports/e1_formal_ued/real_smoke_readiness.json`（布尔绝不
  手写）；`reports/e1_formal_ued/current_blockers.md` 逐项阻断证据；
  本章节就地更正 round-2 被推翻表述。

**本机诚实状态（C3/C4 实跑，JAX_PLATFORMS=cpu，审计 venv）**：
- 单更新入口：BLOCKED，exit 2，12 条阻断（8 接缝合同 +
  ENVCODER_BACKEND_BLOCKED + REFERENCE_CONTRACT_UNFROZEN +
  BLOCKED_SHARED_ANCHOR_MANIFEST + E1_REAL_LLM_NOT_AUTHORIZED）；
  `REAL_ONE_UPDATE_EXECUTED=false`。
- 长跑入口：prepare-only REFUSED，exit 2（Reference 未冻结、
  anchor manifest DRAFT、checkpoint 合同未绑定）；`--launch`
  亦 REFUSED；从未进入训练循环。
- 套件基线保持：993 passed / 5 skipped / 0 failed。
