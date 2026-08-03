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
  数；**T1≡0**，E1 无 TaskGenerator，显式常量；K1=按唯一 artifact
  逐条计，spec_hash+variant 去重；F1=repair 独立计数，本轮单程出码
  ⇒F1≡0，槽位存在且永不与 K1 混合）。
- "每窗 7 次/第 7 次"表述已从全部源码删除（grep 审计 +
  `test_llm_accounting.py` 模块文档断言）。
- 证据：`tests/e1_formal/test_llm_accounting.py`：REUSE 窗⇒0 调用；
  1 窗+10 spec×2 变体⇒board=6、K1=20、T1=0、F1=0；重复
  spec_hash+variant 不重复计数；`LLM_ACCOUNTING_MISMATCH`
  核对失败码。
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

## 待总控冻结项清单

1. **Reference 身份**（G1 身份值 + manifest hash）；
2. **跨方向共享 anchor manifest 冻结**（G3：4 anchor 身份/
   TaskParams/seed/hash + 签署 hash）；
3. **（可选）CC3 bagr_ued copeland 源/SHA**（G4 扩展比对）。
