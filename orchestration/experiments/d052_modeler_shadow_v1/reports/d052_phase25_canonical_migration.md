# D052 Phase 2.5 — Canonical Migration Bundle 交付报告

- **任务**：把已验证的 Modeler B/C 机制证据冻结为 Canonical Migration Bundle，交付 canonical_v2 主线（CC3）。
- **分支**：`henry/d052-modeler-shadow-v1`（未 push；未改 canonical_v2 主分支；无 merge）。
- **包路径**：`mechanism_UED/orchestration/experiments/d052_modeler_shadow_v1/artifacts/d052_phase25_canonical_migration/`
- **本阶段遵守**：未训练；未修补旧 salted-hash launcher；未把旧 selected-8 当正式训练任务；未改 canonical_v2 主分支；无整分支 merge；未设计第二套 D052 框架。
- **完整性**：`sha256sum -c SHA256SUMS` 全部通过；从 bundle 的 judgments 重放 selector，B/C selection_hash **精确复现锚点且确定性一致**。

---

## 1. 最终冻结标志

```
PHASE25_MIGRATION_BUNDLE_COMPLETE = true
MODELER_MATCHED_SELECTION_EFFECT  = CONFIRMED
MODELER_LEARNING_VALUE            = UNTESTED
LEGACY_SALTED_HASH_TRAINING_PATH  = INVALID
LEGACY_SELECTED8_TRAINING_READY   = false
```

**释义**：
- `BUNDLE_COMPLETE=true`：14 个必需文件齐备、自洽、可重放、哈希锚定。
- `MATCHED_SELECTION_EFFECT=CONFIRMED`：在唯一变量=是否附加冻结 profile 的严格匹配协议下，selected-8 变化 4/8（Jaccard 0.333）——Modeler 对选择有真实净效应。
- `LEARNING_VALUE=UNTESTED`：离线无法区分"有益再校准"与"失稳过度修正"；且旧训练路径因 salted-hash 不可解释，学习价值从未也**不能**在旧路径上测。
- `SALTED_HASH_TRAINING_PATH=INVALID`：加盐 hash 进入 reward/termination/success/loss，旧 enhanced 训练路径语义无效。
- `SELECTED8_TRAINING_READY=false`：旧 selected-8（B 或 C）均**不得**作为正式训练任务（target 为加盐占位符）。

---

## 2. 文件清单（14）与 SHA256SUMS

```
51df83d175b49a47a9d3904b592c69ab28edf3b997711bcdd0df77036abe13dc  expected_behavior.json
210a57f74504237f92f30978882df723ec09fb905c23d8a2d75e168b7ce3211d  field_mapping.json
cca913b3e8f6dc4de7674879d2efbc143de4c37555702302f2c4714a5c011c17  judgments_B.jsonl
320950e5f6fa0f027999a6c535aa81b4fbfa9af6208d1d7435af6ff14fd8bbe5  judgments_C.jsonl
ddb0a82aea8910f44998d86f7b4806ae36a7e358b813ad9c5d2217ba4604cb26  prompt_registry.json
4d8b2f20fcb5a1ce4fa9beaa11825e75dadf874a4acc190d9fe1671e5816288a  protocol.json
2b92f3a27ec087f9ad41a58d03109d3f84244cb3ef42b5000050b026d72848fe  ranking_B.json
2b0b55d33fdb6254d19c412314a4bfa6a4ebb993712f6d7264bb57348b08aaf9  ranking_C.json
4ac0cd7b9743cf51afaca2073c5cfac3358835ca76f0329b9154fe12e1aa5c7c  regression_test_spec.md
5245448444577594b83f7d7532340f42b1e7238dc32378a62a5f3982625b621e  role_ablation.json
8e3e28a02d101787cbbd81a89822c399db19939117d6d2aa9d4292830808638b  salted_hash_audit.json
4e5256640ec254b6cbcf9be6efdc378ee3f0ea697f409e29cfc056fc5b0d1eb6  selector_config.json
749fb4468ccc4a3df64041f2b95507f51a53f140abd6a8e946c23239fb89b017  student_profile.json
```
（SHA256SUMS 文件自身不含其本身哈希；`sha256sum -c` 13 项全部"成功"。）

| 文件 | 内容要点 |
|------|----------|
| protocol.json | B/C 严格匹配协议：受控不变量（池/顺序/匿名 ID/模型/temp=0/timeout/schema/normalization/selector/seed/tie-break）、唯一允许差异、6 次调用预算、B 臂为新调用、禁止项 |
| student_profile.json | profile_hash=`223defdf…`；calculator SHA=`bab4388b…`；machine_facts（确定性）与 llm_interpretation（冻结 profile 逐字）分离；来源/round/诚实规则 |
| prompt_registry.json | prompt_version=`d052_phase25_v1`；B/C 各 3 角色完整 prompt 全文 + prompt_hash；raw_summary/profile/candidate_block 分项 hash；唯一差异说明 |
| judgments_B.jsonl / judgments_C.jsonl | 各 96 条；parse_status、raw_scores、role labels（含 glm 回显原值与规范化目标）、judgment_hash、provider/model/attempts |
| selector_config.json | normalization（robust median/IQR，clip=3.0）、Soft Copeland 权重 0.34/0.33/0.33/0.01/0.01、temperature=1.0、rng seed=none、tie-break、selection/pool hash 函数、源码 SHA |
| ranking_B.json / ranking_C.json | B/C 全 32 排序（rank/分数/信号/tier/targets/in_sel8） |
| role_ablation.json | 4/8 变化、Jaccard 0.333、三角色 B/C 相关与均值差、34/96 翻转（tutor 24/critic 10/explorer 0）、leave-one-role-out、全排序 Spearman 0.437 |
| salted_hash_audit.json | hash() 位点（launcher:818→832）；reward/termination/success/success-mask 数据流（base_task.py:52-62/28-50/69-81, setup.py:300）；obs conditioning 说明；**固定 PYTHONHASHSEED 不是修复**的明确声明 |
| field_mapping.json | 旧字段→canonical_v2（新 Schema 不可见，标 UNAVAILABLE）；可直接迁移/需转换/必须丢弃/缺失信息 |
| expected_behavior.json | legacy pool_hash、B/C selection_hash、4/8、0.333；明确"仅历史机制锚点，不要求 canonical 新池复现相同 selected-8" |
| regression_test_spec.md | R1 禁 hash/salted/unknown 目标；R2 B/C 匹配字段；R3 judgment replay；R4 selector determinism；R5 canonical target 语义；R6 profile 完整性 |
| SHA256SUMS | 全包 sha256 清单 |

---

## 3. 核心证据摘要

**B/C 匹配效应（CONFIRMED）**：
- 协议唯一差异 = 是否附加冻结 StudentProfile；其余全部字节级一致（prompt 前缀/后缀相同，C 仅插入 profile JSON 块）。
- selected-8：B=`{0000,0006,0007,0011,0014,0022,0025,0028}`，C=`{0000,0003,0010,0014,0017,0024,0025,0028}`；**变化 4/8，Jaccard 0.333**，selection_hash `82571538…→868a5726…`。
- Tutor 为最大判断移动者（progression 均值 6.84→1.52，24/34 翻转）；最终选择变化为三角色弥散合力（leave-one-out 各使 overlap_with_B=4）。
- 全排序 Spearman(B,C)=0.437；确定性自检通过；6 次调用（itok 25829 / otok 17815 / attempts 9）。

**Salted-hash（INVALID）**：加盐 `hash()`→`relevant_achievements`→进入 reward/termination/success/success-mask（具体文件/函数/行号见 salted_hash_audit.json）。eval pilot 自标 `SUCCESS_MODE="UNDEFINED"`。**固定 PYTHONHASHSEED 仅冻结一个任意分配，不恢复语义，不是修复**；正确修复=显式可逆的 canonical name→enum 映射。

---

## 4. 已知缺失（交给 canonical_v2 / CC3）

1. **canonical_v2 新 Schema 在本服务器不可见**（find + grep 于 `mechanism_UED_continuation_20260715` 均无命中）。`field_mapping.json` 的"新 Schema 字段"列因此全部标 `UNAVAILABLE`，须由 canonical_v2 owner 补全：
   - 候选/任务 Schema（id 命名空间、task_params、difficulty 本体）
   - judgment/score Schema（分数范围、必需字段）
   - StudentProfile/evidence Schema
   - **canonical achievement enum + name→enum 表**（用于再生 target，解除 salted-hash）
2. **学习价值 UNTESTED**：需 canonical_v2 在 canonical target 上跑 A/B smoke 才能判定 Modeler 是否有助学习（离线不可判）。
3. **glm role 回显怪癖**（B/C explorer 各 9 条 builder/survivor）：已透明规范化并记录于 `outputs/bc_{B,C}_explorer_normalization_log.json`；该字段不进入 selector，但 canonical_v2 应在 R3 中固化此规范化。
4. **Phase 2 tutor/critic token 成本未持久化**（历史遗留，见 phase2_shadow.md §6.2）；Phase 2.5 的 6 次调用成本完整。

---

## 5. 合规摘要

- 未训练、未修补旧 launcher、未把旧 selected-8 当正式任务、未改 canonical_v2 主分支、无 merge、无第二套框架。
- 省 Token 铁则：无子代理/background agent；本阶段 0 次新 LLM 调用（纯确定性打包）；仅拉取小型 JSON/CSV/源码哈希。
- 凭证 blind 加载，从未读取/输出 key/token。
- Git：分支 `henry/d052-modeler-shadow-v1`，未 push、未 reset --hard、未 clean -fd、未 force push、未改 shared_r0 原始 artifact。
- 所有哈希为真实计算（profile/prompt/judgment/selection/source SHA），非占位。

---

## 6. 交付与停止

迁移包已完成并冻结。**现停止，将包路径、文件清单、SHA256SUMS 与已知缺失交给 CC3。不启动任何训练。**

CC3 接手建议：先补全 canonical_v2 Schema 与 canonical achievement 映射（解除 salted-hash），按 `regression_test_spec.md` 的 R1–R6 落地守卫，再决定是否在 canonical 池上以本 bundle 的匹配协议复跑并授权 A/B smoke。本 bundle 的协议、prompt、selector、profile 可直接复用。
