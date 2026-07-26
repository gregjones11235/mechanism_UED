# D052 Phase 2.5 真实迁移包对账 —— 最终报告

- 任务：D052_PHASE25_REAL_BUNDLE_RECONCILIATION（§1–§13），经 **D052_PREMERGE_CORRECTION_V2** 合并前修正（①恢复 Henry 旧 D052 无效归档；②critic_reject policy 改为必须显式指定、fail closed），再经 **D052_PREMERGE_SEMANTIC_CLEANUP_V3** 语义清理（③消除 critic policy 标签作用域歧义；④拆分 critic_reject 派生规则（维度 A）与 selector 消费策略（维度 B）为两个独立字段；⑤收紧真实判定产物目录的 .gitignore 豁免为逐文件 allowlist）
- 日期：2026-07-26 ｜ 分支：`henry/d052-canonical-refactor` @ 968768e → 5f9ab74 → 49f9121 → 本轮修正 commit（本地 commit，**未 push**）
- 测试：**312 passed**（283 基线防火墙测试全部保留 + 15 项对账门 + 6 项 v2 修正门 + 8 项 v3 语义清理门；0 failed / 0 error / 0 skip）
- 训练：**0 timestep**；新 LLM 调用：**0**；迁移包原件：**只读**

## 1. 真实包获取与完整性（§2 前置）

真实 Phase 2.5 Canonical Migration Bundle 经项目专用 SSH 密钥从服务器 `oseasy@172.25.14.221` 拉取至 `orchestration/experiments/d052_modeler_shadow_v1/`（与服务器同仓库相对路径，75 文件）。四阶段完整性：服务器自检 13/13 → 交付清单三方互验（服务器 SHA256SUMS == CC Modeler 交付 13 SHA == 本地 control 报告 13 SHA）→ 本地自检 13/13 → 服务器↔本地逐文件 75/75。来源为服务器**未提交未跟踪**工作树（`henry/d052-modeler-shadow-v1 @ e6765fb`，目录状态 `??`）→ `GIT_COMMIT_PROVENANCE_CLAIMED=false`（如实记录，不声称 git 出处）。详见 `reports/phase25_real_bundle_acquisition.json`。

## 2. §3 历史重放：PASS

离线、无 LLM、无 RNG，逐字复刻包自带 wrapper 逻辑，使用包 `selector_config.json` 钉扎的原始 selector 源（SHA `27492e8a…`；其 `robust_normalize`/`_aggregate_soft_copeland` 与工作区已提交 `590fcef4…` 版本 AST 字节一致）。

```
REAL_PHASE25_BUNDLE_REPLAY   = PASS
REAL_PHASE25_B_SELECTION_HASH = 82571538e5299ea9   (复算 == 预期)
REAL_PHASE25_C_SELECTION_HASH = 868a57268d66b90b   (复算 == 预期)
REAL_PHASE25_SELECTED_CHANGE  = 4/8
REAL_PHASE25_MATCHED_PROTOCOL = PASS
legacy_pool_hash              = 1902b71a5d86fa00   (复算 == 预期)
Jaccard = 0.333 ｜ overlap = 4 ｜ 双跑比特一致 ｜ rng_seed = null
```

补充锚点：6 个 prompt_hash 全对；B/C 逐角色严格只差冻结 StudentProfile 插入块（3 角色插入块逐字节一致）；`raw_summary`/`candidate_block`/`profile_json` 注册表哈希全部复现；`profile_hash_sha256=223defdf…` 复现公式 `sha256(canon_json(llm_interpretation))`。**全程无任何锚点不匹配，未改一行代码迁就结果。** 机器数据：`real_bundle_replay_result.json`。

## 3. §4 字段映射：完成（实证）

逐字段对账 JSON+MD：`real_bundle_field_mapping_completed.{json,md}`。摘要见向总监汇报；关键：32/32 候选不可实例化 canonical Candidate（salted 目标 + 缺必填 melee_spawn_multiplier + 16 字符 chash）；192/192 judgment 在派生规则下无损映射；归一化/选择哈希两套体制互不改写。

## 4. §5/§6 契约与只读适配器

新包 `gpu1_aggregation_siege/d052/reconciliation/`（4 模块）：
- `real_bundle.py`：定位、SHA256SUMS 13/13 复验、judgment 防篡改公式（**192/192** 对 `outputs/` 原始记录复验通过，扁平化分数零漂移）；
- `judgment_adapter.py`：原始记录逐字入审计信封；glm 角色回声显式归一化记录（raw/canonical/reason/log_hash；18 条归一化，日志哈希确定性）；**`critic_reject` fail-closed（v2+v3）**：适配器只负责**维度 A（派生规则）**，`DEFAULT_CRITIC_REJECT_DERIVATION_RULE=NONE`，critic 记录必须收到显式命名的派生规则（`decision_reject` 或 `flags_too_hard`），否则整臂抛 `CRITIC_POLICY_REQUIRED`；未知字符串抛 `UNKNOWN_RULE`；派生记录标注 `critic_reject_derivation_rule`/`critic_reject_value`/`derived=true`/"legacy schema has no raw critic_reject bit"；非 critic 记录绝不产生派生；**维度 B（selector 消费策略 hard_veto/soft_penalty/score_only）由 `schemas/selector.py` + `selectors/` 负责，两维度互不替代**；
- `tier_c_gate.py`（v3 新增）：真实 canonical Tier-C 路径的双维度 fail-closed 门——`critic_reject_derivation_rule` 与 `critic_selection_policy` 必须**同时**显式冻结且合法，缺一抛 `TierCPolicyError`（`CRITIC_DERIVATION_RULE_REQUIRED` / `CRITIC_SELECTION_POLICY_REQUIRED`）；并校验 cell 模板保持双字段拆分、PENDING、两条独立 blocker；
- `prompt_profile_contract.py`：prompt 注册表与冻结 profile 契约离线验证 + 真实 model pin 对 ROLE_REGISTRY 差异记录；
- `replay.py`：§3 重放纯函数库（脚本与测试共用）。

## 5. §7 R1–R6 自动化测试：312 全绿

`d052/tests/test_real_bundle_reconciliation.py`（29 测试）：R1 salted/unknown 目标被防火墙以具体 CODE 拒绝（18/21 名未知；25/32 候选在边界被拒）；R2 B/C matched-field + 协议不变量；R3 96×2 覆盖 + 192/192 防篡改 + canonical 实例化 + 派生规则敏感性钉死（40 vs 38）+ 归一化日志；R4 重放全锚点 + 决定性；R5 canonical 池 `executed_as_intended=True` vs legacy 拒绝；R6 profile 完整性（7/67 测量、null 保持）。

**v2 修正门（6 项新增）**：`CCV2` 缺 policy 整臂 fail-closed（`CRITIC_POLICY_REQUIRED`）、单条 critic 记录 fail-closed、非 critic 无 policy 可转换且零派生、未知 policy 字符串 `UNKNOWN_RULE`（臂级+记录级）、显式双规则完整出处链（`derived=true`/规则/值/无 raw bit 注记）且计数仍钉死 40/38、重放 overlap=4 与 Jaccard=0.3333 及全部锚点不受适配器改动影响、Henry 无效归档保留（01_d052/README.md 的 invalid/code-only 声明 + d052_data_removed_by_request.txt）。模板新增断言 `training_authorized=false`。

**v3 语义清理门（8 项新增）**：`test_no_ambiguous_critic_policy_pass_label`（`D052_CRITIC_POLICY` ≠ PASS，为 DEPRECATED 兼容字段且指向 replacement_fields）；`test_real_canonical_critic_fields_are_split`（两模板双字段拆分、PENDING、两条独立 blocker、`tier_c_gate.validate_template_critic_fields` ok）；`test_real_canonical_missing_derivation_rule_blocks`（维度 A 缺失 → 适配器 `CRITIC_POLICY_REQUIRED` + Tier-C 门 `CRITIC_DERIVATION_RULE_REQUIRED`，即使维度 B 已冻结）；`test_real_canonical_missing_selection_policy_blocks`（维度 B 缺失 → `CRITIC_SELECTION_POLICY_REQUIRED`，即使维度 A 已冻结；两维度同时冻结合法才返回双字段记录）；`test_synthetic_engineering_pass_does_not_freeze_real_policy`（synthetic PASS 有作用域注记、真实两维度 UNDECIDED、两 DEFAULT=NONE、两 BLOCKED，三份产物标签互镜）；`test_gitignore_only_allows_frozen_outputs`（临时未来文件被忽略、21 个 allowlist 文件可跟踪、临时文件用后即删）；`test_frozen_output_allowlist_matches_git`（allowlist == `git ls-files outputs/` 且 SHA256/size 逐一匹配磁盘）；`test_historical_replay_unchanged`（B=82571538e5299ea9 / C=868a57268d66b90b / pool=1902b71a5d86fa00 / change=4/8 / overlap=4 / Jaccard=0.3333 / 双跑比特一致）。

## 6. §8/§9 三层证据与 cell

`real_bundle_evidence_tiers.json`：
- **Tier A REAL_LEGACY_PHASE25**：真实数据、legacy 机制；MECHANISM_ONLY；不可训练、不可作表现解读；
- **Tier B SYNTHETIC_CANONICAL_FIXTURE**：968768e 工程测试 PASS（1/8）；**不是科学证据**；其 10 个产物原样保留（本轮产物均以 `real_bundle_*` 命名，零覆盖）；
- **Tier C REAL_CANONICAL_POOL = NOT_RUN**：cell 模板 `CELL_PHASE25_REAL_CANONICAL_B/C` 置于 `gpu1_aggregation_siege/phase25_real_canonical_cell_templates/`，状态 `BLOCKED_PENDING_REAL_CANONICAL_JUDGMENTS`，**未注册**任何 CellRegistry，`intended_total_timesteps=0`。

## 6b. v2 合并前修正（D052_PREMERGE_CORRECTION_V2）

1. **归档恢复**：从 `origin/Henry-branch`（a2726e3，本轮 fetch 时网络抖动，采用上一轮 ls-remote 已远端核实的同名引用）路径限定恢复 6 条：`experiments/henry_dicode_student_upgrade/01_d052/`（12 文件）+ `inventory/d052_data_removed_by_request.txt` 共 13 个 A，manifest/inventory 4 个 M 回退为 Henry-branch 版本；恢复内容与 origin/Henry-branch 逐字节一致（diff 0 行）；相对 Henry-branch 的 D 条目归零。旧 D052 仍为 invalid/code-only，**未**作为科学结果恢复任何数据文件。
2. **critic policy fail-closed**：见 §4 与 §8 冻结标签；历史重放不使用适配器（消费原始 critic_penalty），全部锚点不变。

## 6c. v3 语义清理（D052_PREMERGE_SEMANTIC_CLEANUP_V3）

1. **标签作用域消歧**：旧 `D052_CRITIC_POLICY=PASS` 同时混用了"合成品工程测试通过"与"真实 canonical 科学策略"两层含义。处置（option B 兼容保留）：该字段改为 `DEPRECATED_AMBIGUOUS_DO_NOT_USE`（`deprecated=true` + `replacement_fields`），任何自动门不再消费；新增分层标签 `D052_SYNTHETIC_CRITIC_SELECTOR_ENGINEERING=PASS`（仅工程层）与真实层 `REAL_CANONICAL_CRITIC_REJECT_DERIVATION_RULE=UNDECIDED` / `REAL_CANONICAL_CRITIC_SELECTION_POLICY=UNDECIDED` / 两个 `DEFAULT_*=NONE` / 两个 `*_WITHOUT_*=BLOCKED`。
2. **双维度拆分**：cell 模板 B/C 的歧义 `critic_policy` 单字段拆为 `critic_reject_derivation_rule`（A）与 `critic_selection_policy`（B），均 `PENDING_DIRECTOR_DECISION`，blockers 分为两条独立条目，并新增 `execution_certificate_policy_record`（真实证书必须分别记录两维度状态，缺一 fail closed）；适配器参数/常量/派生键统一为维度 A 命名；新增 `tier_c_gate.py` 双维度门。
3. **.gitignore 收紧**：真实判定产物目录由整目录否定豁免改为"父目录放行 + 目录默认忽略 + 21 个已跟踪文件逐条 allowlist"，未来新文件默认忽略；allowlist 机器清单 `reports/phase25/frozen_output_allowlist.json`（path/sha256/size/purpose/无密/无 checkpoint/批准入库），并做密钥与二进制扫描。
4. **不变量**：synthetic 工程 PASS 不升格为真实冻结策略；历史重放锚点（B=82571538e5299ea9 / C=868a57268d66b90b / pool=1902b71a5d86fa00 / 4/8 / Jaccard=0.3333）不受任何改动影响；两维度均未冻结、均未擅自选择候选。

## 7. 需总监裁定（不在本轮自动解决）

1. critic **两个维度分别冻结**（v3 拆分，互不替代）：维度 A 派生规则 `decision_reject`（B+C=40）vs `flags_too_hard`（B+C=38）；维度 B selector 消费策略 `hard_veto` vs `soft_penalty` vs `score_only`——适配器（A）与 `tier_c_gate`（A+B）均已 fail-closed（`REAL_CANONICAL_CRITIC_REJECT_DERIVATION_RULE=UNDECIDED`、`REAL_CANONICAL_CRITIC_SELECTION_POLICY=UNDECIDED`、两个 `DEFAULT_*=NONE`、缺任一即 `CRITIC_POLICY_REQUIRED` / `TierCPolicyError`），所有候选均**不是**已批准的 canonical 科学定义；冻结前 `REAL_CANONICAL_CONVERSION_WITHOUT_CRITIC_RULE=BLOCKED` 且 `REAL_CANONICAL_SELECTION_WITHOUT_CRITIC_POLICY=BLOCKED`；
2. ROLE_REGISTRY model pin（qwen-turbo/deepseek-chat/glm-4.5-air）vs 真实包型号（qwen-flash-2025-07-28/deepseek-v4-pro/glm-4-flash）——未来 Tier C 调 LLM 前必须决；
3. canonical 池 `task_params` 必填 `melee_spawn_multiplier` 取值策略。

## 8. §13 最终冻结标签

```
REAL_PHASE25_BUNDLE_REPLAY                      = PASS
REAL_PHASE25_B_SELECTION_HASH                   = 82571538e5299ea9
REAL_PHASE25_C_SELECTION_HASH                   = 868a57268d66b90b
REAL_PHASE25_SELECTED_CHANGE                    = 4/8
REAL_PHASE25_MATCHED_PROTOCOL                   = PASS
MODELER_MATCHED_SELECTION_EFFECT                = CONFIRMED
LEGACY_MATCHED_SELECTED_SET_CHANGE              = 4/8
LEGACY_MATCHED_JACCARD                          = 0.333
MODELER_LEARNING_VALUE                          = UNTESTED
LEGACY_PHASE25_SELECTION_EVIDENCE               = MECHANISM_ONLY
LEGACY_PHASE25_SELECTED8_TRAINING_READY         = false
LEGACY_PHASE25_PERFORMANCE_INTERPRETATION_ALLOWED = false
CANONICAL_SYNTHETIC_FIXTURE_ENGINEERING_TEST    = PASS
CANONICAL_SYNTHETIC_MODELER_SELECTION_CHANGE    = 1/8
CANONICAL_SYNTHETIC_RESULT_IS_SCIENTIFIC_EVIDENCE = false
REAL_CANONICAL_POOL_EXPERIMENT                  = NOT_RUN
D052_CRITIC_POLICY                              = DEPRECATED_AMBIGUOUS_DO_NOT_USE
D052_SYNTHETIC_CRITIC_SELECTOR_ENGINEERING      = PASS
REAL_CANONICAL_CRITIC_REJECT_DERIVATION_RULE    = UNDECIDED
REAL_CANONICAL_CRITIC_SELECTION_POLICY          = UNDECIDED
DEFAULT_CRITIC_REJECT_DERIVATION_RULE           = NONE
DEFAULT_CRITIC_SELECTION_POLICY                 = NONE
REAL_CANONICAL_CONVERSION_WITHOUT_CRITIC_RULE   = BLOCKED
REAL_CANONICAL_SELECTION_WITHOUT_CRITIC_POLICY  = BLOCKED
CC3_ARCHIVE_PRESERVATION                        = PASS
CC3_CRITIC_POLICY_FAIL_CLOSED                   = PASS
CC3_CRITIC_LABEL_SCOPE_CLEANUP                  = PASS
CC3_CRITIC_DIMENSIONS_SPLIT                     = PASS
CC3_GITIGNORE_ALLOWLIST                         = PASS
CC3_FROZEN_OUTPUT_SECRET_SCAN                   = PASS
D052_TRAINING_AUTHORIZED                        = false
NEW_TRAINING_RUNS                               = 0
NEW_LLM_CALLS                                   = 0
```

## 9. 交付物清单

- `gpu1_aggregation_siege/d052/reconciliation/`（real_bundle / judgment_adapter / prompt_profile_contract / replay）
- `gpu1_aggregation_siege/d052/tests/test_real_bundle_reconciliation.py`
- `gpu1_aggregation_siege/reports/phase25/real_bundle_{replay_result.json,field_mapping_completed.json,field_mapping_completed.md,evidence_tiers.json,reconciliation_final.md,SHA256SUMS}`
- `gpu1_aggregation_siege/reports/phase25_{local_search_verdict.json,local_fs_search.txt,unreachable_git_objects.txt,real_bundle_acquisition.json}`
- `gpu1_aggregation_siege/phase25_real_canonical_cell_templates/CELL_PHASE25_REAL_CANONICAL_{B,C}.json`
- `orchestration/experiments/d052_modeler_shadow_v1/`（真实包 14 件 + outputs/ 原始记录 + replay_inputs/ + reconciliation/）

**约束合规**：不训练、不调新 LLM、不 push、不改包原件、不覆盖 968768e 合成品、不以 synthetic 顶替真实、不从总结重生成数据、任一锚点不匹配即停（未触发）、critic 两个维度均无隐式默认（fail closed）、synthetic 工程 PASS 不升格为真实冻结策略、Henry 无效归档保留且不升格为科学证据、真实产物目录仅 21 个 allowlist 文件可入库且通过密钥/二进制扫描。

**本轮到此停止，等待总监复核。**
