# D052 Phase 2.5 真实迁移包对账 —— 最终报告

- 任务：D052_PHASE25_REAL_BUNDLE_RECONCILIATION（§1–§13）
- 日期：2026-07-26 ｜ 分支：`henry/d052-canonical-refactor` @ 968768e（本轮单次本地 commit，**未 push**）
- 测试：**298 passed**（283 基线防火墙测试全部保留 + 15 项新增对账门，§10 满足）
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
- `judgment_adapter.py`：原始记录逐字入审计信封；glm 角色回声显式归一化记录（raw/canonical/reason/log_hash；18 条归一化，日志哈希确定性）；`critic_reject` 派生显式标注（默认 `decision=='reject'`，可选 `flags.too_hard`）；
- `prompt_profile_contract.py`：prompt 注册表与冻结 profile 契约离线验证 + 真实 model pin 对 ROLE_REGISTRY 差异记录；
- `replay.py`：§3 重放纯函数库（脚本与测试共用）。

## 5. §7 R1–R6 自动化测试：298 全绿

`d052/tests/test_real_bundle_reconciliation.py`（15 测试）：R1 salted/unknown 目标被防火墙以具体 CODE 拒绝（18/21 名未知；25/32 候选在边界被拒）；R2 B/C matched-field + 协议不变量；R3 96×2 覆盖 + 192/192 防篡改 + canonical 实例化 + 派生规则敏感性钉死（40 vs 38）+ 归一化日志；R4 重放全锚点 + 决定性；R5 canonical 池 `executed_as_intended=True` vs legacy 拒绝；R6 profile 完整性（7/67 测量、null 保持）。

## 6. §8/§9 三层证据与 cell

`real_bundle_evidence_tiers.json`：
- **Tier A REAL_LEGACY_PHASE25**：真实数据、legacy 机制；MECHANISM_ONLY；不可训练、不可作表现解读；
- **Tier B SYNTHETIC_CANONICAL_FIXTURE**：968768e 工程测试 PASS（1/8）；**不是科学证据**；其 10 个产物原样保留（本轮产物均以 `real_bundle_*` 命名，零覆盖）；
- **Tier C REAL_CANONICAL_POOL = NOT_RUN**：cell 模板 `CELL_PHASE25_REAL_CANONICAL_B/C` 置于 `gpu1_aggregation_siege/phase25_real_canonical_cell_templates/`，状态 `BLOCKED_PENDING_REAL_CANONICAL_JUDGMENTS`，**未注册**任何 CellRegistry，`intended_total_timesteps=0`。

## 7. 需总监裁定（不在本轮自动解决）

1. `critic_reject` 派生规则：`decision=='reject'`（40/128）vs `flags.too_hard`（38/128）；
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
```

## 9. 交付物清单

- `gpu1_aggregation_siege/d052/reconciliation/`（real_bundle / judgment_adapter / prompt_profile_contract / replay）
- `gpu1_aggregation_siege/d052/tests/test_real_bundle_reconciliation.py`
- `gpu1_aggregation_siege/reports/phase25/real_bundle_{replay_result.json,field_mapping_completed.json,field_mapping_completed.md,evidence_tiers.json,reconciliation_final.md,SHA256SUMS}`
- `gpu1_aggregation_siege/reports/phase25_{local_search_verdict.json,local_fs_search.txt,unreachable_git_objects.txt,real_bundle_acquisition.json}`
- `gpu1_aggregation_siege/phase25_real_canonical_cell_templates/CELL_PHASE25_REAL_CANONICAL_{B,C}.json`
- `orchestration/experiments/d052_modeler_shadow_v1/`（真实包 14 件 + outputs/ 原始记录 + replay_inputs/ + reconciliation/）

**约束合规**：不训练、不调新 LLM、不 push、不改包原件、不覆盖 968768e 合成品、不以 synthetic 顶替真实、不从总结重生成数据、任一锚点不匹配即停（未触发）。

**本轮到此停止，等待总监复核。**
