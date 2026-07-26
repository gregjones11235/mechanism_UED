# D052 Phase 2.5 真实迁移包 ↔ canonical_v2 字段映射对账（完成版）

- 任务：D052_PHASE25_REAL_BUNDLE_RECONCILIATION §4（经 D052_PREMERGE_CORRECTION_V2 修订：critic policy fail-closed；经 D052_PREMERGE_SEMANTIC_CLEANUP_V3 修订：critic 拆分为两个独立维度——A 派生规则 / B 选择消费策略）
- 日期：2026-07-26 ｜ 分支：henry/d052-canonical-refactor @968768e + 5f9ab74 + 本轮修正
- 机器数据：`gpu1_aggregation_siege/reports/phase25/real_bundle_field_mapping_completed.json`
- canonical 侧来源：本 worktree `gpu1_aggregation_siege/d052/**`（服务器当时不可见，现已在本地）
- legacy 侧来源：真实包 `orchestration/experiments/d052_modeler_shadow_v1/artifacts/d052_phase25_canonical_migration/`（13/13 SHA 校验通过，原件只读）
- 方法：**实证**——用真实包数据对真实 pydantic schema 逐个实例化，失败记录真实错误 CODE；不猜测、不补数、不静默兼容（NO_SILENT_SCHEMA_COERCION）。

状态码：`LOSSLESS`=恒等无损；`LOSSY`=有损/派生；`UNSUPPORTED`=canonical 无对应字段（审计信封保留）；`BLOCKED_REQUIRED_FIELD`=canonical 必填而 legacy 缺失；`MECHANISM_MISMATCH`=机制层不同，不可表达。

---

## A. Candidate / TaskParams / CandidatePool

| source_field (legacy pool) | canonical_field | conversion | status | required_default | provenance | validation_result |
|---|---|---|---|---|---|---|
| `task_id` (d052_r3_0000…) | `Candidate.task_id` | 恒等 | LOSSLESS | — | pool frozen round_4 | PASS（格式可用） |
| `chash` (16字符) | `Candidate.chash` (64字符) | 必须按 `compute_candidate_chash` 重算 | BLOCKED_REQUIRED_FIELD | 无默认（内容哈希不可捏造） | 旧值存审计信封 | 32/32 INVALID_HASH |
| `chash` → `legacy_short_id` 桥接？ | `Candidate.legacy_short_id` | `sha256(f"{id}:{sorted(names)}")[:16]` | UNSUPPORTED | — | — | **0/32 不可复现**（legacy chash 为不透明 salted 值，不作桥接断言） |
| `target_achievements` (salted 占位名) | `target_achievements: List[AchievementRef]` | 名称须经 REGISTRY 解析 | BLOCKED | unknown_target_policy=error | salted_hash_audit.json | 25/32 含未知名；21 个不同名中仅 3 个可解析（collect_wood/defeat_zombie/place_table） |
| `difficulty_tier` | — | — | UNSUPPORTED | — | 审计信封 | canonical 无难度本体字段 |
| `task_params.passive_spawn_multiplier` | `TaskParams.passive_spawn_multiplier` | 数值恒等(>0,finite) | LOSSLESS | — | — | PASS |
| `task_params.mob_health_multiplier` | `TaskParams.mob_health_multiplier` | 数值恒等 | LOSSLESS | — | — | PASS |
| `task_params.mob_damage_multiplier` | `TaskParams.mob_damage_multiplier` | 数值恒等 | LOSSLESS | — | — | PASS |
| （legacy 无） | `TaskParams.melee_spawn_multiplier` | — | BLOCKED_REQUIRED_FIELD | **无诚实默认**（legacy 从未设定） | — | 32/32 缺必填 |
| `description`/`label`/`short_reason`/`source`/`snapshot_hash`/`_prov` | — | extra="forbid" | UNSUPPORTED | — | 审计信封（_prov 为 LLM 生成出处，保留） | 不进 canonical Candidate |
| pool `hash` (16字符=`1902b71a5d86fa00`) | `CandidatePool.pool_hash` (64字符=sha256(有序 chash 列)) | 哈希体制不同 | MECHANISM_MISMATCH | — | 历史锚点（重放已验证） | 不相互改写 |

**结论**：32/32 候选无法实例化为 canonical Candidate；与包内 salted_hash_audit 一致——legacy 目标语义无效，训练路径 INVALID。

## B. RoleJudgment（96×2 臂）

| source_field (bundle jsonl) | canonical_field | conversion | status | provenance / validation |
|---|---|---|---|---|
| `role` (tutor/critic/explorer) | `RoleJudgment.role` (ScoringRole) | 恒等 | LOSSLESS | PASS |
| `task_id`（真实 id） | `candidate_id` | 恒等（anon_id C001..C032 留审计信封） | LOSSLESS | PASS |
| `raw_scores` | `scores: Dict[str,float]` | 恒等（headline 键齐全：progression_score / critic_penalty / novelty_score；辅助键 learnability_score、diversity_score 保留） | LOSSLESS | 192/192 headline 校验 PASS |
| `short_reason` | `rationale` | 恒等（永不参与打分） | LOSSLESS | PASS |
| `provider` | `provider` | 恒等 | LOSSLESS | PASS |
| `model_returned` | `exact_model_id` | 恒等（model_requested 留信封） | LOSSLESS | PASS |
| （registry）`d052_phase25_v1` | `prompt_version` | 取自 prompt_registry | LOSSLESS | PASS |
| `decision` + `flags`（仅 critic） | `critic_reject: bool` | **候选派生规则，无隐式默认**（legacy 无此 raw bit） | LOSSY / **FAIL_CLOSED** | 两个候选：`decision_reject`（B+C True=40）与 `flags_too_hard`（B+C True=38）。**REAL_CANONICAL_CRITIC_REJECT_DERIVATION_RULE=UNDECIDED（维度 A）**：适配器必须收到显式命名的派生规则，否则对 critic 记录整体抛 `CRITIC_POLICY_REQUIRED`；未知字符串抛 `UNKNOWN_RULE`。两个候选均不是已批准的 canonical 科学定义；未冻结前不得生成正式 canonical judgments。此维度与维度 B（selector 如何消费 critic 信号，`REAL_CANONICAL_CRITIC_SELECTION_POLICY=UNDECIDED`）相互独立、不得互相替代。历史重放不使用任何派生（直接用原始 critic_penalty），任何规则选择都不改变历史锚点 |
| `decision`（tutor/explorer） | — | — | UNSUPPORTED | 审计信封（selector 不消费） |
| `flags`/`anon_id`/`arm`/`attempts`/`parse_status`/`source_file`/`model_requested` | — | — | UNSUPPORTED | 审计信封 |
| `judgment_hash_sha256`（对**原始** judgment 的防篡改哈希） | — | extra="forbid" 不允许内嵌 | UNSUPPORTED | 信封保留；R3 防篡改校验照旧对原始记录做 |
| `role_label_in_raw` vs `role_label_normalized_to` | — | glm 角色回声归一化 | LOSSY→审计 | 18 条原始标签≠归一化标签；适配器记录 raw_role_label/canonical_role_label/normalization_reason/normalization_log_hash（§6） |

**结论**：192/192 在**显式指定**的派生规则下映射为合法 RoleJudgment；headline 分数无损。派生记录在 `derived` 中标注 `critic_reject_derivation_rule` / `critic_reject_value` / `derived=true` / "legacy schema has no raw critic_reject bit"。非 critic 记录不产生任何 critic_reject 派生。

## C. NormalizedRoleJudgment / NormalizedRoleScores — `MECHANISM_MISMATCH`

- legacy：robust median/IQR（clip 3.0, ε1e-8）+ 权重软 Copeland（0.34/0.33/0.33/0.01/0.01）+ temperature 1.0；
- canonical：`NormalizedRoleScores` 钉死 `rank_percentile_v1`（[0,1] 秩百分位，确定性 tie_group）。
- **结论**：legacy 归一化信号不可表达为 canonical NormalizedRoleScores。这正是 Tier A（legacy 机制证据）与 Tier C（未来 canonical 池实验）的分界，不做强制。

## D. SelectorConfig

| source_field (selector_config.json) | canonical_field | status | 说明 |
|---|---|---|---|
| 算法=soft copeland | `selector=SOFT_COPELAND` | LOSSLESS | 枚举存在 |
| critic 以 0.01/0.01 权重入分 | 维度 B：`critic_policy=SOFT_PENALTY`（selector 消费策略，`d052/schemas/selector.py` CriticPolicy） | LOSSY | 最接近的 canonical 策略；注意 `REAL_CANONICAL_CRITIC_SELECTION_POLICY=UNDECIDED`——legacy 取值只是映射记录，不是已冻结的真实 canonical 策略 |
| k=8 | `k=8` | LOSSLESS | — |
| `rng_seed=null`（无种子确定性） | `seed: int`（必填） | BLOCKED_REQUIRED_FIELD | 无诚实默认；实证：约定 seed=0 可实例化，但该约定只用于 canonical 侧结构表达，**不改变历史锚点** |
| roles tutor/critic/explorer | `roles` | LOSSLESS | — |
| weights/temperature/clip/epsilon | — | UNSUPPORTED | canonical 选择器不用该权重向量；留审计信封 |
| selection_hash_fn / pool_hash_fn（16字符） | canonical 64字符体制 | MECHANISM_MISMATCH | 互不改写 |
| selector_source_sha256 (27492e8a…) | — | 审计 | 已验证；相关函数与工作区 590fcef4… AST 字节一致 |

## E. SelectionResult — `HASH_REGIME_INCOMPATIBLE`

legacy `B_selection_hash=82571538e5299ea9` / `C_selection_hash=868a57268d66b90b`（16字符，`sha256(json.dumps(sorted(ids)))[:16]`）是**历史锚点**，重放逐位复现（PASS）。canonical `SelectionResult.selection_hash` 为 64 字符、payload 含 (selector, critic_policy, k, seed, ids)——此处 `critic_policy` 是维度 B（selector 消费策略）的基线 schema 命名，与维度 A 派生规则无关。两者不互相改写；canonical SelectionResult 只能由 canonical 选择器运行产生（Tier C，NOT_RUN）。

## F. ExecutionMappingCertificate — `BLOCKED`

- 7/32 候选目标名全部可解析、25/32 含未知名 → `target_is_canonical` 门对 25/32 直接失败；
- 即使名称可解析，legacy launcher 以 salted `hash()` 取模映射到成就槽（salted_hash_audit.json）→ `no_silent_fallback` 门失败 → 全池 `executed_as_intended=False`。
- 与包结论一致：训练路径 INVALID。

## G. CellSpec — `BLOCKED`

legacy cell `soft_copeland_x_original/seed0_1784462982/round_4` 不可表达为 canonical CellSpec（pool_hash/selection_hash 16字符、seed=null、protocol=legacy）。legacy cell = Tier A 证据。新模板 `CELL_PHASE25_REAL_CANONICAL_B/C` → `BLOCKED_PENDING_REAL_CANONICAL_JUDGMENTS`（§9）。

## H. Prompt registry / 角色协议

| legacy（真实包） | canonical（d052.counterfactual.prompts / roles.protocol） | 对账 |
|---|---|---|
| prompt_version `d052_phase25_v1` | `ROLE_PROMPT_VERSION=canonical_v2.roles.v1` | 版本名不同，均保留 |
| 6×`prompt_hash_sha256`（对 full_text） | `PromptSet.prompt_set_hash`（对角色钉扎+conditioning block） | 哈希对象不同；双留不改写 |
| tutor `qw/qwen-flash-2025-07-28` | ROLE_REGISTRY: dashscope/**qwen-turbo** | **MODEL_PIN_MISMATCH**（provider 族一致） |
| critic `ds/deepseek-v4-pro` | deepseek/**deepseek-chat** | 同上 |
| explorer `glm-4-flash` | zhipu/**glm-4.5-air** | 同上 |
| output schemas（per role 文本） | `ROLE_OUTPUT_SCHEMA=role_judgment_v2` | 结构对应 |

真实 model id 经 `RoleJudgment` 的 Optional 出处字段**无损保留**；ROLE_REGISTRY 钉扎差异只影响未来 LLM 调用（本轮不调用）→ flagged 待总监裁定（任何 Tier C 运行前必须先决）。

## I. StudentProfile / Modeler

| source_field (student_profile.json) | canonical_field | status | validation |
|---|---|---|---|
| `machine_facts.per_achievement_completion[].completion_rate` | `StudentProfile.per_achievement_sr` | LOSSY（语义注记：64 episodes 完成率作 SR 代理；evidence_source 保留诚实出处） | **PASS**：构建成功，measured=7/67，mastered=1（WAKE_UP 0.9844），proficient=1，overall=0.030551；其余 60 项按 canonical 保守默认 SR=0.0 |
| `machine_facts.episode_level/skill_chain/dominant_breakpoints/evidence_boundaries` | — | UNSUPPORTED | 审计信封（供 prompt 渲染，非模型字段） |
| `llm_interpretation.curriculum_priorities` | `ModelerJudgment.guidance` | LOSSY | 自由文本 |
| `llm_interpretation.dominant_breakpoints`（4 项全部可解析） | `siege_foci` | LOSSLESS（排序去重由 schema 做） | PASS |
| `skills[].status`（逐成就） | `student_state`（会话级枚举，必填） | BLOCKED_REQUIRED_FIELD | 禁止派生 |
| （无） | `recommendation`（DEPTH/BREADTH/CONSOLIDATE，必填） | BLOCKED_REQUIRED_FIELD | 禁止派生 |
| （无） | `evidence_check` | 需派生规则 | flagged |
| `skills[].best_sr/recent_delta=null` | — | 保持 null | INSUFFICIENT_EVIDENCE，**绝不补数** |
| `profile_hash_sha256=223defdf…` | — | 已复现：`sha256(canon_json(llm_interpretation))` | PASS |

**结论**：冻结解读**逐字节**保留（即 C 臂追加的原文）；不声称 legacy 包拥有严格 canonical ModelerJudgment。

## J. Provenance / 哈希

- canonical 体制：64 字符小写 sha256，canonical JSON（sort_keys, `(",",":")`, ensure_ascii=False）；`protocol_version="canonical_v2"` 钉死；extra 字段禁止。
- legacy 审计保留：chash/snapshot_hash/pool.hash/selection_hash（16）、judgment_hash_sha256/profile_hash/calculator_source/machine_facts.source/selector_source/wrapper_source/prompt_hash×6（64）。
- canonical 冻结配置快照已写入 JSON（67 成就、multi-hot 67、obs 8335、shared_frozen、rank_percentile_v1、三策略 error）。

---

## 冻结标签（D052_PREMERGE_CORRECTION_V2 + D052_PREMERGE_SEMANTIC_CLEANUP_V3）

```
D052_SYNTHETIC_CRITIC_SELECTOR_ENGINEERING        = PASS   (仅合成品工程测试，不冻结真实策略)
REAL_CANONICAL_CRITIC_REJECT_DERIVATION_RULE      = UNDECIDED   (维度 A：派生规则)
REAL_CANONICAL_CRITIC_SELECTION_POLICY            = UNDECIDED   (维度 B：selector 消费策略)
DEFAULT_CRITIC_REJECT_DERIVATION_RULE             = NONE
DEFAULT_CRITIC_SELECTION_POLICY                   = NONE
REAL_CANONICAL_CONVERSION_WITHOUT_CRITIC_RULE     = BLOCKED
REAL_CANONICAL_SELECTION_WITHOUT_CRITIC_POLICY    = BLOCKED
```

- `decision_reject` 与 `flags_too_hard` 都只是维度 A 的**候选**派生规则，均**不是** legacy 原始字段，也**不是**已批准的 canonical scientific definition；`hard_veto` / `soft_penalty` / `score_only` 是维度 B 的候选消费策略，同样未冻结；
- 两个维度**相互独立、不得互相替代**：适配器只处理 A（`judgment_adapter.py`），selector/protocol 只处理 B（`schemas/selector.py` + `selectors/`），Tier-C 门（`tier_c_gate.py`）要求两者同时显式冻结，缺一即 fail closed；
- synthetic 工程 PASS **不得**升格为冻结的真实策略；
- historical legacy replay 不依赖这两个维度（消费原始 `critic_penalty`），任何冻结决定都不改变历史锚点；
- future canonical protocol 必须**分别显式冻结**维度 A 一个规则、维度 B 一个策略；未冻结前不得生成正式 canonical judgments，不得授权 D052 training。

## 需总监裁定的开放项

1. critic 两个维度分别冻结：维度 A 派生规则 `decision_reject`（B+C=40）还是 `flags_too_hard`（B+C=38）？维度 B 消费策略 `hard_veto` / `soft_penalty` / `score_only`？——适配器（A）与 Tier-C 门（A+B）均已 fail-closed（无默认，缺任一即 BLOCKED / `CRITIC_POLICY_REQUIRED`），等待总监**独立**冻结两者；冻结前 Tier C 转换与选择整体 BLOCKED。
2. ROLE_REGISTRY model pin 与真实包 model id 不一致：未来 Tier C 运行用哪套钉扎？
3. legacy `task_params` 缺 `melee_spawn_multiplier`：未来 canonical 池生成时取值策略（属于 Tier C 生成器，不属于本对账）。

约束合规：未改包原件、无 LLM、无训练、无静默兼容、无 synthetic 顶替、无 push、两个 critic 维度均无隐式默认（fail closed）。
