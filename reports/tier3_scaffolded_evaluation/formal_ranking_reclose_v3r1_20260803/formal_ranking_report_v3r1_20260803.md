# Tier3 分段支架评测 — V3R1 正式排名重收口报告（FRONT-first + top-tie-only）

- **任务**：`CC4_RECLOSE_FORMAL_GLOBAL_RANKING_WITH_FRONT_FIRST_AND_TOP_TIE_ONLY`
- **执行者**：CC4（统一评测 / world-set 证据 / Tier3 分段支架评测环境）
- **日期**：2026-08-03（UTC）
- **仓库 / 分支**：`gregjones11235/mechanism_UED` / `henry/tier3-scaffolded-evaluation`
- **基线提交（reclose 起点，冻结）**：`3c36ae620d450314a0e034bc80e73210f4784296`
- **reclose 提交**：包含本报告与新产物的单一提交，提交信息
  `fix(tier3): reclose ranking with front-first top-tie policy`
- **结论**：`FORMAL_RANKING_AUTHORIZED_V3R1=true`，`RANKING_STATUS=ORDERED_WITH_LOWER_TIES`，
  `FORMAL_WINNER=PERSISTENT_RMT16_ORIGINAL_VTRACE_98304`（唯一第一名；低名次并列披露、不阻塞）。

---

## 1. 任务性质与边界（§前言/§十三）

本次收口 **不是重新评测**：

- 未重跑任何 FULL/FRONT/BACK rollout（`ENVIRONMENT_RERUNS=0`）；
- 未重训任何 Student（`STUDENTS_RETRAINED=0`）；
- 未修改任何 checkpoint / state bank / seed / horizon / episode 数 / greedy 策略
  （`CHECKPOINTS_MODIFIED=false`）；
- 未删除候选、未做任何候选级豁免（`CANDIDATE_EXCEPTIONS_USED=0`）；
- 全部工作 = 从 **已冻结、不可变的 V3 episode records / metrics / 证书** 出发，
  纯离线地（a）冻结新的排名维度次序，（b）修复"低名次并列取消唯一第一名"的旧语义，
  （c）消除 BACK/FULL 主事件在 secondary_events 中的重复，（d）生成 V3R1 新收口产物。

旧 V3 收口（`INCONCLUSIVE_FULL_TIE`、winner=null）**在其自身协议下是正确结果**，
不是错误或伪造；V3R1 是总控的正式澄清与重收口。旧 V3/V2 证据 **零覆盖、零改写**。

## 2. 协议变更（§一/§二/§六）

### 2.1 排名维度次序（FRONT-first）

| 字段 | 旧 V3 次序（FULL-first，冻结 V2DT） | 新 V3R1 次序（FRONT-first） |
|---|---|---|
| 1 | `full success_count` | `front_l2 transition_count` |
| 2 | `front_l2 transition_count` | `front_l2 mean graph_distance_progress` |
| 3 | `front_l2 mean graph_distance_progress` | `full success_count` |
| 4 | `back_l2 defeat_count` | `back_l2 defeat_count` |

- `RANKING_PRIMARY_ORDER=FRONT_TRANSITION_FIRST`；FULL success_count **不是**第一字段。
- 词典序全降序；容差 `1e-12`；禁止 reward/长度/规模/参数量/速度/checkpoint 年龄/
  复杂度等任何其它 tie-break；`candidate_id` 只用于输出的确定性序列化，
  **绝不**作为科学 tie-break。
- `RANKING_PROTOCOL=TIER3_FRONT_FIRST_LEXICOGRAPHIC_V1`，
  `FORMAL_RANKING_PROTOCOL=V3R1_FRONT_FIRST_TOP_TIE_ONLY`。

### 2.2 并列语义（ONLY_TOP_TIE_BLOCKS_WINNER=true）

- 旧语义：任意位置出现 ≥2 全元组并列 → 整个排名作废（`INCONCLUSIVE_FULL_TIE`，winner=null）。
  这正是旧 V3 收口的结局：SLOWGRU_PERSISTENT 虽唯一第一，但第 3 位
  BASE_GTRXL≡RESET128 的并列使全体 rank 置空、winner 置空。
- 新语义：**只有第一名等价组内部的并列**才阻塞 winner：
  - 第一名等价组 size=1 → `formal_winner` = 该候选（即使更低名次存在并列）；
  - 第一名等价组 size>1 → winner=null + `INCONCLUSIVE_TOP_TIE`；
  - 更低名次并列 → 如实披露为
    `{"tie_group_rank":<竞赛排名>,"candidate_ids":[...],"tie_scope":"LOWER_POSITION","winner_blocking":false}`，
    组内候选 `student_rank=null` 但记录 `tie_group_rank`；绝不以 candidate_id 杜撰组内次序。

### 2.3 secondary event 去重（NEG20_V3R1_PRIMARY_NONREDUNDANT_SECONDARY）

- FRONT：primary=FRONT_TRANSITION_SUCCESS/FRONT_NO_TRANSITION；defeat_kobold=true 时
  DEFEAT_KOBOLD **保留**为 secondary（FRONT 主谓词只是 floor transition）；
- BACK：defeat_kobold=true → primary=BACK_DEFEAT_KOBOLD_SUCCESS，secondary_events
  **不得**再含 DEFEAT_KOBOLD；
- FULL：defeat_kobold=true → primary=FULL_DEFEAT_KOBOLD_SUCCESS，同理去重；
- 其它合法事件（PLAYER_DIED、KOBOLD_ENGAGED、CORRIDOR_EXIT_REACHED、TIMED_OUT）不变。
- 全部为对现有 episode records 的 **离线重分类**，未重跑环境。
- 主分类（primary_outcome/taxonomy_status/composite/frozen_label）由冻结 V3 分类器
  原样透传并在包装器内逐项复断言 → 四个冻结主指标 **构造级不变**。

## 3. 输入证据与不可变审计（§三/§四）

重收口唯一输入 = 旧 V3 证据目录
`reports/tier3_scaffolded_evaluation/formal_evaluation_evidence_v3_20260801/`（79 文件）：

- 7 × `episode_records.jsonl`、7 × `evaluation_certificate_v3.json`、
  7 × 3 × `evaluation_result_v3.{full,front_l2,back_l2}.json`、
  7 × `READY_FORMAL_V3.json`、7 × `SHA256SUMS_FORMAL_V3`、7 × `provenance_v3.json`；
- `FORMAL_RANKING_SUMMARY_V3.json`（sha `dab522cf7bcc43ed74f0bc1e9cab20c01c98d972d7edceb2717f9dc18445b659`）；
- `FORMAL_EVALUATION_GATE_V3.json`（sha `c529ebf3ddbf37085b85b0a79018d9cc06ce5a096dc744d18a97a3e0c8b72528`）；
- `COMMON_EVALUATOR_V3_READY.json`、`CROSS_GPU_DETERMINISM_PREFLIGHT_V3.json`、
  `V3_REPAIR_AUTHORIZATION.json`、`formal_eval_logs/`、`common_v2/metric_schema.json`。

**开始前 SHA 快照**（`tmp_v3r1_baseline/old_v3.sha` 79 条、`old_v2.sha` 87 条，
保持 untracked），收口后复验：

| 审计项 | 结果 |
|---|---|
| OLD_V3_FILES_MODIFIED | **0** |
| OLD_V3_HASH_DRIFT | **0** |
| V2_FILES_MODIFIED | **0** |
| 旧目录新增/缺失文件 | 0 / 0（V3 与 V2 均零） |

另逐候选重哈希 7 份 `SHA256SUMS_FORMAL_V3`（各 6 文件）全部通过；summary/gate
sidecar 与冻结钉一致。详见 `OLD_V3_IMMUTABILITY_REPORT.json`。

## 4. 主指标恒等（§七）— V3_PRIMARY_METRICS == V3R1_PRIMARY_METRICS

对 7 个候选逐一从冻结 result JSON 重新抽取四个主指标，并与旧 V3 summary 的
rule_tuple 精确相等比对（任何漂移立即 fail closed、不得发布）：

| 候选 | front transition_count | front mean progress | full success_count | back defeat_count | 恒等 |
|---|---|---|---|---|---|
| PERSISTENT_RMT16_ORIGINAL_VTRACE_98304 | 3 | 0.5905970705064548 | 9 | 7 | EXACT |
| SLOWGRU_PERSISTENT_CANONICAL_98304 | 2 | 0.5752855014895730 | 17 | 6 | EXACT |
| BASE_GTRXL_ORIGINAL_VTRACE_98304 | 2 | 0.5650157181747473 | 14 | 8 | EXACT |
| RESET128_RMT16_ORIGINAL_VTRACE_98304 | 2 | 0.5650157181747473 | 14 | 8 | EXACT |
| SLOWGRU_RESET128_CANONICAL_98304 | 2 | 0.5236034412438056 | 17 | 7 | EXACT |
| CONTROL_CONTINUOUS_98304 | 0 | 0.4196479859579006 | 0 | 7 | EXACT |
| BASELINE_TEACHER_CKPT17500（teacher，仅参考） | 2 | 0.5805684102905279 | 19 | 7 | EXACT |

`PRIMARY_METRIC_PARITY=true`。BACK `diagnostics.survival.defeat_count` 与
primary successes 逐候选一致；valid_starts=64/8/8、episode 数=64/8/8 全部核对。

## 5. secondary 去重结果（§六）

共 **140** 条 BACK/FULL defeat 记录移除重复的 DEFEAT_KOBOLD（FULL 90 条 + BACK 50 条）；
FRONT 的 secondary 逐字未动（含 transition∧defeat 复合记录中的 DEFEAT_KOBOLD，合法保留）。
逐候选逐场景的 V3 旧计数、V3R1 新计数、primary_outcome_counts 全部记录于
`RANKING_RECOMPUTATION_AUDIT_V3R1.json`；且复算的 V3 计数与旧 summary 完全一致
（证明重读的就是旧收口发布的同一批记录）。

`SECONDARY_EVENT_PRIMARY_DUPLICATION_REMOVED=true`。

## 6. V3R1 排名结果（§八/§十）

纯函数 `rank_students_v3r1(entries)`（冻结比较元组
`(front_transition_count, front_progress, full_success_count, back_defeat_count)`，
全降序，tol=1e-12）对 6 个 Student 重排（teacher 不入学生排名）：

| 竞赛排名 | 候选 | FRONT-first 元组 (ft, fp, full, back) | student_rank | tie_group_rank | tie 状态 |
|---|---|---|---|---|---|
| 1 | **PERSISTENT_RMT16_ORIGINAL_VTRACE_98304** | (3, 0.5905970705064548, 9, 7) | 1 | 1 | UNIQUE_TOP（formal_winner） |
| 2 | SLOWGRU_PERSISTENT_CANONICAL_98304 | (2, 0.5752855014895730, 17, 6) | 2 | 2 | ORDERED_NO_TIE |
| 3 | BASE_GTRXL_ORIGINAL_VTRACE_98304 | (2, 0.5650157181747473, 14, 8) | null | 3 | LOWER_TIE_GROUP_MEMBER |
| 3 | RESET128_RMT16_ORIGINAL_VTRACE_98304 | (2, 0.5650157181747473, 14, 8) | null | 3 | LOWER_TIE_GROUP_MEMBER |
| 5 | SLOWGRU_RESET128_CANONICAL_98304 | (2, 0.5236034412438056, 17, 7) | 5 | 5 | ORDERED_NO_TIE |
| 6 | CONTROL_CONTINUOUS_98304 | (0, 0.4196479859579006, 0, 7) | 6 | 6 | ORDERED_NO_TIE |
| — | BASELINE_TEACHER_CKPT17500 | (2, 0.5805684102905279, 19, 7) | null | — | TEACHER_REFERENCE_ONLY（排除） |

- `TOP_GROUP=[PERSISTENT_RMT16_ORIGINAL_VTRACE_98304]`，`TOP_GROUP_SIZE=1`，`TOP_TIE=false`；
- `LOWER_TIE_GROUPS=[{tie_group_rank:3, candidate_ids:[BASE_GTRXL…, RESET128…],
  tie_scope:LOWER_POSITION, winner_blocking:false}]`；
- `RANKING_STATUS=ORDERED_WITH_LOWER_TIES`；
- `FORMAL_WINNER=PERSISTENT_RMT16_ORIGINAL_VTRACE_98304`（重算输出，非常量；
  复验器对输入做了全置换不变性验证与"扰动任一指标 → 结果必变"的敏感性负例）。

**新旧对照披露**：`OLD_RULE_ORDER=FULL,FRONT_TRANSITION,FRONT_PROGRESS,BACK`；
`NEW_RULE_ORDER=FRONT_TRANSITION,FRONT_PROGRESS,FULL,BACK`；
`OLD_V3_FORMAL_WINNER=null`；`OLD_V3_RANKING_STATUS=INCONCLUSIVE_FULL_TIE`。
在旧 FULL-first 次序下 SLOWGRU_PERSISTENT 为唯一第一，但第 3 位的全元组并列按旧语义
作废了整个排名；新次序下 PERSISTENT 以 FRONT transition_count=3（唯一达到 3 的候选）
居唯一第一，而同一批低名次并列按新语义仅披露、不阻塞。

## 7. 门禁（§十一）

`FORMAL_RANKING_AUTHORIZED_V3R1=true`，当且仅当以下全部为真（全部为真）：

| 门 | 内容 | 结果 |
|---|---|---|
| R1 | 6/6 Student V3 证据完整（64/8/8，valid starts 齐） | true |
| R2 | teacher 参考完整 | true |
| R3 | 7 份 V3 证书 SHA 全部验证 | true |
| R4 | 旧 V3/V2 文件零修改（快照复验） | true |
| R5 | 主指标恒等（§七，EXACT） | true |
| R6 | FRONT-first 规则代码/协议一致 | true |
| R7 | top-tie-only 测试全部通过（§九 A–K + §六/§七 自检） | true |
| R8 | teacher 排除于学生排名 | true |
| R9 | 无重跑 rollout | true |
| R10 | 无 checkpoint 修改 / 无重训 / 无豁免 | true |
| R11 | 未使用 FULL-first 排序 | true |
| R12 | 未用 candidate_id 作科学 tie-break | true |

保留约束（随门禁一起固化）：`scientific_claim_authorized=false`、
`single_training_seed=true`、`multi_seed_confirmation_skipped_by_director=true`、
`scaffolded_results_can_replace_full_task=false`。
**本结果只可用于"工程阶段 strongest Student 选择"，不构成任何统计 SOTA 主张。**

## 8. 测试与复验（§九/§三/§十二）

| 测试 | 范围 | 结果 |
|---|---|---|
| ranking machine 自检 A–K | 唯一第一/低名次并列/顶并列/FRONT 优先/progress/FULL/BACK 决胜/四级全并列/置换不变/NaN·Inf·缺失 fail-closed/teacher 无豁免 | PASS（142 checks） |
| taxonomy v3r1 自检 | BACK/FULL 去重、FRONT 保留、trade-kill、INVALID_START、fail-closed 透传、560 条真实记录回放 parity（140 次去重命中） | PASS（828 checks） |
| 独立复验器（内嵌模式） | 驱动落盘 summary/gate 后、READY 落盘前 | PASS（2229 checks） |
| 独立复验器（完整模式） | 含 READY / SHA256SUMS_V3R1 / 全部交叉引用 | PASS（2255 checks） |

复验器独立重算：旧证据逐文件重哈希、主指标重抽取、episode records 重分类、
FRONT-first 重排名，并与发布产物逐项比对；验证 winner 为排名函数输出（非常量）、
输入置换不变、teacher 永不入选、READY 与 gate/summary 哈希交叉一致、
旧 `COMMON_EVALUATOR_V3_READY.json` 未被覆盖（V3R1 READY 为独立新文件）。

## 9. 新产物清单（§五）

目录 `reports/tier3_scaffolded_evaluation/formal_ranking_reclose_v3r1_20260803/`：

| 文件 | SHA256 |
|---|---|
| FORMAL_RANKING_SUMMARY_V3R1.json | `4bdba0da2379a5da546718863c73aba859a3046c3c044725ae21648ac5cf9238` |
| FORMAL_EVALUATION_GATE_V3R1.json | `23f540b9a917051913b7f431fea098632841fd0f2c635add3469eabccad13892` |
| RANKING_PROTOCOL_DECISION_V3R1.json | `e9b2b4fe21189c58246c1e29c70078d48a84f83d79800ab7c77642a27678a3a3` |
| RANKING_RECOMPUTATION_AUDIT_V3R1.json | `2e2e78e08d5c9cb51c0fa98bd8928c221cc920cb03d49826ab9082e3a5268fc9` |
| OLD_V3_IMMUTABILITY_REPORT.json | `197fc832e98ee8c84d1298defc4327d8d9a1e4c5b3ff08e4834202df652a9e57` |
| COMMON_EVALUATOR_V3R1_RANKING_READY.json | `9e740fe42ad631459604ad46ea61bc954e4361602711fe6c53cfa5c830d0c1b8` |
| verify_formal_ranking_v3r1.py | `011359939065fb371db5c5255e4c27787d2e46e3fc5045fe8c2b3edcf06bbc90` |
| formal_ranking_report_v3r1_20260803.md | （本报告；哈希于提交时固定） |

schemas：`mechanism_UED.tier3_formal_ranking_summary/v3r1`、
`mechanism_UED.tier3_formal_evaluation_gate/v3r1`、
`mechanism_UED.common_evaluator_v3r1_ranking_ready/v1`。
`SHA256SUMS_V3R1` 覆盖上表机器产物（报告在驱动落盘 sums 之后生成，未列入 sums；
其完整性由本提交的 git tree 固定）。

新增工具（`tools/tier3_scaffolded_evaluation/`）：

| 模块 | LF-SHA256 |
|---|---|
| tier3_ranking_v3r1.py（纯排名机器，JAX-free） | `2b850876bd3e28543a20e7e85d09099ed621dc72496aa3c668bd1487f635a607`（原样 sha） |
| tier3_taxonomy_v3r1.py（secondary 去重包装器） | `4804f19b12d0a3189fb20220620020adf285e3c57d1de7b2b30e4f252c5c735a` |
| tier3_formal_reclose_v3r1.py（重收口驱动，唯一 READY 写者） | `485855959178c5581020ac37cbafe15986e19c7ad6b5999543a558585f6d38cd`（原样 sha） |

冻结依赖钉：taxonomy_v3 LF-SHA
`01f06d09190a70898b11165aed016d5f7f96a1e0ca9366acc81dbd4d9d6a3da2`（未修改）；
metric_schema(common_v2) `8ec4adcdfa6844b276f5f253470e14ea8ad52f1e64c398e5e2658e8a066645c7`。

## 10. 诚实性声明

- 旧 V3 收口在其协议下正确；V3R1 为总控裁定的正式澄清与重收口，非"纠错翻案"；
- 全程无环境重跑、无重训、无 checkpoint/seed/bank/horizon/episode 数变动；
- 旧 V2/V3 证据与原始证书零触碰（逐文件哈希复验为零漂移）；
- 单一训练种子、未经多种子确认 → 科学主张不授权；结果用途严格限定为
  工程阶段 strongest Student 选择；
- 独立复验（`INDEPENDENT_REAUDIT_REQUIRED=true`）已内置并全绿，仍欢迎总控指派
  第三方复核。
