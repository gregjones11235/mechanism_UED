# Tier3 分段支架评测 — V3 复合事件语义修复正式评估报告（2026-08-01）

- **授权任务（总控 ruling）**：`CC4_REPAIR_NEG20_COMPOSITE_EVENT_SEMANTICS_AND_COMPLETE_FORMAL_EVALUATION_V3`
- **裁定**：`AUTHORIZED_COMPOSITE_EVENT_SEMANTIC_REPAIR_V3`
- **新协议**：`FORMAL_EVALUATOR_PROTOCOL=V3_COMPOSITE_EVENT`，`NEG20_PROTOCOL=NEG20_V3_PRIMARY_SECONDARY_EVENTS`
- **执行 git HEAD（marker==execution==closing 三者一致）**：`5f035ed238171729a47633b5c54f0b14da059082`
- **分支 / worktree**：`henry/tier3-scaffolded-evaluation` / `D:/cc4tmp`
- **证据目录**：`reports/tier3_scaffolded_evaluation/formal_evaluation_evidence_v3_20260801/`（77 文件，2.0M，**无任何 npz/pkl/npy/ckpt/orbax**）
- **证据 tarball sha256**：`ecf55214e0d48871d782aa230285862f201e7dea8f5f29954bcb101defa81d0d`
- **离线复验器**：`verify_formal_evaluation_evidence_v3.py`（JAX-free）→ **全绿 checks=754**

---

## §一 V2 归档与 V3 定位

V2 正式评估已**归档**（不可变引用，V3 绝不删/覆盖/改写）：

| 键 | 值 |
|---|---|
| V2_STATUS | `CLOSED_INCONCLUSIVE_PARTICIPATION`（1/6 候选完成，6 个被冻结分类器 NEG20 永久阻断） |
| V2_WINNER | `null` |
| V2_STUDENT_RANKING_VALID | `false` |
| V2 archive summary sha256 | `3e8186417aefeb25729324ce5fb4bc6b56a58087c8d1ee67bc088ad37d5c1ac3` |
| V2 archive gate sha256 | `51d3d6fb8efbc978875823cdc4576443c4d61f308840462c1bfa12da52fddc5b` |
| V2 证据被 V3 修改 | `false`（复验器 G14 + marker/gate/summary 三处独立断言） |

**V3 = 新的可审计 evaluator 语义修复**，不是 V2 的续跑或改写。修复对象唯一：冻结失败分类器对**合法复合事件 `floor transition AND defeat_kobold`** 无法赋单一标签而 FailClosed（NEG20）的问题。该复合事件物理可达（policy descend 2→3 后在 floor3 击杀），V2 把 6 个候选永久阻断，使全局评估只剩 1/6 参与。

## §二 复合事件语义修复（`tier3_taxonomy_v3.py`）

新表示（总控 §二逐字应用）：

- `primary_outcome`：FRONT 以**过渡成功**为主（`FRONT_TRANSITION_SUCCESS` / `FRONT_NO_TRANSITION` / `FRONT_INVALID_START`）；BACK/FULL 以 **DEFEAT_KOBOLD** 为主（`*_DEFEAT_KOBOLD_SUCCESS` / `*_NO_DEFEAT` / `*_INVALID_START`）。
- `secondary_events[]`（排序输出）：`DEFEAT_KOBOLD` / `PLAYER_DIED` / `KOBOLD_ENGAGED` / `CORRIDOR_EXIT_REACHED` / `TIMED_OUT`。
- `taxonomy_status`：`VALID_COMPOSITE_EVENT`（≥2 合法主事件并存，**不 abort**）/ `VALID_SINGLE_EVENT` / `INVALID_START` / `FAIL_CLOSED_<类别>`。

**FRONT 复合判定**：`transition∧defeat` → `primary=FRONT_TRANSITION_SUCCESS`，`secondary` 含 `DEFEAT_KOBOLD`，`taxonomy_status=VALID_COMPOSITE_EVENT`，**不得 FailClosed**。defeat-only **不得**误报 transition success（冻结主谓词只认 transition）。

**FailClosed 仅保留 6 类**（打类别标签）：状态损坏 / 必需字段缺失 / 互相矛盾且无法由事件时序解释的数据（defeat+timed_out、died+timed_out、alias 矛盾、无终局信号）/ 非法值 / 证据哈希不匹配 / 未注册事件类型。`defeat+died` 同步步属"合法多事件并存"（时序可解释）→ 不 fail-close，主 defeat + 次 PLAYER_DIED（证据中 0 次出现，已加自检并披露）。

**taxonomy_v3 模块 LF-SHA256（新钉）**：`01f06d09190a70898b11165aed016d5f7f96a1e0ca9366acc81dbd4d9d6a3da2`

**dense 均值逐位奇偶**：`summarize_v3` 把每条 V3 分类映射回冻结 label 串，构造冻结兼容 classified 列表后**以库方式调用冻结 `tier3_metrics.summarize`**（已验证其零 taxonomy 依赖）→ 逐位一致由构造保证，非自行实现均值。

## §三 最小变更（5 个新文件，冻结面零触碰）

全部位于 `tools/tier3_scaffolded_evaluation/`，冻结引擎模块（evaluator/predicates/runtime/bank/profile/schema）**一字未改**：

| 提交 | 文件 | 内容 |
|---|---|---|
| C1 `f8c17ac` | `tier3_taxonomy_v3.py` | 复合事件分类器（primary/secondary/taxonomy_status；6 类 FailClosed 保留；§四 A–H 自检） |
| C2 `bb88a28` | `tier3_evaluation_certificate_v3.py` | V3 证书（11 冻结钉 + taxonomy_v3 LF-SHA + 协议号；逐臂 reuse provenance；复合披露；rank=null；禁伪装 V2 扫描） |
| C3 `4c5fbe6` | `tier3_formal_evaluation_v3.py` | V3 驱动（FULL 离线复用 R1–R9 / FRONT 离线重分类 / BACK 补跑；冻结 rollout 逐字复刻；`evaluate()` 绝不调用） |
| C4 `e675743` | `tier3_formal_ranking_v3.py` + `tier3_formal_start_marker_v3.py` | V3 排名 + 收口门 G1–G16（冻结规则逐字；V3 READY 唯一写者）+ 修复授权 marker |
| `5f035ed` | （驱动 stage1 修复） | stage1 核验 V2-CLOSED 前置（而非 V2 未启动守卫）；此即执行 HEAD |

新协议号、新 full SHA、新 schema 串（`mechanism_UED.tier3_evaluation_certificate/v3`、`mechanism_UED.tier3_v3_repair_authorization/v1`）——**不伪装 V2**。marker→收口之间无工具提交，HEAD 链干净。

## §四 测试台（A–H，全部 JAX-free）

- **A–E、G**：`tier3_taxonomy_v3.py --self-test` 合成记录。G 覆盖全部 6 保留 FailClosed 类（腐败 timesteps>4096 / action 长度不符、缺字段、不可解释矛盾、非法值、哈希不符、未注册 scenario/事件）。
- **F（精确奇偶）**：对 CONTROL 的 V2 记录做 V3 原生重算，四元组 `== (0, 0, 0.4196479859579006, 7)`（精确 `==`）；7 份 FULL/BACK 块 == V2 发布值（逐位）。阻断候选 FRONT 无 V2 数值 → V3 为首发，不作奇偶声明。复验器 §7 对全部 7 候选四元组逐位复现（`ALL_TUPLES_REPRODUCED`）。
- **H（双重断言）**：对全部 **9 条**已提交 transition∧defeat FRONT 记录，**V3 判 `VALID_COMPOSITE_EVENT`**（primary `FRONT_TRANSITION_SUCCESS`、secondary 含 `DEFEAT_KOBOLD`、记录 SHA 重算匹配）**且冻结 V1 `classify_episode` 仍抛 NEG20**——证明修复是**加性的**、冻结面未改。复验器 §11 导入仓库内提交工具（`tier3_failure_taxonomy` + `tier3_taxonomy_v3`，均 JAX-free）逐条回放验证。
- **COMPOSITE_EVENT_TESTS** = A–H 全绿；**V2_NON_COMPOSITE_PARITY** = 全 7 候选四元组逐位复现 + CONTROL §四F 精确；**ORIGINAL_NEG20_REPRODUCTIONS_FIXED** = **9**。

## §五 复用与补跑（驱动 stage7 分臂）

| 臂 | 处置 | 结果 |
|---|---|---|
| **FULL** | 离线复用 R1–R9（全离线可核：64/64 带评估、逐条记录 SHA 重算、V2 sums 6/6 重哈希、checkpoint/params owner 匹配、runtime/capsule 匹配、schedule 冻结 seeds 200000..200063/4096/greedy、无性能早停、引擎 4 模块 LF-SHA 冻结、V3 重分类可复现） | 7/7 `REUSED_PASS`（reuse_gate R1–R9 全真） |
| **FRONT** | 离线重分类（`classification_only=true`，`environment_rerun=false`；源 V2 episode SHA + 逐记录 bank SHA×8 + v3_classifier_sha 全记录） | 7/7 `REUSED_RECLASSIFIED`（8/8 离线，无补跑） |
| **BACK** | 6 阻断候选 0/8 骨架 → **正式补跑**（同一冻结 profile：back bank seeds 1010000..1010007、4096、greedy；逐字复刻 V2 rollout 循环）；CONTROL 已 8/8 → 复用重签 | 6/6 `COMPLETED`（`V3_FRESH_COMPLETION_RUN`，首次执行非重试，`source_v2_episode_sha256=null`）；CONTROL `REUSED_RESIGNED`（`V2_COMMITTED_EVIDENCE`，源 sha `41dee497…`） |

产物落 `<pool>/cc4/<ID>/formal_evaluation_v3/`（8 文件/候选）；V2 目录 `formal_evaluation_v2dt/` 零触碰。

## §六 统一 V3 证书（7 entries 引用相同钉）

7 份 `evaluation_certificate_v3.json`（schema `mechanism_UED.tier3_evaluation_certificate/v3`）引用**完全相同**的公共钉集（复验器断言 `len(common_pins_seen)==1`）：

| 钉 | sha256 |
|---|---|
| common_evaluator（V3_EVALUATOR_SHA256） | `2978a0f625bc94e18c99649959e8c090f964cd66e5dafd6b93245f144a317037` |
| metric_schema（V3_METRIC_SCHEMA_SHA256） | `8ec4adcdfa6844b276f5f253470e14ea8ad52f1e64c398e5e2658e8a066645c7` |
| evaluation_profile（V3_PROFILE_SHA256） | `0f1d2c1a17ea9802243583fdaa0f7966662dc87738705a9583eca2f566639069` |
| full_profile | `2eceb288785a589f3f7f8b6989be7876bbe8da299128363ee008397d79039c1f` |
| front_bank_content | `21aeb7dcdcb4ffccfb1eedc80f2c6daec1995c242822a0efbec9b947e275d687` |
| back_bank_content | `c632e30dcabea7ff812fa07ce855809ec54e7cbca87747fa2d5ab775431f2566` |
| common_runner | `4113e666c7d6f582b5a158d89cec1972ecf40a478b9ea189ec80901c78cd51f6` |
| environment_lock | `453f1680dafe0f168c25c262f51de59ddc59559676aecd05f8f17389015c2ad3` |
| candidate_runtime_abi | `61e52af6ff64a3071f8b64916c80906275dcb201d37feaa0382ed988d03d7f6a` |
| assembly_manifest_v2 | `aa18d2f4927d9c268b029d324381eaae6e485b9e81b07a3a5927e1ce5cad4420` |
| sha256sums_v2 | `190624ea399bdf420413339d8ca0b2defdf7bec05b2971d303eac3f44670abde` |

外加 `taxonomy_v3_lf_sha256=01f06d09…`、`neg20_protocol=NEG20_V3_PRIMARY_SECONDARY_EVENTS`、`formal_evaluator_protocol=V3_COMPOSITE_EVENT`。每份证书 `student_rank=null`、四类诚实标志位全 `false`、`provenance.git_commit_head=5f035ed2…`、`gpu.visible_gpu_uuids ⊆ {GPU2,GPU3}`。

## §七 排名门（G1–G16）与排名结果

**门全绿**（`FORMAL_EVALUATION_GATE_V3_PASS=true`，17 项门全真，`gate_failures=[]`，`blocked_candidate_ids=[]`）：G1 6/6 student 完成、G2 teacher 完成、G3 无 ENGINE_ABORT、G4 无彩排入正式池、G5 证书全验、G6 钉统一冻结、G7 git HEAD 统一、G7b HEAD 等于或后裔于 marker、G8 规则逐字、G9 注册表 rank null、G10 排名诚实计算、G11 FULL 复用×7、G12 FRONT 重分类 provenance×7、G13 BACK 补跑/复用×7、G14 V2 归档未触碰、G15 V3 marker 核验、G16 GPU ⊆ {GPU2,GPU3}。

**`FORMAL_RANKING_AUTHORIZED_V3=true`**（全完成、无 ENGINE_BLOCKED、无豁免）。flip 策略 `V3_GATE_GREEN`，READY_V3 已翻转（`FORMAL_RANKING_STARTED=PUBLISHED=true`，`pending_gates=[]`）。

**冻结排名规则（逐字，源自 `metric_schema.json`，sha 钉）**：序 `["full success_count","front_l2 transition_count","front_l2 mean graph_distance_progress","back_l2 defeat_count"]`，容差 `1e-12`，四级全平 → `INCONCLUSIVE`。排名机器 `rank_students`/`compare_rule_tuples` 自 V2DT **逐字导入（对象同一）**。

### 发布四元组（复验器从原始 result 独立重算，逐位复现）

| 候选 | full succ | front trans | front progress | back defeat | rank |
|---|---|---|---|---|---|
| SLOWGRU_PERSISTENT_CANONICAL_98304 | 17 | 2 | 0.575285501489573 | 6 | **1** |
| SLOWGRU_RESET128_CANONICAL_98304 | 17 | 2 | 0.5236034412438056 | 7 | **2** |
| BASE_GTRXL_ORIGINAL_VTRACE_98304 | 14 | 2 | 0.5650157181747473 | 8 | **null**（平局） |
| RESET128_RMT16_ORIGINAL_VTRACE_98304 | 14 | 2 | 0.5650157181747473 | 8 | **null**（平局） |
| PERSISTENT_RMT16_ORIGINAL_VTRACE_98304 | 9 | 3 | 0.5905970705064548 | 7 | **5** |
| CONTROL_CONTINUOUS_98304 | 0 | 0 | 0.4196479859579006 | 7 | **6** |
| BASELINE_TEACHER_CKPT17500（参考） | 19 | 2 | 0.5805684102905279 | 7 | null（TEACHER_REFERENCE_ONLY，不入学生排名） |

### 排名结论

- **FORMAL_RANKING_STATUS = `INCONCLUSIVE_FULL_TIE`**
- **FORMAL_WINNER = `null`**
- 平局组：`[[BASE_GTRXL_ORIGINAL_VTRACE_98304, RESET128_RMT16_ORIGINAL_VTRACE_98304]]` —— 两投影变体在冻结四级指标 `(14, 2, 0.5650157181747473, 8)` 上**精确全平**（tol 1e-12）。
- 这是冻结排名机器在**全参与**（6/6 student + 1/1 teacher）下的**诚实实在结果**：冻结单种子指标组无法区分这两个投影变体。偏序仍披露（SLOWGRU_PERSISTENT=1、SLOWGRU_RESET128=2、平局组 rank=null 居 3–4、PERSISTENT=5、CONTROL=6）。
- 与 V2 不同：**无升级**（`escalation=null`）——V2 的 INCONCLUSIVE 源于参与不足（1/6，引擎阻断），V3 的 INCONCLUSIVE 是合法的全参与精确平局。

### 复合事件披露（summary）

- `total_composite_episodes=13`（FRONT，major≥2）：CONTROL 0 / PERSISTENT 3 / RESET128 2 / BASE_GTRXL 2 / SLOWGRU_RESET128 2 / SLOWGRU_PERSISTENT 2 / TEACHER 2。
- 其中 transition∧defeat（== 原 NEG20 复现）= **9**：CONTROL 0 / PERSISTENT 1 / RESET128 2 / BASE_GTRXL 2 / SLOWGRU_RESET128 1 / SLOWGRU_PERSISTENT 1 / TEACHER 2。全部判 `VALID_COMPOSITE_EVENT`（非 abort）。

## §八 禁止事项合规（总控 §八/§九）

| 禁止项 | 状态 |
|---|---|
| 重训 Student | `STUDENTS_RETRAINED=0`，`RETRAINING_PERFORMED=false` |
| 改 checkpoint | `CHECKPOINTS_MODIFIED=false` |
| 候选级豁免 | `CANDIDATE_EXCEPTIONS_USED=0`，`CANDIDATE_EXCEPTION_USED=false` |
| 删 CONTROL / 只排 5 个 | 6/6 全排，CONTROL 在列（rank 6） |
| FULL-only / BACK-only 排名 | `FULL_ONLY_RANKING_USED=false`，`BACK_ONLY_RANKING_USED=false` |
| 改 bank / seeds / episode 数 / horizon | `FROZEN_BANKS_MODIFIED=false`；schedule 冻结（R6） |
| 性能重试 / smoke 替代 | `NO_PERFORMANCE_RETRY` / `NO_SMOKE_SUBSTITUTION_FOR_PERFORMANCE`（BACK 补跑措辞一律 completion，非 retry） |
| 覆盖 V2 证据 | `v2_evidence_modified_by_v3=false`（G14） |
| force push / rebase / amend / merge | 未使用；path-limited add；HEAD 链干净 |
| 降级子指标排名 | `NO_SUBMETRIC_RANKING_DOWNGRADE`（仍用冻结四级序） |
| GPU 纪律 | 仅 GPU2/GPU3（G16 强制；GPU0/1 对 V3 禁） |

## §九 固定格式汇报键

```
V2_STATUS                         = CLOSED_INCONCLUSIVE_PARTICIPATION
V2_WINNER                         = null
V3_DRIVER_COMMIT                  = 5f035ed238171729a47633b5c54f0b14da059082
V3_TAXONOMY_PROTOCOL              = V3_COMPOSITE_EVENT (NEG20_PROTOCOL=NEG20_V3_PRIMARY_SECONDARY_EVENTS)
V3_TAXONOMY_SHA256                = 01f06d09190a70898b11165aed016d5f7f96a1e0ca9366acc81dbd4d9d6a3da2
V3_EVALUATOR_SHA256               = 2978a0f625bc94e18c99649959e8c090f964cd66e5dafd6b93245f144a317037
V3_PROFILE_SHA256                 = 0f1d2c1a17ea9802243583fdaa0f7966662dc87738705a9583eca2f566639069
V3_METRIC_SCHEMA_SHA256           = 8ec4adcdfa6844b276f5f253470e14ea8ad52f1e64c398e5e2658e8a066645c7
COMPOSITE_EVENT_TESTS             = A–H PASS (taxonomy_v3 self-test; §四F 精确奇偶; §四H 双重断言)
V2_NON_COMPOSITE_PARITY           = PASS (7/7 四元组逐位复现; CONTROL==(0,0,0.4196479859579006,7))
ORIGINAL_NEG20_REPRODUCTIONS_FIXED = 9

候选 × {FULL_REUSE, FRONT_RECLASSIFICATION, BACK_COMPLETION/STATUS}:
  PERSISTENT_RMT16_ORIGINAL_VTRACE_98304   : FULL=REUSED_PASS, FRONT=REUSED_RECLASSIFIED(3 composite), BACK=COMPLETED / PASS
  RESET128_RMT16_ORIGINAL_VTRACE_98304     : FULL=REUSED_PASS, FRONT=REUSED_RECLASSIFIED(2), BACK=COMPLETED / PASS
  BASE_GTRXL_ORIGINAL_VTRACE_98304         : FULL=REUSED_PASS, FRONT=REUSED_RECLASSIFIED(2), BACK=COMPLETED / PASS
  CONTROL_CONTINUOUS_98304                 : FULL=REUSED_PASS, FRONT=REUSED_RECLASSIFIED(0), BACK=REUSED_RESIGNED / PASS
  SLOWGRU_RESET128_CANONICAL_98304         : FULL=REUSED_PASS, FRONT=REUSED_RECLASSIFIED(2), BACK=COMPLETED / PASS
  SLOWGRU_PERSISTENT_CANONICAL_98304       : FULL=REUSED_PASS, FRONT=REUSED_RECLASSIFIED(2), BACK=COMPLETED / PASS
  BASELINE_TEACHER_CKPT17500 (reference)   : FULL=REUSED_PASS, FRONT=REUSED_RECLASSIFIED(2), BACK=COMPLETED / PASS

STUDENT_V3_COMPLETE_COUNT         = 6/6
TEACHER_V3_COMPLETE               = true (TEACHER_REFERENCE_ONLY)
ENGINE_BLOCKED_COUNT              = 0
FORMAL_RANKING_AUTHORIZED         = true
FORMAL_RANKING_STATUS             = INCONCLUSIVE_FULL_TIE
FORMAL_WINNER                     = null
CHECKPOINTS_MODIFIED              = false
STUDENTS_RETRAINED                = 0
CANDIDATE_EXCEPTIONS_USED         = 0
FULL_ONLY_RANKING_USED            = false
BACK_ONLY_RANKING_USED            = false
```

## 证据清单与复验

- 证据目录 77 文件：7 候选 bundle（各 8 文件）+ `COMMON_EVALUATOR_V3_READY.json` + `CROSS_GPU_DETERMINISM_PREFLIGHT_V3.json` + `FORMAL_EVALUATION_GATE_V3.json` + `FORMAL_RANKING_SUMMARY_V3.json` + `V3_REPAIR_AUTHORIZATION.json`（各带 `.sha256` sidecar）+ `formal_eval_logs/`（gpu2/gpu3 日志 + 队列 out + summary）+ `common_v2/metric_schema.json`。
- 关键 artifact sha256：summary `dab522cf…`、gate `c529ebf3…`、marker `efa68c85…`、preflight `afaa6b3b…`。
- **离线复验器 `verify_formal_evaluation_evidence_v3.py` 全绿（checks=754）**：权重隔离、发布件+sidecar 重哈希、marker 链（逐字裁定/9 禁令/V2 归档 SHA/taxonomy LF-SHA/git HEAD）、预检（GPU idx {2,3}、未放宽、零差异）、READY_V3 翻转、冻结规则逐字、7 bundle（SHA256SUMS 重哈希 + READY + 证书统一钉 + 逐臂 episode SHA 原始行重算 + reuse provenance）、复合事件层、原生排名重算 → INCONCLUSIVE_FULL_TIE + 平局组 + winner null、§四F CONTROL 精确、§四H 双重断言（9 条 V3 VALID_COMPOSITE_EVENT ∧ 冻结 V1 仍抛 NEG20）、gate 17 项、禁夸扫描、日志证据。

## 升级 / 裁定结论

- **无升级**（`escalation=null`）。V3 在总控授权范围内完成：冻结失败分类器的 NEG20 复合事件语义已修复（加性、冻结面未改），全部 6 个 student + teacher 参考在全冻结合同下完成正式评估，排名门全绿。
- **科学结论**：在冻结单种子四级指标组下，`BASE_GTRXL_ORIGINAL_VTRACE_98304` 与 `RESET128_RMT16_ORIGINAL_VTRACE_98304` 两个投影变体**精确全平**，全局排名诚实地为 `INCONCLUSIVE_FULL_TIE`、无 winner。这不是参与不足（V2 的病因），而是冻结指标组对这两个变体不可区分——如需区分，属新的科研裁定（更换/扩充指标组或多种子），不在本授权范围内，不单方面蔓延。
- 所有诚实标志位（scientific_claim_authorized / scaffolded_results_can_replace_full_task / interface_smoke_substituted_for_performance / teacher_included_in_student_ranking）均 `false`。
