# 正式全局性能评估报告（V2_DYNAMIC_TOPOLOGY，6 student + teacher reference）

日期：2026-08-01　责任人：CC4（统一评测 / world-set 证据 / Tier3 分段支架评测环境）
分支：`henry/tier3-scaffolded-evaluation`　worktree：`D:/cc4tmp`
授权任务：FORMAL GLOBAL PERFORMANCE EVALUATION（总控授权，formal ranking）
二级审计结论（逐字）：`SECONDARY_AUDIT_VERDICT=PASS_FOR_FORMAL_GLOBAL_PERFORMANCE_EVALUATION_START`

---

## 1. 最终裁定

```
FORMAL_GLOBAL_PERFORMANCE_EVALUATION = CLOSED_INCONCLUSIVE_PARTICIPATION
ranking_status        = INCONCLUSIVE_PARTICIPATION
eligible_students     = 1/6（CONTROL_CONTINUOUS_98304）
engine_blocked        = 6（5 student + teacher reference）
FORMAL_EVALUATION_GATE_V2DT_PASS = false（参与门 G1/G2/G3 假；完整性门全真）
flip_policy           = PUBLISH_HONEST_INCONCLUSIVE_UNDER_ENGINE_BLOCK
scientific_claim_authorized = false
升级总控              = 必需（见 §9）
```

**不存在正式优胜者。** 唯一完成候选 CONTROL 不获得排名（<6 参评 → 无权威排序，冻结规则使然，非遗漏）。teacher 指标同样缺失（同样被冻结分类器中止），reference 绑定记 FAIL 并如实发布。

## 2. 总控 8 条硬约束执行核对

| # | 约束 | 执行 |
|---|---|---|
| 1 | V2_DYNAMIC_TOPOLOGY common evaluator | ✓ 全程 V2 引擎（LF-SHA 字节钉，逐证书重验） |
| 2 | 同一套 FRONT/BACK/FULL frozen banks | ✓ 银行内容 SHA 门逐候选核验，未改 |
| 3 | 6 student 参评，teacher 仅 reference 不入排名 | ✓ teacher rank=null、excluded=true、reference_only=true |
| 4 | 不改 checkpoint | ✓ gate `CHECKPOINTS_MODIFIED=false` |
| 5 | 不重训 | ✓ `RETRAINING_PERFORMED=false`、`CONTROL_RETRAINED=false` |
| 6 | 不把 interface smoke 当性能结果 | ✓ 逐证书/READY/summary `interface_smoke_substituted_for_performance=false`；本次为 FULL 64 + FRONT 8 + BACK 8 × 4096 步真实 rollout |
| 7 | 正式证书 + 排名汇总 + per-student 指标 | ✓ 7 证书 + `FORMAL_RANKING_SUMMARY_V2DT.json` + gate |
| 8 | 启动前 SECONDARY_AUDIT_PASS marker | ✓ marker 先于任何正式运行写入（`recorded_at_utc=2026-07-31T10:09:36Z`，sha `b08c1a9b…`，拒覆盖） |

## 3. 候选结果

| 候选 | 类别 | 状态 | 规则四元组 (full succ / front trans / front progress / back defeat) |
|---|---|---|---|
| CONTROL_CONTINUOUS_98304 | STUDENT | **ELIGIBLE_COMPLETE**（无排名，1/6） | 0 / 0 / 0.4196479859579006 / 7 |
| PERSISTENT_RMT16_ORIGINAL_VTRACE_98304 | STUDENT | BLOCKED_ENGINE_ABORT | — |
| RESET128_RMT16_ORIGINAL_VTRACE_98304 | STUDENT | BLOCKED_ENGINE_ABORT | — |
| BASE_GTRXL_ORIGINAL_VTRACE_98304 | STUDENT | BLOCKED_ENGINE_ABORT | — |
| SLOWGRU_RESET128_CANONICAL_98304 | STUDENT | BLOCKED_ENGINE_ABORT | — |
| SLOWGRU_PERSISTENT_CANONICAL_98304 | STUDENT | BLOCKED_ENGINE_ABORT | — |
| BASELINE_TEACHER_CKPT17500 | TEACHER_REFERENCE | BLOCKED_ENGINE_ABORT | —（reference 本就不入排名） |

BLOCKED 统一形态（结构化中止证据，6 份同构）：`evaluation_status=BLOCKED`；READY 门**仅 G4_FORMAL_SCHEDULE_COMPLETE 假**（G12_CERTIFICATE_VERIFIED 真）；`formal_abort.verdict=ENGINE_TAXONOMY_REJECTED_FORMAL_EVALUATION_V2`，`aborted_phase=evaluate_classification`，`scenario=front_l2`，`episodes_completed_before_abort=8/8`；FULL 64/64 完整带评估、FRONT 8/8 rollout 完整但评估缺（中止点）、BACK 0/8（冻结顺序 FULL→FRONT→BACK 下未到达）。

## 4. BLOCKED 根因（冻结引擎分类器，不可由 CC4 放宽）

中止异常类：`tier3_failure_taxonomy.FailClosed`（NEG20 类，区别于驱动/运行时 FailClosed）：

> FAIL CLOSED (NEG20): ambiguous/contradictory terminal signals
> `['front_l2 floor transition AND defeat_kobold (front ends at the transition)']`;
> refusing to silently assign a single label

冻结分类器对"同一条 episode 既到达 front 层楼过渡（FRONT 的成功条件）又击败 kobold（FRONT 范围外的终局信号）"拒绝赋单一标签。**凡在 front 层楼过渡点击败 kobold 的强 policy 必然触发**：5/6 student + teacher 全部触发（6 次触发点分别为 front_l2 bank4/bank6 等，`defeat=True transition=True progress=1.0`）。CONTROL 因 policy 弱、在过渡前早亡，不触发而完整通过。这不是数据腐败、不是基础设施故障、不可由 CC4 豁免或放宽——分类器与引擎模块同属冻结面（LF-SHA 字节钉）。重跑逐位复现（§6）。

## 5. 排名规则与结果

冻结规则（`metric_schema.json`，sha 钉，逐字内嵌 summary）：字典序降序 `["full success_count","front_l2 transition_count","front_l2 mean graph_distance_progress","back_l2 defeat_count"]`，容差 1e-12，四级全平 → INCONCLUSIVE。参评 <6 → `INCONCLUSIVE_PARTICIPATION` + 升级总控（runbook §8）。teacher 出指标但 `rank=null, excluded=true`；排名不回写注册表（`student_rank` 结构 null，G9 核验）。

## 6. 确定性证据（重跑合法性）

四候选首跑日志为**旧字节码进程**崩溃（C9 结构化中止路径同步晚于队列启动；Python linecache 使 traceback 呈现新源码文本）。按 runbook §8 处置：保日志（`*_rawcrash_pre_driver_fix_20260731T135941Z.log` ×4）→ 部分目录移入 `_incomplete_blocked_pre_driver_fix_*`（不提交）→ 以 C9 驱动结构化重跑。重跑是**证据重建**（同一冻结银行/种子/合同，确定性复现中止并落盘结构化 BLOCKED），不是成绩重试——中止点在冻结分类器内，重跑必然同点中止。三次独立逐位复现：

| 证据 | 首跑 | 重跑 | canonical 字段 |
|---|---|---|---|
| RESET128 `full-seed200020` | 4096 步 timed_out | 4096 步 timed_out | 逐位相同（墙钟 1092.50s vs 1100.06s，非确定性字段） |
| PERSISTENT `front_l2-bank4` | 2285 步 | 2285 步 | 逐位相同 |
| TEACHER `front_l2-bank7` | 4096 步 | 4096 步 | 逐位相同 |

跨 GPU 确定性预检（GPU0/GPU1 启用前置）：`CROSS_GPU_DETERMINISM_PREFLIGHT=PASS checks=180`（sha `5c23a8fc…`）；GPU2 彩排对 b0d7e92 的 rollout 逐位相同。

## 7. 披露事项（全部如实）

1. **GPU 解禁**：总控 2026-07-31 裁定解禁 GPU0/GPU1 用于本次评估，四卡并行（每卡单进程顺序队列）；条件全部满足——跨 GPU 预检 180 项先过、注册表 `CC4_GPU_ALLOWED_UUIDS` 扩 4 卡已提交可审计（历史禁令注释保留）。总墙钟约 8h。
2. **设计偏差一（marker/flip 分离）**：冻结验证器 smokev2 步骤 1e 要求启动前 `FORMAL_RANKING_STARTED=false`；故 marker 工具只写 marker，排名工具在收口执行唯一 READY 白名单 RMW（runbook §1）。
3. **设计偏差二（HEAD 策略，C10=`8d46bd3`）**：C3 排名工具的"bundle HEAD == marker HEAD"严格相等写于 C8/C9 驱动硬化提交之前；7 份 bundle 一致携带执行 HEAD `6f5e2705`。放宽为：7 份一致 **且** 等于或 git 可证后裔（`merge-base --is-ancestor`），summary `git_head_policy` 块全披露（marker `b0d7e92` / execution `6f5e2705` / closing `8d46bd3` / 逐 bundle 关系 / marker→execution 提交清单 = 恰好 C8 marker 门形修复 + C9 结构化中止两个纯工具提交）。行为同一性依据：冻结引擎模块 LF-SHA 字节钉逐证书重验 + 180 项预检 + 彩排逐位相同——工具提交不改 rollout/分类行为。
4. **设计偏差三（诚实发布翻转策略，C10）**：参与门 G1/G2/G3 在 BLOCKED 多数下必然假；若"全绿才翻转"则收口永久死锁。修复：完整性门（G4–G10+G7b）全真且剩余失败全部源自 BLOCKED 候选 → 策略 `PUBLISH_HONEST_INCONCLUSIVE_UNDER_ENGINE_BLOCK` 翻转；任何完整性失败 → NO_FLIP（自检含 pin 漂移 → NO_FLIP 负例）。翻转后驱动对任何重跑 fail-close。
5. **单训练种子**：所有候选均为单训练种子产物，summary 逐字记录 `scientific_claim_status="FORMAL_SCIENTIFIC_CLAIM: NOT_AUTHORIZED_SINGLE_TRAINING_SEED"`，不构成科学论断。
6. **墙钟构成**：单候选 80 条真实 episode × 最高 4096 步 × 实测 0.21–0.27s/步；强候选（RMT16/teacher）episode 顶满上限，FULL 单腿 2–3h；重跑约 80% 墙钟为冻结顺序强制的 FULL 腿。

## 8. 证据清单与核验

证据目录：`reports/tier3_scaffolded_evaluation/formal_evaluation_evidence_20260801/`（服务器 tarball 端到端 sha `07f5c01874ddead2ac76800992280edc1e7a301d3632a23f16f94e09629286eb`，270KB/93 条目；**npz/pkl 永不出服务器**，目录含权重隔离断言）。

| 文件 | SHA256 |
|---|---|
| `cc4/FORMAL_RANKING_SUMMARY_V2DT.json` | `3e8186417aefeb25729324ce5fb4bc6b56a58087c8d1ee67bc088ad37d5c1ac3` |
| `cc4/FORMAL_EVALUATION_GATE_V2DT.json` | `51d3d6fb8efbc978875823cdc4576443c4d61f308840462c1bfa12da52fddc5b` |
| `cc4/SECONDARY_AUDIT_PASS.json` | `b08c1a9bf7055ac6b4a200c6f561374a0c34c2dee9ce6dd799212de3eb5f8351` |
| `cc4/POOL_BINDING_GATE_V2DT.json` | `cec167117a7aa8e67a3d5eb60839e711e72d950135553e4035a87e6c9859a352` |
| `cc4/CROSS_GPU_DETERMINISM_PREFLIGHT.json` | `5c23a8fccb3a61ffb3fdfa7be83c7eca24e7b3eff2e6676dcad85cc5eda29f7c` |
| `common_v2/COMMON_EVALUATOR_V2_READY.json`（翻转后，after_sha 前缀 `62a3982ff5ac06b7`） | — |

含 7 份 `formal_evaluation_v2dt/`（各 8 文件：episode_records.jsonl / 3×result / certificate / provenance / READY / SHA256SUMS）与全部日志（7 正式日志 + 4 旧崩溃日志 + 4 队列 + 4 重跑 wave nohup）。

核验记录：排名工具自检 40/40（本地 + 服务器 venv 双跑）；干跑形态确认；**独立离线复验器 `verify_formal_evaluation_evidence.py` checks=527 全绿**（全文件重哈希、marker/preflight/READY 链、7 bundle 结构、BLOCKED 证据、日志证据、原生排名重算比对发布值、禁夸扫描、权重隔离）。

## 9. 升级总控（必需）

冻结 NEG20 分类器在"过渡点击败 kobold"上拒绝赋标签，使 5/6 student + teacher 永久 BLOCKED——**这是冻结语义与强 policy 行为的结构性冲突，CC4 无权裁定**。请总控在以下方向中裁定（任一均需新授权，CC4 不自行放宽）：

1. 维持现状：发布 INCONCLUSIVE_PARTICIPATION 为最终结论（本次已如实发布）；
2. 授权对 FRONT 过渡层 kobold 终局信号的分类器语义澄清（需新审计轮，触及冻结面）；
3. 授权降级指标集（如仅 FULL success_count / 仅 BACK defeat_count 的子排名）——需显式裁定替代规则并重新审计；
4. 其他。

## 10. 诚实边界（逐字）

`scientific_claim_authorized=false`；`scaffolded_results_can_replace_full_task=false`（scaffolded ≠ full task）；`interface_smoke_substituted_for_performance=false`；teacher reference 不入 student ranking；单训练种子；BLOCKED 候选绝不记为成绩、无候选级豁免；不伪造 PASS。

## 11. 连续性记录

服务器 repo HEAD `8d46bd30770d05c5eee11c9c58592e928fdd152d`（C10，ff-only 同步）；本地 HEAD 同。四 GPU 全部释放（1 MiB）；无 CC4 进程；孤儿 PID 106885 未触碰（standing 纪律）。提交链：b736c8c → … → 6f5e2705（C9，正式执行 HEAD）→ 8d46bd3（C10，收口工具）。V1（d0d05ff2）保持 SUPERSEDED_PRE_RANKING。
