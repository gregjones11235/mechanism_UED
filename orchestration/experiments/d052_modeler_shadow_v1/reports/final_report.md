# D052-V2 最终报告：Modeler 四角色影子实验（Phase 0–2）

- **任务**：D052-V2 Modeler + Tutor/Critic/Explorer 四角色任务过滤影子实验
- **分支**：`henry/d052-modeler-shadow-v1`（未 push）
- **目标**：`soft_copeland_x_original` / `seed0_1784462982` / **round 4**（唯一有逐 episode 评测的轮次）
- **输出**：`mechanism_UED/orchestration/experiments/d052_modeler_shadow_v1/`
- **授权状态**：Phase 0–2 已预授权并完成；Phase 3（训练）**未授权**，本报告仅给出推荐。
- **结论速览**：`ENGINEERING_PASS` + `SCIENTIFIC_SIGNAL（强）` + `EVIDENCE_INSUFFICIENT（缺口映射）` → **推荐 `AUTHORIZE_4096_SMOKE`（附强制条件）**，并停止等待用户裁决。

---

## 1. 三阶段判定矩阵

| 维度 | Phase 0 审计 | Phase 1 Pilot(8) | Phase 2 全影子(32) |
|------|-------------|------------------|--------------------|
| 工程 | PASS（确定性已证：pool_hash/selected-8/selection_hash 逐位复现 round1&4） | **ENGINEERING_PASS**（7 硬门全过） | **ENGINEERING_PASS**（OLD 重算复现原始 selected-8 强锚定） |
| 科学信号 | — | **SIGNAL**：Spearman 0.095，50% 判断翻转 | **SIGNAL（强）**：selected-8 变 7/8，Jaccard 0.067，Spearman −0.21，37.5% 翻转 |
| 缺口映射 | 确立 salted-hash 危险：intended-target SR=UNDEFINED | 不适用 | **EVIDENCE_INSUFFICIENT / UNDEFINED**（7 个新进入候选 target 全为 salted 占位符） |

**分类口径（按指令要求区分）**：
- `ENGINEERING_PASS`：全部硬门通过——schema 100% 有效、canonical id 有效、evidence_ids 存在、数值可重算、Soft Copeland 确定性、无静默回退、Modeler 数值忠实（current_sr 与确定性 base 逐位相等，best_sr/recent_delta 全 null）。
- `SCIENTIFIC_SIGNAL`：Modeler profile 对角色判断与最终 selected-8 产生**大幅、方向可解释**的影响（非 no-op）。
- `NO_SIGNAL`：不适用（信号明确存在）。
- `EVIDENCE_INSUFFICIENT`：离线无法判定该影响是"有益再校准"还是"失稳过度修正"；缺口映射因 salted-hash 不可验证。

---

## 2. 五个核心问题的回答

1. **Modeler 是否产出可靠/可重算的 profile？** → **是**。完全由确定性代码计算的 base 数值驱动；Modeler 仅解释不修改；profile 冻结、可复现、机器校验通过（canonical id、evidence_ids、current_sr 逐位相等、status 合法）。
2. **加入 Modeler 是否改变角色判断？** → **是，大幅**。Pilot 50% 翻转、全 32 37.5% 翻转；progression/critic/novelty 平均绝对差均显著（全 32：2.58 / 0.24 / 4.47）。
3. **是否改变 Soft Copeland 排序与 selected-8？** → **是，剧烈**。selected-8 变 7/8（仅 `0022` 保留），selection_hash 改变，全排序 Spearman −0.21（近负相关）。
4. **变化是否映射到真实 Student 缺口？** → **UNDEFINED / EVIDENCE_INSUFFICIENT**。所有新进入候选 target 为 salted 占位符，形式化映射不可行；仅**语义暗示**（偏向 wood/table/crafting=薄弱前沿，退出 "fantasy/magical" 超纲任务）但**不可验证**。
5. **是否应进入 4096 smoke？** → 见 §3 推荐。

---

## 3. 唯一推荐：`AUTHORIZE_4096_SMOKE`（附强制条件）

**理由**：
- 工程已充分验证（`ENGINEERING_PASS`，OLD 复现原始选择），影子管线端到端确定性可用。
- 存在明确强信号：Modeler 非 no-op，它实质性地重排课程（这正是要去 smoke 验证其价值的对象）。
- 离线已穷尽：影子能回答"是否有影响"（是），但**无法**回答"是否有助学习"——后者按设计只能由 4096-step smoke 回答。这正是本实验作为"门禁"应产出的结论。
- 信号 **valence 不明**（Spearman −0.21 的近反转可能是有益再校准，也可能是有害过度修正）——恰是必须做 A/B smoke 来定夺的理由。

**强制条件（缺一不应启动）**：
1. **A/B 对照**：Modeler 条件 selected-8 **vs** 原始 selected-8，相同 seed、相同 4096 步、相同评测协议；不得单臂。
2. **真值读出**：只以 ground-truth Student 学习指标判定（mean_return、逐 canonical achievement 完成率、death_rate），**不得**用 LLM 判断或 intended-target SR 作为成功标准。
3. **预注册成功判据**：启动前写明何种差异算"Modeler 有益/无效/有害"，避免事后解读。
4. **并行 REQUEST_RAW_DATA（机制验证）**：恢复/重生成 canonical target 映射（解除 salted-hash），并补统一评测/纵向数据——用于验证机制（课程是否对准真实缺口）。此项**不阻塞** smoke 的头条读出，但缺它则无法解释"为什么"。
5. **授权边界**：仅限 4096 步 smoke；任何 24576/98304 或更长训练须再次授权。

**备选（若用户不愿在缺口映射 UNDEFINED 下花费 GPU）**：选 `REQUEST_RAW_DATA`——先解除 salted-hash、补纵向评测，再决定是否 smoke。这是更保守但仍合理的路径；本报告之所以头条推荐 smoke，是因为 smoke 的头条读出（是否有助学习）不依赖 target 映射即可解释，且离线已无其他可判定手段。

`STOP_AND_FIX` 不推荐：工程无失败需修复（salted-hash 是 inherited 数据问题，离线不可"修复"）。

---

## 4. 诚实边界（不得越界）

- **未声称**学生性能提高；**未声称** NEW selected-8 更正确；**未声称**缺口已对齐。
- 7/8 的大变动是"影响力"证据，**不是**"改进"证据；离线无法区分有益 vs 有害。
- 角色判断为 LLM 意见（非可验证事实）；"hallucination=0" 仅对 Modeler 数值忠实度成立。
- target 占位符不可信是 OLD/NEW 共享背景 → old-vs-new 对照**内部效度**仍成立（隔离 Modeler 效应有效），受损的是**外部效度**（课程是否对准真实技能）。

---

## 5. 工程与合规摘要

- **确定性**：Phase 0 已证 pool_hash/selected-8/selection_hash 逐位复现 round1&4；Phase 2 OLD 重算复现原始 round-4 selected-8。
- **无静默回退**：两次 provider 问题均**根因诊断 + 透明修复**：(a) deepseek 推理模型空内容→提高 mtok；(b) glm 30s TTFB 超时→`D052_HTTP_TIMEOUT`（默认忠实）。均非绕过。
- **省 Token 铁则**：批处理（每角色每阶段 1 次）；Modeler profile 复用到 Phase 2（0 额外调用）；未重跑 5×5、未多 seed、未无授权 GPU。
- **Git 合规**：分支 `henry/d052-modeler-shadow-v1`，未 push、未 reset --hard、未 clean -fd、未 force push、未改 shared_r0 原始 artifact。
- **只读/不干预**：未 kill 他人进程、未 pkill/killall；仅精确终止本任务一次卡死自有进程（PID 224350）。
- **凭证**：全程 blind 加载，从未读取/输出任何 key/token。
- **已知缺口**：Phase 2 tutor/critic token 成本因进程在 explorer 卡死被杀、未到写盘行而未持久化（判断本身完整有效）；不做补跑。

---

## 6. 产出清单

```
d052_modeler_shadow_v1/
├── manifests/   canonical_achievement_order.json, pilot8_manifest.json
├── outputs/     student_evidence_base.json, student_profile.json,
│                {tutor,critic,explorer}_judgments.jsonl (pilot 8),
│                full32_{tutor,critic,explorer}_judgments.jsonl (32),
│                llm_cost.json, llm_cost_phase2.json
├── analysis/    round4_full_ranking.json,
│                pilot8_{old,new}_ranking.json, pilot8_stats.json,
│                ranking_delta.csv, judgment_delta.csv,
│                full32_{old,new}_ranking.json, full32_stats.json,
│                full32_ranking_delta.csv, full32_judgment_delta.csv
├── tests/       extract_student_evidence.py, soft_copeland_recompute.py,
│                llm_client.py, preflight.py, modeler_step.py, roles_step.py,
│                analyze_pilot.py, phase2_roles_step.py, analyze_full32.py, diag_glm.py
└── reports/     phase0_audit.md, phase1_pilot.md, phase2_shadow.md, final_report.md
```

---

## 7. 停止等待裁决

Phase 0–2 已完成。按指令，**现停止并等待用户裁决**。请用户在以下三者中择一：
- **`AUTHORIZE_4096_SMOKE`**（本报告推荐，须满足 §3 全部强制条件）
- **`REQUEST_RAW_DATA`**（先解除 salted-hash + 补纵向评测，更保守）
- **`STOP_AND_FIX`**（本报告认为不适用）

未经再授权，不会启动任何训练。
