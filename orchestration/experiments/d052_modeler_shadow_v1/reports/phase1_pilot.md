# D052-V2 Phase 1 报告：8 候选 Pilot 影子过滤

- **任务**：D052-V2 Modeler + Tutor/Critic/Explorer 四角色影子实验
- **分支**：`henry/d052-modeler-shadow-v1`（未 push）
- **目标 cell**：`soft_copeland_x_original` / `seed0_1784462982` / **round 4**（唯一有逐 episode 评测的轮次）
- **输出目录**：`mechanism_UED/orchestration/experiments/d052_modeler_shadow_v1/`
- **状态**：**PHASE_1 = ENGINEERING_PASS**，存在 **SCIENTIFIC_SIGNAL（Modeler 显著影响判断）**

---

## 1. 本阶段做了什么

在固定的 8 候选 pilot 子池上，对**同一批候选、同一份原始 Soft Copeland 代码**做受控对照，唯一变量是角色判断的"学生状态输入"：

| 条件 | 学生状态输入 | 判断来源 |
|------|-------------|----------|
| **OLD** | round-4 原始 sparse `snapshot_str[:200]` | 存档 `round_4/judgment_cache.jsonl`（96 条中取这 8 个的 24 条） |
| **NEW** | Modeler 生成的冻结 `StudentProfile` 摘要 | 本阶段三角色各 1 次批量调用（24 条新判断） |

Pilot 8 候选（`manifests/pilot8_manifest.json`, seed=52，确定性）：
`0027, 0005, 0010, 0000, 0008, 0002, 0022, 0012`
选取规则：2 高排名 + 2 低排名 + 2 最大角色分歧 + 2 技能族多样；覆盖 explore/other/resource/survival/tools 五族；其中 5 个属于原始 selected-8。

---

## 2. 硬门（Phase 1 hard gates）逐条核验

| 门 | 结果 | 证据 |
|----|------|------|
| JSON schema 100% 有效 | **PASS** | 24/24 新判断通过机器校验（task_id∈pilot、role 字段、必需分数为数值、decision∈{accept,hold,reject}） |
| canonical achievements 有效 | **PASS** | Modeler profile 7 skill 全为 canonical id，机器校验通过；角色判断不产出 achievement |
| evidence_ids 存在 | **PASS** | Modeler 每个 skill 含非空 evidence_ids（指向 base 字段路径） |
| 数值可重算 | **PASS** | base 数值由确定性代码计算；Modeler `current_sr` 与 base `completion_rate` 逐位相等（误差<1e-6）；Soft Copeland 重算自检：OLD 与 NEW 各跑两次 → 分数与选择**完全一致** |
| hallucination = 0 | **PASS（限定范围）** | Modeler 反伪造门通过（不修改数值、best_sr/recent_delta 全 null、target-SR 标 UNDEFINED）。角色判断属"意见"而非可验证事实，不计入此项 |
| 无静默回退 | **PASS** | Critic 首次返回空内容（deepseek 推理模型 max_tokens 不足）→ 硬失败、显式记录、调高 mtok 后重跑，无静默替换（见 §5） |
| Soft Copeland 确定性 | **PASS** | 自检 PASS；直接 import 原始 `_aggregate_soft_copeland`，权重 0.34/0.33/0.33/0.01/0.01 |

**结论：无一门失败 → ENGINEERING_PASS。**

---

## 3. 科学信号：Modeler 是否改变角色判断与排序？

**是，且幅度很大。**（以下均为 pilot 8 子池内的受控对照）

| 指标 | 数值 | 含义 |
|------|------|------|
| Spearman(OLD 排序, NEW 排序) | **0.095** | 近似零相关 → 排序几乎被重排 |
| 平均 \|Δprogression\|（0–10 尺度） | **3.37** | 大幅 |
| 平均 \|Δcritic_penalty\|（0–1 尺度） | **0.52** | 极大（半量程） |
| 平均 \|Δnovelty\|（0–10 尺度） | **2.54** | 大幅 |
| decision 翻转 | **12 / 24（50%）** | 半数角色判断改变 accept/hold/reject |
| 子池 top-4 重合 | **2 / 4** | 头部集合显著变化 |

排序对比（按 OLD 子池排名）：

| task_id | tier | OLD#→NEW# | score OLD→NEW | full32# | in原selected8 | OLD dec(T/C/E) → NEW dec(T/C/E) |
|---------|------|-----------|---------------|---------|---------------|----------------------------------|
| 0008 | medium | 1→5 | 1.000→0.339 | 3 | ✓ | accept/accept/accept → accept/**reject**/hold |
| 0027 | easy | 2→8 | 0.965→0.000 | 1 | ✓ | accept/reject/accept → **reject/reject**/hold |
| 0005 | easy | 3→2 | 0.900→0.726 | 2 | ✓ | accept/hold/accept → accept/hold/accept |
| 0022 | easy | 4→1 | 0.726→1.000 | 4 | ✓ | accept/reject/accept → accept/**accept**/accept |
| 0012 | easy | 5→3 | 0.555→0.726 | 6 | ✓ | accept/accept/accept → accept/hold/accept |
| 0002 | easy | 6→6 | 0.481→0.305 | 14 | ✗ | accept/reject/accept → hold/hold/hold |
| 0000 | easy | 7→4 | 0.068→0.634 | 31 | ✗ | accept/accept/accept → reject/accept/accept |
| 0010 | easy | 8→7 | 0.000→0.156 | 32 | ✗ | accept/hold/accept → hold/reject/accept |

### 方向性解读（可解释，但未验证为"更正确"）
- **0027**（描述："附魔武器并在奇幻世界击败魔法敌人"，target_achievements=["A","B"] 占位）：profile 揭示学生处早期（WAKE_UP 已掌握、工具/种植/饮水薄弱、death_rate=1.0），NEW 三角色一致判定其远超学生水平 → progression 8.65→1.0、转 reject/reject/hold。OLD sparse snapshot 给出 progression 8.65/accept，疑似因未传达学生真实状态而**校准失当**。这是 Modeler 价值最清晰的单例。
- **0022**（tools 族）：profile 显示 MAKE_WOOD_PICKAXE 完成率仅 0.0156 → NEW 视为高价值、低惩罚（critic 0.85→0.05）的前沿目标，升至 NEW#1。与 profile 的 curriculum_priorities 一致。
- **0008**（生存/饥饿管理）：NEW tutor 更看好（4.75→9.0）但 critic 大幅加罚（0.0→0.9，反映 death_rate=1.0），explorer 降新颖度 → 净降至 NEW#5。说明 profile 放大了角色间分歧（此候选本就是 max-disagreement 选取）。
- **0000**：OLD critic 近满分惩罚（0.99）在 profile 下被大幅下调（→0.05），novelty 0.65→5.0，升至 NEW#4——提示部分 OLD 惩罚可能过度。

---

## 4. 重要诚实边界（不得越界声明）

1. **不声称学生性能提高。** 本阶段完全离线，无法证明新 selected-8 训练效果更好；只能证明判断"系统性不同且方向上可解释"。
2. **大幅 ≠ 更正确。** NEW 判断可能包含**过度修正**（如 0008 的 critic 0.9 主要由 death_rate 强调驱动）。profile 本身基于单一横截面、仅覆盖 8/32 任务的评测，带有不确定性。
3. **子池 selection_hash 相等是退化结果，不含选择信号。** 在 8 候选池上取 top-8 = 全选，OLD/NEW selection_hash 同为 `cc392cddfcaf5d05` 属构造性必然，**不可**解读为"选择未变"。真正的 selected-8 选择问题须在全 32 池（Phase 2）回答。
4. 角色判断为 LLM 意见，非可验证事实；"hallucination=0" 仅对 Modeler 的数值忠实度成立。

---

## 5. 工程异常与处置（透明记录）

- **Critic 首次调用失败**：deepseek-v4-pro 在 mtok=2500 下三次重试均返回空 content（otok≈42，last_raw_head=""），与预检阶段发现的"推理模型 max_tokens 不足→content 为空"同根因。硬门正确拦截（无静默回退）。处置：将 Critic mtok 提至 4000 后重跑成功（attempts=2, otok=805）。该失败 pass 额外消耗约 itok≈3294 / otok≈42，**未**计入 `llm_cost.json`（被成功 pass 覆盖），此处单独披露。
- conda 环境为 `dicode310`（指令文档中的 `sfl` 不存在，Phase 0 已记录）。

---

## 6. LLM 调用成本（本阶段）

| 角色 | provider/model | attempts | itok | otok |
|------|---------------|----------|------|------|
| Modeler | ds / deepseek-v4-pro | 2 | 4681 | 861 |
| Tutor | qw / qwen-flash-2025-07-28 | 1 | 1043 | 1116 |
| Critic | ds / deepseek-v4-pro | 2 | 2185 | 805 |
| Explorer | gl / glm-4-flash | 1 | 1047 | 664 |
| **合计（记录）** | | | **8956** | **3446** |
| + Critic 失败 pass（披露） | | | ~3294 | ~42 |

逻辑调用数：Modeler 1 + Tutor 1 + Critic 1 + Explorer 1 = **4 次**（在"每角色每阶段≤1 次"预算内；Critic 空响应不计为有效评价）。

---

## 7. 产出文件

- `outputs/student_profile.json`（Modeler 冻结 profile，7 skill）
- `outputs/{tutor,critic,explorer}_judgments.jsonl`（各 8 条新判断）
- `outputs/llm_cost.json`
- `analysis/pilot8_old_ranking.json` / `pilot8_new_ranking.json` / `pilot8_stats.json`
- `analysis/ranking_delta.csv` / `judgment_delta.csv`

---

## 8. 判定与下一步

- **PHASE_1 = ENGINEERING_PASS**（7 项硬门全过）。
- **SCIENTIFIC_SIGNAL = 存在**：Modeler profile 对角色判断与排序产生大幅、方向可解释的影响（Spearman≈0.10，50% 判断翻转）。
- **NO 声明**：未声称性能提升、未声称 NEW 更正确。
- **下一步**：进入 **Phase 2**（全 32 候选影子过滤，已预授权），在完整候选池上检验该影响是否稳定/连贯、是否改变真正的 selected-8 与 selection_hash、是否映射到真实 Student 缺口。Phase 3 训练仍需再授权。
