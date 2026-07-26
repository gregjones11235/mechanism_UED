# D052-V2 Phase 2.5 报告：匹配反事实审计 + Salted-Hash 数据流审计

- **任务**：D052-V2 Modeler + Tutor/Critic/Explorer 四角色影子实验
- **分支**：`henry/d052-modeler-shadow-v1`（未 push）
- **cell / round**：`soft_copeland_x_original` / `seed0_1784462982` / **round 4**
- **本阶段指令**：暂不启动 4096 训练；先做匹配反事实审计，排除 OLD/NEW 之间的模型/prompt/provider/配置漂移，严格验证选择变化是否由 Modeler 引起；同时做 salted-hash 静态数据流审计。
- **状态**：**PHASE_2.5 = ENGINEERING_PASS**；**Modeler 净效应 = 已证实（4/8 变化，受控隔离）**；**裁决 = `STOP_AND_FIX_CANONICAL_TARGETS`**（salted-hash 影响训练语义，优先级最高）。
- **训练状态**：**未启动、不会启动任何训练**，停止等待裁决。

---

## 0. 裁决（先给结论）

**`NEXT = STOP_AND_FIX_CANONICAL_TARGETS`**

依据用户裁决规则的优先级：
> 若 salted hash 影响训练语义：NEXT=STOP_AND_FIX_CANONICAL_TARGETS

Phase 2.5a 的静态数据流审计**确证** salted-hash 进入 **reward / termination / success / loss**（非仅报告标签）。该条件一旦成立，**无论 B/C 选择差异大小，裁决都为 STOP_AND_FIX**。

补充：B/C 反事实本身**也满足**"变化有证据支持"的条件（selected-8 变化 4/8 ≥ 2/8，且为受控隔离的纯 Modeler 效应）——即若没有 salted-hash 问题，本会推荐 `AUTHORIZE_4096_SMOKE`。但 salted-hash 问题使其不可执行：在加盐随机奖励函数上训练，任何课程质量结论都不可解释。

---

## 1. Phase 2.5b：B/C 匹配反事实（受控隔离 Modeler 净效应）

### 1.1 设计（严格匹配，唯一变量 = 是否附加 StudentProfile）

| 维度 | B_NO_MODELER | C_WITH_MODELER | 是否相同 |
|------|--------------|----------------|----------|
| 候选池 | 同一 32 候选 | 同一 32 候选 | ✅ |
| 候选顺序 | sorted(task_id) | sorted(task_id) | ✅ |
| 匿名 ID | C001..C032（同一映射） | C001..C032（同一映射） | ✅ |
| 学生原始数值摘要 | 确定性代码计算 | **完全相同** | ✅ |
| Modeler StudentProfile | **不提供** | 附加 Phase-2 冻结 profile | ❌（唯一差异） |
| Tutor/Critic/Explorer prompt 模板 | 同一 | 同一 | ✅ |
| 模型/provider | qw/ds/glm | qw/ds/glm | ✅ |
| temperature / top-p / timeout / schema | 0 / 默认 / 180s / 同一 | 同 | ✅ |
| Soft Copeland 代码 | 原始 `_aggregate_soft_copeland` | 同一 | ✅ |
| 判断来源 | **本次新调用**（非历史 D052） | 本次新调用 | ✅（满足"禁止用历史判断作 B 臂"） |

- **原始数值摘要（B、C 完全相同）**：仅来自 `student_evidence_base.json` 的确定性数字——episode_level 统计、逐 achievement 经验完成率(含 n)、确定性技能链 frontier、低完成断点、证据边界。**不含**任何 Modeler 解释字段（无 status/MASTERED、无 confidence、无 curriculum_priorities、无 uncertainties）。
- **C 臂** = 上述原始摘要 + 冻结 StudentProfile JSON。
- **调用预算**：每角色每臂 1 次批量调用，共 **6 次**（恰好达到上限，无超额）。

### 1.2 工程硬门（ENGINEERING_PASS）

| 门 | 结果 | 证据 |
|----|------|------|
| pool_hash 锚定 | **PASS** | `1902b71a5d86fa00`，与 Phase 2 完全一致（同一冻结池） |
| schema 100% 有效 | **PASS** | B、C 各 96/96 判断通过机器校验（匿名 ID 全映射、分数数值、decision 合法） |
| 确定性自检 | **PASS** | B、C 各自重算两次→selected-8 与分数逐位相同 |
| 无静默回退 | **PASS** | glm role 回显怪癖经**透明确定性修复**（见 §1.5），非绕过 |

### 1.3 各角色 B/C 分数与排序相关性

| 角色 | 信号 | Spearman(秩) | Pearson(值) | 平均\|Δ\| | 均值 B→C |
|------|------|-------------|-------------|-----------|----------|
| Tutor | progression | **0.145** | 0.099 | **5.33** | 6.84 → **1.52** |
| Critic | critic_penalty | 0.818 | 0.697 | 0.18 | 0.60 → 0.70 |
| Explorer | novelty | 0.580 | 0.197 | 0.31 | 7.09 → 6.78 |

**判读**：
- **Tutor 是最大判断移动者**：附加 profile 后，progression 均值从 6.84 暴跌到 1.52（平均\|Δ\|=5.33/10），秩相关仅 0.145（近独立）。profile 提供的"早期学生、死亡率高、技能链薄弱"信息使 tutor 大幅下调几乎全部候选的教学进阶价值。
- Critic 秩相关高（0.818）但均值上移（更严苛），绝对移动小。
- Explorer 移动最小。

### 1.4 judgment 翻转率与全排序

- **decision 翻转：34/96（35.4%）**，按角色：tutor **24**、critic 10、explorer **0**。
- **全 32 排序 Spearman(B,C) = 0.437**（中等正相关——B/C 比 Phase 2 的 OLD/NEW 更相似，因为本设计控制了 prompt/配置漂移）。

### 1.5 selected-8 变化、overlap、Jaccard、替换任务、selection_hash

| 指标 | 数值 |
|------|------|
| B selected-8 | `0000,0006,0007,0011,0014,0022,0025,0028` |
| C selected-8 | `0000,0003,0010,0014,0017,0024,0025,0028` |
| **变化数** | **4 / 8**（≥2/8 阈值 → "变化有证据支持"成立） |
| overlap | 4（`0000,0014,0025,0028`） |
| Jaccard | **0.333** |
| selection_hash | `82571538e5299ea9` → `868a57268d66b90b`（**改变**） |
| 进入（C 独有） | `0003,0010,0017,0024` |
| 退出（B 独有） | `0006,0007,0011,0022` |

### 1.6 角色各自贡献（消融：C 中将单一角色回退为 B）

| 消融 | 与 B 重合 | 与 C 重合 | 解读 |
|------|----------|----------|------|
| 回退 tutor→B | 4 | 5 | tutor 有贡献 |
| 回退 critic→B | 4 | 5 | critic 有贡献 |
| 回退 explorer→B | 4 | 3 | explorer 有贡献 |

**判读**：与 Phase 2（critic 单一主导）不同，本受控隔离下**三个角色的贡献是弥散的**（每个回退都把 overlap_with_B 提到 4）。没有单一主导角色；tutor 在**判断层**移动最大（§1.3/1.4），但最终选择是三角色合力。

### 1.7 与 Phase 2 OLD/NEW 的对比（为何 2.5 更严格）

| | Phase 2（OLD vs NEW） | Phase 2.5（B vs C） |
|---|---|---|
| B/OLD 臂来源 | 历史 round-4 判断（sparse snapshot 条件） | **本次新调用**（原始数值摘要条件） |
| 控制的变量 | 仅"是否 profile"，但 OLD 臂含 prompt/条件漂移 | **同模板/同模型/同摘要**，唯一差异=profile |
| selected-8 变化 | 7/8（Jaccard 0.067） | **4/8（Jaccard 0.333）** |
| 主导角色 | critic | 弥散（tutor 判断层最大） |

**结论**：Phase 2 的 7/8 里有一部分来自 OLD 臂的 prompt/条件漂移；**真正由 Modeler profile 隔离出的净效应是 4/8**。这仍 ≥2/8 阈值、方向可解释（profile 使课程更保守、更指向早期生存/合成），故"变化有证据支持"成立——但幅度比 Phase 2 表象更温和、更可信。

### 1.8 token 与重试

| 调用 | provider/model | attempts | itok | otok |
|------|---------------|----------|------|------|
| B_tutor | qw/qwen-flash | 1 | 2470 | 2328 |
| B_critic | ds/deepseek-v4-pro | 3 | 7517 | 5623 |
| B_explorer | glm-4-flash | 1 | 2452 | 1776 |
| C_tutor | qw/qwen-flash | 1 | 3337 | 2298 |
| C_critic | ds/deepseek-v4-pro | 2 | 6734 | 4043 |
| C_explorer | glm-4-flash | 1 | 3319 | 1747 |
| **合计** | 6 次调用 | **9** | **25829** | **17815** |

（B_critic 3 attempts、C_critic 2 attempts 为 deepseek 推理模型偶发重试，最终成功；非静默回退。）

### 1.9 工程异常与透明处置

- **glm-4-flash role 回显怪癖（B、C 两臂 explorer 均触发）**：glm 返回完整有效的 32 条数组（novelty/diversity 分数、decision 全部合法、32 个匿名 ID 全映射），但把其中 **9 条**（C005/006/012/013/019/020/022/026/027，两臂完全相同的 9 个）的自报 `role` 字段错写成 `'builder'`/`'survivor'`。
  - **处置**：`role` 仅为模型回显的自标签，**不进入 Soft Copeland 信号**（信号按 progression/critic_penalty/novelty 分数键读取）。做**确定性透明修复**：仅将这 9 条的 `role` 回显规范化为 `explorer`，保留原始错标值于 `original_role_echo` 与 `bc_{arm}_explorer_normalization_log.json`，所有分数/决策/理由**逐字不改**，重过硬校验门后写盘。
  - **0 次新 LLM 调用**（若重跑将使总数达 7，违反"最多 6 次"）。两个 `bc_{arm}_explorer_FAILED_raw.json` 作为修复来源保留备查。
  - 这是规范化一个下游不消费的装饰字段，**非数值伪造、非静默回退**；硬门正确捕获并记录了它。

---

## 2. Phase 2.5a：Salted-Hash 静态数据流审计（确证影响训练语义）

### 2.1 加盐位点

- **文件/行**：`workers/gpu0_original/gpu0_training_mechanisms/scripts/launch_d052_pure_dynamic_enhanced.py:818`
- **代码**：`ta = [_all_a[hash(f"{cid}_{a}") % len(_all_a)] for a in ta_names]`
- **问题**：Python 字符串 `hash()` 每进程加盐（PYTHONHASHSEED），映射**不可逆**且跨运行不一致。
- **存储**：`launch_..._enhanced.py:832` `self.relevant_achievements = list(__ta)`；候选类 `_CT`（L826-866）**不覆写** get_reward/is_terminal/is_success。

### 2.2 进入训练语义的调用链（具体文件/函数/行号）

```
launch_d052_pure_dynamic_enhanced.py:1168  task_classes = make_task_classes(sel_cands)
launch_...:1169  task_embeddings = jnp.eye(len(task_classes))   # 任务索引 one-hot（非加盐 achievement）
launch_...:1180  make_train_with_treatments(cfg, task_classes, UPD, task_embeddings=_te, ...)
  → dicode/training/integration.py:37 → dicode/ppo_tr.py:39 make_train
  → dicode/ppo_tr.py:71-121  MultiTaskMiniCraftaxEnv[R](task_classes,...)
  → minicraftax/envs/multitask.py:324  self.reward_fns   = tuple(t.get_reward   for t in tasks)
  → minicraftax/envs/multitask.py:325  self.terminal_fns = tuple(t.is_terminal  for t in tasks)
```

### 2.3 受影响的训练语义（逐条证据）

| 方面 | 文件:行 | 函数 | 关键代码 | 进入 loss? |
|------|---------|------|----------|-----------|
| **REWARD** | `minicraftax/tasks/base_task.py:52-62` | `BaseTask.get_reward` | `mask=zeros.at[relevant_achievements].set(1); reward=(achievement_delta*mask*ACHIEVEMENT_REWARD_MAP).sum()` | ✅（PPO 回报） |
| **TERMINATION** | `base_task.py:28-50` | `BaseTask.is_terminal` | `task_solved=all(current_achievements_bool[relevant_indices]); return done\|is_dead\|defeated_boss\|task_solved` | ✅（截断回报） |
| **SUCCESS** | `base_task.py:69-81` | `BaseTask.is_success` | `task_solved=all(current_achievements_bool[relevant_indices])` | ✅（评测/指标） |
| **SUCCESS_MASK** | `dicode/setup.py:300-319` | `_create_achievement_masks` | `task_achievement_mask.at[i, relevant_achievements].set(True)` → `scoring.py:22`, `evolution.py:321-339` | ✅ |

调用者：reward 经 `minicraftax/envs/base.py:519,682`；termination 经 `base.py:60,225,387,550`。

### 2.4 Observation conditioning（未直接加盐，但路由仍受影响）

- launcher 传 `task_embeddings=jnp.eye(n)`（任务索引 one-hot），`ppo_tr.py:68` 使用该嵌入。
- 加盐的 relevant_achievements multi-hot 路径（`training/__init__.py:335-337` 的 `_generate_embeddings_for_session` one-hot 分支）**未被本 launcher 使用**。
- **但**：任务索引仍路由到一个其 reward/termination 被加盐的任务——故策略仍在学习加盐随机奖励。

### 2.5 eval pilot 自证

- `single_director_20260722/d052_eval/d052_eval_pilot.py:132,139`：`SUCCESS_MODE = "UNDEFINED"  # enhanced: relevant_ach via salted hash(), unrecoverable`。
- 评测脚本**自己**把 success 标为 UNDEFINED。注意：Modeler 使用的逐 achievement 经验完成率是真实/canonical、**未加盐**的——故 Modeler 的证据基础本身可信；受损的是候选 target→训练奖励的映射。

### 2.6 后果

`hash()` 每进程加盐 → `relevant_achievements`（从而每个任务的奖励函数、终止、成功）是**事实上随机的 achievement 分配**，与 LLM 所判断的 spec target_achievements 标签**解耦**。课程选择（基于标签）与训练奖励/终止/成功（基于加盐 hash）**解耦**。任何在这些 enhanced 候选上的训练（含 4096 smoke）都在加盐随机奖励函数上训练，使课程质量结论**不可解释**，直到 target 被规范化为 canonical。

---

## 3. 裁决与下一步

### 3.1 裁决规则核对

| 规则条件 | 本阶段结果 | 触发? |
|----------|-----------|-------|
| B/C selected-8 变化 ≥2/8 且变化有证据支持 | 4/8 变化，受控隔离、确定性、方向可解释 | ✅ 满足 |
| **且** salted hash 不影响训练语义 | **影响**（reward/termination/success/loss） | ❌ 不满足 |
| → 若 salted hash 影响训练语义 | 确证影响 | **✅ 触发 STOP_AND_FIX** |
| 若缺代码或原始数据 | 代码与数据齐备 | 不触发 |

**优先级**：salted-hash 条件凌驾于 B/C 幅度之上。

### 3.2 `NEXT = STOP_AND_FIX_CANONICAL_TARGETS`

**含义**：在做任何训练（含 4096 smoke）之前，必须先把候选的 `target_achievements` 从加盐 `hash()` 占位符**规范化为 canonical achievement 映射**，使 reward/termination/success 与 LLM 所判断的标签一致。否则训练读出不可解释。

**建议的修复方向（供用户裁决，本阶段不执行）**：
1. 用确定性、可逆、跨进程稳定的映射（如 canonical achievement 顺序的显式索引表，或固定 PYTHONHASHSEED + 显式枚举）替换 `hash(f"{cid}_{a}") % len(_all_a)`。
2. 重生成 enhanced 候选的 `relevant_achievements`，使 `is_terminal/is_success/get_reward` 作用于真实目标。
3. 修复后重跑 eval pilot，使 `SUCCESS_MODE` 从 UNDEFINED 变为可计算。
4. 而后再回到本影子实验的 B/C 结论（Modeler 净效应 4/8 已证实）决定是否 `AUTHORIZE_4096_SMOKE`。

### 3.3 诚实边界

- **未声称**学生性能提升；**未声称** C 臂 selected-8 更正确。
- B/C 的 4/8 变化是"Modeler 有真实净效应"的证据，**不是**"改进"证据；valence（有益/有害）离线不可判，且当前因 salted-hash 即便训练也不可解释。
- 角色判断为 LLM 意见；"hallucination=0" 仅对 Modeler 数值忠实度成立。
- glm role 回显修复仅触及下游不消费的装饰字段，已完整记录。

---

## 4. 合规摘要

- **省 Token 铁则**：B/C 共 **6 次**批量调用（恰达上限，无超额）；无逐候选调用；无子代理/background agent；无重跑 5×5；无多 seed。
- **禁止项遵守**：未启动任何训练；未 kill 他人进程、未 pkill/killall；未读/输出任何 key/token（blind 加载）；未复制大 artifact 到本地（仅小型 JSON/CSV/源码）。
- **Git 合规**：分支 `henry/d052-modeler-shadow-v1`，未 push、未 reset --hard、未 clean -fd、未 force push、未改 shared_r0 原始 artifact。
- **无静默回退**：glm role 回显怪癖经透明确定性修复并记录；deepseek 重试为模型偶发、最终成功。

---

## 5. 产出文件

```
d052_modeler_shadow_v1/
├── outputs/  bc_{B,C}_{tutor,critic,explorer}_judgments.jsonl (各 32),
│             bc_{B,C}_explorer_FAILED_raw.json (修复来源),
│             bc_{B,C}_explorer_normalization_log.json,
│             llm_cost_phase25.json (6 调用)
├── analysis/ salted_hash_audit.json,
│             bc_stats.json, bc_{B,C}_ranking.json,
│             bc_judgment_delta.csv, bc_ranking_delta.csv
├── tests/    phase25_bc_step.py, phase25_repair.py, analyze_bc.py
└── reports/  phase2_5_counterfactual.md (本报告)
```

---

## 6. 停止等待裁决

Phase 2.5（a + b）已完成。按指令，**现停止并等待用户裁决，不启动任何训练**。

裁决建议：**`STOP_AND_FIX_CANONICAL_TARGETS`**——先规范化候选 target 映射（解除 salted-hash），修复 eval success 可读性，再决定是否授权 4096 smoke。Modeler 的受控净效应（4/8）已在本阶段证实并归档，可在修复后直接复用。
