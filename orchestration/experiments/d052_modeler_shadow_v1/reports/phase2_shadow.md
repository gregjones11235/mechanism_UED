# D052-V2 Phase 2 报告：全 32 候选影子过滤

- **任务**：D052-V2 Modeler + Tutor/Critic/Explorer 四角色影子实验
- **分支**：`henry/d052-modeler-shadow-v1`（未 push）
- **cell / round**：`soft_copeland_x_original` / `seed0_1784462982` / **round 4**
- **状态**：**PHASE_2 = ENGINEERING_PASS**；**SCIENTIFIC_SIGNAL = 强**（Modeler 剧烈改变 selected-8）；**缺口映射 = EVIDENCE_INSUFFICIENT（UNDEFINED）**

---

## 1. 本阶段做了什么

在**完整 32 候选冻结池**上，对同一批候选、同一份原始 Soft Copeland 代码做影子过滤，唯一变量是角色判断的学生状态输入：
- **OLD** = round-4 原始 `judgment_cache.jsonl`（96 条，sparse snapshot 条件）
- **NEW** = 三角色各 1 次批量调用（96 条，冻结 Modeler StudentProfile 条件）
- **Modeler profile 复用 Phase 1 冻结产物**（profile 由 round-4 Student 评测证据构建，与候选池无关 → 0 次新 Modeler 调用，符合"冻结 profile 供所有角色读取"设定）。

---

## 2. 硬门核验（ENGINEERING）

| 门 | 结果 | 证据 |
|----|------|------|
| JSON schema 100% 有效 | **PASS** | 96/96 新判断通过机器校验（task_id∈pool、role、必需分数为数值、decision 合法）；三角色各 32 行 |
| canonical achievements 有效 | **PASS** | 复用 Phase 1 已验证 profile（7 skill 全 canonical） |
| evidence_ids 存在 | **PASS** | 复用 Phase 1 已验证 profile |
| 数值可重算 + 确定性 | **PASS（强）** | pool_hash=`1902b71a5d86fa00`；OLD 与 NEW 各重算两次→完全一致；**OLD 重算精确复现原始 round-4 selected-8**（与 `round4_full_ranking.json` 的 in_original_selected8 完全吻合） |
| hallucination = 0 | **PASS（Modeler 数值忠实度）** | 复用已验证 profile；角色判断为意见不计入 |
| 无静默回退 | **PASS** | glm 超时被**根因诊断**（30s TTFB 不足）并以透明、默认忠实的 `D052_HTTP_TIMEOUT` 修复，非绕过（见 §6） |
| Soft Copeland 确定性 | **PASS** | 自检 + OLD 复现原始选择双重确认 |

**结论：PHASE_2 = ENGINEERING_PASS。**

---

## 3. 科学信号：Modeler 是否改变 selected-8 与排序？

**是，且极其显著。**

| 指标 | 数值 |
|------|------|
| OLD selected-8 | `0005,0008,0012,0016,0022,0024,0026,0027` |
| NEW selected-8 | `0000,0001,0007,0011,0014,0022,0025,0028` |
| selected-8 重合 | **1 / 8**（仅 `0022` 保留） |
| Jaccard | **0.067** |
| selection_hash | `6a285e8f…` → `24cd8eda…`（**改变**） |
| 全 32 排序 Spearman(OLD,NEW) | **−0.21**（近负相关） |
| 平均 \|Δprogression\|（0–10） | 2.58 |
| 平均 \|Δcritic_penalty\|（0–1） | 0.24 |
| 平均 \|Δnovelty\|（0–10） | **4.47**（explorer 变动最大） |
| decision 翻转 | **36 / 96（37.5%）** |

**进入 / 退出：**
- 新进入（7）：`0000,0001,0007,0011,0014,0025,0028`
- 退出（7）：`0005,0008,0012,0016,0024,0026,0027`

---

## 4. 角色消融：谁驱动了选择变化？

逐一将某角色回退为 OLD、保留另两个 NEW，重算 selected-8：

| 消融 | 与 NEW_all 重合 | 与 OLD 重合 | 解读 |
|------|----------------|-------------|------|
| 全 NEW | 8 | 1 | 基准 |
| **回退 critic** | **4** | **3** | **critic 是主导驱动**：回退后 3 个 OLD 成员回归 |
| 回退 explorer | 6 | 1 | explorer 次要（贡献约 2 个成员） |
| 回退 tutor | 7 | 1 | tutor 影响最小（约 1 个成员） |

**结论**：Modeler profile 主要通过**重塑 critic 的风险判断**（结合 death_rate=1.0 与薄弱技能，重新评估"太难/脚手架错误/对新手致命"）来驱动 selected-8 变化；explorer 次之；tutor 最小。

NEW 判断样例（连贯于早期学生 profile）：
- `0027`（退出）：T "Fantasy distraction" reject；C "Enchanting and magical combat too hard" pen=0.90 reject → 正确剔除超纲任务（与 pilot 一致）。
- `0011`（进入）：T "Crafting table foundation" prog=7.5 accept → 指向薄弱前沿 PLACE_TABLE（sr=0.0625）。
- `0000`（进入）：T "High skill-chain value" accept，但 C "deadly for beginner survival" pen=0.90 reject → tutor/explorer 推入、critic 担忧（角色分歧）。

---

## 5. 是否映射到真实 Student 缺口？——EVIDENCE_INSUFFICIENT（UNDEFINED）

学生薄弱技能（sr<0.2）：`MAKE_WOOD_PICKAXE(0.0156)、PLACE_TABLE(0.0625)、COLLECT_DRINK(0.1094)`。

7 个新进入候选的 `target_achievements` **全部是 salted-hash 占位符**，无法形式化映射到 canonical 技能：

| task_id | target_achievements（占位） | 映射 |
|---------|----------------------------|------|
| 0000 | collect_wood, craft_planks | UNDEFINED |
| 0001 | A, B | UNDEFINED |
| 0007 | collect_wood | UNDEFINED |
| 0011 | craft_table, place_table | UNDEFINED |
| 0014 | collect_wood | UNDEFINED |
| 0025 | place_table | UNDEFINED |
| 0028 | collect_wood | UNDEFINED |

按 Phase 0 结论，enhanced cell 用 Python 每进程加盐 `hash()` 映射 target_achievements，**不可逆**，故这些占位符（无论字面像什么）**不可信**。

- **形式化缺口映射 = UNDEFINED / EVIDENCE_INSUFFICIENT。**
- **仅语义暗示（不可验证）**：多数新进入候选的占位符字面指向 wood/table/crafting——恰好是学生薄弱前沿；退出者含 "fantasy/magical" 超纲任务。这与 profile 的 curriculum_priorities 方向一致，但**因 salted-hash 不可信，不得作为已验证的缺口对齐**。
- 注意：占位符的不可信是 OLD 与 NEW **共享**的背景（两者读取相同 target），故 old-vs-new 对照对"隔离 Modeler 效应"仍内部有效；受损的是**外部效度**（结果课程是否对准真实技能）。

---

## 6. 工程异常与处置（透明）

1. **glm-4-flash 32 条批量超时**（本阶段主要工程问题）：
   - 现象：explorer 32 条批量调用反复失败、进程卡死约 34 分钟。
   - 诊断（单次有界探针）：`WALL=35.6s, ok=False, err=max retries, CONTENT_LEN=0` → 命中 `api()` 硬编码 **30s** HTTPSConnection 超时。glm 经此 paas 端点对大请求为**非流式缓冲**，TTFB≈完整生成时间（32 条≈2600 token，约 60–90s）>30s。原始 launcher 逐候选（小请求）从不触发；指令要求的批处理才暴露。
   - 修复：给 `llm_client.py` 加 `D052_HTTP_TIMEOUT` 环境变量（**默认 30s=忠实**），explorer 重跑设 180s → **attempt=1 成功**（otok=1775）。同时加 `D052_MAX_ATTEMPTS`（默认 48=忠实）以给重跑加有界保险。
   - 期间精确终止了本任务自有的一次卡死进程（PID 224350，非他人进程、非模糊匹配）。
2. **Phase 2 tutor/critic token 成本未持久化**：首次运行在 explorer 卡死、被超时信号杀掉，未到 `llm_cost_phase2.json` 写盘行；孤儿进程 likewise。最终成功重跑幂等跳过 tutor/critic，故 `llm_cost_phase2.json` **仅含 explorer**（itok=2446, otok=1775, attempts=1）。tutor/critic 的 phase2 token 数**丢失**，但二者判断完整有效（各 32 行、已验证）。不做补跑（避免翻倍 LLM 调用）。

---

## 7. LLM 调用成本

| 角色 | 阶段 | provider/model | attempts | itok | otok | 备注 |
|------|------|---------------|----------|------|------|------|
| Modeler | P1 | ds/deepseek-v4-pro | 2 | 4681 | 861 | profile 复用到 P2 |
| Tutor | P1 | qw/qwen-flash | 1 | 1043 | 1116 | |
| Critic | P1 | ds/deepseek-v4-pro | 2 | 2185 | 805 | +失败 pass ~3294/42 |
| Explorer | P1 | gl/glm-4-flash | 1 | 1047 | 664 | |
| Tutor | P2 | qw/qwen-flash | — | **未持久化** | **未持久化** | 判断有效 |
| Critic | P2 | ds/deepseek-v4-pro | — | **未持久化** | **未持久化** | 判断有效 |
| Explorer | P2 | gl/glm-4-flash | 1 | 2446 | 1775 | 180s 超时修复后成功 |

逻辑调用：P1 = 4 次；P2 = 3 次（Modeler 复用 0 次）。均在"每角色每阶段≤1 次"预算内。

---

## 8. 产出文件

- `outputs/full32_{tutor,critic,explorer}_judgments.jsonl`（各 32 条）
- `outputs/llm_cost_phase2.json`（仅 explorer，见 §6.2）
- `analysis/full32_stats.json`、`full32_old_ranking.json`、`full32_new_ranking.json`
- `analysis/full32_ranking_delta.csv`、`full32_judgment_delta.csv`

---

## 9. 判定

- **PHASE_2 = ENGINEERING_PASS**（硬门全过，OLD 复现原始选择强锚定）。
- **SCIENTIFIC_SIGNAL = 强**：Modeler 使 selected-8 变 7/8（Jaccard 0.067）、selection_hash 改变、全排序 Spearman −0.21、37.5% 判断翻转；主导驱动为 critic。
- **EVIDENCE_INSUFFICIENT**：缺口映射 UNDEFINED（salted-hash）；离线**无法**判定该剧变是"有价值的再校准"还是"失稳的过度修正"——二者 offline 不可区分。
- **不声称**：学生性能提升、NEW 更正确、缺口已对齐。
- 下一步见 `final_report.md` 的唯一推荐。
