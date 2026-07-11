# Threshold Probe 结果与瓶颈定位：脚手架泄漏（Scaffolding Leakage）

> 2026-07-09。probe 臂 = +A+B + `mastery_threshold=0.2`（wandb `oun2yfm6`，改名 `AB_threshold02_probe`）。
> 对照 = u1gjqror（同配置，threshold 0.60），同 seed=1 同环境单 flag 差。
> 数据：`experiments_mason/eval/eval_PROBET02_seed0.json`、`run_probe_t02.log`、probe task_graph.graphml。
> **本文档结论：机制臂 ~30 平台的成因最终定位为"生成任务的脚手架泄漏"——且全量审计（302 任务/4 臂）显示泄漏率 100%（含无定向 baseline）：脚手架是本范式的固有生成模式，非机制诱发。**

---

## 1. 五臂终表（官方口径，mean_return，seed 0 同批 1024 世界）

| update (steps) | baseline_ext | +A_ext | +A+B (t=0.6) | **probe (t=0.2)** |
|---|---|---|---|---|
| 300 (39M) | 12.75 | 12.20 | 13.46 | 13.19 |
| 600 (79M) | 17.24 | 18.32 | 17.81 | 16.70 |
| 900 (118M) | 10.84 ▼ | 18.15 | 18.54 | 18.75 |
| 1200 (157M) | 18.27 | 20.00 | 20.25 | **23.19** |
| 1500 (197M) | 18.76 | 12.94 ▼ | 29.16 | 26.38 |
| 1800 (236M) | 18.74 | 21.34 | 28.59 | 28.71 |
| 2100 (275M) | 17.66 | 28.93 | 32.46 | 29.61 |
| 2400 (315M) | 17.88 | 31.07 | 29.81 | **29.06** |

（一代 157M 三臂另存 v1；五臂身份：mc75k0nx / z8jygtyw / u1gjqror / oun2yfm6 + 32v02vi9/85qid2ev。）

## 2. Probe 过程指标（预注册判据逐条对账）

| 判据 | 结果 |
|---|---|
| frontier 解锁 tier 3？ | ✅ session 3 首现；session 1-4 短暂 2↔3 振荡后 **22/25 session 稳定 tier 3** |
| preflight 拒绝率变化？ | **0/76 全收**（0.6 臂为 2/78）——见 §4，全收本身成为证据 |
| 平台抬升？ | ❌ 终点 29.06，与 0.6 臂 29.81 / +A 31.07 同噪声带——**三种机制配置收敛 29-31** |
| tier-3 技能非零？ | ❌ 调度 targets（defeat_lizard / learn_fireball / enter_vault / gnome_warrior）**20 个定向 session 后仍全 0**；仅 make_diamond_sword 0.6 |
| 副作用（tier-2 回撤）？ | 无实证：probe iron 2.7/2.6 vs 正当对照 ARMAB 4.6/2.8（同带）；tier-2.5 照常巩固至 61-66；钻石资源非零（7.4/5.9/4.7）。+A_ext 的 iron 28.9/39.2 为无闸臂独立高点（n=1 方差带）——机会成本仅 hint |
| 中段效应 | ✅ 1200 处 23.19 vs 两个 0.6 臂 20.0/20.3——早解锁买到中段加速 |

其他：零故障；轨迹为五臂最平滑之一（300→2100 单调，末点 -0.5）；振荡期为事实上的 tier-2/3 混合课程（方案二的免费近似）。

## 3. ★ 瓶颈定位：脚手架泄漏（实锤）

**证据链**（每环第一手）：
1. 调度器忠实指向 tier 3（22/25 session，targets 为真身枚举名，含 `defeat_orc_solider` 源码拼法）；
2. 14B 生成的任务代码**系统性内建脚手架**。probe task_graph 抽查实录（14B 自写注释原文）：

```python
# task_19 "MAKE_DIAMOND_SWORD" — generate_world():
# --- ADDED SCAFFOLDING ---
# 1. Give a stone pickaxe and 50 coal for crafting
builder.set_player_inventory({"pickaxe": 1, "coal": 50})
# 2. Place lizards as combat practice
builder.add_mobs_randomly_near(..., min_dist=4, max_dist=8, ...)
# --- END SCAFFOLDING ---
```
   并且 `completed_achievements` 预标记前置链（task_20 一次预标 **14 个**前置成就）——前置依赖被代码级跳过；
3. 任务因此可解 → **preflight 全收**（sr≥0.05 恰恰由脚手架制造）；
4. policy 学会"持现成资源做最后一步"→ 裸 held-out 需从空手走完整链 → **技能不迁移** → 平台。

### 3.1 ★ 全量审计（302 任务 / 4 臂）：泄漏率 100%，范式固有

静态 AST 审计（`scaffold_audit.py`，四签名：S1 送库存 / S2 预标成就 / S3 近距刷怪 min_dist≤8 / S4 起始层>0）：

| 臂 | 生成任务 | 泄漏率 | S1/S2/S3/S4 | 预标均数 |
|---|---|---|---|---|
| probe (t=0.2) | 74 | **100%** | 73/74/72/58 | 4.4 |
| ARMAB (t=0.6) | 68 | **100%** | 66/64/68/34 | 4.4 |
| ARMA_ext | 79 | **100%** | 79/77/78/41 | 4.6 |
| **BASE_ext（无定向）** | 81 | **100%** | 80/75/76/51 | 4.2 |

各深度分桶（tier0 到 tier3）**全部 100%**。

**因果修正**（推翻本文初版的"定向诱发"假说）：baseline 无任何 target 注入、tier-0 任务（收木头级）同样 100% 泄漏 → **脚手架不是对"够不着的目标"的压力反应，是 14B 在本管线 prompt 范式下的默认生成形态**。

**溯源已结案（2026-07-10，双重教学实证）**：
1. **few-shot 教的**：四个种子任务（src/minicraftax/tasks/seed_tasks/*.py，模型看到的全部示例）每一个都包含 set_player_inventory + completed_achievements 预标 + add_mobs_randomly_near；
2. **指令明文教的**：persona prompt 原文——`persona_ambitious_coop.py`: "aim forward **WITH scaffolding**, NEVER 'unsolvable'"、"If the target needs prerequisites the agent lacks, **SCAFFOLD them**: provide the intermediate tools/resources/floor... and **list them as Completed Achievements**"；`persona_feasible.py` / `evolve_mastered_r.py` 同类措辞。S1/S2/S4 三种签名被逐字规定。
→ **责任归属修正：14B 无罪，脚手架是 DiCode 范式的显式设计意图**；缺陷不在"坏示范"，而在范式未闭合的假设（见核心命题）。修复语义 = 补全缺失的另一半（递减/验证），而非删除 scaffold 哲学（它保障可学性，有其道理）。
唯一非饱和信号：S4 起始层跳跃随定向深度上升（probe 78% vs ARMAB 50%）——"深度→更重脚手架"仅存此残余证据。

**随之必须回答的反问：泄漏率同为 100%，机制臂凭什么 +12 分？** 答：泄漏率相同，但**脚手架残余的迁移价值不同**。机制臂的任务终端瞄准 tier-2/3 技能（probe 63/74 为 tier-3 定向）——即便前置链被跳过，policy 至少反复练习了**最后一步**（打蜥蜴/开箱/射箭）；短链技能（tier-2.5：dungeon/bow/chest，脚手架后仅剩一两步）因此迁移成功（→60%+），长链技能（tier-3，被跳过的部分太长）迁移失败（→0）。baseline 的任务连终端都未瞄准深层。**机制的 +12 分 = 终端技能定向的收益；脚手架吃掉的 = 前置链的学习。** 这同时解释了 tier-2.5 成功与 tier-3 失败的分界线位置。

**升维含义**：本管线（含全部五臂、且很可能含同范式先行工作）生成的从来不是"裸课程"，而是 100% 脚手架化课程——所有成绩都是**脚手架课程的迁移残余**。fidelity gate 由"补丁"升格为"必需品"。

### 3.2 ★ 裸复验（C-2 动态层原型）：三个 regime，与迁移模式逐点对齐

工具 = `bare_reverify.py`（剥除 S1 送库存 + 清空 S2 预标，保留刷怪/起始层；冻结 probe 终版 checkpoint(2400)，同 rng 配对 rollout；胶水全部复用 eval_checkpoints / evaluate_new_tasks 生产路径）。10 任务两轮结果：

| regime | 代表任务 | SR(原版)→SR(裸版) | 含义 |
|---|---|---|---|
| **已内化** | task_14 / task_8（tier-2.5 族） | 0.76→0.70 / 0.95→0.58 | 技能真学会 = **迁移成功的那批** |
| **脚手架承重** | task_25 | **0.65→0.01（崩塌 0.64）** | 表现是借来的 = **迁移断裂的位置** |
| **给脚手架也不会** | task_17/19/20（钻石剑族） | ~0→~0 | **准入时"可学"≠预算内"可掌握"**（tier-3 长链，连脚手架版都未学会） |

两个独立测量（裸复验 vs held-out 迁移）讲同一个故事 → 裸复验 = 逐任务"脚手架依赖度"测量仪，**C-2 动态层原型已验证可用**。局限：单 checkpoint、每桶 2-3 任务、任务按关键词选取较粗。（task_25 的 0.65 注：relevant 为双成就 DIAMOND_SWORD+DEFEAT_ORC_SOLIDER，高分来自脚手架装备下的战斗半；剥除后两半全崩。）

### 3.3 与主线 v6fix7 的关系（分层防御）

主线已有直接尝试：v6fix7 的 SiegeLevelValidator（已实现+测试+接入 gen_manager），哲学 = 不禁脚手架、保护"焦点技能的直接执行链"（真采真熔真造：焦点及直接前置禁入 Completed）。**关键差异：它守在 docstring 层（声明式、LLM 自查），本线审计在代码层（AST 实测）**——docstring 干净不保证代码干净（S1/S3/S4 完全不查，S2 仅选择性保护），该假设无人验过。
→ 防御纵深三层两线正好拼齐：**docstring 语义规则(v6fix7) / 代码 AST 审计(scaffold_audit) / 行为层裸复验(C-2)**——联合论文"layered defense"一节的骨架。
→ 现成跨线实验：用 scaffold_audit 审主线 fix7 前后 task graph，一条命令量化 docstring 防御的代码层效果（待与 Alec 对接）。
另注：v6fix7"保护直接链"的设计选择恰被本线断裂带数据背书（短链迁移成功、长链失败——保的正是能活下来的那段）。

**两条正交质量轴**（本轮最重要的概念产出）：
- **可解性（learnability）**：当前 policy 能否推进——编译校验、C-0 lint、preflight 全部只覆盖此轴；
- **忠实性（fidelity）**：任务是否真的要求它声称的技能——**现有全部闸门零覆盖**，泄漏即由此穿过。

**跨尺度呼应**：v6 线（235B）文档记录的同一风险（"LLM 可能直接把成品铁镐塞进初始库存"，v6fix7 以代码硬闸防御）——本线在 14B 上以五臂消融独立复现并定位 → **脚手架泄漏是 code-level UED 的尺度无关缺陷**（联合论文 discussion 素材）。

**公允注记**：LLM 的行为符合人类教学直觉（递工具练最后一步）；缺陷在于无任何机制传达"评测是裸的/脚手架须递减"。修复方向因此不是禁止脚手架，而是约束或验证它（§5）。

## 4. Preflight（★B）终版定位

机制审计结论（代码级）：B-1 写入的 `learnability_score` **全管线零读取方**（采样与激活均读 `priority_score`；training.py 的 learnability 分支与 task_utils 的复活逻辑均从 `latest_sr` 现算，与 B 无关且默认休眠）。故本管线中 **+A+B ≈ +A + ~10min/设计session + rng 偏移**。

据此校准：
- "path quality（-0.6 vs -7.1 回撤）" 降级为 **observed but unattributed**（n=1；同 seed 方差 ±1 分；probe 零干预却最平滑，进一步削弱因果归因）；
- 保留的主张：**拒必有理**（2/2 真 too_easy）、**廉价**（~10min）、**保险角色**（为激进 frontier 推进封顶下行——probe 即受保运行，零赔付≠零价值）；
- **白送的改进项**：将 PLR 采样接入 B 实测的 learnability 字段（"只写不读"→"闸+信号源"），config 级改动，Stage-2 相邻工程。

> **注意：以上校准均不改变 B 的机制地位** —— B 仍是可学性轴的唯一闸门，且 §5 的 C-2 动态检测直接复用 B 的 rollout 机器（"裸初始条件下重跑 preflight"）。三闸终局分工：**C-0 保代码真、B 保可学、C-2 保忠实**——各管一条正交质量轴，互补而非替代；C-2 上线后 B 反而更关键（剥除脚手架后的裸任务更难，正需 B 判定可学边界）。

## 5. 移交：★C-2 / 方案二的精确规格

平台成因既定为忠实性缺陷，下一机制的规格随之明确（候选，待排序）：

- **prompt 溯源（新增，最便宜候选）**：grep 种子任务与 evolve/craftax_coder prompt 是否示范了 `set_player_inventory` / `completed_achievements` 预标模式——若是，"14B 学坏"实为"prompt 教的"，修 prompt 即可能大幅降泄漏，成本近零；
- **C-2 忠实性闸（fidelity gate）**：候选任务在**裸初始条件**下复验——静态层：`scaffold_audit.py` 已就绪（四签名 AST 检测，302 任务实测零 LLM 成本）；动态层：将任务 init 替换为裸初始态后重跑 preflight rollout，SR 崩塌幅度即"脚手架依赖度"量化指标；
- **脚手架递减课程（方案二升级版）**：不禁止脚手架，prompt/代码层要求随 session 递减（对齐 v6 线"隔离演练→代码保障"思路，两线可合写）；
- 判据：fidelity gate 上线后，tier-3 定向任务的"裸复验 SR"与 held-out tier-3 技能增长应恢复相关。

## 6. 局限

1. 全部 n=1；probe vs 0.6 臂差异多在 ±1.5 噪声带内。
2. ~~脚手架抽查 n=3~~ → **已完成全量审计（302 任务，§3.1）**；但四签名清单可能不完备（未检测的脚手架形态，如地形捷径/资源密度操纵）。
3. "终端技能定向解释 +12 分"为事后解释（post-hoc）——可检验预测：若成立，机制臂收益应集中在 target 过的终端技能上（可对 eval JSON 做技能级归因验证）。
4. 振荡期（session 1-4）的混合课程效应与 0.2 阈值本身的效应未分离。
5. "预算不足"假设未完全排除，但 100% 泄漏使其进一步边缘化。
