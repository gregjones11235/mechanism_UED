# 新颖性定位与竞品区分（novelty & differentiation）

> 起草 2026-07-07。用途 = proposal / related work 的定位骨架与差异化论证。
>
> **⚠️ 来源与可信度边界（务必先读）**
> - 本文对 **SCALAR / CODE-SHARP / SkillGraph** 等竞品的机制描述与数字，**全部转引自同仓库 `main` 分支 `fable_research_reports/` 的队友调研档案，本人未读原文**。凡涉及具体机制、数字、arXiv 编号处均标 **`[待原文核实]`**，写进正式 proposal 前必须自己打开原文核对（队友档案里部分条目自己都标了"记忆未核实"）。
> - 真正**现在就成立、不依赖竞品数字**的是第 2、3、5 节的**定位逻辑**（我的 setting 为什么不同、如何区分、诚实边界）。这部分是本文的主体价值。
> - 本文**不替任何主张打包票**。学术定位讲究"不过度声称"——proposal 评审最反感言过其实，一处引错编号或夸大数字即是硬伤。

---

## 1. 一句话新颖性定位

**在"小模型（14B）驱动的 code-level UED"这个尚无人做过的 setting 下，组合 skill-graph 引导生成 + learnability preflight 过滤，让一个弱得多的生成器逼近大模型（235B 级）的课程质量与最终 held-out 性能。**

拆成三个可辩护的支点：
1. **setting 新**：小模型 + code-level UED 的组合无先例（竞品用 GPT-5 / Qwen3-235B）。
2. **动机被竞品自己背书**：已发表工作实证"生成模型换小则 pipeline 崩" `[待原文核实：CODE-SHARP 的 FM-scaling 消融，Qwen3-30B 档案崩坏]` —— 正说明"如何让小模型不崩"是个真问题、且没人解。
3. **两个组件是手段、不是主张**：skill-graph 与 preflight 各有先例（见第 3 节），我们的贡献是**在小模型 setting 下把它们组合起来达成上面的目标**，以及若干实现层的区分点。

---

## 2. 我的 setting 为什么"硬"：四条评测口径

竞品的高分往往建立在放松了某条口径上。把这四条口径列成对照表，让**口径差异本身成为"我们的 setting 更干净"的论据**（这个"口径对照表"的写法直接借鉴队友档案的建议）。

| 口径 | 我们的立场 | 竞品常见放松处 `[待原文核实]` |
|---|---|---|
| **单一裸 policy** | 单个 PPO student 在权重里内化整条链，测试时**无任何辅助** | SCALAR：per-skill 独立 head + planner 测试时拼链；CODE-SHARP：测试时代码路由拼链 |
| **真实初始态** | 从真实初始库存/状态出发 | SCALAR：Frontier Checkpointing 训练时状态空投（eval 时撤除，但 train-eval gap 是其 limitation） |
| **held-out mean return** | 用 held-out 关卡的裸成功率/return 判胜负 | 两篇多为 per-achievement SR / 自定义 benchmark；held-out mean return 口径下 tier4 据称无人碰过 |
| **小模型生成器** | 14B 本地模型 | SCALAR：GPT-5 Thinking；CODE-SHARP：Qwen3-235B |

**核心论点**：在"单一裸 policy + 真实初始态 + held-out mean return"这个最干净的口径下，深层 tier（4+，铁装/钻石/魔法）据队友调研**全领域都还没被真正攻破** `[待原文核实]`。我们**不声称打破这堵墙**，而是研究"**在小模型约束下，skill-graph + preflight 能把课程质量推到多接近大模型**"——这是一个定义清楚、可量化、可消融的问题，不需要"横扫 SOTA"就成立。

---

## 3. 逐个 differentiate（related work 骨架）

每小节结构：它做了什么 / 我的区别 / 我能主张的。**数字与机制细节 `[待原文核实]`。**

### 3.1 vs SkillGraph `[arXiv 2605.12039，待核实]`
- **它**：技能 = 图节点，带类型边编码 prerequisite / co-occurrence，图随轨迹演化，依赖序检索 + 渐进解锁形成自动课程。
- **我的区别**：SkillGraph 是**通用技能图**的课程机制；我的 skill-graph 是**嵌进 code-level UED、用来引导小模型生成任务**的。差异不在"用不用图"，而在**图服务于什么**——我的图为"弱生成器该造哪个 tier 的任务"提供 target 信号（`skill_scheduler` 定位学习前沿）。
- **可主张**：skill-graph 作为**小模型生成的 target 调度器**，据现有调研无直接先例。（注意：不主张"图结构本身"是新的——那是 SkillGraph 的。）

### 3.2 vs CODE-SHARP 的 learnability 准入闸 `[arXiv 2602.10085，待核实]`
- **它**：发现新技能时用 learnability 闸准入——policy 副本试训、增益 Δρ 超阈值（`[待核实：>0.05]`）才收进档案。
- **我的区别**：这几乎就是我 `preflight` 在做的事（cold rollout 判可学习性再准入），**必须主动引用、承认概念先例**。区别在：① 它是**大模型 + 测试时代码路由拼链**的架构，preflight 只是其档案准入的一环；② 我是**小模型 + preflight 过滤 + 让裸 policy 内化**，preflight 是"补偿小模型生成质量差"的关键机制——在小模型 setting 下，preflight 的作用和必要性与大模型场景不同（大模型生成的候选本就质量高，preflight 更多是锦上添花；小模型候选噪声大，preflight 是刚需过滤）。
- **可主张**：**learnability preflight 作为"小模型生成质量补偿"机制**的定位与实证。**不主张 learnability 准入闸的概念是我首创**——这是硬伤风险最高的一处，务必谦逊。

### 3.3 vs SCALAR 的轨迹反馈（PTA） `[arXiv 2603.09036，待核实]`
- **它**：Pivotal Trajectory Analysis——首次非零 SR 时取成功轨迹喂 LLM，修正技能算子规格（一次性触发）。
- **我的区别**：都用了"从 student 轨迹回馈 LLM"，但 PTA 修的是**算子规格**、一次性、per-goal 独立跑；我的方向（若做轨迹链挖掘）是**跨 session 的课程 target 调整**。
- **可主张**：定位为"轨迹反馈修正生成先例"，我的贡献收窄为具体用途上的区别。（此节仅当你确实做轨迹分析时才写；否则可省。）

---

## 4. 我能主张的新颖点（收敛后）

按"最强 → 最弱"排，proposal 里主打前两条：

1. **小模型逼近大模型（最强、最独特）**：在小模型 code-level UED setting 下，用 skill-graph + preflight 逼近大模型课程质量。竞品实证"小模型直接用会崩"，而"如何补偿"没人做——这是最干净、无争议的主张。
2. **两组件在小模型 setting 下的组合与实证**：skill-graph 当生成 target 调度器 + preflight 当质量补偿，两者协同（graph 指方向、preflight 保质量）。
3. **（可选，需谨慎）实现层区分点**：如轨迹链挖掘、preflight 的具体 gating 策略等——这些是工程贡献，别拔高成理论新颖。

---

## 5. 诚实边界（proposal 里明确写出，反而加分）

主动声明"哪些不是我的首创"，是成熟研究者的做法，评审会因此更信任其余主张：

- **learnability preflight 的概念**有先例（CODE-SHARP 准入闸 `[待核实]`）——我的贡献是**小模型 setting 下的应用与作用重定位**，非概念首创。
- **skill graph 结构**有先例（SkillGraph `[待核实]`）——我的贡献是**作为小模型生成 target 调度器的用法**。
- **"切断长链导致迁移失败"**据队友调研已被同行发表 `[待核实：CODE-SHARP 的 CODE-FRP 消融]`——**不能当独立发现**，只能当 motivation 引用。
- **DiCode 本体**是我们的直接基座（arXiv 2602.08194，Imperial/Cully `[待核实]`），我们是"小模型化 + 加两组件"，这个继承关系要讲清楚。

---

## 6. 待办（升级到"可进 proposal"版本）

- [ ] 逐篇打开原文核对：SkillGraph (2605.12039)、CODE-SHARP (2602.10085)、SCALAR (2603.09036)、DiCode (2602.08194) 的标题/作者/年份/核心机制/关键数字——**把所有 `[待原文核实]` 消灭**。
- [ ] 确认队友档案里标 `[记忆，未核实]` 的编号（如 Go-Explore、HER、Deep Skill Chaining）是否要引，引则核对。
- [ ] 与队友（v6/SOTA 线）对齐引用口径，避免同组两条线对同一批竞品的描述打架。
- [ ] 补一段"我们的 held-out 评测协议定义"，让第 2 节的口径主张有可复现的操作定义。
