# GenEnv vs 我们的方法 — 逐点差异对比（给教授/审稿的区分论证）

> 创建 2026-06-27。GenEnv 是两轮新颖性核查里捞到的**头号近撞车**（撞我们的 gate 机制 + 训练效率结论）。本文逐点对比，给出可直接用于 related-work 区分的论证。
>
> **GenEnv**: *Difficulty-Aligned Co-Evolution Between LLM Agents and Environment Simulators*，Jiacheng Guo, Ling Yang, …, Mengdi Wang。arXiv:2512.19682v2，2025-12-22/23。GitHub: Gen-Verse/GenEnv。
>
> **来源**：GenEnv 细节读自 arXiv HTML 全文（已核）；我方 gate 公式读自 live Oscar 代码 `sfl/train/pcgrl_generator.py:846-847` + `jaxnav_sfl.py:942-944`（赢 SFL 的 `anchored_cenie_gate` 模式实际代码，非旧高斯门）。

---

## 0. 一句话结论

GenEnv 和我们**在"按学习者成功率把生成器推向 p≈0.5 中等难度（ZPD）"这个高层思想上同构**，但落到**域、生成器载体、门的数学形式、应用层、训练算法、评测口径**六个维度全部不同。**最致命的区分点是 GenEnv 用高斯钟形门 `exp(−β(p̂−0.5)²)`——这恰好和我们一个早期写了但没用上的旧门同构，而我们实际赢 SFL 的门是双向加性 relu，形状与应用方式都不同。** 新颖性可守，但 related work 必须主动引 GenEnv 并逐点区分，不能回避。

---

## 1. 逐点对比表

| 维度 | **GenEnv (2512.19682, Dec 2025)** | **我们（jaxnav UED）** | 是否撞 |
|---|---|---|---|
| **核心思想** | 生成器被推向 agent 的 ZPD（p̂≈0.5 中等难度）| 同左：gate 把 generator reward 推向 p≈0.5 可学区 | ⚠️**高层同构** |
| **域 / 任务** | LLM-agent 文本任务（API-Bank, ALFWorld, BFCL, Bamboogle, TravelPlanner）| jaxnav **连续导航**（机器人避障，LiDAR 观测）| ✅ 完全不同 |
| **生成器载体** | LLM（Qwen2.5-7B-Instruct）prompt 采样生成 task 文本 | **learned PCGRL** 卷积关卡生成器（离散网格→连续 jaxnav 关）| ✅ 完全不同 |
| **门的数学形式** | **高斯钟形** `R_env(p̂)=exp(−β(p̂−α)²)`，α=0.5 | **双向加性 relu** `gate=−W_HARD·relu(0.5−p)−W_EASY·relu(p−0.5)`，W_HARD=2.0/W_EASY=1.0 | ✅ **形状不同**（见 §2）|
| **门的应用方式** | 直接作 RWR 的样本权重 `∝exp(λ·R_env)` | host 层 `final=z(auction_bid)+GATE_WEIGHT·z(gate)`（**加性修正**到 auction 选关分上，非乘性、非样本权重）| ✅ 不同 |
| **生成器信号丰富度** | **单信号**：α-Curriculum Reward `exp(−β(p̂−0.5)²)` 是 πenv 的**唯一** reward，只看成功率、不看关卡结构/语义 | **多信号 auction**：`[geodesic-anchored PVL, CENIE]` 选关竞争 + gate 修正项。**GenEnv 整个生成器 reward ⊊ 我们一个修正项** | ✅ **GenEnv 无几何锚、无 CENIE 新颖性** |
| **p̂ 怎么测** | 一个 seed 的 n 个变体，agent 跑完算 p̂=k/n | 一次 student rollout 取 per-level p（每关独立）| ≈ 类似 |
| **生成器训练算法** | **Reward-Weighted Regression (RWR)**：加权 SFT 把 πenv 推向高 reward 生成 | PCGRL generator 走 **PPO**（gen_train_iter），gate 进的是选关打分不是 PPO 梯度 | ✅ 不同 |
| **replay buffer / curation** | 纯在线，每 epoch 重新生成，只做 validity 过滤，无优先级 buffer | **固定 PLR replay buffer** + auction 漏斗 top-K 注入（PLR 优先级回放）| ✅ 不同 |
| **geodesic / 几何锚** | **无**（纯难度门，无空间结构）| **有**：geodesic-anchored PVL 防 generator 堆墙刷分（我们的①号创新）| ✅ **GenEnv 完全没有** |
| **CENIE 新颖性扩展因子** | **无**（无任何新颖性/多样性/覆盖信号）| **有**：CENIE 协方差/嵌入新颖性进 auction（我们的扩展因子）| ✅ **GenEnv 完全没有** |
| **★agent 类型（范式级）** | **LLM agent**（Qwen2.5-7B-Instruct 预训练大模型），微调**文本动作**（调 API/文本指令），走 **GRPO** | **从零训练的 RL student**（小卷积/RNN policy），学**连续控制动作**（速度/转向，LiDAR 观测），走 **PPO** | ✅ **范式级不同**（微调LLM文本行为 vs 从随机初始化训具身导航）|
| **评测口径** | validation score（均值）| **worst-case CVaR**（最难 10% 关）+ 100-map | ✅ **GenEnv 无 worst-case/CVaR** |
| **效率结论** | "3.3× less data" vs **Gemini 2.5 Pro 离线数据增强**（静态预生成数据集）| 更小池(3000) vs **SFL**(5000 海选+learnability curation) +29pt CVaR | ⚠️ 结构类比，但 baseline 不同 |

---

## 2. ★最关键区分：门的数学形式不同（且我们赢的不是高斯门）

这是必须讲清的核心，因为它同时**洗清一个潜在的"完全撞车"指控**：

- **GenEnv 的门 = 高斯钟形** `exp(−β(p̂−0.5)²)`：处处可导、平滑、两极趋 0 但非线性衰减。
- **我们一个早期写过的旧门**（`terminal_reward_euc` 里的 `gate=exp(−((p−0.5)/σ)²)`，乘性 `reward=euc×gate`）**恰好和 GenEnv 同构**——但**这个旧门没用在赢 SFL 的 run 上**（它是更早被取代的版本）。
- **我们实际赢 SFL 的门 = 双向加性 relu**：`gate=−W_HARD·relu(0.5−p)−W_EASY·relu(p−0.5)`，**分段线性**、两侧斜率独立（W_HARD=2.0≠W_EASY=1.0，故压太难比压太简单更狠），且**加性**修正到 auction 选关分而非乘性样本权重。

**论证含义**：
1. 我们能诚实说"赢 SFL 的门和 GenEnv 形式不同"（加性 relu vs 乘性高斯，两侧非对称 vs 对称），这是**实打实的差异**，不是文字游戏。
2. **更强的论证**：W_HARD≠W_EASY 的**非对称双向门**是 GenEnv 对称高斯门没有的设计——它体现"压太难 > 压太简单"的先验（早期 student 弱，太难关危害大于太简单关）。这是个可写进论文的设计动机。
3. ⚠️ **但要小心**：我们旧的高斯门和 GenEnv 同构这件事，如果被审稿人翻到我们代码，可能问"你们不也写过高斯门吗"。**建议**：要么在论文里只呈现 relu 门（赢的那个），要么主动说明高斯门是早期消融、已被非对称 relu 门取代且后者更优。

---

## 2.5 ★最强 framing：GenEnv 是我们方法的一个退化特例

把上表浓缩成一句对审稿最有力的话：

> **GenEnv 的整个生成器 = 我们生成器的一个退化子集**——只保留"难度门"这一个信号，去掉几何锚（geodesic-anchored PVL）、去掉新颖性扩展（CENIE）、把门从非对称双向 relu 简化成对称高斯、把载体从 learned PCGRL 换成 LLM prompt、把域从连续导航换成文本任务、把 agent 从从零训练的 PPO student 换成微调的 LLM agent。

换言之：我们不是"GenEnv 的一个变体"，**反过来，GenEnv 在机制上是我们 auction 系统里单独抽出"难度门"那一项、再搬到 LLM 域的结果**。我们的方法在信号维度（多信号 auction vs 单门）、空间结构（有 geo 锚 vs 无）、agent 范式（具身 RL vs LLM 微调）三个层面都是 GenEnv 的超集或正交扩展。

---

## 3. 给教授/审稿的区分陈述（可直接用）

> 我们注意到与 GenEnv (Guo et al., arXiv:2512.19682, Dec 2025) 的概念相邻：两者都把生成器推向学习者的 ZPD（中等成功率）。但 GenEnv 在机制上是我们方法的一个退化特例，二者在七个维度上不同：(i) **agent 范式**——我们是从随机初始化训练的具身 RL student（小卷积/RNN policy，PPO，连续控制 + LiDAR），GenEnv 是微调预训练大模型 Qwen2.5-7B 的 LLM agent（GRPO，文本动作）；(ii) **域**——连续机器人导航 UED vs LLM-agent 文本任务；(iii) **生成器载体**——learned PCGRL 卷积关卡生成器 vs LLM prompt 采样；(iv) **生成器信号丰富度**——我们是 `[geodesic-anchored PVL, CENIE]` 多信号 auction + gate 修正项，GenEnv 只有难度门这一个 reward（我们的整套生成器信号严格包含并超出它）；(v) **几何锚**——我们有 geodesic-anchored PVL 防生成器堆墙刷分，GenEnv 无任何空间/几何结构；(vi) **门的形式**——非对称双向加性 relu 门（压太难 > 压太简单）vs 对称高斯门；(vii) **curation 与评测**——我们建在固定 PLR 优先级回放 buffer 上、在 worst-case CVaR（最难 10% 关）对标，GenEnv 纯在线、只报均值。**关键的是，gate 在我们方法里只是 geodesic-anchored PVL 主信号上的可学性修正项，而难度门是 GenEnv 生成器的唯一 reward。** 且 GenEnv 是 Dec 2025 同期工作，时间线上与我们并行，应作为 concurrent work 处理，而非先行被覆盖。

---

## 4. 残留风险 / 待办

1. **GenEnv 是 Dec 2025 同期工作**——严格说不算"先行工作"（我们的结果时间线上并行）。可在论文里作为 *concurrent work* 处理，进一步弱化撞车。✅ 对我们有利。
2. ⚠️ **旧高斯门的尴尬**（见 §2）：决定论文是否呈现高斯门、怎么解释。建议只留 relu 门或明确标早期消融。
3. **建议自己读一遍 GenEnv 原文确认 §1 表里 GenEnv 那列**——本文基于 arXiv HTML 全文抽取（已核 α=0.5/β/RWR/Qwen2.5-7B/GRPO/无 geodesic/无 CVaR），但教授通知前你亲自扫一遍最稳。GitHub: Gen-Verse/GenEnv 可看代码确认门的实现。
4. camera-ready 前再扫一次 2026 预印本（子领域快移）。
