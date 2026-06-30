# UED 开放问题与领域现状 —— Deep Research 存档（2026-06-30）

> 五个联网调研子任务交叉核验，arXiv ID + 具体数字均查证。
> 可信度三档：**[已核验]**（对照论文/摘要确认）/ **[转述]**（论文自述、二手抓取未独立交叉核对）/ **[推测]**（综合判断）。
> 动机：用户问"既然简单 jaxnav 迷宫已 solved，UED 领域 unsolved 的问题都有哪些"。
> 配套记忆：`ued-benchmark-saturation-map-2026`、`ued-paper-name-corrections-2026`。

## 1. 未解决的硬 benchmark（非饱和的证据）

最重要的发现：**更难的现代 benchmark 远未饱和。** 老 UED 论文"解决"过的经典迷宫（MiniGrid 16×16）和双层 BipedalWalker，已被换成一批所有方法（DR、PLR⊥、ACCEL、SFL、CENIE、TRACED）仍然惨败的测试床。

### 关键 framing 铁证
2025 年 SFL-作者团队的 follow-up "**An Optimisation Framework for UED**"（arXiv:2505.20659，Monette/Letcher/Beukman/Jackson/Rutherford/Goldie/Foerster）原文：
> *"We omit JaxNav, as the single-agent setting's results are **highly saturated** and the multi-agent setting introduces additional optimisation challenges."*

并点名 XLand-Minigrid 是其测试床里最不饱和的：
> *"this environment has results that are **less saturated**, and thus leaves more room for improvement."*

这是用户背景事实最强的已发表确认，也是直接指向 XLand-Minigrid 和多智能体 JaxNav 的路标。

### 饱和度地图

| 环境 | 状态 | 非饱和铁证（含数字） | arXiv |
|---|---|---|---|
| **BipedalWalker-Hardcore** | 未饱和（区分度最强、文档最全） | TRACED@10k Hardcore **86.83±17.96** vs ACCEL **37.59±15.0**（2.3×）；6 地形均值 ~90 vs ~50；Stump/PitGap 仍负 return；原 ACCEL 达最优 ~75%、约 PLR 3× | 2506.19997 / 2203.01302 |
| **Kinetix-Large**（可选） | 未饱和 | 零样本 S~75% / M~35% / **L~10%（非单调）**；Mujoco-Hopper/Walker-Hard 零样本失败、须微调才解；PLR/ACCEL 在此 = DR 无增益（只区分 SFL/DR vs regret 类） | 2410.23208（ICLR25 oral） |
| **完整 Craftax**（可选） | 未饱和（但属开放式 RL 非经典 UED） | 论文明说现有方法（含全局/分幕探索**和 UED**）"fail to make material progress"；Advanced/Very-Advanced 成就基本无解；最好 ~18% of max reward | 2402.16801（ICML24） |
| **大 PerfectMaze** | 各方法基本没解开 | PerfectMazeLarge(51×51) TRACED ~27%±23%@10k vs ACCEL ~20%±25%@20k；XL(100×100) ~10%±14% vs ~12%±28%；旧 8% 锚已作废、无新表 | 2506.19997 |
| **CAMAR**（连续多智能体路由） | 未饱和 + **无人做 UED = 处女地** | labmaze grid MAPPO 仅 **0.568**、IPPO 0.213，off-policy(MADDPG/MASAC)全崩；random grid MAPPO 0.830 vs IPPO 0.410；~100k 步/秒；定位为 benchmark 非 UED 方法 | 2508.12845（2025-08） |
| **XLand-Minigrid hard 档** | 未饱和（原作者点名） | 2505.20659 点名 least-saturated；最难任务 zero reward（exploration wall） | 2312.12044 |
| **多智能体 JaxNav** | 未饱和（SFL 论文自带逐方法表证之） | SFL 论文 Fig5/6 **有**逐方法对照：手设集 SFL 峰值仅 ~0.85、采样集仅 ~0.5（vs 单智能体 99%）；SFL 显著超 baseline = 方法仍有区分度；原作者称"额外优化挑战" | 2408.15099 |
| 单智能体 jaxnav 11×11 | **已饱和（避开）** | 原作者自认 highly saturated；overall 99% CVaR100 | 2408.15099 |
| MiniGrid 标准 maze | **已饱和** | 2505.20659 称结果重叠 | — |
| Craftax-Classic | **已饱和** | PPO <1hr/1GPU 达 ~90% 最优；>human(65%) | 2402.16801 |
| gymnax 经典控制 | **已饱和** | PPO 基本解掉 | — |

**总结：** 已饱和 → MiniGrid-16、BipedalWalker-Basic、Craftax-Classic、单智能体 jaxnav。**仍开放且方法间差距大** → Kinetix-Large/硬控制、完整 Craftax、PerfectMazeXL（<30%、巨大方差）、BipedalWalker-Hardcore（刚被填上）、CAMAR/多智能体 JaxNav。

---

## 1.5 ★多智能体 jaxnav 深度分析（基于 SFL 论文原文，2026-06-30 用户问 2）

> 起因：用户问"SFL 在多智能体 jaxnav 上没饱和吗？遇到了什么困难？"。直接读 SFL 论文（2408.15099）§7.2 + Fig5/6 + §G.5 Fig21 + Appendix A。
> ⚠️ 订正上方表格此前"无 published 逐方法 gap 表"的说法不准确——SFL 论文**就有**逐方法 CVaR/成功率/heatmap 对照（Fig5/6）。准确说法=有表，但绝对水平低 + 后续 UED 论文很少在它上面做干净对照。

### 饱和度：赢但远没到"解决"（双重未饱和）
- SFL **显著超所有 baseline**（DR/ACCEL/PLR-PVL/PLR-MaxMC），与单智能体一致（Fig5/6）。
- **但绝对水平远低于单智能体**：手设测试集 SFL 峰值约 **0.85** 即到顶震荡（vs 单智能体 overall 99%）；采样测试集 SFL 仅爬到 **~0.5**；CVaR α=100% 约 80%+、低 α 尾部更低（Fig5a-d）。
- → 双重未饱和：①**绝对未饱和**（最好的 SFL 也只 0.5~0.85，离天花板远）；②**方法间仍有区分度**（不像单智能体大家挤 99%）。
- 对应 NCC(2505.20659)原话后半段："multi-agent introduces **additional optimisation challenges**"——不是饱和，是**太难/太乱以致难做干净对照**。

### 三个困难（全是多智能体特有，非单智能体放大版）
1. **协调 + 部分可观测叠加，难度本质升级**（§7.2 + Appendix A）：每 agent 仅自身 LiDAR 局部观测、IPPO 独立训、连续空间互相避碰，比单智能体"绕墙到目标"多一层无中心通信的 agent-agent 动态博弈。绝对水平上不去的根因。
2. **learnability 信号被迫妥协**（§7.2 原文）：n agent 的 learnability 定义成 **Σ pᵢ(1−pᵢ)**、关卡分=各 agent regret 均值。把单智能体标量硬摊到多 agent，**没刻画 agent 间协调难度**。=本文档 §2 信号层病的多智能体版（单智能体盲视"长程"，多智能体盲视"协调"，同一类问题）。
3. **生成器造部分不可解关**（§G.5 Fig21 原文 "SFL occasionally generates levels where not all agents can reach their goals"）：多 agent 时更难保证全员同时可解；SFL 称"仍有用"=承认可解性检验/生成质量控制在多 agent 下更难。=本项目单智能体踩过的"不可解关被注入"坑（[[component-mask-fragmented-graph-bug]]）的多 agent 版（要 n 条路同时可达，更严重）。

### 对本项目的含义
多智能体 jaxnav 三个困难全正中本项目已研究透的命门：**信号盲视结构性难度**（已诊断长程版 [[s4p2A-root-cause-difficulty-axis-mismatch]]，这里是协调版）+ **不可解关生成质量**（已用 BFS 根治单智能体版）。区别只是把"结构性难度"从"长测地"换成"多 agent 协调"。

---

## 2. 开放的方法论问题（三层卡点）

### 信号层（最深的一层 —— field #1 开放问题，恰好是本项目反复撞的墙）

**(a) 后悔值近似 ≠ 后悔值 ≠ 可学性。**
- "**No Regrets**"（arXiv:2408.15099，即 SFL 论文）[已核验]：标准评分器 **PVL** 和 **MaxMC** 相关的是**成功率而非后悔值**，于是过度挑选智能体已会解的关，浪费经验；**即便真实后悔值也不总追踪可学性**，错配导致停滞。修法：**可学性 = p·(1−p)**，对照 Perfect Regret oracle 验证。
- LLM 延伸（已核验）：**LILO**（arXiv:2502.12272，Foster 等，2025-02）把同一病理搬到 RLVR LLM 微调；复用可学性估计的 rollout 当训练样本（SFL 缓冲对 LLM 浪费）。
- 这是当下整个领域的核心张力：三个曾被假定一致的目标（后悔近似、真实后悔、可学性）被证明并不一致。

**(b) 可学性只看一局的最终成败，不看完整轨迹。** —— DEGen（见订正 2）；TRACED 也批 regret 由 value-loss 单独近似 = transition-blind。**外部背书本项目"student 信号盲视长程"诊断**（记忆 s4p2A-root-cause-difficulty-axis-mismatch）。

**(c) 没有一个方法同时做到 轨迹感知 + 可学性对齐 + 理论收敛** —— MNA、TRACED、NCC 是三条独立线，未合一。

### 生成层（本项目 generator 坍缩不是个例，是全 field 病）

**生成器/对手模式坍缩。**[已核验，含硬数字]
- "**Stabilizing UED with a Learned Adversary**"（arXiv:2308.10797，Mediratta 等，CoLLAs 2023）：PAIRED RL 对手不稳，策略熵坍缩（P1）+"fall-behind"退化循环（P2），常**输给 DR 和 PLR**。
- 数字：CarRacing-F1 vanilla PAIRED **26.16±15.22** vs Robust PLR **530.94±6.67**；Minigrid 0-60 PAIRED **0.21±0.07** vs ACCEL **0.61±0.03**。修法（高熵 bonus + 双向行为克隆）把 PAIRED 推到 **641.01±7.21** 才终于超 Robust PLR。
- 纯 minimax 对手倾向生成**不可解**关（零学习信号）—— PAIRED 用 regret 而非 minimax 的初始动机（Dennis arXiv:2012.02096）。

**生成范式之争（四阵营，无胜者）：**
- **随机+curate（PLR/PLR⊥/SFL）**：最简单、最差尾部上强，但受限于大空间有效关命中率 [已核验弱点]。
- **RL/学习式对手（PAIRED/DEGen）**：难度控制更稠密但易坍缩、受信用分配之苦。
- **演化（ACCEL）**：编辑后回放；强 baseline 但 PerfectMazeXL 仍 <30%、复杂度爬升慢。
- **生成模型/LLM**：增长最快——Eurekaverse（LLM 写环境代码，2411.01775）、DéjàQ（LLM 变异题目，2601.01931）、ADEPT（diffusion 生成器优化初始噪声朝向最大策略提升，2506.01759）、SHED（层次 MDP + 生成模型造合成轨迹，2310.00301）。
- **[推测]** 领域尚未收敛；2025-26 势头明显倒向生成模型/LLM 驱动，因为随机+curate 在大空间造不出有效硬关——这正是本项目 JaxNav 生成器反复撞的瓶颈。

### 理论实践鸿沟

1. **Nash 不可达/不可测。** 真实 UED 目标带神经网是非凸非凹，原始理论的凸凹 normal-form 构造"不是 UED 问题的合理表示"；收敛无保证、也判不出何时达到。
2. **Regret stagnation。** 智能体掌握所有关的 regret bound 后，对手继续采 regret 无法再降的关→停滞（ReMiDi arXiv:2402.12284 / BLP 证此，对应本项目 curriculum-too-hard）；随机/部分可观下 regret 有不可消的下限。
3. **Regret 原则上也不是可行 score** —— 高 regret ≠ 高学习，真 regret 需最优策略。
- **收敛修法**：NCC（2505.20659）= 首个 zero-sum 可证 ε-first-order-Nash（熵正则化 + 双时间尺度 SGDA），但采纳还薄。
- **目标misgen修法**：arXiv:2507.03068（含 Dennis）**证明** minimax-expected-regret(MMER) 下不会目标误泛化、DR 会；但现有方法不总能达到 MMER 策略。

### Curation 样本效率
- 评分追踪成功率非 regret → buffer 过度回放已解关（直接样本效率损失）；Robust PLR 只在 curated 关更新、丢一半数据；value-loss 评分放大 stochasticity（noisy-TV，motivates TRACED）；ACCEL 复杂度爬升慢（motivates DEGen 动态生成）。TRACED 是最干净的量化效率结果（等迁移、约半更新量 / wall-clock vs ACCEL）。

---

## 3. 近期前沿论文清单（2024-2026）

### A. Foerster / Oxford FLAIR 谱系（SFL 家族）
| 论文 | arXiv | 主要主张 | 指向的下一开放问题 |
|---|---|---|---|
| **No Regrets / SFL**（NeurIPS24） | 2408.15099 | 后悔近似追踪成功非后悔；可学性 p(1−p) 才对；SFL 在 JaxNav 最差尾部胜出 | 仅二值/确定性域；巨大空间有效关生成 |
| **NCC / Optimisation Framework**（RLC25） | 2505.20659 | 零和 UED 可证收敛双时间尺度 SGDA（非凸强凹）；NCC-Learn 在 XLand 最高解决率 | 超越零和/一般和保证；弃用 jaxnav(saturated) |
| **LILO**（2025-02） | 2502.12272 | 把 SFL 可学性搬进 LLM 推理 RL 微调；复用估计 rollout 当训练样本 | 哪些题目随能力变化仍带信号 |
| **DéjàQ**（2026-01） | 2601.01931 | 开放式演化 多样+可学+可验证 的数学/代码题（QD + LLM 变异） | 规模化生成可学题、留在能力内、保持可验证 |
| **DEGen / MNA**（NeurIPS25） | 2601.14957 | 动态生成 denser reward 解 credit assignment + MNA 轨迹感知 regret 代理 | 图像观测、代理理论保证、跨族迁移 |
| **Kinetix**（ICLR25 oral） | 2410.23208 | 数千万 2D 物理任务训通用智能体、零样本泛化 | Large 档 + 硬控制未解；intelligently generate 而非 filter |

### B. 基础 regret-UED 锚点
| 论文 | arXiv | 主张 |
|---|---|---|
| **PAIRED**（2020） | 2012.02096 | 引入 regret（protagonist−antagonist）当设计信号；可解-复杂课程 |
| **Robust PLR / DCD**（NeurIPS21） | 2110.02439 | PLR=UED；PLR⊥ 只在 curated 关更新→Nash 鲁棒性保证、更少数据更好 |
| **ACCEL**（ICML22） | 2203.01302 | 演化/变异高 regret buffer 关；BipedalWalker-Hardcore ~75% 最优、约 PLR 3× |
| **Refining Minimax Regret (ReMiDi/BLP)**（ICML24） | 2402.12284 | MMR 掌握后早停滞；BLP + ReMiDi 继续学（对应本项目 stagnation 发现） |

### C. 相邻 / 最新（2025-2026）
| 论文 | arXiv | 主张 |
|---|---|---|
| **CENIE**（2025） | 2502.05726 | coverage-based novelty（GMM over 覆盖）叠加 regret UED → SOTA |
| **TRACED**（ICLR26） | 2506.19997 | transition-prediction error + Co-Learnability；双足硬核 2×、半更新量超 CENIE |
| **目标误泛化 via MMER**（RLC25） | 2507.03068 | 证 MMER 不会目标误泛化、DR 会；现有方法不总达 MMER |
| **CAMAR**（2025-08） | 2508.12845 | 连续动作多智能体路由 benchmark（非 UED 方法）；~100k 步/秒；RRT/RRT* 混合 |
| **Beyond Fixed Tasks**（2025-11） | 2511.12706 | UED 扩到联合生成 (环境, 任务) 对 |
| **Hierarchical UED**（2026-02） | 2602.09813 | 用学习式策略表示层次化生成课程、更样本高效 |
| **Eurekaverse**（CoRL24） | 2411.01775 | LLM 生成环境代码当课程；真 Go1 四足零样本迁移、超人工设计 |
| **iMac (Imagined Autocurricula)**（NeurIPS25） | 2509.13341 | diffusion 世界模型 + PLR 在想象关上自动课程；Procgen +17-48% vs model-free |
| **OMNI-EPIC**（ICLR25） | 2405.15568 | 基础模型自主生成代码指定下一"可学有趣"任务 |
| **Open-Endedness manifesto**（ICML24 oral） | 2406.04268 | open-ended = novelty + learnability；FM + 开放性正交且强力；**与本项目双向 gate 撞，必引** |

### 谱系要点
FLAIR 主线弧：**Kinetix → DéjàQ**，处在 Open-Endedness 立场论文（2406.04268）和 Dennis 谱系 UED 之下。2026 显著动作 = DéjàQ 把可学性驱动 UED 从网格世界 RL **移植进 LLM 推理**（带可验证奖励的数学/代码）—— UED 被重定义为面向任何学习者的自动课程。

### 无统一综述
未找到 2023-26 的权威专门 UED 综述。最近的 taxonomy 锚点：minimax(arXiv:2311.12716，本项目用的 codebase + DR/PLR/Robust-PLR/ACCEL/PAIRED taxonomy)；Structure in Deep RL 综述(2306.16021)；Open-Endedness 立场论文(framing 非 taxonomy)。

---

## 4. 领域认为应该往哪走（consensus → speculative）

1. **Generate 不要 filter（强共识）**：Kinetix 明确喊话 "intelligently generate levels rather than just filtering"；curation 是 PLR/ACCEL 根本局限。由 OMNI-EPIC/Eurekaverse/iMac 实现。
2. **世界模型 / 基础模型当 environment generator（强、快速增长共识）**：iMac(diffusion WM+PLR)/Genie(2402.15391 当 learned simulator)/PROWL(2605.18803)/Dreaming in Code(2602.08194)。
3. **open-endedness 当框架，novelty + learnability 当核心目标**（Hughes 与 Clune/Cully 两大阵营趋同）：**与本项目双向 gate 思路撞，须引为 prior art**。
4. **离开 gridworld → 连续控制 & 多智能体**（regret 在此明显变弱）：Kinetix/CAMAR/JaxNav；MAESTRO(2303.03376) 多智能体 UED 锚点；Unsupervised Partner Design(2508.06336)。
5. **长程/稀疏奖励要 directed target-relevant 课程**（新、有理论界）：**DISCOVER**(2505.19850，NeurIPS25)论证"解复杂高维任务须解与目标相关的简单任务"、形式化界定到达目标时间不依赖任务空间体积——**外部背书本项目"student 信号盲视长程"诊断**。
6. **更长时域/深度探索**：完整 Craftax 是当头棒喝，社区当北极星。
7. **firmer theory**：NCC 收敛 + MMER 鲁棒性当通向部署的桥（active，回报存疑）。
8. **real robot transfer**：仍只 LLM 课程线交付过（Eurekaverse 四足），classic UED 没上过真机；资源/时域受限 UED 被点名为部署 gap。

---

## 5. 对本项目的要点（bottom line）

1. 本项目的**最差尾部（CVaR）+ 连续导航**框架**正处在领域主轴上**——TRACED/SFL/PerfectMazeXL 数字都确认最差情形是方法仍分化处、大型/连续任务未饱和。
2. 本项目的 **可学性 vs 后悔 vs 成功** 混淆（"难度竞价 ≡ 可学性平移"发现）= 领域第一号开放方法论问题（No Regrets + DEGen）。
3. **DEGen（2601.14957）** = 之前想回忆的那篇，直接论证"只看终局的可学性不够用"，与本项目生成器工作相邻，须定位区分（但那句批评原话须亲验，见订正 2）。
4. **DéjàQ（2601.01931）+ GenEnv（记忆里近撞车）** 确认 FLAIR 正活跃此空间——引用做区分。
5. **选战场建议（★工程成本不计入决策，2026-06-30 用户明确）**：现在是**从零开始**阶段，不考虑任何迁移/移植/重写成本——选战场**只看科学价值**（未饱和度、方法区分度、与本项目命门的契合度、故事新颖性），不看"是不是 JAX / 要不要重写物理引擎"。jaxnav 11×11 该弃；按**科学价值**排（非工程成本）：
   - **BipedalWalker-Hardcore**：区分度最强、文档最全（TRACED vs ACCEL 2.3×），连续控制 UED 经典硬基准。此前因 PyTorch+Box2D 被降级，**现取消该降级**——若科学上最该打就打，重写 Box2D→JAX（如 Kinetix 的 Jax2D）或直接用 PyTorch 栈都接受。
   - **CAMAR**：处女地（无人做 UED）、novelty 最高、连续多智能体。
   - **多智能体 jaxnav**：未饱和（SFL 仅 0.5~0.85）、三个困难全中本项目命门（见 §1.5），故事顺。
   - **Kinetix-L**：~10% 解决率、欺骗性长程、同 FLAIR 谱系。
   - **Axis3 大地图 holdout**：同环境新 OOD 轴，最贴近现有工作但 novelty 中等。
   > 注：迁移成本仍可作为**执行排期**的次级参考（先跑得动的先验证），但**不再是选哪个战场的决策依据**。科学价值优先。

---

## 5.5 ★LLM-generator UED vs SFL 尽调（2026-06-30 用户问：拟做"多 LLM 协作 UED + auction"）

> 起因：用户拟做"多 LLM 智能体协作 UED（可能用 auction 机制）"，问 Eurekaverse 之类 LLM 方法有无和 SFL 比过、性能如何。一个 agent 逐篇核实原文。

### ★★★ 头号结论：LLM-generator 与 regret-UED **从未同台比过**（双刃）
- **没有任何一篇 LLM-as-generator UED 论文**比过 SFL/PLR/ACCEL/DR/PAIRED。Eurekaverse/OMNI/OMNI-EPIC/DiCode 全部只比 **human-designed 课程 + uniform/learning-progress + 自身消融**，且**只报均值、无 CVaR**。
- 反向也真：SFL/ACCEL/PLR 经典 benchmark 表里**零个 LLM generator**。**两条文献线从未放到同一根轴上。**
- → **真空白**：把 LLM(或多 LLM auction) generator 放进 SFL 的 JaxNav + CVaR worst-case 口径 vs regret-UED = 领域首创。
- → **真风险**：从没人做过=**无先验**保证 LLM generator 能在连续导航 CVaR 上追平 SFL；本项目记忆已反复证 CVaR worst-case 极难（generator 输 SFL 负结果 [[generator-pool-collapses-to-extremes]]）。LLM-code 论文全在 Crafter/parkour/BabyAI 等**不用 CVaR 当标尺**的域玩，没在本擂台证明过自己。

### 学生类型分层（决定能否和 SFL 对话）
| 方法 | arXiv | 生成什么 | **学生** | 比 regret-UED? | CVaR? | 单/多 generator |
|---|---|---|---|---|---|---|
| **Eurekaverse**(CoRL24) | 2411.01775 | 环境**代码**(地形,演化) | **RL(PPO)** 四足 | ❌无 | ❌仅均值 | 单 LLM(GPT-4o) |
| **OMNI-EPIC**(ICLR25) | 2405.15568 | 环境**代码** | **RL(DreamerV3)** | ❌无 | ❌ | 单 LLM+VLM |
| **OMNI**(ICLR24) | 2306.01711 | 任务 spec+检查码 | **RL(PPO)** Crafter等 | ❌无(比 uniform/LP) | ❌ | 单 FM |
| **DiCode**(2026-02) | 2602.08194 | 执行环境码(按能力) | **RL** Craftax | ❌无(目前所见) | ❌ | 单 FM；★最新竞品须全文读 |
| **DéjàQ**(2026-01) | 2601.01931 | 数学题变异 | **LLM(Qwen)** ❗ | ❌(类别不符) | ❌(借 p(1−p)做 archive) | 单；学生是 LLM**非 RL**=另一问题 |
| **ADEPT**(2026) | 2506.01759 | 地形(**diffusion 非 LLM**) | RL(PPO) | ❌(比 planner) | ❌ | 单 |
| **ADD**(2024) | 2410.19715 | 地形(diffusion,**用 regret**) | RL | 较近 UED canon | — | 单；regret-guided diffusion |

- **同问题类**(学生=RL agent,只是没比 SFL)：Eurekaverse/OMNI/OMNI-EPIC/DiCode → 你做"多 LLM UED"**学生须是 RL agent**才能和 SFL 对话。
- **不同问题**(学生=LLM)：DéjàQ → 不可比。

### novelty:auction + 多 LLM 协作 UED —— 信号强
- **UED 里 auction/市场机制选 level：完全没找到**（搜出来全是广告竞价,无关）。
- **多协作 LLM 当 UED generator：没找到**（只有人类教学设计的多 LLM persona 对话 2508.16659,非 RL UED）。
- 本项目已有 auction 机制（[[auction-mechanism-is-plr-not-paired]] [[bid-standardization-topk-fix]]）升级成"多 LLM generator 竞价"**看起来真空**。
- ⚠️ 须交叉核 [[novelty-check-currA-gate-prior-art]] 的 **GenEnv(2512.19682)**=已记录近撞车,本次 agent 未重核。
- 相邻参照:**MAESTRO**(2511.19253,单 LLM 生成 reward+task spec,学生 MARL,交通域,无 auction 无 regret-UED 对照)=最近的"LLM 塑 MARL 课程"工作。

### 对方向的判定
- **可入论文的首创点**=第一个让 LLM-generator 上 SFL CVaR 口径同台 vs regret-UED;本项目独有资产=现成 SFL 去偏 CVaR 管线(10k 关)。
- **执行前必锁**:①学生定为 RL agent(非 LLM);②LLM→jaxnav 关的生成-编译-rollout-反馈 loop 可行性(本项目按"不计工程成本"暂不作决策依据,但技术上须验);③GenEnv 撞车核查。
> 数据原件:本 session task a470b70bc015c068a 输出。Genie(2402.15391)=learned simulator 非 UED 生成器,切题外。

---

## 5.6 ★★★方向定型：对标 DiCode（Craftax + 多 LLM teacher 课程，2026-06-30 决定性）

> 用户设定：**multiple LLM teachers + single RL student**，直觉=LLM teacher 在"指导 student 通关 2D Minecraft（完整 Craftax）"上有语义优势（懂科技树依赖）。
> 用户要求先确认 OMNI-EPIC 是否真是此方向最强 SOTA → agent 核实：**不是，已被 DiCode 超越。**

### ★真正的 SOTA = DiCode（arXiv:2602.08194，2026-02，Mitsides/Faldor/Cully，Imperial）
- **不是 OMNI-EPIC**：OMNI-EPIC（2405.15568）不跑 Craftax、无科技树数字；DiCode 是其直接继任（Faldor 同为作者），related-work 原话 "OMNI-EPIC focuses on isolated behaviors, whereas DiCode orchestrates **sequences of environments that scaffold learning**"。
- **DiCode 是什么**：FM（开源 **Qwen3-235B**）按 parent level + agent competence "dream" 出新 level 描述→可执行 Python 代码→编译检查→训练批 = 目标env + 新level + archived(PLR采样)。机制 = **PLR + ACCEL 式变异 + learnability curation + FM 当代码生成器**。学生 = **RL（PPO-GTrXL）**。固定 Craftax 引擎执行代码（无学习式世界模型→保证物理有效）。
- **涌现 teacher 行为**：agent 变强时 FM 自动撤掉资源脚手架，维持成功率~0.5（最近发展区）。

### ★Craftax 课程纪录（DiCode 持有，全 verified from 2602.08194）
- 口径：5 seed、held-out **1024 关** 程序生成 Craftax、所有方法共用 **PPO-GTrXL** agent。
- **主指标 final mean return：DiCode 48.33 vs 最强 baseline PPO-GTrXL 41.54（+16%相对）**。
- **关键：DiCode 直接和 SFL/PLR/DR 比过**（之前顾虑"Craftax 无 regret-UED 对照表"=已被 DiCode 解决，擂台现成）。
- 后期科技树突破（baseline 全 0%）：打地精战士 11% vs 0% / 地精弓手 9% vs 0% / 铁甲 45% vs 14% / 钻石剑 6% vs 3% / 到达地精矿洞2层 30% vs 9%。
- ablation DiCode-OL（开环无反馈）40.91≈PPO-GTrXL baseline → 增益来自**闭环 learnability 课程**，非裸 FM 生成。
- 注：原 Craftax 18% max-reward 天花板是另一指标（max-reward-%），DiCode 用 mean return，不直接换算。

### ★你的 novelty：精确定位 + 确认未被占（verified 多轮搜索）
- DiCode/OMNI-EPIC/OMNI/Eurekaverse/EnvGen **全是单 FM，无 auction、无多 FM 协作**。
- 多 LLM debate/ensemble 文献存在但**学生是 LLM**（非生成课程喂独立 RL student）=另一问题。
- **{多/协作 FM teacher} × {auction 选 generator} × {RL student} × {Craftax 科技树}** 组合 = 真空。
- 最近单轴先例：(a) DiCode = 单 FM 代码生成器 + RL + Craftax；(b) 本项目已有 auction 工作 = auction + RL 但非 FM 生成器。**没人融合 FM 代码生成 + 多生成器 auction。**

### ★方向定型（卖点 + 对标 + baseline）
- **卖点框架**："DiCode 证明**单 FM** 课程在 Craftax 有效；我们证明一个 auction 选择的**竞争性多 FM generator 市场**更好。"
- **要打败的精确目标 = DiCode**（代码开源 github.com/konstantinosmitsides/dreaming-in-code）：主指标 mean return >48.33（同款 PPO-GTrXL + 1024 关 held-out + 5 seed 才可比）；次要 = 超/平 打地精战士11%/铁甲45% 等。
- **天然 ablation**：你的方法设 **N=1 FM ≈ DiCode**，干净。
- **学生类型已锁 = RL agent**（PPO-GTrXL），与 SFL 同问题类，非 LLM 学生（避开 DéjàQ 赛道）。

### 待消不确定性（执行前）
- DiCode 全文精读（机制细节 + 它的 learnability 实现，避免你的 auction 与它的 PLR curation 撞）。
- 多 FM auction 的可行性：N 个 Qwen3-235B 级 FM 调用成本/延迟（按"不计工程成本"暂不作决策依据，但要心里有数）。
- GenEnv(2512.19682) 撞车核（记忆 [[novelty-check-currA-gate-prior-art]] 已标，未重核）。
> 数据原件：本 session task a3ba045b36bec6b12。EnvGen(2403.12014)=Crafter+JSON 配置（DiCode 称其"limited expressivity"）。

---

## 6. 注意事项 / 待亲验

- Kinetix 各档解决率、DEGen/CENIE/TRACED 标量表、ACCEL Bipedal/PerfectMaze 数、CAMAR 逐方法数字 = **[转述]**（抓论文正文/project page 非独立重新推导），**入论文表前须抠 PDF 小数**。
- CAMAR 的"未解"状态依据作者框架，非可取到的逐方法对照表。（多智能体 JaxNav 已订正：SFL 论文 Fig5/6 有逐方法表，见 §1.5。）
- DEGen "只看终局"批评句 = 子任务间有冲突，引用前亲抠 2601.14957 原文。
- 2026 年极新论文（DEGen/NCC/DéjàQ/iMac/COvolve 2603.28386/Dreaming in Code 2602.08194）= 单源、recent，数字按 speculative 对待。

---

## 来源 URL 汇总

SFL/No Regrets 2408.15099 · NCC/Optimisation Framework 2505.20659(github nmonette/NCC-UED) · LILO 2502.12272 · DEGen/MNA 2601.14957 · DéjàQ 2601.01931 · Kinetix 2410.23208 · Craftax 2402.16801 · PAIRED 2012.02096 · Robust PLR 2110.02439 · ACCEL 2203.01302 · ReMiDi/BLP 2402.12284 · CENIE 2502.05726 · TRACED 2506.19997 · 目标误泛化 MMER 2507.03068 · CAMAR 2508.12845 · Stabilizing PAIRED 2308.10797 · MAESTRO 2303.03376 · Unsupervised Partner Design 2508.06336 · Beyond Fixed Tasks 2511.12706 · Hierarchical UED 2602.09813 · Eurekaverse 2411.01775 · iMac 2509.13341 · OMNI-EPIC 2405.15568 · Open-Endedness manifesto 2406.04268 · DISCOVER 2505.19850 · Genie 2402.15391 · ADEPT 2506.01759 · SHED 2310.00301 · minimax taxonomy 2311.12716
