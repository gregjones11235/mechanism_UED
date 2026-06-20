# idea 可行性分析：non-IC 机制 + 异质估计器 UED

*配套文档：[UED实验步骤指导.md](UED实验步骤指导.md)*
*分析日期：2026-06-16*

---

## 0. 文档说明

本文档是对核心 idea「non-IC mechanism（Gap A）+ heterogeneous estimator agents（Gap B）」两个目标的可行性分析，由一次质疑驱动的对话整理而成。所有承重墙级别的理论引用均经过联网核查（来源见文末）。

**两个质疑（讨论起点）：**

1. 「异质」是否纯靠尝试来判断哪些组合更好？
2. 「异质」是否只是把现有毫无关联的 agent 拼装，并无任何数学/理论说明这种合作会更好、能刷 SOTA？

**两个目标（可行性评估对象）：**

- **目标 A：** 特定领域刷 SOTA —— 准确定义为「做出与 MAESTRO/SFL 同量级的顶会贡献」，**不是**取代/动摇 ACCEL 的 SOTA 地位。
- **目标 B：** 建立新范式（auction/机制作为 UED 课程聚合层）。

> ⚠ 关键澄清：本 idea 的「异质」**不是异质 RL 算法的 agent**，而是**异质 estimator（评分函数）的 teacher**——PVL / Gen-SFL / CENIE（见 [UED实验步骤指导.md §4.3.1](UED实验步骤指导.md)）。三类 estimator 回答本质不同的三个问题（Regret 问「这关多难」、Learnability 问「现在学得动吗」、Coverage 问「去过没去过」），是有结构的正交，而非随机拼装。这个区别决定了两个质疑成不成立。

---

## 1. 质疑一：异质是纯靠尝试，还是有理论说明合作更好？

**结论：不是纯靠尝试——多样性收益有硬数学；但也不是定理保证 SOTA——所用聚合方式落在硬定理覆盖之外，且收敛保证在 general-sum 设定下按 NCC 原文就是「没有」。理论的真实角色是「解释方向 + 划定失效边界」，不是「保证赢」。**

分三层：

### ① 多样性带来收益 —— 这一层有硬数学，不靠尝试

ensemble 的 **ambiguity / bias-variance-covariance 分解**是教科书级结论：集成误差含一个 **covariance 项**，base learner 间相关性越低，集成误差越小；若所有成员犯同样的错，covariance 拉满，集成等于无用。

这恰好给出**异质优于 single-winner 端点（=只用一个 estimator）的数学理由**（⚑V5：原引已取消的 §4.2.2「同质 Stage 2 零结果」，现改述为 Stage 3 内部对照）：若所有 teacher 用同一 estimator，打分高度相关（covariance 大）→ 聚合无增益、退化为单 estimator；异质 estimator 打分去相关 → covariance 项被压低 → 聚合有净增益。这不是猜测，是 covariance 项的直接推论。**「为什么异质比同质好」是有数学的：你在压低 covariance 项。** 这也正是 Σ 冗余测量（T3/T4）要验的：异质 estimator 的 score 协方差是否真的低。

> ⚠ **隐蔽软肋 (a)**：ambiguity 分解严格成立的前提是「对输出做平均」（回归/分类的 soft label）。而 [UED实验步骤指导.md §4.3.4](UED实验步骤指导.md) 自己承认：UED 里 teacher 输出是具体关卡、不能平均，只能在分布层组合或 selection。**一旦从「平均」退到「选一个/分布加权」，ambiguity 分解的干净保证不再直接适用。** 这是多样性收益的硬定理恰好不覆盖实际聚合方式的裂缝——审稿人会真正发难的地方之一。

### ② 三类 estimator 正交 —— 这一层是「领域论证」不是「定理」

「Regret/Learnability/Coverage 问三个不同问题」有文献支撑（DIPLR、CENIE 都论证「光靠 regret 会选到冗余关卡，需 diversity/coverage 补」）。但**「三者组合 > 任意单个」在 UED 里没有定理，只有 empirical 证据 + 直觉**。

文档把它放进消融（[§4.3.6](UED实验步骤指导.md)：去掉 CENIE 看掉不掉）是诚实做法——等于承认「组合是否更好要靠实验确认」。**质疑一在这一层部分成立**：哪个组合更好最终要靠消融，不能纯靠理论推出。

### ③ general-sum 收敛 —— 文档诚实地承认「会崩」

NCC 原文（arXiv:2505.20659）核实：收敛保证**只在 zero-sum（regret 类 score）成立**，明文说对非零和 score function "lacks convergence guarantees"。[UED实验步骤指导.md §2.2 / §6.2](UED实验步骤指导.md) 没有隐瞒——反而把「刻画从零和滑向 general-sum、保证崩溃的边界」当成 EGTA 分析的卖点。

> 这是 idea 最聪明也最危险处：它不声称「我的异质系统收敛」，而声称「我来研究它何时不收敛」。失效边界刻画是合法的学术贡献，但意味着「异质合作更好」在理论上恰恰没有保证——你研究的正是它不保证的那个区域。

---

## 2. 质疑二 + 工作量评估：MAESTRO/SFL 的「净增益」长什么样，我的够格吗

### 2.1 顶会 UED 论文的「净增益」窄得惊人（核查结果）

**MAESTRO（arXiv:2303.03376，ICLR 2023）的全部 novelty = 一个观察 + 一个联合采样。**
观察：co-player 强弱依赖环境特征，所以环境课程与对手课程不该独立做，要 joint 采样 (environment, co-player) pair。没有新 estimator，没有新收敛定理。**净增益来源：把一维扩成二维联合。**

**SFL（arXiv:2408.15099，NeurIPS 2024）的全部 novelty = 一个否定 + 一个替换。**
否定：实验证明 PVL/MaxMC 等 regret 近似与真 regret 几乎不相关。替换：score function 从 regret 换成 `p(1-p)` learnability。**净增益来源：换掉一个评分函数 + 用实验打脸旧评分函数。**

> **关键认知修正**：先前担心「我的净增益来源很窄（几乎全压在 non-IC 机制一个旋钮上）」是个误导性批评。顶会 UED 论文的净增益本来就这么窄——MAESTRO 是一个采样维度，SFL 是一个评分函数。**窄不是病，孤证才是病。** 窄但干净、且配一个有说服力的「为什么」+ 反例/消融，才是顶会的标准形态。non-IC 机制旋钮在「窄度」上完全够格，和「joint 采样」「换 score function」是同一量级的单点创新。**它目前缺的不是宽度，是这种「配套论证 + 反例/消融」的厚度。**

### 2.2 三种组合方式的判断

**情况一（不推荐）：non-IC 机制 + 自研 estimator + 蒸馏，三个并列卖点。**
顶会审稿人对「一篇塞三个创新」通常负面——质疑「哪个真正起作用？是不是每个单拎都不够强才凑一起？」。会放大 [§4.3.5](UED实验步骤指导.md) 已警惕的「增益不可归因」问题。

**情况二（推荐，本项目采用）：non-IC 机制为主轴，自研 estimator 作为「为异质组合/机制聚合场景定制」的配套组件。**
关键在自研 estimator 的定位——**不是第四个独立评分函数**（那是情况一的堆量），而是**专门为 auction/机制聚合设计、让 teacher 的 bid 更可分离、更能体现 covariance 下降的 estimator**。故事变成：「提出把机制设计接入 UED（主 novelty），并发现现有 estimator 在 auction 语境下次优，故设计适配的 estimator（支撑组件）」。**这正是 MAESTRO 的结构**（主 novelty = joint 采样，population learning 是支撑组件）。

**情况三（蒸馏定位）：蒸馏是 plan B，不进主线。**
[§6.4.1](UED实验步骤指导.md) 写死：蒸馏仅在主线失效时启用；[§6.4.3](UED实验步骤指导.md) 明确蒸馏与 auction 在「信息聚合」职责重叠、同开互相干扰。蒸馏的 novelty（UED 多 teacher 蒸馏作聚合是空白）足够撑**另一篇**，塞进主线只会稀释叙事。

### 2.3 够格的配方（与 MAESTRO/SFL 对齐）

| | 主 novelty（单点） | 支撑组件 | 理论/实证厚度 |
|---|---|---|---|
| MAESTRO | joint env×co-player 采样 | population learning | minimax-regret @ Nash + 多环境实验 |
| SFL | learnability 换 regret | buffer 维护算法 | 「regret 近似失效」反例实验 |
| **本项目** | **non-IC 机制接入 UED** | **为机制聚合定制的 estimator** | **EGTA 零和→general-sum 边界 + λ-frontier 消融** |

**这一行填满 = MAESTRO/SFL 同量级的顶会工作。** 工作量不是问题——真正的风险是抵抗住「把蒸馏和多个 estimator 都塞进来」的诱惑，守住单主轴 + 厚论证。

---

## 3. 两个目标的可行性结论

### 目标 A（做出 MAESTRO/SFL 量级的顶会贡献）：中等偏高

修正说明：先前用「动摇 ACCEL 地位」当标尺误判为「中等偏低」；按真实目标（MAESTRO/SFL 量级），上调为**中等偏高**。理由：顶会 UED 论文的净增益本就是单点创新（§2.1），non-IC 机制旋钮在量级上对齐；缺口是论证厚度而非创新宽度，而这是可补的（§2.2 情况二）。

> 仍需清醒：PerfectMaze 8% 是极硬、区分度极高的靶子（[§4.1.2](UED实验步骤指导.md)），ACCEL 至今未被稳定超越，这正是 MAESTRO/SFL/NCC 都「绕道」Craftax/Kinetix 立功的原因。做出「特定分支超过已发表 ACCEL/PLR」是现实的（Stage 4 选址正是如此）；不要把目标放在硬靶子上。

### 目标 B（建立新范式）：显著高于目标 A

「auction/机制作为 UED 课程聚合层」确实是空白（[§6.4.4](UED实验步骤指导.md) 两轮 novelty 核查 0 撞车，与 memory `distillation-novelty-blank-check` 一致；"auction × 课程/UED" 入口本身空白）。

**范式贡献不依赖刷赢 SOTA**：[§5.2.3](UED实验步骤指导.md) 明确「若 truthful 反而最优，这本身可发表」。方案设计已把可行性下限焊死在「建立分析框架」上，而非赌「必须赢」。

> ⚠ **重要降级（取代先前定位）**：先前把 [§6.3](UED实验步骤指导.md) 的 EGTA「刻画零和→general-sum 收敛崩溃边界」当成可独立支撑范式贡献的卖点。经第二轮分析（§4.5），这个定位**过高，必须降级**——EGTA 边界刻画不能当主 novelty，且不应靠实验去精确刻画。正确的范式主菜是 **non-IC 机制接入 UED（constructive）+ 让 general-sum 重新稳定的自研 estimator（出路三，§4.5）**。EGTA 退回为支撑材料（小配置解析），不是主菜。

**可行性排序：目标 B > 目标 A。** 若要赌，赌 B；把 A 当作支撑 B 的证据，而非项目成败线。

---

## 4. 两个隐蔽软肋（审稿人真正会发难处，需提前防御）

并非用户最初提的那两个质疑，而是更深的两条：

- **软肋 (a)：ambiguity 分解的「平均」前提不覆盖实际用的 selection/分布聚合**（详见 §1①）。
  - 防御方向：为 selection/分布聚合设定补一个 ambiguity 类的近似论证；或明确承认并把它作为 EGTA 分析要刻画的对象之一。
- **软肋 (b)：Babaioff-Sharma-Slivkins 定理是广告拍卖（pay-per-click）setting，搬到 UED 是借喻不是移植。**
  - 原定理结论：truthful MAB 机制必须分离 explore/exploit，且 worst-case regret 被卡在 Ω(T^(2/3))，劣于最优 bandit。
  - 文档用作「truthful 不是最优、中间 λ 更好」的依据（[§4.3.3](UED实验步骤指导.md)），但 UED 的 teacher-student 不是该论文 setting，故它给的是**动机**（有理由期待中间 λ 更好）而非**保证**（中间 λ 一定更好）。
  - 防御方向：明确把 Babaioff 降格为 "motivation" 而非 "guarantee"；λ-frontier 的实证结果（[§5.2.3](UED实验步骤指导.md)）才是真正的证据。

---

## 4.5 EGTA 边界刻画的定位修正 + 出路三的可行性

本节是第二轮分析的核心产出，回答两个新质疑：(1)「刻画零和→general-sum 崩溃边界」够不够发顶会；(2) 纯靠实验验证是否太耗时。结论：**两个质疑都成立，EGTA 须降级；且不稳定的根源在 score function 层，故自研 estimator 是从根上恢复稳定（出路三）的正确且唯一的杠杆。**

### 4.5.1 EGTA 边界刻画不能当主 novelty（质疑一成立）

参照最近的 UED+理论论文 NCC（arXiv:2505.20659，RLC 2025）：它的主卖点是 **provably convergent 算法（constructive，正面构造「我证明能收敛」）**，且配新 score function（Gen-SFL）+ 实验超越 prior methods 的四件套。

而 EGTA 边界刻画卖的是反面，有两个致命弱点：

- **方向是负面、描述性的**：「证明 X 收敛」是定理（constructive），「刻画 X 何时崩溃」是观察（descriptive）。顶会主贡献偏爱前者。且「general-sum 会崩」本就是 NCC future work 里的脚注级内容——把脚注当主菜，审稿人会问「so what?」。
- **EGTA 单独当主贡献门槛极高**：核查显示，EGTA 作 primary contribution 的论文（PSRO 系、Meta-Game Evaluation arXiv:2405.00243）无一例外是**提出新的 EGTA 方法论/求解器**；「用现成 EGTA 分析一个现象」几乎总是附属分析。[§6.3](UED实验步骤指导.md) 正是后者形态。

> **结论**：EGTA 边界刻画最多是 §2.3 配方里「理论/实证厚度」那一栏的**支撑材料**，与 MAESTRO 的 "minimax-regret @ Nash" 同级，**不是主菜**。

### 4.5.2 纯靠实验精确刻画边界，成本灾难（质疑二成立，比质疑一更要命）

三重叠加：

1. **UED 方差极大**：memory `stage1-failure-was-timeout-not-env` 与 [§7.5.1](UED实验步骤指导.md) 记录 ACCEL 有崩溃 seed（min solved_rate=0.01），3 seed 的 CI 宽到任何两 setting 重叠。要证「λ=0.3 收敛、λ=0.5 崩」，区分「崩」与「seed 运气差」本身就需满种子（8~10）。
2. **边界是连续轴非单点**：刻画边界要在 λ（或零和度）连续轴上扫多点，每点都要满种子确认它在边界哪侧——正是 [§7.5.2](UED实验步骤指导.md) 标红的「Stage 3 更耗时」之源（λ 轴 × 满种子）。
3. **崩溃本身难定义、难测量**：「收敛崩溃」是带噪声的软信号（课程震荡？性能不升？），从噪声里干净切出边界所需 run 数会爆炸。

→ 纯实验刻画边界 = 把项目最贵的部分压在一个不能当主菜的附属分析上，双输。

### 4.5.3 三条出路（出路三是唯一靠谱的，且能靠 estimator 实现）

- **出路一**：EGTA 降级为 N=2–3 小配置解析 payoff 表 + 定性结论，不追求精确边界。成本从「连续轴×满种子」降到「几个离散策略组合的 payoff」。
- **出路二**：把「边界」从因变量改成卖点注脚（一段话 + 一张 N=2 payoff 表，回答审稿人「为什么有时不稳」）。
- **出路三（采用）**：放弃负面描述，换成 constructive 命题——**「我的设计让 general-sum 系统重新稳定」**，与 NCC 同形态。

### 4.5.4 不稳定的根源在 score function 层 —— estimator 是正确且唯一的杠杆

NCC 原文核查（Eq 6 / Theorem 5.1）确认根源（详见 §6 来源表）：

- adversary 最大化 `g(x,y) = yᵀ·s(π,Λ) + α·H(y)`；
- 收敛保证**当且仅当 `s = -J`（score = 负回报 = regret，zero-sum）** 时成立——此时 nonconvex-strongly-concave 结构成立，two-timescale 梯度法（Lin et al. 2020）收敛；
- `s ≠ -J`（learnability/coverage）→ 结构破裂，保证消失。

> **关键**：不稳定**不来自** aggregation 机制、**不来自**训练动态，而来自 score function `s` 偏离 `-J`。这正是 estimator 所在的层。**破坏稳定的是 `s`，能从根上修复的也只能是 `s`（estimator）。** 收敛层（directedness/KL，[§4.3.5](UED实验步骤指导.md)）只是 `s` 破坏结构后的事后稳定化——治标；estimator 治本。

### 4.5.5 前置硬关卡的 deep-research 判定（2026-06-16，23 条 claim 经 3-vote 对抗验证）

原设想的两条路径：
- **路径 A（强）**：把 learnability/coverage 表达成对 `-J` 的重参数化/单调变换，使其仍落在 Theorem 5.1 条件内（扩展 NCC 收敛保证）。
- **路径 B（稳）**：写成 `s = -J + λ·(bounded term)`，扰动分析证明 λ 小于阈值仍收敛（得解析 bound 而非靠实验）。

**判定结果：三个 estimator 里两个把这两条路都走死了。**

| Estimator | 路径 A | 路径 B | 数学障碍（已验证） |
|---|---|---|---|
| **PVL** | 有条件可能 | 有条件可能 | `PVL=(1/T)Σ max(Σ(γλ)^{k-t}δ_k, 0)` = `max(advantage,0)`，与 regret 同族但**不等于 `-J`**：正截断非单调 + 依赖学习中的 critic（有偏）。最接近，但需把「截断非光滑」和「critic 偏差」一起 bound 住 |
| **Gen-SFL learnability** | **失败** | **失败** | `p(1-p)` 是 Bernoulli 方差/回报离散度；式11 `s=σ_λ·N(μ_λ\|µ,σ²)` 是回报标准差×高斯密度。SFL 实测与 PVL 仅弱相关、与 MaxMC 零相关；Beukman 2024 给出「高真 regret + 低 learnability」反例 → 既非 `-J` 的单调变换（A 死），也非小扰动而是**独立项**（B 死） |
| **CENIE coverage** | **失败** | **失败** | `Novelty=-(1/\|X\|)Σ log p(x_t\|λ_Γ)` 是状态-动作访问密度的负对数似然（单位 nats，与回报量纲不同），**可证与回报无关**；且 CENIE **根本不在 score 层相加**——它在采样概率层凸组合 `P_replay=α·P_N+(1-α)·P_R`，Theorem 5.1 的「加性扰动」模板压根不适用 |

> **核心教训**：learnability 和 coverage 之所以有用，正因它们**故意捕捉 regret 之外的信息**——这同一性质恰好让它们无法被写成「regret + 小扰动」。**路径 B 对异质核心（learnability/coverage）是结构性失败，不是调参能救的。原设想的「自研 estimator 伪装成 regret」思路作废。**

### 4.5.6 出路三的新形态：用正则项独立扛收敛，而非把异质 score 伪装成 regret

deep-research 给出四条修补路线，按「前提在 UED 里现不现实」排序：

- **修补 (i)（最现实，新主路径）：proximal/熵正则独立提供 strong-concavity。**
  关键洞察（来自对 NCC Eq.6 的核查，caveat：该 nuance claim 过 2-1）：**NCC 的 strong-concavity 本来就不来自 `s=-J`，而来自它已加的熵正则 `αH(y)`；`s=-J` 只是让 score 项在 y 上线性/zero-sum。** ⇒ 即使换成异质 score，只要保留并调强 `αH(y)`（或换 proximal 项），strong-concavity 仍由正则项提供——**收敛结构不必依赖 score 是不是 regret。**
  - 这把出路三从「必须把异质 score 伪装成 regret」（已证失败）解放为「**让异质 score 自由，用正则独立扛 strong-concavity**」。
  - 代价：收敛到的是**被正则修改过的目标**的不动点，非原始异质目标 ⇒ 必须补「正则偏差有界」论证（见 §4.5.7 openQ）。

- **修补 (ii)：投影到 `-J` 兼容子空间——仅对 PVL 类有效**，对 learnability/coverage 无效（与 `-J` 独立，投影会把信息投没）。

- **修补 (iii)：n-sided PL 条件（arXiv:2602.11835，AAMAS 2026）——前沿但未架桥。** 2026 新结果，把梯度占优推广到 n-player general-sum 并证 GD 收敛到 Nash；但是通用博弈结果，**没人验证过 UED 目标满足 PL 不等式**，且允许多 Nash、收敛率退化。

- **修补 (iv)：改投 bilevel（Hong 2020）——不是干净出路，是换个新障碍。** bilevel TTSA 要求**内层 strongly-convex**，而 student RL 内层是 nonconvex，**前提不成立**；NCC 作者也只把 bilevel 列为 future work 未做。**「改投 bilevel」比「修 minimax」更难，不更易**——这直接否定了 [UED实验步骤指导.md §2.3](UED实验步骤指导.md)「bilevel 是正确框架」的乐观假设（见下方地基甄别）。

> ⚠ **地基甄别（已确认）**：「two-timescale」两个来源数学对象不同、扮演对立角色：
> - **Lin-Jin-Jordan 2020（arXiv:1906.00331，nonconvex-concave minimax GDA）**：NCC 的 Theorem 5.1 收敛证明引的是这个；解 `min_x max_y` 鞍点。
> - **Hong 2020（arXiv:2007.05170，bilevel TTSA，内层 strongly-convex）**：[UED实验步骤指导.md §2.3](UED实验步骤指导.md) 引的是这个。
> - **判定**：异质 score 破坏 minimax 的 strong-concavity 时，改投 bilevel **并不更自然**——其内层 strongly-convex 前提在 nonconvex 的 student RL 内层不成立，是新障碍。**正确出路是修补 (i)（在 minimax 框架内用正则恢复 strong-concavity），不是改投 bilevel。**

### 4.5.7 撞车检查（好消息）+ 待解 open questions

**撞车检查（medium confidence，源自 NCC future work + CENIE 无收敛分析 + n-sided PL 未特化）**：**没有任何已有工作把 non-regret 的 learnability/coverage/diversity score 纳入带收敛保证的 UED 框架**；NCC 自己明文把 general-sum UED 收敛列为 open future work；唯一的 general-sum 收敛结果（n-sided PL）是通用的、没接到 UED 上。**这正是 novelty 缺口——NCC 作者亲手留下的 open problem，撞车风险低。**

**待解 open questions（决定出路三新形态能否落地）**：
1. PVL 的正截断 `max(·,0)` 非光滑性是否破坏 Lin-Jin-Jordan Thm 4.5 所需的 ℓ-smoothness？
2. n-sided PL 条件对任一具体异质 UED 目标（熵正则 Eq.6 + learnability/coverage score）是否真成立？（通用结果与 UED 之间没人架过的桥）
3. CENIE 的扰动作用在**采样分布层**（非 score 层），是否存在「一玩家优化 regret score、异质项作为对采样分布的有界扰动」的收敛框架？（与路径 B 不同的扰动靶点）
4. 能否构造一个 strongly-concave 的 proximal 代理，其不动点可证地在可控偏差内逼近异质 score 的最优？（即修补 (i) 的偏差有界性）

> **所有收敛保证的天花板（caveat 5）**：本节讨论的一切收敛都是到 **ε-近似一阶 Nash / Φ 的驻点**，从不是 global Nash。任何扩展保证都继承此上限。

---

## 5. 待办（落实出路三新形态 + 情况二）

> 前置硬关卡（原「能否写成 `-J+λ·bounded term`」）已由 deep-research 回答：**对 learnability/coverage 结构性失败**——主路径改为「正则独立扛收敛」（修补 (i)，§4.5.6）。证明蓝本见 [proof_skeleton_entropy_regularized_heterogeneous_UED.md](proof_skeleton_entropy_regularized_heterogeneous_UED.md)，已经 Gemini 独立重证 + 文献核实（V2 修正）。**以下待办已据 V2 重排。**

**已解决（V2 交叉验证，2026-06-16）：**
- ✅ 定理 2 偏差上界 `α log n` = Nesterov 2005 smoothing 推论，**可直接引用、无需自证**。
- ✅ 引理 1.3 误判更正：退化的是 smoothness 非 strong-concavity；改用 **two-timescale mirror descent**（KL 内层）绕开边界 Lipschitz。
- ✅ 值近⇒解近：两模型一致，**决定只 claim value suboptimality，不 claim 解接近**（标准做法）。
- ✅ PVL 非光滑：Softplus smoothing，偏差 `O(α)` 与 `α log n` 同阶可吸收。
- ✅ CENIE 采样层障碍：用 `s_i=-log p(x_i)` 重构，内层解析解 = 反密度采样，绕开。

**(1) 唯一剩余真 🟡（最高优先，下一步深挖）：** **mirror descent + 两时间尺度 + 随机 + NC-SC + KL 内层** 的完整收敛定理无单篇现成覆盖（Zhang 2021/Lin-Jin-Jordan 都仍需 smoothness，绕不开边界问题；mirror descent 那条线又没覆盖完整随机两时间尺度组合）。需文献拼接或局部自证。

**(2) 复核义务（不可省）：** proof_skeleton 中 Gemini 的代数（Hessian `-α diag(1/y_i)`、LogSumExp 解析解、`y_i*∝p^{-1/α}` 的**指数符号方向**）仍是单模型输出，须人工/Lean 复核。引用 Zhang 2021（UAI，arXiv:2103.15888）须独立确认覆盖范围，勿与 mirror descent 混引。

**(3) 自研 estimator 的重新定位（情况二）：** 不再是「能写成 regret 变体的 estimator」（已证不可行），而是**为「正则可控的异质聚合」设计的 estimator**——在修补 (i) 框架下让异质项与正则配合、偏差最小的 score 形式（CENIE 反密度采样即一例）。

**(4) 风险对冲：** 若 (1) 的完整定理拼不出，退回 §4.5.3 出路一（EGTA 小配置解析 + 定性结论），收敛性作支撑材料而非主定理。

**(5)（可选）deep-research：** ambiguity 分解在 selection（非平均）聚合下是否有已知扩展（软肋 a 防御，§4）。

---

## 6. 核查来源（2026-06-16）

| 承重墙 | 核到的实际结论 | 来源 |
|---|---|---|
| Babaioff 定理 | truthful MAB 机制须分离 explore/exploit，worst-case regret ≥ Ω(T^(2/3))，劣于最优 bandit；广告拍卖 setting | arXiv:0812.2291 |
| NCC general-sum | 收敛保证仅 zero-sum 成立，非零和 score "lacks convergence guarantees" | arXiv:2505.20659 |
| NCC 收敛条件根源 | Theorem 5.1 / Eq 6：adversary 最大化 `g=yᵀs+αH(y)`，**当且仅当 `s=-J`（score=负回报=regret）** 时 nonconvex-strongly-concave 结构成立、two-timescale（Lin 2020）收敛；根源在 score function 层（非 aggregation、非 dynamics） | arXiv:2505.20659（HTML 全文核查） |
| EGTA 作主贡献门槛 | 作 primary contribution 的论文均为「提出新 EGTA 方法论/求解器」（PSRO 系）；「用现成 EGTA 分析现象」几乎总是附属分析 | arXiv:2405.00243；arXiv:2403.04018 |
| SFL 净增益 | 否定 regret 近似（与真 regret 不相关）+ 换 `p(1-p)` learnability | arXiv:2408.15099 |
| MAESTRO 净增益 | joint 采样 (env, co-player) pair；无新 estimator/收敛定理 | arXiv:2303.03376 |
| ACCEL 靶子 | PerfectMaze 极硬、未被稳定超越 | arXiv:2203.01302 |
| ensemble 多样性收益 | covariance 项：成员相关性越低集成误差越小（ambiguity / bias-variance-covariance 分解） | Chen et al. IJCNN 2011 tutorial；Diversity creation survey (ScienceDirect) |
| DIPLR / CENIE | 光靠 regret 选冗余关卡，需 diversity/coverage 补 | IJCAI 2023；AAMAS 2024 |
| EGTA | 用 meta-game 刻画过复杂以致无法解析的策略域 | arXiv:2403.04018 |
| **estimator 可表达性判定** | PVL=`max(adv,0)` 与 regret 同族但≠`-J`（截断非单调+critic偏差）；learnability `p(1-p)`/式11 与 regret 统计不同、有反例（路A/B 均死）；CENIE=访问密度负对数似然、与回报无关、且在采样概率层凸组合非 score 层（路A/B 均死） | arXiv:2203.01302；2408.15099；2505.20659；2502.05726 |
| **strong-concavity 真来源** | NCC 的 strong-concavity 来自熵正则 `αH(y)`，非 `s=-J`；⇒ 异质 score 下仍可由正则独立提供（修补 (i) 的依据） | arXiv:2505.20659（Eq.6 核查，2-1） |
| **bilevel 不是出路** | Hong 2020 bilevel TTSA 要求内层 strongly-convex，student RL 内层 nonconvex 不满足；NCC 仅列为 future work | arXiv:2007.05170；1906.00331 |
| **general-sum 收敛先例** | n-sided PL（AAMAS 2026）证 n-player general-sum GD 收敛到 Nash，但通用、未特化 UED、多 Nash/率退化 | arXiv:2602.11835 |
| **撞车检查** | 无已有工作把 non-regret score 纳入带收敛保证的 UED；NCC 明文留为 open problem | arXiv:2505.20659；2502.05726 |

*§6 后六行为 2026-06-16 deep-research（19 源 / 89 claim / 25 验证 / 23 confirmed）产出。完整报告见 workflow run `wf_65d9c353-f3f`。*
