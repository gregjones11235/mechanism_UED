# 多Teacher机制驱动UED实验步骤指导

*Multi-Teacher Mechanism-Driven Curriculum for Unsupervised Environment Design*

**Jiayu Zhu (Alec) & Hrithik Nambiar  |  版本日期：2026年6月**

---

## 0. 文档说明

本文档汇总了多次讨论后的完整实验设计，基于以下核心材料：

- Experiment Design Pitch（原始方案）
- 与Hrithik的Slack讨论（bilevel优化、MiniGrid起步策略、聚合方式）
- 文献精读：ACCEL、SFL/No Regrets、NCC、CENIE、TRACED、ReMiDi、MAESTRO
- 关键判断：ACCEL仍是标准SOTA基准；SFL/NCC的learnability可直接融入系统作为非regret类估计器；CENIE可作为第三类覆盖率估计器；bilevel优化（Hong et al. 2023）是general-sum多teacher系统的正确理论框架

---

## 0.4 V5 规划修正指针（2026-06-19，Stage 2 取消 + teacher≡estimator 落定）

> ⚑ **重大规划变更：原"Stage 2（同质多 teacher 种群）"已取消，直接进入异质 estimator + auction 阶段（仍编号 Stage 3）。** 依据见 main.tex §3.1（teacher *j* scoring with estimator `s^(j)`）与三段对话核查（`45b9e91a` line 315 / `d67916a9` 进度）。
>
> **取消原因（核心）**：在本框架里 **teacher ≡ estimator（打分函数），一对一**。"多 teacher" 和 "多种 estimator" 是**同一个维度的两种说法**，不是两个独立可解锁的变量。因此**不存在一个"只增加 teacher 数量、却仍同质（全 PVL）"的中间阶段可作隔离基线**——原 Stage 2 想隔离的"多 teacher 本身有没有用"是个伪变量（多 teacher 必然意味着多 estimator，否则就是同一个 estimator 复制 N 份、退化为单 teacher）。
>
> **三个现成聚合方式（round-robin / EXP3 / argmax）的新定位**：它们是 auction 机制族的 **single-winner 退化端点**（`w=e_{j*}`，§4.3.4 ⚑V2），因此**自动成为 Stage 3 auction 的消融对照基线**，不需要一个独立 Stage 去单独跑它们。
>
> **当前实体**：跑的是 `multi_estimator_plr` runner（**PLR 血统、`n_students=1`、无 teacher 生成器网络**，3 个异质 estimator PVL/SFL/CENIE + non-IC auction）。原 PAIRED 式 `multi_teacher` runner（[STAGE2_实现方案.md](STAGE2_实现方案.md)）**已弃用为主线**，降级为 fallback 探索方向（见 §4.3.2 ⚑边界：会破收敛定理，只能走分层路线）。
>
> **本文档下列小节据此就地改写（标 ⚑V5）**：§1.2 变量层级表、§4.0 配置总览、§4.2（原"同质 Stage 2"已删/改为指针）、§5.1 预期性能表、§7 时间线。Stage 编号保留（Stage 3 异质+auction、Stage 4 推 SOTA 不变）。

---

## 0.5 V2 理论修正指针（2026-06-16，收敛性分析 + 两轮 AI 交叉验证后）

> 本文档原始版本（2026-06）写于收敛性分析之前。此后经 deep-research + Gemini 两轮证明 + Claude 红队，对「异质 estimator + 机制」的**理论支撑与收敛性**有了实质结论，**直接改动以下实验设计**。详细推导见 [idea可行性分析.md](idea可行性分析.md) §4.5 与 [proof_skeleton_entropy_regularized_heterogeneous_UED.md](proof_skeleton_entropy_regularized_heterogeneous_UED.md)。本文档相关小节已就地标注 ⚑V2。

**五条核心修正（影响本文档实验）：**

1. **异质 score 无法伪装成 regret**：learnability/coverage 与 regret 统计独立，不能写成「regret + 小扰动」。收敛保证改由**熵正则 `αH(y)` 独立提供**（strong-concavity 本就来自正则项、非来自 `s=-J`）。
2. **新增 α 轴（熵正则强度），与 §4.3.3 的 λ 并列**：α 是「优化稳定性 ↔ 异质保真度」的解析旋钮，偏差 `Φ_α − Φ_0 ≤ α log n`。
3. **claim（V4 更新，原"路线 B"已升级）**：「**多项式时间收敛到 α-正则化目标 `Φ_α`**」，率 `K=O(1/(α³ε⁴))`（大 batch，与 NCC 同 ε 阶）/ `O(G_s²σ²Δ/(α³ε⁵))`（单样本）。⚠ **原路线 B 的"α→0 因 `e^{2B_s/α}` 指数爆炸不可行"已被推翻**（proof_skeleton §7.9：指数是记账失误，走对偶 LogSumExp 收紧为多项式）。α→0 现仅多项式代价、原则上可达；仍**报告**在固定适度 α（因 α 是「优化代价 K∝α⁻³ vs 偏差 ≤α log n」的解析旋钮，是建模选择非可解性必需）。无截断、无 `e^{2B_s/α}`。
4. **增益归因消融**（⚑V5 改：原"嵌入 Stage 2"）：single-winner 端点（=单 estimator）基线须**也加 α 熵正则**，作为「熵正则本身贡献 vs 异质性贡献」的对照组——异质 N=3 须显著高于"single-winner + 同 α"才能把增益归给异质性（见 §4.2 去向说明）。
5. ⚠ **（V4 改写）原"崩溃边界由理论 `e^{2B_s/α}` 给公式"已失效**：指数收紧为多项式后，理论**不再预测** α 太小会指数级崩溃。「α 太小是否致训练不稳」现降级为**纯经验问题**（多项式率下小 α 只是常数变大、优化变慢，非理论崩溃）。原"免费的理论-经验对应崩溃实验"叙事作废；若仍观测到小 α 不稳，那是工程/方差现象，不能再归因于理论阈值。α 扫描的动机改为纯粹的「偏差 vs 优化代价」权衡（见 §4.3.3 ⚑V4）。

---

## 1. 研究方案概述

### 1.1 核心想法

N个异质teacher agent各自对候选关卡提出评分（出价/bid），选择机制将这些出价聚合成课程。两个正交的研究轴：

- **Gap B — 估计器异质性（主轴，多样性来源）：**
  - 不同teacher使用结构上本质不同的估计器（PVL、Gen-SFL/learnability、CENIE覆盖率）
  - 多样性来自「目标哲学不同」，而非给同一信号加噪声

- **Gap A — 机制truthfulness（副轴，exploration旋钮）：**
  - 从truthful（VCG）到可调非IC的拍卖机制，truthfulness本身作为hyperparameter
  - 理论依据：Babaioff-Sharma-Slivkins定理——强制truthfulness在bandit学习系统中是binding约束，会导致更高的regret

- **收敛层：** directedness正则（CURATE-style）+ KL bound + VCG-style置信区间，防止异质teacher使课程震荡

### 1.2 关键变量层级（控制变量逻辑）

实验逐个解锁变量，不同Stage对应的变量状态：

| 变量 | Stage 1 | ~~Stage 2~~（已取消，⚑V5） | Stage 3 | Stage 4 |
|------|---------|---------|---------|---------|
| 多teacher＝多estimator | ✗（单teacher原版算法） | ~~伪变量，见下~~ | ✓ ON（N=3：PVL+Gen-SFL+CENIE） | ✓ ON |
| 异质估计器 | ✗ | ~~✗（全用PVL，同质）~~ | ✓ ON（PVL+Gen-SFL+CENIE） | ✓ ON |
| Tunable non-IC机制 | ✗ | ~~✗（现成聚合方式）~~ | ✓ ON（VCG→tunable-λ；现成聚合=single-winner端点） | ✓ ON |
| 复杂环境 | ✗（MiniGrid/BipedalWalker） | — | ✗（MiniGrid/BipedalWalker） | ✓ ON（Craftax/Kinetix） |

> ⚑ **V5（2026-06-19）——原 Stage 2 已取消**：因 **teacher ≡ estimator（§0.4）**，"多 teacher（ON）+ 异质估计器（OFF）"这一格**是矛盾的伪状态**——多 teacher 必然就是多 estimator，无法"多 teacher 却仍同质"。故删去独立 Stage 2，直接进 Stage 3。三个现成聚合方式（round-robin/EXP3/argmax）作为 Stage 3 auction 的 **single-winner 退化端点**自动成为消融基线（§4.3.4）。

> ⚠ **注意：** Stage 3的novelty在于Gap A（非IC机制）+ Gap B（异质估计器）同时开启；Stage 1（复现基线）+ Stage 3 内部的 single-winner 端点（=现成聚合）为它建立可归因的对照基线。

### 1.3 因变量

student在held-out / human-designed关卡上的zero-shot泛化性能（minimax-regret proxy）——而非auction welfare。在所有Stage中保持一致，是和ACCEL/PLR直接可比的唯一可靠轴。

---

## 2. 核心概念说明

### 2.1 PVL是什么（三类估计器之一）

PVL（Positive Value Loss）是ACCEL/PLR⊥的默认regret代理，定义为：

```
PVL = max(V_target − V_current, 0)
```

直觉：value function还有多少上升空间 ≈ agent在这个关卡还没学好 ≈ regret大。但SFL/No Regrets（Rutherford et al. 2024）的核心批判是：PVL与真实regret的相关性很弱，反而与成功率（solve rate）相关，是一个有系统性偏差的近似。

> ⚑ **V5**：PVL 是 Stage 3 三类异质估计器中的 **regret 类 teacher**（§4.3.1），与 Gen-SFL（learnability 类）、CENIE（覆盖率类）并列、各自出 bid 进 auction。Stage 1 复现基线（ACCEL/PLR⊥）用纯 PVL，是为了与已发表实现同源、拿可信的对标数字；PVL 的系统性偏差正是引入异质 Gen-SFL/CENIE 要修正的。（原文此处"Stage 2 同质 PVL 控制变量"叙事随 Stage 2 取消而删去。）

### 2.2 General-sum UED是什么

标准UED（ACCEL/PLR）被建模为两人零和博弈：teacher最大化student的regret，student最小化regret，双方利益严格对立（此消彼长）。零和性质让目标函数在teacher变量上强凹，这是NCC能证明收敛的数学基础。

**General-sum（一般和）：** teacher与student之间、以及teacher与teacher之间的收益之和不等于常数，利益关系任意。

你的多teacher系统天生是general-sum的，原因有二：

1. learnability（SFL/NCC）不是regret的零和对立，引入后破坏零和结构
2. N个teacher用不同估计器出价，teacher之间没有严格的零和关系

> ⚠ NCC原文明确："with score functions that are not zero-sum with the policy's negative return, our method lacks convergence guarantees"。这是NCC的保留，不是你方案的死刑——见下方 V4 修正。

> ⚑ **V4 修正（2026-06-19）——"general-sum 必崩"是旧认识，已被推翻；关键是区分"两层 general-sum"**（详证见 [proof_skeleton §8](proof_skeleton_entropy_regularized_heterogeneous_UED.md)）：
>
> - **目标层 general-sum（= 现状，免费）**：异质 score（learnability/coverage 非 `-J`）确实让博弈 non-zero-sum——但你的关键技巧是把非零和性**全部塞进 score 向量 `s`**，而 `s` 只通过对 `y` 线性的项 `yᵀs` 进入目标，强凹性由 `αH(y)` 在**单纯形 `y`** 上独立提供，与 `s` 异不异质**解耦**。所以 **general-sum 的目标可以保住 min-max 形式并收敛**（main.tex 第101行 keeps the min-max **form**；proof_skeleton §7.9 多项式率）。**现状已是 general-sum，收敛没崩。**
> - **决策结构层 general-sum（= 加"会训练的生成器 teacher"才出现，破定理）**：把 teacher 从"对固定 buffer 评分的函数"换成 PAIRED 式生成 level 的网络，对手决策变量从单纯形 `y` 变成非凸网络参数 `φ`，熵正则罩不住 `φ`，有限 buffer 的 `α log n`/对偶 1-光滑论证失效——**这才**退回 NCC 明说无解的 non-zero-sum open problem。
> - **一句话**：只要对手还在单纯形上动，general-sum 免费；一旦对手跳到网络参数上，才付出定理。所以"载体必须是 PLR（共享 buffer 评分）、不是 PAIRED（生成器）"背后的**真正理由是收敛性**（memory `auction-mechanism-is-plr-not-paired`）。

### 2.3 双层优化（Bilevel Optimization）

Hrithik指出这是一个bilevel优化问题——这是一个准确的理论判断。

**双层结构：** teacher（们）+ 机制层在外层决定课程分布，student在内层对给定课程做RL收敛到一个策略。外层teacher的"好不好"取决于内层student最终学成什么样。这是标准的bilevel结构。

min-max（零和UED）是bilevel的特例。bilevel 是"真正想优化"的 ground truth（外层泛化目标只通过内层 student 解依赖课程）。

> ⚑ **V4 修正（2026-06-19）**：早期"引入 general-sum 异质 teacher 后 min-max 框架就不适用"的判断**只对"决策结构层 general-sum"成立**（加生成器 teacher）。对**目标层 general-sum（现状：异质 score）**，本方案恰恰**保留 min-max 形式**并经熵正则获得收敛——min-max relaxation 是"如何拿到可收敛算法"，bilevel 是"真正优化什么"，二者并存不矛盾（见 §2.2 V4 修正 + main.tex 第87/95行）。

**推荐工具：** Two-Timescale Stochastic Algorithm Framework for Bilevel Optimization（Hong et al. 2023，arXiv:2007.05170）——NCC Future Work中明确引用的方向。

**实践含义：** 让student（内层）用快学习率，teacher/机制（外层）用慢学习率，内层相对外层近似准静态收敛。这与"teacher更新应比student慢"的直觉天然一致，工程上就是调学习率比例，不需要计算Hessian逆。

> ⚠ NCC对bilevel持保留态度，是因为他们的目标是证明收敛定理，而bilevel在nonconvex内层下给不出同等干净的保证。你的目标是刻画失效边界，bilevel对你是正确框架——这是两种不同立场，NCC的保留不适用于你的研究目标。

---

## 3. 实验环境与代码库

### 3.1 代码库

- **主库（自 Stage 1 起全程使用）：minimax**（Jiang et al., arXiv:2311.12716）
  - 模块化双课程设计，纯 JAX、可 jit、多设备，相比旧参考实现约 120× 加速
  - 原生支持多 teacher / 多 student 博弈（Stage 3 直接用，无需自己造轮子）
  - **关键：原生内置 `Maze-PerfectMazeXL`（101×101 完美迷宫）等全套 OOD 评估关卡**，正是对标 ACCEL 论文 8% 靶子的那个环境
  - 内置基线开箱即用且带调好的 maze 超参：DR、PAIRED、PLR、Robust-PLR（PLR⊥）、ACCEL、Population-PAIRED
- **JaxUED（Coward et al., arXiv:2403.13091）已弃用为主库。**
  > ⚠ 决策记录（2026-06）：曾用 JaxUED 跑过一轮 Stage 1，但发现 JaxUED 的 maze 是**随机撒墙的 domain randomization、并非完美迷宫算法**，且评估关卡只有 ≤13×13 的手画 prefab，**不含 PerfectMaze、更无 101×101**。因此无法直接对标 ACCEL 论文的 PerfectMaze 8% 协议（该协议是 100 个随机生成完美迷宫的平均解出率）。在 JaxUED 上自写完美迷宫生成器会引入自制偏差、且需自证正确，不如直接用 minimax 现成且经验证的实现。**故 Stage 1 起改用 minimax。** JaxUED 仅可作本地极小规模 sanity 的备选，不产出任何对标 SOTA 的数字。
- 内置基线（Stage 1直接调用）：ACCEL、Robust-PLR（PLR⊥）、PAIRED
- RL算法：PPO（所有Stage统一）
- 评估协议：α-CVaR鲁棒性评估（来自No Regrets/SFL）+ 标准zero-shot解出率/回报

### 3.2 环境层级

| 层级 | 环境 | 用途 | 单卡A100算力 | 说明 |
|------|------|------|-------------|------|
| **Sanity** | minimax AMaze（13×13 训练 → 评估含 `Maze-PerfectMazeXL` 101×101） | Stage 1/3：复现基线、验证管线、Σ 测量 | 实测：ACCEL~1h / PAIRED~4.1h / `multi_estimator_plr`~1h（PLR 血统、n_students=1）／~~Stage2 multi_teacher N=1~3.7h~~（PAIRED 血统，已弃用）（详见 [任务运行时间分析.md](任务运行时间分析.md)） | 已高度饱和；只用于sanity check，不用于证明方法价值。对标靶子=PerfectMazeXL 上的解出率 |
| **Sanity** | BipedalWalker-Hardcore | Stage 1/2：ACCEL未完全解决（≈75%）→有余量 | 数小时至数天/run | 保留为sanity；比MiniGrid更能体现种群增益 |
| **SOTA** | Craftax / Craftax-Classic | Stage 4：长程目标树；大余量；~1h/1B步 | <1天/run | 主推SOTA环境；Gen-SFL在此已验证（NCC论文） |
| **SOTA** | Kinetix | Stage 4：开放式2D物理；SFL已被采纳为课程引擎 | <1天/run | SFL在此的表现是其最强证据；teacher多样性效果更明显 |
| **可选** | XLand-MiniGrid | Stage 4扩展：多智能体（MAESTRO风格） | <1天/run | NCC在此的增益最显著；可选做 |

> ✓ Craftax和Kinetix是真正能推SOTA、体现teacher多样性价值的环境。MiniGrid只是sanity check的起点，不要在MiniGrid的结果上建立过多论点。

---

## 4. 实验阶段设计

### 4.0 实验配置总览

| 阶段 | 选择机制（Gap A） | Teacher / 估计器（Gap B） | 实验环境 | 目标 |
|------|-----------------|--------------------------|---------|------|
| **1. 复现基线** | ACCEL、PLR⊥、PAIRED（原版） | 单 teacher（纯 PVL） | MiniGrid、BipedalWalker | 与已发表数字差距 ≤5% |
| ~~**2. 同质种群**~~ | ~~（已取消，⚑V5）~~ | ~~teacher≡estimator，无"同质多 teacher"中间台阶~~ | — | ~~round-robin/EXP3/argmax 降为 Stage 3 single-winner 端点~~ |
| **3. 异质种群 + 机制** | VCG → tunable-λ non-IC + 收敛层（single-winner 端点=round-robin/EXP3/argmax，作消融基线） | 异质 N=3：PVL、Gen-SFL（NCC式11）、CENIE覆盖率 | MiniGrid、BipedalWalker | 刻画 curriculum质量 vs truthfulness 边界 |
| **4. 推进SOTA** | Stage 3最优配置 | Stage 3最优配置 | Craftax、Kinetix | 超越已发表的ACCEL/PLR |

> ⚑ **V5**：Stage 2 取消后，Stage 1（单 teacher 复现）之后**直接进 Stage 3**。runner = `multi_estimator_plr`（PLR 血统、无 teacher 生成器网络）。

---

### 4.1 Stage 1：当前SOTA复现（第1—2周）

目标：建立可信的自有基线数字，验证 minimax 管线正确。这是整个项目最重要的基础，后续所有"超过ACCEL"都必须和这组数字比，而非和论文数字比。

#### 4.1.1 实验内容

1. 在 minimax 中运行 ACCEL、PLR⊥、PAIRED 原版实现（`--train_runner` 选择，配置见 minimax `configs/maze/`）
2. 环境：minimax AMaze（13×13 训练 → `Maze-PerfectMazeXL` 101×101 评估）+ BipedalWalker-Hardcore
3. 因变量：held-out关卡上的zero-shot解出率/平均回报
4. 每方法至少10个随机种子；报告IQM和95% CI（rliable协议）

#### 4.1.2 推进门槛

- ACCEL/PLR⊥在 PerfectMaze 和 BipedalWalker 上复现数字与已发表数字差距 ≤5%
- **具体靶子（ACCEL 原论文 Parker-Holder et al. 2022, arXiv:2203.01302 原文数字，100 trials 平均）：**
  - **PerfectMaze 101×101（= minimax `Maze-PerfectMazeXL`，`--n_episodes=100`）：ACCEL ≈ 8%（empty-generator 版）/ 7%（DR-generator 版）；PLR⊥ ≈ 4%；DR ≈ 4%；PAIRED 基本解不出。**
  - PerfectMaze 51×51（区分度更大，可作更易看出高下的备选靶子）：ACCEL > 50%，PLR ≈ 25%。
  - BipedalWalkerHardcore ≈ 75% of optimal。
  > ⚠ 8% 是个很难、区分度很高的靶子，差几个百分点就是 SOTA 与否的区别——不是"差不多就行"的软目标。minimax accel.json 默认即 empty-generator 版（`maze_n_walls=0`，从空迷宫 mutate），与 8% 那一栏对齐。

> ⚠ Stage 1只做MiniGrid和BipedalWalker，不做Craftax/Kinetix（留到Stage 4起步时再复现，避免浪费算力）。

#### 4.1.3 这是"第一次复现"

注意：SOTA复现出现在两个时间点——Stage 1（MiniGrid/BipedalWalker）是为了验证管线；Stage 4起步时（Craftax/Kinetix）才是真正意义上的"当前SOTA复现"，因为你真正要超越的战场在那里。

---

### 4.2 ~~Stage 2：同质Teacher种群~~（已取消，⚑V5 2026-06-19）

> ⚑ **本阶段已取消，理由见 §0.4 / §1.2 ⚑V5。** 一句话：**teacher ≡ estimator，"多 teacher 却仍同质" 是矛盾的伪状态**，故不存在一个可隔离"多 teacher 本身有没有用"的独立阶段。Stage 1（单 teacher 复现）之后**直接进 Stage 3**（异质 estimator + auction），跑 `multi_estimator_plr`。

**原 Stage 2 各要素的去向（不是丢弃，是被 Stage 3 吸收）：**

- **三种聚合方式（round-robin / EXP3 / argmax）** → 它们是 auction 机制族的 **single-winner 退化端点**（`w=e_{j*}`，§4.3.4 ⚑V2），**作为 Stage 3 的消融对照基线**自动包含，不需独立阶段单跑。它们仍是"Stage 3 的 tunable non-IC 机制要打败的参照基线"——只是现在内生于 auction 族、而非外挂一个 Stage。
- **N 的 scaling（原 N∈{1,2,4,8} 扫描）** → 取消。Stage 3 直接锁 **N=3**（每类估计器一个，§4.3.2）+ 鲁棒性 N=6。N 不再是"堆 teacher 数量"的自由旋钮，而是"用几类异质 estimator"的结构选择。
- **增益归因消融（原"同质 PVL + α 熵正则"对照组）** → **仍然要做，但内生于 Stage 3**：Stage 3 的消融里，**single-winner 端点（argmax，等价于"只用一个 estimator"）＋同一 α** 就是"熵正则本身 vs 异质性"的归因对照——异质 N=3 配置须显著高于"single-winner + 同 α"才能把增益归给异质性。逻辑不变，只是挂到 Stage 3 的 single-winner 端点上。
- **MiniGrid pivot 判断** → 顺延为 Stage 3 的分支判断（见 §5.2.1）：若 Stage 3 在 MiniGrid 上 single-winner 与异质配置都看不出差距，是 MiniGrid 饱和、提前 pivot to Craftax，不在 MiniGrid 反复调参。

> ⚠ **为什么"多 teacher 零结果"叙事一并删除**：原 §4.2.2 的"同质 PVL + argmax ≈ ACCEL（零结果）"建立在"N 个 teacher 用同一 estimator、打分高度相关"上。该叙事**已不适用**——现在 teacher 一开始就是异质 estimator，不存在"同质 N>1"这个被测对象。"增益不来自堆 teacher 数量"这一点改由 Stage 3 的 single-winner 端点（=只用一个 estimator）作为下界自然体现：若异质 auction 打不过 single-winner 端点，就说明异质性/机制无净增益（触发 §6.4 蒸馏 plan B）。

---

### 4.3 Stage 3：异质种群 + Tunable Non-IC机制（第5—7周）

这是novelty所在。同时开启Gap B（异质估计器）和Gap A（tunable non-IC机制）。

#### 4.3.1 三类异质Teacher估计器

| Teacher类型 | 评分函数 | 适用环境 | 对应文献 |
|------------|---------|---------|---------|
| **Regret类** | PVL（Positive Value Loss） | 所有环境（MiniGrid起） | ACCEL/PLR⊥（Parker-Holder et al. 2022） |
| **Learnability类** | Gen-SFL：σλ·N(µλ\|µ,σ²)（NCC式11） | 连续回报（Craftax/Kinetix）；MiniGrid用原版p(1-p) | No Regrets（Rutherford et al. NeurIPS 2024）+ NCC（Monette et al. RLC 2025） |
| **覆盖率类** | CENIE GMM新颖性（负平均对数似然） | 所有环境；Stage 3第二批引入 | CENIE（Teoh et al. NeurIPS 2024） |

> ✓ 三类估计器回答的是本质不同的三个问题：Regret类问"这关有多难"，Learnability类问"学生现在学得动吗"，覆盖率类问"有没有带学生去没去过的地方"。三个问题互不替代，这是多样性的原理性来源，而非加噪声。

#### 4.3.2 Teacher数量建议

- Stage 3异质实验主要配置：**N=3**（每类估计器一个），是最干净的对应关系
- 鲁棒性验证：**N=6**（每类两个），检验within-type竞争的效果
- EGTA分析配置：**N=2–3**（teacher间博弈在小N下可解析，N≥8时empirical game基本不可分析）

> ⚠ 不要预先设定一个"让涌现发生"的魔法N值。（⚑V5：原"靠 Stage 2 scaling curve 选 N"已取消——N 不再是自由扫描的 teacher 数量，而是"用几类异质 estimator"的结构选择，主配置直接锁 N=3。）

> ⚑ **关键边界（2026-06-19）——这里的"N 个 teacher"全是评分器（estimator），不是会训练的生成器 agent**：N=3 指 PVL/SFL/CENIE 三个**对共享 buffer 评分的函数**，N=6 是每类两个。它们不生成关卡（关卡靠 DR 采样 + PLR 重放）。**收敛定理成立的前提就是"teacher=评分器、对手在单纯形 `y` 上"**（见 §2.2 V4 修正 + proof_skeleton §8）。
>
> **后续若主方案性能不足、想引入"会训练的生成器 teacher agent"（PAIRED 式）作 fallback**：
> - ❌ **直接对抗**（teacher↔student 真 `min_x max_φ`）会**破坏现有收敛定理**（对手变量变非凸 `φ`，熵正则罩不住），退回 NCC 无解的 non-zero-sum open problem，丢掉本篇最硬的理论卖点。
> - ✅ **唯一保住定理的路线 = 分层**：生成器 agent 只负责**生成候选 level 填进 buffer**（类 ACCEL edit/replay），estimator+auction 仍在**固定 buffer** 上做课程——定理罩"给定 buffer 的课程求解"层，agent 刷 buffer 当外部更新。代价：生成层本身无收敛保证（但 ACCEL/PLR 也没有，可接受）。
> - **决策建议**：当前 estimator-only 方案的 novelty **独立成立**（首个给 non-regret 异质 score 配收敛保证的 UED + non-IC auction + 无截断多项式率），不是"等加 agent 的半成品"。加 agent 是另一个 Gap，优先走"分层"。

#### 4.3.3 Gap A：VCG → Tunable-λ Non-IC机制

- **起点：** VCG拍卖（truthful锚点）——每个teacher提交（关卡, bid），VCG选胜者并计算支付
- **扫描：** 逐步增大IC-violation penalty λ，从λ=∞（严格truthful）到λ=0（自由机制）
- **因变量：** student在每个λ下的zero-shot泛化性能（不是auction welfare）
- **假设（per Babaioff-Sharma-Slivkins）：** 某个中间λ——而非truthful极端——最大化student性能，且环境越复杂增益越大

> ⚑ **V2 新增——α 轴（熵正则强度）与 λ 并列扫描**：α 是保证异质 estimator 收敛的正则旋钮（替代「伪装成 regret」的失败路径，见 [idea可行性分析.md](idea可行性分析.md) §4.5）。
> ⚑ **V4 更新（2026-06-17）**：指数 `e^{2B_s/α}` 已被证为记账失误、收紧为多项式 `K=O(1/(α³ε⁴))`（proof_skeleton §7.9）。以下三条据此改写。
> - **理论预测中等 α 最优（机制改变）**：α 太大 → 偏差 `α log n` 大；α 太小 → 优化代价 `K∝α⁻³` 多项式变大（**非指数崩溃**）。仍是「偏差 vs 优化代价」的权衡、中等 α 最优，但小 α 侧是**多项式变慢而非崩溃**。适用 §7.5.3(b) 的粗-细 active-search。
> - **claim**：固定每个 α 跑出「收敛到 `Φ_α`」，`Φ_α−Φ_0 ≤ α log n`（解析保证）。α→0 现多项式可达，但报告在固定适度 α（旋钮取舍）。
> - ⚠ **原"免费理论-经验对应崩溃实验"已作废**：理论不再预测 α 太小指数崩溃，故没有"理论阈值"可让实验去验证吻合。若仍要扫 α→小，只能作为**经验稳定性观察**（多项式率下预期是平滑变慢，非突变崩溃），不能宣称验证了理论崩溃边界。

#### 4.3.4 聚合方式：Selection vs Combination

机制层在技术上可以做两种事：

1. **Selection（选择）：** 从N个teacher提案里选一个关卡输出给student。VCG和tunable non-IC的经典形式。
2. **Combination（组合）：** 在分布层面聚合——每个teacher维护一个关卡优先级分布，机制调整各teacher分布的相对权重，从混合分布采样具体关卡。

注意：UED里teacher的输出是具体关卡（迷宫地图/BipedalWalker参数/Craftax种子），不能像知识蒸馏里的soft label那样直接平均。所以"组合"必须在分布层面做，而非在关卡层面做。

> ✓（⚑V5）Selection（single-winner 端点：round-robin/argmax/exp3）是机制族的退化起点；Stage 3 扩展到分布层面的 Combination（fractional w，类似MAESTRO的shared buffer机制），single-winner 自动作为消融对照。

> ⚑ **V2——CENIE 的 score 层重构（绕开本节的「关卡不能平均」墙）**：原始 CENIE 在采样概率层凸组合 `P_replay=αP_N+(1-α)P_R`，不在 score 的线性项里，无法纳入熵正则收敛框架。**重构**：直接令 coverage score `s_i = -log p(x_i|λ_Γ)`（截断见 §4.3.6 ⚑V2），代入熵正则内层得解析解 `y_i* ∝ p(x_i)^{-1/α}`（反密度幂次采样，低覆盖→指数级高采样）——这正是 CENIE「优先低覆盖」思想的最大熵形式化，且在 `yᵀs` 线性项里、满足收敛框架。**论文措辞须为「CENIE 思想的规范化升级」，非「与原公式数值等价」。**

> ⚑ **V2 新增——Multi-winner 机制：auction 同时采用多个 teacher 的输出（main.tex §3.1 草稿源，2026-06-17）**
>
> **动机**：现有三种聚合（`round_robin/argmax/exp3`，见 `multi_teacher_runner.py`）都是 **single-winner selection**——每步只选一个 teacher 的关卡、`_mask_teacher_update` 只更新被选中 teacher。这是机制族最退化的端点；§4.2.2 预期「N>1 同质 argmax ≈ ACCEL（近乎零增益）」很可能正源于此：从近乎相同的分数里挑一个 ≈ 没聚合。**Multi-winner 是绕开该零增益陷阱的方向，且不碰本节「关卡不能平均」的墙（混合的是采样概率/训练预算，非关卡内容；中标关卡始终是 buffer 里完整合法的整张关卡）。**
>
> 设 N 个 teacher 各出 bid 向量 `s^(j) ∈ ℝⁿ`（对候选 buffer `Θ_n={θ_1,…,θ_n}` 的打分），机制给中标权重 `w ∈ Δ_N`。三个递进方案：
>
> 1. **Multi-unit auction（top-k 多中标）**：一次选 top-k 个 teacher 的关卡组成本 batch 训练集。VCG **原生支持多中标**（标准形式即分配多物品并各算支付），故从 truthful 锚点扩到 multi-winner **不破坏 §4.3.3 的 VCG 理论锚**。落地：student 本就按 mini-batch 训练，一个 batch 塞 k 张不同 teacher 中标的完整关卡。
>
> 2. **分布层混合（Combination，本节方案 2 的形式化）**：`y_mix = Σ_j w_j · y^(j)`，每个 teacher 按 `w_j` 贡献采样概率质量、无人被归零（区别于 argmax）。这正是**熵正则内层解析解**：令混合 score `s̄ = Σ_j w_j s^(j)`，则
>    ```
>    y*(x) = softmax( (1/α) · Σ_j w_j · s^(j)(x) )
>    ```
>    —— 在 `yᵀs` 线性项里、满足 main.tex 定理 1 的 A5 假设（收敛结构由 αH(y) 独立提供，见 §0.5 修正 1）。**「同时采用多 teacher」= 多个 bid 融进一个混合 score。**
>
> 3. **Fractional allocation（分数中标，auction 理论里「同时中标」的标准解）**：把接下来 T 步的**训练预算**按比例 `w_j` 分给各 teacher（divisible-good allocation）。`argmax` 是其 `w = one-hot` 的退化特例 ⇒ **现有基线自动成为新机制族的 single-winner 端点，消融对照天然成立。**
>
> **与旋钮的接口**：multi-winner 把 Gap A 的 tunable-λ 旋钮空间从「选哪个 teacher」扩展到「如何在 teacher 间分配训练预算 `w`」——本质更丰富的机制设计空间，是 Gap A 刷 SOTA 的主要杠杆之一。**α（熵正则）与 λ（IC-violation）之外，`w` 的分配规则（top-k / 软混合 / 学习得到）构成第三类机制变体，须在 Stage 3 消融。**
>
> **关系一览**：single-winner（argmax/exp3，已实现）⊂ top-k multi-unit ⊂ fractional（全 teacher 带比例）；分布混合是 fractional 在采样分布层的实现。**写入 main.tex 时作为 §3.1 机制层 combination 变体的正式数学定义，argmax 列为 single-winner 特例基线。**

#### 4.3.5 收敛层

- **CURATE-style directedness正则（L_dist）：** 防止课程偏离start→target难度方向
- **KL bound（REPS-style）：** 防止课程分布一步跳太远
- **VCG-style置信区间：** 处理teacher出价估计的不确定性

> ⚠ directedness和KL是稳定化组件，不是让方法超过ACCEL的来源。它们的作用是"防止异质teacher把课程搅成震荡的"，不能被宣称为SOTA增益的来源。

#### 4.3.6 CENIE的引入时机

建议Stage 3分两批：

1. **第一批（Stage 3a）：** PVL + Gen-SFL两类teacher + tunable non-IC机制，跑通核心novelty
2. **第二批（Stage 3b）：** 加入CENIE覆盖率teacher（第三类），配合消融实验

加入CENIE时必须配置一个消融：去掉CENIE teacher，看性能是否下降。如果不下降，说明在该环境里覆盖率和learnability高度重合，CENIE冗余——这本身也是干净的发现。

> ⚑ **V2——CENIE 截断的「不可约偏差底板」必须如实报告**：为保证 coverage score `s_i=-log p` 有界（收敛框架的 A2 假设），须截断 `s_i = clip(-log p, -C, C)`。代价：`p < e^{-C}` 的**极罕见 level（最高 Novelty）被压平为同一采样概率**，反密度采样在最长尾失真为均匀。该截断偏差 `E_clip(C)` **只依赖 C、与 α 无关**——即它**不随调小 α 而消失**，是一个**不可约的固定偏差底板**（区别于可随 α 调节的 `α log n` 平滑偏差）。
> - **论文须声明**：coverage teacher 的总偏差 = 可调平滑偏差 `α log n` + 由 C 决定的安全底板 `E_clip(C)`；截断牺牲了对发生概率 `< e^{-C}` 的极罕见环境的区分度，换取最坏情况稳定性（防策略崩到病态环境）。
> - **不可强行 `C→∞` 消偏差**：那会无限放大截断尾部失真、且 `B_s=C` 影响率常数（V4 下是多项式而非指数，但仍是常数）。**C 是稳定性与覆盖保真度之间的权衡，不是自由参数。**

> ⚑ **V4 改写——CENIE 引入前的前置 checklist**：⚠ 原 V2 版基于 `α_min∝2B_s`（指数式，B_s 顶大即区间空、加 GPU 无用因对数依赖）——**已失效**。V4 多项式率下 `α_min=Θ((1/(K_budget·ε⁴))^{1/3})`，**与 B_s 无关、对 K_budget 多项式衰减**（加预算 K^(1/3) 地降 α_min）。新结论：
> 1. **B_s/C 不再卡 α_min**：可行区间唯一的空区间情形是 **`E_clip(C) ≥ ε_bias`**（截断底板本身超过偏差容忍，即 `α_max≤0`）。先算 `E_clip(C)` 与 `ε_bias` 比较即可。
> 2. **若 `E_clip(C) ≥ ε_bias`**：调小 C 抬高 `α_max`（代价 = 尾部失真，本就要报告），直到 `E_clip(C) < ε_bias`。这是**偏差侧**约束，与算力无关。
> 3. **预算侧 α_min 现在可加 GPU 缓解**（V4 新增可能）：因 `α_min∝K_budget^{-1/3}`，加预算确实多项式地降 α_min、打开区间——不再像 V2 那样"加 GPU 没用"。
> - **风险分布（V4 大幅降低）**：可行域基本总非空，唯一退化情形是 coverage 的 C 设得太激进使 `E_clip(C)≥ε_bias`。Stage 3a（PVL+learnability，无截断，`E_clip=0`）**永不为空**。用 [alpha_range_solver.py](alpha_range_solver.py) 一键核。

---

### 4.4 Stage 4：推进SOTA（第8周起）

目标：把Stage 3最优配置搬到Craftax + Kinetix，实现"超越已发表的ACCEL/PLR"。

#### 4.4.1 第二次SOTA复现（Stage 4起步时）

在正式对比之前，先在Craftax和Kinetix上复现已发表的ACCEL、PLR（MaxMC/PVL）、SFL、NCC-Learn的数字。这是"当前SOTA复现"的第二次、也是真正关键的一次。

- **Craftax：** 复现PLR-MaxMC（Matthews et al. 2024的最强UED基线）和NCC-Learn
- **Kinetix：** 复现SFL（已被Kinetix论文采用为课程引擎）

> ⚠ 不能只和MiniGrid上复现的数字比——Craftax/Kinetix上的ACCEL/PLR可能因实现细节和你Stage 1的数字有差异，必须自己跑一遍。

#### 4.4.2 完整基线集

在顶会投稿中，"当前SOTA"基线集必须包括（不能只比ACCEL）：

- ACCEL（Parker-Holder et al., ICML 2022）
- PLR⊥（Robust PLR）
- SFL（Rutherford et al., NeurIPS 2024）
- NCC-Learn（Monette et al., RLC 2025）——目前在XLand/Craftax上最强的learnability方法
- CENIE（Teoh et al., NeurIPS 2024）——如果你的系统用了CENIE teacher
- ADD / TRACED——如果审稿人会问（当前MiniGrid/BipedalWalker上报数最漂亮的方法）

> ✓ 主打"仅用ACCEL就够"是2026年的一个风险：已有多篇2024-2025论文在特定基准上超过ACCEL。你需要和当前活跃挑战者都比。

#### 4.4.3 评估指标

- Mean solve rate / mean return on held-out evaluation set（主指标）
- α-CVaR鲁棒性评估（来自No Regrets，Rutherford et al. 2024）：α%最坏情况关卡上的性能
- 每方法10个种子；IQM + 95% CI（rliable）

---

## 5. 预期结果与分支判断

### 5.1 各配置预期性能对照表

> ⚑ **V5**：原表含两行"N>1 同质 PVL"配置已删（Stage 2 取消，§4.2）。single-winner 端点（argmax，等价"只用一个 estimator"）取代它们作为下界对照。

| 配置 | 相对ACCEL预期性能 | 诊断含义 | 若出现意外结果 |
|------|-----------------|---------|--------------|
| 单 estimator（PVL），argmax＝single-winner 端点 | ≈ ACCEL（sanity通过） | 框架实现正确；退化为 ACCEL/PLR⊥ | 若差距>5%：检查 minimax 配置和超参 |
| 单 estimator，EXP3/UCB bandit | 可能微弱提升 | 增益来自聚合策略的探索性，非异质性 | 若完全无提升：印证MiniGrid太简单，触发pivot to Craftax |
| N=3，异质（PVL+Gen-SFL+CENIE），single-winner（argmax/exp3） | ≥ 单 estimator 端点 | 异质性的净贡献（机制仍退化） | 若 ≈ 单 estimator：异质性在该环境无增益，查 Σ 冗余（T3/T4） |
| N=3，异质，tunable non-IC（fractional w） | 预期明显超过ACCEL（Stage 3目标） | 异质估计器 + non-IC 机制层的净贡献 | 若无提升：消融定位——是异质性无效还是机制无效 |

### 5.2 关键分支判断

#### 5.2.1 MiniGrid Pivot

**触发条件：**（⚑V5 改）Stage 3 在 MiniGrid 上，**异质 auction 配置相对 single-winner 端点（=单 estimator）完全看不出增益**。

**动作：** 不要在MiniGrid上继续调，直接提前进入Craftax实验。MiniGrid已高度饱和（所有顶会方法均如此承认），它看不出差异不代表方法失败。

#### 5.2.2 Teacher折叠问题

**风险：** 所有teacher收敛到相同策略（PAIRED已记录的failure mode，Mediratta et al. CoLLAs 2023）。

**动作：** 加入diversity正则（DIPLR-style，Li et al. IJCAI 2023）或PSRO-style fictitious-play循环。

#### 5.2.3 Truthful机制主导

如果Stage 3发现VCG（λ=∞ truthful）在所有λ中都最优：

**解读：** 这本身也是可发表的结果——它支持了Hrithik的观点（truthful=最优），但需要检验是否是因为PVL噪声底层太低（Babaioff-Sharma-Slivkins的预测是"在有噪声的bandit估计下非truthful更好"）。Either way是有信息量的发现。

---

## 6. 理论框架定位

### 6.1 你的方案 vs SFL的关系（重要定位）

SFL应该被定位为"你框架里的一个特例/一种估计器选择"，而非"要被你系统整体击败的竞争方法"。

当N=1、估计器=learnability、机制=truthful时，你的框架大致退化成SFL那一类（单teacher + 直接learnability评分 + 无strategic aggregation）。

这个"SFL是我的特例"的定位有两个好处：

1. 清楚地告诉审稿人你和最强挑战者的关系
2. 让"我比SFL多了什么"变成可消融的实验——退化到单teacher learnability = 复现SFL，全系统 = 你的方法，性能差 = 机制层和异质性的净贡献

### 6.2 你的方案 vs NCC的关系

**NCC的Gen-SFL（式11：σλ·N(µλ|µ,σ²)）** 直接作为你的learnability类teacher的估计器函数，在连续回报环境（Craftax/Kinetix）里使用。引用：Monette et al. RLC 2025。

**NCC的双时间尺度SGDA** 作为你系统训练时的学习率分离方案。

> ⚑ **V2 勘误（地基甄别）**：原文「Lin et al. 2020，引用 Hong et al. 2023 bilevel 版本」**混淆了两篇不同的 two-timescale**：
> - **Lin-Jin-Jordan 2020（arXiv:1906.00331，nonconvex-concave minimax GDA）**：NCC 的收敛证明实际引的是这个；也是本方案熵正则收敛框架（路线 B）的地基。
> - **Hong et al. 2020（arXiv:2007.05170，bilevel TTSA，内层 strongly-convex）**：原文当作「正确框架」，但 **V2 判定：bilevel 不是出路**——其内层 strongly-convex 前提在 nonconvex 的 student RL 内层不成立，比修 minimax 更难。**本方案留在 minimax 框架内用熵正则恢复收敛，不改投 bilevel。**

**NCC的核心保证**（最优迭代收敛到ε近似一阶Nash均衡）对你的general-sum多teacher系统不直接成立——NCC自己也承认learnability破坏了零和条件。**V2 解法**：用熵正则 `αH(y)` 独立提供 strong-concavity，**重新获得收敛保证**（收敛到 α-正则化目标 `Φ_α`，路线 B），而非停留于「刻画崩溃」。

### 6.3 EGTA分析目标（⚑V2 已降级为支撑材料）

> ⚑ **V2 重大降级（V4 微调理由 3）**：原文把「刻画零和→general-sum 收敛崩溃边界」当 EGTA 主轴。但 (1) 用现成 EGTA 描述已知现象是**附属分析、不够当顶会主 novelty**；(2) 靠满种子实验扫连续轴刻画边界**成本不可承受**；(3) ⚠ **（V4 改）原"崩溃边界由理论 `e^{2B_s/α}` 给公式"已失效**——指数收紧为多项式后理论不再预测崩溃边界；但这**反而更支持降级 EGTA**：既无理论崩溃边界、又不值得用实验去刻画一个多项式平滑变化，EGTA 刻画崩溃边界的动机进一步消失。
>
> **新定位（不变）**：主菜是 **constructive 命题——「多项式时间收敛到 `Φ_α`」+ α 作为解析旋钮**（[idea可行性分析.md](idea可行性分析.md) §4.5）。EGTA 退为**支撑材料**：仅在 N=2–3 小配置做定性 payoff 分析（如「引入 learnability teacher 后 Nash 从纯策略变混合」），佐证理论，**不追求精确边界、不靠连续轴满种子实验**。

Empirical Game-Theoretic Analysis（EGTA）在 N=2–3 小配置下做 empirical payoff matrix 的定性分析（大 N 无法解析），作为收敛性理论结论的实证佐证。

---

### 6.4 备手 idea：知识蒸馏（仅在主方案失败/刷不出 SOTA 时启用）

> 定位：这是 **plan B**，不是主线。主线是「non-IC 机制（Gap A）+ 异质估计器 agent（Gap B）」的 auction-UED。本节给出"当主线失效时改用蒸馏"的思路、触发条件、与 auction 的关系，以及 novelty 依据。

#### 6.4.1 触发条件（写死，避免与主线抢职责）

仅在以下任一情形启用蒸馏，否则**不开**：

1. **机制层无净增益**：Stage 3 的 λ-frontier 显示 VCG 各 λ 都打平（即 Gap A 的 non-IC 机制没带来超过 truthful 锚点的提升）；
2. **异质性无净增益**：消融显示异质估计器组合不优于 single-winner 端点（=单 estimator，⚑V5 原"同质 PVL"）（即 Gap B 也没立住）；
3. **刷不出 SOTA**：Stage 4 在 Craftax/Kinetix 上 auction 配置无法超过已发表 ACCEL/PLR/SFL/NCC。

> ⚠ 若主线 work，**不要引入蒸馏**——它和 auction 在"信息聚合"职责上部分重叠，同时开会互相干扰、且稀释 novelty 叙事。

#### 6.4.2 蒸馏在本框架的两个接入点（都作用在 policy 侧，绕开"关卡不能平均"）

文档 §4.3.4 已立的墙——"teacher 输出是具体关卡，不能像 soft label 那样平均"——只挡住**关卡层蒸馏**，没挡住 policy 层：

- **接入点 A（student 端，加速 bilevel 内层）**：把 N 个 teacher 各自训出的 student 策略（或 value head）蒸馏/正则化进一个主 student，缩短内层 RL 收敛（直接缓解 §7.5.2 的 Stage 3 耗时痛点）。
- **接入点 B（teacher 端，省评估开销）**：把异质 teacher 的 scoring function（PVL/learnability/coverage）蒸成一个统一 surrogate scorer，廉价预筛关卡、只把高分关卡送进贵的真评估（省 §4.3.3 argmax 的 N 倍 rollout）。

> ⚠ **真实风险**：蒸馏改变 student 的 policy 分布 → 可能污染 UED 赖以选关卡的 **regret 信号**。引入前必须验证 regret 估计不被破坏（建议：蒸馏只在 student 更新的辅助 loss 项里加权，且与 regret 计算用的 rollout 解耦）。

#### 6.4.3 关键定位：蒸馏是「知识聚合」而非「多智能体协作」

- **协作**的核心是 agent 间为共同目标**双向适配、动态响应**对方行为。标准蒸馏里 teacher 冻结、单向、student 单向模仿 → **不是协作，是知识转移/聚合**。（例外：co-distillation / mutual learning 多网络互为 teacher 同时训练，才算弱协作。）
- 在本框架里，**多智能体协作发生在 auction/聚合环节**（N 个 teacher 出价、被机制协调成课程）；**蒸馏是其下游的转移步骤**，把 auction 产出的知识压进 student。
- → **论文措辞**：蒸馏应表述为"把多智能体协作（auction）产出的知识转移进 student 的下游环节"，**不可**表述为"另一种多智能体协作方式"（审稿人会反驳蒸馏无双向博弈）。**auction = 协作机制，蒸馏 = 转移机制，职责不同、不可互替。**

#### 6.4.4 novelty 依据（两轮文献核查，2026-06）

- **第一轮**（deep-research）：PAIRED/PLR/PLR⊥/ACCEL/MAESTRO/TRACED + Distral/ensemble/KnowRU/AgentArk，无一在 UED 里用多 teacher 蒸馏作聚合；AgentArk(arXiv:2602.03955) 是 LLM 多 agent debate 轨迹蒸馏，领域/机制都不撞。
- **第二轮**（穷尽式核查）：18 个概念交集入口全扫（跨 arXiv/OpenReview/Semantic Scholar/Google Scholar/NeurIPS·ICML·ICLR·AAMAS proceedings）+ 4 红队角度主动反驳 + 55 候选深挖全文 → **确认撞车 0、红队反例 0**；"auction×课程/UED" 入口本身直接空白。55 候选全因三类出局：①模型压缩多 teacher 蒸馏（术语陷阱，teacher=预训练 NN）②RL ensemble/policy 蒸馏（作迁移/鲁棒性，非课程聚合，环境固定无 UED）③真 UED 但零蒸馏（ACCEL/DivSP/DIPLR/POET 聚合 regret+diversity 标量）。
- **逻辑**：窄命题（UED×多 teacher 蒸馏作聚合）空白 ⇒ 中（+auction）、宽（+non-IC）必空白。
- ⚠ **措辞约束**：可写 *"to the best of our knowledge, no prior work uses multi-teacher distillation as the curriculum aggregation mechanism in UED"*；**不可写"首创/绝无仅有"**——缺证据 ≠ 否证，且空白可能含 negative-result 不发表偏差。

---

## 7. 时间线草图

| 周次 | 阶段 | 主要工作 | 门槛/产出 |
|------|------|---------|---------|
| **第1–2周** | Stage 1 | minimax 上复现 ACCEL、PLR⊥、PAIRED，PerfectMazeXL/BipedalWalker（第一次SOTA复现） | 与已发表数字差距≤5%；minimax 管线验证 |
| ~~第3–4周~~ | ~~Stage 2~~ | ~~（已取消，⚑V5：teacher≡estimator 无同质中间台阶）~~ — single-winner 端点内生于 Stage 3 | — |
| **第3–6周** | Stage 3a | PVL+Gen-SFL两类teacher + VCG→tunable-λ机制（含 single-winner 端点消融），扫λ；Σ 冗余测量（T3/T4）定配置 | curriculum质量 vs truthfulness frontier |
| **第7周** | Stage 3b | 加入CENIE覆盖率teacher；N=3/N=6消融；EGTA分析（N=2–3） | 三类估计器各自净贡献消融 |
| **第8周+** | Stage 4 | Craftax/Kinetix上复现ACCEL/PLR/SFL/NCC（第二次SOTA复现），再用Stage 3最优配置推SOTA | 超越已发表ACCEL/PLR；α-CVaR鲁棒性 |

> ⚠ 与Amy的会面建议放在 Stage 3a 有初步结果之后（⚑V5 原"Stage 2 后"）——此时既有 Stage 1 sanity 数字 + single-winner 端点对照，又有"为什么需要异质性和机制"的实证依据，讨论会更有针对性。

---

## 7.5 实验算力预算与省时间推断策略

> 本节回答两个工程问题：**(A) 缩减种子数是否还能看出方法趋势；(B) Stage 3 相对 Stage 1 基线（ACCEL/PLR⊥）多出哪些开销、以及如何用"有原则的推断"代替"大规模试错"。**（⚑V5：原文以已取消的 Stage 2 为对照基准，改为以 Stage 1 复现基线 + Stage 3 内部 single-winner 端点为基准。）
> 核心立场：Stage 3 不是盲扫参数，而是**带理论预测的验证**——绝大多数维度可以靠已有文献/Stage 1 基线推断出来，不需要试错。

### 7.5.1 种子策略：3 seed 只能"筛除"，不能"确认"

UED 方差极大（本项目 Stage 1 已记录 ACCEL 有崩溃 seed，min solved_rate=0.01）。3 seed 的 95% CI 宽到几乎任何两个 setting 都重叠，**无法支撑"A 显著优于 B"**。但可以这样用：

- **3 seed 只做排除，不做确认**：明显垮的 setting（如 N=8 argmax 直接崩）可放心砍；几个 setting 缠在一起时**不要据此宣布"持平"**，标记为"候选，待加种子"。
- **两阶段种子策略（省时间的正解，而非单纯砍种子）**：
  1. 第一轮所有 setting 跑 **3 seed 粗筛** → 砍掉明显坏的、锁定 top-2~3 候选；
  2. 只对候选**补到 8~10 seed** 出正式数字（IQM + 95% CI，rliable）。
  总算力远小于"所有 setting 都 10 seed"，又保住发表所需的统计强度。
- **⚠ "持平"结论需要窄 CI**：Stage 3 的核心论断（异质 auction vs single-winner 端点是否真有增益，§5.1）恰恰最需要补种子——3 seed 下"看不出差异"与"真的没差异"无法区分。**最想下结论的那个 setting，必须补满种子。**

### 7.5.2 Stage 3 比 Stage 1 基线（ACCEL/PLR⊥ ~1h/run）多出的开销（明确原因，非估计）

1. **多一条连续扫描轴 λ**（VCG truthful → tunable non-IC，§4.3.3）：在 {3 聚合/w 配置} 之上叠加 λ frontier，每个 λ 点都要重训。
2. **异质估计器评估更重**：Gen-SFL 算 learnability 分布、CENIE 维护 GMM 算 novelty，都比纯 PVL 一个 relu 贵。
3. **argmax / VCG 选择层的 N 倍学生 rollout 开销**（§4.3.3）。

> ⚑V5 实测校准：`multi_estimator_plr`（PLR 血统、`n_students=1`、无 teacher 生成器）单 run ≈ ACCEL 的 ~1h 量级（见 [任务运行时间分析.md](任务运行时间分析.md)），**不是** PAIRED 的 ~4h——异质 estimator + CENIE GMM 的额外开销有限，主开销来自 λ 轴的配置数。

→ Stage 3 单配置墙钟 ≈ ACCEL 量级，但**配置数因 λ 轴更多**。这正是"试错不可承受"之处。

### 7.5.3 三条"比试错更好"的推断（每条都来自已有理论/Stage 1 基线，不需新实验去试）

- **(a) N 直接锁 N=3，不扫**（⚑V5 原"靠 Stage 2 scaling curve 推断"）：teacher≡estimator，N=3=三类估计器各一个（§4.3.2），是结构选择而非自由旋钮。**本就没有一整个 N 扫描维度。** 测哪几类 estimator 冗余靠 Σ（T3/T4），不靠扫 N。
- **(b) λ 用粗-细两段 active-search，不用均匀网格**：理论（Babaioff-Sharma-Slivkins，§4.3.3）预测最优 λ 在中间、不在 truthful 极端。先跑 3 锚点 {λ=∞ 严格truthful, λ=中, λ=0 自由}：
  - 若单峰（理论预测）→ 在峰附近细扫；
  - 若单调（VCG 最优）→ 这本身是可发表结果（§5.2.3），直接停。
  **这是 active-search，不是网格试错。**
- **(c) 廉价代理预筛，再上贵环境**：Stage 3 的机制/估计器组合先在 MiniGrid（数小时/run）筛"哪个明显垮"，**只把胜出组合搬到贵的 Craftax/Kinetix**。
  ⚠ MiniGrid 已饱和，只能筛"明显坏的"、不能筛"最好的"（与 §7.5.1 同逻辑）。

> 三条合起来：Stage 3 的 **N 直接锁 N=3（本无扫描维度）、λ 靠单峰假设做 active-search（省网格）、贵环境只跑预筛胜出者（省算力）**。全部来自文档已有理论，不需要先做实验去试。

### 7.5.4 ⚑V2 新增——α 轴（熵正则强度）的解析可行区间，把新维度的成本压到最小

> 路线 B 引入了 α 这条新扫描轴（§4.3.3 ⚑V2）。本节给出**不需任何 run、纯解析**的可行区间，把 α 的搜索从「未知范围的网格」压到「一个窄区间内的 2–3 点」。材料全部来自已有收敛性公式（[proof_skeleton](proof_skeleton_entropy_regularized_heterogeneous_UED.md) §7.5c），无需新推导。

> ⚑ **V4 改写（2026-06-17）**：指数 `e^{2B_s/α}` 已收紧为多项式（proof_skeleton §7.9），故 (2) 的 α_min 公式从指数隐式改为**多项式闭式**，且不再依赖 B_s。完整可执行实现见 [alpha_range_solver.py](alpha_range_solver.py)。

**(1) 上界 `α_max`（偏差闸门，闭式）**：由定理 2，`Φ_α − Φ_0 ≤ α log n`（coverage 还要加截断底板 `E_clip(C)`）。设可容忍偏差 `ε_bias`，则
```
α_max = (ε_bias − E_clip(C)) / log n          （PVL/learnability 取 E_clip=0）
```
> 超过 `α_max`：`Φ_α` 离 `Φ_0` 太远，增益无法干净归给「异质性」。

**(2) 下界 `α_min`（可优化性闸门，V4 多项式闭式）**：由 V4 率 `K=O(1/(α³ε⁴))`（大 batch）。令 `K ≤ K_budget`，解出
```
α_min = Θ( (1/(K_budget · ε⁴))^{1/3} )         （闭式，无需求根；单样本率则 ε⁴→ε⁵）
```
> ⚠ **与旧版关键差异**：α_min **不再依赖 B_s**（旧版 `∝2B_s` 已作废），且对 K_budget 是**多项式衰减**（`∝K_budget^{-1/3}`，旧版是对数）——所以**加算力确实能打开区间**。低于 α_min 只是预算内迭代不够，**不是「α 太小 UED 崩」**（理论已不预测指数崩溃）。

**(3) 可行区间 = `[α_min, α_max]`，两端闭式、零实验。**

> ⚠ **V4 下区间几乎总非空**：唯一空区间情形是 **`E_clip(C) ≥ ε_bias`**（coverage 的 C 太激进，使 `α_max≤0`）——这是偏差侧问题，调小 C 即解。旧版"算力-保真度权衡紧到区间空"的 negative insight 在 V4 下大幅弱化（多项式率下加预算即降 α_min）。Stage 3a（无截断，E_clip=0）**永不为空**。

**(4) 区间内只跑 2–3 点 active-search（同 §7.5.3(b) 的 λ 逻辑）**：
- 锚点：`α_min`、几何中点 `√(α_min·α_max)`、`α_max`；
- 理论预测泛化对 α **单峰**（太小：优化代价 K∝α⁻³ 大、预算内欠优化；太大：偏差 α log n 大；中间好）→ 单峰则峰附近细化，单调则取端点即停。⚠ V4：小 α 侧是**多项式欠优化**而非崩溃；
- **泛化最优 `α*` 解析推不出**（依赖真实任务分布的 nonconvex 泛化曲线），只能实验定——但已被 (3) 夹在窄区间内，3 点足够。

**(5) α 与 λ 用坐标式扫描、不做二维网格（待可分性验证）**：α 管「优化稳定性」、λ（§4.3.3）管「机制探索性」，作用层不同。先固定 λ 定 α、再固定 α 定 λ，把 `|α|×|λ|` 降到 `|α|+|λ|`。
> ⚠ 前提：α 与 λ 的最优近似可分。**在 (4) 的三锚点阶段顺便查**：若发现强耦合（α 最优随 λ 显著漂移），退回局部小网格。

> **合起来**：α 这条新轴的净成本 = 解析夹窄（0 run）+ 区间内 3 点 + 坐标式（而非 ×λ 网格）。**实验量增加有限可控，不翻倍。**

---

## 8. 关键设计约束

- **种子分两阶段：所有 setting 先 3 seed 粗筛，仅 top 候选补到 10 seed；"持平"类结论必须补满种子（§7.5.1）**
- **Stage 3 的 N 直接锁 N=3（teacher≡estimator，本无 N 扫描维度，⚑V5）、λ 用粗-细 active-search、贵环境只跑预筛胜出者（§7.5.3）——不做均匀网格试错**
- **⚑V2 α 轴先解析夹出可行区间 `[α_min, α_max]`（零实验），再区间内 3 点 active-search，与 λ 坐标式扫描不做二维网格（§7.5.4）；若区间为空（`α_max<α_min`）当结论报告，非 bug**

- Teacher多样性必须来自结构上本质不同的估计器，而非给同一个regret信号加噪声
- Non-IC探索必须来自机制结构（拍卖均衡动态），而非手工加bid噪声
- 因变量始终是student的zero-shot泛化，而非auction welfare
- MiniGrid只是sanity testbed，不是用来证明方法价值的地方
- CENIE是第二批加入的第三维度，Stage 3a先不开，配合消融再开
- EGTA分析限制在N=2–3的小配置，否则empirical game不可解析
- Stage 4基线集必须包含SFL和NCC-Learn，不能只比2022年的ACCEL

---

## 9. 关键参考文献

| 方法 | 来源 | 在本方案中的角色 |
|------|------|----------------|
| ACCEL | Parker-Holder et al., ICML 2022, arXiv:2203.01302 | Stage 1/2/4主基线；PVL估计器来源 |
| SFL / No Regrets | Rutherford et al., NeurIPS 2024, arXiv:2408.15099 | learnability class估计器理论来源；Stage 3–4基线之一 |
| NCC | Monette et al., RLC 2025, arXiv:2505.20659 | Gen-SFL式11（连续回报learnability）；bilevel future work引用；Stage 4基线之一 |
| CENIE | Teoh et al., NeurIPS 2024 Oral, arXiv:2502.05726 | 覆盖率class估计器（Stage 3b引入）；Stage 4基线 |
| MAESTRO | Samvelyan et al., ICLR 2023, arXiv:2303.03376 | 多teacher population设计参考；shared buffer机制参考 |
| ReMiDi | Beukman et al., ICML 2024, arXiv:2402.12284 | regret stagnation问题诊断；EGTA分析相关 |
| Bilevel Opt. | Hong et al. 2023, arXiv:2007.05170 | general-sum UED的双层优化框架；两时间尺度学习率分离 |
| Lin et al. 2020 | ICML 2020（nonconvex-concave SGDA） | NCC收敛定理的数学基础；双时间尺度实践依据 |
| Babaioff et al. | EC 2009, arXiv:0812.2291 | "truthful机制在bandit学习系统中有更高regret"——Gap A的理论依据 |
| minimax库 | Jiang et al., arXiv:2311.12716 | **主代码库（Stage 1 起全程）**；原生 PerfectMazeXL 评估 + ACCEL/PLR/PAIRED + 多 teacher |
| JaxUED | Coward et al., arXiv:2403.13091 | ~~备选代码库~~ **已弃用**：maze 为随机撒墙非完美迷宫、无 PerfectMaze/101×101，无法对标 ACCEL 协议（详见 §3.1） |

---

*文档生成时间：2026年6月  |  基于本次对话全部讨论内容*
