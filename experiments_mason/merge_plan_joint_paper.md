# 合并方案：小模型线（Mason）如何并入组内联合论文（Alec 线）

> 起草 2026-07-08。依据 = main 分支 Alec 线文档实读（experiment_design.md / 方法设计_v2.md / 异质teachers_auction性能增益.md / fable_research_reports）+ 本线 baseline_v2 结果。
> 性质 = **给 PI 和团队的提案草稿**，最终结构由 Prof. Xu 拍板。

---

## 1. 两条线现状速览（合并的原材料）

| | Alec 线（主线） | 本线（Mason） |
|---|---|---|
| FM | 3 个 persona proposer（DeepSeek/GLM/Qwen，前沿级） | 单个 qwen2.5-coder:14b（本地单卡） |
| 机制 | persona 分化（供给）+ 四项 bid auction（选择），(1-1/e) 定理 | skill-graph frontier 调度（供给）+ learnability preflight（选择） |
| 口径 | DiCode 官方：1024 held-out 世界 mean return，one-hot 67，2e9 步 | 目前：tier2-4 held-out SR + mean_performance，one-hot 67，157M 步 |
| 现状 | baseline 复现健康（44.6 > 论文最强 baseline 41.54）；**auction 臂中段 −3 分（诚实负结果）** | baseline 完成（157M/64节点/零故障）；+A 跑通；+A+B 修复后待跑 |

## 2. ★关键发现：两条线独立收敛到同一组概念

1. **Learnability**：Alec 的第 4 项 bid = 父关 `p(1-p)`；本线 preflight 的存档分 = 候选 `sr(1-sr)`。**同一个量**，两种用法（他们=先验 bid 项，我们=实测双阈值准入闸）。
2. **Frontier targeting**：Alec 的 Proposer-Feasible（"造 student 能力边缘的关"）≈ 本线 skill-graph scheduler 的 frontier tier 定位。**同一个概念**，两种实现（persona prompt vs 显式技能图）。

**含义**：合并论文里这两个概念应升格为**全文共享机制**，两条线是它们在不同 FM 尺度下的实例化——这让联合论文有一条真正统一的论点，而非两个项目的拼盘。

## 3. 提议的联合论文骨架

**统一论点**：LLM 驱动的 code-level UED 需要对任务生成做**双侧引导**——供给侧（生成什么）与选择侧（准入什么）；且 FM 越小，引导从"锦上添花"变为"生存必需"。

```
§1 Intro：code-level UED（DiCode）+ 双侧引导论点 + 尺度轴
§2 Related work（SCALAR / CODE-SHARP / SkillGraph…，两线共用，需与 novelty_positioning.md 合稿）
§3 方法
   3.1 供给侧引导：persona 分化（235B） / skill-graph 调度（14B）
   3.2 选择侧引导：次模 auction（235B） / learnability preflight（14B）
   3.3 理论（Alec）：(1-1/e) 次模贪心；v2 机制塑造供给
§4 前沿尺度实验（Alec）：baseline 对标 48.33 + auction 消融（含中段负结果的诚实分析）
§5 ★小模型尺度实验（本线的 section）
   5.1 小模型失败模式刻画：语义幻觉而非语法（15×AttributeError / 0×SyntaxError）
   5.2 三组消融：baseline / +A / +A+B（同 seed 同环境，157M 对齐）
   5.3 机制能买回多少：相对裸 14B 的恢复量 +（若口径对齐成功）相对 235B 参照的差距
§6 讨论：尺度 × 机制交互；何时引导必需；负结果的信息量
```

**本线的身份 = §5 + §2 部分 + 失败模式分析**。即使 +A+B 结果平淡，§5.1 的刻画与受控消融本身也是该 section 的骨肉（无人做过小模型 code-level UED 的系统刻画）。

## 4. 本线要做的对齐动作（按优先级）

### 4.1 【必做】采用共享评测协议
- 用 `experiments/training/eval_checkpoints.py`（已在本分支）评本线三个臂的 checkpoint：**num_envs=1024, num_steps=8192, mean episode return**。
- 条件化已天然一致（one-hot 67）。⚠️ 复用 Alec 已排的坑：独立评测脚本默认 `embedding_size=1024` 会导致 restore 崩，须按 conditioning_type 判断用 67（他们已修，核对本分支是否含该修复）。
- 产出 = 三臂在**与 Alec 完全相同口径**下的 mean_return → 可进联合论文同一张表。
- 保留 tier2-4 SR 分解作诊断指标（对应 Alec §9.4 的 skill 分解风格，两线格式统一）。

### 4.2 【必做】步数尺度的诚实表述
- 本线 157M vs Alec 2e9（≈8%）。表述为 **commodity 预算 regime**（单卡 + 本地 14B 的现实预算），比较均注明训练步数；不与 2e9 终值直接比绝对分。
- 若团队认为需要，本线可延长 run（成本线性），但先以 157M 出第一版。

### 4.3 【建议】术语统一
- 全文采用 supply-side / selection-side steering 词汇；本线 A→supply（skill-graph targeting），B→selection（learnability gating）。
- Learnability 概念两线合写一个定义小节：bid 项（先验，父关 p(1-p)）vs 准入闸（实测，候选 sr 双阈值路由）——互补而非重复。

### 4.4 【与 Alec 协调】
- held-out 1024 世界的 seed/生成配置是否与他们完全一致（同一批世界 or 同协议不同实例——前者可配对比较，后者只能同口径比均值）。
- 图表模板与 wandb project 归档方式统一。
- Related work 合稿（本线 novelty_positioning.md + 他们 fable_research_reports 的竞品分析，避免同一竞品两种描述）。

## 5. 给 PI 的三个决策点（提案时列出）

1. **论文定位**：冲 SOTA 叙事 vs 「LLM-UED 机制的诚实解剖」叙事？现状（auction 中段 −3、小模型深层 tier 为 0）更支持后者——解剖式论文里两线的负/平结果都是信息，且本线失败模式刻画是解剖叙事的直接素材。
2. **seed 预算**：两线目前均单 seed；联合投稿前每臂补 seed 2,3 的算力从哪出、优先补哪条线。
3. **作者与分工**：§5 由本线执笔；§2 合稿归属；理论节（Alec）与实验节交叉引用的写法。

## 6. 风险与诚实边界

- 本线 157M 下深层 tier 可能三臂皆 ~0 —— 届时主张收缩为「课程质量中间指标（tier 分布/拒绝率/frontier 命中）的机制效应」+ 失败模式刻画，SR 差异留给更长预算。
- 两线基座差异（persona 多 FM vs 单 FM）意味着 §3 的"同一机制两种实例化"是**概念级对应，不是受控对照**——文中须明说，防评审误读为跨尺度受控实验。
- Alec 线负结果若持续到终局，论文骨架需向"解剖"进一步倾斜——这是 PI 决策，本线保持两种叙事下素材均可用。
