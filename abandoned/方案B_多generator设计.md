# 方案 B：多 generator 注入 + 异质 estimator 打分（完整设计）

> 记录于 2026-06-20。本文档固化「引入会训练的多 generator agent 提高关卡多样性」这一轮设计的所有结论。
> 是 [future_routes.md](future_routes.md) 路线 B 的拍板落地版。后续以此为锚，不再凭记忆重推。
> 配套修正：旧文档 λ 语义写反、bid 标准化实现有 bug（见 §6）。

---

## 0. 一句话定位

把 future_routes 路线 B 落成可执行设计：**N 个会训练的 generator（PCGRL 式 RL）各被一个异质 estimator 信号驱动、分工往固定 buffer 注入 level；异质 estimator + auction 在固定 buffer 上做课程**。分层保住收敛定理，多 generator 提供覆盖多样性。

---

## 1. 用户最初设想的两处误判（已纠正，数据/证明支撑）

### 1.1 「去掉异质 estimator 收敛定理就成立」—— 错
收敛定理的强凹性**只来自 `αH(y)`**，罩的是对手决策变量 `y∈Δ_n`（单纯形）。score `s` 是不是异质，定理不在乎——`s` 只通过对 `y` 线性的项 `yᵀs` 进入目标（proof_skeleton §0 第101行 A5、§8.2）。
- **异质 estimator ≠ 破坏收敛**：异质是「目标层 general-sum」，免费收敛，是 novelty 不是缺陷。
- **真正破坏定理的是引入 generator agent**：对手决策变量从单纯形 `y` 跳到非凸网络参数 `φ`，熵正则罩不住 `φ`（proof_skeleton §8.3）。这跟异质与否**完全正交**。
- → 「引入 generator + 去掉异质来挽救定理」是两个旋钮拧反：既丢免费 novelty，又拿不回被 generator 破坏的定理。

### 1.2 「多 generator 自动带来多样性」—— 需要分化机制，否则 mode collapse
N 个 generator 灌同一个 buffer，若无显式分化激励会**收敛到同一模式**（都造同类关卡），N 退化成 1。
- **解法（核心接缝）**：让 N 个异质 estimator **同时充当 N 个 generator 的训练信号**。estimator-j 既在打分层排序（保收敛），又作为 generator-j 的 reward 驱动其往 buffer 不同区域注入（产多样性）。
- → 「异质 estimator」与「多 generator」不再是两件无关的事，拧成一个统一故事：**异质 estimator 从纯排序器升级为 generator 的多样性引擎**。

---

## 2. 分层架构（唯一保住定理的形态，proof_skeleton §8.4）

```
注入层（覆盖多样性来源，无收敛保证 —— ACCEL/PLR/SFL 同样没有，可接受）
   ├─ generator-A  ←reward= 难度/学习潜力信号   → 造「刚好能学」关卡
   ├─ generator-B  ←reward= 覆盖/新颖信号        → 造 student 没去过的关卡（exploration 主力）
   └─ generator-C  ←reward= 转移新颖信号          → 造动态没见过的关卡
        │  (N estimator 同时是 N generator 的训练信号 = 强制分化、防 mode collapse)
        ↓ 各自填进固定共享 buffer Λ_n（不经过 auction）
打分层（收敛定理罩这一层：固定 buffer + 单纯形 y + αH(y) 强凹）
   ├─ estimator 对 Λ_n 打分（含 learnability p(1-p)，见 §4）
   └─ auction: w=softmax(bids/λ)，s̄_w=Σ w_j s^(j) → 进 PLR 采样
```

- **generator 只往 buffer 填 level，不与 student 做真 min_x max_φ 对抗**。绝不走 PAIRED 式彻底对抗（退回 NCC 明说无解的 non-zero-sum open problem，丢全篇最硬理论卖点）。
- 收敛定理罩「给定 buffer 的课程求解层」，前提是 buffer 刷新被当作外部（proof_skeleton §8.4，🟡 但路线可用）。

---

## 3. generator 实现：方案 B = PCGRL 式 tile-MDP + PPO（已拍板）

deep-research（task w40znqjx0，103 agent/20 源/25 claim 3-vote）确认：**无任何已发表工作把 generator 训练在 SFL learnability p(1-p) 上**——这是真空白（中心 novelty）。但可用正交零件拼装。三套候选，已选 B：

| 方案 | 怎么绕开不可微 | 取舍 |
|---|---|---|
| A：ACCEL 式编辑器 | mutate+选择，不反传过 level 参数（signal-agnostic） | 最快跑通，表达力受限；可作 B 之前的脚手架 |
| **B：PCGRL tile-MDP + PPO（选定）** | RL 不需信号对 level 可微（canonical recipe） | 真 RL generator，表达力+novelty 强 |
| C：CLUTR 冻结 latent + generator | 在固定连续 latent 上生成 | 唯一有硬证据「overcomes non-stationarity」；工程最重，作非平稳兜底 |

**方案 B 落地**：
- generator = RL agent，关卡设计建成「逐 tile 编辑」MDP，PPO 训。
- reward = generator-j 的有方向难度/覆盖/转移信号（生成完一个 level → 跑 student rollout 测 → reward）。
- N 个 generator 各自 reward 用不同 estimator 信号，自然分到 buffer 不同区域。

**被 deep-research 排除的方案**（无验证证据，别在其上建论点）：Gumbel-softmax、CMA-ES、MAP-Elites/QD（仅可作防 collapse 的辅助 novelty bonus，非主信号）。

---

## 4. 训练信号：generator 不用 p(1-p)，p(1-p) 退到打分层

### 4.1 p(1-p) 的两个缺陷（为什么不拿它训 generator）
1. **方向盲**：p=0（过难）和 p=1（饱和）给完全相同的 reward 0，generator 不知该把关卡变难还是变简单 → REINFORCE 梯度方差极大。
2. **会消失的窄带靶子**（非平稳恶性版）：p≈0.5 的 learnable 关卡随 student 变强快速枯竭。
→ ①更根本：信号有方向时追移动靶就容易多了。

### 4.2 generator 用「有方向的信号」（PERM 等先例已回避 p(1-p) 转用难度代理）
| 信号 | 有方向 | 饱和/过难可分 | 非平稳 |
|---|---|---|---|
| p(1-p)（learnability） | ❌ | ❌ 都=0 | 最恶性 |
| **p 直接 / `-(p-0.5)²`**（难度匹配） | ✅ | ✅ | 温和 |
| GoalGAN 式 band [0.1,0.9] | ✅ | ✅ | 温和 |
| **transition-prediction error**（TRACED） | ✅ | ✅ | 温和 |
| marginal benefit（学习增益 delta） | ✅ | ✅ | 中等 |

> ⭐ 本表只列「有方向 vs 非平稳」属性，**各信号的去留判断（哪些要写/哪些降为盲区候选/哪些被同维度代表）见 §5.0 汇总表**。一句话：difficulty-match 主力要写；transition-error/co-learnability 实现贵降为数据驱动盲区候选；GoalGAN band 被 `-(p-0.5)²` 代表（保留为 GAN 技巧）；marginal benefit 非新维度（难度备选测法）。

### 4.3 p(1-p) 退到打分层当 estimator —— generator 正好救活它
- 旧实测：maze 上 learnability 当 estimator **退化为常数 0**（future_routes §1.2/§2.1），因 buffer 里 p≈0.5 关卡几乎没有。
- **但那是在「随机生成的 buffer」上测的**。新设计有 generator 主动造 learnable 关卡 → buffer 里有大量 p≈0.5 关卡 → **learnability estimator 不再退化、第一次在 maze 上活过来**。
- learnability 是收敛定理闸门核验最干净的样本（`p(1-p)∈[0,1/4]` 有界✅、不依赖 y✅，proof_skeleton §4），也是「non-regret score + 收敛保证」novelty 叙事的关键样本 → **打分层强烈建议保留**。

### 4.4 职责分离（一句话）
```
注入层 generator 训练信号 = 有方向的难度/覆盖/转移信号（好训、稳）
打分层 estimator 排序信号 = 含 learnability p(1-p)（异质、保收敛、不训网络）
```

---

## 5. 异质 estimator 选择：按「两两正交」选，防 generator 塌缩

### 5.1 原则
异质 estimator 价值 = 正交性（相关性低才能让 N 个 generator 分化）。反例警告：future_routes §2.2 实测 **ρ(PVL,CENIE)=0.73 偏高**。文献先例 TRACED 已组合 value-loss + transition-error + co-learnability 并实测互补（非冗余）——但 TRACED 那两个贵信号在 SFL 流程里实现成本高（§5.0），**正交维度优先从即插即用的现成信号里凑**。

### 5.0 ⭐ 信号去留汇总（2026-06-20 修正，推翻 §5.2 旧表「transition-error 是主力」）

读完 SFL `get_learnability_set` 流程后核验各候选信号的**实现成本**，结论：候选信号去留命运各异，
**没有一个被淘汰，但只有「即插即用纯打分函数」是现在要写的主力**。

| 信号 | 维度 | 在 SFL 流程的实现成本 | 去留判断 |
|---|---|---|---|
| **`-(p-0.5)²`**（difficulty-match） | 难度/学习潜力 | 一行（`p=success_by_env` 现成） | ✅ **主力·要写** |
| **CENIE 反密度** | 覆盖/新颖 | 已有逻辑可搬（hidden state 密度） | ✅ **主力·要写** |
| **value_disagreement** | 价值不确定性 | jaxued_utils.py:63 **已有函数**（需 ensemble value） | ✅ **主力候选·要写** |
| collision_rate / ep_len / timeout_rate | 行为多样性 | SFL rollout 现成产出 | ✅ 备选维度（即插即用） |
| positive_value_loss / max_mc | regret 系 | jaxued_utils.py 已有 | ✅ 备选（regret 维度） |
| **transition-prediction error** | 转移动态新颖 | **大**：需加 forward model 网络头 + 训练 loss | 🟡 **盲区候选**（V1 正交性表明有需要则**必须做**，非可选） |
| **co-learnability**(TRACED) | 关卡间迁移结构 | **大**：需逐任务梯度步+跨任务评估，改训练循环 | 🟡 **盲区候选**（V1 正交性表明有需要则**必须做**，非可选） |
| GoalGAN 式 band [0.1,0.9] | 难度匹配（同 `-(p-0.5)²`） | — | ⛔ 同维度被代表（band 内部方向盲较弱）；保留为 **GAN 实现技巧**（post-hoc 成功率标签喂判别器） |
| marginal benefit | 学习增益（预期与难度相关） | 需双 student 快照（target student 副产品） | ⛔ 非新维度·难度备选测法；deep-research 打折（处理非平稳说法 0-3 否决） |

**两个「贵」estimator 各自捕捉的独特设计因素（砍掉它们的盲区分析）**：
- **transition-error = 环境动态可预测性**：高亮「转移 `(s,a)→s'` 对 agent 陌生」的关卡（狭窄通道/急转弯）。
  与成功率、状态覆盖都正交（一个关卡可 p≈0.5 + 去过 + 仍动态不可预测）。**盲区影响中等**——
  被 CENIE（状态覆盖）部分填补（没去过的地方动态通常也没学好）。
- **co-learnability = 关卡间迁移结构**：唯一「非单关卡」信号，高亮「学了能带动一片」的课程枢纽点。
  **盲区影响取决于要不要「课程顺序」卖点**——多 generator 覆盖故事下非必需；「generator 学造教学杠杆点」卖点下不可替代。

### 5.2 推荐 N=3（三个正交维度，主力用即插即用信号）
| 维度 | estimator（主力） | generator 被拉向 |
|---|---|---|
| 难度/学习潜力 | **`-(p-0.5)²`**（difficulty-match） | 「刚好能学」关卡 |
| **覆盖/新颖**（exploration 核心） | CENIE 反密度 / count-based novelty | student 没去过的状态区 |
| 价值不确定性 | **value_disagreement**（jaxued 已有，需 ensemble value） | 价值估计最不确定的关卡 |

**修正（推翻旧「gen-C 换 transition-error」表述）**：transition-error 实现贵（要加 forward model 头），
**先不默认进 N=3 主力**（主力优先用三个即插即用信号：难度/覆盖/不确定性）。但**「贵」≠「可选」**：
是否追加 transition-error 由 V1 正交性数据裁决（见 §5.4 决策树）——**若现成信号撑不起 N=3 正交，则贵信号必须实现，不能因为贵而绕过**（正交性是 generator 分化的硬前提，撑不起就没有多样性）。

### 5.3 加分项：显式多样性惩罚
generator reward 加一项「和其他 generator 已造关卡的距离」（inter-generator novelty bonus，MAP-Elites 思想），双保险防 collapse。

### 5.4 地基验证 = 数据驱动决策树（不可跳过）
> ✅ **本决策树已于 2026-06-20 执行完毕，结果见 §5.5：verdict = `ready_pool_sufficient`（现成池够 3 正交，贵信号列 future work）。** 以下为决策树方法论（保留）。

上述正交性除 ρ(PVL,CENIE)=0.73 是实测，其余均为文献定性推断。**落地前必须用 V1 探针测信号在 jaxnav 上的两两相关矩阵**——把「实现哪些 estimator」从拍脑袋变成数据决策：

1. **先测现成池**（零模型改动）：difficulty-match + CENIE + value_disagreement + collision_rate/ep_len + pvl/max_mc，跑一次 jaxnav rollout 出两两相关矩阵 + T3 增益判据（复用 sigma_measure.py 逻辑）。
2. **若现成池里有 3 个两两正交且覆盖核心维度**（难度/覆盖/不确定性）→ 主线用它们；transition-error / co-learnability 列为 future work（论文诚实写「两维度留作扩展」）。
3. **若现成信号两两都偏相关**（像 maze 上 ρ=0.73）→ 现成池撑不起 N=3 正交 → **则贵信号（transition-error 优先，原理上与成功率/覆盖最正交）必须实现，不能因为实现贵而绕过**。正交性是 generator 分化的硬前提：撑不起正交 = generator 会 mode collapse = 整个多样性卖点落空，此时贵信号是刚需不是可选。

→ 决策树的作用是把「先做哪个、贵信号要不要做」交给 V1 数据，而**不是**给「贵信号太麻烦就跳过」开口子。避免两个反向错误：①没验证就硬实现贵信号（可能白干）；②验证表明需要却因为贵而绕过（多样性落空）。这是方案 B 的地基（同 future_routes §4 的「先验地基」逻辑）。

### 5.5 ✅ V1 探针实测结果（2026-06-20，SFL jaxnav，3 seed）—— verdict = 现成池够

决策树已执行。探针（边训边验，载体 = SFL repo `sfl/train/{estimators,probe_orthogonality,train_probe}.py` + 官方 `cenie_density`，接进 `jaxnav_sfl.py` 的旁路 `get_probe_signals`）在 jaxnav 上实测**四真信号**（difficulty-match / learnability / PVL / CENIE，无任何代理）的两两相关。3 seed（`driven-planet-5` / `radiant-dream-6` / `fallen-deluge-7`）全程（update 50–2250，各 45 epoch）一致：

| 信号对 | jaxnav 实测（3 seed 平均范围） | maze 对照 |
|---|---|---|
| **ρ(PVL, CENIE)** | **-0.10 ~ -0.19**（甚至轻微**负**相关） | **+0.73**（高共线） |
| ρ(difficulty, PVL) | 0.05 ~ 0.10 | — |
| ρ(difficulty, CENIE) | -0.08 ~ 0.02 | — |
| ρ(difficulty, learnability) | 1.0（恒等，见下） | — |

**裁决：`ready_pool_sufficient`** —— 三个真维度（难度 / 不确定性PVL / 覆盖CENIE）两两 |ρ|<0.2，现成池足以撑起 N=3 正交。**主线用现成池；transition-error / co-learnability 列 future work，不必现在实现贵信号**（决策树第 2 分支）。

**这正面验证了「转 jaxnav 主场」（§7）**：maze 上 PVL/CENIE 高共线（0.73）是 maze 特有退化，jaxnav 上它们真正正交（甚至反向）。

两个必须诚实标注的点：
1. **ρ(difficulty, learnability)=1.0 是恒等，非 bug**：learnability `p(1-p) = 0.25-(p-0.5)²`，与 difficulty `-(p-0.5)²` 差一个常数+符号，是同维度的线性变换（§5.0 已判同维度）。故四信号里**有效正交维度最多 3 个**，learnability 只是难度维的另一写法，决策树在难度/PVL/CENIE 三维上判。
2. **饱和判据修正（实测教训）**：探针初版判据 `frac_hi+frac_lo>0.8` 把 jaxnav 的健康**双峰分布**（一半 level success≈1 + 一半≈0，p_std≈0.45 很分散）误报为饱和，致非饱和样本只剩 2-3、统计基础虚薄。正确判据 = **p_std<0.05（success_rate 几乎恒定）或 frac_hi>0.95（单极全解开）才算饱和，双峰相关完全可信不算饱和**。修正后非饱和样本 2→45（覆盖全程）。判据只影响 saturated 标记，csv 存了 raw 相关+p_std，可 `rejudge_trace.py` **离线重判不需重训**。详见 memory [[jaxnav-orthogonality-verified]]、[[probe-hidden-oom-subsample]]。

---

## 6. auction 必要性 + bid 标准化修复 ⭐（本轮核心技术产出）

### 6.1 auction 必要性：打分层口径已实测无差异；定 α 移交 Stage4 阶段1（generator-on）
- **旧设计 / 打分层口径（generator-off，固定 buffer 同质来源）**：auction 只是对同一批 level 不同排序，必要性弱，实测 argmax 就够。
  **已用 jaxnav 真实数据坐实（2026-06-22，job 3363545/46/47，`AUCTION_SCORING=true` 且 `GENERATOR_INJECTION=false`，{inf,1.0,3.0}×8~9 seed 跑满 3e8）**：
  三档 sampled_wr AUC（0.926/0.930/0.926）、达 wr≥0.9 收敛速度（393/432/487 upd）、末值 CI **全部重叠**；
  唯一稳定现象是 argmax 早熟但终点略低、uniform 终点略高，差距 ~1pt，无统计显著。**结论：在固定 buffer 上，auction 加权 vs argmax 无增益。**
  → 这是一个**诚实的负对照（下界）**：它精确反驳「auction 增益来自加权动作本身」，把增益（若有）归因逼向 generator 注入层。**不能用它给方案 B 定 α —— 它测错了层**（λ 只在 SFL 海选的同质 buffer 上重加权，N 个 generator 没参与）。
- **新设计 / 真实系统口径（generator-on，多 generator 异质来源）**：auction 职责变为「在来自不同 generator 的异质 level 混合池上，决定优先回放哪些」。single-winner(argmax) vs fractional(混合) **第一次有实质意义**：fractional 防止打分层把注入层辛苦造出的多样性又压回单一模式。**这才是定 α 的正确实验，由 Stage4 阶段1（`GEN_AUCTION_LAMBDA` 主扫描轴、看 learnability 方差结构）回答，见 [STAGE4_实验设计.md](STAGE4_实验设计.md) §1**。
- → **定位**：打分层 λ 消融（generator-off）= 已完成负下界，论文写诚实消融；generator 层 α（generator-on）= Stage4 阶段1 待出。**两个 λ 不可混淆**（`AUCTION_LAMBDA` vs `GEN_AUCTION_LAMBDA`）。**前提是先修好 bid 标准化（§6.2）让 λ 能真正扫到 fractional 区**。

### 6.2 bid 标准化的三个确诊 bug（numpy 推导验证，2026-06-20）
现 `multi_estimator_plr_runner.py:160-197` 的 bid 逻辑有三处 bug：

| # | bug | 根因 | 实测 |
|---|---|---|---|
| ① | α=0 炸弹 | bid 用原始量级，CENIE −354 永远最小 → argmax 恒选 PVL，CENIE 权重恒 0 | bids=[0.014,0,−354] |
| ② | α>0 塌成 uniform | bid 用 **z-score 后全体均值**，而 z-score 定义上零均值 → bids≈[6e-16,0,−2e-16] → softmax 塌成 [.333,.333,.333] | λ 完全失效 |
| ③ | λ 语义写反 | 代码 `softmax(bids/λ)`：λ→0=sharp(argmax)，λ 大=uniform；λ=∞ 特判 argmax | 旧扫描点 {∞,1.0,0.01} 中 ∞ 和 0.01 **同在 argmax 端** |

### 6.3 修复设计：bid = 「z-score 后 top-k 分位均值」
两步，per-level 混合用的 `s^(j)` 与 bid 摘要 `b_j` 用**不同**归一化：

1. **per-level z-score**（保留，用于 `mixed=Σ w_j s^(j)`）：每 estimator 各自减均值除标准差 → 量级可比，比的是相对位置。
2. **bid = z 后 top-k 分位均值**（新）：对每 estimator z 分数排序取最高前 k 个（~12.5%）求均值。
   - 为什么不取全体均值：z 后全体均值恒 0（bug②根因）。
   - 为什么取 top-k：bid 要答「该 estimator 对它最看重的那批 level 意见有多强」，直接对应 PLR「优先回放高分 level」语义；对量级悬殊免疫（z-score 已抹平量级，top-k 提取相对突出度）。

**numpy 验证结果（全过）**：
- 量级悬殊免疫：CENIE 从「权重恒 0」变为拿到 0.44 实质权重。
- λ 扫出连续单调 frontier：λ=∞→[1,0,0]（argmax 端）、λ=1.0→[.25,.28,.47]（三方混合）、λ=3.0→近均匀；端点 λ→0=one-hot argmax、λ→大=uniform。
- 三个 estimator 都参与：λ=1 时 min 权重 0.25（无被压死）。
- NaN 安全：incomplete level（-inf）屏蔽后 mixed 有限值无 NaN、incomplete 位正确 -inf。

### 6.4 三处修复 —— ✅ 已实现并验证（2026-06-20，SFL repo 载体）
独立 estimator 无关模块 `auction_bid.py`（z-score / top-k bid / softmax 权重 / mix），已部署 `oscar:~/_bidtest/`，
配套 `_test_auction_bid.py` 在 Oscar `sfl` 环境（真 JAX）跑 **16/16 通过**。三处修复全部落地：
1. **bid 定义**：z 后全体均值 → z 后 top-k 分位均值（`topk_bids`）。✅ 断言验证 bids 不恒 0。
2. **解除门控**：标准化不再被 `auction_alpha>0` 门控，对任意 N 个 estimator 无条件标准化（`standardize_per_estimator`）。✅
3. **λ 语义对齐**：`auction_weights` 实现 λ=inf/λ→0=argmax 端、λ 大=uniform 端；文档约定已统一（代码温度语义）。✅ 断言验证端点。
   STAGE3 λ 扫描点改为 {∞(argmax), 3.0(近均匀), 1.0(混合), 0.3(偏argmax)}。

**模块设计**：estimator 无关（输入 (N,M) per_est，对任意 N 成立，N=5 已测）→ 直接复用，无论最终用哪些 estimator。
**待验证**：top-k 的 k（分位比例 ~12.5%）是新超参，不敏感（10%~20% 类似 frontier），jaxnav 接进 `get_learnability_set` 跑通时确认一次。
**接线点**：SFL `get_learnability_set`（jaxnav_sfl.py:178-300）的 `argsort(learnability)` 处——把单 estimator 选择换成多 estimator → `mix_scores` → argsort（见 §8 / 集成说明）。

---

## 7. 环境：jaxnav 主场 + minigrid 对照

### 7.1 jaxnav 必要性（两支柱，非平稳是较弱的那个）
- **支柱一（弱，被 target student 削掉）**：jaxnav learnability 平稳 → generator 追的靶动得慢 → 训练易。
- **支柱二（强，决定理论卖点能否落地）**：minigrid learnability 衰减极快（future_routes §4.5：0.246→0.04→0），learnable 关卡窗口期极短 → 打分层 learnability estimator 仍大概率退化；jaxnav 全程维持 0.22 → estimator 真正活着 → 「non-regret score + 收敛保证」novelty 样本站得住。
- → **jaxnav 是理论卖点能落地的关键环境**，从来不主要靠非平稳撑。

### 7.2 「learnability 低 ≠ agent 饱和」（证伪用户设想）
maze 上 learnability 暴跌主因是**随机海选机制失效**（撞不到 p≈0.5 关卡），非 student 真饱和：实测 0<p<1 仅占 0.2%~1% 递减，但 p=0（解不了）占 19% → 有 19% 解不了的关卡说明 agent 没饱和，只是随机生成器喂不出「刚好能学」的关卡。**这正是 generator 价值最强论据**：在随机海选失效处继续造 learnable 关卡。

### 7.3 环境分工
| 环境 | 角色 | 理由 |
|---|---|---|
| **jaxnav** | 主场 | learnability 全程平稳 → estimator 活着 → 理论落地；baseline 全集现成 |
| minigrid/maze | 对照+故事 | learnability 快速枯竭 → 演示「SFL 随机海选末期掉 0.04，generator 能否撑住」 |

### 7.4 对照 SOTA（jaxnav 上，来自 SFL 论文 NeurIPS 2024）
DR、PLR(MaxMC)、PLR(PVL)、PLR-Robust、ACCEL(MaxMC)、ACCEL-Robust、SFL，外加已有 PAIRED、理论出发点 NCC。
评测口径：CVaR worst-case + hand-designed test set + 随机 100-map（单/多 agent）。
**这些 baseline 实现都在 SFL 官方 repo（jaxued 血统，已在 Oscar `sfl` 环境部署，future_routes §179）。**

---

## 8. 工程载体决策：SFL 官方 repo（已拍板）

- **载体 = SFL 官方 repo（jaxued 血统）**，非 minimax。
- 代价：把 multi-estimator auction + α 旋钮 + generator 注入 + bid 修复（§6）移植到 jaxued 血统；minimax 的 Stage3 现有代码（multi_estimator_plr_runner.py）不能直接用，但其 auction/标准化**逻辑**可搬（bid 修复与具体 estimator 无关，对任意 N 成立）。
- 红利：jaxnav 环境 + 整排 baseline 现成，直接同表对比。

---

## 9. 两个红灯（诚实标注，不淡化）

1. **障碍2（在线非平稳）是真未解研究问题**：所有前例（PERM 离线、CLUTR 冻结 manifold）都是「绕开」非平稳，唯一沾边在线处理的 Marginal-Benefit 双快照恰被 deep-research 对抗验证否决（0-3）。**用 target student 周期重评压住（温和但不能省）**；「在线正面解决」作可选加分项，非必需地基。
2. **放弃 regret 换 learnability 无现成收敛背书**：SFL 论文肯定 regret 理论本身没错、只批评其近似。收敛保证只能来自「分层 + 固定 buffer + αH(y)」（proof_skeleton §8.4），那套对 generator 注入层不提供保证（同 ACCEL/PLR）。论文须诚实写。

---

## 10. 落地前必须验证的地基（汇总）

1. ✅ **信号正交性 = 数据驱动决策树**（§5.4/§5.5）：**已完成（2026-06-20）**。jaxnav 3 seed 实测 difficulty/PVL/CENIE 两两 |ρ|<0.2（ρ(PVL,CENIE)≈-0.15 vs maze 0.73）→ **verdict=现成池够，主线用现成池、贵信号(transition-error)列 future work**。这一步同时裁决了「实现哪些 estimator」：N=3 主力即插即用信号，不实现贵信号。
2. **learnability estimator 复活**（§4.3）：jaxnav 上跑 generator，测打分层 learnability estimator 方差是否 >0（对比 minigrid 恒 0）。这验证 jaxnav 主场是否铁证。
3. **bid 修复 frontier**（§6.3/§6.4）：bid 模块本身已 16/16 单测过（Oscar）；剩 jaxnav 接进 `get_learnability_set` 后确认 top-k 的 k、λ 扫出真前沿。
4. **auction 必要性消融**（§6.1）：{argmax, fractional, uniform} 有无性能差。
5. **generator 维持 learnability**（§7.2）：generator 能否把 maze 衰减曲线撑住（目前是 future_routes §177 推断，未实测）—— 方案 B 第一个核心 claim。

---

## 11. 关联文档 / memory
- [future_routes.md](future_routes.md)（路线 A/B 分叉、§4.5 SFL 复现实测、本设计的前身）
- [proof_skeleton_entropy_regularized_heterogeneous_UED.md](proof_skeleton_entropy_regularized_heterogeneous_UED.md) §0/§4/§8（A5 线性进入、闸门核验、两层 general-sum + §8.4 分层保定理）
- [STAGE3_实验设计.md](STAGE3_实验设计.md)（旧 maze 实验矩阵，λ 语义/bid 标准化表述已按 §6 修正）
- deep-research task w40znqjx0（generator learnability 信号空白 + 拼装零件）
- 关联 memory：[[generator-teacher-breaks-convergence]]、[[auction-mechanism-is-plr-not-paired]]、[[stage2-cancelled-direct-to-stage3]]、[[sfl-zero-variance-pq-bimodal]]
