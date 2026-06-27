# STAGE4 Phase2 实验设计优化方向

> # ★★★ 决定性突破（2026-06-26，请先读这段）★★★
>
> **候选池基数是真瓶颈，不是信号/课程/塌缩。** 用户抓出对照不公平：baseline 每 eval 海选
> **5000** 候选关，generator 只 **128**（基数差 39 倍）。把 generator 候选池拉到 2000/3000
> （`GEN_POOL_PER_GEN`，注入数仍 48 不变、PPO 规模仍 64 不变，单变量；分批 `GEN_ROLLOUT_CHUNK=250`
> 避 PCGRL CNN 显存 OOM）后：
>
> | 指标 | currA pool=3000 (fl2mgtjq) | gate pool=2000 (qfo781yy) | baseline | 之前所有 generator arm |
> |---|---|---|---|---|
> | **Nav2** | **1.000** | **1.000** | 0.90 | **全 0** |
> | Nav3 | **1.000** | 0.90 | 0.90 | 0 |
> | **overall** | **1.000** | 0.97 | 0.96 | 0.60 |
>
> **第一次有 generator arm 真正攻破 Nav2（满分，非 0.4 运气），两个独立配置同时攻破→非单 seed 噪声。**
> currA pool=3000 全满分 overall=1.0 **超** baseline。
>
> **★真检验口径（100-map + CVaR，文档 §2.1 的 SOTA 口径，非 5 关试水）——反转 §6.1 负结果：**
>
> | 100-map 指标 | baseline | currA pool=3000 | gate pool=2000 |
> |---|---|---|---|
> | overall win（均值） | 0.979 | 0.986 | 0.972 |
> | **CVaR10 win（最难10%关）** | **0.477** | **0.771 (+29pt)** | 0.605 |
> | worst-bin return（最惨关） | **−4.943（崩盘）** | **−0.872（不崩）** | −3.714 |
>
> **差距全在尾部**：均值三者持平（0.97~0.99，大部分关谁都解），但**最难 10% 关的 CVaR worst-case**
> currA 赢 baseline **+29pt**（0.771 vs 0.477），baseline 在最惨关有 ~47% 崩盘率（return −4.9），currA 几乎不崩。
> **这反转了下方 §6.1 的旧负结果**（λ=1.0 难关 CVaR 输 baseline −22pt）——旧负结果真因是候选池太小(128)，
> 非方法有害。**这正是 UED 该有的卖点：定向供给难关→student 在 worst-case 更鲁棒。** 详见 memory
> `cvar-100map-reverses-negative-result`。（仍单 seed，CVaR 是尾部统计须多 seed 复现。）
>
> **★反直觉真相（推翻本文档下方一整条主线假设）**：`gen_p_learnable_frac` 仍 ≈0、关仍两极，
> Nav2 却满分翻起。**不需要"造可学 p≈0.5 关"，只需要候选基数大**——大 pool 让 PLR 有足够**多样**
> 的关可挑，偶尔掷出的 Nav2/3 结构关被捕捉反复喂 student（=借到 baseline 大基数+多样性机制）。
> **下方"治坍缩/可学区供给/单标量定向必坍"等诊断对结论的指向需整体重评**——它们都假设
> "可学率低=失败主因"，但大基数证明可学率低也能满分赢。真瓶颈是最朴素的"候选基数"。
>
> **必做下一步（最高优先级）**：多 seed 复现（line 286 单 seed 噪声警告仍悬，两配置满分已是强信号但须坐实）。
> 详见 memory `bigpool-breakthrough-nav2-solved` / `generator-pool-size-unfair-vs-baseline`。

---

> # ⚠️⚠️ 重大警告（2026-06-25 晚发现，请先读这段再读全文）⚠️⚠️
>
> **本文档大量结论建立在一个根本性的对照 bug 上，须整体重评。**
>
> **Bug**：generator 给 agent 设的**初始朝向 θ 写成了离散 4 值** `{0°,90°,180°,270°}`
> （`pcgrl_generator.py:119`：`theta = (jnp.pi/2)*jax.random.choice(r_th, jnp.arange(4))`，
> 注释还假引用了一个**不存在的** `barn_test_case`），而 **baseline(DR 随机) 用的是连续
> `uniform(-π,π)`**（jaxmarl `grid_map.py:140`）。两条 pipeline 的初始朝向分布不一致 = **对照不干净**。
>
> **后果**：student 在离散朝向上训练，**学不到"连续转向"的鲁棒寻路**。测试关 **SingleNav2
> （初始 θ=180° 背朝目标 + 起点贴墙角 + 需精细掉头）** 因此对**所有** generator arm
> （pvl/anchored/euc/euc+ALP-bid/euc+ALP-quota/课程 A/B）都得 win_rate=**0**。本文档把这个
> 现象反复解读为"机制坍缩(collapse)"、"curriculum-too-hard"、"Nav2 是所有非 baseline 方案的
> 共性病"——**这些诊断在修复朝向 bug 前全部失真，可能部分或全部是这一行 bug 造成的**。
>
> **须重评的核心结论**：① §坍缩"铁证"（Nav2 全 0 是共性病）；② §curriculum-too-hard 真因；
> ③ s4p2A 难关 CVaR 输 baseline -22pct；④ gate 无效；⑤ 结构课程 A/B 的 Nav 判据。
> **不受影响的**：learnability 方差复活（阶段1 GO）、anchored 离线 ρ 验证、SFL p(1-p) 数学塌缩、
> 可解性设计、ALP=curation 性质判定（这些建立在方差结构/离线 ρ/数学推导上，非 Nav win rate）。
>
> **修复**：`pcgrl_generator.py:119` 改为连续 `jax.random.uniform(r_th, (), minval=-jnp.pi, maxval=jnp.pi)`
> 与 DR 对齐 + 单测 + **重跑 baseline-vs-generator 对照**。详见 memory
> `generator-theta-discrete-bug-pollutes-all-arms` 与 `nav2-nav3-real-difference`。
>
> ---

> 背景：s4p2A（generator 注入 + auction，estimator=[difficulty, anchored/pvl, cenie]）在 test-set
> worst-case CVaR 上系统性输给 SFL baseline（最难 10% 关：l1 0.63 vs base 0.87，输 ~24pct）。
> 根因已确诊（见 memory `s4p2a-root-cause-difficulty-axis-mismatch`）：**难度维度错配**——三个
> estimator 全 student-centric，集体被"高墙密度→瞬时高 value-loss/状态新颖"捷径主导，generator
> 学会**堆墙刷分**；但 test-set 真正最难的关是"中等 fill(0.337) + 起终点远(长程导航)"，所有
> student-centric 信号盲视该维度。anchored=PVL×测地 修复失败（1400 关离线证实 generator 用
> "堆墙+绕路"同时刷高 PVL 和测地，组合捷径变形未消失）。
>
> **本文档不追求"零堆墙关"**。判据修正为（用户 2026-06-24 定）：**buffer 里堆墙关占少数、
> 有价值关（长程/近距离真难）占多数即可**——课程是分布塑形非完美筛选。
>
> 离线已验（1400 关 sweep，sim_formA_raw.npz + auction_bid.mix_scores）：
> `[euc, difficulty, cenie]`（砍 PVL + 引欧氏起终点距离 euc）的 top-175 buffer 构成 =
> **健康长程关 69.1% / 高墙关(fill>0.5) 14.9%**（≈4.6:1），fill 主峰 [0.35,0.45)、尾部少量高墙，
> euc 中位 7.62（长程维度补上）。按修正判据**达标**。CENIE 在"近距离真难"关上 z=2.33（全场最高），
> 证实异质权衡不误杀该类关（用户直觉成立）。

---

## 为什么是 euc 不是 geo（测地）—— 抗操纵的关键

| 量 | generator 能否操纵 | 原因 |
|---|---|---|
| **geo（测地最短路）** | **能** | 摆墙逼出绕路 → geo 变大（已证 1400 关 top-k 41% 堆墙绕路） |
| **euc（起终点欧氏距离）** | **不能** | 起终点位置定了 euc 就定了，摆墙改不了；且起终点由 `place_start_goal_on_map` 放、不归 generator 网络控制 |

→ euc 是**不可被 generator 摆墙操纵**的长程维度信号。这是它能进 auction（驱动层）而 geo 不能的根本原因。
（**前置确认项**：实现前须核实 jaxnav 起终点确由 place_env 采样器放、非 generator 输出——若 generator
能控制起终点位置则 euc 也可被操纵，需退回"euc 当放置约束"形态。）

---

## 三个优化方向（可独立做，亦可叠加）

### 方向 1：提高 difficulty 在 auction 里的权重

**动机**：difficulty=−(p−0.5)² 是"student 说简单/太难就少造、p≈0.5 才多造"的标准反馈闭环。
但 s4p2A 实测 auction 自动给 difficulty 权重仅 **0.068**（被 PVL/CENIE 压低），student 的难度反馈
"声音太小"。砍 PVL 后（方向3）difficulty 权重升到 0.10，仍低。手动抬高让难度反馈说话更响。

**工程复杂度：极低**。无新信号、无新数据。改 auction 权重的两种实现：
- (a) 改 `auction_lambda`（softmax 温度）——但它对所有 estimator 一致，不能单独抬 difficulty。
- (b) 在 `auction_bid.mix_scores` 后对 difficulty 维加固定乘子 / 给 estimator 配固定权重向量，
  绕过纯 bid 自动分配。需在 `auction_bid.py` 加一个可选 `fixed_weights` 参数（~10 行）。

**实验设计**：
1. **先离线扫**（不烧 GPU，复用 sim_formA_raw.npz）：把 difficulty 权重手动设 {0.1(基线), 0.3, 0.5, 0.7}，
   每档算 top-175 buffer 的「堆墙关占比 / 健康长程占比 / 近距离真难命中」三指标，画权重-占比曲线。
   找"堆墙占比明显降且长程/近距离真难不被挤掉"的甜区权重。
2. **若离线有甜区** → 真训练单档：`GEN_ESTIMATOR_IDS=[euc,difficulty,cenie]` + difficulty 固定权重=甜区值，
   10 seed × 3e8，对照 SFL base。看 worst-case CVaR。
3. **风险**：difficulty 权重过高 → 课程塌到 p≈0.5 的"舒适区"关，长程难关（p 可能<0.5 偏低）反被压低。
   离线扫必须监控"长程关是否被挤出 buffer"。difficulty 的 p 两极化塌缩（memory `sfl-zero-variance`）
   在 jaxnav 强 student 上也存在，权重抬太高可能放大该塌缩。

---

### 方向 2：换更强的"学习进度"反馈（learning progress / ALP）

**动机**：difficulty 是**瞬时难度**（这一刻 p 多少）。更接近"我觉得这关我已经会了"的是
**learning progress (LP)**：student 在这**类**关上**还在不在进步**。还在进步=有学习价值（多造）；
已会（进步停滞）=少造。LP 比瞬时 p 更能指向"还在进步空间里的盲区"，且对"堆墙→瞬时混乱"捷径
不敏感（堆墙关若 student 早已不再进步，LP=0，不得分）。代表：ALP-GMM（absolute learning progress）。

**工程复杂度：高（本文档三方向里最高）**。核心难点：现有所有 estimator 都是**单时刻**的——
generator 每次造**新关**（不重复），一次 rollout 出一个分，**没有"同一关的历史表现"**。而 LP/ALP 本质
需要"同一关（或同一参数区域）在两个时间点的 student 表现差"。三种实现路径，复杂度递增：

| 路径 | 机制 | 改动量 | 风险 |
|---|---|---|---|
| **2a. buffer 级 ALP** | 不追踪单关，对 buffer 整体维护"上次评估 win-rate"，本次评估差=LP | 中：buffer 加一列历史 win-rate + 每 epoch 重评估 buffer 内关 | 需 buffer 持久化关卡（现 buffer 存什么要查） |
| **2b. 参数空间 ALP-GMM** | 在关卡描述符空间（如 fill×euc）拟合 GMM，按"该区域 LP"采样（ALP-GMM 原版） | 高：要定义关卡参数空间 + 在线维护 LP-GMM + 改 generator 采样为按 LP 区域 | 描述符要人选（fill/euc）——回到"瞄准维度"，但 LP 是元信号非直接难度 |
| **2c. 双时刻 student 差** | 存 student 的"前一个 ckpt"，同一批注入关上跑两个 ckpt 的 p，LP=\|p_now−p_prev\| | 高：要持有历史 student params + 每批多跑一次旧 student rollout（≈+1× student 算力） | 类似 antagonist 的双 rollout 成本；LP 信号在强 student 上可能普遍小 |

**前置查证**（写实现前必做）：现有 PLR buffer（`get_learnability_set` / 注入路径）到底存不存关卡本身
（还是只存 score+ckpt）？memory `s4p2a-root-cause` §诊断方法论记"注入 level 数组没存盘，ckpt 只有
safetensors 参数"——**若 buffer 不持久化关卡，路径 2a 不可行**，只能走 2b/2c。这决定方向2 工程量。

**实验设计**：
1. **先离线可行性**：用现成 base/l1/anchored 三组 ckpt（它们是 student 训练的不同阶段/不同 run）当
   "不同时刻 student"，在 1400 关上算 LP=\|p(ckptA)−p(ckptB)\|，看 LP 是否：(a) 在堆墙绕路关上低
   （student 早不进步）、(b) 在长程/近距离真难关上高（还在进步）。**这能在写任何训练代码前判 LP 信号
   有没有区分度**——和 antagonist 离线验证同法（数据已在 sim_antag_raw.npz，3 个 base seed + l1 的 per-level p）。
2. **若离线 LP 有区分度** → 选 2a/2c 中工程量较小者实现（依赖前置查证结果），加 `GEN_ESTIMATOR_LP`，
   `[euc, LP, cenie]` 或 `[euc, difficulty, LP, cenie]`，10 seed × 3e8，对照 base + 方向3。
3. **若离线 LP 无区分度**（强 student 上 LP 普遍塌缩）→ 放弃方向2，省下高工程量。

---

### 方向 3：使用 [euc, difficulty, cenie]（砍 PVL + 引 euc）—— 对照 s4p2A 坏结果验证

**动机**：离线已验为三配置中最优（堆墙占比 24.9% vs 旧形式A 29.9%，buffer 健康长程 69%）。改动最小、
不引入 antagonist/LP 重型架构。但**离线正向 ≠ 训练有效**——必须和 s4p2A 的坏结果同口径对比，
才能证明"砍 PVL + 引 euc"真的扭转了 worst-case 负结果（这是用户明确要求的对照验证）。

**工程复杂度：低**。只需：
- (a) 新增 `GEN_ESTIMATOR_EUC`：generator 造关 + place_start_goal 放完起终点后，euc=‖agent_pos−goal_pos‖₂
  （`map_to_env_instance` 里已有 start_xy/goal_xy，直接算，~15 行，不依赖 student rollout，最廉价的 estimator）。
- (b) `compute_terminal_reward` 加 euc 分支。
- (c) 配置 `GEN_ESTIMATOR_IDS=[euc, difficulty, cenie]`、`AUCTION_USE_CENIE=true`、`GENERATOR_INJECTION=true`。

**实验设计（对照矩阵，全部 10 seed × 3e8、对齐 SFL 论文 rliable 口径、PROBE_ORTHOGONALITY=true）**：

| arm | GEN_ESTIMATOR_IDS | 角色 | 数据来源 |
|---|---|---|---|
| **base (SFL)** | — (GENERATOR_INJECTION=false) | 锚 | **已跑完，勿重交**（memory `stage4-baseline-already-done`，7seed 6.6e-4~2.6e-3 方差） |
| **s4p2A (旧坏结果)** | [difficulty, anchored, cenie] | 负结果对照 | 已跑（royal-snowflake-95 等，l1 CVaR 0.63） |
| **★ 新 arm** | **[euc, difficulty, cenie]** | 本方向 | 待提交 |

**判据（按用户修正判据，不要求堆墙归零）**：
1. **主判据**：新 arm 难关 CVaR（最难 10%）是否 ≥ base、且显著 > s4p2A 的 0.63。门槛检验
   （memory `s4p2A-baseline-anchor`）：任一 seed 到不了 base 的 ~0.96 末值就检视 idea。
2. **机制判据**：注入关 fill 分布——主体 fill 0.3-0.45 + euc 大（健康长程多数）、尾部少量 fill>0.5
   （堆墙少数），对齐离线预测的 69%/15%。**须在训练中 log 注入关的 fill 和 euc**（s4p2A 当年没 log，
   只能 PNG 反推——这次代码侧补 log injected-level fill/euc，见 memory 诊断方法论教训）。
3. **抗作弊 vs 泛化分开验**（调研 caveat）：(a) 注入关 fill 分布健康（抗堆墙生效）；
   (b) worst-case CVaR 真提升（泛化生效）——两者独立，不可混为一谈。

**对照价值**：base / s4p2A 数据已在 wandb，新 arm 一跑完即可三方同口径比 CVaR + fill 分布，
直接回答"砍 PVL + 引 euc 是否扭转 s4p2A 负结果"。这是三方向里**唯一能立刻拿到 SOTA 对标结论**的。

---

## 推荐执行顺序

1. **方向 3 先行**（工程最小、唯一能立刻对标 s4p2A 坏结果、离线已正向）——加 euc estimator + log 注入关
   fill/euc，提交 `[euc,difficulty,cenie]` 10seed。这是主线，验证"砍 PVL+引 euc"扭转负结果。
2. **方向 1 离线扫并行**（零 GPU）——在 sim_formA_raw.npz 上扫 difficulty 权重，找甜区；若有甜区，
   作为方向3 的一个变体 arm（`[euc, difficulty(高权重), cenie]`）一并提交。
3. **方向 2 先做离线可行性判定**（零 GPU，用现成 ckpt 当不同时刻 student 算 LP 区分度）——
   **有区分度才投入高工程量实现**；无区分度则放弃，避免在塌缩信号上烧 GPU。

> 共同前置查证（任何实现前）：① jaxnav 起终点是否由 place_env 放而非 generator 控制（决定 euc 抗操纵性）；
> ② 注入 buffer 是否持久化关卡本身（决定方向2 路径2a 可行性）。两项都查代码、不靠记忆。

---

## Baseline（SFL 随机海选）全过程分析 —— idea 攻击靶点画像（2026-06-24）

> 锚：phase2 baseline `s4p2-base`（48lfz699 / 710sivtf / yc00jix1，3 seed，11×11，
> `GENERATOR_INJECTION=False`）。注意 phase1 的 `s4-base` 是不同设置（更差，Nav2/3≈0~0.3），
> **不参与对标，只用 s4p2-base 当锚**。

### Baseline 怎么跑
DR 随机撒墙生成候选 → PLR 按 learnability 维护回放 buffer → 训练混合"新随机关+回放关"。
**无会学习的 generator**，全程 learnability 标签恒 0.250（PLR 占位分，生成端对关卡好坏无感知）。

### 学习曲线（48lfz699，15 点）
```
step        0   482   803  1125  1446  1928  2089  2250
Middle      0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   ← 简单关 ~500step 秒会
Nav1        0   0.91  0.95  1.0   1.0   1.0   1.0   1.0   ← 同上
Nav2        0   0     0.50  0.10  0.70  0.00  0.00  1.0   ← 长程关：剧烈震荡反复丢失
Nav3        0   0     0.50  1.0   0.50  0.00  0.00  1.0   ← 同上
sampled_all 0.02 0.95 0.99  0.98  0.98  0.99  0.99  0.99  ← 主测试集早早饱和
```
learn_mean 0.25→0.11（student 变强后随机关里"有价值的"越来越稀，边际收益递减）；
p_std 全程 0.46（健康双峰，未饱和坍缩）。

### Baseline 的三个缺陷（均指向"随机低效"——idea 攻击点）

| # | 缺陷 | 量化证据 | 可攻击指标 |
|---|---|---|---|
| **1** | **能力反复丢失（不稳定）** | Nav2 首次学会(step1250)后 **40% eval 时刻又掉回 0**；Nav3 **35%**；各 2 次大滑坡；step1928-2089 双双归零 | "掉回0"次数、能力曲线方差 |
| **2** | **算力浪费在简单关（撞大运供给）** | Middle/Nav1 step~500 即满分，**后续 ~75% 训练量对其纯浪费**；随机关常起终点贴一起/大片空白（零学习价值）；PLR 只能从"随机已掷出的"里筛，**掷不出长程关就永远筛不到** | 达到 sing≥0.9 的 step 数 |
| **3** | **长程关学得晚、学得勉强** | Nav2 step1250 才首解（全程 60%）、Nav3 step1100；简单关 step500 已满；长程能力获得被推迟到后半段且始终 0↔1 震荡 never 稳定 | Nav2/3 首次解开 step |

### 诚实记账：baseline 的"好"也真实
sampled_all step482 即 0.95、全程 0.99；末次 5 关全 1.0（sing=1.0）。
**它最终全解了，只是过程颠簸、效率低。**baseline 的病是"过程低效但终点到达"。

### ★bug 修复后 baseline 实测复刻（2026-06-26, mild-sponge-147 / 8lh6h99y, 朝向+碎图 bug 修复后, inj=false, 3e8, seed0）
重跑 baseline 拿同口径对照素材（产 maze dump + 裸 p log）。**这是 generator 必须对标的真实基准。**

**5 关测试（末值 / 首解 step）**：BlankTest 1.0/s100、MiddleTest 1.0/s250、Nav1 1.0/s250、**Nav2 0.90/s450**、**Nav3 0.90/s350**、overall **0.96**。比文档原记录强很多（原说 Nav2 首解 s1250，这次 **s450**——bug 修复或 seed 效应）。

**★能力滑落细粒度（用户要求改口径：看回撤/方差非"掉回0"次数）**——"掉回0=0"严重掩盖了不稳定性：
| 关 | 首解 | 下滑次数 | 累计下滑 | 单次最大下滑 | **最大回撤** | 末值 | 方差 |
|---|---|---|---|---|---|---|---|
| Nav2 | s450 | 7 | 2.11 | 0.80 | **0.80** | 0.90 | 0.054 |
| Nav3 | s350 | 14 | 2.11 | 0.40 | 0.57 | 0.90 | 0.062 |
| overall | s100 | 14 | 0.66 | 0.18 | 0.18 | 0.96 | 0.057 |

**修正"baseline 全程稳定"的错误印象**：Nav2 经历过 **0.80 最大回撤**（从≈1.0 一度暴跌到 0.2，只是没正好到 0 故"掉0次数"漏掉）、Nav3 震荡 14 次。**baseline 是"高末值但高方差、能力反复大幅丢失"**。→ 这反而给 generator 卖点指路：**末值不必超 0.90，只要"接近 baseline 末值 + 方差/回撤更小"稳定性卖点即成立**（前提仍是先让 Nav2 翻起）。

**★裸 p 分布（决定性，坐实 baseline 赢的机制）**：baseline 每帧海选 5000 关。step350 时 p≈0 降到 33%、p≈1 升到 46%、**但中间区 p=0.3~0.9 连续散布上千关**（p≈0.5 有 82、p≈0.66 有 58、p≈0.8 有 126…）；中后期两极化但**可学区始终保留**（step1250 仍有 p≈0.5 的 26 关 + p=0.66~0.9 的 ~90 关）。
→ **baseline 赢的本质 = 大基数(5000) + 随机连续多样性**：learnable_frac 虽只 0.6%，但 5000×0.6%≈30 关落可学区，且中间区连续覆盖，PLR 总能挑到难度合适的关喂 student（这就是它 step1250 学会 Nav2 的素材来源）。**generator 只造 128 关且塌缩堆两极→可学区 0 关→PLR 无可挑**。差距不在单张关样子（图骗人，baseline/generator 单张关看着一样），在**关分布是否覆盖可学区**。素材见 mazes/baseline-dump/（三档图 + p 分桶）。

### 对 idea 的硬约束（必须正视）
当前 generator 方案在三个攻击点上**目前都更差**：quota0.5 Nav2 全程 0、Nav3 末期归零，
连 baseline 末值 1.0 都没到。即：
> **baseline = "过程低效但终点到达"；现 idea = "过程没更好、终点还没到"。**
> 靶子（随机低效）选对了，但当前的箭（euc 定向）射偏——定向到"对角拉直线"这个对 test 无用
> 的方向，反比随机的多样性更糟（euc 的新型 reward-hack：清墙+拉直线，非堆墙）。

立论要立住：不是证明"方向对"（方向已对），而是让 generator **真正定向到 test 需要的难关结构**
（稳定、定向、早期的长程难关供给）。euc 给的是"几何远但拓扑简单"，没补到点上。

---

## 备手方向：随机多样性 + 定向增强（混合课程）

> 用户 2026-06-24 定为**备手**（主攻"治坍缩"）。记录于此防遗忘。

**动机**：baseline 的"好"恰恰来自它**不优化**——DR 随机天然保持结构多样性，所以 Nav2 这种
靠多样性才能覆盖的关它能解（0.7~1.0），而所有 generator 方案 Nav2 全 0。但随机的代价是
低效（75% 算力喂简单关、长程关学得晚、能力反复丢失）。

**思路**：不让 generator **取代**随机，而是在 DR 随机池基础上**加注**定向难关：
```
buffer = α·DR随机关  +  (1-α)·generator定向关
```
- 随机部分保结构多样性（守住 Nav2）；
- generator 部分补稀缺长程关（攻痛点2/3：浪费 + 学得晚）。
- 卖点定位 = **样本效率 + 稳定性**（更快达标、能力不丢失），**不赌终点超越** baseline 末值。

**为什么可能比纯 generator 好**：纯 generator 100% 注入 → 坍缩方向独占 buffer → 多样性全失。
混合保留 α 比例的随机，等于给坍缩一个"多样性下界"，generator 再坍也坍不掉那 α 部分。

**可量化卖点指标**（对照 baseline 三缺陷）：
- 达到 sing≥0.9 的 step 数（攻痛点2）；
- Nav2/3 "掉回0" 次数 / 能力曲线方差（攻痛点1）；
- Nav2/3 首次解开 step（攻痛点3）。

**工程量：低**。注入路径已存在，只需把"全替换"改成"按 α 混合"。
**风险**：α 是超参，太大≈baseline（无增益）、太小≈纯 generator（坍缩）；需扫 α∈{0.3,0.5,0.7}。
**与主线关系**：正交。若"治坍缩"成功，可与混合叠加（定向部分不坍 + 随机部分保底）。

---

## 备手方向：LLM 多智能体讨论 + 工具调用 造关卡（2026-06-26 记录，仅存档未展开）

> 用户灵感来源：用 HeyGen + LLM 对话式设计制作出高质量 AI 视频，联想到"一群 LLM 讨论如何设计关卡，然后调用工具具体制作关卡"。

**核心想法**：多个 LLM agent 协作讨论关卡设计意图（"造一个需要绕过房间的长程导航关"），再调用工具（写 0/1 墙矩阵 + 起终点）落地成具体关卡。

**★直接当训练循环内 generator——不契合（两条硬约束，记录防以后重蹈）**：
1. **吞吐量差 5–6 个数量级**：当前训练每 eval 造 128 关 / 海选 5000 关 × 2289 eval = 单 run 造关几十万~上千万次，全在 JAX GPU 上 jit/vmap（<1s/批）。LLM 调用秒级、不可微、进不了 JAX 训练图。几十万次 LLM 调用成本/时间完全不可行。这是范式不匹配，非工程优化能解。
2. **jaxnav 无语义供 LLM 发挥**：关卡 = 纯几何墙矩阵 + 起终点，没有"诱饵/陷阱/叙事"等语义，LLM 的世界知识优势用不上。LLM 造关范式真正发光在有语义的开放环境（Crafter/Minecraft/文字迷宫）。

**可行的接入点（若将来要捡起，从这三个选，别直接塞进训练循环）**：
- (a) **离线造固定高质量关池**：不进训练循环，LLM 预先造几百关对准 Nav2/3 走廊拓扑当固定课程/test。解决吞吐量（只造一次），但与 UED 核心（student 共演化）脱节，退化成静态课程。
- (b) **LLM 当低频高层调度器**：保留 GPU 上 PCGRL 高频造关，LLM 每 N eval 一次看指标讨论后调 generator 高层参数（课程阈值/信号权重/多样性旋钮）。LLM 在循环外当"教练"不在循环内造关，避开吞吐量。最贴合当前架构。
- (c) **换有语义的开放环境**：放弃当前 jaxnav 积累，换赛道让 LLM 语义优势发挥。代价大。

**Novelty 待核（若推进先做）**：LLM-for-UED/RL-课程生成 学界刚起步，需 deep-research 摸清谁做过 LLM 生成训练关卡/课程、怎么解吞吐量、空白在哪。

**当前定位**：纯备手存档。主线仍是治生成端塌缩（让 generator 关分布覆盖可学区）。此 idea 与主线正交，且 (b) 可与主线叠加（LLM 调的就是治塌缩的那些旋钮）。

---

## 主攻方向：治坍缩（collapse）—— 根因与候选解（2026-06-24）

### 坍缩是所有 non-baseline 方案的共性病（铁证）

全部 7 个 generator 信号家族（pvl / anchored / euc / euc+ALP-bid / euc+ALP-quota）实测：
1. **Nav2 全军覆没（全 0）**，唯独 baseline（不优化的随机）能解 Nav2（0.7~1.0）；
2. **单 seed "好结果"全是噪声**：s4p2-lambda 同配置 3 seed，Nav3 = **1.0 / 0.6 / 0.0**；
3. **没有任何 arm 接近 baseline sing≈0.95**（最好 0.80 且不稳）。

**注入关坍缩指标对照**（start→end）：

| run | fill_hi_frac | euc_mean | 解读 |
|---|---|---|---|
| quota0.5（失败） | 0.46 → **0.08** | 4.6 → **6.1** | 清空高墙关 + 对角拉直线 |
| **d3scan-euc-fac5（表里"最好"）** | 0.40 → **0.02** | 5.8 → **6.6** | **坍缩方向完全一致** |

→ **"最好"的 arm 和失败的 arm 是同一个坍缩**，它 Nav3=1.0 纯属单 seed 运气。
**根因不是信号选错，是"单标量定向优化"这个范式本身必然坍缩**：generator 朝任何单一标量
爬坡，最省力的极大化方式就是坍到一个廉价模式（euc→对角空房间、pvl/diff→堆墙、
anchored→堆墙+绕路），丢掉一切与该标量无关的结构多样性（含 Nav2 所需结构）。

### 治坍缩的候选解（按"是否触及根因"排序）

| 解 | 机制 | 是否触根因 | 工程量 | 风险 |
|---|---|---|---|---|
| **A. 多目标/Pareto** | 同时优化 euc⊥fill⊥拓扑多个不相关目标，buffer 按 Pareto 前沿采样，单方向爬不满全部 | **是**——让单方向极大化无法独占 | 高 | 目标若仍可被同一捷径满足则无效；Pareto 实现复杂 |
| **B. 多样性正则 / quality-diversity** | 显式奖励 buffer 内关卡的结构差异（如 MAP-Elites 按 fill×euc 网格存档，每格留精英），强制覆盖描述符空间 | **是**——直接把"多样性"写进目标 | 中-高 | 描述符要人选（fill/euc/拓扑），选错=覆盖错维度 |
| **C. 混合随机（=备手）** | α 比例随机保多样性下界 | 部分——绕过而非治坍缩 | 低 | α 超参；非"治"是"稀释" |
| **D. 信号换抗坍缩量** | euc→摆墙能影响的测地/必经走廊数（让 generator 没法靠"清墙"刷分） | 部分——堵一个捷径，可能冒出新捷径 | 低 | 文档已记 geo 可被"堆墙绕路"操纵；治标 |

### 关键判断（治坍缩前必须想清）
1. **A/B 才真正触及根因**（单标量→多目标/多样性），C/D 是绕过或治标。用户要"治坍缩"→应锁 A 或 B。
2. **B（quality-diversity / MAP-Elites 思路）与你的 idea 框架最契合**：你要攻击"随机低效"，
   QD 正是"既要定向（每格选精英）又要多样性（覆盖全网格）"——**它本质就是"有结构的随机"**，
   理论上同时解决 baseline 的低效（定向选精英）和 generator 的坍缩（强制铺满网格）。
3. **写任何训练代码前先离线判**：用现成 1400 关 sweep 数据，模拟"若按 MAP-Elites(fill×euc 网格)
   选 buffer"得到的关卡分布，对比纯 euc top-k 的坍缩分布——**离线就能看 QD 是否真能保住多样性**，
   不烧 GPU。同 memory 诊断方法论（先离线验区分度再训练）。

### 待补实验（确认坍缩普遍性，零额外训练）
- 给 quota0(COEF=0) + s4p2-lambda 3seed 跑同样 fill_hi_frac/euc 坍缩曲线 → 坐实"坍缩与信号无关、是范式病"。

---

## ★ QD 离线模拟的意外结果 —— 推翻多个诊断（2026-06-24，关键修正）

> 起因：要为 QD(MAP-Elites, fill×euc 轴)做离线模拟，先提取 test 关描述符 + 对比 DR/generator
> 分布。结果**连续证伪了三个先前诊断**，把真因指向完全不同的方向。诚实记录推翻链。

### 实测数据（全部离线、零 GPU）

**1. test 关描述符**（`make_jaxnav_singleton_collection("multi")` 实例化提取）：
| test 关 | map_shape | fill | euc | student |
|---|---|---|---|---|
| SingleNav1 | 7×11 | 0.597 | 8.94 | 解(1.0) |
| **SingleNav2** | 7×11 | **0.597** | 8.06 | **全 0** |
| SingleNav3 | 7×11 | 0.597 | 8.59 | 时好时坏 |

**2. DR 随机关分布**（`JaxNav(num_agents=1).reset`，N=500，baseline 的关卡来源，11×11）：
- fill: 0.33~**0.52**（median 0.42），**fill>0.55 的关 = 0 个（0%）**
- euc: median 3.58，euc>8 仅 2%
- **落在 SingleNav2 区(fill>0.55 & euc>7)的 DR 关 = 0 个（0%）**

**3. generator 注入池分布**（sim_formA_raw，1400 关，11×11）：
- fill: 20% 在 0.55 以上（DR 是 0%）、euc>8 占 6.8%
- → **generator 造的关比 DR 更难（更高 fill、更长 euc）**

### 推翻链（先前结论 → 证据 → 推翻）

| # | 先前诊断 | 证伪证据 | 状态 |
|---|---|---|---|
| 1 | **尺寸错配**（test 7×11，generator 造 11×11，造不出） | baseline **同样**只造 11×11，却泛化到 7×11 成功（用户点破） | ❌ 推翻 |
| 2 | **分布覆盖不到 test 区**（generator 坍缩丢高fill关 → Nav2 失败） | DR(baseline) **同样** 0% 覆盖 fill>0.55 区，Nav2 却能解 | ❌ 推翻 |
| 3 | **euc 信号把图清空/对角拉直线**（reward-hack 新形态） | 重看末期迷宫图，generator 关墙也密也碎，与 DR 结构肉眼相当；且 generator fill/euc 实际比 DR **更高**非更低 | ❌ 夸大，撤回 |
| 4 | **造了 student 解不开的不可解关** | `gen_n_incomplete` 全程 0（valid_path_check 生效，注入关都可解） | ❌ 排除 |

### 新立诊断（当前证据支持，但仍需印证）

**baseline 和 generator 的训练关在 fill×euc 上近似同分布（都 median fill≈0.42），但 generator
的关系统性更难（fill/euc 上探更高）。baseline 喂大量温和关→扎实打基础→泛化到难关；
generator 把课程推难推偏→student 基础没打牢→泛化更差（Nav2 全 0）。**

→ 这是 UED 经典 **curriculum-too-hard**：generator 自以为"定向供给难关"在提效，实则破坏了
baseline"大量温和多样关打基础"的健康路径。不是关太简单，是**关太难/太偏、挤掉了基础训练**。

**这同时改写两个关键判断：**
1. **fill×euc 作 QD 描述符轴 = 选错**：实测它对 baseline/generator **无判别力**（两者近似同分布），
   且 generator 在该空间更激进，QD 铺满只会铺出更多过难关。**用户最担心的"轴选错"风险 = 实锤。**
2. **真正该控的轴是 student 当前胜率 p / learning progress**：保证课程含足够"p≈0.5 当前可学"的关，
   而非一味高 euc/高 fill。difficulty=−(p−0.5)² 方向其实对，只是被 euc/cenie 压住了——
   呼应 memory `alp-is-curation-not-generation`（LP/进步信号），但**用途要从"加难"转成"控难度在可学区"**。

### 仍开放的问题（不下定论，待印证）
- **假设B 未完成**：baseline 学会 Nav2 的那段(step1100~1250)具体在练什么 p 的关？需取那段 PLR
  buffer 的 p 分布。若证实"baseline 那时大量 p≈0.5 关、generator 大量 p≈0 关"→ 实锤 curriculum-too-hard。
- generator 注入关的 **student 实时 p 分布**（不是池子的离线 p，是训练中那批）——是否大量 p≈0（太难）。
- 结构维度（fill 相同时墙的拓扑）是否仍有隐藏差异——目前肉眼看接近，但未定量。

### 对 QD 方案的修正结论
- **不要用 fill×euc 当轴**（无判别力 + 鼓励更难）。
- QD 思想仍可用，但**描述符轴应含 student 胜率 p / learning progress**，目标是"覆盖各难度档、
  保证可学区不空"，而非"覆盖各几何形态"。
- **下一步应先做假设B 印证**（零 GPU，取 baseline 那段 buffer p 分布），确认真因=curriculum-too-hard
  后，再重新设计 QD 轴。**避免在 fill×euc 上写任何 QD 代码（已证轴错）。**

---

## ★★ 假设B 验证完成 + "堆墙是否真因"裁决（2026-06-24，决定性）

> 用户质疑：anchored/euc 等"弱于 baseline"的方法，真因是否其实是"前期关太难"，而非先前诊断的
> "堆墙作弊/拉距离"？数据裁决：**用户基本正确——前期课程失准是主因，堆墙/拉距离是表象。**

### 证据1：前期 sampled_win_rate 爬升（step 0-600）
```
方法           step100  step150  step200  到0.85需
baseline        0.46     0.71    0.76    ~step300   ← 最快
anchored        0.18     0.51    0.75    ~step400
PVL-lambda      0.25     0.61    0.75    ~step500+
euc3sig         0.10     0.24    0.47    ~step600   ← 最慢
quota0.5        0.09     0.27    0.68    ~step550
```
**所有 generator 前期都比 baseline 慢；越激进的信号(euc/quota)前期越慢(慢约5×)。**
失败发生在**前期**——看末期注入关长什么样根本看不到真因（先前方法论缺陷）。

### 证据2：learnability buffer 分布演化（决定性）
**口径警告**：baseline learnability ∈[0,0.25]（标准 `p(1-p)`，0.25=p≈0.5 最可学）；
generator learnability ∈[0,3.7]（euc+cenie 加权和，**量纲不同，数值不可直接比**，只看分布形状）。

```
baseline (p(1-p) 口径):
  step100: 100关 全部=0.25      ← 前期buffer 100% 是 p≈0.5 完美可学关
  step600: 52%还在0.25 +35%→0.22
  step1200: 散开 0.09~0.25      ← student变强,关渐被学会
  step2250: 48%→~0.00(已学会) +28%还在0.25  ← 始终维持一批可学关
→ 教科书式 PLR 课程:前期锁定可学区,健康衰减

generator quota0.5 (euc+cenie 口径):
  step100: 48关 分散0.07~3.1, mean0.595, 右偏长尾
  全程: 按信号强度铺开成长尾, 非集中在"当前可学"窄峰
→ 追求"信号强(高euc/cenie)"而非"当前可学",前期就没把student放在可学区
```

### 裁决表
| 假设 | 证据 | 裁决 |
|---|---|---|
| **真因=前期课程没对准 student 可学区(curriculum 失准)** | baseline 前期 buffer 100% p≈0.5；generator 按信号铺开非按可学；generator 前期爬升慢5× | ✅ **主因** |
| 次因=堆墙作弊/盲目拉距离 | fill/euc 确被推高,但这是"信号铺开"的表现;且 fill×euc 与 baseline 同分布、与失败无判别力 | ⚠️ **表象/次要** |

**核心结论（改写全部先前诊断）**：
> 「堆墙作弊」「盲目拉开起终点」**不是独立真因，是症状**。真因是 generator 用 euc/cenie/anchored
> 这些信号选关，而**这些信号没有对准 student 当前的"可学区(p≈0.5)"**——baseline 用 `p(1-p)`
> 精确锁定可学关，generator 把 buffer 铺成信号长尾，前期就没让 student 在可学区训练，基础打不牢。

### 对 idea 的最终启示
1. **不是"信号选错该换哪个估计器"，是"选关准则该回到 learnability/可学区"**。difficulty=`−(p−0.5)²`
   和 SFL 的 `p(1−p)` 才是对准可学区的信号；euc/cenie/anchored 是"加难度/加新颖"信号，
   把它们当主选择准则 → 偏离可学区 → curriculum 失准。
2. **idea 若要立住**：异质信号的正确用途可能是**在"可学区内"再做区分/塑形**（如可学关中优先长程），
   而非替代 `p(1−p)` 当主准则。即 learnability 把关(保前期不失准) + 异质信号微调(补盲区)。
3. **下一步零-GPU 可做**：离线在 1400 关上模拟"`p(1-p)` 选 top-k" vs "euc 选 top-k"的 buffer
   p 分布,直接看后者是否前期就偏离 p≈0.5。这能在写训练代码前确认"learnability 把关"方向。

---

## ⚠️ 数据可信度更正 —— sim_alp_raw.npz 作废 + 48关来历钉死（2026-06-24）

> 用户三点纠正（全部成立），触发对前面"可学区空/generator造的关太简单"结论的回查。结论：**那些结论建立在被误用的数据上，全部作废。**

### 1. sim_alp_raw.npz 的真实来历（查清）
- **写它的脚本 = `sim_alp_bucket.py` 在 `SIM_DRY=1` 模式**（FILLS=[0.2,0.5]、PER_FILL=30→48关）。
- 这 48 关是脚本**手动 fill-sweep 撒的随机关**（fill_label 只有 {0.2,0.5}），用 4 个 ckpt 评 p。
- **完全不是 generator 训练时实时造的关**——它是测 ALP 信号本身的离线素材。
- 同理 `sim_alp.py` 用的是 **5 个命名 test 关**（写 sim_alp_testset.npz），也非 generator 关。
- `sim_formA_raw.npz` / `sim_antag_raw.npz` 是 `sim_formA.py` / `sim_antagonist.py` 撒的关（actual_fill 0.33~0.78），同样是离线 sweep 素材，**非 generator 实时关**。

### 2. 据此作废的结论
- ❌ "generator 造的关 85% 已被 student 学会(p≥0.95)、太简单" —— **错**：那是人工 fill-sweep 随机关对强 student 简单，与 generator 无关。
- ❌ "可学区里压根没有关、learnability 把门会致 buffer 全空" —— **撤回**（建立在上条错误数据）。
- ⚠️ 先前所有用 sim_*_raw.npz 推 generator 行为的分析，都须重新审视——这些 npz 只能说明"人工撒的关在各 fill 档的信号值"，**不能代表 generator 实时造什么关**。

### 3. p(1-p) 坍缩是老问题（非新发现）
[[sfl-zero-variance-pq-bimodal]] 16天前已坐实：**maze 上** p(1-p) 在成功率两极(80.7% p=1/19.1% p=0)下数学塌缩，非bug。
[[heterogeneous-score-convergence-finding]] §4.5.5 预警 SFL 依赖在线 student 找 p≈0.5、**适用连续回报环境**。
**⚠️ 关键未确认项**：那是 maze 结论；**jaxnav（连续避障）上 p 是否同样两极塌缩，尚无实测**——这是后续要直接验的（取 generator run 注入关的实时 p 分布，非用离线 sweep npz）。

### 4. 现在唯一可信的证据（真实 generator 训练数据，不依赖任何 npz）
| 证据 | 来源 | 指向 |
|---|---|---|
| generator step100 墙就很密、确实更难 | **用户+我看图**（mazes/ 文件夹） | 前期就造难关 |
| generator fill_mean 前期≈0.49（DR median 0.42） | wandb 实时 | 比随机更密 |
| generator 前期 sampled 爬升慢 5×（0.09 vs 0.46） | wandb 实时 | student 基础打不牢 |

三条互相印证 → **generator 前期就造太难的关（curriculum-too-hard）**，用户最初质疑成立，且**不依赖任何作废的 npz**。

### 5. 下一步必须做的（取真实 generator 注入关数据，非离线 sweep）
- 从 generator run 取**实时注入关的 per-level student p 分布**（训练中那批关，非人工撒的）——
  确认 jaxnav 上 generator 注入关的 p 到底两极塌缩 vs 集中可学区 vs 偏难。这是判 curriculum-too-hard
  和设计修正方案的**唯一可靠依据**。wandb 里若没存 per-level p，需在代码侧补 log 或离线用 ckpt 重跑注入关。

---

## ★★★ injp 离线诊断结果 + 护栏方案定型（2026-06-25，决定性）

### injp 结果：jaxnav 上 generator 注入关 p **两极塌缩，可学区空**（job 3416626）
忠实复现 quota0.5（denim-hill-120）的注入关，对 step600（前期）/ step2250（末期）各取 48 关 per-level student p：
```
step600 (前期): 可学区(0.2≤p≤0.8)=0%  两极(p<0.05 或 p>0.95)=100%
                p≈0(打死) 73% | p≈1(秒会) 27%   ← 前期 73% 直接必死，student 学不动
step2250(末期): 可学区=0%  两极=100%
                p≈0 40% | p≈1 60%             ← 末期涌向另一极（秒会）
```
→ **§5 待确认项裁决：jaxnav 上 p 同样两极塌缩**（与 maze 一致，[[sfl-zero-variance-pq-bimodal]]）。
→ **curriculum-too-hard 实锤**：前期 73% 必死，注入关从无落在可学区。

### 代码级真因（读 pcgrl_generator.py 确认）
`terminal_reward_euc` 是**纯几何量**（euc=起终点欧氏距离），**不看 student p**——只奖励"起终点远"，
generator 就堆墙拉远起终点，根本不管 student 死活。这是 curriculum-too-hard 的**代码级根因**：
reward 没有任何"对准可学区"的项。换 PVL 也治不了（证据1：PVL-lambda 前期同样慢到 step500+，
同属"加难度信号"，文档证据2 已证非"换信号"问题，是"选关准则没对准 p≈0.5"问题）。

### 护栏方案（主线 = 选项一：可学性门调制）
**改 `terminal_reward_euc` 的 reward：base 信号 × 可学性门，双边对准可学区。**
```
reward = euc * gate(p)
  gate(p) = exp(-((p - 0.5) / sigma)^2)    # p≈0.5 时 gate≈1，两极 gate→0
```
- **euc**（或 cenie）保留 → novelty 不丢、异质信号还在；
- **gate(p)** 把 reward 在 p≈0.5 放大、两极压低 → generator 被推着造"student 学得动"的关 → 对准可学区（治本）；
- **p=succ 现成**：difficulty 路径已跑 student rollout，`info["success_rate"]` 直接取，**零额外算力**；
- 为什么不是单边：injp 末期 60% 涌向 p≈1（秒会），单边只挡 p≈0，治不了"太简单"那半。双边才对准 p≈0.5。
- **塌缩风险（已知 tradeoff）**：gate 也是 (p−0.5)² 家族，p 两极时梯度弱。靠 base 信号（euc 始终 finite）
  提供两极处的方向，gate 只做"可学区加权"，缓解但不消除。上线后看 per-level p log 是否真把 p 拉回中部。

### per-level p log（与护栏同改，补先前漏做的）
`all_signals_on_levels` 已返回 `succ`（per-level p），从 `get_generator_set` 透到 metrics → wandb.log
（直方图 + 可学区占比 + 两极占比）。这是**护栏是否生效的仪表盘**（不是再诊断）：上线后看 p 分布
是否从"100% 两极"向"可学区填充"移动。先前 [725] 答应加但只写了离线脚本（injp），训练代码漏做，此次补上。

### 备选方案（选项二：单边难度护栏，hold）
```
reward = euc - P * relu(p_floor - p)    # 只惩罚 p<p_floor（≈0.05~0.1）的"太难"关
```
- 最贴最初"只解决关卡太难"的表述；**100% 不吃 p(1-p) 塌缩**（单边不依赖中间梯度）。
- 缺点：只挡"太难"，被挡的关会涌向 p≈1（injp 末期 60% 秒会佐证），治不了"没对准可学区"的另一半。
- **触发条件**：若选项一上线后 per-level p log 显示 gate 因塌缩失效（p 仍卡两极、未向中部移动），
  退选项二作为更稳的单边止血。

### 升级方向（选项三：课程版 gate，p_center 随时间挪，hold 待固定 gate 结果）
> 用户想法：teacher 前期造简单关、后期造难关（课程学习）。图证据（mazes/）：generator step100
> 墙就密集铺满、learnability 0.61~3.58；baseline step100 墙参差、learnability 全 0.25——肉眼实证
> generator 前期就太难。**关键**：课程的"简单/难"必须用 student p 度量（fill/euc 已证不可靠：
> baseline 墙密关也可 p=0.5；euc 拉远是堆墙捷径）。用 p 度量 → 课程 = gate 中心随时间移动。
```
reward = euc * exp(-((p - p_center(t)) / sigma)^2)
  p_center(t): 0.7(前期,造student大概率赢=简单) → 0.5(中期) → 0.3(后期,偶尔赢=难)
```
- 比固定 p=0.5 gate 更贴"前期简单→后期难"直觉；难度由 student p 锚定（可靠），时间表只控门往哪移；
  绕开定义墙密度/euc 难度轴的坑（那两轴已证不可靠）。
- **为什么先不做**：固定 p=0.5 gate（job 3422090）刚启动。先看它能否把 gen_p_learnable_frac
  从 injp 的 0% 拉起来——能爬证明 gate 机制有效，再加 p_center(t) 时间表升级为课程；不能爬说明
  gate 本身有问题（塌缩），加时间表也白搭。**一次只动一个变量**，避免混淆功劳/归因。
- 时间表实现需把训练进度（如 update_step / TOTAL_UPDATES）传进 terminal_reward_euc（目前签名没有，
  需顺 gen_train_iter → compute_terminal_reward 透一个 progress∈[0,1]）。

---

## ★★★★ gate on/off 对照 + baseline p 验证：诊断彻底改写（2026-06-25，决定性）

### gate 完全无效（on/off 对照铁证）
3422586(ON, euc×gate) vs 3422635(OFF, 纯euc)，唯一差 GEN_LEARNGATE，同 seed/参：
```
              gen_p_learnable_frac(末)  gen_p_extreme_frac(末)  fill_mean  fill_hi_frac
ON  (euc×gate)      0.001                    0.999              0.533       0.521
OFF (纯euc)         0.001                    0.998              0.441       0.188
```
两条曲线**完全重合** → gate 完全无效（乘性 gate 在 p 全两极时 gate≈0、梯度消失=塌缩兑现）。
且 **gate ON 反而堆更多墙**（fill 0.53 vs 0.44，堆墙关 52% vs 19%）——gate 压可学信号后 euc 失制衡。

### baseline p 验证：可学关存在，p(1-p) 不塌缩（推翻旧判断）
baseline(s4p2-base 48lfz699) 海选关 learnability=p(1-p) 分布：**可学 p∈0.2~0.8 = 45%**，两极 48%
（双峰 [0.25]×28 有一批 p≈0.5）。→ **p(1-p) 工作正常，jaxnav 上可学关存在**。
之前以为"p(1-p) 塌缩"是被 generator 坏 pool 误导（[[sfl-zero-variance-pq-bimodal]] 是 maze 结论）。

### 决定性诊断：generator 生成分布塌到两极（同口径对照）
| | 可学 p∈0.2~0.8 | 两极 p≈0/1 | 分布 |
|---|---|---|---|
| generator pool (ON)  | **0%** | 100% | [p≈0.01]×21, [p≈0.99]×107 |
| generator pool (OFF) | **0%** | 100% | [p≈0.01]×26, [p≈0.99]×102 |
| baseline 海选关       | **45%** | 48% | 双峰有 p≈0.5 |
gen_p_perlevel 口径=整个 pool(M=128 注入前) → **pool 里就没有可学关，非 auction 选不中**。
ON run pool **84% 是 p≈0.99 student 秒会(太简单!)** → 与"堆墙造太难"印象相反，主要是**太简单**。

**根因改写**：euc(奖起终点远)+cenie(奖新颖) 驱动 PPO 把 generator 训成只会造极端关的策略，
丧失造中等难度关能力。baseline 不训 generator(随机海选)反保留多样性→捞到 45% 可学关。
→ 问题在 **generator 生成端**，非选关准则、非几何课程（那些假设 pool 有可学关可选/可塑）。

### 当前实验：纯 difficulty 驱动 generator（job 3426176, group=pure-difficulty）
配置：`GEN_ESTIMATOR_IDS=[difficulty]`(N=1) + `GEN_AUCTION_LAMBDA=none` + `AUCTION_USE_CENIE=false`。
**euc/cenie 完全排除**，generator 训练 reward = 纯 `-(p-0.5)²`（对准可学区），选关也 difficulty 自评。
（已在 fallback 路径补 gen_p_* log，N=1/auction=none 也能记 pool p。）

**这不是 baseline**：baseline 是随机海选被动捞 p≈0.5；本 run 是 generator **主动学着造** p≈0.5 关。
用户洞察：要瞄准的正是"随机生成低效率、student 学习不够好"——若主动生成能恢复可学性，则证明
"主动造关"能做随机海选做的事，且更高效/针对性，是加 novelty 的地基。

**判据 + 后续**：
1. 若 `gen_p_learnable_frac` 从 ≈0 恢复（pool 离开两极）→ 坐实 euc/cenie 是塌缩元凶，difficulty 能造可学关。
2. 然后拉 baseline + 本 run 的注入迷宫（存 mazes/），**对比迷宫结构特征**：主动生成 vs 随机海选造的
   可学关有何差异、本 run 对"随机生成低效"解决得如何。
3. 据此再决定**怎么把 novelty 加回来**（在不破坏可学区前提下）——课程 / 可学区内塑形 / 其他。
4. 若纯 difficulty 下 pool 仍塌两极 → 塌缩是更深层(PPO/环境)问题，需另查。

---

## ★★★★★ 决定性纠正：baseline 真实 p 也两极，真差距是结构多样性（2026-06-25）

> 给 baseline 随机海选路径(get_learnability_set)加了真实 per-level p 透出（p_env 现成，零额外算力），
> basep(bf4w9c1z) 与 anchoredPVL(ancp aqbbyx86) 同口径真实 p 对比。**用户坚持要直接 p、不反推 0.25——
> 拿到真相后推翻了所有先前基于"可学性/p 分布"的诊断。**

### 同口径真实 p 对比（不是信号分、不是占位分）
| | baseline海选 | anchoredPVL | pure-difficulty |
|---|---|---|---|
| gen_p_learnable_frac(p∈0.2~0.8) | 0.101→0.009 | 0.005→0.001 | ≈0 全程 |
| gen_p_extreme_frac(p≈0/1) | 0.852→0.987 | 0.993→0.999 | ≈1 全程 |
| test Nav1/2/3 | **1 / 1 / 1** ✅ | 0.83 / 0 / 0.55 | 0.95 / 0 / 0 |

### 推翻"learnability=0.25 ⟺ p=0.5"（用户推断 + 先前文档判断都错）
**baseline buffer 真实 p 也 85%~99% 两极，可学只 10%→1%，根本不锁 p≈0.5。** 图上/文档里的"全 0.25"
**= PLR 占位分**（新加入未评估关的默认 score），**不是真实 p(1-p)**。先前 §假设B "baseline 海选 45% 可学"
作废（那是占位分直方图）。口径警告升级：**连 baseline 的 learnability 标注都是占位分骗人，只有从 rollout
取的 gen_p_* 是真 p。**

### 真相（改写所有诊断）：差距 = 结构多样性，非 p 分布
baseline 与 generator 的 pool p 分布几乎一样（都两极）→ **差距不在 p**。在结构多样性：
- baseline 随机海选 → 结构千变万化 → **偶尔掷出 Nav2/3 走廊拓扑** → PLR 挑出碰巧可学的 → 学会；
- generator 训练后 → 塌缩到单一造关模式 → **永远掷不出那种结构** → 学不会。
**baseline 赢不是锁可学区（它没锁），是随机多样性偶尔命中对的结构。** 这是 PLR "只能从随机已掷出的里筛"
天花板的另一面：随机至少掷出多样结构，generator 塌缩后只造一种。

### anchoredPVL "越来越简单" 的机制（用户问：是 student 不会被慢反映吗？）
**方向对但机制反**：是 **student 学会** 被反映，不是 student 不会。anchored=PVL×测地，PVL=student value loss。
student 变强 → 学会的关 PVL→0 → generator 在这些关拿不到 reward → 被迫换关，但"还能让 PVL 高"的关越来越少
→ 退化造简单关（p_mean 0.45→0.87）或刷无解关（gen_n_incomplete 0→6，student 永远完不成=PVL 持续高=刷分）。
**根因：PVL/p 这类依赖 student 瞬时状态的信号，在 student 稳定后归零失向 → generator 塌简单关。**
与 difficulty "单极无梯度" 同病不同表现。**再次指向：别用依赖 student 瞬时状态的 p/PVL 信号。**

### test 集真相（用户抓出）
make_jaxnav_singleton_collection("single") 渲染 5 关（mazes/test_set/）：**Nav2 == Nav3 逐格同图**
（map_data 完全相同，只起点差 0.6 格），均 7×11。BlankTest/MiddleTest 7×7。**generator 训练 11×11，
test 7×11/7×7，尺寸不匹配（OOD）。** Nav2/Nav3 本质是一个走廊迷宫（#.#.# 交替窄通道），非两个独立难关。

### 方向彻底拨正
- **放弃纯 p 信号路线**（gate / difficulty / 可学性门 / p_center 课程 全在 p 分布做文章，但 p 两极也能赢）。
- 真问题 = **结构多样性 / 定向结构供给**：让 generator 定向造出"随机海选很少掷出的特定结构（走廊拓扑）难关"，
  而非碰运气。这才是 generator 相对随机的价值，呼应用户"随机生成低效"洞察（随机靠撞大运命中结构，
  generator 应定向命中）。
- 信号必须 **不依赖 student 瞬时状态**（否则 student 稳定后归零）——结构/几何量（如走廊数、连通分支、
  测地绕路）是候选，但需避开 anchored 的"堆墙刷无解"陷阱。

---

## ★★★★★ 结构课程方案：事前引导 + 固定时间表（2026-06-25 设计，待 review）

> 用户洞察（接住并采纳）：**"堆墙不是核心，核心一直是 student 能力和课程难度不匹配"。**
> 这把所有诊断统一了——baseline 赢不是不堆墙，是随机海选天然给出从易到难全谱、student 总能找到
> "刚好够得着"的关往上爬；generator 输是塌到单一难度档（全是悬崖），课程难度与 student 能力脱节。
> **题眼 = "匹配"（让难度始终贴 student 能力前沿），不是 p、不是堆墙、不是 novelty。**

### 与先前"课程版 gate"（§463 选项三）的本质区别
| | 选项三 p_center 课程（已否定） | 本方案 结构课程 |
|---|---|---|
| 难度轴 | student 成功率 p | 几何结构量（fill / 测地绕路比） |
| 轴是否漂移 | **会**（p 依赖 student 瞬时态，学会后归零失向） | **不会**（图固有几何，student-independent） |
| 作用层 | 选关/打分（已证 gate on/off 无效，治不了生成塌缩） | **生成端事前引导**（直接塑造生成分布，治根） |
| 先易后难沿 | p（已证 p 两极也能赢，轴本身错） | 结构复杂度 |

### 核心机制：事前引导（不是事后过滤）
用户拍板**事前引导**而非"造完再过滤"。事后拒采样的病：generator 内部策略仍朝造难关训练，只在输出端
拦截 → 一直撞墙（造难→被拒→再造难），梯度混乱、效率低，是 anchored"堆墙刷无解"的同构陷阱。
事前引导 = generator 在"简单阶段"**根本倾向生成简单关**，从生成策略层就被约束，难关压根不被造出来。

PCGRL 是**逐格放置**（narrow rep，363 步逐 tile 放 EMPTY/WALL），天然支持**生成过程中实时塑形**——
不必等整图建完。落地手段（generator 的 PPO reward 上加一项**结构塑形项**，随时间放松）：

```
generator terminal reward (改后) =
    base_signal(difficulty/PVL/...，照旧)          # 保留原信号
  + curriculum_shaping(struct_value, threshold(t)) # 新增：结构超当前档则罚

其中 curriculum_shaping = -β · relu(struct_value - threshold(t))   # 软引导
  struct_value  = 该 map 的结构难度（见下双轴）
  threshold(t)  = 固定时间表，随训练 step 线性/分段抬升的难度上限
  β             = 塑形强度（够大则等价硬引导：超档关 reward 被压到远低于达标关）
```

> 说明：虽然用户倾向"硬性不生成"，但 PCGRL 逐格放置下无法在动作空间里硬禁"未来会让整图超档"的放置
> （超档是**整图**属性，单步看不出）。最接近"事前硬引导"的可实现形式 = **塑形项 β 取大**，使超档关
> terminal reward 远低于达标关 → generator PPO 梯度强烈推离超档 → 收敛后**几乎不再生成**超档关
> （而非生成后过滤）。这与"造完再拒采样"的关键差别：罚的是 **terminal reward（驱动生成策略本身）**，
> 不是在 buffer 注入端删关。β 扫 {1, 3, 10} 验证"多硬才够把生成分布推到达标档"。

### ★ 主轴定稿：测地绝对长度（非 fill、非绕路比）——用户抓出 B 类关推翻 fill
**用户看 baseline_DR-PLR__step100.png 抓到决定性反例**：baseline 的"简单关"有两类机制——
- **A 类**：墙稀疏 + 起终点远（大片空地一条长直线），靠**低 fill** 简单；
- **B 类**：墙很密 + 起终点近（满屏黑格但 start/goal 挨着，灰线极短），**fill 很高却简单**。

**B 类直接证伪 arm-fill**：高 fill ≠ 难。若用 fill 当难度轴，早期会把 B 类有效简单关误判为难关禁掉。
**这实锤了用户判断"堆墙不是核心"**：墙多≠难，决定难易的是 **student 要走的那条路有多长**。

→ **主轴改用「测地绝对长度」= BFS 距离场上 start→goal 的最短路格数**（不是绕路比，不是 fill）。
绕路比也被否（A 类长直线绕路≈1 会被当简单，丢掉"长直线=距离难"）。绝对长度同时正确处理两类：
A 类长直线=中长测地、B 类高墙近距=短测地，**一个轴吃下两类**，且 B 类高 fill 不再被误杀。

### 双轴 = 主轴(geo) + 反例对照(fill)
- **arm-geo（主轴）**：`struct_value = 测地绝对长度`（jit 内定步数 BFS 距离场取 goal 处距离值）。
  避开 Dijkstra×io_callback 卡死坑 [[stage4-stall-root-cause-dijkstra-iocallback]]：用定上界 while/scan 松弛
  （上界=map 格数即收敛），不用 io_callback、不用无界 while。
- **arm-fill（反例对照）**：`struct_value = fill_ratio`。**预期它因误杀 B 类而表现差**——把用户判断
  "堆墙非核心"变成论文里一条消融证据，不是真候选轴。

两轴都 student-independent（图固有几何），**不随训练漂移、不归零**——根治"信号在 student 稳定后失向"。

### ★ 档位标定（看 mazes/ 真实图读测地长度，2026-06-25）
读数来源（11×11 图，对角线≈14 格）：
- **baseline step100（有效简单关，定"易"端）**：灰线测地分布 **2~13 格**，中位 **5–6**，大量集中短端
  （B 类高墙近距 2–4；A 类空地长直线 9–13）。
- **generator pure-difficulty step2250（塌缩，待纠正）**：测地 3~11 但**全直线**（绕路≈1，无走廊结构）。
- **test Nav2/3（目标"难"端）**：start 左上 goal 右下，**真实绕齿测地 ≈16–22 格**（远超对角线），
  generator 全程造不出的长测地+高绕路关。

**三档定稿（测地绝对长度，单位=格；用户拍板 thr_hard 宁高勿低）：**
```
step ∈ [0,   1e8):  thr_easy = 5     # baseline 简单关中位数，早期只许短程；同时放行 B 类高墙近距
step ∈ [1e8, 2e8):  thr_mid  = 10    # 覆盖 A 类长直线(9–13)，放到对角线量级
step ∈ [2e8, 3e8]:  thr_hard = 22    # 锚 Nav2/3 真实绕齿测地(16–22)上限再宽一点，确保末期够到目标难度
```
> thr_easy=5 是测地**上限**：早期只禁长程关，B 类（高墙但路短）和短 A 类都放行 → 正是用户要的
> "墙多但起终点近=简单可以早期出现"。fill 不参与主轴，B 类不被误杀。
> 标定为图估值（火线是 start-goal 直连非真路径，有误差）→ **先开跑，靠训练中 gen_struct_value_mean
> 仪表盘微调**（用户选先开跑不上 Oscar 复核）。

### 调度：固定时间表（用户选，先验证假设）
按训练 step **分段抬升 threshold(t)**，完全 student-independent（数值见上"档位标定"）。
不用 student 自适应解锁（那会重新引入对 student 瞬时态的依赖 = 今天证明会自毁的东西）。
分两步走：**先固定时间表跑通、确认结构课程能赢 baseline，再决定要不要加自适应**。

### 实验矩阵 + 判据
| arm | struct_value | thr 时间表(格/比) | β | 对照 | 角色 |
|---|---|---|---|---|---|
| arm-geo | 测地绝对长度 | 5 → 10 → 22 | {1,3,10} | basep(bf4w9c1z) + pure-difficulty(无塑形) | **主候选** |
| arm-fill | fill_ratio | (待标定, 同法读图) | {1,3,10} | 同上 | **反例对照**(预期差) |

- **主判据**：test Nav1/2/3。基线锚 [[s4p2A-baseline-anchor-and-criterion]]：末值≈0.96，卖点是**更快达标**
  非更高末值；门槛检验 = 任一 seed 到不了 0.96 就检视 idea。**关键看 Nav2/3 能否从 0 翻起来**
  （baseline=1.0，所有 generator 档=0，这是要攻的关）。
- **机制判据**：per-level p 分布（gen_p_learnable_frac 是否从 ~0 抬起）、gen_p_mean 轨迹是否不再单调飘高
  （0.45→0.87 的"越来越简单"是否被时间表压住）、注入迷宫图（mazes/，看早期是否真的结构简单、
   后期才复杂）。
- **交叉判据**：arm-geo 赢而 arm-fill 不赢 → 核心是绕路结构、堆墙非主因（**实锤用户判断**）；
  两者都赢 → "按任意合理难度轴做先易后难匹配"就够（**更强结论：课程匹配 > 具体尺子**）。

### 代码改动点清单（待 review 后动手）
1. **新增 `struct_value` 计算**（`pcgrl_generator.py`）：
   - `geodesic_length(jaxnav_map, start, goal)`（**主轴**）：jit 内定步数 BFS 距离场，取 goal 处距离值
     = 最短路格数。避开 Dijkstra×io_callback（[[stage4-stall-root-cause-dijkstra-iocallback]]），用 while/scan
     定上界松弛（上界=map 格数即收敛）。不可达→返大值（被时间表当超难自然压住）。
   - `fill_ratio_of_map(env_map)`（**反例对照**）：纯逐元素，jit 安全。预期因误杀 B 类表现差。
2. **新增时间表函数** `curriculum_threshold(step, schedule)`：分段返回 thr(t)，纯标量，穿 jit。
3. **改 `gen_train_iter` / `compute_terminal_reward`**：terminal_rew 上加 `-β·relu(struct_value-thr(t))`
   塑形项。需把当前 step 与 schedule/β/arm 配置穿进来（与 gmm_params 同模式作宿主变量穿 jit 边界）。
4. **config 新增**：`CURRICULUM_ARM`(fill/geo/none)、`CURRICULUM_SCHEDULE`(thr_easy/mid/hard + 分段边界)、
   `CURRICULUM_BETA`。none = 关闭塑形（= 当前 pure-difficulty，作对照 arm）。
5. **新增 wandb 指标（含历次踩坑铁律：必记真实 p + 数值结构，不靠图/占位分）**：
   - **课程专属**：`gen_struct_value_mean`（注入关测地长度均值，标定微调靠它）、
     `gen_frac_over_threshold`（超档关比例，应随训练→0 证明事前引导生效）、
     `curriculum_threshold`（当前档，确认时间表推进）。
   - **★ 真实 p（铁律，[[baseline-p-also-bimodal-real-gap-is-diversity]]：图/占位分都骗过人）**：
     注入关从 rollout 取的**真实 per-level p**（gen_p_learnable_frac / gen_p_extreme_frac / gen_p_mean /
     gen_p_median，同口径于 base_vs_anc.py），**禁用 PLR 占位分 learnability**。看课程是否把 p 从两极拉回。
   - **★ 生成迷宫的数值结构（铁律，[[sim-npz-are-offline-sweeps-not-generator]]：判 generator 造关必看
     真实数值非图）**：注入关 `fill_ratio`、测地长度、绕路比、连通分支数的**分布数组**（直方/分位落 wandb），
     而非仅渲染 PNG。便于离线 rejudge 课程是否真把结构从塌缩谱推向 Nav2/3 走廊拓扑。
6. **单测**：fill/geo 度量正确性（含退化图）、时间表分段边界、塑形项符号（达标关塑形=0、超档关<0）。

### 风险与护栏
- **arm-geo 的 BFS 别撞 Dijkstra 卡死坑**：必须 jit 内定步数松弛（不用 io_callback、不用无界 while），
  上界 = map 格数即足够收敛。沿用 [[stage4-stall2-place-start-goal-unbounded-whileloop]] 的"加迭代上界"纪律。
- **invalid 图仍走 GEN_INVALID_PENALTY**：塑形项不替换现有 invalid 罚分（-1.0），叠加即可。
- **β 太大可能压垮 base_signal**：监控 mean_terminal_reward 别被塑形项主导到 base 信号失效；β 扫描正为定此。
- **时间表与 baseline step 对齐**：3e8 总步、分段边界须与 basep 同口径，否则"更快达标"无法横比。

---

## ★ 朝向对照 bug 与 Nav2/Nav3 真相（2026-06-25 晚，决定性发现）

### 1. SingleNav1/2/3 = 同一张地图，只改起点 + 初始朝向
实测 `state.map_data` 逐格对比：**Nav1/Nav2/Nav3 的 7×11 墙体完全相同（diff = 0/77）**，终点都在 col1。
区别只在 agent 的起点格和**初始朝向 θ**（定义在 `jaxnav_singletons.py:294/318/342`）：

| 关 | 起点 (x,y,θ) | 起点格 | 测地格数 | θ 含义 |
|---|---|---|---|---|
| SingleNav1 | (9.5, 1.5, **0.78**) | col9 | 12 | 45°，朝向偏目标 |
| SingleNav2 | (8.5, 1.5, **3.14**) | col8 | **17** | **180°，背朝目标** |
| SingleNav3 | (9.1, 1.5, **3.14**) | col9 | 16 | 180°，背朝目标 |

图见 `mazes/test_set/Nav2_vs_Nav3.png` + `mazes/test_set/test_set_5maps.png`。
**设计意图**（出处 No Regrets/SFL，arXiv:2408.15099，hand-crafted holdout）：JaxNav agent 只观测局部
LiDAR + 目标相对方向 + 自身速度（无全局地图），固定迷宫 + 扰动起点/朝向 = 隔离测试**对初始条件的
鲁棒性/泛化**。Nav2 最难 = θ=180° 背朝目标 + 起点贴墙角 + 测地最长，出生就朝错方向且半包围，
需先精细掉头再绕长程。

### 2. 朝向对照 bug（污染全部 generator arm 的真因）
| | 初始朝向 θ 采样 |
|---|---|
| **baseline (DR, `grid_map.py:140`)** | `uniform(-π, π)` — 连续覆盖 [-180°,180°] |
| **generator (`pcgrl_generator.py:119`)** | `(π/2)·choice([0,1,2,3])` — 只有 {0°,90°,180°,270°} 4 个离散值 |

离线采样验证（N=20000）：DR 连续 6016 个 unique 值、8 扇区均匀；generator 只有 4 个值各占 25%。
**student 在离散朝向上训练 → 学不到连续转向 → Nav2(θ=180°) 必翻车。** 这不是"坍缩"，是一行初始化 bug。

### 3. 结构课程 A/B 实测结果（job 3428935 / 3428936，各 1 seed go/no-go）
列顺序 = [Blank, Middle, Nav1, **Nav2**, **Nav3**]。

| | Nav2 (列4) | Nav3 (列5) | overall 末值 |
|---|---|---|---|
| **Baseline (48lfz699)** | 末值 **1.0**（中途两次归零，震荡） | 末值 **1.0**（震荡） | sing=1.0 |
| **Run A (课程 N=3 同配)** | **全程 0**（46 eval 无一非零） | 末段 0.45~1.0 波动 | ~0.71 |
| **Run B (课程 N=2 去 difficulty)** | **全程 0** | 多在 0~0.3，比 A 差 | ~0.64 |

**结论**：(a) **A > B 实锤 difficulty 删不得**（B 把 Nav3 拖垮，与"加课程后 difficulty 冗余"假设相反，
否掉 Run B 配置）；(b) **但 A/B 的 Nav2 全 0、不稳定、输 baseline 的整体判读全部受朝向 bug 污染**——
Nav2 全 0 极可能就是离散朝向造成的，**修 bug 前这次 A/B 的 go/no-go 不作数**。

### 4. 下一步（修 bug 优先）
1. 修 `pcgrl_generator.py:119` → 连续 `uniform(-π,π)` + 单测。
2. **重跑最直接被污染的两个对照**：baseline-vs-anchoredPVL（看 Nav2 是否从 0 翻起）、l1 难关 CVaR
   （看 -22pct 负结果是否扭转）。这两个实验是判断"过去几天的负结论里有多少是 bug、多少是真机制问题"的关键。
3. 修 bug 后若 Nav2 仍全 0，才回到"坍缩/curriculum-too-hard/QD"那条机制性路线。
