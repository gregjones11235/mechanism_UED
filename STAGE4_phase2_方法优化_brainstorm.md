# Stage4 阶段2 — 方法优化 brainstorm（currA / gate，怎么继续优化）

> 创建 2026-06-26 晚，2026-06-27 重写（修正：上一版把卖点讲反成"靠基数赢"，且只谈 5 关试水没谈真检验）。
>
> **真检验主线证据 = 100-map 测试集 + CVaR worst-case（文档 §2.1 定的 SOTA 口径，非 5 关 singleton 试水）。**
>
> 这份文档回答你提的三个不确定：**(1) 哪个 non-baseline 方法事实上最好；(2) 还能不能继续改善；(3) 还能不能调超参。** 分两大节：先定方法 + 干净实验路线，再展开超参/机制调优空间。§0.1 是 deep research 的新颖性撞车判定。

---

## 0. 核心结果（真检验口径 = 100-map + CVaR，单 seed）

> ★**baseline = SFL 本身（代码铁证）**：`GENERATOR_INJECTION=false` 走 `get_learnability_set` → `score=learnability=p(1−p)` → top-K（jaxnav_sfl.py:208/400/401），这正是 SFL(Rutherford NeurIPS2024) 的核心算法。所以下表不是"超随机 DR baseline"，是 **在 jaxnav 主场用更小池子超近期 SOTA SFL**。详见 [[baseline-is-sfl-itself-beat-with-smaller-pool]]。

| 指标 | **SFL**(learnability curation) | **currA pool=3000** | gate pool=2000 |
|---|---|---|---|
| 候选池基数 | **5000** | **3000**（更小）| **2000**（最小）|
| 可学区(p≈0.5)占比 | 1~21% | **0.07~1.4%** | 0~1.35% |
| overall win rate (100map均值) | 0.979 | **0.986** | 0.972 |
| **★CVaR10 win rate（最难10%关）** | 0.477 | **0.771** | 0.605 |
| **★CVaR10 return（最难10%关）** | 1.269 | **1.550** | 0.949 |
| worst-bin return（最惨那关）| −4.943 | **−0.872** | −3.714 |

**这才是"刷出 new SOTA"的证据所在**：均值三者持平（0.97~0.99，因为 100-map 大部分关不难谁都解），**差距全在尾部 worst-case**——而这正是 UED 方法该有的卖点（定向供给难关 → student 在最难关上更鲁棒）。currA 在最难 10% 关上成功率 0.771 ≈ SFL 0.477 的 1.6 倍；SFL 在最惨关 −4.943（撞墙/超时深度崩盘），currA 只 −0.872（接近解开）= **几乎不崩盘**。

**为什么能赢 SFL**（机制，文档 §6.3 洞察）：SFL 用同样的 learnability，但**没有 generator，只能被动从 env.reset 无偏先验里挑**，挑不出池里本就没有的 Nav2/3 结构难关；你有 learned generator 能**主动定向生成** SFL 永远海选不到的难关。="少而精的生成 > 多而杂的海选"。**注意 PPCG(Justesen 2018) 不是对手**——它是离散网格单调难度课程、早 SFL 6 年、从没在 jaxnav/CVaR 跑过；真正对标对手是 SFL，PPCG 只是 related work 要区分的老近邻。

> 5 关 singleton（Nav1/2/3 等）只是试水关，本文不作为主证据。Nav2 从全 arm=0 翻到 1.0 是攻破信号，但定论看上表。

---

## 0.0 ★卖点的正确表述（修正"靠基数赢"的错误）

我之前从"gate2000 vs currA3000 谁末值高"这个**配置内对比**里推出"基数是关键变量"，然后**错误外推**成"相对 baseline 也是靠基数赢"。同口径数字直接否定这个外推：

- **基数最大的是 SFL（5000），它恰恰是 CVaR 输家。** 三个 generator 配置里，连池子**最小**的 gate(2000) 都在 CVaR 上超过 SFL(5000)。"靠基数赢"在逻辑上不成立——若靠基数，5000 该赢。
- **可学关(p≈0.5)我们少一个数量级**：SFL 1~21%（绝对量几十~上千关），我们 0.07~1.4%（绝对量个位~几十关）。我们在**可学原料更稀缺**的劣势设定下赢了。

**所以真正的卖点 = 关卡质量 / 训练效率，不是数量：**

> **用更小的候选池（3000/2000 < SFL 5000）、少一个数量级的可学关，定向生成的关卡单位价值远高于随机海选，在最难 10% 关的 worst-case CVaR 上超过近期 SOTA SFL（+29pt）。** 同样去喂 PLR，我们用更少的原料喂出了更鲁棒的 student。

"基数"在**我们自己两个配置之间**（gate2000 vs currA3000）确实解释了一部分末值差（0.97→1.0），那是个内部旋钮。但在**我们 vs SFL** 这个真正重要的对比里，**基数是我们的劣势，我们却赢了——赢的只能是质量/效率。** 这是更干净、更强、也更符合 UED 动机的故事。

### ⚠️ "战胜 SFL" 的两种含义（措辞必须分清）
1. **内部对照式战胜（已有，最干净）**：SFL baseline 是我们自己 repo 里 `GENERATOR_INJECTION=false` 跑的，**同环境/评测/代码，唯一变量是"有没有 generator"**。这是最干净的消融对照。✅ 已成立（单 seed）。
2. **论文同表战胜（还没做）**：和 SFL/PLR/ACCEL/PAIRED **论文报告的数字**同表（文档 §2.1，需 10 seed + 外部 baseline）。❌ 还没做。
- 投稿时两种都可说，但**别把 (1) 说成 (2)**。当前能严格主张的是 (1)，且仍是**单 seed**——多 seed(P1) 是坐实前提。

---

## 0.1 新颖性核查判定（deep research，24 源 / 107 agent / 对抗验证，2026-06-26）

**总判定（两轮核查后，可对教授/同事陈述）：三条核心创新都新颖性可守，但有一篇头号近撞车 GenEnv(Dec 2025) 必须主动引用+区分，否则会被认为是它的换域增量。** 两轮共核查 ~20 篇 primary source + 对抗验证 + 前向引用追踪 + 非 UED 术语 + 距离场 reward shaping + 2026 预印本扫描。

| 三条核心创新 | 新颖性判定 | 头号近邻（必引必区分）|
|---|---|---|
| **① geodesic-anchored PVL/value-loss** | ✅ **可守（高置信）** | ACCEL/DEGen/SFL/NCC 全文 geodesic/distance-field=0；无人用测地距离锚定 regret/PVL 信号 |
| **② learnability 当 generator reward 的双向门(push p≈0.5)** | ⚠️ **可守但拥挤**——RL UED 内无人做，但 **GenEnv(2512.19682) 结构同构** | **GenEnv** 的 α-Curriculum Reward `R=exp(−β(p̂−α)²), α=0.5` 就是双向 ZPD 门当 generator reward。差异=域(LLM-agent 文本任务 vs jaxnav 连续导航 UED)+ GenEnv 无 geodesic 锚 |
| **③ 少而精生成超海选 SFL 的训练效率** | ✅ **可守但需正面证据** | 无 RL-UED 先例(对 SFL 5000池 +29pt CVaR)；最近结构类比=GenEnv "3.3× less data"(但 baseline 是 LLM 离线数据增强非 SFL) |

> ★★★ **GenEnv(arXiv:2512.19682, Dec 2025) 是头号风险**——第一轮漏了，第二轮攻盲区 D(2025末-2026预印本)才捞到。它同时撞 ②(双向门当 generator reward) 和 ③(targeted 生成省数据)。**正式陈述新颖性时必须主动提它并区分**(域不同 + 无 geodesic 锚 + 它是 LLM 环境策略我们是 PCGRL 连续导航)，不能说"双向门当 generator reward 无人做过"。这正是攻盲区的价值：没扫 2025末预印本就会在不知情下踩雷。

### 撞车线1 — currA（anchored 信号 + 训练效率结论）

| 子机制 | 判定 | 最近邻先行工作 | 精确差异 |
|---|---|---|---|
| **geodesic-anchored value-loss** | ✅ **新颖**（高置信） | **TRACED**(Su 2025, 2506.19997)：Regret=PVL+α·ATPL，ATPL=transition-dynamics 重建误差；**DEGen**(Mead/Foerster/Hawes 2026, 2601.14957)：更密 generator reward + MNA(advantage) | 三者都不用测地/最短路锚定（DEGen 全文 'geodesic/shortest/BFS/distance-field'=0）。我们用测地距离场几何锚定 PVL 防堆墙=无人做过 |
| **"小池/稀疏可学关却赢大基数 baseline"=训练效率** | ✅ **新颖**（高置信，且应做成正面结果）| **DRED**(Garcin ICML 2024)：变生成**分布**；**Garcin 2023**：变**采样策略**+互信息；**SFL**(2024)：pool 参数固定为超参 | 没人报告"更小候选池+更稀疏可学关，CVaR 仍超大基数随机海选"。⚠️**别再写成"基数才是关键"**(那既贬低自己又撞 sampling-strategy 工作)，写成**训练效率/关卡质量**结论，并用受控消融立正面证据(见 P4) |

### 撞车线2 — gate（双向可学性门 + 自适应课程状态机）★最暴露

| 判定 | 最近邻先行工作 | 精确差异（reviewer 会追，必须显式+实证论证）|
|---|---|---|
| ✅ **坐标未占据但拥挤**（高置信，区分微妙） | **Goal-GAN**(Florensa ICML 2018, 1705.06366)：GAN 生成"中等难度目标"(p∈0.1~0.9 双边窗) | 在**目标空间**生成 goal、非 PLR replay level；是 0.1~0.9 **窗**非 p≈0.5 **推**；GAN adversarial 非 reward 项 |
| | **Setter-Solver**(Racaniere ICLR 2020, 1909.12892)：learned setter 训于 validity+feasibility+coverage | 条件于 f∼Unif(0,1) **全范围**、非固定 p=0.5 门。⚠️"steering toward target difficulty"说法被对抗验证 **0-3 否决**，别引它当固定目标难度机制 |
| | **SFL**(Rutherford NeurIPS 2024, 2408.15099)：learnability=p(1−p) 在 p=0.5 最大 | **纯 curation/selection**(top-K)，"does not generate levels"、无 learned generator。我们把 learnability 当 **generator reward** 而非 curation=载重区分 |
| | **LILO**(Foster NeurIPS 2025, 2502.12272)：优先 p≈0.5 高方差题 | curation 现有 LLM 数学题、非 generator reward、不同 domain |
| | **PPCG**(Justesen 2018, 1806.10729)：按 agent 表现调生成关难度 | ★最接近自适应状态机：但 PPCG 是**单调难度标量**(一个旋钮)、非 learnability 门、非 PLR buffer；我们是双向 frac(p>0.8)/frac(p<0.2) 触发+地板逻辑 |
| | ★**GenEnv**(arXiv:2512.19682, Dec 2025)：learned LLM 环境策略，α-Curriculum Reward `R=exp(−β(p̂−α)²),α=0.5` 当 generator reward | **结构同构！** 它就是双向 ZPD 门当 generator reward。差异=**域**(LLM-agent 文本任务 API-Bank/ALFWorld vs jaxnav 连续导航 UED)+ 它**无 geodesic 锚** + 它用 Reward-Weighted Regression 训 LLM 策略、我们是 PCGRL。**必须主动引用区分** |

**载重新颖性 = 在 RL UED level generation 里，把 learnability 当 generator 训练 reward（而非 curation）+ 配 geodesic 锚 + 连续导航域。** RL UED 内确实无人做(ACCEL/DEGen/SFL/NCC 全把 learnability/regret 放 curation 侧)。但 **GenEnv 在 LLM 域做了同构的双向门当 generator reward**——所以我们的新颖性**不能**主张"双向门当 generator reward 史无前例"，要主张"**在 RL UED / 连续导航 / 配 geodesic 锚** 这个具体设定下"。

### 撞车线3 / 课程母题（A&B）

谱系（两轮已坐实）：**PAIRED→PLR/Robust-PLR(PLR⊥, DCD 统一, Jiang NeurIPS 2021, 2110.02439)→ACCEL→SFL→TRACED/DEGen**；学界共识(Jiang 2023 thesis + SFL/ReMiDi)把这条线定性为 **regret-based**，learnability 是**后来的departure/critique**——这对你有利：你的 learnability-as-generator-reward 是"延伸公认的新方向"，不是重新发现 regret 老线。非 UED 一支 **Goal-GAN / Setter-Solver / ZPDES**；A 类由易到难一支 **PPCG**；LLM 域 **GenEnv**。**我们坐标 = learned PCGRL generator + generator 侧 learnability 门 + geodesic 锚 + jaxnav 连续导航 = 未占据交叉点。**

**ACCEL 已直接核查（第二轮补上）**：ACCEL = evolutionary edit(主要随机 mutation)，编辑+curation **都用 regret=PVL**，**不是 learned generator、不是 PCGRL**(原文明说"PCGRL 用手工 dense reward，与 ACCEL 不同")、**无 learnability 门、无 push toward p≈0.5、无 geometric 锚**。它的"easy criterion"(把低 regret 可解关编辑回 frontier)是最接近难度定向的机制，但是**单向 selection 规则**非双边 generator reward。差异清晰，不撞。

### ⚠️ 投稿前硬边界（两轮后更新）
1. **GenEnv 是头号必引必区分**(见上)，且它 Dec 2025 才出 → 子领域快速移动(DEGen Jan 2026)，**camera-ready 前必须再扫一次 2026 最新预印本**，任何新颖性陈述要带时间戳。
2. ✅ **ACCEL 已直接核查**(上一版的缺口已补)，差异清晰。
3. ⚠️ **三个盲区第二轮未完全闭合**(靠"没搜到"非正面 primary-source 阅读，证据较弱)：(a) PCGRL/PCG-for-curriculum 配 learnability/ZPD reward(盲区 Gap3)；(b) 非 UED 的 DDA/success-rate-targeted generation/teacher-student 难度调控(盲区 B)；(c) 完整的 SFL/ACCEL/Robust-PLR 前向引用图谱(盲区 A 只部分遍历，捞到 DEGen/NCC/GenEnv)。**这三个是残留风险，正式陈述时要标注"基于已搜范围"**。
4. geodesic-anchored PVL 要能扛"它只是 navigation 里老的 distance-based shaping 的特例"这个 objection——关键论点=**我们锚的是 UED SCORE(防 generator 堆墙刷分)，不是 agent 的 reward shaping**，无 source 这么做。这个 framing 建议再做一次专门的 reward-shaping 文献压力测试。

---

## 第一节 · 哪个方法最好 + 怎么干净地坐实

### 1.1 把"方法"拆成三个正交的轴

currA 和 gate 不是两个打包配置，是三个独立旋钮的不同取值。先解耦才能谈"谁更好"：

| 轴 | currA pool=3000 | gate pool=2000 | 说明 |
|---|---|---|---|
| **A. 候选池基数** | 3000 | 2000 | 配置内旋钮。注意：**两者都 < baseline 5000** |
| **B. 信号维度** | 3 维 `[difficulty, anchored, cenie]` | 2 维 `[anchored, cenie]` | currA 带 difficulty |
| **C. reward 门控** | 无 gate | 双向 gate(W_HARD=2.0/W_EASY=1.0)+自适应课程状态机 | gate 推向 p≈0.5 |

**问题本质**：currA 和 gate 三个轴同时不同，混淆没法直接说谁好。要回答"哪个最好"，必须逐轴隔离。

### 1.2 现有证据能下的结论（诚实账，单 seed）

| 结论 | 证据强度 | 依据 |
|---|---|---|
| **两个配置在 CVaR 上都超 baseline**（核心卖点）| **中**（单 seed，但两配置一致超）| currA 0.771 / gate 0.605 vs baseline 0.477 |
| **赢在训练效率/质量非基数** | **强**（逻辑+同口径）| 基数最大的 baseline 反而 CVaR 最低；我们可学关少一个数量级 |
| currA CVaR 高于 gate | **中**（单 seed）| 0.771 vs 0.605 |
| currA 高于 gate 是配置(B/C)还是基数(A)的功劳 | **未知** | A/B/C 三轴在两 run 间同时不同→纠缠 |
| 门控 C 轴本身有没有用 | **未知** | gate 同时改 B 和 C，分不清门控贡献 |

**最该警惕**：全是**单 seed**。CVaR 是尾部统计，单 seed 噪声更大。多 seed 是坐实卖点的第一硬门槛。

### 1.2.5 ★多 seed 实测结果（2026-06-27，统一口径 = returns CVaR10）

> ⚠️ **口径修正**：之前对话用 win_rate 的 CVaR（0.771/0.477 等）已废弃——100-map win_rate 是 96/4 两极分布（96 关满分），CVaR10 在其上失真（mean==CVaR）。**全部改用 returns 的 CVaR10**（连续、有真尾部）。详见记忆 [[win-cvar-distorted-use-returns-cvar]]。**下表是当前唯一可信的 CVaR 横比。**

| run | 配置 | overall_ret | **CVaR10_ret（最难10%）** | worst-bin | 难关数(<1) |
|---|---|---|---|---|---|
| **SFL baseline** | learnability curation, 池5000 | 3.911 | 1.269 | −4.94 | 4 |
| **currA 3维** (fl2mgtjq) | `[difficulty,anchored,cenie]` 池3000 | 3.947 | **1.855** | −0.87 | 6 |
| **currA 去diff s0** (x7jffnye) | `[anchored,cenie]` 池3000 | 3.933 | **1.653** | −2.11 | 7 |
| **currA 去diff s1** (dffsybqy) | `[anchored,cenie]` 池3000 | 3.954 | **1.730** | −4.09 | 4 |
| gate2k (qfo781yy) | 池2000 | 3.868 | 0.949 | −3.71 | 7 |
| gate3k s0 (jkju1i58) | 池3000 | 3.983 | 2.333 | +0.86 | 10 |
| gate3k s1 (f13t8isa) | 池3000 | 3.862 | 0.959 | −5.32 | 6 |

> 口径说明：上面 3 个 **currA 谱系 run 不是同配置 3 seed**——是 1 个 3维(带difficulty,单seed) + 2 个 2维(去difficulty,2seed)。但它们共享 currA 主干（anchored+cenie+geo课程+池3000），合起来给了 currA 路线的多 seed 鲁棒性证据。

**三个结论：**

1. **✅ difficulty 大基数下多余**：currA 3维 CVaR 1.855 → 去 diff 两 seed 1.653/1.730（均值 1.69），只降 ~9%。印证 difficulty weight 全程仅 8% + difficulty≡learnability−0.25 在两极池失区分度（见 [[difficulty-bid-drowned-equals-learnability]]）。方法可简化为 **anchored+cenie 2维 + geo课程**。

2. **★currA 路线稳，gate 路线 seed 撕裂**：去 diff 两 seed CVaR 1.653/1.730（差 0.08，一致）；**gate3k 两 seed 2.333/0.959（差 2.4倍）、worst-bin +0.86/−5.32（一最佳一最崩）**。同样 2维信号+大基数，**gate 的双向门/自适应课程是方差来源、不是稳定性来源**。

3. **✅ currA 去diff 两 seed 都稳超 SFL**：1.653/1.730 vs SFL 1.269（都 +0.4 左右），overall_ret 也都 ≥ SFL。**首次多 seed 一致超 SFL**（之前 currA 3维只单 seed）。

**判定：当前最佳架构 = currA 路线的 2 维简化版（anchored+cenie+geo课程，无 gate、无 difficulty）。** 比 currA 3维几乎不丢、比 gate 路线多 seed 稳定性碾压、比 SFL 两 seed 一致超。gate3k s0 的 2.333 是单 seed 运气（s1 崩到 0.959），worst-case 口径下不能押"一高一崩"的方法。

### 1.3 干净隔离实验矩阵（按优先级）

目标：最终能写出"配置 X 在 CVaR 上确定超 baseline，且贡献来自轴 Z + 训练效率"。

**P0 — 已在跑：gate pool=3000 × 2 seed（job 3446797_0/_1，RUNNING 健康）**
- 隔离 A 轴：gate 开到和 currA 同基数(3000)，看 gate 配置本身是否 ≥ currA。同时是第一次多 seed。
- **判据**：两 seed 的 100-map CVaR10 是否一致 + 是否追平/超 currA 的 0.771。

**P0 状态（已完成 2026-06-27）**：gate3k 2 seed 跑完 → **seed 撕裂**（CVaR 2.333/0.959）。结论=gate 配置不能稳定复现，"gate 更优"是单 seed 假象。见 §1.2.5。

**P1 — 多 seed 坐实 currA（部分完成）**
- ✅ currA 去diff 2 seed 已跑（CVaR 1.653/1.730 一致，都超 SFL 1.269）。currA 3维仍单 seed（1.855）。
- 待补：currA 3维补 seed + 推向 §2.1 的 10 seed 全表。**当前已能说"currA 路线 2维版多 seed 稳超 SFL"**。

**P2 — 门控消融（钉死 C 轴，"gate 到底有没有用"）**
- 固定 pool、固定信号，只开关门控：gate-on vs gate-off @同 pool。
- ⚠️ 风险：记忆 [[generator-pool-collapses-to-extremes]] 记过小基数下 gate on/off 曲线完全重合(gate 无效)。**新证据更不利**：§1.2.5 显示 gate 路线 seed 方差远大于 currA 路线 → gate 的门控/自适应课程很可能是**方差来源**。P2 大概率证 gate 该砍。

**P3 — 信号维度消融（钉死 B 轴，部分完成）**
- ✅ **3维 vs 2维已做**：去 diff（[anchored,cenie]）vs currA 3维，CVaR 只降 ~9%（1.69 vs 1.855）→ **difficulty 大基数下多余**（见 [[difficulty-bid-drowned-equals-learnability]]）。
- 待补：2维 vs 1维`[anchored]`（cenie 是否也可砍）。

**P4 — ★训练效率曲线（把核心卖点做成正面受控结果）**
- pool ∈ {128, 512, 1000, 2000, 3000}，**全部 ≤ baseline 5000**，信号固定，画 **CVaR10 win vs pool** 曲线，**与 baseline(5000) 水平线对照**。
- 这条图是论文核心 figure：**"我们在哪个 pool 起就追平/超过 5000 的 baseline"**——直接量化"用多小的池子就赢"。比单个 +29pt 数字强得多，且把新颖性立在正面受控结果上（回应 §0.1 撞车线1 的要求）。

### 1.4 我的判断（多 seed 之前）

- **要 SOTA 数字 / 主结果**：押 **currA pool=3000**（CVaR 最高），但**必须先过 P1 多 seed**。
- **要最干净的卖点叙事**：押 **"训练效率"本身**（P4 曲线）——它不依赖门控/信号是否 work，且天然回应"为什么不直接堆基数"。currA/gate 是它的两个实例。
- **gate 命运取决于 P0+P2**：P0 若显示 gate@3000 追平 currA 且过程更好(学得早/回撤小)，gate=更高效配置；P2 若显示门控无效，gate 只剩 2 维信号这点区别。

**一句话**：别在多 seed 前选 currA 还是 gate。先 P0→P1→P2 拆三轴。很可能最终结论是"配置都不是关键，**训练效率(小池/稀疏可学关却赢)才是卖点**"——那最好的"方法"是最简配置 + anchored 信号，而非 currA/gate 任一全套。

---

## 第二节 · 还能怎么调（超参 + 机制改进空间）

按"预期收益/风险"排序。每条标 [收益] 和 [前提]。

### 2.1 训练效率方向（A 轴 —— 卖点所在，最该深挖）

1. **★做训练效率 scaling 曲线（P4）** — [收益最高] 不是为了"推大 pool"，而是为了**找到追平 baseline 5000 的最小 pool**。如果 1000 就追平，卖点变成"用 1/5 的池子达到 SOTA"，极强。[前提] 见 P4。

2. **解耦"造关基数"与"测信号基数"** — [收益中，新机制] 现在 pool 里每关都跑完整 student rollout 测 p（贵）。可"造很多关(便宜 CNN inference)，只对子集测 p"，让有效多样性进一步放大而 rollout 成本不爆。[风险] PLR 需 p 排序，没测 p 的关进不了优先级——curation 逻辑要重想。**这本身是个值得单独想的机制创新点，且强化"效率"叙事。**

3. **GEN_ROLLOUT_CHUNK** — [收益低，纯工程] 仅 OOM 时下调，不影响数值。

### 2.2 信号质量方向（B 轴 —— "质量"卖点的核心）

4. **显式加结构多样性/拓扑信号** — [收益高，针对性] 记忆 [[baseline-p-also-bimodal-real-gap-is-diversity]]：baseline 赢靠结构多样性(走廊拓扑)。现在 generator 靠少量关偶然撞到。**显式奖励 Nav2/3 那类拓扑**(连通分量、走廊长度、分支度)→ generator 从"碰运气"升级为"定向供给目标结构"。这把"质量"卖点坐实。[风险] 易 reward-hack，需想清"目标结构"可计算代理。

5. **砍信号到最简** — [收益中，简化] 若 P3 显示 1维 anchored 够，砍 difficulty/cenie。卖点更强("最简即可超 baseline CVaR")。[前提] P3。

6. **anchored-PVL 几何锚调参** — [收益低-中] 锚强度/形式(线性 vs 平方 vs 分段)。记忆记离线 ρ(fill) 0.35→0.004 已很好，边际小。

### 2.3 门控 / 课程方向（C 轴 —— 收益取决于 P2）

7. **W_HARD/W_EASY、gate 目标点** — [收益中] 当前推向 p≈0.5；可把目标 p* 设超参或让它随训练漂移(隐式课程)。[前提] 仅 P2 证门控有效才值得调。

8. **gate 从 reward 项改成 curation 项** — [收益高，方向性] 记忆 [[alp-bid-vs-quota-two-architectures]] 点破过"learnability 是 modulation 不该进 generator reward 竞争"。把可学性约束移到 **PLR 采样/注入层**(只 curate p≈0.5 的关进 buffer)，generator reward 纯追难度/多样性。更符合"generator 造、curator 选"分工。[风险] 改架构。**最可能产出干净新机制。** ⚠️注意：这会让撞车线2从"generator reward"退回"curation"，撞 SFL 更近——权衡。

9. **自适应课程 τ / 升降档逻辑** — [收益中] 当前 τ∈{0.6,0.8}。[前提] **有可能大基数/PLR 已让课程多余**——"课程 vs PLR 自身 curate 谁更必要"值得单独消融。

### 2.4 评测 / 验证（坐实而非调方法）

10. **多 seed**（P1，硬门槛，CVaR 尾部统计噪声大）。
11. **和外部 baseline(SFL/PLR/ACCEL/PAIRED)同表**（文档 §2.1，没它只是"超自己 baseline"非 SOTA）。
12. **per-bin(按难度分桶)win rate 对比图** — 卖点在尾部，可视化"尾部更鲁棒"比单个 CVaR 数字更有说服力。

---

## 第三节 · 建议执行顺序

```
现在 ──► [P0] gate pool=3000 × 2seed（已跑，job 3446797，健康）
            │ 同基数下 gate vs currA？多 seed 一致吗？
            ▼
        [P1] currA pool=3000 × 3-5seed   ← 最高科学优先级，排运气
            │ +29pt CVaR 是真的还是单 seed 运气？
            ▼
        [P2] 门控消融 + [P3] 信号维度消融 @固定pool
            │ 门控/贵信号在我们设定下还有没有用？（很可能可砍→简化）
            ▼
        [P4] ★训练效率曲线（128→3000，对照 baseline 5000）
            │ 用多小的 pool 就追平 baseline？→ 核心 figure
            ▼
        [§2.4] 多 seed 全表 + 外部 baseline 同表 → 真 SOTA 对标
```

**三句话浓缩：**
1. **卖点 = 训练效率/关卡质量（小池、稀疏可学关却在 CVaR 超大基数 baseline），不是基数。** 基数最大的 baseline 恰是 CVaR 输家。
2. **别在多 seed 前选 currA 还是 gate** —— 全是单 seed，CVaR 尾部噪声大。
3. **最可能的优化方向是砍东西+做效率曲线**——P2/P3 很可能显示门控/贵信号可砍，P4 把"用多小的池子就赢"做成正面受控结果，既是更强卖点也回应新颖性核查。
