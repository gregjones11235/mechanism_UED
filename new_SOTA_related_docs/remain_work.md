# 剩余工作 & 当前真实结果状态（2026-06-27）

> 本文记录 new SOTA 主张的**当前真实数据状态**和**待办**。⚠️ 与 `new_SOTA_results_report` 里"+38% 超 SFL"的旧结论**冲突**——那个结论已被证伪（见 §1）。

---

## 1. ★当前真实横比（证伪了"已超 SFL"）

**口径说明（重要）**：下表 CVaR10 用的是**自写脚本 `~/cvar_all.py` 的 return 代理指标**（100-map 训练评测集的 returns 直方图 → worst-10% return 均值），**不是 SFL 官方/论文口径**。
- 官方 CVaR = `sfl/deploy/eval_jaxnav_single_agent_2_analyse.py`：**win-rate 口径 + 10,000 levels + seed0/seed1 去偏**，论文 Fig 3a。
- 官方/论文**没有给出参考数值表**，只有曲线图。
- 故本表只能用于**内部 run 之间相对比较**，不能声称"论文口径 CVaR"或"复现论文数字"。

| | seed | CVaR10_ret | overall_ret | overall_win |
|---|---|---|---|---|
| **SFL baseline** | s0 (8lh6h99y) | 1.269 | 3.911 | 0.979 |
| | s1 (8bv7giqf) | 2.212 | 4.002 | 0.993 |
| | s2 (nns7h0cg) | 2.752 | 4.045 | 0.996 |
| | s3 (48lfz699) | 2.199 | 3.985 | 0.994 |
| | **均值（4 seed）** | **2.108** | 3.986 | 0.991 |
| **currA** | s0 2维 (x7jffnye) | 1.653 | 3.933 | 0.982 |
| | s1 2维 (dffsybqy) | 1.730 | 3.954 | 0.983 |
| | s2 3维 (fl2mgtjq) | 1.855 | 3.947 | 0.986 |
| | **均值（3 seed）** | **1.746** | 3.945 | 0.984 |

**结论：当前数据下 currA 未超 SFL baseline。** 每个指标 baseline 都 ≥ currA（CVaR10 2.108 vs 1.746；baseline 三个高 seed 2.20/2.21/2.75 全部高于 currA 任一 seed 最高 1.855）。

**为什么之前误以为超了**：旧文档"+38%"是只拿了 **baseline 最低的 seed0(1.269)** 单 seed 比出来的假象。baseline 补到多 seed 后，seed0 暴露为低离群点（worst-bin −4.94 也最惨），其余 3 个 seed 都远高于 currA。

---

## 2. 数据口径与 run 清单（避免重复踩坑）

### baseline = SFL 本身
- `GENERATOR_INJECTION=false` 走 learnability top-K 海选 = SFL 算法（非退化 DR）。
- **确定性**：同 seed → 完全相同结果。故 `baseline-dump` 与 `s4p2-base` 的 seed1/seed2 CVaR 逐字相同（2.212/2.752），是同一批的复制品，去重后 baseline 实际 **4 个独立 seed（0/1/2/3）**。
- baseline 配置 = **SFL 官方仓库 `amacrutherford/sampling-for-learnability` 的默认值，一字未改**（Michael Beukman commit）：L=1000, ρ=0.5(FROM_SAMPLED=128/256), LR=2.5e-4。⚠️ 官方默认 ≠ 论文 Table 4（L=2000, ρ=1.0, LR=2.4e-4）——这是 SFL 代码 vs 论文的差异，非本项目问题。

### currA 当前 3 run 口径不干净
- s0/s1：2维 [anchored,cenie]，pool 3000（GEN_POOL_PER_GEN=1500×2），group=`currA-nodiff-3k`
- s2：**3维 [difficulty,anchored,cenie]，pool 2000**，group=`currA-bigpool2k` ← 与 s0/s1 **不同配置**
- 即"currA 3 seed"实为 2 个 2维(3k) + 1 个 3维(2k)，非干净同配置 3 seed。

### checkpoint 位置（官方 deploy 评测要用）
- 训练存于 `checkpoints/multi_robot_ued/{run_name}/model_step2250.safetensors`（按 run name）。
- deploy 期望 `checkpoints/jaxnav_single_agent/{GROUP}/*.safetensors`（已有 s4p2_base/s4p2_l1 用 symlink 指过去）。
- run name 映射：baseline s0/1/2/3 = mild-sponge-147 / twilight-grass-161 / royal-sunset-160 / revived-microwave-85(48lfz699)；currA s0/1/2 = happy-energy-159 / swept-jazz-158 / leafy-valley-154。

### CVaR 计算两条路（务必分清）
- 训练 run **内部不算 CVaR**，只 log overall_win/overall_ret + returns/win_rates 直方图。
- 自写 `~/cvar_all.py` = return 代理指标（上表用的）。
- 官方 `_2_analyse.py` = win-rate 真口径（论文 Fig 3a，需先跑 `_0_generate_levels`→`_1_rollout` 生成 10k-level CSV）。

---

## 3. ρ=1.0（论文配置）实验 —— ⚠️ currA run 课程被误删，结果不可用

**机制假设（用户提出）**：ρ=1.0 时不混入随机 `env.reset` 关，generator 定向供给的难关不被稀释 → 尾部鲁棒性应更强。

| 配置 | run | CVaR10_ret | overall_ret | worst-bin |
|---|---|---|---|---|
| **ρ=1.0** | **SFL** s0 (v3raza8r) | 2.319 | 3.985 | −1.02 |
| | ~~**currA** s0 (zjg5ehp4)~~ | ~~2.373~~ | ~~3.891~~ | ~~−1.99~~ |

### ⚠️★重大更正（2026-06-27）：currA-nodiff-paper (zjg5ehp4) 课程被误删，结果作废

提交 `sbatch_currA_paper.sh` 时**漏传了 5 个 `+CURRICULUM_*` 参数**（CURRICULUM_ARM=geo / BETA / THR_EASY=5 / THR_MID=10 / THR_HARD=22）。课程是 Hydra `+` 前缀新 key、yaml 无默认（见 [[curriculum-hydra-plus-prefix-required]]），漏传=课程关闭。

**后果：zjg5ehp4 同时改了 ρ(0.5→1.0)+L(1000→2000)+LR+buffer(48→60)+课程(有→无误删) 五个变量**，不是干净对照。"currA 2.373 反超 SFL"**不能采信**。

且之前基于此 run 的**裸 p 塌缩诊断（95% p≈1）很可能正是课程被误删导致**，非"anchored 无 gate 救不了塌缩"——那个诊断方向也需重做。

### ★2×2 因子消融完整结果（2026-06-27，论文规格底座，单 seed，return 代理口径）

底座：L=2000, ρ=1.0, LR=2.4e-4, pool 1500×2=3000, GEN_NUM_TO_SAVE=60。
2×2 = {固定课程, adaptive课程} × {无gate, 有gate}，对照 SFL paper-cfg + currA课程误删版。

| run | 课程 | gate | **CVaR10** | overall_ret | worst-bin | 可学区均值[0.2,0.8] |
|---|---|---|---|---|---|---|
| **B1 adaptcurr-nogate (t4t47j3k)** | adaptive | 无 | **2.447** ✅ | 3.944 | **+0.35** | 0.33% |
| currA 课程误删 (zjg5ehp4) | 无 | 无 | 2.373 | 3.891 | −1.99 | 0.32% |
| **SFL paper-cfg (v3raza8r)** | — | — | 2.319 | 3.985 | −1.02 | 4.91% |
| A1 fixedcurr-nogate (6kve106k) | 固定 | 无 | 2.006 | 3.861 | −4.98 | 0.32% |
| A2 fixedcurr-gate (oh0rf40j) | 固定 | 有 | 1.070 | 3.781 | −4.35 | 0.25% |
| B2 adaptcurr-gate (t0aah24d) | adaptive | 有 | 0.438 ❌ | 3.699 | −4.73 | 0.33% |

**三个结论：**
1. **★最佳 = B1（adaptive 课程 + 无 gate）CVaR 2.447 > SFL 2.319**，且 worst-bin **+0.35（唯一最惨关不崩盘正收益）**。
2. **★gate 是毒药，一致拉垮**：固定课程 2.006→1.070(加gate)，adaptive 2.447→0.438(加gate)。每对加 gate 都大跌。坐实 [[generator-pool-collapses-to-extremes]] 并升级"gate 不只无效、是有害"。
3. **课程排序（无gate）：adaptive(2.447) > 无课程(2.373) > 固定(2.006)**。固定课程比没课程还差（[[stage4-realcause-curriculum-too-hard]] 前期太难）；adaptive 按 student 表现升降档才对。

**★悖论（重要洞察）：可学区占比与 CVaR 几乎无关。** 所有 generator run 可学区都 0.25~0.33%（塌缩），唯 SFL 4.91% 高一个数量级；但 B1（可学区 0.33%）CVaR 反超 SFL（4.91%）。→ **真正起作用的是结构多样性/难关质量，不是 p 分布可学区占比**（坐实 [[baseline-p-also-bimodal-real-gap-is-diversity]]）。**"塌缩"可能根本不是要解决的核心问题。**

**可学区官方口径修正**：`gen_p_learnable_frac = mean(p∈[0.2,0.8])`（代码定义，非我之前手数的窄"p≈0.5"桶）。两极=p<0.05或p>0.95。用此宽口径，gate2k/gate3k/currA 可学区峰值仍仅 1.5~1.7%、末值 0、塌缩是共性病。

**待办：B1 单 seed 超 SFL 仅 +0.13，必须补 seed 坐实（吸取 §1 baseline 单 seed 翻盘教训）。** gate 应彻底砍。

配置存档：
- sfl-paper-cfg (v3raza8r)：CVaR 2.319，**无课程问题，可用**
- ~~currA-nodiff-paper (zjg5ehp4)~~：课程误删，由 B1 取代为新最佳
- 课程机制：**固定**=按训练进度三等分切 thr 档(5/10/22)；**adaptive**=按 frac(p>0.8)>τ(0.6) 连2次升档、frac(p<0.2)>τ 连2次降档(有地板)

---

## 4. 待办（按优先级）

### P0 — 坐实 / 证伪 SOTA 主张
1. **currA 补成干净同配置多 seed**：当前 3 run 配置混杂（2维3k×2 + 3维2k×1）。需 2维 pool=3000 跑齐 ≥4 seed 与 baseline 对齐。
2. **超参调优后重比**（见 §5，用户主张当前 currA 是"故意弱化版"：pool 3000<5000 + 超参未调）。
3. **上官方 deploy 评测**（win-rate CVaR + 10k levels + 去偏）得到能写进 paper、能和论文比的真数字。链路已摸清（§2）。

### P1 — 对齐论文严谨度
4. 10 seed（论文 single-agent JaxNav 是 10 seed；当前 baseline 4 / currA 3）。
5. 外部 baseline 同表：DR / PLR / ACCEL（官方 deploy 已有 `rsim_*_single_v4` 结果可复用）。

### P2 — 跨域泛化（加分，非必需）
6. multi-agent JaxNav（geodesic 锚可迁移，性价比最高的第二域）。
7. Minigrid（离散网格，geodesic 锚语义需重想）。
8. XLand-Minigrid（ruleset 采样，generator 范式基本搬不过去）。

---

## 5. 超参调优方向（待补，见对话）

> 用户主张：currA 是故意弱化版（pool 3000 vs SFL 5000、超参未调优）。调优空间分析见下一轮记录。

### 5.1 ★adaptive 课程实际轨迹诊断（2026-06-27，B1=resilient-blaze-166）

拉 wandb 档位轨迹（curriculum_stage）发现 **B1 的自适应课程"快速爬顶就再没动"**：
- 升档路径 `stage 0→1→2`：step 250 升 1、step **350 升到顶档 2**，之后到 2250 **再无变化、零退档**（max_stage 单调）。
- 即 **stage 0/1 各只停留极短（合计 ~300 步=13%），87% 训练全在 hard 档**。
- 退档逻辑（ALLOW_DEMOTE=true）"备而未用"：升顶后 frac(p<0.2) 从 0.13→0.03，**从未连续 2 次 >τ=0.6**，离触发退档差一个数量级。
- **诊断**：升档太易（frac(p>0.8)>0.6 连2次，student 一适应秒升）。"渐进课程"几乎没发生。
- **副推论**：`mazes/` 里 early/mid/late=step 50/1150/2250 对应档位 = stage 0 / **2 / 2**——mid 和 late 都在 hard 档，结构上不该有本质区别；用户"mid 图里有该归 hard 的关"是正常的（采样时课程已在 hard）。

### 5.2 ★难度结构项候选数据检验（2026-06-27，复用 measure_p1150 范式）

对 B1 注入关重跑 student rollout 测 p_true（n=200 可达关，step1150/2250 两 checkpoint），算各几何量 vs 失败率(1-p) Spearman：

| 几何量 | Spearman(与失败率) | 结论 |
|---|---|---|
| detour 绕路比 geo/euc | +0.251 | **有毒相关**：刷分=缩 euc=造短程畸形关（用户已点破 reward-hack），撤 |
| **geo 测地长度（现有主轴）** | **+0.240** | **唯一干净且有效**：刷分方向=拉长测地=长程导航，无捷径。**保留当主轴** |
| euc 欧氏距离 | −0.115 | 负相关：起终点越近反而越难（难关=近距离要绕一大圈） |
| fill 墙占比 | −0.167 | 负（塌缩假象） |
| sclear/gclear 局部封闭度 | +0.09/−0.02 | **几乎零相关**，撤 |

**两个被否的候选**：①朝向失配角——用户否决（破坏 uniform(-π,π) 朝向先验、制造鲁棒性漏洞，应保持均匀不进难度轴）；②绕路比——用户担心 reward-hack，数据坐实（euc 负相关印证"难关=euc小geo大"，detour 的 hack 方向与难关方向一致=有毒激励）。
**核心**：所有相关性都弱（≤0.25，p 99% 两极），印证 [[baseline-p-also-bimodal-real-gap-is-diversity]]——真正区分难度的是**结构多样性/拓扑**，单个几何标量都抓不住。→ 不加新几何轴，把劲用在"geo 主轴拆细 + 防退化"。
**待办（用户提出，未做）**：检验 anchored/cenie **bid** 与 (1-p) 的相关性——bid 是否能当难度轴。anchored 可干净重算（不需 gmm），cenie 需现拟 GMM（注入时 GMM 没落盘，只能近似）。

### 5.3 ★多档 + streak 可配代码改动 + 新 setting 提交（2026-06-27）

**代码改动**（jaxnav_sfl.py，已 scp Oscar + py_compile + 端到端 config 自检 PASS，备份 `.bak_pre_multistage`）：
- 新增 `CURRICULUM_THR_STAGES` config（**Hydra list 语法 `[5,10,16,22]`**，裸逗号 `5,10,16,22` 会被 Hydra 判歧义报错）：给定则覆盖默认三档 [easy,mid,hard]，状态机 `len(_curr_thr_by_stage)` 已动态、加档无需再改。
- 新增 `CURRICULUM_STREAK` config（默认 2，零回归）：替换两处硬编码 `>= 2`（升档+退档）。τ↑+streak↑ 逼 student 中间档多停留。

**两个 go/no-go run（2026-06-27 提交，各 seed0，2 GPU 并行 gpu2104）**：
| job | GROUP | τ | streak | 档 | 作用 |
|---|---|---|---|---|---|
| 3455019 | s4-adaptcurr-nogate-4stage-tau0_8-streak3 | 0.8 | 3 | 5/10/16/22 | 主菜：调严+拆档 |
| 3455020 | s4-adaptcurr-nogate-4stage-tau0_6-streak2 | 0.6 | 2 | 5/10/16/22 | 隔离对照：只拆档不调严 |

**归因三角**：3455019 vs 3455020 隔离"τ/streak 调严"效果；3455020 vs B1(三档) 隔离"档位拆细"效果。判据：看档位轨迹是否在中间档多停留（curriculum_stage）+ CVaR/Nav2 是否改善。
**⚠ 单 seed go/no-go**，趋势好再补 seed（吸取 §1 单 seed 翻盘教训）。

---

## 6. ★SFL 的 p(1−p) 理论怎么论证的（核查原文，2026-06-27）

> 用户想重写 p(1−p) 可学性理论，自提新本质指标。先核查 SFL 原文如何论证 p(1−p)，找它的薄弱处。

**核查对象**：SFL = "No Regrets: Investigating and Improving Regret Approximations for Curriculum Discovery"，Rutherford et al., **NeurIPS 2024**，arXiv:2408.15099（即本项目 baseline 同一篇，repo `amacrutherford/sampling-for-learnability` 对得上）。读 HTML 全文 §4.1。

### 结论：p(1−p) 是**纯口头直觉论证（intuition）**，既无数学推导，也无实验论证。

| 论证类型 | 原文是否使用 |
|---|---|
| 形式数学推导（从 regret bound / 梯度幅度 / 信息增益导出） | ❌ **完全没有**。无任何定理/引理/不等式把 p(1−p) 与 learning rate 联系 |
| 实验论证（证明 p(1−p) 与真实学习速度相关） | ❌ **没有**。论文未做"p(1−p) 高的关确实学得快"的相关性实验；实验全在下游（SFL 整体方法 vs baseline 的 return/鲁棒性），非验证 score 本身 |
| 口头直觉论证 | ✅ **唯一用的就是这个** |

**§4.1 原话**（定义 + 唯一论证）：
> "we define learnability to be p·(1−p)"，p = 成功率。
> "p represents how likely the agent is to obtain positive learning experiences from a level. 1−p represents the maximum potential improvement... Multiplying these yields (probability of improvement) × (improvement potential), i.e., expected improvement."

外加一句**事后旁注**（非论证基础）：
> "p·(1−p) can also be seen as the variance of a Bernoulli distribution... how inconsistent the agent's performance is."

### 这个直觉论证本身的三处逻辑松动（=可攻击入口）

1. **"1−p = 最大改进潜力" 是未证断言**。为什么改进潜力正比于失败率没有依据。它靠"概率 p"那项把 p→0 的关压下去，是两个 hand-wave 互相抵消凑出 p=0.5 最大，非一阶原理导出。
2. **"期望改进" 未定义改进度量**（regret 减少？return 增加？策略熵降？都没说）。
3. **Bernoulli 方差是"can also be seen as"事后观察**，指向"performance 不一致性"，与"learning potential"是两个不同的东西，作者未调和此张力。

### 对本项目"重写理论"的直接启示（三条入口）

1. **本质指标应是"期望策略改进量/梯度幅度"的可计算代理，而非 outcome 方差**。p(1−p) 只看二元成功率，丢掉"为什么失败、离成功多远"。本项目反复撞的 **p 两极塌缩致 p(1−p)→0 失区分度**（[[generator-pool-collapses-to-extremes]]）正是此本质缺陷的实证：所有关 p∈{0,1} 时 p(1−p) 数学塌成 0，**无法刻画"长程导航难度"这类结构维度**（=anchored-PVL/geodesic 路线立论基础）。
2. **真正的可学性应是 regret / value-prediction-loss 的函数**（项目已在用 PVL）。可主张：learnability 本质 = student 在该关的可改进量 ≈ E|advantage| / PVL，而 **p(1−p) 只是它在"二元稀疏奖励 + 确定性环境"下的退化投影**；连续导航（非二元非稀疏）里此投影失效——这正好解释为何 jaxnav 上必须换信号。
3. **形式化机会**：原文连"p=0.5 时学习最快"都没证。可补真推导（如稀疏奖励下 policy gradient 期望幅度 ∝ p(1−p)·|∇log π|），暴露其**隐含假设**（梯度方向项被当常数），指出该假设在结构化长程任务破裂。**把对手直觉升格成有前提的定理、再指出前提失效**=最干净的"重写"姿势。

来源：[arXiv:2408.15099 HTML §4.1](https://arxiv.org/html/2408.15099v1)、[SFL 官方 repo](https://github.com/amacrutherford/sampling-for-learnability)
