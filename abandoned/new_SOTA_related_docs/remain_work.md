# 剩余工作 & 当前真实结果状态（2026-06-28）

> 本文记录 new SOTA 主张的**当前真实数据状态**和**待办**。⚠️ 与 `new_SOTA_results_report` 里"+38% 超 SFL"的旧结论**冲突**——那个结论已被证伪（单 seed 假象，见 §1）。
> **当前真实状态（2026-06-28）**：holdout 三档机制端到端跑通、两 seed 课程轨迹逐帧可复现，但**性能 seed 方差极大**（CVaR10 seed0=1.115 ↔ seed1=2.323），2 seed 均值 1.719 **未超** baseline 4 seed 均值 2.108。**又一次单 seed 翻盘**（先看 seed1 强、补 seed0 暴露崩盘）。s4tight 四档亦仅单 seed、同样未坐实。**结论：当前无任何 generator 路线在多 seed 均值上确证超 baseline，须补 seed（§4 P1）。**

---

## 0. ★★★全量 run 整合大表（2026-06-28，所有 run 一张表）

**口径**：CVaR10 = return 代理（100-map 训练评测集 returns 直方图 → worst-10% return 均值，`~/cvar_all.py` 同口径），**非论文 win-rate 口径**，仅供内部 run 间相对比较。按 CVaR10 降序。

| # | 配置 | 档数 | 下界 | 判据 | seed | **CVaR10** | worstbin | mean_ret | ovr_win | 来源 |
|---|---|---|---|---|---|---|---|---|---|---|
| — | **SFL baseline** | — | — | — | **4** | **2.108** | −1.51 | 3.98 | 0.990 | §1 |
| ★ | **combo1 best-seed**(ts3.2/ρ0.8/clip0.05) | 3 | 无 | frac | s0 | **3.089** 🥇 | **−3.83** | 3.98 | 0.987 | §9.5 |
| 1 | **B1 adaptcurr-nogate** | 3 | 无 | frac | 1 | **2.447** ✅ | +0.35 | 3.95 | 0.992 | §5.4 |
| 2 | s4tight 四档 | 4 | 无 | frac | 1 | 2.379 | +0.74 | — | 0.996 | §5.4 |
| 3 | currA 课程误删 | 无 | 无 | — | 1 | 2.373 | −1.99 | 3.89 | — | §3 |
| 4 | SFL paper-cfg | — | — | — | 1 | 2.319 | −1.02 | 3.99 | — | §3 |
| 5 | currA nodiff（旧） | 3 | 无 | frac | 3 | 1.746 | — | — | — | §1 |
| 6 | holdout 三档（旧） | 3 | 无 | holdout | 2 | 1.719 | −1.00 | 3.83 | 0.985 | §1 |
| 7 | **★C 四档无下界（新）** | 4 | 无 | holdout | 2 | **1.325** | −3.80 | 3.79 | 0.977 | §5.11 |
| 8 | A1 fixedcurr-nogate | 4 | 无 | 固定 | 1 | 1.070 | −4.98 | 3.86 | — | §3 |
| 9 | s4only 四档 | 4 | 无 | frac | 1 | 0.609 | −5.56 | — | — | §5.4 |
| 10 | **fixA 四档有下界（新）** | 4 | **有** | holdout | 2 | 0.444 | −3.49 | 3.69 | 0.974 | §5.11 |
| 11 | B2 adaptcurr-gate | 3 | gate | frac | 1 | 0.438 ❌ | −4.73 | 3.70 | — | §3 |

> **⚠️ combo1 表首项是 best-seed，非均值，会误导，必看**：combo1（B1基底+TS3.2e8+ρ0.8+clip0.05，2seed）刷出**单 seed 史上最高 CVaR10=3.089**，但**另一个 seed 崩盘**：s0=3.089(worstbin−3.83) ↔ **s1=1.679(worstbin−4.98)**。两 seed 均值仅 2.384、且**两个 seed 的 worstbin 都大负**（最难关都崩，不像 B1 的 +0.35）。即"3.089🥇"是幸运 seed 的尖峰，崩盘未修。详见 §9.5。
>
> **作废 run（课程节奏 bug，§5.10）**：旧 8-task 双边下界扫描（A/B/C/D ×2seed，CVaR 0.2~1.3）因 holdout 判据 GRACE/攒考卷开销吃掉 ~39%/档、卡 stage2、stage3 永不进入而**全部作废**，不入表。

### 谁最好 + 三条结论

1. **唯一超 baseline 的是 B1（2.447，三档无下界 frac 判据，worstbin +0.35 最难关不崩盘）。但 B1 单 seed**，§1 铁律：单 seed 翻盘已重演 3 次，未补 seed 不可采信。第 2~4 名（s4tight 2.379 / currA 2.373 / SFL-paper 2.319）全挤在 2.3~2.4 且全单 seed，彼此差距在 seed 噪声内（baseline 自身单 seed 能 1.27↔2.75）。
2. **多 seed 里最高 = C 四档无下界（1.325，2seed），但仍 < baseline 2.108。无任何 generator run 在多 seed 均值上确证超 baseline**（当前最诚实状态，未变）。
3. **两个机制结论（本次坐实）**：① **下界有负影响**——fixA(有下界 0.444) vs C(无下界 1.325)，唯一变量是下界，去下界 CVaR 翻 3 倍；带下界/gate 的（#10/#11/作废 A/D）全垫底 → **下界与 gate 都该砍**。② **四档系统性差于三档**——三档区（2.447/1.746/1.719）整体高于四档区（1.325/1.070/0.609/0.444）；3-stage 无下界 holdout 验证中（§5.11）。

---

## 1. ★当前真实横比（证伪了"已超 SFL"）

**口径说明（重要）**：下表 CVaR10 用的是**自写脚本 `~/cvar_all.py` 的 return 代理指标**（100-map 训练评测集的 returns 直方图 → worst-10% return 均值），**不是 SFL 官方/论文口径**。
- 官方 CVaR = `sfl/deploy/eval_jaxnav_single_agent_2_analyse.py`：**win-rate 口径 + 10,000 levels + seed0/seed1 去偏**，论文 Fig 3a。
- 官方/论文**没有给出参考数值表**，只有曲线图。
- 故本表只能用于**内部 run 之间相对比较**，不能声称"论文口径 CVaR"或"复现论文数字"。

| | seed | CVaR10_ret | overall_ret | overall_win | worstbin |
|---|---|---|---|---|---|
| **SFL baseline** | s0 (8lh6h99y) | 1.269 | 3.911 | 0.979 | −4.94 |
| | s1 (8bv7giqf) | 2.212 | 4.002 | 0.993 | −0.65 |
| | s2 (nns7h0cg) | 2.752 | 4.045 | 0.996 | +0.43 |
| | s3 (48lfz699) | 2.199 | 3.985 | 0.994 | −0.87 |
| | **均值（4 seed）** | **2.108** | 3.986 | 0.991 | **−1.51** |
| **holdout 三档** | s0 (b1aaaec2) | 1.115 | 3.765 | 0.972 | −2.64 |
| | s1 (x1shj21n) | 2.323 | 3.895 | 0.998 | +0.63 |
| | **均值（2 seed）** | **1.719** | 3.830 | 0.985 | **−1.00** |

> **配置**：`s4-holdout-3stage-grace6-tau0_8` = §5.8 holdout 掌握度判据（各档冻结 N=32 历史注入关当固定考卷，student 在其上的真实 p_holdout 判升降档；标定 GRACE=6/COLLECT=4/INTERVAL=3/τ_up=0.8/τ_down=0.3）。底座 = B1 三档 [5,10,22] geo arm + anchored+cenie bid，L=2000/ρ=1.0/LR=2.4e-4/pool 3000。两 seed 课程轨迹逐帧可复现（eval#16 升 thr=10、eval#33 升 thr=22，无退档）——**机制完全跑通，但性能 seed 间方差极大（见下）**。

**结论：holdout 2 seed 均值（1.719）未超 baseline（2.108）。又一次单 seed 翻盘——seed 方差吞掉了好结果。**

1. **★seed 间方差极大，比 seed0/seed1 差异 = 1.115 ↔ 2.323（worstbin −2.64 ↔ +0.63）**，几乎是 baseline 自身跨度（1.27~2.75）的翻版。**这正是 §1 反复警告的剧本第三次重演**（前两次：旧"+38%"假象、B1 单 seed）：先看 seed1=2.323（落 baseline 高 seed 区、worstbin 史上次好）以为强，补 seed0 后暴露 1.115（崩盘 −2.64），均值掉到 baseline 之下。**单 seed 仍不可信，铁律。**
2. **均值横比**：holdout 2 seed 1.719 < baseline 4 seed 2.108。worstbin 均值 −1.00 优于 baseline −1.51，但都是负（整体仍会崩盘）。**当前数据下 holdout 未超 baseline。**
3. **机制本身无问题**：考卷收集/p_holdout/升降档端到端跑通且两 seed 课程轨迹逐帧一致；问题是**性能层面 seed 方差，不是机制 bug**。seed1 证明这套机制能达到 2.323+Nav 全 1.0 的强结果，但不稳定。
4. **N 仍不够**：2 seed 判不出真均值（标准误巨大）。要坐实必须补到 ≥4 seed 与 baseline 同 N（P1 待办）。

**vs s4tight 四档（vcb0osvi，§6，均单 seed return 代理口径）**：s4tight 也只有单 seed（CVaR10 2.379 / worstbin +0.74），**同样未补 seed、同样不能采信单 seed**。holdout 的 seed 方差给 s4tight 也敲了警钟——s4tight 那个 +0.74 很可能也是高 seed 的幸运值，必须补 seed 才知道。

| 指标 | s4tight 四档 s? | holdout s0 | holdout s1 | holdout 均值 |
|---|---|---|---|---|
| CVaR10 | 2.379（单seed） | 1.115 | 2.323 | **1.719** |
| worstbin | +0.74 | −2.64 | +0.63 | −1.00 |
| overall_win | 0.996 | 0.972 | 0.998 | 0.985 |

**旧 currA 数据下线**：上一版本此处列 currA 3 run（CVaR10 均值 1.746），已被 holdout/s4tight 路线取代，移出主横比。三条路线（currA 1.746 / holdout 1.719 / baseline 2.108）目前都在 baseline 之下或持平，**无任何 generator 路线在多 seed 均值上确证超过 baseline**——这是当前最诚实的状态。

---

## 2. 数据口径与 run 清单（避免重复踩坑）

### baseline = SFL 本身
- `GENERATOR_INJECTION=false` 走 learnability top-K 海选 = SFL 算法（非退化 DR）。
- **确定性**：同 seed → 完全相同结果。故 `baseline-dump` 与 `s4p2-base` 的 seed1/seed2 CVaR 逐字相同（2.212/2.752），是同一批的复制品，去重后 baseline 实际 **4 个独立 seed（0/1/2/3）**。
- baseline 配置 = **SFL 官方仓库 `amacrutherford/sampling-for-learnability` 的默认值，一字未改**（Michael Beukman commit）：L=1000, ρ=0.5(FROM_SAMPLED=128/256), LR=2.5e-4。⚠️ 官方默认 ≠ 论文 Table 4（L=2000, ρ=1.0, LR=2.4e-4）——这是 SFL 代码 vs 论文的差异，非本项目问题。

### ★★★SEED 语义 = 固定可复现的确定随机实例（2026-06-29，方法论更新，关键）

> **核心**：SEED 是 JAX PRNG 种子，派生**整个训练的所有随机源**（网络初始化 / PPO 采样 / generator 采样 / minibatch shuffle / 环境关）。**不是只控环境随机。**

1. **配置相同 + SEED 相同 → 完全可复现（逐字相同结果）。** 故"seed0 崩、seed1 好"这类**不是随机噪声，是该 (配置,seed) 实例的确定特性**——可重查、可针对性追查。**→ 今后分析可直接针对"某 seed 为何崩"找线索**（崩盘关结构、该 seed 训练轨迹分叉点），而非笼统归为"seed 方差"。每个 (配置,seed) = 一次独立可复现抽样。
2. **⚠️ 配置不同 + SEED 相同 ≠ 同一随机个体**。如 E1 seed0 vs E2 seed0：相同 SEED 只保证**训练起点**（初始网络权重 + 第一批关）相同，**第一步梯度更新后轨迹即发散**（clip/ρ 不同→权重不同→之后采样/generator/一切全不同）。**故不能把跨配置的同编号 seed 当"同一随机实例换一个旋钮"比**。
3. **由 2 推出的铁律**：观察"E1 崩 s0、E2/combo1 崩 s1"（崩盘 seed 编号不固定）**不能推出"崩盘与 seed 无关"**——它们是 6 次独立抽样，不是"同一 seed 换配置"。崩盘共性须从**崩盘关的结构**找（per-level 诊断），不能从 seed 编号找。

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

### 5.4 ★两个 4stage run 结果（2026-06-27，COMPLETED，return 代理口径，单 seed）

两 job 都跑满 step2250。run：s4tight(3455019)=**desert-oath-168/vcb0osvi**；s4only(3455020)=glamorous-capybara-168/386tm8f4。

| run | 配置 | CVaR10 | worstbin | 档位轨迹(停留占比) | 首次到顶 |
|---|---|---|---|---|---|
| SFL_paper (v3raza8r) | — | 2.319 | −1.02 | 无课程 | — |
| **B1 三档 (t4t47j3k)** | τ0.6/s2 旧最佳 | **2.447** | +0.35 | {0:8.9%,1:4.4%,**2:86.7%**} | @6/45 |
| **s4tight 四档 (vcb0osvi)** | τ0.8/s3 主菜 | 2.379 | **+0.74** | {0:22%,1:6.7%,2:6.7%,**3:64.4%**} | @16/45 |
| s4only 四档 (386tm8f4) | τ0.6/s2 对照 | **0.609** ❌ | −5.56 | {0:11%,1:4.4%,2:4.4%,**3:80%**} | @9/45 |

**★课程升降档逐次事件（curriculum_stage 全轨迹，wandb vcb0osvi/386tm8f4/t4t47j3k）：**
| run | 初始档 | 升档事件 | 降档事件 | 末档 | 冲顶后剩余训练 |
|---|---|---|---|---|---|
| **s4tight 四档** | 0@step50 | ↑0→1@550, ↑1→2@700, ↑2→3@850 | **无（从未降档）** | 3@2250 | step850→2250 ≈ **1400 步(62%)钉死顶档零变化** |
| s4only 四档 | 0 | 三次升档更早(更松)冲到顶档3@~step≤850 | **无** | 3 | 80% 训练在顶档 |
| B1 三档 | 0 | ↑冲顶极早 first_top@6/45(≈step300) | **无** | 2 | 86.7% 训练钉死顶档2 |

**共性铁律**：三个 run **全程只升档、从不降档**（ALLOW_DEMOTE=true 但备而未用），都在训练前 1/3 内一口气冲到顶档后**纹丝不动**。
- **从不降档的铁证**：`gen_p_low_frac`(frac p<0.2，控降档触发) 0.973→mid 0.089→last 0.023 —— 学生学会基础导航后**永远够不到降档阈值 τ**，降档逻辑形同虚设。
- **秒冲顶的铁证**：`gen_p_high_frac`(frac p>0.8，控升档) 0.025→mid 0.906→last 0.976 —— 学生学会后 90%+ 关秒变 trivial，远超升档阈值，课程除一路升顶别无选择。
- s4tight 的"调严"(τ0.8+streak3)唯一作用=把升档**整体推迟**(@550/700/850 vs B1 的≈step300)、底档多停留(9%→22%)，但治标不治本。**自适应课程退化成了一次性单调爬坡，不是真课程。**

**归因三角结论：**
1. **拆档本身无效/有害**：s4only（只拆档不调严）0.609 暴跌、worstbin −5.56，是这批最差；档位轨迹 80% 钉死顶档，和 B1 三档(86.7%)几乎一样秒冲顶 → **多给一档照样秒冲到顶，拆档没让中间档多停留**。
2. **调严有方向性效果但不足**：s4tight（τ0.8+streak3）把冲顶从 @6 推迟到 @16、底档停留 9%→22%，**但中间两档(1,2)合计仍仅 13.4%**，顶档仍 64%。"调严"治标(推迟冲顶)不治本。
3. **谁都没超过 B1**：s4tight 2.379 < B1 2.447。

**★为什么 s4tight 的 CVaR10 略低、worstbin 却明显高（两指标反向）：**
- worstbin = returns 直方图**最左端点**（单个最惨关）；CVaR10 = 最差 **10% 一整段**的均值。
- s4tight 把"灾难性崩盘的极少数关救起来"（端点 +0.74，全场唯一连最惨关都正收益），代价是"中等偏难那批关整体略降"，把最难 10% 段均值拖下 3%。
- 机制对应：调严让学生在低/中档**多待一会**、基础更牢 → **最极端崩盘被托住**(worstbin↑)；但冲顶后仍钉死顶档 62%、渐进课程没真发生 → 难关整段没系统受益(CVaR10↓)。
- **2.379 vs 2.447 仅差 3%，单 seed，完全在 seed 噪声内**（§1 baseline 单 seed 能 1.27↔2.75 翻盘）→ 目前**判不出 s4tight 比 B1 好或差，只能说打平、尾部形态不同**。

**为何升档秒冲顶 + 从不降档（铁证）**：`gen_p_low_frac`(控降档) 0.973→0.089→0.023，学生学会后**永远够不到降档阈值 τ**；`gen_p_high_frac`(控升档) 0.025→0.906→0.976，学生学会后 90%+ 关秒变 trivial、远超升档阈值 → 课程除一路升顶别无选择。**根因不在课程调度参数，在 generator 生成分布塌缩**（见下）。

### 5.5 ★s4tight 真实迷宫 + p 轨迹（2026-06-27，已存 `mazes/s4tight-4stage-tau0_8-streak3/`）

迷宫图：每段 20 关全图 early_50/mid_1150/late_2250（命名对齐 B1=`mazes/adaptcurr-nogate/` 可横比）。源 dump=`_maze_dump/<group>/seed0/inj_*.npz`，是 `log_buffer` 后的 `instances_top`=**真注入 buffer 的关**。

**裸 p 轨迹（3000 关注入池分桶，get_p.py vcb0osvi）：**
| step | 50 | 350 | 800 | 1150 | 1600 | 2250 |
|---|---|---|---|---|---|---|
| p≈0 | 97% | 27% | 4% | 9% | 6% | 2% |
| 中间(可学区) | <1% | ~2% | <1% | ~1% | <1% | <1% |
| p≈1 | 2% | 71% | 96% | 90% | 94% | 97% |

形态=**"全难急塌为全易"**：早期 generator 堆一池 p≈0 关(学生菜全失败)，学生学会基础导航后**同批结构瞬间全变 p≈1**；可学区从头到尾 <2%，从未建立。坐实 [[generator-pool-collapses-to-extremes]]。

**⚠️ 自我纠错（重要，防再犯）**：本轮我一度用**手写 BFS** 判注入关"半数不可达、generator 造退化畸形关、闸失效"——**全错，已撤回**。①肉眼看迷宫图全部可达、是正常连通迷宫；②即便换成项目**修复后的** `bfs_dist_field`(从训练同款 `sfl/train/pcgrl_generator.py` import) 复算结果一致，说明那不是 BFS 实现差异，而是我**拿"起终点随机落不同连通块"统计现象去脑补成 bug**，且没核实注入档 valid_path_check 配置。**结论：不对注入关做可达性/geo 断言**，迷宫图只渲染 墙/S/G/朝向，让图说话。[[component-mask-fragmented-graph-bug]] 的修复已在代码里生效，不要再把可达性问题往那上面套。

**能站住的迷宫观察（仅定性、不带可达性指控）**：early_50 起终点多挤在局部(generator 早期没学会布局)；mid_1150 出现需绕行的长程关(结构质量较好)；late_2250 仍在顶档但好关没系统性保持。配合 p轨迹+档位轨迹三者一致 → **课程在调测地阈值，但 generator 产出分布不响应这根轴**=档位怎么调都超不过 B1 的根因。

### 5.6 ★★★信号区分度实验：回答"p(1-p)塌缩为何仍赢" + 课程判据该换 CENIE（2026-06-27，job 3456683）

**实验**：对 s4tight 注入关(ckpt1150/2250 各 100 关)重跑 student rollout，分 p 桶算 PVL/anchored-PVL/|adv|/CENIE 的桶内 std + Spearman(信号,1-p) + 训练时间轴饱和度。脚本 `signal_discrim.py`(GPU sbatch)。CENIE 用当前 ckpt 在该批 hidden 现拟 GMM(近似,注入时未落盘)。

**发现1（悖论核心答案）：p 两极桶内(p(1-p)≡0)所有梯度类信号仍有大 std = p(1-p)是盲投影。**
ckpt1150 的 p≈0 桶(n=22,22关全p≈0、SFL眼里"完全无差别")：**|adv| std=1.516（≈桶均值1.97的77%）、pvl std=0.845、cenie std=100**。即这22个"看起来一样失败"的关**梯度幅度天差地别**。→ **p≈0 不是零学习信号**，generator 注入的 p=0 关里藏着高 advantage 关(学生预测错得离谱=大梯度)。**候选②(梯度幅度是本质、p(1-p)是其二元盲投影)实锤。**

**发现2（反转，更深）：Spearman(PVL,1-p)=−0.68 是负的 → SFL 真盲区在 p≈1 一侧。**
PVL 越高失败率越低(p越高)。机制：student 已强，p≈0 关价值预测稳定预期失败、无 surprise → PVL 低(0.31/0.07)；真正高 PVL 的是"快做对但没完全对"(p偏高)的关。**而 p(1-p) 在 p≈1 也给 0 分，把这批高 PVL 的"快会"关当 trivial 丢掉**。你的 anchored-PVL 选关把它们捞起来 = **塌向 p≈1 仍赢的直接机制**。SFL 的 p(1-p) 双零盲区(p≈0 和 p≈1 都给0)恰好漏掉 p≈1 侧的可学关。

**发现3（课程判据，决定性）：PVL/adv 会饱和、CENIE 不饱和 → 判据应换 CENIE。**
| 信号 | std@1150 | std@2250 | 比值 |
|---|---|---|---|
| pvl | 1.53 | 0.79 | **0.51 腰斩** |
| anchored | 1.23 | 0.73 | 0.59 |
| absadv | 0.92 | 0.62 | 0.67 |
| **cenie** | 84.1 | 94.0 | **1.12 不降反升** |

PVL/adv 随 student 变强而价值预测变准 → 整体塌下、**后期失分辨率**(和 p 一样的病)。CENIE(结构新颖性=hidden落已访问分布稀有处)**与student强弱无关、全程保持分辨率**。

**→ 课程升降档判据的根因诊断**：当前用 `frac(p>0.8)`/`frac(p<0.2)`(p判据)，与被诊断的病(p盲)同源，且 p 后期饱和 → 自适应课程退化成单调爬坡(§5.4铁证)。**换 PVL 也不行(同样饱和)**。**唯一全程不饱和的方向盘是 CENIE。** 这是比"调τ/streak"深一层的改法：根因是判据信号饱和，非阈值松紧。

**本质特征结论（双轴）**：①可学量本质=**梯度幅度(PVL/adv)**，p(1-p)是其退化盲投影，且漏 p≈1 侧；②但梯度信号饱和，长期课程方向盘须用**CENIE(结构新颖性,不饱和)**。①②=候选②(梯度)+候选①(结构多样性)分工，非竞争。候选④(定向命中Nav2/3)已被用户证伪(CVaR是100关尾部非3关)。

**⚠️ caveat**：单 seed、单 run(s4tight)、CENIE 用现拟 GMM 近似(注入时 GMM 未落盘)。Spearman 负号需在 B1/SFL 上复核是否普适。但 p 两极桶内 std>0 这个核心事实稳健(不依赖 GMM)。

### 5.7 ★pvl/absadv 全过程记录代码改动 + |adv| 利用方案 + 待办（2026-06-27）

**代码改动（已 scp Oscar + py_compile + import 自检 PASS，备份 `.bak_pre_advlog`）**：
之后所有 run 除 p 外**同步落 per-level 裸PVL 和 mean|adv| 的全过程**（用户要求1）。零额外 rollout——从 `all_signals_on_levels` 已有的 traj/adv 顺路算：
- `pcgrl_generator.py`：`all_signals_on_levels` 在 line~775(裸PVL覆盖前)存 `pvl_raw_perlevel` + 算 `absadv_perlevel=mean|adv|(T,agent维)`，返回 dict 带出；`get_generator_set` line~1893 仿 gen_p_* 塞 `gen_pvl_mean/median/perlevel` + `gen_absadv_mean/median/perlevel`。
- `jaxnav_sfl.py`：转发白名单加 4 标量键 + 仿 gen_p_hist 加 `gen_pvl_hist`/`gen_absadv_hist`（无损分桶直方图）。
- ⚠ 仅 auction 路径(_ainfo 带出)+adv 分支(anchored/pvl/anchored_cenie/...)有定义；euc/baseline 路径键缺失自动跳过（零回归）。**当前所有在跑 arm 都走 anchored_cenie，覆盖。**
- ⚠ s4tight/B1 是旧代码跑的**只有 p**，pvl/absadv 全过程**须等新 run**。

**★|adv| 的利用方案（用户问2，定位=与 CENIE 互补的第二把刀）**：
|adv|=梯度幅度绝对量（不分高估低估），与 PVL(只看低估=regret)不同。实验铁证 |adv| 在 **p≈0 桶 std 最大(1.516>PVL 0.845)** → **它是"看似全失败"区分辨率最高的刀**，正好补 p(1-p)+PVL 都看不清的 p≈0 区。两个用法：
1. **选关层（最有价值）**：auction 加一路 |adv| estimator 专攻 p≈0 侧高梯度真难关 + 保留 anchored-PVL(管 p≈1 侧"快会"关) → **两侧夹击覆盖 p(1-p) 双零盲区**。直接补 SFL "p≈0 和 p≈1 都给0分"的致命漏洞。
2. **课程层（局部门控）**：CENIE 管"升档去哪(探索方向,不饱和)"，|adv| 管"这档练够没(火候)"——mean|adv|还高=还有梯度没吃完=别升档；掉到平台=榨干=升档。本质优于 p 判据(p说"做对了就升",|adv|说"没东西可学了才升")。**坑**：|adv| 也轻微饱和(0.92→0.62)，故只当局部门控、不当长期方向盘，须配不饱和的 CENIE。
- 推荐组合：选关层 auction=[|adv|(攻p≈0), anchored-PVL(攻p≈1), cenie(多样性)] 三路覆盖三区；课程层 升档=CENIE、停留/退档=mean|adv|，**彻底弃 p 判据**。

**待办（用户要求3，需新 run 数据）**：
- 升级 `get_p.py`→取 p/pvl/absadv 三量全过程轨迹（gen_*_hist 无损分桶，[[how-to-fetch-raw-p-genphist]] 范式），按 step + 课程升降档(curriculum_stage)对照画三量演化，看 pvl/absadv 全过程如何变、如何影响升降档。**等下一个新代码 run 跑完才有数据。**
- 优先级建议：先复跑 signal_discrim 在 B1/SFL 复核 Spearman 负号普适性(§5.6 caveat)；再起新 run 验 |adv| 选关 / CENIE 课程判据。

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

---

## 7. ★★★顶档信号诊断：冲顶不是病 + Δ|adv| 是吃饱判据（2026-06-28，job 3474994 `signal_topdwell.py`）

**实验**：B1(resilient-blaze-166) + s4tight(desert-oath-168) 各取 4 个 ckpt(step 600/1150/1700/2250），**各用自己末期顶档注入关(inj_2150/2200/2250 三帧=60关)做固定考卷**，4 个 ckpt 全在这同一批关上重跑 student rollout → 算 p/|adv|/CENIE 绝对值 + Δp/Δ|adv|(相邻ckpt) + 逐关"吃饱vs学不会"判别。固定考卷是关键：所有 ckpt 同批关才能算干净的 Δ。

### 顶档纵向数据（固定考卷）

| ckpt | **B1** p | B1 \|adv\| | B1 CENIE | **s4tight** p | s4t \|adv\| | s4t CENIE |
|---|---|---|---|---|---|---|
| 600 | 0.567 | 1.978 | 95.4 | 0.598 | 2.498 | 76.7 |
| 1150 | 0.665 | 2.276 | 104.7 | 0.723 | 2.764 | 103.6 |
| 1700 | 0.848 | 2.236 | 114.6 | 0.800 | 2.520 | 102.8 |
| 2250 | **0.967** | **2.468↑** | 98.3 | **0.917** | **2.118↓** | 95.6 |

Δ|adv| 末段(1700→2250)：**B1 = +0.232（升）** vs **s4tight = −0.402（降）**。

### 三个关键结论（用户确认）

1. **★"冲到顶档"不是坏事——顶档关压根没"学不会"的。** 固定考卷上 B1 顶档 p 从 0.567 **单调爬到 0.967**、末段仍在涨（Δp 全程>0）。末期判别：还在学 13-17%，停滞 78-85%，而**停滞里 83% 是吃饱(p>0.8)，"学不会"(p<0.2)只有 2-3%（1-2 关）**，且那 1-2 关 |adv|≈0.1（不是有梯度没吃到，是真·无解关，student 已放弃）。**→ 顶档钉死不是病，是 student 在顶档持续吃到东西的证据；"降档"在顶档几乎无触发场景（推翻"该修降档"的旧方向，§5.4/§5.6 的"自适应退化成单调爬坡"不再是缺陷而是正常）。**
2. **★|adv| 上 B1 优于 s4tight，且与 CVaR 输赢同向。** 两 run p 都爬到 ~0.9+，但 **B1 末段 |adv| 回升(2.236→2.468，还有梯度可吃=没吃饱) vs s4tight 持续衰减(2.52→2.12，梯度榨干=已吃饱)**。对应 CVaR：B1 2.447（还在涨没到顶）> s4tight 2.379（已榨干）。**Δ|adv| 是唯一能区分"还在吃 vs 榨干"且与输赢方向一致的量。**
3. **★p 仅在注入关坍塌（都是新关），固定考卷上不坍塌。** §5.6"p 饱和/两极塌缩"是针对**注入池**（每帧新关、generator 分布塌两极）；**固定考卷的 p 是连续单调不饱和的掌握度曲线**。重要订正：p 本身没病，"p 不能用"只在注入池语义下成立。

### 5 个量的顶档诊断价值排序

| 量 | 顶档诊断价值 |
|---|---|
| **Δ\|adv\|** | ✅✅ **最优"吃饱"判据**：>0=还有梯度别动(B1)；<0=榨干该加难(s4tight)。直接编码火候，与 CVaR 同向 |
| **\|adv\|** | ✅ 关键：绝对值升降区分"还在吃 vs 榨干" |
| **p(固定考卷)** | ✅ 意外好用：单调不饱和量掌握度；但只说"会不会"，不说"还有没有可学" |
| Δp | ⚠️ 顶档全程>0、区分不出（没真停滞），分辨率低 |
| CENIE | ⚠️ 两 run 都先升后落(非单调)，顶档纵向不如 |adv| 干净 |

### → 超参/机制调优方向（基于本诊断）

- **弃"降档"思路**：顶档不需要降档（学不会只占 2-3% 且是无解关）。课程的事是**"何时再加难"**，不是"何时退"。
- **吃饱判据换 Δ|adv|**：mean|adv| 掉头向下（Δ|adv|<0 连 streak 次）= 榨干 = 该升档/加难；仍在升 = 别动。替换当前 frac(p>0.8) 判据。
- **B1 可能"还没吃饱就训练结束"**：B1 末段 |adv| 仍在升 → 顶档梯度没吃完。**延长训练 / 给更难的关 / 升 thr** 可能让 B1 更高（当前 2.447 或非其上限）。

**⚠️ caveat**：B1/s4tight 均单 seed；CENIE 用现拟 GMM 近似；4 ckpt 粗粒度(间隔~24%训练/段)。Δ|adv| 趋势稳健但精细拐点需更密 ckpt。

---

## 8. ★★★调参方法论 + 公平性宪法（2026-06-28，核对 SFL 论文 Table 4）

> **选型已定**：后续主力 = **B1（三档无下界 adaptcurr-nogate，CVaR 2.447）/ s4tight（四档 τ0.8 streak3，CVaR 2.379）**。调参围绕 SFL 论文 Table 4 这类细节参数进行。

### 8.1 ★PPO 超参不该动（核对论文 Table 4 后的硬结论）

核对 SFL 论文（arXiv:2408.15099 v3）Appendix C Table 4「单智能体 JaxNav」官方 PPO 值 vs 我们 yaml：

| 参数 | 论文官方(JaxNav单agent) | 我们 yaml | 一致 |
|---|---|---|---|
| **PPO clip range (CLIP_EPS)** | **0.04** | 0.04 | ✅ |
| Adam LR | **2.4e-4** | sbatch覆盖2.4e-4 | ✅ |
| **entropy coef** | **0.0** | 0.0 | ✅ |
| PPO epochs / minibatches | 4 / 4 | 4 / 4 | ✅ |
| PPO steps / # envs | 512 / 256 | 512 / 256 | ✅ |
| γ / λGAE | 0.99 / 0.95 | 0.99 / 0.95 | ✅ |
| Anneal LR / max grad norm / VF coef | yes / 0.5 / 0.5 | True / 0.5 / 0.5 | ✅ |
| FC / Hidden dim | 512 / 512 | 512 / 512 | ✅ |
| Number of Updates | 2250 | 3e8→2250 | ✅ |

**★重大自我纠正**：曾判定「CLIP_EPS=0.04 异常低、是可疑 bug、该调 0.1-0.2；ENT_COEF=0 该开」——**全错，已撤回**。论文 C.1 明写「conducted an **extensive sweep** on JaxNav ensuring robust DR performance, tuning **only UED-specific** parameters」。0.04 clip + 0.0 熵是作者在 JaxNav（连续控制+高度部分可观测 LiDAR）上专门扫出的最优，不是疏忽。**拿 Minigrid/Atari 的"常规 0.2"去套已调好的连续导航环境是错误的。PPO 超参已与论文对齐，不动。**

### 8.2 ★★★公平性宪法（顶会 NeurIPS/AAAI 衡量下"能否声称打败 SFL"）

**核心问题**：若 new method 的最优 CLIP_EPS 与 SFL 不同（如调大更好），能否声称赢 SFL？

**答案：能，但有严格前提，否则是 unfair comparison（desk-reject 级硬伤）。**

- SFL 论文协议（C.1）：**PPO 超参对所有方法统一固定，只调 UED 专属参数**。这是 UED 领域标准公平协议。
- 真正的公平标准不是"统一超参"，而是 **each method tuned to its best with equal tuning budget**（各调到最优、预算对等）。
- **要让审稿人买账须证明三件事**：① SFL 也重新 sweep 该超参取其最优、your method 也 sweep 取最优，两个**最优对最优**比（不能"我调了 SFL 卡 0.04"）；② **调参预算对等**（不能给自己扫 20 点给 SFL 扫 3 点）；③ **解释为何你的方法需要不同超参，且理由 intrinsic**（弱="试出来 0.1 好"=p-hacking；强="我的 generator 造的关测地更长/更难【数据】，student 梯度信号更强【§7 |adv| 数据】，故需更大 clip"）。

**最干净的 2×2 对照**（排除"超参本身带来的提升"）：

| 实验 | 目的 |
|---|---|
| SFL @ SFL最优clip(0.04) | SFL 真实战力 |
| SFL @ your最优clip | 排除"clip 本身的提升" |
| yours @ 0.04 | 机制在统一底座下赢不赢 |
| yours @ your最优clip | 你的真实战力 |

- **若 yours@0.04 就赢 SFL@0.04（统一底座、机制单独赢）= 最强论证、无可辩驳。**
- 若 yours 只在自己 clip 下赢、yours@0.04 输 → 必须走"内在性质"论证 + 给 SFL 也调 clip，审稿更严。

**★实操优先级（诚实）**：当前 generator 路线在统一 0.04 下尚未多 seed 坐实赢 SFL（B1 单 seed 2.447 vs SFL 2.108）。**故先在统一 0.04 底座下多 seed 把"机制赢"坐实（最干净路径，也=§4 P1 补 seed），别急着追超参甜点**（那是更难被审稿人接受的路）。

### 8.3 ★论文背书的可调方向（Appendix I 消融，有数据支撑）

**UED/SFL 专属参数**。论文 Appendix I 已扫，对 JaxNav 的结论：

| 参数 | 论文官方值 | 论文消融结论(JaxNav) | 我们现状 | 调优方向 |
|---|---|---|---|---|
| **N(候选池/BatchSize)** | **5000** | "Sampling more levels → improved performance"(图24单调↑) | generator pool **3000** | **★加到5000+**(论文直接背书，且=§4"故意弱化"修正) |
| **ρ(sample ratio)** | 1.0 | "higher ρ preferable on JaxNav" | 已1.0 | ✅不动 |
| **K(buffer size)** | 100 | "smaller buffer outperforms larger on JaxNav" | NUM_TO_SAVE 60 | ✅偏小方向对 |
| **L(rollout length)** | 2000 | "L=1000略降，4000无增益vs2000" | 已2000 | ✅不动 |
| **T(update period)** | 50 | "增大T降性能" | EVAL_FREQ 50 | ✅不动 |
| **降序选关** | — | "decreasing order of learnability > 随机top-K"(图24末格) | 我们 auction top-K | ⚠️可试：注入按信号降序而非随机 |

**结论**：① **候选池 3000→5000** 是唯一有论文单调背书的调优点（且修正"故意弱化"）；② **按信号降序注入**（非随机 top-K）论文证更好，可低成本试；③ 其余 SFL 专属参数已在论文最优点附近，不动。

### 8.4 ★★★完整可调参数总表（2026-06-28，核实 B1=t4t47j3k 实跑值 + 论文 Table4）

> 后续调参查阅基准。**🔒锁死=不动 / 🟡UED可调 / 🎯generator自有(主战场)**。当前值=B1 实跑值（wandb 核实）。

#### A. PPO 底座（混合：🔒纯共享底座锁死 / 🔬因训练分布改变而最优点可能漂移者→做对等sweep）

**★关键区分（修正前一版把 CLIP_EPS 误标🔒锁死，与 §8.2 自相矛盾）**：我们的方法**本质改变了训练分布**（generator 造更难长程关 vs SFL 海选随机关）。「统一 PPO」公平协议的前提是"所有方法看同一训练分布"——**我们打破了这个前提**，故对训练分布敏感的 PPO 参数（首推 CLIP_EPS）**不该锁死，而该做"对 SFL 和 ours 对等 sweep、各取最优"的实验**（§8.2 的 2×2 对照）。unfair 是"给自己调给 SFL 卡死"，不是"sweep 本身"。

| 参数 | 当前值(B1) | 论文Table4 | 判定 |
|---|---|---|---|
| **CLIP_EPS** | 0.04 | 0.04 | 🔬**对等sweep**：训练分布变难→最优clip可能漂移；扫{0.04,0.1,0.2}×{SFL,ours}各取最优(§8.2)。论证须配 §7 \|adv\| 数据(难关梯度更强→需更大clip)=intrinsic 理由 |
| ENT_COEF | 0.0 | 0.0 | 🔬对等sweep(次优先)：更难分布或需更多探索；同样须对SFL对等扫 |
| LR (learning.LR) | 2.4e-4 | 2.4e-4 | 🔬对等sweep(次优先)：anneal下可一并看 |
| UPDATE_EPOCHS | 4 | 4 | 🔬可纳入(难关多榨数据)，对等扫 |
| ANNEAL_LR | True | yes | 🔒锁(协议共享) |
| NUM_MINIBATCHES | 4 | 4 | 🔒锁 |
| NUM_STEPS | 512 | 512 | 🔒锁 |
| NUM_ENVS | 256 | 256 | 🔒锁 |
| GAMMA / GAE_LAMBDA | 0.99 / 0.95 | 0.99 / 0.95 | 🔒锁 |
| MAX_GRAD_NORM | 0.5 | 0.5 | 🔒锁 |
| VF_COEF | 0.5 | 0.5 | 🔒锁 |
| FC_DIM / HIDDEN_SIZE | 512 / 512 | 512 / 512 | 🔒锁 |
| CLIP_EPS_SCALE / USE_LAYER_NORM | False / False | — | 🔒锁(论文未开) |
| TOTAL_TIMESTEPS | 3e8(=2250 upd) | 2250 upd | 🔬§7延长训练实验候选 |

**🔬对等sweep层的铁律**：① 对 SFL 和 ours **同等预算**各扫一遍；② 报告 2×2（SFL@SFL最优 / SFL@ours最优 / ours@0.04 / ours@ours最优）；③ 最强主张=ours@0.04 已赢 SFL@0.04（机制单独赢），其上再叠超参增益；④ 若仅 ours@自己clip赢，须用 intrinsic 理由（§7 难关\|adv\|更强）站住。**先做哪个：仍是先统一0.04多seed坐实机制赢，再做CLIP_EPS对等sweep加分**（顺序不变，但 CLIP_EPS 是确定要做的实验、非锁死项）。

#### B. SFL/UED 专属参数（🟡可调，但多数已在论文最优附近）

| 参数 | 当前值(B1) | 论文官方 | 论文消融结论 | 判定 |
|---|---|---|---|---|
| **N 候选池**(generator pool=POOL_PER_GEN×N_gen) | **3000**(1500×2) | **5000** | 更大→更好(图24单调↑) | 🎯**★加到5000** |
| ρ sample ratio(NUM_ENVS_FROM_SAMPLED/总) | 1.0(256/256,TO_GEN=0) | 1.0 | 高ρ更好 | 🟡已最优,不动 |
| L ROLLOUT_STEPS | 2000 | 2000 | 1000略降/4000无增益 | 🟡已最优,不动 |
| T EVAL_FREQ(update period) | 50 | 50 | 增大降性能 | 🟡已最优,不动 |
| K buffer(GEN_NUM_TO_SAVE) | 60 | 100 | 小buffer更好 | 🟡偏小方向对,不动 |
| BATCH_SIZE(海选池,baseline路径) | 1000 | — | — | 🟡generator路径不走 |
| 注入选关方式 | auction随机top-K | 降序 | 降序>随机top-K(图24末格) | 🟡可低成本试 |

#### C. Generator / 课程 / auction 专属（🎯主战场——论文无此参数，new method 贡献面，不涉公平性）

| 参数 | 当前值(B1) | 说明 | 调优空间 |
|---|---|---|---|
| **CURRICULUM_THR_STAGES** | [5,10,22](B1三档) | 课程档位(测地格数) | 🎯三档vs四档(§0三档更优)；顶档22是否对齐test真实测地(待测) |
| **CURRICULUM_HOLDOUT_TAU_UP** | (B1用frac判据,旧) | 升档阈值 | 🎯§5.4"调严"有益,试0.8/0.85/0.9 |
| **CURRICULUM_STREAK** | 2(B1) | 连续N次升档 | 🎯试3/5(s4tight用3有益) |
| CURRICULUM_BETA | 1.0 | 塑形项强度-β·relu(geo-thr) | 🎯可扫 |
| CURRICULUM_ARM | geo | 难度轴=测地绝对长度 | 🟡§5.2证geo唯一干净轴,不换 |
| CURRICULUM_ADAPTIVE / ALLOW_DEMOTE | True / True | 自适应升降档 | 🟡§7证降档基本无触发,保留 |
| 课程判据 | frac(p)(B1) / holdout(新) | 升降档信号源 | 🎯§7:Δ\|adv\|判"顶档吃饱"(用于延长/加难,非升档闸) |
| **AUCTION_SIGNAL_MODE** | anchored(B1!) | auction信号集 | 🎯B1=anchored,后续run=absadv_anchored_cenie,须对齐 |
| **GEN_ESTIMATOR_IDS** | [anchored,cenie](B1) | N路estimator | 🎯§5.7建议[\|adv\|,anchored,cenie]三路夹击 |
| GEN_AUCTION_LAMBDA | 1.0 | auction温度 | 🟡§auction-ablation证打分层无差异,Stage4定 |
| AUCTION_USE_CENIE | True | CENIE维开关 | 🟡保留 |
| GEN_POOL_PER_GEN | 1500 | 每gen候选池(→N总) | 🎯★1500→2500(配ROLLOUT_CHUNK避OOM) |
| GEN_NUM_LEVELS_PER_GEN | 64 | 每gen PPO级数 | 🟡 |
| GEN_ROLLOUT_CHUNK | 250 | 分批避OOM | 🟡随pool调 |
| GEN_PPO_EPOCHS / GEN_LR / GEN_OUTER_STEPS | 4 / 2.5e-4 / 1 | generator自身PPO | 🟡正交于student,暂不调 |
| GEN_MAP_SIZE / GEN_MAX_BOARD_SCANS | 11 / 3 | 关卡尺寸/编辑遍数 | 🔒环境对齐,不动 |

#### ★调参执行顺序（信号强度排序）

1. **先补 seed**（前提）：B1/s4tight 各 ≥3 seed，否则调参被单 seed 方差淹没（§1铁律/§8.2）。
2. **候选池 3000→5000**（论文唯一单调背书 + 修"故意弱化"）。
3. **课程"调严" τ_up/streak**（§5.4 方向有益，扫 τ∈{0.8,0.9}×streak∈{3,5}）。
4. **estimator 三路** [\|adv\|,anchored,cenie]（§5.7，攻 p≈0/p≈1/多样性三区）。
5. （可选）降序注入、Δ\|adv\| 判顶档延长/加难。

每步**单变量改 + 多 seed**，不一次动多个。

### 8.5 ★★★参数含义 + 对"崩盘/吃不饱"双目标的对症性（2026-06-29，用户要求逐参数解释）

> **两个待解问题的精确定义**：
> - **崩盘**：overall_win 健康（绝大多数关会做）但**最难 ~10% 关灾难失败**（worstbin 大负，如 E2 s1 = −4.71）。PPO 诊断量(approx_kl/ratio/clipfrac)全程健康 → **不是优化发散，是策略收敛到"尾部脆"的解**（在难关上自信地犯错）。
> - **吃不饱**（§7）：训练结束时固定考卷上 |adv| 还在升 = 顶档关里还有梯度没吃完。两条解法：(a) 更多时间/更新次数榨现有关；(b) 持续供给新难度的关。
> **⚠️ 张力**：榨梯度（修吃不饱）与保守更新（修崩盘）常是反向旋钮，难同时调。唯一机制上同时利好两者的 = **|adv| 进 bid 选关**（持续供给高梯度关 + 补尾部盲区），且**从未正式跑过**（§8.6）。

#### A. PPO 底座块（含 §8.4 漏讲的 PPO 参数）

| 参数 | 当前值 | 含义 | 对**崩盘**作用 | 对**吃不饱**作用 |
|---|---|---|---|---|
| **CLIP_EPS** | 0.04 | PPO 信任域。ratio 限在[1−ε,1+ε]，ε大=每步策略变化大(激进) | 🎯**核心**：ε=0.1 增大seed方差(E2 s0更好/s1崩)；**调小0.02**或更稳。论文0.04为DR调 | ε大→步子大→吃更快，但易冲过头 |
| **ENT_COEF** | 0.0 | 熵正则。>0鼓励探索，=0允许策略变确定 | 🎯🎯**最对症**：=0→难关上策略过度确定地走错；**开0.001~0.01**强制保留探索防"自信犯错" | 探索多→可能发现新可学方向 |
| **LR** | 2.4e-4 | 学习率/步长 | 🎯调小→收敛更平坦(鲁棒)解；大→易到尖锐(脆)解 | 大→吃更快但易过冲 |
| **MAX_GRAD_NORM** | 0.5 | 梯度裁剪上限 | 🎯**候选**：难关大advantage→大梯度→**调小0.25**让更新更稳防推到坏区 | 限制单步幅度 |
| **UPDATE_EPOCHS** | 4 | 每批数据重复训几遍 | 调大→过拟合当前批→**尾部罕见关反而欠拟合** | (b)路榨梯度:调大同批榨更多,但与崩盘同向风险 |
| **NUM_MINIBATCHES** | 4 | 每epoch切几份各更新一次(4×epoch4=16次/批) | 调大→minibatch小→梯度噪声大→偏向不稳定 | **(b)路榨梯度**:调大=更多更新次数榨更多;但只榨现有关+与崩盘同向风险,非好杠杆;🔒需对等sweep |
| **VF_COEF** | 0.5 | value loss权重 | value不准→GAE方向错→难关更新错 | value准→advantage准 |
| **GAMMA/GAE_LAMBDA** | 0.99/0.95 | 折扣/GAE平滑 | 长程导航关(崩盘正是这类)对γ敏感;0.99已较高 | — |
| ANNEAL_LR/NUM_STEPS/NUM_ENVS/FC/HIDDEN | True/512/256/512/512 | 退火/rollout/并行env/网络 | 🔒协议锁,容量已调 | — |

#### B. SFL/UED 专属块

| 参数 | 当前值 | 含义 | 对**崩盘**作用 | 对**吃不饱**作用 |
|---|---|---|---|---|
| **N候选池** | 3000 | generator海选范围 | 🎯池大→尾部结构覆盖更全,可能减崩;§8.3论文背书5000 | 多样性高→更可能有可吃关 |
| **ρ sample ratio** | 1.0 | 注入关vs随机reset关比例 | ρ=1全注入,generator造的关若结构单一→某些尾部结构没见过→崩 | 全注入=全是定向难关 |
| **K buffer(NUM_TO_SAVE)** | 60 | 注入buffer保留数 | 小buffer→过度集中某类难关→其它尾部欠覆盖 | — |
| **注入选关方式** | 随机top-K | top-K后随机/降序 | 弱相关;§8.3论文证降序更好 | — |

#### C. Generator/课程/auction 块

| 参数 | 当前值 | 含义 | 对**崩盘**作用 | 对**吃不饱**作用 |
|---|---|---|---|---|
| **GEN_ESTIMATOR_IDS** | [anchored,cenie] | auction真正进竞价的信号源(注意≠SIGNAL_MODE标签,见§8.6) | 🎯🎯**强相关**:当前信号若系统性漏某类尾部难关→那类从不训练→崩;**加\|adv\|路补盲区** | 🎯🎯**\|adv\|选关=持续供给"还有梯度可吃"的关**(§5.7 p≈0区分辨率最高),非榨现有关 |
| **CURRICULUM_THR_STAGES** | [5,10,22] | 课程难度档(测地格数) | 🎯顶档22:若test尾部关测地>22→训练没见过→崩;**调高顶档**(不超test真实测地) | 留更难关让后期继续有梯度吃 |
| **CURRICULUM_P_THRESHOLD(τ_up)** | 0.6 | 升档阈值frac(p>0.8)>τ | 太早升顶→中间难度没练扎实→尾部脆 | §7:p判据后期饱和致秒冲顶,换Δ\|adv\|判据更准 |
| **CURRICULUM_STREAK** | 2 | 升档需连续N次 | 调大→每档多练→基础更牢 | 同左 |
| **CURRICULUM_BETA** | 1.0 | 课程塑形强度-β·relu(geo−thr) | 控generator往当前档推的力度 | — |
| **CURRICULUM_ARM** | geo | 难度轴=测地长度 | 若崩盘关难度不在geo轴→课程没对准 | §5.2证geo唯一干净轴 |
| **GEN_AUCTION_LAMBDA** | 1.0 | auction温度(选关贪心度) | 🟡打分层已证无差异 | — |
| **GEN_POOL_PER_GEN** | 1500 | 每gen候选(×2=池3000) | 同N候选池 | — |

#### ★对症小结（修崩盘 / 修吃不饱 各按机制相关度排序）

- **修崩盘**(尾部脆/自信犯错)：① ENT_COEF 开0.001~0.01(最对症,强制探索) → ② CLIP_EPS↓0.02 / MAX_GRAD_NORM↓0.25(更保守更新) → ③ THR顶档加高(若崩在超长程关) / estimator加\|adv\|(补尾部盲区)。
- **修吃不饱**(顶档还有梯度没吃完)：① estimator加\|adv\|选关(供给高梯度关,非榨现有) → ② THR顶档加高(留更难关) → ③ 换Δ\|adv\|课程判据(吃饱才升档) → ④(下策)UPDATE_EPOCHS/NUM_MINIBATCHES榨现有关(与崩盘同向风险,只榨现有关)。
- **★同时利好两者且空白**：**estimator [absadv,anchored,cenie]**——选关层持续供给高梯度关(治吃不饱)+捞p≈0侧尾部难关(治崩盘),从未正式跑(§8.6)。

### 8.6 ★核查：|adv| 进 bid 从未正式跑过（2026-06-29）

核查 Oscar 所有脚本的 `GEN_ESTIMATOR_IDS`（**这才是真正进 auction 竞价的信号源；`AUCTION_SIGNAL_MODE` 只是标签名**）：

| 脚本 | SIGNAL_MODE(标签) | **ESTIMATOR_IDS(真bid)** | \|adv\|真进bid |
|---|---|---|---|
| `sbatch_absadv_dry.sh` | absadv_anchored_cenie | **[absadv,anchored,cenie]** | ✅ 但 TOTAL=**4e6**(冒烟,1/75正式量) + sacct无执行记录 |
| fix_3/fix_A/fix_C/fixcurr | absadv_anchored_cenie | **[anchored,cenie]** | ❌ 名字带absadv但实际没进bid |
| advadj_3stage/4stage | absadv_anchored_cenie | **[anchored,cenie]** | ❌ |

**结论：estimator 三路 [absadv,anchored,cenie] 100% 空白**——只在 4e6 冒烟脚本配过(还没执行记录)，所有正式 run 的 estimator 实际只有 [anchored,cenie]。**§8.4 执行顺序第4步从未做过。** 跑它前需先验证 absadv 真能进 auction 不报错(dry 没跑过)。

#### ★★★ 8.6 订正 + 工程根因 + absadv reward 实现（2026-06-29 晚，读代码坐实）

**订正 1（开关语义读反了）**：上表说"GEN_ESTIMATOR_IDS 才是真正进 auction 竞价的信号源"——**读反了**。读代码（pcgrl_generator.py line 760-878 + jaxnav_sfl.py line 940-954）坐实：
- **`AUCTION_SIGNAL_MODE` 决定 auction bid 维度（`per_est` 矩阵）** ← 这才是"选关 bid"的真开关。`absadv_anchored_cenie`→bid 3维含 absadv；`anchored_cenie`→bid 2维不含。
- **`GEN_ESTIMATOR_IDS` 决定 N 个 generator 各自的 terminal reward 信号（造关）** ← 是"造关 reward"，**不是 bid**。
- 两者是**独立的两层**（reward=造 / bid=挑）。要 absadv 真进 bid，**两个都要设对**。

**订正 2（根因=缺工程，非配置）**：absadv "从未跑通"的真因**不是没配**，是 **`compute_terminal_reward` 根本没有 absadv 分支**（只支持 difficulty/pvl/cenie/anchored/euc，传 absadv 当 reward 直接 `raise ValueError`）。即 **absadv 在旧代码里只能当 bid（per_est 有），不能当 reward（terminal_reward 缺实现）**。`GEN_ESTIMATOR_IDS=[absadv,...]` 跑到 generator 训练步必崩（旧 4e6 dry 太短没触发故没暴露）。

**工程修复（2026-06-29 已实现 + 验证）**：
- 新增 `terminal_reward_absadv`（pcgrl_generator.py，仿 terminal_reward_pvl）：reward = per-level `mean|adv|`，让 generator **主动造高梯度关**（事前 reward 版，区别于只在 bid 挑关）。
- 加常量 `GEN_ESTIMATOR_ABSADV="absadv"` + `compute_terminal_reward` 分派 + 报错信息更新。
- probe 加 absadv 维（probe_orthogonality.py：four_signals→five_signals，复用已有 advantages 零额外算）→ 可测 `rho/absadv~anchored`、`absadv~CENIE`（回答"|adv| 是否与 anchored 共线=话语权过大"）。
- dry 验证（job 3535912，8e6 跑到 generator 训练步）：`terminal_reward_absadv` 端到端不崩，.err 无报错。

**架构澄清**：`GEN_ESTIMATOR_IDS=[absadv,anchored,cenie]` = **3 个独立 generator 网络**（make_generator_state 各 init 独立 params，防 mode collapse），各被一种 reward 驱动造关；拼成候选池后 auction 按 SIGNAL_MODE 维度 bid 挑 60 关注入。

**本轮对照实验（2026-06-29，各 1 seed，无课程，底座=currA-nodiff-paper）**：
| 组 | job | GEN_ESTIMATOR_IDS(reward) | AUCTION_SIGNAL_MODE(bid) | absadv 角色 |
|---|---|---|---|---|
| 组1 reward+bid | 3535916 | [absadv,anchored,cenie] | absadv_anchored_cenie | 事前造高梯度关 + 事后 bid 挑 |
| 组2 reward-only | 3535917 | [absadv,anchored,cenie] | anchored_cenie | 只事前造，bid 不认（靠 anchored/cenie 挑） |
| 基线 currA | (已有) | [anchored,cenie] | (anchored 默认) | 完全不参与，官方 CVaR10=85.73 |

**2×2 拆解**：组2 vs currA = "让 generator **造**高梯度关有没有用"（纯 reward 价值）；组1 vs 组2 = "auction **挑**关时认 absadv 有没有额外价值"（纯 bid 价值）。跑完上官方 deploy 评测出 win-rate CVaR 定论。

> **信号强度现状**：terminal_reward_absadv 裸返回 mean|adv|（无显式系数）。三信号量级相近（anchored/cenie bid~2.0、|adv|~1.97，§5.6），且 bid 层 z-score 兜底量级悬殊 → 裸 1.0 是合理起点。GEN_AUCTION_LAMBDA=1.0（bid top-K 尖度）是 absadv 话语权的真旋钮，generator-on 下其最优值未知（[[auction-ablation-launched]] 的"λ无差异"是 generator-off 旧测）。强度扫描留第二轮。

---

## 9. ★本轮提交：延长训练 + CLIP_EPS 对等 sweep + s4tight 三档对照（2026-06-28，进行中）

> **分工**（已定）：**B1 + baseline 各补 seed 0–9（共 20 run）全交同学**（外部 GPU 机器，文档 [补seed_运行指导.md](./补seed_运行指导.md) 已就绪、论文同底座两档）；**Oscar 2 GPU 专心跑本节的 E 组 + s4t3**，分工不重复。

### 9.1 Oscar 提交队列（8 run，2 GPU 自动按 QOSMaxGRESPerUser 节流消化）

| 顺序 | job | GROUP | 配置（vs B1 底座只改的变量） | seed | 墙钟 |
|---|---|---|---|---|---|
| 1 | tuneE1 (3478968) | tune-E1-extend | **只延长** TOTAL_TIMESTEPS 3e8→**4e8** (clip 0.04) | 0,1 | ~77min |
| 2 | tuneE2 (3478969) | tune-E2-clip01 | **只加 clip** CLIP_EPS 0.04→**0.1** (TS 3e8) | 0,1 | ~58min |
| 3 | tuneE3 (3478970) | tune-E3-both | **叠加** TS 4e8 + clip 0.1 | 0,1 | ~77min |
| 4 | s4t3 (3479177) | s4-adaptcurr-nogate-3stage-tau0_8-streak3 | **s4tight 三档版**：THR_STAGES [5,10,16,22]→**[5,10,22]** | 0,1 | ~77min |

> **E 组底座 = B1（abl_B1 逐字核对）**：frac判据 P_THRESHOLD=0.6 / anchored+[anchored,cenie] / 三档[5,10,22] / pool3000 / save60 / ρ1.0 / L2000 / LR2.4e-4。E1/E2/E3 各只改标注的那一个（或两个）变量。
> **s4t3 底座 = s4tight 四档（4stage_tight 逐字核对）**：与 B1 仅差课程调度三参（τ0.8/streak3/四档）；s4t3 再把四档→三档，**只改档数**这一个变量。

### 9.2 判读方案（结果出来直接套）

**E 组 2×2 归因**（基准 = B1 单 seed 2.447，由同学补 seed 那批提供 10 seed 锚点）：
- **E1 vs B1 = 纯"延长训练"效果**。§7 预测：B1 末段 |adv| 仍在升（2.236→2.468）=没吃饱 → 延长应让 CVaR↑。验证用 `gen_absadv_hist`（§5.7 代码已落）看延长后 |adv| 是否吃到平台。
- **E2 vs B1 = 纯 CLIP_EPS 效果**（§8.2 公平性 sweep 的 ours 侧；SFL 侧的对等 sweep 后续补）。
- **E3 vs E1/E2 = 交互**：叠加增益是否 > 各自单独。

**s4t3 归因**：vs s4tight 四档(2.379)=纯档数效果；vs B1(2.447)=纯调严效果。配 curriculum_stage 轨迹看中间档是否多停留（§5.4 四档没让中间档多停留）。

### 9.4 ★E1（只延长 4e8）结果（2026-06-28，COMPLETED 2seed，return 代理口径）

run：E1_s0=aza64cq2 / E1_s1=j22yfjai。两 seed 跑满 step3050（≈3052 upd，2h36m，比 B1 慢因 1.33× 步数 + probe 开销 + 节点速度）。

| | seed | CVaR10 | worstbin | ovr_win | ovr_ret |
|---|---|---|---|---|---|
| **B1 基准**(3e8/clip0.04) | s0(单) | 2.447 | +0.35 | 0.992 | 3.951 |
| **E1 只延长**(4e8/clip0.04) | s0 | 1.910 | −0.90 | 0.989 | 3.852 |
| | s1 | 2.515 | −1.12 | 0.993 | 3.947 |
| | **均值** | **2.213** | −1.01 | — | — |

**★|adv| 吃饱判据（机制证据，比 CVaR 抗 seed 噪声）**：取 `gen_absadv_mean` 全过程末1/4斜率：
- E1 s0：2.391→2.415（Δ+0.024，**走平**）；E1 s1：2.774→2.349（Δ**−0.425，下降**）。
- 对比 §7 B1@3e8 末段 |adv| **还在升**（2.236→2.468，Δ+0.232）。

**两个结论：**
1. **★延长确实让"没吃饱"变"吃饱"**：B1@3e8 |adv| 还在升 → E1@4e8 |adv| 走平/下降，梯度被吃完。**§7 "B1 没吃饱"假设方向证实**。
2. **★但吃饱没换来 CVaR 收益**：E1 均 2.213、worstbin −1.01（比 B1 +0.35 还差）。延长让 student 在顶档关更熟练（ovr_win 0.989/0.993），**但最难 10% 该崩还崩**。坐实 §0 洞察：**决定 CVaR 的是难关结构多样性，不是在现有关上吃多少梯度**——延长只把现有关吃透、没引入新难关结构，故尾部不改善。

**⚠️ caveat**：E1 两 seed CVaR 撕裂(1.91↔2.52，§1 剧本重演)，且 B1 仅单 seed 对照，"延长无 CVaR 收益"严格说待 B1 补 seed 坐实；但 |adv| 走平的机制证据稳健(两 seed 一致吃饱)，不依赖 CVaR 噪声。**初步倾向：延长训练不是 CVaR 的杠杆，E3(延长+clip)若也不涨则确认延长该砍。**

### 9.6 ★★★口径矛盾：return CVaR 的"崩盘"在 success-rate 口径下消失（2026-06-29，决定性）

**崩盘关结构诊断**（job 3501437，6 ckpt 在固定 100 关评测集 `sampled_tc_100e_1a.pkl` 逐关 rollout，崩 seed 按 §2：E1崩s0/E2崩s1/combo1崩s1）：

| run | 好seed平均p | 崩seed平均p | 崩盘专属失败关 n | 这些关 geo |
|---|---|---|---|---|
| E1 | 0.990 | 0.979 | 2 | 8.0 |
| E2 | 1.000 | 0.980 | 2 | 8.5 |
| combo1 | 0.990 | 0.981 | 2 | 9.5 |

全体可达关 geo=5.7±3.3；崩盘专属失败关 geo=8.7（偏长程，但 n 太小 statistically 弱）。

**★决定性矛盾**：wandb 的 CVaR/worstbin（`~/cvar_all.py` = **return 代理口径**）说 E2 s1（0.397/−4.71）、combo1 s1（1.679/−4.98）**崩盘**；但本诊断的 **success-rate 口径**（`_success_from_traj`，=SFL 论文官方口径）说这些"崩盘 seed"在 100 关上**平均 p≈0.98、只比好 seed 多崩 2 关**，根本不崩。

**机制解释（核查 SFL 论文全文坐实）**：
- **SFL 论文官方口径 = success rate（win rate），不是 return**。CVaR 协议(§6)是"average **success rate** on worst α%"，Fig 3a y轴=Avg Win Rate%；学习性 p(1−p) 的 p=success rate。论文附录 G.1 明说"return 与 success rate **强相关**，故主文只展示 success rate"。
- jaxnav return 受 **shaping reward 污染**(附录A)：距离 shaping wg·Δdist + 碰撞 −4 + 贴墙 −0.1/步 + 超时 −0.01/步。一关**成功到达但路径绕远/贴墙/差点撞** → return 大负但 success=1。worstbin −4.71 极可能是**碰撞惩罚 −4**(撞墙)或贴墙累积，**不是任务失败**。
- **故"崩盘"是 return 代理口径制造的：崩 seed 不是"失败更多关"，是"成功了但 return 灾难性低"**(绕路/贴墙/差点撞)。换论文官方 success-rate CVaR，这些 seed 可能根本不崩。

**★影响全项目的口径订正**：内部一直用的 `~/cvar_all.py`(return 代理) ≠ SFL 论文官方(success-rate CVaR + 10k levels + seed去偏，Fig 3a)。§2 早警告"非论文口径仅内部相对比较"，但本诊断**首次暴露两口径给出相反结论**(return说崩/success说不崩)。**§0 大表与 §9 全部 CVaR 数字都是 return 代理口径，不能写进 paper，须用官方口径重测**(见下 §9.7 待办)。

### 9.7 ★待办：上 SFL 官方测试口径（2026-06-29，用户拍板）

**目标**：对 **combo1/E1/E2 共 6 seed + SFL baseline 4 seed** 跑 SFL 官方 deploy 评测（**win-rate CVaR + 10,000 levels + seed0/seed1 去偏**，论文 Fig 3a 口径），得到能写进 paper、能和论文 baseline(DR/PLR/ACCEL) 同表比的真数字。
- 链路(§2)：`sfl/deploy/eval_jaxnav_single_agent_2_analyse.py`，需先 `_0_generate_levels`→`_1_rollout` 生成 10k-level CSV。
- checkpoint 用各 run 的 `model.safetensors`(final)；combo1 用此而非 step1850(§9.5 校正)。
- **预期**：若官方 success-rate 口径下"崩盘"消失，则 §9.4/9.5 基于 return 代理的"崩盘/撕裂"结论需大幅修正，combo1/E 组的真实战力须重判。

### 9.8 ★官方评测链路勘探 + 适配方案（2026-06-29，已验证模板）

**官方脚本现成**（SFL repo 自带 `sfl/deploy/eval_jaxnav_single_agent_{0,1,2}.py`），非重搭，只需适配。链路已验证（learn-peak 等 17 group 在 seed_0/seed_1 都跑通、CSV 10001 行）。

**链路三步**：
1. `_0_generate_levels.py`：生成 10k 评测集 `eval_single_agent_10000e.pkl`（**已存在，不用重跑**；valid_path_check=True 保证可解）。
2. `_1_rollout.py`：`GROUP=sys.argv[-1]` 命令行传 group；从 `checkpoints/jaxnav_single_agent/{GROUP}/*.safetensors` 读每个 seed 的 ckpt，在 10k 评测集上每关 rollout `N_EPISODES=10`，输出 per-level win_rates+returns CSV 到 `eval_10000_envs_seed_{SEED}/{group}/{seed}.csv`。
3. `_2_analyse.py`：win-rate CVaR + 去偏。`SEEDS=10`/`TOTAL=10000` 硬编码。

**★去偏机制（核实）**：`get_vals(group,0)` 选最差关、`get_vals(group,1)` 报成绩。**两批=同一评测集 pkl 跑两次不同 rollout SEED**（不是两批 levels！learn-peak 两批 return 仅差小数点后4位=纯 rollout 随机）。故 `_1_rollout` 对每 group 跑 2 次（SEED=0→seed_0 目录 / SEED=1→seed_1 目录）。

**★s4p2 模板验证**（用户提醒"当时只探索没做完"）：
- ✅ symlink 逻辑**正确可照搬**：`{group}/{label}_seed{N}.safetensors → ../../multi_robot_ued/{run_name}/model_step2250.safetensors`（3 个 base_seed 链接有效）。
- ⚠️ **但 `_1_rollout` 从没真对 s4p2 跑出结果**（seed_0/1 目录都无 s4p2 CSV）→ **必须先用现有 seed 端到端试跑一次验证**，再批量。

**适配 4 处**（seed 补完后做）：
1. symlink：4 组(baseline/combo1/E1/E2)×6 seed 的 `model.safetensors`(final) → `checkpoints/jaxnav_single_agent/{group}/`。
2. `_1_rollout`：`NUM_ENVS_TO_SAMPLE` 1000→**10000**；SEED 走 argv 跑 0/1 两遍。
3. `_2_analyse`：`CLEAN_NAMES` 加 4 个 group + `SEEDS` 10→**6**。
4. 统一 config.yaml 已在 `checkpoints/jaxnav_single_agent/config.yaml`（含 jaxnav 环境参数）。

**执行**：先 1 个 group×现有 seed 端到端试跑（验证 `_1_rollout` 真能加载 ckpt+输出有效 CSV），通过再批量 4 组。

### 9.5 ★combo1 组合 run 结果（2026-06-29，COMPLETED 2seed，return 代理口径）

**配置**（用户设计，B1 基底同时改 3 参数，非单变量归因）：`TOTAL_TIMESTEPS=3.2e8` + `ρ=0.8`(FROM_SAMPLED=204/TO_GENERATE=52) + `CLIP_EPS=0.05`。其余=B1。group=`combo-ts3_2-rho0_8-clip0_05`。run：s0=4kout71w / s1=42k9h5zp。
> **设计意图**：三参都往"温和+多样"方向调冲修崩盘——CLIP 0.05 比 E2 的 0.1 温和(避激进崩盘)；ρ 0.8 混入 20% 随机 reset 关增结构多样性(原以为能加稳定性)；TS 3.2e8 略延长。
> **降序注入**：用户原列第 4 参数，核查发现 generator 架构(60关一次性全注入固定buffer+训练shuffle)无现成开关、语义可能无效(≠PLR replay)、需改代码 → **本 run 去掉降序，只跑 3 参数**(待 §8.6 后另议)。

| | seed | CVaR10 | worstbin | ovr_win | ovr_ret |
|---|---|---|---|---|---|
| **B1 基底**(3e8/ρ1.0/clip0.04) | s0(单) | 2.447 | +0.35 | 0.992 | 3.951 |
| **combo1**(3.2e8/ρ0.8/clip0.05) | **s0** | **3.089** 🥇史上最高 | **−3.83** | 0.987 | 3.976 |
| | **s1** | **1.679** | **−4.98** | 0.982 | 3.929 |
| | 均值 | 2.384 | −4.41 | — | — |

**数据事实（不归因，3参数无法拆开）：**
1. combo1 s0 = **单 seed 史上最高 CVaR10=3.089**（超 B1 2.447 与所有 E 组）。
2. **崩盘未修**：两 seed worstbin 都大负(−3.83/−4.98)，均值 worstbin −4.41，比 B1 单 seed +0.35 差；"加 20% 随机关增稳定性"的预期未兑现，s1 仍重崩(1.679/−4.98 与 E2 s1 的 0.40/−4.71 同形态)。
3. **两 seed 撕裂(3.089↔1.679)** 延续 §1 剧本，3 参数捆绑无法归因。

**★再现的跨 run 模式（值得专门查）**：B1(6/27 旧代码)是**唯一 worstbin 为正(+0.35)** 的 run；**所有 6/28-29 新跑的 run（E1/E2/combo1 共 6 个 seed）worstbin 全负(−0.90~−4.98)**。可能性：①6/28 改 pcgrl_generator/jaxnav_sfl 动了训练逻辑(非只加log)污染所有新 run；②或 B1 的 +0.35 本就是其单幸运 seed 特例(补 seed 后也会变负)=崩盘是方法固有性质。**两者都需先用 B1 原脚本在当前代码重跑一 seed 才能区分(诊断待办)。**

**⚠️ 铁律**（§1）：E 组/s4t3 每臂仅 2 seed，**不能下"坐实赢"结论**。本轮产出 = **趋势 + 是否值得补 seed**。真正坐实要等同学 B1 10 seed 回来，再决定哪个 E 臂值得补到 10 seed。CVaR 用 `~/cvar_all.py` return 代理同口径；2 seed 一致才算信号，分裂（如 §0 worstbin 撕裂）则标记需补 seed。

### 9.3 LR 为何本轮不动（用户问，记录决策）

用户问"困难档 student 学得慢，直接加 LR 是否有用"。**否决**：§7 固定考卷诊断证 B1 顶档 student 不是"学得慢"——p 从 0.567 单调爬到 0.967、末段 Δp>0、|adv| 还在回升。瓶颈是"**还在高效吃梯度但步数用完**"（"没吃饱就结束"），不是"学不动"。此时加 LR 是在梯度吃得好的情况下放大步长，更可能过冲而非加速。正确解法=延长训练/加难（=E1/E3）。LR 列为 §8.4 "🔬对等 sweep 次优先"，待延长后 |adv| 不再回升、CVaR 卡住才单独扫，避免与"延长"耦合难归因。延长倍数取 1.33×（4e8）= 用户定。
