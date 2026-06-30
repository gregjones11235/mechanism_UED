# remain_work2 —— 官方口径评测存档（baseline 体检 + 全方法座次）（2026-06-29）

> 本文档记录 **SFL 官方测试口径（论文 Fig 3a：win-rate CVaR + 10k levels + 去偏）** 下的两类数据：
> §1-3 = SFL baseline 稳定性体检（对比基准锚点）；**§4 = 全方法官方 CVaR 座次（所有探索过的非 baseline 方法）**。
> 内部 return 代理口径的数据见 remain_work.md（注意：§4 证明内部口径几乎无预测力）。

## 0. 口径说明（关键）

- **数据源 = SFL 官方评测口径（论文 Fig 3a）**，非内部 `cvar_all.py` 的 return 代理口径。
- 评测脚本：`sfl/deploy/eval_jaxnav_single_agent_1_rollout_official.py` →(去偏两批 seed_0/seed_1)→ `analyse_official.py`（CVaR 算法逐字抄官方 `_2_analyse.py` line 143-160，一字未改）。
- **指标 = win rate（success rate）**，干净成败，不含 jaxnav shaping reward 污染。
- 评测集 = 官方固定 **10,000 关**（`eval_single_agent_10000e.pkl`，valid_path_check=True 保证可解，4 chunk×2500）。
- 去偏 = seed_0 批**选**最差 α% 关、seed_1 批**读**这些关成绩（消除选择偏差）。
- **n = 6 个训练 seed**（官方论文用 10 seed；6<10 只影响误差棒宽度 std/√n，不影响口径权威性）。

baseline 6 个训练 run（wandb，project=multi_robot_ued，entity=gregjones11235-brown-university）：
`cqdy03uu / rqdqn5da / hsmqjgqh / pk8abkr8 / kusioig2 / v3raza8r`（group=sfl-paper-cfg）。
配置 = SFL 官方仓库 `amacrutherford/sampling-for-learnability` 默认值一字未改（L=1000, ρ=0.5 FROM_SAMPLED）。

---

## 1. 官方 win-rate CVaR（去偏，6 seed 均值）

| 档 | CVaR1% | CVaR2% | CVaR5% | **CVaR10%** | CVaR20% | CVaR50% | CVaR100%(overall) |
|---|---|---|---|---|---|---|---|
| **mean** | 28.38 | 57.39 | 82.09 | **91.03** | 95.49 | 98.15 | 99.03 |
| **std** | 5.33 | 4.39 | 1.92 | **0.96** | 0.47 | 0.19 | 0.09 |
| min | 19.10 | 49.41 | 78.63 | 89.31 | 94.65 | 97.82 | 98.87 |
| max | 36.72 | 64.15 | 85.13 | 92.56 | 96.25 | 98.45 | 99.19 |
| 极差 | 17.62 | 14.74 | 6.50 | **3.25** | 1.61 | 0.64 | 0.31 |

（单位 %。CVaR_α = 最差 α% 关的去偏平均 win rate。）

### 逐 seed 明细（CVaR10 / overall / 失败关分布）

| seed | run | CVaR1% | CVaR5% | **CVaR10%** | overall_win | zero(=0)关 | partial关 |
|---|---|---|---|---|---|---|---|
| s0 | cqdy03uu | 36.72 | 85.13 | 92.56 | 99.185 | 35 | 317 |
| s1 | rqdqn5da | 19.10 | 78.63 | 89.31 | 98.852 | 50 | 359 |
| s2 | hsmqjgqh | 27.33 | 81.76 | 90.87 | 99.016 | 38 | 341 |
| s3 | pk8abkr8 | 28.21 | 82.13 | 91.05 | 99.026 | 42 | 370 |
| s4 | kusioig2 | 31.84 | 82.90 | 91.44 | 99.066 | 40 | 354 |
| s5 | v3raza8r | 27.10 | 81.99 | 90.97 | 99.043 | 39 | 310 |

---

## 2. 稳定性体检（刷 SOTA 三项检验）

### ✅ 检验 1：CVaR test 的 seed 间方差 —— 主指标极稳

| 档 | mean | std | 极差 | 变异系数 |
|---|---|---|---|---|
| **CVaR10%（paper 主指标）** | 91.03 | **0.96** | 3.25 | **1.1%** |
| CVaR5% | 82.09 | 1.92 | 6.50 | 2.3% |
| CVaR1%（极尾100关） | 28.38 | 5.33 | 17.62 | 18.8% |
| overall win | 99.03 | 0.10 | 0.33 | 0.1% |

- **CVaR10 变异系数仅 1.1%**，6 seed 全挤在 [89.31, 92.56]。
- 唯一方差大的是 **CVaR1%（极尾 100 关，std 5.33）**——这是统计本性（仅 100 关、全 0/1 两极的最难关，谁来都抖），**非 baseline 独有的不稳**。即便此档 baseline 均值 28% 仍远高于 idea（4~6%）。
- **判定：baseline 在 paper 主指标 CVaR10 上无幸运 seed 问题，单 seed 都基本可信。CVaR1% 档须多 seed。**

### ✅ 检验 2：训练稳定性（末段震荡）—— 稳

| run | win末25%std | return末25%std | final_return |
|---|---|---|---|
| cqdy03uu | 0.0040 | 0.037 | 3.954 |
| rqdqn5da | 0.0043 | 0.047 | 4.030 |
| hsmqjgqh | 0.0033 | 0.033 | 3.994 |
| pk8abkr8 | 0.0039 | 0.040 | 3.964 |
| kusioig2 | 0.0040 | 0.040 | 3.951 |
| v3raza8r | 0.0044 | 0.035 | 3.985 |

- win rate 末 25% std ≈ 0.004，return 末 25% std ≈ 0.04，**末段平台稳定无大震荡**。
- final return 跨 6 run：mean=3.980，**std=0.027**，range=[3.951, 4.030]，训练终点高度一致。

### ✅ 检验 3：student 表现回落（peak→final）—— 无回落

| run | win峰@位置 | win_final | 峰后回落 | ret早25%→末25% |
|---|---|---|---|---|
| cqdy03uu | 0.9955@89% | 0.9906 | +0.0050 | 1.573→3.930 |
| rqdqn5da | 0.9996@98% | 0.9996 | +0.0000 | 1.678→3.954 |
| hsmqjgqh | 0.9997@84% | 0.9949 | +0.0048 | 1.446→4.005 |
| pk8abkr8 | 0.9940@53% | 0.9896 | +0.0045 | 1.719→3.907 |
| kusioig2 | 0.9977@67% | 0.9905 | +0.0072 | 1.770→3.926 |
| v3raza8r | 0.9974@96% | 0.9939 | +0.0035 | 1.585→3.942 |

- 6 run 的「峰后回落」**全为正数或 0**（final≈peak，差距 ≤0.007 win / ≤0.055 return，均在末段噪声内），**无任何 run 训到后期掉下来**。
- 轨迹健康单调收敛：return 早 25% ~1.6 → 末 25% ~3.95，爬升到平台后稳住。

---

## 3. 综合结论

| 检验项 | baseline | 判定 |
|---|---|---|
| CVaR seed 间方差（CVaR10） | std 0.96，变异系数 1.1% | ✅ 极稳 |
| 训练稳定性 | 末段 std 0.004(win)/0.04(ret) | ✅ 稳 |
| student 回落 | 峰后回落全 ≥0 | ✅ 无回落 |

**baseline 三项全过，是干净可靠的对比基准。** 91% CVaR10 由 6 seed 稳定复现，非幸运 seed。

**对刷 SOTA 的含义**：基准如此稳定，idea 的失败不能甩锅给「baseline 运气好」。须以此为锚——任一 idea 想声称超 baseline，CVaR10 须**稳定**摸到 ≥89.3（baseline 最低 seed），单高 seed 不算。

> 数据原件：Oscar `~/sampling-for-learnability/official_cvar_results.json`（4 组全量），分析脚本 `analyse_official.py / base_stability.py / base_train_curve.py / base_tail_curve.py`。

---

## 4. ★★★ 全方法官方 win-rate CVaR 座次（2026-06-29，决定性）

> 项目首次把**所有探索过的非 baseline 方法**统一上官方口径排在一起。
> job=3533300（去偏两批，16 个 model）+ 早先 official-combo1/e1/e2（各 6 seed）。
> ckpt→方法映射经 config（档数/下界 BETA_LO/judge）+ 内部 CVaR 数字双重确认（C=1.325、fixA=0.444 精确吻合）。
> 数据原件：Oscar `official_cvar_all_methods.json`，脚本 `analyse_all.py`。

### 4.1 座次（按官方 CVaR10 降序）

| 排名 | 方法 | n | **CVaR10%** | CVaR5% | CVaR1% | ovr_win | vs base | 内部CVaR10(return口径) | 判据/下界 |
|---|---|---|---|---|---|---|---|---|---|
| — | **SFL baseline** | 6 | **91.03** | 82.1 | 28.4 | 99.03 | — | — | — |
| 1 | s4tight 四档 | 1 | 86.25 | 72.6 | 12.2 | 98.58 | **−4.8** | 2.379 | frac/无 |
| 2 | currA-nodiff-paper | 1 | 85.73 | 72.0 | 11.3 | 98.46 | −5.3 | 2.373 | (多变量污染) |
| 3 | s4only 四档 | 1 | 83.38 | 67.4 | 10.8 | 98.26 | −7.7 | 0.609 | frac/无 |
| 4 | **E1 (extend 4e8)** | 6 | 81.56 | 63.8 | 5.8 | 98.08 | −9.5 | 2.213 | frac/无,延长 |
| 5 | B1 adaptcurr-nogate | 1 | 81.04 | 64.7 | 10.3 | 97.99 | −10.0 | **2.447** | frac/无 |
| 6 | combo1 (ts3.2/ρ0.8/clip0.05) | 6 | 80.41 | 61.8 | 6.1 | 97.95 | −10.6 | 2.384 | frac/无,3参组合 |
| 7 | B2 adaptcurr-gate | 1 | 80.23 | 62.7 | 8.8 | 97.93 | −10.8 | 0.438 | frac/gate |
| 8 | A1 fixedcurr-nogate | 1 | 79.01 | 59.6 | 4.4 | 97.84 | −12.0 | 1.07 | 固定/无 |
| 9 | E2 (clip 0.1) | 6 | 76.42 | 55.6 | 4.5 | 97.58 | −14.6 | — | frac/无,clip0.1 |
| 10 | currA-bigpool2k | 1 | 73.82 | 48.5 | 3.4 | 97.32 | −17.2 | 1.855 | 3维bid pool2k |
| 11 | currA-nodiff-3k | 2 | 71.43 | 44.0 | 1.4 | 97.09 | −19.6 | 1.69 | 2维bid pool3k |
| 12 | A2 fixedcurr-gate | 1 | 71.25 | 46.4 | 2.5 | 97.09 | −19.8 | 1.07 | 固定/gate |
| 13 | C 四档无下界 | 2 | 71.02 | 47.0 | 3.2 | 96.95 | −20.0 | 1.325 | holdout/无 |
| 14 | holdout 三档 | 2 | 65.98 | 41.7 | 2.3 | 96.37 | −25.1 | 1.719 | holdout/无 |
| 15 | fixA 四档有下界 | 2 | 45.04 | 23.0 | 3.6 | 93.53 | −46.0 | 0.444 | holdout/有 |
| 16 | **g1RB (absadv-bid, 4500池)** | 1 | **36.99** | 11.9 | 3.3 | 92.81 | **−54.0** | — | absadv入bid/4500池 |
| 17 | **g2RO (absadv-gen, 4500池)** | 1 | **31.82** | 5.1 | 1.9 | 92.43 | **−59.2** | — | absadv驱动gen/4500池 |

逐 seed CVaR10（看方差）：currA-3k[74.7,68.2] / C四档[74.9,67.2] / holdout[63.5,68.4] / fixA[33.2,56.8 撕裂]。

### 4.1.1 ★absadv 路线负结果存档（2026-06-29，朝向 bug 已排除）

g1RB/g2RO = N=3 estimators=`[absadv, anchored, cenie]` + 4500 大候选池（job 3535916/3535917，各跑满 1h，model_step2250）。官方口径（job 3537048 去偏两批）：

| run | wandb | absadv 用法 | CVaR10 | overall |
|---|---|---|---|---|
| **g1RB** | 5m52apom (royal-sky-222) `absadv-reward-bid-nocurr` | 进 **reward + bid**(auction竞价) | 36.99 | 92.88 |
| **g2RO** | ol3xtf72 (radiant-butterfly-222) `absadv-reward-only-nocurr` | **仅进 reward**(无bid) | 31.82 | 92.42 |

**★诊断价值**：无论 absadv 进 bid 还是仅进 reward 都垫底 → 不是"bid 层用错"，是 **absadv 信号本身有害，从 reward 层就把 generator 带偏**。排除"换个用法能救"。

**★GPU 排除（用户 2026-06-29 质疑）**：g1RB/g2RO 都在 **gpu3001=L40s**（SFL官方标准GPU，[[stage4-generator-10x-slow-anchor]] 锚点同型号），与 baseline 同型号；日志末尾 `[probe][step2250]`+`Job END` 正常退出、`model_step2250` 落盘 = **跑满 3e8 没欠训没截断**。5000/100 新 run(3537316/3537317)也在 gpu3001。**故负结果与 GPU 算力无关，是 absadv 真实负结果。**

**★真因铁证(probe step2250)**：`PVL~absadv ρ=0.89`(全程均值0.785)——**absadv 与 PVL 高度共线**。而 PVL 正是 SFL 论文 Fig2 钉死的"相关 success rate、不预测 learnability"失败信号（整篇论文在批判 PVL/MaxMC）。故 absadv 驱动 generator = **让 generator 去优化一个 SFL 已证无效的信号** → 造关质量差 → 垫底。**absadv 有害的精确原因 = 它≈PVL（SFL 反面教材）**，非笼统"信号坏"。


- **两个都垫底**——CVaR10 仅 37/32，比 §4 此前最差的 fixA(45) 还低，**比 baseline 输 54~59pt**。
- **连 overall(CVaR100) 都掉**：baseline 99.03 → g1RB 92.81 / g2RO 92.43，掉 6~7pt。**不是尾部 robust 差，是整体能力退步**——absadv 信号有害到拖累均值。
- **大池没救回来**：4500 池非但没帮上，比 3000 池 currA(85) 还差一大截。"扩池+absadv 合并增益"被证伪 = **合并损害**。

**朝向 bug 已排除（时间线铁证）**：`pcgrl_generator.py:182` 当前是连续 `uniform(-π,π)`（2026-06-25 修复）；文件 mtime=6/29 22:14:38，g1RB 训练启动=6/29 22:21:17（启动晚于修复 7min），加载的就是已修代码。**故此负结果可信，不是 [[generator-theta-discrete-bug-pollutes-all-arms]] 污染。**

**probe 数据（step50/100）**：absadv 与其他信号几乎正交（CENIE~absadv 0.2 / PVL~absadv 0.16 / difficulty~absadv 0.01-0.08），不冗余——但正交≠有用，absadv 当 generation 信号把生成分布带偏（同 [[alp-is-curation-not-generation-signal]]/[[generator-pool-collapses-to-extremes]] 模式：某些信号当 generation 会塌缩/带偏）。

**判定：absadv 这条路这批失败，且失败程度全表最重。** 数据原件 Oscar `official_cvar_g1g2.json`，脚本 `analyse_g1g2.py`。

### 4.1.2 在途：座次前二名升 5000/100 同口径重测（2026-06-29 提交）

§4 此前所有 generator 方法用 **3000/60**（baseline=5000/100），即带池子 handicap 输 baseline（图24证 N单调↑→更好）。为去掉 handicap，把座次前二名升到 baseline 同口径重跑：

| Job ID | run-name | wandb | 改动 | 原版CVaR10/vs base |
|---|---|---|---|---|
| 3537316_0 | s4tight-5k100 | qjs2l268 | GEN_POOL_PER_GEN 1500→2500(池5000) + GEN_NUM_TO_SAVE 60→100 | 86.25 / −4.8 |
| 3537317_0 | currA-nodiff-5k100 | fus6brmp | 同上 | 85.73 / −5.3 |

其余全不变（estimators=[anchored,cenie]、s4tight带adaptive课程[5,10,16,22]/currA无课程、TIMESTEPS=3e8、chunk=250、seed、PPO）。已确认池5000/buffer100生效、无OOM、无Hydra崩、进训练循环。脚本 `sbatch_s4tight_5k100.sh`/`sbatch_currA_paper_5k100.sh`。

#### ★★★结果（2026-06-30，反转，决定性）：升 5000/100 不升反降，"池子handicap"假说被证伪

官方口径（job 3538972 去偏两批，gpu3001=L40s，ckpt graceful-resonance-224/fallen-disco-224）：

| 方法 | 池/buffer | **CVaR10%** | CVaR1% | overall | vs base | vs 原版3000/60 |
|---|---|---|---|---|---|---|
| SFL baseline | 5000/100 | **91.03** | 28.4 | 99.03 | — | — |
| s4tight 原版 | 3000/60 | 86.25 | 12.2 | 98.58 | −4.8 | — |
| **s4tight 5k100** | **5000/100** | **84.89** | 12.0 | 98.41 | **−6.1** | **↓1.4** |
| currA 原版 | 3000/60 | 85.73 | 11.3 | 98.46 | −5.3 | — |
| **currA 5k100** | **5000/100** | **83.86** | 15.7 | 98.31 | **−7.2** | **↓1.9** |

**★证伪 §4 末尾"池子handicap"假说**：此前认为 generator 输 baseline 是因带 3000/60 小池(图24证N单调↑→好)。但**同口径升5000/100 反而各降 1.4/1.9pt**，gap 不缩反扩。**−5pt 差距不是池子小造成的。**

**★真因（呼应问题2"无方向性=天花板"）**：加大池对 baseline≠对 generator。
- baseline=SFL 加大N = 从**更多随机关**挑learnability top-K，候选多样性真增 → 图24单调升。
- generator 加大 GEN_POOL_PER_GEN = 让**同一塌缩的generator网络**多造关（[[generator-pool-collapses-to-extremes]]）。生成分布本就塌缩同质，多造=**同质关更多**，多样性没增反可能强化塌缩偏好。**加大池≠加大多样性**。
- → generator 瓶颈**不在候选池规模，在生成分布的多样性/方向性**；补池补不了方向性。这是干净可入论文的负结果。

数据原件 `official_cvar_5k100.json`，脚本 `analyse_5k100.py`。

### 4.2 三条决定性结论

**结论 1：官方口径下无任何方法超 baseline。** 最好的 s4tight 也输 4.8pt。combo1/E1/E2（仅有的 6 seed、最可信）排第 4/6/9，全输 9~15pt。frac+无下界路线真实水平在 80~86 一档，整体输 baseline 5~15pt。

**结论 2：★内部 return 口径几乎无预测力，过去判断被它误导。** 内部排名 vs 官方排名严重错位：B1 内部第1→官方第5；s4only 内部垫底(0.609)→官方第3（反超 B1）；B2(gate) 内部倒2→官方第7；holdout 内部 1.719→官方倒2。Spearman(内部,官方)≈0。根因 = return 被 jaxnav shaping 污染（碰撞−4/贴墙/绕路），"成功但路径差"被误判崩盘（§9.6 诊断本表全面坐实）。**⚠️ remain_work.md §0 大表 / §9 全部 return 口径数字不能判方法好坏，须以本座次重判。**

**结论 3：唯一两口径一致的结构结论——holdout 判据 + 下界把方法搞坏。** 官方垫底三名全是 holdout 脉络（holdout三档66/C四档71/fixA45）；fixA(有下界45) vs C(无下界71)差 26pt，"下界有害"官方口径更悬殊。靠前的全是 frac(p)+无下界。→ **holdout 判据、下界、gate 都该砸。**

> **重要前提（用户 2026-06-29 指出）**：本批所有 generator 方法用的是 **N=3000/buffer60**，而 baseline=SFL 默认 **N=5000/buffer100**。即对比是**故意用更少候选池/buffer 去打 baseline**（论文图24 N 单调↑→更大更好）。故"全输 baseline"须叠加此 handicap 解读——见 remain_work.md 后续"N 劣势下如何击败 baseline"分析。

---

## 5. ★★★ 外部基准调研：jaxnav 单智能体官方已"饱和"+ 更难测试集候选（2026-06-29，5 个联网 agent）

> 动机：用户问"有没有比 jaxnav 更难、SFL 未饱和的测试集"。五个并行联网 agent 的结论汇总。**这一节改变战略判断。**

### 5.1 ⚠️ 决定性铁证：SFL/jaxnav 原班人马官方承认单智能体"highly saturated"并弃用

**NCC 论文**（"An Optimisation Framework for UED"，arXiv:**2505.20659**，RLC/RLJ 2025 #238），作者含 **Alexander Rutherford（SFL 一作）+ Beukman + Foerster**（SFL/jaxnav 原班人马）原文：

> *"we decline to use JaxNav as the single-agent setting's results are **highly saturated**, and the multi-agent setting introduces additional optimisation challenges."*

- 含义①：**jaxnav 单智能体确实饱和——不是我们复现不行，是环境在此设定下不再有方法区分度**（§1-4 baseline overall 99% CVaR100 也印证）。
- 含义②：原作者"逃"向**更难离散域**（Minigrid 60-wall holdout / XLand / Craftax），**没人去补"更难连续导航 holdout"——这是干净的开放缺口**。
- 含义③（写论文用）：可**反过来把这句"highly saturated"当动机**——"原作者承认小地图饱和，故我们提出更难的连续导航 holdout，并证明那里 SFL 随机海选失效、定向 generator 不失效"。
- ⚠️ **战略转向**：不要再把卖点定为"jaxnav 单智能体小地图超 SFL"——那张桌子原作者自己掀了，审稿人会引 NCC 直接打。须往**大地图/长测地 holdout（Axis3）**或**多智能体/CAMAR**走。

### 5.2 独立佐证："learnability 只看最终成败，盲视轨迹/长程"

**DEGen**（arXiv:**2601.14957**，2026，Foerster 组）原文：

> *"SFL performs poorly... **Learnability only considers the final outcome of an episode, rather than the full student trajectory.**"*

- 与我们 [[s4p2A-root-cause-difficulty-axis-mismatch]]（student-centric 信号盲视长程导航）**独立撞上**——别人已替我们论证"p(1−p) 标量对轨迹/长程盲"。这是"无方向性=天花板"的外部背书。

### 5.3 更难测试集候选清单（★完整版，不主观删减——五个 agent 提到的每一个都列）

> 排序按"对我们最有用"降序，但**全部列出**，不因有硬伤就省略（用户 2026-06-29 要求完整记录）。

| # | 环境 | JAX | 连续控制 | 有UED对照 | SFL/UED 不饱和证据（含数字） | 迁移成本 | 故事新颖性 | 链接 |
|---|---|---|---|---|---|---|---|---|
| 1 | **jaxnav 大地图 holdout (Axis3)** | ✅ | ✅ | ✅自带 | 长测地关随机生成器撒不出；网格尺寸参数化；SFL 论文从没测过大地图 | **最低(改config重训)** | 中(同环境新OOD轴) | jaxnav 本身 |
| 2 | **Kinetix** | ✅ | ✅(也支持多离散) | ✅含SFL | **5B步后:S≈85-90% / M≈40-50% / L≈5-15%(非单调)**；PLR/ACCEL无增益over DR(附录L)；微调才能解 Car-Ramp 等纯RL失败任务 | 低 | 高(同实验室继任者) | arXiv:2410.23208,ICLR2025 Oral / github FLAIROx/Kinetix / kinetix-env.github.io |
| 3 | **CAMAR (多智能体连续路由)** | ✅ | ✅(Holonomic+DiffDrive) | ❌**处女地(无UED)** | MAPPO random_grid 0.83 但 **labmaze_grid 仅 0.568**；IPPO 0.41/0.21；6张程序化地图(random_grid/labmaze/movingai/caves_cont Perlin/string) | 中(已JAX连续,加UED层) | **最高(无人做UED)** | arXiv:2508.12845(2025-08) / github AIRI-Institute/CAMAR |
| 4 | **jaxnav 多智能体(≤32 agent)** | ✅ | ✅ | ✅SFL已给 | SFL 论文已测 1/2/4/10 agent，多智能体 worst-case 主导更难；原作者称多智能体"引入额外优化挑战" | 低(jaxnav原生) | 中 | jaxnav 本身 / arXiv:2408.15099 |
| 5 | **BARN(静态)** | ❌(Gazebo/ROS) | ✅(差速+LiDAR) | ❌无UED | **RL(TD3/SAC/DDPG):MLP 65±4% / GRU 82±4% / safe-RL 74±2%**，DWA 82%；论文原话"no method 100%"；ICRA2024冠军 0.476/0.5 上限仍有差距 | 中(地图30×30二值栅格易移植,Gazebo物理贵) | 高(现实BARN对标) | github dperille/jackal-map-creation / arXiv:2210.04839 / 2008.13315 |
| 6 | **DynaBARN(动态障碍)** | ❌ | ✅ | ❌ | **可调30%~90%**:Dyna-LfLH config(DWA65/LfLH40/BC85/Dyna-LfLH90)；harder config(LfH-CP paper)整体仅22-31%；60环境×难度档 | 中(动态障碍物理需自写) | 高(动态障碍=jaxnav future work) | cs.gmu.edu/~xiao/Research/DynaBARN / arXiv:2403.17231 / 2509.26513 |
| 7 | **BipedalWalker(-Hardcore)** | ❌(PyTorch DCD) | ✅(4连续力矩) | ✅ACCEL/CENIE/TRACED | **整体solved率 ACCEL 0.29 / TRACED 0.36**；多地形仍负return(Stairs−29/PitGap−39)；24维obs,8地形参数 | **高(全Box2D地形JAX重写)** | 低(UED经典但拥挤) | arXiv:2506.19997(TRACED) / 2203.01302(ACCEL) / facebookresearch/dcd |
| 8 | **CarRacing UED** | ❌(PyTorch DCD) | ✅ | ✅(DCD套件) | 同DCD家族,连续控制 | 高(同BipedalWalker) | 低 | facebookresearch/dcd |
| 9 | **Overcooked-UED** | ✅ | ❌(离散5动作) | ✅全DCD(DR/PLR/PAIRED/ACCEL) | **泛化近0**:Table1新布局 DR 4.21/PLR 0.68/PAIRED 13.12/ACCEL 0.61 vs Oracle 216.76；最佳partner泛化仅14.6%±7.7 | 低 | 中(多智能体协调) | arXiv:2406.17949 |
| 10 | **Craftax(full,65成就)** | ✅ | ❌(离散网格生存) | ⚠️UED失败showcase | **最好仅~15-16%最大奖励**(PPO-RNN 15.3%/PQN-RNN 16%)；最难两类成就从未达成；UED中仅plain PLR略胜DR,Robust PLR/ACCEL无用,E3B反害 | 低 | 中(UED开问题citation) | arXiv:2402.16801 / craftaxenv.github.io |
| 11 | **Craftax-Classic(22成就)** | ✅ | ❌ | — | **已饱和**:PPO <1hr/1B步达~90%最优;SOTA(ITC) 72.5%return | — | 低(已解) | 同上 |
| 12 | **XLand-MiniGrid(高难档)** | ✅ | ❌(离散meta-RL) | ⚠️非原生UED,SFL有结果 | README明说"baselines不会解最难benchmark";trivial→深规则树,3M任务;但meta-RL非控制 | 中 | 中 | arXiv:2312.12044 / github dunnolab/xland-minigrid |
| 13 | **大PerfectMaze(M/L/XL)** | ✅(minimax) | ❌(离散迷宫) | ✅全(PLR/PAIRED/ACCEL/DR/SFL) | 21×21/51×51/**101×101**,>5k步horizon,标准OOD holdout;TRACED PerfectMazeLarge仅27%±23% | — | 低(SFL在maze家族恰饱和/塌缩) | arXiv:2311.12716 / 2110.02439 |
| 14 | **NAVIX(MiniGrid JAX版)** | ✅ | ❌(离散) | ❌(只DDQN/PPO/SAC) | minigrid家族,SFL在此恰饱和(我方记忆:p∈{0,1}双峰塌缩,方差→0) | — | 低 | arXiv:2407.19396 |
| 15 | **MABrax(多智能体Brax)** | ✅ | ✅(关节力矩) | ❌无UED框架 | JAX多智能体连续控制,Ant/HalfCheetah/Humanoid各关节;难但无UED curriculum baseline | 中 | 低(无UED框架) | arXiv:2311.10090 / github FLAIROx/JaxMARL |
| 16 | **gymnax classic_control** | ✅ | ⚠️(Pendulum/MountainCarCont,平凡) | ❌ | 经典控制PPO基本解掉=饱和;真连续控制要转Brax | — | 低 | github RobertTLange/gymnax |
| 17 | **社交/人群导航(SocNavGym/CrowdNav/SoNIC)** | ❌(PyTorch) | ✅ | ❌ | 动态人群避障,真OOD但非JAX非UED | 高(从零移植) | 中(社交导航) | arXiv:2304.14102 / 2407.17460 |

### 5.4 Kinetix 详细任务说明（最强"换环境"候选，arXiv:2410.23208 / ICLR 2025 Oral）

**它是什么**：基于自研 JAX 物理引擎 **Jax2D** 的开放式 2D 刚体物理 RL 环境空间。**同 Foerster 实验室**（Matthews/Beukman/Foerster），**直接把 SFL 当 SOTA UED 方法**。

**一个 task 由什么组成**：场景 = 圆形 + 凸多边形 + 关节(joint) + 推进器(thruster) 四种实体。**统一目标**：让绿色形状碰到蓝色形状、同时避开红色形状。

**任务多样性（同一引擎能表达的范围）**：
- **机器人运动 Locomotion**：Half-Cheetah / Walker / Hopper——多连杆刚体靠马达控制关节（等价于 MuJoCo 那套，但 2D）。
- **抓取 Grasping**：小车操纵机械臂搬运物体。
- **视频游戏**：Pinball（推进器控制挡板）等。
- **经典 RL**：Cartpole / Lunar Lander / Swing-Up 在框架内重现。

**observation**：**符号化实体集合**——每个实体由物理属性数组(位置/旋转/速度)描述，排列无关、支持可变实体数（像 LLM 处理可变 token）。**不是图像、不是栅格**。

**action**：多离散(马达/推进器 开/关，马达可正反) 或 连续(马达 [−1,1]、推进器 [0,1])。

**为什么比 jaxnav 难（4 个真实难度来源）**：
1. **真物理推理**：要理解刚体动力学、碰撞、约束、**多体交互**——jaxnav 只是质点在栅格里避墙，Kinetix 是连杆机器人/可抓取物体的接触动力学。
2. **任务多样性极广**：从导航到抓取到打游戏，**一个 policy 要泛化跨语义任务**，jaxnav 只有"导航"一种语义。
3. **欺骗性长程**：如 **Car-Ramp** 要先**远离**目标攒动量再冲——纯 RL 完全失败，必须微调才解。jaxnav 无此类欺骗结构。
4. **随机生成大量不可解/平凡关**：实体越多(L档)越可能不可解，**有用数据比例下降**——这恰好是 learnability/SFL 的主战场，也是它在 L 档失效的原因。

**S/M/L 分档标准**：按**场景中每种实体的数量**（实体少=简单）。74 个手设 holdout 关。

**饱和度（5B 交互后零样本解决率）**：**S≈85-90% / M≈40-50% / L≈5-15%（非单调上升）**。PLR/ACCEL 相比 DR 无增益(附录L)→ 选 SFL。**L 档是远未饱和的硬证据。**

### 5.5 连续控制 UED 经典难基准：TRACED/BipedalWalker（备查）

- TRACED（arXiv:2506.19997, 2025）在 BipedalWalker 6 地形整体 solved rate 仅 **0.36**（ACCEL 0.29），多地形仍负 return——远未饱和，但 **PyTorch DCD**，移植 JAX 成本高。

> 数据原件：本 session tasks 目录 6 个 agent 输出；关键 URL 见上表与待建记忆 [[harder-benchmark-candidates]]。
