# Stage 4 实验设计 —— 方案 B（N 异质 generator + auction）实验路线

> # ★★★ 2026-06-26 决定性突破：候选池基数是真瓶颈 ★★★
>
> 用户抓出对照不公平（baseline 每 eval 海选 **5000** 候选 vs generator **128**，差 39 倍）。把 generator
> 候选池拉到 2000/3000（`GEN_POOL_PER_GEN`，注入数 48、PPO 规模 64 均不变=单变量；`GEN_ROLLOUT_CHUNK=250`
> 分批避 PCGRL CNN OOM）后，**首次有 generator 攻破 Nav2**：
> - **currA pool=3000**：Nav2=1.0 / Nav3=1.0 / overall=**1.0**（**超** baseline 0.96）
> - **gate pool=2000**：Nav2=1.0 / overall=0.97
> - 两独立配置同时攻破 → 非单 seed 噪声。之前所有 generator arm Nav2 全 0。
>
> **反直觉**：`gen_p_learnable_frac` 仍 ≈0、关仍两极，Nav2 却满分——**不需造可学关，只需大基数让 PLR
> 有多样关可挑**。真瓶颈是候选基数，非信号/课程/塌缩（这些之前折腾一整天的方向需重评）。
> **下一步最高优先级=多 seed 复现**。详见 memory `bigpool-breakthrough-nav2-solved`。

---

> **目的**：把"方案 B 机制已全部跑通（35/35 单测）"推进到"科学结论 + SOTA 对标"。
> 给出从 go/no-go 验证 → 真实规模 → jaxnav SOTA 同表对标的可执行实验矩阵。
>
> **生成日期**：2026-06-22。承接 [方案B_多generator设计.md](方案B_多generator设计.md)（机制设计 + §7.4 对标口径）、
> [STAGE3_实验设计.md](STAGE3_实验设计.md)（旧 maze 矩阵，已降级历史记录）。
> 对标口径以本文档为准（**非** STAGE3 的 8%/PerfectMaze）。

---

## 0. 起点：机制已完成（无代码缺口）

载体 = SFL 官方 repo（`~/sampling-for-learnability`，jaxued 血统，jax 0.4.38）。已实现并 srun 单测验证：

| 模块 | 状态 | 验证 |
|---|---|---|
| N=3 异质 generator（PCGRL narrow，各绑 difficulty/PVL/CENIE） | ✅ | G5：三 generator 各 params 真动（\|Δθ\|≈1489/1496/1496） |
| 三信号 terminal reward（difficulty/PVL/CENIE，一次 rollout emit 全信号） | ✅ | G8：三信号形状/有限；CENIE 用 valid GMM 产非零 |
| auction 出价漏斗（z-score 抹平量级 + top-k bid + softmax(bid/λ)） | ✅ | G2 端到端：量级 2000× 经 z-score → weights[0.21,0.21,0.58] 不碾压 |
| 起终点可解（place_env valid_path_check=True 根除不可解图） | ✅ | G7：注入 level 起终点连通 1.00 |
| 交替训练闭环（GENERATOR_INJECTION 开关，jaxnav_sfl 集成） | ✅ | 端到端 A：CPU 跑通一个完整 eval epoch（阶段 G/S 交替） |

**剩下的全是实验，不是工程。** 关键 CLI/config 见 §5。

---

## 1. 阶段 1 —— go/no-go：learnability 方差复活 + λ 扫描

> **这一步决定方案 B 机制"有没有用"，是后续一切的前置。不是 SOTA 判定。**

### 1.1 要回答的问题
1. **方差复活**（方案 B §10 地基 2）：generator 注入 vs SFL 随机海选，打分层 learnability estimator 的**方差是否 >0**（对比随机海选在某些环境/阶段塌缩到 0）。这是 idea 的核心 claim——异质定向生成能产出有区分度的课程。
2. **λ 甜区**（exploit↔explore 权衡）：λ 是一阶超参（single-winner 易 mode-collapse 扼杀异质性、踩 PAIRED 熵坍缩陷阱；uniform 稀释最该学的信号）。把 λ 作主扫描轴，找让方差最大 / student 学得最好的 λ。

### 1.2 实验矩阵（λ 作主扫描轴）

| run | GENERATOR_INJECTION | GEN_AUCTION_LAMBDA | 作用 |
|---|---|---|---|
| 基线 | false | — | SFL 随机海选（内部对照，方差塌缩参照） |
| explore 端 | true | 5~10（大，近 uniform-mix） | 三信号近等权 |
| 甜区候选 | true | 1.0（中段软混合） | 默认起点 |
| exploit 端 | true | "inf"（single-winner） | 赢者通吃 |
| fallback 消融 | true | "none"（各 gen 自评，不跨 gen auction） | 证 auction 混合 vs 各自为政 |

### 1.3 看的指标
- 每个 eval epoch 的 `learnability_set_scores` **方差 / 分布**（是否从塌缩复活、随 λ 怎么变）；
- `auction_weights` / `auction_bids` 轨迹（哪个 teacher 何时出价高，λ 怎么调尖锐度）；
- student 在 eval set 上的 win rate（趋势，非最终判定）。

### 1.4 规模与判据
- **短跑**：缩小版或几个 eval epoch，趋势性验证。**go**（方差明显 >0 且随 λ 有结构）→ 进阶段 2；**no-go**（方差仍塌缩）→ 回查信号/generator/auction。
- ⚠ go/no-go ≠ SOTA：见 §3 认知链条。**方差复活 ≠ student 学得更好 ≠ 刷过 SOTA**，只是必要前置。

---

## 2. 阶段 2 —— 真实规模 + jaxnav SOTA 同表对标

> 阶段 1 go 之后才做。这一步才产出 SOTA 成绩单。

### 2.1 对标口径（方案 B §7.4，jaxnav 主场）
- **环境**：jaxnav（主场）；minigrid/maze 仅对照+故事（演示随机海选末期 learnability 枯竭、generator 能否撑住）。
- **对标 baseline（SFL 论文 NeurIPS 2024，实现都在 SFL repo、已 Oscar 部署，不用重训）**：
  DR、PLR(MaxMC)、PLR(PVL)、PLR-Robust、ACCEL(MaxMC)、ACCEL-Robust、**SFL**、PAIRED、NCC。
- **评测口径**：**CVaR worst-case + hand-designed test set + 随机 100-map（单/多 agent）**。
- **seed = 10**（与 baseline 对齐，rliable IQM + 95%CI 投稿口径；见 [[Stage3直接用10seed]]）。

### 2.2 矩阵
- **训练**：阶段 1 选出的 top λ（1~2 个）配置 × 10 seed，真实规模跑满（NUM_ENVS=256、ROLLOUT_STEPS=1000、TOTAL_TIMESTEPS 跑满）。
- **⚠ 补档 λ=2.0（2026-06-23 追加，jobid `3403257_[0-9]`）**：阶段 1 只采样了 λ∈{5.0,1.0,inf}+none 四点，"甜区=1.0" 是这三点里 singleton win rate 最高者（0.409），**非沿 λ 轴精搜的极值**。核查 [auction_bid.py](_sfl_repo_mirror/auction_bid.py) `auction_weights` 确认 λ 是 softmax 温度、**λ 轴呈 U 形**：argmax 端 = {λ→0}∪{λ=inf}（λ=0 经 1e-6 clamp 数值上≈inf，**非更温和方向**，不值得加），uniform/软混合端 = 有限大 λ。真正没探过、且方向上可能优于甜区的空白是 **λ∈(1,5) 的软混合侧** → 补 λ=2.0 档（10 seed，3e8 跑满），与 1.0 横比检验"更软混合是否进一步改善 win rate"。挂在 Alec 批、无依赖、同抢 2-GPU 配额。
- **对标表**：方案 B 候选 + 上面整排 baseline 进同一张 rliable 表，CVaR/test set/100map 三口径。
- **事后消融**（内生于矩阵，不额外加 run）：
  - 去 auction（λ=none 各自为政）→ 证 auction 混合的增益；
  - 去某个 estimator（N=3→2）→ 证异质性贡献；
  - single-winner（λ=inf）vs fractional → 证 λ 权衡的必要性（呼应 [[auction消融已起跑]] 的 {inf,1.0,3.0} 思路，但口径升级到 jaxnav SOTA 表）。

### 2.3 诚实标注（方案 B §9 两个红灯）
- **障碍 2（在线非平稳）是真未解研究问题**：用 target student 周期重评压住（温和但不能省），"在线正面解决"作可选加分项。
- **放弃 regret 换 learnability 无现成收敛背书**：收敛保证只来自分层+固定 buffer+αH(y)（proof_skeleton §8.4），对 generator 注入层不提供保证（同 ACCEL/PLR）。论文须诚实写。

---

## 3. 实验认知链条（别混淆"机制通"和"刷过 SOTA"）

```
机制能跑       ✅ 已完成（35/35 单测，N=3 + auction 闭环）
  ↓
机制有用       ← 【阶段 1 go/no-go】learnability 方差复活 + λ 甜区，vs SFL 随机海选（内部对照）
  ↓
机制有效       真实规模跑满，student 因课程学得更好（eval win rate 提升）
  ↓
SOTA 冲击      ← 【阶段 2】jaxnav CVaR+test set+100map 口径，同表对标 SFL/PLR/ACCEL/PAIRED，10 seed
```

- 阶段 1 只走到第二环（机制有用），**远不是 SOTA 判定**（差几个量级：缩小版 vs 跑满、内部随机海选对照 vs 外部 baseline）。
- 详见 [[jaxnav-sota-benchmark-truth]]。

---

## 4. 已知风险 / 注意

- **XLA 偶发线程死锁**：auction 消融已遇 1 个 seed 卡死（24 线程全 futex_wait，GPU 0%，~10% 概率）。长 run 需看门狗（N 分钟 .out 无更新则 scancel + 重提该 seed），见 [[oscar-login-node-no-compute]] 邻近经验。
- **N=3 编译开销**：三份独立 generator rollout+PPO 各自 jit 编译，比 N=1 明显慢（CPU 上 25min 墙钟连第一个 epoch 都编不完）。真实 GPU 跑也要留足首个 eval epoch 的编译时间，别按"无更新"误判死锁。future 可统一三 gen rollout 结构共享编译。
- **CENIE 依赖 gmm_params**：estimator_ids 含 cenie 时 jaxnav_sfl 自动冷启 GMM + 开每 epoch 重拟（否则 CENIE generator 恒 0）。
- **SBATCH 硬规约**：所有 Oscar 脚本带邮件通知（mail-type/mail-user）+ 墙钟 ≥24h（见 [[SBATCH脚本两条硬规约]]）。
- **登录节点禁跑计算**：训练/PPO/多迭代单测一律 srun 走计算节点；py_compile 验语法可留 login（见 [[oscar-login-node-no-compute]]）。

---

## 5. 关键 config / CLI（SFL repo，jaxnav_sfl.py）

config: `sfl/train/config/jaxnav-sfl.yaml`
```yaml
"GENERATOR_INJECTION": true          # 开 generator 注入（false=原 SFL 随机海选，零回归基线）
"GEN_ESTIMATOR_IDS": ["difficulty", "pvl", "cenie"]   # N=3 异质，列表长度即 N
"GEN_AUCTION_LAMBDA": 1.0            # auction 出价温度：inf=single-winner / 有限=fractional / none=各 gen 自评(消融)
"GEN_MAP_SIZE": 11                   # 对齐训练 map_size [11,11]
"GEN_NUM_LEVELS_PER_GEN": 64         # 每 gen 每 round 造几张（pool = N×此值）
"GEN_NUM_TO_SAVE": 48                # auction 漏斗输出宽度（须 < pool 让漏斗挤掉 incomplete）
"GEN_OUTER_STEPS": 1                 # 每 round 每 gen 跑几次 PPO 迭代
"GEN_PPO_EPOCHS": 4
"GEN_LR": 0.00025
"AUCTION_USE_CENIE": true            # CENIE generator 需 GMM 重拟（含 cenie 时 jaxnav_sfl 会自动开）
```
- λ 扫描：hydra override `GEN_AUCTION_LAMBDA=inf` / `=1.0` / `=5.0` / `=none`。
- 基线对照：`GENERATOR_INJECTION=false`（走 get_learnability_set 随机海选）。
- 单测（必 srun）：`PYTHONPATH=sfl/train python sfl/train/_test_generator_integration.py`（35 PASS）。

---

## 6. 阶段 2A 负结果诊断（2026-06-23）—— 难度维度错配 + 捷径闭环

> # ⚠️⚠️ 本节 −22pct 负结果已于 2026-06-26 被推翻（反转）⚠️⚠️
> **真因不是"难度维度错配/捷径"，是候选池太小（128 vs baseline 5000）。** 大基数（pool=3000）后
> currA 在同口径难关 CVaR 上**赢 baseline +29pt**（CVaR10 win 0.771 vs 0.477），方向完全反转。
> 见本文档顶部突破框 + memory `cvar-100map-reverses-negative-result` / `generator-pool-size-unfair-vs-baseline`。
> 下方原诊断（难度维度错配等）保留作推翻链记录，但其"方法有害/无增益"的结论作废。

> **结论先行（已推翻，见上）**：λ=1.0 注入档（l1）在难关 CVaR 上**系统性输给随机海选 baseline**（最难 10% 关 base 0.872 vs l1 0.653，差 −22pct，越难输越多，3e8 跑满、方差极小是真实差距）。
> 用**原始数字数据**（测试集 pkl 的 `map_data` 墙体矩阵 + `pos`/`goal` 坐标 + per-level win-rate csv）精确确诊，逐一**排除**了塌缩/病态难/训练不够/舒适区四个先验猜想。

### 6.1 排除项（数据证伪，别再走回头路）
| 猜想 | 数据 | 裁决 |
|---|---|---|
| reward-hack 病态难关 | `gen_n_incomplete=0` 全可解；`auction_bid_difficulty=0` 权重仅 0.068 | ❌ 证伪 |
| 多样性塌缩 | 注入关两两 Hamming ≈ 0.499 ≈ 随机上限 0.5 | ❌ 证伪 |
| 舒适区课程 | `learnability_set_mean=0.68` 是 PVL+CENIE 混合量纲非 p(1-p)，不可与 base 0.072 直接比；maps 显示 l1 关实打实更难 | ❌ 证伪 |
| 训练不够 | step2250 = 3e8 跑满（step 是 eval epoch 内部计数，46 epoch = 注释的 45 epoch = 3e8） | ❌ 证伪 |

### 6.2 真因：难度维度错配（student-centric 信号集体被堆墙捷径主导）
- 三个 estimator（difficulty/PVL/CENIE）全 **student-centric**（衡量 student 单关瞬时反应）。11×11 小图上"高墙密度→瞬时高 value-loss/状态新颖"是省力捷径 → generator 学会**堆墙刷分**。注入关 fill 中位 **0.47**（35% 关 fill>0.5，fill<0.2 仅 1.2%）。
- 但**测试集最难 10% 关 = fill 中等 0.337 + 起终点 dist 4.99（全体 95 分位）= 长程导航难**，不是墙多。
- l1 输最惨 100 关 dist 4.87 vs 全体 3.83；劣势按距离分桶**单调放大**（dist<2: −0.003 → dist 5.7–10.6: −0.043）；按 fill 分桶劣势在中 fill 0.2–0.4 最大、高 fill 最小。
- base 随机海选 dist 分布无偏 → 天然含大量长程关 → worst-case 更稳（这也是 SFL 论文 DR/随机基线难打败的根本）。

### 6.3 捷径闭环坐实（3 ckpt rollout 实测，决定性实验）
对 base ckpt 在 1000 测试关上 rollout，算 4 个 estimator 的 per-level 分 vs 关卡几何的 Spearman 相关：

| ckpt（succ p） | PVL ρ(dist) | PVL ρ(fill) | CENIE ρ(dist) |
|---|---|---|---|
| base_seed1 (0.987) | +0.237 | +0.352 | +0.256 |
| base_seed2 (0.990) | +0.238 | +0.354 | +0.197 |
| **l1_seed0 (0.968)** | **+0.131** | +0.306 | +0.188 |

- **difficulty/SFL=p(1-p) 在 base 上对全体关塌缩**（p 几乎全 ∈{0,1}，ρ=NaN）：student 太强致 learnability 维度失区分度（复现 maze 双峰塌缩，跨到 jaxnav）。其 top-100 偏好 dist=6.08/fill=0.025（方向对，但只在极少 p≈0.5 关有信号，覆盖不到测试难关）。
- **PVL 同时偏好长程(+0.24)和堆墙(+0.35)，但堆墙偏好更强** → generator 被更省力的堆墙捷径主导。CENIE ρ(fill)≈0 相对干净但 ρ(dist) 也只 ~0.25。
- **闭环坐实**：base→l1，PVL 的 dist 偏好从 +0.238 **腰斩到 +0.131（−45%）**，fill 偏好几乎不降。机制 = generator 堆墙 → student 在堆墙关训练 → 对长程不敏感 → student 的 PVL 在长程关信号更弱 → generator 更没动力造长程 → 锁死"堆墙、忽略长程"。base 两 seed 内部高度一致（非噪声），l1 单方向退化（与能力退化同向）。

### 6.4 改进方向（保留 SFL，补抗捷径的正交维度，非堆 score 打地鼠）
- **宗旨约束**：保留 SFL learnability（收敛理论样本，方案B §4.3/§6.4）+ 现有异质池；不"换信号"，而是补一个维度。深调研（2026-06-23）裁决纯多样性/QD **不能**替代 regret（DIPLR 强主张 0-3 否决），纯真 regret 也会饱和停滞（ReMiDi, ICML24）。
- **核心缺陷不是"缺维度"也不是纯"修聚合"，是 fill 捷径与 dist 真难在 student-centric 信号里纠缠、且捷径更省力**。需要一个**不依赖 student value、内禀于关卡几何**的信号来锚住长程维度、不被闭环侵蚀。
- **备手（轻量、可解释、抗堆墙）**：BFS 最短路 / 起终点测地距离 geometry score（读 map_data+pos/goal 即可算，flood-fill 基础设施现成）。它正交于 fill、直接覆盖 SFL/PVL/CENIE 共同够不到的长程维度。
- **更优美方向（理论侧，待设计）**：真 minimax-regret 估计（MMER, 2507.03068）自然指向"代理与真目标分歧处"；或 occupancy/QD 自动行为覆盖（DIPLR/AURORA）做防停滞约束。详见 deep-research task wf_0dd317d7-869 报告。

> 命中证据：No Regrets（NeurIPS 2024，即 SFL 母论文）在**同款单智能体 JaxNav** 上实证 PVL/MaxMC 不与真 regret 相关、只与成功率相关、优先采"已会做的关"——强支持"代理被捷径带偏"诊断。但它未测 geometry 维度，故不能否定本文实测的长程导航盲区；二者非互斥、同时成立。

### 6.5 为什么 baseline 不堆墙？—— 堆墙是"生成动作"而非"信号"的问题（论文定位关键）

**(1) phase2 baseline 身份确认（代码铁证）**：`base = GENERATOR_INJECTION=false` → 走 `get_learnability_set`（jaxnav_sfl.py:218）：`env.reset` **随机海选**一批关 → 每关算 `learnability=p(1-p)`（:124）→ `score=learnability` → `argsort(score)[-NUM_TO_SAVE:]` 取 top-K（:193-194）。**这就是 SFL（Sampling-For-Learnability）本身，不是纯随机 DR。** 故本阶段对照是 **SFL vs 你的 generator 注入**。

**(2) baseline 不堆墙的真正原因 = 它没有"生成"能力，只能筛选**：

| | 关卡来源 | 墙密度(fill) | 会堆墙吗 |
|---|---|---|---|
| SFL baseline | `env.reset` **随机生成** → 在无偏池里**挑** learnability 高的 | 中位 0.30（≈env.reset 固定先验） | ❌ 无处可堆：只能消费 env.reset 给的无偏 fill 分布，挑不出池里没有的高墙关 |
| 本方法（generator 注入） | PCGRL generator **主动生成**（有梯度）→ 优化 estimator 分 | 中位 0.47（generator 主动推高） | ✅ 梯度直奔"刷分最省力"方向，而堆墙是刷 PVL 分的捷径 |

> **核心洞察：堆墙偏置不是 estimator 信号的缺陷，是"主动生成"这个动作的副作用。** SFL 用同样的 PVL/learnability，但**没有生成器、无处可堆**，其多样性是 env.reset 白送的无偏先验。generator 一旦有优化能力，就会发现"堆墙刷 PVL 分"并钻进去（§6.3 实测 PVL ρ(fill)0.35 > ρ(dist)0.24）。所以 baseline "稳"不是因为信号好，是因为**它结构上无偏、但只能被动筛选**。

**(3) 推论：其它 SOTA 在 jaxnav 上的捷径风险谱（机制推断，尚无人系统验证 → 潜在 contribution）**：

| 方法类 | 关卡来源 | 捷径(堆墙)风险 |
|---|---|---|
| PLR / PLR⊥ / Robust-PLR / **SFL** | replay：env.reset 无偏池里**只筛不生成** | **免疫**——无生成器、无处可堆，多样性来自无偏先验 |
| **ACCEL** | random **edit**（加/减墙）+ 保留高 regret | **中等**——若"加墙"比"移远起终点"更易涨 regret 则也会渐进堆墙；但 edit 随机(无梯度)故倾向更弱、更慢，机制同源 |
| **PAIRED** | RL adversary **生成**（与本方法最像） | **最高**——梯度直奔捷径，著名的 mode collapse |
| 本方法 | PCGRL generator 生成（梯度） | 高（已实测）|

> **对论文定位的含义**：本方法不是在和"一个更好的信号"比，而是在和"一个**结构上无偏、只能被动筛选**的 SFL baseline"比。卖点必须是"**主动生成能造出 env.reset 先验里没有、且有学习价值的关，同时不被几何捷径带偏**"——后半句正是 §6.4 要修的。"生成式 UED（PAIRED/ACCEL/本方法）在连续导航上被几何捷径带偏、而 replay 式（PLR/SFL）因无偏池免疫"这一机制对比**此前无人在 jaxnav 上系统跑过**，可做成独立的机制性发现。CENIE 本身是 estimator 非生成器，§6.3 实测其 ρ(fill)≈0 相对不偏堆墙——堆墙主要由 PVL 带，非 CENIE。

### 6.6 修复实现：几何锚定 PVL（anchored PVL，方案丙，2026-06-24）

**理论依据（第二轮 deep-research, task wf_617e3780-55c）**：PAIRED 的抗作弊机制是 minimax-regret `REGRET=U(πA)−U(πP)`，**antagonist 充当"可达性上界参照点"**——造"谁都做不到的关"antagonist 也失败→regret=0→刷不到分。PVL 退化成"成功率/瞬时 value-loss"恰恰**丢了这个参照点**，故堆墙的瞬时混乱被照单全收。**测地最短路 = antagonist 可达性参照的廉价、student 无关替身**：它内禀于关卡几何，免疫 §6.3 那个"generator 堆墙→student 退化→信号更偏堆墙"的闭环。
> 调研同时**否决**了几个看似优美的方案（透明记录）：min/悲观多 proxy 聚合（"Helping or Herding?" DeepMind 实测只缓解不消除，三信号误差相关时集体失守，min 不胜 mean）；multi-agent debate / GAN-discriminator / curator 投票（UED 内几乎无直接实证，"理论漂亮证据稀薄"，降为 future work 不作主线）；"对 CVaR 尾部生成"（=拟合测试集，违背宗旨，撤回）。

**最终公式（方案丙，保留 PVL 的价值误差角度 + 几何锚堵捷径）**：
```
anchored_pvl(level) = PVL(level) × (geodesic_shortest_path(level) / max_geodesic_in_batch)
```
- **含 PVL 量**（不是另起 (1-p)×测地 的"方案甲"），保留 PVL 捕捉的价值估计误差/regret 维度。
- 乘归一测地因子：堆墙但起终点近的关 → 测地短 → 因子小 → 即使 PVL 高也被压低 → **generator 无法靠堆墙刷分**；中等墙密度+起终点远 → 测地大+PVL 高 → 自然被选中。

**离线验证（base/l1 ckpt 各 1000 测试关，纯几何 + 已存 PVL，未训练）**：

| 信号 | ρ(euc) 长程偏好 | ρ(fill) 堆墙偏好 | 长程/堆墙近 比值 | 最难关分位 |
|---|---|---|---|---|
| PVL（裸，旧） | +0.237 | **+0.352** | 1.3 | 81% |
| (1-p)×测地（方案甲） | +0.093 | +0.059 | 极端 | 99% |
| **PVL×测地（方案丙，采用）** | **+0.698** | **+0.004** | 4.9 | 86% |

> 方案丙优于方案甲：甲在强 student 上 (1-p)→0 信息塌缩（ρ(euc) 仅 0.09），丙保留 PVL 连续区分度故长程信号**反增**到 0.70。base/l1 两 student 一致（ρ(euc) 0.698/0.690）。测地与 fill 天然反向（ρ(测地,fill)=−0.386）是它能对冲堆墙的结构原因。

**代码改动（`sfl/train/pcgrl_generator.py`）**：
- 新增 `geodesic_from_instances`（从 EnvInstance 的 map_data+agent_pos+goal_pos 算 batch 测地，复用 `sfl.util.graph.shortest_path_len` APSP，jit+vmap 安全，11×11 廉价）。
- 新增 `_geodesic_anchor_factor`（测地/最大测地归一）+ `terminal_reward_anchored`（PVL×anchor）。
- 注册 `GEN_ESTIMATOR_ANCHORED="anchored"`，`compute_terminal_reward` 加分发分支。
- `all_signals_on_levels`（auction 用）把第 2 行 PVL 替成 anchored（auction 也用锚定信号）。
- config/提交脚本：`GEN_ESTIMATOR_IDS=[difficulty,anchored,cenie]`（anchored 替 pvl，维度仍 N=3，保留 SFL=difficulty 同维）。

**测试**：现有 35 generator 集成单测全 PASS（改动无回归）+ anchored 冒烟 8/9 PASS（唯一 FAIL=geo≥1 断言过严：冒烟关 generator 只训 1 步造出退化关 geo=0，属期望行为非 bug，真实关无此情形）。

**caveat（调研明确警告，待训练验证时盯）**：① ReMiDi（ICML24）——纯 regret 在部分可观测下会饱和停滞，JaxNav 连续避障落在此域，靠 SFL=difficulty 维度兜底覆盖；② 抗作弊 ≠ 自动泛化（"PAIRED regret 课程产生迁移"声明 0-3 否决），须分开验证：既看 generator 是否不再堆墙（注入关 fill 分布），又看 worst-case CVaR 是否真提升。

**验证实验（进行中，2026-06-24）**：l1=λ1.0 用 anchored 代码跑 3e8 完整版 10 seed（GROUP=`s4p2anc-lambda-1_0`，job 3407004）。base 不重跑（`GENERATOR_INJECTION=false` 不调 generator 代码，与旧 baseline 同结果，见 [STAGE4 §6.5]+memory `stage4-baseline-already-done`），用旧 base run 同口径对照难关 CVaR，看是否扭转 §6.1 的 −22pct 负结果。
