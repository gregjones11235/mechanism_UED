# Stage 4 实验设计 —— 方案 B（N 异质 generator + auction）实验路线

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
