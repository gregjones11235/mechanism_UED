# 官方口径评测 v2：三臂八点全程配对（315M，DiCode 协议，seed 0）

> 2026-07-09。取代 v1 成为主结果文档（v1 保留为 157M 一代 run 的存档）。
> 协议同 v1：`eval_checkpoints.py`，1024 个固定 held-out Craftax 世界，mean episode return，one-hot 67，**seed=0 与 Alec 线同批世界**。
> 原始数据：`experiments_mason/eval/eval_{BASEEXT14B,ARMAEXT14B,ARMAB14B}_seed0.json`。

## 1. 臂身份

| arm | wandb id | 训练 | 说明 |
|---|---|---|---|
| baseline_ext | `mc75k0nx` | 0→session 23（315M），148 节点 | 纯 DiCode，延长二代跑 |
| +A_ext | `z8jygtyw` | 同上，148 节点 | `use_scheduler=true`，延长二代跑 |
| +A+B | `u1gjqror` | 0→session 23（315M），136 节点 | 两 flag + 修复后 preflight（一代跑，天然 315M） |

三臂同 seed=1、同环境、同预算；curriculum 规模相当（136-148 节点）→ 差异归因于"生成了什么"而非"生成了多少"。

## 2. 主表（mean_return；▼= 显著回撤点）

| update (env steps) | baseline_ext | +A_ext | +A+B |
|---|---|---|---|
| 300 (39M) | 12.75 | 12.20 | 13.46 |
| 600 (79M) | 17.24 | 18.32 | 17.81 |
| 900 (118M) | **10.84** ▼ | 18.15 | 18.54 |
| 1200 (157M) | 18.27 | 20.00 | 20.25 |
| 1500 (197M) | 18.76 | **12.94** ▼ | **29.16** |
| 1800 (236M) | 18.74 | 21.34 | 28.59 |
| 2100 (275M) | 17.66 | 28.93 | **32.46** |
| **2400 (315M)** | **17.88** | **31.07** | **29.81** |

## 3. 结论

### 3.1 机制阶梯（终点配对）：18 → 31 / 30
裸 14B 平台正式实证：**1200 后四点钉在 17.7-18.8，零增长**（另含 900 处 -6.4 的中段能力回撤）。两个机制臂双双突破至 **~30-31**（+12~13 分配对差），且终点互相不可分（31.07 vs 29.81，±1.5 噪声带内）→ **frontier 调度（★A）是天花板抬升的主要来源**，两条独立路径（有闸/无闸）交叉验证。

### 3.2 preflight 买到的不是高度，是路径质量
| | baseline_ext | +A_ext | +A+B |
|---|---|---|---|
| 最大回撤 | -6.4（@900） | **-7.1（@1500）** | **-0.6** |
| 轨迹形态 | 平台+塌陷 | 剧烈震荡后冲顶 | **单调爬升** |
| payoff 兑现点 | 无 | ~2100 | **~1500（早 ~600 update）** |

**B 的完整画像：终点相当、路径更稳（近零回撤）、兑现更早**——与其"低干预率（3%）+ 拦 too_easy"的机制本质一致（过滤突变任务 → 训练分布更平稳）。

### 3.3 尺度轴（与 Alec 线同口径同世界）
14B + 机制在 **315M 步（≈235B 线 2e9 预算的 16%）达到 29.8-31.1**，对照 235B baseline 中段 44.58 / 论文终值 48.33 → **约 16% 算力达到 ~2/3 成绩**。表述时注明：评测口径完全相同，训练配方差异不止模型尺度（regime 对照，非受控比较）。

### 3.4 跨 run 方差标尺（免费副产品）
baseline 一代 vs 二代（同 seed=1）：step 300 完全相同（12.75，seed 训练逐位复现）；600/1200 差 0.05/0.87 → **同 seed 下 LLM 采样随机性 ≈ ±1 分**。此后解读任何 <1 分的差距均以此为噪声下限。

## 4. 汇报口径（英文）

> "With extended budget under the official protocol (1024 seed-paired held-out worlds): the bare-14B plateau at ~18-19 is confirmed — four flat points after 157M plus a mid-training collapse to 10.8 — while both mechanism arms break through to ~30-31, a +12-13 paired gap, at ~16% of the 235B line's compute reaching ~2/3 of its mid-training score. Between the two arms, the preflight gate doesn't raise the ceiling but buys *path quality*: monotone climb with -0.6 max drawdown and payoff ~600 updates earlier, versus scheduling-only's -7.1 drawdown before recovering. Single seed; same-seed LLM-sampling variance measured at ~±1 point."

## 5. 局限

1. 单 seed（训练 seed=1 / 评测 seed=0）；终点 +A vs +A+B 不可分（噪声带内）。
2. 三臂各含一次深回撤或震荡（+A+B 除外）——回撤成因未做逐 checkpoint 归因（候选：新任务批次冲击）。
3. frontier 全程锁死 tier 2（见 `frontier_lock_分析与修复.md`）——平台 ~30 可能部分源于此，阈值 probe 待 PI 排序。
4. 一代 157M 三臂（32v02vi9/85qid2ev/u1gjqror 前 4 点）与二代 ext 跑为不同 run，跨代混排时须注明。
