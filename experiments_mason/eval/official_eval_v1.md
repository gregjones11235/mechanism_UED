# 官方口径评测 v1：三臂配对 mean_return（DiCode 协议，seed 0）

> 2026-07-08。工具 = `experiments/training/eval_checkpoints.py`（Alec 线所写、两分支同版，含 one-hot→67 修复）。
> 协议 = DiCode 官方：完整 Craftax、**固定 1024 个 held-out 程序生成世界**、mean episode return、one-hot 67 条件化。
> **seed=0 ⇒ 与 Alec 线（235B baseline 44.58 / auction 40.94 @update 10400）看的是同一批世界**，数字可进同一张表（补 compute 列）。
> 原始结果：`experiments_mason/eval/eval_{BASE14B,ARMA14B,ARMAB14B}_seed0.json`（含全 67 skill 分解）。

## 运行注记（复现要点）

- 本管线 defaults 未挂 `eval` 组 → 三个参数需 `+` 前缀：`+eval.ckpt_root=... '+eval.steps=[...]' +eval.tag=...`；
- 需带与训练一致的 gen_manager 覆盖（local_qwen14b ×2 + nomic）+ 两个 SERVER_URL 环境变量（脚本会初始化 GenManager）；
- `use_wandb=false`；逐臂顺序跑（显存）；每 checkpoint 约几分钟。
- checkpoint 留存实况：baseline 与 +A 为 300-1200（每 100）；+A+B（跑到 2400）滚动窗口只余 300 倍数 + 末 10 个 → **三臂共有配对点 = [300,600,900,1200]**。

## 主表（mean_return；括号 = mean_performance）

| update (env steps) | baseline `32v02vi9` | +A `85qid2ev` | +A+B `u1gjqror` |
|---|---|---|---|
| 300 (39M) | 12.75 (5.64) | 12.65 (5.60) | **13.46** (5.96) |
| 600 (79M) | 17.29 (7.65) | 17.06 (7.55) | **17.81** (7.88) |
| 900 (118M) | 18.10 (8.01) | 16.83 (7.45) | **18.54** (8.20) |
| **1200 (157M)** | 19.14 (8.47) | **23.36** (10.34) | 20.25 (8.96) |
| 1500 (197M) | — | — | 29.16 (12.90) |
| 1800 (236M) | — | — | 28.59 (12.65) |
| 2100 (275M) | — | — | **32.46** (14.36) ← 峰值 |
| 2400 (315M) | — | — | 29.81 (13.19) |

## 发现

1. **+A+B 是唯一在全部 4 个配对点均优于 baseline 的臂**（+0.71/+0.52/+0.44/+1.11）。温和但逐点一致——与 preflight"稳定提升课程质量、干预小而准（拒绝率 ~3%，均为 too_easy）"的机制画像吻合。
2. **两臂 payoff 时刻错开**：+A 在 ~1200 兑现（16.83→23.36，单点 +6.5 尖峰，此前 900 还落后 baseline 1.27）；+A+B 在 **~1500 兑现（20.25→29.16，一步 +8.9）**，随后 29-32.5 平台。157M 截止线恰切在 +A+B 起跳前 ⇒ 1200 格"+A > +A+B"很可能是 **payoff 时序差而非 B 有害**（单 seed 不能断言，两臂轨迹形状支持此读法）。
3. **尺度轴 headline**：14B + 显式机制在 **~315M 步（≈DiCode 2e9 预算的 16%）达到 mean_return ~30-32.5**，对照同口径下 235B baseline 中段 44.58（Alec 复现）/ 论文终值 48.33 —— **约 16% 算力摸到 235B 成绩的 2/3**。⚠️ 表述时注明：评测口径完全相同，但训练配方差异不止模型尺度一处（步数预算、单模型 vs 官方配比等），此为 regime 对照而非受控比较。
4. 交叉验证：baseline@1200 = 19.14 与 in-training eval（~19）一致，两条独立评测管线互证。

## 汇报口径（英文）

> "Under the official DiCode protocol (1024 fixed held-out Craftax worlds, seed-paired with the 235B line): the full method (+A+B) is the only arm above baseline at **all four** matched checkpoints (+0.4 to +1.1), while scheduling-only (+A) shows a single large end-point gain (+4.2 at 157M) after a mid-training investment phase. Beyond the matched window, +A+B keeps climbing to ~30-32.5 mean return at 315M steps — roughly two-thirds of the 235B baseline's mid-training score at ~16% of its compute (same eval protocol; training recipes differ in more than model scale). Single seed; end-point rankings carry noise of ~±1.5."

## 局限

1. 单 seed（=1 训练 / seed 0 评测）；+A 的 1200 为单点尖峰，排名不稳；平台期震荡 ±1.5 提示口径噪声带，配对点 <1 的差距勿过度解读。
2. 1500-2400 段无 baseline/+A 对照（checkpoint 已被滚动窗口清除），只能臂内描述。若需对照，须延长跑 baseline/+A（成本线性）。
3. 与 Alec 线同表时 compute 列必填；两线训练配方差异须在表注声明。

## 下一步

1. JSON + 本文档 push 至 `experiments_mason/eval/`；表格 + headline 发教授与 Alec（口径已统一，可直接讨论进 §5）。
2. 用 JSON 里的 67-skill 分解画三臂技能级对比（iron/diamond/dungeon 族的官方口径版本）。
3. 候选第二轮：延长 baseline/+A 至 2400 补齐长预算对照；seed 2,3；too_easy 阈值 0.85→0.7 敏感性。
