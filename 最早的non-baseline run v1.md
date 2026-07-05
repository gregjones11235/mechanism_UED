# 最早的 non-baseline run: v1 (auctionC-s0-v1) 数据事实

> 记录 2026-07-03。这是**最早那个 auction (non-baseline) run**,即 experiment_design.md §10 反复
> 分析的 C 档。核心特征:**wandb step 标签有 +1900 污染**(续跑接力造成),分析必须先平移。
> 本文档固化 v1 的真实性能事实,避免以后再拿污染 step 得出"v1 全程落后"的错误结论。

---

## 1. 关键文件名(Oscar 日志 + wandb)

**wandb**:
- entity: `gregjones11235-brown-university`
- project: `DiCode-auction`
- run id: `dicode-auctionC-s0-v1`  (URL 里 `/runs/dicode-auctionC-s0-v1`)
- 指标: `evaluation/mean_return`(held-out 完整 Craftax 1024,= 论文 48.33 口径)
- run 最终 summary: `evaluation/mean_return=40.38`, `evaluation/mean_performance=17.87`
- ⚠️ wandb 里 v1 的 `_step` 是**污染标签**;`run.history(keys=[M], pandas=False)` 可取。

**Oscar .out 日志**(v1 是**多次 sbatch 续跑接力**,同一逻辑 run 分几个 job,step 递增):
| job .out | step 段(污染标签) | 说明 |
|---|---|---|
| `dicode-aucC-s0-3590029.out` | step 0→400 | 起始段 |
| `dicode-aucC-s0-3590474.out` | step 300→1000 | 续跑 |
| `dicode-aucC-s0-3591148.out` | step 1200→1900 | 续跑 |
| `dicode-aucC-s0-3593784.out` | **step 1900→10900** (5.4MB, 主体) | 续跑主段,`Resuming` |
- 更早的探路 job: `dicode-auc-gng-3580337.out`(gng=go-no-go 冒烟)、`3582100`/`3582377`(短)。
- 路径: `/users/jzhu223/*.out`。查 log 用 WSL ssh 喂脚本(见 [[oscar-workflow-env-cheatsheet]])。

---

## 2. ★★★ step 1900 污染(核心陷阱)

experiment_design.md §10.6 定论:**该 run step 标签永久污染(LR 正确但标签错位)**。
- **真实步 = wandb_label − 1900**。
- 成因:续跑接力时 step 计数偏移(3593784 从 `Resuming` + `step 1900` 起,但那其实是真实步 ~0 附近)。
- **任何 v1 vs baseline / vs r2 的对齐,必须先对 v1 做 −1900 平移**,否则整体错位 1900 步 →
  会把 v1 的"早期领先窗口"对到错误位置 → 误得"v1 全程落后"的错误结论(2026-07-03 本人踩过)。
- r2 run(v5y-s0-r2 / v5yA-s0-r2)是**从头干净跑,无污染**,step 即真实步,不平移。

---

## 3. ★ v1 真实性能曲线(已 −1900 平移,held-out evaluation/mean_return)

| 真实step | v1 | baseline | v1−base | 阶段 |
|---|---|---|---|---|
| 1000 | 24.4 | 18.5 | **+6.0** | 早期领先 |
| **2000** | **32.1** | 20.6 | **+11.6** | 🔥 峰值领先("超很多"的来源) |
| 3000 | 35.0 | 34.3 | +0.7 | baseline 追上 |
| 4000 | 37.3 | 36.6 | +0.8 | 拉锯 |
| 6000 | 39.5 | 39.5 | 0.0 | 打平 |
| 8000 | 40.3 | 42.2 | −1.9 | 被反超 |
| 10000 | 41.0 | 44.3 | −3.2 | 反超扩大 |

- **v1 早期(真实步 1000-2000)大幅超 baseline(+6 ~ +11.6)** —— 用一半真实步就到 baseline
  双倍步才到的 tier-2 水平(样本效率优势,§10.5)。
- **中期(~step 3000,tier-3 目标出现)baseline 追平,后期(step 8000+)反超** —— v1 final 40.4 < base 47.0。

---

## 3b. ★ v1 早期中标关的 proposer 构成(诊断 v5 早期偏弱,2026-07-03)

v1 是 **3 proposer 异质**,persona 按 index 配对(日志 `[auction] Loaded personas:
['breadth','ambitious','feasible'] (paired by index)`):
- **proposer_0 = breadth** / proposer_1 = **ambitious** / proposer_2 = feasible

中标关(auction top-k=10)的 proposer 构成,从 `[auction][voice] by_proposer` 汇总:

| proposer | persona | 早期中标占比(s1-11≈真实步<2000) | 全程 |
|---|---|---|---|
| proposer_0 | **breadth** | **28%** (60关中17关) | 19% (500中93) |
| proposer_1 | **ambitious** | **52%** (31关) | 54% (272) |
| proposer_2 | feasible | 20% (12关) | 27% (135) |

**读法**:
- v1 早期爆发主力 = **ambitious 52% + breadth 28%**(冲 + 广铺的组合);feasible 20% 造稳的 tier-2。
- breadth 早期(28%)比全程(19%)更活跃 —— 早期确实在广铺简单成就。

**对 v5(2×ambitious-coop + modeler)的含义 —— 早期偏弱的可能机制**:
- v5 用 **2 个同质 ambitious-coop** 替换了 v1 的 breadth/ambitious/feasible **三元异质**。
- 相比 v1 早期,v5 **同时丢了 breadth 的广度(28%)+ feasible 的稳度(20%)**,只留 ambitious 类。
- 且 v5 的 ambitious-coop 受 modeler 方向指导 + 合作补位约束,可能没 v1 ambitious 那么"放开冲"。
- ⟹ 早期"广铺 + 快速刷简单成就"的供给变薄,可能是 v5 早期没复现 v1 早期领先的原因之一。
  (待观察:两 r2 run 到真实步 1000-2000 是否仍压在 baseline 附近未爆发。)



**不是 auction 选择机制问题(机制全程健康),而是 proposer 课程"难度锚点太激进"**:
- graphml 证据:C 档造关目标 tier 中期漂移到 tier-3——s1-10 纯 t2(72%)→ s26-45 t3=72%。
- 后果:student 在深层关大量失败(进不去 mines/打不过 orc)→ 走不到"打铁"那步 → 独立铁器链关
  供给不足 → **RL 灾难性遗忘**:make_iron_pickaxe 斜率 −0.68、collect_iron −0.12(倒退!)。
- baseline 单 FM 更保守、持续供给独立 tier-2 关,把铁器链练透 → 反超。

**这正是 v5 要解决的问题**:modeler 诊断当前状态、把课程锚在 student 的 learnable band,
防止 ambitious 过早把目标推到 tier-3(见 v5_design.md §0 动机 + §8)。

---

## 5. 对当前工作的含义
- 任何**含 v1 的横比**都必须 v1 −1900 平移;两个 r2 run + baseline 不平移(它们 step 真实)。
- v5 的核心待验证命题:能否**复现 v1 的早期领先**,同时**避免中期 tier3 漂移被反超**。
  观察窗口 = 两 r2 run 到真实步 1000-2000(看早期领先)+ step 3000+(看 tier3 是否守住)。
- 相关记忆: [[wandb-fetch-eval-curves-local]]、[[dicode-step-semantics-and-official-eval]]、
  [[v5-runs-and-selection-variants-2026-07-03]]。
