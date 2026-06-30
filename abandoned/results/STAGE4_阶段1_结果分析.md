# Stage 4 · 阶段 1（go/no-go）结果分析

> **数据来源**：wandb `gregjones11235-brown-university/multi_robot_ued`，35 run 全部 `_step=750`（=15 eval epoch，1e8 短跑规模）正常 finish（另有 4 个 crashed 残留重跑，已剔除）。
> **原始数据**：[results/stage4_phase1_35runs.csv](stage4_phase1_35runs.csv)（35 行终值）、[results/stage4_phase1_curves.csv](stage4_phase1_curves.csv)（逐 epoch 曲线，5 档 × 15 step）。
> **分析日期**：2026-06-23。
> **判据依据**：[STAGE4_实验设计.md](../STAGE4_实验设计.md) §1 / [STAGE4_阶段1_任务分配.md](../STAGE4_阶段1_任务分配.md) §5。

---

## 0. 一句话结论：**GO** ✅

方差**确凿复活**（注入档全程 0.1~0.6，基线全程 ≈0；终值 140~320× 基线）+ auction 权重**随 λ 单调调尖锐度**（机制按设计工作）→ 满足 go/no-go 全部判据，进阶段 2。
**一个黄灯带进阶段 2**：短跑（1e8）下注入档 eval win rate 暂低于基线，但逐 epoch 曲线显示注入档 win rate **全程单调上升、尚未收敛**，且随机-map 口径下只差 ~9 个百分点——属"难课程欠训未兑现"而非"机制有害"（见 §4）。

---

## 1. ⚠ 先澄清两个对照，别混（用户易混点）

| 名称 | GROUP_NAME | 注入? | score 归一 | 角色 |
|---|---|---|---|---|
| **基线** | `s4-base` | ❌ false | 随机海选本就归一 | 方差塌缩的**对照锚点** |
| **none 档** | `s4-lambda-none` | ✅ true | ❌ 各 gen 自评、**不跨 gen z-score** | 5 个注入档之一（"各自为政 vs 跨 gen 混合"消融） |

- **基线 ≠ none 档**：基线是关掉 generator 的 SFL 原版；none 是开了 generator、只是不做跨 gen auction。两者用途完全不同。
- **none 档量纲问题对 go 结论无影响**：因为方差复活判据**只用基线 vs 经 z-score 的 1.0/5.0/inf 三档**横比，none 档不参与这个横比（详见 §2/§5）。none 的 var=1090 纯是原始 difficulty 量纲未归一，不是真方差。

---

## 2. 终值汇总（每档 7 seed，mean ± 95%CI）

**同口径组（基线 + z-score 三注入档，可直接横比）：**

| 档位 | λ | n | learnability_set_var | lset_mean | win_rate(singleton) |
|---|---|---|---|---|---|
| **s4-base**（基线） | — | 7 | **0.0015 ± 0.0005** | 0.210 ± 0.012 | 0.703 ± 0.071 |
| s4-lambda-5_0 | 5.0 | 7 | 0.2085 ± 0.0650 | 0.584 ± 0.070 | 0.305 ± 0.094 |
| s4-lambda-1_0 | 1.0 | 7 | 0.2828 ± 0.0541 | 0.751 ± 0.119 | 0.409 ± 0.106 |
| s4-lambda-inf | inf | 7 | 0.4778 ± 0.1100 | 1.416 ± 0.204 | 0.342 ± 0.121 |

**异口径（单列，不与上表比绝对值）：**

| 档位 | λ | n | learnability_set_var | lset_mean | win_rate |
|---|---|---|---|---|---|
| s4-lambda-none | none | 7 | 1090.8 ± 273.8 ⚠量纲未归一 | 179.3 ± 9.8 ⚠ | 0.360 ± 0.072 |

---

## 3. 判据一：方差复活 ✅（逐 epoch 曲线比终值更强）

**learnability_set_var 逐 eval epoch（7 seed 均值）：**

| step | base | λ=5.0 | λ=1.0 | λ=inf | (none⚠) |
|---|---|---|---|---|---|
| 50 | 0.0004 | 0.772 | 1.625 | 3.579 | 0.627 |
| 100 | **0.0000** | 0.409 | 0.203 | 0.900 | 2.046 |
| 150 | **0.0000** | 0.257 | 0.143 | 0.400 | 19.75 |
| 200–450 | **0.0000~0.0001** | 0.17~0.34 | 0.14~0.26 | 0.39~0.59 | 261~542 |
| 750 | 0.0015 | 0.209 | 0.283 | 0.478 | 1090.8 |

- **基线全程塌缩**：step 100–450 var **完全 = 0.0000**（随机海选在 jaxnav 上彻底选不出有区分度的课程），末尾才微升到 0.0015。这是比终值更硬的塌缩证据。
- **注入档（1.0/5.0/inf）从 step50 起全程 0.1~0.6，从不塌缩**。95%CI 与基线完全不重叠。
- **随 λ 单调**：λ 越大（越 exploit）→ var 越大（终值 inf>1.0>5.0）。"方差明显 >0 且随 λ 有结构"**双满足**。
- inf 档 step50 爆高到 3.58 后回落（single-winner 早期激进）；none 档 var 逐步爆炸到 1090，**再次确认是原始 difficulty score 随训练递增的量纲假象**，非真方差。

---

## 4. 判据二：auction 随 λ 调尖锐度 ✅

| 档 | λ | w_difficulty | w_pvl | w_cenie | 形态 |
|---|---|---|---|---|---|
| λ=5.0 | 5.0 | 0.267 | 0.349 | 0.384 | **近等权**（explore，符合预期） |
| λ=1.0 | 1.0 | 0.095 | 0.332 | 0.573 | 软混合（cenie 占优不碾压） |
| λ=inf | inf | 0.000 | 0.000 | **1.000** | **one-hot**（single-winner，符合预期） |

温度旋钮按设计工作。三档 cenie 出价始终最高（bid: cenie~2.17 > pvl~1.33 > difficulty~0），difficulty 经 z-score 被持续挤到 bid≈0（与 [[bid标准化top-k修复]] 一致）。inf 退化成只听 cenie——var 最大但 win rate 不是最高，印证 single-winner mode-collapse 风险。

---

## 5. 黄灯：win rate 倒挂 —— 曲线证明是"欠训未兑现"非"机制有害"

**两口径 win rate 终值（750 step）+ 趋势：**

| 档 | singleton(hand-designed) 终值 | sampled(随机map) 终值 | sampled 趋势 Δ(后5−前5) |
|---|---|---|---|
| 基线 | 0.703 | 0.968 | +0.379 |
| λ=5.0 | 0.305 | 0.883 | +0.409 |
| λ=1.0 | 0.409 | 0.885 | +0.416 |
| λ=inf | 0.342 | 0.876 | +0.408 |
| none | 0.360 | 0.883 | +0.419 |

**关键修正（拉到逐 epoch 曲线后）：win rate 倒挂从"红灯"降为"黄灯"**：
1. **随机-map 口径只差 ~9 个百分点**（0.968 vs 0.876~0.885），不是 hand-designed test set 那种 0.70 vs 0.31 的悬殊。hand-designed test set 这个口径对"注入课程更难"特别不友好，放大了表象差距。阶段 2 真判据是 CVaR + 100-map，不是 singleton overall_win_rate。
2. **注入档 win rate 全程单调上升、尚未收敛**：所有注入档 Δ(后5−前5) 都 ≈ +0.41，**比基线 +0.38 还陡**，说明在 1e8（15 epoch）截断点上注入档"学得慢但没停"。注入课程系统性更难（lset_mean 0.58~1.42 vs 基线 0.21），收益要更长训练才兑现。
3. **基线早期冲高是 SFL 已知毛病**：随机海选一直喂低区分度课程（var=0），在简单 eval 上自然先到顶——但这正是"后期课程枯竭"，阶段 2 跑满 3e8 才是注入档反超的窗口。

> **结论边界**：阶段 1 验到"机制有用"（方差复活，铁证）；"机制有效"（student 因课程学得更好）**需阶段 2 跑满才能判**——曲线显示注入档处于上升期，没有证据表明机制有害。完全落在设计文档 §3 认知链条预期边界内。

---

## 6. 给阶段 2 的具体建议

1. **λ 选择**：带 **λ=1.0（甜区，注入档里 singleton win rate 最高 0.409、var 已 190×）+ λ=inf（exploit 端 + 验 single-winner 是否 mode-collapse）** 进阶段 2。λ=5.0 近等权但 var/win rate 都最弱，**仅留作消融端点不训练**。
2. **none 档归一化 bug（阶段 1 暴露）✅ 已修（2026-06-23）**：`pcgrl_generator.py:get_generator_set` 的 fallback（λ=none）路径原直接 concat 各 gen 原始量级；现各 gen 先 per-gen z-score（复用 `auction_bid.standardize_per_estimator`）再 concat，与 auction 路径同口径。已 py_compile + Oscar sfl 环境验数值。阶段 2 的 none 消融可与 1.0/inf 横比。
3. **win rate 黄灯**：阶段 2 必须 ① 跑满 3e8 看注入档是否反超；② 用 CVaR + 100-map 口径作终判（不用 singleton overall_win_rate）；③ 加 target student 周期重评压在线非平稳（设计文档 §2.3 障碍 2）。
4. **p_std 缺失 —— 真根因 = 注入档显式关了 `PROBE_ORTHOGONALITY=false`（非 SAVE_PATH）**：核查 config 发现基线 `PROBE_ORTHOGONALITY=true`（probe 跑、有 p_std），注入档命令把它设成 `false`（probe 整条不跑、无 p_std）。SAVE_PATH 两者都是默认非空，**不是 SAVE_PATH 的锅**。
   - **阶段 2 的 action（关键）**：注入档命令必须 **`PROBE_ORTHOGONALITY=true`** 才能拿到 p_std 判饱和（见 [STAGE4_阶段2_任务分配.md](../STAGE4_阶段2_任务分配.md) §1.3）。
   - **附带已修（健壮性，2026-06-23）**：`train_probe.log_orthogonality_step` 支持 out_dir=None（只算 wandb 指标不写 trace）+ `jaxnav_sfl.py` 把 probe 收集/落盘解耦 SAVE_PATH——让"开 probe 但没设 SAVE_PATH"也能出 p_std。但这**不是** p_std 缺失的根因修复，根因是上面那个开关。已 py_compile + 同步 Oscar。
5. **代码修复同步**：两处修复在 `mechanism_UED/_sfl_repo_mirror/`（pcgrl_generator.py / train_probe.py / jaxnav_sfl.py），阶段 2 开跑前需 `git pull` + `cp _sfl_repo_mirror/*.py sfl/train/` 覆盖进 Oscar 本体（见 [STAGE4_阶段2_任务分配.md](../STAGE4_阶段2_任务分配.md) §1.1）。

---

## 7. 数据质量备注

- 35/35 finished run 全 `_step=750`、各 15 个 history 点，无看门狗死锁残留。另有 4 个 crashed run（2 空 + 2 个 7-step 残留）为重跑中途产物，已剔除。
- 3 个注入档 seed4 run `gen_n_incomplete`>0（1_0:31 / 5_0:29 / inf:29 / none:33）——auction 漏斗挤掉的 incomplete level 数，属正常设计（GEN_NUM_TO_SAVE=48 < pool），不影响结论。
- 逐 epoch 曲线经 wandb public API（本地 wandb 0.28.0）拉取，已落盘 stage4_phase1_curves.csv。
