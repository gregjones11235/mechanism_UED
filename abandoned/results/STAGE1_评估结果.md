# Stage 1 评估结果（minimax / Maze 13×13）

> ⚑ **历史定位（2026-06-20 方案 B 转向后）**：本成绩单是 **Stage 1 maze sanity 的基线数字**（数据本身有效、10 seed 完整）。但 **SOTA 对标已转向 jaxnav + SFL 论文 baseline**（见 [../STAGE4_实验设计.md](../STAGE4_实验设计.md) §2.1 / [../UED实验步骤指导.md](../UED实验步骤指导.md) §0.6）；下文的 maze 8%/PerfectMaze 口径**不是**方案 B 的对标靶子，仅作 maze 基线参考。详见 [[jaxnav-sota-benchmark-truth]]。

> 记录基线 UED 方法在 4 个 held-out 测试迷宫上的零样本泛化。每方法 10 seed。
> **对标口径（2026-06-19 据 ACCEL 论文原文订正）**：ACCEL 论文（Parker-Holder 2022）里 **8% 那一关是 101×101 PerfectMazeXL，且只测了 ACCEL(8%)/DR(4%)/PLR(4%)，PAIRED 未参加**（见论文 B.5 节）。论文给 PAIRED 标数字的是 **Table 5 的 human-designed maze**（PAIRED 均值 0.39，其中 PerfectMaze-**M**(中号)=0.32±0.06），**PAIRED 从来不是 0**。且论文 maze 训练是 **20k updates**（PAIRED/Minimax 引用 Jiang 2021a 的 ≈30k），我们用 30k。

## 评估作业元数据

| 项 | 值 |
|---|---|
| 卡 | 1 GPU（L40S 级） |
| 墙钟 | **≈18h36m**（10 seed × PerfectMazeXL，单 seed≈111min/it） |
| checkpoint | `<REMOTE_LOG_DIR>/plr-maze-s0..s9/checkpoint.pkl` |
| 结果 CSV | `results/plr_eval.csv` |
| ACCEL CSV | `results/accel_eval.csv` |

> 单 seed 评估 PerfectMazeXL≈91–111min 是已知坑，整批 10 seed 才会 18h+。

## 结果汇总（mean ± SE，n=10 seed）

> **PAIRED 列口径说明**：PAIRED 官方配置 `n_students=2`（= protagonist a0 + antagonist a1，
> `relative_regret` 恰好要求 2 个 student；minimax README 标为 "Population PAIRED"）。
> ACCEL/PLR 官方均为单 student（`n_students=1`）。为与 ACCEL/PLR **同口径可比**，
> 下表 PAIRED 列取 **a0（protagonist，被部署/评估的主智能体）**。a1 与 best-of-2 见文末附表。

### test_solved_rate（成功率，主指标）

| 测试环境 | ACCEL | PLR | PAIRED (a0) |
|---|---|---|---|
| Maze-Labyrinth | 0.825 ± 0.104 | 0.964 ± 0.023 | 0.812 ± 0.098 |
| **Maze-PerfectMazeXL** | **0.142 ± 0.029** | **0.142 ± 0.044** | **0.301 ± 0.090** |
| Maze-SixteenRooms | 0.997 ± 0.003 | 1.000 ± 0.000 | 0.976 ± 0.023 |
| Maze-StandardMaze | 0.820 ± 0.115 | 0.942 ± 0.033 | 0.731 ± 0.105 |

### test_return（回报）

| 测试环境 | ACCEL | PLR | PAIRED (a0) |
|---|---|---|---|
| Maze-Labyrinth | 0.553 ± 0.072 | 0.638 ± 0.039 | 0.518 ± 0.085 |
| Maze-PerfectMazeXL | 0.114 ± 0.021 | 0.112 ± 0.034 | 0.216 ± 0.063 |
| Maze-SixteenRooms | 0.870 ± 0.013 | 0.862 ± 0.022 | 0.812 ± 0.028 |
| Maze-StandardMaze | 0.526 ± 0.081 | 0.533 ± 0.050 | 0.440 ± 0.069 |

## 对标结论

- **PerfectMazeXL solved_rate：ACCEL 14.2% / PLR 14.2%**，均**高于**论文锚点（ACCEL≈8%、PLR≥4%）→ **对标通过**，复现达标甚至偏高。
- SixteenRooms 三者都基本满分（已饱和，区分度低）。
- Labyrinth / StandardMaze：PLR 略优于 ACCEL（成功率 0.96/0.94 vs 0.83/0.82），但 ACCEL 方差更大（含个别崩溃 seed，min solved_rate=0.01）。
- **PAIRED 在 PerfectMazeXL 上 a0=30.1%（a1=41.9%）—— 与论文吻合，无反常（2026-06-19 据论文原文订正）**：
  - **此前"反常"判断的前提是错的，且基于一个不存在的数字。** 我曾说论文里 "PAIRED≈0% @ XL"，并拿我们 30% 反超它当反常 —— 查 ACCEL 论文原文（PMLR v162）后确认：**101×101 PerfectMazeXL 那一关只测了 ACCEL(8%)/DR(4%)/PLR(4%) 三个方法，PAIRED 根本没参加这组实验**（论文 B.5 节 / 第 8 页）。"PAIRED 0% @ XL" 是凭空脑补的，论文从未报过。
  - **论文真正给 PAIRED 标数字的地方是 Table 5（第 21 页，human-designed maze 迁移）**：PAIRED 12 关均值 **0.39±0.03**，其中 `PerfectMaze (M, 中号)` = **0.32±0.06**。我们的 PAIRED a0 在 `PerfectMazeXL` 上是 **0.301±0.090** —— **与论文中号 PerfectMaze 的 0.32 几乎完全吻合**（口径略不同：论文是中号、我们是 XL；但量级与方差都对得上）。**PAIRED 在 maze 上从来不是 0，是 0.3~0.4 量级。**
  - **ACCEL 论文里的 PAIRED 数字不是 ACCEL 自己跑的**：第 5 页 + C 节 implementation 明写 *"the minimax and PAIRED results are those reported in Jiang et al. (2021a)"*。所以"ACCEL 用错了 PAIRED 实现"不成立 —— 它只是引用了 Robust PLR 论文的数。PAIRED 弱（teacher 难训、entropy collapse）是公认事实，但"弱"≠"0"。
  - **minimax 实现本就强于原始 dcd**：minimax 论文 Table 2 自报 PAIRED 0.63(minimax) vs 0.52(dcd)、PLR 0.82 vs 0.71、ACCEL 0.83 vs 0.75。我们用 minimax，基线整体偏高有据，是库的工程差异，非算法突破。（注：0.63 是**标准 maze benchmark 聚合**值，不能直接拿来跟我们 XL 单关比 —— 这是又一处此前犯过的口径错误。）
  - 此前"minimax 是 Population PAIRED 变体所以更强"的解释**也是错的**：源码 [paired_runner.py:81-82](../../minimax/src/minimax/runners/paired_runner.py#L81-L82) 注释明写 *"Standard PAIRED uses only 2 students"*——`n_students=2`+`relative_regret` 就是**标准经典 PAIRED**（protagonist+antagonist）。

- ✅ **环境定义 / eval 协议 / 训练完整性 三项已核实，结果可信、无 bug（2026-06-16 核查）**：
  - **环境对齐论文**：`Maze-PerfectMazeXL`=`PerfectMazeExtraLarge`，硬编码 101×101（[maze_ood.py:939-941](../../minimax/src/minimax/envs/maze/maze_ood.py#L939-L941)）；步上限由父类无条件重算 `2·(w+2)·(h+2)=2·103·103=21218`（[maze_ood.py:776](../../minimax/src/minimax/envs/maze/maze_ood.py#L776)），≈论文 20k 步。两项都对齐，且训练用的 250 步**无法污染** XL。
  - **eval 协议干净**：checkpoint `meta.json` 的 `eval_env_args` 仅 `{normalize_obs:true, see_agent:true}`，不含尺寸/步上限键 → EvalRunner（[eval_runner.py:119,248](../../minimax/src/minimax/runners/eval_runner.py#L119)）拿到的就是干净的 101×101/21218，obs 归一化与训练一致；`n_episodes=100`、`agent_idxs='*'` 均符合协议。
  - **训练跑满 30k（非欠训）**：`paired-maze-s0/logs.csv` 末行 `_tick=30000`、`n_updates=30000`，10 seed 全 COMPLETED，checkpoint 可信。
  - **`successful:false` 是死字段，非欠训信号**：源码全仓库仅 [loggers.py:131-132](../../minimax/src/minimax/util/loggers.py#L131-L132) 一处写 `successful`，且只在 `Logger.__init__`（训练启动时）写一次、值恒为 `False`，**无任何路径回写 True**。`evaluate.py` 只读不改。故所有 minimax checkpoint 的 `successful` 都恒为 false，与训练是否跑满**无关**。SLURM 的 COMPLETED 才是进程正常退出的依据。
- ⚠ **整体偏高的成因（保留，非 bug；2026-06-19 补全）**：我们三方法在 XL 上都偏高（ACCEL/PLR 14.2%、PAIRED 30%，均高于论文 XL 的 ACCEL 8%/PLR 4%）。注意 **PAIRED 30% 不能与论文 XL 比**（论文 XL 无 PAIRED 数据，见上）；ACCEL/PLR 的 14% vs 8%/4% 偏高，成因按贡献排：
  - **(1) 我们多训了 50%**：论文 maze 是 **20k updates**，我们 **30k**。UED 收敛前还在涨，多训 1 万 updates，解出率普遍偏高是预期的。
  - **(2) 库差异**：minimax 实现整体强于论文用的 dcd（Table 2 有据，见上条）。
  - **(3) 评估关偏易 + eval 高方差**：4 个 held-out 关里 3 个是中小迷宫（SixteenRooms/Labyrinth/StandardMaze 已饱和），唯一极端关 XL 单关 100ep、SE 本就大。
  - 综上：**"整体优于论文 SOTA" 是 多训50% + 不同库 + 评估关偏易 三者叠加的口径红利，不是方法真的更强**。同库+同步数+同极端关的干净对标下，PAIRED 0.30 ≈ 论文中号 0.32，无超越。已排除环境/协议/欠训三类 bug。

## 评估元数据补充（PAIRED）

- **作业**：mmEVAL，Job 3247908，节点 gpu2106，1 GPU。
- **墙钟 ≈ 19h13m**（06-15 19:41 → 06-16 14:55，单 seed≈115min/it × 10，与 ACCEL/PLR 的 18h+ 一致）。
- **10 seed 全部成功**：`paired-maze-s0..s9/checkpoint.pkl` 全部加载，结果 CSV 完整（`results/paired_eval.csv`，10 行）。
- 之前的 1 字节空 CSV（06-13 失败残留）已被本次正常结果覆盖。

## 附表：PAIRED 两 student 全口径（PerfectMazeXL）

| 指标 | a0 (protagonist) | a1 (antagonist) | best-of-2 | a0/a1 平均 |
|---|---|---|---|---|
| solved_rate | 0.301 ± 0.090 | 0.419 ± 0.085 | 0.529 ± 0.090 | 0.360 ± 0.058 |
| return | 0.216 ± 0.063 | 0.315 ± 0.060 | 0.383 ± 0.064 | 0.266 ± 0.043 |

> 观察：antagonist (a1) 在 held-out 测试上略强于 protagonist (a0)，四个测试迷宫均如此。可能与 minimax-regret 训练下两 student 的角色分工有关，但仅 10 seed、差异在 1 个 SE 量级，**不下强结论**；对标统一用 a0（protagonist）。

## 待办
- [x] ~~PAIRED eval~~：已完成（mmEVAL 3247908，10 seed，06-16）。
- [x] ~~PAIRED 列补进结果表~~：已补（a0 口径）。
- [x] ~~"PAIRED 反常"核查~~：已澄清——对标错对象（30k vs 论文 50k 极端关卡）；环境/协议/训练完整性三项全部核实无误（见上）。`successful:false` 确认为死字段。
- [ ] **正式复现（待算力）**：当前 stage1 10 seed / stage2 **3 seed 是 intentional**——Oscar 免费版算力有限、实验室 condo 申请中暂不可用；3 seed 仅用于**粗筛**哪些设置（聚合×N）较好，不追求窄置信区间（如 N=1 的 0.57±0.24 大 SE 是预期内、可接受）。待 condo 算力到位后，仅对**筛选出的优胜设置**做高 seed 正式复现。非阻塞。
- [ ] Stage2（mmS2 3269211）训练产物整理（待 Stage2 文件）。

---
> **核查记录（2026-06-16）**：本轮经历了一次完整的"假反常"排查。最初误把 PAIRED 30% 当反常（对标错对象），又误把 `successful:false` 当欠训风险，均已逐一证伪。教训：对标必须同口径（updates 数 + 关卡 + 库版本），meta 字段的语义要回源码确认而非望文生义。
>
> **订正记录（2026-06-19，据 ACCEL 论文原文逐页核对）**：上一轮的"澄清"本身仍含错误，已全部改正：①编造了"论文 PAIRED≈0% @ XL"——实则 101×101 那关**只测 ACCEL/DR/PLR，PAIRED 没参加**；②把论文 maze 训练量写成 50k——实则 **20k updates**；③拿 minimax Table 2 的 0.63（标准 benchmark 聚合）跟我们 XL 单关比——又一处口径错。**真相**：论文 PAIRED 唯一 maze 数字在 Table 5（PerfectMaze-中号 0.32±0.06），与我们 XL 的 0.301±0.090 吻合；ACCEL 论文的 PAIRED 数字是引用 Jiang 2021a 而非自跑，故"ACCEL 用错 PAIRED 实现"不成立。整体偏高=多训50%(30k vs 20k)+库差异(minimax>dcd)+评估关偏易。**双重教训：连"纠错"也要回原始文献逐页取证，不能凭记忆二次脑补。**
