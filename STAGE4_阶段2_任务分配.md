# Stage 4 · 阶段 2（真实规模 + jaxnav SOTA 同表对标）任务分配 —— Alec & Henry 四卡并行

> **目标**：跑完 [STAGE4_实验设计.md](STAGE4_实验设计.md) §2「真实规模 + jaxnav SOTA 同表对标」。
> 阶段 1 已 **GO**（方差复活铁证 + auction 随 λ 调尖锐度，见 [results/STAGE4_阶段1_结果分析.md](results/STAGE4_阶段1_结果分析.md)）。
> **人力/算力**：Alec 2× A6000 + Henry 2× A6000 = **4 卡并行**。
> **预计墙钟**：方案 B 候选档训练 ≈ **12 h**（30 run，4 卡并行，见 [任务运行时间分析.md](任务运行时间分析.md) §7.3）；+评测/制表 ≈ 半天 → **端到端约 1 天**。
>
> **生成日期**：2026-06-23。
> **与阶段 1 的差异**：seed 7→**10**（对齐 baseline rliable 口径）、TOTAL_TIMESTEPS 1e8→**3e8**（15→45 eval epoch 跑满）、
> 评测口径升级到 **CVaR + hand-designed test set + 100-map**、新增**外部 baseline 同表对标**。

---

## 0. 实验是什么（一句话）

阶段 1 验到"机制有用"（方差复活）；阶段 2 验"机制**有效**"（student 因课程学得更好）+ 出 **SOTA 成绩单**：
1. 把阶段 1 选出的 **top λ（=1.0 甜区 + inf exploit 端）** × **10 seed** 真实规模（3e8）跑满；
2. 在 **CVaR worst-case + hand-designed test set + 随机 100-map** 口径下，与外部 baseline（DR/PLR/ACCEL/SFL/PAIRED/NCC）进**同一张 rliable 表**。

**阶段 1 的头号待答问题**：注入档 eval win rate 在短跑下倒挂（黄灯）。阶段 2 跑满 3e8 看是否反超——这是"机制有效"的判据。

---

## 1. 共同前置（两人开跑前都要确认）

### 1.1 ⚠ 代码已更新（阶段 1 暴露的两个 bug 已修，2026-06-23）

阶段 1 暴露并已修复两处，**两人必须用更新后的镜像**（`mechanism_UED/_sfl_repo_mirror/`，按 §1.2 覆盖进 `sfl/train/`）：

1. **none 档 score 归一化**（`pcgrl_generator.py:get_generator_set`）：fallback（λ=none）路径原直接 concat 各 gen 原始量级 score
   （致 learnability_set_var~1090、与 auction 档不可比）；现各 gen 先 per-gen z-score 再 concat（与 auction 路径同口径）。
   **影响阶段 2 的 none 消融**：修后 none 与 1.0/inf 同量纲、可横比。
2. **probe/p_std 解耦 SAVE_PATH（健壮性修复）+ ⚠ 真根因是开关没开**（`train_probe.py` + `jaxnav_sfl.py`）：
   - **阶段 1 注入档缺 p_std 的真根因**：核查 config 发现注入档命令显式 `PROBE_ORTHOGONALITY=false`（基线是 true），整条 probe 不跑 → 无 p_std。**不是 SAVE_PATH 的锅**（SAVE_PATH 两者都默认非空 `checkpoints/multi_robot_ued`）。
   - **阶段 2 的 action（关键，写进 §1.3 命令）**：注入档必须 **`PROBE_ORTHOGONALITY=true`** 才有 p_std。
   - **附带已修**：`log_orthogonality_step` 支持 out_dir=None（只算 wandb 指标不写 trace）+ probe 收集/落盘解耦 SAVE_PATH，让"开 probe 但没设 SAVE_PATH"也能出 p_std。是健壮性兜底，非根因。

> 已 `py_compile` 通过 + Oscar sfl 环境验过 z-score 数值正确（各 gen 均值≈0/std≈1、-inf 占位保留）。
> **代码已同步 Oscar**（2026-06-23，直接 scp 覆盖 `~/sampling-for-learnability/sfl/train/`，旧版备份为 `*.bak_pre_p2fix`，三文件 py_compile 通过）。
> 若换机器：`cp ../mechanism_UED/_sfl_repo_mirror/*.py sfl/train/`（先 `git pull` 拿最新）。

### 1.2 环境（同阶段 1，无变化）

jax 0.4.38 血统、SFL 本体 + 覆盖镜像、CENIE 隐藏依赖（scikit-learn/joblib/threadpoolctl）。
搭建步骤完全照 [STAGE4_阶段1_任务分配.md](STAGE4_阶段1_任务分配.md) §1.2 Step 0~3，此处不重复。运行前自检（GPU 可见 + Stage4 模块 import）同 §1.3。

### 1.3 单个 run 的命令模板（阶段 2 规模：3e8）

```bash
python sfl/train/jaxnav_sfl.py \
    SEED=<seed 0..9> \
    GROUP_NAME=<见 §2 表分组名> \
    WANDB_MODE=online \
    GEN_ESTIMATOR_IDS="[difficulty,pvl,cenie]" \
    AUCTION_USE_CENIE=true \
    PROBE_ORTHOGONALITY=true \
    learning.TOTAL_TIMESTEPS=300000000 \
    <注入档参数：见 §2 表>
```

- **阶段 2 三处关键变更（相对阶段 1）**：
  1. `learning.TOTAL_TIMESTEPS=300000000`（3e8，=45 eval epoch 跑满，**SOTA 对标必须跑满**，不得截断）；
  2. `SEED=0..9`（**10 seed**，对齐 baseline rliable IQM+95%CI 口径）；
  3. `PROBE_ORTHOGONALITY=true`（**注入档必须开**——阶段 1 注入档显式关了它致无 p_std；开了才能判注入档饱和。基线本就默认 true）。
- **SAVE_PATH 无需显式设**：config 默认 `checkpoints/multi_robot_ued` 非空，checkpoint 照常落盘（**阶段 1 实测 35/35 run 都有 final + 4 个 step checkpoint**，见 §4.1）。若想分目录可显式设。
- 其余 config（NUM_ENVS=256 / ROLLOUT_STEPS=1000 / EVAL_FREQ=50 等）**一律默认不 override**——保证两人可比、且与阶段 1 同规模。
- ⚠ **probe 有额外 rollout 开销**：阶段 1 注入档关 probe 是为省这部分。开 PROBE_ORTHOGONALITY=true 后单 run 墙钟会略增（probe 每 eval epoch 多一次 rollout），§7 推算的 ~1.1 h 是关 probe 的值，开 probe 后保守按 ~1.3 h/run 估，整套墙钟相应上调（已在 §7.2 ×1.3 余量内）。

### 1.4 看门狗（同阶段 1，阈值不变）

- 第一个 eval epoch 给足 20 min（含编译）；第 2 个起单 epoch >10 min 无新日志 → 判死锁，kill 重跑。
- 阶段 2 单 run ~1.1~1.45 h（45 epoch），比阶段 1 长 3×，**总墙钟长但单 epoch 速度不变**，看门狗按 epoch 判不变。
- ⚠ 阶段 2 长训下 difficulty 激励的病态图率可能上升（见记忆 `stage4-stall2`），已加 `MAX_PLACE_TRIES=64` 上界 + 注入端过滤，理论不再卡死；若仍 GPU 97% 满频且 mtime 死，按阶段 1 §1.4 第 2 种卡死处置。
- **SBATCH 硬规约**：所有脚本带邮件通知（mail-type/mail-user）+ 墙钟 ≥24h（见记忆 `sbatch-script-mandatory-rules`）。

---

## 2. 实验矩阵（方案 B 候选档：3 训练档 × 10 seed = 30 run）

| 档位 | 注入档参数 | GROUP_NAME | seeds | run 数 | 作用 |
|---|---|---|---|---|---|
| **基线** | `GENERATOR_INJECTION=false` | `s4p2-base` | 0–9 | 10 | SFL 随机海选，内部对照（win rate 反超判据锚点） |
| **甜区 (λ=1.0)** | `GENERATOR_INJECTION=true GEN_AUCTION_LAMBDA=1.0` | `s4p2-lambda-1_0` | 0–9 | 10 | 软混合，阶段 1 注入档 win rate 最高，**主候选** |
| **exploit (λ=inf)** | `GENERATOR_INJECTION=true GEN_AUCTION_LAMBDA=inf` | `s4p2-lambda-inf` | 0–9 | 10 | single-winner，var 上限 + 验 mode-collapse |
| **小计** | | | | **30** | |

- **λ=5.0（explore 端）不进阶段 2 训练**：阶段 1 实测它 var 最小、win rate 最低（0.305），无优势，仅作事后消融的 explore 端点保留（阶段 1 数据已够）。
- **none（去 auction 消融）是否进阶段 2**：见 §2.1 决策。

### 2.1 ⚠ 待用户确认的两个范围问题

1. **none 消融档（+10 run，→40 run，+3 h 墙钟）**：设计文档 §2.2 把"去 auction（λ=none 各自为政）"列为**内生消融**（证 auction 混合的增益）。
   修复后 none 已与其他档同口径可比。**建议纳入**（多 3 h 换一条关键消融），但若算力紧可砍。**默认按纳入（40 run）分配，砍则去掉 none 那 5+5。**
2. **外部 baseline（DR/PLR/ACCEL/SFL/PAIRED/NCC）是否需重训**：设计文档 §2.1 称"实现都在 SFL repo、已 Oscar 部署，不用重训"。
   **但"已部署"≠"已有 10 seed 跑满结果"**——若 wandb 上没有这些 baseline 的 10-seed run，阶段 2 还要补训它们（每个 baseline 同 ~1 h/run × 10 seed，6 个 baseline = 60 run ≈ +15 h）。
   **这是阶段 2 最大的不确定算力项，开跑前必须核实**（见 §5 第 0 步）。本分配先只管方案 B 候选档；baseline 重训需求确认后另行追加。

---

## 3. 人员划分（按对照完整性切，基线最优先）

> 划分原则同阶段 1：每人手上的档能自成对照、基线最优先（win rate 反超的判据锚点）。
> Alec 拿 **基线 + 甜区（主候选）**；Henry 拿 **exploit 端 + none 消融**。

### 3.1 Alec（2 卡）— 20 run · 骨架档（基线 + 主候选）

| 档位 | GROUP_NAME | seeds | run 数 | 优先级 |
|---|---|---|---|---|
| **基线**（最高优先，先跑） | `s4p2-base` | 0–9 | 10 | ① 最先（反超判据锚点） |
| 甜区 (λ=1.0) | `s4p2-lambda-1_0` | 0–9 | 10 | ② 主候选 |
| **小计** | | | **10 波×2卡=5 波** | |

### 3.2 Henry（2 卡）— 20 run · exploit + 消融

| 档位 | GROUP_NAME | seeds | run 数 |
|---|---|---|---|
| exploit (λ=inf) | `s4p2-lambda-inf` | 0–9 | 10 |
| none 消融 (λ=none) | `s4p2-lambda-none` | 0–9 | 10 |
| **小计** | | | **20** |

> 若 §2.1 决定**不跑 none**：Henry 只跑 exploit 10 run，Alec/Henry 共 30 run；此时可把甜区/exploit 的 seed 在两人间再均摊加速。
> **接替预案**：同阶段 1 §4——配置纯由 (GROUP_NAME, SEED, λ档) 决定，无跨 run 依赖；Henry 失败时 Alec 照 wandb 缺口补齐。

### 3.3 每人 2 卡内部排法（同阶段 1）

- 每卡一次 1 run（jaxnav N=3 单 run 占满一卡）。2 卡 = 2 run 并发。
- 一档 10 seed = ⌈10/2⌉ = 5 波，按 `seed 0,1 → 2,3 → … → 8,9` 放行。
- `WANDB_MODE=online` + 上表 GROUP_NAME，wandb 自动按档分组。
- ⚠ 阶段 2 单 run ~1~1.45 h（vs 阶段 1 ~0.4 h），Alec 20 run = 10 波 × ~1.3 h ≈ **13 h**（含波次取整）；Henry 同理。

### 3.4 ✅ checkpoint 保存已验证（不是问题）

> 曾担心阶段 1 注入档没设 SAVE_PATH 致 checkpoint 不存（会让阶段 2 评测无 checkpoint 可用，致命）。**核查后是虚惊**：
> `SAVE_PATH` 在 yaml config 有默认值 `checkpoints/multi_robot_ued`（非 None），命令不传也照常落盘。

- **阶段 1 实测**：35/35 run 全部有 `model.safetensors`（final）+ 4 个 `model_step*.safetensors`（中间 checkpoint），在 `~/sampling-for-learnability/checkpoints/multi_robot_ued/<run.name>/`。
- **阶段 2 不会有问题**：同一默认 SAVE_PATH，checkpoint 逻辑（`jaxnav_sfl.py:1126-1142`，按 `checkpoint_steps` 存多份 + final）正常工作。评测脚本直接读这些 final checkpoint 跑 CVaR/100-map。
- ⚠ **唯一注意**：阶段 1 的 step checkpoint 只有 4 个（NUM_CHECKPOINTS 默认值在 15 epoch 下取了 ~4 个点）；阶段 2 是 45 epoch，若要画细粒度效率曲线（见 §5），可适当调大 `NUM_CHECKPOINTS` 多存几个中间点，或直接用 wandb 的逐 epoch eval 指标（无需 checkpoint）。

---

## 4. 跑完看什么（产出 SOTA 成绩单）

对照 STAGE4 §2，阶段 2 的判据**升级到 SOTA 口径**（不再是阶段 1 的内部趋势）：

1. **机制有效（头号判据）**：注入档（1.0/inf）的 **eval win rate 是否反超 / 追平基线**（阶段 1 短跑倒挂在 3e8 下是否兑现）。
   - 看 **sampled 100-map 口径**（阶段 1 已差距仅 ~9pct）和 **CVaR worst-case**，**不**用 singleton overall_win_rate 作终判。
2. **SOTA 同表对标**：方案 B 候选 + 外部 baseline 进同一张 **rliable 表（IQM + 95%CI）**，三口径（CVaR / hand-designed test set / 随机 100-map）。
3. **事后消融**（内生于矩阵，不额外加 run）：
   - 去 auction（λ=none vs 1.0/inf）→ 证 auction 混合增益（**已修归一化，可横比**）；
   - single-winner（λ=inf）vs fractional（λ=1.0）→ 证 λ 权衡必要性 + 验 inf 是否 mode-collapse；
   - 注入档 `probe/p_std`（**已修可拿到**）→ 看注入课程是否饱和（双峰口径 `p_std<0.05` 才算）。

**诚实标注（设计文档 §2.3 两个红灯，论文须写）**：
- 障碍 2（在线非平稳）是真未解研究问题，用 target student 周期重评压住（**阶段 2 须加，不能省**）；
- 放弃 regret 换 learnability 对 generator 注入层无收敛背书（同 ACCEL/PLR），论文诚实写。

---

## 5. ⭐ 为什么"同轮数 + 简单标准关测"不能验"效率更高"（评测设计的核心认知）

> **背景**：阶段 1 出现 win rate 倒挂后，一个自然的质疑是——"注入档和基线训练轮数一样，
> 如果我的方法效率更高，那即使现在拿到相同标准关卡测，也该 注入档 > 基线 才对。"
> **这个推理藏着一个前提错误，会直接误导阶段 2 的评测设计。这里讲清楚，免得阶段 2 用错标尺白跑。**

### 5.1 错在哪：baseline 和注入档练的不是同一件事

"效率更高 = 同轮数下能力更强" 只在**两种方法练同一件事、只是我学得快**时成立。但 UED 里两者练的根本不是一回事：

| | 基线（随机海选） | 注入档（定向 generator） |
|---|---|---|
| 喂给 student 的课程 | 低区分度、偏简单（var≈0，选不出难关） | 系统性更难（lset_mean 0.21→1.4，定向造难关） |
| student 在练什么 | 大量简单关，**很快刷满简单能力** | 难关，**学得慢，但学的是更难的技能** |
| 同样 15 epoch 后 | 简单能力已饱和 | 难技能还在半路 |

类比：A 刷 300 道简单题、B 刷 100 道难题（同样时间）。考一张**简单卷**——A 暂时高分。
这不证明 A 能力强，只证明这张卷偏向 A 练的东西。

### 5.2 关键：阶段 1 的 "标准关" 恰恰偏向基线

阶段 1 的 hand-designed test set（SingleNav/Blank/Middle）是**相对简单**的关卡。在这个标尺上：
- 基线一直练类似难度 → 同分布、直接高分；
- 注入档把预算花在**更难**的关卡 → 在简单 eval 上"牛刀还没磨好" → 暂时低分。

**这不是效率低，是训练分布 ≠ 测试分布 + 难任务需更多步收敛。** 阶段 1 数据已自证这一点：
> 关卡越接近"随机分布"，差距越小——sampled 随机 100-map 上注入档 0.88 vs 基线 0.97（差 ~9pct），
> 而简单 hand-designed 关上是 0.41 vs 0.70（差 ~29pct）。**说明注入档的劣势集中在"简单 in-distribution 关"，正是它没主练的那端。**

### 5.3 因此："放简单标准关测"是对注入档最不利的测法

- 注入档的优势**定义上**出现在 **难关 / worst-case / 泛化分布**，不是简单 in-distribution 关；
- 在简单标准关上，即使跑满 3e8，注入档也可能只是**持平**基线（简单关基线练得多）——**赢面在难端**；
- 所以"同标准简单关注入档该赢"这个期待**本身就不该成立**，用它判机制失败是误判。

### 5.4 阶段 2 必须用的正确标尺（直接落到评测设计）

1. **标尺要够难/够全**：在**难关卡 + CVaR worst-case + 随机 100-map**上测（设计文档 §2.1 口径）。
   难关上基线暴露短板、注入档优势才显现。**绝不能只用简单 hand-designed test set 的 overall_win_rate 作终判。**
2. **看效率曲线，不是单点**："效率更高"的正确定义 = **注入档在 step X < 750(满) 就达到基线 final 的（难关）能力**。
   阶段 2 要画 **win-rate（难关/CVaR）vs training step** 曲线对比，而非固定终点测一次。逐 epoch eval 指标 wandb 已落，无需额外 checkpoint。
3. **效率指标具体定义**（写进阶段 2 产出）：对每条曲线，记 `step_to_reach(baseline_final_level)`——
   注入档达到"基线 3e8 终点难关 win rate"所需的 step 数。注入档该值 < 基线（=750/满）即"效率更高"；
   若注入档 final 难关 win rate 直接 > 基线 final，则是"能力更强"。两者任一成立即 UED 意义上的优质方法。

> **一句话**：阶段 2 验"更强/更高效"必须用 **难关 + CVaR + 效率曲线**，不能用简单标准关单点测——
> 后者对注入档系统性不利，会把"预算花在难处、暂未在易处兑现"误判成"机制无效"。

---

## 6. 开跑前的 0 步骤（必做，定算力）

> 阶段 2 算力的最大变数是外部 baseline 是否要重训（§2.1 第 2 点）。**两人开跑方案 B 候选档前，先派一人花 10 min 核实**：

1. **核实 baseline 现状**：上 wandb（`gregjones11235-brown-university/multi_robot_ued`）数 DR/PLR(MaxMC)/PLR(PVL)/PLR-Robust/ACCEL(MaxMC)/ACCEL-Robust/SFL/PAIRED/NCC 各有几个 **3e8 跑满 + 10 seed** 的 finished run。
   - 若齐 → 阶段 2 只跑方案 B 候选档（30~40 run，~12~15 h），直接进同表。
   - 若缺 → 列缺口，追加 baseline 重训（每个 ~1 h/run × 10 seed），算力另算（6~9 baseline × 10 seed ≈ +15~22 h，可两人接着方案 B 档后排）。
2. **确认 none 消融纳不纳入**（§2.1 第 1 点）：纳入则 40 run、不纳入则 30 run。**默认纳入。**
3. 确认评测脚本（CVaR + test set + 100-map）就绪：阶段 2 每个 final checkpoint 要跑评测（轻量，分钟级），见 [STAGE4_实验设计.md](STAGE4_实验设计.md) §2.1 口径。

---

## 7. 与阶段 1 的对照速查

| 维度 | 阶段 1（go/no-go） | 阶段 2（SOTA） |
|---|---|---|
| TOTAL_TIMESTEPS | 1e8（15 epoch） | **3e8（45 epoch）** |
| seed | 7（0–6） | **10（0–9）** |
| 训练档 | 5 档（base+5.0+1.0+inf+none） | **3 档（base+1.0+inf）** [+none 消融可选] |
| SAVE_PATH | 不设 | **必设**（评测要 checkpoint） |
| 评测口径 | 内部 learnability var + win rate 趋势 | **CVaR + test set + 100-map + rliable 表** |
| 对标 | 内部随机海选对照 | **外部 baseline 同表** |
| 单 run 墙钟 | ~0.4 h | ~1.1~1.45 h |
| 整套墙钟（4 卡） | ~数 h | **~12 h（30run）/ ~15 h（40run）** + 评测半天 |
| 判据 | 方差复活 + λ 结构 → go ✅（已达） | win rate 反超 + SOTA 表 |
