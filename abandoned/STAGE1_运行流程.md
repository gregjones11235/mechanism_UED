# Stage 1 标准运行流程（minimax / SLURM 集群）

> ⚑ **历史定位（2026-06-20 方案 B 转向后）**：本文档是 **Stage 1 maze 基线（minimax + PerfectMazeXL）的执行手册**，对标 ACCEL 8% 仅为 sanity check。**项目主场已转 jaxnav + SFL repo**（见 [UED实验步骤指导.md](UED实验步骤指导.md) §0.6 + [STAGE4_实验设计.md](STAGE4_实验设计.md)）；本流程作 maze 基线复现的历史参考保留。单 run 墙钟以 [任务运行时间分析.md](任务运行时间分析.md) 的 sacct 实测为准（ACCEL~1h / PAIRED~4.1h）。

> 复现 ACCEL / PLR⊥ / PAIRED 在 minimax AMaze 上的基线，最终在 **Maze-PerfectMazeXL (101×101)** 对标
> 论文靶子：**ACCEL ≈ 8% / PLR⊥ ≈ 4% / PAIRED ≈ 0**（100 episodes 均值，差距 ≤5% 即达标）。
>
> 关联文档：实验全局设计见 `UED实验步骤指导.md`。
>
> 📌 本文用到的占位符（`<CLUSTER_USER>` 等）为集群专有私有值，统一放在仓库根目录 `.env`（未纳入版本库）；
> 训练/评估/部署的具体 SLURM 脚本同样为集群专有，请各自按 `.env` 填好并自行维护。

---

## ⚠ 头号铁律：PerfectMazeXL 绝不进训练循环

**训练脚本的 `--test_env_names` 只能是 3 个小迷宫**（`Maze-SixteenRooms,Maze-Labyrinth,Maze-StandardMaze`），
**绝不能加 `Maze-PerfectMazeXL`**。

原因（2026-06 实测教训，4 天全失败的真因）：
- PerfectMazeXL 是 101×101 巨型迷宫，单次评估 ≈ **91 分钟**。
- 训练 `--test_interval=100`（每 100 步评一次）→ 30000 步要评 300 次 → **≈19 天**，必撞 24h 墙超时(TIMEOUT)。
- 而纯训练只需 **~1 小时**。评估炸弹拖慢约 **300 倍**。

> ⚠ **2026-06-19 注**：此处"~1 小时"是 **ACCEL（plr runner, n_students=1）** 的实测单 run（sacct ≈1:02）。**勿套用到 PAIRED / Stage 2**：PAIRED（修复后）≈ **4.1h/run**，Stage 2 multi_teacher N=1≈3.7h、N=2≈7h、N=4≈14h。各阶段真实墙钟见 [任务运行时间分析.md](任务运行时间分析.md)。

**PerfectMazeXL 的评估放在训练【之后】用 `oscar_minimax_eval.sh` 单独整体评一次。** 这才是正确分工，也是修复手段。

| 脚本 | 何时跑 | 评什么 | 产出 |
|---|---|---|---|
| `oscar_minimax_{accel,plr,paired}.sh`（训练） | 第 ① 步 | 训练期仅评 3 个小迷宫（快） | 10 个 seed 的 checkpoint |
| `oscar_minimax_eval.sh`（评估） | 第 ② 步（训练全完成后） | **PerfectMazeXL 100 episodes** | **对标 8% 的解出率** |

---

## ⚠ 所有 SBATCH 脚本的两条硬规约（每个新脚本必查）

1. **必须有邮件通知**——`#SBATCH --mail-type=END,FAIL` + `#SBATCH --mail-user=<CLUSTER_EMAIL>`。
   缺了不会报错，但作业完成/失败你收不到邮件（2026-06 eval 脚本就漏了这两行，白等）。
2. **墙钟 `-t` 至少写 24:00:00（eval 用 48:00:00）**——宁可申请多、提前结束，绝不要申请少而撞 `TIME LIMIT` 被砍。
   eval 脚本曾只给 `-t 01:00:00`，而 PerfectMazeXL 单 seed ≈ 91 分钟 × 10 seed，1h 内连一个都跑不完 → `CANCELLED DUE TO TIME LIMIT`、空 CSV。

> **`-t` 是「作业能跑多久」的上限，不是排队时长。** `-t` 写 1h 还是 48h 都不影响何时开跑，往大了写（48h）不会让你多等。
> eval 串行 10 seed × ~91 分钟 ≈ 15 小时是「作业自身运行」时长，不是排队；48h 墙钟完全罩得住。
>
> ⚠ **但「array 不必并行 / 几分钟就全开跑」是错的（2026-06-14 sacct 实测推翻）。** 真实并发上限是 **2 个 GPU**：
> 10 个 seed 是**两两成对、每隔约一个"单 run 时长"才开跑下一对**，不是 10 个一起跑。（此处 64 分钟是 ACCEL 单 run；PAIRED≈4.1h、Stage 2 各 N 见 [任务运行时间分析.md](任务运行时间分析.md)，放行间隔随之拉长。）
> 实测 ACCEL array：首对秒开，末对排队 4h21m，整套 submit→全完成 **≈5h24m**（不是 1h）。
> 更糟：**第二套作业要等第一套腾出那 2 张卡**——PLR submit 后排队 **5h09m** 才开跑（ACCEL 还占着卡），整套 ≈10h45m。
> 结论：你的有效配额≈2 GPU，30 个 run 实际是「2 路串行」跑掉约 **16 小时**墙钟。这正是 Stage 2 必须正视的瓶颈。
> 详见 `results/STAGE1_作业并发实测.md`。

---

## 标准两步流程

### 前置（每次开工一次性）
1. 传脚本回 Oscar（在 **WSL** 跑，远程路径 `~` 必须加单引号，否则被本地展开）：
   ```bash
   oscar-put ./oscar_minimax_accel.sh ./oscar_minimax_plr.sh ./oscar_minimax_paired.sh ./oscar_minimax_eval.sh '~/mechanism_UED/'
   ```
2. 确认环境正常（可选，存疑时跑一次性自检作业，几秒出结果）：
   ```bash
   sbatch oscar_gpu_check.sh && cat gpucheck-*.out   # 看末行 RESULT: ✅
   ```

### 第 ① 步：训练
干净重跑前先清旧 checkpoint（`from_last_checkpoint=True` 会从旧断点续，混旧配置会污染结果）：
```bash
# 预览要删的训练目录
ls -d ~/logs/minimax/{accel,plr,paired}-maze-s* 2>/dev/null
# 确认后删除（或 mv 到备份目录）
rm -rf ~/logs/minimax/{accel,plr,paired}-maze-s*
```
提交训练（默认 `--array=0-9` = 10 个 seed）：
```bash
cd ~/mechanism_UED
sbatch oscar_minimax_accel.sh
# 先让 ACCEL 跑顺、确认速度正常后，再交另外两套（避免 30 作业挤爆 GPU 配额 QOSMaxGRESPerUser）
sbatch oscar_minimax_plr.sh
sbatch oscar_minimax_paired.sh
squeue -u <CLUSTER_USER>
```

### ★ 提交后必做：早期验证速度（别盲等！）
提交 ~10 分钟后，第一个 seed 开始写日志，立刻查评估间隔：
```bash
awk -F, 'NR>1{d=$2-p; print $1, "间隔" d "秒"} {p=$2}' ~/logs/minimax/accel-maze-s0/logs.csv | grep -E "^[0-9]+00 " | head
```
- ✅ 整百评估点间隔 **几秒~几十秒** → 评估炸弹已拆，30000 步几小时跑完，放心等。
- ❌ 整百点间隔 **几千秒** → 立刻 `scancel <jobid>`，说明 test_env_names 又混进了 PerfectMazeXL 或别的大评估，别让 10 个 seed 空烧。

### 第 ② 步：评估（训练全部完成后）
```bash
sbatch oscar_minimax_eval.sh accel     # 自动匹配 accel-maze-s* 全部 10 个 seed 汇总
sbatch oscar_minimax_eval.sh plr
sbatch oscar_minimax_eval.sh paired
# 评估很快(几分钟)，也可在 login 节点直接 bash oscar_minimax_eval.sh accel
```
看结果：
```bash
cat ~/logs/minimax/results/accel_eval.csv
```
**看 PerfectMazeXL 那一列均值，对照靶子：ACCEL≈8% / PLR⊥≈4% / PAIRED≈0。** 差距 ≤5% 即 Stage 1 达标。

---

## 三个训练脚本的关键参数（已与官方 accel.json 对齐）

| 参数 | 值 | 说明 |
|---|---|---|
| `-t`（墙钟） | 48:00:00 | 拆评估炸弹后纯训练 ~1h（ACCEL）/ ~4h（PAIRED），48h 是保险 |
| `--mem` | 8G（PAIRED 16G） | 原 32G 仅用 ~9%，降低缩短排队 |
| `--from_last_checkpoint` | True | 支持续跑（三脚本已统一开启） |
| `--n_total_updates` | 30000 | 官方值，勿改 |
| `--n_parallel` | 32 | 官方值 |
| `--n_devices` | 1 | 单卡 |
| `--test_env_names` | 3 小迷宫 | **绝不加 PerfectMazeXL** |
| GPU 自检 | 命令前一行 | JAX 认不出 GPU 则秒退，不空烧 48h |

---

## 排障：作业失败时的标准诊断顺序

别靠瞬时 `nvidia-smi` 或 `sacct` 的 CPU 时间猜（都不能定性）。按权威数据走：

1. **看是不是超时**：`sacct -j <jobid> --format=JobID,State,Elapsed,TotalCPU,ExitCode` → `State=TIMEOUT` 即撞墙。
2. **看 GPU 真用没用上**：`jobstats <jobid>` → 看 **GPU utilization**。高(80%+)=在训；≈0=没绑上 GPU。
3. **看时间花哪了（定位评估炸弹）**：
   ```bash
   awk -F, 'NR>1{d=$2-p; print $1, "间隔" d "秒"} {p=$2}' ~/logs/minimax/<xpid>/logs.csv | grep -E "^[0-9]+00 "
   ```
   整百 update 点间隔远大于非整百点 → 周期评估在拖慢，检查 `--test_env_names` 是否混入大迷宫。
4. **看环境本身**：`sbatch oscar_gpu_check.sh`，确认 JAX=CudaDevice、jax/jaxlib/plugin 全 0.5.3 配套。
