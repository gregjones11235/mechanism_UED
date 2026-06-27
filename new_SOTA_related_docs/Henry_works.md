# Henry 运行指导 —— 补 seed 三档 × 10 seed（4 GPU 并行）

> **目标**：给 new SOTA 主张补干净的多 seed 横比。当前最佳 run（B1 adaptive-no-gate）只有**单 seed** 超 SFL +0.13，
> 必须补 ≥10 seed 才能坐实（吸取教训：baseline seed 间 CVaR 能差 1.27↔2.75，单 seed 会翻盘）。
>
> **你的环境**：4× GPU 并行，**不用 Oscar / 不是布朗账号**——所以本文是**自包含**的，不依赖任何 Oscar 路径或 slurm。
> **生成日期**：2026-06-27。本文写法参考 [STAGE4_阶段2_任务分配.md](../STAGE4_阶段2_任务分配.md)，但**为你换机器场景重写了环境部分**。

---

## 0. 你要跑什么（一句话 + 矩阵）

三个训练档 × 10 seed = **30 run**，4 GPU 并行（每卡一次 1 run，jaxnav 单 run 占满一卡）。

| 档名 | 是什么 | run 数 |
|---|---|---|
| **baseline_thesis_5000/100** | SFL 本体随机海选（=对标 baseline），池 5000 / 保留 top-100 | 10 |
| **adaptive_no_gate_3000/60** | 我们的方法（当前最佳 B1 配置）：generator 注入 + 自适应课程，池 3000 / 保留 60 | 10 |
| **adaptive_no_gate_5000/100** | 同上但放大到 池 5000 / 保留 100（与 baseline 同基数，验"放大是否再涨"） | 10 |
| **小计** | | **30** |

> **为什么是这三档**：① baseline 是对标锚点（必须同口径自己跑，不能借别人的数）；
> ② 3000/60 复现当前最佳、把单 seed 补成 10 seed；③ 5000/100 是"放大到与 baseline 同基数"的关键对照——
> 当前 3000/60 是"用更小池子超 baseline"，5000/100 验证同基数下能否拉开更大差距。

⏱ **墙钟估算**：单 run 3e8（45 eval epoch）≈ **1.1~1.5 h**（开了 probe，保守按 1.5 h）。
30 run / 4 卡 = ⌈30/4⌉ = **8 波 × 1.5 h ≈ 12 h**。一天内能跑完。

---

## 1. 环境搭建（换机器，从零，照做即可）

> jax **必须锁 0.4.38 血统**（SFL 本体在 0.4.38 上验过；新版 jax 的 API 变动会让 jaxmarl/jaxued 崩）。CENIE 有隐藏依赖。

### Step 0 — 取 SFL 本体（提供 jaxnav 环境 + `sfl/` 包结构）
```bash
git clone https://github.com/amacrutherford/sampling-for-learnability.git
cd sampling-for-learnability
```

### Step 1 — 覆盖我们打过补丁的训练代码（关键，别用本体旧版）

我们的改动（generator 注入 + auction + 自适应课程 + 一堆 bug 修复）都在 `mechanism_UED/_sfl_repo_mirror/`。
先 `git pull` 我们的 `mechanism_UED` 仓库拿最新，然后覆盖进本体的 `sfl/train/`：

```bash
# 假设 mechanism_UED 与 sampling-for-learnability 是同级目录
cp    ../mechanism_UED/_sfl_repo_mirror/*.py   sfl/train/   # 改过的 .py（jaxnav_sfl/pcgrl_generator/...）
cp -r ../mechanism_UED/_sfl_repo_mirror/envs   sfl/train/   # ⚠⚠ generator 注入核心依赖
```

> ⚠⚠ **`envs/` 绝不能漏**：`pcgrl_generator.py` 有 5 处 `from envs.pcgrl_env import ...`、`models_pcgrl.py:13` 顶层也 import。
> 漏了一开跑就 `ModuleNotFoundError: No module named 'envs'`。三个档里的两个 adaptive 档都是注入档，必炸。
> `envs/` 是 pcgrl-jax 裁剪版（env.py / pcgrl_env.py / pathfinding.py / probs/ / reps/ ...），本地 mirror 与我们能跑的版本逐文件一致。

### Step 2 — Python 环境 + 锁版本
```bash
conda create -n sfl python=3.10 -y && conda activate sfl
pip install -r requirements.txt          # SFL 自带：jaxmarl / jaxued / hydra / pandas / wandb

# 锁 jax 0.4.38 血统（务必这一组版本，别让 pip 自动升级）
pip install "jax==0.4.38" "jaxlib==0.4.38" "flax==0.10.2" "optax==0.2.5" "chex==0.1.90" \
            "orbax-checkpoint==0.6.4"

# ⚠ GPU：上面是 CPU jaxlib。换成你的 CUDA 版本对应的 GPU wheel，例如：
# pip install "jax[cuda12]==0.4.38" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
#   （按你机器的 CUDA driver 选 cuda11/cuda12；务必仍是 0.4.38）

# CENIE 隐藏依赖（auction 第二信号 cenie 要用，不装一开跑就 ImportError）
pip install scikit-learn joblib threadpoolctl

# 其它常见缺失
pip install "gymnax==0.0.9" "Pillow>=10"
```

### Step 3 — 开跑前自检（30 秒，必做）
```bash
conda activate sfl
# (a) GPU 可见，且应看到你 4 张卡
python -c "import jax; print('JAX devices:', jax.devices())"
# (b) jax 版本对
python -c "import jax, jaxlib; print('jax', jax.__version__, '| jaxlib', jaxlib.__version__)"   # 应 0.4.38 / 0.4.38
# (c) generator 注入依赖 import 通（不通=envs/ 没 cp 对）
cd sampling-for-learnability
python -c "from envs.pcgrl_env import PCGRLEnv; print('envs OK')"
python -c "import sklearn, joblib, threadpoolctl; print('cenie deps OK')"
```
三条全过再开跑。任何一条挂了先修环境，别硬跑。

### Step 4 — wandb 登录（结果要进同一个 project 才能横比）
```bash
wandb login           # 用 gregjones11235 的 wandb API key（向 Alec/你队友要）
# 结果会进 project: gregjones11235-brown-university/multi_robot_ued
```

---

## 2. 三档的精确命令（直接抄，只改 SEED）

> 所有 run 都跑满 **3e8**（45 eval epoch）。**不得截断**——SOTA 对标必须跑满。
> 公共参数（三档都带）：`PROBE_ORTHOGONALITY=true`（注入档要它才有 p_std 仪表盘）、`WANDB_MODE=online`、`learning.TOTAL_TIMESTEPS=300000000`。

### 档 A — baseline_thesis_5000/100（SFL 海选，对标锚点）

> baseline 池 = `BATCH_SIZE(1000) × NUM_BATCHES(5) = 5000`、`NUM_TO_SAVE=100`，**这就是 yaml 默认值**，所以**一个池参数都不用传**，只要关掉 generator 注入。

```bash
python sfl/train/jaxnav_sfl.py \
    SEED=<0..9> \
    GROUP_NAME=henry-baseline-5000-100 \
    GENERATOR_INJECTION=false \
    PROBE_ORTHOGONALITY=true \
    WANDB_MODE=online \
    learning.TOTAL_TIMESTEPS=300000000
```
（无 generator、无课程、无 auction——纯 SFL learnability top-100 海选。）

### 档 B — adaptive_no_gate_3000/60（当前最佳 B1，复现+补 seed）

> 池 3000 = `GEN_POOL_PER_GEN(1500) × 2 个 estimator`。保留 `GEN_NUM_TO_SAVE=60`。无 gate（signal_mode=anchored，不带 gate）。

```bash
python sfl/train/jaxnav_sfl.py \
    SEED=<0..9> \
    GROUP_NAME=henry-adaptcurr-nogate-3000-60 \
    GENERATOR_INJECTION=true \
    GEN_ESTIMATOR_IDS="[anchored,cenie]" \
    GEN_POOL_PER_GEN=1500 \
    GEN_NUM_TO_SAVE=60 \
    GEN_ROLLOUT_CHUNK=250 \
    AUCTION_USE_CENIE=true \
    AUCTION_SIGNAL_MODE=anchored \
    +CURRICULUM_ARM=geo \
    +CURRICULUM_ADAPTIVE=true \
    +CURRICULUM_ALLOW_DEMOTE=true \
    +CURRICULUM_P_THRESHOLD=0.6 \
    +CURRICULUM_BETA=1.0 \
    +CURRICULUM_THR_EASY=5 \
    +CURRICULUM_THR_MID=10 \
    +CURRICULUM_THR_HARD=22 \
    PROBE_ORTHOGONALITY=true \
    WANDB_MODE=online \
    learning.TOTAL_TIMESTEPS=300000000
```
> **「无 gate」由 `AUCTION_SIGNAL_MODE=anchored` 决定**（有 gate 的那版是 `anchored_cenie_gate`，我们已证它是毒药、不跑）。
> 不要传 `GEN_LEARNGATE`——那是另一条 `euc` 信号路径的门，对 `anchored` 模式无效，B1 也没传它。

### 档 C — adaptive_no_gate_5000/100（放大到与 baseline 同基数）

> 与档 B **只差两个数**：`GEN_POOL_PER_GEN=2500`（×2=5000）、`GEN_NUM_TO_SAVE=100`。其余完全一致。

```bash
python sfl/train/jaxnav_sfl.py \
    SEED=<0..9> \
    GROUP_NAME=henry-adaptcurr-nogate-5000-100 \
    GENERATOR_INJECTION=true \
    GEN_ESTIMATOR_IDS="[anchored,cenie]" \
    GEN_POOL_PER_GEN=2500 \
    GEN_NUM_TO_SAVE=100 \
    GEN_ROLLOUT_CHUNK=250 \
    AUCTION_USE_CENIE=true \
    AUCTION_SIGNAL_MODE=anchored \
    +CURRICULUM_ARM=geo \
    +CURRICULUM_ADAPTIVE=true \
    +CURRICULUM_ALLOW_DEMOTE=true \
    +CURRICULUM_P_THRESHOLD=0.6 \
    +CURRICULUM_BETA=1.0 \
    +CURRICULUM_THR_EASY=5 \
    +CURRICULUM_THR_MID=10 \
    +CURRICULUM_THR_HARD=22 \
    PROBE_ORTHOGONALITY=true \
    WANDB_MODE=online \
    learning.TOTAL_TIMESTEPS=300000000
```

### ⚠ 三个最容易踩的坑（务必看）

1. **`+CURRICULUM_*` 前面那个 `+` 不能掉**：这 8 个 key 是 yaml 里没声明的新 key，Hydra struct 模式下不加 `+` 会**秒崩**（`Could not override 'CURRICULUM_ARM'... use +`）。baseline 档（档 A）**没有**课程，所以不带这些。
2. **`GEN_ESTIMATOR_IDS="[anchored,cenie]"` 的引号和方括号**：列表要整个用引号包住，逗号后别加空格（`[anchored,cenie]` 不是 `[anchored, cenie]`）。
3. **池基数 = `GEN_POOL_PER_GEN × estimator 数`**：这里 2 个 estimator，所以 1500→3000、2500→5000。别把 `GEN_POOL_PER_GEN` 直接写成 3000/5000（那会变成 6000/10000，可能 OOM）。

---

## 3. 4 卡怎么排（10 seed/档）

- 每卡一次 1 run（jaxnav 单 run 占满一卡显存）。4 卡 = 4 run 并发。
- 用 `CUDA_VISIBLE_DEVICES` 把 run 钉到指定卡，例如同时起 4 个：
  ```bash
  CUDA_VISIBLE_DEVICES=0 python sfl/train/jaxnav_sfl.py SEED=0 ... &
  CUDA_VISIBLE_DEVICES=1 python sfl/train/jaxnav_sfl.py SEED=1 ... &
  CUDA_VISIBLE_DEVICES=2 python sfl/train/jaxnav_sfl.py SEED=2 ... &
  CUDA_VISIBLE_DEVICES=3 python sfl/train/jaxnav_sfl.py SEED=3 ... &
  wait
  ```
- 跑法建议：**先把档 A（baseline）10 seed 跑完**（它是对标锚点，最优先），再跑档 B、档 C。
  或三档交错，只要 `GROUP_NAME` 对、wandb 会自动按组聚合。
- run 之间**无依赖**，挂了哪个 seed 就单独补哪个 seed（配置纯由 GROUP_NAME+SEED 决定）。

---

## 4. 看门狗（判死锁，照阈值）

- **第一个 eval epoch 给足 20 min**（含 JAX 编译，正常就是慢）。第 2 个起单 epoch >10 min 无新日志 → 判死锁，kill 重跑该 seed。
- 正常单 run 全程 ~1.1~1.5 h、45 个 eval epoch。日志里每 50 step 一个 eval 打印。
- 若某 seed GPU 占着但 util 0%（或满频但 mtime 不动）→ 卡死，kill 重跑。注入档历史上有过几种卡死，但相关 bug 都已修（BFS 替换碎图检测、place 上界、io_callback 剥离），现在理论不再卡；真遇到就 kill 重跑，单 seed 偶发不影响整体。

---

## 5. 跑完给什么（交回的产出）

跑完**不用你算 CVaR**，只要确认：

1. **wandb 上三个 group 各 10 个 finished run**（`henry-baseline-5000-100` / `henry-adaptcurr-nogate-3000-60` / `henry-adaptcurr-nogate-5000-100`），每个都跑到 `_step≈2250`（45 epoch 满）。
2. **checkpoint 落盘了**：每个 run 在 `checkpoints/multi_robot_ued/<run_name>/` 下应有 `model.safetensors`（final）+ 几个 `model_step*.safetensors`。这些后面 Alec 跑官方 deploy 评测要用。
3. 把三个 group 名 + 哪些 seed 成功/失败回报给 Alec，缺口由我们补。

> **判据预告**（你不用管，给你个数感）：当前单 seed B1 的 CVaR10≈2.447、SFL≈2.319。
> 补完 10 seed 后看的是**均值是否仍 adaptive > baseline**、以及 5000/100 是否比 3000/60 拉开更大差距。

---

## 6. 速查表

| 项 | 值 |
|---|---|
| jax 版本 | **0.4.38**（jaxlib 同） |
| CENIE 依赖 | scikit-learn / joblib / threadpoolctl |
| wandb project | `gregjones11235-brown-university/multi_robot_ued` |
| baseline 池 = | BATCH_SIZE(1000) × NUM_BATCHES(5) = **5000**（yaml 默认，不用传） |
| adaptive 池 = | GEN_POOL_PER_GEN × 2(estimator)：1500→3000、2500→5000 |
| 跑满步数 | 3e8（`learning.TOTAL_TIMESTEPS=300000000`），45 eval epoch |
| 单 run 墙钟 | ~1.1~1.5 h |
| `envs/` 漏了 | → `ModuleNotFoundError: No module named 'envs'` |
| `+CURRICULUM_*` 漏 `+` | → Hydra struct 秒崩 |
