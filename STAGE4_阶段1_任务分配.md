# Stage 4 · 阶段 1（go/no-go）任务分配 —— Alec & Henry 四卡并行

> **目标**：并行跑完 [STAGE4_实验设计.md](STAGE4_实验设计.md) §1「go/no-go：learnability 方差复活 + λ 扫描」。
> **人力/算力**：Alec 2× A6000 + Henry 2× A6000 = **4 卡并行**。
> **容错前提**：本分配保证 **Alec 可无缝接替 Henry**——Henry 那半若失败/掉线，Alec 拿这份文档即可独立续完全部 35 run。
>
> **生成日期**：2026-06-22。
> **注意**：本阶段是 **go/no-go 短跑**（趋势验证，**非** SOTA 判定）。SOTA 对标是阶段 2 的事，别混。
>
> **✅ 本文档已完全确定，无需任何人讨论。** 短跑规模（`TOTAL_TIMESTEPS=1e8`）、seed 划分、看门狗阈值
> 全部钉死，命令可直接复制（§1.5）。Alec / Henry 各自照表提交即可；Henry 那半失败时 Alec 照 §4 独立补齐。

---

## 0. 实验是什么（一句话）

跑 **5 档配置 × 7 seed = 35 个 run**，看两件事：
1. **方差复活**：generator 注入（方案 B）相比 SFL 随机海选基线，`learnability_set_scores` 的**方差是否 >0**（基线在某些阶段会塌缩到 0）。
2. **λ 甜区**：把 `GEN_AUCTION_LAMBDA` 当主扫描轴，找让方差最大 / student 学得最好的 λ。

**go** = 方差明显 >0 且随 λ 有结构 → 进阶段 2；**no-go** = 方差仍塌缩 → 回查信号/generator/auction。

---

## 1. 共同前置（两人开跑前都要确认，已由 Alec 完成代码侧）

### 1.1 代码缺口已补 ✅
`learnability_set_scores`（方差主指标）两条路径都已落盘，无需改动。
**唯一缺口已补**：`gen_metrics`（auction_weights / bids 轨迹，STAGE4 §1.3 第二指标）原先没进 wandb，
现已在 `jaxnav_sfl.py` 的 `test_metrics` 字典补落盘（基线 `GENERATOR_INJECTION=false` 时 `gen_metrics=None`，
不落这些键，**零回归**）。补丁已 py_compile 通过。
- 新增 wandb 键：`learnability_set_var`、`auction_weight_{difficulty,pvl,cenie}`、`auction_bid_{...}`、`gen_mean_score`、`gen_injected`、`gen_n_incomplete`。
- **两人必须用同一份打过补丁的 `jaxnav_sfl.py`**（Henry 从 Alec 处同步该文件，别用旧版，否则 auction 曲线拿不到）。

### 1.2 环境与运行命令（两人各自机器一致）
```bash
conda activate sfl          # jax 0.4.38 血统（minimax 的 0.5.3 锁不适用 SFL repo）
cd <sampling-for-learnability 根目录>
# 自检：JAX 必须认到 GPU，否则秒退别空烧
python -c "import jax; print(jax.devices())"   # 必须看到 cuda/gpu device
python -c "import sys; sys.path.insert(0,'sfl/train'); import pcgrl_generator, auction_bid, cenie_density; print('import OK')"
```

### 1.3 单个 run 的命令模板（已含短跑规模，照填即可）
```bash
python sfl/train/jaxnav_sfl.py \
    SEED=<seed> \
    GROUP_NAME=<见 §2 表分组名> \
    WANDB_MODE=online \
    GEN_ESTIMATOR_IDS="[difficulty,pvl,cenie]" \
    AUCTION_USE_CENIE=true \
    learning.TOTAL_TIMESTEPS=100000000 \
    <注入档参数：见 §2 表>
```

### 1.5 五档的完整可复制命令（直接照抄，把 `SEED=` 改成 0..6）
```bash
# 基线（s4-base）—— 注意：基线不带 GEN_* 注入参数
python sfl/train/jaxnav_sfl.py SEED=0 GROUP_NAME=s4-base WANDB_MODE=online \
  GEN_ESTIMATOR_IDS="[difficulty,pvl,cenie]" AUCTION_USE_CENIE=true \
  learning.TOTAL_TIMESTEPS=100000000 GENERATOR_INJECTION=false

# explore 端（s4-lambda-5_0）
python sfl/train/jaxnav_sfl.py SEED=0 GROUP_NAME=s4-lambda-5_0 WANDB_MODE=online \
  GEN_ESTIMATOR_IDS="[difficulty,pvl,cenie]" AUCTION_USE_CENIE=true \
  learning.TOTAL_TIMESTEPS=100000000 GENERATOR_INJECTION=true GEN_AUCTION_LAMBDA=5.0

# 甜区候选（s4-lambda-1_0）
python sfl/train/jaxnav_sfl.py SEED=0 GROUP_NAME=s4-lambda-1_0 WANDB_MODE=online \
  GEN_ESTIMATOR_IDS="[difficulty,pvl,cenie]" AUCTION_USE_CENIE=true \
  learning.TOTAL_TIMESTEPS=100000000 GENERATOR_INJECTION=true GEN_AUCTION_LAMBDA=1.0

# exploit 端（s4-lambda-inf）
python sfl/train/jaxnav_sfl.py SEED=0 GROUP_NAME=s4-lambda-inf WANDB_MODE=online \
  GEN_ESTIMATOR_IDS="[difficulty,pvl,cenie]" AUCTION_USE_CENIE=true \
  learning.TOTAL_TIMESTEPS=100000000 GENERATOR_INJECTION=true GEN_AUCTION_LAMBDA=inf

# fallback 消融（s4-lambda-none）
python sfl/train/jaxnav_sfl.py SEED=0 GROUP_NAME=s4-lambda-none WANDB_MODE=online \
  GEN_ESTIMATOR_IDS="[difficulty,pvl,cenie]" AUCTION_USE_CENIE=true \
  learning.TOTAL_TIMESTEPS=100000000 GENERATOR_INJECTION=true GEN_AUCTION_LAMBDA=none
```
- **短跑规模（已钉死，所有 35 run 必须用这个值，不得改动）**：追加 hydra override
  ```
  learning.TOTAL_TIMESTEPS=100000000
  ```
  （即 `1e8`。**完整命令见 §1.5**。）
  - **为什么是 1e8**：满跑是 `TOTAL_TIMESTEPS=3e8` → `NUM_UPDATES≈2250` → `EVAL_FREQ=50` → **45 eval epoch**（实测）。
    截到 1e8 = 满跑的 1/3 → `NUM_UPDATES≈750` → **15 eval epoch**。这是从实测 45-epoch/3e8 的线性换算直接推出的确定值，
    非估计。15 个 eval epoch 足够看出方差是否复活 + λ 是否有结构（实测方差/正交曲线在 50–2250 步全程都有结构）。
  - 其余 config（`NUM_ENVS` / `EVAL_FREQ` / `ROLLOUT_STEPS` 等）**一律用默认值，不 override**——保证两人 35 run 完全可比。

### 1.4 看门狗（已钉死阈值，两人都按这个判死锁）
- **XLA 线程死锁 ~10% 概率**：所有线程 futex_wait、GPU 0%。判死后 **kill 该 run + 用原 (GROUP_NAME, SEED) 重跑**。
- **判死阈值（确定值）**：
  - **第一个 eval epoch：给足 20 分钟。** 含 XLA 编译，N=3 generator rollout 编译比打分层重
    （打分层实测 eval0=110s；注入路径更重，留足余量，20 min 内别动）。看到 `[DIAG] eval_step=0` 打印即证在跑。
  - **第 2 个 epoch 起：单 epoch 超过 10 分钟无新日志 → 判死锁，kill 重跑。**
    （打分层纯运行实测 49s/epoch；注入路径更慢但同量级，10 min 是 ~12× 余量，超出必是死锁非慢。）
- **CENIE 自动开 GMM**：`GEN_ESTIMATOR_IDS` 含 `cenie` 时 runner 自动冷启 GMM + 每 epoch 重拟，无需手动开。

---

## 2. 实验矩阵（5 档 × 7 seed = 35 run）

| 档位 | 注入档参数（填进 §1.3 模板末行） | GROUP_NAME | 作用 |
|---|---|---|---|
| **基线** | `GENERATOR_INJECTION=false` | `s4-base` | SFL 随机海选，方差塌缩参照（**最关键对照**） |
| **explore 端** | `GENERATOR_INJECTION=true GEN_AUCTION_LAMBDA=5.0` | `s4-lambda-5_0` | 近 uniform，三信号近等权 |
| **甜区候选** | `GENERATOR_INJECTION=true GEN_AUCTION_LAMBDA=1.0` | `s4-lambda-1_0` | 软混合，默认起点 |
| **exploit 端** | `GENERATOR_INJECTION=true GEN_AUCTION_LAMBDA=inf` | `s4-lambda-inf` | single-winner 赢者通吃 |
| **fallback 消融** | `GENERATOR_INJECTION=true GEN_AUCTION_LAMBDA=none` | `s4-lambda-none` | 各 gen 自评，不跨 gen auction |

- **seed = 7**（每档 seed 0..6）。go/no-go 用 7 seed 看 λ 甜区趋势够稳；阶段 2 才上 10 seed 对齐 baseline。
- `GEN_AUCTION_LAMBDA` 取值由 runner 解析：`none`→各 gen 自评 / `inf`→single-winner / 有限值→fractional 混合。

---

## 3. 人员划分（按对照完整性切，不按机械等分）

> **划分原则**：每个人手上的档要能**自成对照**，且**基线最优先**（它是方差复活的判据锚点，没它整个实验不成立）。
> Alec 拿 **基线 + 两个端点**（结论骨架）；Henry 拿 **中段两档**（填曲线、可被 Alec 接替）。

### 3.1 Alec（2 卡）— 18 run · 骨架档
| 档位 | GROUP_NAME | seeds | run 数 |
|---|---|---|---|
| **基线**（最高优先，先跑） | `s4-base` | 0–6 | 7 |
| exploit 端（λ=inf） | `s4-lambda-inf` | 0–6 | 7 |
| 甜区候选（λ=1.0）的 **前 4 seed** | `s4-lambda-1_0` | 0–3 | 4 |
| **小计** | | | **18** |

> Alec 先把 **基线 7 个**跑出来——基线方差塌缩是 go/no-go 的判据底座，必须最早拿到。

### 3.2 Henry（2 卡）— 17 run · 填充档
| 档位 | GROUP_NAME | seeds | run 数 |
|---|---|---|---|
| explore 端（λ=5.0） | `s4-lambda-5_0` | 0–6 | 7 |
| fallback 消融（λ=none） | `s4-lambda-none` | 0–6 | 7 |
| 甜区候选（λ=1.0）的 **后 3 seed** | `s4-lambda-1_0` | 4–6 | 3 |
| **小计** | | | **17** |

> 甜区档（λ=1.0）被**两人按 seed 切开**（Alec 0–3 / Henry 4–6），这是故意的：它是最可能成为最终甜区的一档，
> 两人分摊既加速、又让"Alec 接替 Henry"时甜区档已有一半在 Alec 手里，接起来最顺。

### 3.3 每人 2 卡内部怎么排
- 每张卡一次跑 1 个 run（jaxnav N=3 单 run 已占满一张卡）。2 卡 = 2 run 并发。
- 一个档 7 seed = ⌈7/2⌉ = 4 波。各自按 `seed 0,1 → 2,3 → 4,5 → 6` 顺序放行。
- 用 `WANDB_MODE=online` + 上表 `GROUP_NAME`，wandb 自动按档分组，两人结果汇到同一 project 直接对比。

---

## 4. Alec 接替 Henry 的预案（容错核心）

**触发条件**：Henry 机器宕机 / 环境跑不起来 / 看门狗反复杀同一 seed / 失联。

**接替步骤**（Alec 单人即可续完，无需 Henry 在场）：
1. **看 wandb**：按 GROUP_NAME 数 Henry 已完成的 run（每个 finished run = 一个 (档, seed)）。
   - Henry 三档分组名固定：`s4-lambda-5_0`、`s4-lambda-none`、`s4-lambda-1_0`（seed 4–6）。
2. **列缺口**：Henry 应交 17 run，wandb 缺哪些 (档,seed) 就补哪些——配置全在 §2/§3 表里，无隐藏状态。
3. **Alec 直接续跑**：用 §1.3 同一命令模板，把缺的 (GROUP_NAME, SEED) 填进去，在 Alec 自己 2 卡上排。
   - 每 run 自包含、无跨 run 依赖、无 checkpoint 接力问题 → 谁的卡上跑都一样。
4. **总量校验**：最终 wandb 上 5 个 group 各 7 个 finished run（基线/5.0/1.0/inf/none），共 35，即全齐。

> **为什么接替无痛**：① 所有 run 配置纯由 (GROUP_NAME, SEED, λ档) 决定，写死在本文档表里；
> ② seed 固定 0–6，可精确点名缺口；③ 单 run 无状态依赖、无 checkpoint 续跑陷阱。
> Alec 接替 = 把 Henry 的 17 run 当成自己待办表的延伸，照表补齐即可。

---

## 5. 跑完看什么（产出 go/no-go 结论）

对照 STAGE4 §1.3，每人/汇总后看三组指标（全在 wandb，按 GROUP 分组）：

1. **方差复活（主判据）**：各档 `learnability_set_var`（或离线对 `learnability_set_scores` 数组算 var/p_std）
   随 eval epoch 的曲线。**基线 `s4-base` 应见塌缩段；注入档应见方差 >0**。
   - ⚠ jaxnav 上 learnability 是**双峰分布**（一半 success≈1 一半≈0），用 **p_std** 判饱和（`p_std<0.05` 才算饱和，双峰不算），别用旧 `frac_hi+frac_lo` 判据误报饱和。
2. **auction 轨迹**：`auction_weight_{difficulty,pvl,cenie}` / `auction_bid_{...}` 随 epoch——
   哪个 teacher 何时出价高、λ 怎么调尖锐度（λ=inf 应近 one-hot，λ=5.0 应近等权）。
3. **student win rate 趋势**：eval set win rate（趋势性，**非最终 SOTA 判定**）。

**判据**：方差明显 >0 且随 λ 有结构 → **go**（写一句话结论 + 选出 top 1–2 λ 进阶段 2）；
方差仍塌缩 → **no-go**（回查信号/generator/auction，别进阶段 2）。

---

## 6. 耗时分析（跑完补，现在写不出来）

> 本阶段是 generator 注入路径（`GENERATOR_INJECTION=true`，扫 `GEN_AUCTION_LAMBDA`），GPU 上**尚无跑满实测**。
> 现有 52min 是**旧 auction 打分层路径**（job 3363545/46/47：`AUCTION_SCORING=true` 且 `GENERATOR_INJECTION=false`，
> 扫的是 `AUCTION_LAMBDA` 不是 `GEN_AUCTION_LAMBDA`），不含 N=3 generator rollout，**不能外推墙钟、也不能用来定本阶段的 α**
> —— 那批已实测三档 sampled_wr AUC/收敛速度 CI 全重叠（打分层口径 auction 无增益，详见 [方案B_多generator设计.md](方案B_多generator设计.md) §6.1）。
> **第一批 run 出来后**，用真实 epoch 墙钟仿
> [任务运行时间分析.md](任务运行时间分析.md) 补一份诚实的耗时表（现在写=瞎猜）。
> —— 这不影响开跑：§1–§5 已完全确定，照着提交即可，墙钟分析是事后补记。
