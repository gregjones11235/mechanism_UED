# 异质 estimator 信号层 + V1 正交性探针 —— 集成与部署记录

对应设计：`mechanism_UED/方案B_多generator设计.md` §5（信号去留 + 数据驱动决策树）、§6（auction bid）。
本轮产出「写 estimators」的全部代码 + 边训边验探针，已部署到 Oscar。**真实 verdict 需跑训练才出**
（本机无 jaxnav/sfl env，合成测试只证代码正确，不证信号正交——见 §「真实数据 caveat」）。

---

## 1. 文件清单（载体 = SFL repo `sampling-for-learnability`）

| 文件 | 职责 | 测试 |
|---|---|---|
| `estimators.py` | 纯 JAX 信号层：difficulty-match `-(p-0.5)²`、learnability `p(1-p)`、组装 `(N,M)` 矩阵、两两相关、决策树报告 | `_test_estimators.py` **22/22** |
| `auction_bid.py` | estimator 无关 bid：z-score + top-k bid + λ-frontier 权重（已有，上轮产出） | `_test_auction_bid.py` 16/16 |
| `probe_orthogonality.py` | 探针信号管道：`compute_four_signals`（四真信号→`(4,M)`）+ `write_reports`（相关 csv/npz/png/verdict） | `_test_probe_pipeline.py` **16/16** |
| `train_probe.py` | 训练时探针入口：`log_orthogonality_step`（host 拟 GMM + 相关 + 饱和判定）+ `summarize_trace`（跨 epoch 总裁决） | `_test_train_probe.py` **17/17** |
| `cenie_density.py` | 官方 GMM 密度（从 `minimax/util/rl/` 复制，自包含，仅 numpy/jax/flax） | 官方实现 |
| `jaxnav_sfl.py` | 训练主文件，3 处接线 + 多 checkpoint（见 §3） | py_compile OK（Oscar sfl env） |

**四个真信号（无任何代理）**：difficulty-match、learnability、PVL(`positive_value_loss`)、CENIE(`-log p(GRU hidden)`)。
为什么不用代理见 `estimators.py` 顶部「信号边界」：value_disagreement 单 critic 拿不到（换真实 PVL）；
CENIE 必须用官方 GMM+hidden（不用 outcome k-NN——那是另一个信号不是退化版）。

---

## 2. Oscar 部署状态（已完成）

```
~/sampling-for-learnability/sfl/train/
  estimators.py            ← 新增
  probe_orthogonality.py   ← 新增
  train_probe.py           ← 新增
  cenie_density.py         ← 新增（从 ~/minimax/src/minimax/util/rl/ 复制）
  jaxnav_sfl.py            ← 已打补丁（备份 jaxnav_sfl.py.bak_preprobe）
```

**已验证**：sfl env 下 `import train_probe` OK（整条链通：train_probe→probe_orthogonality→estimators+
cenie_density+`sfl.util.jaxued.jaxued_utils`）；`jaxnav_sfl.py` py_compile OK；`get_probe_signals` 已嵌入。

**回滚**：`cp sfl/train/jaxnav_sfl.py.bak_preprobe sfl/train/jaxnav_sfl.py`（其余 4 个新文件删除即可，互不影响训练）。

### import 路径（部署时踩过的两个坑，已修）
- PVL：`from sfl.util.jaxued.jaxued_utils import positive_value_loss`（**不是** `from jaxued...`；jaxnav_plr.py 同款）。
- CENIE：`from cenie_density import ...`（**不在** sfl env 装 minimax 包——会依赖冲突；改用自包含单文件副本）。

---

## 3. jaxnav_sfl.py 三处接线（diff 摘要）

1. **import**（顶部）：`from train_probe import log_orthogonality_step, summarize_trace`。
2. **`get_probe_signals(rng, network_params)`**（新函数，紧跟 `get_learnability_set` 后）：
   独立 rollout，emit 四信号原料——success_rate/num_episodes（难度+learnability）、dones+GAE
   advantages（PVL）、GRU hidden+per_level_index（CENIE）。**独立于 get_learnability_set，零回归**
   （后者返回 top-100 筛选结果、结构不兼容；探针要全量 M=5000）。
3. **eval for-loop**（jit 外）：每 epoch 调 `get_probe_signals`→`device_get`→`log_orthogonality_step`，
   wandb.log `probe/*` 指标；checkpoint 改为**多份带 step 后缀**（`model_step{N}.safetensors`，
   保留 `model.safetensors` 向后兼容）；训练末 `summarize_trace` 出总裁决。
   开关：`config["PROBE_ORTHOGONALITY"]`（默认 True）。探针异常被 try/except 兜住，不影响训练。

---

## 4. 怎么跑（边训边验）

```bash
ssh oscar
cd ~/sampling-for-learnability
conda activate sfl
# 用现成 SFL 配置（jaxnav 主场 multi_robot_ued, Grid-Rand-Poly, num_agents=1）
python sfl/train/jaxnav_sfl.py        # 读 sfl/train/config/jaxnav-sfl.yaml
# （建议挂 sbatch：墙钟≥24h + 邮件通知，见 mechanism_UED memory「SBATCH脚本两条硬规约」）
```

产物（`checkpoints/multi_robot_ued/<run>/probe/`）：
- `orthogonality_trace.csv` —— 每 eval epoch 一行：update_count、verdict、max_abs_offdiag、
  六对 ρ、p_std、frac_sat_hi/lo、saturated。
- `verdict_summary.txt` —— 训练末总裁决（**只采非饱和 epoch 的多数**）。
- wandb `probe/rho/*`、`probe/max_abs_offdiag`、`probe/saturated` 曲线（看相关随训练漂移）。

---

## 5. 怎么读 verdict（喂 §5.4 决策树）

`summarize_trace` 的 `verdict` 三态：
- **`ready_pool_sufficient`** → 现成池有 ≥3 个两两 |ρ|<0.5 正交信号 → 主线用现成池；
  transition-error / co-learnability 列 future work。
- **`must_implement_expensive_signal`** → 现成池撑不起 N=3 正交（像 maze ρ(PVL,CENIE)=0.73）→
  **按 §5.4 贵信号必须实现，不能因实现贵而绕过**（正交是 generator 分化硬前提）。
- **`inconclusive_all_saturated`** → 所有 epoch success_rate 都贴 1.0（信号塌角落）→ 换更难
  test set 或看更早 epoch。

**注意 difficulty~learnability 必然 ρ=1.0**：learnability=`p(1-p)`=`0.25-(p-0.5)²`，与 difficulty=`-(p-0.5)²`
差一个常数+符号，是同维度的线性变换（§5.0 已判定同维度）。所以四信号里**有效正交维度最多 3 个**
（难度 / 不确定性PVL / 覆盖CENIE）——决策树该在这三维上判，learnability 只是难度维的另一种写法。

---

## 6. 三个诚实标注的设计取舍（影响解读，非 bug）

1. **饱和防护是关键**：现有 jaxnav ckpt 全是末期（win rate≈1.0），事后单点分析会得退化相关。
   边训边验 + 饱和判定（`p_std<0.02` 或两极占比>80% → 标 saturated，总裁决排除）正是为此。
   → **必须重训出非饱和阶段轨迹**，别拿现有末期 ckpt 下结论。

2. **GAE 的 last_val 用末步 value 近似**（非重跑 apply 拿真 bootstrap）：ROLLOUT_STEPS=1000 长
   rollout，末步 bootstrap 经 γλ 衰减可忽略；PVL 是 level 间相对信号，常数偏差不改相关。对测正交无害。

3. **per_level_index 假设 num_agents=1**（jaxnav-sfl.yaml 确认）。多 agent 场景 `a_env = a_idx %
   BATCH_SIZE` 的 hidden→level 映射要复核。

---

## 7. 真实数据 caveat（最重要）

**本轮所有 PASS = 代码逻辑正确，≠ 信号正交。** 单元测试用合成（numpy 随机）数据，只隔离验证
信号公式 / NaN 安全 / 相关矩阵 / 控制流。**真实 ρ 只能由 jaxnav 真 rollout 得出**——要跑上面 §4 的
训练，读 `orthogonality_trace.csv` 才有可喂决策树的数字。合成相关值是无意义随机数，绝不能当结论。
```
本机限制：WSL 无 sfl env / 无 jaxnav / Y 盘未挂 /mnt → 真 rollout 只能在 Oscar。
```
