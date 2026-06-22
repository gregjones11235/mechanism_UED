# 异质 Estimator / 多 Generator 路线设计与结论

> 记录于 2026-06-20。本文档汇总「基于 SFL/CENIE/PVL 改装异质 estimator」这一设计阶段的
> 所有用数据钉死的结论、路线分叉、与待验证的地基。后续设计以此为锚，不再凭记忆重推。

---

## 0. 背景：当前 Stage3 框架的本质

- 跑的是 `multi_estimator_plr` runner（**PLR 血统**，n_students=1，**无 generator 网络**）。
- 三个 estimator = PVL / SFL / CENIE，对**同一个共享 PLR buffer** 的同一批 level 打分，
  auction 用 `w=softmax(bid/λ)` 混合。⚠ **λ 语义修正（2026-06-20）**：代码里 λ 是 softmax 温度——
  **λ=∞（特判）及 λ→0 都是 argmax/单赢家端，λ 大才是均匀端**。旧表述「λ→0 → 均匀」**写反了**，
  这正是 §1.1 实测「{∞,1.0,0.01} 三点都落 argmax 端」的部分原因（∞ 和 0.01 本就同端）。见
  [方案B_多generator设计.md](方案B_多generator设计.md) §6.2 bug③。
- teacher ≡ 打分函数，不是生成器。收敛定理成立前提就是「teacher=评分器、对手在单纯形 y 上」。

---

## 1. 用数据钉死的诊断结论（全部有实测支撑）

### 1.1 λ 在当前实现下是「空旋钮」（实测 Δb）
- 实测 bids = [PVL, SFL, CENIE] = [+0.011, 0.0, −337]（α=0 未标准化，诊断作业 3356495）。
- 三类 bid 量级差 ~3e4 倍。CENIE 是大负数 → argmax(bid) 永远是 PVL，**CENIE 权重在任何 λ 下都≈0**。
- `softmax(bid/λ)`：λ=1.0 时 Δb/λ≫1 → 已接近 one-hot；λ=0.01 更彻底。
  → {∞,1.0,0.01} 三个点**都落在 argmax 端**，扫不到「中间混合区」。
- 实测后果：λ=∞ 与 λ=1.0 的 test 性能统计上分不开（0.893 vs 0.875，CI 重叠）。
- **根因是 bid 没标准化**（z-score 标准化被 `auction_alpha>0` 门控，本轮 α=0 没开）。
- ⭐ **修正（2026-06-20，numpy 推导验证）**：根因比这更深，共三个 bug——① α=0 不标准化（本条）；
  ② **即使开 α 路标准化也没用**：现实现 bid = z-score 后**全体均值**，而 z-score 定义上零均值
  ⇒ bid≈0 ⇒ softmax 塌成 uniform，λ 同样失效（只是塌到另一端）；③ λ 语义写反（见 §0）。
  **修复 = bid 改用「z-score 后 top-k 分位均值」**，已 numpy 验证 λ 能扫出连续 frontier、CENIE
  拿到实质权重。完整设计与验证见 [方案B_多generator设计.md](方案B_多generator设计.md) §6。
- λ-frontier 那 30 个 run 已全部取消（继续跑无意义，全在同一端点）。

### 1.2 SFL 在当前框架下结构性退化为 0（非 bug、非公式塌缩）
- 实测：512 个 level 上 SFL 全 0（n_unique=1）；p 分布 mean=0.809，80.7% 在 p=1、19.1% 在 p=0、
  仅 0.2% 在中间（诊断作业 3351567）。
- **不是** p(1-p) 公式塌缩，**是 p 的分布极度两极**（强 student 把大多 level 学到 p∈{0,1}），
  两端 p(1-p)=0，中间地带只占 0.2%。

#### ⭐ 关键修正（2026-06-20，复查 done_counts 实测，推翻「p 二值化」说）
- 复查 diagp-3352589 实测：**done_counts mean=10.7、max=252、全0占比=0%** —— 每个 level
  其实平均完成了 **~11 个 episode**，p **已经是多 episode 估计**，不是单 episode 二值化！
- 所以「n_eval=1 + rollout 短 → p 二值化」这个说法**错了**。p 有能力取中间值（1/11, 5/11…）。
- **SFL≈0 的真因更硬**：maze 上对训过的 student，「将学未学」(p≈0.5) 的关卡**本身就极稀少**
  （diagp 实测 0<p<1 仅占 1%，且越训越少：2500步 1.0% → 10000步 0.0%）。
  一个迷宫要么 student 策略能解(p→1)、要么解不了(p→0)，几乎没有「有时解出」的中间态——
  **是关卡难度相对 student 的真双峰，不是采样/估计问题**。
- 退化机制因此修正为：
  1. **maze 关卡相对 student 真双峰**：learnable(p≈0.5) 关卡密度极低（~1%，递减），这是环境性质。
  2. **SFL 被当 estimator 在 PVL 主导 buffer 上打分**，learnable level 几乎不在池里
     （论文 Table 5：PLR buffer learnability 恒=0.01）。
  3. SFL 丧失主动海选环节（见 §2）。
  → 修复方向不是「改 n_eval/拉长 rollout」（已证无用，本就多 episode），而是「加大采样量
    海选 + 看 maze 上 learnable 关卡绝对数量够不够」（见 §4）。

### 1.3 SFL 的本质：随机生成 + 智能筛选（论文 Algorithm 1+2）
- **SFL 没有任何 generator agent**，关卡纯 DR 随机生成（Algorithm 2 第一行 `B ← N random levels`）。
- 它的全部「智能」在**筛选**：rollout N 个随机 level → 算 p(1-p) → 留 top-K learnable。
- 默认超参（Minigrid）：**N=40000**、K=1000、T=100（每 100 update 重新海选）、多 episode 估 p。
- learnable level 在随机池里只占 ~0.2%，所以 N 必须极大（4 万）才能撞到几十个。
- **关键认知**：SFL ≠ 打分函数，是一个**自带采样策略的完整方法**。把它降格成「对别人选的
  buffer 打分」就会退化（=我们 Stage3 踩的雷）。论文 Table 5 "SFL(Selected)" 海选后 learnability
  达 0.22（接近 0.25 上限），"SFL(All Sampled)" 不筛只有 0.01。

### 1.4 环境层面：maze 是 SFL 最不利的环境（文档 §3.2）
- 你们用的 minimax AMaze 13×13 ≈ ACCEL/SFL 都用的 Minigrid，**环境同源、差别不大**。
- 真正差别不在环境，在**没跑 SFL 的海选机制**：SFL Minigrid 用 N=40000，你们 n_parallel=32，
  差 ~1250 倍采样量 + 没做 top-K → learnable level 撞不到。
- 文档明确警告（§161）：「Craftax/Kinetix 才是体现 teacher 多样性价值的环境，MiniGrid 只是
  sanity check，不要在 MiniGrid 结果上建立过多论点」。
- maze 用二值 p(1-p)，易两极塌缩；Craftax/Kinetix 用连续回报 + NCC 式11 的连续 Gen-SFL，
  learnability 不塌缩 —— **learnability 的真正主场在连续回报环境**。

---

## 2. 设计核心矛盾：「覆盖更多关卡」无法靠改 score 函数达成

- 用户核心目标：**让多个 teacher 的观点覆盖到更多关卡可能性**。
- 当前 auction 里三个 teacher 评的是**同一个共享 buffer 的同一批 level** → 只是「对同一批关卡的
  不同排序」，**不是「覆盖更多关卡」**。覆盖范围由「谁生成候选 level」决定，而非由 score 函数决定。
- 故「覆盖更多关卡」必须改「谁生成候选 level」（注入层），不能只改打分。

### 「注入层 / 打分层」分层设计
```
注入层（扩大候选池，各自往池里加 level，不经过 auction）：
   ├─ SFL          → 随机采 N + 留 learnability top-K
   ├─ generator-A  → 各自偏好的关卡
   └─ generator-B  → ...
                          ↓ 汇成共享候选池
打分层（对池里 level 排序，进 auction，min-max 只在此层）：
   ├─ PVL    → 打分
   └─ CENIE  → 打分
```
- 注入时不经过 auction（各加各的 = 覆盖扩大的来源）；注入的 level 被打分层一起排序。
- auction 选的是「池里哪些 level 优先回放」，**不是「哪个 generator 赢」**。
- PVL/CENIE 相关性高（ρ=0.73）不再是问题——它们只负责排序，覆盖由注入层保证。

---

## 3. 两条路线分叉（核心决策点，尚未拍板）

### 路线 A：随机生成 + 多筛选器（理论干净）
- 注入器 = 同一批 DR 随机 level + **不同筛选标准**（SFL 按 learnability 筛、覆盖型按 CENIE 筛…）。
- **无任何学习型 agent**，随机采样=DR，**不破坏凸性、收敛理论免费**。
- 效率低（大海捞针，N 要大）、覆盖中、novelty 中（SFL 变体）。
- SFL 在 Craftax/Kinetix 是 SOTA 且全程无 generator → 证明路线 A 足够强。

### 路线 B：多 generator agent 协作/竞争（用户倾向，高风险高回报）
- 注入器 = **学习型 generator agent**，主动构造高 learnability 关卡（替代 SFL 被动随机海选）。
- 合理内核（数据支撑）：SFL 随机生成效率极低（99.8% rollout 浪费在非 learnable 关卡）；
  generator 若能主动瞄准 p≈0.5 则效率天差地别；多 generator 分工覆盖不同 learnable 子区域。
- **如做成 = 顶会级贡献**（首个多 generator 协作 + learnability + 收敛保证的 UED，几乎无人做）。

#### 路线 B 的三个真实障碍（每个都是开放研究问题）
1. **generator 的训练信号**（最难）：learnability=p(1-p) 依赖 student 当前能力、对 generator 参数
   不可微（关卡离散）、且内层 non-stationary。PAIRED 用 regret 的 minimax 结构能 work，但
   **learnability 不是 regret**，PAIRED 理论用不上，须自设 generator 训练机制。
2. **收敛理论**：多 generator 协作/竞争 = **多人 general-sum 博弈**；learnability 让博弈**非零和**
   （generator 要 p≈0.5、student 要 p=1，非纯对抗），纳什存在性都要重证。bilevel(Hong) 已被
   memory 否决（非凸内层）。**理论上是开放问题**，可能是最硬骨头也是最大 novelty。
3. **载体**：从 PLR 系换到 PAIRED 系（multi_teacher_runner），训练慢 3-4 倍，auction/λ/α 全要
   重新适配到「generator 输出关卡」语义。

| 维度 | 路线A | 路线B |
|---|---|---|
| 效率 | 低 | **高** |
| 覆盖 | 中 | **高** |
| Novelty | 中 | **高** |
| 收敛理论 | 干净（免费） | **开放难题** |
| 工程 | 轻 | 重 |
| 风险 | 低 | **高** |

---

## 4. 待验证的地基（决定 A/B 都成立与否）

**前置问题（比路线选择更先）：maze 上随机采 N 个 level，到底有没有、有多少 learnable level？**

- 若 maze 上随机 5000 个里 learnable 的几乎没有 → generator 再强也没有 learnable 关卡的目标空间，
  **B 在 maze 上同样地基不稳**；A 的海选也救不了 → 必须上 Craftax/Kinetix。
- 若海选 top-K 的 learnability 能到 0.22（论文 Table 5 量级）→ 证明 maze 上 SFL 可救、是机制没
  跑对而非环境问题 → A 的注入器路线成立。

**验证脚本**：在已有 ckpt（s3sk-Linf-A0-s0，跑满 30000）上跑「随机采 N=5000 + 多 episode 估 p
+ 按 learnability 排序看 top-K」，报告 p 分布 + top-K learnability。两种结果都指导下一步。

### B 的命门问题（地基过后立刻要答）
**generator 用什么信号训练才能朝高 learnability 生成？** 在搞清这个之前，B 的收敛理论和工程都是
空中楼阁。建议地基验证后用 deep-research 调研「generator 优化 learnability（而非 regret）」的现有机制。

---

## 4.5 ⭐ SFL 官方复现实证（2026-06-20）—— 回答"SFL 凭什么 learnability≠0 而我=0"

在 Oscar 独立 conda 环境 `sfl` 部署 SFL 官方 repo（jax 0.4.38 + flax 0.10.2 + jaxmarl/jaxued），
按官方原生流程**边训边海选**（每 EVAL_FREQ 步用当前 student 重新海选），learnability 进 wandb
字段 `learnability_set_mean_score`（top-K 海选 buffer 的平均 learnability）。

**实测对比（决定性，从 wandb 云端拉完整轨迹，非末值）：**
| 环境（边训边海选 2seed） | 初期 learnability | 末期 | 全程均值 | 衰减 |
|---|---|---|---|---|
| **Minigrid（离散）** | **0.246**（≈上限0.25） | **0.040/0.051** | **0.13/0.15** | 快速暴跌 |
| **JaxNav（连续）** | **0.232** | 0.19 | **0.22/0.22** | 缓慢维持 |
| 你之前：maze + **训满 ckpt 事后测** | — | **0.000** | — | 踩在曲线最低谷 |

**⭐ 关键修正（推翻早前"离散环境上限低"的末值误判）：**
两个环境**训练初期 learnability 都接近 0.25（理论上限）**！差别只在**衰减速度**，不在初值：
- **Minigrid（离散）**：student 学得快 → 关卡迅速被掌握 → learnable 关卡快速枯竭 →
  learnability 从 0.246 单调暴跌到 0.04。
- **JaxNav（连续）**：任务更难/学得慢 → 始终有大量"将学未学"关卡 → learnability 稳定 0.19-0.25。

**根因精确化（你最初问题"SFL 凭什么≠0 而我=0"的完整答案）：**
1. **时机是决定性的**：learnability 是训练【过程量】且**随训练单调衰减**。SFL 全程边训边海选，
   捕捉的是高位的早中期（Minigrid 初期 0.246！）。你在**训满 ckpt** 上测 → 正好踩在衰减曲线
   **最低谷**（0.04，更激进训满测到 0.00）。不是环境不行，是测在了曲线尾部。
2. **环境只影响衰减速度，不影响初值**：离散 maze 早期 learnability 照样到 0.25，只是衰减快。
   故"离散 maze 是 learnability 弱环境"应修正为"**离散 maze 是 learnability 衰减快的环境**"。
   论文倒 U 图用 JaxNav 是因为它**全程维持高位**、图好看，不是 Minigrid 初期没有 learnability。

**对路线 B 的含义（强化）**：maze 上 learnable 关卡在**训练早中期大量存在**（learnability 高达 0.25），
只是随 student 变强快速枯竭。这正是 generator 的用武之地——**当随机海选的 learnable 关卡枯竭时
（Minigrid 末期 0.04），generator 可主动构造维持高 learnability**，把 Minigrid 的快速衰减曲线
"撑"成 JaxNav 那样的缓慢维持。地基不仅没塌，而且 generator 的价值正是"对抗 learnability 衰减"。

**复现环境部署记录**（独立 conda `sfl`，不碰 minimax 的 jax 0.5.3 锁）：
- jax/jaxlib 0.4.38（jaxmarl pin jax<=0.4.38）；**flax 必须 0.10.2**（0.10.4 缺 `jax.api_util.debug_info`；
  0.8.5 用废弃 tracer `.level`；0.10.2 正好卡 jax 0.4.38 兼容窗口）。
- requirements.txt 缺漏需补：distrax(--no-deps) + tensorflow-probability + decorator + wandb + tqdm + imageio。
- matplotlib 3.10 移除 `tostring_rgb`：minigrid_sfl.py 和 jaxnav_sfl.py 都改 `buffer_rgba`（RGB→RGBA）。
- 训练入口：`sfl/train/minigrid_sfl.py`（config minigrid-sfl，BATCH_SIZE=5000×NUM_BATCHES=5, ROLLOUT_STEPS=2000）；
  `sfl/train/jaxnav_sfl.py`（config jaxnav-sfl，BATCH_SIZE=1000×5, ROLLOUT_STEPS=1000）。hydra 覆盖 SEED/GROUP_NAME/WANDB_MODE。
- wandb entity=默认（gregjones11235-brown-university）；项目 minigrid / multi_robot_ued。

## 5. 关联 memory / 文档
- [[sfl-zero-variance-pq-bimodal]]（SFL 全 0，本文档 §1.2 是其精确化版本）
- [[generator-teacher-breaks-convergence]]（路线 B 障碍2 的来源）
- [[heterogeneous-score-convergence-finding]]（bilevel 不是出路、撞车风险低、熵正则扛收敛）
- [[auction-mechanism-is-plr-not-paired]]（路线 B 障碍3 载体切换）
- UED实验步骤指导.md §3.2（环境层级）、§6.1（SFL 是你框架的特例）、§232（连续 Gen-SFL）
- proof_skeleton_entropy_regularized_heterogeneous_UED.md §8（两层 general-sum）
