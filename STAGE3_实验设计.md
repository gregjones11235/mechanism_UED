# Stage 3 实验设计

> **目的**：把四个 verdict（T1–T4）的判据、`multi_estimator_plr` 的实测墙钟、以及 SFL/h_drift 的实测结论，落成一份可执行的 Stage 3 实验矩阵 + 工作量 + 双人 4-GPU 分配方案。
>
> **生成日期**：2026-06-19。基于：①四个 verdict（`gemini scripts/experiment optimization theorem/T{1,2,3,4}_redteam_verdict.md`）；②`任务运行时间分析.md` 实测墙钟；③本轮 measure/diag 实测（SFL 全程方差 0、ρ(PVL,CENIE) 跨 seed 漂移）。
>
> ⚠ **本文档区分"设计矩阵"与"实现现状"**：设计有 α/λ/w 三轴，但当前 `multi_estimator_plr` runner **只实现了 λ 一根轴**（=auction_lambda）。α 轴和"w 独立于 λ"需先改代码。见 §2.3。

---

## 0. 一句话结论

- **单 run 墙钟实测 ≈ 1.1h**（N=3 异质 PVL/SFL/CENIE + auction，PLR 血统、n_students=1；实测 1h06–08m）。比 Stage2 PAIRED 血统（N=3≈14h）快 ~13×。
- **可跑的轴现在只有 λ**（`--auction_lambda`）。**α 轴 + w-独立轴需先写代码**（§2.3）。
- **seed = 10**（与 Stage1 基线 ACCEL/PLR/PAIRED 的 10 seed 对齐，满足 rliable IQM+95%CI 投稿口径）。
- **粗筛保留，用"骨架优先"版**（先 λ×argmax×α* 看 frontier，再展开）——比 12 配置全 3-seed 粗筛省一半。
- **双人 4-GPU = 真并行**（不同学校、独立配额、不互抢）。按**配置整块切**分配（同一配置的 10 seed 不拆两边），墙钟约砍半。

---

## 1. 实验矩阵（设计层）

四个 verdict 把 Stage 3 拆成几根**正交轴**，每根由对应 verdict 管：

| 轴 | 设计配置数 | 含义 | 管它的 verdict | verdict 判定（实测修正后） |
|---|---|---|---|---|
| **α**（熵正则强度） | 2 | 收敛↔异质保真权衡，偏差 `Φ_α−Φ_0≤α log n` | T1+T2 | α 钉 `α*=(3c/4·log n)^{4/7}` + 邻域 1 点（×3→**×2**，L2 泛化层证不出故留邻域）；λ-无关 |
| **λ**（IC-violation 惩罚） | 3 | VCG-truthful ↔ non-IC 自由机制 | （主 novelty，不被削减） | 地板 {∞, 中, 0} |
| **w**（分配丰度） | 2 | single-winner ↔ fractional | T3 | maze 上 ρ 漂移→**不能事前判 argmax 够不够**→跑满（argmax + fractional） |
| **estimator/N** | N=3 锁死 | 用几类异质 estimator | T4 | maze 上 h_drift 失效→**不能事前砍**→跑满 N=3 + 事后消融 |

**核心配置数**（设计层）= α(2) × λ(3) × w(2) = **12 配置**。
argmax 端（w=single-winner）同时是**归因消融下界**（"只用一个 estimator + 同 α" vs "异质 auction"），内生于矩阵、不额外加 run。

---

## 2. 本轮实测对设计的三处硬修正

### 2.1 SFL 在 maze 上不可测（根因坐实，非 bug）
- 实测（meplr-arch s0/s1 各 14–15 个 archive ckpt，2500→37500）：SFL 对角方差 **全程恒 0**；diag 出 p 分布从 step=2500 起就两极（p∈{0,1}，0<p<1 仅 1%/0%/0%）。
- 根因：`LEARNABILITY=p(1-p)`（ued_scores.py:29），maze 关卡成功率天生二元 → p(1-p)≡0。与 idea可行性分析 §4.5.5 预警一致（SFL 适用连续回报环境）。
- ⚠ **未证伪的可能**：PerfectMaze（完美迷宫、尺寸连续可调）上 SFL 尚未实测；理论预判也会两极但未验。见 §6 待定 #4。
- **对设计的含义**：SFL 仍是 Stage3 一类 teacher（N=3 锁死），但在 maze 上其 bid 退化为常数。**靠事后消融（去 SFL 看性能降不降）定其贡献**，而非事前砍。SFL 主场是连续回报环境（Craftax/Kinetix）。

### 2.2 h_drift 失效（跨 seed 复现）→ T3/T4 事前省实验拿不到
- 实测：ρ(PVL,CENIE) 跨 step 漂移 **s0 range=0.73、s1 range=0.76**（两 seed 一致，远超 tol=0.15）。
- 按 T4 verdict 规则（波动>30%⇒H_drift 失效⇒退回事后消融）：**w 轴跑满、estimator 轴跑满+事后消融**。
- **四个 verdict 的省实验净账**：
  - **T1+T2（解析省）✅ 拿到**：α 轴 ×2、λ-无关不重扫。这是确定收益。
  - **T3+T4（实测省）❌ 没拿到**：h_drift 失效退回跑满。这是 verdict 正确预留的兜底，是诚实可写进论文的结论（异质 estimator 事前冗余删减在 maze 不可行，因 score 协方差随训练漂移）。

### 2.3 实现现状：α 旋钮已实现（路 A + score 标准化），完整矩阵可跑

| 设计轴 | 实现现状 | CLI |
|---|---|---|
| **λ** | ✅ 已实现 | `--auction_lambda`（∞=argmax/single-winner；有限=softmax 混合） |
| **α**（熵正则） | ✅ **已实现（2026-06-19，路 A + score 标准化）** | `--auction_alpha`（0=未启用走 PLR 原 rank/temp；>0=启用） |
| **w**（独立于 λ） | ⚠ 未独立：当前 `w=softmax(bids/λ)`，λ 同时管 single-winner↔fractional。设计的"w 作为独立第三类机制变体"暂未拆（λ 已覆盖该轴的连续谱，够用）。 | （并入 λ） |

**α 实现细节（路 A，已 CPU sanity 验证）**：
- **数学对象**：采样分布 = 熵正则内层解析解 `y*∝softmax(s̄_w/α)`（proof_skeleton A5），替换 PLR 的 rank/temp 变换。`α>0` 时 `_get_replay_dist` 走 `softmax(scores/α)`；`α=0`（默认）走原 rank 路 → **基线 ACCEL/PLR/PAIRED 零污染**。
- **score 标准化**：α 路启用时，混合前对每个 estimator 的 `s^(j)` 做 z-score（减均值除标准差）。**必须**——实测 CENIE~1e3 vs PVL~1e-4 量级悬殊，不标准化则 softmax 被 CENIE 支配、退化为"只听 CENIE"。
- **改动 4 处**（全向后兼容）：`plr.py`（PLRBuffer/PLRManager 加 `score_temp_alpha` + `_get_replay_dist` 分支）、`plr_runner.py`（透传）、`multi_estimator_plr_runner.py`（`auction_alpha` + z-score 标准化）、`arguments.py`（`--auction_alpha`）。
- **验证**（`_test_alpha_path.py`，CPU）：α=0 输出 `[0.48,0.24,0.16,0.12]`（标准 rank）；α=1 输出与 `softmax([2,1,0,-1])` 逐位相等；α=0.5 更尖锐（top1 0.865>0.644）。三条全过。

**含义**：**完整 12 配置矩阵（α×λ×w）现在就能跑**，无待写代码（w 并入 λ 谱）。

---

## 3. seed 策略：10 seed + 骨架优先粗筛

### 3.1 为什么 seed=10（不是 8）
- 文档三处独立写死正式口径 = **10 seed + IQM + 95%CI（rliable）**（§4.1.2 / §4.4.3 / OSCAR_部署指南）。
- **Stage1 基线已是 10 seed**（`accel/plr/paired-maze-s0..s9`）。Stage3 候选要和基线进同一张 rliable 对比表 → seed 数须对齐，否则审稿人问"为什么主方法 seed 比基线少"。
- §7.5.1："持平类结论必须补满种子"。Stage3 核心论断（异质 auction vs single-winner 有没有增益）最可能落到"持平"、最需窄 CI → 主候选必须 10。

### 3.2 粗筛：骨架优先（比 12 配置全 3-seed 省一半）
§7.5.1 两段式 + §7.5.3 active-search：
1. **骨架轮**：先跑 λ 的 3 点 × argmax × α*（=3 配置 × 3 seed = 9 run），看 **λ-frontier 形状**。这是最便宜的"机制有没有用"信号。
2. **展开轮**：据 frontier 决定在哪些 λ 上展开 fractional / 第二个 α 值（避免对注定崩的 λ 浪费）。
3. **正式轮**：top-2~3 候选补到 **10 seed**，出 IQM+95%CI。

> ⚠ §7.5.1 铁律：3 seed **只做排除不做确认**。明显垮的 setting 可砍；几个缠在一起的**标记"候选待加种子"，不得宣布持平**。

---

## 4. 工作量估算（单 run ≈ 1.1h，有效并发见 §5）

### 4.1 骨架轮（最高性价比第一步）
| 轮 | 配置 | seed | run | 墙钟（2 GPU 串行） |
|---|---|---|---|---|
| 骨架（λ×argmax，固定 α*） | 3 | 3 | 9 | ⌈9/2⌉×1.1h ≈ **5h** |

→ **5 小时就能看到 λ-frontier 的初步形状**，判断 non-IC 机制有没有信号。这是最高性价比的第一步——若连 λ-frontier 都没信号，再决定 α 轴是否值得展开。

### 4.2 完整矩阵（α 旋钮已实现 §2.3，现在即可跑）
| 阶段 | 配置 | seed | run | 墙钟（2 GPU 串行） |
|---|---|---|---|---|
| 粗筛（展开后）| ≤12 | 3 | ≤36 | ≈ **20h** |
| 正式（top-3 候选）| ~3 | **10** | 30 | ⌈30/2⌉×1.1h ≈ **17h** |
| measure Σ / h_drift / T3T4 | — | — | ~6 | 每个 ~1min（已有脚本，近免费） |
| **训练小计** | | | **~66 run** | **~37h（2 GPU 串行）** |
| PerfectMaze eval（对标 8%）| top 候选 | 10 | — | 单 seed ~91min；top-2 候选×10 ≈ **30h+** |

### 4.3 对比：选对 runner 的红利
同样 ~66 run：
- **PLR 血统（现状，1.1h/run）**：~37h。
- PAIRED 血统（Stage2 multi_teacher，N=3≈14h/run）：~460h（19 天）。
- **选 multi_estimator_plr 省了 ~12×。**

---

## 5. 双人 4-GPU 任务分配（不同学校、独立配额、真并行）

### 5.1 前提与硬约束
- **真并行**：两人各自 2-GPU 配额（`QOSMaxGRESPerUser` 各算各的，不互抢）→ 4 卡同时跑，墙钟 ≈ 单人的 1/2。
- **约束①（不同集群）**：队友 GPU 不一定是 Oscar → minimax 锁定环境、ckpt、measure 脚本要在两边各部署一份。
- **约束②（ckpt 分散）**：measure/h_drift/PerfectMaze eval 要读训练 ckpt → 训练分散则 ckpt 分散，后处理前要归集到一处。
- **约束③（rliable 口径一致）**：同一配置的 10 seed 最好在**同一集群**跑完，避免两边硬件/数值差异引入 seed 间不可比。

### 5.2 分配原则：**按配置整块切，不按 seed 切**
每个完整配置（含其全部 10 seed）整块分给一个人。**绝不**把同一配置的 10 seed 拆两边。

**示例（正式轮 3 候选 × 10 seed = 30 run）**：
| 人 | 负责配置 | run | 墙钟（自己的 2 GPU） |
|---|---|---|---|
| 你（Oscar） | 候选 A（10 seed）+ 候选 C（10 seed） | 20 | ⌈20/2⌉×1.1h ≈ 11h |
| 队友 | 候选 B（10 seed） | 10 | ⌈10/2⌉×1.1h ≈ 6h |

→ 不均时把零头 seed 也按"同配置不拆"原则微调；总并行墙钟 ≈ **max(两人) ≈ 11h**（对比单人 17h）。

### 5.3 后处理归集
- 训练完，队友把其 ckpt（或仅 logs.csv + 需 measure 的 archive ckpt）rsync 到 Oscar。
- measure Σ / h_drift / rliable 汇总 / PerfectMaze eval **统一在 Oscar 跑**（脚本都在这），保证判据层口径一致。
- PerfectMaze eval（91min/seed）是大头，也可两人分摊：各 eval 自己训的候选，CSV 汇总到一处喂 rliable。

### 5.4 基线复用
Stage1 的 ACCEL/PLR/PAIRED 10-seed 基线**已在 Oscar**，不必重跑。Stage3 候选只需和这些已有数字进同一张 rliable 表。

---

## 6. 待定变量（影响工作量，需你拍板）

1. ~~α 旋钮是否实现~~ → **已实现**（§2.3）。剩余可选：w 是否拆成独立于 λ 的第三轴（当前 λ 谱已覆盖 single-winner↔fractional，拆与否影响消融粒度，非必需）。
2. **fractional / 中间 λ / α>0 的真实墙钟**：当前 1.1h 是 argmax(λ=∞, α=0) 实测；softmax 混合 + α 路预计差异 <10%，但未实测（骨架轮会顺带测到）。
3. **是否纳入 BipedalWalker**（§4.4 第二环境）：纳入 → run 数 ×2、墙钟 ×2（BipedalWalker 单 run 墙钟另需实测）。
4. **PerfectMaze 上 SFL 是否补测**（§2.1）：理论预判两极但未验；若补测 = 1 个 diag 作业 ~5min。
5. **PerfectMaze eval 对几个候选做**：每候选 10 seed × 91min ≈ 15h；建议只对最终 top-1~2 做。

---

## 7. 推荐执行顺序

1. **立刻**（无需改代码）：跑骨架轮（λ×argmax，3 配置×3 seed，~5h），看 λ-frontier 有没有信号。**这一步决定整个 Stage3 值不值得继续投入。**
2. 若 frontier 有信号 → 实现 α 旋钮（§2.3 #1），展开粗筛。
3. 锁 top 候选 → 双人分配（§5）补 10 seed。
4. 归集 ckpt → measure Σ + h_drift + 事后消融（去 SFL / 去 CENIE）。
5. top-1~2 候选 → PerfectMaze eval → rliable 表对标 8%。

---

## 附：关键 CLI（multi_estimator_plr）
```
--train_runner multi_estimator_plr
--estimators PVL,SFL,CENIE        # N=3 异质；事后消融时去掉某个
--auction_lambda inf              # ∞=argmax(single-winner)；有限值=softmax fractional 混合
# α 熵正则旋钮：⚠ 待实现（§2.3）
--archive_interval 2500           # 存历史 ckpt 供 measure/h_drift
--maze_height 13 --maze_width 13 --maze_n_walls 60 --maze_replace_wall_pos True
```
