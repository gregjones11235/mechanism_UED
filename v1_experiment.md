# v1 实验参考 —— DiCode 复现的评测协议 + Log 记录清单（2026-06-30）

> 用途：正式复现 DiCode 前固化两件事——(1) 训练/评测协议（对标口径），(2) 一次 run 能拿到哪些 log（用于设计我自己的多 LLM Proposer + auction 方法）。
> 来源：DiCode 论文（arXiv:2602.08194）原文 + 官方开源代码深度核验（head `35d4ae4`）。
> 关联：[方法设计_v1.md](方法设计_v1.md)、[DiCode复现_接入点评估.md](DiCode复现_接入点评估.md)。
> 正式复现参数（用户拍板 2026-06-30）：**全量 2e9 + Qwen3-235B-Thinking（DeepInfra 同款 FP8）+ 开 wandb**。

---

## 1. 训练 / 评测协议（对标口径，必须照搬才能公平比 48.33）

### 1.1 训练分布（student 在哪训练）
- **不是纯生成关卡**：训练预算分布（论文 Table 5「With Newly Generated Envs」列）：
  - **20% 原始 target Craftax**（Target Env Worker Proportion = 0.20）
  - **53% 新生成关卡**（New Env Worker Proportion = 0.53）
  - **27% archive 回放**（Replay Env Worker Proportion = 0.27）
- 即 **80% 生成/回放课程 + 20% 真实 Craftax**。student 一直也在见真环境，非纯靠生成关。
- 生成关卡的 reward/termination 结构镜像 target 环境（继承原生 reward + goal completion bonus B_t，非新 reward）。

### 1.2 评测协议（SOTA 48.33 怎么测出来的）★
论文 §4.1 原文：「During training, we **archive policy checkpoints at 50 uniformly spaced intervals**. We **evaluate each checkpoint on a fixed held-out test set of 1024 procedurally generated Craftax instances**, reporting mean return and standard error across seeds.」

- **训练与评测完全分离**：
  1. 训练时存 **50 个均匀时间间隔的 checkpoint**。
  2. 每个 checkpoint 拿去在**固定的 1024 个 held-out（未见过的）原始程序生成 Craftax 世界**上评测。
  3. held-out 1024 与训练用的生成关卡**完全隔离** → 测的是**泛化能力**（训练分布外的真 Craftax）。
- **SOTA 48.33** = 训练末期（2e9 步）那个 checkpoint 在 1024 held-out 上的 **mean return**，跨 **5 seed** 取均值 + standard error。
- **per-achievement success rate**（论文 Figure 3 Achievement Breakdown：CRAFTING/COMBAT/GATHERING/DESCENDING 四类，含 Gnome Warrior/Archer 等；Figure 5 全成就聚合）也在此 held-out 评测里测。
- 卖点 = late-game 成就（Gnome Warrior/Archer）baseline 塌到 0%，DiCode 非零。

### 1.3 用户理解确认（2026-06-30）
- ✅「训练只在生成关卡（+少量真环境）」——基本对，修正=有 20% 原始 target。
- ✅「训练后用 50 个 checkpoint 在 1024 原始 Craftax 测」——对。
- ✅「1024 是完整原始 Craftax 难度、测泛化」——对（后期成就极难，性能能上去=泛化强）。注：1024 是标准程序生成世界（随机地形/资源/敌人），非特殊「全石头」地形。

---

## 2. 一次 Run 能拿到的 Log（用于设计我的方法）★

核验结论：**三类关键信息全都有记录**。

### 2.1 Student 失败在哪个具体成就 ✅
- **评测层**：`evaluation/skill_{成就名}`（0-100%），每个 Craftax 成就分别记录（如 `evaluation/skill_gnome_warrior`）。= Figure 3 数据源。来源 `craftax_evaluation.py:243-249` → `online_evaluation.py:80-83`。
- **训练层（更细）**：每个生成关卡带 `achievement_srs` 字典（`scoring.py:170-178`），记 student 在**那一关**每个成就的成功率 → 能定位「哪个生成关、哪个成就卡 0」。

### 2.2 有无性能回降 ✅
- wandb `evaluation/mean_return` + `evaluation/skill_*` 是**时序**（按 session），直接看曲线掉没掉。
- 更细：`task_graph.graphml` 每个 task 有 `performance_history`（按 session 记 sr/achievement_srs/mean_return）→ 追踪单关跨时间表现，回降发生在哪关/哪成就可查。来源 `gen_manager.py:254-275`、`run_dicode.py:204-206`。

### 2.3 每次 LLM 沟通的完整内容 ✅（对设计 auction 最关键）
- **`curriculum/generation_table`**（wandb table + 本地 `curriculum_generation_table.json`，跨 resume 持久化）。列：
  - `parent_task_id` / `parent_task`（从哪个父关进化）
  - `LLM_reasoning`（**FM 完整推理文本**）
  - `generated_docstring`（生成的关卡描述 = 式8 产物）
  - `final_code`（生成的关卡代码 = 式9 产物）
  - `compilation_status`（编译成功/失败）
- 来源 `evolution_efficient.py:330-364`（`_log_generation_results`）+ `logging_utils.py:149-192`。

### 2.4 其他 wandb 指标（完整清单）
- **Student 训练**（每 update，`ppo_tr.py:220-234`）：`train/total_loss`、`value_loss`、`actor_loss`、`entropy`、`grad_norm_mean`、`grad_norm_max`、`global_step`。
- **Session 级**（每 session，`logging_utils.py:64-83`）：`curriculum/num_tasks_activated/compiled`、`activation_success_ratio`、`training/avg_sr_trained`、`avg_lp_trained`、`curriculum/num_newly_{A,B,C,D}`（任务难度分级数）、`archive_{A,B,C,D}_pct`、`archive_total_tasks`、`curriculum/interactive_graph`、`system/worker_wait/execution_time`、`efficiency_ratio`。
- **评测**（`online_evaluation.py:79-83`）：`evaluation/mean_return`、`mean_performance`（=return/226×100）、`average_episode_length`、`skill_*`。

### 2.5 本地输出文件（一次 run 落盘）
| 文件 | 来源 | 内容 |
|---|---|---|
| `task_graph.graphml` | `gen_manager.py:188` | NetworkX DiGraph：所有 task + 完整 `performance_history` + parent-child + learnability/PVL/MaxMC score + reasoning + code |
| `curriculum_generation_table.json` | `logging_utils.py:171` | LLM 交互记录本地副本（reasoning/docstring/code/编译状态）|
| `runtime_analysis/timings.csv` | `runtime_analysis.py:51-57` | 每 session 各组件耗时（FM 等待 vs 训练 vs 编译）|
| `runtime_analysis/runtime_breakdown.png` | `runtime_analysis.py:59-82` | 运行时间堆叠图 |
| RL checkpoint 目录 | `setup.py:68-77` | agent 权重（50 个间隔 ckpt）|

> ⚠️ 落盘位置：`task_graph.graphml` 存**当前工作目录**（`gen_manager.py:188`）。正式跑设 `hydra.run.dir` 到 scratch + job chdir 进去 → graphml/runtime/wandb 本地副本都落在 scratch 的 output dir。正式跑脚本须理清这些路径，确保跑完能完整导出。

---

## 3. 对我的方法设计的直接价值

| 我的方法零件 | 依赖的 log | 怎么用 |
|---|---|---|
| **auction bid - Coverage** | `achievement_srs`（细粒度成就 SR） | 算「哪条依赖链/成就没覆盖」= 互补覆盖度 |
| **auction bid - AmbitionGain** | `evaluation/skill_*`（target 上各成就缺口） | 攻的依赖链深度 × target 缺口 |
| **N-Proposer 对比** | `generation_table`（parent + reasoning + docstring）| N 个提案落表对比分歧/互补；parent tracking 看谱系 |
| **Critic 补盲** | per-achievement 失败数据 | 判断「archive 缺哪条链」的输入 |
| **性能跟踪/回降诊断** | `performance_history` 时序 | 验证课程是否真提升、有无回降 |

→ **结论**：官方 log 足够支撑我设计多 LLM Proposer + auction + Critic 的全部信号，无需在复现阶段额外加 logging（正式跑先用官方原生记录，要补再补）。

---

## 4. 待办（启动正式复现前）
1. 正式 sbatch：开 wandb（绕过 `_log_callback` 在 `use_wandb=false` 时无条件调 `wandb.log` 的 bug，见短跑诊断）、全量 2e9、Qwen3-235B-Thinking。
2. checkpoint 续跑机制：单 seed 估 ~12-13h（[[deepinfra-fm-latency-measured]]），可能跨 12h 墙钟 → 申请更长墙钟或确认 `load_checkpoint` 续跑可用。
3. 输出路径理清：保证 `task_graph.graphml` + `curriculum_generation_table.json` + wandb 数据跑完能完整导出（§2.5 ⚠️）。
4. 评测口径对齐：确认官方评测确实在 1024 固定 held-out 上跑、存 50 ckpt（§1.2），保证主表能与 48.33 同口径比。

---

## 5. 完整超参数表（论文 ↔ 仓库逐项核对，2026-06-30）★

**核对结论：23 项全部一致**。开源仓库 `conf/training/default.yaml` + `conf/dicode_manager/default.yaml` 的默认值就是论文跑出 48.33 的配置，**可直接跑，复现忠实**。我们的复现脚本只改了 FM provider（local→deepinfra 同款模型）+ wandb 落点，**训练参数一字未动**。

### 5.1 PPO 优化（论文 Table 2 「General Optimization」）
| 超参 | 论文 | 仓库 key | 值 | 一致 |
|---|---|---|---|---|
| Number of Workers（并行环境） | 1,024 | `num_envs` | 1024 | ✅ |
| Steps per Worker（rollout 长度） | 128 | `num_steps` | 128 | ✅ |
| Initial Learning Rate | 2×10⁻⁴ | `lr` | 2e-4 | ✅ |
| Min Learning Rate | — | `min_lr` | 2e-6 | （anneal 终点）|
| LR Schedule | anneal | `anneal_lr` | true | ✅ |
| Discount γ | 0.999 | `gamma` | 0.999 | ✅ |
| GAE λ | 0.8 | `gae_lambda` | 0.8 | ✅ |
| Clip Range ε | 0.2 | `clip_eps` | 0.2 | ✅ |
| Entropy Coefficient | 0.002 | `ent_coef` | 0.002 | ✅ |
| Value Function Coefficient | 0.5 | `vf_coef` | 0.5 | ✅ |
| Epochs | 4 | `update_epochs` | 4 | ✅ |
| Number of Minibatches | 8 | `num_minibatches` | 8 | ✅ |
| Max Gradient Norm | 1.0 | `max_grad_norm` | 1.0 | ✅ |
| Activation | ReLU | `activation` | relu | ✅ |
| Total Timesteps | 2×10⁹ | `total_timesteps` | 2_005_401_600 | ✅ |

### 5.2 GTrXL 网络架构（论文 Table 2 「Network Architecture (GTrXL)」）
| 超参 | 论文 | 仓库 key | 值 | 一致 |
|---|---|---|---|---|
| Embedding Size | 256 | `embed_size` | 256 | ✅ |
| QKV Features | 256 | `qkv_features` | 256 | ✅ |
| Number of Heads | 8 | `num_heads` | 8 | ✅ |
| Number of Layers | 2 | `num_layers` | 2 | ✅ |
| Hidden Layer Size | 256 | `hidden_layers` | 256 | ✅ |
| Memory Window | 128 | `window_mem` | 128 | ✅ |
| Gradient Window | 64 | `window_grad` | 64 | ✅ |
| Gating Mechanism | True | `gating` | true | ✅ |
| Gating Bias | 2.0 | `gating_bias` | 2.0 | ✅ |

### 5.3 课程机制（论文 Table 5「Foundation Model」「With Newly Generated Envs」）
| 超参 | 论文 | 仓库 key | 值 | 一致 |
|---|---|---|---|---|
| 生成频率 v | 2 iterations | `evolution_interval` | 2 | ✅ |
| Updates per Curriculum Iteration | 100 | `max_updates_per_session` | 100 | ✅ |
| Target Env Worker Proportion | 0.20 | `original_task_proportion` | 0.2 | ✅ |
| Replay Env Worker Proportion | 0.27 | (派生) | — | ✅ |
| New Env Worker Proportion | 0.53 | (派生) | — | ✅ |
| Num Unique New Envs | 10 | `num_generation_tasks` | 10 | ✅ |
| Num Unique Replayed Envs | 5 | `training_sample_size_n` 相关 | 16(总采样) | 注* |
| parent 选择信号 | learnability | `score_function` | learnability | ✅ |
| active buffer 容量 | — | `active_task_capacity` | 100 | — |

> 注*：`training_sample_size_n=16` 是每轮训练总采样数（含 new+replay+target），论文的 5 replayed/10 new 是其中分解；reward 含 goal completion bonus（`completion_bonus_scale=2.0`/`completion_bonus_min=20.0`/`bonus_type=dynamic`/`dynamic_bonus_k=2.0`）。

### 5.4 Foundation Model（论文 Table 5）
| 超参 | 论文 | 仓库 / 我们的复现 | 一致 |
|---|---|---|---|
| Model ID | Qwen/Qwen3-235B-A22B-Thinking-2507-FP8 | DeepInfra: Qwen/Qwen3-235B-A22B-Thinking-2507（FP8）| ✅ 同款 |
| Max Tokens | 32,768 | 32768 | ✅ |
| Temperature | 0.6 | 0.6 | ✅ |
| Top-p | 0.95 | 0.95 | ✅ |
| 托管方式 | HuggingFace API（作者低优先级排队基础设施）| DeepInfra 弹性端点（更快，见 §7 of 方法设计_v1）| 我们的优化 |

### 5.5 易混点 / 配置噪音（已澄清，不影响）
- **论文 Table 2「Hyperparameters specific to SFL」**（Buffer 4000 / Rollout Length 1,500 / Update Period 640 / Sample Ratio 1.0）= **SFL baseline 专属**，**不是 DiCode 配置**，别拿来对 DiCode。
- 仓库 `training/default.yaml` 顶注「EXACT CONFIGURATION FROM SCRIPT B」=作者从某 Script B 抄的精确配置，对得上论文表。
- 仓库 `wandb_project: SIACE` / `wandb_entity: airl-lab`（training 层旧值）被顶层 `config.yaml` 覆盖，无影响。

---

## 6. 正式复现 Job（2026-06-30 提交）
- **Job ID**：3575920（seed 0，go/no-go 单 seed 先跑通）
- **配置**：全量 2e9 + Qwen3-235B-Thinking（DeepInfra）+ 开 wandb（entity `gregjones11235-brown-university`，project `DiCode-repro`）
- **墙钟**：48h（免费 plan 上限）；**续跑就绪**：固定 `WANDB_RUN_ID=dicode-repro-s0-v1` + 固定 output dir `/oscar/scratch/jzhu223/dicode_outputs/repro_s0_v1` + `load_checkpoint=true` → 中断/超时后重交脚本自动从最近 ckpt + task_graph 续跑（机制：`setup.py:80-99` 自动 restore ckpt + 从 wandb resumed run 恢复 step 计数）。
- **预期**：单 seed ~12-13h（DeepInfra 加速后，见 [[deepinfra-fm-latency-measured]]），一个 48h job 应能跑完。跑通后再上多 seed 对标 5 seed。
- **唯一改动 vs 官方**：FM provider（local→deepinfra 同款模型）+ wandb 落点。训练参数 0 改动（§5 核对一致）。

---

## 7. 机制设计辨析 —— 「auction」名实与 top-k 的性能边界（2026-06-30 讨论）★

> 触发问题（用户）：auction mechanism 领域里有没有「多 agent 竞争/协作提升整体输出质量」的范式？我现在只用「基于 bid 的 top-k」，理论上是不是不如真正的 auction？
> 结论先行：**就「课程质量」而言 top-k 不输（次模目标下贪心已 (1-1/e)-最优）；就「机制完备性/抗操纵」而言 top-k 确实缺一层（无 payment/incentive）。后者决定「auction」这个命名当前名不副实。**

### 7.1 我当前设计的机制定位（要诚实）
当前「基于 bid 的 top-k」**严格说不是 auction**，是 **scoring-based curation / 加权 top-k selection**（本质 = Coverage 集合函数下的贪心分配）。缺 auction 的两个定义性要素：
- **无价格（payment）**：winner 选出后不付任何代价。
- **无策略空间（strategic bidding）**：bid 是 auctioneer（我）**单方面算**出来的（Coverage/Endorsement/AmbitionGain 都是我给 Proposer 打的分），Proposer 自己**不报价**。
→ 写论文时要么把名字精确化为 *scoring-based curation*，要么把机制升级成真 auction（见 7.4）。

### 7.2 相关范式全景（按与本场景契合度排序）
| 范式 | 核心 | 比 top-k 多给的 | 与本场景的张力 |
|---|---|---|---|
| **A. Combinatorial Auction**（组合拍卖） | bidder 对**物品组合**报价，auctioneer 解 winner-determination 最大化社会福利 | 选 k 个**互补**关 = 组合分配问题；top-k 是「独立打分各取前 k」忽略选了 A 后 B 的边际变化 | WDP 是 NP-hard，但**次模目标下贪心 = (1-1/e) 近似** → 我的贪心 top-k 恰是其多项式可解特例（卖点：不是退而求其次，是因目标次模故贪心可证最优） |
| **B. VCG / payment** | 让每个 winner 付它给别人造成的外部性 → 诚实报价成占优策略 | **抗操纵（strategy-proof）**：防某 Proposer 夸大覆盖标签挤掉别人（尤其 Endorsement 互评可串谋/自夸） | 需 Proposer 有**私有估值且策略性行动**；当前 LLM-Proposer 只生成提案、不报价，故 VCG 暂无用武之地——除非把架构改成 Proposer 自报覆盖+置信度（= 把假 auction 变真 auction 的关键一步） |
| **C. Fisher Market / 竞争均衡** | 把训练名额当稀缺资源，Proposer 竞价，价格由供需均衡决定 | **动态价格**：稀缺覆盖（没人攻的深链）价高→激励去攻；饱和覆盖价低→自动去重。top-k 是静态阈值 | new_direction「多 FM 市场」卖点的真正形式化；工程量大 |
| **D. Cooperative game / Shapley value** | 给每个 Proposer 在所有联盟里的边际贡献均值 | 不是选择机制，是**归因/消融工具**：证「异质底座每个 FM 都有不可替代的边际贡献」 | 直接支撑 B<A<C |

### 7.3 「top-k 是否不如真 auction」的精确回答（区分两种性能）
- **(a) 课程质量（选出的 k 关好不好）→ top-k 不输。** Coverage 次模 ⇒ 贪心 top-k 已 **(1-1/e)-最优**；真组合拍卖的 WDP 在次模目标下也只能做到这个近似比（除非 P=NP）。**就最大化覆盖质量，top-k 没留下性能。**
- **(b) 抗操纵 / 激励相容（Proposer 会不会学会骗分）→ top-k 确实输。** top-k 无 incentive 层，一旦 Proposer 策略性（或 reviewer 追问「FM 学会夸大覆盖怎么办」），**无任何理论保护**；VCG 在此给 strategy-proofness。
→ 故诚实结论：**不是「性能」输给 auction（质量维度已最优），是「机制完备性」缺一块**——无 payment/incentive ⇒ (i) 命名名不副实，(ii) 抗操纵无理论保证。

### 7.4 v1 / v2 双线（2026-06-30 用户拍板：v1/v2 同步推进，v2 为后续工作重点）
| 线 | 做法 | bid 来源 | 与 DiCode 隔离 | 论文价值 |
|---|---|---|---|---|
| **v1（scoring-based curation）** | 客观计算 bid（Coverage 从 multi-hot 算、Endorsement 从互评算）→ 贪心 top-k；证 **(1-1/e) 质量最优** | auctioneer **客观算**，Proposer 无法影响 ⇒ 无说谎可能 ⇒ 不需 strategy-proof | 极干净（N=1 去 Critic 逐字退化回 DiCode） | 稳、好审、归因清晰 |
| **v2（机制塑造供给 / 价格引导生产）★后续重点** | 让 Proposer 在生成 `<docstring>` 同时多吐 `<bid>`（自报覆盖+置信度+学习增益）→ 自报值进入选择/价格 → 价格信号反馈给下一轮生成 | Proposer **自报私有估值**（制造操纵动机，strategy-proof 才成真定理） | 变脏（Proposer 看到机制规则 ⇒ 提案分布本身变了，不再是「只改选择」而是「改了生成条件」） | 高，且组合真空 |

**v2 三个澄清（2026-06-30 用户三问 → 结论）**：
1. **不必换 agent**：v2 = v1 的同一 Proposer **一次调用里多输出一个自报 bid 字段**（prompt 多要一段结构化自评），不需第二个 agent/换底座。**但**：v2 的全部理论价值来自「bid 从客观计算→主观自报」制造的**操纵动机**——若自报值不真实影响 winner/payment，Proposer 无动机说谎，机制空转、退化回 v1。故 v2 必须设计成**自报值真实影响 winner 与 price**。
2. **v1/v2 同步跑（非 future work）**：升级消融骨架 = 在 B<A<C 上叠一个正交维度「机制是否反馈影响 Proposer 行为」。新对照 = **「Proposer 报价但忽略报价仍用客观 bid」vs「自报值真进入价格并反馈生成」**——干净科学问题：*让生成者参与定价机制，是否改变其生成内容、提升课程质量*（UED 内无人做过此对照）。**代价**：v2 基线要重新对齐，否则赢了说不清赢在「自报」还是「机制反馈」。
3. **★机制塑造供给 ⇒ 可突破固定池 (1-1/e) 上界（最重要 framing）**：(1-1/e) 有一个**隐藏前提=候选池固定外生**，它只约束 *selection 最优性*，不上界 *池子质量*。v2 让机制反馈塑造生成池后，目标从 `max over k-subset of FIXED pool` 变为 `max over 机制 m 的 Coverage(top-k(pool(m)))`——**两个不同优化问题**。(1-1/e) 完全没上界外层 ⇒ **v2 系统整体性能可严格超过 v1 贪心 top-k 的任何可达值**。比喻：(1-1/e)=从固定一篮苹果挑最好 k 个；v2=让果农先种出更好的苹果再挑。**这正是经济学里 auction/market 的真正功能（价格引导生产，而非分配既有价值），搬进 UED 课程生成 = 干净组合真空。**

### 7.5 数学理论可做点（本讨论衍生，待形式化）
1. **★Coverage 次模性 + 贪心 (1-1/e)（v1 定理）**：把「覆盖的依赖链/成就」建模为集合覆盖，证 Coverage 满足 diminishing returns（单调次模）⇒ 贪心 top-k 继承 Nemhauser 保证。**衡量 selection 最优性（池子固定）。最硬、风险最低。**
2. **★★机制塑造供给 ⇒ 突破固定池上界（v2 命题，比 (1-1/e) 更强更新颖）**：构造「机制反馈使 Proposer 把概率质量移到未覆盖依赖链」的过程，证其极限 Coverage **严格 > 无反馈池的期望 Coverage**。本质 = **mechanism-induced distribution shift**：机制不分配既有价值、而**创造**价值。⚠️ **半理论半经验**：可在纸上证条件命题「若反馈把分布往未覆盖区移，则覆盖提升」，但「反馈真的让 LLM 这么移」是经验断言、靠实验验证前件 → 正是「v1/v2 同步跑」对的原因（v2 实验去验证理论的经验前件）。这是 conditional theorem + empirical verification 组合。
3. **多 Proposer 可分离性下界**：构造单 FM（单一提案分布）有常数概率漏掉某条依赖链、而 N 异质提案覆盖概率 →1 的 separation，把「正交性」从直觉变定理，支撑 intro 立论链第 3 步。
4. **Endorsement / 自报诚实的 mechanism design**：见 §7.7 机制选型。
5. **Critic 补盲收敛**（风险高，可能撞 GenEnv α-Curriculum）：archive 覆盖率单调逼近 target 全集 → **暂列 future work**。

### 7.6 v2「价格引导生产」框架下的机制选型 —— VCG 不是最强（2026-06-30 讨论）★
> 触发问题（用户）：v2「价格引导生产」框架下，有没有比 VCG 更强的机制？

**关键判断：在「价格引导生产」框架下，VCG 是个错配的机制。** VCG 是**静态、一次性、配置型（allocative）**机制——「给定一组已有 bidder 的私有估值，如何诚实分配既有物品」。它 ✅ 保当轮自报诚实（strategy-proof），但 ❌ **完全没有跨轮「价格信号引导生产」功能**（VCG 的 payment 为诚实而非为「告诉 Proposer 下次往哪生产」）。故 VCG 守住 §7.4 抗操纵，但**没碰** §7.4-3 的真卖点（突破固定池上界）。→ 要「价格引导生产」，须看别的机制。

| 机制 | 对「价格引导生产」契合 | 理论红利 | 新颖度 | 备注 |
|---|---|---|---|---|
| **① Walrasian / Fisher 竞争均衡** | ★★★（字面本体） | **福利第一定理**（均衡 Pareto 最优）+ 次模/gross-substitutes 下均衡存在性（Kelso-Crawford）正好接次模 Coverage | 中 | 维护每个成就/依赖链的**影子价格**：饱和成就价→0（别再造）、无人覆盖深链价→高（值得造）；价格反馈给 Proposer 当下一轮生成上下文。**价格本身=生产信号**，正是 §7.4-3 的机制实现 |
| **② All-pay / Contest（锦标赛）** | ★★★（努力引导） | **均衡努力闭式解**（Tullock/all-pay）；指导**怎么设计 top-k 奖励梯度（第1名 vs 第k名待遇差）最大化整体提案质量** | **★★★最高** | 利用本场景独有结构：**所有 Proposer 不管中不中标都已付出生成成本（LLM 调用+推理）= all-pay 定义性结构**。把「努力」映射「提案质量/野心」。把 LLM 多 Proposer 课程生成形式化成 contest 用均衡指导奖励——无人做过 |
| **③ Posted-price + prophet inequality** | ★★（流式匹配） | **prophet 竞争比 1/2**（对抗最优提案出现顺序未知） | 中 | 每成就维度挂事前公布价格阈值，提案「买得起」就注入，价格按历史覆盖在线更新；**在线/增量/Proposer 看价再生成**，天然适配「每 cycle 生成」流式课程 |
| **④ (协作侧) Market scoring rule（Hanson）** | — | proper scoring rule 让自报预测诚实 + market scoring 聚合多 Proposer 共识 | 中 | 若 Proposer 自报「学生会学到多少」，给 **Endorsement 维度**一个诚实聚合的理论基础 |
| VCG | ★（只保诚实不引导生产） | strategy-proof | 低 | **降级为「抗操纵的一个零件」，非 v2 核心** |

**v2 选型倾向**：主推 **① Walrasian 价格信号做生产引导（价格当生产信号 + 福利第一定理）** + **② contest 视角做奖励设计（利用 all-pay 结构 + 均衡努力闭式解，新颖度最高）** 的组合；VCG 降级为抗操纵零件。**理由**：① 是「价格引导生产」的字面本体且带福利定理；② 利用了「所有 Proposer 都已付出生成成本」这个本场景独有、其他机制没利用的结构，新颖度最高。

### 7.7 ① Walrasian vs ② Contest 的风险对比 + 分层方案（2026-06-30 讨论）★
> 触发问题（用户）：① 和 ② 谁风险更高（e.g. 不诚实 LLM 错误地占据主导）？

**关键：两者风险性质不同，答案随维度反转。**

| 风险维度 | ① Walrasian | ② Contest |
|---|---|---|
| **不诚实 LLM 占主导**（用户主顾虑） | **低** | **高** |
| **机制存在性 / 收敛** | **高** | **低** |
| **实现 / 调试复杂度** | **高** | **低** |
| **半经验前件脆弱性** | 信号复杂（要 LLM 看懂价格） | 信号简单（奖励梯度闭式） |

**为何 ② Contest 在「不诚实占主导」上更脆弱（三点结构性原因）**：
1. **All-pay 均衡内生鼓励虚张**：提高均衡努力的代价 = 参与者把资源砸在「显得有竞争力」而非真实价值上 → Proposer 学会把提案**包装得野心大、覆盖标签标得广**而非真造好关。**奖励梯度越陡（越榨努力），虚张激励越强**——是 all-pay 内生张力非 bug。
2. **赢家通吃放大单点错误**：陡分档下，一个**自信但错误**的 LLM（烂提案标高覆盖）能直接挤掉一群诚实谦虚的 Proposer = 用户说的「不诚实占主导」。Walrasian 价格**连续/边际**：报高只压低该成就影子价一点，被他人稀释，不会一票定生死。
3. **Contest 缺天然诚实锚**：优化「努力强度」，无内建机制把自报值钉回真实。

**为何 ① Walrasian 抗操纵结构性更低**：价格是市场出清均衡产物，Proposer 是 **price-taker**（竞争充分时操纵自己 bid 改不了均衡价多少）；且可**外接客观锚**——影子价由「实际注入后学生在该成就的真实 SR 提升」事后校准（§2 log 的 `achievement_srs` 正好提供），自报值无法长期偏离真实。Contest 的努力无此事后地面真值校准接口。

**但账要算全 —— ① 有 ② 没有的硬风险**：均衡存在依赖估值满足 **gross-substitutes**，LLM 自报估值很可能**不满足** → 市场可能**不出清/无均衡/价格震荡**，机制直接失效；且每轮要解均衡（不动点/tâtonnement 迭代），不收敛就卡住。即 ② 是「机制能跑但可能被骗」，① 是「机制可能根本跑不起来」。

**结论排序**：抗操纵风险 ② > ①；落地存在性风险 ① > ②。**用户主顾虑（不诚实占主导）方向上 ② 更危险。**

**★推荐分层方案（v2 主架构）**：
- **主干 = ① Walrasian 影子价格**（连续、可事后校准、抗单点操纵）决定「哪些成就/依赖链值得造」。
- **子模块 = ② Contest 受约束**：仅在**同一成就内**用温和奖励梯度激励努力强度 → 拿 ② 的「引导努力」红利，但虚张激励被价格层客观锚**封顶**，不吃「赢者通吃放大错误」的毒。
- **Fallback = v1 客观 top-k**：若 tâtonnement 不收敛/无均衡，退回 v1（永远能跑）→ 正好让 v1 成为 v2 的安全网（又一个 v1/v2 同步跑的理由）。
- **VCG**：保留为抗操纵零件（自报值进价格前的诚实约束），非核心。

---

## 8. Proposer 底座选型 + thinking-vs-instruct 探索点（2026-06-30，deep-research 对抗验证）★

### 8.1 ★关键发现：thinking 模型对多 LLM 协作可能是错配（deep-research，18条3-vote验证）
触发问题：多 LLM 协作用 thinking 还是 non-thinking 效果好？6角度26源112claim→25验证→**18确认**。结论对本方法有实质影响：

1. **thinking/强 reasoning 让协作受益更少，不是更多**（high, 3-0）：MAD/debate 仅在「模型弱+任务难」时有用；模型越强 debate 增益越小，甚至系统性降准确率（sycophancy/误差放大）。源 arXiv:2505.22960 / 2502.08788 / 2509.05396。
2. **★最致命：强+高对齐模型→多样性塌缩**（high, 3-0, ACL2026 arXiv:2604.18005）：「更强、高度对齐的模型产生**递减的边际多样性**，尽管 per-sample 质量更高」；塌缩**主要来自交互结构非模型能力不足**。→ 直击命门：C 档卖点=proposer 间分歧/互补覆盖(auction Coverage 要有东西可选)，而三个强 thinking 模型可能给**高质量但高度相似**的提案→Coverage 无互补可挖→C 档优势消失。
3. **异质底座是对的杠杆**（high, arXiv:2602.03794）：2异质≈16同质;同质 N≈4 饱和,异质能到 N≈8。→ 我做 C 档(异质底座)方向**对**,赢因是**谱系不同非用了thinking**。
4. **Self-MoA 约束**（high, arXiv:2502.00674）：混不同模型非无脑更好,存在**质量vs多样性 Pareto**——只有混入模型**质量相近**时多样性才正收益;混弱模型拉低池子反害。→ 三底座要**质量相近**。
5. **诚实边界**：**无任何 source 直接做过 thinking vs instruct 头对头对照**;以上全是「能力/对齐→多样性」趋势**推断**非实测。

### 8.2 ★天然优势（可写 intro）：independent-propose ≠ debate
arXiv:2604.18005 说多样性塌缩「主要来自**交互结构**」。本方法**恰好规避**——N proposer **独立 dream、不共享中间结果**（方法设计§2.1）= interaction 之前的 independent ideation，正是文献建议的保多样性结构。**故本方法不是 debate（会塌多样性），是 independent-propose-then-auction（保多样性）**——相对 MAD/debate 的天然结构优势。

### 8.3 ★探索点：thinking-vs-instruct 做成消融（填文献空白）
8.1-5 是**有争议、无直接实测**的问题(文献 open question 第1条明列：「无 source 测过 thinking ensemble vs instruct ensemble under matched compute」)。**我有现成 N-Proposer 框架,跑两版 C 档(全Thinking / 全Instruct)即填补此空白=可能独立小贡献点**。
- 预期(据 8.1-2 推断):Instruct 版多样性更高→auction 互补信号更强→C 档优势更明显;Thinking 版质量高但趋同→auction 没东西可选。**若实测反转(thinking 更好)亦是发现**。
- 成本提醒:thinking 慢且贵(proposer 只出 NL 描述不需 reasoning,thinking token 纯浪费);此点也支持 default 用 instruct。

### 8.4 Proposer 底座选型（2026-06-30 DeepInfra 实测核实 + 用户拍板）
**核实**:之前拟选(Qwen3-235B/DeepSeek-V3.1/GLM-4.6)已**非最新梯队**;DeepInfra 当前开源旗舰=Qwen3.5-397B-A17B / DeepSeek-V4-Pro / GLM-5.2 / Kimi-K2.6 / MiniMax-M2.7。
- **用户拍板 C 档 = 三个最新开源旗舰(全中国谱系)**：`Qwen3.5-397B-A17B` + `DeepSeek-V4-Pro` + `GLM-5.2`。质量最高且相近(满足 8.1-4 Self-MoA 约束)。
- **A 档** = N×同一模型(persona 异质);**B 档** = 单 FM 基线(=DiCode)。
- ⚠️**limitation(写论文)**：三底座同为中国实验室谱系,训练数据/对齐范式可能有共性→异质性或不如跨国谱系。须论证「底座架构/训练谱系仍足够不同」或在消融里量化 proposer 间真实分歧(用 auction 的 Coverage 互补度当代理指标)。
- 复现锚点(防「换强模型作弊」指控)由 **B 档=DiCode 同款 Qwen3-235B-Thinking** 守住;C 档换最新旗舰是「异质增益」的额外卖点,非作弊(B<A<C 隔离归因)。
