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

---

## 9. 中段 checkpoint 评测：C 档(auction) vs baseline，step 9900/10200/10300/10400（2026-07-01）★

### 9.1 评测设定（DiCode 官方口径，已核对论文原文）
- **口径**：论文 §4.1 + Fig2 + Appendix A.3——完整 Craftax，固定 **1024 个 held-out 程序生成世界**，报 **mean episode return**（DiCode 终值 48.33，最强 baseline 41.54）。student 观测的任务条件向量=**multi-hot 成就编码**（论文 §3 原文），即我们的 `conditioning_type=one_hot`（67 维）——两 run 与论文同口径，对标干净。
- **入口**：`experiments/training/eval_checkpoints.py`（新写，复用官方 `online_evaluation.run_session_evaluation`→`craftax_evaluation.main`；`num_envs=1024, num_steps=8192`）。**同 seed(0) → 两档看同一批 1024 held-out 世界 → 严格配对对比**。
- ⚠️坑修复：官方独立脚本 `online_evaluation.main` 写死 `embedding_size=1024`（embedding 模型维），在 one-hot 训练下会让 obs 维错(base+1024 vs ckpt base+67)导致 restore 崩；已改按 conditioning_type 判断 → one-hot 用 67。日志确认 `emb_size used for dummy env = 67`。
- step 语义：`'step'=global_update_step`（每 session +100，总 15300=2e9÷1024÷128）；评测点在训练 ~65-68%（中段，非终局）。
- 取 step 逻辑：两档 ckpt 目录**共同存在**且 ≤C档当时最新(10400)的交集。10100 因 baseline 滚动窗口(max10+keep300)挤出，剔除保对称。最终 4 点：9900/10200/10300/10400。

### 9.2 结果：mean_return（同 seed 同 held-out 世界）

| step  | baseline | C 档(auction) | 差 (C−base) |
|-------|----------|---------------|-------------|
| 9900  | 42.05    | 40.44         | −1.61       |
| 10200 | 43.94    | 41.33         | −2.61       |
| 10300 | 44.29    | 40.14         | −4.15       |
| 10400 | 44.58    | 40.94         | −3.64       |
| **均值** | **43.71** | **40.71**   | **−3.00**   |

### 9.3 结论：C 档在中段系统性落后 baseline ~3 分（诚实负结果）
1. **baseline 复现健康**：稳定上升 42.0→44.6，已超论文最强 baseline 41.54，向 48.33 逼近。
2. **C 档全程低 ~3 分且无上升趋势**（40~41 波动）；auction 此刻**无增益反拖累**。

### 9.4 差在哪（skill 分解，step 10400，base vs C）
| skill | base | C档 | 说明 |
|-------|------|-----|------|
| make_iron_armour   | 34.2 | 16.2 | 中后段铁装备链 |
| make_iron_sword    | 68.7 | 54.1 | 同上 |
| collect_iron       | 89.1 | 83.1 | |
| defeat_orc_mage    | 73.9 | 57.5 | 战斗中段 |
| enter_gnomish_mines| 17.7 | 4.6  | 深层探索 |

C 档在**中后段工具链(铁装备)+深层探索(gnomish mines)**明显更弱——正是需要「长程、有前置依赖课程」才练得出的能力。说明 **auction 造的课程没能把 student 推进到深层技能**。与已知隐患一致：竞争压力未回生成层(三必要条件缺第③)、by_proposer 曾坍缩 ambitious 垄断 → 课程质量没起来。

### 9.5 必须声明的 caveat（勿过度解读）
- **单 seed**、仅 4 个相邻中段 step、训练 ~68% 未到终局；趋势可能变。
- 差值(配对同 seed 同世界)可信度 > 均值绝对水平。
- 这是中段快照，非最终 48.33 对标结论；C 档要翻盘需课程质量在后段起来（当前无此迹象）。

### 9.6 操作记录（可复用）
- 评测走"暂停 baseline 腾 GPU"(配额 2 并发)；按《训练暂停处理规范.md》：scancel→备份 task_graph.graphml→清 orbax tmp→重交 eval；eval 完重交 baseline 无损续跑(11100 ckpt)。删最新半写 ckpt 续跑对性能无损(LR 随 opt_state 精确恢复)。
- 结果 JSON：`/oscar/scratch/jzhu223/dicode_outputs/eval_step10400/{base,carm}/eval_*_seed0.json`（含全 67 skill）。

---

## 10. C 档(auction) vs baseline(单FM DiCode) 全程性能分析（2026-07-02 重写，前版多处基于污染数据已作废）★★★

> 本章经多轮质疑逐层纠错重写。**唯一有效标准 = 官方 held-out mean_return（1024 未见 Craftax 世界，对标 48.33）**；跨 run 比较**必须用真实训练步对齐**（见 §10.2 污染）。之前用"各自训练关 sr / 同 label step / 结构多样性"等做的判断全部作废。数据源：wandb `evaluation/mean_return` + per-skill（官方 held-out）+ ckpt eval 交叉验证。

### 10.1 决定性污染：C 档 step 标签虚高 1900（欠费崩溃所致）
DeepInfra 初始只充 $5，欠费在早期反复打断两个 run。**baseline 打断时已存有 ckpt（disk 最早 step 300）→ 正确恢复**（`Restoring...step 3000` + `VERIFICATION opt count 96000`）。**C 档打断时首个 ckpt 还没落盘 → 每次 `No RL agent checkpoint, Starting from scratch` 重来，但 wandb step 计数照累加**。最终 job 3593784 从"标签 step 1900 的全新模型"起连续训练到 11300（`Global:` 从 2200 连续 +100，无跳变）。
- **精确偏移 = 1900 个 global update**（日志明写 resume 标签，非估算；seed training 那 300 步 1900→2200 也是真实训练）。
- **∴ C 档任意 label step N 的 ckpt = 真实训练 N−1900 步。** 评测用的 label 10400 = **真实 8500 步 = 总进度 55.6%**（总目标 15300）。C 档 wandb 最新真实步 9300 = 61%；baseline = 74%。**两者都远未到 DiCode 深层成就(tier3/4)的发力期(80-100%)。**
- **LR 未受污染**（曾误判，已更正）：LR schedule 的 count = optax 内部 step，只随真实更新累加，不含虚标签（ppo_tr.py:143）。scratch 新模型从 count=0、LR 满值 2e-4 正确退火。**C 档权重与 LR 同步正确，唯一错是 step 标签。** 详见《训练暂停处理规范.md》附录 A。

### 10.2 auction 机制本身健康（推翻"坍缩"旧担忧）
- **by_proposer 三 proposer 全程共存不坍缩**：p1 主导(~5.4/16)、p2(~2.8)、p0 最弱(~1.84，5/44 session 中标 0)——被边缘化的是 proposer_0 非 proposer_2。无 persona 被逐出。
- **bid 四项均衡**：全程 lrn 36% / end 33% / amb 20% / cov 10%（cov 结构性最弱，仅 1 次 WARN）。无任一项归零或 >70% 主导。
- **无崩溃/不可解**；task 编译失败率 ~30%（baseline ~29%，**两者相同**，是 DiCode「LLM 造 Craftax 关」固有特性，非 C 档劣势）。
- **proposer 一直在瞄深层**：中期 73% 的关瞄准 tier-3，auction 也给深层关更高 learnability（tier3=0.099 > tier1=0.059）。**ambitious 有发挥，生成层没失灵。**

### 10.3 全程 held-out 曲线（官方标准 + 真实步对齐）
两 run 起点对等（都从 scratch + seed training，wandb step200 held-out≈11-12，几乎相同）。真实步对齐后：
| 真实step | baseline | C档 | 形态 |
|---|---|---|---|
| 300-600 | 14-18 | 14-18 | 起步持平 |
| 700-2100 | 18-22 | 21-33 | **C 档飙升领先 +6~+12** |
| 2200-3900 | 27-37 | 34-37 | C 档仍领先，收窄 |
| ~3900(反超点) | — | — | baseline 追平 |
| 4600-8500 | 38-43 | 37-41 | baseline 小幅领先 −1~−3 |
| 8900 | 41.4 | 41.4 | 收敛同一区间 |

**形状 = "C 档陡升早饱和 vs baseline 匀速持续爬"**：C 档 real 2200 就冲到 35（baseline 要 real 3000+ 才到），然后 real 2200→8900 只涨 6 分（近躺平）；baseline 从 27 匀速爬到 42（涨 15 分不停）。**C 档不是变差，是先到天花板等 baseline。**

### 10.4 ★公平比较结论（同真实步，绝对成就数）
把 mean_return 换算成"平均每 episode 多解锁几个成就"（Σ达成率差/100）。**同真实步、公平区间(real 5000-9300)平均**：
| tier | baseline 领先(成就/episode) |
|---|---|
| tier-1 | +0.262 |
| tier-2 | +0.382 |
| tier-3 | **+0.096（几乎打平）** |
| **合计** | **+0.74 个成就/episode（≈mean_return +1.07）** |

- **平移前(同 label，不公平) baseline 领先 +1.65 成就/−3.6 分；平移后(公平)缩到 +0.74/−1.07** —— **约一半表观落后是"C 档少训 1900 步"的假象，不是机制差。**
- 剩余 +0.74 不是 trivial(约 1 个成就)但也不大，**集中在 tier-1/tier-2 基础技能（iron 链 make_iron_armour/sword/gnomish_mines）；tier-3 深层几乎打平(+0.096)**。
- **极不稳定**：逐点抖动 ±1 成就(real 9000 处 C 档反超 −1.05，9300 又 +1.49 = eval 噪声)；但差距未持续扩大(real5000→8000 稳定 +0.7~1.1)。
- tier-3 斜率两者持平(base +0.70/1k vs C +0.64/1k)，C 档 **tier-2 后段斜率 +1.40/1k 是 baseline(+0.66) 两倍**——C 档仍在快速补 tier-2，未跑偏。

### 10.5 综合判断（当前最可信，仍非定论）
1. **auction 无明显劣势**：公平对齐后仅落后 ~0.74 成就(基础技能)，**tier-3/tier-4 打平**；**early 阶段有确凿样本效率优势**(用一半真实步就到 baseline 双倍步数才到的 tier-2 水平)。
2. **共同天花板 = tier-2 ~70% + tier-3 ~12%**，两方法都突破不了，是 RL student 能力墙/Craftax 深层难度，非 auction 过错。反馈到生成层(促分化)解决不了此墙(proposer 已在瞄深层、student 学不动)。
3. **评在最不利时点**：C 档仅真实 55-61% 进度，tier-2 还在陡升、tier-3/4(DiCode 真正战场)两者都才 12%/0% 未发力。**用当前数据判 auction 好坏为时过早；"C 档押注深层、后期赢 tier-3"是合理但未兑现的假设(tier-3 斜率目前持平，无反超迹象)。**

### 10.6 待跟进
- **唯一能定论深层战场胜负的方式**：两 run 都跑到接近终局(15300)，同真实步比 tier-3/tier-4。C 档现 61%。
- 该 C 档 run step 标签永久污染(LR 正确但标签错位)，只能靠"真实步=label−1900"平移分析；若要发论文级干净数据，需从头重跑不中断的 C 档(额度已充足)。
- 数据脚本：wandb 取数见记忆 [[wandb-fetch-eval-curves-local]]；曲线/tier JSON 在 scratchpad(`_wandb_curves.json`/`_skill_tiers.json`)；graphml 分析(目标 tier/学习率)脚本在 Oscar。

### 10.7 ★被反超的最终根因：课程目标漂移到 tier-3，铁器链因灾难性遗忘倒退（2026-07-02）
用 graphml（C 档造关的目标 tier 随 session）+ wandb（铁器链成就 held-out 斜率）双证据定位到根因。

**(a) C 档一直造铁器链，但目标 tier 中期漂移到 tier-3**（graphml，IRON-chain 关占比 by session band）：
| session 段 | 铁器链关占比 | 目标 tier 分布 |
|---|---|---|
| s1-10 (early领先) | 66% | **t2=72%**（纯 tier-2 目标）|
| s11-25 (追平区) | 87% | t2=35%, **t3=65%** |
| s26-45 (被反超) | 92% | t2=24%, **t3=72%** |
| s46+ (最新) | 85% | t2=25%, t3=60% |
- early 用**纯 tier-2 关**冲高 → tier-2 held-out 早早领先（=early +6~12 来源）。
- 中期 ambitious 倾向让**目标升到 tier-3**（gnomish mines/orc），铁器链从"主菜"沦为深层关的**前置配料**。reasoning 里明确写 "Deep Bottleneck Hypothesis" 攻 gnomish/orc。

**(b) 铁器链核心成就 held-out 实际在倒退**（wandb，斜率 real 5000→9300/1k）：
| 成就 | baseline 斜率 | C档 斜率 |
|---|---|---|
| make_iron_sword | +3.62 | +2.14 |
| make_iron_armour | +2.71 | +0.41 |
| make_iron_pickaxe | +0.50 | **−0.68（倒退）** |
| collect_iron | +1.05 | **−0.12（倒退）** |
- baseline 四个全稳升；C 档停滞或**倒退**。（注：之前"tier-2 均值斜率 +1.40 在追"是被 tier-2 里饱和浅成就拉高的假象，拆到铁器链核心成就是退的。）

**根因机制**：C 档中后期 72% 关瞄 tier-3，student 在深层关**大量失败（sr≈0，连进不去 mines/打不过 orc）**→ 走不到"打铁"那步→独立干净的铁器链关供给不足→**student 对铁器链既没练透又因长期不在纯铁器情境训练而遗忘/退化（RL 灾难性遗忘）**。baseline 单 FM 课程更保守、持续供给独立 tier-2 关，把铁器链稳稳练上去，于是反超。

**定性结论**：被反超**不在 auction 选择机制**（机制全程健康），而在 **proposer 课程"难度锚点太激进"**——ambitious 倾向让课程过早把目标升到 student 还够不着的 tier-3，导致 tier-2 未练透就被遗忘。这**正面印证** UED 核心命题：课程必须锚在 student 的 learnable band，跳太快有害。

**注（2026-07-02 用户决定）**：当前 C 档 run（auctionC_s0_v1）因 step 标签污染 + 上述课程缺陷，将被**弃用重跑**（额度已充足），用改进后的方法从头跑干净 run。本章所有 C 档数据仅作诊断依据，不作最终性能结论。改进方向见 §11。

### 10.8 ★★最终根因（公平 session 对齐 + baseline 对照，收敛结论，2026-07-02）
前面 §10.7 及多轮探索中的若干中间假设经公平对照后**修正/推翻**，最终收敛如下。

**方法论修正**：跨 run 公平对齐**必须用 session 号**（每 session 恒 100 update = 真实训练进度代理），**不能用 label step**（C 档 label 因崩溃继承虚标签 1900 且中间有跳变，session 8→9 从 step1000 跳到 3000，故"真实=label−1900"只是粗近似）。之前按 label/四分位混算的分箱不可比。

**公平对照（baseline vs C 档，相同 session band）**：
| session | tier3占比 base/C | 学不会率 base/C | tier2占比 base/C |
|---|---|---|---|
| A 早期(s1-10) | 5% / 16% | 4% / 2% | 65% / 72% |
| B(s11-20) | 16% / 62% | 18% / 20% | **83% / 38%** |
| C(s21-35) | 48% / 73% | 25% / 24% | 51% / 25% |
| D(s36-50) | 61% / 62% | **44% / 32%** | 38% / 25% |
| E(s51+) | 60% / 61% | 28% / 27% | 39% / 24% |

**被推翻的中间假设（都不成立）**：
1. ❌"可学关被 ambition/breadth 的 bid 压制落选"——日志中标 bid 明细显示 **feasible(proposer_2) 中标率不降反升(前期20%→反超区34%)，且 lrn 是最强中标驱动项(反超区 59% 中标关由 lrn 主导)**。可学关赢得了 auction，没被压落选。
2. ❌"ambitious 灌垃圾深层关/课程质量差"——C 档学不会率**不比 baseline 高**(B:20vs18, C:24vs25)，D 段甚至更低(32vs44)。C 档 tier3 关质量不差。
3. ❌"C 档灾难性遗忘/课程失控是独有病"——baseline 同样 tier3 冲到 60%、学不会率 D 段 44%(比 C 档还高)。深层难关两个方法都大量造。

**成立的根因 = tier 份额配比过早上移**：C 档在 B/C 段(session 11-35)tier3 占比显著高于 baseline(62% vs 16%、73% vs 48%)，**tier2 巩固份额被挤**(B 段 tier2 仅 38% vs baseline 83%)。baseline B 段**死磕 tier2(83%)**把它练透 → tier2 held-out 持续上涨 → 最终反超。**不是学不会、不是可学关落选，是 tier2(held-out 主力得分层)的巩固份额被 tier3 过早挤占。**

**机制源头**：auction 的 **Coverage(深层权重 tier1-4 = 1/2/4/8×) + AmbitionGain(gap×depth 偏深)** 两项(占一半 bid 权重)系统性把 archive 的 tier 重心比单 FM 更早上移。即使每个关单独合格、feasible 也能中标，**群体配比**偏向深层。

**改进方向（有据）**：不是调单个 bid 权重(feasible 已能中标)，而是**控制 archive 的 tier 份额配比**——在 student 掌握 tier2 前维持足够 tier2 份额(如 baseline B 段)。候选：per-tier 配额 / 降低 Coverage 深层权重倾斜 / 按 student 当前能力动态设定可造最深 tier。详见 §11 讨论。

**诚实边界**：held-out 真实差距本就小(公平对齐仅 +0.74 成就/episode，两者终值都收敛 ~41)；上述份额差是**当前数据下最站得住的差异解释**，但 C 档仅 55-61% 进度、tier3/4 未发力，最终胜负仍需干净重跑验证。

### 10.9 ★★★根因精确化 + §10.8 假设#2 修正（2026-07-02，按 proposer 拆中标关）
用户点破：§10.8 排除"ambitious 灌垃圾关"用错了标准——**"垃圾关"应按 student 当前相对难度定义**：对 B 阶段(session 11-20)的 student，tier3 关就是垃圾关(学不会、浪费训练份额)，与其"绝对质量"无关。据此拆 B 阶段中标关的 proposer 来源(日志 winner bid 明细的 `<proposer_X>` + amb_raw 作深浅代理)：

**B 阶段(s11-20) 50 个中标关按 proposer**：
| proposer | 中标数(占比) | 高amb(深层)关 | lrn_raw均 |
|---|---|---|---|
| **ambitious** | **30 (60%)** | **19 个深层** | 0.66 |
| feasible | 13 (26%) | 1 (几乎全浅可学) | 0.88 |
| breadth | 7 (14%) | 2 | 0.89 |

**证据**：
- **B 段深层(tier3)关几乎 100% 来自 ambitious**(19/30，feasible 仅 1、breadth 2)。
- **ambitious 独占 60% 中标名额**(top-10 里约 6 个)，把 feasible 的 tier2 可学关(amb 低、lrn_raw 0.88、10/13 高可学)挤到仅 26% 份额。
- breadth 中标从 A 段 15 萎缩到 B 段 7，进一步让 ambitious 独大。
- C 反超区(s21-35)同样：ambitious 43(54%) vs feasible 27(34%)，深层关 29/43 来自 ambitious。

**§10.8 假设#2 修正**：❌"ambitious 灌垃圾关不成立(学不会率不比 baseline 高)" → ✅ **成立**。按 student 相对难度，ambitious 在 B/C 段灌入深层关(对当前 student=垃圾关)、独占多数中标名额，正是 tier2 巩固份额被挤、held-out 被反超的直接机制。之前用"绝对学不会率"排除是错的标准。(注：feasible 的可学关本身没被压落选——它 lrn 主导能中标——但 top-10 名额被 ambitious 占去大半，feasible 拿不到更多份额。两者不矛盾：可学关"能中标"≠"中标够多"。)

**精确根因**：不是笼统"tier 份额上移"，而是 **ambitious proposer 中期靠 amb+cov bid 独占 ~60% 中标名额、灌入对当前 student 过深的关，挤掉 feasible 的可学 tier2 关份额**。auction top-k 是固定 10 名额的零和竞争，ambitious 赢得多→feasible 必然少。

**改进方向(精确化)**：核心是**限制单 persona(尤其 ambitious)的中标份额** 或 **按 student 当前能力动态压制"超前"的关**。候选：
- **per-proposer 中标配额**(如每 persona 最多 k/3)——保证 feasible 的可学关有保底份额；
- **动态难度闸**：按 student 当前 tier 掌握度，压制超出其能力 1 档以上的关的 bid(把"是否超前"作为 gate 而非让 amb 无上限加分)；
- **让 amb bid 随 student 能力自适应**：student 未掌握 tier2 时，tier3 的 amb 增益应被打折(gap 大但"够不着"应扣分，而非现在的 gap 大就高分)。
这与项目"竞争驱动异质 teacher 分化"卖点相容：配额/难度闸让三 persona 沿难度轴分工(feasible 守巩固、ambitious 探边界)，而非 ambitious 单方通吃。
