# DiCode 复现 —— 接入点评估报告（2026-06-30）

> 目标（本步）：**跑通 DiCode 开源复现 = B 档擂台 + 评估"我的方法"接入点**（不追 48.33）。
> 决策（用户拍板 2026-06-30）：执行环境=**Oscar 集群**；本步目标=**先跑通+评估接入点**；FM=**先看 repo 默认配置**。
> repo：`github.com/konstantinosmitsides/dreaming-in-code`（arXiv:2602.08194，已 clone 到 scratchpad 读代码）。
> 关联方法设计：[方法设计_v1.md](方法设计_v1.md)。

---

## 0. 结论先行（TL;DR）

1. **开源 repo 真实可用**：JAX + PPO-GTrXL + Hydra + WandB + Craftax，主入口 `experiments/training/run_dicode.py`。依赖干净（标准 JAX 栈 + craftax + openai + networkx），Oscar 装得起。
2. **我的方法接入点已精确锁定**：改动**只在 `TaskGenerator.evolve_mastered`（描述层=式8）**，`evolve_tasks` 的代码层（`generate_code_only`=式9）、archive、selection、PPO 训练**全部不动**。隔离极干净，N=1 去 Critic 天然 ≈ DiCode。
3. **架构是论文的两阶段（描述→代码）**：`evolve_tasks` 先调 `task_generator` LLM 出 `<docstring>` 描述，再调 `env_generator` LLM 出 `<code>` 代码。
4. **三个工程坑/红利**（决定接入工程量）：
   - 🟢 **`interestingness_critic` 已在 conf 声明但代码零引用、未初始化** → 我的 Critic 补盲 = 接入这个现成骨架（不是从零设计角色）。
   - 🟢 **provider 框架已支持多后端**（local/gemini/openai/openrouter），接 HF 只需补一处。
   - 🔴 **`llm.py` 的 `query()` 只接了 `local` 和 `gemini`**（openai/openrouter 只接进了 `get_embedding`，没接进 `query`）。**接 HF 走生成必须补 `query()` 的 provider 分支**——这是 FM 接入的第一个具体工程项。

---

## 1. 整体架构（4 步生成循环 → 代码落点）

主循环 `run_dicode.py:main`（Phase 3 第114行 `while global_env_steps < total_timesteps`）：
- 生成在**后台线程**（`dispatch_evolution_worker`），与 PPO 训练解耦 → 加 N-Proposer 不阻塞训练，红利。
- 调用链（已交叉验证）：

```
run_dicode.py:main (while loop, L114)
 └─ evolution_efficient.py:dispatch_evolution_worker (L95)
     └─ evolution_efficient.py:evolve_and_validate_tasks (L149)
         └─ gen_manager.evolve_tasks() (L172)                    ← 顶层编排，两阶段
             ├─ TaskGenerator.evolve_mastered() (gen_manager L1411)   ★②描述层=我的注入点
             │    └─ _query_and_parse_responses() (L770)
             │         └─ self.llm.query(sys, user_prompts) (L829)   ← 单 FM 调用，要换成 N 个
             │              └─ _parse_generation_response 提取 <docstring> (L944)
             └─ EnvGenerator.generate_code_only() (gen_manager L1442)  ③代码层=复用不改
                  └─ self.llm.query() (L1215) → _extract_file 提取 <code> (L1330)
         └─ env_generator.check_compilation() (evolution_efficient L188)  ④编译检查=复用
```

| DiCode 步骤 | 代码位置 | 我的改动 |
|---|---|---|
| ① 选 parent（learnability 式7） | `selection.py:select_tasks_for_evolution` (L137-180) | **不改** |
| ② 生成描述（式8） | `gen_manager.py:TaskGenerator.evolve_mastered` (L711)，prompt=`prompts/dicode/evolve.py` | **★核心改动**（N-Proposer + Critic + auction） |
| ③ 生成代码（式9） | `gen_manager.py:EnvGenerator.generate_code_only` (L1184)，prompt=`prompts/cl_/gen_env.py` | **不改**（对选出的 k 关各出代码） |
| ④ 编译检查 | `gen_manager.py:check_compilation` (L1252) | 复用 + 微增可解性检查（接旧 generator-solvability 经验） |
| 训练/reward/archive | `training.py` / `dicode_manager` config / `TaskArchive` | **不改**（继承 DiCode reward 结构，含 goal completion bonus） |

---

## 2. 我的方法的精确接入点（最小改动面）

**唯一需要动的类 = `TaskGenerator`（gen_manager.py L621-1012）。** 三处改动：

### 改动 A — N 个异质 Proposer（替换单 FM）
- 现状：`TaskGenerator.__init__` 接受**单个** `task_generator_llm`，存为 `self.llm`（L637）；`evolve_mastered` 第 770/829 行对所有 parent 发给这 `self.llm` 一个 FM。
- 改：`__init__` 接受 **LLM 列表** `self.llms = [proposer_1..N]`（异质底座/persona）；`evolve_mastered` 对每个 parent **对 N 个 Proposer 各发一遍**，得 N×parent 个候选 `<docstring>`。
- persona 多样性在 system prompt 这层注入（A 档）；异质底座在 conf 这层配不同 model（C 档）。

### 改动 B — auction 选 top-k（在 evolve_mastered 末尾）
- 在 `evolve_mastered` 返回前（L772 `_organize_data` 之前/之后），对 N×parent 候选算 bid，选 top-k。
- bid 三项（方法设计 §2.3，与 learnability 正交）：Coverage（互补覆盖）+ Endorsement（Proposer 互评）+ AmbitionGain（依赖链深度×target 缺口）。
- 选出的 top-k 才进 `tasks_to_generate_code_for`（evolve_tasks L1432）→ 送代码层。

### 改动 C — Critic 补盲（接入现成但未用的 interestingness_critic）
- `interestingness_critic` 在 `conf/gen_manager/default.yaml:6` 已声明，但 `GenManager.__init__`（L1343-1385）**没初始化它，代码零引用**。
- 改：`GenManager.__init__` 加一个 `critic = LLM(...self.config.interestingness_critic...)`，传入 TaskGenerator。
- Critic 做两件事（方法设计 §2.2）：(主动) 读 archive 覆盖摘要补 ambitious 提案 h_{N+1}；(护栏) 给每个候选打 [可解性/可学性/覆盖新颖性] 标签砍掉不可解。

**→ ablation 干净**：N=1 + 去 Critic + 去 auction = 原样 DiCode（B 档）。

---

## 3. FM 接入（"先看 repo 默认配置"的落地）

### provider 框架（`src/dicode/dreaming/llm.py`）
- `LLM._create_client`（L40）支持 4 provider：`local`（vLLM OpenAI 兼容）/ `gemini` / `openai` / `openrouter`。
- 🔴 **关键坑**：`LLM.query`（L221-227）只 dispatch `local` 和 `gemini`；`openrouter`/`openai` **没接进 query**（只接进 `get_embedding` L232）。
  → **接 HF 生成必须补 `query()` 的 provider 分支**（HF Inference 是 OpenAI 兼容，可走 `openrouter` 风格：自定 base_url + HF token）。工程量小（~10 行 + 一个 `_query_batch` 复用 `_query_local_gen`）。

### conf 里现成的 provider 配置（`conf/gen_manager/llm/`）
| 文件 | provider | 默认 model | 备注 |
|---|---|---|---|
| `local_gen.yaml` | local | **Qwen/Qwen3-235B-A22B-Thinking-2507-FP8** | DiCode 同款=复现锚点；需本地 vLLM server，235B 跑不动 |
| `openrouter.yaml` | openrouter | mistralai/mistral-large-2411 | OpenAI 兼容，base_url 可改成 HF Inference |
| `openai.yaml` | openai | o1-preview | 闭源，禁用（丢复现性） |
| `gemini.yaml` | gemini | gemini-2.5-pro | 闭源，禁用 |
| `local_embed.yaml` | local | Qwen/Qwen3-Embedding-0.6B | embedding，auction 的 Coverage/相似度要它 |

- **复现锚点**：默认就是 Qwen3-235B（DiCode 同款）。Oscar 上若能起 vLLM server 跑 235B 最忠实；否则走 HF Inference 同款 Qwen3-235B（方法设计已定 HF Inference Providers）。
- `.env` 需 6 个 key：`OPENAI/GEMINI/OPENROUTER_API_KEY`、`GENERATION/EMBEDDING_SERVER_URL`、`WANDB_API_KEY`。接 HF → 复用 `OPENROUTER_API_KEY` 槽位放 HF token + base_url 指 HF。

---

## 4. 关键配置锚点（对齐 SOTA 口径 + 我方法的 N/k）

`conf/training/default.yaml`（= 论文 Script B 的精确配置）：
- `env_name: Craftax-Symbolic-v1`，`num_envs: 1024`，`num_steps: 128`
- **`total_timesteps: 2_005_401_600`（≈2e9 = 5×4e8）** = SFL 论文同款预算量级 → 跑满到 48.33 **算力很大**（本步不追）。
- GTrXL：`num_layers: 2`、`num_heads: 8`、`embed_size: 256`、`window_mem: 128`、`gating: true`。

`conf/dicode_manager/default.yaml`：
- **`score_function: "learnability"`**（式7 parent 选择确认；另有 `pvl`/`max_mc` 选项 = 与我们旧 jaxnav PVL 工作同源）。
- **`original_task_proportion: 0.2`** = 方法设计 "20% target" 确认。
- `num_generation_tasks: 10`、`additional_num_parents: 2` → **每 cycle 产出 12 个描述**（各 ×1）。我的 N-Proposer 会变 12×N。
- `training_sample_size_n: 16`（每轮训练采样关数）、`active_task_capacity: 100`（buffer 容量）。
- `completion_bonus_*`：DiCode 给生成关加 goal completion bonus（B_t，自适应）→ 印证"沿用 DiCode reward 结构，不引入新 reward 项"。

`conf/gen_manager/default.yaml`：`num_generations: 5`、`num_examples: 6`（Explore 核实 `num_generations` 当前代码未用，仅 `num_examples` 用于选相似示例）。

---

## 5. 下一步（建议执行顺序）

1. **【环境】Oscar 上搭起来**：scp repo（先 cp 到 C 盘，[[oscar-file-access-sshfs]]）→ 建 conda/uv 环境装依赖 → 装 `jax[cuda12]` → 配 `.env`（WandB + FM key）。遵循 [[oscar-workflow-env-cheatsheet]]（登录节点禁跑计算）。
2. **【跑通短跑】** 先用便宜 FM（HF DeepSeek-V3.2 或小模型）+ 砍小 `total_timesteps` 做 go/no-go：确认能编译、能起训练、生成 worker 不崩、能落 WandB。**不追 48.33**，只验证管线通。
3. **【FM 接入】** 补 `llm.py:query()` 的 openrouter/openai 分支（红色坑），让 HF Inference 走通生成。
4. **【接入点冒烟】** 在 `evolve_mastered` 接 N=1 → 确认改造骨架不破坏原行为（= B 档自检）。
5. 之后才进方法设计 §6：接 N-Proposer + auction（A 档）→ Critic → C 档异质 + B<A<C 对照。

> ⚠️ 算力提醒：跑满 2e9 steps 到 48.33 是大活，**本步只做到"管线通 + 接入点清楚"**。完整复现 48.33 留到接入点改造完成、确认要正式对标时再上多 seed（遵循 [[seed3-coarse-screen-is-intentional]] / [[stage4-generator-10x-slow-anchor]] 算力纪律）。
