# Baseline v2 实验记录：干净的小模型 DiCode Baseline（修复后重跑）

> 作者：Mason　｜　日期：2026-07-07　｜　分支：`skill-preflight-ued_Mason`　｜　代码 commit：`d58965e`
>
> **结论速览**：修复 v1 暴露的全部问题后重跑，得到一个**干净、健康、零致命故障**的 baseline：
> curriculum 全程增长（task_graph 4→64 节点，无冻结）、embeddings 全 200 OK、Ollama 满负载稳定、
> loss 收敛正常。**本 run 可作为 Phase 5 三组消融的对照基准（arm: baseline）。**
>
> 前置阅读：`baseline_实验记录.md`（v1，记录首次 run 的 curriculum 冻结 + embeddings 404 诊断）。
> 本文是**独立的第二次 run**，不是 v1 的续跑；v1 数据不参与最终对比。

---

## 1. Run 身份（对比时按此索引）

| 项 | 值 |
|---|---|
| wandb run id | **`32v02vi9`** |
| wandb run name | `singleLLM_baseline`（原名 DiCode-run-1783460542） |
| wandb project | `mechanism_UED / Skill_Preflight_UED` |
| output dir | `outputs/2026-07-07_214222_740288/`（含 rl_checkpoints + task_graph.graphml） |
| 本地日志 | `/workspace/baseline_run.log` |
| 代码 commit | `d58965e`（鲁棒性补丁 + flag-gated Phase 3 hooks，**hook 全关**） |
| arm | **baseline（纯 DiCode，无 skill graph、无 preflight）** |

**hook 关闭已验证**：`grep -a "SkillGraph\|Preflight" baseline_run.log` 无输出 → 跑的是纯 DiCode 路径。

---

## 2. 运行配置

### 2.1 硬件与环境

| 项 | 配置 |
|---|---|
| 平台 | RunPod，**NVIDIA A100-SXM4-80GB**（driver 580.126.20 / CUDA 13.0），网络卷 `/workspace` |
| Python 环境 | `/workspace/venv`（python 3.12，**建在网络卷上**，换 pod 后 `source` 即可复用） |
| JAX 生态 | **jax/jaxlib 0.6.2**、flax 0.10.7、optax 0.2.5、orbax-checkpoint 0.11.18、chex 0.1.89、distrax 0.1.5、craftax 1.4.5 |
| 生成模型 | Ollama 0.31.1 serving `qwen2.5-coder:14b`（**49/49 层全 GPU**，占 ~15G） |
| Embedding | Ollama serving `nomic-embed-text` |

> ⚠️ **jax 版本必须锁 0.6.2**。装成最新（0.10.2）会导致训练发散（loss → 1e15），见 §4.1。
> 版本来源 = v1 健康 run（`uqw2fb4y`）的 `wandb/.../files/requirements.txt` 快照。

### 2.2 环境变量（三组消融必须一致）

```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75      # 给 14B 留 ~20G，防止它被挤下 GPU
export OLLAMA_KEEP_ALIVE=-1                     # 14B 常驻，不被驱逐
export GENERATION_SERVER_URL=http://localhost:11434/v1   # ★ 默认是 :5000，必须覆盖
export EMBEDDING_SERVER_URL=http://localhost:11434/v1    # ★ 默认是 :5000，必须覆盖
```

Ollama 启动（**必须带模型路径，且在 JAX 启动前预热**）：
```bash
OLLAMA_MODELS=/workspace/ollama_models OLLAMA_KEEP_ALIVE=-1 nohup ollama serve > /workspace/ollama_server.log 2>&1 &
sleep 5
# 预热：趁 GPU 空着，让 14B 把 49 层全 offload 上 GPU
curl -s http://localhost:11434/api/generate -d '{"model":"qwen2.5-coder:14b","prompt":"hi","stream":false}' > /dev/null
grep -a "offloaded" /workspace/ollama_server.log | tail -1   # 期望 "offloaded 49/49 layers to GPU"
```

### 2.3 启动命令

```bash
python experiments/training/run_dicode.py \
  seed=1 \
  use_wandb=true wandb_entity=mechanism_UED wandb_project=Skill_Preflight_UED \
  training.total_timesteps=500000000 \
  gen_manager/llm@gen_manager.task_generator=local_qwen14b \
  gen_manager/llm@gen_manager.env_generator=local_qwen14b \
  gen_manager.embedding_model.model=nomic-embed-text \
  gen_manager.task_generator.max_tokens=8192 \
  gen_manager.env_generator.max_tokens=8192
```

**关键点：**
- `total_timesteps=5e8` **不能砍小**——LR schedule 按它归一化，砍到 3e6 会让 warmup/decay 压成尖峰，训练在 step ~105 必定发散（见 §4.2）。实际跑到目标 session 数后手动 `pkill` 停止。
- `max_tokens=8192`（原 32768）——32768 会让 KV cache 吃掉 6G 显存，挤得 14B 只能部分 offload。
- embedding model 覆盖为 `nomic-embed-text`（config 默认写的是 `Qwen/Qwen3-Embedding-0.6B`，Ollama 里不存在 → 404）。

---

## 3. 运行结果

### 3.1 规模

| 项 | 值 |
|---|---|
| 完成 session | **10（完整，含训练+评估+存图）** |
| global_update_step | **1200**（seed 200 + 10 session × 100） |
| **global_env_steps** | **157,286,400（≈1.573e8）** — wandb 实测；= 1200 update × 131072（131072 = num_envs × rollout_length = 2¹⁷） |
| 单 session 墙钟 | seed 1210s；后续 session 744–1120s |
| 14B 生成吞吐 | **69.8 tokens/s**（全 GPU offload 后） |

### 3.2 ★ Curriculum 全程增长（干净 baseline 的核心标志）

```
Saving task graph with  4 nodes   （seed）
                       16 nodes
                       28 nodes
                       40 nodes
                       52 nodes
                       64 nodes   （session 10 结束）
```
**单调递增，5 个设计轮次，每轮 +12 节点，零冻结。**
wandb 侧 `curriculum/num_tasks_compiled_cumulative` 与 `num_tasks_activated_cumulative` 同步阶梯上涨。

> 对照 v1：v1 从 Session 2 之后**永远冻在 16 节点**，后半程在同一批任务上反复训练 —— 这正是 v1 不能当 baseline 的原因。

> 注：`curriculum/num_tasks_sampled` 涨到 ~13 后走平是**正常的**——它是每 session 的采样容量（恒定值），不是累计量。判断 curriculum 死活只看 `*_cumulative` 曲线和 task_graph 节点数。

### 3.3 生成情况

- 设计轮次：session 1/3/5/7/9（DiCode 隔 session 设计一次，偶数 session 用现有 archive 训练）。
- 每轮 `12 designs created`；通过校验进入训练的新任务：session 3 → 9 个，session 5 → 8 个，session 7 → 8 个。
- wandb `num_tasks_compiled_cumulative` 到 session 8 累计 **25**。

### 3.4 训练健康度

- `train/value_loss` / `total_loss`：每个 session 一个驼峰后回落（新 session 重编译 + 新任务导致的正常波动），峰值在**个位数量级**（~1.5 / ~0.9），无发散。
- `train/entropy`：3.7 → ~0.9，平滑下降（curriculum 变难时略回升，正常）。
- `train/grad_norm`：全程 2–12，有界。
- checkpoint 正常保存。

---

## 4. 本次修复的问题清单（v1 → v2）

### 4.1 🔴 训练发散：jax 版本（重装环境引入）

- **现象**：seed training 在 step ~105 处 loss 爆到 1e15，entropy 塌陷。
- **根因**：换 pod 后重建环境，`pyproject.toml` 未 pin jax 版本，`pip install -e .` 拉到最新 **jax 0.10.2**；该版本与本代码库数值行为不兼容。
- **修复**：精确对齐 v1 健康 run（`uqw2fb4y`）的版本快照 → **jax 0.6.2 全家**。
- **诊断技巧**：wandb 每个 run 都存了当时的 `wandb/run-*/files/requirements.txt`，是复现环境的黄金参照。

### 4.2 🔴 训练发散：`total_timesteps` 砍太小（实验设计失误）

- **现象**：降级 jax 后仍在 step ~105 发散（幅度降到 1e7 / 6e4，但爆点不变）。
- **根因**：为了做"快速 smoke test"把 `total_timesteps` 从 5e8 砍到 3e6。**LR schedule 按 total_timesteps 归一化**，压缩 167 倍后 warmup/decay 变成尖峰，step ~105 附近的高 LR 直接把 GTrXL 训炸。
- **判据**：三次发散都精确在 step ~105 → 确定性事件触发，不是随机数值噪声（真正的精度/硬件发散爆点会飘）。
- **修复**：`total_timesteps` 保持 5e8，靠**手动停止**控制跑多久。
- **教训**：**smoke test 不能靠砍 `total_timesteps` 来缩短**。

### 4.3 🔴 embeddings 404（v1 遗留）

- **根因（两层）**：① `local_embed.yaml` 的 `base_url` 默认 `localhost:5000`，不是 Ollama 的 11434；② 它指定的 model 是 `Qwen/Qwen3-Embedding-0.6B`，**Ollama 里根本没这个模型**。
- **修复**：`ollama pull nomic-embed-text` + `export EMBEDDING_SERVER_URL=http://localhost:11434/v1` + 命令行覆盖 `gen_manager.embedding_model.model=nomic-embed-text`。
- **验证**：本 run 全程 `POST /v1/embeddings "HTTP/1.1 200 OK"`，零 404。
- 附注：`embedding_size` 字段（config 里写 1024）**不被代码校验**，换 nomic（768 维）无需改它。

### 4.4 🔴 Ollama 持续 500 / curriculum 冻结（v1 的致命病）

- **v1 根因**：显存被 JAX 挤爆 → 14B 被驱逐 / 后端崩 → 持续 500 → 生成全失败 → `No new tasks` → curriculum 冻死；且主线程死等 worker（`Waited: 11914s`），整台机器空转 3.3 小时。
- **修复（三管齐下）**：
  1. `OLLAMA_KEEP_ALIVE=-1`（14B 常驻不被驱逐）；
  2. `XLA_PYTHON_CLIENT_MEM_FRACTION=0.75`（80G 中留 ~20G 给 Ollama）；
  3. **代码补丁：worker 同步超时熔断**（`run_dicode.py`，`worker_sync_timeout_s` 默认 600s）——即使生成侧再挂，RL 也继续用现有 archive 训练，不再陪跑数小时。
- **验证**：本 run 全程零 500、零 `No new tasks were designed`；`nvidia-smi` 实测 JAX 68G + Ollama 15G 稳定共存。

### 4.5 🟠 14B 生成龟速（0.36 t/s → 69.8 t/s，提速 ~194×）

- **现象**：生成一个任务要几十分钟，`tg = 0.36 tokens/s`。
- **根因**：Ollama 日志 `offloaded 33/49 layers to GPU` —— **14B 有 15 层留在 CPU**。因为 Ollama 是在 JAX 已占 ~60G 显存**之后**加载的，只剩 11G，装不下完整 14B（还要 32768 的 KV cache 吃 6G）。
- **修复**：① `max_tokens` 32768 → 8192（KV cache 6G → 1.5G）；② `XLA_PYTHON_CLIENT_MEM_FRACTION` 0.85 → 0.75；③ **加载顺序**：先起 Ollama 并预热 14B（趁 GPU 空着抢占），**再**启动 JAX 训练。
- **验证**：`offloaded 49/49 layers to GPU`，`tg = 69.80 t/s`。

### 4.6 🟡 环境重建（换 pod 后 venv 丢失）

- 旧 venv 建在容器临时盘上，换 pod 即消失。**本次建在 `/workspace/venv`（网络卷）**，以后换 pod 只需 `source /workspace/venv/bin/activate`。
- 漏网依赖：`pyproject.toml` 未列 `gymnasium`，需手动 `pip install gymnasium`。

### 4.7 🟢 代码补丁（commit `d58965e`，已 push）

| 补丁 | 内容 | 本 run 的效果 |
|---|---|---|
| R1 fence-strip | `_extract_file` 剥掉 14B 泄漏的 ` ```python ` 围栏 | **本 run 零 SyntaxError**（v1 有围栏导致的语法错） |
| R2 sync-timeout | worker 超时则跳过本轮生成、继续训练 | 未触发（Ollama 全程稳定），但作为保险 |

---

## 5. 小模型生成质量观测（对方法动机的量化证据）

校验阶段（`gen_manager.check_compilation`）会**实际执行 14B 生成的代码**，跑不通的丢弃并打印 Traceback。
这些 Traceback 是**正常的拒绝行为，不是故障**（流程继续，curriculum 照常增长）。

全 run 错误类型计数：

| 错误类型 | 次数 | 典型样本 |
|---|---|---|
| `AttributeError` | **15** | `type object 'Achievement' has no attribute 'DESCEND'` —— 14B **幻觉出 Craftax 里不存在的成就枚举** |
| `TypeError` | 10 | 参数类型/数量用错 |
| `SyntaxError` | **0** | （v1 有；fence-strip 补丁修复） |

**规模参考**：5 个设计轮次 × 12 候选 = **60 个候选**；task_graph 节点 4→64（+60，每个候选都入图）；
`num_tasks_compiled_cumulative` 到 session 8 累计 25。

> ⚠️ **不要用 25/60 直接算通过率**：单个候选在重试中可能报多次错，错误计数 ≠ 失败候选数。
> 若论文需要精确通过率，应从日志按候选粒度重新统计（每个 `Validating 12 generated tasks...` 块内统计成功/失败个数）。

**对方法的意义**：`AttributeError` 占多数（15/25）说明小模型最典型的失败模式是**幻觉不存在的 API/枚举**。
这正是本项目两个组件要缓解的：skill-graph 引导生成方向（减少乱造）、preflight 过滤不可学习/无效任务（减少浪费）。
这是"小模型 code-level UED 需要补偿机制"的**第一手量化证据**。

---

## 6. 干净性验证（可复现的检查命令）

```bash
# ① curriculum 全程增长（应单调递增，无重复停滞）
grep -a "Saving task graph with" /workspace/baseline_run.log

# ② 零致命故障（应无输出）
grep -aiE "No new tasks were designed|CUDA error|out of memory|RESOURCE_EXHAUSTED|Killed|Segmentation fault" /workspace/baseline_run.log

# ③ hook 确实关闭（应无输出 → 纯 DiCode）
grep -a "SkillGraph\|Preflight" /workspace/baseline_run.log

# ④ embeddings 无 404（应全是 200 OK）
grep -a "v1/embeddings" /workspace/baseline_run.log | grep -c "200 OK"
grep -ac "404" /workspace/baseline_run.log
```

本 run 四项全部通过。

---

## 7. 局限与注意事项（诚实记录）

1. **单 seed（seed=1）**。目前是"受控单点对照"，三组消融同 seed 可看趋势；若要报统计显著性，需每组补 seed=2,3 跑均值±方差。
2. **方法组的对齐目标 = `global_env_steps 157,286,400`（update_step 1200 / session 10）**。+A 与 +A+B 都应跑到同一 session 数（10）后停止，使三组步数一致；否则性能差无法归因于方法本身。
3. **生成通过率未精确统计**（见 §5 警告）。
4. **深层 tier 仍为 0**：seed 评估中 `make_iron_pickaxe` / `collect_diamond` / 战斗魔法类均为 0 —— 这与 v1 一致，也正是本项目要攻的墙。**baseline 在深层 tier 崩到 0，是方法组要对比的起点，不是 bug。**
5. **环境不可变性**：本 run 的环境（jax 0.6.2 / 显存 0.75 / max_tokens 8192 / seed 1 / 同一 Ollama 实例）是三组消融的公共基线。**跑方法组前不得更改任何一项**，否则该组不可与本 baseline 比较。

---

## 8. 下一步

严格按 `methods_run_checklist.md` 执行：

1. 环境不变性检查（jax 0.6.2 / Ollama 两模型 / 14B 49/49 / 显存已释放）；
2. ~~确认 `MiniCraftaxTrain.default_params` 存在~~ → **✅ 已于 2026-07-07 在 pod 上确认为 `True`**
   （`Task(smart_absolute_path('src/minicraftax/tasks/seed_tasks/survive.py')).env` → `hasattr(..., 'default_params') == True`）。
   preflight 的 B-1 hook 可直接使用 `_raw.default_params`，**无需改代码**；
3. **+A**：`+skill_preflight.use_scheduler=true` → 盯 `[SkillGraph] frontier tier ...` → **跑到 session 10（= 157,286,400 steps）** → 记 run id；
4. **+A+B**：再加 `+skill_preflight.use_preflight=true` → 盯 `[Preflight] kept X/Y` → 同样跑到 session 10 → 记 run id；
5. 三组进同一 wandb workspace 对比：**主判据 = tier2-4 held-out 裸 SR**（能否把 baseline 崩到 0 的深层 tier 拉起来），辅以 mean_performance 与 curriculum 演化质量。

**运维要点**：停训练一律用 `pkill -9 -f run_dicode`（Ctrl-C 杀不干净，会残留占显存）；**Ollama 全程不要关**（三组共用同一个常驻 14B）。
