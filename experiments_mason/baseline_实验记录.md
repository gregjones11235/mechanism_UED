# Baseline 实验记录：小模型 DiCode Baseline（首次运行 + 问题诊断）

> 作者：Mason　｜　日期：2026-07-07　｜　分支：`skill-preflight-ued_Mason`
> 结论速览：首次 baseline run 跑通了环境与训练管线，但**中途 curriculum 冻结 + 全程 embeddings 404**，
> 导致该 run 不能作为干净的对照 baseline，需修复后重跑。本文记录运行配置、观测、诊断与修复方向。

---

## 1. 目的

在小模型（本地 14B，替换 DiCode 默认的 235B）上跑一次纯 DiCode baseline（不加 skill graph、不加 preflight），
作为后续方法（skill graph + preflight）三组消融的对照基准。

---

## 2. 运行配置

| 项 | 配置 |
|---|---|
| 平台 | RunPod，NVIDIA H100 NVL 94GB（driver 580.159.04 / CUDA 13.0） |
| 生成模型 | 本地 Ollama serving `qwen2.5-coder:14b`（env_generator + task_generator 均覆盖为 14B） |
| 训练 | DiCode 原版 PPO-GTrXL，`training.total_timesteps=5e8`（实际手动/卡死中止于 ~7.86e7） |
| wandb | entity `mechanism_UED`，project `Skill_Preflight_UED`，run `DiCode-run-1783387797` |

启动命令（要点）：
```bash
uv run experiments/training/run_dicode.py \
  seed=1 use_wandb=true \
  wandb_entity=mechanism_UED wandb_project=Skill_Preflight_UED \
  training.total_timesteps=500000000 \
  gen_manager/llm@gen_manager.env_generator=local_qwen14b \
  gen_manager/llm@gen_manager.task_generator=local_qwen14b
```

> 关键 Hydra 覆盖点：生成模型的 key 是 `gen_manager.env_generator` / `gen_manager.task_generator`
> （不是 `env_coder_llm`）。GPU JAX 需 `LD_LIBRARY_PATH` 指向 venv 内 `nvidia/*/lib`（driver 580 下）。

---

## 3. 运行观测

- 训练管线跑通：wandb 正常 syncing，进入 Seed Training，14B 成功生成可编译环境
  （`Generated full 9-level base world`），checkpoint 正常保存（ckpt 到 update_step 600）。
- 训练健康指标正常：`train/value_loss`、`train/total_loss` 先升后收敛，`train/entropy` 3.5→~0.7，
  梯度范数无爆炸。
- 深层成就开始有信号：`evaluation/skill_make_iron_sword` 在 ~6.5e7 步后从 0 爬到 ~0.08；
  但 iron_pickaxe / iron_armour / diamond / defeat_gnome 等更深成就仍为 0。
- **进度**：约 5h40m 墙钟内，只推进到 `session=4`、`global_env_steps≈7.86e7`、`update_step=600`。

---

## 4. 问题诊断（本 run 不能作为干净 baseline 的原因）

### 4.1 ⚠️ 中途 step 冻结（后半程空转）

- `global_env_steps` 停在 7.86e7、`update_step` 停在 600 后，**其后约 3.3 小时一步未增**。
- 即前 ~48 分钟（Seed + Session 1–4）是真实训练（7.86e7 步为真），
  之后 ~3.3 小时为**卡死空转**（时间在流，steps 不增），并非"训练慢"。
- 换算佐证：`update_step=600` 对应 `7.86e7 ÷ 600 ≈ 131072 = num_envs × rollout_length`（合理）；
  Seed=200 updates，每 session +100（S1→300…S4→600），S4 后冻结。

### 4.2 ⚠️ curriculum 冻结（最致命）

- 日志 `Saving task graph with N nodes` 从 Session 2 之后一直是 **16 nodes 不变**（S2→S4 未增长）。
- 说明 archive 只在 Session 2 扩张过一次（4→16），之后**新任务不再进入** —— 生成后期失败。
- 后果：后半程（S3–S4）agent 一直在**同一批 16 个任务上反复训练**。
- DiCode/UED baseline 的核心价值在于 **curriculum 随训练不断演化**；此 run 后半段是"静态薄 curriculum"，
  **不能代表一个正常的 DiCode baseline**。

### 4.3 全程 embeddings 404

- 从 Session 1 起持续报 embeddings 404（embedding 服务/端点未正确配置）。
- 影响：依赖 embedding 的采样/多样性路径可能退化为 rank-based fallback，进一步影响 curriculum 演化。

### 4.4 环境层面的已知噪音（无害）

- Ollama 需重装 + `OLLAMA_CONTEXT_LENGTH=32768`（DiCode 生成 prompt ~1.8e4 tokens）。
- 若不设 `LD_LIBRARY_PATH`，JAX 会 fallback CPU（`device: TFRT_CPU_0`）——本 run 已修正为 `cuda:0`。
- `.env` 中 `WANDB_API_KEY` 占位值（6 字符）会覆盖 netrc 导致认证失败，需填真实 key 或删除该行。

---

## 5. 修复方向（重跑干净 baseline 前）

1. **embeddings 404**：定位 embedding provider/端点配置（`EMBEDDING_SERVER_URL` / embedding_model），
   确认本地/远端 embedding 服务可用；这是 curriculum 依赖的一环。
2. **curriculum 冻结 / 生成后期失败**：排查为何 Session 2 后 archive 停在 16 节点、新任务不再进入
   （生成 worker 是否静默失败、check_compilation 后是否未激活、select_tasks_for_evolution 是否返回空）。
3. **稳定性**：确认 Ollama 在长跑中不掉线（生成失败会连锁导致 curriculum 停滞）。
4. 修复后重跑：目标 `global_env_steps≈8e7`（干净、curriculum 全程增长），单 seed 看趋势即可。

> 判断"干净"的两个监控指标：
> - `global_env_steps`（切时间轴）：中途不应出现水平段（水平=卡死）。
> - `task_graph 节点数` vs `global_env_steps`：步数增长时节点数应持续增长（不增=生成又停了）。

---

## 6. 对方法工作的影响

- **不影响方法代码**：skill_scheduler / preflight 及其单测独立于此 run，均已完成。
- **对方法动机的印证**：小模型在单卡上的生成吞吐 + 稳定性是真实瓶颈（生成停滞会拖垮 curriculum），
  这正是 skill graph（聚焦生成方向）+ preflight（过滤无效生成、减少浪费）要缓解的问题。
- **下一步**：修复 embeddings + curriculum 冻结 → 重跑干净 baseline → 应用 Phase 3 hook →
  在相同 steps（~8e7）、相同 seed 下跑三组消融（纯 baseline / +skill graph / +preflight）对比。
