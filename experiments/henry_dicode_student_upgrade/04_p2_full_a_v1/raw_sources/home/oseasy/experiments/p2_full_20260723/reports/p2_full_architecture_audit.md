# Full P2 — 只读架构审计报告

- 日期: 2026-07-23
- 范围: Henry 健康基座 GTrXL-PPO（只读）+ P2-v1-lite 现有模块（只读）+ ckpt17500 参数树（只读 dump）
- 约束遵守: 未改任何 frozen 源码；ckpt17500/session175/P2-v0/P2-v1-lite/D052 全只读；GPU0 UUID 绑定；无子 Agent。
- 证据: `p2_full_20260723/evidence/architecture_inspection.json`

---

## 1. Henry GTrXL 网络结构（健康 Student 的真实长上下文策略）

健康基座 = `dicode.network.ActorCriticTransformer`（Gated Transformer-XL，GTrXL），由
`dicode/transformer/transformerXL.py::Transformer` + `rel_multi_head.py::RelMultiHeadDotProductAttention` 构成。

**前向签名**：`network(memories, obs, mask) -> (pi, value, memory_out)`
- `memories`: `[batch, window_mem, num_layers, embed_size]` —— Transformer-XL 记忆库（长历史）。
- `obs`: 当前观测（eval 单步）或观测窗口（train，`[batch, window_grad, obs_dim]`）。
- `mask`: `[batch, num_heads, 1, window_mem+1]`（eval）/ `[B, H, window_grad, window_mem+window_grad]`（train，因果+padding）。
- transformer 对 `[memories ; obs]` 做相对位置注意力 → 输出 `x`；**actor 头与 critic 头都消费 `x`**：
  - actor: `x → actor_ln1 → act → actor_ln2 → act → actor_out(43) → Categorical`
  - critic: `x → critic_ln1 → act → critic_ln2 → act → critic_out(1)`

**三种 forward**：
- `__call__` / `model_forward_eval`: 单步 rollout，返回 `(pi, value, memory_out)`（memory_out = 本步新隐状态）。
- `model_forward_train`: 观测窗口，返回 `(pi, value)`（不返回 memory）。

**记忆维护（rollout 循环，ppo_tr.py）**：`memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(memories_out)`
—— 长度 `window_mem` 的滚动记忆，最新在尾部；done 时清零。

**关键结论（决定 Full P2 设计）**：
- **健康 Student 本身就是真正的长上下文策略**：actor 在 eval 真实 attend `window_mem+1=129` 步、在 train 真实 attend `window_mem+window_grad=128+64=192` 步。
- 长上下文不需要新增模块；它就是 GTrXL 的 `window_mem=128` 记忆库。
- "episode 远大于 128" 的瓶颈不是"actor 看不到长历史"，而是**稀有成功轨迹无法反复利用 + Kobold 奖励无法向早期搜索传播**——这靠**整轨迹 off-policy 重放 + 多步回报信用分配 + hindsight** 解决，而非把记忆从 128 扩到更大。

## 2. 检查点参数树（ckpt17500，只读 dump）

- **总参数 = 4,906,028；叶子 = 80；params SHA256 = `5dfe67dda87ef15aa716276730de7685d73ac4096761abbc405fc7198cc6cd61`**
- 结构（顶层 `params/`）：
  - `transformer/encoder/{kernel[8335,256], bias[256]}` —— obs 编码器（2,133,760 参数，最大叶子；obs_dim 8335→256）。
  - `transformer/tf_layers_{0,1}/`（2 层 GTrXL），每层：
    - `attention1/{query,key,value,out}/{kernel[256,8,32],bias[8,32]}` + `pos_embed_mat/kernel[256,8,32]` + `r_r_bias[8,32]` + `r_w_bias[8,32]`（Transformer-XL 相对位置注意力，8 头×32 维）。
    - `dense1/{kernel[256,256],bias}`、`dense2/{kernel,bias}`（FFN，GELU）。
    - `gate1/{Dense_0..5/kernel[256,256], gating_bias[256]}`、`gate2/{...}`（GTrXL 门控，每门 6 Dense）。
    - `ln1/{scale,bias}`、`ln2/{scale,bias}`（pre-norm LayerNorm）。
  - `actor_ln1/ln2/{kernel[256,256],bias[256]}`、`actor_out/{kernel[256,43],bias[43]}` —— actor 头。
  - `critic_ln1/ln2/{kernel[256,256],bias}`、`critic_out/{kernel[256,1],bias[1]}` —— critic 头。
- **完整 80 叶子清单见 `evidence/architecture_inspection.json`。**

**零观测参考策略（兼容初始化 bit-exact 基准）**：固定 `obs=0, memory=0, mask=1`（batch=2）下，
ckpt17500 → `value=3.5761`、`entropy=0.9791`、top5 actions=[12,18,4,11,21] probs=[0.718,0.144,0.056,0.039,0.026]。
Full P2 兼容初始化后必须在同一固定输入下复现该 logits（residual=0 等价 → bit-exact）。

## 3. 动作/观测/目标条件接口

- **action_space.n = 43**（离散，Categorical）。
- **obs_dim = 8335**（ symbolic obs + 尾部拼接的 task/goal 向量）。
- **task/goal 向量 = 67 维**（`get_achievement_multi_hot`，Craftax 67 个 achievement；`embedding_size EMB=67`）。
- **目标条件方式**：multi-task wrapper 把 67 维 task 向量**拼到 obs 末尾**（`CraftaxAugObsTrain.get_obs = concat([symbolic_obs, task_vector])`）。
  → hindsight 重标 = 替换 obs 尾部 67 维为新目标 multi-hot（见 §7）。**注意：goal 向量 67 维 ≠ transformer encoder_size 256 维，勿混淆。**

## 4. Rollout 记忆接口（ppo_tr.py / p2_v1_core.collect_rollout）

- 每步：更新 `mem_idx`/`mem_mask`（done 清零、one-hot OR 进 mask）→ `model_forward_eval(memories, obs, mask)` 得 `(pi, value, mem_out)` → 采样 action、记 behavior `log_prob` → `memories = roll(-1).at[-1].set(mem_out)` → env.step。
- `Transition`（Henry）= `(done, action, value, reward, log_prob, memories_mask, memories_indices, obs, info)`。
- P2-v1 `collect_rollout` 对齐修正后：`obs[t]=决策obs`、`memory[t]=step前memory`、`next_obs[t]=step后obs`，并按 env slot 持久化 pending episode（跨 rollout 收集整 episode）。

## 5. Replay transition schema（P2-v1-lite，可复用）

`trajectory_replay.py::Trajectory` / `ReplaySample` 字段：
- `observations[T,8335]`、`actions[T]`、`rewards[T]`、`dones[T]`、`values[T]`、**`log_probs[T]`（behavior μ，已保存）**、
  `initial_memory[window_mem,layers,embed]`、`achievements[T,67]`、`target_achievements[67]`、
  **`memory_sequence[T,window_mem,layers,embed]`（逐step记忆，磁盘杀手，见§10）**、`next_observations[T,8335]`、
  `next_value`(bootstrap)、`episode_done`、**`collected_update_count`（policy version，已保存）**、`trajectory_id`。
- Buffer：固定容量环（默认 256），确定性 `RandomState(seed)` 采样，**序列严格 >128 步**，仅接受 `done=True` 的完整 episode（Gate3/4）；correct memory slice + bootstrap（done→0，否则 values[end]）；`state_dict/from_state_dict`（含 rng_state）；`hash_digest` 证据。
- **behavior logprob 与 policy_version 均完整保存** → 满足 off-policy IS 校正与 policy-lag 门的原料要求。

## 6. 当前 long_context_learner 是否真正参与 Actor forward？—— 否（critic-only）

`long_context_learner.py::LongContextLearner`：
- **on-policy 主更新**（`ppo_update`/`native_ppo_loss`）：标准 clipped PPO，窗口化 transformer train forward（每窗口前缀 window_mem 记忆），**正常更新 actor 头 + 共享 trunk + critic 头**。这是健康的主路径。
- **replay 辅助更新**（`_replay_aux_loss`/`update`）：**仅 value-only 半梯度 lambda-return 损失**；actor 项默认 `replay_actor_update=False` 关闭；即便开启也是"把普通 off-policy 数据直接套 PPO ratio"（指令明令禁止）。
- **方案2 critic-only 隔离**：`_is_replay_updatable_path`（路径含 "critic" 才可更新）+ `optax.masked` 专用优化器 + `_select`（非 critic 叶子保持原数组 bit-exact）→ **replay 只改 critic 头，actor 头 + 共享 trunk bit-exact 不变**。
- 结论：**replay 路径目前完全不更新 actor / 不更新长上下文 trunk**。这正是"P2-v1-lite ≠ Full P2"的根本。Full P2 必须反转此隔离。

## 7. 当前 hindsight 修改了哪些字段（可复用）

`hindsight.py`（Gate5/6 不弱化）：
- `apply_goal_conditioning`：替换 obs/next_obs **尾部 67 维**为新目标 multi-hot → 真实改变 goal-conditioned 网络输入。
- `recompute_reward_for_goal`：`r_g[t]=max(ach[t,g]-ach[t-1,g],0)` → 新目标下重算 reward（首次达成 +1）→ 改变 return/TD target/loss。
- `relabel_sample`/`relabel_trajectory`：改 `observations`(goal条件)、`rewards`、`target_achievements`；**透传** behavior log_probs/values/memory/bootstrap/`collected_update_count`（供 IS/policy-lag）。
- Gate5（正向）：目标只取自字面达成；Gate6（负向）：拒绝伪造/未达目标。
- 结论：hindsight 已真实改 goal+reward+success(target)；下游 V-trace 用新 reward 重算 TD target → actor/value loss 与梯度全变（满足 Gate4）。

## 8. Optimizer 与 checkpoint 结构

- **Optimizer**（Henry/P2-v1）：`optax.chain(clip_by_global_norm(max_grad_norm=1.0), adam(lr=2e-4, eps=1e-5))`；P2 Stage4 `anneal_lr=False`（**常数 LR=2e-4**）。P2-v1 另有 critic-only 辅助 `optax.masked` 优化器（Full P2 移除）。
- **Checkpoint**（`checkpointing.py`）：orbax `PyTreeCheckpointer` 存 TrainState(params+opt_state，bit-exact) + `replay_meta.pkl`(pickle: replay state + rng_key + action_rng_state + global_step + update_count + pending_state + collector_state + aux_opt_state) + `manifest.json`；`restore_full_checkpoint` / `checkpoint_inventory` / `compatible_weight_restore_report`（记录 restored/skipped/newly-initialized 叶子）。
- **GPU-saved orbax 不能在 CPU 恢复**（restore/verify 必须 GPU0）。

## 9. 可复用模块清单（P2-v1-lite → Full P2）

| 模块 | 复用方式 |
|---|---|
| `RolloutBatch` 数据结构 | 直接复用（on-policy 主路径） |
| `_gae` / `reference_gae`（显式 terminal bootstrap） | 直接复用（on-policy GAE + 测试参考） |
| `_causal_roll` / `_window_train_mask` / `_replay_window_mask` | **核心复用**：长序列窗口掩码（>128 序列喂 GTrXL train forward） |
| `windowize_batch` | 复用（on-policy 主路径窗口化） |
| `native_ppo_loss` / `ppo_update` | 复用（on-policy PPO 主更新，即 Control 算法） |
| `compute_on_policy_gae` | 复用（bootstrap V(next)，batch==1 tile 修补） |
| `_replay_forward`（mem_timeline 长上下文前向） | **改造复用**：改为 re-burn-in 重建记忆（见§10） |
| off-policy 诊断（behavior_lp/ratio/ESS/policy_lag） | 复用（V-trace 需 ρ=π/μ，ESS/lag 门） |
| `hindsight.py` 全部 | 直接复用（Gate4 重标） |
| `trajectory_replay.py`（schema+采样+Gate3/4） | 复用，**但 Trajectory 不再存 memory_sequence**（§10） |
| `pending_episodes.py`（跨rollout整episode） | 复用，**逐step不再存 mem_pre/mask_pre**（§10） |
| `checkpointing.py`（orbax+replay_meta+inventory+compat报告） | 复用 + 扩展（EMA target、滚动剥离） |
| `rng_utils.py`（可保存 action RNG） | 直接复用 |
| `collect_rollout`（对齐+持久pending） | 复用（丢 mem_pre/mask_pre 存储） |

## 10. 必须替换/移除的 critic-only 模块

| 模块 | 处置 |
|---|---|
| `_replay_aux_loss`（value-only + PPO-ratio-on-replay actor 项） | **替换**为整轨迹 **V-trace** actor-critic 损失（ρ/c 截断 IS 校正） |
| critic-only 隔离：`_is_replay_updatable_path` / `_aux_tx`(masked) / `_aux_mask` / `_aux_opt_state` / `_select` | **移除/反转**：replay 联合更新 actor+trunk+value；新硬门 `REPLAY_ACTOR_UPDATE_CONTROLLED`（KL/IS/lag/grad-clip/loss-cap） |
| `p2_v1_update` 中 `replay_actor_update=False` | **替换**为受控 Full P2 replay actor-critic 更新 |
| `Trajectory.memory_sequence` / `ReplaySample.memory_sequence` 逐step记忆存储 | **移除存储**，改 **re-burn-in**（从存的 obs 用当前参数重算记忆）——磁盘杀手根因 |
| `pending` 逐step `mem_pre/mask_pre` | **移除存储**（on-policy 路径仍用瞬时内存，不落盘） |

## 11. 磁盘杀手根因（上轮 replay_meta 撑爆磁盘）

- `memory_sequence[T, window_mem=128, layers=2, embed=256]` ≈ **256 KB/步**。长 episode（健康 Student 在 S4_dark 搜索瓶颈下可达 1000–4096 步）单条记忆序列数百 MB；容量 256 → 数十 GB，且被序列化进**每个** checkpoint 的 `replay_meta.pkl`。
- **Full P2 决策：丢弃 memory_sequence 存储，改 R2D2 式 re-burn-in**——replay 只存 obs/act/rew/done/val/logp/next_obs/ach + init_mem；训练时用当前参数对采样序列前 `burn_in` 步跑 `forward_eval` 重建记忆（无梯度），再在 loss 区做 V-trace。既省盘（obs 主导），又对 V-trace 更一致（当前参数态表示）。
- obs 本身 8335 维≈33 KB/步仍是主成本 → replay 容量须有界（冻结 capacity=64），且 replay_meta 仅在前沿 checkpoint 序列化 + **滚动剥离**（只留最新前沿），峰值 ≈ 一份 buffer。

## 12. 配置（P2-v1 Stage4 已解析 Cfg，作为公平基线）

`num_envs=16, num_steps=128, num_minibatches=2, update_epochs=1, window_mem=128, window_grad=64, embed_size=256, num_heads=8, num_layers=2, qkv_features=256, hidden_layers=256, activation(tanh/relu见cfg), gating=True, lr=2e-4(常数,anneal=False), min_lr=2e-6, gamma=0.999, gae_lambda=0.8, clip_eps=0.2, vf_coef=0.5, ent_coef=0.002, max_grad_norm=1.0, condition_on_task=True`。
action_dim=43, obs_dim=8335, task_emb=67。1 update = 16×128 = 2048 env steps。
