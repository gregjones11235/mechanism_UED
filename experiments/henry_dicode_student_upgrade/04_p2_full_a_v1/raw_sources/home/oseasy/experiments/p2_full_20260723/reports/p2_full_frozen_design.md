# P2-Full-A — 冻结算法设计（v2.1，权威）

- 日期: 2026-07-23（v2.1；v2 取代 v1，v1 见 `p2_full_frozen_design_v1_SUPERSEDED.md`；v2.1 增补见 §19）
- 状态: **DESIGN FROZEN v2.1 — v2 总体方案已批准；v2.1 为用户强制 KL 事务门修订（§19），算法主干/Control 协议不变**
- **正式命名（§命名边界）**: 
  > **P2-Full-A = native GTrXL + sequence replay + V-trace actor/value + hindsight AWR actor/value learning**
- **命名边界（强制）**: 本方案**不**声称已扩展到"完整 episode 显式 Transformer 上下文"。长上下文 = GTrXL 原生 `window_mem=128` 记忆库 + 序列 replay 的多步信用分配；**没有**把 Transformer 显式上下文窗口扩展到整条 episode。Phase-2 residual adapter 继续**关闭**（§13）。
- 目标: 一个能在 Official FULL 自然破零 Tier3（ENTER_SEWERS>0 且 DEFEAT_KOBOLD>0）的更强 Student。
- 唯一算法: P2-Full-A（单一方案，不实现第二套）。

**v1→v2 修订摘要（用户强制 4 项）**：
1. **Hindsight 与 V-trace 分离**（§3/§5）：原目标轨迹用 stored behavior logprob 做 V-trace actor/value；**重标轨迹禁用原目标 behavior logprob 计算 IS ratio**——重标 Value/Q 用重标 TD target，重标 Actor 用受 KL/权重裁剪/policy-lag 约束的 **advantage-weighted behavior cloning (AWR)**。新增"无跨目标错误 ratio"测试（G4.4）。
2. **Memory 恢复改为稀疏 anchor**（§2）：**禁止**对任意中段序列仅从零 memory burn-in 128 步；每 128 步存一次 **pre-action GTrXL memory anchor**；replay 从最近 anchor 重放到 loss window 起点；episode 起点必有 anchor；anchor 纳入轨迹守恒/checkpoint/exact-resume 测试；**不恢复**逐步 memory_sequence 全量存储。
3. **Control 选择规则冻结**（§14）：LR 网格 {2e-4, 6e-5, 2e-5}，每组 24576 步、固定同 64 worlds；健康门 = SR 较 Baseline 降幅 ≤8pp + floor3 reach ≥80% Baseline + 无 NaN/Inf + policy KL<阈值；多个 LR 过门取**最高 LR**；全失败 → 停并标 `CONTROL_PROTOCOL_UNHEALTHY`。
4. **命名边界**：见上。

---

## 0. 为什么是这个方案（审计结论驱动）

审计证明健康 Student 的 GTrXL 记忆（`window_mem=128`）已是 actor 在训练/推理都真实使用的长上下文。瓶颈不在"看不到长历史"，而在"稀有成功轨迹无法反复利用 + Kobold 奖励无法向早期搜索传播"。P2-Full-A 因此**不新增参数模块**，补上缺失的三件事：
1. **整轨迹/长序列 off-policy 重放**（反复利用稀有成功轨迹）；
2. **V-trace 多步回报**（原目标轨迹，把晚期奖励沿轨迹向早期搜索状态传播）；
3. **hindsight AWR actor/value**（重标轨迹，把"事后才达成的子目标"变成密集学习信号，且不与 V-trace 共用错误的跨目标 ratio）。
兼容初始化 = ckpt17500 参数树 100% 加载，初始 policy 与健康 Student bit-exact 一致。

## 1. 冻结定义对照（§三）

| §三 要求 | P2-Full-A 满足方式 |
|---|---|
| 1 真正的长上下文策略模型 | GTrXL `window_mem=128` 记忆，actor 逐位置 attend（eval 129 / train 192）+ 序列 replay 多步信用分配。**未扩展显式整 episode 上下文。** |
| 2 整轨迹或长序列 replay | `ReplayBuffer` 存完整 episode（>128）+ **稀疏 memory anchor**；采样整 episode 或 L_seq=512 长序列 |
| 3 hindsight goal/reward relabel | `hindsight.py`（goal obs-条件 + reward 重算 + target，Gate5/6）→ 驱动 **AWR** 路径（§5），非 V-trace ratio |
| 4 off-policy Actor+Value/TD 联合学习 | **原目标 = V-trace**（ρ/c 截断 IS）；**重标 = AWR**（加权 BC + 重标 TD value，无 IS ratio）。actor+trunk+value 联合更新 |
| 5 replay 直接改变策略行为 | 受控 actor replay 更新（KL/IS/lag/grad/AWR-权重门），actor 头+trunk 有限非零变化 |
| 6 健康 checkpoint 兼容初始化 | ckpt17500 → 同一参数树 100% 叶子加载（§9） |
| 7 策略漂移稳定约束 | ρ̄/c̄ 截断 + AWR 权重裁剪 + policy-lag 拒绝 + KL 阈值 + grad-clip + loss 权重上限 + EMA target + value-target clip |

明令禁止项规避：不"只存轨迹但 actor 仍按 128 PPO 学"；不"hindsight 只影响 critic"（AWR 改 actor）；不"replay 只更新 value head"（联合更新）；不"普通 off-policy 套 PPO ratio"（V-trace ρ/c；重标走 AWR 不用 ratio）；不"critic-only 冒充 Full P2"（隔离机制移除）；**不"用原目标 behavior logprob 给重标轨迹算 IS ratio"（跨目标 ratio 禁止，§5/G4.4）**。

## 2. 长上下文长度 / 稀疏 anchor / loss 区间 / 序列采样

- **长上下文长度** = `window_mem = 128`（GTrXL 记忆库；actor 真实 attend）。
- **稀疏 memory anchor（v2 修订，取代零记忆 burn-in）**：
  - **禁止**对任意中段序列仅从零 memory burn-in 128 步（中段 memory 反映整段 episode 历史，零记忆重建是错误的）。
  - 采集时**每 128 步**保存一次 **pre-action GTrXL memory anchor**：在步 `t ∈ {0,128,256,…}` 动作**之前**快照当前 `memories[window_mem,layers,embed]`，记 `anchor_step=t`。
  - **episode 起点必有 anchor**（step 0 = `initial_memory`，fresh episode 为零记忆）。
  - anchor 数 `n_anchors = ceil(L/128)`（步 0,128,…,128·(n−1)）。
  - **replay 记忆重建**：对 loss window 起点 `s`，取最大 `anchor_step ≤ s` 的 anchor 记忆，用**当前参数**对其间 `obs[anchor_step : s]` 跑 `model_forward_eval`（无梯度）重放到 `s`，得 pre-action memory at `s`，再做 loss 区 train forward。因 anchor 间隔 128，**重放长度 ≤128 步**且记忆正确（anchor 已携带真实 episode 历史）。anchor 记忆在喂入 loss 区前 `stop_gradient`。
  - **不恢复逐步 `memory_sequence` 全量存储**（磁盘杀手根因，审计§11）；anchor 比 memory_sequence 省 ~128×。
- **loss 区间** = 序列有效步（anchor 重建记忆之后），按 `window_grad = 64` 切窗。每窗 GTrXL train forward 前缀 128 记忆（由 anchor 重放得到）。
- **序列长度** `L_seq = 512`（可配置）。采样：episode ≤ L_seq → 整条（真·整轨迹）；否则 contiguous L_seq 窗口（窗口起点可为任意步，记忆由其前方最近 anchor 重放重建，**不要求窗口内含 128 步前缀**——anchor 已提供正确记忆）。
- **每次 replay 更新采样 K = 4 条序列**，batched。1 次 replay 更新 / 1 次 PPO 主更新。
- **序列采样**：`ReplayBuffer.sample`（确定性 RandomState），仅完整 episode（done=True）、长度 >128。容量 **capacity = 64**。

## 3. V-trace off-policy 校正（仅原目标轨迹，冻结公式，离散动作）

**适用范围**：原目标（采集时 goal-conditioning）轨迹。behavior 策略 μ = 采集时存储的 `log_probs`（即产生这些动作的真实行为策略），当前策略 π。
- 重要性比 `ρ_t = π(a_t|x_t)/μ(a_t)`，截断 `ρ̄_t = min(ρ̄, ρ_t)`（**ρ̄=1.0**），`c_t = min(c̄, ρ_t)`（**c̄=1.0**）。
- TD 误差 `δ_t = ρ̄_t·(r_t + γ·V_target(x_{t+1})·(1−done_t) − V_online(x_t))`，γ=0.999。
- **V-trace 目标**（逆序递推）：`v_t = V_online(x_t) + Σ_{k=t}^{L-1} γ^{k-t}(Π_{i=t}^{k-1} c_i·(1−done_i)) δ_k`；done 后不 bootstrap（done→0）。`v_t` 裁剪 `[vt_clip_min,vt_clip_max]=[-50,300]`。
- **Value loss**：`L_vtrace_value = 0.5·mean_t (V_online(x_t) − sg(v_t))²`。
- **Actor loss**（IMPALA V-trace 策略梯度）：
  `L_vtrace_actor = −mean_t[ log π(a_t|x_t)·ρ̄_t·(r_t + γ·sg(v_{t+1})·(1−done_t) − sg(V_online(x_t))) ] − ent_coef·mean_t H(π_t)`。
- `V_online`=在线网络；`V_target`=EMA target（§4）。behavior logprob μ 必用，禁止把旧数据当 on-policy。
- **μ 的合法性**：原目标轨迹下 μ 是真实行为策略，ρ=π/μ 是正确的 on→off 校正。**此 ratio 只用于原目标轨迹**；重标轨迹**不得**复用此 μ（§5）。

## 4. Target / EMA 网络策略（冻结）

- **EMA target 网络**：一份参数 EMA 副本，用于 TD 目标 `V_target(x_{t+1})`、bootstrap、AWR 优势基线。
- 更新：每次 PPO+replay 更新后 `θ_target ← τ·θ_target + (1−τ)·θ_online`，**τ=0.995**；初值 `θ_target=θ_online`（ckpt17500）。
- EMA target 进 checkpoint（exact resume），不做梯度更新。`V_online` 用于 loss 的 `V_online(x_t)` 与 actor 前向。

## 5. Hindsight → AWR actor/value（v2 修订：与 V-trace 分离，Gate4）

**核心原则：重标轨迹禁止使用原目标 behavior logprob 计算 IS ratio。** 重标改变 goal-conditioning（obs 尾部 67 维）与 reward，原 μ 不再是重标策略的行为分布，`π_relabeled/μ_original` 是**跨目标错误 ratio**，禁用。重标轨迹改用 **AWR（advantage-weighted behavior cloning）+ 重标 TD value**。

**重标流程**（复用 `hindsight.py`，Gate5/6 不弱化）：
- `relabel_sample`：从该轨迹**字面达成**的 achievement 选目标（默认最小索引/按覆盖度采样），替换 obs/next_obs 尾部 67 维 goal multi-hot，重算 `r'_t = max(ach[t,g']−ach[t−1,g'],0)`，更新 target_achievements；透传 behavior log_probs/policy_version（仅供诊断，**不进 AWR ratio**）。
- Gate5 只用真实达成目标；Gate6 拒绝伪造/未达目标（`ValueError`）。无达成目标 → 该序列跳过 hindsight（仅走原目标 V-trace），不伪造。

**重标 Value/Q（重标 TD target，无 IS ratio）**：
- 重标折扣回报 `G'_t = Σ_{k=t}^{L-1} γ^{k-t}(Π_{i=t}^{k-1}(1−done_i)) r'_k + γ^{L-t}(Π(1−done))·V_target(x_L)`（done 后不 bootstrap）；裁剪 `[-50,300]`。
- `L_awr_value = 0.5·mean_t (V'_online(x'_t) − sg(G'_t))²`。`V'_online(x'_t)` = 当前网络在**重标 obs** 下的 value。**无 importance ratio。**

**重标 Actor（受约束 advantage-weighted behavior cloning）**：
- 重标优势 `A'_t = sg(G'_t − V_target(x'_t))`（重标 TD 优势，stop_gradient）。
- AWR 权重 `w_t = min(w_max, exp(A'_t / β))`（stop_gradient）。**冻结 β=1.0（AWR 温度），w_max=20.0（权重裁剪上限）**。
- **AWR actor loss**（对实际动作的加权行为克隆，**不含 IS ratio**）：
  `L_awr_actor = −mean_t[ w_t · log π(a_t | x'_t) ] + λ_kl·mean_t KL(π(·|x'_t) ∥ sg(π_before(·|x'_t)))`。
  - π(a_t|x'_t) = 当前策略在**重标 obs** 下对**所采动作** a_t 的概率。
  - **KL 约束**：软罚项 `λ_kl=0.01`（对更新前策略 π_before，stop_gradient）+ 硬门 `KL(π_before∥π_after) < KL_MAX_AWR=0.05`（§6）。
- **不使用** `log_probs`(μ) 作为分母或任何 ratio 因子（G4.4 验证）。

**合并 replay 目标（单次受控梯度步）**：
`L_replay = w_vtrace·(L_vtrace_actor + vf_coef·L_vtrace_value) + w_awr·(L_awr_actor + vf_coef·L_awr_value)`。
- 冻结 `w_vtrace=0.5`，`w_awr=0.5`，`vf_coef=0.5`，`ent_coef=0.002`；`w_vtrace+w_awr ≤ W_REPLAY_MAX=1.0`。
- 一次 replay 更新 = 一次合并梯度步，`clip_by_global_norm(1.0)`，受 §6 全部门约束。诊断按 vtrace/awr 分别记录（vtrace 记 ratio/ESS；awr 记权重 w 的 max/mean、KL）。
- 每条采样序列**同时**产生原目标 V-trace 项；若该序列有 eligible 子目标，**额外**产生重标 AWR 项。

## 6. 受控 Actor replay 更新（硬门 REPLAY_ACTOR_UPDATE_CONTROLLED）

旧"replay 不得改 actor"硬门**取消**。新硬门要求每次 replay actor 更新同时满足（否则该次更新 fail-closed、不写坏 checkpoint、非零退出）：
- `actor_params_finite_nonzero_change`：actor 头叶子有限且非零变化。
- `trunk_params_finite_nonzero_change`：transformer/encoder 叶子有限且非零变化。
- `value_params_changed`：critic 头变化。
- `policy_kl_finite_below_threshold`：固定探针批上 `KL(π_before∥π_after)` 有限且 < **KL_MAX=0.05**；AWR 路径另满足 `KL_MAX_AWR=0.05`。
- `vtrace_importance_ratio_bounded`：原目标 `ρ_t` 截断在 ρ̄=1.0；记录 raw ratio max/mean、ESS fraction。
- `awr_weight_bounded`：AWR `w_t ≤ w_max=20`；记录 w max/mean、AWR KL。
- `no_cross_goal_ratio`：AWR 梯度路径不引用 behavior `log_probs` 作 ratio（G4.4 结构+扰动双重验证）。
- `policy_lag_reject_gate`：`lag = update_count − collected_update_count > MAX_POLICY_LAG(=16)` → 丢弃该序列 actor 项（V-trace 与 AWR 都丢；诊断仍记录）。
- `grad_norm_clipped`：`clip_by_global_norm(1.0)`；梯度非有限 → fail-closed。
- `replay_loss_weight_capped`：`w_vtrace+w_awr ≤ W_REPLAY_MAX=1.0`。
- `no_one_step_collapse`：更新后固定探针 obs 上 entropy > ENT_FLOOR=0.05、logits 有限。
- PPO/行为采集路径在 replay 后仍可运行。
- **v2.1 事务门（§19）**：上述 `policy_kl_finite_below_threshold` 升级为**事务门**——候选更新 KL 超 `kl_replay_max=0.05` 时按 actor 步长 1.0/0.5/0.25/0.125 重试；首个过门者提交，全失败 → `KL_REJECTED_UPDATE` 且**回滚 policy-affecting params+对应 opt_state+基于该候选的 EMA target**；**纯 critic 专有 head 独立提交**。门覆盖 encoder+GTrXL/shared trunk+goal/context+actor head 全部 policy-affecting 参数。

## 7. 训练循环（每个 update）

1. `collect_rollout`（on-policy，16 env×128 步；持久 pending 收整 episode；存 behavior logprob + policy version；**每 128 步存 pre-action memory anchor + episode 起点 anchor；不存逐步 memory_sequence**）。
2. 完成 episode 插入 `ReplayBuffer`（capacity 64；含 obs/act/rew/done/val/logp/next_obs/ach + **anchors**）。
3. `compute_on_policy_gae` + `ppo_update`（原生 PPO 主更新 = Control 算法，更新 actor+trunk+critic）。`update_count += 1`。
4. 若 `replay.can_sample()`：采 K=4 序列 → anchor 重建记忆（§2）→ 原目标 V-trace 项（§3）+（eligible 时）重标 AWR 项（§5）→ 合并受控梯度步（§6）。`update_count += 1`。
5. EMA target 软更新（§4）。
6. 记 JSONL：ppo_* 与 replay_*（vtrace loss/ratio/ESS、awr loss/w/KL、grad_norm、policy KL、lag、hindsight_goal、seq_len、n_anchors）+ 守恒计数（含 anchor 守恒）。
7. 前沿步（0/24576/49152/73728/98304）写 checkpoint（§8）+ 滚动剥离旧前沿 replay_meta。

## 8. Checkpoint schema（兼容 + exact resume，含 anchor）

每个前沿 checkpoint `<step>/`：
- `default/`（orbax）：`{params, opt_state}`（在线 TrainState，bit-exact）。
- `ema_params/`：EMA target 参数。
- `replay_meta.pkl`：replay buffer state（obs/act/rew/done/val/logp/next_obs/ach + **memory_anchors + anchor_steps**；**无逐步 memory_sequence**）+ rng_key + action_rng_state + global_step + update_count + gradient_updates + pending_state（含 **pending anchors**）+ collector_state（obsv/env_state/memories/mem_mask/mem_idx）。
- `manifest.json`：step/global_step/update_count/gradient_updates/counters/replay_size/**anchor 计数**/gamma/gae/gpu_uuid/各 SHA。
- **滚动剥离**：新前沿写完后删上一前沿 `replay_meta.pkl`；峰值 ≈ 一份 buffer。
- **exact resume**：params+opt+EMA+rng+action_rng+collector+pending（含 anchors）+global_step+update_count bit-exact；replay buffer（含 anchors）best-effort 前沿重载（文档注明）。
- restore/verify 必须 GPU0（GPU-saved orbax 不能 CPU 恢复）。

## 9. 旧 checkpoint 兼容初始化（ckpt17500）

- P2-Full-A 网络 = 与 ckpt17500 **完全相同**的 `ActorCriticTransformer`（action_dim=43/encoder_size=256/num_heads=8/qkv=256/num_layers=2/gating）。
- 加载：`load_weights_only(ckpt17500)` → **80/80 叶子全匹配，无未加载叶子，无新初始化叶子**（`compatible_weight_restore_report` 记 restored=80/skipped=[]/newly_initialized=[]）。
- EMA target 初始 = 加载后在线参数。优化器**重置**（fresh Adam，不继承 ckpt17500 opt_state = 兼容初始化非优化器续训，文档注明）。
- **bit-exact 验证**：固定 `obs=0,memory=0,mask=1`（batch=2）下复现 ckpt17500：`value=3.5761, entropy=0.9791, top action=12(0.718)`；params SHA256 == `5dfe67dd…`。
- **不静默跳过任何不匹配参数**：叶子数/形状不符 → fail-closed。

## 10. Gate 1–4 测试清单（CPU 单测）+ Gate 5（GPU0 smoke）

**Gate 1 — 长上下文前向 + 稀疏 anchor**（通过前禁止 Actor replay）：
- G1.1 历史前缀不同 → 后续 logits 不同。
- G1.2 padding 内容不影响有效区间（改 mask 外 padding → 有效步 logits 不变）。
- G1.3 episode A 不污染 episode B（done 清零；拼接 vs 分开 → 各自 logits 一致）。
- G1.4 **anchor 记忆重建 == 在线 rollout 记忆（bit-exact）**：对一段 obs，从 anchor 重放得到的 pre-action memory == 在线 rollout 在同一步的 memory。
- G1.5 interrupted/resumed bit-exact（checkpoint round-trip 后前向一致）。
- G1.6 长序列前向序列长度 >128（真实读 >128 步）。
- **G1.7 anchor 守恒**：`n_anchors == ceil(L/128)`；`anchor_steps == [0,128,…]`；episode 起点 anchor 存在且 == initial_memory；中段序列记忆由最近 anchor 重建（**不从零 memory**）。

**Gate 2 — Sequence TD / Value 学习**（暂不开 Actor replay）：
- G2.1 采样序列 >128；anchor 重建记忆正常。
- G2.2 原目标 V-trace TD target 有限（无 NaN/Inf）且在 [-50,300]。
- G2.3 bootstrap 边界正确：done 后不 bootstrap；非终结用 EMA V_target(x_{t+1})。
- G2.4 mask/padding 不进 loss（loss 仅有效步）。
- G2.5 EMA target 正确（τ=0.995；初值=在线）。
- G2.6 value 与 long-context（encoder+transformer）参数有限非零梯度。
- G2.7 确定性 V-trace 参考比对（手写 numpy V-trace vs 实现，固定 ρ/c/r/v）。
- G2.8 重标 TD value target 有限且在 [-50,300]（无 IS ratio）。
- G2.9 无 NaN/Inf。

**Gate 3 — 受约束 Actor replay（V-trace 原目标路径）**：
- G3.1 一次 replay actor update 后：actor 参数有限非零变化、trunk 有限非零变化、value 变化。
- G3.2 policy KL 有限且 < KL_MAX=0.05。
- G3.3 原目标 IS ratio 截断（ρ̄=1.0）、记录 raw ratio/ESS。
- G3.4 policy-lag 拒绝门：lag>16 的序列 actor 项被丢弃（V-trace+AWR）。
- G3.5 grad-clip(1.0)、`w_vtrace+w_awr ≤ W_REPLAY_MAX`。
- G3.6 无一步坍塌（entropy>0.05、logits 有限）；PPO 采集路径仍可运行。
- G3.7 `REPLAY_ACTOR_UPDATE_CONTROLLED=true`。

**Gate 4 — Hindsight AWR（与 V-trace 分离）**：
- G4.1 同一轨迹重标前后：goal 不同、reward/return 不同、AWR actor loss 不同、重标 value target 不同、actor 梯度不同（逐条 not allclose）。
- G4.2 Gate5 正向（只用真实达成目标）/ Gate6 负向（伪造目标 raise）。
- G4.3 重标不改 behavior logprob / policy version（透传保留，供诊断）。
- **G4.4 无跨目标错误 ratio（核心）**：
  - (a) **扰动测试**：大幅扰动重标序列的 behavior `log_probs` → **AWR actor 梯度不变**（allclose）；扰动原目标序列的 `log_probs` → **V-trace actor 梯度改变**（not allclose）。证明 AWR 不依赖 μ、V-trace 依赖 μ。
  - (b) **结构断言**：AWR loss 函数签名/实现不以 behavior `log_probs` 作为 ratio 分母或乘子。
- G4.5 AWR 权重裁剪：`w_t ≤ w_max=20`；KL 软罚+硬门（KL_MAX_AWR=0.05）生效。

**Gate 5 — P2-Full-A 集成 smoke（GPU0）**：
- 1 update/2048 → 2 updates/4096。核验：PPO 采集正常；replay 真实采样；hindsight accepted>0（有 eligible 时）；V-trace actor update>0；AWR actor update>0（有 eligible 时）；long-context 梯度>0；value 梯度>0；KL/IS/ESS/AWR-w/lag 有限；trajectory 守恒（collected==inserted，**含 anchor 守恒**）；checkpoint round-trip bit-exact；无 NaN/Inf。
- **4096 步无 eligible 长轨迹时不降门槛**：报 `DATA_UNREACHABLE`（附 episode 长度分布），而非改阈值。

## 11. 磁盘 / 显存峰值估算（v2：anchor 省 ~128×）

- **参数**：4,906,028×4B ≈ 19.6 MB；Adam opt_state ≈ 39 MB；EMA target ≈ 19.6 MB。
- **单 checkpoint**：params+opt ≈ 60 MB + ema ≈ 20 MB = ~80 MB；5 前沿 ≈ 400 MB。
- **memory anchor（v2）**：每 anchor = window_mem(128)×layers(2)×embed(256)×4B = **256 KB**；每 128 步一个 → **~2 KB/步均摊**（vs memory_sequence 256 KB/步，省 128×）。L=512 → 4 anchor = 1 MB/episode；L=4096 → 32 anchor = 8 MB/episode。
- **Replay buffer（RAM 驻留，无 memory_sequence）**：obs+next_obs ≈ 67 KB/步为主；L≈2000–4096 → 单条 ~134–275 MB + anchors ~4–8 MB。capacity=64 → **~9–18 GB（RAM，122 G 充裕）**。
- **磁盘峰值**：滚动剥离后仅最新前沿 `replay_meta.pkl`（≈ 一份 buffer ~9–18 GB，含 anchors）+ 5 params checkpoint ≈ **~10–19 GB**。/home 可用 50 G → 可行。
- **磁盘看门狗**：可用 < **12 GB** 立即暂停。
- **显存**：GTrXL ~5 M 参数 + PPO 主更新（16×128 窗口化）已在 GPU0 正常运行；replay 增 K=4×L_seq=512 扫描前向/反向（≈2048 步，含 anchor 重放 ≤128 步/窗）≈ 与 PPO 主更新同量级 → 预估 < 上轮峰值。Level A smoke 实测确认。
- **序列长度上限**：L_seq=512 bound 单次 replay 显存/时间。

## 12. 开发文件清单（写入 p2_full_20260723/，不改 frozen 源码）

```
p2_full_20260723/
  reports/p2_full_architecture_audit.md            (报告1，审计)
  reports/p2_full_frozen_design.md                 (报告2，v2 权威)
  reports/p2_full_frozen_design_v1_SUPERSEDED.md   (v1 存档)
  src/
    vtrace.py             原目标 V-trace 目标/损失（ρ/c 截断，逆序递推，确定性）
    awr.py                hindsight AWR：重标 TD value + 加权 BC actor（β/w_max/λ_kl/KL 门，无 IS ratio）
    memory_anchor.py      稀疏 anchor 采集 + 从最近 anchor 重放重建 pre-action memory（≤128 步）
    full_p2_learner.py    P2-Full-A 学习器：anchor 重建 + V-trace(原目标) + AWR(重标) 合并受控更新
                          + EMA target + 受控门（复用 long_context_learner 窗口/mask/GAE/PPO 主更新；移除 critic-only 隔离）
    full_p2_core.py       collect_rollout(每128步存anchor、去 memory_sequence) + full_p2_update 编排
    replay_buffer.py      复用 trajectory_replay，Trajectory 含 anchors、去 memory_sequence，capacity=64
    hindsight.py          复用（拷贝，Gate5/6 不改）
    checkpointing.py      复用 + EMA target + anchors + 滚动剥离
    pending_episodes.py   复用（存 pending anchors，去逐步 mem_pre/mask_pre）
    rng_utils.py          复用
    compat_init.py        ckpt17500 兼容初始化 + bit-exact 验证 + 叶子报告
  tests/
    test_gate1_longctx_anchor.py test_gate2_vtrace_value.py test_gate3_actor_replay.py
    test_gate4_hindsight_awr.py test_no_cross_goal_ratio.py test_vtrace_reference.py
    test_awr_reference.py test_memory_anchor.py test_compat_init.py
    test_checkpoint_roundtrip.py test_replay_anchor_conservation.py
  commands/
    run_control_calibration.sh   Control 健康校准（lr 网格 + §14 选择规则）
    run_fullp2_smoke.sh          Gate5 集成 smoke（2048/4096）
    run_short_screen.sh          Level B 24576 短程筛选
    run_formal_train.sh          Level C 98304 正式训练
    run_formal_eval.sh           Level D 512+512 正式评估
  logs/  evidence/
```
Henry 基座 / session175 / ckpt17500 / P2-v0 / P2-v1-lite 全只读，import 不修改。

## 13. Residual long-context adapter（冻结关闭，Phase-2 可选）

`base_feature + long_context_residual`（residual 初值 0）作为**冻结关闭**扩展保留：仅当 P2-Full-A 显示方向性信号、且证据表明"记忆长度=128 是限制因素"时才启用；届时 residual-zero 接入，新增叶子标记 newly_initialized，初始 policy 仍 bit-exact 等于健康 Student。**v1 不实现**。注意：启用它属于"扩展显式长上下文"，与当前命名边界（未扩展整 episode 显式上下文）一致地标注为 Phase-2，**不计入 P2-Full-A**。

## 14. Control 健康协议与选择规则（v2 冻结）

- 起点 ckpt17500，原始 Henry GTrXL-PPO（`native_ppo_loss`/`ppo_update`，replay/hindsight 关），静态 S4_dark 训练任务。
- **LR 网格（预定义，禁看结果后无限调参）**：`{2e-4（原默认）, 6e-5（0.3×）, 2e-5（0.1×）}`；其余超参冻结（γ0.999/λ0.8/clip0.2/vf0.5/ent0.002/gradnorm1.0/num_envs16/num_steps128）。
- **每组 24576 步，固定相同 64 worlds 评估**。
- **健康门（提前冻结，全部满足才算通过）**：
  1. Stage4 SR 较 Baseline 下降 **≤ 8 个百分点**；
  2. floor3 reach **≥ 80% × Baseline**；
  3. 无 NaN/Inf；
  4. policy KL < 冻结阈值 `KL_MAX_RUN=0.1`（累计）。
- **选择规则**：多个 LR 通过 → 选**通过健康门的最高 LR**；该 LR 冻结用于 P2-Full-A 的 on-policy 基础部分（同一 LR）。
- **全部失败** → 停止正式比较，标 `CONTROL_PROTOCOL_UNHEALTHY`。

## 15. 实验组（§七）

- **A. Baseline**：ckpt17500，不训练。
- **B. Control**：ckpt17500 + 原始 Henry GTrXL-PPO（§14 冻结 LR）+ 静态 S4_dark。
- **C. P2-Full-A**：ckpt17500 兼容初始化 + native GTrXL + sequence replay + V-trace actor/value + hindsight AWR actor/value（同冻结 LR）。
- 主比较 C vs B；辅 C vs A、B vs A。一致项：训练任务/env 数/训练 seed/env steps/γ0.999/λ0.8/checkpoint 时刻/评估世界/max episode steps/obs-action 接口。仅 C 新增 sequence replay + V-trace + AWR。

## 16. 训练评估阶梯（§九）与成功判据（§十）

- **Level A**：CPU 单测（Gate1–4）→ 2048 GPU smoke → 4096 activation smoke。通过才训练。
- **Level B**：Control 与 P2-Full-A 各 24576 步，同 64 Stage4 worlds 评估（DK/ENTER_SEWERS/floor3/conditional kill/death-timeout/ep len/policy KL/death-to-stair/上游保持）。同时满足（Control 健康 + P2-Full-A 无灾难退化 + V-trace & AWR actor 路径真实工作 + P2-Full-A 方向性改善或失败行为明显改善）才扩长。
- **Level C**：98304 步，存 0/24576/49152/73728/98304。
- **Level D**：512 fresh S4_dark + 512 fresh Official FULL（自然 floor0 出生，无脚手架/arrival reset），每 episode ≤4096 步。
- **成功判据**：S4_dark SR(P2-Full-A)−SR(Control) ≥ 8pp；Official FULL Tier3 自然破零（ENTER_SEWERS>0 且 DEFEAT_KOBOLD>0）；能力保护（floor2 不崩、Tier1/2 保持、不显著低于 Baseline）；行为证据（floor3↑、death-to-stair↓、覆盖↑、重复走旧路↓、条件击杀不恶化）；复现（二批 512 worlds 或第二 seed 再 Tier3 非零）。单次 1/512 仅记 `TIER3_BREAKZERO_SIGNAL`，不称稳定突破。

## 17. 停止条件（§十一，立即暂停）

Control 协议不健康 / NaN/Inf / checkpoint 不可恢复 / replay 轨迹或 anchor 守恒失败 / episode 边界污染 / off-policy ratio 失控 / 跨目标 ratio 出现 / AWR 权重或 KL 超阈 / Actor KL 超冻结阈值 / Actor 一步更新后灾难性漂移 / GPU UUID 绑定错误 / 输出目录串线 / 旧 checkpoint 被修改 / 磁盘 < 12 GB。**不自动改算法规避失败。**

## 18. Token / 监控（§十二）

无子 Agent（SUBAGENT_COUNT=0）；禁高频 nvidia-smi/ps/tail；训练本地阻塞等待；公共源码只读一次；测试审计批量；指标写 JSONL/CSV；不逐 Gate 长篇汇报。汇报节点：①审计+冻结设计完成（已报）②Gate1–4 全过或 BLOCKED ③4096 smoke 完成 ④24576 短程完成 ⑤98304+512 评估完成。中间只报 PASS/FAIL、核心指标、异常、证据路径、SHA256、下一动作。仅 GPU0（UUID 绑定）。

## 19. v2.1 增补 — KL 事务门 / 三类 KL / actor 步长重试 / critic 隔离（用户强制修订）

**触发**：4096 GPU0 smoke 出现 `policy_kl=0.239 > KL_REPLAY_MAX=0.05` 却仍提交策略更新，定级 `CONTROLLED_ACTOR_UPDATE_FAIL / LEVEL_A_INCOMPLETE`。禁止直接进入 98304 或正式 512 评估。以下修订把 §6 的 KL 报告门升级为**事务门**，算法主干（§2–§5）与 Control 协议（§14）不变。

### 19.1 KL 事务门的参数覆盖范围（强制）
事务门必须覆盖**全部 policy-affecting 参数**：
- **encoder**（obs→embed，在 Transformer 内）、**GTrXL / shared trunk**（Transformer 全部层）、**goal/context 模块**、**actor head**（`actor_ln1/actor_ln2/actor_out`）。
- **纯 critic 专有 head**（`critic_ln1/critic_ln2/critic_out`）**不在 actor 前向中**，可**独立提交**，不受 KL 门约束；但任何参与 Actor 前向的 shared trunk **不得**绕过 KL 门。
- 实现：按参数叶子路径的模块名划分——`critic_ln1/critic_ln2/critic_out` 为 critic-only，其余全部为 policy-affecting。冻结参数计数：critic-only **131,841**，policy-affecting **4,774,187**，合计 **4,906,028**（== ckpt17500）。`test_kl_transactional_gate` 用结构计数 + 扰动双重证明：扰动 critic-only 叶子策略 logits 不变（value 变），扰动 actor 叶子 logits 非均匀变化。

### 19.2 超限回滚（强制）
候选更新 KL 超限时，必须回滚：
1. **policy-affecting params**（恢复为更新前）；
2. **对应 optimizer state**（policy-affecting 叶子的 Adam moments 回滚；共享 Adam count 随 policy 侧回滚，取保守语义，已在实现 docstring 注明）；
3. **基于该候选的 EMA target 更新**（policy-affecting target 叶子冻结）。
纯 critic 专有 head 的 params / opt moments / EMA target **照常提交**（独立于门）。

### 19.3 actor 步长重试（强制）
依次尝试 actor 步长缩放 **{1.0, 0.5, 0.25, 0.125}**：仅缩放 policy-affecting 部分的更新步（critic-only 始终全步）。首个使 `KL(π_candidate ∥ π_current) ≤ kl_replay_max` 的 scale 被接受并提交；**全部失败 → `KL_REJECTED_UPDATE`，不提交任何策略更新**（critic 仍提交）。探针 KL 在**原目标 loss window** 上计算（与 §6 探针一致）。

### 19.4 三类 KL 必须区分（强制，不得混用阈值）
| 名称 | 冻结阈值 | 约束对象 |
|---|---|---|
| `kl_replay_max` | **0.05** | **单次** replay 更新（§19.3 事务门）。仅约束单步。 |
| `kl_run_max`（== §14 `KL_MAX_RUN`） | **0.1** | 一次 24576 步运行的**累计** policy KL（Control 健康门）。 |
| `kl_baseline_cumulative` | 独立诊断 | 相对 **Baseline（ckpt17500）** 策略的累计 KL，单独追踪/报告。 |
**明确：0.05 绝不用于约束 24576 步累计 KL**；累计 KL 用 0.1（`kl_run_max`）。

### 19.5 smoke 预算 manifest（强制记录）
4096 smoke 必须记录并断言：`num_envs=16`、`rollout_steps=128`、`num_updates=2`（更新机会数：1/2048→2/4096）、`steps_per_env=256`、`total_env_steps=4096`。写入 `smoke_summary.json` 与 checkpoint manifest。

### 19.6 Level A 正式 PASS 判据（修订）
用 §14 网格选定的 LR 重跑 4096 Full P2 smoke；**只有出现至少一次 `KL ≤ 0.05` 的 accepted policy replay update（`accepted_policy_updates ≥ 1`）**，Level A 才正式 PASS。`accepted_policy_updates` 与 `kl_rejected_updates` 记入 summary。2048 smoke 合法地 0 更新、KL_REJECTED 是合法门结果（均非管线失败）。

### 19.7 v2.1 测试与实现（已 CPU 全绿）
- `src/full_p2_learner.py`：`critic_only_mask/classify_params/_select_where/_select_opt_critic/_window_log_softmax/_kl_mean`；`full_p2_update` 重写为事务门（§19.1–19.4）。`FullP2Config` 增 `kl_replay_max=0.05 / kl_run_max=0.1 / actor_step_scales=(1.0,0.5,0.25,0.125)`。
- `tests/test_kl_transactional_gate.py`：KL.partition / KL.criticinv / KL.accept / KL.scale / KL.rollback（回滚 bit-exact：149 个 opt 叶子回滚、12 个 critic opt 叶子推进、policy logits bit-exact、value 变）。
- Gate2/3/4 组合更新测试：LR 调至可过门区间并**强化**断言（`policy_committed==True` 且 `policy_kl≤0.05`），拒绝路径由门测试覆盖（非弱化）。
