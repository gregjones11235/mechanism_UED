> ⚠️ **SUPERSEDED — 本冻结设计 v1 已被 v2 取代（2026-07-23）。**
> 取代原因：用户批准进入 Level A 前要求 4 项强制修订：
> (1) Hindsight 与 V-trace 分离（重标轨迹禁用原目标 behavior logprob 计算 IS ratio，改 AWR）；
> (2) Memory 恢复由"零记忆 burn-in"改为"稀疏 anchor（每 128 步存 pre-action memory anchor）"；
> (3) Control 选择规则冻结（SR 降幅≤8pp、floor3 reach≥80% Baseline、多 LR 过门取最高 LR）；
> (4) 命名边界 = **P2-Full-A**，不得声称已扩展到完整 episode 显式 Transformer 上下文。
> **权威版本见 `p2_full_frozen_design.md`（v2）。本文件仅作 provenance 留存，不得据此实现。**

---

# Full P2 — 冻结算法设计（单一方案）

- 日期: 2026-07-23
- 状态: **DESIGN FROZEN（待用户确认后进入 Gate 1 实现）**
- 目标: 一个能在 Official FULL 自然破零 Tier3（ENTER_SEWERS>0 且 DEFEAT_KOBOLD>0）的更强 Student。
- 唯一算法: **原生 GTrXL 长上下文 + 整轨迹/长序列 V-trace actor-critic 重放 + hindsight 联合训练**（下称 Full P2）。
- 不实现第二套算法；§五的 residual long-context adapter **冻结关闭**，作为 Phase-2 可选扩展（见 §13）。

---

## 0. 为什么是这个方案（审计结论驱动）

审计证明健康 Student 的 GTrXL 记忆（`window_mem=128`）已是 actor 在训练/推理都真实使用的长上下文。瓶颈不在"看不到长历史"，而在"稀有成功轨迹无法反复利用 + Kobold 奖励无法向早期搜索传播"。Full P2 因此**不新增参数模块**，而是补上缺失的三件事：
1. **整轨迹/长序列 off-policy 重放**（反复利用稀有成功轨迹）；
2. **V-trace 多步回报**（把晚期 Kobold/子目标奖励沿轨迹向早期搜索状态传播）；
3. **受控 off-policy actor 更新**（replay 真正改变 actor + 长上下文 trunk，而非仅 critic）。
兼容初始化 = ckpt17500 参数树 100% 加载，初始 policy 与健康 Student bit-exact 一致。

## 1. 冻结定义对照（§三）

| §三 要求 | Full P2 满足方式 |
|---|---|
| 1 真正的长上下文策略模型 | GTrXL `window_mem=128` 记忆，actor 逐位置 attend（eval 129 步 / train 192 步）+ 整轨迹 V-trace 长程信用分配 |
| 2 整轨迹或长序列 replay | `TrajectoryReplayBuffer` 存完整 episode（>128），采样整 episode 或 L_seq=512 长序列 |
| 3 hindsight goal/reward relabel | 复用 `hindsight.py`（goal obs-条件 + reward 重算 + target，Gate5/6） |
| 4 off-policy Actor+Value/TD 联合学习 | **V-trace** actor-critic：actor 与 value 与 trunk 联合更新，ρ/c 截断 IS 校正 |
| 5 replay 直接改变策略行为 | 受控 actor replay 更新（KL/IS/lag/grad 门），actor 头+trunk 有限非零变化 |
| 6 健康 checkpoint 兼容初始化 | ckpt17500 → 同一参数树 100% 叶子加载（§9） |
| 7 策略漂移稳定约束 | ρ̄/c̄ 截断 + policy-lag 拒绝 + KL 阈值 + grad-clip + loss 权重上限 + EMA target + value-target clip |

明令禁止项的规避：不"只存轨迹但 actor 仍按 128 PPO 学"（actor 走 V-trace replay）；不"hindsight 只影响 critic"（改 reward→改 actor loss）；不"replay 只更新 value head"（联合更新）；不"普通 off-policy 套 PPO ratio"（用 V-trace ρ/c 截断）；不"critic-only 冒充 Full P2"（隔离机制移除）。

## 2. 长上下文长度 / burn-in / loss 区间 / 序列采样

- **长上下文长度** = `window_mem = 128`（GTrXL 记忆库；actor 真实 attend）。
- **burn-in 长度** = `B = window_mem = 128`。re-burn-in：用**当前参数**对采样序列前 B 步跑 `model_forward_eval` 逐步重建记忆（R2D2 式），其记忆 carry 在 B 边界 `stop_gradient` 后喂入 loss 区；burn-in 步不计 loss。**不存 memory_sequence**（磁盘杀手根因，审计§11）。
- **loss 区间** = 步 `[B, L_seq)`，按 `window_grad = 64` 切窗（6 窗 @ L_seq=512）。每窗 GTrXL train forward 前缀 128 记忆。
- **序列长度** `L_seq = 512`（可配置）。采样规则：episode 长度 ≤ L_seq → 用整条（真·整轨迹）；否则采样 contiguous L_seq 窗口，要求其前缀有 ≥B 的真实历史（即 start_step ≥ 0，burn-in 用窗口内前 B 步重建）。
- **每次 replay 更新采样 K = 4 条序列**，batched scan。1 次 replay actor-critic 更新 / 1 次 PPO 主更新。
- **序列采样方法**：复用 `TrajectoryReplayBuffer.sample`（确定性 RandomState），仅完整 episode（done=True）、长度 >128。容量 **capacity = 64**（有界，磁盘见 §11）。

## 3. V-trace off-policy 校正（冻结公式，离散动作）

对采样序列 loss 区步 `t ∈ [B, L)`，behavior 策略 μ（存 `log_probs`），当前策略 π：
- 重要性比 `ρ_t = π(a_t|x_t)/μ(a_t)`，截断 `ρ̄_t = min(ρ̄, ρ_t)`（**ρ̄ = 1.0**），`c_t = min(c̄, ρ_t)`（**c̄ = 1.0**）。
- TD 误差 `δ_t = ρ̄_t · (r_t + γ·V_target(x_{t+1})·(1-done_t) − V_online(x_t))`，γ=0.999。
- **V-trace 目标**（逆序递推）：`v_t = V_online(x_t) + Σ_{k=t}^{L-1} γ^{k-t} (Π_{i=t}^{k-1} c_i·(1-done_i)) δ_k`；终止后不 bootstrap（done→0）。`v_t` 裁剪到 `[vt_clip_min, vt_clip_max]=[-50, 300]`。
- **TD target / value loss**：`L_value = 0.5·mean_t (V_online(x_t) − sg(v_t))²`（sg=stop_gradient）。
- **Actor replay loss**（IMPALA V-trace 策略梯度）：
  `L_actor = −mean_t [ log π(a_t|x_t) · ρ̄_t · (r_t + γ·sg(v_{t+1})·(1-done_t) − sg(V_online(x_t))) ] − ent_coef·mean_t H(π_t)`。
- **总 replay 损失**：`L_replay = L_actor + vf_coef·L_value`（vf_coef=0.5, ent_coef=0.002）。
- `V_online` = 当前（在线）网络；`V_target` = **EMA target 网络**（§4）。
- behavior logprob 必用（μ），禁止把旧数据当 on-policy。

## 4. Target / EMA 网络策略（冻结）

- **EMA target 网络**：维护一份参数 EMA 副本，用于 TD 目标里的 `V_target(x_{t+1})` 与 bootstrap。
- 更新规则：每个 PPO+replay 更新后 `θ_target ← τ·θ_target + (1−τ)·θ_online`，**τ = 0.995**（软更新）；初始化 `θ_target = θ_online`（ckpt17500）。
- EMA target 参数进 checkpoint（exact resume）。**不**对其做梯度更新。
- 在线网络 `V_online` 用于 loss 里的 `V_online(x_t)` 与 actor 前向。
- 备用冻结选项（默认关）：周期硬拷贝 target（每 K 更新）；本方案冻结软 EMA。

## 5. Hindsight 重标规则（复用，Gate4）

- 每次 replay 更新，对每条采样序列调用 `relabel_sample`：从该轨迹**字面达成**的 achievement 集合中选目标（默认最小索引，或按覆盖度采样），替换 obs/next_obs 尾部 67 维 goal multi-hot，重算 `r_g[t]=max(ach[t,g]−ach[t−1,g],0)`，更新 target_achievements。
- 重标后**重跑 V-trace**：reward 变 → δ_t/v_t 变 → value target 变 → actor loss 变 → actor 梯度变。
- Gate5（正向）只用真实达成目标；Gate6（负向）拒绝伪造/未达目标（`ValueError`）。无达成目标时该序列跳过 hindsight（用原 reward），不伪造。
- Gate4 测试须证同一轨迹重标前后：goal 不同、reward/return 不同、actor loss 不同、value target 不同、actor 梯度不同。

## 6. 受控 Actor replay 更新（新硬门 REPLAY_ACTOR_UPDATE_CONTROLLED）

旧"replay 不得改 actor"硬门**取消**。新硬门要求每次 replay actor 更新同时满足（否则该次更新 fail-closed、不写坏 checkpoint、非零退出）：
- `actor_params_finite_nonzero_change`：actor 头叶子有限且非零变化。
- `trunk_params_finite_nonzero_change`：transformer/encoder 叶子有限且非零变化。
- `value_params_changed`：critic 头变化。
- `policy_kl_finite_below_threshold`：replay 序列上 `KL(π_before ∥ π_after)` 有限且 < **KL_MAX = 0.05**（每序列均值）。
- `importance_ratio_bounded`：`ρ_t` 截断在 ρ̄=1.0；记录 raw ratio max/mean、ESS fraction。
- `policy_lag_reject_gate`：`lag = update_count − collected_update_count > MAX_POLICY_LAG(=16)` → 丢弃该序列 actor 项（仍记录诊断），不强行更新。
- `grad_norm_clipped`：`clip_by_global_norm(1.0)`；梯度非有限 → fail-closed。
- `replay_loss_weight_capped`：replay 总损失权重 `w_replay ≤ W_REPLAY_MAX = 1.0`（相对 PPO 主更新），冻结 `w_replay = 0.5`。
- `no_one_step_collapse`：更新后在固定探针 obs 上 entropy 不崩塌（> ENT_FLOOR=0.05）、logits 有限。
- PPO/行为采集路径在 replay 后仍可运行（下一次 rollout 正常）。

## 7. 训练循环（每个 update）

1. `collect_rollout`（on-policy，16 env × 128 步，行为采集；持久 pending 收整 episode；存 behavior logprob + policy version；**不存 memory_sequence**）。
2. 完成 episode 插入 `TrajectoryReplayBuffer`（capacity 64）。
3. `compute_on_policy_gae` + `ppo_update`（原生 PPO 主更新 = Control 算法，更新 actor+trunk+critic）。`update_count += 1`。
4. 若 `replay.can_sample()`：采 K=4 序列 → 逐条 hindsight relabel → re-burn-in + V-trace actor-critic 联合更新（§3/§5/§6，受控门）。`update_count += 1`（aux 计数独立于 gradient_updates）。
5. EMA target 软更新（§4）。
6. 记 JSONL：ppo_* 与 replay_*（loss/grad_norm/KL/IS/ESS/lag/hindsight_goal/seq_len）+ 守恒计数。
7. 前沿步（0/24576/49152/73728/98304）写 checkpoint（§8）+ 滚动剥离旧前沿 replay_meta。

## 8. Checkpoint schema（兼容 + exact resume）

每个前沿 checkpoint 目录 `<step>/`：
- `default/`（orbax）：`{params, opt_state}`（在线 TrainState，bit-exact）。
- `ema_params/`（orbax 或 npz）：EMA target 参数。
- `replay_meta.pkl`（pickle）：replay buffer state（**无 memory_sequence**，仅 obs/act/rew/done/val/logp/next_obs/ach/init_mem）+ rng_key + action_rng_state + global_step + update_count + gradient_updates + pending_state + collector_state（obsv/env_state/memories/mem_mask/mem_idx）。
- `manifest.json`：step/global_step/update_count/gradient_updates/counters/replay_size/gamma/gae/gpu_uuid/各 SHA。
- **滚动剥离**：新前沿写完后删上一前沿的 `replay_meta.pkl`（评估/续训只需 params + 新前沿 replay）；峰值 ≈ 一份 buffer。
- **exact resume**：params+opt+EMA+rng+action_rng+collector+pending+global_step+update_count bit-exact；replay buffer 内容 best-effort（前沿重载，文档注明）。
- restore/verify 必须 GPU0（GPU-saved orbax 不能 CPU 恢复）。

## 9. 旧 checkpoint 兼容初始化（ckpt17500）

- Full P2 网络 = 与 ckpt17500 **完全相同**的 `ActorCriticTransformer`（同 action_dim=43/encoder_size=256/num_heads=8/qkv=256/num_layers=2/gating）。
- 加载：`load_weights_only(ckpt17500)` → **80/80 叶子全部匹配加载，无未加载叶子，无新初始化叶子**（`compatible_weight_restore_report` 记录 restored=80 / skipped=[] / newly_initialized=[]）。
- EMA target 初始 = 加载后的在线参数。优化器**重置**（fresh Adam，不继承 ckpt17500 opt_state，文档注明 = 兼容初始化非优化器续训）。
- **bit-exact 验证**：固定 `obs=0, memory=0, mask=1`（batch=2）下，加载后 policy 必须复现 ckpt17500 参考：`value=3.5761, entropy=0.9791, top action=12(0.718)`；params SHA256 == `5dfe67dd…`。
- **不静默跳过任何不匹配参数**：若叶子数/形状不符 → fail-closed。

## 10. Gate 1–4 测试清单（CPU 单测 + GPU smoke）

**Gate 1 — 长上下文前向**（通过前禁止 Actor replay）：
- G1.1 历史前缀不同 → 后续 logits 不同（同 obs，不同 burn-in 前缀 → 不同 logits）。
- G1.2 padding 内容不影响有效区间（改 mask 外 padding → 有效步 logits 不变）。
- G1.3 episode A 不污染 episode B（done 清零记忆；两条 episode 拼接 vs 分开 → 各自 logits 一致）。
- G1.4 re-burn-in 与 rollout 记忆一致：对同一段 obs，re-burn-in 重建的记忆 == 在线 rollout 产生的记忆（bit-exact）。
- G1.5 interrupted/resumed bit-exact（checkpoint round-trip 后前向一致）。
- G1.6 长序列前向序列长度 >128（真实读 >128 步）。

**Gate 2 — Sequence TD / Value 学习**（暂不开 Actor replay）：
- G2.1 采样序列长度 >128；re-burn-in 正常。
- G2.2 V-trace TD target 有限（无 NaN/Inf）且在 [-50,300]。
- G2.3 bootstrap 边界正确：done 后不 bootstrap（δ 用 0）；非终结用 EMA V_target(x_{t+1})。
- G2.4 mask/padding 不进 loss（loss 仅在 [B,L) 有效步）。
- G2.5 EMA target 正确（软更新 τ=0.995；初值=在线）。
- G2.6 value 与 long-context（encoder+transformer）参数有限非零梯度。
- G2.7 确定性 V-trace 参考比对（手写 numpy V-trace vs 实现，固定 ρ/c/r/v）。
- G2.8 无 NaN/Inf。

**Gate 3 — 受约束 Actor replay**：
- G3.1 一次 replay actor update 后：actor 参数有限非零变化、trunk 有限非零变化、value 变化。
- G3.2 policy KL 有限且 < KL_MAX=0.05。
- G3.3 IS ratio 截断（ρ̄=1.0）、记录 raw ratio/ESS。
- G3.4 policy-lag 拒绝门：lag>16 的序列 actor 项被丢弃（诊断仍记录）。
- G3.5 grad-clip(1.0)、loss 权重 ≤ W_REPLAY_MAX。
- G3.6 无一步坍塌（entropy>0.05、logits 有限）；PPO 采集路径仍可运行。
- G3.7 `REPLAY_ACTOR_UPDATE_CONTROLLED=true`。

**Gate 4 — Hindsight 联合训练**：
- G4.1 同一轨迹重标前后：goal 不同、reward/return 不同、actor loss 不同、value target 不同、actor 梯度不同（逐条断言 not allclose）。
- G4.2 Gate5 正向（只用真实达成目标）/ Gate6 负向（伪造目标 raise）。
- G4.3 重标不改 behavior logprob / policy version（IS 仍有效）。

**Gate 5 — Full P2 集成 smoke（GPU0）**：
- 1 update / 2048 步 → 2 updates / 4096 步。核验：PPO 采集正常；replay 真实采样；hindsight accepted>0（有 eligible 数据时）；Actor replay update>0；long-context 梯度>0；value 梯度>0；KL/IS/ESS/lag 有限；trajectory 守恒（collected==inserted）；checkpoint round-trip bit-exact；无 NaN/Inf。
- **4096 步无 eligible 长轨迹时不降门槛**：报告 `DATA_UNREACHABLE`（说明 episode 长度分布），而非改阈值。

## 11. 磁盘 / 显存峰值估算

- **参数**：4,906,028 × 4B ≈ 19.6 MB；Adam opt_state（2 矩）≈ 39 MB；EMA target ≈ 19.6 MB。
- **单个 checkpoint**：params+opt（orbax）≈ 60 MB + ema ≈ 20 MB = ~80 MB；5 个前沿 ≈ 400 MB。
- **Replay buffer（无 memory_sequence，RAM 驻留）**：obs+next_obs ≈ 67 KB/步；长 episode L≈2000–4096 → 单条 ~134–275 MB + init_mem 256 KB。capacity=64 → **~9–18 GB（RAM，122 G 充裕）**。
- **磁盘峰值**：滚动剥离后仅最新前沿 `replay_meta.pkl`（≈ 一份 buffer ~9–18 GB）+ 5 params checkpoint（~400 MB）≈ **~10–19 GB**。当前 /home 可用 50 G → 可行。
- **磁盘看门狗**：可用 < **12 GB** 立即暂停（停止条件）。
- **显存**：GTrXL ~5 M 参数 + PPO 主更新（16 env×128 步窗口化）已在上轮 GPU0 正常运行；replay 增加 K=4×L_seq=512 的扫描前向/反向（≈ 2048 步，与 PPO 主更新同量级）→ 预估总显存 < 上轮峰值，GPU0（充裕）可承载。Level A smoke 实测确认。
- **序列长度上限**：L_seq=512 截断超长 episode 采样窗口，bound 单次 replay 显存/时间。

## 12. 开发文件清单（写入 p2_full_20260723/，不改 frozen 源码）

```
p2_full_20260723/
  reports/p2_full_architecture_audit.md      (本报告1)
  reports/p2_full_frozen_design.md           (本报告2)
  src/
    vtrace.py                V-trace 目标/损失（ρ/c 截断，逆序递推，确定性）
    full_p2_learner.py       Full P2 学习器：re-burn-in 长序列前向 + V-trace actor-critic
                             联合更新 + 受控门 + EMA target（复用 long_context_learner 的
                             窗口/mask/GAE/PPO 主更新；移除 critic-only 隔离）
    full_p2_core.py          collect_rollout(去 memory_sequence 存储) + full_p2_update 编排
    replay_buffer.py         复用 trajectory_replay，Trajectory 去 memory_sequence，capacity=64
    hindsight.py             复用（拷贝，Gate5/6 不改）
    checkpointing.py         复用 + EMA target + 滚动剥离
    pending_episodes.py      复用（去逐step mem_pre/mask_pre）
    rng_utils.py             复用
    compat_init.py           ckpt17500 兼容初始化 + bit-exact 验证 + 叶子报告
  tests/
    test_gate1_longctx.py test_gate2_vtrace_value.py test_gate3_actor_replay.py
    test_gate4_hindsight.py test_vtrace_reference.py test_compat_init.py
    test_checkpoint_roundtrip.py test_replay_conservation.py
  commands/
    run_control_calibration.sh   Control 健康校准（lr 网格）
    run_fullp2_smoke.sh          Gate5 集成 smoke（2048/4096）
    run_short_screen.sh          Level B 24576 短程筛选
    run_formal_train.sh          Level C 98304 正式训练
    run_formal_eval.sh           Level D 512+512 正式评估
  logs/  evidence/
```
Henry 基座 / session175 / ckpt17500 / P2-v0 / P2-v1-lite 全只读，import 不修改。

## 13. Residual long-context adapter（冻结关闭，Phase-2 可选）

§五建议的 `base_feature + long_context_residual`（residual 初值 0）作为**冻结关闭**的扩展保留：仅当 Full P2 v1（原生 128 记忆 + V-trace）显示方向性信号、且证据表明"记忆长度=128 是限制因素"时才启用。届时新增长上下文模块（如更长记忆/递归摘要）residual-zero 接入，新增叶子在兼容初始化报告中标记 newly_initialized，初始 policy 仍 bit-exact 等于健康 Student（residual=0）。**v1 不实现，避免同时多套算法。**

## 14. Control 健康协议（§八，先校准再冻结）

- 起点 ckpt17500，原始 Henry GTrXL-PPO（= `native_ppo_loss`/`ppo_update`，replay/hindsight 关），静态 S4_dark 训练任务。
- **协议网格（预先定义，禁止看结果后无限调参）**：LR ∈ {2e-4（原默认）, 6e-5（0.3×）, 2e-5（0.1×）}；其余超参冻结（γ0.999/λ0.8/clip0.2/vf0.5/ent0.002/gradnorm1.0/num_envs16/num_steps128）。
- 分段保存 2048/4096/8192/12288/24576；固定 64 worlds 工程评估；记 policy KL/LR/grad norm/entropy/SR。
- **Control 保留门（开跑前冻结）**：无 NaN/Inf；floor3 reach 不灾难性坍塌；S4_dark SR 不低于 Baseline 预设保留比例（冻结：SR ≥ 0.5×Baseline 且 floor3_reach ≥ 0.5×Baseline）；policy KL ≤ KL_MAX_RUN=0.1（累计）。
- 选定首个满足保留门的 LR（按 原默认→0.3×→0.1× 顺序）即冻结，用于 Full P2 的 on-policy 基础部分（同一 LR）。
- 若三档 LR 全灾难性坍塌 → 停止正式比较，报告 `CONTROL_PROTOCOL_UNHEALTHY`。

## 15. 实验组（§七）

- **A. Baseline**：ckpt17500，不训练。
- **B. Control**：ckpt17500 + 原始 Henry GTrXL-PPO（冻结 LR）+ 静态 S4_dark。
- **C. Full P2**：ckpt17500 兼容初始化 + 长上下文 + 整轨迹 replay + hindsight + off-policy actor/value（同冻结 LR）。
- 主比较 C vs B；辅 C vs A、B vs A。一致项：训练任务/env 数/训练 seed/env steps/γ0.999/λ0.8/checkpoint 时刻/评估世界/max episode steps/obs-action 接口。仅 C 新增长上下文+off-policy replay。

## 16. 训练评估阶梯（§九）与成功判据（§十）

- **Level A**：CPU 单测（Gate1–4）→ 2048 GPU smoke → 4096 activation smoke。通过才训练。
- **Level B**：Control 与 Full P2 各 24576 步，同 64 Stage4 worlds 评估（DK/ENTER_SEWERS/floor3/conditional kill/death-timeout/ep len/policy KL/death-to-stair/上游保持）。同时满足（Control 健康 + Full P2 无灾难退化 + replay actor 路径真实工作 + Full P2 方向性改善或失败行为明显改善）才扩长。
- **Level C**：98304 步，存 0/24576/49152/73728/98304。
- **Level D**：512 fresh S4_dark + 512 fresh Official FULL（自然 floor0 出生，无脚手架/arrival reset），每 episode ≤4096 步。
- **成功判据**：S4_dark SR(FullP2)−SR(Control) ≥ 8pp；Official FULL Tier3 自然破零（ENTER_SEWERS>0 且 DEFEAT_KOBOLD>0）；能力保护（floor2 不崩、Tier1/2 保持、不显著低于 Baseline）；行为证据（floor3↑、death-to-stair↓、覆盖↑、重复走旧路↓、条件击杀不恶化）；复现（二批 512 worlds 或第二 seed 再 Tier3 非零）。单次 1/512 仅记 `TIER3_BREAKZERO_SIGNAL`，不称稳定突破。

## 17. 停止条件（§十一，立即暂停）

Control 协议不健康 / NaN/Inf / checkpoint 不可恢复 / replay 轨迹守恒失败 / episode 边界污染 / off-policy ratio 失控 / Actor KL 超冻结阈值 / Actor 一步更新后灾难性漂移 / GPU UUID 绑定错误 / 输出目录串线 / 旧 checkpoint 被修改 / 磁盘 < 12 GB。**不自动改算法规避失败。**

## 18. Token / 监控（§十二）

无子 Agent；禁高频 nvidia-smi/ps/tail；训练本地阻塞等待；公共源码只读一次；测试审计批量；指标写 JSONL/CSV；不逐 Gate 长篇汇报。汇报节点：①审计+冻结设计完成（本次）②Gate1–4 全过或 BLOCKED ③4096 smoke 完成 ④24576 短程完成 ⑤98304+512 评估完成。中间只报 PASS/FAIL、核心指标、异常、证据路径、SHA256、下一动作。
