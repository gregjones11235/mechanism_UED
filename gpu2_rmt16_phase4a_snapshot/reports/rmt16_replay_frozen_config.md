# RMT16 × P2-Replay — 冻结 Replay 配置对照（逐项 vs P2-Full-A v2.1）

- 阶段: RMT16_MEMORY_CARRY_X_P2_REPLAY_PHASE4A
- 负责人: CC2/总监B（仅 GPU2/GPU3）
- 来源权威: `p2_full_20260723/reports/p2_full_frozen_design.md`（v2.1，已完整读取）
- 原则: **完全复用 P2-Full-A 冻结 Replay 配置，不自行调任何系数**。RMT16 仅在 anchor 状态 schema 上做"网络适配扩展"（多带 16 个 RMT tokens + seg_buf），**不改动任何 Replay 数值系数**。
- 状态: 待 CC1 冻结 RMT16 网络/step0 SHA 后，复核 Replay 两臂网络与 step0 参数与 CC1 冻结版本逐位一致，方可启动 24576。

---

## 一、逐项对照表（指令第四节 ↔ P2-Full-A v2.1）

| # | Replay 配置项 | 指令要求 | P2-Full-A v2.1 冻结值 | 本实验采用 | 是否一致 |
|---|---|---|---|---|---|
| 1 | buffer 类型 | whole-episode buffer | `ReplayBuffer` 存完整 episode（done=True，长度>128） | 同 | ✅ 一致 |
| 2 | 容量 | capacity=64 episodes | `capacity = 64` | 64 | ✅ 一致 |
| 3 | 序列长度 | sequence length=512 | `L_seq = 512`（episode≤512 整条；否则 contiguous 512 窗口） | 512 | ✅ 一致 |
| 4 | 每次序列数 | 4 条连续序列/Replay update | `K = 4` 序列/update，batched | 4 | ✅ 一致 |
| 5 | 每次 replay transitions | 2048/update | K×L_seq = 4×512 = 2048 | 2048 | ✅ 一致（4×512 导出） |
| 6 | replay/PPO 频率 | （隐含 1:1） | 1 次 replay 更新 / 1 次 PPO 主更新 | 1:1 | ✅ 一致 |
| 7 | 原目标 V-trace Actor | original-goal V-trace Actor | IMPALA V-trace 策略梯度，ρ̄=1.0、c̄=1.0、γ=0.999 | 同 | ✅ 一致 |
| 8 | 原目标 V-trace Value | original-goal V-trace Value | `v_t` 逆序递推，clip[-50,300]；`L=0.5·mean(V_online−sg(v_t))²` | 同 | ✅ 一致 |
| 9 | 重要性比截断 | importance ratio | `ρ_t=π/μ`，`ρ̄_t=min(1.0,ρ_t)`，`c_t=min(1.0,ρ_t)`；behavior μ 必用 | ρ̄=c̄=1.0 | ✅ 一致 |
| 10 | ESS 监控 | importance ratio/ESS | 记录 raw ratio max/mean、ESS fraction | 同 | ✅ 一致 |
| 11 | hindsight AWR Actor | hindsight AWR Actor | 加权 BC：`w_t=min(w_max,exp(A'/β))`，β=1.0，w_max=20.0；λ_kl=0.01 软罚 + KL_MAX_AWR=0.05 硬门；**无 IS ratio** | 同 | ✅ 一致 |
| 12 | hindsight AWR Value | hindsight AWR Value | 重标 TD 回报 G'，clip[-50,300]；`L=0.5·mean(V'_online−sg(G'))²`，无 ratio | 同 | ✅ 一致 |
| 13 | hindsight 目标规则 | （复用 P2） | `relabel_sample` 只取字面达成 achievement；Gate5 正向 / Gate6 拒绝伪造；无达成→跳过 hindsight（仅走原目标 V-trace） | 同 | ✅ 一致 |
| 14 | 合并 replay loss | （隐含） | `L_replay = w_vtrace(L_vtrace_actor+vf·L_vtrace_value) + w_awr(L_awr_actor+vf·L_awr_value)`；w_vtrace=0.5，w_awr=0.5，vf_coef=0.5，ent_coef=0.002，w_vtrace+w_awr≤W_REPLAY_MAX=1.0 | 同 | ✅ 一致 |
| 15 | EMA target | EMA target | τ=0.995，初值=在线（ckpt17500）；每次 PPO+replay 后软更新；进 checkpoint；不做梯度 | 同 | ✅ 一致 |
| 16 | policy lag | policy lag | `lag=update_count−collected_update_count > MAX_POLICY_LAG(=16)` → 丢该序列 actor 项（V-trace+AWR），诊断仍记 | MAX_POLICY_LAG=16 | ✅ 一致 |
| 17 | 事务 KL 门 | transactional KL gate ≤0.05 | `kl_replay_max=0.05`；超限按 actor 步长 {1.0,0.5,0.25,0.125} 重试；首个过门提交，全失败→`KL_REJECTED_UPDATE` 回滚 policy-affecting params+对应 opt_state+基于该候选的 EMA；critic-only head 独立提交 | 同 | ✅ 一致 |
| 18 | 三类 KL 分离 | （隐含） | kl_replay_max=0.05（单步）/ kl_run_max=0.1（24576 累计）/ kl_baseline_cumulative（诊断）。0.05 绝不约束累计 | 同 | ✅ 一致 |
| 19 | grad clip | （隐含） | `clip_by_global_norm(1.0)`；梯度非有限→fail-closed | 1.0 | ✅ 一致 |
| 20 | 一步坍塌保护 | （隐含） | 更新后探针 entropy>ENT_FLOOR=0.05 且 logits 有限 | ENT_FLOOR=0.05 | ✅ 一致 |
| 21 | KL 参数覆盖 | （隐含 §19.1） | 事务门覆盖 encoder+GTrXL/shared trunk+goal/context+actor head 全部 policy-affecting 参数；critic-only(critic_ln1/ln2/out) 独立 | 同（RMT16 的 rmt_read/update_attn、rmt_read/update_ln、rmt_gate 均参与 actor 前向 → 归 policy-affecting，受门约束） | ✅ 一致（RMT 新模块归入 policy-affecting） |
| 22 | 采样规则 | 冻结 P2 采样 | 确定性 RandomState，仅完整 episode、长度>128；**不加成功/质量/TD-error 优先采样** | 同 | ✅ 一致（明令不新增优先采样） |
| 23 | 稀疏 memory anchor | 每128步精确 anchor | 每128步存 pre-action GTrXL memory anchor；episode 起点必有 anchor；replay 从最近 anchor 重放≤128步重建；anchor 记忆 stop_gradient；不存逐步 memory_sequence | 同（**schema 扩展见第二节**） | ✅ 系数一致；schema 适配扩展 |
| 24 | 不跨 true done | 不跨 episode | replay 序列不跨 episode；done 后不 bootstrap | 同 | ✅ 一致 |

**结论：24/24 项 Replay 数值系数与 P2-Full-A v2.1 完全一致，零自调系数。** 唯一差异是 anchor 状态 schema 的 RMT16 适配扩展（见下），属网络结构适配，非 Replay 算法/系数改动。

---

## 二、RMT16 anchor schema 适配扩展（唯一与 P2-Full-A 不同处，非系数改动）

P2-Full-A 的 anchor 仅含 GTrXL 短记忆 `memories[window_mem=128, layers=2, embed=256]`。RMT16 在同 GTrXL 之上**额外**有持久记忆模块，故 anchor 必须**同时**快照以下全部状态，否则恢复后首次前向无法逐位一致：

| anchor 字段（每128步、pre-action） | 形状/含义 | P2 是否已有 | 说明 |
|---|---|---|---|
| `gtrxl_memories` | [128, layers, embed] GTrXL 短记忆 | ✅ 已有 | 与 P2 完全相同 |
| `rmt_mem_tokens` | [16, 256] 16 个 RMT persistent memory tokens | ➕ RMT 新增 | **核心**：Persistent 臂恢复真实 tokens；Reset128 臂在边界按定义置零 |
| `rmt_seg_buf` | [128, 256] 段内累积缓冲 | ➕ RMT 新增 | update 每128步读它；anchor 须带其当前内容 |
| `rmt_seg_count` | 标量，段计数 | ➕ RMT 新增 | 决定 update 触发相位 |
| `episode_position` | 步 t（anchor_step） | ✅ 已有 | t∈{0,128,256,…} |
| `done_mask` | done/mask 状态 | ✅ 已有 | 不跨 true done |
| `rng_state` | 相关 RNG | ✅ 已有 | exact resume 逐位 |
| `behavior_logprob` / `policy_version` | 行为策略信息 | ✅ 已有 | V-trace μ；AWR 仅供诊断 |

- **anchor 数** `n_anchors = ceil(L/128)`，anchor_steps = [0,128,…]，episode 起点 anchor 存在（== 初始状态）。与 P2 相同。
- **重建**：对 loss window 起点 s，取最大 anchor_step≤s 的 anchor，用当前参数对 obs[anchor_step:s] 跑无梯度前向重放到 s（≤128步），得 pre-action 全状态（含 RMT tokens），再做 loss 区 train forward。anchor 状态 stop_gradient。
- **Persistent vs Reset128 唯一差异（指令第二节/第五节）**：
  - Persistent-Replay：恢复采集时真实 RMT tokens，并跨128步继续更新；512步 replay 序列内部的128步边界**不**清零 tokens。
  - Reset128-Replay：在每个128步窗口边界（**包括512步 replay 序列内部的边界**）按定义把 rmt_mem_tokens 置零，单个窗口内正常读写。
  - 两臂除 carry/reset 这一处外，所有代码/参数/Replay数据/超参逐位一致（config diff 只含 carry/reset）。

---

## 三、共同训练配置（指令第三节，两臂一致）

| 项 | 值 | 来源 |
|---|---|---|
| 健康起点 | ckpt17500（params SHA d4e85af58b7f87d6） | 与原 RMT16 相同 step0 |
| 主更新 | Original PPO（update_epochs=1, num_minibatches=2） | bakeoff |
| LR | 2e-5 | 指令 |
| Adam eps | 1e-5 | 指令 |
| γ / GAE λ | 0.999 / 0.8（严禁改动） | 指令 |
| clip / ent_coef / vf_coef / max_grad_norm | 0.2 / 0.002 / 0.5 / 1.0 | bakeoff |
| num_envs / rollout | 16 / 128 | 指令 |
| online transitions/update | 2048（=16×128） | 指令 |
| goal | DEFEAT_KOBOLD（Stage4-native） | 指令 |
| seed / det-ops | 42 / `--xla_gpu_deterministic_ops=true` | 指令 |
| total_steps | 24576（12 updates） | 指令 |
| 第二 seed / Official FULL | 无 / 无 | 指令 |
| 保存点 | 0/4096/8192/12288/16384/20480/24576 | 指令 |
| 自动续训到 98304 | **禁止** | 指令 |

---

## 四、18 工程硬门映射（指令第六节 → 测试覆盖）

| 门 | 内容 | 覆盖测试 |
|---|---|---|
| 1 | Persistent/Reset128 step0 参数逐位一致 | test_step0_bitexact |
| 2 | 网络 schema 完全一致 | test_schema_identical |
| 3 | config diff 只含 carry/reset | test_config_diff |
| 4 | 原 PPO 主更新未改变 | test_ppo_unchanged（feature-off bit-exact，gate=0） |
| 5 | episode buffer 守恒 | test_buffer_conservation（collected==inserted） |
| 6 | anchor/token roundtrip 逐位一致 | test_anchor_roundtrip |
| 7 | exact resume 逐位一致 | test_exact_resume（params/opt/EMA/rng/anchor bit-exact） |
| 8 | replay 序列不跨 episode | test_no_cross_episode |
| 9 | tokens 生命周期正确 | test_token_lifecycle（Persistent 跨边界保持 / Reset128 边界清零，含512内部边界） |
| 10 | vector env 不串线 | test_env_isolation（扰动 env0 只动 env0） |
| 11 | EMA/optimizer/replay/RNG 完整恢复 | test_resume_state |
| 12 | 无 NaN/Inf | test_finite |
| 13 | entropy 健康 | test_entropy_floor（>0.05） |
| 14 | tokens 与读写参数有 finite 非零梯度 | test_memory_grad |
| 15 | 所有 commit KL≤0.05 | test_kl_transactional_gate |
| 16 | rejected update 不污染状态 | test_reject_rollback（policy/opt/EMA bit-exact 回滚） |
| 17 | policy lag/ESS/ratio 监控有效 | test_lag_ess_ratio |
| 18 | Replay 和在线样本计数正确 | test_sample_counts |

先 4096 smoke（两臂），通过后再 24576。24576 阻塞于 CC1 冻结信息复核（指令第一节同步门）。

---

## 五、Replay 噪声审计字段（指令第七节，落盘）

buffer episode count / stored transition count / DK·floor3·death·timeout episode 比例 / episode 长度分布 / 各类型 episode 被采样次数 / 单一 episode 最大采样占比 / policy lag / ratio p50·p95·max / ESS / commit·reject 数量 / hindsight 有效率 / token norm·相似度·秩 / attention·gate / memory on·off action KL / top-action flip / 深 episode 中的 memory 读取强度。
采样规则用冻结 P2（不新增成功/质量/TD-error 优先）。
