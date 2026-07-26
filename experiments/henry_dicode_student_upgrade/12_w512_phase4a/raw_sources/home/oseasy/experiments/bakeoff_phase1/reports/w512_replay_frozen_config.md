# W512 × P2 Replay — 冻结配置报告

来源：P2-Full-A (p2_full_20260723/src/)
冻结时间：2026-07-25
状态：FROZEN — 不得调参

## 源码SHA256

| 文件 | SHA256 |
|------|--------|
| run_p2_full_levelB.py | 4c2fbb273dfbab6abddc2d7618a9cf3fbbbb00e9e20088b5d2e01dfc4f731449 |
| full_p2_learner.py | c374f0aa3ce4ad284a2ed7d7f0f96ce0c4db754f21babcca47b3d0d80e6d83b4 |
| full_p2_core.py | 2e0fa3c67e5e78aa9ad11efd96d6e654adb8458ee895b026cb63c570e16ad09f |
| replay_buffer.py | c36c95b406031c0267d0458bb21745485e5803855f13d10c9417a37f69f06216 |
| vtrace.py | 1eb3a5cb289bfe73ce81d32dcde178d848a3c545bc3a46c34b5c659b7d90f7c5 |
| awr.py | 3ae7fa6dcea0dccc374e2cea03973f28a4d4e737812995dd07ca1619a2f09943 |
| hindsight.py | 1fc01d1c4a4843c6b4debd895a9efcc01985f4eaa6e7ed049b58280be68c9efb |
| memory_anchor.py | 49ac6241dac2df48654c625f71caec0545d9e37766816ffd763acd155343f8fe |
| pending_episodes.py | 8ce77dd3cabf8278b3f0633cd16a0d359fc0e29824d3f5879700f4324ebe3d31 |
| checkpointing.py | 93a526487081b2164c59c26ba49eb61ab7b123a38bc9765fccaf4c76ad0d3442 |
| rng_utils.py | 1894bc30901c5b0d5ebcc46b2e2d0a32ec43cd43fb087fc3c90934a8ca2de447 |

## Replay配置（从P2-Full-A冻结代码直接提取）

### Buffer
- **类型**: whole-episode ring buffer (ReplayBuffer)
- **capacity**: 64 完整episode
- **seed**: 42

### 采样
- **K_BATCH**: 4 条连续序列/update
- **L_SEQ**: 129 (loss-window length)
- **MIN_SEQUENCE_LENGTH**: 129 (strictly > 128)
- **replay transitions/update**: 4 × 129 = 516

### 更新频率
- **warmup**: 无显式warmup；replay在buffer中首个length>=129的episode出现后开始
- **replay频率**: 每个rollout（128步）后尝试一次replay update（当can_sample()=True）
- **UNREACHABLE_GUARD_ROLLOUTS**: 6（连续6个rollout有eligible数据但无update则HARD STOP）

### V-trace
- **rho_bar**: 1.0 (importance ratio truncation)
- **c_bar**: 1.0 (trace-cutting truncation)
- **gamma**: 0.999
- **vt_clip_min**: -50.0
- **vt_clip_max**: 300.0
- **w_vtrace**: 0.5

### AWR (hindsight)
- **beta**: 1.0 (temperature)
- **w_max**: 20.0 (weight cap)
- **lambda_kl**: 0.01 (KL regularization)
- **w_awr**: 0.5

### Loss组合
```
loss = w_vtrace * (vtrace_actor + vf_coef * vtrace_value)
     + w_awr   * (awr_actor    + vf_coef * awr_value)
```

### KL门控
- **kl_replay_max**: 0.05 (单次replay update KL门；transactional rollback)
- **kl_run_max**: 0.10 (累积run-level KL上限)
- **actor_step_scales**: (1.0, 0.5, 0.25, 0.125) (KL breach时重试)

### EMA target
- **ema_tau**: 0.995

### Policy lag
- **max_policy_lag**: 16

### 优化器
- **lr**: 2e-5 (constant)
- **adam_eps**: 1e-5
- **grad_clip**: 1.0

### PPO主更新
- **ent_coef**: 0.002
- **ent_floor**: 0.05
- **vf_coef**: 0.5

### 环境
- **NUM_ENVS**: 16
- **ROLLOUT_STEPS**: 128
- **STEPS_PER_ROLLOUT**: 2048
- **OPTIMISTIC_RESET_RATIO**: 16
- **MASTER_SEED**: 42
- **total_steps**: 24576

### Checkpoint
- **SAVE_STEPS**: (0, 4096, 8192, 12288, 24576)

## 与用户指令的差异

用户指令中列出 "replay sequence length=512" 和 "replay transitions/update=4×512=2048"。
实际P2-Full-A冻结代码中 L_SEQ=129, K_BATCH=4, transitions/update=516。
**本报告以实际冻结代码为准。** 若需使用512，须用户明确确认覆盖。

## 冻结声明

以上所有数值从P2-Full-A冻结源码直接提取，不得调参。
W512×Replay实验必须严格复用上述配置。
