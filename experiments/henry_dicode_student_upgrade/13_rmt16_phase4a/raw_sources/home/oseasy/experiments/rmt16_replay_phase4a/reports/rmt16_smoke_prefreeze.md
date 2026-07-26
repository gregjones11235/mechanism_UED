# RMT16 Phase4A — Smoke 预冻结清单 (prefreeze)

- 状态: **VALID_MATCHED_SMOKE**
- 生成时间(UTC): 2026-07-25T16:21:17Z
- 健康起点 ckpt17500: `/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/base_ckpt_v7fix55_armA_s0/rl_checkpoints/17500`
- seed: 42  EMB(n_achievements): 67  ACTION_DIM: 43  OBS_DIM: 8335

## 1-4. 代码 SHA (sha256)
| 模块 | SHA256 |
|---|---|
| 1. RMT16 网络 `network_rmt16.py` | `b5c37d7aa2e9cac1b4b395111262b4d8a11e20fd75a2930670336a68d86b8632` |
| 2. RMT16 memory 生命周期 `rmt16_memory.py` | `17e1a614c404e4edf176de7e8f9bd3f241059257fb24962d0df148960c7f6500` |
| 3. 训练器(4臂统一驱动) `train_rmt16_p2replay.py` | `0763abc46dd6aa066156b495a6d9aff312580f7e1d72568878b182dc973f3cc3` |
| V-trace(冻结) `vtrace.py` | `1eb3a5cb289bfe73ce81d32dcde178d848a3c545bc3a46c34b5c659b7d90f7c5` |
| AWR(冻结) `awr.py` | `3ae7fa6dcea0dccc374e2cea03973f28a4d4e737812995dd07ca1619a2f09943` |
| hindsight(冻结) `hindsight.py` | `1fc01d1c4a4843c6b4debd895a9efcc01985f4eaa6e7ed049b58280be68c9efb` |
| replay_buffer(冻结) `replay_buffer.py` | `c36c95b406031c0267d0458bb21745485e5803855f13d10c9417a37f69f06216` |
| memory_anchor(冻结) `memory_anchor.py` | `49ac6241dac2df48654c625f71caec0545d9e37766816ffd763acd155343f8fe` |
| full_p2_learner(冻结) `full_p2_learner.py` | `c374f0aa3ce4ad284a2ed7d7f0f96ce0c4db754f21babcca47b3d0d80e6d83b4` |
| pending_episodes(冻结) `pending_episodes.py` | `8ce77dd3cabf8278b3f0633cd16a0d359fc0e29824d3f5879700f4324ebe3d31` |
| rng_utils(冻结) `rng_utils.py` | `1894bc30901c5b0d5ebcc46b2e2d0a32ec43cd43fb087fc3c90934a8ca2de447` |
| 4a. Replay buffer(RMT扩展) `rmt_replay_buffer.py` | `21e31565a56f806c9fc96fbb9a1840c621d4ea8d0a8fec818ed72134614303cd` |
| 4b. Replay anchor/重建 `rmt_memory_anchor.py` | `92c56b6375878e789fae2fddee0bf5a4fef25ad4eec83e67ab8c91ec65ea68e8` |
| 4c. Replay collector `rmt_collect.py` | `291d5726866cb60cdccd0e28b6ade5b7212bac2efe9646a1155d9b68bd985057` |
| 4d. Replay learner `rmt_replay_learner.py` | `a575019f2833a10b38e348cb70c74a73c26ed756e3c26ecd5c03c44b0d878db5` |
| 4e. Replay hindsight `rmt_hindsight.py` | `2447a7981dedac417c9187a391f47b0b8486d33c5bd13830537ce25660073985` |
| PPO 主更新 `rmt_ppo.py` | `2d2d943e0ed16bb68ad37df44205ee07397d8751f92e6aa7d39c33509a524b3c` |

## 5-7. step0 参数
- 5. tree 顶层 keys (12): `['actor_ln1', 'actor_ln2', 'actor_out', 'critic_ln1', 'critic_ln2', 'critic_out', 'rmt_gate', 'rmt_read_attn', 'rmt_read_ln', 'rmt_update_attn', 'rmt_update_ln', 'transformer']`
- RMT 参数 keys: `['rmt_gate', 'rmt_read_attn', 'rmt_read_ln', 'rmt_update_attn', 'rmt_update_ln']`
- 6. 参数叶子数: **101**   参数元素总数: **5433389**
- 7. step0 参数 SHA256: `2f8cd875993ae10385dbb5dae530a557a0eb1008541b98de416cc7ae7ba2d93b`
   - build#2 SHA256: `2f8cd875993ae10385dbb5dae530a557a0eb1008541b98de416cc7ae7ba2d93b`  (确定性: 一致)
   - ckpt17500 base 内层 SHA256: `d4e85af58b7f87d689fadea12eec70c852fa098a09f5ea8907448684b3bf60f5`
   - 两臂 step0 公共参数: **逐位相同**（carry_mode 为运行时标志，不进入 params；两次独立构建 SHA 一致证明初始化确定性）

## 8. RMT token shape/dtype/初始化
- mem_tokens 状态 shape (per env): (16, 16, 256)  dtype=float32  init=全0
- seg_buf 状态 shape (per env): (16, 128, 256)  dtype=float32  init=全0
- seg_count 状态 shape (per env): (16,)  dtype=int32  init=0
- rmt_gate 参数 shape: (1,)  init=0.0  (实测 max|gate|=0.000e+00 -> bit-exact no-op at init)
- 读: cross-attn(query=h_t, kv=mem_tokens)+LN，每步；z_t = h_t + tanh(rmt_gate)·rmt_ctx
- 写: 每128步 mem_tokens ← mem_tokens + LN(attn(query=mem_tokens, kv=seg_buf))

## 9. GTrXL 短期 memory shape 与生命周期
- memories shape (per env): (16, 128, 2, 256)  (window_mem×num_layers×embed)
- mem_mask shape (per env): (16, 8, 1, 129)
- 生命周期: 每步 roll+写入 mem_out；true done 时清零并把 mem_idx 复位到 window_mem。这是健康 Student 原有语义，**两臂完全相同**，Reset128 臂不额外清空 GTrXL memory。

## 10. Persistent / Reset128 配置 diff
详见 `rmt16_persistent_vs_reset128_config_diff.txt`。唯一差异 = carry_mode (RMT tokens 在128步段边界 carry-updated vs zero)。无网络/loss/LR/Replay/optimizer 差异。

## 硬要求自检
- 两臂 step0 公共参数逐位相同: PASS
- 参数元素数完全一致: PASS (单一 params 构建，两臂共享, n=5433389)
- 参数 schema 完全一致: PASS (同一 treedef)
- config diff 仅 carry/reset 生命周期: PASS (见 diff 文件)
- rmt_gate=0 (bit-exact at init): PASS

**结论: VALID_MATCHED_SMOKE**
