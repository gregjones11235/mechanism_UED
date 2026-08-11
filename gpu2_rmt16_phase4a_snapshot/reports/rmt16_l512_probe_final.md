# RMT16 L512 可达性探针 — 最终报告（NOT FOR FORMAL SCIENCE）

**裁定标签：`L512_REACHABILITY_BOTH`**

探针**只回答**：固定 16384 预算内，完整 episode 能否达到 512 步，以及直接终止原因。
**不回答**（明确非结论）：Persistent 更优 / Reset128 更优 / Carry 有效 / Carry 无效 / RMT 提升性能。

- PROBE_SOURCE_COMMIT（探针执行代码，字节一致 PASS）：`0b405dd85224a368d4029ff6f7818c3855d9487b`
- base commit：`64546e0be9ae0aafd235bb8db1f6675460d22adf`；remote：`https://github.com/gregjones11235/mechanism_UED.git`；branch：`henry/rmt16-l512-reachability-probe`
- 冻结 step0 params SHA：`2f8cd875993ae10385dbb5dae530a557a0eb1008541b98de416cc7ae7ba2d93b`（两臂活体复现）
- ckpt17500 base SHA：`d4e85af58b7f87d6…`；seed=42；replay OFF；hindsight OFF；online PPO only。
- 启动 UTC：2026-07-26T11:26:35Z。GPU0/GPU1 全程未触碰。

### PERSISTENT (carry_mode=persistent, GPU-8df11537-ab79-722d-606f-411966196c4c)
- updates / env_steps / online_ppo: 8 / 16384 / 8
- completed episodes: 20
- count_ge_512 / fraction: 6 / 0.3
- first_ge512: {"first_ge512_update": 4, "first_ge512_global_step": 8241, "first_ge512_episode_id": 2, "first_ge512_length": 562}
- termination_reason_counts: {"player_death": 16, "task_success": 4}
- episode lengths (sorted): [49, 70, 85, 127, 128, 147, 155, 186, 255, 383, 395, 397, 428, 466, 562, 570, 586, 726, 753, 792]
- final_params_sha256: 38b080142bbf92b1b6aee29ca8906045a06de94def383a7b207dfd74961dbf9a
- base_sha256 (ckpt17500): d4e85af58b7f87d689fadea12eec70c852fa098a09f5ea8907448684b3bf60f5
- early_stop_used: False
- end_utc: 2026-07-26T11:41:25Z

### RESET128 (carry_mode=reset128, GPU-f56a59b4-99f3-f2e5-11c6-d01685de8abd)
- updates / env_steps / online_ppo: 8 / 16384 / 8
- completed episodes: 21
- count_ge_512 / fraction: 5 / 0.23809523809523808
- first_ge512: {"first_ge512_update": 4, "first_ge512_global_step": 8241, "first_ge512_episode_id": 2, "first_ge512_length": 562}
- termination_reason_counts: {"player_death": 17, "task_success": 4}
- episode lengths (sorted): [49, 70, 85, 127, 128, 147, 155, 192, 251, 255, 273, 376, 383, 397, 428, 466, 562, 570, 586, 726, 792]
- final_params_sha256: 6ccfe16aa4f6f1d1cff8e9f59a770467575d0edbf1497eff3a25bba860c56f14
- base_sha256 (ckpt17500): d4e85af58b7f87d689fadea12eec70c852fa098a09f5ea8907448684b3bf60f5
- early_stop_used: False
- end_utc: 2026-07-26T11:40:31Z

## 13 项完成核验（每臂）
```json
{
  "PERSISTENT": {
    "update_count_eq_8": true,
    "total_env_steps_eq_16384": true,
    "online_ppo_update_count_eq_8": true,
    "replay_update_count_eq_0": true,
    "hindsight_update_count_eq_0": true,
    "early_stop_triggered_false": true,
    "seed_42": true,
    "source_sha_eq_PROBE_SOURCE_COMMIT": true,
    "step0_sha_eq_frozen": true,
    "unique_output_dir": true,
    "no_resume_fresh_step0_from_ckpt17500": true,
    "no_silent_fallback_wrapper_raises": true,
    "full_episode_lengths_and_termination_recorded": true,
    "base_sha256_is_ckpt17500": true
  },
  "RESET128": {
    "update_count_eq_8": true,
    "total_env_steps_eq_16384": true,
    "online_ppo_update_count_eq_8": true,
    "replay_update_count_eq_0": true,
    "hindsight_update_count_eq_0": true,
    "early_stop_triggered_false": true,
    "seed_42": true,
    "source_sha_eq_PROBE_SOURCE_COMMIT": true,
    "step0_sha_eq_frozen": true,
    "unique_output_dir": true,
    "no_resume_fresh_step0_from_ckpt17500": true,
    "no_silent_fallback_wrapper_raises": true,
    "full_episode_lengths_and_termination_recorded": true,
    "base_sha256_is_ckpt17500": true
  }
}
```
all_13_items_pass_all_arms = True

## 退出码说明
exit codes NOT captured to file (launch used bare `nohup &`); exit 0 INFERRED per arm from: PROBE_SUMMARY emitted (driver prints it immediately before sys.exit(0)) + 8/8 updates + global_step=16384 + no Traceback/NaN/HARD STOP. If literal exit codes are required, this is a known limitation of the current launch wrapper.
