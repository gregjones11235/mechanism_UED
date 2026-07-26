# W512 Phase4A Artifact Manifest
Generated: 2026-07-26T01:46:52Z
Total artifacts: 44

## Frozen Labels

- **W512_PPO_TRAINING_HEALTH** = FAILED
- **W512_REPLAY_STABILIZATION_SIGNAL** = True
- **W512_LONG_MEMORY_CAUSAL_CANDIDATE** = False
- **W512_PERFORMANCE_UPGRADE** = False
- **W512_PHASE4A_VERDICT** = REPLAY_ELIMINATES_CARRY

## Frozen Causal Quantities

| Quantity | Value |
|----------|-------|
| CARRY_NO_REPLAY | +8.20pp (p=6.3e-05) |
| CARRY_WITH_REPLAY | -1.95pp (p=0.42) |
| REPLAY_EFFECT_PERSISTENT | +24.22pp (p<1e-6) |
| REPLAY_EFFECT_RESET | +34.38pp (p<1e-6) |
| MEMORY_REPLAY_INTERACTION | -10.16pp |

## Frozen 6-Arm Results

| Arm | DK SR | n |
|-----|-------|---|
| Baseline | 39.45% | 101 |
| Control | 36.33% | 93 |
| W512_Persistent_PPO | 10.94% | 28 |
| W512_Reset128_PPO | 2.73% | 7 |
| W512_Persistent_P2Replay | 35.16% | 90 |
| W512_Reset128_P2Replay | 37.11% | 95 |

## Artifacts

| # | Type | Path | SHA256 (prefix) |
|---|------|------|------------------|
| 1 | evaluation_result | `/home/oseasy/experiments/bakeoff_phase1/reports/cc1_corrected_eval/a_side_unified_eval.json` | 2be1c5583382a509... |
| 2 | evaluation_result | `/home/oseasy/experiments/bakeoff_phase1/reports/w512_p2replay_eval.json` | 9e7c882e14fc3522... |
| 3 | evaluation_result | `/home/oseasy/experiments/bakeoff_phase1/reports/w512_p2replay_causal.json` | 8e7eacf5fa69c136... |
| 4 | evaluation_result | `/home/oseasy/experiments/bakeoff_phase1/reports/w512_memory_replay_interaction.json` | f38184d2f3644113... |
| 5 | evaluation_result | `/home/oseasy/experiments/bakeoff_phase1/reports/w512_memory_replay_interaction.md` | edc5490002400715... |
| 6 | evaluator_code | `/home/oseasy/experiments/bakeoff_phase1/eval_a_side_unified.py` | dcf7fe207bb485c4... |
| 7 | evaluator_code | `/home/oseasy/experiments/bakeoff_phase1/eval_w512_p2replay.py` | f76bb53ca20f3f13... |
| 8 | evaluator_code | `/home/oseasy/experiments/bakeoff_phase1/compute_causal_p2replay.py` | 085eb4a8ff50fb26... |
| 9 | training_code | `/home/oseasy/experiments/bakeoff_phase1/w512_p2_replay/src/w512_pending_episodes.py` | 0b62681057603761... |
| 10 | training_code | `/home/oseasy/experiments/bakeoff_phase1/w512_p2_replay/src/w512_compat_init.py` | 48d9523527c0e9ba... |
| 11 | training_code | `/home/oseasy/experiments/bakeoff_phase1/w512_p2_replay/src/launcher_w512_p2.py` | 7668200ccaf0f858... |
| 12 | training_code | `/home/oseasy/experiments/bakeoff_phase1/w512_p2_replay/src/w512_p2_core.py` | 52a35c2104777e46... |
| 13 | training_code | `/home/oseasy/experiments/bakeoff_phase1/w512_p2_replay/src/w512_p2_learner.py` | 612d41462aff5920... |
| 14 | training_code | `/home/oseasy/experiments/bakeoff_phase1/w512_p2_replay/src/run_w512_p2_levelB.py` | 016ccb598d9638d4... |
| 15 | training_code | `/home/oseasy/experiments/bakeoff_phase1/w512_p2_replay/src/w512_replay_buffer.py` | 552a929d95a0e406... |
| 16 | training_code | `/home/oseasy/experiments/bakeoff_phase1/gpu0_w512/src/ppo_tr_w512.py` | b8590c48a8e3f5d0... |
| 17 | training_code | `/home/oseasy/experiments/bakeoff_phase1/gpu0_w512/src/network_w512.py` | 8d1824d2e37e387e... |
| 18 | training_code | `/home/oseasy/experiments/bakeoff_phase1/gpu0_w512/src/w512_memory.py` | ee89fd0b3dd4bb79... |
| 19 | checkpoint | `/home/oseasy/experiments/bakeoff_phase1/gpu0_w512/train_24576/checkpoints/0` | 1597c74010dce8b1... |
| 20 | checkpoint | `/home/oseasy/experiments/bakeoff_phase1/gpu0_w512/train_24576/checkpoints/24576` | 2d89713e783d8a88... |
| 21 | checkpoint | `/home/oseasy/experiments/bakeoff_phase1/gpu0_w512/train_24576/checkpoints/4096` | 6ea82552744323e4... |
| 22 | checkpoint | `/home/oseasy/experiments/bakeoff_phase1/gpu0_w512_reset128_training/train_24576/checkpoints/0` | 846e48e628af9935... |
| 23 | checkpoint | `/home/oseasy/experiments/bakeoff_phase1/gpu0_w512_reset128_training/train_24576/checkpoints/24576` | a43c0c16858164db... |
| 24 | checkpoint | `/home/oseasy/experiments/bakeoff_phase1/gpu0_w512_reset128_training/train_24576/checkpoints/4096` | 156718cc85f70543... |
| 25 | checkpoint | `/home/oseasy/experiments/bakeoff_phase1/w512_p2_replay/w512_persistent_p2replay_24576/checkpoints/0` | c69c6c1ae0121481... |
| 26 | checkpoint | `/home/oseasy/experiments/bakeoff_phase1/w512_p2_replay/w512_persistent_p2replay_24576/checkpoints/12288` | 6b76cb06c38d5129... |
| 27 | checkpoint | `/home/oseasy/experiments/bakeoff_phase1/w512_p2_replay/w512_persistent_p2replay_24576/checkpoints/16384` | 7388a7376bae0718... |
| 28 | checkpoint | `/home/oseasy/experiments/bakeoff_phase1/w512_p2_replay/w512_persistent_p2replay_24576/checkpoints/20480` | a61aceb09568b023... |
| 29 | checkpoint | `/home/oseasy/experiments/bakeoff_phase1/w512_p2_replay/w512_persistent_p2replay_24576/checkpoints/24576` | 60ed1cf9c7319da2... |
| 30 | checkpoint | `/home/oseasy/experiments/bakeoff_phase1/w512_p2_replay/w512_persistent_p2replay_24576/checkpoints/4096` | 3f6cb18bbd65aa90... |
| 31 | checkpoint | `/home/oseasy/experiments/bakeoff_phase1/w512_p2_replay/w512_persistent_p2replay_24576/checkpoints/8192` | 958284374b69d4f9... |
| 32 | checkpoint | `/home/oseasy/experiments/bakeoff_phase1/w512_p2_replay/w512_reset128_p2replay_24576/checkpoints/0` | 07a70363b888e7f2... |
| 33 | checkpoint | `/home/oseasy/experiments/bakeoff_phase1/w512_p2_replay/w512_reset128_p2replay_24576/checkpoints/12288` | 832a8f1e44a64368... |
| 34 | checkpoint | `/home/oseasy/experiments/bakeoff_phase1/w512_p2_replay/w512_reset128_p2replay_24576/checkpoints/16384` | 5de194141ab65398... |
| 35 | checkpoint | `/home/oseasy/experiments/bakeoff_phase1/w512_p2_replay/w512_reset128_p2replay_24576/checkpoints/20480` | be904e9967e7f4ca... |
| 36 | checkpoint | `/home/oseasy/experiments/bakeoff_phase1/w512_p2_replay/w512_reset128_p2replay_24576/checkpoints/24576` | 4f2a154d11a02635... |
| 37 | checkpoint | `/home/oseasy/experiments/bakeoff_phase1/w512_p2_replay/w512_reset128_p2replay_24576/checkpoints/4096` | ed001821c3c75a64... |
| 38 | checkpoint | `/home/oseasy/experiments/bakeoff_phase1/w512_p2_replay/w512_reset128_p2replay_24576/checkpoints/8192` | b7a184ec4a03293a... |
| 39 | training_log | `/home/oseasy/experiments/bakeoff_phase1/gpu0_w512/train_24576/logs/training_log.jsonl` | 259c43cf0698a894... |
| 40 | training_log | `/home/oseasy/experiments/bakeoff_phase1/gpu0_w512_reset128_training/train_24576/logs/training_log.jsonl` | 3b7583d19315ca4f... |
| 41 | training_log | `/home/oseasy/experiments/bakeoff_phase1/w512_p2_replay/w512_persistent_p2replay_24576/run/training_log.jsonl` | f99a902cc5dc59c9... |
| 42 | training_log | `/home/oseasy/experiments/bakeoff_phase1/w512_p2_replay/w512_reset128_p2replay_24576/run/training_log.jsonl` | 3c5be6d4fa514aed... |
| 43 | source_checkpoint | `/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/base_ckpt_v7fix55_armA_s0/rl_checkpoints/17500` | N/A |
| 44 | reference_code | `/home/oseasy/experiments/p2_full_20260723/src/full_p2_learner.py` | c374f0aa3ce4ad28... |

## Authorizations (all false)

- TRAINING_TO_98304_AUTHORIZED = False
- W512_SECOND_SEED_AUTHORIZED = False
- UPDATE_HORIZON_ON_W512_AUTHORIZED = False
- P2_FULL_B_AUTHORIZED = False

## Provenance

服务器非git repo。所有SHA256在manifest生成时计算。SHA256为权威provenance。
所有科学产物标记为只读。不得修改、覆盖或重新评估。
