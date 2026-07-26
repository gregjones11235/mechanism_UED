# W512 Memory × Replay Interaction: Final Report

## Scientific Question

Does P2 Replay (V-trace + AWR off-policy training) causally interact with W512's
cross-128-step long-history carry? Specifically: is the +8.20pp carry effect observed
under PPO training a genuine long-memory benefit, or an artifact of PPO's interaction
with the W512 architecture?

## Experimental Design

**2×2 Factorial:**
- Factor 1 (Carry): Persistent (long_buf/long_mask persist across rollout boundary) vs Reset128 (cleared every 128 steps)
- Factor 2 (Training): PPO (from Part A) vs P2 Replay (V-trace + AWR, frozen P2-Full-A config)

**Common initialization:** ckpt17500, seed=42, deterministic ops, Stage4-native, goal=DEFEAT_KOBOLD
**Training budget:** num_envs=16, rollout=128, LR=2e-5, Adam eps=1e-5, gamma=0.999, GAE lambda=0.8, total_steps=24576
**P2 Replay config (frozen P2-Full-A):** L_SEQ=129, K_BATCH=4, capacity=64, kl_replay_max=0.05, kl_run_max=0.10, ema_tau=0.995, max_policy_lag=16, w_vtrace=0.5, w_awr=0.5, rho_bar=1.0, c_bar=1.0, beta=1.0, w_max=20.0, lambda_kl=0.01, actor_step_scales=(1.0,0.5,0.25,0.125), ent_floor=0.05

**Evaluation protocol (frozen):** 256 worlds, seed=42, stochastic policy, max_steps=4096, S4_dark native start, pre-step achievements, DistributedMultiTaskOptimisticLogWrapper

## Results

### 6-Arm Comparison Table

| Arm | DK SR | n_success | floor3 | death | timeout | ep_len |
|-----|-------|-----------|--------|-------|---------|--------|
| Baseline (ckpt17500) | 39.45% | 101/256 | 43.36% (111) | 147 | 8 | 986 |
| Control @24576 | 36.33% | 93/256 | 43.36% (111) | 156 | 7 | 862 |
| W512-Persistent (PPO) | 10.94% | 28/256 | 17.58% (45) | 226 | 2 | 622 |
| W512-Reset128 (PPO) | 2.73% | 7/256 | 6.64% (17) | 249 | 0 | 575 |
| **W512-Persistent-P2Replay** | **35.16%** | **90/256** | **41.02% (105)** | 162 | 4 | 954 |
| **W512-Reset128-P2Replay** | **37.11%** | **95/256** | **44.92% (115)** | 152 | 9 | 1025 |

### Core Causal Quantities

| Quantity | Value | p-value | 95% CI | Interpretation |
|----------|-------|---------|--------|----------------|
| CARRY_NO_REPLAY (PPO) | +8.20pp | 6.3e-05 | [+4.69, +12.11] | Significant carry under PPO |
| CARRY_WITH_REPLAY | -1.95pp | 0.424 | [-5.86, +1.95] | No carry under Replay |
| REPLAY_EFFECT_PERSISTENT | +24.22pp | <1e-6 | [+18.75, +29.69] | Replay massively helps Persistent |
| REPLAY_EFFECT_RESET | +34.38pp | <1e-6 | [+28.52, +40.23] | Replay helps Reset128 even more |
| MEMORY_REPLAY_INTERACTION | -10.16pp | — | — | Replay eliminates carry |
| P2Replay_Pers vs Control | -1.17pp | 0.832 | [-8.20, +5.86] | No difference |
| P2Replay_Rst vs Control | +0.78pp | 0.920 | [-7.03, +8.20] | No difference |
| P2Replay_Pers vs Baseline | -4.30pp | 0.254 | [-10.94, +2.34] | No difference |
| P2Replay_Rst vs Baseline | -2.34pp | 0.566 | [-8.98, +4.30] | No difference |

### Verdict: **REPLAY_ELIMINATES_CARRY**

## Interpretation

### 1. PPO catastrophically degrades W512 performance
W512 + PPO training for 24576 steps collapses DK success from 39.45% (baseline) to
10.94% (Persistent) and 2.73% (Reset128). This is a -25 to -37pp catastrophe. The W512
architecture (GTrXL-128 + 384-step raw history cross-attention + zero-init gate) is
fundamentally incompatible with on-policy PPO at this budget.

### 2. P2 Replay recovers W512 to baseline performance
With P2 Replay (V-trace + AWR), W512 recovers to 35.16% (Persistent) and 37.11%
(Reset128) — statistically indistinguishable from both Control (36.33%) and Baseline
(39.45%). The zero-init gate allows the network to ignore the W512 additions and
behave like the base GTrXL-128 when trained with a stable off-policy method.

### 3. The carry effect was a PPO artifact, not a long-memory benefit
Under PPO, Persistent outperformed Reset128 by +8.20pp (p=6.3e-05). Under Replay,
the direction reverses to -1.95pp (p=0.42, n.s.). The MEMORY_REPLAY_INTERACTION is
-10.16pp: replay eliminates the carry effect entirely.

**Mechanism:** Under PPO's catastrophic training dynamics, the Reset128 arm suffered
more because clearing long_buf/long_mask at each rollout boundary introduced additional
discontinuity in an already unstable optimization landscape. The Persistent arm's
slightly better performance was not due to beneficial long-memory carry — it was due
to less disruption of an already-failing optimization. When training is stabilized
by replay, both modes converge to the same baseline-level performance.

### 4. Reset128 is slightly better than Persistent with replay (non-significant)
The Reset128 arm achieves 37.11% vs 35.16% for Persistent (+1.95pp, p=0.42). While
not significant, this is the opposite direction from the PPO carry effect and suggests
that if anything, periodic resetting of long state is mildly beneficial under stable
training — possibly because it prevents stale long-state representations from
accumulating.

## Training Diagnostics

### P2Replay Training (both arms, 24576 steps = 12 rollouts)

| Metric | Persistent | Reset128 |
|--------|-----------|----------|
| Updates | 11 | 11 |
| Accepted (KL pass) | 11 | 11 |
| KL rejected | 0 | 0 |
| Total episodes | 21 | 23 |
| Replay size | 21 | 23 |
| NaN/Inf | No | No |
| Total time | 938.7s | 955.8s |
| Final loss | 29.77 | 23.12 |
| Final KL | 0.001 | 0.001 |

Both arms trained stably with all 11 updates accepted by the KL gate. No hard-stop
triggers. KL values decreased over training (from ~0.025 to ~0.001), indicating
convergence.

## Frozen Provenance

### Checkpoints
- W512-Persistent-P2Replay @24576: params_sha=7f23a9cba37882fe1664068996b63782cfde4f84d8fb1672f9c3c5cf1d50b4b8
- W512-Reset128-P2Replay @24576: params_sha=c6271247b73fc1c9c121efa705c5336269e860ed324a51e55367cf7348185864
- Common init (both arms): params_sha=5942526301c66766edadae3bbd28ad680d2c323b97acca41265aa16e8bfdcbdd

### Evaluator
- eval_w512_p2replay.py (SHA computed at runtime)
- Protocol: 256 worlds, seed=42, stochastic, max_steps=4096, S4_dark native

### Code
- Training: /home/oseasy/experiments/bakeoff_phase1/w512_p2_replay/src/
- Evaluator: /home/oseasy/experiments/bakeoff_phase1/eval_w512_p2replay.py
- Causal analysis: /home/oseasy/experiments/bakeoff_phase1/compute_causal_p2replay.py

### Source checkpoint
- ckpt17500: /home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/base_ckpt_v7fix55_armA_s0/rl_checkpoints/17500

## Authorizations (unchanged)
- TRAINING_TO_98304_AUTHORIZED = false
- P2_FULL_B_FINAL_AUTHORIZED = false
- UPDATE_HORIZON_PHASE_AUTHORIZED = false
