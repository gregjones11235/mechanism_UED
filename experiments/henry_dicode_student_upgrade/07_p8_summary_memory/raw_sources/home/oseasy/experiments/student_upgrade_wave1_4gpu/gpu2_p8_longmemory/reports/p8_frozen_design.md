# P8-LONGMEM-SUMMARY — Frozen Design (GPU2, Director B, wave1)

> FROZEN before any performance is seen. Structure below does NOT change in response to
> results. Date: 2026-07-24. Purpose: isolate whether the original GTrXL's 128-step window
> limits dark-search, by adding a long-term summary channel that extends effective history to
> >=1024 steps WITHOUT changing the short-term GTrXL or any inherited weight.

## 0. What is kept (all inherited bit-exact from healthy ckpt17500)

- Original obs encoder (`transformer/encoder`, 8335->256) — the "CNN/encoder".
- Short-term GTrXL memory: `transformer/tf_layers_{0,1}` attention (window_mem=128,
  num_heads=8, qkv=256, embed=256, gating=true, gating_bias=2.0).
- Actor head (`actor_ln1/2/out`), Value head (`critic_ln1/2/out`).
- All PPO/opt config (frozen config table, §6). Short-term memory reset behaviour unchanged
  (resets on the raw `done` exactly as the original trainer).

## 1. Long-term summary channel (the ONLY addition) — FROZEN structure

| frozen knob | value | rationale |
|---|---|---|
| summary_interval (K) | 64 steps | "每32或64步"; 64 chosen to bound param/compute |
| num_summaries (N) | 16 | "最近若干summary，覆盖至少1024步": 16 x 64 = 1024 |
| summary embedding dim | 256 (=embed_size) | matches GTrXL hidden width |
| aggregation method | mean of the last K encoder outputs, then `summary_proj` Dense(256->256, tanh) | deterministic, O(1)/step via a running sum+count accumulator |
| how Actor reads summaries | single-head dot-product attention, query = current encoder output, keys/values = the N summary tokens, masked by a validity mask -> context (256) | lets the actor selectively read long history |
| where it enters | added to the ACTOR branch only (per directive: "Actor读取...长期summary"); the Value head input is unchanged | matches directive; keeps value==teacher at init |
| `summary_to_actor` projection | Dense(256->256), **kernel & bias initialised to ZERO** | => long-term contribution is exactly 0 at init |
| effective history coverage | 1024 steps | meets requirement |
| reset rule | long-term ring buffer + accumulator cleared on **true_done only** (`info["returned_episode"]`); persists across rollout boundaries; NOT cleared on optimistic-reset truncation | "长期状态跨rollout保存；true done重置" |

Consequence of the zero-init `summary_to_actor`: **at initialisation the P8 model is
bit-identical in its outputs to the healthy teacher** (logits and value). This makes the
"feature-off path" exact and satisfies "新模型初始logits/value接近教师" with equality, not
approximation. The behavioural distillation (§3) then trains ONLY the new long-memory params
(summary_proj, summary attention, summary_to_actor) on teacher trajectories while keeping
logits/value close to the teacher; inherited GTrXL/encoder/head weights are FROZEN during it.

## 2. Param budget (reported, frozen)

New params only (inherited net unchanged):
- `summary_proj`: 256x256 + 256 = 65,792
- summary attention (single head, 256): q/k/v 256x256 each + out 256x256 + biases = ~263,168
- `summary_to_actor`: 256x256 + 256 = 65,792 (zero-init)
- (running sum/count accumulator and the summary ring buffer are STATE, not params)
- **Total new trainable params ≈ 0.39 M** on top of the unchanged teacher net.

## 3. Behavioural distillation (init, GPU2)

1. Generate a FIXED teacher trajectory set from healthy GTrXL ckpt17500 under the frozen
   config (S4_dark, seed=42): record per step (obs, teacher_logits, teacher_value). Pinned
   RNG => reproducible bit-for-bit.
2. Start P8 at zero-init long path (so outputs==teacher exactly).
3. Train ONLY the new long-memory params to minimise
   `KL(student_logits||teacher_logits) + value_mse` on the fixed set (inherited params frozen),
   capped epochs, early-stop on a held-out split. Because it starts at loss~=0 with the long
   path zeroed, the inherited behaviour is preserved; the long-memory params learn an
   informative-but-non-disruptive summary readout.
4. Persist ONE artifact: `distill/P8_DISTILLED_INIT/` (full params + teacher dataset sha +
   held-out KL/value error + param-count report + the bit-equal inheritance proof).

## 4. Migration gate (BEFORE long training) — all must PASS else ENGINEERING_FAIL

On a frozen 64-world evaluator vs Baseline (ckpt17500), same protocol:
- DK SR drop vs Baseline <= 5 pp; floor3 drop vs Baseline <= 5 pp;
- **feature-off path correct**: with the long-term channel disabled the model reproduces the
  teacher/Baseline (drop ~= 0);
- long-term state persists across a rollout boundary (summary buffer at the seam is carried,
  not zeroed);
- true_done resets the long-term buffer; optimistic truncation does NOT;
- entropy normal (above the `guard_session_entropy_min` floor), no NaN/Inf;
- checkpoint round-trip + EXACT resume (restore full state incl. long-term buffer/accumulator/
  short-term memory/RNG/collector -> bit-identical continuation under det-ops);
- 4096-step smoke stable.

On PASS -> train to 98304, save at 0/4096/24576/49152/73728/98304.

## 5. Engineering/runner mechanics

- The runner_state is extended with the long-term state
  `(summ_ring[N,256], summ_valid[N], accum_sum[256], accum_count)` per env, threaded through
  the inner `_env_step` scan and the outer `_update_step` scan exactly like the existing
  short-term `memories` — so it survives rollout boundaries and is checkpointed.
- `_env_step`: run the original GTrXL forward (unchanged) to get (pi,value,memories_out);
  in parallel update the long-term accumulator from `encoded`; every K steps commit a summary
  token into the ring; the actor logits use `x + summary_to_actor(attn_over_summaries)`.
  true_done (from `info["returned_episode"]`) clears the long-term state.
- For the loss (`model_forward_train`), the long-term readout is recomputed by scanning the
  summary-commit logic over the segment from the segment-start long-term state (stored once
  per update, exactly as `memories_previous` is today) so the policy used for log_prob matches
  the rollout policy (deterministic, bit-exact).
- Vector-env isolation: all long-term ops are vmap-leading over env with no cross-env mixing.

## 6. Frozen config (both for P8 and the shared baseline)

deterministic ops; seed=42; LR=2e-5; Adam eps=1e-5; gamma=0.999; GAE lambda=0.8; rollout=128;
num_envs=16; minibatches=2; epochs=1; clip=0.2; vf=0.5; ent=0.002; gradnorm=1.0; anneal_lr=
false; value_target_clip=[-50,300]; Stage4-native (S4_dark); goal DEFEAT_KOBOLD;
total_steps=98304 (48 updates); saves at 0/4096/24576/49152/73728/98304.

## 7. Forbidden

No replay, no V-trace, no hindsight, no novelty bonus, no NavAux/privileged aux heads.
No second seed, no 512-world, no Official FULL, no hyperparameter search.
