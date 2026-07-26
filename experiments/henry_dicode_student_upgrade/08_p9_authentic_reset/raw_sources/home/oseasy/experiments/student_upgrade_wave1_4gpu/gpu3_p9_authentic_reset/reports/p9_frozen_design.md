# P9-AUTHENTIC-RESET — Frozen Design (GPU3, Director B, wave1)

> FROZEN before any performance is seen. Structure / ratio / categories do NOT change in
> response to results. Date: 2026-07-24. Purpose: test whether the student fails because it
> rarely PRACTICES the critical floor-2 (Gnomish Mines) phase — by resetting half of training
> episodes into AUTHENTIC reached states harvested from the healthy student's own real
> trajectories. NO network change: original healthy GTrXL-PPO from ckpt17500.

## 0. What is unchanged

- Network = original `ActorCriticTransformer` (encoder + short-term GTrXL window=128 +
  actor/value heads), loaded from healthy ckpt17500. **No new params, no architecture change.**
- PPO/opt config = frozen table (§6), identical to the shared baseline.
- The trainer is the canonical continuous GTrXL-PPO loop (same as the Control re-train), with
  ONLY the episode-reset source modified (§3).

## 1. Authentic state library (READ-ONLY, from real ckpt17500 trajectories)

Roll out the healthy student under the frozen config (seed=42, S4_dark) and, at qualifying
moments, snapshot the FULL resumable state. No synthetic states; nothing hand-added (no extra
gear / health / resources / map info). Five categories (FROZEN):

| category | selection signal (read from sim state at that instant only) |
|---|---|
| `floor2_reached` | first step `player_level` enters the Gnomish-Mines floor (`ENTER_GNOMISH_MINES` just became true) |
| `mid_clear` | on the mines floor with monsters-killed in [1,7] (mid 8-monster corridor) |
| `gate_unlocked` | mines floor, down-stair gate has just unlocked (8 kills -> stair released) |
| `saw_stair_lost` | down-stair was visible in the local view on a recent step but is not visible now (agent walked off) |
| `near_floor3_failed` | reached deep into the mines / adjacent to sewers entry but the episode ended without `ENTER_SEWERS` |

Each snapshot stores (per env): the FULL wrapper `env_state` (inner Craftax state + logging +
optimistic-reset bookkeeping), `obs`, GTrXL `memories` (128-window), `memories_mask`,
`memories_mask_idx`, the env-step RNG `_rng` that produced the NEXT step, the `action` taken,
the category label, and (for validation) the recorded NEXT `(obs, env_state)`. Selection uses
only the instant's sim state — no future information enters the snapshot.

## 2. Correctness gates (the real validation — independent of category labels)

1. **Bit-restorable**: pickle round-trip of every snapshot reproduces all leaves byte-exactly.
2. **One-step-transition match**: restore a snapshot `(env_state, obs, memories, mask, midx,
   _rng, action)`, run ONE `env.step(_rng, env_state, action)` (through the same optimistic-reset
   wrapper) and the resulting `(obs_next, env_state_next)` is BIT-IDENTICAL to the recorded
   next state. This proves a restored authentic state continues exactly as the real trajectory
   did — the soundness of the whole reset mechanism, regardless of how the moment was labelled.
3. **No future/leak**: follows from (2) + construction (snapshot holds only past/present info).
4. **4096 smoke stable** (no NaN/Inf, entropy above floor).
5. **Checkpoint exact resume**: save/restore full training state INCLUDING the reset-sampler RNG
   and per-env episode-phase bookkeeping -> bit-identical continuation under det-ops.

## 3. Training reset policy (FROZEN ratio; never tuned by results)

At each episode boundary, for each env independently, draw from a fixed Bernoulli(0.5):
- **natural** (50%): standard Stage4 start via `env.reset` (fresh random world, floor-0).
- **authentic** (50%): restore a uniformly-sampled snapshot from the library
  (set env_state, obs, memories, mask, midx, RNG) and continue from there.
The 50/50 mix is FIXED for the whole run. The authentic-reset branch only changes WHERE an
episode starts; the PPO objective, the network, and the natural-start branch are untouched.

## 4. Evaluation discipline (FROZEN)

- Final evaluation uses **100% natural Stage4 starts** — NO reset injection, on the frozen
  256-world evaluator, same protocol as Baseline/Control. This measures whether practicing
  authentic reached states transfers to the real deployment distribution.
- Extra P9 metrics: per-category reset sample counts; natural vs authentic training-phase
  success rates; final natural-start DK SR / floor3 / conditional kill / death-timeout /
  episode length.

## 5. 98304 positive gate (vs same-step Control)

natural-eval DK SR >= +8 pp; floor3 >= Control; >=1 death/search metric improved; no numeric or
entropy collapse. Allowed labels only: EXPLORATORY_POSITIVE_SIGNAL / NO_POSITIVE_SIGNAL /
ENGINEERING_FAIL. No second seed, no 512-world, no Official FULL, no hyperparameter search.

## 6. Frozen config (shared)

deterministic ops; seed=42; LR=2e-5; Adam eps=1e-5; gamma=0.999; GAE lambda=0.8; rollout=128;
num_envs=16; minibatches=2; epochs=1; clip=0.2; vf=0.5; ent=0.002; gradnorm=1.0; anneal_lr=
false; value_target_clip=[-50,300]; Stage4-native (S4_dark); goal DEFEAT_KOBOLD;
total_steps=98304; saves at 0/4096/24576/49152/73728/98304.

## 7. Forbidden

No network change; no synthetic success states; no hand-added gear/health/resources/map;
no tuning the 50/50 ratio by results; no reset injection at evaluation; no replay/V-trace/
hindsight/novelty/NavAux.
