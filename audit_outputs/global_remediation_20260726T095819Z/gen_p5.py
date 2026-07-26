#!/usr/bin/env python
# CC4 remediation [5/6]: Base GTrXL matched Replay control config + diff + cell manifest + resume plan.
# Config only; NO training launched (READY_NOT_AUTHORIZED).
import json, os, hashlib
BASE=os.getcwd()
OUT=open(os.path.join(BASE,"audit_outputs","_remediation_outdir.txt")).read().strip()
def J(p,o):
    with open(p,"w",encoding="utf-8") as f: json.dump(o,f,indent=2,ensure_ascii=False)

yaml="""# base_gtrxl_matched_replay_config.yaml
# CANONICAL matched control: Base GTrXL + the P2 Replay bundle, NO architecture change.
# Purpose: isolate the P2 Replay effect from architecture/training-bundle effects (fills MISS-1 /
# BASE_GTRXL_MATCHED_REPLAY_CONTROL). Status: READY_NOT_AUTHORIZED -- DO NOT launch training.
# Every field MUST match the P2-Full-A arm it controls for; the ONLY allowed difference is `network`.

experiment_id: BASE_GTRXL_MATCHED_REPLAY
status: READY_NOT_AUTHORIZED
authorization: "training NOT authorized this round; launching is forbidden until explicit re-authorization"

# ---- THE SINGLE DIFFERENCE ----
network:
  class: ActorCriticTransformer        # Base GTrXL window-attention trunk
  long_memory_module: NONE             # explicitly NOT W512 / RMT / SlowGRU / EventMemory
  note: "This is the only field that differs from P2-Full-A. Everything below is identical to P2."

# ---- initialization (identical to P2) ----
init:
  checkpoint: teacher17500
  params_sha256: d4e85af58b7f87d6
  method: "base trunk inherited bit-exact; no fresh long-memory leaves (there are none)"

# ---- P2 Replay bundle (identical to P2-Full-A) ----
replay:
  enabled: true
  capacity: 64                          # ReplayBuffer(capacity=64) per P2 requirement matrix
  L_SEQ: 512                            # MUST equal the P2 arm controlled for. CONFLICT NOTE: W512 repro
                                        # used 129 (run_p2_full_smoke.py:66 'formal run uses 512'); RMT16 /
                                        # P2-Full-A v2.1 frozen config specify 512. FREEZE canonical L_SEQ on
                                        # the server BEFORE running, and record it in provenance (MISS-6).
  k_batch: 4
  vtrace: true
  awr_hindsight: true
  ema:
    enabled: true
    tau: 0.995
  policy_lag_gate:
    enabled: true
    max_lag: 16
  transactional_kl_gate:
    enabled: true
  anchor_interval: 128

# ---- PPO / optimizer (identical to P2) ----
ppo:
  optimizer: Adam
  learning_rate: 2.0e-5
  adam_eps: 1.0e-5
  gamma: 0.999
  grad_clip_global_norm: 1.0
  num_envs: 16
  rollout_steps: 128
  transitions_per_update: 2048          # online (+2048 replay when replay on)
  total_environment_steps: 24576        # screen; longrun variant = 98304 (separate authorization)

# ---- evaluation (CANONICAL) ----
evaluation:
  protocol: CANONICAL_EVALUATOR_V1
  evaluator_sha256: "REQUIRED (canonical evaluator file SHA at run time)"
  worlds: 256
  evaluation_seed: 42                   # seed42 line; paired with CONTROL24576_BASELINE / TEACHER17500_BASELINE
  world_manifest: canonical_worlds_256_seed42.json
  world_set_hash: "REQUIRED (materialized on JAX host before run; GATE2/3)"
  action_mode: stochastic
  max_timesteps: 4096
  target_achievement: DEFEAT_KOBOLD

# ---- controls this arm enables ----
enables_comparisons:
  - "BASE_GTRXL_MATCHED_REPLAY vs P2-Full-A  => isolates ARCHITECTURE (network) effect under identical Replay"
  - "BASE_GTRXL_MATCHED_REPLAY vs CONTROL24576_BASELINE => isolates REPLAY-bundle effect under identical Base GTrXL"
"""
open(os.path.join(OUT,"base_gtrxl_matched_replay_config.yaml"),"w",encoding="utf-8").write(yaml)

diff="""# Base GTrXL Matched Replay Control — diff vs P2-Full-A

This is the **missing strict control** (MISS-1). It is NOT a performance bug; it is a design gap that
prevents isolating the P2 Replay effect from architecture/training-bundle effects. Status:
`BASE_GTRXL_MATCHED_REPLAY_CONTROL = READY_NOT_AUTHORIZED` (config built; **training not launched**).

## Single difference
| Field | P2-Full-A | BASE_GTRXL_MATCHED_REPLAY |
|---|---|---|
| **network** | GTrXL + long-memory/replay-augmented trunk (e.g. W512/RMT/SlowGRU/EventMem) | **Base GTrXL `ActorCriticTransformer`, NO long-memory module** |

Everything else is held identical:
- init: teacher17500 (`d4e85af58b7f87d6`)
- Replay bundle: capacity 64, L_SEQ (see conflict), K_BATCH 4, V-trace, AWR/hindsight, EMA tau 0.995,
  policy-lag gate (16), transactional KL gate, anchor_interval 128
- PPO: Adam lr 2e-5 eps 1e-5, gamma 0.999, grad-clip 1.0, num_envs 16, rollout 128, 2048/update, 24576 total
- eval: CANONICAL_EVALUATOR_V1, 256 worlds, seed42, stochastic, max4096, DEFEAT_KOBOLD

## Open item that MUST be frozen before any run
- **L_SEQ conflict (MISS-6):** W512 repro = 129; RMT16 / P2-Full-A v2.1 = 512. The matched arm must use the
  SAME L_SEQ as the P2-Full-A arm it controls for. Freeze canonical L_SEQ on the server and record it in
  provenance before launching.

## Comparisons this unlocks
1. `BASE_GTRXL_MATCHED_REPLAY` vs `P2-Full-A` → isolates the **architecture** contribution under identical Replay.
2. `BASE_GTRXL_MATCHED_REPLAY` vs `CONTROL24576_BASELINE` (93/256) → isolates the **Replay-bundle** contribution
   under identical Base GTrXL — the comparison that is currently impossible (GATE15 verifies the config match).

## GATE15
The regression gate asserts this config equals the P2 bundle on every field EXCEPT `network.class` /
`long_memory_module`. Implemented in `tools/regression_gates.py`.
"""
open(os.path.join(OUT,"base_gtrxl_matched_replay_diff.md"),"w",encoding="utf-8").write(diff)

manifest={
 "cell_id":"BASE_GTRXL_MATCHED_REPLAY",
 "status":"READY_NOT_AUTHORIZED",
 "config_file":"base_gtrxl_matched_replay_config.yaml",
 "config_sha256":hashlib.sha256(open(os.path.join(OUT,"base_gtrxl_matched_replay_config.yaml"),"rb").read()).hexdigest(),
 "purpose":"isolate P2 Replay effect from architecture (fills MISS-1)",
 "single_difference":"network = Base GTrXL (no long-memory module); all other fields identical to P2-Full-A",
 "blocked_on":["canonical L_SEQ freeze (MISS-6)","world_set_hash materialization (JAX host)","explicit training authorization"],
 "training_launched":False,"new_training_runs":0,
 "gate":"GATE15 (config matches P2 bundle except network) in tools/regression_gates.py"}
J(os.path.join(OUT,"base_gtrxl_matched_replay_cell_manifest.json"),manifest)

plan="""# Global Exact-Resume Test Plan (FIXED)

Status: **harness + schema delivered; training-type tests NOT run** (require explicit re-authorization).

## Distinction (frozen)
- `CHECKPOINT_SAVE_VALID`: save→load roundtrip yields a usable state (necessary, not sufficient).
- `EXACT_RESUME_BITEXACT`: continuation bit-exact — continuous run **A** vs new-process-restore **B2** must
  satisfy `A@4096 == B1@4096` AND `A@8192 == B2@8192` over the **FULL state**, not just params.

## Standard test (to run only when authorized)
- **A (continuous):** init → run continuously to step 8192 → serialize full state → SHA per component.
- **B (restore):** init → run to 4096 → save checkpoint → **exit process** → new process → restore from
  checkpoint → continue to 8192 → serialize full state → SHA per component.
- **Gate:** PASS iff every required component SHA matches between A@8192 and B2@8192 (and A@4096==B1@4096).

## Full state components compared (`exact_resume_schema.json`)
params, optimizer_state, target/EMA params, global_step, update_count, JAX RNG, action RNG, env_state,
observation, GTrXL memory/mask/index, replay buffer, replay sampling RNG, pending episodes, policy version,
RMT/W512 extra state, next-batch trajectory IDs, per-update metrics (+ done/true_done).

## GATE14 (missing-component detection) — verified by harness self-test
The harness flags a checkpoint that LACKS any required component instead of silently passing:
- **P7-like** (params+carry only) → missing `optimizer_state`, `global_step`, `env_state`, `jax_rng` ✓ detected.
- **RMT16-like** (replay-on) → missing `env_state` ✓ detected.
- A single `jax_rng` diff → bit-exact FAIL ✓ detected (this is exactly what params-only checks miss).
- Replay-enabled comparison includes replay components; a replay-sampling-RNG diff → FAIL ✓ detected.

## Per-experiment remediation required before the test can pass
| Experiment | Current | Required remediation |
|---|---|---|
| SlowGRU | BITEXACT (1a4232e6) | none — reference standard |
| EventMem | BITEXACT (67ee581c) | none — reference standard |
| P9 | continuation CLAIMED (text 9ba3f2b9) | produce `compare_resume` artifact via this harness |
| P8 | roundtrip only | add independent A/B continuation script |
| P7 | params+carry only; in-memory mini | extend disk ckpt: optimizer/Adam moments + global_step + env_state + rng + manifest; record params SHA |
| RMT16 | train_state.pkl missing env_state; tests NOT_FOUND | add env_state to ckpt + restore path + `test_exact_resume(gate7)`/`test_resume_state(gate11)` |
| W512 | UNVERIFIED | add a resume smoke script with the full-state continuation gate |

## Discipline
Resume gaps are IMPLEMENTATION/EVIDENCE gaps — NOT FAIL, NOT performance failures. No training-type Exact
Resume test is executed this round; the harness `--run-continuation` path raises NOT_AUTHORIZED.
"""
open(os.path.join(OUT,"global_exact_resume_test_plan_fixed.md"),"w",encoding="utf-8").write(plan)
print("WROTE base_gtrxl_matched_replay_config.yaml/.diff/.cell_manifest + global_exact_resume_test_plan_fixed.md")
