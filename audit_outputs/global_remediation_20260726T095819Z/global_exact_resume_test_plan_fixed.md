# Global Exact-Resume Test Plan (FIXED)

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
