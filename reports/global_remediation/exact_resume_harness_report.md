> GLOBAL_EVALUATION_REMEDIATION (CC4) · repo mechanism_UED @ Henry-branch · HEAD=UNVERIFIED (git access BLOCKED)
> Environment: JAX/Craftax ABSENT, experiment server UNREACHABLE, GPU idle (not used), no CC2/CC3 processes.
> Discipline: read-only w.r.t. CC2/CC3 & old audit; NEW_TRAINING_RUNS=0; MISSING/UNVERIFIED never relabeled FAIL.

# Exact-Resume Harness Report

Distinction (frozen): CHECKPOINT_SAVE_VALID (roundtrip; necessary not sufficient) vs EXACT_RESUME_BITEXACT
(continuation A@4096==B1@4096 AND A@8192==B2@8192 over the FULL state, not just params).

tools/exact_resume_harness.py — pure-logic state-diff engine, runs without JAX. `--self-test` PASS 6/6:
ident_full_bitexact, rng_diff_detected, p7_missing_detected, rmt_missing_env_state_detected,
replay_components_compared, replay_rng_diff_detected. `--run-continuation` raises NOT_AUTHORIZED
(training-type A/B continuation needs JAX + a training step + explicit re-authorization — NOT run this round).

## Per-experiment coverage (exact_resume_schema.json)
SlowGRU=BITEXACT(1a4232e6); EventMem=BITEXACT(67ee581c); P9=CLAIMED-text-only(9ba3f2b9); P8=roundtrip-only;
P7=params+carry only (missing optimizer/global_step/env_state/rng/manifest); RMT16=train_state.pkl missing
env_state, no restore path, gate7/gate11 tests NOT_FOUND; W512=UNVERIFIED. Resume gaps are
IMPLEMENTATION/EVIDENCE gaps — NOT FAIL. Plan: global_exact_resume_test_plan_fixed.md.
