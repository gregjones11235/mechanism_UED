# E1 CC2-Director: smoke handoff blockers

## Status

**`DIRECTOR_SMOKE_HANDOFF_READY`** — code + contract tests are
complete; the real Smoke is NOT authorized or executed.

## Remaining blockers (all fail-closed this round)

1. **Shared runtime absent** — `dicode.shared_runtime` does not
   exist; every shared contract resolves
   `BLOCKED_WAITING_SHARED_RUNTIME_<CONTRACT>`.
2. **Runtime-bundle Signer Registry EMPTY** —
   `AUTHORIZED_BUNDLE_SIGNERS=()`; no director-signed PRODUCTION
   bundle can verify (`RUNTIME_BUNDLE_SIGNER_UNAUTHORIZED`).
3. **No real LLM provider authorized** —
   `AUTHORIZED_REAL_LLM_PROVIDERS=()`; the six-role board never falls
   back to replay.
4. **Real EnvCoder backend blocked** — `ENVCODER_BACKEND_BLOCKED`;
   only the authorized 13-stage validation surface exists (TEST_ONLY).
5. **Reference identity contract unfrozen** (G1).
6. **Shared anchor manifest DRAFT_UNFROZEN** (G3).
7. **No real probe / update / round-trip / smoke signers** — the
   signer whitelists are EMPTY; nothing real is signed or consumed on
   the production path.
8. **Training budget undecided** —
   `BLOCKED_WAITING_DIRECTOR_BUDGET_DECISION`; the formal experiment
   never starts on an unresolved budget.

## Authorization (all false)

- REAL_LLM_EXECUTED / REAL_ENVCODER_EXECUTED /
  REAL_CANDIDATE_PROBE_EXECUTED / REAL_OPTIMIZER_UPDATE_EXECUTED /
  REAL_FULL_STATE_ROUND_TRIP / E1_REAL_SMOKE_AUTHORIZED /
  FORMAL_EXPERIMENT_AUTHORIZED — **all false**.

## Only next step

The director signs the runtime bundle, freezes the shared assets and
approves the Smoke; then `run_e1_real_one_update.py
--director-runtime-bundle <signed> --check-only` verifies, and only a
human-approved Smoke executes. This round: check-only + tests only.
