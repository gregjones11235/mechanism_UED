# E1 CC2-Student: director handoff blockers

## Status

**`E1_DUAL_STUDENT_CONSUMER_READY`** — code + contract tests are
complete for the dual-Student selection / mount / continuity /
read-only-vs-training capability split; the real Smoke is NOT
authorized or executed.

## Remaining blockers (all fail-closed this round)

1. **Shared student registry absent** — `dicode.student_adapters`
   does not exist; every mount resolves
   `STUDENT_SHARED_REGISTRY_UNBOUND` (read-only mount is NOT ready,
   training is NEVER implied).
2. **Runtime-bundle Signer Registry EMPTY** —
   `AUTHORIZED_BUNDLE_SIGNERS=()`; no director-signed PRODUCTION
   bundle can verify, so no real Student selection can be injected.
3. **No real LLM provider authorized** — the six-role board never
   falls back to replay.
4. **Reference identity contract unfrozen** (G1) and the shared
   anchor manifest DRAFT_UNFROZEN (G3).
5. **No canonical DiCode training runtime bound** — training happens
   ONLY through the director-injected `CanonicalDiCodeOneUpdateRuntime`
   + `CanonicalDiCodeRunStateCheckpoint`; until then the Director
   Smoke handoff stays BLOCKED and no update is ever forged.

## Authorization (all false)

- REAL_LLM_EXECUTED / REAL_CANDIDATE_PROBE_EXECUTED /
  REAL_OPTIMIZER_UPDATE_EXECUTED / REAL_FULL_STATE_ROUND_TRIP /
  E1_REAL_SMOKE_AUTHORIZED / FORMAL_EXPERIMENT_AUTHORIZED — **all
  false**.

## Only next step

The director signs the runtime bundle (with a `student` selection of
one of the two allowed candidates), freezes the shared assets and
approves the Smoke; then `run_e1_real_one_update.py
--director-runtime-bundle <signed> --student-candidate-id <id>
--check-only` verifies. This round: check-only + tests only.
