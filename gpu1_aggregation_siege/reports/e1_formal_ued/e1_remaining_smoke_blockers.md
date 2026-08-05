# E1 CC2-Repair: remaining smoke blockers

## Status

**`E1_OBJECT_LEVEL_CONSUMER_READY`** — code + tests complete for the
unified student-selection bundle schema, the removed production
synthetic fallback, the two-level check-only and the reachable
pipeline path. OBJECT_LEVEL_CHECK_ONLY_OK and the Director Smoke
handoff are NOT granted this round.

## Remaining blockers

1. **Shared FormalAssetRegistry absent** — no real StudentInitContract
   / StudentIdentity / StudentAdapter / Reference / Probe / Anchor /
   one-update runtime / runstate checkpoint objects are resolved; the
   object-level check-only honestly BLOCKS.
2. **Runtime Signer Registry EMPTY** — `AUTHORIZED_BUNDLE_SIGNERS=()`;
   no director-signed PRODUCTION bundle can verify.
3. **No real LLM provider authorized**; the six-role board never
   falls back to replay.
4. **Reference contract unfrozen (G1)** + **anchor manifest
   DRAFT_UNFROZEN (G3)**.
5. **No canonical DiCode training runtime bound** — training only
   flows through the director-injected
   `CanonicalDiCodeOneUpdateRuntime` +
   `CanonicalDiCodeRunStateCheckpoint`; no update is ever forged.

## Authorization (all false)

REAL_LLM_EXECUTED / REAL_ENVCODER_EXECUTED /
REAL_CANDIDATE_PROBE_EXECUTED / REAL_OPTIMIZER_UPDATE_EXECUTED /
REAL_FULL_STATE_ROUND_TRIP / E1_REAL_SMOKE_AUTHORIZED /
FORMAL_EXPERIMENT_AUTHORIZED — all false.

## Only next step

The director injects the shared FormalAssetRegistry + signs the
runtime bundle; then the object-level check-only can pass and the
director approves the Smoke. This round: check-only + tests only.
