# D3Q Phase-2 Incident 05 — post-run GPU gate blocked after all preflight arms completed

classification: D3Q_PHASE2_INCIDENT_05
schema_version: 1
recorded_utc: 2026-08-16T08:26:00Z
severity: blocker_in_recovery

## Symptom

Preflight run `d3q_preflight_20260816T065423Z` completed ALL 6 arms
(driver.rc=0, remote summary status PASS, all 6 RESULT.json collected) but the
orchestrator's POST-run GPU gate then failed:

```json
{"detail": ["GPU-8df11537-ab79-722d-606f-411966196c4c, 3753771"], "reason": "gpu2_external_app", "status": "BLOCKED"}
```

The exception preceded the result-document write, so no
D3Q_PREFLIGHT_ORCHESTRATOR_RESULT.json was produced; the guarded cleanup then
verified no live driver/replay processes and removed the remote exec root.
Full arm evidence was already collected locally (tar collection happens before
the post gate).

## External process

- PID 3753771: `/home/oseasy/miniconda3/envs/dicode310/bin/python -m pytest tests/e3_litesim -q`
- lstart: Sun Aug 16 16:14:07 2026 CST = 2026-08-16T08:14:07Z (ps lstart).
- Not owned by D3Q (another workstream's test suite). JAX multi-device
  mapping: 28010 MiB on GPU0, 262 MiB contexts on GPU1/GPU2/GPU3.
- Overlap window: small_r3 replay (~08:08-08:22Z) — the only arm overlapping.

## Interference analysis (quantitative)

small_r3 (10 candidates, overlapped) vs small_r2 (10 candidates, clean):

| phase                    | small_r2 | small_r3 |
|--------------------------|----------|----------|
| preflight_eval_lower_compile | 665.0s | 560.2s |
| preflight_eval_execute   | 232.9s   | 202.6s   |
| scoring_cpu              | 19.8s    | 19.8s    |
| session_wall             | 936.4s   | 801.4s   |

The overlapped arm is FASTER in both GPU-relevant phases; a co-resident
context stealing compute would have inflated them. A 262 MiB context on a
48 GiB card poses no memory pressure. Acceptance numerics are computed under
`--xla_gpu_deterministic_ops=true` and are unaffected by scheduling.

## Recovery (implemented, tested)

Following the incident-02 precedent (`recover-completed-chunk`, allowed reason
`gpu2_external_app`), the orchestrator gains `recover-completed-run`:

- allowed reasons whitelist: `gpu2_external_app` only;
- fail-closed gates: remote summary PASS; remote_run_rc.txt == 0 and
  driver.rc == 0; every arm PASS; per-arm evidence files present and
  SHA256-hashed into D3Q_PREFLIGHT_RECOVERY.json; per-arm interference check
  (execute-per-candidate within 1.25x of the median of the other arms);
  external PID verified gone on the server; post-run GPU/ollama gates re-pass.
- Writes D3Q_PREFLIGHT_ORCHESTRATOR_RESULT.json (status PASS + `recovery`
  block) and D3Q_PREFLIGHT_RECOVERY.json (audit + evidence hashes).
- 7 new tests cover happy path and every fail-closed branch.

## Disposition

Run `recover-completed-run --artifact-dir d3q_artifacts/d3q_preflight_20260816T065423Z
--reason gpu2_external_app --external-pid 3753771` once PID 3753771 exits,
then proceed to finalize.
