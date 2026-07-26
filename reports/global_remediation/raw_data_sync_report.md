> GLOBAL_EVALUATION_REMEDIATION (CC4) · repo mechanism_UED @ Henry-branch · HEAD=UNVERIFIED (git access BLOCKED)
> Environment: JAX/Craftax ABSENT, experiment server UNREACHABLE, GPU idle (not used), no CC2/CC3 processes.
> Discipline: read-only w.r.t. CC2/CC3 & old audit; NEW_TRAINING_RUNS=0; MISSING/UNVERIFIED never relabeled FAIL.

# Raw Data Sync Report

Status: **BLOCKED** (environment). Connectivity probes: github raw TCP REACHABLE, but git via proxy
(127.0.0.1) down and direct HTTPS handshake fails after 21s; experiment server ADDRESS_UNKNOWN/UNREACHABLE.
Single bounded probes, no polling. Server originals not moved/renamed/mtime-changed. No missing-field
inference (NO_SILENT_ASSUMPTION).

Consequence: cannot freeze origin HEAD (FU-8 BLOCKED) and cannot sync W512/P7/P8/P9 per-world data. Sync
deferred to a connected host; expected paths + minimum data needs recorded per experiment in
server_raw_data_manifest.json. Missing items MD-1..MD-10 in global_missing_raw_data_updated.json.

## L_SEQ resolution (MISS-6) — primary evidence found
- smoke/repro = 129 (run_p2_full_smoke.py:66 comment "formal run uses 512"; run_p2_full_levelB.py:74; posthoc_attribution.py:65)
- formal/RMT16 = 512 (requirement matrix; RMT16/P2-Full-A v2.1 frozen config)
Interpretation: both exist as distinct configs; the canonical freeze must pin which applies to each reported
number. W512 repro used 129; RMT16 uses 512; matched control pins 512 with an explicit CONFLICT NOTE.
