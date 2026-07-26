# Raw data status re-adjudication matrix (CC4 premerge hardening -- four)

- UTC: `2026-07-26T13:24:56Z` ; task: `GLOBAL_EVALUATION_PREMERGE_EVIDENCE_HARDENING`
- Did NOT conclude from historical summaries: inspected raw_data_manifest.json / server_raw_data_manifest.json / server_raw_data_SHA256SUMS, the real input paths of the recompute script gen_recompute.py, and the local tar extract; recomputed SHA256 / array length / evaluator SHA live; checked git tracking.

## Two evidence layers (must be distinguished)
1. **Committed Git manifest layer**: `sync_status=BLOCKED`; `server_raw_data_SHA256SUMS` is **EMPTY** (2 comment lines, no synced files); records **expected server paths only** (REMOTE_PATH_ONLY level); `locally_available_for_recompute` notes phase2_unified_eval.json (10 arms) recomputed.
2. **Local archive layer**: tar extract `D:/Projects/dicode-codex-director/audit_outputs/global_raw_data_extract_20260726T110032Z` (read-only snapshot of remote /home/oseasy/experiments); extract-root SHA256SUMS has 1536 lines; **NOT git-tracked** (tracked_filecount=0) => **not in current branch**.

## RAW_DATA_SYNC is NOT COMPLETE
Per-world arrays exist ONLY in the local tar extract (not in the current git checkout), and the committed manifest server SHAs are EMPTY => strict COMPLETE conditions not met. Per directive, archive-only => **LOCAL_ARCHIVE_ONLY_VERIFIED**; never written as 'current git checkout raw data complete'.

## Per-line adjudication

### W512 -- **LOCAL_ARCHIVE_ONLY_VERIFIED**
- source path(server expected): `bakeoff_phase1/<arm>/eval/*.json (server /home/oseasy/experiments; expected per raw_data_manifest.json)`
- local path(tar extract): `home/oseasy/experiments/bakeoff_phase1/eval_results/eval_w512_full_per_world.jsonl`
- raw array filename: `eval_w512_full_per_world.jsonl` ; field: `dk_success` ; array length: **256**
- SHA256: `471803f3953b9a0e97153a1bf90086af05742c7114690f66edbb5e387c2d6cea`
- reported / recompute-match: 135 / True
- evaluator: `eval_a_side_unified.py (docstring claims seed=42; DATA seed100000 -- flagged divergence)`
- evaluator SHA: `dcf7fe207bb485c47b2669e6c0eb187556d1a4724dd3417a81a83fc88abe5828`
- world seed: **100000** ; world-set identity: **seed100000**
- registered in extract SHA256SUMS: True ; in current Git branch: **False**
- recomputation input status: READ_BY_gen_recompute.py (RAW_FILE per-world)
- completeness: per-world complete (256). NOTE: W512_P2Replay sub-claim has NO per-world array (collapsed 90/95 aggregate only) => that sub-claim = SUMMARY_ONLY / UNVERIFIED_FROM_SUMMARY_ONLY

### P7 -- **LOCAL_ARCHIVE_ONLY_VERIFIED**
- source path(server expected): `gpu1_p7_egomap/eval/*.jsonl (server; expected per manifest)`
- local path(tar extract): `home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu1_p7_egomap/eval_paired_out/p7_egomap_eval/results/p7_egomap_98304_episodes.jsonl`
- raw array filename: `p7_egomap_98304_episodes.jsonl` ; field: `DEFEAT_KOBOLD` ; array length: **256**
- SHA256: `7a05fea8822b553ed8d3ead97cc3615465f3ad85a9c87114d1d4ce99d77971a1`
- reported / recompute-match: 42 / True
- evaluator: `eval_p7_egomap_paired_256.py (p7 eval_paired_out)`
- evaluator SHA: `c082db8b82e86b971d8943bd9275ba8b709ffdc0da198fb236c52ccd56c08325`
- world seed: **100000** ; world-set identity: **seed100000**
- registered in extract SHA256SUMS: True ; in current Git branch: **False**
- recomputation input status: READ_BY_gen_recompute.py (RAW_FILE per-world)
- completeness: per-world complete (256); paired control p7_control_98304_episodes.jsonl also present in extract

### P8 -- **LOCAL_ARCHIVE_ONLY_VERIFIED**
- source path(server expected): `gpu2_p8_longmemory/eval/{migration,final}/*.json (server; expected per manifest)`
- local path(tar extract): `home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu2_p8_longmemory/eval/p8_final_gate.json`
- raw array filename: `p8_final_gate.json (success_per_world EMBEDDED per step)` ; field: `results[*].success_per_world` ; array length: **256 per step (CTL_24576, CTL_49152, CTL_73728, CTL_98304, P8_24576, P8_49152, P8_73728, P8_98304)**
- SHA256: `2467ecf464cc232c978df1592d5e5dcf52b0b3ab2a08fd52f20904b17a59ee8f`
- reported / recompute-match: per-step n_success embedded in gate JSON / verified per arm in gen_recompute (0 mismatch)
- evaluator: `P8 gate evaluator (eval_p8_final.py)`
- evaluator SHA: `da0baba39200e5102c4f546d20b616a17546bd648c42263e247ce09026dfd6c8` (closed_loop=True)
- world seed: **42** ; world-set identity: **seed42**
- registered in extract SHA256SUMS: True ; in current Git branch: **False**
- recomputation input status: READ_BY_gen_recompute.py (RAW_FILE per-world, EMBEDDED in gate JSON)
- completeness: per-world embedded for steps + paired CTL_*; migration gate (p8_migration_gate.json) also present

### P9 -- **LOCAL_ARCHIVE_ONLY_VERIFIED**
- source path(server expected): `gpu3_p9_authentic_reset/eval/*.json (server; expected per manifest)`
- local path(tar extract): `home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu3_p9_authentic_reset/eval/p9_final_gate.json`
- raw array filename: `p9_final_gate.json (success_per_world EMBEDDED per step)` ; field: `results[*].success_per_world` ; array length: **256 per step (CTL_24576, CTL_49152, CTL_73728, CTL_98304, P9_24576, P9_49152, P9_73728, P9_98304)**
- SHA256: `7c006c19522656085eb10ceeb58ae8de4f11e4b0755e2a45b30320077ff62776`
- reported / recompute-match: per-step n_success embedded in gate JSON / verified per arm in gen_recompute (0 mismatch)
- evaluator: `P9 gate evaluator (eval_p9_final.py)`
- evaluator SHA: `eff1d77cae6d229a89c2580009076d909f86e7f636244b909cfbc3cb41abd627` (closed_loop=True)
- world seed: **42** ; world-set identity: **seed42**
- registered in extract SHA256SUMS: True ; in current Git branch: **False**
- recomputation input status: READ_BY_gen_recompute.py (RAW_FILE per-world, EMBEDDED in gate JSON)
- completeness: per-world embedded for steps + paired CTL_*. NOTE: P9 exact-resume = snapshot-restore check (P9_VALIDATE), NOT an A/B continuation gate

## World-set separation
- seed100000: W512 (bakeoff), P7; seed42: P8, P9. **Never pool/pair across sets.**

## Conclusions
- W512 / P7 / P8 / P9 = **LOCAL_ARCHIVE_ONLY_VERIFIED** (arrays locally present, SHA recomputable, length 256 verifiable, gen_recompute.py really reads them, evaluator/world traceable; **but not in git**).
- W512_P2Replay sub-claim: no per-world array (only collapsed 90/95 aggregate) => **SUMMARY_ONLY / UNVERIFIED_FROM_SUMMARY_ONLY**; no strong claim.
- P8/P9 per-world arrays are EMBEDDED in the gate JSON, and each gate's evaluator_sha256 forms a closed loop with the source .py SHA (P8 da0baba39200, P9 eff1d77cae6d).

## Discipline
- Did NOT modify raw per-world arrays; did NOT modify the 54 frozen files; read-only throughout.
