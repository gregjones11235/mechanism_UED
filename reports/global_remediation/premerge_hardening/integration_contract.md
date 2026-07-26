# Integration & scientific-claims contract (CC4 premerge hardening -- nine)

- UTC: `2026-07-26T13:35:02Z` ; branch `henry/reviewed-global-evaluation` ; HEAD `2a89f393113d26a4e022646ba9d26c4d8c2b0dad`

## 1. CC4 allowed integration file range
- Reviewed baseline already in branch: 54 frozen remediation files (45 audit_outputs/global_remediation_20260726T095819Z/ + 11 reports/global_remediation/) committed at 2a89f393; fresh-checkout 54 OK / 0 FAILED.
- New this round:
  - `reports/global_remediation/premerge_hardening/  (all premerge-hardening reports, CSV/JSON/MD)`
  - `reports/global_remediation/world_set_materialization_runbook.md`
  - `reports/global_remediation/world_set_materialization_runbook.json`
  - `tools/global_evaluation/materialize_world_set_twice.py`
- NOT entering integration:
  - audit_outputs/global_raw_data_extract_20260726T110032Z/ (local tar extract; NOT git-tracked)
  - audit_outputs/global_evidence_closure_20260726T110032Z/ (evidence closure; delivered via evidence tar, not this branch)
  - audit_outputs/global_world_set_v1/ (world-set recipe deliverable; evidence tar, not this branch)
  - CC3 independent clone (C:/Users/Lenovo/Desktop/.../mechanism_UED_Henry_worktree) -- DO NOT TOUCH
  - D:/cc4_gen/* (ephemeral report generators; never committed)

## 2. .gitattributes scope -- **CC4_GITATTRIBUTES_SCOPE = PASS**
- .gitattributes files in worktree: ['.gitattributes'] (single root file)
- rules: `audit_outputs/global_remediation_20260726T095819Z/** -text` ; `reports/global_remediation/** -text`
- exact match expected: True ; all 54 frozen files under a -text rule: True
- effect: -text on the two evidence dirs ONLY; no global wildcard.

## 3. 54 frozen files: no text conversion in merge
- fresh-checkout sha256sum -c: {'OK': 54, 'FAILED': 0, 'cwd': 'worktree root (BASE-equivalent)', 'sumfile': 'audit_outputs/global_remediation_20260726T095819Z/SHA256SUMS'}
- CC4_FRESH_CHECKOUT_INTEGRITY = PASS

## 4/5. CC2 / CC3 untouched
- CC2_FILES_TOUCHED = false ; CC3_FILES_TOUCHED = false (CC3 = READ-ONLY D052 refactor).

## 6. world-set status
- GLOBAL_WORLD_RECIPE = PASS ; GLOBAL_WORLD_SET_HASH = BLOCKED_SOURCE_UNVERIFIED
- deliverable tier = WORLD_RECIPE_MANIFEST + WORLD_INDEX_MANIFEST (SPEC ONLY) ; is_materialized_world = False

## 7. raw-data status
- W512=LOCAL_ARCHIVE_ONLY_VERIFIED ; P7=LOCAL_ARCHIVE_ONLY_VERIFIED ; P8=LOCAL_ARCHIVE_ONLY_VERIFIED ; P9=LOCAL_ARCHIVE_ONLY_VERIFIED
- RAW_DATA_SYNC_COMPLETE = False ; W512_P2Replay sub-claim = SUMMARY_ONLY

## 8. Exact Resume status
- HARNESS=READY ; EXECUTION=NOT_RUN ; BITEXACT=NOT_CLAIMED (self-test PASS != real bit-exact pass)
- per experiment: P2=PARTIAL, W512=BLOCKED_MISSING_STATE, RMT16=BLOCKED_MISSING_STATE, D052=NOT_APPLICABLE, REFERENCE_LC_family=PARTIAL

## 9. matched Replay status
- BASE_GTRXL_MATCHED_REPLAY_CONTROL = READY_NOT_AUTHORIZED ; match_verdict = SPEC_SINGLE_DIFFERENCE_CONFIRMED + FIELD_VERIFICATION_PARTIAL
- field counts = {'identical': 23, 'unverified_must_freeze': 5, 'different': 0} ; is NOT_MATCHED = False ; L_SEQ NOT auto-picked

## 10. Allowed vs forbidden scientific claims

### ALLOWED
- remediation engineering framework is complete
- archived metrics are recomputable according to the ACTUAL raw-data status (per-world arrays from the local archive; 0 mismatch on recomputed arms)
- Exact Resume harness is READY
- matched Replay config design is READY (single-difference architecture confirmed)

### FORBIDDEN
- the world set has been really materialized
- checkpoints have been re-evaluated
- training has been reproduced
- Exact Resume is fully bit-exact
- matched Replay has been run
- the long-memory mechanism has been validated

## Discipline
- no push, no merge, no rebase, no force push, no git add .
- no training, no formal evaluation, no real Exact Resume continuation, no matched Replay run
- 54 frozen files unmodified; SHA256SUMS not rewritten; no fabricated world_set_hash
- recipe/index manifest never called a materialized world
