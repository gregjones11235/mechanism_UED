> GLOBAL_EVALUATION_REMEDIATION (CC4) · repo mechanism_UED @ Henry-branch · HEAD=UNVERIFIED (git access BLOCKED)
> Environment: JAX/Craftax ABSENT, experiment server UNREACHABLE, GPU idle (not used), no CC2/CC3 processes.
> Discipline: read-only w.r.t. CC2/CC3 & old audit; NEW_TRAINING_RUNS=0; MISSING/UNVERIFIED never relabeled FAIL.

# Global Remediation Architecture

## Scope
Fixes the 9 global problems from the Phase-1 causal audit via 8 remediation workstreams + unified provenance
+ 15 regression gates. All deliverables are NEW files in `audit_outputs/global_remediation_20260726T095819Z/`.

## Workstream → artifact map
| # | Problem | Fix | Key artifact |
|---|---|---|---|
| 1 | inconsistent evaluator/inference mode | Canonical Evaluator | global_evaluator_canonical_spec.json, tools/canonical_evaluator.py |
| 2 | world set not frozen/hashed | World Manifest | world_manifests/canonical_worlds_256_seed{42,100000}.json, tools/build_world_manifest.py |
| 3 | baseline identity/metric confusion | Baseline identity | global_baseline_registry_fixed.csv, tools/baseline_id_validation_tests.py |
| 4 | wrong achievement-tier versions | Official tiers | official_achievement_tiers.json, tools/tier_registry_test.py |
| 5 | W512/P7/P8/P9 raw data not synced | Raw-data sync | server_raw_data_manifest.json (BLOCKED), global_missing_raw_data_updated.json |
| 6 | checkpoint only proves saveable | Exact-Resume tool | exact_resume_schema.json, tools/exact_resume_harness.py |
| 7 | missing Base GTrXL matched control | Matched-Replay control | base_gtrxl_matched_replay_config.yaml (READY_NOT_AUTHORIZED) |
| 8 | reports lack provenance manifest | Unified provenance | tools/canonical_evaluator.build_provenance, evaluation_provenance.json |
| 9 | L_SEQ 129/512 not frozen | L_SEQ resolution | global_missing_raw_data_updated.json (primary evidence) |

## Recompute policy
Only Phase2 has local per-world data → recomputed (0 mismatch). Server-only lines stay EVIDENCE_UNVERIFIED
(no summary substitution). Statistics: McNemar + paired bootstrap(12345) + Wilson + Clopper-Pearson.

## Regression gates (this run)
PASS=13 · PARTIAL=1 · BLOCKED=1 · FAIL=0

| Gate | Name | Status |
|---|---|---|
| GATE1 | CANONICAL_EVALUATOR_SINGLE_SOURCE | PASS |
| GATE2 | WORLD_MANIFEST_FROZEN | PASS |
| GATE3 | WORLD_SET_HASH_AVAILABLE | PARTIAL |
| GATE4 | PARTIAL_RESTORE_HARD_FAIL | PASS |
| GATE5 | MEMORY_ISOLATION_PROBE | PASS |
| GATE6 | DONE_RESET_PROBE | PASS |
| GATE7 | OFFICIAL_TIER_SOURCE | PASS |
| GATE8 | TIER_COUNTS_AND_FROZEN_FACTS | PASS |
| GATE9 | BASELINE_SINGLE_IDENTITY | PASS |
| GATE10 | PAIRED_COMPARISON_ELIGIBILITY | PASS |
| GATE11 | RESUME_STATE_TESTS_PRESENT | BLOCKED |
| GATE12 | RAW_DATA_BEFORE_STRONG_CLAIM | PASS |
| GATE13 | ACTION_MODE_EXPLICIT | PASS |
| GATE14 | EXACT_RESUME_MISSING_COMPONENT_DETECT | PASS |
| GATE15 | MATCHED_REPLAY_CONFIG_MATCHES_P2_EXCEPT_NETWORK | PASS |
