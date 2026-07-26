# Pure-logic gate report (CC4 premerge hardening -- ten)

- UTC: `2026-07-26T13:36:53Z`
- Regression gates: **PASS=13, PARTIAL=1, BLOCKED=1, FAIL=0** (total 15)
- Allowed final state satisfied (FAIL=0, PARTIAL=1, BLOCKED=1): **True**

## Regression gates

| Gate | Status | Name |
|---|---|---|
| GATE1 | **PASS** | CANONICAL_EVALUATOR_SINGLE_SOURCE |
| GATE2 | **PASS** | WORLD_MANIFEST_FROZEN |
| GATE3 | **PARTIAL** | WORLD_SET_HASH_AVAILABLE |
| GATE4 | **PASS** | PARTIAL_RESTORE_HARD_FAIL |
| GATE5 | **PASS** | MEMORY_ISOLATION_PROBE |
| GATE6 | **PASS** | DONE_RESET_PROBE |
| GATE7 | **PASS** | OFFICIAL_TIER_SOURCE |
| GATE8 | **PASS** | TIER_COUNTS_AND_FROZEN_FACTS |
| GATE9 | **PASS** | BASELINE_SINGLE_IDENTITY |
| GATE10 | **PASS** | PAIRED_COMPARISON_ELIGIBILITY |
| GATE11 | **BLOCKED** | RESUME_STATE_TESTS_PRESENT |
| GATE12 | **PASS** | RAW_DATA_BEFORE_STRONG_CLAIM |
| GATE13 | **PASS** | ACTION_MODE_EXPLICIT |
| GATE14 | **PASS** | EXACT_RESUME_MISSING_COMPONENT_DETECT |
| GATE15 | **PASS** | MATCHED_REPLAY_CONFIG_MATCHES_P2_EXCEPT_NETWORK |

- PARTIAL: GATE3 WORLD_SET_HASH_AVAILABLE (JAX absent; GLOBAL_WORLD_SET_HASH=BLOCKED_SOURCE_UNVERIFIED).
- BLOCKED: GATE11 RESUME_STATE_TESTS_PRESENT (0 binaries; REAL_EXACT_RESUME_EXECUTION=NOT_RUN).

## Individual pure-logic self-tests

| Test | Result |
|---|---|
| baseline_identity | **PASS** |
| action_mode_consistency | **PASS** |
| tier_registry | **PASS** |
| exact_resume_harness_self_test | **PASS** |
| canonical_evaluator_schema | **PASS** |

- tier_registry: GATE7/GATE8 vs installed craftax = BLOCKED_ON_CRAFTAX (craftax ABSENT), NOT FAIL.

## Build / hygiene
- python -m compileall (frozen tools): rc=0 => PASS
- python -m compileall (new script): rc=0 => PASS
- git diff --check: rc=0 => (clean)

## Statement
- No gate was lowered to convert PARTIAL/BLOCKED into PASS. PARTIAL=1 (world_set_hash) and BLOCKED=1 (resume-state tests) are retained; FAIL=0. tier_registry GATE7/8 vs installed craftax = BLOCKED_ON_CRAFTAX (craftax ABSENT), NOT FAIL.
