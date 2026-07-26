# Pure-logic gate report (round-6 revision -- thirteen)

- UTC: `2026-07-26T15:10:12Z`
- Frozen GATE1-15 (re-run live, frozen script unmodified): **PASS=13 PARTIAL=1 BLOCKED=1 FAIL=0**
- GATE3 = **PARTIAL** (retained; GLOBAL_WORLD_SET_HASH=BLOCKED_SOURCE_UNVERIFIED). GATE11 = **BLOCKED** (resume-state tests).

## CC4 addendum gates (16-19)

| Gate | Status | Name |
|---|---|---|
| GATE16 | **PASS_STATIC_IMPLEMENTATION** | ACTUAL_CRAFTAX_WORLD_MATERIALIZATION_PATH |
| GATE17 | **PASS** | KEY_ONLY_HASH_REJECTED |
| GATE18 | **PARTIAL_ENVIRONMENT_BLOCKED** | EVALUATION_SEED_ENTERS_REAL_RNG_PATH |
| GATE19 | **BLOCKED** | MATERIALIZER_EVALUATOR_SHARED_BUILDER |

- **GATE16 PASS_STATIC_IMPLEMENTATION**: materializer implemented + compiles + self-test PASS + anchor-check vs real canonical sources PASS; reset chain verbatim-anchored; real run NOT_RUN (no JAX/craftax).
- **GATE17 PASS**: assert_materialized rejects prototype shape=True; prototype entry fails closed exit2=True; NEG06 PASS.
- **GATE18 PARTIAL_ENVIRONMENT_BLOCKED**: STATIC: EVAL_SEED=42 -> PRNGKey -> split chain confirmed in source (anchor-check). REAL per-world bytes change on numeric-seed change NOT verifiable on this JAX-less host (NEG02=BLOCKED_ENVIRONMENT). NOT asserted PASS.
- **GATE19 BLOCKED**: strict shared importable builder=BLOCKED (evaluator read-only + inline, task via exec of absolute path). Weak honest substitute: static anchor test PASS (12 anchors, 0 mismatches); wrapper byte-identical (2ded41d8) to the copy the evaluator loads.

- Serializer self-test: **PASS** ; anchor-check vs real sources: **PASS** ; negative tests: PASS=8 BLOCKED=2 FAIL=0
- **ALL_FAIL_ZERO = True**

## Frozen GATE1-15 (live re-run)
```
GATE1   PASS     CANONICAL_EVALUATOR_SINGLE_SOURCE
GATE2   PASS     WORLD_MANIFEST_FROZEN
GATE3   PARTIAL  WORLD_SET_HASH_AVAILABLE
GATE4   PASS     PARTIAL_RESTORE_HARD_FAIL
GATE5   PASS     MEMORY_ISOLATION_PROBE
GATE6   PASS     DONE_RESET_PROBE
GATE7   PASS     OFFICIAL_TIER_SOURCE
GATE8   PASS     TIER_COUNTS_AND_FROZEN_FACTS
GATE9   PASS     BASELINE_SINGLE_IDENTITY
GATE10  PASS     PAIRED_COMPARISON_ELIGIBILITY
GATE11  BLOCKED  RESUME_STATE_TESTS_PRESENT
GATE12  PASS     RAW_DATA_BEFORE_STRONG_CLAIM
GATE13  PASS     ACTION_MODE_EXPLICIT
GATE14  PASS     EXACT_RESUME_MISSING_COMPONENT_DETECT
GATE15  PASS     MATCHED_REPLAY_CONFIG_MATCHES_P2_EXCEPT_NETWORK
```

## Discipline
- frozen GATE1-15 unmodified
- no gate lowered to PASS
- FAIL=0 required and met
- BLOCKED / PARTIAL_ENVIRONMENT_BLOCKED used where environment-blocked, never relabeled PASS
