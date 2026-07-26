> GLOBAL_EVALUATION_REMEDIATION (CC4) · repo mechanism_UED @ Henry-branch · HEAD=UNVERIFIED (git access BLOCKED)
> Environment: JAX/Craftax ABSENT, experiment server UNREACHABLE, GPU idle (not used), no CC2/CC3 processes.
> Discipline: read-only w.r.t. CC2/CC3 & old audit; NEW_TRAINING_RUNS=0; MISSING/UNVERIFIED never relabeled FAIL.

# Global Remediation Test Report

All tests are pure-logic / file-assertion; NO JAX, NO GPU, NO training.

| Tool | Command | Result |
|---|---|---|
| action_mode_consistency_test.py | --self-test | SELF_TEST_PASS |
| tier_registry_test.py | run | PURE_PYTHON_SELF_CHECK_PASS (vs craftax: BLOCKED_ON_CRAFTAX) |
| baseline_id_validation_tests.py | --self-test | BASELINE_VALIDATION_SELF_TEST_PASS |
| exact_resume_harness.py | --self-test | EXACT_RESUME_HARNESS_SELF_TEST_PASS (6/6) |
| canonical_evaluator.py | --dry-run | GATE4 HARD-FAIL verified; provenance built; paired_eligible gated |
| regression_gates.py | run | PASS=13 PARTIAL=1 BLOCKED=1 FAIL=0 |

## GATE 1–15
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

PARTIAL (GATE3) and BLOCKED (GATE11) reflect environment limits (no JAX; CC2 domain) and are explicitly NOT FAIL.
