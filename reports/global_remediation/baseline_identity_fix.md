> GLOBAL_EVALUATION_REMEDIATION (CC4) · repo mechanism_UED @ Henry-branch · HEAD=UNVERIFIED (git access BLOCKED)
> Environment: JAX/Craftax ABSENT, experiment server UNREACHABLE, GPU idle (not used), no CC2/CC3 processes.
> Discipline: read-only w.r.t. CC2/CC3 & old audit; NEW_TRAINING_RUNS=0; MISSING/UNVERIFIED never relabeled FAIL.

# Baseline Identity Fix

Bare "Baseline" is FORBIDDEN. Exactly two single-identity baselines (GATE9 PASS):
- TEACHER17500_BASELINE — teacher17500, params d4e85af58b7f87d6, 101/256 = 39.453125%
- CONTROL24576_BASELINE — control_RUN2/ckpt/24576, params ece6fa99…bdabf55, 93/256 = 36.328125%

## Paired-comparison rule (GATE10 PASS)
A paired delta requires IDENTICAL: evaluator_sha256, world_set_hash, success_definition, denominator,
action_mode. checkpoint_path is checked separately (must be KNOWN; MAY differ — it is the varied factor).
Any mismatch → `PAIRED_COMPARISON_NOT_ALLOWED`. tools/baseline_id_validation_tests.py self-test PASSED.
Cross-evaluator / cross-world-set results are never directly compared.
