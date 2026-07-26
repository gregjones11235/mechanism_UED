> GLOBAL_EVALUATION_REMEDIATION (CC4) · repo mechanism_UED @ Henry-branch · HEAD=UNVERIFIED (git access BLOCKED)
> Environment: JAX/Craftax ABSENT, experiment server UNREACHABLE, GPU idle (not used), no CC2/CC3 processes.
> Discipline: read-only w.r.t. CC2/CC3 & old audit; NEW_TRAINING_RUNS=0; MISSING/UNVERIFIED never relabeled FAIL.

# Metric Recomputation Report

Recompute policy: recompute ONLY where per-world data is local (Phase2, 10 arms) → **0 mismatch** vs reported.
Server-only lines (W512/P7/P8/P9/P2/RMT16) are NOT recomputed and NOT substituted from aggregates →
EVIDENCE_UNVERIFIED. Full per-row provenance: evaluator_sha, world_set_hash(REQUIRED), world_recipe_hash,
checkpoint_sha, action_mode, evidence_level. Artifact: global_metric_recomputation_fixed.csv
(arm rows + EVIDENCE_UNVERIFIED rows + 13 paired rows). Claim scope: global_claim_scope_matrix_fixed.csv
(17 claims). Mismatches: global_report_metric_mismatches_fixed.json = 0.

Statistics: paired McNemar (discordant), paired bootstrap CI (seed 12345, 20000), Wilson, Clopper-Pearson.
Signal = p<0.05 AND bootstrap CI not crossing 0. Phase2 causal deltas all match the audit verification.
