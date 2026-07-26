> GLOBAL_EVALUATION_REMEDIATION (CC4) · repo mechanism_UED @ Henry-branch · HEAD=UNVERIFIED (git access BLOCKED)
> Environment: JAX/Craftax ABSENT, experiment server UNREACHABLE, GPU idle (not used), no CC2/CC3 processes.
> Discipline: read-only w.r.t. CC2/CC3 & old audit; NEW_TRAINING_RUNS=0; MISSING/UNVERIFIED never relabeled FAIL.

# Canonical Evaluator Spec (CANONICAL_EVALUATOR_V1)

Single official evaluation protocol. Anchor = eval_phase2_unified.py (SHA
224514026aefd273a6647e055fc2e1a434760dc5f4b6b9acd0624bbba57035a1, verified == local file).

## Frozen fields (with source-line citations)
- action_mode ∈ {stochastic, argmax}; stochastic `pi.sample(seed=a_rng)` :146; `policy_mode="stochastic"` :200
- EnvParams(max_timesteps=4096) :82 · EVAL_SEED=42 :77 · NUM_ENVS=256 · spawn_floor=2
- wrapper DistributedMultiTaskOptimisticLogWrapper(s4_base, PRNGKey(0), condition_on_task=True,
  optimistic_reset_ratio=16, mode="score", bonus_type="none") :104/:121
- success=seen|(info_acc>0) :190 · timeout :191/194 · died :192 · floor3=max_floor>=3 :196 · cond_kill :198
- per-world arrays :206–208 · self-SHA :80/:417

## Mandatory assertions
- A1 print action_mode at startup · A2 write action_mode to output · A3 behavioral test (argmax==1 unique,
  stochastic varies) · A4 partial-restore HARD-FAIL (RestoreLeafMismatch; no silent fallback) ·
  A5 memory-isolation probe (GATE5) · A6 done-reset probe (GATE6).
- argmax memory-off :277 is a DIAGNOSTIC probe, NEVER the policy mode.

## Reference impl + dry-run
tools/canonical_evaluator.py `--dry-run` → prints/writes action_mode, GATE4 HARD-FAIL verified, provenance
built, paired_eligible gated. 7 fixes recorded in global_evaluator_diff.md (E1–E7); 10-row registry in
global_evaluator_registry_fixed.csv (P7_BROKEN QUARANTINE; W512_A_SIDE_UNIFIED dcf7fe20; W512_P2REPLAY f76bb53c).
