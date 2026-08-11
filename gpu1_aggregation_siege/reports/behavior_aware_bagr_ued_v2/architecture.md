# BA-BAGR-UED (D052-v2) Architecture — engineering dry run

`d052/bagr_ued/` — a GLOBAL UED controller + Tier3 FRONT bottleneck signal +
multi-role Student failure-behavior review board + counterfactual environment
induction. NOT a Tier3-only trainer, NOT trajectory imitation, NOT expert
demonstration, NOT action guidance, NOT reward shaping, NOT a hand-crafted
curriculum.

## Closed loop (mock evidence this round)

Student training rollout (mock) -> TrainingTrajectoryEvidenceAdapter
-> DeterministicEventExtractor (8 plugin detectors, symbolic adapter)
-> BehaviorClipSelector -> ReviewBoard [StudentModeler -> BehaviorAuditor
-> CausalFailureAnalyst -> InterventionTutor -> Explorer -> CriticSkeptic]
-> ReviewBoardReconciler (rule-based, provenance-bound)
-> CounterfactualEnvironmentBuilder (control + single-axis + capped factorial)
-> GlobalTaskParamsProposer (MOCK adapter; REAL = BLOCKED_EXTERNAL_DEPENDENCY)
-> LegalityGate -> front_regret + global_regret + behavioral_gap (SEPARATE)
-> learnability + learning_progress + diversity -> Soft Copeland (>=8 inputs,
alpha split visible) -> BudgetAllocator (12 UED + 4 global anchors)
-> ProposalArchive refresh (DRY RUN).

## Boundaries

* TRAINING_AUTHORIZED=False,
  FORMAL_EVALUATION_AUTHORIZED=False,
  REAL_LLM_CALLS_AUTHORIZED=False
  (real_llm_calls=0,
  mock_llm_calls=6).
* Two independent fail-closed guards: TrajectorySupervisionGuard (no
  supervision keys, no direct action advice in ANY output) and
  FormalEvaluationLeakageGuard (no FORMAL_FRONT/BACK/FULL, FROZEN_BANK, or
  certificate-private-state provenance enters the board).
* Deterministic: every hash is canonical-JSON sha256; detector provenance
  carries detector_source_sha256; selection is bit-identical replay.

## Dry-run numbers (this window)

anomalies=4 clips=2
accepted_findings=4
supported_hypotheses=13
accepted_interventions=7
legal_descriptors=18
budget: 12 UED + 4 anchors
plan_status=OK
