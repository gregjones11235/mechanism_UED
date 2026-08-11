# Role responsibility matrix

| Role | Answers | MUST NOT output | Output schema |
|---|---|---|---|
| StudentModeler | current Student capability snapshot, recurring difficulties | causes, proposals, advice | StudentModelSnapshot |
| BehaviorAuditor | WHAT behaviors, WHERE is evidence, recurrence, severity, possibly-incidental note | failure causes, env proposals, action advice | behavior_findings[] |
| CausalFailureAnalyst | MULTIPLE competing causes per finding (closed vocabulary), testable prediction, counterfactual variables | single-cause attribution, correlation-as-causation, Student actions | causal_hypotheses[] |
| InterventionTutor | environment-induction axes (legal TaskParams vocabulary), controls, counterfactual groups, expected GLOBAL effect | "flee/walk/don't sleep", action sequences, reward changes | intervention_hypotheses[] |
| Explorer | environment families DIFFERENT from Tutor's axes; novelty, difference, prediction, global value, side effects | Tier3-only framing, action advice | alternative_environment_proposals[] |
| CriticSkeptic | 9 review dimensions; SEPARATE reject-derivation vs selection evidence; critic penalties | freezing REAL_CANONICAL_* rules (stay PENDING) | critique_items[] + two evidence blocks |
| ReviewBoardReconciler | rule-based decisions with full provenance bindings (NOT majority vote) | letting role outputs directly override selector/curriculum | ReconciliationResult |

Every reconciled item binds: role output hash, evidence span hash, prompt
version, backend/model identity, reconciliation rule version
(`bagr_ued.reconcile.v1`).
