"""Artifact writer for reports/behavior_aware_bagr_ued_v2/ (task section 16).

Writes the 16 required artifacts from one DryRunResult, then SHA256SUMS last
(over every other artifact). Deterministic JSON (sort_keys, separators). The
markdown audits quote the run's own evidence (hashes, statuses), never
performance claims.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

from d052.bagr_ued import (behavior_auditor, behavior_taxonomy,
                           causal_failure_analyst, constants as C,
                           critic_skeptic, explorer, intervention_tutor,
                           mock_llm_backend, review_board, student_modeler)
from d052.bagr_ued.formal_evaluation_leakage_guard import FormalLeakageViolation
from d052.bagr_ued.review_contracts import RoleEnvelope
from d052.bagr_ued.trajectory_supervision_guard import GuardViolation

ARTIFACTS = (
    "architecture.md",
    "role_responsibility_matrix.md",
    "trajectory_evidence_policy.md",
    "behavior_taxonomy.json",
    "behavior_review_board_contract.json",
    "unsafe_rest_synthetic_trace.json",
    "unsafe_rest_review_outputs.json",
    "counterfactual_environment_plan.json",
    "ued_nature_audit.md",
    "global_not_tier3_only_audit.md",
    "trajectory_supervision_guard_report.json",
    "formal_leakage_guard_report.json",
    "dry_run_certificate.json",
    "test_report.json",
    "dependency_blockers.json",
    "SHA256SUMS",
)


# NOTE: every artifact is written with explicit LF line endings so the
# recorded SHA256 hashes match the git-normalized (LF) blob content and stay
# verifiable with `sha256sum -c` on any platform (repo uses core.autocrlf).
def _dump_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False,
                  default=str)
        f.write("\n")


def _dump_md(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


#: role -> (module, output schema class name, must-NOT-output list)
_ROLE_CONTRACT = (
    (C.ROLE_STUDENT_MODELER, student_modeler, "StudentModelSnapshot",
     ["causes", "proposals", "action advice"]),
    (C.ROLE_BEHAVIOR_AUDITOR, behavior_auditor, "BehaviorAuditorOutput",
     ["failure causes", "environment proposals", "action advice"]),
    (C.ROLE_CAUSAL_FAILURE_ANALYST, causal_failure_analyst,
     "CausalAnalystOutput",
     ["single-cause attribution", "correlation asserted as causation",
      "direct Student actions"]),
    (C.ROLE_INTERVENTION_TUTOR, intervention_tutor, "InterventionTutorOutput",
     ["action instructions", "fixed action sequences", "reward modifications"]),
    (C.ROLE_EXPLORER, explorer, "ExplorerOutput",
     ["Tier3-only framing", "action advice"]),
    (C.ROLE_CRITIC_SKEPTIC, critic_skeptic, "CriticSkepticOutput",
     ["freezing REAL_CANONICAL_CRITIC_REJECT_DERIVATION_RULE",
      "freezing REAL_CANONICAL_CRITIC_SELECTION_POLICY",
      "merging reject-derivation evidence with selection evidence"]),
)


def _review_board_contract() -> dict:
    """The versioned contract of the six-role board (artifact #5)."""
    backend = mock_llm_backend.DeterministicMockBackend()
    roles = [dict(role=role, sequence=i, module=mod.__name__,
                  prompt_version=mod.PROMPT_VERSION, output_schema=schema,
                  backend_id=backend.backend_id, model_id=backend.model_id,
                  context_fold_key=review_board._CONTEXT_KEY[role],
                  must_not_output=must_not)
             for i, (role, mod, schema, must_not) in enumerate(_ROLE_CONTRACT)]
    return dict(
        schema="bagr_ued.review_board_contract.v1",
        bagr_ued_version=C.BA_BAGR_UED_VERSION,
        role_execution_order=list(C.REVIEW_BOARD_ROLES),
        roles=roles,
        envelope_contract=dict(
            model="RoleEnvelope",
            fields=sorted(RoleEnvelope.model_fields),
            note="every role output is wrapped: role output hash + prompt "
                 "version + backend/model identity + sequence"),
        gates=dict(
            trajectory_supervision_guard=dict(
                scope="EVERY role parsed output + the whole dry-run result",
                codes=[GuardViolation.TRAJECTORY_SUPERVISION_KEY_FORBIDDEN,
                       GuardViolation.DIRECT_ACTION_ADVICE_FORBIDDEN,
                       GuardViolation.SERIALIZED_GUARD_LIMIT_EXCEEDED],
                failure_mode="FAIL_CLOSED",
                forbidden_supervision_keys=sorted(
                    C.FORBIDDEN_SUPERVISION_KEYS),
                forbidden_supervision_key_aliases=sorted(
                    C.FORBIDDEN_SUPERVISION_KEY_ALIASES),
                serialized_string_parsing=dict(
                    policy="trim-then-JSON-looking strings are parsed and "
                           "the full guard re-runs inside; parse failure "
                           "falls back to plain-text NL patterns (never a "
                           "lenient skip); limit excess fails closed",
                    max_parse_depth=C.MAX_SERIALIZED_PARSE_DEPTH,
                    max_string_length=C.MAX_SERIALIZED_STRING_LENGTH,
                    max_container_items=C.MAX_SERIALIZED_CONTAINER_ITEMS)),
            formal_evaluation_leakage_guard=dict(
                scope="board input context + evidence sources + regret inputs",
                codes=[FormalLeakageViolation.FORMAL_EVALUATION_LEAKAGE,
                       FormalLeakageViolation.FORBIDDEN_PROVENANCE_KEY,
                       FormalLeakageViolation.SOURCE_NOT_DECLARED],
                failure_mode="FAIL_CLOSED",
                forbidden_sources=sorted(C.FORBIDDEN_EVIDENCE_SOURCES),
                allowed_sources=sorted(C.ALLOWED_EVIDENCE_SOURCES))),
        reconciliation_rule_version=C.RECONCILIATION_RULE_VERSION,
        reconciler_note="rule-based provenance-bound decisions; NOT majority "
                        "vote; role outputs are CANDIDATE hypotheses only",
        pending_rules=dict(
            real_canonical_critic_reject_derivation_rule=
                C.REAL_CANONICAL_CRITIC_REJECT_DERIVATION_RULE,
            real_canonical_critic_selection_policy=
                C.REAL_CANONICAL_CRITIC_SELECTION_POLICY,
            note="both must remain PENDING; CriticSkepticOutput validators "
                 "refuse any frozen value (CRITIC_RULE_FROZEN_FORBIDDEN / "
                 "CRITIC_POLICY_FROZEN_FORBIDDEN)"),
        real_llm_backend=dict(
            status="NOT_CONNECTED", mock_backend_id=backend.backend_id,
            real_calls_this_round=int(backend.real_calls)))


def write_all(result, raw_rollout: dict, reports_dir: str, *,
              guard_reports: dict, test_report: Optional[dict] = None) -> dict:
    """Write every artifact; return {filename: sha256}."""
    os.makedirs(reports_dir, exist_ok=True)
    d = result.model_dump()
    cert = d["dry_run_certificate"]
    rec = d["reconciliation"]

    # ------------------------------------------------------------------ json
    _dump_json(os.path.join(reports_dir, "behavior_taxonomy.json"),
               behavior_taxonomy.taxonomy_document())
    _dump_json(os.path.join(reports_dir, "unsafe_rest_synthetic_trace.json"),
               dict(schema="bagr_ued.synthetic_trace.v1",
                    source_allowed="SYNTHETIC_TEST_TRACE / "
                                   "GENERATIVE_TRAINING_ENV only",
                    rollout=raw_rollout))
    _dump_json(os.path.join(reports_dir, "unsafe_rest_review_outputs.json"),
               dict(schema="bagr_ued.review_outputs.v1",
                    board=d["board"], reconciliation=rec,
                    ued_nature_assertions=d["ued_nature_assertions"]))
    _dump_json(os.path.join(reports_dir, "counterfactual_environment_plan.json"),
               dict(schema="bagr_ued.counterfactual_plan.v1",
                    plan=d["counterfactual_plan"],
                    descriptors=d["descriptors"],
                    rejected_descriptors=d["rejected_descriptors"],
                    proposal_distribution_hash=d["proposal_distribution_hash"]))
    _dump_json(os.path.join(reports_dir, "behavior_review_board_contract.json"),
               _review_board_contract())
    _dump_json(os.path.join(reports_dir, "trajectory_supervision_guard_report.json"),
               guard_reports["supervision"])
    _dump_json(os.path.join(reports_dir, "formal_leakage_guard_report.json"),
               guard_reports["leakage"])
    _dump_json(os.path.join(reports_dir, "dry_run_certificate.json"), cert)
    _dump_json(os.path.join(reports_dir, "test_report.json"),
               test_report or dict(status="NOT_RUN_YET"))
    _dump_json(os.path.join(reports_dir, "dependency_blockers.json"), dict(
        schema="bagr_ued.dependency_blockers.v1",
        blocking_external_dependencies=[
            dict(id="REAL_TASKPARAMS_ADAPTER",
                 status=C.REAL_TASKPARAMS_ADAPTER,
                 note="real Global TaskParams adapter not delivered; mock "
                      "adapter + legality gate used; real fields never guessed"),
            dict(id="REAL_LLM_BACKEND",
                 status="NOT_CONNECTED",
                 note="review board roles run on the deterministic mock "
                      "backend; real providers require director authorization "
                      "(REAL_LLM_CALLS_AUTHORIZED=false)"),
            dict(id="REAL_CANONICAL_CRITIC_REJECT_DERIVATION_RULE",
                 status=C.REAL_CANONICAL_CRITIC_REJECT_DERIVATION_RULE,
                 note="must stay PENDING until a director decision"),
            dict(id="REAL_CANONICAL_CRITIC_SELECTION_POLICY",
                 status=C.REAL_CANONICAL_CRITIC_SELECTION_POLICY,
                 note="must stay PENDING until a director decision"),
            dict(id="REAL_CANONICAL_POOL",
                 status="OUT_OF_SCOPE_THIS_ROUND",
                 note="the 4 global anchors are synthetic ids this round"),
            dict(id="CC4_COMMON_EVALUATOR_SLOWGRU_FAMILY",
                 status="BLOCKED_COMMON_EVALUATOR_SLOWGRU_FAMILY_NOT_REGISTERED",
                 note="carried from CC3 binding task; unrelated to this dry "
                      "run but part of the same student-pool work stream"),
        ]))

    # ------------------------------------------------------------------- md
    _dump_md(os.path.join(reports_dir, "architecture.md"), f"""\
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

* TRAINING_AUTHORIZED={C.TRAINING_AUTHORIZED},
  FORMAL_EVALUATION_AUTHORIZED={C.FORMAL_EVALUATION_AUTHORIZED},
  REAL_LLM_CALLS_AUTHORIZED={C.REAL_LLM_CALLS_AUTHORIZED}
  (real_llm_calls={cert.get('real_llm_calls')},
  mock_llm_calls={cert.get('mock_llm_calls')}).
* Two independent fail-closed guards: TrajectorySupervisionGuard (no
  supervision keys, no direct action advice in ANY output) and
  FormalEvaluationLeakageGuard (no FORMAL_FRONT/BACK/FULL, FROZEN_BANK, or
  certificate-private-state provenance enters the board).
* Deterministic: every hash is canonical-JSON sha256; detector provenance
  carries detector_source_sha256; selection is bit-identical replay.

## Dry-run numbers (this window)

anomalies={len(d['anomalies'])} clips={len(d['clips'])}
accepted_findings={len(rec['accepted_behavior_findings'])}
supported_hypotheses={len(rec['supported_causal_hypotheses'])}
accepted_interventions={sum(1 for i in rec['accepted_intervention_hypotheses'] if i['decision'] == 'accepted')}
legal_descriptors={len(d['descriptors'])}
budget: {cert.get('ued_slots_allocated')} UED + {cert.get('anchor_slots_allocated')} anchors
plan_status={d['budget_plan']['status']}
""")

    _dump_md(os.path.join(reports_dir, "role_responsibility_matrix.md"), """\
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
""")

    _dump_md(os.path.join(reports_dir, "trajectory_evidence_policy.md"), """\
# Trajectory evidence policy

ALLOWED evidence: current Student generative-training trajectories — action
semantics (resolved via EXTERNAL symbolic adapter; no hardcoded Craftax
action integers, no raw state leaf indices), state/resource change summaries,
threat/damage/achievement/progress/death/timeout events, limited windows
around anomalies.

FORBIDDEN: formal FRONT/BACK/FULL state payloads; formal evaluation per-state
trajectories; formal map/ladder positions; expert trajectories; Reference
action sequences as demonstration; hidden policy state as supervision; manual
correct-action labels.

Guards: A. TrajectorySupervisionGuard — rejects in ANY output the keys
recommended_actions / action_sequence_to_follow / waypoints /
expert_demonstration / policy_override / hidden_state_override / reward_delta
/ reward_shaping, and any direct action-advice text (bilingual patterns).
B. FormalEvaluationLeakageGuard — rejects FORMAL_FRONT / FORMAL_BACK /
FORMAL_FULL / FROZEN_BANK / FORMAL_EVALUATION_CERTIFICATE_PRIVATE_STATE
provenance anywhere in the input. Both fail closed with greppable codes.
""")

    _dump_md(os.path.join(reports_dir, "ued_nature_audit.md"), f"""\
# UED-nature audit

Method identity assertions (this run): {json.dumps(d['ued_nature_assertions'], sort_keys=True, ensure_ascii=False)}

* Environment induction only: interventions move legal TaskParams mutation
  axes; no role emits Student actions.
* Final environment VALUE is never LLM judgment alone: Soft Copeland consumes
  rollout-evidence-based scores (mock this round; real rollout validation
  required before production).
* Reward shaping: none representable in any schema; supervision guard
  additionally rejects reward_delta / reward_shaping keys.
* Curriculum is NOT hand-crafted: proposals arise from deterministic
  extraction -> role review -> reconciliation -> counterfactual construction.
""")

    families = sorted({p["environment_family"] for p in
                       next(e["parsed_json"] for e in d["board"]["envelopes"]
                            if e["role"] == C.ROLE_EXPLORER)
                       ["alternative_environment_proposals"]})
    _dump_md(os.path.join(reports_dir, "global_not_tier3_only_audit.md"), f"""\
# Global-not-Tier3-only audit

* TRAINING_SCOPE={C.TRAINING_SCOPE}, TIER3_ONLY_TRAINING={C.TIER3_ONLY_TRAINING},
  GLOBAL_SIGNAL_REQUIRED={C.GLOBAL_SIGNAL_REQUIRED}.
* Taxonomy covers GLOBAL anomalies: dangerous resting, resource waste, combat
  freeze, exploration loops, tool misuse, long-term planning failure — not a
  floor2->floor3-specific taxonomy.
* Explorer families proposed this run: {families}
  (every proposed family's primary axes are DISJOINT from the mutation axes
  the Tutor already covered in this window => no Tier3-only collapse; the
  disjoint remainder count is trace-dependent, never padded).
* Budget reserves {C.GLOBAL_CANONICAL_ANCHORS} GLOBAL canonical anchors every
  window, unconditionally.
""")

    # ------------------------------------------------------------------ sums
    return finalize_sha256sums(reports_dir)


def finalize_sha256sums(reports_dir: str) -> dict:
    """(Re)write SHA256SUMS over every artifact except itself; return map."""
    sums = {}
    for name in ARTIFACTS:
        if name == "SHA256SUMS":
            continue
        path = os.path.join(reports_dir, name)
        if not os.path.isfile(path):
            continue
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        sums[name] = h.hexdigest()
    with open(os.path.join(reports_dir, "SHA256SUMS"), "w", encoding="utf-8",
              newline="\n") as f:
        for name in sorted(sums):
            f.write(f"{sums[name]}  {name}\n")
    return sums
