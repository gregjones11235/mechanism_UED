"""ReviewBoard assembly (task sections 1 / 10).

Runs the six review roles IN THE FIXED ORDER against one evidence window,
wrapping every output in a RoleEnvelope (role output hash + prompt version +
backend/model identity + sequence):

    StudentModeler -> BehaviorAuditor -> CausalFailureAnalyst ->
    InterventionTutor -> Explorer -> Critic/Skeptic

Hard boundaries enforced here (fail-closed, per role):
  * FormalEvaluationLeakageGuard over the board INPUT context before any role
    runs (no formal/bank provenance reaches a role);
  * TrajectorySupervisionGuard over EACH role's parsed output immediately
    after parsing — a role emitting recommended_actions / "don't sleep" /
    "move away from the monster" fails the whole board closed;
  * real_llm_calls counted from the backend and asserted 0 by the controller.

Role outputs generate CANDIDATE hypotheses only; they never override the
selector or curriculum — that is the Reconciler + Soft Copeland + Budget
chain's deterministic job.
"""
from __future__ import annotations

from typing import List

from d052.bagr_ued import behavior_auditor, causal_failure_analyst, constants as C
from d052.bagr_ued import critic_skeptic, explorer, intervention_tutor
from d052.bagr_ued import student_modeler
from d052.bagr_ued.event_extractor import AnomalyCandidate
from d052.bagr_ued.behavior_clip_selector import BehaviorClip
from d052.bagr_ued.formal_evaluation_leakage_guard import (
    FormalEvaluationLeakageGuard)
from d052.bagr_ued.review_contracts import ReviewBoardOutput, RoleEnvelope
from d052.bagr_ued.symbolic_behavior_clip import (
    assert_valid_symbolic_clip_payload,
    build_symbolic_clip_payload,
)
from d052.bagr_ued.trajectory_evidence import TrajectoryEvidenceBundle
from d052.bagr_ued.trajectory_supervision_guard import TrajectorySupervisionGuard

_ROLE_MODULES = (
    student_modeler,
    behavior_auditor,
    causal_failure_analyst,
    intervention_tutor,
    explorer,
    critic_skeptic,
)

#: base-context key each role's parsed output is folded under for later roles
_CONTEXT_KEY = {
    C.ROLE_STUDENT_MODELER: "student_model_snapshot",
    C.ROLE_BEHAVIOR_AUDITOR: "behavior_findings",
    C.ROLE_CAUSAL_FAILURE_ANALYST: "causal_hypotheses",
    C.ROLE_INTERVENTION_TUTOR: "intervention_hypotheses",
    C.ROLE_EXPLORER: "alternative_environment_proposals",
    C.ROLE_CRITIC_SKEPTIC: "critic_output",
}


def build_base_context(bundle: TrajectoryEvidenceBundle,
                       anomalies: List[AnomalyCandidate],
                       clips: List[BehaviorClip],
                       detector_manifest: List[dict]) -> dict:
    """The shared evidence context every role receives (symbolic only).

    CC3 fix2 (§12): the six roles receive BOUNDED, de-identified, per-step
    SYMBOLIC behavior clips — not only anomaly labels and clip metadata. Each
    clip payload is built from the admitted evidence bundle, then validated
    fail-closed (both guards + raw-exposure scan + source admissibility +
    payload hash + step/byte limits) BEFORE it enters the context. A clip
    that fails validation fails the whole board closed. Formal-evaluation
    trajectories remain forbidden at the bundle source level.
    """
    leakage = FormalEvaluationLeakageGuard()
    supervision = TrajectorySupervisionGuard()
    symbolic_clips = []
    for c in clips:
        payload = build_symbolic_clip_payload(bundle, c)
        # fail-closed per clip: guards + raw exposure + hash + limits
        assert_valid_symbolic_clip_payload(
            payload, leakage_guard=leakage, supervision_guard=supervision)
        symbolic_clips.append(payload.model_dump())
    return dict(
        bundle_id=bundle.bundle_id,
        source=bundle.source.value,
        symbolic_adapter_version=bundle.symbolic_adapter_version,
        global_scope=dict(training_scope=C.TRAINING_SCOPE,
                          tier3_only_training=C.TIER3_ONLY_TRAINING,
                          global_signal_required=C.GLOBAL_SIGNAL_REQUIRED),
        detector_manifest=detector_manifest,
        anomalies=[a.model_dump() for a in anomalies],
        clips=[c.model_dump() for c in clips],
        # CC3 fix2 (§12): real per-step symbolic behavior evidence + metadata
        symbolic_behavior_clips=symbolic_clips,
        symbolic_clip_contract=dict(
            schema_version="bagr_ued.symbolic_clip.v1",
            max_clip_steps=C.MAX_CLIP_STEPS,
            max_clips_per_review_window=C.MAX_CLIPS_PER_REVIEW_WINDOW,
            max_event_semantics_per_step=C.MAX_EVENT_SEMANTICS_PER_STEP,
            max_resource_fields=C.MAX_RESOURCE_FIELDS,
            max_serialized_payload_bytes=C.MAX_SERIALIZED_PAYLOAD_BYTES,
            raw_action_integer_exposed=False,
            raw_state_exposed=False,
            formal_trajectory_exposed=False),
    )


class ReviewBoard:
    def __init__(self, backend) -> None:
        self.backend = backend
        self.supervision_guard = TrajectorySupervisionGuard()
        self.leakage_guard = FormalEvaluationLeakageGuard()

    def run(self, bundle: TrajectoryEvidenceBundle,
            anomalies: List[AnomalyCandidate],
            clips: List[BehaviorClip],
            detector_manifest: List[dict]) -> ReviewBoardOutput:
        context = build_base_context(bundle, anomalies, clips, detector_manifest)

        # gate 1: no formal-evaluation provenance reaches any role
        leak = self.leakage_guard.assert_clean(context, label="board_context")

        envelopes: List[RoleEnvelope] = []
        for seq, module in enumerate(_ROLE_MODULES):
            envelope = module.run(context, self.backend, sequence=seq)
            # gate 2: no supervision / action advice in ANY role output
            self.supervision_guard.assert_clean(
                envelope.parsed_json, label=f"role_output:{module.ROLE}")
            envelopes.append(envelope)
            # fold the parsed output into the context for downstream roles
            key = _CONTEXT_KEY[module.ROLE]
            if key == "behavior_findings":
                context[key] = envelope.parsed_json["behavior_findings"]
                # CC3 fix2 (§13): provisional out-of-taxonomy hypotheses are
                # surfaced for downstream visibility ONLY — the reconciler's
                # acceptance rules and the selector/budget/archive chain never
                # consume this key.
                context["provisional_anomaly_hypotheses"] = \
                    envelope.parsed_json.get(
                        "provisional_anomaly_hypotheses", [])
            elif key == "causal_hypotheses":
                context[key] = envelope.parsed_json["causal_hypotheses"]
            elif key == "intervention_hypotheses":
                context[key] = envelope.parsed_json["intervention_hypotheses"]
            elif key == "alternative_environment_proposals":
                context[key] = envelope.parsed_json[
                    "alternative_environment_proposals"]
            else:
                context[key] = envelope.parsed_json

        return ReviewBoardOutput(
            bundle_id=bundle.bundle_id,
            envelopes=envelopes,
            supervision_guard_status="PASS",
            leakage_guard_status="PASS" if leak["passed"] else "FAIL",
            real_llm_calls=int(self.backend.real_calls),
        )

    def parsed(self, board: ReviewBoardOutput, role: str) -> dict:
        for e in board.envelopes:
            if e.role == role:
                return e.parsed_json
        raise KeyError(f"ROLE_NOT_RUN: {role}")
