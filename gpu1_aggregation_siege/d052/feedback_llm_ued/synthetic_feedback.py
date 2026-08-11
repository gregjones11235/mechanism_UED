"""Synthetic test-trace feedback factory (SOURCE_SYNTHETIC_TEST_TRACE).

Unit-test scaffolding ONLY: builds well-formed SimulatorFeedbackRecords with
controlled metric values so tests can drive the diagnostician / comparator /
designer deterministically without running the probe funnel. Every record is
labelled SYNTHETIC_TEST_TRACE — an allowed loop source, but one a controller
report must never confuse with real candidate-probe feedback.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.feedback_contracts import (
    CandidateEnvironment,
    ProbeMetrics,
)
from d052.feedback_llm_ued.simulator_feedback_store import (
    SimulatorFeedbackRecord,
)


def synthetic_candidate(*, candidate_id: str, family: str,
                        axes: Optional[List[str]] = None) -> CandidateEnvironment:
    axis = (axes or [C.MUTATION_AXES[0]])[0]
    return CandidateEnvironment(
        candidate_id=candidate_id, environment_family=family,
        axis_values={axis: "medium"}, variant_id=f"syn-{candidate_id}",
        variant_kind="synthetic", mutation_axes=[axis],
        provenance=dict(source=C.SOURCE_SYNTHETIC_TEST_TRACE,
                        generator="feedback_llm_ued.synthetic.v1"))


def synthetic_feedback_record(*, feedback_id: str,
                              candidate: CandidateEnvironment,
                              plan_id: str, window: int,
                              student_success_rate: float,
                              expected_signature: Dict[str, float],
                              behavior_activation: float = 0.5,
                              front_progress: float = 0.4,
                              reference_success_rate: float = 0.9,
                              global_retention: float = 0.9,
                              learnability: float = 0.6,
                              distinguishes_hypothesis_ids: Optional[List[str]] = None,
                              student_identity_hash: str = "",
                              ) -> SimulatorFeedbackRecord:
    metrics = ProbeMetrics(
        stage="full",
        student_success_rate=student_success_rate,
        student_behavior_activation=behavior_activation,
        student_front_progress=front_progress,
        reference_success_rate=reference_success_rate,
        reference_mean_progress=round(min(1.0, reference_success_rate * 0.9), 6),
        reference_behavior_activation=round(min(1.0, behavior_activation + 0.2), 6),
        global_retention=global_retention,
        regret=round(max(0.0, reference_success_rate - student_success_rate), 6),
        learnability=learnability,
        simulator_transitions=C.ROLLOUT_LENGTH * (
            C.STAGE2_STUDENT_EPISODES_MIN + C.STAGE2_REFERENCE_EPISODES_MIN),
        probe_source=C.SOURCE_SYNTHETIC_TEST_TRACE)
    return SimulatorFeedbackRecord(
        feedback_id=feedback_id,
        candidate_id=candidate.candidate_id,
        candidate_hash=candidate.candidate_hash,
        source_plan_id=plan_id, window=window,
        environment_family=candidate.environment_family,
        mutation_axes=list(candidate.mutation_axes),
        axis_values=dict(candidate.axis_values),
        distinguishes_hypothesis_ids=list(distinguishes_hypothesis_ids or []),
        stage2_metrics=metrics,
        reference_stats={"episode_success_rate": reference_success_rate},
        expected_signature=dict(expected_signature),
        provenance=dict(source=C.SOURCE_SYNTHETIC_TEST_TRACE,
                        note="synthetic test trace, not a simulator probe"),
        student_identity_hash=student_identity_hash)
