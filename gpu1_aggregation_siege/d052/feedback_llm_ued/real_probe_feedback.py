"""P0-3: real Simulator-Probe consumption + provenance-bound feedback.

Production-path hard rules (master directive):

* the production path may NOT derive probe metrics from a candidate hash —
  candidate-hash-derived numbers are the deterministic SYMBOLIC runner's
  business; a real probe's metrics come ONLY from real Student/Reference
  episodes executed by the shared CandidateProbeRunner;
* every SimulatorFeedbackRecord produced on the production path binds:
  source window, source plan, hypothesis ids, candidate-environment hash,
  changed axes, held constants, predicted metrics, observed metrics,
  residual, CI-sample count, Student/Reference checkpoint hashes, the seed
  bank, transitions and full provenance (``RealProbeProvenance``);
* unknown, wrong-window, wrong-plan, wrong-family or wrong-identity
  inputs ALL fail closed — never coerced, never dropped.

This module consumes (never re-implements): the ``SimulatorProbeRunner``
Protocol surface, ``ProbeMetrics`` / ``CandidateEnvironment`` contracts,
``assert_episode_budget``, the Reference output guard, and the existing
``derive_seed_bank``. It never imports the symbolic runner as a data
source; the runner ids below are a REJECTION list, not a dependency.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Protocol, Sequence, Tuple, \
    runtime_checkable

from pydantic import Field, model_validator

from d052.bagr_ued.hashing import verify_content_hash
from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.execution_mode import FeedbackLaunchGate
from d052.feedback_llm_ued.feedback_contracts import (
    CandidateEnvironment,
    ProbeMetrics,
)
from d052.feedback_llm_ued.formal_isolation import ReferenceOutputGuard
from d052.feedback_llm_ued.real_simulator_probe import (
    RealProbeBlocked,
    derive_seed_bank,
)
from d052.feedback_llm_ued.shared_runtime_binding import (
    ReferenceBindingIdentity,
)
from d052.feedback_llm_ued.simulator_feedback_store import (
    SimulatorFeedbackRecord,
)
from d052.feedback_llm_ued.simulator_probe import assert_episode_budget
from d052.feedback_llm_ued.student_binding import StudentBindingIdentity
from d052.schemas.common import CanonicalModel, is_sha256_hex

# ---------------------------------------------------------------------------
# production runner honesty
# ---------------------------------------------------------------------------
#: runner ids that may NEVER serve a production-path probe. This is a
#: rejection list: the symbolic runner derives metrics from the candidate
#: hash and the preflight seam is construction-blocked locally — citing
#: either as a real-probe source is a hard error.
FORBIDDEN_PRODUCTION_RUNNER_IDS = frozenset({
    "mock.feedback_llm_ued.symbolic_probe.v1",
    "feedback_llm_ued.craftax_preflight_probe.v1",
})

_GUARD = ReferenceOutputGuard()


def assert_real_runner(runner: object) -> None:
    """Fail closed unless ``runner`` is a genuine real-simulator runner."""
    if getattr(runner, "real_simulator", None) is not True:
        raise RealProbeBlocked(
            "PROBE_RUNNER_NOT_REAL: production probes require a runner "
            "with real_simulator=True (candidate-hash-derived symbolic "
            "metrics are forbidden on the production path)")
    runner_id = getattr(runner, "runner_id", "")
    if runner_id in FORBIDDEN_PRODUCTION_RUNNER_IDS:
        raise RealProbeBlocked(
            f"PRODUCTION_PATH_FORBIDDEN_RUNNER: runner_id={runner_id!r} "
            "is on the symbolic/blocked-seam rejection list; a real probe "
            "must come from the shared CandidateProbeRunner")


# ---------------------------------------------------------------------------
# the shared CandidateProbeRunner surface (consume-only contract)
# ---------------------------------------------------------------------------
@runtime_checkable
class SharedProbeResult(Protocol):
    """One stage's real episode results, signed by the shared runner."""

    metrics: Dict[str, object]
    simulator_transitions: int
    episode_count: int
    student_checkpoint_hash: str
    reference_checkpoint_hash: str


@runtime_checkable
class SharedCandidateProbeRunner(Protocol):
    """The shared runner this direction consumes (one owner, read-only).

    Responsible ONLY for real reset/rollout/transition accounting and
    evidence signing — never ranking or selection (that belongs to the
    Soft Copeland layer).
    """

    runner_id: str
    real_simulator: bool

    def probe_candidate(self, *, candidate_hash: str,
                        environment_family: str,
                        axis_values: Dict[str, str],
                        held_constant_axes: Dict[str, str],
                        stage: str, student_episodes: int,
                        reference_episodes: int,
                        seed_bank: Tuple[int, ...]) -> SharedProbeResult: ...


#: the ProbeMetrics fields a real probe result MUST provide (missing keys
#: fail closed — REAL_PROBE_METRICS_INCOMPLETE; nothing is defaulted)
REQUIRED_METRIC_KEYS = (
    "student_success_rate", "student_behavior_activation",
    "student_front_progress", "reference_success_rate",
    "reference_mean_progress", "reference_behavior_activation",
    "global_retention", "regret", "learnability",
    "simulator_transitions", "too_hard", "too_easy",
)


def metrics_from_shared_result(result: SharedProbeResult, *, stage: str
                               ) -> ProbeMetrics:
    """Convert a shared-runner result into the loop's ProbeMetrics, fail
    closed on any missing/illegal value (NO silent defaults)."""
    metrics = getattr(result, "metrics", None)
    if not isinstance(metrics, dict):
        raise RealProbeBlocked(
            "REAL_PROBE_METRICS_MISSING: the shared probe result carries "
            "no metrics mapping")
    missing = [k for k in REQUIRED_METRIC_KEYS if k not in metrics]
    if missing:
        raise RealProbeBlocked(
            f"REAL_PROBE_METRICS_INCOMPLETE: missing keys {missing} — a "
            "real probe must report every metric explicitly")
    try:
        probe_metrics = ProbeMetrics(
            stage=stage,
            student_success_rate=float(metrics["student_success_rate"]),
            student_behavior_activation=float(
                metrics["student_behavior_activation"]),
            student_front_progress=float(metrics["student_front_progress"]),
            reference_success_rate=float(metrics["reference_success_rate"]),
            reference_mean_progress=float(
                metrics["reference_mean_progress"]),
            reference_behavior_activation=float(
                metrics["reference_behavior_activation"]),
            global_retention=float(metrics["global_retention"]),
            regret=float(metrics["regret"]),
            learnability=float(metrics["learnability"]),
            simulator_transitions=int(metrics["simulator_transitions"]),
            too_hard=bool(metrics["too_hard"]),
            too_easy=bool(metrics["too_easy"]),
            probe_source=C.SOURCE_CANDIDATE_PROBE)
    except RealProbeBlocked:
        raise
    except Exception as exc:
        raise RealProbeBlocked(
            f"REAL_PROBE_METRICS_ILLEGAL: {type(exc).__name__}: {exc}") \
            from exc
    _GUARD.assert_clean(probe_metrics.model_dump(),
                        label=f"real_probe:{stage}")
    return probe_metrics


class RealProbeFeedbackRunner:
    """Presents the shared CandidateProbeRunner on the loop's
    ``SimulatorProbeRunner`` surface (probe_calls / total_transitions /
    probe()). Construction fails closed without gate authorization AND a
    genuine shared runner.
    """

    real_simulator = True
    status = "READY"

    def __init__(self, *, shared_runner: SharedCandidateProbeRunner,
                 gate: FeedbackLaunchGate,
                 student_identity_hash: str,
                 seed_bank_source=derive_seed_bank) -> None:
        gate.assert_real_probe_allowed()
        assert_real_runner(shared_runner)
        if not is_sha256_hex(student_identity_hash):
            raise RealProbeBlocked(
                "PROBE_STUDENT_IDENTITY_HASH_NOT_SHA256: "
                f"{student_identity_hash!r}")
        self._runner = shared_runner
        self._student_identity_hash = student_identity_hash
        self._seed_bank_source = seed_bank_source
        self.runner_id = ("feedback_llm_ued.real_probe_adapter.v1::"
                          f"{shared_runner.runner_id}")
        self.probe_calls = 0
        self.total_transitions = 0

    def probe(self, candidate: CandidateEnvironment, *, stage: str,
              student_episodes: int,
              reference_episodes: int) -> ProbeMetrics:
        assert_episode_budget(stage, student_episodes, reference_episodes)
        if student_episodes <= 0 or reference_episodes <= 0:
            raise RealProbeBlocked(
                "ILLEGAL_PROBE_EPISODES: must be positive")
        seed_bank = self._seed_bank_source(
            candidate.candidate_hash, stage=stage,
            n=max(student_episodes, reference_episodes))
        result = self._runner.probe_candidate(
            candidate_hash=candidate.candidate_hash,
            environment_family=candidate.environment_family,
            axis_values=dict(candidate.axis_values),
            held_constant_axes=dict(candidate.held_constant_axes),
            stage=stage, student_episodes=student_episodes,
            reference_episodes=reference_episodes,
            seed_bank=tuple(seed_bank))
        transitions = int(getattr(result, "simulator_transitions", 0))
        if transitions <= 0:
            raise RealProbeBlocked(
                "REAL_PROBE_NO_TRANSITIONS: a real probe must account "
                "positive simulator transitions")
        metrics = metrics_from_shared_result(result, stage=stage)
        if metrics.simulator_transitions != transitions:
            raise RealProbeBlocked(
                "REAL_PROBE_TRANSITION_MISMATCH: result.simulator_"
                f"transitions={transitions} but metrics report "
                f"{metrics.simulator_transitions}")
        self.probe_calls += 1
        self.total_transitions += transitions
        return metrics


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
class RealProbeProvenance(CanonicalModel):
    """The complete provenance of one production-path probe feedback."""

    source_window: int = Field(ge=0)
    source_plan_id: str = Field(min_length=1)
    distinguishes_hypothesis_ids: List[str] = Field(default_factory=list)
    candidate_id: str = Field(min_length=1)
    candidate_hash: str = Field(min_length=1)
    environment_family: str = Field(min_length=1)
    changed_axes: Dict[str, str] = Field(default_factory=dict)
    held_constant_axes: Dict[str, str] = Field(default_factory=dict)
    predicted_metrics: Dict[str, float] = Field(default_factory=dict)
    observed_residual: Dict[str, float] = Field(default_factory=dict)
    #: CI-sample count = real episodes executed (Student + Reference)
    ci_sample_count: int = Field(default=0, ge=0)
    student_identity_hash: str = Field(min_length=1)
    reference_identity_hash: str = Field(min_length=1)
    student_checkpoint_hash: str = ""
    reference_checkpoint_hash: str = ""
    seed_bank: List[int] = Field(default_factory=list)
    simulator_transitions: int = Field(default=0, ge=0)
    runner_id: str = Field(min_length=1)
    provenance_hash: str = ""

    @model_validator(mode="after")
    def _validate_and_hash(self) -> "RealProbeProvenance":
        if self.environment_family not in C.ENVIRONMENT_FAMILIES:
            raise ValueError(
                f"UNKNOWN_ENVIRONMENT_FAMILY: {self.environment_family!r}")
        if not is_sha256_hex(self.candidate_hash):
            raise ValueError(
                f"CANDIDATE_HASH_NOT_SHA256: {self.candidate_hash!r}")
        for field_name in ("student_identity_hash",
                           "reference_identity_hash",
                           "student_checkpoint_hash",
                           "reference_checkpoint_hash"):
            value = getattr(self, field_name)
            if value and not is_sha256_hex(value):
                raise ValueError(
                    f"PROVENANCE_HASH_NOT_SHA256: {field_name}={value!r}")
        if self.runner_id in FORBIDDEN_PRODUCTION_RUNNER_IDS:
            raise ValueError(
                f"PRODUCTION_PATH_FORBIDDEN_RUNNER: {self.runner_id!r}")
        if not self.distinguishes_hypothesis_ids:
            raise ValueError(
                "PROVENANCE_WITHOUT_HYPOTHESES: a real probe must bind at "
                "least one hypothesis it distinguishes")
        if self.ci_sample_count <= 0:
            raise ValueError(
                "PROVENANCE_WITHOUT_EPISODES: ci_sample_count must count "
                "the real episodes executed")
        for key, value in self.predicted_metrics.items():
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"NON_NUMERIC_PREDICTED_METRIC: {key}={value!r}")
        computed = verify_content_hash(self.model_dump(),
                                       hash_field="provenance_hash",
                                       carried=self.provenance_hash,
                                       kind="RealProbeProvenance")
        object.__setattr__(self, "provenance_hash", computed)
        return self


def compute_residual(predicted: Dict[str, float],
                     observed: ProbeMetrics) -> Dict[str, float]:
    """predicted - observed for every predicted key present in the metrics
    (keys the probe does not report are omitted, never defaulted)."""
    dump = observed.model_dump()
    residual: Dict[str, float] = {}
    for key, value in predicted.items():
        if key in dump and isinstance(dump[key], (int, float)):
            residual[key] = round(float(value) - float(dump[key]), 6)
    return residual


# ---------------------------------------------------------------------------
# feedback record assembly (fail-closed binding)
# ---------------------------------------------------------------------------
def build_real_feedback_record(*, feedback_id: str,
                               candidate: CandidateEnvironment,
                               source_window: int,
                               source_plan_id: str,
                               known_hypothesis_ids: Sequence[str],
                               predicted_signature: Dict[str, float],
                               stage_metrics: ProbeMetrics,
                               reference_stats: Dict[str, float],
                               student_binding: StudentBindingIdentity,
                               reference_binding: ReferenceBindingIdentity,
                               runner_id: str,
                               seed_bank: Sequence[int],
                               ci_sample_count: int,
                               student_checkpoint_hash: str = "",
                               reference_checkpoint_hash: str = "",
                               expected_observed_match: str,
                               match_detail: Optional[dict] = None
                               ) -> Tuple[SimulatorFeedbackRecord,
                                          RealProbeProvenance]:
    """Assemble one production-path feedback record + its provenance.

    Every binding violation fails closed:
      * wrong window          -> FEEDBACK_WINDOW_MISMATCH
      * wrong plan            -> FEEDBACK_PLAN_MISMATCH
      * wrong family          -> FEEDBACK_FAMILY_MISMATCH
      * wrong/missing student -> STUDENT_IDENTITY_MISMATCH
      * missing reference     -> REFERENCE_IDENTITY_MISSING
      * unknown hypothesis    -> UNKNOWN_HYPOTHESIS_ID
    """
    if source_window < 0:
        raise RealProbeBlocked(
            f"FEEDBACK_WINDOW_MISMATCH: source_window={source_window}")
    if not candidate.distinguishes_hypothesis_ids:
        raise RealProbeBlocked(
            f"PROVENANCE_WITHOUT_HYPOTHESES: candidate "
            f"{candidate.candidate_id!r} distinguishes no hypothesis")
    known = set(known_hypothesis_ids)
    unknown = [h for h in candidate.distinguishes_hypothesis_ids
               if h not in known]
    if unknown:
        raise RealProbeBlocked(
            f"UNKNOWN_HYPOTHESIS_ID: {sorted(unknown)} are not in the "
            "window's known ledger hypotheses")
    if student_binding.candidate_id != C.STRONG_STUDENT_CANDIDATE_ID:
        raise RealProbeBlocked(
            f"STUDENT_IDENTITY_MISMATCH: binding candidate_id="
            f"{student_binding.candidate_id!r} but direction two is bound "
            f"to {C.STRONG_STUDENT_CANDIDATE_ID!r}")
    if reference_binding is None or not reference_binding.identity_hash:
        raise RealProbeBlocked(
            "REFERENCE_IDENTITY_MISSING: a production feedback record "
            "requires the shared Reference identity hash")
    if stage_metrics.probe_source != C.SOURCE_CANDIDATE_PROBE:
        raise RealProbeBlocked(
            f"PROBE_SOURCE_NOT_ALLOWED: {stage_metrics.probe_source!r}")

    residual = compute_residual(predicted_signature, stage_metrics)
    provenance = RealProbeProvenance(
        source_window=source_window,
        source_plan_id=source_plan_id,
        distinguishes_hypothesis_ids=list(
            candidate.distinguishes_hypothesis_ids),
        candidate_id=candidate.candidate_id,
        candidate_hash=candidate.candidate_hash,
        environment_family=candidate.environment_family,
        changed_axes=dict(candidate.axis_values),
        held_constant_axes=dict(candidate.held_constant_axes),
        predicted_metrics=dict(predicted_signature),
        observed_residual=residual,
        ci_sample_count=ci_sample_count,
        student_identity_hash=student_binding.identity_hash,
        reference_identity_hash=reference_binding.identity_hash,
        student_checkpoint_hash=student_checkpoint_hash,
        reference_checkpoint_hash=reference_checkpoint_hash,
        seed_bank=[int(s) for s in seed_bank],
        simulator_transitions=stage_metrics.simulator_transitions,
        runner_id=runner_id)

    record = SimulatorFeedbackRecord(
        feedback_id=feedback_id,
        candidate_id=candidate.candidate_id,
        candidate_hash=candidate.candidate_hash,
        source_plan_id=source_plan_id,
        window=source_window,
        environment_family=candidate.environment_family,
        mutation_axes=list(candidate.mutation_axes),
        axis_values=dict(candidate.axis_values),
        held_constant_axes=dict(candidate.held_constant_axes),
        distinguishes_hypothesis_ids=list(
            candidate.distinguishes_hypothesis_ids),
        stage1_metrics=(stage_metrics
                        if stage_metrics.stage == "fast" else None),
        stage2_metrics=(stage_metrics
                        if stage_metrics.stage == "full" else None),
        reference_stats=dict(reference_stats),
        expected_signature=dict(predicted_signature),
        expected_observed_match=expected_observed_match,
        match_detail=dict(match_detail or {}),
        provenance=dict(
            real_probe_provenance_hash=provenance.provenance_hash,
            runner_id=runner_id,
            ci_sample_count=ci_sample_count,
            seed_bank_size=len(provenance.seed_bank),
            production_path=True,
            symbolic_metrics_forbidden=True),
        student_identity_hash=student_binding.identity_hash,
        #: finally populated on the production path (the field existed but
        #: could never be honestly filled before the shared Reference
        #: binding landed)
        reference_identity_hash=reference_binding.identity_hash,
        student_parameter_tree_hash=student_binding.parameter_tree_hash,
        student_checkpoint_step=student_binding.checkpoint_global_step,
        student_roles=(C.STUDENT_ROLE_SEARCH,),
        memory_compatibility_status=C.MEMORY_COMPATIBILITY_NOT_APPLICABLE)
    return record, provenance


__all__ = [
    "FORBIDDEN_PRODUCTION_RUNNER_IDS", "assert_real_runner",
    "SharedProbeResult", "SharedCandidateProbeRunner",
    "REQUIRED_METRIC_KEYS", "metrics_from_shared_result",
    "RealProbeFeedbackRunner", "RealProbeProvenance", "compute_residual",
    "build_real_feedback_record",
]
