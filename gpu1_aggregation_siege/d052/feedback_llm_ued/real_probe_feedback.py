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

P0-8 (CC3 follow-up audit): the production path consumes ONLY the
immutable, registry-signed :class:`CandidateProbeResult` — duck-typed
stand-ins are refused (``REAL_PROBE_RESULT_NOT_SIGNED``), a missing
signature is refused (``REAL_PROBE_RESULT_UNSIGNED``), and the carried
``result_hash`` is recomputed and compared (tamper fails closed). The
result's episode accounting must balance per role
(requested == completed + failed/rejected), the requested counts must
match what this seam asked for, the CI-sample count is the number of
actually COMPLETED (valid) episodes — never the requested count — and
both checkpoint hashes are mandatory valid sha256. Feedback provenance
additionally binds ``changed_axes`` only inside the candidate's declared
``mutation_axes`` (P0-8) and fails closed on duplicate/stale/missing
evidence (P0-9).

This module consumes (never re-implements): the ``SimulatorProbeRunner``
Protocol surface, ``ProbeMetrics`` / ``CandidateEnvironment`` contracts,
``assert_episode_budget``, the Reference output guard, and the existing
``derive_seed_bank``. It never imports the symbolic runner as a data
source; the runner ids below are a REJECTION list, not a dependency.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, \
    Tuple, runtime_checkable

from pydantic import ConfigDict, Field, model_validator

from d052.bagr_ued.hashing import canonical_sha256, verify_content_hash
from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.executable_env_artifact import (
    ExecutableEnvironmentArtifact,
    assert_candidate_artifact_binding,
    bind_candidate_to_artifact,
)
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
    """P0-8: one stage's real episode results.

    The shared runner MUST return the immutable registry-signed
    :class:`CandidateProbeResult`; this protocol only documents the
    minimum surface of that contract — duck-typed stand-ins are refused
    by :func:`consume_signed_probe_result` (REAL_PROBE_RESULT_NOT_SIGNED).
    """

    stage: str
    metrics: Dict[str, object]
    simulator_transitions: int
    student_episodes_requested: int
    student_episodes_completed: int
    student_episodes_failed_or_rejected: int
    reference_episodes_requested: int
    reference_episodes_completed: int
    reference_episodes_failed_or_rejected: int
    student_checkpoint_hash: str
    reference_checkpoint_hash: str
    issuer_runner_id: str
    result_hash: str


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


# ---------------------------------------------------------------------------
# P0-8: the immutable, registry-signed probe result (the ONLY shape the
# production path consumes)
# ---------------------------------------------------------------------------
class CandidateProbeResult(CanonicalModel):
    """One stage's real episode results — immutable and registry-signed.

    * frozen: any post-construction mutation is refused by pydantic;
    * signed: ``result_hash`` MUST be carried (the registry/owner computes
      it over the content) and is recomputed-and-compared at construction
      — unsigned (``REAL_PROBE_RESULT_UNSIGNED``) and tampered
      (``CONTENT_HASH_MISMATCH``) results fail closed;
    * balanced: per role, requested == completed + failed/rejected;
    * honest checkpoints: both checkpoint hashes are mandatory valid
      sha256 (the runner signs which Student/Reference weights actually
      rolled out).
    """

    model_config = ConfigDict(frozen=True)

    stage: str = Field(min_length=1)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    simulator_transitions: int = Field(gt=0)
    #: P0-8 episode accounting (per role):
    #: requested == completed + failed_or_rejected, always
    student_episodes_requested: int = Field(ge=0)
    student_episodes_completed: int = Field(ge=0)
    student_episodes_failed_or_rejected: int = Field(ge=0)
    reference_episodes_requested: int = Field(ge=0)
    reference_episodes_completed: int = Field(ge=0)
    reference_episodes_failed_or_rejected: int = Field(ge=0)
    #: P0-8: mandatory valid sha256 checkpoint hashes
    student_checkpoint_hash: str = Field(min_length=1)
    reference_checkpoint_hash: str = Field(min_length=1)
    #: the shared runner that executed and signed this result
    issuer_runner_id: str = Field(min_length=1)
    result_hash: str = ""

    @model_validator(mode="after")
    def _validate_and_hash(self) -> "CandidateProbeResult":
        for role, requested, completed, failed in (
                ("student", self.student_episodes_requested,
                 self.student_episodes_completed,
                 self.student_episodes_failed_or_rejected),
                ("reference", self.reference_episodes_requested,
                 self.reference_episodes_completed,
                 self.reference_episodes_failed_or_rejected)):
            if requested != completed + failed:
                raise ValueError(
                    f"REAL_PROBE_EPISODE_ACCOUNTING_MISMATCH: {role} "
                    f"requested={requested} != completed={completed} + "
                    f"failed_or_rejected={failed}")
            if requested <= 0:
                raise ValueError(
                    f"REAL_PROBE_EPISODES_NOT_REQUESTED: {role} episodes "
                    f"requested={requested} — a real probe must request "
                    "positive episodes for both roles")
        for field_name in ("student_checkpoint_hash",
                           "reference_checkpoint_hash"):
            value = getattr(self, field_name)
            if not is_sha256_hex(value):
                raise ValueError(
                    f"REAL_PROBE_CHECKPOINT_HASH_NOT_SHA256: {field_name}="
                    f"{value!r} — a real probe result must carry the valid "
                    "sha256 checkpoint hashes the runner signed")
        if self.issuer_runner_id in FORBIDDEN_PRODUCTION_RUNNER_IDS:
            raise ValueError(
                f"PRODUCTION_PATH_FORBIDDEN_RUNNER: issuer_runner_id="
                f"{self.issuer_runner_id!r} is on the symbolic/blocked-seam "
                "rejection list")
        if not self.result_hash:
            raise ValueError(
                "REAL_PROBE_RESULT_UNSIGNED: a CandidateProbeResult must "
                "carry the registry-issued result_hash computed over its "
                "content — unsigned results are never consumed")
        computed = verify_content_hash(self.model_dump(),
                                       hash_field="result_hash",
                                       carried=self.result_hash,
                                       kind="CandidateProbeResult")
        object.__setattr__(self, "result_hash", computed)
        return self

    @property
    def valid_episode_count(self) -> int:
        """P0-8: the CI-sample count source — actually COMPLETED (valid)
        episodes only; failed/rejected episodes never count."""
        return (self.student_episodes_completed
                + self.reference_episodes_completed)


def sign_probe_result(payload: Mapping[str, Any]) -> CandidateProbeResult:
    """Owner/registry-side helper: build and SIGN one immutable result.

    The signature is the canonical sha256 over the complete content
    (exactly what ``model_dump()`` will reproduce, including the
    canonical_v2 protocol field); consumption then recomputes and
    compares. Direction two calls this ONLY inside tests / fixtures — in
    production the shared runtime owner signs.
    """
    body = dict(payload)
    body.pop("result_hash", None)
    body.setdefault(
        "protocol_version",
        CandidateProbeResult.model_fields["protocol_version"].default)
    signature = canonical_sha256(body)
    return CandidateProbeResult(**body, result_hash=signature)


def consume_signed_probe_result(raw: object, *, expected_issuer: str,
                                stage: str, requested_student: int,
                                requested_reference: int
                                ) -> CandidateProbeResult:
    """P0-8 consume-only gate: accept ONLY the immutable registry-signed
    result, bound to THIS probe call. Fail-closed ladder:

      * not a CandidateProbeResult / mapping  -> REAL_PROBE_RESULT_NOT_SIGNED
      * mapping that fails the schema/signature -> REAL_PROBE_RESULT_ILLEGAL
      * wrong stage                           -> REAL_PROBE_STAGE_MISMATCH
      * signed by a different issuer          -> REAL_PROBE_RESULT_ISSUER_MISMATCH
      * requested != what the seam asked for  -> REAL_PROBE_EPISODE_REQUEST_MISMATCH
      * zero completed (valid) episodes       -> REAL_PROBE_NO_VALID_EPISODES
    """
    if isinstance(raw, CandidateProbeResult):
        result = raw
    elif isinstance(raw, Mapping):
        try:
            result = CandidateProbeResult(**dict(raw))
        except Exception as exc:
            raise RealProbeBlocked(
                f"REAL_PROBE_RESULT_ILLEGAL: mapping payload failed the "
                f"signed-result contract: {type(exc).__name__}: {exc}"
            ) from exc
    else:
        raise RealProbeBlocked(
            "REAL_PROBE_RESULT_NOT_SIGNED: the production path consumes "
            "ONLY the immutable registry-signed CandidateProbeResult, got "
            f"{type(raw).__name__} — duck-typed stand-ins are refused")
    if result.stage != stage:
        raise RealProbeBlocked(
            f"REAL_PROBE_STAGE_MISMATCH: result stage={result.stage!r} "
            f"but this probe call requested stage={stage!r}")
    if result.issuer_runner_id != expected_issuer:
        raise RealProbeBlocked(
            "REAL_PROBE_RESULT_ISSUER_MISMATCH: result is signed by "
            f"{result.issuer_runner_id!r} but the bound shared runner is "
            f"{expected_issuer!r} — a result signed by anyone else is "
            "refused")
    if (result.student_episodes_requested != requested_student
            or result.reference_episodes_requested != requested_reference):
        raise RealProbeBlocked(
            "REAL_PROBE_EPISODE_REQUEST_MISMATCH: result reports "
            f"requested student={result.student_episodes_requested} / "
            f"reference={result.reference_episodes_requested} but this "
            f"probe call requested student={requested_student} / "
            f"reference={requested_reference}")
    if result.valid_episode_count <= 0:
        raise RealProbeBlocked(
            "REAL_PROBE_NO_VALID_EPISODES: every requested episode failed "
            "or was rejected — zero valid episodes can support no "
            "feedback (NO_SILENT_FALLBACK)")
    return result


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
        #: per-candidate evidence trail (candidate_id -> list of per-stage
        #: evidence dicts): what the ProbeMetrics interface alone cannot
        #: carry — the checkpoint hashes the shared runner signed, the
        #: seed bank, the episode counts (= CI-sample count), and the
        #: executable artifact the probe executed. Feedback record
        #: builders bind provenance from this trail; it is derived
        #: observation data, never a source of metrics.
        self.probe_evidence: Dict[str, List[dict]] = {}
        #: P0-2 (CC3 follow-up audit): the executable environment
        #: artifacts this probe is allowed to execute — artifact_id ->
        #: immutable artifact. Probes without a bound artifact fail
        #: closed; there is no symbolic stand-in.
        self._executable_artifacts: Dict[str,
                                         ExecutableEnvironmentArtifact] = {}

    # ---------------------------------------------------------- artifacts
    def bind_executable_artifacts(self, artifacts
                                  ) -> None:
        """Bind derived executable artifacts (one per environment family).

        A rebind of the SAME artifact_id must be byte-identical (equal
        artifact_hash); a conflicting rebind fails closed. Artifacts whose
        runtime adapter is on the symbolic/blocked-seam rejection list are
        refused.
        """
        for art in artifacts:
            if art.runtime_adapter_id in FORBIDDEN_PRODUCTION_RUNNER_IDS:
                raise RealProbeBlocked(
                    "PRODUCTION_PATH_FORBIDDEN_RUNNER: artifact runtime "
                    f"adapter {art.runtime_adapter_id!r} is on the "
                    "rejection list")
            existing = self._executable_artifacts.get(art.artifact_id)
            if existing is not None \
                    and existing.artifact_hash != art.artifact_hash:
                raise RealProbeBlocked(
                    "EXECUTABLE_ARTIFACT_REBIND_MISMATCH: artifact_id="
                    f"{art.artifact_id!r} is already bound with hash "
                    f"{existing.artifact_hash!r}; refusing conflicting "
                    f"hash {art.artifact_hash!r}")
            self._executable_artifacts[art.artifact_id] = art

    def lookup_executable_artifact(self, *, environment_family: str
                                   ) -> Optional[ExecutableEnvironmentArtifact]:
        """The (single) artifact bound for one environment family, if any."""
        bound = [a for a in self._executable_artifacts.values()
                 if a.environment_family == environment_family]
        if len(bound) > 1:
            raise RealProbeBlocked(
                "EXECUTABLE_ARTIFACT_FAMILY_CONFLICT: family "
                f"{environment_family!r} has {len(bound)} bound artifacts")
        return bound[0] if bound else None

    def bind_candidates_to_executable_artifacts(self, candidates
                                                ) -> List[CandidateEnvironment]:
        """P0-2 controller seam: every production candidate enters the
        probe funnel as a BOUND copy of its family's executable artifact
        (new candidate_hash; unbound families fail closed)."""
        bound: List[CandidateEnvironment] = []
        for cand in candidates:
            artifact = self.lookup_executable_artifact(
                environment_family=cand.environment_family)
            if artifact is None:
                raise RealProbeBlocked(
                    "EXECUTABLE_ARTIFACT_MISSING: family "
                    f"{cand.environment_family!r} has no bound executable "
                    "environment artifact; production candidates may not "
                    "be probed without one")
            bound.append(bind_candidate_to_artifact(cand, artifact))
        return bound

    def probe(self, candidate: CandidateEnvironment, *, stage: str,
              student_episodes: int,
              reference_episodes: int) -> ProbeMetrics:
        assert_episode_budget(stage, student_episodes, reference_episodes)
        if student_episodes <= 0 or reference_episodes <= 0:
            raise RealProbeBlocked(
                "ILLEGAL_PROBE_EPISODES: must be positive")
        #: P0-2 (CC3 follow-up audit): a production probe executes ONLY a
        #: bound executable environment artifact. Missing artifact for the
        #: family, an unbound candidate, or any id / hash / family
        #: mismatch fails closed BEFORE any episode runs.
        artifact = self.lookup_executable_artifact(
            environment_family=candidate.environment_family)
        if artifact is None:
            raise RealProbeBlocked(
                "EXECUTABLE_ARTIFACT_MISSING: no executable environment "
                f"artifact is bound for family "
                f"{candidate.environment_family!r}; a production probe "
                "refuses to run without one")
        assert_candidate_artifact_binding(candidate, artifact)
        seed_bank = self._seed_bank_source(
            candidate.candidate_hash, stage=stage,
            n=max(student_episodes, reference_episodes))
        raw = self._runner.probe_candidate(
            candidate_hash=candidate.candidate_hash,
            environment_family=candidate.environment_family,
            axis_values=dict(candidate.axis_values),
            held_constant_axes=dict(candidate.held_constant_axes),
            stage=stage, student_episodes=student_episodes,
            reference_episodes=reference_episodes,
            seed_bank=tuple(seed_bank))
        #: P0-8: consume ONLY the immutable registry-signed result, bound
        #: to this exact probe call (stage / issuer / requested counts);
        #: anything else fails closed before any metric is read
        result = consume_signed_probe_result(
            raw, expected_issuer=self._runner.runner_id, stage=stage,
            requested_student=student_episodes,
            requested_reference=reference_episodes)
        transitions = int(result.simulator_transitions)
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
        self.probe_evidence.setdefault(candidate.candidate_id, []).append(
            dict(stage=stage,
                 seed_bank=[int(s) for s in seed_bank],
                 #: P0-8: the balanced episode accounting the signed
                 #: result reported (requested == completed +
                 #: failed/rejected, per role)
                 student_episodes_requested=(
                     result.student_episodes_requested),
                 student_episodes_completed=(
                     result.student_episodes_completed),
                 student_episodes_failed_or_rejected=(
                     result.student_episodes_failed_or_rejected),
                 reference_episodes_requested=(
                     result.reference_episodes_requested),
                 reference_episodes_completed=(
                     result.reference_episodes_completed),
                 reference_episodes_failed_or_rejected=(
                     result.reference_episodes_failed_or_rejected),
                 #: P0-8: the CI-sample count counts ONLY actually
                 #: completed (valid) episodes — never requested ones
                 ci_sample_count=result.valid_episode_count,
                 simulator_transitions=transitions,
                 student_checkpoint_hash=result.student_checkpoint_hash,
                 reference_checkpoint_hash=(
                     result.reference_checkpoint_hash),
                 #: P0-8: the registry signature of the consumed result
                 result_hash=result.result_hash,
                 #: P0-2: the exact executable artifact these episodes ran
                 #: against — feedback provenance binds the SAME hash
                 executable_artifact_id=artifact.artifact_id,
                 executable_artifact_hash=artifact.artifact_hash))
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
    #: P0-2 (CC3 follow-up audit): the content hash of the executable
    #: environment artifact the probe episodes actually ran against — the
    #: SAME hash the candidate carried into the probe. A production probe
    #: provenance without it is illegal (min_length=1, sha256-checked).
    executable_artifact_hash: str = Field(min_length=1)
    changed_axes: Dict[str, str] = Field(default_factory=dict)
    held_constant_axes: Dict[str, str] = Field(default_factory=dict)
    predicted_metrics: Dict[str, float] = Field(default_factory=dict)
    observed_residual: Dict[str, float] = Field(default_factory=dict)
    #: CI-sample count = actually COMPLETED (valid) real episodes
    #: (Student + Reference) — failed/rejected episodes never count (P0-8)
    ci_sample_count: int = Field(default=0, ge=0)
    student_identity_hash: str = Field(min_length=1)
    reference_identity_hash: str = Field(min_length=1)
    #: P0-8: mandatory — the signed probe result's checkpoint hashes
    student_checkpoint_hash: str = Field(min_length=1)
    reference_checkpoint_hash: str = Field(min_length=1)
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
        if not is_sha256_hex(self.executable_artifact_hash):
            raise ValueError(
                "EXECUTABLE_ARTIFACT_HASH_NOT_SHA256: "
                f"{self.executable_artifact_hash!r}")
        #: P0-8: all four hashes are MANDATORY valid sha256 — no empty
        #: escape hatch for the checkpoint hashes
        for field_name in ("student_identity_hash",
                           "reference_identity_hash",
                           "student_checkpoint_hash",
                           "reference_checkpoint_hash"):
            value = getattr(self, field_name)
            if not is_sha256_hex(value):
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
        #: P0-9: a provenance without its seed bank or with zero
        #: transitions is stale/incomplete — refused
        if not self.seed_bank:
            raise ValueError(
                "PROVENANCE_WITHOUT_SEED_BANK: the complete seed bank the "
                "episodes ran under is part of the provenance")
        if self.simulator_transitions <= 0:
            raise ValueError(
                "PROVENANCE_WITHOUT_TRANSITIONS: a real probe provenance "
                "must account positive simulator transitions")
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
                               student_checkpoint_hash: str,
                               reference_checkpoint_hash: str,
                               expected_observed_match: str,
                               executable_artifact_hash: str,
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
      * missing/illegal artifact hash -> EXECUTABLE_ARTIFACT_HASH_MISSING /
                                        EXECUTABLE_ARTIFACT_HASH_NOT_SHA256
      * missing/non-sha256 checkpoint hashes ->
                                REAL_PROBE_CHECKPOINT_HASH_MISSING /
                                REAL_PROBE_CHECKPOINT_HASH_NOT_SHA256 (P0-8)
      * changed_axes not inside mutation_axes ->
                                CHANGED_AXES_NOT_IN_MUTATION_AXES (P0-8)
    """
    if source_window < 0:
        raise RealProbeBlocked(
            f"FEEDBACK_WINDOW_MISMATCH: source_window={source_window}")
    #: P0-8: the checkpoint hashes the shared runner signed are MANDATORY
    #: valid sha256 — a production feedback record without them cannot
    #: attest which Student/Reference weights rolled out the episodes
    for role, value in (("student", student_checkpoint_hash),
                        ("reference", reference_checkpoint_hash)):
        if not value:
            raise RealProbeBlocked(
                f"REAL_PROBE_CHECKPOINT_HASH_MISSING: {role}_checkpoint_"
                "hash is mandatory for a production feedback record")
        if not is_sha256_hex(value):
            raise RealProbeBlocked(
                f"REAL_PROBE_CHECKPOINT_HASH_NOT_SHA256: {role}_checkpoint"
                f"_hash={value!r}")
    #: P0-8: every changed axis must be a DECLARED mutation axis of the
    #: candidate — an undeclared changed axis is a silent protocol drift
    undeclared = sorted(set(candidate.axis_values)
                        - set(candidate.mutation_axes))
    if undeclared:
        raise RealProbeBlocked(
            f"CHANGED_AXES_NOT_IN_MUTATION_AXES: candidate "
            f"{candidate.candidate_id!r} changed axes {undeclared} that "
            f"are not among its declared mutation axes "
            f"{sorted(candidate.mutation_axes)}")
    #: P0-2: the feedback record must bind the SAME executable artifact
    #: hash the candidate carried into the probe — never empty, never hex-
    #: shaped garbage
    if not executable_artifact_hash:
        raise RealProbeBlocked(
            "EXECUTABLE_ARTIFACT_HASH_MISSING: a production feedback "
            "record must bind the executable artifact hash its probe "
            "executed")
    if not is_sha256_hex(executable_artifact_hash):
        raise RealProbeBlocked(
            "EXECUTABLE_ARTIFACT_HASH_NOT_SHA256: "
            f"{executable_artifact_hash!r}")
    if candidate.executable_artifact_hash \
            and candidate.executable_artifact_hash \
            != executable_artifact_hash:
        raise RealProbeBlocked(
            "EXECUTABLE_ARTIFACT_HASH_MISMATCH: the candidate entered the "
            f"probe bound to {candidate.executable_artifact_hash!r} but "
            f"the feedback record binds {executable_artifact_hash!r}")
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
        executable_artifact_hash=executable_artifact_hash,
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
            executable_artifact_hash=executable_artifact_hash,
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
        #: P0-16 (dual student): the FULL Student identity the feedback was
        #: probed under — window k+1 may only consume the matching Student
        student_candidate_id=student_binding.candidate_id,
        student_memory_mode=student_binding.memory_mode,
        student_memory_spec_hash=student_binding.memory_spec_hash,
        runtime_bundle_hash=student_binding.runtime_bundle_hash,
        memory_compatibility_status=C.MEMORY_COMPATIBILITY_NOT_APPLICABLE)
    return record, provenance


__all__ = [
    "FORBIDDEN_PRODUCTION_RUNNER_IDS", "assert_real_runner",
    "SharedProbeResult", "SharedCandidateProbeRunner",
    "CandidateProbeResult", "sign_probe_result",
    "consume_signed_probe_result",
    "REQUIRED_METRIC_KEYS", "metrics_from_shared_result",
    "RealProbeFeedbackRunner", "RealProbeProvenance", "compute_residual",
    "build_real_feedback_record",
]
