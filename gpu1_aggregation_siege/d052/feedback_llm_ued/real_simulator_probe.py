"""P0-2: real Craftax probe INTERFACE (injection seam, fail-closed).

There is no JAX/Craftax interpreter in this environment and this module never
imports one. It defines the complete interface a real probe flows through:

    ExecutableCandidate  ->  RealTaskParamsAdapter  ->  ProbeExecutionContext
                          ->  CraftaxProbeExecutor (reset/step/terminal)
                          ->  RealCraftaxProbeRunner (gate-authorized)

Everything real is INJECTED: a fake-real test wires scripted StepEnvs and
policies; on the training host a closure that imports Craftax supplies the
env factory. Authorization is decided ONCE by ``FeedbackLaunchGate``
(``assert_real_probe_allowed``) — while ``REAL_SIMULATOR_PROBE_AUTHORIZED``
is false, constructing a runner raises. The capability flag
``REAL_SIMULATOR_PROBE`` stays False regardless: a scripted env is not
Craftax, and this module refuses to pretend otherwise.

Determinism doctrine: no random source anywhere. Probe seeds and seed banks
are derived from the candidate hash; runtime identity (seed, Student binding)
is carried OUTSIDE ``CandidateEnvironment`` and therefore never enters the
candidate hash.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from d052.bagr_ued.hashing import canonical_sha256, text_sha256
from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.execution_mode import FeedbackLaunchGate
from d052.feedback_llm_ued.feedback_contracts import (
    CandidateEnvironment,
    ProbeMetrics,
)
from d052.feedback_llm_ued.formal_isolation import ReferenceOutputGuard
from d052.feedback_llm_ued.simulator_probe import assert_episode_budget
from d052.feedback_llm_ued.student_binding import StudentBindingIdentity
from d052.schemas.common import is_sha256_hex

PROBE_ROLE_STUDENT = "student"
PROBE_ROLE_REFERENCE = "reference"
PROBE_ROLES = frozenset({PROBE_ROLE_STUDENT, PROBE_ROLE_REFERENCE})

RUNNER_ID = "feedback_llm_ued.real_craftax_probe.v1"


class RealProbeBlocked(RuntimeError):
    """Fail-closed refusal of the real-probe seam."""


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# ---------------------------------------------------------------------------
# StepEnv protocol: the slice of a Craftax-style env the executor consumes.
# ---------------------------------------------------------------------------
@runtime_checkable
class StepEnv(Protocol):
    def reset(self, seed: int) -> object:
        ...

    def step(self, action: int) -> Tuple[object, float, bool, dict]:
        ...


EnvFactory = Callable[["ExecutableCandidate", str], StepEnv]
PolicyFactory = Callable[["ExecutableCandidate", str], Callable[[object], int]]


# ---------------------------------------------------------------------------
# ExecutableCandidate: candidate + runtime identity OUTSIDE the candidate hash
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExecutableCandidate:
    """A candidate paired with runtime-only identity (seed + Student binding).

    The runtime fields are deliberately NOT part of ``CandidateEnvironment``:
    the candidate hash recomputes from the candidate payload alone, so seeds
    and identity can vary per probe without ever changing what the candidate
    IS. Construction re-verifies the candidate hash (tamper fails closed).
    """

    candidate: CandidateEnvironment
    probe_seed: int
    student_identity_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.probe_seed, int) or self.probe_seed < 0:
            raise ValueError(f"ILLEGAL_PROBE_SEED: {self.probe_seed!r}")
        if not is_sha256_hex(self.student_identity_hash):
            raise ValueError(
                "STUDENT_IDENTITY_HASH_NOT_SHA256: "
                f"{self.student_identity_hash!r}")
        payload = self.candidate.model_dump()
        payload.pop("candidate_hash", None)
        recomputed = canonical_sha256(payload)
        if recomputed != self.candidate.candidate_hash:
            raise ValueError(
                "CANDIDATE_HASH_MISMATCH: candidate_hash="
                f"{self.candidate.candidate_hash!r} but recomputed="
                f"{recomputed!r}")

    @property
    def candidate_hash(self) -> str:
        return self.candidate.candidate_hash


# ---------------------------------------------------------------------------
# RealTaskParamsAdapter: canonical payload + hash-recompute binding
# ---------------------------------------------------------------------------
class RealTaskParamsAdapter:
    """Mock-namespaced TaskParams payload adapter (real codec is external).

    The real Craftax TaskParams field names are UNKNOWN while the external
    dependency is blocked, so this adapter only ever emits the audited
    mock-namespaced whitelist and binds any externally provided payload
    through a hash RECOMPUTATION — a claimed hash that does not reproduce
    byte-for-byte is rejected, never coerced.
    """

    @staticmethod
    def payload_for(candidate: CandidateEnvironment) -> Tuple[dict, str]:
        payload = candidate.model_dump()
        unknown = set(payload) - C.MOCK_TASKPARAMS_FIELD_WHITELIST
        if unknown:
            raise ValueError(
                f"UNAUTHORIZED_TASK_PARAMS_FIELD: {sorted(unknown)}")
        return payload, canonical_sha256(payload)

    @staticmethod
    def bind(payload: Dict[str, object], claimed_hash: str) -> Dict[str, object]:
        """Verify an externally provided payload against its claimed hash."""
        unknown = set(payload) - C.MOCK_TASKPARAMS_FIELD_WHITELIST
        if unknown:
            raise ValueError(
                f"UNAUTHORIZED_TASK_PARAMS_FIELD: {sorted(unknown)}")
        recomputed = canonical_sha256(payload)
        if recomputed != claimed_hash:
            raise ValueError(
                "TASK_PARAMS_HASH_MISMATCH: claimed="
                f"{claimed_hash!r} recomputed={recomputed!r}")
        return payload


# ---------------------------------------------------------------------------
# ProbeExecutionContext: identity + seeds + budget + seed policy
# ---------------------------------------------------------------------------
def derive_seed_bank(candidate_hash: str, *, stage: str,
                     n: int) -> Tuple[int, ...]:
    """Deterministic per-episode seed bank derived from the candidate hash.

    No random source: seed i = first 8 hex chars of
    ``text_sha256("seed-bank:<candidate_hash>:<stage>:<i>")``. Recomputable
    anywhere, auditable, and independent of the Student/Reference identity.
    """
    if not is_sha256_hex(candidate_hash):
        raise ValueError(f"CANDIDATE_HASH_NOT_SHA256: {candidate_hash!r}")
    if n <= 0:
        raise ValueError(f"ILLEGAL_SEED_BANK_SIZE: {n}")
    return tuple(int(text_sha256(
        f"seed-bank:{candidate_hash}:{stage}:{i}")[:8], 16) for i in range(n))


@dataclass(frozen=True)
class ProbeExecutionContext:
    """Everything one real probe execution is allowed to depend on."""

    stage: str
    student_identity_hash: str
    seed_bank: Tuple[int, ...]
    student_episodes: int
    reference_episodes: int
    checkpoint_global_step: int = 0
    seed_policy: str = C.SEED_POLICY_JAX_PRNG_SEEDED
    max_steps_per_episode: int = C.ROLLOUT_LENGTH
    #: None at the runner-protocol level; the controller stamps the window.
    window: Optional[int] = None

    def __post_init__(self) -> None:
        assert_episode_budget(self.stage, self.student_episodes,
                              self.reference_episodes)
        if self.student_episodes <= 0 or self.reference_episodes <= 0:
            raise ValueError("ILLEGAL_PROBE_EPISODES: must be positive")
        if not is_sha256_hex(self.student_identity_hash):
            raise ValueError(
                "PROBE_STUDENT_IDENTITY_HASH_NOT_SHA256: "
                f"{self.student_identity_hash!r}")
        if not isinstance(self.checkpoint_global_step, int) or \
                self.checkpoint_global_step < 0:
            raise ValueError(
                f"ILLEGAL_CHECKPOINT_STEP: {self.checkpoint_global_step!r}")
        if self.seed_policy not in C.SEED_POLICIES:
            raise ValueError(f"UNKNOWN_SEED_POLICY: {self.seed_policy!r}")
        if self.max_steps_per_episode <= 0:
            raise ValueError(
                f"ILLEGAL_MAX_STEPS: {self.max_steps_per_episode!r}")
        if self.window is not None and self.window < 0:
            raise ValueError(f"ILLEGAL_PROBE_WINDOW: {self.window!r}")
        need = max(self.student_episodes, self.reference_episodes)
        if len(self.seed_bank) < need:
            raise ValueError(
                f"SEED_BANK_UNDERPROVISIONED: {len(self.seed_bank)} seeds "
                f"for {need} episodes")
        if any(not isinstance(s, int) or s < 0 for s in self.seed_bank):
            raise ValueError("ILLEGAL_SEED_BANK_ENTRY: seeds must be ints >= 0")


# ---------------------------------------------------------------------------
# Episode aggregates + executor
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EpisodeAggregate:
    """Coarse per-role episode statistics (no action-guidance carriers)."""

    role: str
    n_episodes: int
    success_rate: float
    mean_progress: float
    behavior_activation_rate: float
    mean_episode_length: float
    transitions: int


class CraftaxProbeExecutor:
    """Runs banked-seed episodes against an injected env/policy pair.

    reset/step/terminal only: every episode starts from ``env.reset(seed)``
    with a fresh env instance, steps until terminal or horizon, and only the
    whitelisted coarse scalars (success / progress / behavior_active) are
    extracted from ``info`` — any other info key (including forbidden
    carriers) is ignored, and the produced metrics pass the
    ReferenceOutputGuard before leaving the executor.
    """

    def __init__(self, env_factory: EnvFactory,
                 policy_factory: PolicyFactory) -> None:
        if not callable(env_factory) or not callable(policy_factory):
            raise ValueError("PROBE_FACTORIES_MUST_BE_CALLABLE")
        self._env_factory = env_factory
        self._policy_factory = policy_factory
        self._guard = ReferenceOutputGuard()
        self.probe_calls = 0
        self.total_transitions = 0

    def run_episodes(self, executable: ExecutableCandidate,
                     context: ProbeExecutionContext, *,
                     role: str) -> EpisodeAggregate:
        if role not in PROBE_ROLES:
            raise ValueError(f"ILLEGAL_PROBE_ROLE: {role!r}")
        n = (context.student_episodes if role == PROBE_ROLE_STUDENT
             else context.reference_episodes)
        if len(context.seed_bank) < n:          # fail-closed re-check
            raise ValueError(
                f"SEED_BANK_EXHAUSTED: {len(context.seed_bank)} < {n}")
        successes = 0
        progress_sum = 0.0
        active_steps = 0
        total_steps = 0
        for i in range(n):
            seed = context.seed_bank[i]
            env = self._env_factory(executable, role)
            policy = self._policy_factory(executable, role)
            obs = env.reset(seed)
            info: dict = {}
            steps = 0
            for _ in range(context.max_steps_per_episode):
                obs, _reward, done, info = env.step(policy(obs))
                steps += 1
                if info.get("behavior_active"):
                    active_steps += 1
                if done:
                    break
            successes += int(bool(info.get("success", False)))
            progress_sum += _clamp01(float(info.get("progress", 0.0)))
            total_steps += steps
        aggregate = EpisodeAggregate(
            role=role, n_episodes=n,
            success_rate=round(successes / n, 6),
            mean_progress=round(progress_sum / n, 6),
            behavior_activation_rate=(round(active_steps / total_steps, 6)
                                      if total_steps else 0.0),
            mean_episode_length=round(total_steps / n, 6),
            transitions=total_steps)
        self._guard.assert_clean(
            dict(success_rate=aggregate.success_rate,
                 mean_progress=aggregate.mean_progress,
                 behavior_activation_rate=aggregate.behavior_activation_rate,
                 mean_episode_length=aggregate.mean_episode_length),
            label=f"probe:{executable.candidate.candidate_id}:{role}")
        return aggregate

    def probe(self, executable: ExecutableCandidate,
              context: ProbeExecutionContext) -> ProbeMetrics:
        student = self.run_episodes(executable, context,
                                    role=PROBE_ROLE_STUDENT)
        reference = self.run_episodes(executable, context,
                                      role=PROBE_ROLE_REFERENCE)
        regret = max(0.0, reference.success_rate - student.success_rate)
        learnability = round(
            max(0.0, 1.0 - abs(student.success_rate - 0.4) / 0.45), 4)
        made_progress = (student.behavior_activation_rate > 0.35
                         or student.mean_progress > 0.25)
        transitions = student.transitions + reference.transitions
        metrics = ProbeMetrics(
            stage=context.stage,
            student_success_rate=student.success_rate,
            student_behavior_activation=student.behavior_activation_rate,
            student_front_progress=student.mean_progress,
            reference_success_rate=reference.success_rate,
            reference_mean_progress=reference.mean_progress,
            reference_behavior_activation=reference.behavior_activation_rate,
            #: honest default: no training update has executed this round, so
            #: nothing can have been forgotten; a real retention measurement
            #: requires the training seam (BLOCKED) and replaces this value.
            global_retention=1.0,
            regret=round(regret, 6),
            learnability=learnability,
            simulator_transitions=transitions,
            too_hard=(student.success_rate < C.PREFLIGHT_LEARNABLE_LOW
                      and not made_progress),
            too_easy=student.success_rate >= C.PREFLIGHT_TOO_EASY,
            probe_source=C.SOURCE_CANDIDATE_PROBE)
        self._guard.assert_clean(
            metrics.model_dump(),
            label=f"probe:{executable.candidate.candidate_id}:"
                  f"{context.stage}")
        self.probe_calls += 1
        self.total_transitions += transitions
        return metrics


# ---------------------------------------------------------------------------
# RealCraftaxProbeRunner: the gate-authorized runner (SimulatorProbeRunner)
# ---------------------------------------------------------------------------
class RealCraftaxProbeRunner:
    """Real-simulator runner conforming to the SimulatorProbeRunner protocol.

    Construction FAILS CLOSED unless the launch gate allows a real probe
    (EXECUTION_MODE_REAL + REAL_SIMULATOR_PROBE_AUTHORIZED). Even then, the
    runner is only as real as the injected env factory: this round's tests
    inject scripted fake-real envs, so ``C.REAL_SIMULATOR_PROBE`` stays False
    and every produced metric set says ``probe_source=CANDIDATE_PROBE`` —
    never a formal-evaluation source.
    """

    runner_id = RUNNER_ID
    real_simulator = True

    def __init__(self, env_factory: EnvFactory,
                 policy_factory: PolicyFactory, *,
                 gate: FeedbackLaunchGate,
                 student_binding: StudentBindingIdentity) -> None:
        gate.assert_real_probe_allowed()       # LaunchGateBlocked if refused
        self.status = "READY_GATE_ALLOWED"
        self._gate = gate
        self._binding = student_binding
        self._adapter = RealTaskParamsAdapter()
        self._executor = CraftaxProbeExecutor(env_factory, policy_factory)

    # -- SimulatorProbeRunner surface ---------------------------------------
    def probe(self, candidate: CandidateEnvironment, *, stage: str,
              student_episodes: int, reference_episodes: int) -> ProbeMetrics:
        assert_episode_budget(stage, student_episodes, reference_episodes)
        payload, payload_hash = self._adapter.payload_for(candidate)
        self._adapter.bind(payload, payload_hash)   # recompute-verified
        executable = ExecutableCandidate(
            candidate=candidate,
            probe_seed=int(candidate.candidate_hash[:8], 16),
            student_identity_hash=self._binding.identity_hash)
        context = ProbeExecutionContext(
            stage=stage,
            student_identity_hash=self._binding.identity_hash,
            seed_bank=derive_seed_bank(
                candidate.candidate_hash, stage=stage,
                n=max(student_episodes, reference_episodes)),
            student_episodes=student_episodes,
            reference_episodes=reference_episodes,
            checkpoint_global_step=self._binding.checkpoint_global_step,
            seed_policy=C.SEED_POLICY_JAX_PRNG_SEEDED)
        return self._executor.probe(executable, context)

    @property
    def probe_calls(self) -> int:
        return self._executor.probe_calls

    @property
    def total_transitions(self) -> int:
        return self._executor.total_transitions
