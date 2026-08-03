"""P0-2 real Craftax probe interface: fake-real tests only.

A scripted StepEnv stands in for Craftax — no jax import, no real rollout, no
API. The tests prove the seam's mechanics (hash binding, fail-closed gates,
banked-seed episode execution, protocol conformance) while every REAL
capability flag stays False.
"""
import pytest

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.execution_mode import (
    EXECUTION_MODE_MOCK_DRY_RUN,
    EXECUTION_MODE_REAL,
    FeedbackLaunchGate,
    LaunchGateBlocked,
)
from d052.feedback_llm_ued.real_simulator_probe import (
    PROBE_ROLE_REFERENCE,
    PROBE_ROLE_STUDENT,
    CraftaxProbeExecutor,
    ExecutableCandidate,
    ProbeExecutionContext,
    RealCraftaxProbeRunner,
    RealTaskParamsAdapter,
    derive_seed_bank,
)
from d052.feedback_llm_ued.simulator_probe import (
    SimulatorProbeRunner,
    run_staged_funnel,
)
from d052.feedback_llm_ued.student_binding import local_symbolic_binding
from d052.feedback_llm_ued.synthetic_feedback import synthetic_candidate

HORIZON = 6
GOOD_HASH = "a" * 64
ALT_HASH = "b" * 64


class FakeStepEnv:
    """Scripted stand-in for a Craftax env: the seed decides the outcome."""

    def __init__(self, easy=False, horizon=HORIZON, leak=False):
        self._easy = easy
        self._horizon = horizon
        self._leak = leak
        self._t = 0
        self._seed = 0

    def reset(self, seed):
        self._t = 0
        self._seed = seed
        return seed % 7

    def step(self, action):
        self._t += 1
        done = self._t >= self._horizon
        info = dict(success=True if self._easy else self._seed % 2 == 0,
                    progress=min(1.0, self._t / self._horizon),
                    behavior_active=(self._seed + self._t) % 3 == 0)
        if self._leak:
            # forbidden action-guidance carrier: the executor MUST ignore it
            info["trajectory"] = [1, 2, 3]
        return (self._seed + self._t) % 7, 0.0, done, info


def make_factories(leak=False):
    def env_factory(executable, role):
        return FakeStepEnv(easy=(role == PROBE_ROLE_REFERENCE), leak=leak)

    def policy_factory(executable, role):
        return lambda obs: 0

    return env_factory, policy_factory


def make_candidate(i=0):
    return synthetic_candidate(candidate_id=f"c-real-{i}",
                               family=C.ENVIRONMENT_FAMILIES[i % 7])


def make_executable(candidate, identity=GOOD_HASH, seed=1234):
    return ExecutableCandidate(candidate=candidate, probe_seed=seed,
                               student_identity_hash=identity)


def make_context(candidate, stage="fast", student_episodes=2,
                 reference_episodes=1, **over):
    n = max(student_episodes, reference_episodes)
    base = dict(stage=stage,
                student_identity_hash=GOOD_HASH,
                seed_bank=derive_seed_bank(candidate.candidate_hash,
                                           stage=stage, n=n),
                student_episodes=student_episodes,
                reference_episodes=reference_episodes)
    base.update(over)
    return ProbeExecutionContext(**base)


class TestExecutableCandidate:
    def test_runtime_identity_never_enters_candidate_hash(self):
        cand = make_candidate()
        e1 = make_executable(cand, identity=GOOD_HASH, seed=1)
        e2 = make_executable(cand, identity=ALT_HASH, seed=999)
        assert e1.candidate_hash == e2.candidate_hash == cand.candidate_hash

    def test_tampered_candidate_hash_fails_closed(self):
        cand = make_candidate(1)
        object.__setattr__(cand, "candidate_hash", ALT_HASH)   # simulate tamper
        with pytest.raises(ValueError, match="CANDIDATE_HASH_MISMATCH"):
            make_executable(cand)

    def test_illegal_seed_and_identity_rejected(self):
        cand = make_candidate(2)
        with pytest.raises(ValueError, match="ILLEGAL_PROBE_SEED"):
            make_executable(cand, seed=-1)
        with pytest.raises(ValueError,
                           match="STUDENT_IDENTITY_HASH_NOT_SHA256"):
            make_executable(cand, identity="zz")


class TestRealTaskParamsAdapter:
    def test_payload_is_whitelisted_and_bind_roundtrips(self):
        cand = make_candidate()
        payload, h = RealTaskParamsAdapter.payload_for(cand)
        assert set(payload) <= C.MOCK_TASKPARAMS_FIELD_WHITELIST
        assert len(h) == 64
        assert RealTaskParamsAdapter.bind(payload, h) == payload

    def test_tampered_payload_or_hash_fails_closed(self):
        cand = make_candidate(1)
        payload, h = RealTaskParamsAdapter.payload_for(cand)
        with pytest.raises(ValueError, match="TASK_PARAMS_HASH_MISMATCH"):
            RealTaskParamsAdapter.bind(payload, ALT_HASH)
        forged = dict(payload)
        forged["variant_id"] = "forged-variant"
        with pytest.raises(ValueError, match="TASK_PARAMS_HASH_MISMATCH"):
            RealTaskParamsAdapter.bind(forged, h)

    def test_unauthorized_field_rejected(self):
        with pytest.raises(ValueError,
                           match="UNAUTHORIZED_TASK_PARAMS_FIELD"):
            RealTaskParamsAdapter.bind({"action_sequence": []}, "x" * 64)


class TestSeedBankAndContext:
    def test_seed_bank_deterministic_and_hash_derived(self):
        cand = make_candidate()
        b1 = derive_seed_bank(cand.candidate_hash, stage="fast", n=4)
        b2 = derive_seed_bank(cand.candidate_hash, stage="fast", n=4)
        assert b1 == b2 and len(b1) == 4
        assert all(isinstance(s, int) and s >= 0 for s in b1)
        # stage participates: different stage -> different bank
        assert b1 != derive_seed_bank(cand.candidate_hash, stage="full", n=4)
        with pytest.raises(ValueError, match="CANDIDATE_HASH_NOT_SHA256"):
            derive_seed_bank("zz", stage="fast", n=1)
        with pytest.raises(ValueError, match="ILLEGAL_SEED_BANK_SIZE"):
            derive_seed_bank(cand.candidate_hash, stage="fast", n=0)

    def test_context_fail_closed_ladder(self):
        cand = make_candidate(1)
        with pytest.raises(ValueError,
                           match="FAST_PROBE_EPISODE_BUDGET_EXCEEDED"):
            make_context(cand, stage="fast", student_episodes=3,
                         reference_episodes=1)
        with pytest.raises(ValueError,
                           match="FULL_PROBE_STUDENT_EPISODES_OUT_OF_RANGE"):
            make_context(cand, stage="full", student_episodes=2,
                         reference_episodes=2)
        with pytest.raises(ValueError, match="UNKNOWN_SEED_POLICY"):
            make_context(cand, seed_policy="D20_ROLL")
        with pytest.raises(ValueError, match="SEED_BANK_UNDERPROVISIONED"):
            make_context(cand, seed_bank=(1,))
        with pytest.raises(ValueError,
                           match="PROBE_STUDENT_IDENTITY_HASH_NOT_SHA256"):
            make_context(cand, student_identity_hash="zz")
        with pytest.raises(ValueError, match="ILLEGAL_PROBE_WINDOW"):
            make_context(cand, window=-2)

    def test_valid_context_carries_seeded_jax_policy(self):
        ctx = make_context(make_candidate(2))
        assert ctx.seed_policy == C.SEED_POLICY_JAX_PRNG_SEEDED
        assert ctx.window is None            # stamped by the controller later
        assert ctx.max_steps_per_episode == C.ROLLOUT_LENGTH


class TestCraftaxProbeExecutorFakeReal:
    def test_aggregates_match_scripted_expectation(self):
        cand = make_candidate()
        executable = make_executable(cand)
        ctx = make_context(cand, stage="fast", student_episodes=2,
                           reference_episodes=1)
        env_factory, policy_factory = make_factories()
        executor = CraftaxProbeExecutor(env_factory, policy_factory)
        metrics = executor.probe(executable, ctx)

        bank = ctx.seed_bank
        expected_sr = round(sum(1 for s in bank[:2] if s % 2 == 0) / 2, 6)
        assert metrics.student_success_rate == expected_sr
        assert metrics.reference_success_rate == 1.0      # easy reference env
        # every episode runs the full horizon: 2 active steps out of 6
        assert metrics.student_behavior_activation == round(2 / 6, 6)
        assert metrics.student_front_progress == 1.0
        assert metrics.simulator_transitions == (2 + 1) * HORIZON
        assert metrics.global_retention == 1.0   # no training executed yet
        assert metrics.regret == round(max(0.0, 1.0 - expected_sr), 6)
        assert metrics.probe_source == C.SOURCE_CANDIDATE_PROBE
        assert metrics.too_hard is False          # progress was made
        assert executor.probe_calls == 1
        assert executor.total_transitions == (2 + 1) * HORIZON

    def test_two_executors_are_bit_identical(self):
        cand = make_candidate(1)
        executable = make_executable(cand)
        ctx = make_context(cand, stage="full", student_episodes=4,
                           reference_episodes=2)
        m1 = CraftaxProbeExecutor(*make_factories()).probe(executable, ctx)
        m2 = CraftaxProbeExecutor(*make_factories()).probe(executable, ctx)
        assert m1.model_dump() == m2.model_dump()

    def test_forbidden_info_carrier_is_ignored(self):
        cand = make_candidate(2)
        executable = make_executable(cand)
        ctx = make_context(cand)
        executor = CraftaxProbeExecutor(*make_factories(leak=True))
        metrics = executor.probe(executable, ctx)   # guard must pass
        dumped = metrics.model_dump()
        assert "trajectory" not in dumped
        assert not any("trajectory" in str(v) for v in dumped.values())

    def test_illegal_role_rejected(self):
        cand = make_candidate(3)
        executor = CraftaxProbeExecutor(*make_factories())
        with pytest.raises(ValueError, match="ILLEGAL_PROBE_ROLE"):
            executor.run_episodes(make_executable(cand), make_context(cand),
                                  role="teleport")


class TestRealCraftaxProbeRunnerGate:
    def test_unauthorized_construction_fails_closed_both_modes(self):
        env_factory, policy_factory = make_factories()
        binding = local_symbolic_binding()
        for mode in (EXECUTION_MODE_MOCK_DRY_RUN, EXECUTION_MODE_REAL):
            gate = FeedbackLaunchGate(mode)
            with pytest.raises(LaunchGateBlocked,
                               match="REAL_SIMULATOR_PROBE_NOT_ALLOWED"):
                RealCraftaxProbeRunner(env_factory, policy_factory, gate=gate,
                                       student_binding=binding)
        assert C.REAL_SIMULATOR_PROBE is False

    def test_fake_real_authorized_runner_probes_and_conforms(self,
                                                             monkeypatch):
        monkeypatch.setattr(C, "REAL_SIMULATOR_PROBE_AUTHORIZED", True)
        gate = FeedbackLaunchGate(EXECUTION_MODE_REAL)
        runner = RealCraftaxProbeRunner(*make_factories(), gate=gate,
                                        student_binding=local_symbolic_binding())
        assert isinstance(runner, SimulatorProbeRunner)
        assert runner.real_simulator is True
        assert runner.status == "READY_GATE_ALLOWED"

        cand = make_candidate()
        fast = runner.probe(cand, stage="fast", student_episodes=2,
                            reference_episodes=1)
        assert fast.stage == "fast"
        bank = derive_seed_bank(cand.candidate_hash, stage="fast", n=2)
        assert fast.student_success_rate == round(
            sum(1 for s in bank if s % 2 == 0) / 2, 6)
        full = runner.probe(cand, stage="full", student_episodes=4,
                            reference_episodes=2)
        assert full.stage == "full"
        assert runner.probe_calls == 2
        assert runner.total_transitions == (3 + 6) * HORIZON
        # honesty: the capability flag is NOT flipped by a scripted env
        assert C.REAL_SIMULATOR_PROBE is False

    def test_fake_real_runner_drives_the_staged_funnel(self, monkeypatch):
        monkeypatch.setattr(C, "REAL_SIMULATOR_PROBE_AUTHORIZED", True)
        gate = FeedbackLaunchGate(EXECUTION_MODE_REAL)
        runner = RealCraftaxProbeRunner(*make_factories(), gate=gate,
                                        student_binding=local_symbolic_binding())
        candidates = [make_candidate(i) for i in range(3)]
        batch = run_staged_funnel(candidates, runner, window=0)
        assert batch.funnel_stats["raw"] == 3
        assert batch.funnel_stats["static_rejects"] == 0
        assert runner.total_transitions > 0
        assert batch.funnel_stats["total_simulator_transitions"] == \
            runner.total_transitions
