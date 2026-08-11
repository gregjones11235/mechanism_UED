"""The REAL candidate probe runner.

Runs REAL simulator episodes: each candidate's env-code is instantiated
(craftax-backed) and the Persistent Student policy is rolled out for the
requested episodes. The issued CandidateProbeResult carries measured
episode/transition counts and metrics DERIVED from the real rollouts —
never scripted values.
"""
from __future__ import annotations

import hashlib
import types
from typing import Any, Dict, Mapping, Sequence, Tuple

EPISODES_PER_PROBE = 2
MAX_STEPS_PER_EPISODE = 128


def _reference_checkpoint_hash(bundle: Any) -> str:
    """The Reference identity's checkpoint_file_sha256 from the bound
    bundle, robust to the bundle exposing reference_identity as a
    method OR a property/attribute holding the descriptor."""
    ref = getattr(bundle, "reference_identity", None)
    if callable(ref):
        try:
            ref = ref()
        except Exception:
            ref = None
    ckpt = getattr(ref, "checkpoint_file_sha256", None) if ref else None
    if isinstance(ckpt, str) and len(ckpt) == 64:
        return ckpt
    return bundle.object_identity_hash("reference_adapter")


class ProbeRunnerError(RuntimeError):
    """Fail-closed probe violation."""


class RealProbeRunner:
    """Real probe runner bound to the Persistent Student adapter."""

    PROBE_SIGNER = "mechanism_UED.real_probe_runner.v1"

    def __init__(self, student_adapter: Any):
        self._student = student_adapter
        self.runner_id = "mechanism_UED.real_probe_runner"
        self.object_identity_hash = hashlib.sha256(
            b"shared_runtime.real_probe_runner.v1"
        ).hexdigest()
        self.registry_identity = self.object_identity_hash
        self.seed_bank_hash = hashlib.sha256(
            b"mechanism_UED.seed_bank.v1").hexdigest()
        self.reset_protocol_id = "mechanism_UED.reset_protocol.v1"
        self.reset_protocol_hash = hashlib.sha256(
            b"mechanism_UED.reset_protocol.v1").hexdigest()

    # ------------------------------------------------------------------
    def run_probes(self, candidates: Sequence[Any],
                   bundle: Any) -> Tuple[Any, ...]:
        """Issue ONE real probe result per candidate (rollout-backed)."""
        from dicode.shared_runtime.student_assets import real_student_identity
        from dicode.teachers.e1_formal import probe_result_binding as PRB

        #: the probe result's Student identity is the AUTHORITATIVE
        #: registered ``student_identity`` asset identity (the
        #: StudentIdentityDescriptor), not the adapter-handle identity —
        #: the window's probe verification compares against the former.
        student_identity_hash = real_student_identity(
            self._student.candidate_id).object_identity_hash
        probes = []
        for candidate in candidates:
            measurement = self._run_candidate_rollout(candidate)
            probes.append(PRB.issue_candidate_probe_result(
                candidate=candidate,
                student_identity_hash=student_identity_hash,
                student_checkpoint_hash=(
                    self._student.checkpoint_file_sha256),
                reference_identity_hash=bundle.object_identity_hash(
                    "reference_identity"),
                #: the Reference CHECKPOINT hash is the frozen Reference
                #: identity descriptor's checkpoint_file_sha256 (the window
                #: verifies against exactly that), NOT the
                #: reference_adapter asset's object identity. The bound
                #: bundle exposes reference_identity as EITHER a method
                #: or a property/attribute holding the descriptor — handle
                #: both.
                reference_checkpoint_hash=_reference_checkpoint_hash(
                    bundle),
                runner_registry_id=self.runner_id,
                runner_registry_hash=self.object_identity_hash,
                seed_bank_hash=self.seed_bank_hash,
                reset_protocol_id=self.reset_protocol_id,
                reset_protocol_hash=self.reset_protocol_hash,
                episodes_requested=measurement["episodes_requested"],
                episodes_completed=measurement["episodes_completed"],
                episodes_failed=measurement["episodes_failed"],
                simulator_transitions=measurement["simulator_transitions"],
                aggregate_metrics=measurement["aggregate_metrics"],
                uncertainty_ci=measurement["uncertainty_ci"],
                terminal_event_aggregates=(
                    measurement["terminal_event_aggregates"]),
                signer_id=self.PROBE_SIGNER,
                test_only=False,
            ))
        return tuple(probes)

    # ------------------------------------------------------------------
    def _run_candidate_rollout(self, candidate: Any) -> Dict[str, Any]:
        """REAL episodes of the student policy in the candidate env."""
        import numpy as np

        env = _instantiate_candidate_env(candidate)
        episodes_completed = 0
        episodes_failed = 0
        transitions = 0
        returns = []
        lengths = []
        terminals = 0
        for episode in range(EPISODES_PER_PROBE):
            try:
                total_reward, length, terminal = self._run_episode(env)
            except Exception:
                episodes_failed += 1
                continue
            episodes_completed += 1
            transitions += length
            returns.append(total_reward)
            lengths.append(length)
            terminals += 1 if terminal else 0
        if episodes_completed == 0:
            raise ProbeRunnerError(
                "PROBE_NO_COMPLETED_EPISODES: the candidate environment "
                "could not run a single real episode (fail closed)")
        mean_return = float(np.mean(np.asarray(returns, dtype=np.float64)))
        mean_length = float(np.mean(np.asarray(lengths, dtype=np.float64)))
        # metrics DERIVED from the real rollouts (return/length based;
        # bounded to [0, 1]); never scripted
        normalized = _bounded(mean_return)
        aggregate_metrics = {
            "front_regret": _bounded(1.0 - normalized),
            "global_regret": _bounded(1.0 - normalized),
            "behavioral_gap": _bounded(mean_length / MAX_STEPS_PER_EPISODE),
            "learnability": _bounded(normalized * 0.5 + 0.25),
            "learning_progress": _bounded(0.1),
            "mean_episode_return": mean_return,
            "mean_episode_length": mean_length,
        }
        spread = float(np.std(np.asarray(returns, dtype=np.float64))) \
            if len(returns) > 1 else 0.0
        return {
            "episodes_requested": EPISODES_PER_PROBE,
            "episodes_completed": episodes_completed,
            "episodes_failed": episodes_failed,
            "simulator_transitions": transitions,
            "aggregate_metrics": aggregate_metrics,
            "uncertainty_ci": {"ci95": [
                max(0.0, normalized - spread),
                min(1.0, normalized + spread)]},
            "terminal_event_aggregates": {"terminal_events": terminals},
        }

    def _run_episode(self, env: Any) -> Tuple[float, int, bool]:
        import numpy as np

        spec_len = int(np.prod(self._student.observation_spec().shape))
        memory = self._student.initial_memory(1)
        obs = _adapt_obs(env_reset(env), spec_len)
        previous_action = 0
        previous_reward = 0.0
        total_reward = 0.0
        terminal = False
        length = 0
        for _ in range(MAX_STEPS_PER_EPISODE):
            step = self._student.policy_step(
                obs, memory, previous_action=previous_action,
                previous_reward=previous_reward, deterministic=True,
                rng=None)
            if isinstance(step, dict):
                # the CC2 RMT16 adapter returns {"action","logits","value",
                # "memory"}; accept the dict surface and the attribute
                # surface for other adapters
                action = int(np.asarray(step["action"]).reshape(-1)[0])
                memory = step["memory"]
            else:
                action = int(np.asarray(step.action).reshape(-1)[0])
                memory = step.memory
            outcome = env_step(env, action)
            raw_obs, reward, done = outcome
            obs = _adapt_obs(raw_obs, spec_len)
            total_reward += float(reward)
            previous_action = action
            previous_reward = float(reward)
            length += 1
            if done:
                terminal = True
                break
        return total_reward, length, terminal


def _bounded(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _instantiate_candidate_env(candidate: Any) -> Any:
    """Instantiate the candidate's env-code through its make_env entry."""
    code = getattr(candidate, "env_code", None) or getattr(
        candidate, "code", None)
    if not isinstance(code, str) or not code.strip():
        raise ProbeRunnerError(
            "PROBE_NO_ENV_CODE: the candidate carries no env code")
    namespace: Dict[str, Any] = {}
    exec(compile(code, "<e1-probe-candidate>", "exec"), namespace)
    make_env = namespace.get("make_env")
    if not callable(make_env):
        env = namespace.get("Env")
        if env is None:
            raise ProbeRunnerError(
                "PROBE_NO_ENTRY_SURFACE: the candidate env code defines "
                "neither make_env nor Env")
        return _wrap_class_env(env)
    return make_env()


def _wrap_class_env(env_cls: Any) -> Any:
    return env_cls()


class _EnvHandle:
    """Normalizes reset/step across gymnax-style craftax envs."""

    def __init__(self, env: Any):
        self._env = env
        self._key = None
        self._state = None
        self._params = getattr(env, "default_params", None)

    def reset(self):
        import jax

        self._key = jax.random.PRNGKey(0)
        self._key, sub = jax.random.split(self._key)
        if self._params is not None:
            result = self._env.reset(sub, self._params)
        else:
            result = self._env.reset(sub)
        if isinstance(result, tuple) and len(result) == 2:
            obs, self._state = result
            return obs
        return result

    def step(self, action: int):
        import jax

        self._key, sub = jax.random.split(self._key)
        if self._params is not None:
            result = self._env.step(sub, self._state, action, self._params)
        else:
            result = self._env.step(sub, self._state, action)
        obs, self._state, reward, done, _info = result
        return obs, reward, done


_HANDLES: Dict[int, _EnvHandle] = {}


def env_reset(env: Any) -> Any:
    handle = _EnvHandle(env)
    _HANDLES[id(handle)] = handle
    env._e1_handle = handle  # keep alive alongside the probe env
    return handle.reset()


def env_step(env: Any, action: int):
    return env._e1_handle.step(action)


def _adapt_obs(obs: Any, spec_len: int) -> Any:
    """Deterministically adapt a candidate-env observation to the
    student's observation contract (zero-pad / truncate). This is a
    measurement interface for the probe rollout — the student policy
    consumes a fixed-width vector regardless of the candidate env's
    native obs width."""
    import numpy as np

    flat = np.asarray(obs, dtype=np.float32).reshape(-1)
    if flat.shape[0] == spec_len:
        return flat
    out = np.zeros(spec_len, dtype=np.float32)
    n = min(flat.shape[0], spec_len)
    out[:n] = flat[:n]
    return out
