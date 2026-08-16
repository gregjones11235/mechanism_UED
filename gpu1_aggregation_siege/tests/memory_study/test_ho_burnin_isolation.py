"""Burn-in isolation tests: determinism, mode structure, mechanical FailClosed,
structural env isolation, and both step_fn adapters."""
from __future__ import annotations

import inspect

import pytest

from dicode.memory_study.ho_burnin import (
    RNG_STREAM_ID_BURNIN,
    burnin_history,
    wrap_backend_policy_forward_eval,
    wrap_tier3_projection_policy,
)
from dicode.memory_study.ho_contract import (
    FailClosed,
    HOMode,
    IsolationContext,
    hash_pytree,
)
from dicode.memory_study.ho_capture_bank import _payload_sha
from dicode.memory_study.ho_contract import HistoryCapture

SEG = ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))


def _capture():
    sha = _payload_sha("C0", SEG, 11, "P1", 2, 0)
    return HistoryCapture(capture_id="C0", obs_segment=SEG, source_seed=11,
                          capture_policy_id="P1", bank_hash="BANK",
                          payload_sha256=sha)


def _ctx(params):
    return IsolationContext(
        params_sha_before=hash_pytree(params),
        env_state_payload_hash=None,
        rng_stream_id=RNG_STREAM_ID_BURNIN,
        task_embedding_hash="T", timestep=0, inventory_hash="I",
        position_hash="X", entities_hash="E")


def _step(params, memory, obs_row):
    # deterministic memory accumulation; never touches params
    return memory + (tuple(obs_row),)


def test_ho_real_feeds_full_segment():
    params = {"w": [1.0]}
    mem, receipt = burnin_history(_step, params, (), _capture(),
                                  HOMode.HO_REAL, _ctx(params))
    assert mem == SEG
    assert receipt.burnin_steps == 3
    assert receipt.verdict == "PASS"


def test_ho_zero_feeds_zero_shaped_segment():
    params = {"w": [1.0]}
    mem, receipt = burnin_history(_step, params, (), _capture(),
                                  HOMode.HO_ZERO, _ctx(params))
    assert mem == ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    assert receipt.burnin_steps == 3


def test_base_is_identity_with_receipt():
    params = {"w": [1.0]}
    memory = ("seed-memory",)
    mem, receipt = burnin_history(None, params, memory, _capture(),
                                  HOMode.BASE, _ctx(params))
    assert mem is memory
    assert receipt.burnin_steps == 0
    assert receipt.ho_mode == "base"


def test_determinism_two_runs_equal():
    params = {"w": [1.0, 2.0]}
    m1, r1 = burnin_history(_step, params, (), _capture(),
                            HOMode.HO_REAL, _ctx(params))
    m2, r2 = burnin_history(_step, params, (), _capture(),
                            HOMode.HO_REAL, _ctx(params))
    assert m1 == m2
    assert r1.params_sha_before == r2.params_sha_before
    assert r1.params_sha_after == r2.params_sha_after


def test_params_mutation_fails_closed():
    params = {"w": [1.0]}

    def evil_step(p, memory, obs_row):
        p["w"] = [999.0]          # mutate params during burn-in
        return memory + (obs_row,)

    with pytest.raises(FailClosed, match="params_invariant"):
        burnin_history(evil_step, params, (), _capture(),
                       HOMode.HO_REAL, _ctx(params))


def test_env_state_context_fails_closed():
    params = {"w": [1.0]}
    ctx = IsolationContext(
        params_sha_before=hash_pytree(params),
        env_state_payload_hash="ENVHASH",
        rng_stream_id=RNG_STREAM_ID_BURNIN,
        task_embedding_hash="T", timestep=0, inventory_hash="I",
        position_hash="X", entities_hash="E")
    with pytest.raises(FailClosed, match="env_state_structurally_absent"):
        burnin_history(_step, params, (), _capture(), HOMode.HO_REAL, ctx)


def test_params_snapshot_disagreement_fails_closed():
    params = {"w": [1.0]}
    ctx = IsolationContext(
        params_sha_before="WRONG_SNAPSHOT",
        env_state_payload_hash=None,
        rng_stream_id=RNG_STREAM_ID_BURNIN,
        task_embedding_hash="T", timestep=0, inventory_hash="I",
        position_hash="X", entities_hash="E")
    with pytest.raises(FailClosed, match="PARAMS_SNAPSHOT_DISAGREEMENT"):
        burnin_history(_step, params, (), _capture(), HOMode.HO_REAL, ctx)


def test_burnin_signature_has_no_env_parameter():
    sig = inspect.signature(burnin_history)
    assert "env_state" not in sig.parameters
    assert "env" not in sig.parameters


def test_non_homode_rejected():
    params = {"w": [1.0]}
    with pytest.raises(FailClosed, match="UNKNOWN_HO_MODE"):
        burnin_history(_step, params, (), _capture(), "ho_real", _ctx(params))


def test_wrap_tier3_projection_policy_adapter():
    class FakePolicy:
        def __init__(self):
            self.ms = None
            self.calls = []

        def __call__(self, obs, env_state):
            assert env_state is None
            self.calls.append(tuple(obs))
            self.ms = (self.ms or ()) + (sum(obs),)
            return 7  # action must be discarded by burn-in

    pol = FakePolicy()
    step = wrap_tier3_projection_policy(pol)
    params = {"w": [1.0]}
    mem, receipt = burnin_history(step, params, (), _capture(),
                                  HOMode.HO_REAL, _ctx(params))
    assert mem == (3.0, 7.0, 11.0)
    assert pol.calls == [list(r) for r in SEG] or pol.calls == [tuple(r) for r in SEG]
    assert receipt.burnin_steps == 3


def test_wrap_tier3_policy_rejects_non_policy():
    with pytest.raises(FailClosed, match="NOT_A_TIER3_PROJECTION_POLICY"):
        wrap_tier3_projection_policy(object())


def test_wrap_backend_policy_forward_eval_adapter():
    class FakeBackend:
        def policy_forward_eval(self, params, memory, obs):
            # obs arrives batched (1, obs_dim)
            rows = [tuple(r) for r in obs]
            new_memory = memory + tuple(rows)
            return ("pi", "value", "mem_out", new_memory)

    step = wrap_backend_policy_forward_eval(FakeBackend())
    params = {"w": [1.0]}
    mem, receipt = burnin_history(step, params, (), _capture(),
                                  HOMode.HO_REAL, _ctx(params))
    assert mem == SEG
    assert receipt.burnin_steps == 3


def test_wrap_backend_rejects_missing_forward():
    with pytest.raises(FailClosed, match="NOT_A_TRAINING_BACKEND"):
        wrap_backend_policy_forward_eval(object())