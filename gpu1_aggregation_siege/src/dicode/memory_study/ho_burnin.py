"""HO burn-in execution.

Feeds a result-blind history segment through a student's memory update WITHOUT
any environment interaction, gradient step, or RNG consumption. The step_fn
minimal protocol is:

    step_fn(params, memory, obs_row) -> new_memory

Adapters wrap either frozen tier3 projection policies or a
StudentTrainingBackend.policy_forward_eval implementation (wrappers live in
this module by design, so the probe never touches owner internals directly).

Determinism contract: identical (step_fn, params, memory, segment) inputs MUST
produce leaf-equal memory outputs; burn-in consumes no RNG stream of its own
(the declared rng_stream_id is provenance-only).
"""
from __future__ import annotations

import inspect
from typing import Any, Callable

from .ho_contract import (
    FailClosed,
    HOMode,
    HistoryCapture,
    IsolationContext,
    IsolationReceipt,
    hash_pytree,
)

StepFn = Callable[[Any, Any, Any], Any]

#: Provenance label recorded on every receipt: burn-in never consumes the
#: env/rollout RNG stream; it runs a pure forward pass.
RNG_STREAM_ID_BURNIN = "ho_burnin_independent_stream_v1"


def burnin_history(step_fn: StepFn, params: Any, memory: Any,
                   capture: HistoryCapture, mode: HOMode,
                   ctx: IsolationContext):
    """Run HO burn-in and return (memory_out, receipt).

    BASE    : no forward steps; memory returned unchanged (control arm).
    HO_ZERO : all-zero segment shaped exactly like capture.obs_segment.
    HO_REAL : the captured segment itself.

    Raises FailClosed on any isolation violation (params mutated, env state
    present, context hash mismatch, malformed capture).
    """
    if not isinstance(mode, HOMode):
        raise FailClosed("UNKNOWN_HO_MODE: %r" % (mode,))
    if not isinstance(capture, HistoryCapture):
        raise FailClosed("CAPTURE_REQUIRED: burn-in provenance needs a "
                         "HistoryCapture for every mode (BASE included)")
    capture.validate()
    if ctx.env_state_payload_hash is not None:
        raise FailClosed("ISOLATION_VIOLATION: env_state_structurally_absent "
                         "(burn-in may not receive environment state)")
    params_sha_before = hash_pytree(params)
    if params_sha_before != ctx.params_sha_before:
        raise FailClosed(
            "PARAMS_SNAPSHOT_DISAGREEMENT: caller-declared params_sha_before "
            "does not match the params object actually supplied")

    if mode is HOMode.BASE:
        segment = ()
    elif mode is HOMode.HO_ZERO:
        width = len(capture.obs_segment[0])
        segment = tuple(tuple(0.0 for _ in range(width))
                        for _ in capture.obs_segment)
    else:  # HO_REAL
        segment = capture.obs_segment

    memory_out = memory
    steps = 0
    if mode is not HOMode.BASE:
        if not callable(step_fn):
            raise FailClosed("STEP_FN_REQUIRED for mode=%s" % mode.value)
        for obs_row in segment:
            memory_out = step_fn(params, memory_out, obs_row)
            steps += 1

    params_sha_after = hash_pytree(params)
    receipt = IsolationReceipt.issue(
        ho_mode=mode.value,
        params_sha_before=params_sha_before,
        params_sha_after=params_sha_after,
        ctx=ctx,
        burnin_steps=steps)
    return memory_out, receipt


# Structural isolation guard (mechanical, import-time): burn-in must never
# grow an environment parameter. Tests additionally introspect this.
assert "env_state" not in inspect.signature(burnin_history).parameters


def wrap_tier3_projection_policy(policy: Any) -> StepFn:
    """Adapt a frozen tier3 projection policy (reset()/__call__(obs, env_state)
    with memory carried in policy.ms) to the step_fn protocol.

    The action produced during burn-in is DISCARDED: burn-in only advances the
    carried memory. env_state is passed as None by contract; the frozen
    policies ignore it (their signatures keep it for evaluator parity).
    """
    if not hasattr(policy, "ms") or not callable(policy):
        raise FailClosed("NOT_A_TIER3_PROJECTION_POLICY: expected callable "
                         "with an .ms memory attribute")

    def step_fn(params: Any, memory: Any, obs_row: Any) -> Any:
        policy.ms = memory
        policy(obs_row, None)
        return policy.ms

    return step_fn


def wrap_backend_policy_forward_eval(backend: Any) -> StepFn:
    """Adapt StudentTrainingBackend.policy_forward_eval(params, memory, obs) ->
    (pi, value, memory_out, new_memory) to the step_fn protocol by adding the
    batch axis to a single observation row and keeping only new_memory."""
    fwd = getattr(backend, "policy_forward_eval", None)
    if not callable(fwd):
        raise FailClosed("NOT_A_TRAINING_BACKEND: policy_forward_eval missing")

    def step_fn(params: Any, memory: Any, obs_row: Any) -> Any:
        if hasattr(obs_row, "shape"):
            obs_batch = obs_row[None, :]
        else:
            obs_batch = [list(obs_row)]
        _pi, _value, _memory_out, new_memory = fwd(params, memory, obs_batch)
        return new_memory

    return step_fn
