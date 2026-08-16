"""State-start vectorized short rollouts.

Core litesim requirement: rollouts begin from FrontierStateBank states (or
canonical resets) WITHOUT a full env.reset, stay strictly on-policy (batch is
tagged with the generating policy hash / student version), and carry the
architecture-correct entering recurrent state (SlowGRU longstate included).
"""
from __future__ import annotations

from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..runtime import recurrent_state
from ..runtime.hashing import hash_pytree
from .rollout_batch import OnPolicyRolloutBatch


def batched_reset(env, env_params, keys):
    """Vectorized reset: craftax reset takes a single key; vmap batches it."""
    return jax.vmap(lambda k: env.reset(k, env_params))(keys)


def batched_get_obs(env, state):
    """The base env is single-env; batch via vmap (as the project wrappers do)."""
    return jax.vmap(env.get_obs)(state)


def batched_step(env, keys, state, actions, env_params):
    return jax.vmap(env.step, in_axes=(0, 0, 0, None))(
        keys, state, actions, env_params)


def _to_jnp_tree(tree):
    return jax.tree_util.tree_map(jnp.asarray, tree)


def collect_rollouts(*, env, env_params, backend, params,
                     start_states: Sequence[Any],
                     start_memories: Sequence[dict],
                     horizon: int, rng, deterministic: bool = True,
                     student_version: str = "",
                     frontier_family: str = "",
                     start_state_ids: Sequence[str] = (),
                     architecture_family: str = "slice",
                     allow_memory_reset_experiment: bool = False,
                     collect_trace: bool = True,
                     collect_memory_trace: bool = False) -> OnPolicyRolloutBatch:
    """Vectorized short-horizon rollouts from given (state, memory) pairs."""
    state = _concat_batch(list(start_states))
    batch = int(np.asarray(state.player_position).shape[0])
    entering_np = _concat_batch(
        [{k: np.asarray(v) for k, v in m.items()} for m in start_memories])
    recurrent_state.assert_state_start_alignment(
        entering_np, batch, architecture_family=architecture_family,
        allow_memory_reset_experiment=allow_memory_reset_experiment)
    memory = _to_jnp_tree(entering_np)
    obs = batched_get_obs(env, state)
    policy_hash = hash_pytree(params)

    obs_l, act_l, rew_l, done_l, lp_l, val_l = [], [], [], [], [], []
    trace = [state] if collect_trace else []
    mem_trace = [entering_np] if collect_memory_trace else []
    terminal = np.array(["horizon"] * batch, dtype=object)
    rng, sub = jax.random.split(rng)
    for t in range(int(horizon)):
        pi, value, _mem_out, new_memory = backend.policy_forward_eval(
            params, memory, obs)
        rng, key_a, key_s = jax.random.split(rng, 3)
        action = pi.mode() if deterministic else pi.sample(seed=key_a)
        logp = pi.log_prob(action)
        step_keys = jax.random.split(key_s, batch)
        obs_next, state, reward, done, info = batched_step(
            env, step_keys, state, action, env_params)
        done_np = np.asarray(done).astype(bool)
        for b in range(batch):
            if done_np[b] and terminal[b] == "horizon":
                terminal[b] = "terminal"
        # mask post-terminal reward (no auto-reset env: keep dynamics frozen out)
        reward = jnp.where(done, 0.0, reward)
        obs_l.append(np.asarray(obs)); act_l.append(np.asarray(action))
        rew_l.append(np.asarray(reward)); done_l.append(done_np.astype(np.float64))
        lp_l.append(np.asarray(logp)); val_l.append(np.asarray(value))
        if collect_trace:
            trace.append(state)
        if collect_memory_trace:
            mem_trace.append({k: np.asarray(v) for k, v in new_memory.items()})
        obs = obs_next
        memory = new_memory

    _pi, bootstrap, _m, _nm = backend.policy_forward_eval(params, memory, obs)

    return OnPolicyRolloutBatch(
        obs=np.stack(obs_l), actions=np.stack(act_l), rewards=np.stack(rew_l),
        dones=np.stack(done_l), old_logp=np.stack(lp_l),
        old_value=np.stack(val_l),
        entering_memory={k: np.asarray(v) for k, v in entering_np.items()},
        bootstrap_value=np.asarray(bootstrap),
        start_state_ids=list(start_state_ids) or [f"reset#{i}" for i in range(len(start_states))],
        frontier_family=frontier_family,
        rollout_length=int(horizon), horizon=int(horizon),
        policy_hash=policy_hash, student_version=student_version,
        terminal_reason=list(terminal), trace=trace,
        memory_trace=mem_trace)


def _concat_batch(trees: Sequence[Any]) -> Any:
    """Concatenate already-batched pytrees along the batch (env) axis."""
    if len(trees) == 1:
        return trees[0]
    return jax.tree_util.tree_map(
        lambda *xs: np.concatenate([np.asarray(x) for x in xs], axis=0), *trees)


def collect_full_rollouts(*, env, env_params, backend, params, num_envs: int,
                          horizon: int, rng, student_version: str = "",
                          deterministic: bool = True) -> OnPolicyRolloutBatch:
    """Canonical full-episode rollouts (benchmark / FULL_ONLY arm)."""
    rng, key_r = jax.random.split(rng)
    keys = jax.random.split(key_r, num_envs)
    _obs0, state = batched_reset(env, env_params, keys)
    mem = _to_jnp_tree(backend.init_runner_memory(num_envs))
    return collect_rollouts(
        env=env, env_params=env_params, backend=backend, params=params,
        start_states=[state], start_memories=[{k: np.asarray(v) for k, v in mem.items()}],
        horizon=horizon, rng=rng, deterministic=deterministic,
        student_version=student_version, frontier_family="full_only",
        start_state_ids=["canonical_reset"])