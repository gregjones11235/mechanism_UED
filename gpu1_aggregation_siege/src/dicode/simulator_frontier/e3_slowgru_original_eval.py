"""Evaluation-only Original Craftax protocol for an E3 SlowGRU RunState.

This module never trains, calls an LLM, or writes W&B data.  The production
entrypoint restores one canonical RunState, mounts the existing authenticated
SlowGRU adapter, and evaluates the caller-supplied (updated) parameters.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA = "simulator_frontier.e3_slowgru_original_task_eval/v1"
CANDIDATE = "SLOWGRU_PERSISTENT_CANONICAL_98304"
ARCHITECTURE = "SLOWGRU"
POLICY = "stochastic_categorical_jax_split_schedule"


class E3SlowGRUOriginalEvalError(RuntimeError):
    """Fail-closed evaluation contract error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise E3SlowGRUOriginalEvalError(message)


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pytree_sha256(value: Any) -> str:
    import jax

    digest = hashlib.sha256()
    structure = str(jax.tree_util.tree_structure(value)).encode("utf-8")
    digest.update(structure)
    for leaf in jax.tree_util.tree_leaves(value):
        array = np.asarray(jax.device_get(leaf))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def finite_pytree(value: Any) -> bool:
    import jax

    try:
        for leaf in jax.tree_util.tree_leaves(value):
            array = np.asarray(jax.device_get(leaf))
            if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
                return False
        return True
    except Exception:
        return False


def load_evaluation_runstate(stem: str) -> dict[str, Any]:
    """Restore and validate the complete canonical RunState read-only."""
    from dicode.simulator_frontier.runstate_codec import (
        RunStateCheckpointManager,
        fresh_process_restore,
    )

    restored = RunStateCheckpointManager().restore(stem)
    state, metadata = restored["run_state"], restored["metadata"]
    _require(state.get("candidate_id") == CANDIDATE,
             f"candidate_id must be {CANDIDATE!r}")
    _require(state.get("architecture_family") == ARCHITECTURE,
             f"architecture_family must be {ARCHITECTURE!r}")
    _require(state.get("params") is not None and state.get("opt_state") is not None,
             "RunState must contain params and opt_state")
    _require(finite_pytree(state["params"]), "RunState params contain NaN/Inf")
    _require(finite_pytree(state["opt_state"]), "RunState opt_state contains NaN/Inf")
    # The CLI prepends ``src`` to sys.path in-process.  Propagate that exact
    # source root to the independent restore child instead of assuming the
    # package is installed site-wide.
    source_root = str(Path(__file__).resolve().parents[2])
    fresh = fresh_process_restore(stem, extra_pythonpath=source_root)
    _require(fresh.get("restored") is True,
             "fresh-process RunState restore did not report restored=true")
    for key in ("global_update_step", "current_session_idx"):
        _require(int(fresh[key]) == int(metadata[key]),
                 f"fresh-process {key} differs from metadata")
    return {"state": state, "metadata": metadata, "fresh_restore": fresh}


def original_task_protocol(*, seed: int, num_envs: int,
                           num_steps: int) -> dict[str, Any]:
    _require(int(seed) >= 0, "seed must be >= 0")
    _require(int(num_envs) >= 1, "num_envs must be >= 1")
    _require(int(num_steps) >= 1, "num_steps must be >= 1")
    return {
        "environment": "minicraftax.envs.craftax.CraftaxAugObsTrain",
        "seed": int(seed),
        "num_worlds": int(num_envs),
        "horizon": int(num_steps),
        "max_timesteps": int(num_steps),
        "policy": POLICY,
        "achievement_schema": "craftax.craftax.constants.Achievement",
        "memory_semantics": (
            "fresh SlowGRU memory per world batch; persistent carry within episode"
        ),
    }


def dicode_original_rng_prefix(seed: int) -> tuple[Any, Any]:
    """Return the post-reset carry key and reset key from DiCode's exact chain.

    The three splits correspond to online_evaluation.run_session_evaluation,
    craftax_evaluation.main, and make_evaluate respectively.
    """
    import jax

    rng = jax.random.PRNGKey(int(seed))
    rng, evaluator_rng = jax.random.split(rng)
    evaluator_rng, evaluate_rng = jax.random.split(evaluator_rng)
    evaluate_rng, reset_rng = jax.random.split(evaluate_rng)
    return evaluate_rng, reset_rng


def dicode_next_step_keys(rng: Any) -> tuple[Any, Any, Any]:
    """Match DiCode's per-step action split followed by environment split."""
    import jax

    rng, action_rng = jax.random.split(rng)
    rng, step_rng = jax.random.split(rng)
    return rng, action_rng, step_rng


def dicode_slowgru_policy_step(*, adapter: Any, params: Any,
                               observations: Any, memory: Any,
                               action_rng: Any,
                               num_envs: int) -> tuple[np.ndarray, Any]:
    """Forward updated SlowGRU params, then sample with DiCode's JAX key.

    ``SlowGRUStudentAdapter`` accepts a NumPy ``Generator`` on its stochastic
    convenience path, whereas DiCode samples a whole categorical batch from
    one JAX key.  Request logits through the deterministic adapter path and
    deliberately ignore its argmax action; sampling remains owned here and
    therefore follows ``craftax_evaluation.make_evaluate`` exactly.
    """
    import jax
    import jax.numpy as jnp

    output = adapter.policy_step(
        params, observations, memory, None, None, None, True)
    _require(isinstance(output, Mapping),
             "SlowGRU policy_step must return a mapping")
    _require("logits" in output,
             "SlowGRU policy_step output is missing logits")
    new_memory = output.get("new_memory", output.get("memory"))
    _require(new_memory is not None,
             "SlowGRU policy_step output is missing new_memory/memory")

    logits_np = np.asarray(jax.device_get(output["logits"]))
    _require(logits_np.ndim == 2,
             f"policy logits must be rank-2, got {logits_np.shape}")
    _require(logits_np.shape[0] == int(num_envs),
             f"policy logits batch {logits_np.shape[0]} != {num_envs}")
    _require(logits_np.shape[1] >= 1,
             "policy logits action dimension must be non-empty")
    _require(np.issubdtype(logits_np.dtype, np.number)
             and bool(np.isfinite(logits_np).all()),
             "policy logits must be finite numeric values")
    actions = np.asarray(jax.device_get(jax.random.categorical(
        action_rng, jnp.asarray(logits_np), axis=-1)), dtype=np.int32)
    _require(actions.shape == (int(num_envs),),
             f"sampled action shape {actions.shape} != ({num_envs},)")
    return actions, new_memory


def accumulate_first_episode_step(
    *, returns: Any, lengths: Any, finished: Any,
    achievement_flags: Mapping[str, Any], reward: Any, new_done: Any,
    info: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Accumulate first-episode metrics from authoritative step ``info``.

    Craftax may expose an auto-reset state after a terminal step.  DiCode's
    evaluator therefore accumulates the ``Achievements/*`` values returned in
    ``info`` while the world is active; it does not inspect the next EnvState.
    """
    rewards = np.asarray(returns, dtype=np.float64).copy()
    episode_lengths = np.asarray(lengths, dtype=np.int64).copy()
    done_before = np.asarray(finished, dtype=np.bool_)
    reward_array = np.asarray(reward, dtype=np.float64)
    done_after_step = np.asarray(new_done, dtype=np.bool_)
    _require(reward_array.shape == done_before.shape,
             "reward and finished shapes differ")
    _require(done_after_step.shape == done_before.shape,
             "new_done and finished shapes differ")
    active = ~done_before
    rewards += reward_array * active
    episode_lengths += active.astype(np.int64)
    updated: dict[str, np.ndarray] = {}
    for name, old_flags in achievement_flags.items():
        key = f"Achievements/{name.lower()}"
        _require(key in info, f"Original Task info missing required key {key!r}")
        values = np.asarray(info[key])
        _require(values.shape == done_before.shape,
                 f"info {key!r} shape {values.shape} != {done_before.shape}")
        updated[name] = np.asarray(old_flags, dtype=np.bool_) | (
            (values > 0) & active
        )
    return rewards, episode_lengths, done_before | done_after_step, updated


def aggregate_world_metrics(*, returns: Any, lengths: Any, finished: Any,
                            achievement_flags: Mapping[str, Any],
                            max_floor: Any | None,
                            death: Any | None = None,
                            timeout: Any | None = None) -> dict[str, Any]:
    rewards = np.asarray(returns, dtype=np.float64)
    episode_lengths = np.asarray(lengths, dtype=np.int64)
    done = np.asarray(finished, dtype=np.bool_)
    _require(rewards.ndim == episode_lengths.ndim == done.ndim == 1,
             "per-world returns/lengths/finished must be rank-1")
    _require(len(rewards) > 0 and len(rewards) == len(episode_lengths) == len(done),
             "per-world metric lengths differ or are empty")
    _require(np.isfinite(rewards).all(), "per-world returns contain NaN/Inf")
    _require((episode_lengths >= 0).all(), "episode lengths must be non-negative")

    count_finished = int(done.sum())
    _require(count_finished > 0,
             "Original Task evaluation finished zero worlds within horizon")
    completed_rewards = rewards[done]
    completed_lengths = episode_lengths[done]

    achievements: dict[str, Any] = {}
    skill_rates: dict[str, float] = {}
    for name, values in sorted(achievement_flags.items()):
        flags = np.asarray(values, dtype=np.bool_)
        _require(flags.shape == done.shape,
                 f"achievement {name!r} per-world shape mismatch")
        rate = float(flags[done].sum() / count_finished)
        skill_rates[f"skill_{name}"] = rate
        achievements[name] = {
            "rate": rate,
            "per_world": flags.tolist(),
        }

    def optional_metric(value: Any | None, reason: str) -> dict[str, Any]:
        if value is None:
            return {"value": None, "per_world": None, "reason": reason}
        array = np.asarray(value)
        _require(array.shape == done.shape, "optional per-world metric shape mismatch")
        return {"value": float(np.mean(array)), "per_world": array.tolist(),
                "reason": None}

    floor_metric = optional_metric(
        max_floor, "EnvState exposes no authoritative player_level field")
    return {
        "mean_return": float(np.mean(completed_rewards)),
        "median_return": float(np.median(completed_rewards)),
        "per_world_returns": rewards.tolist(),
        "mean_episode_length": float(np.mean(completed_lengths)),
        "median_episode_length": float(np.median(completed_lengths)),
        "per_world_episode_lengths": episode_lengths.tolist(),
        "finished_count": count_finished,
        "unfinished_count": int(len(done) - count_finished),
        "per_world_finished": done.tolist(),
        "death": optional_metric(
            death, "Original Task env info has no authoritative death field"),
        "timeout": optional_metric(
            timeout, "Original Task env info has no authoritative timeout field"),
        "max_floor": floor_metric,
        "achievements": achievements,
        "skill_rates": skill_rates,
        "task_success": {
            "value": None,
            "per_world": None,
            "reason": (
                "DiCode Original Task evaluator defines no authoritative "
                "aggregate task-success predicate; use skill_rates"
            ),
        },
    }


def run_original_task_rollout(*, adapter: Any, params: Any, seed: int,
                              num_envs: int, num_steps: int) -> dict[str, Any]:
    """Run the frozen Original Task protocol without parameter updates."""
    import jax
    import jax.numpy as jnp
    from craftax.craftax.constants import Achievement
    from craftax.craftax.craftax_state import EnvParams
    from dicode.task_utils import get_achievement_multi_hot
    from dicode.wrappers import BatchEnvWrapper
    from minicraftax.envs.craftax import CraftaxAugObsTrain

    eval_env = CraftaxAugObsTrain()
    relevant_achievements = tuple(eval_env.relevant_achievements)
    _require(bool(relevant_achievements),
             "Original Task relevant_achievements is empty")
    task_vector = get_achievement_multi_hot(relevant_achievements)
    task_embeddings = jnp.tile(
        jnp.asarray(task_vector[None, :], dtype=jnp.float32),
        (int(num_envs), 1),
    )
    base_env = CraftaxAugObsTrain(
        condition_on_task=True, conditioning_type="one_hot",
        embedding_size=len(task_vector), task_embeddings=task_embeddings)
    env_params = base_env.default_params.replace(max_timesteps=int(num_steps))
    env = BatchEnvWrapper(base_env, int(num_envs))

    rng, reset_rng = dicode_original_rng_prefix(int(seed))
    observations, env_state = env.reset(reset_rng, env_params)
    _require(tuple(np.shape(observations)) == (int(num_envs), 8335),
             f"Original Task observation shape {np.shape(observations)} != "
             f"({num_envs}, 8335)")
    memory = adapter.initial_memory(int(num_envs))
    rewards = np.zeros(int(num_envs), dtype=np.float64)
    lengths = np.zeros(int(num_envs), dtype=np.int64)
    finished = np.zeros(int(num_envs), dtype=np.bool_)
    achievement_flags = {
        achievement.name: np.zeros(int(num_envs), dtype=np.bool_)
        for achievement in Achievement
    }

    for _ in range(int(num_steps)):
        rng, action_rng, step_rng = dicode_next_step_keys(rng)
        actions, memory = dicode_slowgru_policy_step(
            adapter=adapter, params=params, observations=observations,
            memory=memory, action_rng=action_rng, num_envs=int(num_envs))
        observations, env_state, reward, new_done, info = env.step(
            step_rng, env_state, jnp.asarray(actions), env_params)
        reward_np = np.asarray(jax.device_get(reward), dtype=np.float64)
        done_np = np.asarray(jax.device_get(new_done), dtype=np.bool_)
        info_np = jax.tree_util.tree_map(
            lambda value: np.asarray(jax.device_get(value)), info)
        rewards, lengths, finished, achievement_flags = (
            accumulate_first_episode_step(
                returns=rewards, lengths=lengths, finished=finished,
                achievement_flags=achievement_flags, reward=reward_np,
                new_done=done_np, info=info_np,
            )
        )
        # The adapter consumes this previous-terminal mask on the next policy
        # call, resetting SlowGRU memory before any auto-reset episode step.
        memory = dict(memory)
        memory["true_done"] = done_np
        if bool(finished.all()):
            break

    return aggregate_world_metrics(
        returns=rewards, lengths=lengths, finished=finished,
        achievement_flags=achievement_flags, max_floor=None,
        death=None, timeout=None)


def atomic_write_new_json(path: str, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    _require(not target.exists(), f"output already exists: {path}")
    _require(target.parent.is_dir(),
             f"output parent directory does not exist: {target.parent}")
    temporary = target.with_name(target.name + f".tmp.{os.getpid()}")
    try:
        with open(temporary, "x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True,
                      ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
