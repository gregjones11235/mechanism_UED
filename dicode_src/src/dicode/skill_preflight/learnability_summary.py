"""Contracts and JAX reducers for the fused learnability preflight path."""

from __future__ import annotations

import jax.numpy as jnp

from dicode.skill_preflight.contract import PreflightOptimizationContractError


def require_learnability_fused_contract(score_function: str) -> None:
    """Fail closed unless the fused summary is used with learnability.

    PVL and MaxMC need trajectory/value information that the fused path
    deliberately does not produce.  This check must run before rollout
    construction so an invalid configuration cannot silently keep all tasks.
    """
    if score_function != "learnability":
        raise PreflightOptimizationContractError(
            "performance.learnability_fused_preflight_summary requires "
            f"dicode_manager.score_function=learnability, got {score_function!r}"
        )


def accumulate_learnability_counts(
    finished_counts,
    success_counts,
    task_ids,
    returned_episode,
    is_success,
):
    """Accumulate exact per-task episode and success counts in int32.

    Invalid task ids are ignored, matching the legacy scorer's equality mask
    over the known ``range(num_tasks)``. A success is counted only on a
    returned episode, exactly as ``is_success & task_done_mask`` did.
    """
    num_tasks = finished_counts.shape[0]
    task_ids = jnp.asarray(task_ids, dtype=jnp.int32)
    returned_episode = jnp.asarray(returned_episode, dtype=jnp.bool_)
    is_success = jnp.asarray(is_success, dtype=jnp.bool_)

    valid = (task_ids >= 0) & (task_ids < num_tasks)
    safe_task_ids = jnp.clip(task_ids, 0, max(num_tasks - 1, 0))
    finished = (valid & returned_episode).astype(jnp.int32)
    successes = (valid & returned_episode & is_success).astype(jnp.int32)
    return (
        finished_counts.at[safe_task_ids].add(finished),
        success_counts.at[safe_task_ids].add(successes),
    )
