"""CPU tests for the learnability-only fused preflight summary."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from dicode.skill_preflight.contract import PreflightOptimizationContractError
from dicode.skill_preflight.learnability_summary import (
    accumulate_learnability_counts,
    require_learnability_fused_contract,
)
from dicode.skill_preflight.preflight import route
from dicode.skill_preflight.preflight_route import preflight_route


ROOT = Path(__file__).parents[4]
RUN_DICODE = ROOT / "experiments" / "training" / "run_dicode.py"
PPO_TR = ROOT / "src" / "dicode" / "ppo_tr.py"
ONLINE_EVAL = ROOT / "src" / "dicode" / "evaluation" / "online_evaluation.py"
CONFIG = ROOT / "conf" / "config.yaml"


def _load_score_helper():
    """Load only run_dicode's pure helper, without importing its driver deps."""
    tree = ast.parse(RUN_DICODE.read_text(encoding="utf-8"))
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "_learnability_scores_from_counts"
    )
    namespace = {"PreflightOptimizationContractError": PreflightOptimizationContractError}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(RUN_DICODE), "exec"), namespace)
    return namespace[node.name]


def test_count_reducer_boundaries_and_dtype():
    finished, successes = accumulate_learnability_counts(
        jnp.zeros((3,), dtype=jnp.int32),
        jnp.zeros((3,), dtype=jnp.int32),
        jnp.array([0, 1, 1, 2, -1, 3]),
        jnp.array([True, True, False, True, True, True]),
        jnp.array([True, False, True, False, True, True]),
    )
    assert finished.dtype == jnp.int32
    assert successes.dtype == jnp.int32
    assert finished.tolist() == [1, 1, 1]
    assert successes.tolist() == [1, 0, 0]


def test_minimal_scores_zero_finished_and_priority_boundaries():
    convert = _load_score_helper()
    scores = convert([0, 4, 2], [0, 1, 2], 3)
    assert scores == {
        "0": {"sr": -1.0, "priority_score": 0.0},
        "1": {"sr": 0.25, "priority_score": 0.1875},
        "2": {"sr": 1.0, "priority_score": 0.0},
    }
    with pytest.raises(PreflightOptimizationContractError):
        convert([1], [1], 2)
    with pytest.raises(PreflightOptimizationContractError):
        convert([1], [2], 1)


def _toy_rollout(key, *, fused):
    """Tiny JAX rollout mirroring the production two-split step contract."""
    num_tasks = 3
    num_envs = 5
    initial = (
        jnp.arange(num_envs, dtype=jnp.int32),  # env state
        jnp.zeros((num_envs,), dtype=jnp.int32),  # memory
        jnp.int32(17),  # immutable train-state stand-in
        key,
    )

    if not fused:
        def legacy_step(carry, _):
            env_state, memory, train_state, rng = carry
            rng, action_rng = jax.random.split(rng)
            action = jax.random.randint(action_rng, (num_envs,), 0, 3)
            memory = memory + action
            rng, env_rng = jax.random.split(rng)
            noise = jax.random.randint(env_rng, (num_envs,), 0, 2)
            env_state = env_state + action + noise
            task_ids = jnp.arange(num_envs, dtype=jnp.int32) % num_tasks
            returned = (env_state % 2) == 0
            success = returned & ((env_state % 3) == 0)
            return (env_state, memory, train_state, rng), (
                action, task_ids, returned, success
            )

        final, trajectory = jax.lax.scan(legacy_step, initial, None, length=7)
        actions, task_ids, returned, success = trajectory
        finished = jnp.zeros((num_tasks,), dtype=jnp.int32)
        successes = jnp.zeros((num_tasks,), dtype=jnp.int32)
        finished, successes = accumulate_learnability_counts(
            finished, successes, task_ids.reshape(-1), returned.reshape(-1), success.reshape(-1)
        )
        return final, actions, finished, successes

    def fused_step(carry, _):
        env_state, memory, train_state, rng, finished, successes = carry
        rng, action_rng = jax.random.split(rng)
        action = jax.random.randint(action_rng, (num_envs,), 0, 3)
        memory = memory + action
        rng, env_rng = jax.random.split(rng)
        noise = jax.random.randint(env_rng, (num_envs,), 0, 2)
        env_state = env_state + action + noise
        task_ids = jnp.arange(num_envs, dtype=jnp.int32) % num_tasks
        returned = (env_state % 2) == 0
        success = returned & ((env_state % 3) == 0)
        finished, successes = accumulate_learnability_counts(
            finished, successes, task_ids, returned, success
        )
        return (env_state, memory, train_state, rng, finished, successes), action

    fused_initial = initial + (
        jnp.zeros((num_tasks,), dtype=jnp.int32),
        jnp.zeros((num_tasks,), dtype=jnp.int32),
    )
    final, actions = jax.lax.scan(fused_step, fused_initial, None, length=7)
    return final[:4], actions, final[4], final[5]


def test_small_jax_legacy_and_fused_counts_rng_env_state_memory_train_state_actions_match():
    key = jax.random.PRNGKey(123)
    legacy = jax.jit(lambda k: _toy_rollout(k, fused=False))(key)
    fused = jax.jit(lambda k: _toy_rollout(k, fused=True))(key)
    for old_leaf, new_leaf in zip(jax.tree_util.tree_leaves(legacy),
                                  jax.tree_util.tree_leaves(fused)):
        assert jnp.array_equal(old_leaf, new_leaf)


class _Archive:
    def __init__(self):
        self.events = []

    def update_node_learnability(self, task_id, value):
        self.events.append(("learnability", task_id, value))

    def update_node_status(self, task_id, value):
        self.events.append(("status", task_id, value))

    def set_task_active_status(self, task_id, value):
        self.events.append(("active", task_id, value))


def test_minimal_scores_preserve_route_and_archive_updates():
    convert = _load_score_helper()
    fused_scores = convert([10, 10, 0], [3, 10, 0], 3)
    legacy_scores = {
        "0": {"sr": 0.3, "other_historical_fields": object()},
        "1": {"sr": 1.0, "other_historical_fields": object()},
        "2": {"sr": -1.0, "other_historical_fields": object()},
    }
    old_archive, new_archive = _Archive(), _Archive()
    old_kept, new_kept = [], []
    ids = ["learnable", "easy", "unfinished"]
    preflight_route(legacy_scores, ids, old_kept, old_archive, route)
    preflight_route(fused_scores, ids, new_kept, new_archive, route)
    assert new_kept == old_kept
    assert new_archive.events == old_archive.events


def test_fused_contract_fails_closed_for_non_learnability():
    require_learnability_fused_contract("learnability")
    for score_function in ("pvl", "max_mc", "unknown"):
        with pytest.raises(PreflightOptimizationContractError):
            require_learnability_fused_contract(score_function)


def test_production_fast_path_has_no_large_scoring_payload_or_gae():
    source = PPO_TR.read_text(encoding="utf-8")
    fast = source.split("# [BC fast path]", 1)[1].split(
        "# --------------------------\n\t\t# The Evaluation Step", 1
    )[0]
    for forbidden in (
        "Transition(", ".log_prob(", "_calculate_gae(",
        '"advantages":', '"traj_batch":',
    ):
        assert forbidden not in fast
    assert '"finished_counts": finished_counts' in fast
    assert '"success_counts": success_counts' in fast
    assert "dtype=jnp.int32" in fast
    assert "length=NUM_UPDATES" in fast
    assert "config.validation.num_steps" in fast


def test_default_off_and_fail_closed_wiring_precedes_rollout():
    config = CONFIG.read_text(encoding="utf-8")
    assert "learnability_fused_preflight_summary: false" in config

    driver = RUN_DICODE.read_text(encoding="utf-8")
    guard = driver.index("require_learnability_fused_contract(")
    rollout = driver.index("_pf_raw = evaluate_new_tasks(")
    assert guard < rollout
    assert "_learnability_scores_from_counts" in driver

    online = ONLINE_EVAL.read_text(encoding="utf-8")
    assert "if not _fused_learnability:" in online
    assert 'return {"learnability_summary": summary}' in online
