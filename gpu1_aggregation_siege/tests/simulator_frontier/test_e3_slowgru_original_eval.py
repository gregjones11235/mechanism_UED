from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "src" / "dicode" / \
    "simulator_frontier" / "e3_slowgru_original_eval.py"
SPEC = importlib.util.spec_from_file_location("e3_slowgru_original_eval_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
E3SlowGRUOriginalEvalError = MODULE.E3SlowGRUOriginalEvalError
POLICY = MODULE.POLICY
aggregate_world_metrics = MODULE.aggregate_world_metrics
accumulate_first_episode_step = MODULE.accumulate_first_episode_step
atomic_write_new_json = MODULE.atomic_write_new_json
dicode_next_step_keys = MODULE.dicode_next_step_keys
dicode_original_rng_prefix = MODULE.dicode_original_rng_prefix
dicode_slowgru_policy_step = MODULE.dicode_slowgru_policy_step
original_task_protocol = MODULE.original_task_protocol


def test_per_world_aggregation_and_missing_metrics_are_honest():
    metrics = aggregate_world_metrics(
        returns=[1.0, 3.0, 8.0], lengths=[4, 6, 8],
        finished=[True, True, False],
        achievement_flags={"COLLECT_WOOD": [True, False, True]},
        max_floor=None, death=None, timeout=None)
    assert metrics["mean_return"] == 2.0
    assert metrics["median_return"] == 2.0
    assert metrics["mean_episode_length"] == 5.0
    assert metrics["unfinished_count"] == 1
    assert metrics["achievements"]["COLLECT_WOOD"]["rate"] == pytest.approx(1 / 2)
    assert metrics["skill_rates"]["skill_COLLECT_WOOD"] == pytest.approx(1 / 2)
    for key in ("death", "timeout", "max_floor"):
        assert metrics[key]["value"] is None
        assert metrics[key]["per_world"] is None
        assert metrics[key]["reason"]


def test_task_success_is_not_invented():
    metrics = aggregate_world_metrics(
        returns=[0, 0], lengths=[1, 1], finished=[True, True],
        achievement_flags={"A": [True, True], "B": [False, True]},
        max_floor=[0, 1])
    assert metrics["task_success"]["value"] is None
    assert metrics["task_success"]["per_world"] is None
    assert "no authoritative" in metrics["task_success"]["reason"]


def test_zero_finished_fails_closed_for_json_safe_statistics():
    with pytest.raises(E3SlowGRUOriginalEvalError, match="finished zero worlds"):
        aggregate_world_metrics(
            returns=[1.0, 2.0], lengths=[3, 3], finished=[False, False],
            achievement_flags={"A": [True, False]}, max_floor=None)


def test_terminal_info_captures_achievement_even_if_next_state_reset():
    # The next EnvState is deliberately absent: terminal achievements must be
    # read from step info, which survives an auto-reset state replacement.
    rewards, lengths, finished, flags = accumulate_first_episode_step(
        returns=[0.0], lengths=[0], finished=[False],
        achievement_flags={"COLLECT_WOOD": [False]}, reward=[2.5],
        new_done=[True], info={"Achievements/collect_wood": [1.0]},
    )
    assert rewards.tolist() == [2.5]
    assert lengths.tolist() == [1]
    assert finished.tolist() == [True]
    assert flags["COLLECT_WOOD"].tolist() == [True]

    # Any later auto-reset episode data is masked out.
    rewards, lengths, finished, flags = accumulate_first_episode_step(
        returns=rewards, lengths=lengths, finished=finished,
        achievement_flags=flags, reward=[99.0], new_done=[False],
        info={"Achievements/collect_wood": [0.0]},
    )
    assert rewards.tolist() == [2.5]
    assert lengths.tolist() == [1]
    assert flags["COLLECT_WOOD"].tolist() == [True]


def test_old_uppercase_info_key_is_rejected_without_fallback():
    with pytest.raises(E3SlowGRUOriginalEvalError,
                       match="Achievements/collect_wood"):
        accumulate_first_episode_step(
            returns=[0.0], lengths=[0], finished=[False],
            achievement_flags={"COLLECT_WOOD": [False]}, reward=[1.0],
            new_done=[True], info={"Achievements/COLLECT_WOOD": [1.0]},
        )


def test_atomic_output_refuses_existing(tmp_path):
    out = tmp_path / "result.json"
    atomic_write_new_json(str(out), {"ok": True})
    assert json.loads(out.read_text())["ok"] is True
    with pytest.raises(E3SlowGRUOriginalEvalError, match="already exists"):
        atomic_write_new_json(str(out), {"ok": False})


def test_protocol_matches_existing_dicode_original_task_contract():
    protocol = original_task_protocol(seed=42, num_envs=1024, num_steps=8192)
    assert protocol["seed"] == 42
    assert protocol["num_worlds"] == 1024
    assert protocol["horizon"] == 8192
    assert protocol["max_timesteps"] == 8192
    assert protocol["policy"] == POLICY
    assert protocol["achievement_schema"] == "craftax.craftax.constants.Achievement"


@pytest.mark.parametrize("kwargs", [
    {"seed": -1, "num_envs": 1, "num_steps": 1},
    {"seed": 42, "num_envs": 0, "num_steps": 1},
    {"seed": 42, "num_envs": 1, "num_steps": 0},
])
def test_protocol_rejects_invalid_dimensions(kwargs):
    with pytest.raises(E3SlowGRUOriginalEvalError):
        original_task_protocol(**kwargs)


def test_policy_forward_is_deterministic_but_sampling_matches_dicode_jax():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    logits = np.asarray([[1.0, 2.0, -3.0], [0.5, -0.5, 4.0]],
                        dtype=np.float32)
    forwarded_memory = {"carry": object()}

    class FakeAdapter:
        def __init__(self):
            self.call = None

        def policy_step(self, *args):
            self.call = args
            return {
                # Deliberately wrong adapter action: the evaluator must ignore
                # it and sample the full logits batch with the DiCode key.
                "action": np.asarray([2, 2], dtype=np.int32),
                "logits": logits,
                "new_memory": forwarded_memory,
            }

    adapter = FakeAdapter()
    params = {"w": np.ones((1,), dtype=np.float32)}
    observations = np.zeros((2, 8335), dtype=np.float32)
    memory = {"old": True}
    key = jax.random.PRNGKey(17)
    actions, actual_memory = dicode_slowgru_policy_step(
        adapter=adapter, params=params, observations=observations,
        memory=memory, action_rng=key, num_envs=2)

    assert adapter.call[0] is params
    assert adapter.call[1] is observations
    assert adapter.call[2] is memory
    assert adapter.call[3:] == (None, None, None, True)
    expected = np.asarray(jax.random.categorical(
        key, jnp.asarray(logits), axis=-1), dtype=np.int32)
    assert actions.shape == (2,)
    assert np.array_equal(actions, expected)
    assert actual_memory is forwarded_memory


@pytest.mark.parametrize("logits, match", [
    (None, "missing logits"),
    (np.zeros((2,), dtype=np.float32), "rank-2"),
    (np.zeros((3, 4), dtype=np.float32), "batch"),
    (np.asarray([[0.0, np.nan], [1.0, 0.0]], dtype=np.float32), "finite"),
])
def test_policy_sampling_rejects_missing_bad_shape_or_nonfinite_logits(
        logits, match):
    jax = pytest.importorskip("jax")

    class FakeAdapter:
        def policy_step(self, *args):
            result = {"memory": {"carry": True}}
            if logits is not None:
                result["logits"] = logits
            return result

    with pytest.raises(E3SlowGRUOriginalEvalError, match=match):
        dicode_slowgru_policy_step(
            adapter=FakeAdapter(), params={},
            observations=np.zeros((2, 8335), dtype=np.float32), memory={},
            action_rng=jax.random.PRNGKey(1), num_envs=2)


def test_original_task_seed_schedule_matches_dicode_split_prefix():
    jax = pytest.importorskip("jax")
    root = jax.random.PRNGKey(42)
    _, evaluator_rng = jax.random.split(root)       # online_evaluation.py
    _, evaluate_rng = jax.random.split(evaluator_rng)  # craftax_evaluation.main
    expected_rng, expected_reset = jax.random.split(evaluate_rng)  # make_evaluate
    actual_rng, actual_reset = dicode_original_rng_prefix(42)
    assert np.array_equal(actual_rng, expected_rng)
    assert np.array_equal(actual_reset, expected_reset)

    expected_rng, expected_action = jax.random.split(expected_rng)
    expected_rng, expected_step = jax.random.split(expected_rng)
    actual_rng, actual_action, actual_step = dicode_next_step_keys(actual_rng)
    assert np.array_equal(actual_rng, expected_rng)
    assert np.array_equal(actual_action, expected_action)
    assert np.array_equal(actual_step, expected_step)


def test_source_uses_authoritative_relevant_achievements_and_info():
    text = MODULE_PATH.read_text(encoding="utf-8")
    assert "eval_env.relevant_achievements" in text
    assert "get_achievement_multi_hot(relevant_achievements)" in text
    assert "list(Achievement)" not in text
    assert 'f"Achievements/{name.lower()}"' in text
    assert 'f"Achievements/{name}"' not in text
    assert "env_state.achievements" not in text
