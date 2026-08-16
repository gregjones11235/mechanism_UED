from pathlib import Path

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
ppo_tr = pytest.importorskip("dicode.ppo_tr")
scoring = pytest.importorskip("dicode.scoring")
omegaconf = pytest.importorskip("omegaconf")


def _info(T=4, B=2):
	info = {
		"task_id": jnp.tile(jnp.arange(B, dtype=jnp.int32), (T, 1)),
		"returned_episode": jnp.ones((T, B), dtype=jnp.bool_),
		"is_success": jnp.ones((T, B), dtype=jnp.bool_),
		"returned_episode_lengths": jnp.ones((T, B), dtype=jnp.int32),
		"returned_episode_returns": jnp.ones((T, B), dtype=jnp.float32),
		"discount": jnp.ones((T, B), dtype=jnp.float32),
		"rmt_state": jnp.ones((T, B, 3), dtype=jnp.float32),
	}
	for name in scoring.get_achievement_names():
		info[f"Achievements/{name}"] = jnp.ones((T, B), dtype=jnp.float32)
	return info


def _transition(T=4, B=2):
	from dicode.network import Transition

	shape = (T, B)
	return Transition(
		done=jnp.asarray([[False, False], [False, True], [False, False], [True, True]]),
		action=jnp.zeros(shape, dtype=jnp.int32),
		value=jnp.ones(shape, dtype=jnp.float32),
		reward=jnp.ones(shape, dtype=jnp.float32),
		log_prob=jnp.zeros(shape, dtype=jnp.float32),
		memories_mask=jnp.zeros((T, B, 2), dtype=jnp.bool_),
		memories_indices=jnp.zeros((T, B, 2), dtype=jnp.int32),
		obs=jnp.zeros((T, B, 3), dtype=jnp.float32),
		info=_info(T, B),
	)


@pytest.mark.parametrize(
	("score_function", "expected_top", "expected_transition"),
	[
		("learnability", set(), {"done": False, "reward": False, "value": False}),
		("pvl", {"advantages"}, {"done": False, "reward": False, "value": False}),
		("max_mc", set(), {"done": False, "reward": True, "value": True}),
	],
)
def test_compact_payload_has_exact_static_fields(score_function, expected_top, expected_transition):
	traj = _transition()
	assert not np.array_equal(np.asarray(traj.done), np.asarray(traj.info["returned_episode"]))
	data = ppo_tr._build_scoring_data(
		traj,
		jnp.ones((4, 2), dtype=jnp.float32),
		score_function,
		compact=True,
	)
	assert set(data) == {"traj_batch", *expected_top}
	compact = data["traj_batch"]
	assert compact.done is not None if expected_transition["done"] else compact.done is None
	assert compact.reward is not None if expected_transition["reward"] else compact.reward is None
	assert compact.value is not None if expected_transition["value"] else compact.value is None
	assert compact.obs is None
	assert compact.action is None
	assert compact.log_prob is None
	assert compact.memories_mask is None
	assert compact.memories_indices is None
	assert set(compact.info) == {
		"task_id",
		"returned_episode",
		"is_success",
		"returned_episode_lengths",
		"returned_episode_returns",
		*[key for key in _info() if key.startswith("Achievements/")],
	}


def test_compact_and_full_scores_are_equivalent_for_all_modes():
	traj = _transition()
	num_tasks = 2
	mask = np.ones((num_tasks, len(scoring.get_achievement_names())), dtype=bool)
	completed = np.zeros_like(mask)
	for score_function in ("learnability", "pvl", "max_mc"):
		config = omegaconf.OmegaConf.create(
			{
				"dicode_manager": {"score_function": score_function, "mode": "task"}
			}
		)
		advantages = jnp.ones((4, 2), dtype=jnp.float32)
		full = ppo_tr._build_scoring_data(traj, advantages, score_function, compact=False)
		compact = ppo_tr._build_scoring_data(traj, advantages, score_function, compact=True)
		full_scores = scoring.calculate_scores_from_snapshot(
			full, num_tasks, mask, completed, config
		)
		compact_scores = scoring.calculate_scores_from_snapshot(
			compact, num_tasks, mask, completed, config
		)
		assert full_scores.keys() == compact_scores.keys()
		for task_id in full_scores:
			full_metrics = full_scores[task_id]
			compact_metrics = compact_scores[task_id]
			assert full_metrics["priority_score"] == compact_metrics["priority_score"]
			assert full_metrics["sr"] == compact_metrics["sr"]
			assert full_metrics["achievement_srs"] == compact_metrics["achievement_srs"]
			assert full_metrics["average_episode_length"] == compact_metrics[
				"average_episode_length"
			]
			assert full_metrics["mean_return"] == compact_metrics["mean_return"]


def test_unknown_score_function_fails_closed():
	with pytest.raises(ValueError, match="Unknown score_function"):
		ppo_tr._compact_scoring_fields("future_metric")


def test_none_safe_flatten_and_default_configuration():
	assert ppo_tr._flatten_scoring_leaf(None, 2) is None
	arr = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)
	assert ppo_tr._flatten_scoring_leaf(arr, 2).shape == (6, 4)
	package_root = Path(__file__).parents[4]
	config_text = (package_root / "conf/training/default.yaml").read_text(encoding="utf-8")
	source_text = (package_root / "src/dicode/ppo_tr.py").read_text(encoding="utf-8")
	assert "compact_scoring_payload: false" in config_text
	assert "score_function: ${dicode_manager.score_function}" in config_text
	assert "_scan_with_retained_suffix" not in source_text
	assert "compact_score_function = config.score_function" in source_text
	assert "jax.lax.scan(\n\t\t\t_update_step" in source_text


def test_single_scan_rng_carry_is_bitwise_equal_under_jit():
	def step(carry, _):
		x, rng = carry
		rng, sample = jax.random.split(rng)
		x = x + jax.random.normal(sample, ())
		return (x, rng), {"x": x, "unused": jnp.square(x)}

	init = (jnp.asarray(0.0, dtype=jnp.float32), jax.random.PRNGKey(3))
	full = jax.jit(lambda c: jax.lax.scan(step, c, None, length=7))(init)
	def compact_step(carry, i):
		carry, output = step(carry, i)
		return carry, {"x": output["x"]}

	compact = jax.jit(
		lambda c: jax.lax.scan(compact_step, c, None, length=7)
	)(init)
	assert jax.tree_util.tree_all(
		jax.tree_util.tree_map(jnp.array_equal, full[0], compact[0])
	)
	assert jnp.array_equal(full[1]["x"], compact[1]["x"])
