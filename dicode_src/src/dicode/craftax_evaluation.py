import distrax
import jax
import jax.numpy as jnp
import numpy as np
from collections import OrderedDict
import hashlib
import threading
from flax import linen as nn
from flax import struct
from flax.linen.initializers import constant, orthogonal
from dicode.network import ActorCriticTransformer, Transition
from dicode.wrappers import BatchEnvWrapper
from minicraftax.envs.craftax import CraftaxAugObsTrain
from dicode.necropsy import necro_init, necro_step
from dicode.runtime_analysis import tracker


_COMPILED_EVALUATOR_CACHE = OrderedDict()
_COMPILED_EVALUATOR_CACHE_LOCK = threading.RLock()


def clear_compiled_evaluator_cache():
	"""Clear the run-scoped evaluator cache (used by tests and new runs)."""
	with _COMPILED_EVALUATOR_CACHE_LOCK:
		_COMPILED_EVALUATOR_CACHE.clear()


def _get_cached_evaluator(key):
	with _COMPILED_EVALUATOR_CACHE_LOCK:
		if key not in _COMPILED_EVALUATOR_CACHE:
			return None
		value = _COMPILED_EVALUATOR_CACHE.pop(key)
		_COMPILED_EVALUATOR_CACHE[key] = value
		return value


def _put_cached_evaluator(key, value, max_entries=8):
	with _COMPILED_EVALUATOR_CACHE_LOCK:
		_COMPILED_EVALUATOR_CACHE[key] = value
		while len(_COMPILED_EVALUATOR_CACHE) > max(1, int(max_entries)):
			_COMPILED_EVALUATOR_CACHE.popitem(last=False)


def _pytree_signature(value):
	try:
		leaves, treedef = jax.tree_util.tree_flatten(value)
		return (str(treedef), tuple((tuple(getattr(x, "shape", ())), str(getattr(x, "dtype", type(x)))) for x in leaves))
	except Exception:
		return (type(value).__name__, repr(value))


def _get_or_compile_evaluator(key, jit_fn, args, enabled, max_entries=8):
	if enabled:
		cached = _get_cached_evaluator(key)
		if cached is not None:
			return cached, True
	if enabled:
		with tracker.span("eval_compile", cache_hit=False, task_signature=str(key)):
			compiled = jit_fn.lower(*args).compile()
	else:
		compiled = jit_fn
	if enabled:
		_put_cached_evaluator(key, compiled, max_entries)
	return compiled, False


def _cache_enabled(config):
	performance = config.get("performance", {}) if hasattr(config, "get") else {}
	return bool(performance.get("eval_compile_cache", False))


def _cache_key(config, eval_embedding, detail, input_shape, train_state=None, rng=None):
	training = config.training
	evaluation = config.evaluation
	static = tuple((name, str(getattr(training, name, None))) for name in (
		"activation", "hidden_layers", "embed_size", "num_heads", "qkv_features",
		"num_layers", "gating", "gating_bias", "condition_on_task", "conditioning_type",
		"window_mem",
	)) + tuple((name, str(getattr(evaluation, name, None))) for name in ("num_envs", "num_steps"))
	if hasattr(config, "eval"):
		static += (("max_timesteps", str(config.eval.get("max_timesteps", 8192))),)
	h = hashlib.sha256()
	if eval_embedding is not None:
		arr = np.asarray(eval_embedding)
		h.update(str(arr.shape).encode())
		h.update(str(arr.dtype).encode())
		h.update(arr.tobytes())
	return (static, tuple(input_shape or ()), getattr(training, "conditioning_type", None),
			bool(detail), h.hexdigest(), _pytree_signature(train_state), _pytree_signature(rng))
# --- 2. Transformer Network Class ---
# Imported from dicode.network


def make_evaluate(config, env, env_params, detail=False):
	# detail=True additionally returns per-env first-episode forensics in metrics['_details']
	# (return / length / finished / died / floor_at_done / max_floor) for failure autopsies.
	# Default False -> byte-identical to the original aggregate-only path.
	num_envs = config.evaluation.num_envs
	num_steps = config.evaluation.num_steps

	def _state_core(st):
		# walk wrapper nesting (0-2 levels) until the CraftaxState with player_level
		for _ in range(3):
			if hasattr(st, "player_level"):
				return st
			st = getattr(st, "env_state")
		raise AttributeError("player_level not found in env state pytree")
	def evaluate(train_state, rng):
		network = ActorCriticTransformer(
			action_dim=env.action_space(env_params).n,
			activation=config.training.activation,
			hidden_layers=config.training.hidden_layers,  # Mapping layer_size to hidden_layers
			encoder_size=config.training.embed_size,  # Mapping embedding_size to encoder_size/embed_size
			num_heads=config.training.num_heads,
			qkv_features=config.training.qkv_features,
			num_layers=config.training.num_layers,
			gating=config.training.gating,
			gating_bias=config.training.gating_bias,
		)

		rng, reset_rng = jax.random.split(rng)
		obsv, env_state = env.reset(reset_rng, env_params)

		# --- 3. Determine Info Structure (Crucial Step) ---
		# We need to know what 'info' looks like to create a zero-filled accumulator.
		# We use eval_shape to do this without actually running the computation.
		def get_info_shape():
			dummy_action = jnp.zeros((num_envs,), dtype=jnp.int32)
			_, _, _, _, info = env.step(reset_rng, env_state, dummy_action, env_params)
			return info
		
		info_structure = jax.eval_shape(get_info_shape)

		# Initialize accumulated_stats with zeros matching info_structure
		accumulated_stats = jax.tree.map(
			lambda x: jnp.zeros((num_envs,) + x.shape[1:], dtype=jnp.float32), 
			info_structure
		)

		# Transformer State Init
		memories = jnp.zeros(
			(
				num_envs,
				config.training.window_mem,
				config.training.num_layers,
				config.training.embed_size,
			)
		)
		memories_mask = jnp.zeros(
			(
				num_envs,
				config.training.num_heads,
				1,
				config.training.window_mem + 1,
			),
			dtype=jnp.bool_,
		)
		memories_mask_idx = jnp.zeros((num_envs,), dtype=jnp.int32) + (
			config.training.window_mem + 1
		)

		finished_mask = jnp.zeros((num_envs,), dtype=jnp.bool_)
		accumulated_reward = jnp.zeros((num_envs,), dtype=jnp.float32)
		accumulated_length = jnp.zeros((num_envs,), dtype=jnp.float32)
		done_prev = jnp.zeros((num_envs,), dtype=jnp.bool_)
		floor_at_done = jnp.zeros((num_envs,), dtype=jnp.int32)
		health_at_done = jnp.zeros((num_envs,), dtype=jnp.float32)
		max_floor = jnp.zeros((num_envs,), dtype=jnp.int32)
		necro = necro_init(num_envs, _state_core(env_state), detail)

		init_runner_state = (
			train_state,
			env_state,
			memories,
			memories_mask,
			memories_mask_idx,
			obsv,
			done_prev,
			0,
			finished_mask,
			accumulated_reward,
			accumulated_length,
			accumulated_stats,
			floor_at_done,
			health_at_done,
			max_floor,
			necro,
			rng,
		)

		# --------------------------
		# The Evaluation Step (NO UPDATE)
		# --------------------------

		def _env_step(carry, _):
			(
				train_state,
				env_state,
				memories,
				memories_mask,
				memories_mask_idx,
				last_obs,
				done,
				step_env_currentloop,
				finished_mask,
				acc_reward,
				acc_length,
				acc_stats,
				floor_at_done,
				health_at_done,
				max_floor,
				necro,
				rng,
			) = carry

			# ... (Copy logic from make_train's _env_step) ...
			memories_mask_idx = jnp.where(
				done,
				config.training.window_mem,
				jnp.clip(memories_mask_idx - 1, 0, config.training.window_mem),
			)
			memories_mask = jnp.where(
				done[:, None, None, None],
				jnp.zeros(
					(
						num_envs,
						config.training.num_heads,
						1,
						config.training.window_mem + 1,
					),
					dtype=jnp.bool_,
				),
				memories_mask,
			)
			memories_mask_idx_ohot = jax.nn.one_hot(
				memories_mask_idx, config.training.window_mem + 1
			)
			memories_mask_idx_ohot = memories_mask_idx_ohot[:, None, None, :].repeat(
				config.training.num_heads, 1
			)
			memories_mask = jnp.logical_or(memories_mask, memories_mask_idx_ohot)

			rng, _rng = jax.random.split(rng)
			# Note: model_forward_eval
			pi, _, memories_out = network.apply(
				train_state.params,
				memories,
				last_obs,
				memories_mask,
				method=network.model_forward_eval,
			)
			action = pi.sample(seed=_rng)
			# log_prob = pi.log_prob(action)

			memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(memories_out)

			rng, step_rng = jax.random.split(rng)
			next_obsv, next_env_state, reward, next_done, info = env.step(step_rng, env_state, action, env_params)

			# memory_indices = jnp.arange(0, config.training.window_mem)[
			# 	None, :
			# ] + step_env_currentloop * jnp.ones((config.evaluation.num_envs, 1), dtype=jnp.int32)

			# transition = Transition(
			# 	done,
			# 	action,
			# 	value,
			# 	reward,
			# 	log_prob,
			# 	memories_mask.squeeze(),
			# 	memory_indices,
			# 	last_obs,
			# 	info,
			# )

			# --- METRIC ACCUMULATION LOGIC ---
			
			# 1. Calculate Mask: We only accept data if the env is NOT finished yet.
			# Convert bool mask to float (0.0 or 1.0) for multiplication
			active_mask = (1.0 - finished_mask.astype(jnp.float32))
			
			# 2. Accumulate Reward
			new_acc_reward = acc_reward + (reward * active_mask)
			new_acc_length = acc_length + (1 * active_mask).astype(jnp.int32)

			# 3. Accumulate Info/Achievements
			# We iterate over every key in 'info' (Achievements, discount, etc.)
			# If info contains {k: v}, we do: acc[k] += v * active_mask
			# Since 'v' is already (v * done) inside the env, this effectively captures
			# the value at the moment of termination and ignores all subsequent zeros.
			def accumulate_leaf(acc, new_val):
				# Ensure dimensions match for broadcasting if necessary
				# Usually info values are (num_envs,), so direct mul is fine.
				return acc + (new_val * active_mask)
			
			new_acc_stats = jax.tree.map(accumulate_leaf, acc_stats, info)

			# 4. Update Finished Mask
			next_finished_mask = jnp.logical_or(finished_mask, next_done)

			# 4b. [detail] first-episode forensics
			core = _state_core(next_env_state)
			lvl = core.player_level.astype(jnp.int32)
			hp = core.player_health.astype(jnp.float32)
			first_done_now = jnp.logical_and(next_done, jnp.logical_not(finished_mask))
			floor_at_done = jnp.where(first_done_now, lvl, floor_at_done)
			health_at_done = jnp.where(first_done_now, hp, health_at_done)
			max_floor = jnp.where(finished_mask, max_floor, jnp.maximum(max_floor, lvl))
			necro = necro_step(necro, _state_core(env_state), core, active_mask, first_done_now, detail)

			return (
				train_state,
				next_env_state,
				memories,
				memories_mask,
				memories_mask_idx,
				next_obsv,
				next_done,
				step_env_currentloop + 1,
				next_finished_mask,
				new_acc_reward,
				new_acc_length,
				new_acc_stats,
				floor_at_done,
				health_at_done,
				max_floor,
				necro,
				rng,
			), _

		(final_carry), _ = jax.lax.scan(
			_env_step, init_runner_state, None, num_steps
		)

		# --- 5. Post-Processing ---
		final_raw_rewards = final_carry[9]
		final_raw_lengths = final_carry[10]
		final_stats = final_carry[11]
		finished_mask = final_carry[8]
		_floor_at_done = final_carry[12]
		_health_at_done = final_carry[13]
		_max_floor = final_carry[14]
		_necro = final_carry[15]

		count_finished = finished_mask.sum()
		
		# Helper arrays where Unfinished = NaN (for min/max/median)
		rewards_for_stats = jnp.where(finished_mask, final_raw_rewards, jnp.nan)
		lengths_for_stats = jnp.where(finished_mask, final_raw_lengths, jnp.nan)

		# --- A. Basic Statistics ---
		def get_stats(data_array, name):
			# Safe aggregation handles NaNs automatically
			return {
				f"{name}": jnp.where(count_finished > 0, jnp.nanmean(data_array), -jnp.inf),
			}


		metrics = {}
		metrics.update(get_stats(rewards_for_stats, "mean_return"))
		metrics["mean_performance"] = metrics["mean_return"] / 226.0 * 100.0 
		metrics.update(get_stats(lengths_for_stats, "average_episode_length"))
		# --- C. Achievements ---
		for key, val in final_stats.items():
			if "Achievements" in key:
				skill_name_raw = key.split("/")[-1]
				valid_stats = jnp.where(finished_mask, val, 0.0)
				mean_stat = jnp.where(count_finished > 0, valid_stats.sum() / count_finished, 0.0)
				metrics[f"skill_{skill_name_raw}"] = mean_stat

		if detail:
			metrics["_details"] = {
				"return": final_raw_rewards, "length": final_raw_lengths,
				"finished": finished_mask, "died": _health_at_done <= 0,
				"floor_at_done": _floor_at_done, "max_floor": _max_floor, **_necro,
			}
		return metrics

	return evaluate


def main(config, rng, train_state=None, eval_embedding=None, detail=False):
	# 1. Create the base environment
	if eval_embedding is not None:
		embedding_size = eval_embedding.shape[1]
		env = CraftaxAugObsTrain(
			condition_on_task=config.training.condition_on_task,
			conditioning_type="embedding",
			embedding_size=embedding_size,
			task_embeddings=eval_embedding,
		)
	else:
		env = CraftaxAugObsTrain()

	env_params = env.default_params.replace(
		max_timesteps=int(config.eval.get("max_timesteps", 8192)) if hasattr(config, "eval") else 8192,
	)

	env = BatchEnvWrapper(env, num_envs=config.evaluation.num_envs)


	rng, _rng = jax.random.split(rng)
	evaluate_fn = make_evaluate(config, env, env_params, detail=detail)
	use_cache = _cache_enabled(config)
	key = _cache_key(config, eval_embedding, detail,
					(getattr(eval_embedding, "shape", None) or (config.evaluation.num_envs,)), train_state, _rng)
	evaluate_jit = jax.jit(evaluate_fn)
	max_entries = int((config.get("performance", {}) if hasattr(config, "get") else {}).get("compiled_cache_max_entries", 8))
	evaluate_compiled, cache_hit = _get_or_compile_evaluator(
		key, evaluate_jit, (train_state, _rng), use_cache, max_entries)
	with tracker.span("eval_execute", cache_hit=cache_hit, task_signature=str(key)):
		metrics = evaluate_compiled(train_state, _rng)
		if tracker.enabled:
			for leaf in jax.tree_util.tree_leaves(metrics):
				if hasattr(leaf, "block_until_ready"):
					leaf.block_until_ready()
	return metrics


if __name__ == "__main__":
	main()
