import distrax
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax import struct
from flax.linen.initializers import constant, orthogonal
from dicode.network import ActorCriticTransformer, Transition
from dicode.wrappers import BatchEnvWrapper
from minicraftax.envs.craftax import CraftaxAugObsTrain
# --- 2. Transformer Network Class ---
# Imported from dicode.network


def make_evaluate(config, env, env_params):
	num_envs = config.evaluation.num_envs
	num_steps = config.evaluation.num_steps
	def evaluate(train_state, rng):
		action_dim = env.action_space(env_params).n  # v6 problem-2: width of the behaviour action histogram
		network = ActorCriticTransformer(
			action_dim=action_dim,
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

		# v6 §3.8 (c): the STATIC achievement-key order (jit-external; used to label the co-occurrence
		# arrays the jitted evaluate() returns). Same iteration order as the jitted loop over
		# final_stats, so column k of the returned cooc arrays corresponds to cooc_names[k].
		cooc_names_static = [
			k.split("/")[-1] for k in info_structure.keys() if "Achievements" in k
		]

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
		# v6 problem-2 (behaviour fingerprint): per-env action histogram over the episode, accumulated
		# only while the env is unfinished (active_mask), exactly like reward/length. Appended as the LAST
		# carry element so every existing final_carry[...] index below is unchanged.
		accumulated_action_hist = jnp.zeros((num_envs, action_dim), dtype=jnp.float32)

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
			rng,
			accumulated_action_hist,
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
				rng,
				acc_action_hist,
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

			# 2b. v6 problem-2: accumulate the action histogram, masked to unfinished envs (same active_mask
			# used for reward/length, so an env's counts freeze at its termination step). one_hot(action)
			# is [num_envs, action_dim]; multiply by the per-env active_mask column.
			action_onehot = jax.nn.one_hot(action, acc_action_hist.shape[1], dtype=jnp.float32)
			new_acc_action_hist = acc_action_hist + action_onehot * active_mask[:, None]

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
				rng,
				new_acc_action_hist,
			), _

		(final_carry), _ = jax.lax.scan(
			_env_step, init_runner_state, None, num_steps
		)

		# --- 5. Post-Processing ---
		final_raw_rewards = final_carry[9]
		final_raw_lengths = final_carry[10]
		final_stats = final_carry[11]
		finished_mask = final_carry[8]

		count_finished = finished_mask.sum()
		
		# Helper arrays where Unfinished = NaN (for min/max/median)
		rewards_for_stats = jnp.where(finished_mask, final_raw_rewards, jnp.nan)
		lengths_for_stats = jnp.where(finished_mask, final_raw_lengths, jnp.nan)
		# v6 problem-2: finished-episode lengths with unfinished ZEROED (not NaN), so multihotᵀ·length
		# sums only real winning episodes' steps without NaN contamination.
		lengths_for_behav = jnp.where(finished_mask, final_raw_lengths, 0.0).astype(jnp.float32)

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
		# v6 §3.8 (c): while aggregating per-skill SR, ALSO build a per-env achievement multi-hot so we
		# can compute co-occurrence — "in the FINISHED episodes where the student reached deep skill i,
		# which other skills j did it also reach?". We collect one per-env 0/1 column per achievement
		# (in the deterministic order we iterate final_stats), then form count[i] and cooc[i][j] as
		# fixed-shape arrays (jit-safe). These are handed off to the CPU-side CooccurrenceLog, which
		# accumulates them across sessions (tier3 successes are sparse). See auction/cooccurrence_log.py.
		cooc_cols = []  # each is a per-env 0/1 vector (achievement reached in a FINISHED episode)
		for key, val in final_stats.items():
			if "Achievements" in key:
				skill_name_raw = key.split("/")[-1]
				valid_stats = jnp.where(finished_mask, val, 0.0)
				mean_stat = jnp.where(count_finished > 0, valid_stats.sum() / count_finished, 0.0)
				metrics[f"skill_{skill_name_raw}"] = mean_stat
				# per-env reached-in-a-finished-episode indicator (val may be >1 if the env counts
				# repeats; clamp to 0/1 so co-occurrence is "did it happen", not "how many times").
				reached = jnp.where(finished_mask & (val > 0), 1.0, 0.0)
				cooc_cols.append(reached)

		# multihot: [num_envs, num_ach]; count[i] = sum_env multihot[:,i]; cooc[i,j] = multihotᵀ·multihot.
		# Both are jax arrays (jit-safe). Column order == cooc_names_static (returned by main). These
		# _cooc_* keys are POPPED by run_session_evaluation before wandb logging, so they never pollute
		# the scalar metric log.
		if cooc_cols:
			multihot = jnp.stack(cooc_cols, axis=1)             # [num_envs, num_ach]
			metrics["_cooc_count"] = multihot.sum(axis=0)        # [num_ach]
			metrics["_cooc_matrix"] = multihot.T @ multihot      # [num_ach, num_ach]
			# v6 §3.8 (c) SR guard (user 2026-07-05): the # of FINISHED episodes this session, i.e. the
			# SR denominator. Accumulated by CooccurrenceLog so the relative MIN_SR guard can read
			# count[deep]/total. POPPED alongside the other _cooc_* keys before wandb logging.
			metrics["_cooc_total"] = count_finished

			# v6 problem-2 (behaviour fingerprint): group the per-env action histogram + episode length by
			# WHICH deep achievement each finished episode reached. multihot[:,i] selects the envs that won
			# achievement i, so multihotᵀ · action_hist sums those winners' action histograms per skill,
			# and multihotᵀ · length sums their episode lengths. Column order == cooc_names_static (same as
			# _cooc_*); action column order == craftax Action enum (BehaviorFingerprintLog.ACTION_NAMES).
			# POPPED by run_session_evaluation before wandb logging (big arrays, not scalars).
			final_action_hist = final_carry[13]                          # [num_envs, action_dim]
			metrics["_behav_action"] = multihot.T @ final_action_hist    # [num_ach, action_dim]
			metrics["_behav_steps"] = multihot.T @ lengths_for_behav     # [num_ach]

		return metrics

	# Return the jittable evaluate() plus the STATIC co-occurrence column labels (jit-external), so the
	# caller can map the returned _cooc_count / _cooc_matrix columns back to achievement names.
	return evaluate, cooc_names_static


def main(config, rng, train_state=None, eval_embedding=None):
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
		max_timesteps=8192,
	)

	env = BatchEnvWrapper(env, num_envs=config.evaluation.num_envs)


	rng, _rng = jax.random.split(rng)
	evaluate_fn, cooc_names = make_evaluate(config, env, env_params)
	evaluate_jit = jax.jit(evaluate_fn)
	metrics = evaluate_jit(train_state, _rng)
	# v6 §3.8 (c): attach the STATIC co-occurrence column labels (python list, jit-external) so the
	# caller (run_session_evaluation) can feed _cooc_count / _cooc_matrix into CooccurrenceLog by name.
	if "_cooc_count" in metrics:
		metrics["_cooc_names"] = cooc_names
	# v6 problem-2: attach the STATIC action column labels for _behav_action. Rows of _behav_* use the
	# same cooc_names order (achievements); columns use craftax Action enum order. BehaviorFingerprintLog
	# remaps by these names, so a future action-enum reorder can't silently misalign the histogram.
	if "_behav_action" in metrics:
		from auction.behavior_fingerprint_log import ACTION_NAMES
		metrics["_behav_action_names"] = list(ACTION_NAMES)
		metrics["_behav_names"] = cooc_names  # row labels (achievements), same as cooc
	return metrics


if __name__ == "__main__":
	main()
