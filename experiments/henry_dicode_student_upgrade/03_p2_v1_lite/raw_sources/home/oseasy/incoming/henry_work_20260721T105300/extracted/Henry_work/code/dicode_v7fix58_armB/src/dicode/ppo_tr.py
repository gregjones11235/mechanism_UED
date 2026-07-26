import time

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from flax import struct
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

from dicode.network import ActorCriticTransformer, Transition
from dicode.train_guard import record_update, reset_session_stats
from dicode import sil as sil_lib

from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv, MultiTaskMiniCraftaxEnvR
from dicode.wrappers_cl import (
	DistributedMultiTaskOptimisticLogWrapper,
	MultiTaskOptimisticLogWrapperAllTasks,
)


# --- 2. Transformer Network Class ---
# --- 2. Transformer Network Class ---
# Imported from dicode.network


def _cfg_get(config, key, default):
	"""Config lookup tolerant of both OmegaConf DictConfig and plain namespaces."""
	try:
		value = config.get(key, default)
	except AttributeError:
		value = getattr(config, key, default)
	return default if value is None else value


# --- Helper Functions for Transformer Logic ---
indices_select = lambda x, y: x[y]
batch_indices_select = jax.vmap(indices_select)
roll_vmap = jax.vmap(jnp.roll, in_axes=(-2, 0, None), out_axes=-2)
batchify = lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1],) + x.shape[2:])


def make_train(
	config,
	task_classes,
	num_training_updates,
	task_embeddings=None,
	task_distribution_proportions=None,
	initial_global_update_step=0,
):
	"""Sets up the environment, network, and returns the JIT-compiled train function."""
	# --- Environment Setup (IDENTICAL TO OLD CODE) ---
	NUM_UPDATES = num_training_updates
	num_tasks = len(task_classes)

	# v7fix4.8 guardrail: GAE targets are clipped to the physically achievable return range.
	# Craftax total return is bounded (~226 human max; observed best ~46), but the raw target
	# advantages+value is UNBOUNDED — with gamma=0.999 a local value over-estimate bootstraps
	# itself into a runaway (fast arm banner s191: value_loss 0.4 -> 1e6+, policy destroyed
	# through the shared trunk). Clipping is a strict no-op on healthy training (targets stay
	# >=5x inside these bounds) and removes the runaway's mathematical precondition.
	vt_clip_min = float(_cfg_get(config, "value_target_clip_min", -50.0))
	vt_clip_max = float(_cfg_get(config, "value_target_clip_max", 300.0))

	# v7fix56 P1' SIL (self-imitation, design doc §2.2). Static config — when sil=false
	# (the default) NONE of the branches below trace, so the compiled graph is the
	# status-quo one (S4 no-op discipline, same as the guard/clip precedents).
	sil_on = bool(_cfg_get(config, "sil", False))
	sil_pools = [str(w) for w in (_cfg_get(config, "sil_pools", []) or [])]
	sil_on = sil_on and len(sil_pools) > 0
	if sil_on:
		sil_slots = int(_cfg_get(config, "sil_pool_slots", 48))
		sil_windows = int(_cfg_get(config, "sil_windows_per_update", 8))
		sil_writes = int(_cfg_get(config, "sil_writes_per_update", 4))
		sil_stale_upd = int(_cfg_get(config, "sil_staleness_sessions", 5)) * int(
			config.max_updates_per_session
		)
		sil_vf_coef = float(_cfg_get(config, "sil_vf_coef", 0.01))
		sil_prioritized = bool(_cfg_get(config, "sil_prioritized", True))

	static_env_params = StaticEnvParams()
	env_params = EnvParams(max_timesteps=4096)

	if config.mode != "reward":
		if task_embeddings is not None:
			embedding_size = task_embeddings.shape[1]
			print(f"Using embedding size: {embedding_size}")
			base_env = MultiTaskMiniCraftaxEnv(
				task_classes,
				static_env_params,
				env_params,
				config.condition_on_task,
				conditioning_type="embedding",
				embedding_size=embedding_size,
				completion_bonus_scale=config.completion_bonus_scale,
				completion_bonus_min=config.completion_bonus_min,
				bonus_type=config.bonus_type,
				dynamic_bonus_k=config.dynamic_bonus_k,
			)
		else:
			base_env = MultiTaskMiniCraftaxEnv(
				task_classes,
				static_env_params,
				env_params,
				config.condition_on_task,
				completion_bonus_scale=config.completion_bonus_scale,
				completion_bonus_min=config.completion_bonus_min,
				bonus_type=config.bonus_type,
				dynamic_bonus_k=config.dynamic_bonus_k,
			)
	else:
		if task_embeddings is not None:
			embedding_size = task_embeddings.shape[1]
			base_env = MultiTaskMiniCraftaxEnvR(
				task_classes,
				static_env_params,
				env_params,
				config.condition_on_task,
				conditioning_type="embedding",
				embedding_size=embedding_size,
				completion_bonus_scale=config.completion_bonus_scale,
				completion_bonus_min=config.completion_bonus_min,
			)
		else:
			base_env = MultiTaskMiniCraftaxEnvR(
				task_classes,
				static_env_params,
				env_params,
				config.condition_on_task,
				completion_bonus_scale=config.completion_bonus_scale,
				completion_bonus_min=config.completion_bonus_min,
			)

	if task_distribution_proportions is None:
		# Default to uniform distribution if not provided
		task_distribution_proportions = jnp.ones(num_tasks) / num_tasks

	env = DistributedMultiTaskOptimisticLogWrapper(
		base_env,
		jax.random.PRNGKey(0),  # We need a key for the permutation in the wrapper
		config.num_envs,
		num_tasks,
		config.optimistic_reset_ratio,
		task_distribution_proportions,
		task_embeddings,
	)
	env_params = env.default_params

	# --- Network Setup (CHANGED TO TRANSFORMER) ---
	# NOTE: You must ensure your config has these keys (embed_size, num_heads, etc.)
	# We map existing config keys where possible, or expect new ones.
	network = ActorCriticTransformer(
		action_dim=env.action_space(env_params).n,
		activation=config.activation,
		hidden_layers=config.hidden_layers,  # Mapping layer_size to hidden_layers
		encoder_size=config.embed_size,  # Mapping embedding_size to encoder_size/embed_size
		num_heads=config.num_heads,
		qkv_features=config.qkv_features,
		num_layers=config.num_layers,
		gating=config.gating,
		gating_bias=config.gating_bias,
	)

	# --- Optimizer Setup ---
	TOTAL_GLOBAL_UPDATES = (
		(
			config.total_timesteps
			// config.num_envs
			// config.num_steps
			// config.max_updates_per_session
		)
		+ 1
	) * config.max_updates_per_session

	def linear_schedule(count):
		u = count // (config.num_minibatches * config.update_epochs)
		frac = 1.0 - u / TOTAL_GLOBAL_UPDATES
		# v7fix4.8 ROOT-CAUSE FIX: past the nominal horizon (TOTAL_GLOBAL_UPDATES=15400 under the
		# paper config) frac goes NEGATIVE and the LR crosses zero at update ~15556 — from there
		# every step is gradient ASCENT (maximize loss, actively minimize entropy). This is the
		# deterministic root cause of the fast-arm s191 collapse (predicted zero-crossing 15556,
		# observed runaway onset 15583); only runs that outlive the horizon are exposed. Clamp so
		# training past the horizon proceeds at min_lr, as the schedule always intended.
		frac = jnp.maximum(frac, 0.0)
		lr = config.min_lr + (config.lr - config.min_lr) * frac
		# v7fix4.9 SECOND ANNEAL LEG (post-horizon re-anneal, 2026-07-14). min_lr=2e-6 (1% of
		# base) proved too small to ABSORB LEVEL-POOL ROTATIONS: the s195 relay-level influx
		# ignited a session-long value flare (v_loss 0.2 -> 107) whose gradient monopolized the
		# clipped budget and starved the actor. lr_restart=3e-5 is evidence-anchored — inside
		# BOTH the documented climb band (s117-151 ran at 4.8e-5 -> 5.9e-6) and the
		# flare-self-heal band (four flares healed in-session at ~2.3-3.6e-5). The leg warms up
		# from min_lr over lr_restart_warmup updates (no 15x LR step onto residual value misfit)
		# and decays linearly back to min_lr at lr_restart_horizon, so training never LIVES at
		# the floor — it only touches it when the run is meant to end. lr_restart=0 disables.
		if (_cfg_get(config, "lr_restart", 0.0) or 0.0) > 0.0:
			span = config.lr_restart_horizon - config.lr_restart_at
			frac2 = jnp.clip((config.lr_restart_horizon - u) / span, 0.0, 1.0)
			leg2 = config.min_lr + (config.lr_restart - config.min_lr) * frac2
			warm = jnp.clip((u - config.lr_restart_at) / config.lr_restart_warmup, 0.0, 1.0)
			leg2 = config.min_lr + (leg2 - config.min_lr) * warm
			lr = jnp.where(u >= config.lr_restart_at, leg2, lr)
		return lr

	if config.anneal_lr:
		tx = optax.chain(
			optax.clip_by_global_norm(config.max_grad_norm),
			optax.adam(learning_rate=linear_schedule, eps=1e-5),
		)
	else:
		tx = optax.chain(
			optax.clip_by_global_norm(config.max_grad_norm),
			optax.adam(config.lr, eps=1e-5),
		)

	def train(rng, train_state=None, current_original_return=0.0, sil_state=None, sil_beta=0.0):
		"""The core JIT-compiled function."""
		obs_dim = env.observation_space(env_params).shape[0]

		# v7fix56 P1' SIL: cold-start the ring buffers on first use (resume keeps them
		# only within a process; across job restarts the pool restarts empty — accepted
		# in the design doc §2.2 "首版空池冷启动").
		if sil_on and sil_state is None:
			sil_state = sil_lib.init_sil_state(
				len(sil_pools), sil_slots, config.num_steps, obs_dim,
				config.window_mem, config.num_layers, config.embed_size, config.num_heads,
			)

		# --- Initialization ---
		if train_state is None:
			rng, _rng = jax.random.split(rng)

			# Transformer Init Shapes
			init_obs = jnp.zeros((2, obs_dim))
			init_memory = jnp.zeros((2, config.window_mem, config.num_layers, config.embed_size))
			init_mask = jnp.zeros((2, config.num_heads, 1, config.window_mem + 1), dtype=jnp.bool_)

			network_params = network.init(_rng, init_memory, init_obs, init_mask)

			train_state = TrainState.create(
				apply_fn=network.apply,
				params=network_params,
				tx=tx,
			)

		rng, _rng = jax.random.split(rng)
		obsv, env_state = env.reset(_rng, env_params)
		env_state = env_state.replace(
			running_original_return=jnp.full(
				(config.num_envs,), current_original_return, dtype=jnp.float32
			)
		)

		# --- Initialize Transformer Memory State ---
		memories = jnp.zeros(
			(config.num_envs, config.window_mem, config.num_layers, config.embed_size)
		)
		memories_mask = jnp.zeros(
			(config.num_envs, config.num_heads, 1, config.window_mem + 1), dtype=jnp.bool_
		)
		memories_mask_idx = jnp.zeros((config.num_envs,), dtype=jnp.int32) + (config.window_mem + 1)
		done = jnp.zeros((config.num_envs,), dtype=jnp.bool_)

		rng, _rng = jax.random.split(rng)

		# Current loop counter needed for memory index calculation
		init_step_env_currentloop = 0

		# Current update step counter
		init_update_step = 0

		initial_runner_state = (
			train_state,
			env_state,
			memories,
			memories_mask,
			memories_mask_idx,
			obsv,
			done,
			init_step_env_currentloop,
			init_update_step,
			_rng,
		)

		def _log_callback(metrics, step):
			# Unpack the tuple (now includes max; +2 SIL scalars when sil is on)
			if sil_on:
				t_loss, v_loss, a_loss, ent, g_norm_mean, g_norm_max, s_loss, s_fill = metrics
			else:
				t_loss, v_loss, a_loss, ent, g_norm_mean, g_norm_max = metrics

			to_log = {
				"train/total_loss": t_loss,
				"train/value_loss": v_loss,
				"train/actor_loss": a_loss,
				"train/entropy": ent,
				"train/grad_norm_mean": g_norm_mean,
				"train/grad_norm_max": g_norm_max,  # <--- New
				"global_step": step,
			}
			if sil_on:
				to_log["train/sil_loss"] = s_loss
				to_log["train/sil_fill"] = s_fill
			wandb.log(to_log)
			# v7fix4.8 guardrail: feed the session watchdog (training.py reads the verdict
			# after the session and reverts weights/archive/notebook if it trips).
			record_update(float(v_loss), float(ent))

		# --------------------------
		# The Transformer PPO update step
		# --------------------------
		def _update_step(update_carry, unused_scan_input):
			# v7fix56 P1' SIL: the scan carry additionally threads the SIL ring buffers.
			if sil_on:
				runner_state, sil_state_c = update_carry
			else:
				runner_state = update_carry
			# === DEBUG: PRINT LEARNING RATE ===
			# # 1. Unpack just enough to get the optimizer state
			# _train_state = runner_state[0]

			# # 2. Extract the step count from Optax state.
			# # Optax states are trees; for Adam/Clip chains, the count is usually the first leaf.
			# _step_count = jax.tree_util.tree_leaves(_train_state.opt_state)[0]

			# # 3. Calculate current LR using the schedule function defined in the outer scope
			# _current_lr = linear_schedule(_step_count)

			# # 4. Print using JAX debug (Safe inside JIT/Scan)
			# # Use formatting to make it readable.
			# # We condition it to print only periodically if you want,
			# # but printing every update is usually fine.
			# jax.debug.print("Step: {x} | LR: {y:.8f}", x=_step_count, y=_current_lr)
			# ==================================
			# === A. COLLECT TRAJECTORIES ===
			def _env_step(runner_state, _):
				(
					train_state,
					env_state,
					memories,
					memories_mask,
					memories_mask_idx,
					last_obs,
					done,
					step_env_currentloop,
					update_step,
					rng,
				) = runner_state

				# 1. Reset memories mask if done
				memories_mask_idx = jnp.where(
					done, config.window_mem, jnp.clip(memories_mask_idx - 1, 0, config.window_mem)
				)
				memories_mask = jnp.where(
					done[:, None, None, None],
					jnp.zeros(
						(config.num_envs, config.num_heads, 1, config.window_mem + 1),
						dtype=jnp.bool_,
					),
					memories_mask,
				)

				# 2. Update memories mask with the potential additional step
				memories_mask_idx_ohot = jax.nn.one_hot(memories_mask_idx, config.window_mem + 1)
				memories_mask_idx_ohot = memories_mask_idx_ohot[:, None, None, :].repeat(
					config.num_heads, 1
				)
				memories_mask = jnp.logical_or(memories_mask, memories_mask_idx_ohot)

				# 3. Select Action
				rng, _rng = jax.random.split(rng)
				pi, value, memories_out = network.apply(
					train_state.params,
					memories,
					last_obs,
					memories_mask,
					method=network.model_forward_eval,
				)
				action = pi.sample(seed=_rng)
				log_prob = pi.log_prob(action)

				# 4. Update Cache: Roll memory and add new output
				memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(memories_out)

				# 5. Step Env
				rng, _rng = jax.random.split(rng)
				obsv, env_state, reward, done, info = env.step(_rng, env_state, action, env_params)

				env_state = env_state.replace(
					running_original_return=jnp.full(
						(config.num_envs,), current_original_return, dtype=jnp.float32
					)
				)

				# 6. Compute memory indices for training
				memory_indices = jnp.arange(0, config.window_mem)[
					None, :
				] + step_env_currentloop * jnp.ones((config.num_envs, 1), dtype=jnp.int32)

				transition = Transition(
					done,
					action,
					value,
					reward,
					log_prob,
					memories_mask.squeeze(),
					memory_indices,
					last_obs,
					info,
				)

				carry = (
					train_state,
					env_state,
					memories,
					memories_mask,
					memories_mask_idx,
					obsv,
					done,
					step_env_currentloop + 1,
					update_step,
					rng,
				)
				return carry, (transition, memories_out)

			# Save previous memories to concatenate later (so first step of batch has context)
			memories_previous = runner_state[2]

			(final_state_carry), (traj_batch, memories_batch) = jax.lax.scan(
				_env_step, runner_state, None, config.num_steps
			)

			(
				train_state,
				final_env_state,
				final_memories,
				final_mask,
				final_mask_idx,
				final_obs,
				done,
				final_step_loop,
				update_step,
				rng,
			) = final_state_carry

			# === B. CALCULATE ADVANTAGES (GAE) ===
			# For GAE we need the value of the *next* state (final_obs)
			_, last_val, _ = network.apply(
				train_state.params,
				final_memories,
				final_obs,
				final_mask,
				method=network.model_forward_eval,
			)

			def _calculate_gae(traj_batch, last_val):
				def _get_advantages(carry, transition):
					gae, next_value = carry
					done, value, reward = transition.done, transition.value, transition.reward
					delta = reward + config.gamma * next_value * (1 - done) - value
					gae = delta + config.gamma * config.gae_lambda * (1 - done) * gae
					return (gae, value), gae

				_, advantages = jax.lax.scan(
					_get_advantages,
					(jnp.zeros_like(last_val), last_val),
					traj_batch,
					reverse=True,
					unroll=16,
				)
				# v7fix4.8: bound value targets to the achievable return range (no-op when healthy).
				return advantages, jnp.clip(
					advantages + traj_batch.value, vt_clip_min, vt_clip_max
				)

			advantages, targets = _calculate_gae(traj_batch, last_val)

			# Prepare scoring data (standard PPO interface for your calculator)
			# We strip out transformer specifics here because scoring doesn't need them
			scoring_traj = traj_batch.replace(
				obs=None, action=None, log_prob=None, memories_mask=None, memories_indices=None
			)
			# Explicitly set fields to None if the NamedTuple structure allows, or just reconstruct
			# Actually, Transition has new fields. The calculator expects `info` etc.
			# It should be fine as long as `scoring.py` accesses attributes by name.

			scoring_data = {"traj_batch": scoring_traj, "advantages": advantages}

			# NEW: Metric Logging for Original Task (Last Task ID)
			# The original task is always appended to the end, so its ID is num_tasks - 1
			original_task_idx = num_tasks - 1
			task_mask = traj_batch.info["task_id"] == original_task_idx
			# We only care about episodes that actually returned (finished)
			valid_mask = traj_batch.info["returned_episode"] * task_mask

			# # Calculate metrics only for the valid episodes of the original task
			# # We add epsilon to valid_mask.sum() to avoid division by zero if no original task episodes finished
			# metric = jax.tree.map(
			# 	lambda x: (x * valid_mask).sum() / (valid_mask.sum() + 1e-8),
			# 	traj_batch.info,
			# )

			# if config.debug and config.use_wandb:
			#     # Calculate cumulative steps based on global update count to ensure continuity
			# 	current_global_update = initial_global_update_step + update_step
			# 	current_env_steps = current_global_update * config.num_steps * config.num_envs

			# 	def callback(metric, global_update, env_steps):
			# 		to_log = create_log_dict(metric, config)
			# 		batch_log(global_update, to_log, config, env_steps)

			# 	jax.debug.callback(callback, metric, current_global_update, current_env_steps)

			# === C. UPDATE NETWORK (TRANSFORMER LOSS) ===

			# Concatenate previous memories so the first steps of the batch have context
			memories_batch = jnp.concatenate(
				[jnp.swapaxes(memories_previous, 0, 1), memories_batch], axis=0
			)

			def _update_epoch(update_state, unused):
				def _update_minbatch(train_state, batch_info):
					traj_batch, memories_batch, advantages, targets = batch_info

					# advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

					def _loss_fn(params, traj_batch, memories_batch, gae, targets):
						# --- TRANSFORMER SPECIFIC: MEMORY BATCHING ---
						# Construct memory batch from indices
						memories_batch = batch_indices_select(
							memories_batch, traj_batch.memories_indices[:, :: config.window_grad]
						)
						memories_batch = batchify(memories_batch)

						# Create Mask for Window Grad
						memories_mask = traj_batch.memories_mask.reshape(
							(
								-1,
								config.window_grad,
							)
							+ traj_batch.memories_mask.shape[2:]
						)
						memories_mask = jnp.swapaxes(memories_mask, 1, 2)
						# Concatenate with 0s
						memories_mask = jnp.concatenate(
							(
								memories_mask,
								jnp.zeros(
									memories_mask.shape[:-1] + (config.window_grad - 1,),
									dtype=jnp.bool_,
								),
							),
							axis=-1,
						)
						# Roll
						memories_mask = roll_vmap(
							memories_mask, jnp.arange(0, config.window_grad), -1
						)

						# Reshape Obs and Batch
						obs = traj_batch.obs.reshape(
							(
								-1,
								config.window_grad,
							)
							+ traj_batch.obs.shape[2:]
						)
						traj_batch_r, targets_r, gae_r = jax.tree_util.tree_map(
							lambda x: jnp.reshape(x, (-1, config.window_grad) + x.shape[2:]),
							(traj_batch, targets, gae),
						)

						# Network Output (Train Mode)
						pi, value = network.apply(
							params,
							memories_batch,
							obs,
							memories_mask,
							method=network.model_forward_train,
						)
						log_prob = pi.log_prob(traj_batch_r.action)

						# Value Loss
						value_pred_clipped = traj_batch_r.value + (value - traj_batch_r.value).clip(
							-config.clip_eps, config.clip_eps
						)
						value_losses = jnp.square(value - targets_r)
						value_losses_clipped = jnp.square(value_pred_clipped - targets_r)
						value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

						# Actor Loss
						ratio = jnp.exp(log_prob - traj_batch_r.log_prob)
						gae_r = (gae_r - gae_r.mean()) / (gae_r.std() + 1e-8)
						loss_actor1 = ratio * gae_r
						loss_actor2 = (
							jnp.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * gae_r
						)
						loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

						entropy = pi.entropy().mean()
						total_loss = (
							loss_actor + config.vf_coef * value_loss - config.ent_coef * entropy
						)
						return total_loss, (value_loss, loss_actor, entropy)

					grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
					(total_loss, (value_loss, loss_actor, entropy)), grads = grad_fn(
						train_state.params, traj_batch, memories_batch, advantages, targets
					)

					grad_norm = optax.global_norm(grads)
					train_state = train_state.apply_gradients(grads=grads)
					return train_state, (total_loss, value_loss, loss_actor, entropy, grad_norm)

				(train_state, traj_batch, memories_batch, advantages, targets, update_step, rng) = (
					update_state
				)
				rng, _rng = jax.random.split(rng)

				# Batch Permutation
				permutation = jax.random.permutation(_rng, config.num_envs)
				batch = (traj_batch, memories_batch, advantages, targets)

				# Swap axes to (Envs, Steps, ...)
				batch = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), batch)
				# Shuffle envs
				shuffled_batch = jax.tree_util.tree_map(
					lambda x: jnp.take(x, permutation, axis=0), batch
				)

				# Create Minibatches
				minibatches = jax.tree_util.tree_map(
					lambda x: jnp.reshape(x, [config.num_minibatches, -1] + list(x.shape[1:])),
					shuffled_batch,
				)

				train_state, (total_loss, value_loss, loss_actor, entropy, grad_norm) = jax.lax.scan(_update_minbatch, train_state, minibatches)
				return (
					train_state,
					traj_batch,
					memories_batch,
					advantages,
					targets,
					update_step,
					rng,
				), (total_loss, value_loss, loss_actor, entropy, grad_norm)

			update_state = (
				train_state,
				traj_batch,
				memories_batch,
				advantages,
				targets,
				update_step,
				rng,
			)
			update_state, rl_info = jax.lax.scan(
				_update_epoch, update_state, None, config.update_epochs
			)

			# --- v7fix56 P1' SIL: ring-buffer write + ONE self-imitation step, after the PPO
			# epochs (Oh et al. 2018 ordering). The step is gated by lax.cond so beta=0 (or an
			# empty pool) leaves params AND opt state bit-identical (an Adam apply with zero
			# grads would still move params through stale momentum). The (R-V)+ weight against
			# the CURRENT value head provides the self-annealing decay of §2.2.
			if sil_on:
				train_state_sil = update_state[0]
				# fold_in derives an independent stream WITHOUT advancing the main rng:
				# beta=0 / empty-pool runs stay BIT-IDENTICAL to sil=false (S.6a), and a SIL
				# arm shares the control arm's exact trajectory until beta first fires.
				_rng_s = jax.random.fold_in(update_state[-1], 20260719)
				current_global_update = initial_global_update_step + update_step
				prio_steps = jnp.maximum(targets - traj_batch.value, 0.0)
				for _p, _wall in enumerate(sil_pools):
					_key = sil_lib.resolve_achievement_key(traj_batch.info.keys(), _wall)
					_sv = sil_lib.episode_success_steps(traj_batch.done, traj_batch.info[_key])
					sil_state_c = sil_lib.write_pool(
						sil_state_c, _p, traj_batch.obs, traj_batch.action, targets,
						traj_batch.memories_mask, memories_batch, _sv, prio_steps,
						current_global_update, sil_writes,
					)
				sil_batch, sil_any = sil_lib.sample_pools(
					sil_state_c, _rng_s, sil_windows, current_global_update,
					sil_stale_upd, sil_prioritized,
				)

				def _sil_loss_fn(params, b):
					n_win = sil_windows * len(sil_pools)
					mem_idx = (
						jnp.arange(config.window_mem)[None, :] + jnp.arange(config.num_steps)[:, None]
					)
					mem_idx = jnp.broadcast_to(
						mem_idx[None], (n_win, config.num_steps, config.window_mem)
					)
					memories_b = batch_indices_select(b["mem"], mem_idx[:, :: config.window_grad])
					memories_b = batchify(memories_b)
					sil_mask = b["mmask"].reshape((-1, config.window_grad) + b["mmask"].shape[2:])
					sil_mask = jnp.swapaxes(sil_mask, 1, 2)
					sil_mask = jnp.concatenate(
						(
							sil_mask,
							jnp.zeros(
								sil_mask.shape[:-1] + (config.window_grad - 1,), dtype=jnp.bool_
							),
						),
						axis=-1,
					)
					sil_mask = roll_vmap(sil_mask, jnp.arange(0, config.window_grad), -1)
					obs_b = b["obs"].reshape((-1, config.window_grad) + b["obs"].shape[2:])
					act_r, ret_r, sval_r = jax.tree_util.tree_map(
						lambda x: jnp.reshape(x, (-1, config.window_grad) + x.shape[2:]),
						(b["action"], b["ret"], b["svalid"]),
					)
					pi, value = network.apply(
						params, memories_b, obs_b, sil_mask, method=network.model_forward_train
					)
					log_prob = pi.log_prob(act_r)
					w = jnp.maximum(ret_r - value, 0.0)
					sval_f = sval_r.astype(jnp.float32)
					denom = sval_f.sum() + 1e-8
					pol_loss = -(log_prob * jax.lax.stop_gradient(w) * sval_f).sum() / denom
					val_loss = 0.5 * ((w**2) * sval_f).sum() / denom
					return pol_loss + sil_vf_coef * val_loss, (pol_loss, val_loss)

				(sil_loss, (_sil_pol, _sil_val)), sil_grads = jax.value_and_grad(
					_sil_loss_fn, has_aux=True
				)(train_state_sil.params, sil_batch)
				sil_scale = jnp.asarray(sil_beta, jnp.float32) * sil_any.astype(jnp.float32)
				sil_grads = jax.tree_util.tree_map(lambda g: g * sil_scale, sil_grads)
				train_state_sil = jax.lax.cond(
					sil_scale > 0.0,
					lambda ts: ts.apply_gradients(grads=sil_grads),
					lambda ts: ts,
					train_state_sil,
				)
				update_state = (train_state_sil,) + update_state[1:]
				sil_fill = sil_lib.pool_fill_fraction(sil_state_c)

			losses_and_ent = rl_info[:4]
			grad_norms = rl_info[4]

			# 1. Calculate Means for losses (Standard)
			losses_mean = jax.tree_util.tree_map(lambda x: jnp.mean(x), losses_and_ent)

			# 2. Calculate Mean AND Max for grad_norm (Diagnostic)
			gn_mean = jnp.mean(grad_norms)
			gn_max = jnp.max(grad_norms)

			# 3. Pack it all up for the callback
			# Structure: (t_loss, v_loss, a_loss, ent, g_norm_mean, g_norm_max [, sil_loss, sil_fill])
			if sil_on:
				metrics_to_log = (*losses_mean, gn_mean, gn_max, sil_loss, sil_fill)
			else:
				metrics_to_log = (*losses_mean, gn_mean, gn_max)

			current_step = initial_global_update_step + update_step
			jax.debug.callback(_log_callback, metrics_to_log, current_step)
			

			# D. PREPARE FOR NEXT STEP
			train_state = update_state[0]
			rng = update_state[-1]
			# Reset loop counter to 0 for the next block of rollouts
			next_runner_state = (
				train_state,
				final_env_state,
				final_memories,
				final_mask,
				final_mask_idx,
				final_obs,
				done,
				0,
				update_step + 1,
				rng,
			)

			if sil_on:
				return (next_runner_state, sil_state_c), scoring_data
			return next_runner_state, scoring_data

		# --- Run fixed-iteration training loop ---
		initial_update_carry = (
			(initial_runner_state, sil_state) if sil_on else initial_runner_state
		)
		(final_update_carry, scan_scoring_data) = jax.lax.scan(
			_update_step, initial_update_carry, None, length=NUM_UPDATES
		)
		if sil_on:
			final_runner_state, sil_state_final = final_update_carry
		else:
			final_runner_state = final_update_carry

		final_train_state = final_runner_state[0]

		# --- Process outputs for the "Smart Calculator" (IDENTICAL TO OLD CODE) ---
		k = config.scoring_window_updates
		scoring_window_data = jax.tree.map(lambda x: x[-k:], scan_scoring_data)

		# Flatten k and num_steps
		flat_traj = jax.tree.map(
			lambda x: x.reshape(-1, *x.shape[2:]), scoring_window_data["traj_batch"]
		)
		flat_advantages = scoring_window_data["advantages"].reshape(
			-1, *scoring_window_data["advantages"].shape[2:]
		)

		final_scoring_window_data = {"traj_batch": flat_traj, "advantages": flat_advantages}

		num_env_steps_done = NUM_UPDATES * config.num_envs * config.num_steps

		out = {
			"train_state": final_train_state,
			"metrics": {
				"scoring_window_data": final_scoring_window_data,
				"num_updates_done": NUM_UPDATES,
				"num_env_steps_done": num_env_steps_done,
			},
		}
		if sil_on:
			out["sil_state"] = sil_state_final
		return out

	return train


def make_eval(config, task_classes, num_training_updates, task_embeddings=None):
	"""Identical to make_train, but deletes the Policy/Value update step.
	It runs rollouts and GAE (for scoring) but returns the train_state UNCHANGED.
	"""
	# --- 1. Environment Setup (Same as Train) ---
	NUM_UPDATES = num_training_updates
	num_tasks = len(task_classes)
	static_env_params = StaticEnvParams()
	env_params = EnvParams(max_timesteps=4096)
	# env_params = EnvParams()

	# ... (Copy the Env setup logic from make_train exactly) ...
	if config.dicode_manager.mode != "reward":
		if task_embeddings is not None:
			embedding_size = task_embeddings.shape[1]
			base_env = MultiTaskMiniCraftaxEnv(
				task_classes,
				static_env_params,
				env_params,
				config.training.condition_on_task,
				conditioning_type="embedding",
				embedding_size=embedding_size,
				completion_bonus_scale=config.dicode_manager.completion_bonus_scale,
				completion_bonus_min=config.dicode_manager.completion_bonus_min,
			)
		else:
			base_env = MultiTaskMiniCraftaxEnv(
				task_classes,
				static_env_params,
				env_params,
				config.training.condition_on_task,
				completion_bonus_scale=config.dicode_manager.completion_bonus_scale,
				completion_bonus_min=config.dicode_manager.completion_bonus_min,
			)
	else:
		if task_embeddings is not None:
			embedding_size = task_embeddings.shape[1]
			base_env = MultiTaskMiniCraftaxEnvR(
				task_classes,
				static_env_params,
				env_params,
				config.training.condition_on_task,
				conditioning_type="embedding",
				embedding_size=embedding_size,
			)
		else:
			base_env = MultiTaskMiniCraftaxEnvR(
				task_classes, static_env_params, env_params, config.training.condition_on_task
			)

	# base_env = LogWrapper(base_env)

	env = MultiTaskOptimisticLogWrapperAllTasks(
		base_env,
		config.validation.num_envs,
		num_tasks,
		config.validation.optimistic_reset_ratio,
		task_embeddings,
	)
	env_params = env.default_params

	# env = CraftaxAugObsTrain()
	# env_params = env.default_params

	# env = MultiTaskOptimisticLogWrapperAllTasks(
	# 	env,
	# 	config.validation.num_envs,
	# 	1,
	# 	config.validation.optimistic_reset_ratio,
	# 	None,
	# )
	# env = LogWrapper(env)
	# env = OptimisticResetVecEnvWrapper(env, num_envs=config.validation.num_envs, reset_ratio=config.validation.optimistic_reset_ratio)

	# --- 2. Network Setup (Same as Train) ---
	network = ActorCriticTransformer(
		action_dim=env.action_space(env_params).n,
		activation=config.training.activation,
		hidden_layers=config.training.hidden_layers,
		encoder_size=config.training.embed_size,
		num_heads=config.training.num_heads,
		qkv_features=config.training.qkv_features,
		num_layers=config.training.num_layers,
		gating=config.training.gating,
		gating_bias=config.training.gating_bias,
	)

	# (No Optimizer needed for Eval, but we define 'tx' to satisfy TrainState creation if needed)
	tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-4))

	def eval_loop(rng, train_state=None):
		"""The JIT-compiled evaluation loop."""
		obs_dim = env.observation_space(env_params).shape[0]

		# --- Init (Same as Train) ---
		if train_state is None:
			# Should usually be passed in, but handle None case just in case
			rng, _rng = jax.random.split(rng)
			init_obs = jnp.zeros((2, obs_dim))
			init_memory = jnp.zeros(
				(
					2,
					config.training.window_mem,
					config.training.num_layers,
					config.training.embed_size,
				)
			)
			init_mask = jnp.zeros(
				(2, config.training.num_heads, 1, config.training.window_mem + 1), dtype=jnp.bool_
			)
			network_params = network.init(_rng, init_memory, init_obs, init_mask)
			train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

		rng, _rng = jax.random.split(rng)
		obsv, env_state = env.reset(_rng, env_params)

		# Transformer State Init
		memories = jnp.zeros(
			(
				config.validation.num_envs,
				config.training.window_mem,
				config.training.num_layers,
				config.training.embed_size,
			)
		)
		memories_mask = jnp.zeros(
			(
				config.validation.num_envs,
				config.training.num_heads,
				1,
				config.training.window_mem + 1,
			),
			dtype=jnp.bool_,
		)
		memories_mask_idx = jnp.zeros((config.validation.num_envs,), dtype=jnp.int32) + (
			config.training.window_mem + 1
		)
		done = jnp.zeros((config.validation.num_envs,), dtype=jnp.bool_)

		rng, _rng = jax.random.split(rng)
		init_runner_state = (
			train_state,
			env_state,
			memories,
			memories_mask,
			memories_mask_idx,
			obsv,
			done,
			0,
			_rng,
		)

		# --------------------------
		# The Evaluation Step (NO UPDATE)
		# --------------------------
		def _eval_update_step(runner_state, unused):
			# 1. Run the Environment Rollout (Same as Train)
			#    We use the exact same _env_step logic to ensure Transformer memory
			#    is handled correctly during rollout.

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
							config.validation.num_envs,
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
				pi, value, memories_out = network.apply(
					train_state.params,
					memories,
					last_obs,
					memories_mask,
					method=network.model_forward_eval,
				)
				action = pi.sample(seed=_rng)
				log_prob = pi.log_prob(action)

				memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(memories_out)

				rng, _rng = jax.random.split(rng)
				obsv, env_state, reward, done, info = env.step(_rng, env_state, action, env_params)

				memory_indices = jnp.arange(0, config.training.window_mem)[
					None, :
				] + step_env_currentloop * jnp.ones(
					(config.validation.num_envs, 1), dtype=jnp.int32
				)

				transition = Transition(
					done,
					action,
					value,
					reward,
					log_prob,
					memories_mask.squeeze(),
					memory_indices,
					last_obs,
					info,
				)
				return (
					train_state,
					env_state,
					memories,
					memories_mask,
					memories_mask_idx,
					obsv,
					done,
					step_env_currentloop + 1,
					rng,
				), transition

			(final_state_carry), traj_batch = jax.lax.scan(
				_env_step, runner_state, None, config.validation.num_steps
			)

			(
				train_state,
				final_env_state,
				final_memories,
				final_mask,
				final_mask_idx,
				final_obs,
				done,
				final_step_loop,
				rng,
			) = final_state_carry

			# 2. Calculate Advantages (GAE)
			#    We STILL need this because your 'Smart Calculator' uses advantages
			#    to compute PVL (Positive Value Loss) scores.
			_, last_val, _ = network.apply(
				train_state.params,
				final_memories,
				final_obs,
				final_mask,
				method=network.model_forward_eval,
			)
			# last_val = last_val.squeeze(0)

			def _calculate_gae(traj_batch, last_val):
				def _get_advantages(carry, transition):
					gae, next_value = carry
					done, value, reward = transition.done, transition.value, transition.reward
					delta = reward + config.training.gamma * next_value * (1 - done) - value
					gae = (
						delta
						+ config.training.gamma * config.training.gae_lambda * (1 - done) * gae
					)
					return (gae, value), gae

				_, advantages = jax.lax.scan(
					_get_advantages,
					(jnp.zeros_like(last_val), last_val),
					traj_batch,
					reverse=True,
					unroll=16,
				)
				return advantages

			advantages = _calculate_gae(traj_batch, last_val)

			# 3. Prepare Data for Output
			scoring_traj = traj_batch.replace(
				obs=None, action=None, log_prob=None, memories_mask=None, memories_indices=None
			)

			scoring_data = {"traj_batch": scoring_traj, "advantages": advantages}

			# 4. CRITICAL DIFFERENCE: NO UPDATE STEP
			# We do not run _update_epoch. We do not calculate gradients.
			# We simply return the state (ready for next rollout) and the data.

			# Reset loop counter to 0
			next_runner_state = (
				train_state,
				final_env_state,
				final_memories,
				final_mask,
				final_mask_idx,
				final_obs,
				done,
				0,
				rng,
			)

			return next_runner_state, scoring_data

		# Run the Scan
		(final_runner_state, scan_scoring_data) = jax.lax.scan(
			_eval_update_step, init_runner_state, None, length=NUM_UPDATES
		)

		final_train_state = final_runner_state[0]

		# Process for Calculator
		# k = config.scoring_window_updates
		# scoring_window_data = jax.tree.map(lambda x: x[-k:], scan_scoring_data)
		scoring_window_data = scan_scoring_data

		flat_traj = jax.tree.map(
			lambda x: x.reshape(-1, *x.shape[2:]), scoring_window_data["traj_batch"]
		)
		flat_advantages = scoring_window_data["advantages"].reshape(
			-1, *scoring_window_data["advantages"].shape[2:]
		)

		final_scoring_window_data = {"traj_batch": flat_traj, "advantages": flat_advantages}

		num_env_steps_done = NUM_UPDATES * config.validation.num_envs * config.validation.num_steps

		return {
			"train_state": final_train_state,  # Unchanged!
			"metrics": {
				"scoring_window_data": final_scoring_window_data,
				"num_updates_done": NUM_UPDATES,
				"num_env_steps_done": num_env_steps_done,
			},
		}

	return eval_loop


# =================================================================
# === TOP-LEVEL API (UNCHANGED) ===================================
# =================================================================


def run_training_session(
	config,
	rng,
	task_classes,
	num_training_updates,
	task_embeddings=None,
	train_state=None,
	task_distribution_proportions=None,
	global_update_step=0,
	current_original_return=0.0,
	sil_state=None,
	sil_beta=0.0,
):
	config_t = config.training
	# v7fix4.8 guardrail: fresh watchdog stats for this session (fed per update by _log_callback).
	reset_session_stats()
	train_fn = make_train(
		config_t,
		task_classes,
		num_training_updates,
		task_embeddings,
		task_distribution_proportions,
		global_update_step,
	)
	train_jit = jax.jit(train_fn)
	print("JIT compiling and running training session (Transformer)...")
	start_time = time.time()
	sil_enabled = bool(_cfg_get(config_t, "sil", False)) and bool(
		list(_cfg_get(config_t, "sil_pools", []) or [])
	)
	if sil_enabled:
		# beta as a device scalar: traced by shape/dtype, so a per-session beta change
		# never forces an extra recompile beyond the per-session jit we already pay.
		results = train_jit(
			rng, train_state, current_original_return, sil_state,
			jnp.asarray(sil_beta, jnp.float32),
		)
	else:
		results = train_jit(rng, train_state, current_original_return)
	print(f"Session finished in {time.time() - start_time:.2f} seconds.")
	return results


def run_evaluation_rollouts(
	config, rng, task_classes, num_training_updates, task_embeddings=None, train_state=None
):
	if train_state is None:
		raise ValueError("run_evaluation_rollouts requires a valid train_state.")
	eval_fn = make_eval(config, task_classes, num_training_updates, task_embeddings)
	eval_jit = jax.jit(eval_fn)
	print("JIT compiling and running evaluation rollouts (Transformer)...")
	start_time = time.time()
	results = eval_jit(rng, train_state)
	print(f"Session finished in {time.time() - start_time:.2f} seconds.")
	return results
