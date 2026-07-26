"""LC-RMT16-RESET128 PPO trainer: identical to ppo_tr_rmt16.py but clears
mem_tokens at every 128-step rollout boundary (RESET128 ablation).


Based on dicode/ppo_tr.py (read-only).  Adds:
  - rmt16_state carry through rollout (mem_tokens + segment buffer)
  - mem_tokens snapshot stored per step for PPO training forward pass
  - Segment-boundary token update wired to network.rmt_update_attn
  - rmt16_state checkpoint save/restore

Hyperparams MUST match canonical Control freeze record.
"""
import functools, time
import distrax, flax.linen as nn, jax, jax.numpy as jnp, numpy as np, optax
from flax.training.train_state import TrainState
from flax import struct

from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv
from dicode.wrappers_cl import DistributedMultiTaskOptimisticLogWrapper
from dicode.train_guard import record_update, reset_session_stats

from network_rmt16 import ActorCriticTransformerRMT16
import rmt16_memory as rmtm

# ---- Transition ----
@struct.dataclass
class Transition:
    done:        jnp.ndarray
    action:      jnp.ndarray
    value:       jnp.ndarray
    reward:      jnp.ndarray
    log_prob:    jnp.ndarray
    memories_mask:   jnp.ndarray
    memories_indices: jnp.ndarray
    obs:         jnp.ndarray
    info:        jnp.ndarray

# ---- helpers ----
indices_select = lambda x, y: x[y]
batch_indices_select = jax.vmap(indices_select)
roll_vmap = jax.vmap(jnp.roll, in_axes=(-2, 0, None), out_axes=-2)
batchify = lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1],) + x.shape[2:])


def make_train(config, task_classes, num_training_updates,
               task_embeddings=None, task_distribution_proportions=None,
               initial_global_update_step=0):
    NUM_UPDATES = num_training_updates
    num_tasks = len(task_classes)
    vt_clip_min = float(getattr(config, "value_target_clip_min", -50.0))
    vt_clip_max = float(getattr(config, "value_target_clip_max", 300.0))

    static_env_params = StaticEnvParams()
    env_params = EnvParams(max_timesteps=4096)
    embedding_size = task_embeddings.shape[1] if task_embeddings is not None else 67
    base_env = MultiTaskMiniCraftaxEnv(
        task_classes, static_env_params, env_params, config.condition_on_task,
        conditioning_type="embedding", embedding_size=embedding_size)
    if task_distribution_proportions is None:
        task_distribution_proportions = jnp.ones(num_tasks) / num_tasks
    env = DistributedMultiTaskOptimisticLogWrapper(
        base_env, jax.random.PRNGKey(0), config.num_envs, num_tasks,
        config.optimistic_reset_ratio, task_distribution_proportions, task_embeddings)
    env_params = env.default_params

    # ---- RMT16 config ----
    rmt_cfg = rmtm.RMT16Config(
        num_tokens=getattr(config, "rmt_num_tokens", 16),
        segment_len=config.num_steps,  # 128 = rollout length
        encoder_size=config.embed_size)

    network = ActorCriticTransformerRMT16(
        action_dim=env.action_space(env_params).n,
        activation=config.activation,
        hidden_layers=config.hidden_layers,
        encoder_size=config.embed_size,
        num_heads=config.num_heads,
        qkv_features=config.qkv_features,
        num_layers=config.num_layers,
        gating=config.gating,
        gating_bias=config.gating_bias,
        rmt_num_tokens=rmt_cfg.num_tokens)

    TOTAL_GLOBAL_UPDATES = (
        (config.total_timesteps // config.num_envs // config.num_steps
         // config.max_updates_per_session) + 1
    ) * config.max_updates_per_session

    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.lr, eps=1e-5))

    def train(rng, train_state=None, rmt_state_init=None,
              resume_runner_state=None):
        obs_dim = env.observation_space(env_params).shape[0]

        if resume_runner_state is not None:
            runner_state = resume_runner_state
        else:
            if train_state is None:
                rng, _rng = jax.random.split(rng)
                init_obs  = jnp.zeros((2, obs_dim))
                init_mem  = jnp.zeros((2, config.window_mem, config.num_layers, config.embed_size))
                init_mask = jnp.zeros((2, config.num_heads, 1, config.window_mem + 1), dtype=jnp.bool_)
                init_tok  = jnp.zeros((2, rmt_cfg.num_tokens, config.embed_size))
                network_params = network.init(
                    _rng, init_mem, init_obs, init_mask,
                    mem_tokens=init_tok)
                train_state = TrainState.create(
                    apply_fn=network.apply, params=network_params, tx=tx)

            rng, _rng = jax.random.split(rng)
            obsv, env_state = env.reset(_rng, env_params)

            memories = jnp.zeros((config.num_envs, config.window_mem, config.num_layers, config.embed_size))
            memories_mask = jnp.zeros((config.num_envs, config.num_heads, 1, config.window_mem + 1), dtype=jnp.bool_)
            memories_mask_idx = jnp.full((config.num_envs,), config.window_mem + 1, dtype=jnp.int32)
            done = jnp.zeros((config.num_envs,), dtype=jnp.bool_)

            if rmt_state_init is None:
                rmt_state = rmtm.rmt16_init(config.num_envs, rmt_cfg)
            else:
                rmt_state = rmt_state_init

            runner_state = (train_state, env_state, memories, memories_mask,
                            memories_mask_idx, obsv, done, rmt_state,
                            0, 0, _rng)

        def _update_step(runner_state, _unused):
            # === A. ROLLOUT ===
            def _env_step(carry, _):
                (ts, es, mems, mmask, midx, obs, dn, rmt_st,
                 step_loop, upd_step, rng) = carry

                # RESET128: clear mem_tokens at rollout boundary (every num_steps steps)
                at_boundary = jnp.logical_and(jnp.greater(step_loop, 0),
                                               jnp.equal(jnp.mod(step_loop, config.num_steps), 0))
                rmt_st = {**rmt_st,
                    "mem_tokens": jnp.where(at_boundary, jnp.zeros_like(rmt_st["mem_tokens"]), rmt_st["mem_tokens"])}

                midx = jnp.where(dn, config.window_mem,
                                 jnp.clip(midx - 1, 0, config.window_mem))
                mmask = jnp.where(dn[:, None, None, None],
                                  jnp.zeros_like(mmask), mmask)
                ohot = jax.nn.one_hot(midx, config.window_mem + 1)
                ohot = ohot[:, None, None, :].repeat(config.num_heads, 1)
                mmask = jnp.logical_or(mmask, ohot)

                # RMT16: read mem_tokens BEFORE forward
                mtok = rmt_st["mem_tokens"]

                rng, _rng = jax.random.split(rng)
                pi, value, mem_out, h_t = network.apply(
                    ts.params, mems, obs, mmask,
                    mem_tokens=mtok,
                    method=network.model_forward_eval)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                mems = jnp.roll(mems, -1, axis=1).at[:, -1].set(mem_out)

                rng, _rng = jax.random.split(rng)
                obs_next, es, reward, dn, info = env.step(_rng, es, action, env_params)

                # RMT16: store h_t and update tokens at segment boundary
                def _update_fn(tokens, seg_buf):
                    return network.apply(
                        ts.params, tokens, seg_buf,
                        method=network.update_rmt_tokens)

                rmt_st_new = rmtm.rmt16_step(
                    rmt_st, h_t, dn, _update_fn, rmt_cfg)

                mem_indices = (jnp.arange(0, config.window_mem)[None, :]
                               + step_loop * jnp.ones((config.num_envs, 1), dtype=jnp.int32))
                transition = Transition(dn, action, value, reward, log_prob,
                                        mmask.squeeze(), mem_indices, obs, info)
                carry = (ts, es, mems, mmask, midx, obs_next, dn, rmt_st_new,
                         step_loop + 1, upd_step, rng)
                # Store mem_tokens snapshot for training forward pass
                return carry, (transition, mem_out, mtok)

            mems_prev = runner_state[2]

            (final_carry), (traj, mems_batch, mtok_seq) = \
                jax.lax.scan(_env_step, runner_state, None, config.num_steps)

            (ts, es, mems_f, mmask_f, midx_f, obs_f, dn_f, rmt_st_f,
             step_f, upd_step, rng) = final_carry

            # === B. GAE ===
            mtok_f = rmt_st_f["mem_tokens"]
            _, last_val, _, _ = network.apply(
                ts.params, mems_f, obs_f, mmask_f,
                mem_tokens=mtok_f,
                method=network.model_forward_eval)

            def _gae(carry, tr):
                gae, nv = carry
                d, v, r = tr.done, tr.value, tr.reward
                delta = r + config.gamma * nv * (1 - d) - v
                gae = delta + config.gamma * config.gae_lambda * (1 - d) * gae
                return (gae, v), gae
            _, advantages = jax.lax.scan(_gae, (jnp.zeros_like(last_val), last_val),
                                         traj, reverse=True, unroll=16)
            targets = jnp.clip(advantages + traj.value, vt_clip_min, vt_clip_max)

            mems_batch = jnp.concatenate(
                [jnp.swapaxes(mems_prev, 0, 1), mems_batch], axis=0)

            # === C. PPO UPDATE ===
            def _update_epoch(update_state, _):
                def _update_minbatch(ts_mb, batch_info):
                    traj_b, mems_b, adv_b, tgt_b, mtok_b = batch_info

                    def _loss_fn(params, traj_b, mems_b, gae, targets, mtok_b):
                        mb = batch_indices_select(
                            mems_b, traj_b.memories_indices[:, ::config.window_grad])
                        mb = batchify(mb)
                        mm = traj_b.memories_mask.reshape(
                            (-1, config.window_grad) + traj_b.memories_mask.shape[2:])
                        mm = jnp.swapaxes(mm, 1, 2)
                        mm = jnp.concatenate((mm, jnp.zeros(
                            mm.shape[:-1] + (config.window_grad - 1,), dtype=jnp.bool_)), axis=-1)
                        mm = roll_vmap(mm, jnp.arange(0, config.window_grad), -1)

                        obs_r = traj_b.obs.reshape(
                            (-1, config.window_grad) + traj_b.obs.shape[2:])
                        traj_r, tgt_r, gae_r = jax.tree_util.tree_map(
                            lambda x: jnp.reshape(x, (-1, config.window_grad) + x.shape[2:]),
                            (traj_b, targets, gae))
                        # RMT16: reshape mem_tokens into windows
                        mtok_r = mtok_b.reshape(
                            (-1, config.window_grad) + mtok_b.shape[2:])

                        pi, value = network.apply(
                            params, mb, obs_r, mm,
                            rmt_tokens_seq=mtok_r,
                            method=network.model_forward_train)
                        log_prob = pi.log_prob(traj_r.action)

                        # Value loss
                        v_clip = traj_r.value + (value - traj_r.value).clip(
                            -config.clip_eps, config.clip_eps)
                        vl = 0.5 * jnp.maximum(
                            jnp.square(value - tgt_r),
                            jnp.square(v_clip - tgt_r)).mean()
                        # Actor loss
                        ratio = jnp.exp(log_prob - traj_r.log_prob)
                        gae_n = (gae_r - gae_r.mean()) / (gae_r.std() + 1e-8)
                        la1 = ratio * gae_n
                        la2 = jnp.clip(ratio, 1 - config.clip_eps, 1 + config.clip_eps) * gae_n
                        la = -jnp.minimum(la1, la2).mean()
                        ent = pi.entropy().mean()
                        total = la + config.vf_coef * vl - config.ent_coef * ent
                        return total, (vl, la, ent)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (total, (vl, la, ent)), grads = grad_fn(
                        ts_mb.params, traj_b, mems_b, adv_b, tgt_b, mtok_b)
                    gn = optax.global_norm(grads)
                    ts_mb = ts_mb.apply_gradients(grads=grads)
                    return ts_mb, (total, vl, la, ent, gn)

                (ts_e, traj_b, mems_b, adv_b, tgt_b, mtok_b,
                 upd_s, rng_e) = update_state
                rng_e, _rng = jax.random.split(rng_e)
                perm = jax.random.permutation(_rng, config.num_envs)
                batch = (traj_b, mems_b, adv_b, tgt_b, mtok_b)
                batch = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), batch)
                batch = jax.tree_util.tree_map(lambda x: jnp.take(x, perm, axis=0), batch)
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [config.num_minibatches, -1] + list(x.shape[1:])),
                    batch)
                ts_e, info = jax.lax.scan(_update_minbatch, ts_e, minibatches)
                return (ts_e, traj_b, mems_b, adv_b, tgt_b, mtok_b,
                        upd_s, rng_e), info

            update_state = (ts, traj, mems_batch, advantages, targets,
                            mtok_seq, upd_step, rng)
            update_state, rl_info = jax.lax.scan(
                _update_epoch, update_state, None, config.update_epochs)
            ts = update_state[0]
            rng = update_state[7]

            losses = rl_info[:4]
            gn = rl_info[4]
            loss_mean = jax.tree_util.tree_map(jnp.mean, losses)
            gn_mean = jnp.mean(gn)
            gn_max = jnp.max(gn)
            metrics = (loss_mean[0], loss_mean[1], loss_mean[2], loss_mean[3],
                       gn_mean, gn_max)

            global_step = (initial_global_update_step + upd_step) * config.num_steps * config.num_envs
            jax.debug.callback(_log_cb, metrics, global_step)
            jax.debug.callback(_guard_cb, loss_mean[2], loss_mean[3])

            carry = (ts, es, mems_f, mmask_f, midx_f, obs_f, dn_f, rmt_st_f,
                     step_f, upd_step + 1, rng)
            return carry, metrics

        def _log_cb(metrics, step):
            pass  # wandb disabled

        def _guard_cb(actor_loss, entropy):
            record_update(float(actor_loss), float(entropy))

        runner_state, scan_info = jax.lax.scan(
            _update_step, runner_state, None, NUM_UPDATES)
        return runner_state, scan_info

    return train
