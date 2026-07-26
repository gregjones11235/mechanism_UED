"""LC-W512 PPO trainer: Original Henry PPO + 384-step raw history.

Based on dicode/ppo_tr.py (read-only).  Adds:
  - w512_state carry through rollout (delay buffer + long buffer)
  - long buffer snapshot stored per step for PPO training forward pass
  - w512_state checkpoint save/restore

Hyperparams MUST match canonical Control freeze record:
  update_epochs=1, num_minibatches=2, lr=2e-5 constant,
  gamma=0.999, gae_lambda=0.8, clip=0.2, ent_coef=0.002,
  vf_coef=0.5, max_grad_norm=1.0, num_envs=16, num_steps=128,
  optimistic_reset_ratio=16, mode=score, seed=42.
"""
import functools, time
import distrax, flax.linen as nn, jax, jax.numpy as jnp, numpy as np, optax
from flax.training.train_state import TrainState
from flax import struct

from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv
from dicode.wrappers_cl import DistributedMultiTaskOptimisticLogWrapper
from dicode.train_guard import record_update, reset_session_stats

from network_w512 import ActorCriticTransformerW512
import w512_memory as w5m

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

    # ---- W512 config ----
    w5_cfg = w5m.W512Config(
        long_size=getattr(config, "w512_long_size", 384),
        delay_size=getattr(config, "w512_delay_size", 128),
        encoder_size=config.embed_size)

    network = ActorCriticTransformerW512(
        action_dim=env.action_space(env_params).n,
        activation=config.activation,
        hidden_layers=config.hidden_layers,
        encoder_size=config.embed_size,
        num_heads=config.num_heads,
        qkv_features=config.qkv_features,
        num_layers=config.num_layers,
        gating=config.gating,
        gating_bias=config.gating_bias,
        long_size=w5_cfg.long_size)

    TOTAL_GLOBAL_UPDATES = (
        (config.total_timesteps // config.num_envs // config.num_steps
         // config.max_updates_per_session) + 1
    ) * config.max_updates_per_session

    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.lr, eps=1e-5))

    def train(rng, train_state=None, w512_state_init=None,
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
                init_lbuf = jnp.zeros((2, w5_cfg.long_size, config.embed_size))
                init_lmsk = jnp.zeros((2, w5_cfg.long_size), dtype=jnp.bool_)
                network_params = network.init(
                    _rng, init_mem, init_obs, init_mask,
                    long_buf=init_lbuf, long_mask=init_lmsk)
                train_state = TrainState.create(
                    apply_fn=network.apply, params=network_params, tx=tx)

            rng, _rng = jax.random.split(rng)
            obsv, env_state = env.reset(_rng, env_params)

            memories = jnp.zeros((config.num_envs, config.window_mem, config.num_layers, config.embed_size))
            memories_mask = jnp.zeros((config.num_envs, config.num_heads, 1, config.window_mem + 1), dtype=jnp.bool_)
            memories_mask_idx = jnp.full((config.num_envs,), config.window_mem + 1, dtype=jnp.int32)
            done = jnp.zeros((config.num_envs,), dtype=jnp.bool_)

            if w512_state_init is None:
                w512_state = w5m.w512_init(config.num_envs, w5_cfg)
            else:
                w512_state = w512_state_init

            runner_state = (train_state, env_state, memories, memories_mask,
                            memories_mask_idx, obsv, done, w512_state,
                            0, 0, _rng)

        def _update_step(runner_state, _unused):
            # === A. ROLLOUT ===
            def _env_step(carry, _):
                (ts, es, mems, mmask, midx, obs, dn, w5st,
                 step_loop, upd_step, rng) = carry

                midx = jnp.where(dn, config.window_mem,
                                 jnp.clip(midx - 1, 0, config.window_mem))
                mmask = jnp.where(dn[:, None, None, None],
                                  jnp.zeros_like(mmask), mmask)
                ohot = jax.nn.one_hot(midx, config.window_mem + 1)
                ohot = ohot[:, None, None, :].repeat(config.num_heads, 1)
                mmask = jnp.logical_or(mmask, ohot)

                # W512: read long buffer BEFORE forward
                lbuf  = w5st["long_buf"]
                lmask = w5st["long_mask"]

                rng, _rng = jax.random.split(rng)
                pi, value, mem_out, h_t = network.apply(
                    ts.params, mems, obs, mmask,
                    long_buf=lbuf, long_mask=lmask,
                    method=network.model_forward_eval)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                mems = jnp.roll(mems, -1, axis=1).at[:, -1].set(mem_out)

                rng, _rng = jax.random.split(rng)
                obs_next, es, reward, dn, info = env.step(_rng, es, action, env_params)

                # W512: update delay/long buffers with pre-fusion h_t
                w5st_new = w5m.w512_step(w5st, h_t, dn, w5_cfg)

                mem_indices = (jnp.arange(0, config.window_mem)[None, :]
                               + step_loop * jnp.ones((config.num_envs, 1), dtype=jnp.int32))
                transition = Transition(dn, action, value, reward, log_prob,
                                        mmask.squeeze(), mem_indices, obs, info)
                carry = (ts, es, mems, mmask, midx, obs_next, dn, w5st_new,
                         step_loop + 1, upd_step, rng)
                # Store long buffer snapshot for training forward pass
                return carry, (transition, mem_out, lbuf, lmask)

            mems_prev = runner_state[2]

            (final_carry), (traj, mems_batch, lbuf_seq, lmask_seq) = \
                jax.lax.scan(_env_step, runner_state, None, config.num_steps)

            (ts, es, mems_f, mmask_f, midx_f, obs_f, dn_f, w5st_f,
             step_f, upd_step, rng) = final_carry

            # === B. GAE ===
            lbuf_f  = w5st_f["long_buf"]
            lmask_f = w5st_f["long_mask"]
            _, last_val, _, _ = network.apply(
                ts.params, mems_f, obs_f, mmask_f,
                long_buf=lbuf_f, long_mask=lmask_f,
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
                    traj_b, mems_b, adv_b, tgt_b, lbuf_b, lmask_b = batch_info

                    def _loss_fn(params, traj_b, mems_b, gae, targets, lbuf_b, lmask_b):
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
                        # W512: reshape long buffer into windows
                        lbuf_r = lbuf_b.reshape(
                            (-1, config.window_grad) + lbuf_b.shape[2:])
                        lmask_r = lmask_b.reshape(
                            (-1, config.window_grad) + lmask_b.shape[2:])

                        pi, value = network.apply(
                            params, mb, obs_r, mm,
                            w512_buf_seq=lbuf_r, w512_mask_seq=lmask_r,
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
                        ts_mb.params, traj_b, mems_b, adv_b, tgt_b, lbuf_b, lmask_b)
                    gn = optax.global_norm(grads)
                    ts_mb = ts_mb.apply_gradients(grads=grads)
                    return ts_mb, (total, vl, la, ent, gn)

                (ts_e, traj_b, mems_b, adv_b, tgt_b, lbuf_b, lmask_b,
                 upd_s, rng_e) = update_state
                rng_e, _rng = jax.random.split(rng_e)
                perm = jax.random.permutation(_rng, config.num_envs)
                batch = (traj_b, mems_b, adv_b, tgt_b, lbuf_b, lmask_b)
                batch = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), batch)
                batch = jax.tree_util.tree_map(lambda x: jnp.take(x, perm, axis=0), batch)
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [config.num_minibatches, -1] + list(x.shape[1:])),
                    batch)
                ts_e, info = jax.lax.scan(_update_minbatch, ts_e, minibatches)
                return (ts_e, traj_b, mems_b, adv_b, tgt_b, lbuf_b, lmask_b,
                        upd_s, rng_e), info

            update_state = (ts, traj, mems_batch, advantages, targets,
                            lbuf_seq, lmask_seq, upd_step, rng)
            update_state, rl_info = jax.lax.scan(
                _update_epoch, update_state, None, config.update_epochs)
            ts = update_state[0]
            rng = update_state[8]

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

            carry = (ts, es, mems_f, mmask_f, midx_f, obs_f, dn_f, w5st_f,
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
