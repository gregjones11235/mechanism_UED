"""P2-v1 shared rollout/update core.

This module implements the corrected transition collection and training update
order required by the P2-v1 P0 fixes:

  - obs_t, action_t, reward_t, value_t, log_prob_t, next_obs_t are aligned.
  - memory_t is the Transformer memory BEFORE obs_t/action_t.
  - terminal reset memory does not overwrite the terminal transition memory.
  - action sampling uses a checkpointable local RNG (no global np.random).
  - every rollout update runs a native on-policy PPO main update.
  - replay auxiliary update is optional and never replaces the PPO main update.

It does not depend on Craftax; launchers/tests provide a vectorized env with:
    reset(rng) -> (obs, state)
    step(rng, state, actions) -> (obs, state, reward, done, info)
where obs/reward/done have leading env dimension.
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from long_context_learner import LongContextLearner, RolloutBatch
from pending_episodes import PendingEpisodeBuffers
from rng_utils import action_rng_state, sample_actions
from trajectory_replay import Trajectory, TrajectoryReplayBuffer


def collect_rollout(
    ts: TrainState,
    network,
    env,
    env_state,
    obsv,
    memories,
    mem_mask,
    mem_idx,
    rng: jax.random.PRNGKey,
    action_rng: np.random.Generator,
    num_envs: int,
    rollout_steps: int,
    window_mem: int,
    num_heads: int,
    target_achievement: np.ndarray,
    collected_update_count: int = 0,
    pending: Optional[PendingEpisodeBuffers] = None,
) -> dict:
    """Collect one aligned rollout.

    Episode buffers are PERSISTENT across rollouts (方案B): pass the ``pending``
    returned by the previous call so an episode can span several rollouts and
    exceed ``rollout_steps`` in length.  If ``pending`` is None a fresh set of
    per-slot buffers is created (backward-compatible single-rollout behaviour).

    Returns a dict with:
      batch: RolloutBatch
      trajectories: list[Trajectory] for completed episodes in this rollout
      obsv, env_state, memories, mem_mask, mem_idx, rng: updated env/memory state
      action_rng_state: checkpointable action RNG state after collection
      pending: PendingEpisodeBuffers carrying any in-progress episodes over to
               the next rollout (checkpointable)
    """

    @jax.jit
    def jit_forward(params, mem, obs, mask):
        pi, value, mem_out = network.apply(
            params, mem, obs, mask, method=network.model_forward_eval)
        return pi.logits, value, mem_out

    # Persistent per-slot episode buffers carried across rollouts (方案B).
    if pending is None:
        pending = PendingEpisodeBuffers(
            num_envs, first_episode_id=0,
            first_policy_version=int(collected_update_count))

    all_obs = []
    all_act = []
    all_rew = []
    all_don = []
    all_val = []
    all_lp = []
    all_next_obs = []
    all_mem_pre = []
    all_mask_pre = []

    trajectories = []

    for _ in range(rollout_steps):
        # Memory index/mask for the current pre-step memory.
        mem_idx = jnp.clip(mem_idx - 1, 0, window_mem)
        ohot = jax.nn.one_hot(mem_idx, window_mem + 1)
        ohot = ohot[:, None, None, :].repeat(num_heads, 1)
        mem_mask = jnp.logical_or(mem_mask, ohot)

        # Pre-step memory is the memory used to choose this step's action.
        mem_pre = np.asarray(memories).copy()
        mask_pre = np.asarray(mem_mask).copy()

        rng, a_rng = jax.random.split(rng)
        logits, value, mem_out = jit_forward(ts.params, memories, obsv, mem_mask)
        logits_np = np.asarray(logits)
        value_np = np.asarray(value)
        mem_out_np = np.asarray(mem_out)

        probs = np.asarray(jax.nn.softmax(jnp.asarray(logits), axis=-1))
        actions_np = sample_actions(action_rng, probs)
        logp_np = np.log(probs[np.arange(num_envs), actions_np] + 1e-12)

        obs_pre = np.asarray(obsv).copy()

        rng, s_rng = jax.random.split(rng)
        next_obsv, env_state, reward_j, done_j, info = env.step(
            s_rng, env_state, actions_np)
        reward_np = np.asarray(reward_j)
        done_np = np.asarray(done_j).astype(bool)
        next_obs_np = np.asarray(next_obsv).copy()

        # Post-step memory is the memory AFTER processing obs_t/action_t.
        post_memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(mem_out_np)

        ach_data = np.zeros((num_envs, target_achievement.shape[-1]), dtype=np.float32)
        try:
            est = env_state.env_state
            if hasattr(est, "achievements"):
                ach_data = np.asarray(est.achievements).astype(np.float32)
        except Exception:
            pass

        all_obs.append(obs_pre)
        all_act.append(actions_np)
        all_rew.append(reward_np)
        all_don.append(done_np)
        all_val.append(value_np)
        all_lp.append(logp_np)
        all_next_obs.append(next_obs_np)
        all_mem_pre.append(mem_pre)
        all_mask_pre.append(mask_pre)

        for e in range(num_envs):
            buf = pending.slots[e]
            if buf["init_mem"] is None:
                # Memory before the first step of the episode.
                buf["init_mem"] = mem_pre[e].copy()
            buf["obs"].append(obs_pre[e].copy())
            buf["act"].append(int(actions_np[e]))
            buf["rew"].append(float(reward_np[e]))
            buf["don"].append(bool(done_np[e]))
            buf["val"].append(float(value_np[e]))
            buf["lp"].append(float(logp_np[e]))
            buf["next_obs"].append(next_obs_np[e].copy())
            # Memory AFTER step t; for terminal t this is the terminal memory,
            # not the reset-zero memory.
            buf["mem_pre"].append(np.asarray(post_memories[e]).copy())
            buf["mask_pre"].append(mask_pre[e].copy())
            buf["ach"].append(ach_data[e].copy())

            if done_np[e]:
                L = len(buf["obs"])
                if L > 0 and buf["init_mem"] is not None:
                    mem_seq = np.stack(buf["mem_pre"])
                    traj = Trajectory(
                        observations=np.stack(buf["obs"]),
                        actions=np.array(buf["act"], dtype=np.int32),
                        rewards=np.array(buf["rew"], dtype=np.float32),
                        dones=np.array(buf["don"], dtype=bool),
                        values=np.array(buf["val"], dtype=np.float32),
                        log_probs=np.array(buf["lp"], dtype=np.float32),
                        initial_memory=buf["init_mem"],
                        achievements=np.stack(buf["ach"]),
                        target_achievements=np.asarray(target_achievement),
                        memory_sequence=mem_seq,
                        next_observations=np.stack(buf["next_obs"]),
                        collected_update_count=int(collected_update_count),
                    )
                    trajectories.append(traj)
                # Auto-reset isolation: the terminal transition stays in the
                # finished episode; the reset observation begins a NEW episode
                # (fresh buffer + new episode_id) on the next step.
                pending.reset_slot(e, policy_version=int(collected_update_count))

        # Terminal reset happens AFTER the terminal transition's memory has
        # been recorded above, so reset cannot overwrite terminal memory.
        memories = jnp.where(
            done_np[:, None, None, None],
            jnp.zeros_like(post_memories), post_memories)
        mem_mask = jnp.where(
            done_np[:, None, None, None],
            jnp.zeros_like(mem_mask), mem_mask)
        mem_idx = jnp.where(done_np, window_mem, mem_idx)
        obsv = next_obsv

    batch = RolloutBatch(
        obs=jnp.asarray(np.stack(all_obs, axis=1)),
        action=jnp.asarray(np.stack(all_act, axis=1)),
        reward=jnp.asarray(np.stack(all_rew, axis=1)),
        done=jnp.asarray(np.stack(all_don, axis=1)),
        value=jnp.asarray(np.stack(all_val, axis=1)),
        log_prob=jnp.asarray(np.stack(all_lp, axis=1)),
        next_obs=jnp.asarray(np.stack(all_next_obs, axis=1)),
        memory=jnp.asarray(np.stack(all_mem_pre, axis=1)),
        mask=jnp.asarray(np.stack(all_mask_pre, axis=1)),
        memory_after_final=np.asarray(memories),
        mask_after_final=np.asarray(mem_mask),
        num_envs=num_envs,
        num_steps=rollout_steps,
    )

    return {
        "batch": batch,
        "trajectories": trajectories,
        "obsv": obsv,
        "env_state": env_state,
        "memories": memories,
        "mem_mask": mem_mask,
        "mem_idx": mem_idx,
        "rng": rng,
        "action_rng_state": action_rng_state(action_rng),
        "pending": pending,
    }


def p2_v1_update(
    ts: TrainState,
    learner: LongContextLearner,
    batch: RolloutBatch,
    replay: TrajectoryReplayBuffer,
    update_count: int,
    replay_aux: bool = True,
    relabel_callback=None,
) -> Tuple[TrainState, dict]:
    """Run the P2-v1 update order: PPO main update then optional replay aux."""

    advantages, targets = learner.compute_on_policy_gae(ts.params, batch)
    ts, ppo_metrics = learner.ppo_update(ts, batch, advantages, targets)
    update_count += 1

    metrics = dict(ppo_metrics)
    metrics["update_count"] = int(update_count)
    metrics["on_policy_main_update"] = True
    metrics["replay_aux_update"] = False

    if replay_aux and replay.can_sample():
        sample = replay.sample()
        if relabel_callback is not None:
            sample, relabel_info = relabel_callback(sample)
            metrics.update(relabel_info)
        ts, aux_metrics = learner.update(
            ts, sample,
            current_update_count=update_count,
            replay_actor_update=False,
        )
        update_count += 1
        metrics["replay_aux_update"] = True
        metrics["replay_aux_total_loss"] = aux_metrics["total_loss"]
        metrics["replay_aux_value_loss"] = aux_metrics["value_loss"]
        metrics["replay_aux_actor_loss"] = aux_metrics["actor_loss"]
        metrics["replay_aux_actor_enabled"] = aux_metrics["actor_enabled"]
        metrics["replay_aux_grad_norm"] = aux_metrics["grad_norm"]
        metrics["replay_aux_seq_len"] = aux_metrics["sequence_length"]
        metrics["replay_aux_episode_done"] = aux_metrics["episode_done"]
        metrics["replay_aux_bootstrap_value"] = aux_metrics["bootstrap_value"]
        metrics["replay_aux_behavior_log_prob_mean"] = aux_metrics["behavior_log_prob_mean"]
        metrics["replay_aux_policy_version"] = aux_metrics["policy_version"]
        metrics["replay_aux_policy_lag"] = aux_metrics["policy_lag"]
        metrics["replay_aux_importance_ratio_mean"] = aux_metrics["importance_ratio_mean"]
        metrics["replay_aux_ess_fraction"] = aux_metrics["ess_fraction"]
        metrics["replay_aux_hindsight_goal_index"] = aux_metrics["hindsight_goal_index"]
        metrics["update_count"] = int(update_count)

    return ts, metrics
