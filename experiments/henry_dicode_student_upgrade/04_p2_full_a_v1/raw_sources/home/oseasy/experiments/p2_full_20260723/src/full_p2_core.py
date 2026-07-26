"""P2-Full-A rollout collection: complete-episode trajectories with SPARSE anchors.

Mirrors p2_v1_core.collect_rollout's exact memory/mask/idx dynamics, but:
  * NO per-step memory_sequence storage; instead a pre-action ENTERING-state anchor
    (memory, mask, idx) is snapshotted at episode steps 0,128,256,... (the state at
    the TOP of the iteration, before the mem_idx decrement — bit-exact the state
    memory_anchor.derive_anchor_entering_state reproduces, see Gate G1.4).
  * Episodes persist across rollouts via PendingEpisodeBuffers, so a completed episode
    can exceed rollout_steps and reach MIN_SEQUENCE_LENGTH=129 for replay.
  * Action sampling uses the checkpointable rng_utils Generator (no global np.random).
  * On done, a Trajectory (with anchors) is emitted; auto-reset begins a new episode.

This is the REPLAY data channel for the P2-Full-A combined update (V-trace + AWR);
there is no separate on-policy PPO main update in P2-Full-A — V-trace's rho correction
handles the on->off correction on the original-goal replay trajectories.
"""
import numpy as np
import jax
import jax.numpy as jnp

import rng_utils as RU
from pending_episodes import PendingEpisodeBuffers
from replay_buffer import Trajectory, ANCHOR_INTERVAL


def make_jit_forward(network):
    @jax.jit
    def jit_forward(params, mem, obs, mask):
        pi, value, mem_out = network.apply(
            {"params": params}, mem, obs, mask, method=network.model_forward_eval)
        return pi.logits, value, mem_out
    return jit_forward


def collect_rollout(
    env, env_state, network, params, obsv,
    memories, mem_mask, mem_idx,
    rng, action_rng,
    pending: PendingEpisodeBuffers,
    target_achievement,
    rollout_steps: int,
    window_mem: int, num_heads: int,
    collected_update_count: int = 0,
    jit_forward=None,
):
    """Run rollout_steps vectorized env steps; emit completed Trajectories (w/ anchors).

    Returns (trajectories, carry, stats). `carry` holds the updated env/memory/rng
    state to feed the next rollout (memory/mask/idx persist across rollouts).
    """
    num_envs = int(np.asarray(obsv).shape[0])
    if jit_forward is None:
        jit_forward = make_jit_forward(network)

    trajectories = []
    ep_returns, ep_lengths = [], []

    for _ in range(rollout_steps):
        # ---- snapshot ENTERING state (pre-decrement) for anchor capture ----
        mem_in = np.asarray(memories).copy()       # [E, wm, layers, embed]
        mask_in = np.asarray(mem_mask).copy()      # [E, heads, 1, wm+1]
        idx_in = np.asarray(mem_idx).copy()        # [E]
        for e in range(num_envs):
            k = len(pending.slots[e]["obs"])       # episode-local step about to be collected
            if k % ANCHOR_INTERVAL == 0:
                pending.add_anchor(e, k, mem_in[e].copy(), mask_in[e].copy(), int(idx_in[e]))

        # ---- exact rollout dynamics ----
        mem_idx = jnp.clip(mem_idx - 1, 0, window_mem)
        ohot = jax.nn.one_hot(mem_idx, window_mem + 1)
        ohot = ohot[:, None, None, :].repeat(num_heads, 1)
        mem_mask = jnp.logical_or(mem_mask, ohot)

        rng, a_rng = jax.random.split(rng)
        logits, value, mem_out = jit_forward(params, memories, obsv, mem_mask)
        value_np = np.asarray(value); mem_out_np = np.asarray(mem_out)
        probs = np.asarray(jax.nn.softmax(jnp.asarray(logits), axis=-1))
        actions_np = RU.sample_actions(action_rng, probs)
        logp_np = np.log(probs[np.arange(num_envs), actions_np] + 1e-12)
        obs_pre = np.asarray(obsv).copy()

        rng, s_rng = jax.random.split(rng)
        next_obsv, env_state, reward_j, done_j, info = env.step(s_rng, env_state, actions_np)
        reward_np = np.asarray(reward_j)
        done_np = np.asarray(done_j).astype(bool)
        next_obs_np = np.asarray(next_obsv).copy()
        post_memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(mem_out_np)

        ach_data = np.zeros((num_envs, target_achievement.shape[-1]), np.float32)
        try:
            est = env_state.env_state
            if hasattr(est, "achievements"):
                ach_data = np.asarray(est.achievements).astype(np.float32)
        except Exception:
            pass

        for e in range(num_envs):
            buf = pending.slots[e]
            if buf["init_mem"] is None:
                buf["init_mem"] = mem_in[e].copy()       # entering memory at episode step 0
            buf["obs"].append(obs_pre[e].copy())
            buf["act"].append(int(actions_np[e]))
            buf["rew"].append(float(reward_np[e]))
            buf["don"].append(bool(done_np[e]))
            buf["val"].append(float(value_np[e]))
            buf["lp"].append(float(logp_np[e]))
            buf["next_obs"].append(next_obs_np[e].copy())
            buf["ach"].append(ach_data[e].copy())

            if done_np[e]:
                L = len(buf["obs"])
                if L > 0 and buf["init_mem"] is not None:
                    traj = Trajectory(
                        observations=np.stack(buf["obs"]),
                        actions=np.array(buf["act"], np.int32),
                        rewards=np.array(buf["rew"], np.float32),
                        dones=np.array(buf["don"], bool),
                        values=np.array(buf["val"], np.float32),
                        log_probs=np.array(buf["lp"], np.float32),
                        initial_memory=buf["init_mem"],
                        achievements=np.stack(buf["ach"]),
                        target_achievements=np.asarray(target_achievement),
                        next_observations=np.stack(buf["next_obs"]),
                        memory_anchors=np.stack(buf["anchor_mem"]),
                        anchor_steps=np.array(buf["anchor_step"], np.int64),
                        anchor_masks=np.stack(buf["anchor_mask"]),
                        anchor_idxs=np.array(buf["anchor_idx"], np.int64),
                        collected_update_count=int(collected_update_count),
                    )
                    trajectories.append(traj)
                    ep_returns.append(float(np.sum(buf["rew"])))
                    ep_lengths.append(int(L))
                pending.reset_slot(e, policy_version=int(collected_update_count))

        # terminal reset AFTER the terminal transition is recorded (isolation)
        memories = jnp.where(done_np[:, None, None, None],
                             jnp.zeros_like(post_memories), post_memories)
        mem_mask = jnp.where(done_np[:, None, None, None],
                             jnp.zeros_like(mem_mask), mem_mask)
        mem_idx = jnp.where(done_np, window_mem, mem_idx)
        obsv = next_obsv

    stats = {
        "completed_episodes": len(trajectories),
        "mean_ep_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
        "mean_ep_length": float(np.mean(ep_lengths)) if ep_lengths else 0.0,
        "pending_transitions": pending.total_pending_transitions(),
        "pending_anchors": pending.total_pending_anchors(),
    }
    carry = {"env_state": env_state, "obsv": obsv, "memories": memories,
             "mem_mask": mem_mask, "mem_idx": mem_idx, "rng": rng}
    return trajectories, carry, stats
