"""W512 × P2 Replay rollout collection: complete-episode trajectories with SPARSE
anchors for BOTH GTrXL memory AND W512 long state.

Adapted from p2_full_20260723/src/full_p2_core.py (frozen P2-Full-A).
Changes:
  - forward_eval returns (logits, value, mem_out, h_t) via W512 network
  - w512_step updates long_buf/long_mask/delay after each step
  - Anchors store W512 long state (long_buf, long_mask, delay_buf, delay_idx,
    delay_count, long_idx) alongside GTrXL memory
  - carry_mode: "persistent" (carry across rollout boundary) or "reset128"
    (clear long_buf/long_mask at rollout boundary)

P2-Full-A replay hyperparameters are UNCHANGED (frozen).
"""
import numpy as np
import jax
import jax.numpy as jnp

import rng_utils as RU
from w512_replay_buffer import W512Trajectory, ANCHOR_INTERVAL


def make_jit_forward_w512(network, w5_cfg):
    """JIT'd W512 forward: (params, mem, obs, mask, long_buf, long_mask)
    -> (logits, value, mem_out, h_t)."""
    @jax.jit
    def jit_forward(params, mem, obs, mask, long_buf, long_mask):
        pi, value, mem_out, h_t = network.apply(
            {"params": params}, mem, obs, mask,
            long_buf=long_buf, long_mask=long_mask,
            method=network.model_forward_eval)
        return pi.logits, value, mem_out, h_t
    return jit_forward


def collect_rollout_w512(
    env, env_state, network, params, obsv,
    memories, mem_mask, mem_idx,
    w512_state,
    rng, action_rng,
    pending,
    target_achievement,
    rollout_steps: int,
    window_mem: int, num_heads: int,
    w5_cfg,
    carry_mode: str = "persistent",
    collected_update_count: int = 0,
    jit_forward=None,
):
    """Run rollout_steps vectorized env steps with W512 forward; emit completed
    W512Trajectories (with GTrXL + W512 anchors).

    carry_mode:
      "persistent" - long_buf/long_mask persist across rollout boundary (only
                     true done resets)
      "reset128"   - long_buf/long_mask cleared at rollout boundary (start of
                     this function call)

    Returns (trajectories, carry, stats).
    """
    import w512_memory as w5m

    num_envs = int(np.asarray(obsv).shape[0])
    if jit_forward is None:
        jit_forward = make_jit_forward_w512(network, w5_cfg)

    # RESET128: clear long_buf/long_mask at rollout boundary
    if carry_mode == "reset128":
        w512_state = {**w512_state,
            "long_buf":  jnp.zeros_like(w512_state["long_buf"]),
            "long_mask": jnp.zeros_like(w512_state["long_mask"])}

    trajectories = []
    ep_returns, ep_lengths = [], []

    for step_i in range(rollout_steps):
        # ---- snapshot ENTERING state for anchor capture ----
        mem_in = np.asarray(memories).copy()
        mask_in = np.asarray(mem_mask).copy()
        idx_in = np.asarray(mem_idx).copy()
        # W512 long state snapshot for anchors
        w5_in = {k: np.asarray(v).copy() for k, v in w512_state.items()}

        for e in range(num_envs):
            k = len(pending.slots[e]["obs"])
            if k % ANCHOR_INTERVAL == 0:
                pending.add_anchor(e, k, mem_in[e].copy(), mask_in[e].copy(),
                                   int(idx_in[e]))
                # Store W512 long state at anchor
                pending.slots[e]["w512_anchor_state"].append({
                    kk: vv[e].copy() for kk, vv in w5_in.items()
                })

        # ---- GTrXL mask/idx advance ----
        mem_idx = jnp.clip(mem_idx - 1, 0, window_mem)
        ohot = jax.nn.one_hot(mem_idx, window_mem + 1)
        ohot = ohot[:, None, None, :].repeat(num_heads, 1)
        mem_mask = jnp.logical_or(mem_mask, ohot)

        # ---- W512 forward ----
        rng, a_rng = jax.random.split(rng)
        logits, value, mem_out, h_t = jit_forward(
            params, memories, obsv, mem_mask,
            w512_state["long_buf"], w512_state["long_mask"])
        value_np = np.asarray(value)
        mem_out_np = np.asarray(mem_out)
        probs = np.asarray(jax.nn.softmax(jnp.asarray(logits), axis=-1))
        actions_np = RU.sample_actions(action_rng, probs)
        logp_np = np.log(probs[np.arange(num_envs), actions_np] + 1e-12)
        obs_pre = np.asarray(obsv).copy()

        rng, s_rng = jax.random.split(rng)
        next_obsv, env_state, reward_j, done_j, info = env.step(
            s_rng, env_state, actions_np)
        reward_np = np.asarray(reward_j)
        done_np = np.asarray(done_j).astype(bool)
        next_obs_np = np.asarray(next_obsv).copy()
        post_memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(mem_out_np)

        # ---- W512 long state update ----
        w512_state = w5m.w512_step(w512_state, h_t, done_j, w5_cfg)

        # ---- achievement data ----
        ach_data = np.zeros((num_envs, target_achievement.shape[-1]), np.float32)
        try:
            est = env_state.env_state
            if hasattr(est, "achievements"):
                ach_data = np.asarray(est.achievements).astype(np.float32)
        except Exception:
            pass

        # ---- per-env episode bookkeeping ----
        for e in range(num_envs):
            buf = pending.slots[e]
            if buf["init_mem"] is None:
                buf["init_mem"] = mem_in[e].copy()
                buf["init_w512_state"] = {
                    kk: vv[e].copy() for kk, vv in w5_in.items()}
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
                    traj = W512Trajectory(
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
                        # W512 extensions
                        init_w512_state=buf.get("init_w512_state"),
                        w512_anchor_states=list(buf.get("w512_anchor_state", [])),
                    )
                    trajectories.append(traj)
                    ep_returns.append(float(np.sum(buf["rew"])))
                    ep_lengths.append(int(L))
                pending.reset_slot_w512(e, policy_version=int(collected_update_count))

        # ---- terminal reset (GTrXL memory) ----
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
             "mem_mask": mem_mask, "mem_idx": mem_idx, "rng": rng,
             "w512_state": w512_state}
    return trajectories, carry, stats
