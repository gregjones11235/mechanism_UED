"""RMT16 × P2-Replay — rollout collection emitting RMTTrajectories with sparse anchors.

EXTENDS (does not modify) the frozen P2-Full-A `full_p2_core.collect_rollout`. Mirrors its
exact GTrXL memory/mask/idx dynamics and complete-episode pending-buffer logic, but the
network is RMT16 (forward returns h_t and reads 16 persistent tokens) and each anchor
ALSO snapshots the RMT state (mem_tokens, seg_buf, seg_count). The single per-step RMT
transition is rmt_memory_anchor.rmt_advance_tokens — the SAME function reconstruction
uses, so anchor round-trip is bit-exact by construction (directive gate 6/7).

carry_mode (directive §二/§五, the ONLY Persistent vs Reset128 difference):
  * persistent : tokens updated at each 128-step segment boundary and carried across
                 rollout boundaries; cleared only on TRUE done.
  * reset128   : tokens cleared to 0 at every 128-step segment boundary (single-window
                 read/write only); also cleared on true done.
After a true done the env's RMT state is fully reset so the new episode's step-0 anchor
is the fresh initial state (episode-start anchor == initial state, P2 conservation).
"""
import numpy as np
import jax
import jax.numpy as jnp

import rng_utils as RU
from pending_episodes import PendingEpisodeBuffers
from replay_buffer import ANCHOR_INTERVAL
from rmt_replay_buffer import RMTTrajectory
import rmt16_memory as rmtm
from rmt_memory_anchor import make_apply_eval_rmt, make_update_fn, rmt_advance_tokens


def _fresh_rmt_slot() -> dict:
    return {
        "obs": [], "act": [], "rew": [], "don": [], "val": [], "lp": [],
        "next_obs": [], "ach": [],
        "init_mem": None,
        "anchor_mem": [], "anchor_mask": [], "anchor_idx": [], "anchor_step": [],
        # RMT additions
        "init_rmt_tokens": None, "init_rmt_segbuf": None, "init_rmt_segcount": None,
        "anchor_rmt_tokens": [], "anchor_rmt_segbuf": [], "anchor_rmt_segcount": [],
    }


class RMTPendingEpisodeBuffers(PendingEpisodeBuffers):
    """Pending buffers carrying BOTH GTrXL and RMT anchors per env slot."""

    def __init__(self, num_envs, first_episode_id=0, first_policy_version=0):
        self.num_envs = int(num_envs)
        self.next_episode_id = int(first_episode_id)
        self.slots = []
        self.episode_id = []
        self.policy_version = []
        for _ in range(self.num_envs):
            self.slots.append(_fresh_rmt_slot())
            self.episode_id.append(self.next_episode_id)
            self.policy_version.append(int(first_policy_version))
            self.next_episode_id += 1

    def reset_slot(self, e, policy_version):
        self.slots[e] = _fresh_rmt_slot()
        self.episode_id[e] = self.next_episode_id
        self.policy_version[e] = int(policy_version)
        self.next_episode_id += 1
        return self.episode_id[e]

    def add_rmt_anchor(self, e, tokens, segbuf, segcount):
        s = self.slots[e]
        s["anchor_rmt_tokens"].append(np.asarray(tokens).copy())
        s["anchor_rmt_segbuf"].append(np.asarray(segbuf).copy())
        s["anchor_rmt_segcount"].append(int(segcount))

    def total_pending_rmt_anchors(self):
        return sum(len(s["anchor_rmt_tokens"]) for s in self.slots)


def collect_rollout_rmt(
    env, env_state, network, params, obsv,
    memories, mem_mask, mem_idx, rmt_state,
    rng, action_rng,
    pending: RMTPendingEpisodeBuffers,
    target_achievement,
    rollout_steps: int,
    window_mem: int, num_heads: int,
    rmt_cfg, carry_mode: str,
    collected_update_count: int = 0,
    apply_eval_rmt=None,
    env_params=None,  # Craftax EnvParams (NOT network params); driver must pass explicitly
):
    """Run rollout_steps vectorized env steps; emit completed RMTTrajectories (w/ anchors).

    Returns (trajectories, carry, stats). carry holds env/GTrXL/RMT/rng state for the next
    rollout (memory/mask/idx AND rmt_state persist across rollouts for Persistent)."""
    assert env_params is not None, 'collect_rollout_rmt: env_params (Craftax EnvParams) must be passed explicitly'
    num_envs = int(np.asarray(obsv).shape[0])
    if apply_eval_rmt is None:
        apply_eval_rmt = make_apply_eval_rmt(network)
    update_fn = make_update_fn(network, params)

    trajectories = []
    ep_returns, ep_lengths = [], []

    # ---- rollout-start ENTERING state (for the on-policy PPO main update scan) ----
    rollout_start = {
        "memories": jnp.asarray(memories), "mem_mask": jnp.asarray(mem_mask),
        "mem_idx": jnp.asarray(mem_idx),
        "rmt_state": jax.tree_util.tree_map(jnp.asarray, rmt_state),
    }
    rl_obs, rl_act, rl_val, rl_rew, rl_lp, rl_don = [], [], [], [], [], []

    for _ in range(rollout_steps):
        # ---- snapshot ENTERING state (pre-action) for anchor capture ----
        mem_in = np.asarray(memories).copy()
        mask_in = np.asarray(mem_mask).copy()
        idx_in = np.asarray(mem_idx).copy()
        tok_in = np.asarray(rmt_state["mem_tokens"]).copy()
        segbuf_in = np.asarray(rmt_state["seg_buf"]).copy()
        segcount_in = np.asarray(rmt_state["seg_count"]).copy()
        for e in range(num_envs):
            k = len(pending.slots[e]["obs"])       # episode-local step about to be collected
            if k % ANCHOR_INTERVAL == 0:
                pending.add_anchor(e, k, mem_in[e].copy(), mask_in[e].copy(), int(idx_in[e]))
                pending.add_rmt_anchor(e, tok_in[e].copy(), segbuf_in[e].copy(), int(segcount_in[e]))

        # ---- forward (read ENTERING tokens) ----
        rng, a_rng = jax.random.split(rng)
        logits, value, mem_out, h_t = apply_eval_rmt(
            params, memories, obsv, mem_mask, rmt_state["mem_tokens"])
        value_np = np.asarray(value); mem_out_np = np.asarray(mem_out)
        probs = np.asarray(jax.nn.softmax(jnp.asarray(logits), axis=-1))
        actions_np = RU.sample_actions(action_rng, probs)
        logp_np = np.log(probs[np.arange(num_envs), actions_np] + 1e-12)
        obs_pre = np.asarray(obsv).copy()

        rng, s_rng = jax.random.split(rng)
        next_obsv, env_state, reward_j, done_j, info = env.step(s_rng, env_state, actions_np, env_params)
        reward_np = np.asarray(reward_j)
        done_np = np.asarray(done_j).astype(bool)
        next_obs_np = np.asarray(next_obsv).copy()
        h_t_j = jnp.asarray(h_t)

        # ---- accumulate rollout-aligned per-step arrays (for the PPO main update) ----
        rl_obs.append(obs_pre.copy()); rl_act.append(actions_np.copy())
        rl_val.append(value_np.copy()); rl_rew.append(reward_np.copy())
        rl_lp.append(logp_np.copy()); rl_don.append(done_np.copy())

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
                buf["init_mem"] = mem_in[e].copy()
                buf["init_rmt_tokens"] = tok_in[e].copy()
                buf["init_rmt_segbuf"] = segbuf_in[e].copy()
                buf["init_rmt_segcount"] = int(segcount_in[e])
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
                    traj = RMTTrajectory(
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
                        rmt_initial_tokens=buf["init_rmt_tokens"],
                        rmt_initial_segbuf=buf["init_rmt_segbuf"],
                        rmt_initial_segcount=int(buf["init_rmt_segcount"]),
                        rmt_anchor_tokens=np.stack(buf["anchor_rmt_tokens"]),
                        rmt_anchor_segbuf=np.stack(buf["anchor_rmt_segbuf"]),
                        rmt_anchor_segcount=np.array(buf["anchor_rmt_segcount"], np.int64),
                    )
                    trajectories.append(traj)
                    ep_returns.append(float(np.sum(buf["rew"])))
                    ep_lengths.append(int(L))
                pending.reset_slot(e, policy_version=int(collected_update_count))

        # ---- advance GTrXL memory (terminal reset AFTER recording; isolation) ----
        post_memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(mem_out_np)
        memories = jnp.where(done_np[:, None, None, None], jnp.zeros_like(post_memories), post_memories)
        mem_mask = jnp.where(done_np[:, None, None, None], jnp.zeros_like(mem_mask), mem_mask)
        mem_idx = jnp.where(done_np, window_mem, mem_idx)
        # ---- advance RMT state (shared transition); true done -> FULLY fresh ----
        rmt_state = rmt_advance_tokens(
            rmt_state, h_t_j, jnp.asarray(done_np), update_fn, rmt_cfg, carry_mode)
        rmt_state = {
            "mem_tokens": jnp.where(done_np[:, None, None], 0.0, rmt_state["mem_tokens"]),
            "seg_buf":    jnp.where(done_np[:, None, None], 0.0, rmt_state["seg_buf"]),
            "seg_count":  jnp.where(done_np, 0, rmt_state["seg_count"]),
        }
        obsv = next_obsv

    stats = {
        "completed_episodes": len(trajectories),
        "mean_ep_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
        "mean_ep_length": float(np.mean(ep_lengths)) if ep_lengths else 0.0,
        "pending_transitions": pending.total_pending_transitions(),
        "pending_anchors": pending.total_pending_anchors(),
        "pending_rmt_anchors": pending.total_pending_rmt_anchors(),
    }
    carry = {"env_state": env_state, "obsv": obsv, "memories": memories,
             "mem_mask": mem_mask, "mem_idx": mem_idx, "rmt_state": rmt_state, "rng": rng}
    # rollout-aligned batch for the on-policy PPO main update: arrays [rollout_steps, num_envs, ...]
    rollout = {
        "start": rollout_start,
        "obs": np.stack(rl_obs), "actions": np.stack(rl_act),
        "values": np.stack(rl_val), "rewards": np.stack(rl_rew),
        "log_probs": np.stack(rl_lp), "dones": np.stack(rl_don),
        # entering mem_tokens per step [rollout_steps, num_envs, num_tokens, D] (PPO read sequence)
        "last_value": None,   # filled by caller via a final forward on carry state
    }
    return trajectories, carry, rollout, stats
