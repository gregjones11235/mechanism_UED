"""W512 × P2-Replay — rollout collection emitting W512Trajectories with sparse anchors.

EXTENDS (does not modify) the frozen P2-Full-A collection design, mirroring `rmt_collect.py`.
The network is W512 (forward returns h_t and reads the 384 long buffer) and each anchor ALSO
snapshots the W512 raw-history state (delay line + long ring + seg_step). The single per-step
transition is the SAME one reconstruction uses (w512_memory_anchor.w512_step_forward / its
modular helpers), so anchor round-trip is bit-exact by construction (directive gate 6/7; the
foundation self-test G6 proved this reconstruction convention).

carry_mode = reset128 (fixed by the directive): the 384 long buffer (long_buf/long_mask) is
CLEARED at every 128-step EPISODE-LOCAL segment boundary via the carried seg_step counter
(w512_reset128_clear). This aligns exactly with the replay anchors (episode steps 0,128,256,...),
so reconstruction from an anchor re-applies the identical clear bit-exactly. The delay line is NOT
cleared at the boundary (matches the historical reset128). After a TRUE done the env's whole state
(GTrXL memory + W512 buffers + seg_step) is reset to fresh so the new episode's step-0 anchor is
the fresh initial state (episode-start anchor == initial state, P2 conservation).

Done convention (mirrors the proven historical ppo_tr_w512_reset128._env_step):
  * done_enter : done carried INTO this step (= done_new of the previous step). Drives the GTrXL
                 mem_idx/mem_mask advance (w512_advance_mask) BEFORE the forward.
  * done_new   : done produced by THIS step's env.step. Drives the W512 buffer advance
                 (w5m.w512_step) + seg_step reset, and the true-done isolation, AFTER the forward.
Within a single episode done is False everywhere except the terminal transition.

NOTE on the GTrXL mask carry: unlike rmt_collect (which only resets mem_mask/mem_idx on done and
never applies the per-step decrement/one_hot during collection), W512 collection applies the FULL
w512_advance_mask before every forward — the SAME convention as w512_step_forward / the loss scan /
derive_anchor_entering_state. This is REQUIRED so the snapshotted anchors lie in the reconstruction
convention and replay rebuilds the collected trajectory bit-exactly (the foundation G6 anchor
round-trip assumes this convention). It also makes collection old_logp == PPO re-forward new_logp
(valid PPO ratio). The frozen replay coefficients / conservation rules / provenance fields are all
inherited unchanged from the RMT/P2 design.
"""
import numpy as np
import jax
import jax.numpy as jnp

import rng_utils as RU
from pending_episodes import PendingEpisodeBuffers
from replay_buffer import ANCHOR_INTERVAL
from w512_replay_buffer import W512Trajectory
import w512_memory as w5m
from w512_memory_anchor import (
    make_apply_eval_w512, w512_reset128_clear, w512_advance_mask, w512_reset_state_on_done,
)
# PRECISE resolved-env-step provenance (pure, no JAX) — reused from the frozen Phase4A counters.
from phase4a_v2_counters import completion_resolved_env_step


def _fresh_w512_slot() -> dict:
    return {
        "obs": [], "act": [], "rew": [], "don": [], "val": [], "lp": [],
        "next_obs": [], "ach": [],
        "init_mem": None,
        "anchor_mem": [], "anchor_mask": [], "anchor_idx": [], "anchor_step": [],
        # W512 additions (per-anchor ENTERING raw-history state; 1:1 with anchor_step)
        "init_w512": None,
        "anchor_w512_delay_buf": [], "anchor_w512_delay_idx": [], "anchor_w512_delay_count": [],
        "anchor_w512_long_buf": [], "anchor_w512_long_mask": [], "anchor_w512_long_idx": [],
        "anchor_w512_seg_step": [],
        # diagnostic-only per-episode running max floor (READ-ONLY logging; not stored)
        "diag_max_floor": 0,
    }


class W512PendingEpisodeBuffers(PendingEpisodeBuffers):
    """Pending buffers carrying BOTH GTrXL and W512 anchors per env slot."""

    def __init__(self, num_envs, first_episode_id=0, first_policy_version=0):
        self.num_envs = int(num_envs)
        self.next_episode_id = int(first_episode_id)
        self.slots = []
        self.episode_id = []
        self.policy_version = []
        for _ in range(self.num_envs):
            self.slots.append(_fresh_w512_slot())
            self.episode_id.append(self.next_episode_id)
            self.policy_version.append(int(first_policy_version))
            self.next_episode_id += 1

    def reset_slot(self, e, policy_version):
        self.slots[e] = _fresh_w512_slot()
        self.episode_id[e] = self.next_episode_id
        self.policy_version[e] = int(policy_version)
        self.next_episode_id += 1
        return self.episode_id[e]

    def add_w512_anchor(self, e, w512_st_np):
        """Snapshot the ENTERING w512 state (numpy) for env e at the current anchor step."""
        s = self.slots[e]
        s["anchor_w512_delay_buf"].append(np.asarray(w512_st_np["delay_buf"]).copy())
        s["anchor_w512_delay_idx"].append(int(w512_st_np["delay_idx"]))
        s["anchor_w512_delay_count"].append(int(w512_st_np["delay_count"]))
        s["anchor_w512_long_buf"].append(np.asarray(w512_st_np["long_buf"]).copy())
        s["anchor_w512_long_mask"].append(np.asarray(w512_st_np["long_mask"]).copy())
        s["anchor_w512_long_idx"].append(int(w512_st_np["long_idx"]))
        s["anchor_w512_seg_step"].append(int(w512_st_np["seg_step"]))

    def total_pending_w512_anchors(self):
        return sum(len(s["anchor_w512_long_buf"]) for s in self.slots)


def collect_rollout_w512(
    env, env_state, network, params, obsv,
    memories, mem_mask, mem_idx, w512_state, done_enter,
    rng, action_rng,
    pending: W512PendingEpisodeBuffers,
    target_achievement,
    rollout_steps: int,
    window_mem: int, num_heads: int,
    w512_cfg, segment_len: int = 128,
    collected_update_count: int = 0,
    apply_eval_w512=None,
    env_params=None,   # Craftax EnvParams (NOT network params); driver must pass explicitly
    outer_update_index=None,
    policy_version=None,
):
    """Run rollout_steps vectorized env steps; emit completed W512Trajectories (w/ anchors).

    Returns (trajectories, carry, rollout, stats). carry holds env/GTrXL/W512/rng state AND the
    carried done_enter for the next rollout. The W512 state persists across rollouts (reset128
    clears only at episode-local 128 boundaries; true done resets fully)."""
    assert env_params is not None, 'collect_rollout_w512: env_params (Craftax EnvParams) must be passed explicitly'
    if outer_update_index is None:
        outer_update_index = int(collected_update_count)
    if policy_version is None:
        policy_version = int(outer_update_index)
    num_envs = int(np.asarray(obsv).shape[0])
    if apply_eval_w512 is None:
        apply_eval_w512 = make_apply_eval_w512(network)

    trajectories = []
    ep_returns, ep_lengths = [], []
    episode_records = []

    # ---- rollout-start ENTERING state (for the on-policy PPO main update scan) ----
    rollout_start = {
        "memories": jnp.asarray(memories), "mem_mask": jnp.asarray(mem_mask),
        "mem_idx": jnp.asarray(mem_idx),
        "w512_state": jax.tree_util.tree_map(jnp.asarray, w512_state),
        "done_enter0": jnp.asarray(done_enter).astype(jnp.bool_),
    }
    rl_obs, rl_act, rl_val, rl_rew, rl_lp, rl_don = [], [], [], [], [], []

    done_enter = jnp.asarray(done_enter).astype(jnp.bool_)

    for _rollout_step_i in range(rollout_steps):
        # ---- snapshot ENTERING state (pre-clear) for anchor capture ----
        mem_in = np.asarray(memories).copy()
        mask_in = np.asarray(mem_mask).copy()
        idx_in = np.asarray(mem_idx).copy()
        w512_in = jax.tree_util.tree_map(lambda x: np.asarray(x).copy(), w512_state)
        for e in range(num_envs):
            k = len(pending.slots[e]["obs"])       # episode-local step about to be collected
            if k % ANCHOR_INTERVAL == 0:
                pending.add_anchor(e, k, mem_in[e].copy(), mask_in[e].copy(), int(idx_in[e]))
                pending.add_w512_anchor(e, {kk: vv[e] for kk, vv in w512_in.items()})

        # ---- PHASE 1: reset128 clear + GTrXL mask advance + forward (pre env.step) ----
        # Identical to the first half of w512_step_forward (G8: inline == step_forward bit-exact).
        w512_clr = w512_reset128_clear(w512_state, segment_len)
        mem_idx_adv, mem_mask_adv = w512_advance_mask(
            mem_idx, mem_mask, done_enter, window_mem, num_heads)
        rng, a_rng = jax.random.split(rng)
        logits, value, mem_out, h_t = apply_eval_w512(
            params, memories, obsv, mem_mask_adv, w512_clr["long_buf"], w512_clr["long_mask"])
        value_np = np.asarray(value); mem_out_np = np.asarray(mem_out)
        probs = np.asarray(jax.nn.softmax(jnp.asarray(logits), axis=-1))
        actions_np = RU.sample_actions(action_rng, probs)
        logp_np = np.log(probs[np.arange(num_envs), actions_np] + 1e-12)
        obs_pre = np.asarray(obsv).copy()

        rng, s_rng = jax.random.split(rng)
        next_obsv, env_state, reward_j, done_j, info = env.step(s_rng, env_state, actions_np, env_params)
        reward_np = np.asarray(reward_j)
        done_np = np.asarray(done_j).astype(bool)   # done_new
        next_obs_np = np.asarray(next_obsv).copy()
        h_t_j = jnp.asarray(h_t)

        # ---- READ-ONLY termination-reason capture (additive _term_* info keys) ----
        _has_term = "_term_player_level" in info
        if _has_term:
            _info_level = np.asarray(info["_term_player_level"]).astype(np.int64)
            _info_health = np.asarray(info["_term_player_health"]).astype(np.float32)
            _info_timestep = np.asarray(info["_term_timestep"]).astype(np.int64)
            _info_isdead = np.asarray(info["_term_is_dead"]).astype(bool)
            _info_donesteps = np.asarray(info["_term_done_steps"]).astype(bool)
            _info_issuccess = np.asarray(info["is_success"]).astype(bool)
        else:
            _info_level = np.zeros(num_envs, np.int64)
            _info_health = np.zeros(num_envs, np.float32)
            _info_timestep = np.zeros(num_envs, np.int64)
            _info_isdead = np.zeros(num_envs, bool)
            _info_donesteps = np.zeros(num_envs, bool)
            _info_issuccess = np.zeros(num_envs, bool)
        _ach_keys = [k for k in info.keys() if k.startswith("Achievements/")]

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
                buf["init_w512"] = {kk: vv[e].copy() for kk, vv in w512_in.items()}
            buf["obs"].append(obs_pre[e].copy())
            buf["act"].append(int(actions_np[e]))
            buf["rew"].append(float(reward_np[e]))
            buf["don"].append(bool(done_np[e]))
            buf["val"].append(float(value_np[e]))
            buf["lp"].append(float(logp_np[e]))
            buf["next_obs"].append(next_obs_np[e].copy())
            buf["ach"].append(ach_data[e].copy())
            if _has_term and int(_info_level[e]) > buf["diag_max_floor"]:
                buf["diag_max_floor"] = int(_info_level[e])

            if done_np[e]:
                L = len(buf["obs"])
                if L > 0 and buf["init_mem"] is not None:
                    episode_start_version = int(pending.policy_version[e])
                    episode_end_version = int(policy_version)
                    episode_version_span = episode_end_version - episode_start_version
                    assert episode_end_version >= episode_start_version, (
                        f"policy_version_end {episode_end_version} < start "
                        f"{episode_start_version} (env {e})")
                    assert episode_version_span >= 0, (
                        f"policy_version_span {episode_version_span} < 0 (env {e})")
                    iw = buf["init_w512"]
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
                        collected_update_count=int(outer_update_index),
                        outer_update_index=int(outer_update_index),
                        policy_version_start=episode_start_version,
                        policy_version_end=episode_end_version,
                        policy_version_span=episode_version_span,
                        policy_version_at_collection=episode_start_version,
                        w512_initial_delay_buf=iw["delay_buf"],
                        w512_initial_delay_idx=int(iw["delay_idx"]),
                        w512_initial_delay_count=int(iw["delay_count"]),
                        w512_initial_long_buf=iw["long_buf"],
                        w512_initial_long_mask=iw["long_mask"],
                        w512_initial_long_idx=int(iw["long_idx"]),
                        w512_initial_seg_step=int(iw["seg_step"]),
                        w512_anchor_delay_buf=np.stack(buf["anchor_w512_delay_buf"]),
                        w512_anchor_delay_idx=np.array(buf["anchor_w512_delay_idx"], np.int64),
                        w512_anchor_delay_count=np.array(buf["anchor_w512_delay_count"], np.int64),
                        w512_anchor_long_buf=np.stack(buf["anchor_w512_long_buf"]),
                        w512_anchor_long_mask=np.stack(buf["anchor_w512_long_mask"]),
                        w512_anchor_long_idx=np.array(buf["anchor_w512_long_idx"], np.int64),
                        w512_anchor_seg_step=np.array(buf["anchor_w512_seg_step"], np.int64),
                    )
                    trajectories.append(traj)
                    ep_returns.append(float(np.sum(buf["rew"])))
                    ep_lengths.append(int(L))
                    _is_dead_e = bool(_info_isdead[e]); _done_steps_e = bool(_info_donesteps[e])
                    _is_success_e = bool(_info_issuccess[e])
                    _cands = []
                    if _done_steps_e:
                        _cands.append("time_limit")
                    if _is_success_e:
                        _cands.append("task_success")
                    if _is_dead_e:
                        _cands.append("player_death")
                    _done_reason = _cands[0] if len(_cands) == 1 else "unknown"
                    _ach_reached = [k.split("/", 1)[1] for k in _ach_keys
                                    if float(np.asarray(info[k])[e]) > 0.0]
                    episode_records.append(dict(
                        episode_id=int(pending.episode_id[e]), env_id=int(e), length=int(L),
                        update_index=int(outer_update_index), rollout_step=int(_rollout_step_i),
                        completion_resolved_env_step=completion_resolved_env_step(
                            outer_update_index, num_envs, rollout_steps, _rollout_step_i, e),
                        outer_update_index=int(outer_update_index),
                        policy_version_start=episode_start_version,
                        policy_version_end=episode_end_version,
                        policy_version_span=episode_version_span,
                        policy_version=int(policy_version),
                        policy_version_deprecated=True,
                        policy_version_alias_of="policy_version_end",
                        completion_global_step=int(outer_update_index) * (num_envs * rollout_steps)
                            + int(_rollout_step_i),
                        completion_global_step_deprecated=True,
                        terminated=bool(done_np[e] and not _done_steps_e), truncated=_done_steps_e,
                        done_reason=_done_reason, done_reason_ambiguous=bool(len(_cands) > 1),
                        done_reason_candidates=_cands,
                        target_achievement_reached=_is_success_e,
                        achievements_reached=_ach_reached,
                        achievements_reached_count=len(_ach_reached),
                        max_floor_reached=int(buf["diag_max_floor"]),
                        final_floor=int(_info_level[e]), final_health=float(_info_health[e]),
                        term_is_dead=_is_dead_e, term_done_steps=_done_steps_e,
                        term_is_success=_is_success_e, term_timestep=int(_info_timestep[e]),
                        episode_return=float(np.sum(buf["rew"])),
                        carry_mode="reset128", network_family="W512",
                        has_term_signals=bool(_has_term)))
                pending.reset_slot(e, policy_version=int(policy_version))

        # ---- PHASE 2: advance state (uses done_new); identical to w512_step_forward's 2nd half ----
        post_memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(mem_out_np)
        # W512 buffer advance on the POST-CLEAR state used for the forward (w5m.w512_step resets
        # done envs internally); seg_step resets on done_new else +1.
        new_w512 = w5m.w512_step(w512_clr, h_t_j, jnp.asarray(done_np), w512_cfg)
        new_w512 = {
            **new_w512,
            "seg_step": jnp.where(jnp.asarray(done_np), 0, w512_clr["seg_step"] + 1).astype(jnp.int32),
        }
        # TRUE-DONE isolation: fresh GTrXL memory + fresh W512 state for done envs, so the new
        # episode's step-0 anchor is the fresh initial state (P2 conservation).
        post_memories = jnp.where(done_np[:, None, None, None], jnp.zeros_like(post_memories), post_memories)
        new_mask = jnp.where(done_np[:, None, None, None], jnp.zeros_like(mem_mask_adv), mem_mask_adv)
        new_idx = jnp.where(done_np, window_mem, mem_idx_adv)
        new_w512 = w512_reset_state_on_done(new_w512, jnp.asarray(done_np))

        memories, mem_mask, mem_idx, w512_state = post_memories, new_mask, new_idx, new_w512
        done_enter = jnp.asarray(done_np).astype(jnp.bool_)   # next step's entering done
        obsv = next_obsv

    stats = {
        "completed_episodes": len(trajectories),
        "mean_ep_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
        "mean_ep_length": float(np.mean(ep_lengths)) if ep_lengths else 0.0,
        "pending_transitions": pending.total_pending_transitions(),
        "pending_anchors": pending.total_pending_anchors(),
        "pending_w512_anchors": pending.total_pending_w512_anchors(),
        "episode_records": episode_records,
    }
    carry = {"env_state": env_state, "obsv": obsv, "memories": memories,
             "mem_mask": mem_mask, "mem_idx": mem_idx, "w512_state": w512_state,
             "done_enter": done_enter, "rng": rng}
    rollout = {
        "start": rollout_start,
        "obs": np.stack(rl_obs), "actions": np.stack(rl_act),
        "values": np.stack(rl_val), "rewards": np.stack(rl_rew),
        "log_probs": np.stack(rl_lp), "dones": np.stack(rl_don),
        "last_value": None,   # filled by caller via a final forward on carry state
    }
    return trajectories, carry, rollout, stats
