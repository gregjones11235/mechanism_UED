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
# Phase4A-v2 (CC2 directive §二): PRECISE resolved-env-step provenance (pure, no JAX).
from phase4a_v2_counters import completion_resolved_env_step


def _fresh_rmt_slot() -> dict:
    return {
        "obs": [], "act": [], "rew": [], "don": [], "val": [], "lp": [],
        "next_obs": [], "ach": [],
        "init_mem": None,
        "anchor_mem": [], "anchor_mask": [], "anchor_idx": [], "anchor_step": [],
        # RMT additions
        "init_rmt_tokens": None, "init_rmt_segbuf": None, "init_rmt_segcount": None,
        "anchor_rmt_tokens": [], "anchor_rmt_segbuf": [], "anchor_rmt_segcount": [],
        # Phase4A directive 3: diagnostic-only per-episode running max floor. NOT part of the
        # stored RMTTrajectory; reset with the slot; used only for read-only termination logging.
        "diag_max_floor": 0,
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
    # ---- Phase4A-v2 (CC2 directive §三): split the overloaded update count ----
    # outer_update_index : the OUTER rollout+PPO loop index (authoritative episode update
    #                      index). Replaces the overloaded `collected_update_count` for
    #                      episode stamping / logging.
    # policy_version     : the ACCEPTED policy version (increments only on committed,
    #                      policy-changing updates). Used to stamp pending-episode
    #                      policy_version (NOT the outer loop index).
    # Both default to None -> fall back to `collected_update_count` for strict backward
    # compatibility with replay_mode=off, where every meaning still coincides (bit-exact).
    outer_update_index=None,
    policy_version=None,
):
    """Run rollout_steps vectorized env steps; emit completed RMTTrajectories (w/ anchors).

    Returns (trajectories, carry, stats). carry holds env/GTrXL/RMT/rng state for the next
    rollout (memory/mask/idx AND rmt_state persist across rollouts for Persistent)."""
    assert env_params is not None, 'collect_rollout_rmt: env_params (Craftax EnvParams) must be passed explicitly'
    # ---- Phase4A-v2 (CC2 directive §二/§三): resolve the split counters ----
    if outer_update_index is None:
        outer_update_index = int(collected_update_count)   # deprecated-compat fallback
    if policy_version is None:
        policy_version = int(outer_update_index)           # off-path == old behaviour (bit-exact)
    num_envs = int(np.asarray(obsv).shape[0])
    if apply_eval_rmt is None:
        apply_eval_rmt = make_apply_eval_rmt(network)
    update_fn = make_update_fn(network, params)

    trajectories = []
    ep_returns, ep_lengths = [], []
    episode_records = []   # Phase4A directive 3: per-episode termination records (READ-ONLY logging)

    # ---- rollout-start ENTERING state (for the on-policy PPO main update scan) ----
    rollout_start = {
        "memories": jnp.asarray(memories), "mem_mask": jnp.asarray(mem_mask),
        "mem_idx": jnp.asarray(mem_idx),
        "rmt_state": jax.tree_util.tree_map(jnp.asarray, rmt_state),
    }
    rl_obs, rl_act, rl_val, rl_rew, rl_lp, rl_don = [], [], [], [], [], []

    for _rollout_step_i in range(rollout_steps):
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

        # ---- RMT16 Phase4A READ-ONLY termination-reason capture (CC2 directive 3) ----
        # Terminal signals come from the wrapper's additive _term_* info keys (captured pre-reset
        # inside the wrapper). When probe_term is OFF those keys are ABSENT -> _has_term False ->
        # diagnostics degrade to inert zeros. Host-side reads only; no effect on rollout/PPO/RNG.
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
            # Phase4A directive 3: running max floor (diagnostic only; _info_level valid every step)
            if _has_term and int(_info_level[e]) > buf["diag_max_floor"]:
                buf["diag_max_floor"] = int(_info_level[e])

            if done_np[e]:
                L = len(buf["obs"])
                if L > 0 and buf["init_mem"] is not None:
                    # ---- Phase4A-v2.1 (CC2 §二): episode policy-version RANGE provenance ----
                    # pending.policy_version[e] is the version in force when this episode BEGAN
                    # (set by the reset_slot that opened the slot). It must be read HERE, before
                    # the completing reset_slot below overwrites it with the new episode's start.
                    # `policy_version` (the current accepted version) is the END version. A long
                    # episode that crosses outer rollouts therefore has span = end - start >= 0;
                    # it is NOT correct to stamp the whole trajectory with the end version alone.
                    episode_start_version = int(pending.policy_version[e])
                    episode_end_version = int(policy_version)
                    episode_version_span = episode_end_version - episode_start_version
                    assert episode_end_version >= episode_start_version, (
                        f"policy_version_end {episode_end_version} < start "
                        f"{episode_start_version} (env {e})")
                    assert episode_version_span >= 0, (
                        f"policy_version_span {episode_version_span} < 0 (env {e})")
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
                        # Phase4A-v2 (§三): collected_update_count kept as the OUTER loop index
                        # for strict legacy schema compat.
                        collected_update_count=int(outer_update_index),
                        outer_update_index=int(outer_update_index),
                        # Phase4A-v2.1 (§二): episode policy-version RANGE (start/end/span).
                        # policy_version_at_collection is the DEPRECATED alias of START (it is
                        # NO LONGER the end/current version — that was the bug fixed here).
                        policy_version_start=episode_start_version,
                        policy_version_end=episode_end_version,
                        policy_version_span=episode_version_span,
                        policy_version_at_collection=episode_start_version,
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
                    # ---- Phase4A directive 3: per-episode termination record (READ-ONLY) ----
                    # done_reason mapped host-side from DIRECT terminal signals; NO inference.
                    # Exactly one candidate -> that reason; zero (e.g. boss-only) or >1 (ambiguous)
                    # -> 'unknown', with candidates + ambiguous flag retained for the report.
                    # optimistic_reset/wrapper_reset are never a done CAUSE here (wrapper only
                    # auto-resets already-done envs; it never truncates a live episode).
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
                    episode_record = dict(
                        episode_id=int(pending.episode_id[e]), env_id=int(e), length=int(L),
                        update_index=int(outer_update_index), rollout_step=int(_rollout_step_i),
                        # Phase4A-v2 (§二): PRECISE resolved env step at completion (authoritative).
                        completion_resolved_env_step=completion_resolved_env_step(
                            outer_update_index, num_envs, rollout_steps, _rollout_step_i, e),
                        outer_update_index=int(outer_update_index),
                        # Phase4A-v2.2 (§四): the episode record carries the policy-version
                        # RANGE (start/end/span), values IDENTICAL to the RMTTrajectory built
                        # just above. Recompute provenance (update_index/rollout_step/env_id/
                        # length/episode_id) is unchanged, so frozen recompute stays 20/6,
                        # 21/5, 8979, BOTH.
                        policy_version_start=episode_start_version,
                        policy_version_end=episode_end_version,
                        policy_version_span=episode_version_span,
                        # §四 compat: the old UNSCOPED policy_version field is no longer an
                        # authoritative field; it is an explicit DEPRECATED alias of
                        # policy_version_end (the completion version).
                        policy_version=int(policy_version),
                        policy_version_deprecated=True,
                        policy_version_alias_of="policy_version_end",
                        # DEPRECATED (§二): NOT a precise resolved step (drops *num_envs on the
                        # rollout_step term, the per-env env_id offset and the +1). Kept ONLY for
                        # historical recomparison against pre-v2 records.
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
                        carry_mode=carry_mode, has_term_signals=bool(_has_term))
                    # Phase4A-v2.2 (§四): PRE-WRITE invariants — the record's range must be
                    # self-consistent AND exactly equal to the just-built trajectory's range
                    # (and the trajectory's deprecated alias must equal START). Any violation
                    # aborts collection loudly rather than writing an inconsistent record.
                    assert episode_record["policy_version_start"] >= 0, (
                        f"episode record policy_version_start < 0 (env {e})")
                    assert episode_record["policy_version_end"] >= (
                        episode_record["policy_version_start"]), (
                        f"episode record policy_version_end < start (env {e})")
                    assert episode_record["policy_version_span"] == (
                        episode_record["policy_version_end"]
                        - episode_record["policy_version_start"]), (
                        f"episode record policy_version_span != end - start (env {e})")
                    assert traj.policy_version_start == episode_record["policy_version_start"], (
                        f"traj/record policy_version_start mismatch (env {e})")
                    assert traj.policy_version_end == episode_record["policy_version_end"], (
                        f"traj/record policy_version_end mismatch (env {e})")
                    assert traj.policy_version_span == episode_record["policy_version_span"], (
                        f"traj/record policy_version_span mismatch (env {e})")
                    assert traj.policy_version_at_collection == (
                        episode_record["policy_version_start"]), (
                        f"traj.policy_version_at_collection != record start (env {e})")
                    assert episode_record["policy_version"] == (
                        episode_record["policy_version_end"]), (
                        f"deprecated policy_version alias != policy_version_end (env {e})")
                    episode_records.append(episode_record)
                # Phase4A-v2.1 (§二.3): open the NEXT episode with start version == the CURRENT
                # rollout's accepted policy_version. This is correct because:
                #   * after this auto-reset the new episode's FIRST steps (the rest of THIS
                #     rollout) are still generated by the current rollout's policy;
                #   * the PPO update happens only AFTER the whole rollout completes, so no newer
                #     accepted version exists yet at this point;
                #   * therefore the new episode's policy_version_start is the current rollout
                #     policy version. (In replay_mode=off this equals the outer loop index, so
                #     the off path stays bit-exact.)
                pending.reset_slot(e, policy_version=int(policy_version))

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
        "episode_records": episode_records,   # Phase4A directive 3 (READ-ONLY; host-side)
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
