import sys

PATH = "rmt_collect.py"   # run from src dir
N = "\n"


def block(lines):
    return N.join(lines) + N


# ---- R1: add diagnostic-only running max-floor to the pending slot (NOT part of RMTTrajectory) ----
R1_OLD = block([
    '        "anchor_rmt_tokens": [], "anchor_rmt_segbuf": [], "anchor_rmt_segcount": [],',
    "    }",
])
R1_NEW = block([
    '        "anchor_rmt_tokens": [], "anchor_rmt_segbuf": [], "anchor_rmt_segcount": [],',
    "        # Phase4A directive 3: diagnostic-only per-episode running max floor. NOT part of the",
    "        # stored RMTTrajectory; reset with the slot; used only for read-only termination logging.",
    '        "diag_max_floor": 0,',
    "    }",
])

# ---- R2: episode_records accumulator alongside ep_returns/ep_lengths ----
R2_OLD = block([
    "    trajectories = []",
    "    ep_returns, ep_lengths = [], []",
])
R2_NEW = block([
    "    trajectories = []",
    "    ep_returns, ep_lengths = [], []",
    "    episode_records = []   # Phase4A directive 3: per-episode termination records (READ-ONLY logging)",
])

# ---- R3: name the rollout loop index (for completion_global_step) ----
R3_OLD = "    for _ in range(rollout_steps):\n"
R3_NEW = "    for _rollout_step_i in range(rollout_steps):\n"

# ---- R4: capture TERMINAL signals from wrapper _term_* info keys (host-side; inert when absent) ----
R4_OLD = block([
    "        h_t_j = jnp.asarray(h_t)",
    "",
    "        # ---- accumulate rollout-aligned per-step arrays (for the PPO main update) ----",
])
R4_NEW = block([
    "        h_t_j = jnp.asarray(h_t)",
    "",
    "        # ---- RMT16 Phase4A READ-ONLY termination-reason capture (CC2 directive 3) ----",
    "        # Terminal signals come from the wrapper's additive _term_* info keys (captured pre-reset",
    "        # inside the wrapper). When probe_term is OFF those keys are ABSENT -> _has_term False ->",
    "        # diagnostics degrade to inert zeros. Host-side reads only; no effect on rollout/PPO/RNG.",
    '        _has_term = "_term_player_level" in info',
    "        if _has_term:",
    '            _info_level = np.asarray(info["_term_player_level"]).astype(np.int64)',
    '            _info_health = np.asarray(info["_term_player_health"]).astype(np.float32)',
    '            _info_timestep = np.asarray(info["_term_timestep"]).astype(np.int64)',
    '            _info_isdead = np.asarray(info["_term_is_dead"]).astype(bool)',
    '            _info_donesteps = np.asarray(info["_term_done_steps"]).astype(bool)',
    '            _info_issuccess = np.asarray(info["is_success"]).astype(bool)',
    "        else:",
    "            _info_level = np.zeros(num_envs, np.int64)",
    "            _info_health = np.zeros(num_envs, np.float32)",
    "            _info_timestep = np.zeros(num_envs, np.int64)",
    "            _info_isdead = np.zeros(num_envs, bool)",
    "            _info_donesteps = np.zeros(num_envs, bool)",
    "            _info_issuccess = np.zeros(num_envs, bool)",
    '        _ach_keys = [k for k in info.keys() if k.startswith("Achievements/")]',
    "",
    "        # ---- accumulate rollout-aligned per-step arrays (for the PPO main update) ----",
])

# ---- R5: running max-floor update inside the per-env accumulation loop ----
R5_OLD = '            buf["ach"].append(ach_data[e].copy())\n'
R5_NEW = block([
    '            buf["ach"].append(ach_data[e].copy())',
    "            # Phase4A directive 3: running max floor (diagnostic only; _info_level valid every step)",
    '            if _has_term and int(_info_level[e]) > buf["diag_max_floor"]:',
    '                buf["diag_max_floor"] = int(_info_level[e])',
])

# ---- R6: build the per-episode termination record at done (before reset_slot) ----
R6_OLD = block([
    '                    ep_returns.append(float(np.sum(buf["rew"])))',
    "                    ep_lengths.append(int(L))",
    "                pending.reset_slot(e, policy_version=int(collected_update_count))",
])
R6_NEW = block([
    '                    ep_returns.append(float(np.sum(buf["rew"])))',
    "                    ep_lengths.append(int(L))",
    "                    # ---- Phase4A directive 3: per-episode termination record (READ-ONLY) ----",
    "                    # done_reason mapped host-side from DIRECT terminal signals; NO inference.",
    "                    # Exactly one candidate -> that reason; zero (e.g. boss-only) or >1 (ambiguous)",
    "                    # -> 'unknown', with candidates + ambiguous flag retained for the report.",
    "                    # optimistic_reset/wrapper_reset are never a done CAUSE here (wrapper only",
    "                    # auto-resets already-done envs; it never truncates a live episode).",
    "                    _is_dead_e = bool(_info_isdead[e]); _done_steps_e = bool(_info_donesteps[e])",
    "                    _is_success_e = bool(_info_issuccess[e])",
    "                    _cands = []",
    "                    if _done_steps_e:",
    '                        _cands.append("time_limit")',
    "                    if _is_success_e:",
    '                        _cands.append("task_success")',
    "                    if _is_dead_e:",
    '                        _cands.append("player_death")',
    '                    _done_reason = _cands[0] if len(_cands) == 1 else "unknown"',
    '                    _ach_reached = [k.split("/", 1)[1] for k in _ach_keys',
    "                                    if float(np.asarray(info[k])[e]) > 0.0]",
    "                    episode_records.append(dict(",
    "                        episode_id=int(pending.episode_id[e]), env_id=int(e), length=int(L),",
    "                        update_index=int(collected_update_count), rollout_step=int(_rollout_step_i),",
    "                        completion_global_step=int(collected_update_count) * (num_envs * rollout_steps)",
    "                            + int(_rollout_step_i),",
    "                        terminated=bool(done_np[e] and not _done_steps_e), truncated=_done_steps_e,",
    "                        done_reason=_done_reason, done_reason_ambiguous=bool(len(_cands) > 1),",
    "                        done_reason_candidates=_cands,",
    "                        target_achievement_reached=_is_success_e,",
    "                        achievements_reached=_ach_reached,",
    "                        achievements_reached_count=len(_ach_reached),",
    '                        max_floor_reached=int(buf["diag_max_floor"]),',
    "                        final_floor=int(_info_level[e]), final_health=float(_info_health[e]),",
    "                        term_is_dead=_is_dead_e, term_done_steps=_done_steps_e,",
    "                        term_is_success=_is_success_e, term_timestep=int(_info_timestep[e]),",
    '                        episode_return=float(np.sum(buf["rew"])),',
    "                        carry_mode=carry_mode, has_term_signals=bool(_has_term)))",
    "                pending.reset_slot(e, policy_version=int(collected_update_count))",
])

# ---- R7: expose episode_records via stats ----
R7_OLD = block([
    '        "pending_rmt_anchors": pending.total_pending_rmt_anchors(),',
    "    }",
])
R7_NEW = block([
    '        "pending_rmt_anchors": pending.total_pending_rmt_anchors(),',
    '        "episode_records": episode_records,   # Phase4A directive 3 (READ-ONLY; host-side)',
    "    }",
])

REPLS = [("R1", R1_OLD, R1_NEW, 1), ("R2", R2_OLD, R2_NEW, 1), ("R3", R3_OLD, R3_NEW, 1),
         ("R4", R4_OLD, R4_NEW, 1), ("R5", R5_OLD, R5_NEW, 1), ("R6", R6_OLD, R6_NEW, 1),
         ("R7", R7_OLD, R7_NEW, 1)]
with open(PATH, "r", newline="") as f:
    src = f.read()
for tag, old, new, n in REPLS:
    c = src.count(old)
    if c != n:
        print("ABORT %s expected %d got %d" % (tag, n, c))
        sys.exit(1)
    src = src.replace(old, new)
with open(PATH, "w", newline="") as f:
    f.write(src)
print("COLLECTOR_PATCHED (%d replacements)" % len(REPLS))
