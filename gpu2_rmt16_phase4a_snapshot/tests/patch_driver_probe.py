import sys

PATH = "train_rmt16_p2replay.py"   # run from src dir
N = "\n"


def block(lines):
    return N.join(lines) + N


# ---- D1: argparse additions (probe / debug-early-stop / equiv_dump) ----
D1_OLD = 'ap.add_argument("--save_every", type=int, default=2)         # updates between saves (2 => every 4096)\n'
D1_NEW = block([
    'ap.add_argument("--save_every", type=int, default=2)         # updates between saves (2 => every 4096)',
    "# Phase4A probe (CC2 directive 2/3 + addendum): reachability probe + A/B no-perturbation gate.",
    'ap.add_argument("--probe", action="store_true",',
    '                help="L512 reachability probe: record-only, fixed full horizon, replay learner+hindsight OFF")',
    'ap.add_argument("--early_stop_len", type=int, default=0,',
    '                help="DEBUG-ONLY non-comparative early stop (0=OFF; formal probe MUST keep 0)")',
    'ap.add_argument("--equiv_dump", action="store_true",',
    '                help="emit per-update deterministic equivalence hashes for the A/B no-perturbation gate")',
])

# ---- D2: PROBE flag + enforce replay off ----
D2_OLD = 'REPLAY_ON = (args.replay == "on")\n'
D2_NEW = block([
    'REPLAY_ON = (args.replay == "on")',
    "PROBE = bool(args.probe)",
    "if PROBE:",
    '    assert args.replay == "off", "probe requires --replay off (replay learner + hindsight must be OFF)"',
])

# ---- D3: _arr_hash helper next to _params_sha (_params_sha already hashes any pytree's leaves) ----
D3_OLD = block([
    "def _params_sha(params):",
    "    h = hashlib.sha256()",
    "    for v in jax.tree_util.tree_leaves(params):",
    "        h.update(np.ascontiguousarray(np.asarray(v)).tobytes())",
    "    return h.hexdigest()",
])
D3_NEW = block([
    "def _params_sha(params):",
    "    h = hashlib.sha256()",
    "    for v in jax.tree_util.tree_leaves(params):",
    "        h.update(np.ascontiguousarray(np.asarray(v)).tobytes())",
    "    return h.hexdigest()",
    "",
    "def _arr_hash(*arrays):",
    "    h = hashlib.sha256()",
    "    for a in arrays:",
    "        h.update(np.ascontiguousarray(np.asarray(a)).tobytes())",
    "    return h.hexdigest()",
])

# ---- D4: probe + equiv output paths ----
D4_OLD = 'audit_path = os.path.join(LOG_DIR, f"{ARM}_replay_audit.jsonl")\n'
D4_NEW = block([
    'audit_path = os.path.join(LOG_DIR, f"{ARM}_replay_audit.jsonl")',
    "# Phase4A directive 2/3: probe output files (only written when PROBE) + equiv gate file.",
    'probe_episodes_path = os.path.join(LOG_DIR, f"{ARM}_probe_episodes.jsonl")',
    'probe_updates_path = os.path.join(LOG_DIR, f"{ARM}_probe_updates.jsonl")',
    'equiv_path = os.path.join(LOG_DIR, f"{ARM}_equiv.jsonl")',
    "PROBE_GE_LEN = 512   # directive length>=512 eligibility threshold (RECORD only; never a stop)",
])

# ---- D5: probe cumulative state ----
D5_OLD = "accepted_policy_updates = 0; kl_rejected_updates = 0; hindsight_eligible = 0; hindsight_attempts = 0\n"
D5_NEW = block([
    "accepted_policy_updates = 0; kl_rejected_updates = 0; hindsight_eligible = 0; hindsight_attempts = 0",
    "# Phase4A directive 2/3: probe cumulative state (used only when PROBE).",
    "probe_completed_episodes = []     # cumulative per-episode termination records",
    "probe_first_ge512 = None          # first length>=512 episode (RECORD only; never a stop)",
])

# ---- D6: per-update equivalence dump (A/B no-perturbation gate; only with --equiv_dump) ----
D6_OLD = block([
    "    update_count += 1",
    "    online_ppo_update_count += 1",
    '    assert ppo_metrics["ppo_finite"], "HARD STOP NaN/Inf in PPO update"',
])
D6_NEW = block([
    "    update_count += 1",
    "    online_ppo_update_count += 1",
    '    assert ppo_metrics["ppo_finite"], "HARD STOP NaN/Inf in PPO update"',
    "    # ---- A/B training-no-perturbation gate artifacts (CC2 addendum; only with --equiv_dump) ----",
    "    # Deterministic hashes of the rollout + post-update params/optimizer/RMT state. A (probe OFF)",
    "    # and B (probe ON) both emit these; an exact match proves the probe instrumentation does not",
    "    # perturb training. Host-side reads only; no effect on training numerics / RNG stream.",
    "    if args.equiv_dump:",
    "        equiv = dict(update=u, global_step=(u + 1) * STEPS_PER_UPDATE,",
    '                     actions_hash=_arr_hash(rollout["actions"]),',
    '                     rewards_hash=_arr_hash(rollout["rewards"]),',
    '                     dones_hash=_arr_hash(rollout["dones"]),',
    '                     ard_hash=_arr_hash(rollout["actions"], rollout["rewards"], rollout["dones"]),',
    "                     params_sha=_params_sha(params),",
    "                     ppo_opt_sha=_params_sha(ppo_opt_state),",
    "                     rmt_state_sha=_params_sha(rmt_state),",
    "                     memories_sha=_params_sha(memories),",
    "                     mem_mask_sha=_params_sha(mem_mask),",
    "                     mem_idx_sha=_params_sha(mem_idx),",
    '                     ppo_actor=float(ppo_metrics["ppo_actor"]),',
    '                     ppo_entropy=float(ppo_metrics["ppo_entropy"]),',
    '                     ppo_value=float(ppo_metrics.get("ppo_value", 0.0)),',
    "                     online_ppo_update_count=online_ppo_update_count)",
    "        with open(equiv_path, \"a\") as f:",
    '            f.write(json.dumps(equiv, default=str) + "\\n")',
])

# ---- D7: per-update probe aggregation (RECORD only; never a stop) ----
D7_OLD = block([
    "    assert replay.counters.trajectories_collected == replay.counters.trajectories_inserted, \\",
    '        "HARD STOP conservation: collected != inserted"',
])
D7_NEW = block([
    "    assert replay.counters.trajectories_collected == replay.counters.trajectories_inserted, \\",
    '        "HARD STOP conservation: collected != inserted"',
    "    # ---- Phase4A probe: per-episode records + per-update aggregation (directive 2/3; RECORD only) ----",
    "    if PROBE:",
    '        _new_eps = stats.get("episode_records", [])',
    "        probe_completed_episodes.extend(_new_eps)",
    "        with open(probe_episodes_path, \"a\") as f:",
    "            for _ep in _new_eps:",
    "                f.write(json.dumps(_ep, default=str) + \"\\n\")",
    "        for _ep in _new_eps:",
    "            if probe_first_ge512 is None and int(_ep[\"length\"]) >= PROBE_GE_LEN:",
    "                probe_first_ge512 = dict(",
    "                    first_ge512_update=int(_ep[\"update_index\"]),",
    "                    first_ge512_global_step=int(_ep[\"completion_global_step\"]),",
    "                    first_ge512_episode_id=int(_ep[\"episode_id\"]),",
    "                    first_ge512_length=int(_ep[\"length\"]))",
    "        _lens = [int(_e[\"length\"]) for _e in probe_completed_episodes]",
    "        _reasons = {}",
    "        for _e in probe_completed_episodes:",
    "            _reasons[_e[\"done_reason\"]] = _reasons.get(_e[\"done_reason\"], 0) + 1",
    "        _la = np.array(_lens, float) if _lens else np.array([0.0])",
    "        probe_upd = dict(update=u, global_step=(u + 1) * STEPS_PER_UPDATE, arm=ARM,",
    "            completed_episode_count_cumulative=len(probe_completed_episodes),",
    "            completed_episode_count_this_update=len(_new_eps),",
    "            pending_episode_count=sum(1 for _ps in pending.slots if len(_ps[\"obs\"]) > 0),",
    "            replay_buffer_trajectory_count=len(replay),",
    "            P50=float(np.percentile(_la, 50)), P75=float(np.percentile(_la, 75)),",
    "            P90=float(np.percentile(_la, 90)), P95=float(np.percentile(_la, 95)),",
    "            P99=float(np.percentile(_la, 99)), max_len=int(max(_lens) if _lens else 0),",
    "            count_ge_129=sum(1 for _L in _lens if _L >= 129),",
    "            count_ge_256=sum(1 for _L in _lens if _L >= 256),",
    "            count_ge_512=sum(1 for _L in _lens if _L >= 512),",
    "            fraction_ge_512=(sum(1 for _L in _lens if _L >= 512) / len(_lens) if _lens else 0.0),",
    "            termination_reason_counts=_reasons,",
    "            first_ge512=probe_first_ge512)",
    "        with open(probe_updates_path, \"a\") as f:",
    "            f.write(json.dumps(probe_upd, default=str) + \"\\n\")",
    "        print(f\"[probe u{u}] eps_cum={len(probe_completed_episodes)} this={len(_new_eps)} \"",
    "              f\"max_len={probe_upd['max_len']} ge512={probe_upd['count_ge_512']} \"",
    "              f\"first_ge512={probe_first_ge512}\", flush=True)",
])

# ---- D8: DEBUG-ONLY early stop (default OFF; formal probe must keep 0) ----
D8_OLD = block([
    "    # 4. checkpoint at save points",
    "    if (u + 1) % args.save_every == 0 or (u + 1) == args.total_updates:",
    '        save_ckpt(global_step, params, ppo_opt_state, replay_opt_state, target_params, "save")',
])
D8_NEW = block([
    "    # 4. checkpoint at save points",
    "    if (u + 1) % args.save_every == 0 or (u + 1) == args.total_updates:",
    '        save_ckpt(global_step, params, ppo_opt_state, replay_opt_state, target_params, "save")',
    "    # Phase4A probe DEBUG-ONLY early stop (default OFF; NON-COMPARATIVE debugging only). The formal",
    "    # probe MUST keep args.early_stop_len == 0 so both arms run the identical full fixed horizon.",
    "    if PROBE and args.early_stop_len and probe_first_ge512 is not None \\",
    "            and int(probe_first_ge512[\"first_ge512_length\"]) >= args.early_stop_len:",
    "        print(f\"PROBE_DEBUG_EARLY_STOP fired at u{u} (early_stop_len={args.early_stop_len}); \"",
    "              f\"NON-COMPARATIVE debug only -- formal probe must NOT stop here.\", flush=True)",
    "        break",
])

# ---- D9: probe final summary + exit(0) BEFORE the replay-horizon final gate ----
D9_OLD = "# ---- Phase4A per-arm final gates v2 (CC2 directive: reset128 read branch must be CONNECTED) ----\n"
D9_NEW = block([
    "# ---- Phase4A probe final summary (directive 4; RECORD only; NOT formal science) ----",
    "if PROBE:",
    "    _lens = [int(_e[\"length\"]) for _e in probe_completed_episodes]",
    "    _reasons = {}",
    "    for _e in probe_completed_episodes:",
    "        _reasons[_e[\"done_reason\"]] = _reasons.get(_e[\"done_reason\"], 0) + 1",
    "    probe_summary = dict(",
    "        arm=ARM, carry_mode=args.carry_mode, probe=\"REACHABILITY_ONLY\",",
    "        not_for_formal_science=True,",
    "        replay=args.replay,",
    "        replay_note=\"replay learner + hindsight OFF; buffer collection of complete done episodes ON\",",
    "        total_updates=args.total_updates,",
    "        total_env_steps=args.total_updates * STEPS_PER_UPDATE,",
    "        online_ppo_update_count=online_ppo_update_count,",
    "        completed_episode_count=len(probe_completed_episodes),",
    "        all_episode_lengths_sorted=sorted(_lens),",
    "        count_ge_512=sum(1 for _L in _lens if _L >= 512),",
    "        fraction_ge_512=(sum(1 for _L in _lens if _L >= 512) / len(_lens) if _lens else 0.0),",
    "        first_ge512=probe_first_ge512,",
    "        termination_reason_counts=_reasons,",
    "        final_params_sha256=_params_sha(params), base_sha256=base_sha,",
    "        step0_params_in=\"ckpt/0/full_state.pkl\",",
    "        early_stop_used=bool(args.early_stop_len and probe_first_ge512 is not None),",
    "        timestamp_utc=time.strftime(\"%Y-%m-%dT%H:%M:%SZ\", time.gmtime()))",
    "    with open(os.path.join(LOG_DIR, f\"{ARM}_probe_summary.json\"), \"w\") as f:",
    "        json.dump(probe_summary, f, indent=2, default=str)",
    "    print(\"RMT16_PROBE=REACHABILITY_ONLY NOT_FOR_FORMAL_SCIENCE\", flush=True)",
    "    print(\"PROBE_SUMMARY=\" + json.dumps(probe_summary, default=str), flush=True)",
    "    sys.exit(0)",
    "",
    "# ---- Phase4A per-arm final gates v2 (CC2 directive: reset128 read branch must be CONNECTED) ----",
])

REPLS = [(t, o, nw, 1) for (t, o, nw) in
         [("D1", D1_OLD, D1_NEW), ("D2", D2_OLD, D2_NEW), ("D3", D3_OLD, D3_NEW),
          ("D4", D4_OLD, D4_NEW), ("D5", D5_OLD, D5_NEW), ("D6", D6_OLD, D6_NEW),
          ("D7", D7_OLD, D7_NEW), ("D8", D8_OLD, D8_NEW), ("D9", D9_OLD, D9_NEW)]]
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
print("DRIVER_PATCHED (%d replacements)" % len(REPLS))
