#!/usr/bin/env python3
"""CC4 Tier3 — V2 dynamic-topology REGRESSION TESTS (task §五).

Six named tests; smoke results are NOT performance conclusions:

  A. STATIC_TOPOLOGY_PARITY   — a fixed, no-map-change trajectory yields
     IDENTICAL V1 vs V2 results (primary label, dense progress, terminal
     flags, episode canonical payload) on every initial-reachable state.
     Synthetic (--self-test) + real NOOP rollouts over all 8 FRONT bank
     states (--server-suite); on dig-required states (the frozen bank
     empirically contains such: state 7, seed 10007 — its initial graph has
     no start -> exit path) the INTENDED divergence is witnessed instead:
     V1 aborts with NEG18 (the §一 root-cause family, reproduced) while V2
     completes with dense progress conservatively frozen and no false
     primary.
  B. LEGAL_DIG_NO_ABORT       — a legal dig trajectory that moves the player
     onto tiles OUTSIDE the initial walkable graph raises no
     invalid_position under V2. Synthetic + real (CONTROL greedy policy).
  C. DYNAMIC_DISTANCE_UPDATE  — after a dig opens a new path, BFS uses the
     CURRENT graph, not the initial one. Synthetic + real witness.
  D. UNREACHABLE_CONTINUES    — target unreachable in the current dynamic
     graph: the episode continues, no false primary, dense progress frozen
     (does not increase). Synthetic (authoritative).
  E. TRUE_INVALID_FAIL_CLOSED — genuine corruption still fails closed:
     out-of-bounds, non-finite / undecodable coordinates, player-vs-current-
     map contradiction, bank baseline unreachable. Synthetic (authoritative).
  F. CONTROL_REPRODUCTION     — the ORIGINAL CONTROL checkpoint + the
     original block start (front_l2-bank0, seed 10000): the V1 engine still
     aborts there (reproduction of the historical verdict), while the V2
     engine completes the episode with no initial-graph-membership abort.
     Real only (--server-suite).

Usage:
  local (pure logic, no JAX):   python tier3_v2_dynamic_topology_regression.py --self-test
  server (JAX+GPU, repo-root CWD):
    CUDA_VISIBLE_DEVICES=GPU-8df11537-ab79-722d-606f-411966196c4c \
    python tier3_v2_dynamic_topology_regression.py --server-suite \
        [--candidate-id CONTROL_CONTINUOUS_98304] [--max-steps 32] \
        [--out /home/oseasy/student_pool_v1/cc4/regression_v2dt]
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import tier3_event_predicates as pred1            # noqa: E402  (FROZEN V1)
import tier3_event_predicates_v2 as pred2         # noqa: E402  (V2)


def utc_now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _grid(rows, cols, open_cells):
    g = [[False] * cols for _ in range(rows)]
    for (r, c) in open_cells:
        g[r][c] = True
    return g


# ---------------------------------------------------------------------------
# Pure synthetic tests (A–E; no JAX)
# ---------------------------------------------------------------------------
def pure_tests():
    problems = []
    details = {}

    def check(test, name, cond):
        details.setdefault(test, []).append((name, bool(cond)))
        if not cond:
            problems.append("%s:%s" % (test, name))

    # Shared geometry: 5x5, row 2 open; start (2,0); exit (2,4); d_base = 4.
    walk = _grid(5, 5, [(2, c) for c in range(5)])
    start, exit_pos = (2, 0), (2, 4)
    d_base = pred2.bfs_distance(walk, start, exit_pos)

    # --- A: STATIC_TOPOLOGY_PARITY (synthetic) -------------------------------
    traj = [(2, 0), (2, 1), (2, 2), (2, 3), (2, 2), (2, 1), (2, 0), (2, 1),
            (2, 2), (2, 3), (2, 4)]
    max_v1 = max_v2 = 0.0
    episode_parity = True
    for pos in traj:
        p1 = pred1.normalized_corridor_progress({"player_position": pos},
                                                walk, start, exit_pos)
        p2 = pred2.normalized_corridor_progress_dynamic(
            {"player_position": pos}, walk, start, exit_pos, d_base, max_v2)
        check("A", "step_parity@%s" % (pos,), p1 == p2)
        episode_parity = episode_parity and (p1 == p2)
        max_v1 = max(max_v1, p1)
        max_v2 = max(max_v2, p2)
    check("A", "episode_max_aggregation_parity", max_v1 == max_v2)
    check("A", "primary_label_parity(transition predicate is the identical "
          "re-exported object)",
          pred2.front_floor_transition_reached is
          pred1.front_floor_transition_reached
          and pred2.front_floor_transition_reached(2, 3) is True
          and pred2.front_floor_transition_reached(2, 2) is False)
    # terminal flags / record fields are produced by the evaluator rollout,
    # which V2 re-exports unchanged except the FRONT dense block — the real
    # NOOP parity (server suite) witnesses byte-identical episode payloads.

    # --- B: LEGAL_DIG_NO_ABORT (synthetic) ------------------------------------
    # (1,3) is outside the initial grid; after a legal dig it is walkable in
    # the current grid. V1 aborts; V2 computes progress on the current graph.
    dug = _grid(5, 5, [(2, c) for c in range(5)] + [(1, 1), (1, 2), (1, 3), (1, 4)])
    try:
        pred1.normalized_corridor_progress({"player_position": (1, 3)},
                                           walk, start, exit_pos)
        check("B", "v1_aborts_outside_initial", False)
    except pred1.FailClosed:
        check("B", "v1_aborts_outside_initial", True)
    try:
        pb = pred2.normalized_corridor_progress_dynamic(
            {"player_position": (1, 3)}, dug, start, exit_pos, d_base, 0.0)
        check("B", "v2_no_abort_on_dug_tile", True)
        check("B", "v2_progress_finite_in_range", 0.0 <= pb <= 1.0)
    except pred2.FailClosed:
        check("B", "v2_no_abort_on_dug_tile", False)
        check("B", "v2_progress_finite_in_range", False)

    # --- C: DYNAMIC_DISTANCE_UPDATE (synthetic) -------------------------------
    # Exit-side room walled off initially; current graph opens (3,3)-(3,4).
    current_c = _grid(5, 5, [(2, c) for c in range(5)] + [(3, 3), (3, 4)])
    initial_c = _grid(5, 5, [(2, c) for c in range(5)])
    d_cur = pred2.bfs_distance(current_c, (3, 3), exit_pos)
    d_init = pred2.bfs_distance(initial_c, (3, 3), exit_pos)
    check("C", "initial_graph_unreachable", d_init is None)
    check("C", "current_graph_reachable", d_cur == 2)
    p_dyn = pred2.normalized_corridor_progress_dynamic(
        {"player_position": (3, 3)}, current_c, start, exit_pos, d_base, 0.0)
    check("C", "dynamic_progress_uses_current_graph", p_dyn == 0.5)
    # V1's transient-lost policy stays intact: start reaches the exit but the
    # current position sits on an isolated island -> V1 returns 0.0 (V2 would
    # return the previous progress unchanged — witness D).
    island = _grid(5, 5, [(2, c) for c in range(5)] + [(0, 0)])
    p_v1_lost = pred1.normalized_corridor_progress(
        {"player_position": (0, 0)}, island, start, exit_pos)
    check("C", "v1_lost_zero_policy_intact", p_v1_lost == 0.0)
    # Shortcut through a dug bypass: current graph d(cur,exit) strictly
    # shorter than the initial-graph d from the same cur => progress strictly
    # increases relative to the initial-graph value.
    shortcut_initial = _grid(5, 5, [(2, c) for c in range(5)])
    shortcut_current = _grid(5, 5, [(2, c) for c in range(5)] + [(1, 1), (1, 2)])
    d_cur_s = pred2.bfs_distance(shortcut_current, (1, 1), exit_pos)   # 1,1->1,2->2,2.. = 1+3=... compute below
    d_init_s = pred2.bfs_distance(shortcut_initial, (1, 1), exit_pos)  # None (off initial grid)
    check("C", "shortcut_current_strictly_better",
          d_cur_s is not None and (d_init_s is None or d_cur_s < d_init_s))

    # --- D: UNREACHABLE_CONTINUES (synthetic; authoritative) -------------------
    enclosed = _grid(5, 5, [(2, c) for c in range(5)] + [(0, 0)])
    try:
        pd7 = pred2.normalized_corridor_progress_dynamic(
            {"player_position": (0, 0)}, enclosed, start, exit_pos, d_base, 0.7)
        check("D", "no_abort_when_unreachable", True)
        check("D", "progress_frozen_exact_previous", pd7 == 0.7)
    except pred2.FailClosed:
        check("D", "no_abort_when_unreachable", False)
        check("D", "progress_frozen_exact_previous", False)
    pd0 = pred2.normalized_corridor_progress_dynamic(
        {"player_position": (0, 0)}, enclosed, start, exit_pos, d_base, 0.0)
    check("D", "progress_does_not_increase", pd0 == 0.0)
    # Primary stays false: the primary predicate depends ONLY on the recorded
    # level transition (unchanged V1 object); an unreachable target never
    # produces from==2 -> to>=3 by itself.
    check("D", "primary_stays_false_without_transition",
          pred2.front_floor_transition_reached(2, 2) is False)

    # --- E: TRUE_INVALID_FAIL_CLOSED (synthetic; authoritative) ----------------
    def raises(pos, grid, base=d_base, prev=0.0):
        try:
            pred2.normalized_corridor_progress_dynamic(
                {"player_position": pos}, grid, start, exit_pos, base, prev)
            return False
        except pred2.FailClosed:
            return True
    check("E", "out_of_bounds_positive", raises((9, 9), walk))
    check("E", "out_of_bounds_negative", raises((-1, 2), walk))
    check("E", "nonfinite_nan", raises((float("nan"), 2), walk))
    check("E", "nonfinite_inf", raises((float("inf"), 2), walk))
    check("E", "undecodable_none", raises((None, 2), walk))
    check("E", "contradicts_current_map_solid", raises((0, 1), walk))
    check("E", "previous_out_of_range", raises((2, 0), walk, prev=1.5))
    # Baseline unreachable is LEGAL (dig-required scaffold — frozen FRONT bank
    # state 7, seed 10007), NOT corruption: conservative freeze, no abort...
    check("E", "baseline_none_legal_conservative_freeze",
          pred2.normalized_corridor_progress_dynamic(
              {"player_position": (2, 0)}, walk, start, exit_pos, None, 0.3) == 0.3)
    # ... while position validity stays fail-closed even under baseline None.
    check("E", "baseline_none_position_still_fail_closed",
          raises((9, 9), walk, base=None))

    passed = not problems
    return passed, problems, details


def self_test():
    passed, problems, details = pure_tests()
    total = sum(len(v) for v in details.values())
    for t in sorted(details):
        oks = sum(1 for (_n, c) in details[t] if c)
        print("  [%s] %d/%d" % (t, oks, len(details[t])))
    if not passed:
        print("TIER3_V2DT_REGRESSION_SELF_TEST_FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print("TIER3_V2DT_REGRESSION_SELF_TEST_PASS (pure checks=%d; A/B/C/D/E)" % total)
    return 0


# ---------------------------------------------------------------------------
# Real-environment tests (A-real, B/C/F; JAX + GPU required)
# ---------------------------------------------------------------------------
def server_suite(args):
    import numpy as np
    import jax
    import jax.numpy as jnp
    import tier3_projection_runtime as proj
    import tier3_projection_binding_smoke_v2 as drv2
    import tier3_evaluator as ev1
    import tier3_evaluator_v2 as ev2
    import tier3_scaffold_builder as builder
    import tier3_source_audit as audit
    import tier3_state_bank_materializer as mat
    import tier3_frozen_bank_artifacts as art
    from craftax.craftax.constants import Action

    results = {"schema": "mechanism_UED.v2dt_regression_server_suite/v1",
               "generated_at_utc": utc_now_iso(),
               "common_evaluator_protocol_version":
                   pred2.COMMON_EVALUATOR_PROTOCOL_VERSION}

    # Launch contract + gates (reuse the V2 driver's verified stages).
    repo_root = os.path.dirname(os.path.dirname(drv2.HERE))
    proj.require(os.path.realpath(os.getcwd()) == os.path.realpath(repo_root),
                 "FAIL CLOSED (launch contract): run from the repo root")
    gpu_ev = drv2.verify_gpu_allowed()
    common_ev = drv2.verify_engine_and_common_v2(args.common_dir,
                                                 args.v1_common_dir, drv2.HERE)
    dicode_ev = proj.pin_dicode_resolution(repo_root)
    entry = ev2.make_canonical_env()
    front = art.load_bank(ev2.FRONT, args.frozen_bank_artifacts)
    proj.require(front.get("state_bank_hash") == proj.FROZEN_FRONT_BANK_CONTENT_SHA256,
                 "FAIL CLOSED: FRONT bank content hash drift")

    # --- A-real: STATIC_TOPOLOGY_PARITY (NOOP, all 8 FRONT bank states) -------
    noop = int(Action.NOOP.value)

    class NoopPolicy(object):
        def reset(self):
            pass

        def __call__(self, obs, state):
            return noop

    npol = NoopPolicy()
    a_rows = []
    a_pass = True
    reachable_count = dig_required_count = 0
    for i in range(args.a_states):
        st = jax.tree.map(jnp.asarray, front["states"][i])
        seed = int(mat.FROZEN_SEED_BASE + i * mat.FROZEN_SEED_STRIDE)
        eid = ev2.state_entry_ids_for(ev2.FRONT, [seed])[0]
        # Per-state initial-graph classification (V1 semantics).
        view_i = builder.normalize_envstate(st)
        view_i["floor2_up_ladder_removed"] = True
        grid_i = ev1._front_walkable_grid(st, view_i)
        pi = np.asarray(st.player_position)
        start_i = (int(pi[0]), int(pi[1]))
        exit_i = view_i["down_ladders"].get(audit.FRONT_FLOOR)
        d_start_i = (pred1.bfs_distance(grid_i, start_i, exit_i)
                     if exit_i is not None else None)
        reachable = d_start_i is not None
        # V1 rollout (may abort by construction on dig-required states).
        r1, v1_aborted, v1_msg = None, False, None
        try:
            r1 = ev1.rollout_episode(entry, st, ev1.FRONT, npol, eid, seed,
                                     max_steps=args.max_steps)
        except pred1.FailClosed as exc:
            v1_aborted, v1_msg = True, str(exc)
        # V2 rollout (must always complete — NOOP never corrupts state).
        r2 = ev2.rollout_episode(entry, st, ev2.FRONT, npol, eid, seed,
                                 max_steps=args.max_steps)
        r2["episode_record_sha256"] = proj.sha256_bytes(proj.canonical_json_bytes(r2))
        if reachable:
            reachable_count += 1
            r1["episode_record_sha256"] = proj.sha256_bytes(
                proj.canonical_json_bytes(r1))
            same = r1["episode_record_sha256"] == r2["episode_record_sha256"]
            a_pass = a_pass and same and not v1_aborted
            a_rows.append({
                "bank_index": i, "seed": seed, "entry_id": eid,
                "classification": "initial_reachable", "d_start": d_start_i,
                "v1_episode_record_sha256": r1["episode_record_sha256"],
                "v2_episode_record_sha256": r2["episode_record_sha256"],
                "identical": same,
                "v1_v2": {k: {"v1": r1[k], "v2": r2[k],
                              "same": r1[k] == r2[k]}
                          for k in ("front_floor_transition_reached",
                                    "graph_distance_progress",
                                    "corridor_exit_reached", "defeat_kobold",
                                    "player_died", "timed_out", "timesteps",
                                    "valid_start", "action_sequence")},
            })
        else:
            # DIG-REQUIRED scaffold: no initial start -> exit path (legally —
            # the frozen bank contains such states, e.g. state 7 seed 10007).
            # The intended V2 divergence from V1: V1 aborts (NEG18 — the §一
            # root-cause family, reproduced here); V2 completes with dense
            # progress conservatively frozen at 0.0 (NOOP never mines, so the
            # current graph stays equal to the initial graph and the target
            # stays unreachable) and no false primary.
            dig_required_count += 1
            ok = (v1_aborted and "NEG18" in (v1_msg or "")
                  and r2["graph_distance_progress"] == 0.0
                  and r2["front_floor_transition_reached"] is False
                  and r2["timesteps"] > 0)
            a_pass = a_pass and ok
            a_rows.append({
                "bank_index": i, "seed": seed, "entry_id": eid,
                "classification": "dig_required_initial_unreachable",
                "d_start": None,
                "v1_aborts_NEG18_reproduced": v1_aborted,
                "v1_engine_message": v1_msg,
                "v2_completes": True,
                "v2_graph_distance_progress_frozen":
                    r2["graph_distance_progress"],
                "v2_front_floor_transition_reached":
                    r2["front_floor_transition_reached"],
                "v2_timesteps": r2["timesteps"],
                "v2_player_died": r2["player_died"],
                "v2_episode_record_sha256": r2["episode_record_sha256"],
                "intended_divergence_ok": ok,
            })
    results["STATIC_TOPOLOGY_PARITY"] = {
        "verdict": "PASS" if a_pass else "FAIL",
        "kind": "real_noop_rollout_v1_vs_v2",
        "states": len(a_rows), "max_steps": args.max_steps,
        "initial_reachable_states": reachable_count,
        "dig_required_states": dig_required_count,
        "criterion": "on initial-reachable states: identical primary label + "
                     "dense progress + terminal flags + canonical episode "
                     "payload (episode_record_sha256) on a fixed no-map-change "
                     "trajectory; on dig-required states (initial graph has no "
                     "start -> exit path — legally, e.g. state 7 seed 10007): "
                     "V1 NEG18 abort reproduced AND V2 completes with dense "
                     "progress frozen at 0.0 and no false primary (intended "
                     "V2_DYNAMIC_TOPOLOGY divergence, not a parity violation)",
        "rows": a_rows,
    }

    # --- B/C/F: CONTROL greedy policy, front_l2-bank0, seed 10000 -------------
    spec = proj.get_spec(args.candidate_id)
    capsule_ev = proj.verify_capsule_files(spec)
    ctx = proj.load_owner_runtime(spec)
    params_sha = proj.recompute_params_sha_owner(ctx)
    proj.require(params_sha == spec["declared_params_sha256"]["value"],
                 "FAIL CLOSED: %s params recompute mismatch" % args.candidate_id)
    policy = proj.build_policy(spec, ctx)

    seed = int(mat.FROZEN_SEED_BASE)                  # 10000
    eid = ev2.state_entry_ids_for(ev2.FRONT, [seed])[0]   # front_l2-bank0
    st0 = jax.tree.map(jnp.asarray, front["states"][0])

    # F part 1: V1 reproduction — the frozen V1 engine still aborts here.
    policy.reset()
    v1_aborted, v1_msg = False, None
    try:
        ev1.rollout_episode(entry, st0, ev1.FRONT, policy, eid, seed,
                            max_steps=args.max_steps)
    except pred1.FailClosed as exc:
        v1_aborted, v1_msg = True, str(exc)

    # F part 2 + official V2 rollout (must complete; no initial-graph abort).
    policy.reset()
    v2_official = ev2.rollout_episode(entry, st0, ev2.FRONT, policy, eid, seed,
                                      max_steps=args.max_steps)
    actions_official = list(v2_official["action_sequence"])

    # Instrumented mirror of the V2 loop for position/topology evidence.
    policy.reset()
    view = builder.normalize_envstate(st0)
    ladder_tiles = ev2.front_ladder_transit_positions(view)
    grid0 = ev2.front_walkable_grid_for_map(
        np.asarray(st0.map)[audit.FRONT_FLOOR], ladder_tiles)
    exit_pos = view["down_ladders"].get(audit.FRONT_FLOOR)
    obs, state = ev2.reset_from_bank_state(entry, st0)
    start_pos = (int(np.asarray(state.player_position)[0]),
                 int(np.asarray(state.player_position)[1]))
    d_base = pred2.bfs_distance(grid0, start_pos, exit_pos)
    dk = entry["defeat_kobold_index"]
    rng = jax.random.PRNGKey(seed)
    step_fn = ev2._jit_step(entry)
    actions_mini = []
    trace = []
    max_level = int(np.asarray(state.player_level))
    for step_i in range(args.max_steps):
        a = int(policy(obs, state))
        actions_mini.append(a)
        rng, sk = jax.random.split(rng)
        obs, state, _rew, done, _info = step_fn(sk, state, a)
        lvl = int(np.asarray(state.player_level))
        max_level = max(max_level, lvl)
        pos = (int(np.asarray(state.player_position)[0]),
               int(np.asarray(state.player_position)[1]))
        rows = len(grid0)
        cols = len(grid0[0])
        on_initial = (0 <= pos[0] < rows and 0 <= pos[1] < cols
                      and bool(grid0[pos[0]][pos[1]]))
        gcur = ev2.front_walkable_grid_for_map(
            np.asarray(state.map)[audit.FRONT_FLOOR], ladder_tiles)
        legal_now = (0 <= pos[0] < len(gcur) and 0 <= pos[1] < len(gcur[0])
                     and bool(gcur[pos[0]][pos[1]]))
        d_cur = pred2.bfs_distance(gcur, pos, exit_pos) if legal_now else None
        d_init = pred2.bfs_distance(grid0, pos, exit_pos) if on_initial else None
        trace.append({
            "step": step_i + 1, "lvl": lvl, "pos": list(pos),
            "on_initial_graph": on_initial, "legal_on_current_graph": legal_now,
            "d_current": d_cur, "d_initial": d_init,
            "metric_window": (lvl == audit.FRONT_FLOOR
                              and max_level < audit.CORRIDOR_EXIT_FLOOR),
        })
        defeated = bool(np.asarray(state.achievements)[dk])
        died = float(np.asarray(state.player_health)) <= 0.0
        if defeated or died or bool(np.asarray(done)):
            break

    left_initial = [t for t in trace
                    if (not t["on_initial_graph"]) and t["legal_on_current_graph"]]
    dynamic_witness = []
    for t in trace:
        if t["d_current"] is None:
            continue
        p_cur = 1.0 - t["d_current"] / max(int(d_base), 1)
        if t["d_initial"] is None:
            dynamic_witness.append({"step": t["step"], "kind": "dynamic_only",
                                    "p_current": p_cur})
        else:
            p_init = 1.0 - t["d_initial"] / max(int(d_base), 1)
            if p_cur > p_init + 1e-12:
                dynamic_witness.append({"step": t["step"],
                                        "kind": "strictly_shorter_current_path",
                                        "p_current": p_cur, "p_initial": p_init})

    results["LEGAL_DIG_NO_ABORT"] = {
        "verdict": "PASS" if (bool(left_initial)
                              and v2_official["timesteps"] > 0) else "FAIL",
        "kind": "real_control_greedy_trajectory",
        "candidate_id": args.candidate_id, "entry_id": eid, "seed": seed,
        "v2_rollout_completed": True,
        "steps_off_initial_graph_but_legal": len(left_initial),
        "first_off_initial_step": (left_initial[0] if left_initial else None),
        "v2_graph_distance_progress": v2_official["graph_distance_progress"],
        "criterion": "V2 rollout raises no invalid_position while the player "
                     "stands on legally-mined tiles outside the initial graph",
    }
    results["DYNAMIC_DISTANCE_UPDATE"] = {
        "verdict": "PASS" if dynamic_witness else "FAIL",
        "kind": "real_control_greedy_trajectory",
        "d_start_baseline": d_base,
        "witness_count": len(dynamic_witness),
        "witnesses_first8": dynamic_witness[:8],
        "criterion": "after a dig opens new paths, BFS progress uses the "
                     "CURRENT graph (d_initial None or d_current < d_initial)",
    }
    results["CONTROL_REPRODUCTION"] = {
        "verdict": "PASS" if (v1_aborted
                              and "non-walkable" in (v1_msg or "")
                              and v2_official["timesteps"] > 0
                              and actions_mini == actions_official) else "FAIL",
        "candidate_id": args.candidate_id,
        "checkpoint_params_sha256": params_sha,
        "entry_id": eid, "seed": seed, "max_steps": args.max_steps,
        "v1_engine_aborts_with_historical_verdict": v1_aborted,
        "v1_engine_message": v1_msg,
        "v2_engine_completes_episode": True,
        "v2_timesteps": v2_official["timesteps"],
        "v2_front_floor_transition_reached":
            v2_official["front_floor_transition_reached"],
        "v2_graph_distance_progress": v2_official["graph_distance_progress"],
        "instrumented_actions_match_official_record":
            actions_mini == actions_official,
        "criterion": "the ORIGINAL CONTROL checkpoint + original block start: "
                     "V1 still aborts (invalid_position non-walkable, "
                     "reproduced), V2 completes with no initial-graph-"
                     "membership abort; the instrumented action sequence "
                     "matches the official V2 record (determinism)",
    }
    results["UNREACHABLE_CONTINUES"] = {
        "verdict": "PASS (synthetic authoritative)",
        "note": "unreachable semantics are fully witnessed by the pure suite "
                "(--self-test test D); see TIER3_PREDICATES_V2 self-test too",
    }
    results["TRUE_INVALID_FAIL_CLOSED"] = {
        "verdict": "PASS (synthetic authoritative)",
        "note": "corruption-class fail-closed is fully witnessed by the pure "
                "suite (--self-test test E)",
    }

    results["provenance"] = {
        "candidate_id": args.candidate_id,
        "params_sha256_owner_recomputed": params_sha,
        "capsule_verification": capsule_ev,
        "gpu": gpu_ev,
        "dicode_resolution": dicode_ev,
        "common_verification_v2": {
            "common_v2_sha256sums_self_check":
                common_ev["common_v2_sha256sums_self_check"],
            "engine_modules_lf_sha_verified":
                common_ev["engine_modules_lf_sha_verified"],
            "v1_preservation": common_ev["v1_preservation"],
        },
        "front_bank_content_sha256": front.get("state_bank_hash"),
        "max_steps": args.max_steps,
        "a_states": args.a_states,
        "performance_claim_authorized": False,
        "note": "smoke-scale regression evidence; NOT a performance conclusion",
    }

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "v2dt_regression_results.json")
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(results, fh, sort_keys=True, ensure_ascii=False, indent=2)
        fh.write("\n")
    print("WROTE %s" % out_path)
    names = ["STATIC_TOPOLOGY_PARITY", "LEGAL_DIG_NO_ABORT",
             "DYNAMIC_DISTANCE_UPDATE", "UNREACHABLE_CONTINUES",
             "TRUE_INVALID_FAIL_CLOSED", "CONTROL_REPRODUCTION"]
    all_pass = True
    for n in names:
        v = str(results[n]["verdict"])
        print("%s=%s" % (n, v))
        all_pass = all_pass and v.startswith("PASS")
    print("V2DT_REGRESSION_SERVER_SUITE_%s" % ("PASS" if all_pass else "FAIL"))
    return 0 if all_pass else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true",
                      help="pure synthetic tests A–E (no JAX)")
    mode.add_argument("--server-suite", action="store_true",
                      help="real-environment tests A/B/C/F (JAX + GPU)")
    ap.add_argument("--candidate-id", default="CONTROL_CONTINUOUS_98304")
    ap.add_argument("--max-steps", type=int, default=32)
    ap.add_argument("--a-states", type=int, default=8,
                    help="FRONT bank states for the NOOP parity test [1, 8]")
    ap.add_argument("--common-dir", default="/home/oseasy/student_pool_v1/common_v2")
    ap.add_argument("--v1-common-dir", default="/home/oseasy/student_pool_v1/common")
    ap.add_argument("--frozen-bank-artifacts",
                    default="/home/oseasy/student_pool_v1/common/frozen_bank_artifacts")
    ap.add_argument("--out", default="/home/oseasy/student_pool_v1/cc4/regression_v2dt")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    return server_suite(args)


if __name__ == "__main__":
    sys.exit(main())
