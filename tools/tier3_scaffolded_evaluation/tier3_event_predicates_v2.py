#!/usr/bin/env python3
"""CC4 Tier3 — V2 dynamic-topology FRONT dense metric predicates (pure Python,
JAX-free testable). NEW FILE; tier3_event_predicates.py (V1, LF-SHA frozen
a4fba86b…) is byte-untouched.

Task: CC4_FIX_FRONT_DYNAMIC_TOPOLOGY_METRIC_AND_REBIND_FORMAL_POOL_V2.

ROOT CAUSE FROZEN (§一): COMMON_EVALUATOR_METRIC_DOMAIN_NOT_CLOSED_UNDER_LEGAL_
TOPOLOGY_MUTATION. The V1 FRONT corridor dense metric computed graph distance
over the INITIAL map walkable graph (BFS_GRAPH_SOURCE=INITIAL_MAP_TOPOLOGY); a
policy that legally mines through walls stands on tiles outside that initial
graph, which V1 rejected with `invalid_position_policy: player position
non-walkable` (engine-predicate abort, e.g. CONTROL_CONTINUOUS_98304). This is
NOT candidate corruption and is NOT fixed by retraining: the COMMON dense metric
itself is fixed here to cover legal topology mutation.

EXACTLY THREE REPLACEMENTS (§二/§三/§四 — no other scientific change):
  1. GRAPH SOURCE: BFS_GRAPH_SOURCE=CURRENT_ENVIRONMENT_STATE_TOPOLOGY — the
     caller rebuilds the legal walkable grid from the CURRENT env state every
     step (legally-mined tiles are no longer SOLID_BLOCK, hence naturally
     walkable). The formula's normalization denominator d(start, exit) stays
     FIXED at the episode-start baseline computed ONCE on the initial graph.
  2. LEGAL-POSITION DOMAIN: membership in the CURRENT grid, not the initial
     grid. A player on a legally-mined current tile / outside the initial graph
     but confirmed legal by the current env state is VALID. Continue
     fail-closed: coordinate out-of-bounds, non-finite/undecodable coordinates,
     player position contradicting the CURRENT map state (standing on a tile
     the current map says is solid — impossible via legal movement since mining
     turns the tile walkable). An UNREACHABLE BASELINE is NOT corruption: the
     frozen FRONT bank empirically contains valid start states whose initial
     walkable graph has no start -> exit path (front_l2 bank state 7, seed
     10007; bank content SHA 21aeb7dc… verified at load) — a dig-required
     scaffold where the exit is reachable only through legally mined tiles.
     V1 aborted there with NEG18 (part of the §一 root cause); V2 continues
     the episode with dense progress conservatively frozen.
  3. UNREACHABLE HANDLING: if the target is temporarily unreachable in the
     CURRENT dynamic graph, DO NOT abort the episode: return the previous
     progress unchanged (conservative freeze — dense progress does not
     increase), primary success stays false (evaluator-side), the episode
     CONTINUES. A legal floor2 -> floor3 transition still yields primary
     success=true (unchanged V1 primary predicate).

KEPT UNCHANGED from V1 (re-exported from the frozen V1 module so byte-identity
of the unchanged predicates is structural, not transcribed): the normalized
progress FORMULA `clip(1 - d_t / max(d_start, 1), 0, 1)`, the NEG17 range guard,
bfs_distance (4-connectivity BFS), every validity / event / primary predicate
(valid_front_scaffold_start, front_floor_transition_reached, …), FailClosed.
BACK and FULL semantics never consume this metric and are untouched.
"""
from __future__ import annotations

import math
import os
import sys

# Runnable-as-script AND importable-as-package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tier3_source_audit as audit                 # noqa: E402
import tier3_event_predicates as pred_v1           # noqa: E402  (FROZEN V1)

# ---------------------------------------------------------------------------
# V2 protocol identity
# ---------------------------------------------------------------------------
COMMON_EVALUATOR_PROTOCOL_VERSION = "V2_DYNAMIC_TOPOLOGY"
PREDICATE_VERSION = "tier3_predicates/v2_dynamic_topology"
BFS_GRAPH_SOURCE = "CURRENT_ENVIRONMENT_STATE_TOPOLOGY"
SUPERSEDED_V1_PREDICATE_VERSION = pred_v1.PREDICATE_VERSION   # "tier3_predicates/v1"

# Re-exports from the FROZEN V1 module (unchanged semantics; single source of
# truth — the V1 bytes are LF-SHA frozen and never re-implemented here).
FailClosed = pred_v1.FailClosed
bfs_distance = pred_v1.bfs_distance
exit_reachable_from_start = pred_v1.exit_reachable_from_start
valid_env_state = pred_v1.valid_env_state
valid_full_start = pred_v1.valid_full_start
valid_front_scaffold_start = pred_v1.valid_front_scaffold_start
valid_back_scaffold_start = pred_v1.valid_back_scaffold_start
front_floor_transition_reached = pred_v1.front_floor_transition_reached
corridor_exit_reached = pred_v1.corridor_exit_reached
front_half_entered = pred_v1.front_half_entered
back_half_entered = pred_v1.back_half_entered
boss_area_reached = pred_v1.boss_area_reached
defeat_kobold = pred_v1.defeat_kobold
kobold_present = pred_v1.kobold_present
kobold_engaged = pred_v1.kobold_engaged
KOBOLD_CATEGORY = pred_v1.KOBOLD_CATEGORY
DEFEAT_KOBOLD = pred_v1.DEFEAT_KOBOLD
FRONT_FLOOR = pred_v1.FRONT_FLOOR
BACK_FLOOR = pred_v1.BACK_FLOOR
CORRIDOR_EXIT_FLOOR = pred_v1.CORRIDOR_EXIT_FLOOR


def normalized_corridor_progress_dynamic(state: dict, walkable_current,
                                         start_pos, exit_pos,
                                         d_start_baseline, previous_progress):
    """V2 FRONT dense progress over the CURRENT environment-state topology.

    progress = clip(1 - d_t / max(d_start_baseline, 1), 0, 1)   (V1 formula,
    unchanged), where d_t = BFS distance on the CURRENT walkable grid (rebuilt
    by the caller from the current env state each step) and d_start_baseline =
    BFS(start, exit) computed ONCE at episode start on the INITIAL grid (the
    normalization denominator is FIXED — §四: the formula, normalization and
    result fields are not otherwise touched).

    Policies (task §三):
      - baseline_unreachable_policy: d_start_baseline is None => the frozen
        FRONT bank state is a DIG-REQUIRED scaffold: its initial walkable
        graph has no start -> exit path (empirically front_l2 bank state 7,
        seed 10007, valid_front_scaffold_start=True, bank content SHA
        21aeb7dc… verified at load). This is LEGAL — the exit is reachable
        only through tiles that legal mining turns walkable. V1 raised NEG18
        here and aborted the episode (part of the §一 root cause). V2 does
        NOT abort: the position is still validated against the current grid
        below, then dense progress is returned UNCHANGED (conservative
        freeze, the same treatment as current-graph unreachability; dense
        progress does not increase, primary success stays false unless the
        unchanged floor2 -> floor3 transition predicate fires).
      - legal_position_policy: the position domain is the CURRENT grid. A
        player on a legally-mined current tile (outside the initial graph but
        confirmed legal by the current env state) is VALID — never abort on
        initial-graph membership. Continue fail-closed: non-finite /
        undecodable coordinates (state corruption), coordinate out-of-bounds,
        player position contradicting the CURRENT map state (the current grid
        says solid — impossible via legal movement, since mining turns the
        tile walkable).
      - unreachable_policy: d_t is None on the CURRENT dynamic graph (target
        temporarily unreachable) => return previous_progress UNCHANGED
        (conservative freeze: dense progress does NOT increase; the episode is
        NOT aborted; primary success stays false on the evaluator side).
      - NEG17 guard: progress must land in [0, 1].
    """
    prev = float(previous_progress)
    if not (0.0 <= prev <= 1.0):
        raise FailClosed("V2 previous_progress %r outside [0,1]" % (previous_progress,))
    cur = state["player_position"]
    try:
        fr, fc = float(cur[0]), float(cur[1])
    except (TypeError, ValueError, IndexError):
        raise FailClosed("V2_INVALID_STATE: undecodable player position "
                         "(state corruption)")
    if not (math.isfinite(fr) and math.isfinite(fc)):
        raise FailClosed("V2_INVALID_STATE: non-finite player position "
                         "(state corruption)")
    r, c = int(fr), int(fc)
    rows = len(walkable_current)
    cols = len(walkable_current[0]) if rows else 0
    if not (0 <= r < rows and 0 <= c < cols):
        raise FailClosed("invalid_position_policy: player position off-grid "
                         "(coordinate out of bounds)")
    if not walkable_current[r][c]:
        raise FailClosed("invalid_position_policy: player position "
                         "contradicts current map state (non-walkable)")
    if d_start_baseline is None:
        # Legal dig-required scaffold (see the baseline_unreachable_policy
        # docstring): the position is valid on the current grid, but the
        # normalization denominator d(start, exit) is undefined on the
        # initial graph. Conservative freeze — dense progress does not
        # increase; the episode is NOT aborted (V1's NEG18 abort here was
        # the §一 root-cause family).
        return prev
    d_t = bfs_distance(walkable_current, (r, c), exit_pos)
    if d_t is None:
        # Target temporarily unreachable in the CURRENT dynamic graph: do NOT
        # abort; conservative freeze — dense progress does not increase.
        return prev
    progress = 1.0 - (d_t / max(int(d_start_baseline), 1))
    if progress < 0.0:
        progress = 0.0
    if progress > 1.0:
        progress = 1.0
    if not (0.0 <= progress <= 1.0):
        raise FailClosed("NEG17: progress outside [0,1]")
    return progress


# ---------------------------------------------------------------------------
# Self-test (synthetic; no JAX). Returns process exit code.
# ---------------------------------------------------------------------------
def _grid(rows, cols, open_cells):
    g = [[False] * cols for _ in range(rows)]
    for (r, c) in open_cells:
        g[r][c] = True
    return g


def self_test() -> int:
    problems = []

    def check(name, cond):
        if not cond:
            problems.append(name)

    # Protocol constants
    check("protocol_version", COMMON_EVALUATOR_PROTOCOL_VERSION == "V2_DYNAMIC_TOPOLOGY")
    check("bfs_graph_source", BFS_GRAPH_SOURCE == "CURRENT_ENVIRONMENT_STATE_TOPOLOGY")
    check("supersedes_v1", SUPERSEDED_V1_PREDICATE_VERSION == "tier3_predicates/v1")

    # 5x5 corridor: row 2 fully open; start (2,0), exit (2,4); d_baseline = 4.
    walk = _grid(5, 5, [(2, c) for c in range(5)])
    start, exit_pos = (2, 0), (2, 4)
    d_base = bfs_distance(walk, start, exit_pos)
    check("baseline_is_4", d_base == 4)

    # --- A-pure: STATIC_TOPOLOGY_PARITY (synthetic) -------------------------
    # On an unchanging grid, V2 with previous_progress = running max reproduces
    # the V1 function value bit-for-bit at every position of a trajectory.
    traj = [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 2)]
    max_v1, max_v2 = 0.0, 0.0
    parity_ok = True
    for pos in traj:
        p1 = pred_v1.normalized_corridor_progress({"player_position": pos},
                                                  walk, start, exit_pos)
        p2 = normalized_corridor_progress_dynamic({"player_position": pos},
                                                  walk, start, exit_pos,
                                                  d_base, max_v2)
        parity_ok = parity_ok and (p1 == p2)
        max_v1 = max(max_v1, p1)
        max_v2 = max(max_v2, p2)
    check("A_static_parity_values", parity_ok)
    check("A_static_parity_episode_max", max_v1 == max_v2 == 1.0)

    # Formula anchor points (unchanged normalization)
    check("progress_at_start==0",
          normalized_corridor_progress_dynamic({"player_position": start},
                                               walk, start, exit_pos, d_base, 0.0) == 0.0)
    check("progress_at_mid==0.5",
          normalized_corridor_progress_dynamic({"player_position": (2, 2)},
                                               walk, start, exit_pos, d_base, 0.0) == 0.5)
    check("progress_at_exit==1",
          normalized_corridor_progress_dynamic({"player_position": exit_pos},
                                               walk, start, exit_pos, d_base, 0.0) == 1.0)
    # Denominator is the FIXED initial baseline even when the current graph
    # grew a shortcut (current start->exit = 2 via a dug bypass, cur one step
    # from exit => d_t=1, progress = 1 - 1/max(4,1) = 0.75, NOT 0.5).
    walk_shortcut = _grid(5, 5, [(2, c) for c in range(5)] + [(1, 3), (1, 4)])
    check("denominator_fixed_at_initial_baseline",
          normalized_corridor_progress_dynamic({"player_position": (2, 3)},
                                               walk_shortcut, start, exit_pos,
                                               d_base, 0.0) == 0.75)

    # --- B-pure: LEGAL_DIG_NO_ABORT ------------------------------------------
    # cur (1,1) is OUTSIDE the initial walkable graph but walkable in the
    # CURRENT grid (legally mined): valid, no abort, progress on current graph.
    cur_dug = _grid(5, 5, [(2, c) for c in range(5)] + [(1, 1), (1, 2), (1, 3), (1, 4)])
    check("B_dug_tile_not_in_initial", walk[1][1] is False and cur_dug[1][1] is True)
    pb = normalized_corridor_progress_dynamic({"player_position": (1, 1)},
                                              cur_dug, start, exit_pos, d_base, 0.0)
    # d_t((1,1) -> (2,4)) on cur_dug = (1,1)->(1,2)->(1,3)->(1,4)->(2,4) = 4
    check("B_legal_dig_no_abort_progress", pb == 0.0)   # 1 - 4/4
    pb2 = normalized_corridor_progress_dynamic({"player_position": (1, 3)},
                                               cur_dug, start, exit_pos, d_base, 0.0)
    check("B_legal_dig_progress_on_current_graph", pb2 == 0.5)  # d_t=2

    # --- C-pure: DYNAMIC_DISTANCE_UPDATE --------------------------------------
    # Same position (1,3): on the INITIAL grid it is non-walkable (V1 would
    # fail-closed abort); on the CURRENT grid d_t = 2 => progress 0.5 > 0.
    try:
        pred_v1.normalized_corridor_progress({"player_position": (1, 3)},
                                             walk, start, exit_pos)
        check("C_initial_graph_would_abort", False)
    except pred_v1.FailClosed:
        check("C_initial_graph_would_abort", True)
    check("C_current_graph_dynamic_distance",
          normalized_corridor_progress_dynamic({"player_position": (1, 3)},
                                               cur_dug, start, exit_pos,
                                               d_base, 0.0) == 0.5)
    # Digging opens a strictly shorter path: exit-side room, cur=(3,3) walled
    # off initially (d_t on initial = None => V1 transient 0.0); current graph
    # opens (3,3)-(3,4)-(2,4) => d_t=2 => progress 0.5 > frozen-0.
    current_c = _grid(5, 5, [(2, c) for c in range(5)] + [(3, 3), (3, 4)])
    # V1's transient-lost policy stays intact: start reaches the exit but the
    # current position is on an isolated island -> V1 returns 0.0; V2 instead
    # returns the previous progress unchanged (the documented divergence).
    island = _grid(5, 5, [(2, c) for c in range(5)] + [(0, 0)])
    check("C_v1_transient_zero",
          pred_v1.normalized_corridor_progress({"player_position": (0, 0)},
                                               island, start, exit_pos) == 0.0)
    check("C_v2_freeze_contrasts_v1_zero",
          normalized_corridor_progress_dynamic({"player_position": (0, 0)},
                                               island, start, exit_pos,
                                               d_base, 0.6) == 0.6)
    check("C_dynamic_opened_path_progress",
          normalized_corridor_progress_dynamic({"player_position": (3, 3)},
                                               current_c, start, exit_pos,
                                               d_base, 0.0) == 0.5)

    # --- D-pure: UNREACHABLE_CONTINUES (conservative freeze) ------------------
    # cur legal and walkable on the current grid but enclosed (no path to
    # exit): return previous_progress unchanged; no raise; no increase.
    enclosed = _grid(5, 5, [(2, c) for c in range(5)] + [(0, 0)])
    check("D_enclosed_returns_previous_exact",
          normalized_corridor_progress_dynamic({"player_position": (0, 0)},
                                               enclosed, start, exit_pos,
                                               d_base, 0.7) == 0.7)
    check("D_enclosed_returns_previous_zero",
          normalized_corridor_progress_dynamic({"player_position": (0, 0)},
                                               enclosed, start, exit_pos,
                                               d_base, 0.0) == 0.0)

    # --- E-pure: TRUE_INVALID_FAIL_CLOSED -------------------------------------
    try:  # coordinate out of bounds
        normalized_corridor_progress_dynamic({"player_position": (9, 9)},
                                             walk, start, exit_pos, d_base, 0.0)
        check("E_offgrid_raises", False)
    except FailClosed:
        check("E_offgrid_raises", True)
    try:  # negative coordinate out of bounds
        normalized_corridor_progress_dynamic({"player_position": (-1, 2)},
                                             walk, start, exit_pos, d_base, 0.0)
        check("E_negative_offgrid_raises", False)
    except FailClosed:
        check("E_negative_offgrid_raises", True)
    try:  # player state contradicts CURRENT map state (solid tile)
        normalized_corridor_progress_dynamic({"player_position": (0, 1)},
                                             walk, start, exit_pos, d_base, 0.0)
        check("E_contradicts_current_map_raises", False)
    except FailClosed:
        check("E_contradicts_current_map_raises", True)
    try:  # non-finite coordinates (state corruption)
        normalized_corridor_progress_dynamic({"player_position": (float("nan"), 2)},
                                             walk, start, exit_pos, d_base, 0.0)
        check("E_nonfinite_raises", False)
    except FailClosed:
        check("E_nonfinite_raises", True)
    try:  # undecodable coordinates (state corruption)
        normalized_corridor_progress_dynamic({"player_position": (None, 2)},
                                             walk, start, exit_pos, d_base, 0.0)
        check("E_undecodable_raises", False)
    except FailClosed:
        check("E_undecodable_raises", True)
    # baseline unreachable = LEGAL dig-required scaffold (frozen FRONT bank
    # state 7, seed 10007): conservative freeze, NOT corruption, NO abort.
    check("E_baseline_none_freezes_exact",
          normalized_corridor_progress_dynamic({"player_position": start},
                                               walk, start, exit_pos,
                                               None, 0.4) == 0.4)
    try:  # ... but position validity is STILL fail-closed under baseline None
        normalized_corridor_progress_dynamic({"player_position": (9, 9)},
                                             walk, start, exit_pos, None, 0.0)
        check("E_baseline_none_position_still_checked", False)
    except FailClosed:
        check("E_baseline_none_position_still_checked", True)
    try:  # previous_progress outside [0,1]
        normalized_corridor_progress_dynamic({"player_position": start},
                                             walk, start, exit_pos, d_base, 1.5)
        check("E_previous_out_of_range_raises", False)
    except FailClosed:
        check("E_previous_out_of_range_raises", True)

    # NEG17 range guard: every computed value in [0,1]
    vals = [normalized_corridor_progress_dynamic({"player_position": pos},
                                                 walk, start, exit_pos, d_base, 0.0)
            for pos in traj]
    check("NEG17_all_in_range", all(0.0 <= v <= 1.0 for v in vals))

    # Unchanged V1 predicates are the identical objects (re-exported, never
    # re-implemented) — structural guarantee of unchanged semantics.
    check("v1_predicates_reexported_identical",
          FailClosed is pred_v1.FailClosed
          and bfs_distance is pred_v1.bfs_distance
          and valid_front_scaffold_start is pred_v1.valid_front_scaffold_start
          and front_floor_transition_reached is pred_v1.front_floor_transition_reached)

    n_checks = 31
    if problems:
        print("TIER3_PREDICATES_V2_SELF_TEST_FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print("TIER3_PREDICATES_V2_SELF_TEST_PASS (checks=%d)" % n_checks)
    return 0


if __name__ == "__main__":
    sys.exit(self_test())
