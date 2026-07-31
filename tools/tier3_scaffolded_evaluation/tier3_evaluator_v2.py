#!/usr/bin/env python3
"""CC4 Tier3 — V2 dynamic-topology evaluator (deterministic; inference-only).
NEW FILE; tier3_evaluator.py (V1, LF-SHA frozen 54ae18db…) is byte-untouched.

Task: CC4_FIX_FRONT_DYNAMIC_TOPOLOGY_METRIC_AND_REBIND_FORMAL_POOL_V2.

This module is the V1 evaluator with EXACTLY ONE overridden code path:
rollout_episode's FRONT dense-metric block. Everything else — canonical env
construction, bank reset, episode-record validation (NEG19), failure taxonomy
(NEG20), metric aggregation, schedules, BACK and FULL semantics, terminal
labels, result fields, the frozen contract (action_mode=greedy_argmax,
max_timesteps=4096) — is RE-EXPORTED from the frozen V1 module, so the
unchanged behaviour is byte-identical by construction, not by transcription.

The three V2 replacements (task §二/§三/§四; see tier3_event_predicates_v2.py
for the predicate-level contract):
  1. BFS_GRAPH_SOURCE=CURRENT_ENVIRONMENT_STATE_TOPOLOGY — the FRONT walkable
     grid is rebuilt from the CURRENT env state.map[FRONT_FLOOR] every step
     (same BlockType walkable rule + the same LADDER_TILE_TRANSIT OR-in as V1;
     legally-mined tiles are no longer SOLID_BLOCK, hence naturally walkable).
  2. Legal-position domain = the CURRENT grid (never abort on initial-graph
     membership); genuine corruption (out-of-bounds / non-finite / player-vs-
     current-map contradiction / baseline-unreachable) still fails closed.
  3. Target temporarily unreachable in the CURRENT dynamic graph: the episode
     CONTINUES, primary stays false, dense progress freezes (does not
     increase). A legal floor2 -> floor3 transition still gives primary
     success=true (unchanged V1 primary predicate).

The normalization denominator d(start, exit) is computed ONCE at episode start
on the INITIAL graph and stays FIXED (§四: the formula, normalization and
result fields are otherwise untouched). BACK and FULL never consume the
corridor metric and are untouched.
"""
from __future__ import annotations

import os
import sys

# Runnable-as-script AND importable-as-package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tier3_source_audit as audit                 # noqa: E402
import tier3_evaluator as v1                       # noqa: E402  (FROZEN V1)
import tier3_event_predicates_v2 as pred2          # noqa: E402  (V2 predicates)

COMMON_EVALUATOR_PROTOCOL_VERSION = pred2.COMMON_EVALUATOR_PROTOCOL_VERSION
BFS_GRAPH_SOURCE = pred2.BFS_GRAPH_SOURCE
SUPERSEDED_V1_ENGINE_MODULE = "tier3_evaluator.py"

# ---------------------------------------------------------------------------
# Re-export the UNCHANGED V1 surface (single source of truth). The binding
# driver consumes this module as `ev`; every name below is the frozen V1
# object itself. FailClosed / require are the V1 EVALUATOR's own (distinct
# from the predicate FailClosed the driver catches separately) — preserved
# exactly so evaluator-level contract violations still crash the driver
# fail-closed instead of being caught as engine-predicate verdicts.
# ---------------------------------------------------------------------------
FailClosed = v1.FailClosed
require = v1.require
SCHEMA = v1.SCHEMA
RESULT_VERSION = v1.RESULT_VERSION
FULL = v1.FULL
FRONT = v1.FRONT
BACK = v1.BACK
ACTION_MODE = v1.ACTION_MODE
MAX_TIMESTEPS = v1.MAX_TIMESTEPS
REQUIRED_EPISODE_KEYS = v1.REQUIRED_EPISODE_KEYS
FLOOR_ENTRY_ACHIEVEMENTS = v1.FLOOR_ENTRY_ACHIEVEMENTS
validate_episode_record = v1.validate_episode_record
frozen_contract = v1.frozen_contract
evaluate = v1.evaluate
make_canonical_env = v1.make_canonical_env
reset_from_bank_state = v1.reset_from_bank_state
assert_front_reset_equivalence = v1.assert_front_reset_equivalence
state_entry_ids_for = v1.state_entry_ids_for
assert_output_dir_fresh = v1.assert_output_dir_fresh
performance_start_schedule = v1.performance_start_schedule
screening_start_schedule = v1.screening_start_schedule
FULL_SMOKE_SEED_BASE = v1.FULL_SMOKE_SEED_BASE
_jit_step = v1._jit_step
_jit_reset = v1._jit_reset
_eval_device_identity = v1._eval_device_identity
_git_commit_head = v1._git_commit_head
_sha256_lf_file = v1._sha256_lf_file


# ---------------------------------------------------------------------------
# V2 FRONT walkable-grid construction (same rule as V1 _front_walkable_grid,
# parameterized on the CURRENT map floor so it can be rebuilt every step).
# ---------------------------------------------------------------------------
def front_ladder_transit_positions(view):
    """The floor-2 down_ladder (corridor exit) + up_ladder tile positions from
    the normalized start view. Ladders are fixed map features of the frozen
    scaffold (their tiles do not move during an episode); the same
    LADDER_TILE_TRANSIT exception as V1 OR-s them into every (re)built grid so
    graph-distance progress stays well-defined at the exact transit tiles
    regardless of the underlying BlockType."""
    positions = []
    for key in ("down_ladders", "up_ladders"):
        pos = (view.get(key) or {}).get(audit.FRONT_FLOOR)
        if pos is None:
            continue
        positions.append((int(pos[0]), int(pos[1])))
    return positions


def front_walkable_grid_for_map(map_floor, ladder_positions):
    """Evaluator-only walkable mask for the front floor, built from a GIVEN
    map[FRONT_FLOOR] (the current env state's, in V2): cells whose BlockType
    is in the resolved craftax land-creature walkable set (SOLID_BLOCK / WATER
    / LAVA excluded, exactly as game_logic.move_player collides — identical
    rule to V1), with the ladder transit tiles OR-ed in (LADDER_TILE_TRANSIT,
    identical exception to V1). Legally-mined tiles are no longer SOLID_BLOCK
    in the current map, so they are naturally walkable — this is how legal
    topology mutation enters the graph. A ladder position off the map grid
    fails closed (broken state)."""
    import numpy as np
    walk_values = {int(v) for v in audit.resolve_walkable_blocktype_values()}
    m = np.asarray(map_floor)
    rows = len(m)
    cols = len(m[0]) if rows else 0
    grid = [[bool(int(b) in walk_values) for b in row] for row in m]
    for (r, c) in ladder_positions:
        require(0 <= r < rows and 0 <= c < cols,
                "FAIL CLOSED: FRONT ladder transit tile (%d,%d) is off-grid "
                "(%dx%d) — broken state" % (r, c, rows, cols))
        grid[r][c] = True                       # LADDER_TILE_TRANSIT
    return grid


# ---------------------------------------------------------------------------
# V2 rollout — byte-for-byte the V1 rollout EXCEPT the FRONT dense block
# (three marked replacements). BACK / FULL paths, primary event, terminal
# flags, the episode record fields and the running-max aggregation are
# untouched (§四).
# ---------------------------------------------------------------------------
def rollout_episode(entry, start_state, scenario, policy_fn, episode_id, rng_seed,
                    kobold_type_id=None, max_steps=None):
    """Roll out ONE episode under the frozen Tier3 contract (V2 dynamic
    topology). Identical to tier3_evaluator.rollout_episode (V1) except the
    FRONT dense-progress block: the walkable graph is rebuilt from the
    CURRENT env state every step (BFS_GRAPH_SOURCE=
    CURRENT_ENVIRONMENT_STATE_TOPOLOGY), the legal-position domain is the
    current grid, and a temporarily unreachable target freezes progress
    instead of aborting. See module docstring + tier3_event_predicates_v2.
    """
    import tier3_scaffold_builder as builder
    import tier3_state_serializer as ser
    import tier3_event_predicates as pred        # V1 predicates for unchanged checks
    require(ser.have_jax_craftax(),
            "FAIL CLOSED (BLOCKED_ENVIRONMENT): rollout requires JAX+craftax")
    require(scenario in (FULL, FRONT, BACK), "FAIL CLOSED: unknown scenario %r" % scenario)
    import jax
    import numpy as np
    envns, table, ctor = entry["envns"], entry["task_embeddings"], entry["ctor"]
    dk = entry["defeat_kobold_index"]
    steps_cap = MAX_TIMESTEPS if max_steps is None else int(max_steps)
    require(1 <= steps_cap <= MAX_TIMESTEPS,
            "FAIL CLOSED: max_steps %d outside [1, %d]" % (steps_cap, MAX_TIMESTEPS))

    view = builder.normalize_envstate(start_state)
    if scenario == FRONT:
        view["floor2_up_ladder_removed"] = True
        valid_start = pred.valid_front_scaffold_start(view)
        kb = None
    elif scenario == BACK:
        kb = builder.resolve_kobold_type_id(kobold_type_id)
        valid_start = pred.valid_back_scaffold_start(view, kb)
    else:
        valid_start = pred.valid_full_start(view)
        kb = None

    rec = {
        "episode_id": str(episode_id), "scenario": scenario,
        "valid_start": bool(valid_start), "terminal_label": "",
        "front_floor_transition_reached": False, "corridor_exit_reached": False,
        "defeat_kobold": False, "player_died": False, "timed_out": False,
        "kobold_engaged": False, "timesteps": 0, "graph_distance_progress": None,
        "action_sequence": [],
    }
    if not valid_start:
        return rec                                    # INVALID_START; zero steps

    obs, state = reset_from_bank_state(entry, start_state)
    start_level = int(np.asarray(state.player_level))
    start_pos = (int(np.asarray(state.player_position)[0]),
                 int(np.asarray(state.player_position)[1]))
    kb_baseline = {}
    if scenario == BACK:
        for m in view["mobs"]:
            if (m["category"] == pred.KOBOLD_CATEGORY and int(m["type_id"]) == kb
                    and m["mask"] and float(m["health"]) > 0):
                kb_baseline[m["level"]] = max(kb_baseline.get(m["level"], 0.0),
                                              float(m["health"]))
    ladder_tiles = []
    exit_pos = None
    d_start_baseline = None
    if scenario == FRONT:
        # V2 replacement #1 (graph source): the normalization denominator is
        # the baseline d(start, exit) computed ONCE on the INITIAL graph; the
        # per-step grid is rebuilt from the current state below. The frozen
        # bank guarantees a finite baseline; None => bank payload corruption
        # class -> fail closed (NEG18 retained ONLY as a corruption detector).
        ladder_tiles = front_ladder_transit_positions(view)
        walkable_initial = front_walkable_grid_for_map(
            np.asarray(start_state.map)[audit.FRONT_FLOOR], ladder_tiles)
        exit_pos = view["down_ladders"].get(audit.FRONT_FLOOR)
        if exit_pos is not None:
            d_start_baseline = pred2.bfs_distance(walkable_initial, start_pos,
                                                  exit_pos)
            if d_start_baseline is None:
                raise pred2.FailClosed(
                    "V2_BASELINE_UNREACHABLE: frozen FRONT bank initial graph "
                    "has no start -> exit path — bank payload corruption class "
                    "(NEG18 corruption detector; legal play never produces this)")

    step_fn = _jit_step(entry)                       # 总控 §二: rollout execution path
    actions = []
    rng = jax.random.PRNGKey(int(rng_seed))
    max_level, max_progress, steps = start_level, 0.0, 0
    defeated = died = alias_seen = engaged = env_done = False
    for _ in range(steps_cap):
        action = int(policy_fn(obs, state))
        actions.append(action)
        rng, sk = jax.random.split(rng)
        obs, state, _rew, done, _info = step_fn(sk, state, action)
        steps += 1
        lvl = int(np.asarray(state.player_level))
        max_level = max(max_level, lvl)
        if scenario == FRONT and lvl >= audit.CORRIDOR_EXIT_FLOOR:
            alias_seen = True
        if bool(np.asarray(state.achievements)[dk]):
            defeated = True
        if float(np.asarray(state.player_health)) <= 0.0:
            died = True
        if scenario == BACK and not engaged:
            rb = state.ranged_mobs
            fl = lvl
            h = np.asarray(rb.health)[fl]; msk = np.asarray(rb.mask)[fl]
            tid = np.asarray(rb.type_id)[fl]; cd = np.asarray(rb.attack_cooldown)[fl]
            base = kb_baseline.get(fl)
            for slot in range(int(h.shape[0])):
                if msk[slot] and int(tid[slot]) == kb:
                    if base is not None and float(h[slot]) < base:
                        engaged = True
                    if base is not None and float(h[slot]) <= 0.0:
                        engaged = True
                    if int(cd[slot]) > 0:
                        engaged = True
        if (scenario == FRONT and exit_pos is not None
                and lvl == audit.FRONT_FLOOR and max_level < audit.CORRIDOR_EXIT_FLOOR):
            # V2_DYNAMIC_TOPOLOGY dense block (replaces the V1 static block).
            # Computed ONLY while the player is on the front floor and before
            # the exit floor is reached (unchanged V1 window: a floor-1
            # excursion carries floor-1 coordinates meaningless on the floor-2
            # grid; the metric freezes permanently at the transition). The
            # walkable graph is rebuilt from the CURRENT env state each step;
            # legal-position membership is the current grid (a legally-mined
            # tile is valid); a temporarily unreachable target returns the
            # previous progress unchanged (conservative freeze — the episode
            # is NOT aborted, primary stays false). Genuine corruption
            # (out-of-bounds / non-finite / player-vs-current-map
            # contradiction) still raises pred2.FailClosed and propagates.
            # The running-max aggregation is unchanged (§四).
            pos = (int(np.asarray(state.player_position)[0]),
                   int(np.asarray(state.player_position)[1]))
            walkable_cur = front_walkable_grid_for_map(
                np.asarray(state.map)[audit.FRONT_FLOOR], ladder_tiles)
            p = pred2.normalized_corridor_progress_dynamic(
                {"player_position": pos}, walkable_cur, start_pos, exit_pos,
                d_start_baseline, max_progress)
            max_progress = max(max_progress, p)
        env_done = bool(np.asarray(done))
        if defeated or died or env_done:
            break
    timed_out = not (defeated or died) and (steps >= steps_cap or env_done)
    transition = (scenario == FRONT
                  and pred.front_floor_transition_reached(start_level, max_level))
    rec.update({
        "front_floor_transition_reached": bool(transition),
        "corridor_exit_reached": bool(alias_seen),
        "defeat_kobold": bool(defeated),
        "player_died": bool(died),
        "timed_out": bool(timed_out),
        "kobold_engaged": bool(engaged),
        "timesteps": steps,
        "graph_distance_progress": float(max_progress) if scenario == FRONT else None,
        "action_sequence": list(actions),
    })
    return rec


# ---------------------------------------------------------------------------
# CLI: the V1 run_evaluation / certificate CLI path (CC2 screening territory)
# is UNCHANGED — it delegates to the frozen V1 module verbatim. The V2 metric
# fix is consumed through THIS module's rollout_episode by the V2 binding
# driver (tier3_projection_binding_smoke_v2.py) and the V2 common assembly.
# ---------------------------------------------------------------------------
def main(argv=None):
    return v1.main(argv)


if __name__ == "__main__":
    sys.exit(main())
