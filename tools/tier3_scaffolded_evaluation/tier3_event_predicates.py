#!/usr/bin/env python3
"""CC4 Tier3 — boundary event predicates (pure Python, JAX-free testable).

Every predicate is a pure function of a NORMALIZED state view plus frozen,
source-grounded constants. No predicate invents a field: each reads only fields
that exist in the audited MiniCraftax EnvState (see tier3_source_audit.py).
Achievement tests resolve the achievement by NAME (symbolic) — never by a
hard-coded integer — so the exact craftax enum index is bound at runtime on a
craftax==1.4.5 host (BLOCKED_ENVIRONMENT here). The Kobold mob type_id is passed
in (resolved from craftax constants at runtime), never assumed.

The NORMALIZED state view is a plain dict (documented below) so the whole
predicate / progress / taxonomy layer is exercisable with synthetic states and
no JAX. A JAX-side adapter (in the builder/evaluator) maps a real flax EnvState
pytree into this view; that adapter is the only JAX-touching part and is
BLOCKED_ENVIRONMENT on this host.

NORMALIZED state view (keys):
  _normalized: True
  player_level: int                         (== floor)
  player_health: float
  player_position: (row, col)
  timestep: int
  achieved: set[str]                        (achievement NAMES achieved)
  mobs: list[ {category, level, position:(r,c), health, mask:bool, type_id, attack_cooldown} ]
  monsters_killed: {level:int -> count:int}
  down_ladders: {level:int -> (r,c)}
  up_ladders:   {level:int -> (r,c)}
  inventory: dict
  boss_progress: int
  floor2_up_ladder_removed: bool
  map: {level:int -> 2D list[int]}          (optional; required for progress)
  item_map: {level:int -> 2D list[int]}     (optional)
"""
from __future__ import annotations

import os
import sys
from collections import deque

# Runnable-as-script AND importable-as-package: make sibling modules importable
# regardless of invocation mode.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tier3_source_audit as audit  # noqa: E402

PREDICATE_VERSION = "tier3_predicates/v1"

# Frozen V1 boundary floors (source-grounded in the canonical S4 task 45fdd17c).
FRONT_FLOOR = audit.FRONT_FLOOR                 # 2
BACK_FLOOR = audit.BACK_FLOOR                   # 3
CORRIDOR_EXIT_FLOOR = audit.CORRIDOR_EXIT_FLOOR  # 3

DEFEAT_KOBOLD = "DEFEAT_KOBOLD"

# Canonical starting kit / gate (from the task's generate_world). Used to check
# that a FULL start is well-formed; a scaffold may standardize these legally.
CANONICAL_START_INVENTORY = dict(audit.CANONICAL_TASK_FACTS["starting_inventory"])
CANONICAL_MONSTERS_KILLED_FLOOR2 = audit.CANONICAL_TASK_FACTS["monsters_killed"]["2"]


class FailClosed(Exception):
    """Hard stop for invalid states / unreachable scaffolds."""


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def is_normalized(state: dict) -> bool:
    return isinstance(state, dict) and state.get("_normalized") is True


def require_fields(state: dict, fields):
    missing = [f for f in fields if f not in state]
    if missing:
        raise FailClosed(f"normalized state missing required field(s): {sorted(missing)}")


def normalize(state: dict) -> dict:
    """Accept an already-normalized dict (synthetic) or pass through.

    Real EnvState -> normalized conversion lives in the JAX-side adapter; this
    function only validates the normalized contract so predicates stay pure.
    """
    if not is_normalized(state):
        raise FailClosed("state is not a normalized Tier3 view (adapter required for real EnvState)")
    require_fields(state, ["player_level", "player_health", "player_position", "timestep", "achieved"])
    state.setdefault("mobs", [])
    state.setdefault("monsters_killed", {})
    state.setdefault("down_ladders", {})
    state.setdefault("up_ladders", {})
    state.setdefault("inventory", {})
    state.setdefault("boss_progress", 0)
    state.setdefault("floor2_up_ladder_removed", False)
    return state


def _achieved(state: dict) -> set:
    return set(state.get("achieved", set()))


# ---------------------------------------------------------------------------
# Mob helpers (Kobold type_id is injected, never hard-coded)
# ---------------------------------------------------------------------------
# RESOLVED binding (craftax==1.4.5; see tier3_source_audit.resolve_kobold_binding):
# the Kobold is a RANGED-category mob (ranged type_id 3; MOB_ACHIEVEMENT_MAP
# [MobType.RANGED, 3] == DEFEAT_KOBOLD). The earlier "melee" assumption was WRONG —
# MeleeMobType does not even exist in craftax 1.4.5 constants. Predicates still take
# the type_id as a parameter; only the category is frozen here.
KOBOLD_CATEGORY = "ranged"


def live_mobs_on_floor(state: dict, floor: int):
    return [m for m in state.get("mobs", [])
            if m.get("mask") and m.get("level") == floor and float(m.get("health", 0)) > 0]


def kobold_present(state: dict, kobold_type_id: int, floor: int | None = None) -> bool:
    """True if a live Kobold (RANGED-category mob with the given type_id) exists.

    floor=None => any floor (presence); else restrict to that floor.
    """
    for m in state.get("mobs", []):
        if not m.get("mask") or float(m.get("health", 0)) <= 0:
            continue
        if int(m.get("type_id", -1)) != int(kobold_type_id):
            continue
        if m.get("category") != KOBOLD_CATEGORY:
            continue
        if floor is not None and m.get("level") != floor:
            continue
        return True
    return False


def kobold_max_health(state: dict, kobold_type_id: int, floor: int) -> float:
    hs = [float(m.get("health", 0)) for m in state.get("mobs", [])
          if m.get("mask") and int(m.get("type_id", -1)) == int(kobold_type_id)
          and m.get("level") == floor and m.get("category") == KOBOLD_CATEGORY]
    return max(hs) if hs else 0.0


# ---------------------------------------------------------------------------
# Validity predicates
# ---------------------------------------------------------------------------
def player_alive(state: dict) -> bool:
    return float(state.get("player_health", 0)) > 0


def valid_env_state(state: dict) -> bool:
    """Structural well-formedness shared by every scenario start."""
    try:
        normalize(state)
    except FailClosed:
        return False
    if not player_alive(state):
        return False
    if int(state.get("timestep", -1)) != 0:
        return False
    return True


def valid_full_start(state: dict) -> bool:
    """A well-formed canonical Stage4 DEFEAT_KOBOLD initial state (floor 2)."""
    if not valid_env_state(state):
        return False
    if int(state["player_level"]) != FRONT_FLOOR:
        return False
    if DEFEAT_KOBOLD in _achieved(state):
        return False  # already solved -> not a valid START
    return True


def front_floor_transition_reached(from_level: int, to_level: int,
                                   start_floor: int = FRONT_FLOOR,
                                   exit_floor: int = CORRIDOR_EXIT_FLOOR) -> bool:
    """Frozen V1 FRONT_L2 PRIMARY event (episode-level, not per-state):
    FRONT_FLOOR_TRANSITION_REACHED — the player level moved from the start floor
    (2) to the exit floor (3) at some point during the episode.

    Pure helper over two observed levels (the evaluator passes the recorded
    start level and the final/peak level of the episode). success == player
    level went 2 -> 3, i.e. `from_level <= start_floor and to_level >=
    exit_floor` when the recorded start is exactly the scaffold start. Kept as
    a pure function of the recorded levels so it stays JAX-free and unit
    testable; the evaluator owns the episode history that supplies the levels.
    """
    return int(from_level) == int(start_floor) and int(to_level) >= int(exit_floor)


def corridor_exit_reached(state: dict) -> bool:
    """Per-state predicate: player_level >= CORRIDOR_EXIT_FLOOR (3).

    Source basis: player_level is the floor; the kobold is on floor 3, reached
    by descending. REJECTED_ALTERNATIVE: 'standing on the floor-2 down_ladder
    tile' (more granular but the actual game event is the floor transition).

    收口 (fast-track): the FRONT_L2 PRIMARY event is the episode-level
    `front_floor_transition_reached` (player level 2 -> 3). This per-state
    predicate is retained ONLY as PENDING_EQUIVALENCE_ALIAS — it is reported as
    `corridor_exit_reached` but must NOT be used as the primary success metric
    until real-map evidence proves the floor transition necessarily passes
    through the target corridor. An episode with transition=True but this alias
    False at every recorded state is a contradiction -> FailClosed (taxonomy).
    """
    return int(state["player_level"]) >= CORRIDOR_EXIT_FLOOR


def valid_front_scaffold_start(state: dict) -> bool:
    """A valid FRONT_L2 diagnostic start: on the dark corridor floor, alive,
    not already past the exit, not already solved.

    NEG14: a front start already beyond the exit (player_level >=
    CORRIDOR_EXIT_FLOOR) is INVALID.
    """
    if not valid_env_state(state):
        return False
    if int(state["player_level"]) != FRONT_FLOOR:
        return False
    if corridor_exit_reached(state):
        return False  # NEG14
    if DEFEAT_KOBOLD in _achieved(state):
        return False
    return True


def front_half_entered(state: dict) -> bool:
    return int(state["player_level"]) == FRONT_FLOOR


def valid_back_scaffold_start(state: dict, kobold_type_id: int) -> bool:
    """A valid BACK_L2 diagnostic start: on the kobold floor, alive, a LIVE
    kobold present, and DEFEAT_KOBOLD not already achieved.

    NEG15: back start already has DEFEAT_KOBOLD -> INVALID.
    NEG16: back state has no live Kobold (kill task requires one) -> INVALID.
    """
    if not valid_env_state(state):
        return False
    if int(state["player_level"]) != BACK_FLOOR:
        return False
    if DEFEAT_KOBOLD in _achieved(state):
        return False  # NEG15
    if not kobold_present(state, kobold_type_id, floor=BACK_FLOOR):
        return False  # NEG16
    return True


def back_half_entered(state: dict) -> bool:
    return int(state["player_level"]) == BACK_FLOOR


def boss_area_reached(state: dict) -> bool:
    """Per-state predicate (kept for vocabulary completeness ONLY): player on
    the kobold/target floor (player_level == BACK_FLOOR).

    Source basis: the kobold (target) is on floor 3 (BACK_FLOOR). REJECTED
    ALTERNATIVE: boss_progress > 0 — REJECTED because boss_progress drives the
    Necromancer boss-spawn mechanic, which is a different system from the
    DEFEAT_KOBOLD target (audit: CANONICAL_TASK_FACTS / achievements list).

    收口 (fast-track): BACK_L2 identity is BOSS_COMBAT_SCAFFOLDED — it evaluates
    combat against a live Kobold on floor 3 (primary metric
    P_DEFEAT_KOBOLD_GIVEN_VALID_BACK_START). boss_area_reached,
    time_to_boss_area and BACK_BOSS_NOT_FOUND are N/A for BACK_L2: the start is
    ALREADY on floor 3 next to the Kobold, so "search for the boss area" is not
    something this scaffold measures, and the scaffold must not claim to
    evaluate it.
    """
    return int(state["player_level"]) == BACK_FLOOR


def defeat_kobold(state: dict) -> bool:
    """DEFEAT_KOBOLD achievement set (symbolic; index bound on a craftax host)."""
    return DEFEAT_KOBOLD in _achieved(state)


def kobold_engaged(state: dict, kobold_type_id: int) -> bool:
    """Frozen V1 engagement marker: a kobold on the player's floor has taken
    damage (health below the floor's kobold max) OR an attack cooldown is active.

    Documented as a per-state proxy; precise combat-contact semantics are
    refined via episode history in the evaluator (damage_dealt / damage_received
    metrics). failure_rule_version is recorded on every classification.
    """
    floor = int(state["player_level"])
    engaged = False
    for m in state.get("mobs", []):
        if not m.get("mask") or m.get("level") != floor or m.get("category") != KOBOLD_CATEGORY:
            continue
        if int(m.get("type_id", -1)) != int(kobold_type_id):
            continue
        if float(m.get("health", 0)) <= 0:
            continue
        if int(m.get("attack_cooldown", 0)) > 0:
            engaged = True
    # damage-taken proxy: any kobold below its starting max health on this floor
    # is detected by the evaluator comparing to the scaffold's recorded max.
    return engaged


# ---------------------------------------------------------------------------
# Dense progress: GRAPH_DISTANCE over evaluator-only traversability (BFS).
# ---------------------------------------------------------------------------
def bfs_distance(walkable, start, goal):
    """4-connectivity shortest path length on a 2D bool walkable grid.

    Returns None if goal is unreachable from start (or either is off-grid /
    non-walkable). Pure Python; evaluator-only — the grid is never exposed to
    the Student.
    """
    rows = len(walkable)
    cols = len(walkable[0]) if rows else 0

    def ok(rc):
        r, c = rc
        return 0 <= r < rows and 0 <= c < cols and walkable[r][c]

    if not ok(start) or not ok(goal):
        return None
    if tuple(start) == tuple(goal):
        return 0
    seen = {tuple(start)}
    q = deque([(tuple(start), 0)])
    while q:
        (r, c), d = q.popleft()
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if (nr, nc) in seen or not ok((nr, nc)):
                continue
            if (nr, nc) == tuple(goal):
                return d + 1
            seen.add((nr, nc))
            q.append(((nr, nc), d + 1))
    return None


def exit_reachable_from_start(walkable, start, exit_pos) -> bool:
    return bfs_distance(walkable, start, exit_pos) is not None


def normalized_corridor_progress(state: dict, walkable, start_pos, exit_pos):
    """progress = clip(1 - d(current, exit) / max(d(start, exit), 1), 0, 1).

    Policies (frozen V1, per spec §十):
      - unreachable_state_policy: if the START cannot reach the exit, the world
        is an invalid front scaffold -> FailClosed (NEG18 unless an explicit
        blocked label is supplied by the caller).
      - invalid_position_policy: if the current position is off-grid/non-walkable
        -> FailClosed.
      - transient dead-end: if current cannot reach exit but start could, the
        player is lost -> progress 0.0 (monotonicity NOT guaranteed).
    """
    d_start = bfs_distance(walkable, start_pos, exit_pos)
    if d_start is None:
        raise FailClosed("NEG18: corridor exit unreachable from scaffold start (no blocked label)")
    cur = tuple(state["player_position"])
    rows = len(walkable)
    cols = len(walkable[0]) if rows else 0
    if not (0 <= cur[0] < rows and 0 <= cur[1] < cols):
        raise FailClosed("invalid_position_policy: player position off-grid")
    if not walkable[cur[0]][cur[1]]:
        raise FailClosed("invalid_position_policy: player position non-walkable")
    d_t = bfs_distance(walkable, cur, exit_pos)
    if d_t is None:
        return 0.0  # lost / transiently unreachable; monotonicity not guaranteed
    progress = 1.0 - (d_t / max(d_start, 1))
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
def _mk_state(**over):
    s = {
        "_normalized": True,
        "player_level": FRONT_FLOOR,
        "player_health": 9.0,
        "player_position": (5, 5),
        "timestep": 0,
        "achieved": set(),
        "mobs": [],
        "monsters_killed": {FRONT_FLOOR: 8},
        "down_ladders": {FRONT_FLOOR: (9, 9)},
        "up_ladders": {FRONT_FLOOR: (1, 1)},
        "inventory": dict(CANONICAL_START_INVENTORY),
        "boss_progress": 0,
        "floor2_up_ladder_removed": True,
    }
    s.update(over)
    return s


def _kobold(floor=BACK_FLOOR, health=8.0, type_id=3):
    return {"category": KOBOLD_CATEGORY, "level": floor, "position": (8, 8),
            "health": health, "mask": True, "type_id": type_id, "attack_cooldown": 0}


def self_test() -> int:
    problems = []
    KOBOLD = 3  # craftax==1.4.5 resolved binding: Kobold = RANGED type_id 3 (see tier3_source_audit)

    def check(name, cond):
        if not cond:
            problems.append(name)

    # validity
    check("valid_full_start", valid_full_start(_mk_state()))
    check("valid_full_start_rejects_solved", not valid_full_start(_mk_state(achieved={DEFEAT_KOBOLD})))
    check("valid_full_start_rejects_dead", not valid_full_start(_mk_state(player_health=0.0)))
    check("valid_full_start_rejects_nonzero_timestep", not valid_full_start(_mk_state(timestep=3)))
    check("valid_front_start", valid_front_scaffold_start(_mk_state()))
    check("front_start_rejects_beyond_exit(NEG14)",
          not valid_front_scaffold_start(_mk_state(player_level=CORRIDOR_EXIT_FLOOR)))
    back_ok = _mk_state(player_level=BACK_FLOOR, mobs=[_kobold(type_id=KOBOLD)])
    check("valid_back_start", valid_back_scaffold_start(back_ok, KOBOLD))
    check("back_start_rejects_solved(NEG15)",
          not valid_back_scaffold_start(_mk_state(player_level=BACK_FLOOR, achieved={DEFEAT_KOBOLD},
                                                  mobs=[_kobold(type_id=KOBOLD)]), KOBOLD))
    check("back_start_rejects_no_kobold(NEG16)",
          not valid_back_scaffold_start(_mk_state(player_level=BACK_FLOOR, mobs=[]), KOBOLD))

    # events
    check("corridor_exit_reached_floor3", corridor_exit_reached(_mk_state(player_level=3)))
    check("corridor_not_reached_floor2", not corridor_exit_reached(_mk_state(player_level=2)))
    check("front_half_entered", front_half_entered(_mk_state(player_level=2)))
    check("back_half_entered", back_half_entered(_mk_state(player_level=3)))
    check("boss_area_reached", boss_area_reached(_mk_state(player_level=3)))
    check("boss_area_not_floor2", not boss_area_reached(_mk_state(player_level=2)))
    check("defeat_kobold", defeat_kobold(_mk_state(achieved={DEFEAT_KOBOLD})))
    check("kobold_present", kobold_present(back_ok, KOBOLD, floor=BACK_FLOOR))
    check("kobold_absent_wrong_type", not kobold_present(back_ok, KOBOLD + 1, floor=BACK_FLOOR))
    # Kobold is RANGED (craftax 1.4.5): a same-type_id mob in a DIFFERENT category
    # must NOT count as a Kobold.
    melee_imp = _kobold(type_id=KOBOLD)
    melee_imp["category"] = "melee"
    check("kobold_category_is_ranged",
          not kobold_present(_mk_state(player_level=BACK_FLOOR, mobs=[melee_imp]),
                             KOBOLD, floor=BACK_FLOOR))
    # FRONT_FLOOR_TRANSITION_REACHED — episode-level primary event (player level 2 -> 3)
    check("front_transition_2to3", front_floor_transition_reached(2, 3))
    check("front_transition_2to4", front_floor_transition_reached(2, 4))
    check("front_transition_not_from3", not front_floor_transition_reached(3, 3))
    check("front_transition_not_stay2", not front_floor_transition_reached(2, 2))

    # progress (GRAPH_DISTANCE) on a small grid
    # 5x5 grid, walkable row 2 fully open; start (2,0), exit (2,4) => d_start=4
    walk = [[False] * 5 for _ in range(5)]
    for c in range(5):
        walk[2][c] = True
    start = (2, 0)
    exit_pos = (2, 4)
    prog_start = normalized_corridor_progress(_mk_state(player_position=start), walk, start, exit_pos)
    prog_mid = normalized_corridor_progress(_mk_state(player_position=(2, 2)), walk, start, exit_pos)
    prog_exit = normalized_corridor_progress(_mk_state(player_position=exit_pos), walk, start, exit_pos)
    check("progress_at_start==0", abs(prog_start - 0.0) < 1e-9)
    check("progress_at_mid==0.5", abs(prog_mid - 0.5) < 1e-9)
    check("progress_at_exit==1", abs(prog_exit - 1.0) < 1e-9)
    # NEG17 guard
    check("progress_in_range", all(0.0 <= p <= 1.0 for p in (prog_start, prog_mid, prog_exit)))
    # NEG18: unreachable exit -> FailClosed
    walk_blocked = [[False] * 5 for _ in range(5)]
    walk_blocked[2][0] = True
    walk_blocked[2][1] = True  # exit (2,4) isolated
    try:
        normalized_corridor_progress(_mk_state(player_position=(2, 0)), walk_blocked, (2, 0), (2, 4))
        check("NEG18_unreachable_raises", False)
    except FailClosed:
        check("NEG18_unreachable_raises", True)
    # invalid position -> FailClosed
    try:
        normalized_corridor_progress(_mk_state(player_position=(0, 0)), walk, start, exit_pos)
        check("invalid_position_raises", False)
    except FailClosed:
        check("invalid_position_raises", True)

    if problems:
        print("TIER3_PREDICATES_SELF_TEST_FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print("TIER3_PREDICATES_SELF_TEST_PASS (checks=29)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(self_test())
