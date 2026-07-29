#!/usr/bin/env python3
"""CC4 Tier3 — scaffold builder (the ONLY legal scaffold mechanism = WorldBuilder).

A scaffold spec is a PURE, declarative description of how a diagnostic start state
is produced from the canonical Stage4 DEFEAT_KOBOLD task facts (audited in
tier3_source_audit.py) using the repository's OWN WorldBuilder API — the same
mechanism the repo's combat seed task uses under the comment "ADDED SCAFFOLDING".
No scaffold invents fields, adds observation channels, or injects privileged
information; every legality flag is declared and machine-checked here, and the
negative tests (tier3_negative_tests.py) actively try to violate each one.

Two diagnostic scaffolds (frozen V1, 收口 fast-track):
  * FRONT_L2: start on the floor-2 dark corridor (canonical entry). Removes the
    upstream resource prep / dungeon entry (floors 0-1). Keeps corridor navigation,
    multi-mob survival, and the floor transition. PRIMARY event =
    FRONT_FLOOR_TRANSITION_REACHED (player level 2 -> 3); the per-state
    corridor_exit_reached predicate is PENDING_EQUIVALENCE_ALIAS only.
  * BACK_L2: identity = BOSS_COMBAT_SCAFFOLDED. Start on floor 3 next to a LIVE
    Kobold (RANGED category, ranged type_id 3, canonical HP 8.0), DEFEAT_KOBOLD
    false at t0. Removes the floor-2 corridor bottleneck. Keeps engagement,
    combat, survival, DEFEAT_KOBOLD. boss-area SEARCH is out of scope:
    boss_area_reached / time_to_boss_area / BACK_BOSS_NOT_FOUND are N/A.

MATERIALIZATION (actually calling WorldBuilder to mint EnvState pytrees) requires a
JAX + craftax==1.4.5 host and is GUARDED: without JAX+craftax ``materialize_start``
FAILS CLOSED with BLOCKED_ENVIRONMENT and never emits a state. The FRONT
materialization is byte-identical to the canonical S4 generate_world under the same
world rng (verified: all pytree leaves + observation equal). ``normalize_envstate``
maps a real EnvState pytree into the JAX-free normalized predicate view.
"""
from __future__ import annotations

import os
import sys

# Runnable-as-script AND importable-as-package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tier3_source_audit as audit        # noqa: E402
import tier3_event_predicates as pred     # noqa: E402
import tier3_state_serializer as ser      # noqa: E402

SCHEMA = "mechanism_UED.tier3_scaffold_spec/v1"
SPEC_VERSION = "tier3_scaffold_spec/v1"

FRONT = "front_l2"
BACK = "back_l2"

# Legality invariants every scaffold MUST satisfy (frozen; mirrored in the configs
# and enforced by the negative tests). Any flag False -> fail closed.
SCAFFOLD_LEGALITY_FLAGS = [
    "no_privileged_information",
    "no_extra_observation_channel",
    "no_hidden_boss_direction",
    "no_shortest_path_hint",
    "no_future_monster_action",
    "no_arm_specific_state",
    "same_action_space",
    "same_observation_schema",
    "common_state_bank_for_all_arms",
]

# Keys that MUST NEVER appear anywhere in a scaffold spec / state-bank manifest:
# a scaffold is blind to arms, checkpoints, params and results (NEG08 / NEG26).
FORBIDDEN_RESULT_BLINDNESS_KEYS = {
    "arm", "arm_id", "arm_name", "checkpoint", "checkpoint_id", "checkpoint_sha",
    "params_sha", "params", "result", "results", "reward", "return", "returns",
    "student", "student_id", "selection", "selected_by", "performance", "score",
    "base", "replay", "persistent", "reset128", "d052",
}


class FailClosed(Exception):
    """Hard stop on any scaffold-legality / source-identity violation."""


def require(cond, msg):
    if not cond:
        raise FailClosed(msg)


# ---------------------------------------------------------------------------
# Canonical facts (from the audited task; never invented)
# ---------------------------------------------------------------------------
def canonical_starting_inventory() -> dict:
    return dict(audit.CANONICAL_TASK_FACTS["starting_inventory"])


def validate_inventory(inventory: dict):
    """NEG12: every inventory value must be a non-negative int with a str key."""
    require(isinstance(inventory, dict),
            "FAIL CLOSED (NEG12): inventory is not a dict")
    for k, v in inventory.items():
        require(isinstance(k, str) and k,
                "FAIL CLOSED (NEG12): inventory key %r is not a non-empty str" % (k,))
        require(isinstance(v, int) and not isinstance(v, bool) and v >= 0,
                "FAIL CLOSED (NEG12): inventory[%r]=%r is not a non-negative int" % (k, v))
    return True


def validate_player_position(position, grid_rows=None, grid_cols=None):
    """NEG13: position must be an (row, col) pair of ints >= 0; if a grid size is
    supplied the position must lie inside it."""
    require(isinstance(position, (tuple, list)) and len(position) == 2,
            "FAIL CLOSED (NEG13): player_position %r is not an (row,col) pair" % (position,))
    r, c = position
    require(isinstance(r, int) and isinstance(c, int) and not isinstance(r, bool)
            and not isinstance(c, bool) and r >= 0 and c >= 0,
            "FAIL CLOSED (NEG13): player_position %r has negative/non-int coordinates" % (position,))
    if grid_rows is not None and grid_cols is not None:
        require(r < grid_rows and c < grid_cols,
                "FAIL CLOSED (NEG13): player_position %r outside grid %dx%d"
                % (position, grid_rows, grid_cols))
    return True


# ---------------------------------------------------------------------------
# Spec construction (pure, declarative)
# ---------------------------------------------------------------------------
def _base_legality() -> dict:
    return {k: True for k in SCAFFOLD_LEGALITY_FLAGS}


def build_front_spec() -> dict:
    f = audit.CANONICAL_TASK_FACTS
    return {
        "schema": SCHEMA,
        "spec_version": SPEC_VERSION,
        "scenario": FRONT,
        "identity_class": "TIER3_FRONT_DIAGNOSTIC_SCAFFOLD",
        "purpose": "MECHANISM_DIAGNOSIS_ONLY",
        "primary_event": "FRONT_FLOOR_TRANSITION_REACHED (player level %d -> %d)"
                         % (audit.FRONT_FLOOR, audit.CORRIDOR_EXIT_FLOOR),
        "primary_metric": "P_FRONT_FLOOR_TRANSITION_REACHED_GIVEN_VALID_START",
        "dense_metric": "GRAPH_DISTANCE_PROGRESS",
        "boundary_predicates": {
            "start": "valid_front_scaffold_start",
            "primary_event": "front_floor_transition_reached (episode-level: player level 2 -> 3)",
            "exit_alias": "corridor_exit_reached (PENDING_EQUIVALENCE_ALIAS: reported but NOT the "
                          "primary metric until real-map evidence proves the floor transition "
                          "necessarily passes through the target corridor)",
        },
        "start_floor": audit.FRONT_FLOOR,
        "exit_floor": audit.CORRIDOR_EXIT_FLOOR,
        "builder": {
            "mechanism": "WorldBuilder (minicraftax.world_builder; repo-native scaffolding)",
            "world_builder_sha256": audit.SOURCE_FILES["world_builder"]["sha256"],
            "calls": [
                "r, _r = jax.random.split(world_rng); WorldBuilder(_r, static_params, params)  # canonical rng sequence",
                "set_starting_floor(%d)" % audit.FRONT_FLOOR,
                "set_monsters_killed(%d, %d)" % (audit.FRONT_FLOOR,
                                                 f["monsters_killed"]["2"]),
                "set_player_inventory(<canonical starting kit>)",
                "state = build(r)",
                "state.replace(item_map=state.item_map.at[%d, up_ladder].set(ItemType.NONE.value))  # canonical floor-2 up-ladder removal" % audit.FRONT_FLOOR,
            ],
            "starting_inventory": canonical_starting_inventory(),
            "monsters_killed": dict(f["monsters_killed"]),
            "floor2_up_ladder_removed": f["floor2_up_ladder_removed"],
        },
        "removes": "upstream resource preparation and dungeon entry (floors 0-1)",
        "keeps": "floor-2 DARK corridor navigation, multi-mob survival, floor transition to floor 3",
        "legality": _base_legality(),
        "observation_schema": "canonical_craftax_symbolic (UNCHANGED)",
        "action_space": "canonical_craftax_action_set (UNCHANGED)",
        "scaffolded_results_can_replace_full_task": False,
    }


def build_back_spec() -> dict:
    f = audit.CANONICAL_TASK_FACTS
    kb = audit.CRAFTAX_RUNTIME_BINDINGS["kobold"]
    return {
        "schema": SCHEMA,
        "spec_version": SPEC_VERSION,
        "scenario": BACK,
        "identity_class": "BOSS_COMBAT_SCAFFOLDED",
        "purpose": "MECHANISM_DIAGNOSIS_ONLY",
        "primary_metric": "P_DEFEAT_KOBOLD_GIVEN_VALID_BACK_START",
        "dense_metric": "none",
        "boundary_predicates": {
            "start": "valid_back_scaffold_start",
            "engaged": "kobold_engaged",
            "defeat": "defeat_kobold",
            "boss_area": "boss_area_reached (N/A for BACK_L2 — vocabulary only; see na_metrics)",
        },
        "na_metrics": ["boss_area_reached", "time_to_boss_area", "BACK_BOSS_NOT_FOUND"],
        "na_reason": "the BACK start is ALREADY on floor 3 next to a live Kobold; boss-area "
                     "search is not exercised, so BACK_L2 must not claim to evaluate it",
        "start_floor": audit.BACK_FLOOR,
        "boss_floor": audit.BACK_FLOOR,
        "require_live_kobold_at_start": True,
        "forbid_defeat_kobold_at_start": True,
        "kobold_type_id_binding": "RESOLVED (craftax==1.4.5): RANGED category, ranged type_id %d, "
                                  "canonical max health %s (MOB_ACHIEVEMENT_MAP[MobType.RANGED, %d] == "
                                  "DEFEAT_KOBOLD; MOB_TYPE_HEALTH_MAPPING[%d, MobType.RANGED])"
                                  % (kb["type_id"],
                                     audit.CRAFTAX_RUNTIME_BINDINGS["kobold_canonical_max_health"],
                                     kb["type_id"], kb["type_id"]),
        "builder": {
            "mechanism": "WorldBuilder (minicraftax.world_builder; repo-native scaffolding)",
            "world_builder_sha256": audit.SOURCE_FILES["world_builder"]["sha256"],
            "calls": [
                "r, _r = jax.random.split(world_rng); WorldBuilder(_r, static_params, params)  # canonical rng sequence",
                "set_starting_floor(%d)" % audit.BACK_FLOOR,
                "set_player_inventory(<canonical starting kit>)",
                "mob_rng = jax.random.fold_in(r, 0xBAC)",
                "add_mobs_randomly_near(mob_rng, %d, 'ranged', KOBOLD_TYPE_ID, 1, "
                "target_pos=player_position, min_dist=2, max_dist=5)  # LIVE Kobold near the player" % audit.BACK_FLOOR,
                "state = build(r)",
            ],
            "starting_inventory": canonical_starting_inventory(),
        },
        "removes": "the floor-2 dark corridor bottleneck and boss-area search (start is already on floor 3)",
        "keeps": "kobold engagement, combat, survival, DEFEAT_KOBOLD",
        "legality": _base_legality(),
        "observation_schema": "canonical_craftax_symbolic (UNCHANGED)",
        "action_space": "canonical_craftax_action_set (UNCHANGED)",
        "scaffolded_results_can_replace_full_task": False,
    }


def build_spec(scenario: str) -> dict:
    if scenario == FRONT:
        return build_front_spec()
    if scenario == BACK:
        return build_back_spec()
    raise FailClosed("FAIL CLOSED: unknown scaffold scenario %r (allowed: %s, %s)"
                     % (scenario, FRONT, BACK))


# ---------------------------------------------------------------------------
# Legality validation (NEG08 / NEG09 / NEG10 / NEG11 / NEG26)
# ---------------------------------------------------------------------------
def _walk_keys(obj, acc):
    if isinstance(obj, dict):
        for k, v in obj.items():
            acc.add(str(k).lower())
            _walk_keys(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _walk_keys(v, acc)


def assert_no_arm_specific_metadata(spec: dict):
    """NEG08 / NEG26: a scaffold spec must be blind to arms/checkpoints/results.

    No key anywhere in the spec may name an arm, a checkpoint, params, or a result,
    and there must be no per-arm override block. A scaffold is identical for every
    arm (Base/Replay/Persistent/Reset128/future D052).
    """
    keys = set()
    _walk_keys(spec, keys)
    bad = sorted(keys & FORBIDDEN_RESULT_BLINDNESS_KEYS)
    require(not bad,
            "FAIL CLOSED (NEG08/NEG26): scaffold spec contains arm/checkpoint/result-specific "
            "key(s): %s. A scaffold must be identical for every arm and selected WITHOUT "
            "reference to Student performance." % bad)
    require(spec.get("legality", {}).get("no_arm_specific_state") is True,
            "FAIL CLOSED (NEG08): legality.no_arm_specific_state is not True")
    return True


def validate_scaffold_legality(spec: dict):
    """Every legality flag MUST be True; observation/action identity UNCHANGED."""
    legality = spec.get("legality", {})
    for flag in SCAFFOLD_LEGALITY_FLAGS:
        require(legality.get(flag) is True,
                "FAIL CLOSED: scaffold legality flag %r is not True (scenario=%r)"
                % (flag, spec.get("scenario")))
    # NEG09: no extra observation channel; observation schema unchanged.
    require("UNCHANGED" in spec.get("observation_schema", ""),
            "FAIL CLOSED (NEG09): observation_schema changed by the scaffold")
    # NEG10: action space unchanged.
    require("UNCHANGED" in spec.get("action_space", ""),
            "FAIL CLOSED (NEG10): action_space changed by the scaffold")
    # NEG11: no hidden boss direction / no privileged info / no shortest-path hint.
    require(legality.get("no_hidden_boss_direction") is True
            and legality.get("no_privileged_information") is True
            and legality.get("no_shortest_path_hint") is True,
            "FAIL CLOSED (NEG11): scaffold would inject privileged directional information")
    require(spec.get("scaffolded_results_can_replace_full_task") is False,
            "FAIL CLOSED: a scaffold must declare scaffolded_results_can_replace_full_task=False")
    # builder source SHA must be the audited WorldBuilder.
    require(spec["builder"]["world_builder_sha256"]
            == audit.SOURCE_FILES["world_builder"]["sha256"],
            "FAIL CLOSED: builder world_builder_sha256 != audited WorldBuilder source")
    assert_no_arm_specific_metadata(spec)
    return True


# ---------------------------------------------------------------------------
# Source-identity binding (NEG02 builder realpath/SHA; NEG03 task SHA)
# ---------------------------------------------------------------------------
def bind_builder_source_identity(imported_file=None):
    """NEG02: prove the builder source we bind to IS the audited WorldBuilder file
    (realpath + freshly-computed SHA256). On a JAX host `imported_file` would be
    ``inspect.getsourcefile(WorldBuilder)``; here it defaults to the audited on-disk
    path (so the pure SHA/realpath discipline is exercised). Any mismatch -> FailClosed.
    """
    meta = audit.SOURCE_FILES["world_builder"]
    requested = str(audit.resolve_source_path("world_builder"))
    imported = imported_file or requested
    ident = ser.verify_executed_source_identity(
        requested, imported, "world_builder", expected_sha256=meta["sha256"])
    return ident


def verify_canonical_task_source(expected_sha256=None):
    """NEG03: the canonical Stage4 task source must match its audited full SHA.

    Returns the identity record; fails closed on any mismatch. `expected_sha256`
    defaults to the audited canonical SHA — a negative test injects a wrong value to
    prove the check actually rejects.
    """
    meta = audit.SOURCE_FILES["canonical_s4_task"]
    expected = expected_sha256 if expected_sha256 is not None else meta["sha256"]
    path = audit.resolve_source_path("canonical_s4_task")
    require(path.exists(),
            "FAIL CLOSED (BLOCKED_ENVIRONMENT): canonical s4_task_code.py not on disk at %r; "
            "task source identity cannot be re-verified here." % str(path))
    actual = audit.sha256_file(path)
    require(actual == expected,
            "FAIL CLOSED (NEG03): canonical task source sha256 %s != expected %s "
            "(wrong / non-canonical / CRLF-mangled task definition)"
            % (actual[:16], expected[:16]))
    return {"role": "canonical_s4_task", "path": str(path),
            "realpath": audit.realpath_of(path), "sha256": actual,
            "expected_sha256": expected, "match": True}


# ---------------------------------------------------------------------------
# Materialization (JAX host only; FAILS CLOSED without JAX+craftax)
# ---------------------------------------------------------------------------
# Back-scaffold Kobold placement constants (validated prototype, craftax==1.4.5).
BACK_KOBOLD_MOB_COUNT = 1
BACK_KOBOLD_MIN_DIST = 2
BACK_KOBOLD_MAX_DIST = 5
BACK_KOBOLD_MOB_RNG_SALT = 0xBAC  # fold_in salt; part of the frozen scaffold identity

MAX_TIMESTEPS = 4096  # canonical shared contract (all three scenarios)


def resolve_kobold_type_id(kobold_type_id=None) -> int:
    """Resolve the Kobold type_id: an explicit value wins, else bind it live from
    the craftax constants (must agree with the frozen audit binding)."""
    if kobold_type_id is not None:
        return int(kobold_type_id)
    binding = audit.resolve_kobold_binding()  # FailClosed without craftax
    require(binding["category"] == pred.KOBOLD_CATEGORY,
            "FAIL CLOSED: live Kobold category %r != frozen %r"
            % (binding["category"], pred.KOBOLD_CATEGORY))
    return int(binding["type_id"])


def materialize_start(scenario: str, rng_seed: int = None, kobold_type_id=None,
                      world_rng=None):
    """Mint ONE real diagnostic start EnvState via WorldBuilder.

    Exactly one of ``rng_seed`` (int -> PRNGKey) or ``world_rng`` (a raw JAX key) must
    be given. The raw-key form lets the evaluator mint with the EXACT world key a
    canonical reset would use (reset_env splits its rng and feeds world_gen the second
    half), which is how FRONT == canonical-reset equivalence is proven.

    Without JAX+craftax: raises FailClosed (BLOCKED_ENVIRONMENT) and emits nothing.
    On a JAX+craftax==1.4.5 host this performs the EXACT canonical WorldBuilder rng
    sequence — `r, _r = split(world_rng); WorldBuilder(_r); build(r)` — so the FRONT
    start is leaf-identical to the canonical S4 generate_world output under the same
    world rng (including the floor-2 up-ladder removal). BACK adds one LIVE ranged
    Kobold near the player via add_mobs_randomly_near before build().

    Returns the RAW world state (generate_world-equivalent output). The evaluator
    applies MultiTaskMiniCraftaxEnv reset post-processing (task_params / achievements
    / floor-entry achievements) when injecting a bank state — exactly as canonical
    reset_env does after world_gen.
    """
    require((rng_seed is None) != (world_rng is None),
            "FAIL CLOSED: materialize_start needs exactly one of rng_seed / world_rng")
    spec = build_spec(scenario)
    validate_scaffold_legality(spec)
    bind_builder_source_identity()
    verify_canonical_task_source()
    require(ser.have_jax_craftax(),
            "FAIL CLOSED (BLOCKED_ENVIRONMENT): scaffold materialization requires JAX AND "
            "craftax (jax=%s, craftax=%s). No state is minted and no state-bank hash is "
            "emitted on this host. Run on the authorized JAX+craftax experiment host."
            % (ser.have_jax(), ser.have_craftax()))
    # --- JAX host path (imported lazily so this module stays import-safe here) ---
    import jax
    from craftax.craftax.constants import ItemType
    from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
    from minicraftax.world_builder import WorldBuilder
    if world_rng is None:
        world_rng = jax.random.PRNGKey(int(rng_seed))
    r, _r = jax.random.split(world_rng)          # canonical rng sequence
    static_params = StaticEnvParams()
    params = EnvParams(max_timesteps=MAX_TIMESTEPS)
    b = WorldBuilder(_r, static_params, params)
    inv = canonical_starting_inventory()
    if scenario == FRONT:
        b.set_starting_floor(audit.FRONT_FLOOR)
        b.set_monsters_killed(audit.FRONT_FLOOR,
                              audit.CANONICAL_TASK_FACTS["monsters_killed"]["2"])
        b.set_player_inventory(inv)
        state = b.build(r)
        # canonical floor-2 up-ladder removal, exactly as s4_task_code does:
        up = b.ladders_up[audit.FRONT_FLOOR]
        state = state.replace(
            item_map=state.item_map.at[audit.FRONT_FLOOR, up[0], up[1]].set(ItemType.NONE.value))
    elif scenario == BACK:
        type_id = resolve_kobold_type_id(kobold_type_id)
        b.set_starting_floor(audit.BACK_FLOOR)
        b.set_player_inventory(inv)
        mob_rng = jax.random.fold_in(r, BACK_KOBOLD_MOB_RNG_SALT)
        b.add_mobs_randomly_near(mob_rng, audit.BACK_FLOOR, pred.KOBOLD_CATEGORY, type_id,
                                 BACK_KOBOLD_MOB_COUNT, target_pos=b.player_position,
                                 min_dist=BACK_KOBOLD_MIN_DIST, max_dist=BACK_KOBOLD_MAX_DIST)
        state = b.build(r)
    else:
        raise FailClosed("FAIL CLOSED: unknown scenario %r" % scenario)
    return state


def normalize_envstate(state, include_map: bool = False) -> dict:
    """Map a REAL (mini)craftax EnvState pytree -> the JAX-free normalized predicate
    view documented in tier3_event_predicates.py. This adapter is the ONLY place that
    touches JAX arrays on the normalization path; everything downstream is pure Python.

    No field is invented: every key comes from audited EnvState fields
    (player_level/player_health/player_position/timestep/achievements, the
    melee/passive/ranged mob arrays, monsters_killed, down_ladders/up_ladders,
    boss_progress and the inventory counters).
    """
    require(ser.have_jax_craftax(),
            "FAIL CLOSED (BLOCKED_ENVIRONMENT): normalize_envstate requires JAX+craftax")
    import numpy as np
    from craftax.craftax.constants import Achievement
    ach_names = [a.name for a in Achievement]
    ach = np.asarray(state.achievements)
    require(int(ach.shape[0]) == len(ach_names),
            "FAIL CLOSED: achievements length %d != len(Achievement) %d" % (ach.shape[0], len(ach_names)))
    achieved = {ach_names[i] for i in range(len(ach_names)) if bool(ach[i])}

    mobs = []
    for category, mob_array in (("passive", state.passive_mobs),
                                ("melee", state.melee_mobs),
                                ("ranged", state.ranged_mobs)):
        mask = np.asarray(mob_array.mask)
        pos = np.asarray(mob_array.position)
        health = np.asarray(mob_array.health)
        type_id = np.asarray(mob_array.type_id)
        cooldown = np.asarray(mob_array.attack_cooldown)
        n_floors, n_slots = mask.shape
        for lvl in range(n_floors):
            for slot in range(n_slots):
                mobs.append({
                    "category": category,
                    "level": int(lvl),
                    "position": (int(pos[lvl, slot, 0]), int(pos[lvl, slot, 1])),
                    "health": float(health[lvl, slot]),
                    "mask": bool(mask[lvl, slot]),
                    "type_id": int(type_id[lvl, slot]),
                    "attack_cooldown": int(cooldown[lvl, slot]),
                })

    monsters_killed = {int(lvl): int(v) for lvl, v in enumerate(np.asarray(state.monsters_killed))}
    down_ladders = {int(lvl): (int(p[0]), int(p[1]))
                    for lvl, p in enumerate(np.asarray(state.down_ladders))}
    up_ladders = {int(lvl): (int(p[0]), int(p[1]))
                  for lvl, p in enumerate(np.asarray(state.up_ladders))}

    # Inventory counters are flat EnvState fields (audited canonical kit keys).
    inventory = {}
    for k in dict(audit.CANONICAL_TASK_FACTS["starting_inventory"]).keys():
        if hasattr(state, k):
            inventory[k] = int(getattr(state, k))

    view = {
        "_normalized": True,
        "player_level": int(state.player_level),
        "player_health": float(state.player_health),
        "player_position": (int(state.player_position[0]), int(state.player_position[1])),
        "timestep": int(state.timestep),
        "achieved": achieved,
        "mobs": mobs,
        "monsters_killed": monsters_killed,
        "down_ladders": down_ladders,
        "up_ladders": up_ladders,
        "inventory": inventory,
        "boss_progress": int(state.boss_progress),
        "floor2_up_ladder_removed": False,  # caller sets True for FRONT (it performs the removal)
    }
    if include_map:
        view["map"] = {int(lvl): np.asarray(m).tolist()
                       for lvl, m in enumerate(np.asarray(state.map))}
        view["item_map"] = {int(lvl): np.asarray(m).tolist()
                            for lvl, m in enumerate(np.asarray(state.item_map))}
    return view


# ---------------------------------------------------------------------------
# Spec document emission (for schemas/tier3_scaffold_spec_v1.json)
# ---------------------------------------------------------------------------
def scaffold_spec_doc() -> dict:
    front = build_front_spec()
    back = build_back_spec()
    validate_scaffold_legality(front)
    validate_scaffold_legality(back)
    return {
        "schema": SCHEMA,
        "spec_version": SPEC_VERSION,
        "source_audit_schema": audit.SCHEMA,
        "legal_builder_api": audit.LEGAL_BUILDER_API,
        "canonical_task_sha256": audit.SOURCE_FILES["canonical_s4_task"]["sha256"],
        "world_builder_sha256": audit.SOURCE_FILES["world_builder"]["sha256"],
        "scenarios": {"front_l2": front, "back_l2": back},
        "legality_flags": SCAFFOLD_LEGALITY_FLAGS,
        "forbidden_result_blindness_keys": sorted(FORBIDDEN_RESULT_BLINDNESS_KEYS),
        "materialization_status": ser.environment_status(),
        "scaffolded_results_can_replace_full_task": False,
    }


# ---------------------------------------------------------------------------
# Self-test (pure; runs on this host).
# ---------------------------------------------------------------------------
def self_test() -> int:
    problems = []

    def check(name, cond):
        if not cond:
            problems.append(name)

    front = build_front_spec()
    back = build_back_spec()
    check("front_legality_ok", validate_scaffold_legality(front) is True)
    check("back_legality_ok", validate_scaffold_legality(back) is True)
    check("inventory_valid", validate_inventory(canonical_starting_inventory()) is True)

    # NEG12: invalid inventory rejected.
    try:
        validate_inventory({"wood": -1})
        check("NEG12_inventory_negative_rejected", False)
    except FailClosed:
        check("NEG12_inventory_negative_rejected", True)
    try:
        validate_inventory({"wood": 1.5})
        check("NEG12_inventory_nonint_rejected", False)
    except FailClosed:
        check("NEG12_inventory_nonint_rejected", True)

    # NEG13: invalid position rejected.
    try:
        validate_player_position((-1, 3), 48, 48)
        check("NEG13_position_negative_rejected", False)
    except FailClosed:
        check("NEG13_position_negative_rejected", True)
    try:
        validate_player_position((5, 99), 48, 48)
        check("NEG13_position_outside_rejected", False)
    except FailClosed:
        check("NEG13_position_outside_rejected", True)
    check("position_valid_ok", validate_player_position((5, 5), 48, 48) is True)

    # NEG08: arm-specific metadata rejected.
    tainted = build_front_spec()
    tainted["arm_id"] = "persistent"
    try:
        assert_no_arm_specific_metadata(tainted)
        check("NEG08_arm_metadata_rejected", False)
    except FailClosed:
        check("NEG08_arm_metadata_rejected", True)

    # NEG09/10: changed observation/action identity rejected.
    t2 = build_front_spec()
    t2["observation_schema"] = "augmented_with_exit_arrow"
    try:
        validate_scaffold_legality(t2)
        check("NEG09_obs_change_rejected", False)
    except FailClosed:
        check("NEG09_obs_change_rejected", True)
    t3 = build_back_spec()
    t3["legality"]["no_hidden_boss_direction"] = False
    try:
        validate_scaffold_legality(t3)
        check("NEG11_hidden_boss_dir_rejected", False)
    except FailClosed:
        check("NEG11_hidden_boss_dir_rejected", True)

    # Source identity (pure SHA/realpath discipline).
    ident = bind_builder_source_identity()
    check("builder_source_identity_match", ident["identity_match"] is True)
    task_ident = verify_canonical_task_source()
    check("canonical_task_source_match", task_ident["match"] is True)

    # Materialization: FAILS CLOSED without JAX+craftax; on a JAX host it must mint
    # states that pass the frozen start predicates.
    try:
        st_f = materialize_start(FRONT, 0)
        # Reached only on a JAX+craftax host.
        check("materialize_blocked_on_this_host", ser.have_jax_craftax())
        view_f = normalize_envstate(st_f)
        view_f["floor2_up_ladder_removed"] = True
        check("front_materialized_valid_start", pred.valid_front_scaffold_start(view_f))
        kb_id = audit.CRAFTAX_RUNTIME_BINDINGS["kobold"]["type_id"]
        st_b = materialize_start(BACK, 0)
        view_b = normalize_envstate(st_b)
        check("back_materialized_valid_start", pred.valid_back_scaffold_start(view_b, kb_id))
        check("back_materialized_dk_false_t0", pred.DEFEAT_KOBOLD not in view_b["achieved"])
        check("back_materialized_floor3", view_b["player_level"] == audit.BACK_FLOOR)
    except FailClosed:
        check("materialize_blocked_on_this_host", not ser.have_jax_craftax())

    if problems:
        print("TIER3_SCAFFOLD_BUILDER_SELF_TEST_FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print("TIER3_SCAFFOLD_BUILDER_SELF_TEST_PASS (scenarios=2, legality=9 flags, env=%s)"
          % ser.environment_status())
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # On a JAX host the self-test's REAL mint checks import minicraftax from the
    # audited source tree (repo-relative, no absolute paths).
    _src = str(audit.repo_root() / "dicode_src" / "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)
    if "--self-test" in argv:
        return self_test()
    if "--json" in argv:
        import json
        print(json.dumps(scaffold_spec_doc(), indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    print("usage: tier3_scaffold_builder.py --self-test | --json")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
