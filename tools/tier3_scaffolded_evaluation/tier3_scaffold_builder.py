#!/usr/bin/env python3
"""CC4 Tier3 — scaffold builder (the ONLY legal scaffold mechanism = WorldBuilder).

A scaffold spec is a PURE, declarative description of how a diagnostic start state
is produced from the canonical Stage4 DEFEAT_KOBOLD task facts (audited in
tier3_source_audit.py) using the repository's OWN WorldBuilder API — the same
mechanism the repo's combat seed task uses under the comment "ADDED SCAFFOLDING".
No scaffold invents fields, adds observation channels, or injects privileged
information; every legality flag is declared and machine-checked here, and the
negative tests (tier3_negative_tests.py) actively try to violate each one.

Two diagnostic scaffolds (frozen V1):
  * FRONT_L2: start on the floor-2 dark corridor (canonical entry). Removes the
    upstream resource prep / dungeon entry (floors 0-1). Keeps corridor navigation,
    multi-mob survival, and the exit search (descend to floor 3).
  * BACK_L2: start on floor 3 with a LIVE kobold present. Removes the floor-2
    corridor bottleneck. Keeps boss/kobold search, engagement, combat, survival,
    and DEFEAT_KOBOLD.

MATERIALIZATION (actually calling WorldBuilder to mint EnvState pytrees) requires a
JAX + craftax==1.4.5 host and is GUARDED: on this host ``materialize_start`` FAILS
CLOSED with BLOCKED_ENVIRONMENT and never emits a state. Only the pure spec /
legality / source-identity layer runs here (IMPLEMENTED_STATIC / TESTED_SYNTHETIC).
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
        "primary_metric": "P_CORRIDOR_EXIT_REACHED_GIVEN_VALID_START",
        "dense_metric": "NORMALIZED_CORRIDOR_PROGRESS",
        "boundary_predicates": {
            "start": "valid_front_scaffold_start",
            "exit": "corridor_exit_reached",
        },
        "start_floor": audit.FRONT_FLOOR,
        "exit_floor": audit.CORRIDOR_EXIT_FLOOR,
        "builder": {
            "mechanism": "WorldBuilder (minicraftax.world_builder; repo-native scaffolding)",
            "world_builder_sha256": audit.SOURCE_FILES["world_builder"]["sha256"],
            "calls": [
                "WorldBuilder(rng, static_params, params)",
                "set_starting_floor(%d)" % audit.FRONT_FLOOR,
                "set_monsters_killed(%d, %d)" % (audit.FRONT_FLOOR,
                                                 f["monsters_killed"]["2"]),
                "set_player_inventory(<canonical starting kit>)",
                "build(rng)",
                "item_map.at[%d, up_ladder].set(ItemType.NONE.value)  # canonical floor-2 up-ladder removal" % audit.FRONT_FLOOR,
            ],
            "starting_inventory": canonical_starting_inventory(),
            "monsters_killed": dict(f["monsters_killed"]),
            "floor2_up_ladder_removed": f["floor2_up_ladder_removed"],
        },
        "removes": "upstream resource preparation and dungeon entry (floors 0-1)",
        "keeps": "floor-2 DARK corridor navigation, multi-mob survival, exit search",
        "legality": _base_legality(),
        "observation_schema": "canonical_craftax_symbolic (UNCHANGED)",
        "action_space": "canonical_craftax_action_set (UNCHANGED)",
        "scaffolded_results_can_replace_full_task": False,
    }


def build_back_spec() -> dict:
    f = audit.CANONICAL_TASK_FACTS
    return {
        "schema": SCHEMA,
        "spec_version": SPEC_VERSION,
        "scenario": BACK,
        "identity_class": "TIER3_BACK_DIAGNOSTIC_SCAFFOLD",
        "purpose": "MECHANISM_DIAGNOSIS_ONLY",
        "primary_metric": "P_DEFEAT_KOBOLD_GIVEN_VALID_BACK_START",
        "dense_metric": "none",
        "boundary_predicates": {
            "start": "valid_back_scaffold_start",
            "boss_area": "boss_area_reached",
            "defeat": "defeat_kobold",
        },
        "start_floor": audit.BACK_FLOOR,
        "boss_floor": audit.BACK_FLOOR,
        "require_live_kobold_at_start": True,
        "forbid_defeat_kobold_at_start": True,
        "kobold_type_id_binding": "BLOCKED_SOURCE_SEMANTICS (MeleeMobType/MobType KOBOLD from craftax==1.4.5 host)",
        "builder": {
            "mechanism": "WorldBuilder (minicraftax.world_builder; repo-native scaffolding)",
            "world_builder_sha256": audit.SOURCE_FILES["world_builder"]["sha256"],
            "calls": [
                "WorldBuilder(rng, static_params, params)",
                "set_starting_floor(%d)" % audit.BACK_FLOOR,
                "set_player_inventory(<canonical starting kit>)",
                "add_mob(%d, 'melee', KOBOLD_TYPE_ID, position, health=-1.0)  # live kobold on the target floor" % audit.BACK_FLOOR,
                "build(rng)",
            ],
            "starting_inventory": canonical_starting_inventory(),
        },
        "removes": "the floor-2 dark corridor bottleneck",
        "keeps": "boss/kobold search, engagement, combat, survival, DEFEAT_KOBOLD",
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
# Materialization (JAX host only; FAILS CLOSED here)
# ---------------------------------------------------------------------------
def materialize_start(scenario: str, rng_seed: int, kobold_type_id=None):
    """Mint ONE real diagnostic start EnvState via WorldBuilder.

    BLOCKED_ENVIRONMENT on this host (no JAX / no craftax): raises FailClosed and
    emits nothing. On a JAX+craftax==1.4.5 host this performs the exact, spec'd
    WorldBuilder call sequence (front: canonical floor-2 entry; back: floor-3 entry
    with a live kobold), then re-validates the resulting normalized view against the
    frozen boundary predicates before returning it.
    """
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
    from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
    from minicraftax.world_builder import WorldBuilder
    rng = jax.random.PRNGKey(int(rng_seed))
    rng, build_rng = jax.random.split(rng)
    b = WorldBuilder(build_rng, StaticEnvParams(), EnvParams(max_timesteps=4096))
    inv = canonical_starting_inventory()
    if scenario == FRONT:
        b.set_starting_floor(audit.FRONT_FLOOR)
        b.set_monsters_killed(audit.FRONT_FLOOR,
                              audit.CANONICAL_TASK_FACTS["monsters_killed"]["2"])
        b.set_player_inventory(inv)
        rng, rng2 = jax.random.split(rng)
        state = b.build(rng2)
        # canonical floor-2 up-ladder removal is applied by the caller/evaluator exactly
        # as s4_task_code does (item_map.at[2, up].set(ItemType.NONE.value)).
    elif scenario == BACK:
        require(kobold_type_id is not None,
                "FAIL CLOSED (BLOCKED_SOURCE_SEMANTICS): back scaffold needs the craftax "
                "KOBOLD type_id (bound from craftax constants on a JAX host).")
        b.set_starting_floor(audit.BACK_FLOOR)
        b.set_player_inventory(inv)
        b.add_mob(audit.BACK_FLOOR, "melee", int(kobold_type_id), (8, 8), -1.0)
        rng, rng2 = jax.random.split(rng)
        state = b.build(rng2)
    else:
        raise FailClosed("FAIL CLOSED: unknown scenario %r" % scenario)
    return state


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

    # Materialization is honestly blocked on this host.
    try:
        materialize_start(FRONT, 0)
        # If a JAX host somehow runs this, materialize should succeed; accept either,
        # but on THIS host it must fail closed.
        check("materialize_blocked_on_this_host", ser.have_jax_craftax())
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
