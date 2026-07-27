#!/usr/bin/env python3
"""CC4 Tier3 — real-source audit (single source of truth for SHAs and facts).

This module is the audited ground truth that every other Tier3 component binds
to. It records, for each relevant real source file: the repo-relative (or
recorded absolute) path, its realpath, its full SHA256, its Python import path,
and its role. It also freezes the MiniCraftax EnvState field list, the Mobs
field list, and the canonical Stage4 DEFEAT_KOBOLD task facts — all extracted
by reading the REAL source (never invented). Anything that cannot be strictly
defined from source on this host (because the external `craftax` package is not
installed) is recorded under CRAFTAX_DEPENDENT_BINDINGS and surfaces downstream
as BLOCKED_SOURCE_SEMANTICS / BLOCKED_ENVIRONMENT rather than being guessed.

Run `python tier3_source_audit.py --self-test` to re-hash the in-repo source
files and confirm they still match the frozen SHAs (drift detector).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

SCHEMA = "mechanism_UED.tier3_source_audit/v1"


class FailClosed(Exception):
    """Hard stop. Any audit/identity mismatch raises this and exits non-zero."""


def repo_root() -> Path:
    # tools/tier3_scaffolded_evaluation/tier3_source_audit.py -> repo root
    return Path(__file__).resolve().parents[2]


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def realpath_of(path: str | os.PathLike) -> str:
    return os.path.realpath(os.path.abspath(str(path)))


# ---------------------------------------------------------------------------
# Audited source files. `relpath` is repo-relative when in_repo=True; otherwise
# `abspath` records where the canonical copy was audited (external extract).
# ---------------------------------------------------------------------------
SOURCE_FILES = {
    "env_state_definition": {
        "relpath": "dicode_src/src/minicraftax/craftax_state.py",
        "sha256": "7ed6eed02495fa6f0992ebe3e7a2c89b56d2c8d0798915fed76c60e3a5be770b",
        "module_import": "minicraftax.craftax_state",
        "key_symbols": ["EnvState", "TaskParams"],
        "role": "defines the full MiniCraftax EnvState pytree (top-level fields)",
        "in_repo": True,
    },
    "multitask_env": {
        "relpath": "dicode_src/src/minicraftax/envs/multitask.py",
        "sha256": "c8f2d5c3c23476c92ab3897f47bef4df7f202a3bd57360fc1bd4cb92b9498bae",
        "module_import": "minicraftax.envs.multitask",
        "key_symbols": ["MultiTaskMiniCraftaxEnv", "reset_env", "step_env", "get_obs"],
        "role": "canonical environment the evaluator wraps (== CC4 V3 ENV_SOURCE_SHA256)",
        "in_repo": True,
    },
    "craftax_env": {
        "relpath": "dicode_src/src/minicraftax/envs/craftax.py",
        "sha256": "be90ee9c9cb4977f07ba52b58676166dea3446f9f40e87a26ef552fbec54104a",
        "module_import": "minicraftax.envs.craftax",
        "key_symbols": ["CraftaxEnv", "achievement list (DEFEAT_KOBOLD present)"],
        "role": "base craftax-style env; enumerates Achievement members incl DEFEAT_KOBOLD",
        "in_repo": True,
    },
    "env_base": {
        "relpath": "dicode_src/src/minicraftax/envs/base.py",
        "sha256": "34e9e3392e8fe73069389387f022e4adf32b1ece1ce0d55730560434920db572",
        "module_import": "minicraftax.envs.base",
        "key_symbols": ["achievement<->player_level floor mapping"],
        "role": "ENTER_* achievement := player_level thresholds (floor identity)",
        "in_repo": True,
    },
    "world_builder": {
        "relpath": "dicode_src/src/minicraftax/world_builder.py",
        "sha256": "96536bbf955376b75c44208d80f452e1907d976cd49685a0e97e3a752679b50d",
        "module_import": "minicraftax.world_builder",
        "key_symbols": ["WorldBuilder", "build", "set_starting_floor", "add_mob",
                        "set_player_inventory", "set_monsters_killed", "place_block"],
        "role": "the ONLY legal programmatic world/state builder (scaffold mechanism)",
        "in_repo": True,
    },
    "game_mechanics": {
        "relpath": "dicode_src/src/minicraftax/game_mechanics.py",
        "sha256": "1bb9a4a64fde852c970b32dc3e049d472856490eaf68d67f64dd241319d3a65a",
        "module_import": "minicraftax.game_mechanics",
        "key_symbols": ["achievements.at[Achievement.X.value].set", "mob position/mask access"],
        "role": "how achievements are set and how mob arrays are indexed (real access pattern)",
        "in_repo": True,
    },
    "base_task": {
        "relpath": "dicode_src/src/minicraftax/tasks/base_task.py",
        "sha256": "9b2cb995a807c625fde933a5edf8266dfbf32af3aac9c767f40e41e50586b1fa",
        "module_import": "minicraftax.tasks.base_task",
        "key_symbols": ["BaseTask", "get_task_params", "generate_world (interface)"],
        "role": "task interface (generate_world / get_task_params) the canonical S4 task implements",
        "in_repo": True,
    },
    "combat_seed_task": {
        "relpath": "dicode_src/src/minicraftax/tasks/seed_tasks/combat.py",
        "sha256": "d9ede70921dc96e14a974efb20481b5eb225df4828793604b172e6a964e3fae5",
        "module_import": "minicraftax.tasks.seed_tasks.combat",
        "key_symbols": ["Env(BaseTask)", "WorldBuilder scaffold pattern (comment: ADDED SCAFFOLDING)"],
        "role": "proves WorldBuilder-based scaffolding is the repo's own legitimate mechanism",
        "in_repo": True,
    },
    "wrapper": {
        "relpath": "dicode_src/src/dicode/wrappers_cl.py",
        "sha256": "2ded41d81a98c712620dc1633262f2d185ce7dd22e7cc447db22a6ad04b0ddd8",
        "module_import": "dicode.wrappers_cl",
        "key_symbols": ["DistributedMultiTaskOptimisticLogWrapper"],
        "role": "optimistic-reset wrapper (== CC4 V3 WRAPPER_SHA256); reset = pure split chain",
        "in_repo": True,
    },
    # ---- external (audited from the frozen raw-data extract; recorded SHAs) ----
    "canonical_s4_task": {
        "abspath": ("D:/Projects/dicode-codex-director/audit_outputs/"
                    "global_raw_data_extract_20260726T110032Z/home/oseasy/experiments/"
                    "p2_v1_20260722/evidence/s4_task_code.py"),
        "sha256": "45fdd17c5b34b9f32a7f85b8030437f74d63d16bed2d6f2c683d80454e4d824d",
        "module_import": "cc4_canonical_s4_task (loaded via exec of absolute path)",
        "key_symbols": ["Env(BaseTask)", "generate_world", "get_task_params"],
        "role": "CANONICAL Stage4 DEFEAT_KOBOLD task (== CC4 V3 TASK_SHA256); FULL scenario contract",
        "in_repo": False,
    },
    "canonical_evaluator": {
        "abspath": ("D:/Projects/dicode-codex-director/audit_outputs/"
                    "global_raw_data_extract_20260726T110032Z/home/oseasy/experiments/"
                    "student_upgrade_wave1_4gpu/eval_phase2_unified.py"),
        "sha256": "224514026aefd273a6647e055fc2e1a434760dc5f4b6b9acd0624bbba57035a1",
        "module_import": "eval_phase2_unified (STATIC PROTOCOL ANCHOR; NOT executed by CC4)",
        "key_symbols": ["EVAL_SEED=42", "NUM_ENVS=256", "NUM_STEPS=4096"],
        "role": "STATIC_PROTOCOL_ANCHOR_NOT_EXECUTED (== CC4 V3 EVALUATOR_SHA256)",
        "in_repo": False,
    },
    "rejected_p2v0_s4_task": {
        "abspath": ("D:/Projects/dicode-codex-director/audit_outputs/"
                    "global_raw_data_extract_20260726T110032Z/home/oseasy/experiments/"
                    "P2-v0-exploratory-invalid-for-attribution/evidence/s4_task_code.py"),
        "sha256": "df7cde78bc4ce1067a543063d0a23037046ea9a5975ca953076e59f93e29e6f5",
        "module_import": "(REJECTED — INVALID for attribution)",
        "key_symbols": [],
        "role": "REJECTED_ALTERNATIVE: P2-v0 exploratory task; MUST NOT be used as canonical",
        "in_repo": False,
    },
}


# ---------------------------------------------------------------------------
# EnvState top-level fields — extracted verbatim from craftax_state.py (7ed6eed0).
# (name, kind). `map/item_map/mob_map/light_map/down_ladders/up_ladders/
#  chests_opened/monsters_killed` are per-level (3D map arrays indexed
#  [player_level, row, col]); player_position is [row, col].
# ---------------------------------------------------------------------------
ENVSTATE_TOP_FIELDS = [
    ("task_id", "int"),
    ("map", "ndarray[num_levels,H,W]"),
    ("item_map", "ndarray[num_levels,H,W]"),
    ("mob_map", "ndarray[num_levels,H,W]bool"),
    ("light_map", "ndarray[num_levels,H,W]"),
    ("down_ladders", "ndarray[num_levels,2]"),
    ("up_ladders", "ndarray[num_levels,2]"),
    ("chests_opened", "ndarray[num_levels]bool"),
    ("monsters_killed", "ndarray[num_levels]int"),
    ("player_position", "ndarray[2]"),
    ("player_level", "int"),
    ("player_direction", "int"),
    ("player_health", "float"),
    ("player_food", "int"),
    ("player_drink", "int"),
    ("player_energy", "int"),
    ("player_mana", "int"),
    ("is_sleeping", "bool"),
    ("is_resting", "bool"),
    ("player_recover", "float"),
    ("player_hunger", "float"),
    ("player_thirst", "float"),
    ("player_fatigue", "float"),
    ("player_recover_mana", "float"),
    ("player_xp", "int"),
    ("player_dexterity", "int"),
    ("player_strength", "int"),
    ("player_intelligence", "int"),
    ("inventory", "craftax.Inventory"),
    ("melee_mobs", "craftax.Mobs"),
    ("passive_mobs", "craftax.Mobs"),
    ("ranged_mobs", "craftax.Mobs"),
    ("mob_projectiles", "craftax.Mobs"),
    ("mob_projectile_directions", "ndarray"),
    ("player_projectiles", "craftax.Mobs"),
    ("player_projectile_directions", "ndarray"),
    ("growing_plants_positions", "ndarray"),
    ("growing_plants_age", "ndarray"),
    ("growing_plants_mask", "ndarray"),
    ("potion_mapping", "ndarray"),
    ("learned_spells", "ndarray"),
    ("sword_enchantment", "int"),
    ("bow_enchantment", "int"),
    ("armour_enchantments", "ndarray"),
    ("boss_progress", "int"),
    ("boss_timesteps_to_spawn_this_round", "int"),
    ("light_level", "float"),
    ("achievements", "ndarray[len(Achievement)]bool"),
    ("state_rng", "PRNGKey"),
    ("timestep", "int"),
    ("fractal_noise_angles", "tuple[4]"),
    ("running_original_return", "float"),
    ("task_params", "TaskParams"),
]

# Mobs dataclass fields — extracted from WorldBuilder._generate_empty_mobs /
# build() in world_builder.py (96536bbf). Shapes are per (num_levels, max_mobs).
MOBS_FIELDS = [
    ("position", "ndarray[num_levels,max_mobs,2]int"),
    ("health", "ndarray[num_levels,max_mobs]float"),
    ("mask", "ndarray[num_levels,max_mobs]bool"),
    ("attack_cooldown", "ndarray[num_levels,max_mobs]int"),
    ("type_id", "ndarray[num_levels,max_mobs]int"),
]

MAP_LAYOUT = {
    "indexing": "map[player_level, row, col]; mob_map[player_level, row, col]",
    "floor_identity": "player_level (int); num_levels == 9 (world_builder build splices 9 levels)",
    "achievement_floor_mapping": {
        "ENTER_DUNGEON": "player_level >= 1",
        "ENTER_GNOMISH_MINES": "player_level >= 2",
        "ENTER_SEWERS": "player_level >= 3",
        "ENTER_VAULT": "player_level >= 4",
        "note": "from envs/base.py and envs/multitask.py achievements.at[...].set(...)",
    },
}

# WorldBuilder legal API (the ONLY permitted scaffold mechanism).
LEGAL_BUILDER_API = [
    "WorldBuilder(rng, static_params, params)",
    "set_starting_floor(level)",
    "set_player_stats(dexterity, strength, intelligence)  # clamped [1,5]",
    "set_player_inventory(dict)  # via inventory.replace(**dict)",
    "set_weapon_enchantments(sword, bow)  # clipped [0,2]",
    "set_armour_enchantments(helmet, chestplate, leggings, boots)  # clipped [0,2]",
    "set_learned_spells(fireball, iceball)",
    "set_monsters_killed(level, count)  # clamp >=0; unlocks ladders",
    "place_block(level, block_type, position)",
    "fill_area(level, block_type, top_left, bottom_right)",
    "add_mob(level, mob_name, type_id, position, health=-1.0)  # mob_name in {melee,ranged,passive}",
    "add_mobs_randomly_near(rng, level, mob_name, type_id, n, target_pos, min_dist, max_dist, on_blocks)",
    "place_randomly / place_randomly_near / place_adjacent_to_existing",
    "build(rng) -> EnvState  # health=9.0 food/drink/energy/mana=9 achievements=zeros boss_progress=0 timestep=0",
]

# Canonical Stage4 DEFEAT_KOBOLD task facts — extracted from s4_task_code.py
# (45fdd17c). These pin the FULL scenario and therefore the front/back split.
CANONICAL_TASK_FACTS = {
    "label": "DEFEAT_KOBOLD",
    "relevant_achievements": ["DEFEAT_KOBOLD"],
    "completed_achievements": [],
    "task_params": {"needs_depletion_multiplier": 0.3},
    "starting_floor": 2,
    "monsters_killed": {"2": 8},
    "starting_inventory": {
        "wood": 7, "stone": 27, "coal": 3, "iron": 3, "sapling": 1,
        "pickaxe": 3, "sword": 3, "bow": 1, "arrows": 7, "torches": 10,
    },
    "floor2_up_ladder_removed": True,
    "lighting": "S4_dark (full dark, no radius assist)",
    "kobold_floor": 3,
    "kobold_requirement": "must descend floor2 -> floor3 and kill the kobold",
    "achievement_embedding_dims": 67,
    "build_health_food_drink_energy_mana": {"player_health": 9.0, "food": 9, "drink": 9, "energy": 9, "mana": 9},
}

# Derived frozen V1 boundary floors (from CANONICAL_TASK_FACTS, source-grounded).
FRONT_FLOOR = 2          # canonical entry floor (dark corridor)
BACK_FLOOR = 3           # kobold floor (boss/target area)
CORRIDOR_EXIT_FLOOR = 3  # reaching floor 3 == exiting the floor-2 dark corridor

# Things that genuinely require a craftax==1.4.5 host to bind exactly. These are
# NOT guessed; downstream they surface as BLOCKED_SOURCE_SEMANTICS/BLOCKED_ENVIRONMENT.
CRAFTAX_DEPENDENT_BINDINGS = [
    "Achievement.DEFEAT_KOBOLD integer index (achievements array is len(Achievement); index from craftax.craftax.constants.Achievement)",
    "MeleeMobType / MobType value for KOBOLD (add_mob type_id; from craftax.craftax.constants)",
    "ItemType.NONE integer value (used to remove the floor-2 up-ladder)",
    "BlockType walkable set (for graph-distance traversability: PATH vs WALL vs DARKNESS vs ...)",
    "craftax.Inventory full field list (nested in EnvState.inventory)",
    "craftax.Mobs is imported from craftax.craftax.craftax_state (field list audited via WorldBuilder usage)",
    "craftax.craftax.game_logic.get_distance_map (real distance-map primitive)",
    "static_params.num_levels / map_size exact values",
]


def resolve_source_path(role: str) -> Path:
    meta = SOURCE_FILES[role]
    if meta.get("in_repo"):
        return repo_root() / meta["relpath"]
    return Path(meta["abspath"])


def verify_sources(require_external: bool = False):
    """Re-hash every audited source that exists on disk.

    Returns (per_file_results, all_required_ok). In-repo files are REQUIRED to
    match. External (extract) files are verified if present; if absent they are
    reported as RECORDED_EXTERNAL and do not fail the in-repo audit unless
    require_external=True.
    """
    results = {}
    all_ok = True
    for role, meta in SOURCE_FILES.items():
        path = resolve_source_path(role)
        entry = {
            "role": role,
            "expected_sha256": meta["sha256"],
            "path": str(path),
            "realpath": realpath_of(path) if path.exists() else None,
            "module_import": meta["module_import"],
            "in_repo": meta.get("in_repo", False),
        }
        if path.exists():
            actual = sha256_file(path)
            entry["actual_sha256"] = actual
            entry["match"] = (actual == meta["sha256"])
            entry["status"] = "MATCH" if entry["match"] else "SHA_MISMATCH"
            if not entry["match"]:
                all_ok = False
        else:
            entry["actual_sha256"] = None
            entry["match"] = False
            if meta.get("in_repo") or require_external:
                entry["status"] = "MISSING_REQUIRED"
                all_ok = False
            else:
                entry["status"] = "RECORDED_EXTERNAL_NOT_ON_DISK"
        results[role] = entry
    return results, all_ok


def source_identity_doc() -> dict:
    """A runtime-source-identity-style document (no secrets) for certificates."""
    results, all_ok = verify_sources()
    return {
        "schema": SCHEMA,
        "repo_root": str(repo_root()),
        "all_required_sources_match": all_ok,
        "sources": results,
        "envstate_top_field_count": len(ENVSTATE_TOP_FIELDS),
        "mobs_fields": [n for n, _ in MOBS_FIELDS],
        "legal_builder_api": LEGAL_BUILDER_API,
        "canonical_task_facts": CANONICAL_TASK_FACTS,
        "craftax_dependent_bindings": CRAFTAX_DEPENDENT_BINDINGS,
        "boundary_floors": {
            "FRONT_FLOOR": FRONT_FLOOR,
            "BACK_FLOOR": BACK_FLOOR,
            "CORRIDOR_EXIT_FLOOR": CORRIDOR_EXIT_FLOOR,
        },
    }


def _self_test() -> int:
    doc = source_identity_doc()
    problems = []
    # In-repo sources must all MATCH.
    for role, entry in doc["sources"].items():
        if entry["in_repo"] and entry["status"] != "MATCH":
            problems.append(f"{role}: {entry['status']} ({entry['path']})")
    # The canonical wrapper/env SHAs must equal the CC4 V3 frozen anchors.
    if doc["sources"]["wrapper"]["expected_sha256"] != "2ded41d81a98c712620dc1633262f2d185ce7dd22e7cc447db22a6ad04b0ddd8":
        problems.append("wrapper SHA != CC4 V3 WRAPPER_SHA256")
    if doc["sources"]["multitask_env"]["expected_sha256"] != "c8f2d5c3c23476c92ab3897f47bef4df7f202a3bd57360fc1bd4cb92b9498bae":
        problems.append("multitask SHA != CC4 V3 ENV_SOURCE_SHA256")
    # EnvState must contain the fields the predicates rely on.
    names = {n for n, _ in ENVSTATE_TOP_FIELDS}
    for required in ["player_level", "player_health", "player_position", "achievements",
                     "melee_mobs", "map", "item_map", "down_ladders", "up_ladders",
                     "monsters_killed", "timestep", "inventory", "boss_progress"]:
        if required not in names:
            problems.append(f"EnvState missing required field: {required}")
    if problems:
        print("TIER3_SOURCE_AUDIT_SELF_TEST_FAIL")
        for p in problems:
            print("  -", p)
        return 1
    n_match = sum(1 for e in doc["sources"].values() if e["status"] == "MATCH")
    n_ext = sum(1 for e in doc["sources"].values() if e["status"] == "RECORDED_EXTERNAL_NOT_ON_DISK")
    print(f"TIER3_SOURCE_AUDIT_SELF_TEST_PASS (in-repo MATCH={n_match}, recorded-external={n_ext}, "
          f"envstate_fields={len(ENVSTATE_TOP_FIELDS)}, mobs_fields={len(MOBS_FIELDS)})")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--self-test" in argv:
        return _self_test()
    if "--json" in argv:
        print(json.dumps(source_identity_doc(), indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    print("usage: tier3_source_audit.py --self-test | --json")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
