#!/usr/bin/env python3
"""CC4 Tier3 — boundary schema (mechanism_UED.tier3_boundary_schema/v1).

The SINGLE authoritative event vocabulary for the Tier3 decomposed evaluation.
Each event records its semantic definition, the real EnvState source fields it
reads, the source-file SHA those fields were audited from, the predicate version
and the SHA of the predicate CODE that implements it, whether it needs hidden
evaluator-only state, its visibility to the Student (always false — boundary
judgements are evaluator-side), and its fail-closed conditions.

The frozen V1 definitions (and their REJECTED_ALTERNATIVEs) for the ambiguous
boundaries (corridor exit, boss area) are recorded here so downstream components
and reviewers see exactly one frozen choice.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tier3_source_audit as audit  # noqa: E402
import tier3_event_predicates as predicates  # noqa: E402

SCHEMA = "mechanism_UED.tier3_boundary_schema/v1"
PREDICATE_VERSION = predicates.PREDICATE_VERSION

REQUIRED_EVENT_KEYS = [
    "event_name",
    "semantic_definition",
    "source_fields",
    "source_file_sha256",
    "predicate_version",
    "predicate_code_sha256",
    "requires_hidden_evaluator_state",
    "visible_to_student",
    "fail_closed_conditions",
]

_SRC = audit.SOURCE_FILES


def predicate_code_sha256() -> str:
    """SHA256 of the predicate source CODE CONTENT (binds schema -> exact impl).

    Line-ending INDEPENDENT by construction: the raw bytes are CRLF-normalized
    to LF before hashing. Under core.autocrlf=true a clean worktree file may
    legitimately carry either LF or CRLF bytes (git treats both as the same
    content; the index blob is always LF), so hashing the raw worktree bytes
    makes the frozen binding flip with the checkout state (observed drift:
    LF=05ac6edc... vs CRLF=d66fe614... for the SAME blob d20ead15...). The
    LF-normalized digest equals the SHA256 of the blob's stored content and is
    therefore the single stable "source code SHA" across every checkout form.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "tier3_event_predicates.py")
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()


def _event(name, definition, source_fields, source_role, hidden, fail_closed):
    return {
        "event_name": name,
        "semantic_definition": definition,
        "source_fields": source_fields,
        "source_file_sha256": _SRC[source_role]["sha256"],
        "source_file_role": source_role,
        "predicate_version": PREDICATE_VERSION,
        "predicate_code_sha256": predicate_code_sha256(),
        "requires_hidden_evaluator_state": hidden,
        "visible_to_student": False,  # ALL boundary judgements are evaluator-only
        "fail_closed_conditions": fail_closed,
    }


def build_events():
    p = predicate_code_sha256()
    events = [
        _event(
            "VALID_FULL_START",
            "A well-formed canonical Stage4 DEFEAT_KOBOLD initial state: floor 2 "
            "entry (player_level==2), alive (player_health>0), timestep==0, "
            "DEFEAT_KOBOLD not yet achieved. Mirrors canonical s4_task_code "
            "generate_world output (set_starting_floor(2), monsters_killed[2]=8, "
            "winner-median kit, floor-2 up-ladder removed).",
            ["player_level", "player_health", "timestep", "achievements",
             "monsters_killed", "inventory", "item_map", "up_ladders"],
            "canonical_s4_task", False,
            ["missing required field", "player_health<=0", "timestep!=0",
             "DEFEAT_KOBOLD already achieved"],
        ),
        _event(
            "VALID_FRONT_SCAFFOLD_START",
            "A valid FRONT_L2 diagnostic start: on the dark corridor floor "
            "(player_level==2), alive, timestep==0, NOT already past the corridor "
            "exit (player_level < 3), DEFEAT_KOBOLD not achieved. Upstream resource "
            "prep and dungeon entry (floors 0-1) are removed by construction.",
            ["player_level", "player_health", "timestep", "achievements"],
            "canonical_s4_task", False,
            ["player_level!=2 (NEG: front start beyond exit => player_level>=3)",
             "player_health<=0", "timestep!=0", "DEFEAT_KOBOLD already achieved"],
        ),
        _event(
            "VALID_BACK_SCAFFOLD_START",
            "A valid BACK_L2 diagnostic start: on the kobold floor (player_level==3), "
            "alive, timestep==0, a LIVE kobold present on floor 3 (RANGED-category mob "
            "with the craftax Kobold type_id — resolved binding: ranged type_id 3, "
            "canonical max health 8.0; mask=True, health>0), and DEFEAT_KOBOLD false at "
            "t0. The floor-2 corridor bottleneck is removed by construction.",
            ["player_level", "player_health", "timestep", "achievements",
             "ranged_mobs(position/health/mask/type_id)"],
            "canonical_s4_task", True,
            ["player_level!=3", "player_health<=0", "timestep!=0",
             "DEFEAT_KOBOLD already achieved at t0 (NEG15)",
             "no live Kobold on floor 3 (NEG16)"],
        ),
        _event(
            "FRONT_HALF_ENTERED",
            "The player is on the floor-2 dark corridor (player_level==2).",
            ["player_level"], "canonical_s4_task", False,
            ["missing player_level"],
        ),
        _event(
            "FRONT_HALF_PROGRESS",
            "Dense GRAPH_DISTANCE_PROGRESS in [0,1] computed by GRAPH_DISTANCE: "
            "BFS shortest-path distance over an evaluator-only traversability mask "
            "derived from map[player_level]; progress = clip(1 - d(current,exit)/"
            "max(d(start,exit),1),0,1). exit = floor-2 down_ladder position. The "
            "traversability mask / map topology is evaluator-only and never enters "
            "the Student observation. Monotonicity NOT guaranteed (dead-end => 0).",
            ["player_position", "map", "down_ladders"],
            "world_builder", True,
            ["NEG17 progress outside [0,1]",
             "NEG18 exit unreachable from start without blocked label",
             "invalid_position (off-grid / non-walkable)"],
        ),
        _event(
            "CORRIDOR_EXIT_REACHED",
            "Per-state predicate: player_level >= 3 (the kobold floor). 收口 status = "
            "PENDING_EQUIVALENCE_ALIAS: the FRONT_L2 PRIMARY event is the episode-level "
            "FRONT_FLOOR_TRANSITION_REACHED (player level transitions 2 -> 3). This "
            "per-state predicate is reported as 'corridor_exit_reached' but is NOT the "
            "primary success metric until real-map evidence proves the floor transition "
            "necessarily passes through the target corridor. REJECTED_ALTERNATIVE: "
            "'standing on the floor-2 down_ladder tile' (more granular, but the actual "
            "game event is the floor transition recorded by player_level).",
            ["player_level"], "canonical_s4_task", False,
            ["missing player_level",
             "episode transition True but corridor_exit_reached alias never True "
             "(contradiction -> FailClosed)"],
        ),
        _event(
            "BACK_HALF_ENTERED",
            "The player is on the kobold/target floor (player_level==3).",
            ["player_level"], "canonical_s4_task", False,
            ["missing player_level"],
        ),
        _event(
            "BOSS_AREA_REACHED",
            "Per-state predicate (vocabulary completeness only): player on the "
            "kobold/target floor (player_level==3); the kobold IS the DEFEAT_KOBOLD "
            "target and lives on floor 3. 收口: N/A for BACK_L2 — BACK_L2 identity is "
            "BOSS_COMBAT_SCAFFOLDED (primary metric "
            "P_DEFEAT_KOBOLD_GIVEN_VALID_BACK_START); the BACK start is already on "
            "floor 3 next to a live Kobold, so boss-area SEARCH is out of scope and the "
            "metrics boss_area_reached / time_to_boss_area / BACK_BOSS_NOT_FOUND are "
            "reported N/A, never as evidence. REJECTED_ALTERNATIVE: boss_progress>0 — "
            "REJECTED because boss_progress drives the Necromancer boss-spawn mechanic "
            "(a different system), per the audited achievement list and "
            "CANONICAL_TASK_FACTS.",
            ["player_level", "boss_progress(rejected)"], "canonical_s4_task", True,
            ["missing player_level"],
        ),
        _event(
            "KOBOLD_ENGAGED",
            "Frozen V1 per-state engagement proxy: a kobold (RANGED category, ranged "
            "type_id 3) on the player's floor has an active attack_cooldown OR (via "
            "evaluator episode history) has taken damage / dealt damage. Precise "
            "combat-contact is refined by the kobold_damage_dealt / "
            "damage_received_after_engagement episode metrics.",
            ["ranged_mobs(type_id/attack_cooldown/health)", "player_level"],
            "game_mechanics", True,
            ["missing ranged_mobs", "kobold type_id unbound (needs craftax host)"],
        ),
        _event(
            "DEFEAT_KOBOLD",
            "The DEFEAT_KOBOLD achievement is set: achievements[Achievement."
            "DEFEAT_KOBOLD.value] is true. The achievement NAME is resolved "
            "symbolically (the integer index is bound from craftax.craftax.constants "
            "on a craftax==1.4.5 host; resolved value 41). This is the FULL task "
            "primary metric and the BACK_L2 primary metric "
            "(P_DEFEAT_KOBOLD_GIVEN_VALID_BACK_START) target.",
            ["achievements"], "craftax_env", False,
            ["achievements array missing / wrong length",
             "Achievement.DEFEAT_KOBOLD index unbound (needs craftax host)"],
        ),
    ]
    # Sanity: predicate_code_sha256 consistent across all events.
    for e in events:
        e["predicate_code_sha256"] = p
    return events


FROZEN_V1_DEFINITIONS = {
    "FRONT_PRIMARY_EVENT": "FRONT_FLOOR_TRANSITION_REACHED: episode-level success event — "
        "player level transitions from FRONT_FLOOR (2) to CORRIDOR_EXIT_FLOOR (3)",
    "FRONT_PRIMARY_METRIC": "P_FRONT_FLOOR_TRANSITION_REACHED_GIVEN_VALID_START",
    "FRONT_DENSE_METRIC": "GRAPH_DISTANCE_PROGRESS",
    "FRONT_HALF_START_PREDICATE": "valid_front_scaffold_start: player_level==2 AND player_health>0 "
        "AND timestep==0 AND NOT corridor_exit_reached AND DEFEAT_KOBOLD not achieved",
    "FRONT_HALF_EXIT_PREDICATE": "corridor_exit_reached: player_level >= 3",
    "CORRIDOR_EXIT_REACHED_STATUS": "PENDING_EQUIVALENCE_ALIAS — per-state predicate "
        "(player_level >= 3) reported as 'corridor_exit_reached' but NOT the FRONT_L2 "
        "primary metric until real-map evidence proves the floor transition necessarily "
        "passes through the target corridor",
    "BACK_IDENTITY": "BOSS_COMBAT_SCAFFOLDED — BACK_L2 evaluates combat against a live "
        "Kobold on floor 3, NOT boss-area search",
    "BACK_PRIMARY_METRIC": "P_DEFEAT_KOBOLD_GIVEN_VALID_BACK_START",
    "BACK_NA_METRICS": ["boss_area_reached", "time_to_boss_area", "BACK_BOSS_NOT_FOUND"],
    "BACK_HALF_START_PREDICATE": "valid_back_scaffold_start: player_level==3 AND player_health>0 AND "
        "timestep==0 AND live_kobold_present(floor3, RANGED category, ranged type_id 3) AND "
        "DEFEAT_KOBOLD false at t0",
    "BOSS_AREA_PREDICATE": "boss_area_reached: player_level == 3 (N/A for BACK_L2; vocabulary only)",
    "KOBOLD_BINDING": "RANGED category, ranged type_id 3, canonical max health 8.0 "
        "(resolved from craftax==1.4.5 MOB_ACHIEVEMENT_MAP / MOB_TYPE_HEALTH_MAPPING)",
    "DEFEAT_KOBOLD_PREDICATE": "defeat_kobold: achievements[Achievement.DEFEAT_KOBOLD.value] == True "
        "(resolved index 41 on craftax==1.4.5)",
    "FRONT_PROGRESS_METHOD": "GRAPH_DISTANCE (BFS over evaluator-only traversability from map[player_level])",
    "progress_range": [0, 1],
    "progress_monotonicity_expected": False,
    "progress_monotonicity_not_guaranteed_reason": "player may enter dead-ends / oscillate; d_t can increase",
    "unreachable_state_policy": "FAIL_CLOSED (NEG18) if exit unreachable from start without explicit blocked label",
    "invalid_position_policy": "FAIL_CLOSED if current position off-grid or non-walkable",
}

REJECTED_ALTERNATIVES = [
    {"boundary": "CORRIDOR_EXIT_REACHED", "rejected": "standing on floor-2 down_ladder tile",
     "reason": "the recorded game event is the floor transition (player_level); tile-precise "
               "definition adds fragility without changing the diagnosed capability"},
    {"boundary": "BOSS_AREA_REACHED", "rejected": "boss_progress > 0",
     "reason": "boss_progress drives the Necromancer boss-spawn mechanic, not the DEFEAT_KOBOLD target"},
    {"boundary": "DEFEAT_KOBOLD detection", "rejected": "hard-coded achievement integer index",
     "reason": "must resolve Achievement.DEFEAT_KOBOLD symbolically from craftax constants"},
    {"boundary": "FRONT progress", "rejected": "raw Manhattan distance / screen pixels / tile color",
     "reason": "not source-grounded; ignores walls/doors; can over-credit off-corridor positions"},
    {"boundary": "BACK_L2 scope", "rejected": "BACK_L2 also evaluates boss-area search "
     "(boss_area_reached / time_to_boss_area / BACK_BOSS_NOT_FOUND)",
     "reason": "the BACK scaffold starts ALREADY on floor 3 next to a live Kobold; "
               "boss-area search is not exercised by this start, so those metrics are N/A "
               "and BACK_L2 identity is BOSS_COMBAT_SCAFFOLDED only"},
]


def build_schema():
    return {
        "schema": SCHEMA,
        "predicate_version": PREDICATE_VERSION,
        "predicate_code_sha256": predicate_code_sha256(),
        "boundary_floors": {
            "FRONT_FLOOR": audit.FRONT_FLOOR,
            "BACK_FLOOR": audit.BACK_FLOOR,
            "CORRIDOR_EXIT_FLOOR": audit.CORRIDOR_EXIT_FLOOR,
        },
        "events": build_events(),
        "frozen_v1_definitions": FROZEN_V1_DEFINITIONS,
        "rejected_alternatives": REJECTED_ALTERNATIVES,
        "source_audit_schema": audit.SCHEMA,
        "all_events_visible_to_student_false": True,
    }


def self_test() -> int:
    problems = []
    doc = build_schema()
    expected_events = {
        "VALID_FULL_START", "VALID_FRONT_SCAFFOLD_START", "VALID_BACK_SCAFFOLD_START",
        "FRONT_HALF_ENTERED", "FRONT_HALF_PROGRESS", "CORRIDOR_EXIT_REACHED",
        "BACK_HALF_ENTERED", "BOSS_AREA_REACHED", "KOBOLD_ENGAGED", "DEFEAT_KOBOLD",
    }
    got = {e["event_name"] for e in doc["events"]}
    if got != expected_events:
        problems.append(f"event set mismatch: missing={expected_events - got} extra={got - expected_events}")
    actual_psha = predicate_code_sha256()
    for e in doc["events"]:
        for k in REQUIRED_EVENT_KEYS:
            if k not in e:
                problems.append(f"{e.get('event_name')}: missing key {k}")
        if e.get("visible_to_student") is not False:
            problems.append(f"{e['event_name']}: visible_to_student must be False")
        if e.get("predicate_code_sha256") != actual_psha:
            problems.append(f"{e['event_name']}: predicate_code_sha256 stale")
        # source_file_sha256 must be one of the audited SHAs
        audited_shas = {m["sha256"] for m in audit.SOURCE_FILES.values()}
        if e.get("source_file_sha256") not in audited_shas:
            problems.append(f"{e['event_name']}: source_file_sha256 not an audited source")
    # Drift check vs committed schema JSON, if present.
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                             "schemas", "tier3_boundary_schema_v1.json")
    json_path = os.path.abspath(json_path)
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as fh:
            committed = json.load(fh)
        if committed.get("predicate_code_sha256") != actual_psha:
            problems.append("committed schema JSON predicate_code_sha256 drifted from predicates source")
    if problems:
        print("TIER3_BOUNDARY_SCHEMA_SELF_TEST_FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print(f"TIER3_BOUNDARY_SCHEMA_SELF_TEST_PASS (events={len(doc['events'])}, "
          f"predicate_code_sha256={actual_psha[:12]}...)")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--self-test" in argv:
        return self_test()
    if "--emit" in argv:
        out = argv[argv.index("--emit") + 1]
        with open(out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(build_schema(), fh, indent=2, ensure_ascii=False, sort_keys=True)
            fh.write("\n")
        print(f"emitted {out}")
        return 0
    if "--json" in argv:
        print(json.dumps(build_schema(), indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    print("usage: tier3_boundary_schema.py --self-test | --json | --emit <path>")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
