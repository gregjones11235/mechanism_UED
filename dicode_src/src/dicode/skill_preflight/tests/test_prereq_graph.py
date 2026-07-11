"""Offline tests for the direct-prerequisite graph (C-2-lite §1 data layer).

Run on the pod:
    cd /workspace/mechanism_UED/dicode_src
    uv run pytest src/dicode/skill_preflight/tests/test_prereq_graph.py -v
"""
from __future__ import annotations

from auction.craftax_achievements import ACHIEVEMENT_DEPTH, ALL_ACHIEVEMENTS
from dicode.skill_preflight.prereq_graph import (
    DIRECT_PREREQS,
    FLOOR_ENTRY_LADDER,
    floor_grants,
    inventory_grants,
)


def test_full_coverage_and_valid_names():
    assert set(DIRECT_PREREQS) == set(ALL_ACHIEVEMENTS)
    for a, ps in DIRECT_PREREQS.items():
        assert ps <= ALL_ACHIEVEMENTS, f"{a}: unknown prereq names {ps - ALL_ACHIEVEMENTS}"
        assert a not in ps


def test_prereqs_never_deeper_than_self():
    """A prerequisite may sit in the same tier but never a deeper one."""
    for a, ps in DIRECT_PREREQS.items():
        for p in ps:
            assert ACHIEVEMENT_DEPTH[p] <= ACHIEVEMENT_DEPTH[a], (
                f"{a} (tier {ACHIEVEMENT_DEPTH[a]}) depends on deeper "
                f"{p} (tier {ACHIEVEMENT_DEPTH[p]})"
            )


def test_known_chains():
    # The diamond-sword chain the whole leak analysis is anchored on.
    assert "make_iron_pickaxe" in DIRECT_PREREQS["collect_diamond"]
    assert "collect_diamond" in DIRECT_PREREQS["make_diamond_sword"]
    assert {"collect_iron", "collect_coal", "place_furnace"} <= DIRECT_PREREQS["make_iron_pickaxe"]
    # Ground-truth mob floors (craftax 1.4.5 FLOOR_MOB_MAPPING): orcs on floor 1.
    assert DIRECT_PREREQS["defeat_orc_solider"] == {"enter_dungeon"}
    assert DIRECT_PREREQS["defeat_gnome_warrior"] == {"enter_gnomish_mines"}
    assert DIRECT_PREREQS["defeat_lizard"] == {"enter_sewers"}
    # Floor ladder is a strict chain.
    for i in range(1, len(FLOOR_ENTRY_LADDER)):
        assert FLOOR_ENTRY_LADDER[i - 1] in DIRECT_PREREQS[FLOOR_ENTRY_LADDER[i]]


def test_floor_grants():
    assert floor_grants(0) == frozenset()
    assert floor_grants(1) == {"enter_dungeon"}
    assert floor_grants(3) == {"enter_dungeon", "enter_gnomish_mines", "enter_sewers"}
    assert len(floor_grants(99)) == len(FLOOR_ENTRY_LADDER)  # clamped


def test_inventory_grants_levelled_and_flat():
    g = inventory_grants({"pickaxe": 3, "coal": 50})
    # A level-3 (iron) pickaxe subsumes wood/stone/iron pickaxe capability.
    assert {"make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe"} <= g
    assert "make_diamond_pickaxe" not in g
    assert "collect_coal" in g
    # Unknown / non-numeric / zero values are ignored.
    assert inventory_grants({"mystery_item": 5, "wood": "lots", "iron": 0}) == frozenset()
