"""v7fix4 P1: WALL_NATIVE_FLOOR habitat map — cross-checked against the SAME mob floor table the
proposers have always been prompted with (prompts/dicode/mobs.py), so code and prompt can never
drift apart. The v7fix3 root cause was exactly this knowledge living ONLY in prompts: the modeler
(sole author of relay_r0_floor / prereq_tree) never saw it and anchored lizard's relay at floor 2
while lizards inhabit floor 3 (Sewers)."""

import re

from auction.craftax_achievements import (
    ALL_ACHIEVEMENTS,
    FLOOR_ENTRANCES,
    MAX_DUNGEON_FLOOR,
    WALL_NATIVE_FLOOR,
    native_floor_of,
)

# mob table Name column -> the defeat_* achievement it backs (misspelled 'orc_solider' is the
# upstream craftax enum, kept verbatim). Passive mobs map to their eat_* achievements.
_MOB_TO_ACHIEVEMENT = {
    "Zombie": "defeat_zombie",
    "Skeleton": "defeat_skeleton",
    "Gnome Warrior": "defeat_gnome_warrior",
    "Gnome Archer": "defeat_gnome_archer",
    "Orc Soldier": "defeat_orc_solider",
    "Orc Mage": "defeat_orc_mage",
    "Lizard": "defeat_lizard",
    "Kobold": "defeat_kobold",
    "Knight": "defeat_knight",
    "Knight Archer": "defeat_archer",
    "Troll": "defeat_troll",
    "Deep Thing": "defeat_deep_thing",
    "Pigman": "defeat_pigman",
    "Fire Elemental": "defeat_fire_elemental",
    "Frost Troll": "defeat_frost_troll",
    "Ice Elemental": "defeat_ice_elemental",
    "Bat": "eat_bat",
    "Snail": "eat_snail",
}


def _parse_mob_floor_table() -> dict[str, list[int]]:
    """Parse the markdown mob table the personas are prompted with -> {mob name: [floors]}."""
    from dicode.dreaming.prompts.dicode.mobs import context

    out: dict[str, list[int]] = {}
    for line in context.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 7 or cells[0] in ("Name", "") or set(cells[0]) <= {"-"}:
            continue
        floors = [int(x) for x in re.findall(r"\d+", cells[-1])]
        if floors:
            out[cells[0]] = floors
    return out


def test_mob_walls_match_the_prompted_floor_table():
    """Every mob-backed wall's native floor == the SHALLOWEST floor in the persona table."""
    table = _parse_mob_floor_table()
    assert len(table) >= 18, f"mob table parse broke: {sorted(table)}"
    for mob, ach in _MOB_TO_ACHIEVEMENT.items():
        assert mob in table, f"{mob} vanished from prompts/dicode/mobs.py"
        expect = min(table[mob])
        got = WALL_NATIVE_FLOOR.get(ach)
        assert got == expect, (
            f"{ach}: code says floor {got}, the prompted mob table says {table[mob]} "
            f"(native must be min={expect}) — code and prompt drifted"
        )


def test_necromancer_is_boss_floor():
    # The necromancer is the boss-level encounter, not a spawn-table mob — pinned to floor 8.
    assert WALL_NATIVE_FLOOR["defeat_necromancer"] == MAX_DUNGEON_FLOOR == 8
    assert WALL_NATIVE_FLOOR["damage_necromancer"] == 8


def test_floor_entrances_shape():
    assert sorted(FLOOR_ENTRANCES) == list(range(1, MAX_DUNGEON_FLOOR + 1))
    for floor, name in FLOOR_ENTRANCES.items():
        assert name in ALL_ACHIEVEMENTS and name.startswith("enter_")
        assert WALL_NATIVE_FLOOR[name] == floor  # entrance lives on its own floor
    assert FLOOR_ENTRANCES[3] == "enter_sewers"  # the v7fix3 broken link, pinned


def test_all_keys_are_real_achievements_and_bounded():
    assert set(WALL_NATIVE_FLOOR) <= ALL_ACHIEVEMENTS
    assert all(0 <= f <= MAX_DUNGEON_FLOOR for f in WALL_NATIVE_FLOOR.values())
    # every deep-tier combat/entrance achievement is covered (nothing important defaults to 0)
    for ach in ALL_ACHIEVEMENTS:
        if ach.startswith(("defeat_", "enter_")) or ach == "damage_necromancer":
            assert ach in WALL_NATIVE_FLOOR, f"{ach} missing from the habitat map"


def test_native_floor_of_tolerant():
    assert native_floor_of("defeat_lizard") == 3          # the v7fix3 case, pinned
    assert native_floor_of("  DEFEAT_LIZARD  ") == 3      # case/whitespace tolerant
    assert native_floor_of("enter_sewers") == 3
    assert native_floor_of("collect_wood") == 0           # unlisted basics -> natural spawn
    assert native_floor_of("smelt_iron") == 0             # hallucinated names never crash a gate
    assert native_floor_of(None) == 0
    assert native_floor_of(3) == 0
