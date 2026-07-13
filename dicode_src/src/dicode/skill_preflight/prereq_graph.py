"""Direct-prerequisite graph over the 67 Craftax achievements (C-2-lite §1 data layer).

Each achievement maps to the set of achievements that must be *individually* performable
before it becomes practicable — the DIRECT prerequisites only (one hop), not the transitive
closure. This is the dependency-edge data the prereq-frontier scheduling criterion
(skill_scheduler.pick_target, frontier_mode="prereq") and the scaffold gate (scaffold_gate.py)
consume. DEPTH_TIERS in auction.craftax_achievements remains the coarse 4-tier bucketing;
this graph is the fine-grained per-node structure underneath it.

CURATION PROVENANCE (2026-07-11, verified against craftax==1.4.5 wheel source — the exact
version DiCode pins; see auction/craftax_achievements.py header for the same pin argument):

  * Mining pickaxe levels (game_logic.py): stone/coal need pickaxe>=1 (wood), iron needs >=2
    (stone), diamond needs >=3 (iron), sapphire/ruby mining needs >=4 (diamond).
  * Crafting recipes (game_logic.py do_crafting): wood_pickaxe = wood@table;
    stone tools = wood+stone@table; iron tools = wood+stone+iron+coal@table(+furnace);
    diamond_pickaxe = wood+3 diamond@table; diamond_sword = wood+diamond@table;
    iron_armour = 3 iron+3 coal; diamond_armour = 3 diamond; arrow = wood+stone; torch = wood+coal.
  * Floor order (constants.py LEVEL_ACHIEVEMENT_MAP): 1 dungeon, 2 gnomish mines, 3 sewers,
    4 vault, 5 troll mines, 6 fire realm, 7 ice realm, 8 graveyard.
  * Mob->floor (constants.py FLOOR_MOB_MAPPING x renderer texture index order):
    floor1 dungeon = orc_soldier/orc_mage/snail; floor2 gnomish mines = gnome_warrior/
    gnome_archer/bat; floor3 sewers = lizard/kobold/snail; floor4 vault = knight/knight_archer;
    floor5 troll mines = troll/deep_thing/bat; floor6 fire = pigman/fire_elemental;
    floor7 ice = frost_troll/ice_elemental; floor8 graveyard = necromancer.
    NOTE the counter-intuitive ground truth: ORCS ARE ON FLOOR 1 (dungeon), gnomes on floor 2.
  * Books/potions/gems/bow drop from chests (game_logic loot fields incl. bow); chests
    appear from floor 1 -> open_chest gates the book/potion/gem/bow line. NOTE the
    COLLECT_*/FIND_BOW/MAKE_* family are INVENTORY-STATE achievements (fire on
    inventory.x > 0 / gear level, game_logic ~2871-2972) — chest loot legitimately grants
    them, which also makes ITEM_*_GRANTS below literally game-true, not an approximation.
  * 2026-07-13 HOTFIX (found via C2lite_full_2e9 magic-line flat-zero autopsy): chest loot
    is FLOOR-GATED for two item classes (game_logic l.144-152): bow = floor-1 first chest;
    BOOKS = floor-3/4 (sewers/vault) FIRST chest ONLY. Gems/potions/resources are
    probability-gated, no floor condition (consistent with sapphire/ruby rising in the run
    while learn_* stayed hard-zero). learn_fireball/learn_iceball therefore additionally
    require enter_sewers. Both the hand curation AND the autoextract spike missed the floor
    gate (the extractor read loot FIELDS but not the level condition on the loot line) —
    extractor must capture player_level gates; add to the half-day B-tier work.
  * 2026-07-11 autoextract-spike corrections (prereq_autoextract_spike.py found three
    missed station edges + one imprecise edge vs. do_crafting ground truth): iron armour
    needs table AND furnace (l.713); diamond armour and torch need table; find_bow's
    precise edge is open_chest (was the looser enter_dungeon).
  * Enchantment tables (world_gen_configs.py): ICE table in SEWERS (floor 3, consumes
    sapphire), FIRE table in VAULTS (floor 4, consumes ruby). Canonical (cheapest) enchant
    path = sewers + sapphire.

SEMANTICS / CURATION RULES:
  * Edges are CONJUNCTIVE: every listed prerequisite is required (matches how the scheduler
    gates candidacy). Where the game offers alternatives (e.g. sapphire is minable with a
    diamond pickaxe OR looted from chests) we list only the CANONICAL CHEAPEST path and note
    the alternative in _ALTERNATIVE_PATHS below — a conjunctive edge to the expensive branch
    would wrongly block scheduling.
  * "Free" overworld achievements (punchable mobs, drinking, waking up, finding the first
    ladder) get an EMPTY prereq set: they are always schedulable.
  * Like DEPTH_TIERS, this is a hand-curated design artifact, revisable; unlike DEPTH_TIERS
    it is per-node and recipe-faithful. If craftax is ever bumped, re-verify the provenance
    items above.
"""
from __future__ import annotations

from auction.craftax_achievements import ALL_ACHIEVEMENTS

# --- The graph -------------------------------------------------------------------------

DIRECT_PREREQS: dict[str, frozenset[str]] = {
    # -- tier 1: overworld basics --------------------------------------------------------
    "collect_wood": frozenset(),
    "place_table": frozenset({"collect_wood"}),
    "eat_cow": frozenset(),                    # punchable
    "collect_sapling": frozenset(),
    "collect_drink": frozenset(),
    "make_wood_pickaxe": frozenset({"collect_wood", "place_table"}),
    "make_wood_sword": frozenset({"collect_wood", "place_table"}),
    "place_plant": frozenset({"collect_sapling"}),
    "defeat_zombie": frozenset(),              # possible bare-handed
    "collect_stone": frozenset({"make_wood_pickaxe"}),
    "place_stone": frozenset({"collect_stone"}),
    "eat_plant": frozenset({"place_plant"}),
    "defeat_skeleton": frozenset(),
    "make_stone_pickaxe": frozenset({"collect_wood", "collect_stone", "place_table"}),
    "make_stone_sword": frozenset({"collect_wood", "collect_stone", "place_table"}),
    "wake_up": frozenset(),

    # -- tier 2: furnace/iron line, first descent ----------------------------------------
    "place_furnace": frozenset({"collect_stone"}),
    "collect_coal": frozenset({"make_wood_pickaxe"}),
    "collect_iron": frozenset({"make_stone_pickaxe"}),
    "make_iron_pickaxe": frozenset({
        "collect_wood", "collect_stone", "collect_iron", "collect_coal",
        "place_table", "place_furnace",
    }),
    "make_iron_sword": frozenset({
        "collect_wood", "collect_stone", "collect_iron", "collect_coal",
        "place_table", "place_furnace",
    }),
    "make_arrow": frozenset({"collect_wood", "collect_stone", "place_table"}),
    "make_torch": frozenset({"collect_wood", "collect_coal", "place_table"}),
    "place_torch": frozenset({"make_torch"}),
    "make_iron_armour": frozenset({
        "collect_iron", "collect_coal", "place_table", "place_furnace",
    }),
    "enter_dungeon": frozenset(),              # overworld down-ladder; no hard gate
    "enter_gnomish_mines": frozenset({"enter_dungeon"}),
    "find_bow": frozenset({"open_chest"}),      # FIND_BOW = inventory.bow>0; bow is chest loot
    "fire_bow": frozenset({"find_bow", "make_arrow"}),
    "eat_bat": frozenset({"enter_gnomish_mines"}),      # bats: floors 2/5/6/7, earliest = 2
    "eat_snail": frozenset({"enter_dungeon"}),          # snails: floors 1/3/4, earliest = 1
    "open_chest": frozenset({"enter_dungeon"}),

    # -- tier 3: diamond line, deeper floors, floor-1..3 combat, magic --------------------
    "collect_diamond": frozenset({"make_iron_pickaxe"}),
    "make_diamond_sword": frozenset({"collect_wood", "collect_diamond", "place_table"}),
    "make_diamond_pickaxe": frozenset({"collect_wood", "collect_diamond", "place_table"}),
    "make_diamond_armour": frozenset({"collect_diamond", "place_table"}),
    "enter_sewers": frozenset({"enter_gnomish_mines"}),
    "enter_vault": frozenset({"enter_sewers"}),
    "enter_troll_mines": frozenset({"enter_vault"}),
    "defeat_gnome_warrior": frozenset({"enter_gnomish_mines"}),
    "defeat_gnome_archer": frozenset({"enter_gnomish_mines"}),
    "defeat_orc_solider": frozenset({"enter_dungeon"}),   # ground truth: orcs on floor 1
    "defeat_orc_mage": frozenset({"enter_dungeon"}),
    "defeat_lizard": frozenset({"enter_sewers"}),
    "defeat_kobold": frozenset({"enter_sewers"}),
    "defeat_troll": frozenset({"enter_troll_mines"}),
    "collect_sapphire": frozenset({"open_chest"}),        # canonical: chest loot
    "collect_ruby": frozenset({"open_chest"}),            # canonical: chest loot
    "learn_fireball": frozenset({"open_chest", "enter_sewers"}),  # books: floor-3/4 FIRST chest only
    "cast_fireball": frozenset({"learn_fireball"}),
    "learn_iceball": frozenset({"open_chest", "enter_sewers"}),   # same floor gate as fireball
    "cast_iceball": frozenset({"learn_iceball"}),
    "drink_potion": frozenset({"open_chest"}),
    "enchant_sword": frozenset({"make_wood_sword", "collect_sapphire", "enter_sewers"}),
    "enchant_armour": frozenset({"make_iron_armour", "collect_sapphire", "enter_sewers"}),

    # -- tier 4: deep floors and their mobs ------------------------------------------------
    "enter_fire_realm": frozenset({"enter_troll_mines"}),
    "enter_ice_realm": frozenset({"enter_fire_realm"}),
    "enter_graveyard": frozenset({"enter_ice_realm"}),
    "defeat_knight": frozenset({"enter_vault"}),
    "defeat_archer": frozenset({"enter_vault"}),          # knight_archer, vault ranged mob
    "defeat_deep_thing": frozenset({"enter_troll_mines"}),
    "defeat_pigman": frozenset({"enter_fire_realm"}),
    "defeat_fire_elemental": frozenset({"enter_fire_realm"}),
    "defeat_frost_troll": frozenset({"enter_ice_realm"}),
    "defeat_ice_elemental": frozenset({"enter_ice_realm"}),
    "damage_necromancer": frozenset({"enter_graveyard"}),
    "defeat_necromancer": frozenset({"damage_necromancer"}),
}

# Documented alternatives NOT encoded as edges (conjunctive semantics would over-block):
_ALTERNATIVE_PATHS: dict[str, str] = {
    "collect_sapphire": "also minable with a diamond pickaxe (pickaxe>=4)",
    "collect_ruby": "also minable with a diamond pickaxe (pickaxe>=4)",
    "enchant_sword": "FIRE table in vaults + ruby is the alternative recipe",
    "enchant_armour": "FIRE table in vaults + ruby is the alternative recipe",
    "defeat_zombie": "canonically eased by make_wood_sword, but bare-handed is possible",
}

# --- Floor entry ladder (LEVEL_ACHIEVEMENT_MAP order; index = floor number) --------------
# Used by the scaffold gate to translate builder.set_starting_floor(n) into the set of
# enter_* achievements the task short-circuits.
FLOOR_ENTRY_LADDER: tuple[str, ...] = (
    "enter_dungeon",        # floor 1
    "enter_gnomish_mines",  # floor 2
    "enter_sewers",         # floor 3
    "enter_vault",          # floor 4
    "enter_troll_mines",    # floor 5
    "enter_fire_realm",     # floor 6
    "enter_ice_realm",      # floor 7
    "enter_graveyard",      # floor 8
)


def floor_grants(starting_floor: int) -> frozenset[str]:
    """enter_* achievements short-circuited by starting on ``starting_floor`` (>0)."""
    n = max(0, min(int(starting_floor), len(FLOOR_ENTRY_LADDER)))
    return frozenset(FLOOR_ENTRY_LADDER[:n])


# --- Inventory grants ---------------------------------------------------------------------
# builder.set_player_inventory({...}) items -> the achievement whose in-episode performance
# they substitute. Two families:
#   * levelled gear ("pickaxe"/"sword"/"armour": value = tier level 1..4)
#   * flat resources / items (value = count; any positive count substitutes the collect/make)
ITEM_LEVEL_GRANTS: dict[str, dict[int, str]] = {
    "pickaxe": {1: "make_wood_pickaxe", 2: "make_stone_pickaxe",
                3: "make_iron_pickaxe", 4: "make_diamond_pickaxe"},
    "sword": {1: "make_wood_sword", 2: "make_stone_sword",
              3: "make_iron_sword", 4: "make_diamond_sword"},
    "armour": {1: "make_iron_armour", 2: "make_diamond_armour"},
}

ITEM_FLAT_GRANTS: dict[str, str] = {
    "wood": "collect_wood",
    "stone": "collect_stone",
    "coal": "collect_coal",
    "iron": "collect_iron",
    "diamond": "collect_diamond",
    "sapphire": "collect_sapphire",
    "ruby": "collect_ruby",
    "sapling": "collect_sapling",
    "arrows": "make_arrow",
    "torches": "make_torch",
    "bow": "find_bow",
    "books": "learn_fireball",   # a granted book substitutes the chest->book acquisition leg
    "potions": "drink_potion",
}


def inventory_grants(items: dict[str, object]) -> frozenset[str]:
    """Achievements substituted by a set_player_inventory items dict.

    Levelled gear grants the achievement for its level AND all lower levels of the same
    line (an iron pickaxe subsumes wood/stone pickaxe capability). Flat items grant on any
    positive count. Unknown keys / non-numeric values are ignored (conservative).
    """
    granted: set[str] = set()
    for key, raw in (items or {}).items():
        try:
            val = int(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if val <= 0:
            continue
        if key in ITEM_LEVEL_GRANTS:
            for lvl, ach in ITEM_LEVEL_GRANTS[key].items():
                if lvl <= val:
                    granted.add(ach)
        elif key in ITEM_FLAT_GRANTS:
            granted.add(ITEM_FLAT_GRANTS[key])
    return frozenset(granted)


# --- Integrity checks (import-time, same style as craftax_achievements) -------------------
assert set(DIRECT_PREREQS) == set(ALL_ACHIEVEMENTS), (
    f"prereq graph must cover all 67 achievements; "
    f"missing={set(ALL_ACHIEVEMENTS) - set(DIRECT_PREREQS)}, "
    f"extra={set(DIRECT_PREREQS) - set(ALL_ACHIEVEMENTS)}"
)
for _a, _ps in DIRECT_PREREQS.items():
    _bad = _ps - ALL_ACHIEVEMENTS
    assert not _bad, f"{_a} has unknown prerequisite names: {_bad}"
    assert _a not in _ps, f"{_a} lists itself as a prerequisite"


def _assert_acyclic() -> None:
    """DFS cycle check (67 nodes; trivial cost at import)."""
    WHITE, GREY, BLACK = 0, 1, 2
    colour = {a: WHITE for a in DIRECT_PREREQS}

    def visit(a: str, stack: tuple[str, ...]) -> None:
        colour[a] = GREY
        for p in DIRECT_PREREQS[a]:
            if colour[p] == GREY:
                raise AssertionError(f"prereq cycle: {' -> '.join(stack + (a, p))}")
            if colour[p] == WHITE:
                visit(p, stack + (a,))
        colour[a] = BLACK

    for a in DIRECT_PREREQS:
        if colour[a] == WHITE:
            visit(a, ())


_assert_acyclic()
