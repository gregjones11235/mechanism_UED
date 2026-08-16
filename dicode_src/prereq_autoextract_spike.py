#!/usr/bin/env python3
"""SPIKE: auto-extract the achievement prerequisite graph from craftax source (v1.4.5).

Purpose (2026-07-11): answer the "how much of prereq_graph.py is human prior?" question with
numbers — which edges are mechanically derivable from the environment definition (and thus
count as O(environment-spec) knowledge, defensible against the UED-purity critique), and
which require human judgment. Compares its own output against the hand-curated
DIRECT_PREREQS edge by edge.

Extraction components (all static AST / literal parsing, no craftax import, no jax):
  A. recipes     do_crafting: can_craft_<item> inventory comparisons + station gates
  B. mining      can_mine_<block> = inventory.pickaxe >= n
  C. floors      constants.LEVEL_ACHIEVEMENT_MAP -> entry ladder chain
  D. mobs        renderer texture lists x FLOOR_MOB_MAPPING x FLOOR_MOB_SPAWN_CHANCE
  E. chest loot  fields incremented under is_opening_chest; read_book -> LEARN_*;
                 learned_spells -> CAST_*
  F. placing     is_placing_* inventory consumption -> Achievement.PLACE_*

Usage:
  python prereq_autoextract_spike.py --craftax-src /path/to/craftax/craftax [--diff]
  (--diff requires running from dicode_src with PYTHONPATH=src:. so DIRECT_PREREQS imports)
"""
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

# ---- static glue tables (the ONLY human input; each entry is a naming/spelling fact, ------
# ---- not a design decision) ---------------------------------------------------------------

# craftax's own enum misspelling + renderer-name vs enum-name mismatches.
MOB_NAME_TO_ACHIEVEMENT = {
    "zombie": "defeat_zombie", "skeleton": "defeat_skeleton",
    "gnome_warrior": "defeat_gnome_warrior", "gnome_archer": "defeat_gnome_archer",
    "orc_soldier": "defeat_orc_solider",  # upstream enum misspells "solider"
    "orc_mage": "defeat_orc_mage",
    "lizard": "defeat_lizard", "kobold": "defeat_kobold",
    "knight": "defeat_knight", "knight_archer": "defeat_archer",  # enum drops "knight_"
    "troll": "defeat_troll", "deep_thing": "defeat_deep_thing",
    "pigman": "defeat_pigman", "fire_elemental": "defeat_fire_elemental",
    "frost_troll": "defeat_frost_troll", "ice_elemental": "defeat_ice_elemental",
    "cow": "eat_cow", "bat": "eat_bat", "snail": "eat_snail",
}
INGREDIENT_FIELD_TO_ACHIEVEMENT = {
    "wood": "collect_wood", "stone": "collect_stone", "coal": "collect_coal",
    "iron": "collect_iron", "diamond": "collect_diamond",
    "sapphire": "collect_sapphire", "ruby": "collect_ruby",
    "sapling": "collect_sapling",
}
PICKAXE_LEVEL_TO_ACHIEVEMENT = {
    1: "make_wood_pickaxe", 2: "make_stone_pickaxe",
    3: "make_iron_pickaxe", 4: "make_diamond_pickaxe",
}
STATION_TO_ACHIEVEMENT = {"is_at_crafting_table": "place_table", "is_at_furnace": "place_furnace"}


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def _attr_chain(node: ast.AST) -> str:
    """'state.inventory.wood' style dotted name for an Attribute chain."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


# ---- A + B + F: game_logic.py -------------------------------------------------------------

def extract_from_game_logic(src: str) -> dict[str, set[str]]:
    tree = ast.parse(src)
    edges: dict[str, set[str]] = {}

    def add(a: str, p: str) -> None:
        edges.setdefault(a, set()).add(p)

    # index every assignment name -> list of value nodes (jax code reassigns freely)
    assigns: dict[str, list[ast.AST]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            assigns.setdefault(node.targets[0].id, []).append(node.value)

    def inventory_fields_ge(nodes: list[ast.AST]) -> set[str]:
        """inventory fields appearing in `<x>.inventory.<field> >= n` or
        `new_inventory.<field> >= n` comparisons anywhere under the given nodes."""
        fields: set[str] = set()
        for value in nodes:
            for cmp in ast.walk(value):
                if not (isinstance(cmp, ast.Compare) and len(cmp.ops) == 1
                        and isinstance(cmp.ops[0], (ast.GtE, ast.Gt))):
                    continue
                chain = _attr_chain(cmp.left)
                m = re.search(r"(?:^|\.)(?:new_)?inventory\.([a-z_]+)$", chain)
                if m:
                    fields.add(m.group(1))
        return fields

    def names_under(nodes: list[ast.AST]) -> set[str]:
        out: set[str] = set()
        for value in nodes:
            for n in ast.walk(value):
                if isinstance(n, ast.Name):
                    out.add(n.id)
        return out

    # A. recipes: can_craft_<item> (+ station gates from is_crafting_<item>)
    for name, values in assigns.items():
        m = re.fullmatch(r"can_craft_([a-z_]+)", name)
        if not m:
            continue
        item = m.group(1)
        ach = f"make_{item}"
        for field in inventory_fields_ge(values):
            if field in INGREDIENT_FIELD_TO_ACHIEVEMENT:
                add(ach, INGREDIENT_FIELD_TO_ACHIEVEMENT[field])
        gate_names = names_under(assigns.get(f"is_crafting_{item}", []))
        for station, sach in STATION_TO_ACHIEVEMENT.items():
            if station in gate_names:
                add(ach, sach)

    # B. mining: can_mine_<block> = inventory.pickaxe >= n
    for name, values in assigns.items():
        m = re.fullmatch(r"can_mine_([a-z_]+)", name)
        if not m:
            continue
        block = m.group(1)
        ach = INGREDIENT_FIELD_TO_ACHIEVEMENT.get(block)
        if not ach:
            continue  # tree/stalagmite etc. — no matching achievement
        for value in values:
            for cmp in ast.walk(value):
                if (isinstance(cmp, ast.Compare) and len(cmp.ops) == 1
                        and isinstance(cmp.ops[0], ast.GtE)
                        and _attr_chain(cmp.left).endswith("inventory.pickaxe")
                        and isinstance(cmp.comparators[0], ast.Constant)):
                    lvl = int(cmp.comparators[0].value)
                    if lvl in PICKAXE_LEVEL_TO_ACHIEVEMENT:
                        add(ach, PICKAXE_LEVEL_TO_ACHIEVEMENT[lvl])

    # F. placing: is_placing_<thing> consumes inventory fields; PLACE achievement nearby.
    #    Pattern: `<field>=... - k * is_placing_<thing>` + Achievement.PLACE_X set with it.
    place_consumes: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg in INGREDIENT_FIELD_TO_ACHIEVEMENT:
            names = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
            for nm in names:
                if nm.startswith("is_placing_"):
                    place_consumes.setdefault(nm, set()).add(node.arg)
    src_text = src
    for placer, fields in place_consumes.items():
        # find which PLACE_* achievement this placer sets (textual co-occurrence window)
        for m in re.finditer(r"Achievement\.(PLACE_[A-Z_]+)\.value", src_text):
            window = src_text[m.end(): m.end() + 200]
            if placer in window:
                ach = m.group(1).lower()
                for f in fields:
                    edges.setdefault(ach, set()).add(INGREDIENT_FIELD_TO_ACHIEVEMENT[f])

    # E1. chest loot: inventory fields incremented under is_opening_chest
    loot_fields: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and isinstance(node.value, ast.AST):
            names = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
            if "is_opening_chest" in names and node.arg:
                loot_fields.add(node.arg)
    for f in loot_fields & set(INGREDIENT_FIELD_TO_ACHIEVEMENT):
        add(INGREDIENT_FIELD_TO_ACHIEVEMENT[f], "open_chest")
    if "bow" in loot_fields:
        add("find_bow", "open_chest")
    if "potions" in loot_fields:
        add("drink_potion", "open_chest")
    if "books" in loot_fields:
        # E2. read_book: requires inventory.books, sets LEARN_FIREBALL/LEARN_ICEBALL
        for m in re.finditer(r"Achievement\.(LEARN_[A-Z]+)\.value", src_text):
            add(m.group(1).lower(), "open_chest")
        # E3. cast_spell: gated on learned_spells set by read_book
        for spell in ("fireball", "iceball"):
            if f"CAST_{spell.upper()}" in src_text:
                add(f"cast_{spell}", f"learn_{spell}")

    return edges


# ---- C + D: constants.py + renderer.py ----------------------------------------------------

def _literal_int_rows(call_or_list: ast.AST) -> list[list[int]]:
    """Rows of ints from nested jnp.array([...jnp.array([..]),..]) literals."""
    rows = []
    for node in ast.walk(call_or_list):
        if isinstance(node, ast.List) and node.elts \
                and all(isinstance(e, ast.Constant) and isinstance(e.value, (int, float))
                        for e in node.elts):
            rows.append([int(e.value) for e in node.elts])
    return rows


def extract_floor_ladder(constants_src: str) -> list[str]:
    """LEVEL_ACHIEVEMENT_MAP -> ordered enter_* names (index = floor)."""
    tree = ast.parse(constants_src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id == "LEVEL_ACHIEVEMENT_MAP":
            names = []
            for n in ast.walk(node.value):
                if isinstance(n, ast.Attribute) and n.attr == "value" \
                        and isinstance(n.value, ast.Attribute):
                    names.append(n.value.attr.lower())
            return names
    return []


def extract_mob_edges(constants_src: str,
                      floor_ladder: list[str]) -> dict[str, set[str]]:
    """FLOOR_MOB_MAPPING x texture-list index order x spawn chance -> defeat/eat edges.

    The load_mob_texture_set calls (index order = mob type id) live in constants.py's
    texture-loading function, alongside the mapping arrays."""
    # texture lists, in variable order: melee / passive / ranged
    lists: dict[str, list[str]] = {}
    tree = ast.parse(constants_src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple):
            tgt_names = [e.id for e in node.targets[0].elts if isinstance(e, ast.Name)]
            for kind in ("melee", "passive", "ranged"):
                if any(t == f"{kind}_mob_textures" for t in tgt_names):
                    pngs = [c.value for c in ast.walk(node.value)
                            if isinstance(c, ast.Constant) and isinstance(c.value, str)
                            and c.value.endswith(".png")]
                    lists[kind] = [p[:-4] for p in pngs]

    ctree = ast.parse(constants_src)
    mapping_rows: list[list[int]] = []
    chance_rows: list[list[float]] = []
    for node in ast.walk(ctree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "FLOOR_MOB_MAPPING":
                mapping_rows = _literal_int_rows(node.value)
            elif node.targets[0].id == "FLOOR_MOB_SPAWN_CHANCE":
                for n in ast.walk(node.value):
                    if isinstance(n, ast.List) and n.elts and all(
                            isinstance(e, ast.Constant) for e in n.elts):
                        chance_rows.append([float(e.value) for e in n.elts])

    # earliest floor per mob (spawn chance > 0), column order (passive, melee, ranged)
    earliest: dict[str, int] = {}
    cols = ("passive", "melee", "ranged")
    for floor, row in enumerate(mapping_rows):
        for ci, kind in enumerate(cols):
            if kind not in lists or ci >= len(row):
                continue
            chance = chance_rows[floor][ci] if floor < len(chance_rows) else 1.0
            if chance <= 0:
                continue
            idx = row[ci]
            if idx < len(lists[kind]):
                mob = lists[kind][idx]
                if mob not in earliest or floor < earliest[mob]:
                    earliest[mob] = floor

    edges: dict[str, set[str]] = {}
    for mob, floor in earliest.items():
        ach = MOB_NAME_TO_ACHIEVEMENT.get(mob)
        if not ach:
            continue
        if floor == 0:
            edges.setdefault(ach, set())  # overworld -> no prereq
        elif floor - 1 < len(floor_ladder):
            edges.setdefault(ach, set()).add(floor_ladder[floor - 1])
    return edges


# ---- main ---------------------------------------------------------------------------------

def extract_all(craftax_dir: Path) -> dict[str, set[str]]:
    gl = _read(craftax_dir / "game_logic.py")
    cn = _read(craftax_dir / "constants.py")

    edges = extract_from_game_logic(gl)
    ladder = extract_floor_ladder(cn)
    for i, ach in enumerate(ladder):
        if i == 0:
            edges.setdefault(ach, set())            # enter_dungeon: overworld, no prereq
        else:
            edges.setdefault(ach, set()).add(ladder[i - 1])
    for a, ps in extract_mob_edges(cn, ladder).items():
        edges.setdefault(a, set()).update(ps)
    return edges


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--craftax-src", required=True,
                    help="path to the craftax/craftax package dir (contains game_logic.py)")
    ap.add_argument("--diff", action="store_true",
                    help="diff against hand-curated DIRECT_PREREQS (needs PYTHONPATH=src:.)")
    args = ap.parse_args()

    extracted = extract_all(Path(args.craftax_src))

    print(f"== auto-extracted: {len(extracted)} achievements, "
          f"{sum(len(v) for v in extracted.values())} edges ==")
    for a in sorted(extracted):
        print(f"  {a:24s} <- {sorted(extracted[a]) or '[]'}")

    if not args.diff:
        return

    from dicode.skill_preflight.prereq_graph import DIRECT_PREREQS
    hand = {a: set(ps) for a, ps in DIRECT_PREREQS.items()}
    auto_covered, agree, extra_auto, missing_auto, uncovered = [], [], {}, {}, []
    for a, hps in hand.items():
        if a not in extracted:
            uncovered.append(a)
            continue
        auto_covered.append(a)
        aps = extracted[a]
        if aps == hps:
            agree.append(a)
        else:
            if aps - hps:
                extra_auto[a] = sorted(aps - hps)
            if hps - aps:
                missing_auto[a] = sorted(hps - aps)

    print(f"\n== diff vs hand-curated DIRECT_PREREQS ==")
    print(f"achievements auto-covered : {len(auto_covered)}/67")
    print(f"  exact edge-set agreement: {len(agree)}")
    print(f"  auto has EXTRA edges    : {len(extra_auto)}")
    for a, e in sorted(extra_auto.items()):
        print(f"    {a:24s} auto-extra: {e}")
    print(f"  auto MISSING hand edges : {len(missing_auto)}")
    for a, m in sorted(missing_auto.items()):
        print(f"    {a:24s} auto-missing: {m}")
    print(f"achievements NOT auto-covered ({len(uncovered)}):")
    for a in sorted(uncovered):
        print(f"    {a:24s} (hand: {sorted(hand[a]) or '[]'})")


if __name__ == "__main__":
    main()
