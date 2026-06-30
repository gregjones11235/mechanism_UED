"""Craftax (full) achievement ground-truth set — authoritative constant for the auction layer.

Source of truth: ``craftax.craftax.constants.Achievement`` (the FULL Craftax env, not classic).
Verified against both ``main`` and tag ``v1.4.5`` of MichaelTMatthews/Craftax — both have
**67 members, value 0..66, identical names** (2026-06-30). DiCode pins ``craftax==1.4.5``
(``uv.lock``), so this is the exact set behind:
  - the SOTA 48.33 evaluation (``evaluation/skill_*`` in wandb, see v1_experiment.md §2.1),
  - DiCode's late-game selling point (deep achievements baselines collapse to 0%).

The wandb ``skill_<name>`` keys map to enum members via ``Achievement[name.upper()]``
(DiCode ``online_evaluation.py:229``), i.e. ``skill_collect_wood`` <-> ``COLLECT_WOOD``.
So our ``Proposal.achievements`` field (lowercase enum names) joins directly with the eval log.

This module has ZERO external dependencies (no craftax import) so the auction layer can be
developed/tested offline. If craftax ever bumps the enum, re-run the verification and update here.
"""

from __future__ import annotations

# (name, value) in enum order. Names are the lowercase enum members = the wandb skill_* suffix.
_ACHIEVEMENTS_ORDERED: tuple[tuple[str, int], ...] = (
    ("collect_wood", 0),
    ("place_table", 1),
    ("eat_cow", 2),
    ("collect_sapling", 3),
    ("collect_drink", 4),
    ("make_wood_pickaxe", 5),
    ("make_wood_sword", 6),
    ("place_plant", 7),
    ("defeat_zombie", 8),
    ("collect_stone", 9),
    ("place_stone", 10),
    ("eat_plant", 11),
    ("defeat_skeleton", 12),
    ("make_stone_pickaxe", 13),
    ("make_stone_sword", 14),
    ("wake_up", 15),
    ("place_furnace", 16),
    ("collect_coal", 17),
    ("collect_iron", 18),
    ("collect_diamond", 19),
    ("make_iron_pickaxe", 20),
    ("make_iron_sword", 21),
    ("make_arrow", 22),
    ("make_torch", 23),
    ("place_torch", 24),
    ("make_diamond_sword", 25),
    ("make_iron_armour", 26),
    ("make_diamond_armour", 27),
    ("enter_gnomish_mines", 28),
    ("enter_dungeon", 29),
    ("enter_sewers", 30),
    ("enter_vault", 31),
    ("enter_troll_mines", 32),
    ("enter_fire_realm", 33),
    ("enter_ice_realm", 34),
    ("enter_graveyard", 35),
    ("defeat_gnome_warrior", 36),
    ("defeat_gnome_archer", 37),
    ("defeat_orc_solider", 38),  # NOTE: misspelled "solider" in upstream craftax enum — kept verbatim.
    ("defeat_orc_mage", 39),
    ("defeat_lizard", 40),
    ("defeat_kobold", 41),
    ("defeat_troll", 42),
    ("defeat_deep_thing", 43),
    ("defeat_pigman", 44),
    ("defeat_fire_elemental", 45),
    ("defeat_frost_troll", 46),
    ("defeat_ice_elemental", 47),
    ("damage_necromancer", 48),
    ("defeat_necromancer", 49),
    ("eat_bat", 50),
    ("eat_snail", 51),
    ("find_bow", 52),
    ("fire_bow", 53),
    ("collect_sapphire", 54),
    ("learn_fireball", 55),
    ("cast_fireball", 56),
    ("learn_iceball", 57),
    ("cast_iceball", 58),
    ("collect_ruby", 59),
    ("make_diamond_pickaxe", 60),
    ("open_chest", 61),
    ("drink_potion", 62),
    ("enchant_sword", 63),
    ("enchant_armour", 64),
    ("defeat_knight", 65),
    ("defeat_archer", 66),
)

# Canonical name -> value and value -> name maps.
ACHIEVEMENT_TO_VALUE: dict[str, int] = {name: val for name, val in _ACHIEVEMENTS_ORDERED}
VALUE_TO_ACHIEVEMENT: dict[int, str] = {val: name for name, val in _ACHIEVEMENTS_ORDERED}

# The full ground-truth universe (frozenset of lowercase names). This is the legal value
# space for Proposal.achievements and the universe over which Coverage submodularity is proven.
ALL_ACHIEVEMENTS: frozenset[str] = frozenset(ACHIEVEMENT_TO_VALUE)

NUM_ACHIEVEMENTS: int = len(_ACHIEVEMENTS_ORDERED)  # == 67

assert NUM_ACHIEVEMENTS == 67, f"expected 67 craftax achievements, got {NUM_ACHIEVEMENTS}"
assert max(ACHIEVEMENT_TO_VALUE.values()) == 66
assert len(ALL_ACHIEVEMENTS) == 67  # no duplicate names

# --- Coarse depth tiers (for AmbitionGain "dependency-chain depth" axis) ------------------
# These are a *coarse, hand-curated* progression ordering derived from Craftax's tech tree,
# NOT an authoritative craftax constant. Used only as the depth signal for AmbitionGain
# (v1_experiment.md §7.5 / AmbitionGain = chain depth x target gap). Deeper tier == later-game,
# == where DiCode's selling point lives (baselines collapse to 0% on tiers 3-4).
# Tier assignment is a design choice and may be revised; keep it conservative and documented.
DEPTH_TIERS: dict[int, frozenset[str]] = {
    1: frozenset({  # early game: wood/stone basics, easy mobs, survival
        "collect_wood", "place_table", "eat_cow", "collect_sapling", "collect_drink",
        "make_wood_pickaxe", "make_wood_sword", "place_plant", "defeat_zombie",
        "collect_stone", "place_stone", "eat_plant", "defeat_skeleton",
        "make_stone_pickaxe", "make_stone_sword", "wake_up",
    }),
    2: frozenset({  # mid game: furnace/iron/coal, torches, first dungeon descent
        "place_furnace", "collect_coal", "collect_iron", "make_iron_pickaxe",
        "make_iron_sword", "make_arrow", "make_torch", "place_torch", "make_iron_armour",
        "enter_gnomish_mines", "enter_dungeon", "find_bow", "fire_bow",
        "eat_bat", "eat_snail", "open_chest",
    }),
    3: frozenset({  # late game: diamond gear, deeper floors, gnome/orc combat, magic
        "collect_diamond", "make_diamond_sword", "make_diamond_pickaxe", "make_diamond_armour",
        "enter_sewers", "enter_vault", "enter_troll_mines",
        "defeat_gnome_warrior", "defeat_gnome_archer", "defeat_orc_solider", "defeat_orc_mage",
        "defeat_lizard", "defeat_kobold", "defeat_troll",
        "collect_sapphire", "collect_ruby",
        "learn_fireball", "cast_fireball", "learn_iceball", "cast_iceball",
        "drink_potion", "enchant_sword", "enchant_armour",
    }),
    4: frozenset({  # deepest game: fire/ice realms, graveyard, elementals, necromancer, knights
        "enter_fire_realm", "enter_ice_realm", "enter_graveyard",
        "defeat_deep_thing", "defeat_pigman", "defeat_fire_elemental", "defeat_frost_troll",
        "defeat_ice_elemental", "damage_necromancer", "defeat_necromancer",
        "defeat_knight", "defeat_archer",
    }),
}

# Sanity: tiers partition the full set exactly (every achievement in exactly one tier).
_tier_union = frozenset().union(*DEPTH_TIERS.values())
assert _tier_union == ALL_ACHIEVEMENTS, (
    f"depth tiers must partition all achievements; "
    f"missing={ALL_ACHIEVEMENTS - _tier_union}, extra={_tier_union - ALL_ACHIEVEMENTS}"
)
assert sum(len(v) for v in DEPTH_TIERS.values()) == NUM_ACHIEVEMENTS, "tiers overlap"

ACHIEVEMENT_DEPTH: dict[str, int] = {
    name: tier for tier, names in DEPTH_TIERS.items() for name in names
}


def depth_of(achievement: str) -> int:
    """Coarse tech-tree depth tier (1=early .. 4=deepest) for an achievement name."""
    return ACHIEVEMENT_DEPTH[achievement]
