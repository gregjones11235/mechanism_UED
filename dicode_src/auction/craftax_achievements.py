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

from collections.abc import Mapping

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


# --- Student per-tier mastery & "reachable ceiling" (ability-gate, 2026-07-02) --------------
#
# Shared by BOTH the ambition bid-gate (ambition.py: discount gap on unreachable tiers) and the
# prompt-gate (persona prompts: "don't target above the reachable tier"). Defining the ceiling in
# ONE place keeps the two gates logically consistent — a tier the prompt forbids is the same tier
# the bid discounts.
#
# Rationale (v1_experiment.md §10.9): ambitious systematically drifts the curriculum to tier-3
# before the student has consolidated tier-2, crowding out feasible's learnable levels. The gate
# anchors ambition to the student's ability: it may probe ONE tier beyond mastery (the learnable
# frontier) but gap on tiers further out is discounted / forbidden.

MASTERY_THRESHOLD_DEFAULT: float = 0.60  # a tier counts as "mastered" at >= this mean SR


def tier_mastery(
    target_gap: Mapping[str, float],
    *,
    missing_is_mastered: bool = False,
) -> dict[int, float]:
    """Mean *mastery* (1 - gap) per depth tier, from the AmbitionGain target_gap map.

    target_gap[a] = 1 - student_SR_on_target in [0,1] (see auction_integration.profile_to_target_gap).
    So mastery of achievement a = 1 - target_gap[a].

    ★ missing_is_mastered (default False): how to treat an achievement ABSENT from target_gap when
    averaging over the FULL tier membership. This is a real correctness fork (2026-07-02):
      - False (default, SAFE for the gate): absent => NOT measured => NOT mastered (gap 1.0,
        mastery 0.0). A tier the eval never reported reads as unmastered, so reachable_ceiling stays
        conservative and won't wave through an out-of-reach tier just because it's missing. This is
        the same "absent = (NOT YET), don't hide it" philosophy as the profile formatter.
      - True (legacy / lenient): absent => gap 0 => mastery 1.0. Only correct when the caller has
        already guaranteed every achievement is present (the real Craftax held-out eval emits all 67,
        including 0.00). Use only for that guaranteed-complete case.

    Returns {tier: mean_mastery in [0,1]} for tiers 1..4.
    """
    default_gap = 0.0 if missing_is_mastered else 1.0
    out: dict[int, float] = {}
    for tier, names in DEPTH_TIERS.items():
        total = 0.0
        for a in names:
            total += 1.0 - float(target_gap.get(a, default_gap))
        out[tier] = total / len(names) if names else 1.0
    return out


def reachable_ceiling(
    target_gap: Mapping[str, float],
    *,
    threshold: float = MASTERY_THRESHOLD_DEFAULT,
    missing_is_mastered: bool = False,
) -> int:
    """The deepest tier the student is allowed to be *pushed toward* right now.

    Walk tiers shallow->deep: the student may target up to (and including) the first tier whose
    ALL shallower tiers are mastered (mean mastery >= threshold). Concretely, the ceiling is the
    shallowest not-yet-mastered tier — the student's current learnable frontier — capped at the
    deepest tier. Everything strictly deeper than the ceiling is "unreachable" (gap discounted by
    the bid gate, forbidden by the prompt gate).

    Examples (threshold 0.70):
      t1=0.9,t2=0.4,...           -> ceiling 2  (t1 mastered, t2 is the frontier)
      t1=0.9,t2=0.8,t3=0.3,...    -> ceiling 3
      all >= 0.70                 -> ceiling 4  (may aim at the deepest tier)
      t1=0.5,...                  -> ceiling 1  (not even tier-1 solid; stay at tier-1)
    """
    mastery = tier_mastery(target_gap, missing_is_mastered=missing_is_mastered)
    max_tier = max(DEPTH_TIERS)
    for tier in sorted(DEPTH_TIERS):
        if mastery.get(tier, 0.0) < threshold:
            return tier
    return max_tier


def tier_overreach_factor(
    tier: int,
    ceiling: int,
    *,
    decay: float = 0.3,
) -> float:
    """Soft-decay multiplier applied to a bid signal for a tier vs the reachable ceiling.

    tier <= ceiling  -> 1.0 (within reach, no penalty).
    tier  > ceiling  -> decay ** (tier - ceiling)  (each tier beyond the ceiling shrinks the signal
                        geometrically). With decay=0.3: 1 tier over -> 0.30, 2 over -> 0.09.

    This is the "soft weakening" chosen over a hard zero (2026-07-02 user decision): ambition keeps
    a small, shrinking voice on deep tiers (it should still probe the frontier, per the
    competition-drives-niche selling point) but can no longer out-bid feasible's learnable levels
    on tiers the student cannot reach.
    """
    if tier <= ceiling:
        return 1.0
    return float(decay) ** (tier - ceiling)


# --- Skill families (for Proposer-Breadth coverage) ----------------------------------------
#
# Four coarse families (方法设计_v2.md §2.3, DiCode Figure 3). Used ONLY as the
# {ARCHIVE_FAMILY_COVERAGE} tally fed to the Breadth persona so it can target the
# under-represented family. NOT an authoritative craftax constant.
#
# Mapped by name PREFIX with a few explicit exceptions (some names don't fit a clean prefix):
#   COMBAT  : defeat_*, damage_*
#   GATHER  : collect_*, eat_*, drink_*
#   CRAFT   : make_*, place_*, enchant_*
#   EXPLORE : enter_*, find_*, open_*, cast_*, learn_*, fire_*, and survival misc (wake_up)
FAMILIES: tuple[str, ...] = ("COMBAT", "GATHER", "CRAFT", "EXPLORE")

_FAMILY_EXCEPTIONS: dict[str, str] = {
    "wake_up": "EXPLORE",      # survival misc — group with exploration/other
    "fire_bow": "EXPLORE",     # using the bow (ranged action), not crafting/combat-kill
    "find_bow": "EXPLORE",
    "drink_potion": "GATHER",  # consuming a resource
    "enchant_sword": "CRAFT",
    "enchant_armour": "CRAFT",
    "damage_necromancer": "COMBAT",
}


def family_of(achievement: str) -> str:
    """Coarse skill family (COMBAT/GATHER/CRAFT/EXPLORE) for an achievement name."""
    if achievement in _FAMILY_EXCEPTIONS:
        return _FAMILY_EXCEPTIONS[achievement]
    if achievement.startswith("defeat_") or achievement.startswith("damage_"):
        return "COMBAT"
    if achievement.startswith("collect_") or achievement.startswith("eat_") or achievement.startswith("drink_"):
        return "GATHER"
    if achievement.startswith("make_") or achievement.startswith("place_") or achievement.startswith("enchant_"):
        return "CRAFT"
    # enter_/find_/open_/cast_/learn_/fire_ -> exploration & magic & interaction
    return "EXPLORE"


ACHIEVEMENT_FAMILY: dict[str, str] = {name: family_of(name) for name in ALL_ACHIEVEMENTS}

# Sanity: every achievement maps to exactly one of the four families.
assert set(ACHIEVEMENT_FAMILY.values()) <= set(FAMILIES), (
    f"unexpected family label: {set(ACHIEVEMENT_FAMILY.values()) - set(FAMILIES)}"
)
assert len(ACHIEVEMENT_FAMILY) == NUM_ACHIEVEMENTS
