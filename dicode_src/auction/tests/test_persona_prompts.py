"""Offline tests for the v2 persona-differentiated proposers + skill families.

No jax/craftax/LLM needed. Covers:
  1. Skill-family mapping (family_of) partitions all 67 achievements into COMBAT/GATHER/CRAFT/EXPLORE.
  2. The three persona prompt modules load, .format() cleanly (no missing/extra placeholders), and
     keep DiCode's hardcore blocks; ARCHIVE_FAMILY_COVERAGE appears ONLY in Breadth's user_prompt.
  3. The archive family-coverage tally logic (reimplemented against a fake archive graph, matching
     gen_manager._compute_archive_family_coverage) counts each family once per level that touches it.

Persona modules are loaded directly from file (bypassing dicode/__init__'s heavy deps), same trick
as test_evolve_auction_dataflow.py.
"""

import importlib.util
from collections import Counter
from pathlib import Path

from auction.craftax_achievements import (
    ALL_ACHIEVEMENTS,
    FAMILIES,
    NUM_ACHIEVEMENTS,
    family_of,
    ACHIEVEMENT_FAMILY,
)

_PROMPTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "src" / "dicode" / "dreaming" / "prompts" / "dicode"
)


def _load(module_file):
    path = _PROMPTS_DIR / module_file
    spec = importlib.util.spec_from_file_location(module_file[:-3], path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_SYS_FIELDS = dict(CONSTANTS="C", MOBS="M", GAME_MECHANICS="G", WORLD_GEN="W", API_DOCS="A")
_USER_FIELDS = dict(
    MASTERED_TASK="mt",
    TASK_PERFORMANCE_CONTEXT="tp",
    GLOBAL_AGENT_PROFILE="gp",
    ARCHIVE_FAMILY_COVERAGE="afc",
)
_PERSONAS = {
    "ambitious": "persona_ambitious.py",
    "feasible": "persona_feasible.py",
    "breadth": "persona_breadth.py",
}
_HARDCORE_BLOCKS = [
    "KNOWLEDGE BASE",
    "<reasoning>",
    "<docstring>",
    "Relevant Achievements",
    "Completed Achievements",
    "CRITICAL RULE",
    "SPECIFICITY REQUIREMENT",
]


# --- 1. family mapping -------------------------------------------------------------------------

def test_families_partition_all_67():
    assert len(ACHIEVEMENT_FAMILY) == NUM_ACHIEVEMENTS == 67
    assert set(ACHIEVEMENT_FAMILY.values()) <= set(FAMILIES)
    # every achievement mapped
    assert set(ACHIEVEMENT_FAMILY) == set(ALL_ACHIEVEMENTS)
    counts = Counter(ACHIEVEMENT_FAMILY.values())
    assert sum(counts.values()) == 67


def test_family_spot_checks():
    cases = {
        "defeat_zombie": "COMBAT",
        "damage_necromancer": "COMBAT",
        "collect_wood": "GATHER",
        "eat_cow": "GATHER",
        "drink_potion": "GATHER",
        "make_torch": "CRAFT",
        "place_furnace": "CRAFT",
        "enchant_sword": "CRAFT",
        "enter_dungeon": "EXPLORE",
        "cast_fireball": "EXPLORE",
        "wake_up": "EXPLORE",
        "fire_bow": "EXPLORE",
    }
    for ach, exp in cases.items():
        assert family_of(ach) == exp, f"{ach}: expected {exp}, got {family_of(ach)}"


# --- 2. persona prompt modules ------------------------------------------------------------------

def test_persona_system_prompts_format_cleanly():
    for name, f in _PERSONAS.items():
        m = _load(f)
        s = m.system_prompt.format(**_SYS_FIELDS)  # raises KeyError if a field is missing
        # no leftover single-brace placeholders
        assert "{" not in s and "}" not in s, f"{name} system has leftover placeholder"


def test_persona_system_prompts_keep_hardcore_blocks():
    for name, f in _PERSONAS.items():
        s = _load(f).system_prompt.format(**_SYS_FIELDS)
        for block in _HARDCORE_BLOCKS:
            assert block in s, f"{name} system prompt lost hardcore block: {block!r}"


def test_persona_user_prompts_format_cleanly():
    for name, f in _PERSONAS.items():
        u = _load(f).user_prompt.format(**_USER_FIELDS)
        assert "{" not in u and "}" not in u, f"{name} user has leftover placeholder"


def test_archive_field_only_in_breadth():
    assert "ARCHIVE_FAMILY_COVERAGE" in _load(_PERSONAS["breadth"]).user_prompt
    assert "ARCHIVE_FAMILY_COVERAGE" not in _load(_PERSONAS["ambitious"]).user_prompt
    assert "ARCHIVE_FAMILY_COVERAGE" not in _load(_PERSONAS["feasible"]).user_prompt


def test_personas_are_distinct():
    # The three system prompts must actually differ (persona differentiation, not copies).
    systems = {name: _load(f).system_prompt for name, f in _PERSONAS.items()}
    assert systems["ambitious"] != systems["feasible"]
    assert systems["ambitious"] != systems["breadth"]
    assert systems["feasible"] != systems["breadth"]
    # each carries its distinctive intent keyword
    assert "AMBITIOUS" in systems["ambitious"]
    assert "FEASIBILITY-FIRST" in systems["feasible"]
    assert "BREADTH" in systems["breadth"]


# --- 3. archive family-coverage tally -----------------------------------------------------------

# Reimplementation of gen_manager._compute_archive_family_coverage against a plain dict-of-nodes,
# so we test the counting logic without importing gen_manager (jax/craftax-heavy).

_RELEVANT_RE = None


def _parse_relevant(docstring):
    # Load the adapter's parser directly (bypass dicode/__init__).
    global _RELEVANT_RE
    ai_path = (
        Path(__file__).resolve().parents[2]
        / "src" / "dicode" / "dreaming" / "auction_integration.py"
    )
    spec = importlib.util.spec_from_file_location("auction_integration_t", ai_path)
    ai = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ai)
    return ai.parse_relevant_achievements(docstring)


def _family_coverage(nodes):
    """nodes: list of {'description': docstring}. Mirror of the gen_manager method."""
    counts = {fam: 0 for fam in FAMILIES}
    for data in nodes:
        achs = _parse_relevant(data.get("description", "") or "")
        if not achs:
            continue
        for fam in {family_of(a) for a in achs}:
            counts[fam] += 1
    return " | ".join(f"{fam}: {counts[fam]}" for fam in FAMILIES)


def _doc(*achs_upper):
    return f"x\nRelevant Achievements: {', '.join(achs_upper)}\nWorld:\n- p"


def test_family_coverage_empty_archive_all_zero():
    assert _family_coverage([]) == "COMBAT: 0 | GATHER: 0 | CRAFT: 0 | EXPLORE: 0"


def test_family_coverage_counts_each_family_once_per_level():
    nodes = [
        {"description": _doc("DEFEAT_ZOMBIE", "DEFEAT_SKELETON")},   # COMBAT (once, 2 combat achs)
        {"description": _doc("COLLECT_WOOD")},                        # GATHER
        {"description": _doc("MAKE_TORCH", "DEFEAT_ZOMBIE")},         # CRAFT + COMBAT (mixed level)
        {"description": "no achievements here"},                      # ignored
    ]
    out = _family_coverage(nodes)
    # COMBAT touched by levels 1 & 3 = 2 ; GATHER by 2 = 1 ; CRAFT by 3 = 1 ; EXPLORE = 0
    assert out == "COMBAT: 2 | GATHER: 1 | CRAFT: 1 | EXPLORE: 0", out
