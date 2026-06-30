"""Tests for the DiCode<->auction adapter (docstring parsing + proposal conversion).

Uses the real docstring format from DiCode's own prompt (paper p85).
"""

import importlib.util
from pathlib import Path

# Load auction_integration directly by file path, bypassing dicode/__init__.py (which imports
# dotenv and other heavy DiCode deps not needed — or installed — for the offline adapter test).
_MOD_PATH = (
    Path(__file__).resolve().parents[2]
    / "src" / "dicode" / "dreaming" / "auction_integration.py"
)
_spec = importlib.util.spec_from_file_location("auction_integration", _MOD_PATH)
_ai = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ai)

parse_relevant_achievements = _ai.parse_relevant_achievements
parsed_response_to_proposal = _ai.parsed_response_to_proposal

# Verbatim shape of a generated docstring (paper p85).
_REAL_DOCSTRING = """Floor 2 contains 2 Gnome Warriors within Manhattan distance 3-6 of the ladder.
The task eliminates floor 0 backtracking while introducing the critical combat
dependency the agent neglects in the parent task.
Relevant Achievements: ENTER_GNOMISH_MINES, DEFEAT_GNOME_WARRIOR
Completed Achievements: MAKE_STONE_PICKAXE, MAKE_STONE_SWORD, COLLECT_WOOD,
COLLECT_STONE, COLLECT_COAL, PLACE_TABLE, PLACE_FURNACE
World:
- Player: Starts on floor 1
"""


def test_parses_relevant_only_not_completed():
    achs = parse_relevant_achievements(_REAL_DOCSTRING)
    assert achs == frozenset({"enter_gnomish_mines", "defeat_gnome_warrior"})
    # Completed Achievements (prerequisites) must NOT be counted as coverage
    assert "make_stone_pickaxe" not in achs
    assert "collect_wood" not in achs


def test_uppercase_mapped_to_lowercase():
    achs = parse_relevant_achievements("Relevant Achievements: COLLECT_DIAMOND\nWorld:")
    assert achs == frozenset({"collect_diamond"})


def test_missing_section_returns_empty():
    assert parse_relevant_achievements("Just prose, no achievements here.") == frozenset()
    assert parse_relevant_achievements("") == frozenset()


def test_unknown_tokens_dropped():
    achs = parse_relevant_achievements(
        "Relevant Achievements: COLLECT_WOOD, FLY_TO_MARS, defeat_archer\nWorld:"
    )
    # FLY_TO_MARS is not a real achievement -> dropped; the two valid ones kept
    assert achs == frozenset({"collect_wood", "defeat_archer"})


def test_wrapped_list_across_lines():
    doc = "Relevant Achievements: COLLECT_WOOD,\nCOLLECT_STONE, MAKE_WOOD_PICKAXE\nWorld:"
    achs = parse_relevant_achievements(doc)
    assert achs == frozenset({"collect_wood", "collect_stone", "make_wood_pickaxe"})


def test_parsed_response_to_proposal():
    parsed = {"description": _REAL_DOCSTRING, "reasoning": "because combat"}
    p = parsed_response_to_proposal(
        parsed, proposal_id="task_42", proposer_id="qwen", parent_task_id="task_5"
    )
    assert p.proposal_id == "task_42"
    assert p.proposer_id == "qwen"
    assert p.parent_task_id == "task_5"
    assert p.achievements == frozenset({"enter_gnomish_mines", "defeat_gnome_warrior"})
    assert p.reasoning == "because combat"


def test_proposal_with_no_achievements_is_legal():
    parsed = {"description": "prose only", "reasoning": "r"}
    p = parsed_response_to_proposal(
        parsed, proposal_id="t", proposer_id="m", parent_task_id="p"
    )
    assert p.achievements == frozenset()
