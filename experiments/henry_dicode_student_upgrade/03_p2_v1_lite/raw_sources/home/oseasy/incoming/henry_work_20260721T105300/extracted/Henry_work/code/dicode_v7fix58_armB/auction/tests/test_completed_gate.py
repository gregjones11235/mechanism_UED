"""Offline tests for the §3.4 Completed-admission gate (v6).

Pins the invariant: an UNMASTERED focus-link may NOT stay compressed in ``Completed`` — the gate
pulls it into ``Relevant`` (so it is trained) and rewrites the docstring. Mastered links in Completed
are left alone; no focus (empty unmastered) is a strict no-op that cannot disturb the baseline.
"""

from auction.completed_gate import (
    enforce_completed_gate,
    parse_completed_achievements,
)
from auction.craftax_achievements import ALL_ACHIEVEMENTS


DOC = """\
Objective: Learn to defeat a gnome warrior on floor 2.
Description: The agent starts on floor 2 with a diamond sword.
Relevant Achievements: DEFEAT_GNOME_WARRIOR
Completed Achievements: MAKE_DIAMOND_SWORD, PLACE_TORCH, COLLECT_DIAMOND
World:
- Player: floor 2 with diamond sword.
"""


def test_parse_completed():
    got = parse_completed_achievements(DOC)
    assert got == frozenset({"make_diamond_sword", "place_torch", "collect_diamond"})


def test_parse_completed_absent():
    assert parse_completed_achievements("Relevant Achievements: DEFEAT_TROLL\n") == frozenset()


def test_gate_moves_unmastered_link_to_relevant():
    # place_torch is an unmastered survival link -> must be pulled out of Completed into Relevant.
    new, moved = enforce_completed_gate(DOC, {"place_torch"})
    assert moved == ["place_torch"]
    # Completed no longer contains place_torch; Relevant now does.
    assert "place_torch" not in parse_completed_achievements(new)
    from auction.completed_gate import _RELEVANT_RE, _parse_names
    rel = _parse_names(_RELEVANT_RE.search(new).group(2))
    assert "place_torch" in rel
    assert "defeat_gnome_warrior" in rel  # original relevant preserved
    # mastered links stay compressed
    assert "make_diamond_sword" in parse_completed_achievements(new)
    assert "collect_diamond" in parse_completed_achievements(new)


def test_gate_moves_multiple():
    new, moved = enforce_completed_gate(DOC, {"place_torch", "collect_diamond"})
    assert set(moved) == {"place_torch", "collect_diamond"}
    comp = parse_completed_achievements(new)
    assert comp == frozenset({"make_diamond_sword"})


def test_gate_noop_when_no_unmastered():
    # No active siege focus -> empty unmastered set -> strict no-op (baseline undisturbed).
    new, moved = enforce_completed_gate(DOC, set())
    assert moved == []
    assert new == DOC


def test_gate_noop_when_unmastered_not_in_completed():
    # An unmastered skill that the proposer did NOT compress is not our concern here.
    new, moved = enforce_completed_gate(DOC, {"defeat_orc_mage"})
    assert moved == []
    assert new == DOC


def test_gate_only_moves_mastered_stays():
    # Mix: one unmastered (move), the rest mastered (stay).
    new, moved = enforce_completed_gate(DOC, {"place_torch", "defeat_orc_mage", "eat_bat"})
    assert moved == ["place_torch"]  # only the one actually in Completed AND unmastered


def test_gate_preserves_rest_of_docstring():
    new, _ = enforce_completed_gate(DOC, {"place_torch"})
    assert "Objective: Learn to defeat a gnome warrior" in new
    assert "World:" in new
    assert "- Player: floor 2 with diamond sword." in new


def test_gate_all_names_valid_after_rewrite():
    new, _ = enforce_completed_gate(DOC, {"place_torch"})
    # every token we emitted is a real achievement (no corruption)
    for section in (parse_completed_achievements(new),):
        assert section <= ALL_ACHIEVEMENTS
