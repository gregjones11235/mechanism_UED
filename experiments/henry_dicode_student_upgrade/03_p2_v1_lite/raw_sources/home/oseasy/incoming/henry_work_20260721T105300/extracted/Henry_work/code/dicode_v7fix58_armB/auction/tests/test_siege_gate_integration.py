"""Integration: SiegeNotebook -> unmastered_links -> Completed gate, end to end (v6 §3.4/§3.5).

This exercises the exact chain gen_manager.evolve_mastered_coop runs each session (minus the LLM
calls): the modeler proposes a focus + prereq tree, the notebook folds it through its B-layer rules,
the still-unmastered links are computed from live SR, and the Completed gate rewrites a proposer
docstring so those links are trained, not compressed. No jax/craftax-env/LLM needed.
"""

from auction.completed_gate import (
    enforce_completed_gate,
    parse_completed_achievements,
)
from auction.completed_gate import _RELEVANT_RE, _parse_names
from auction.siege_notebook import MATURITY_MIN_SNAPSHOTS, SiegeNotebook, mastery_from_sr


def _mature(extra):
    from auction.siege_notebook import MATURITY_MIN_MASTERED, MATURITY_SKILL_SR
    prof = {f"solid_{i}": MATURITY_SKILL_SR + 10 for i in range(MATURITY_MIN_MASTERED)}
    prof.update(extra)
    return prof


def _relevant(doc):
    m = _RELEVANT_RE.search(doc)
    return set(_parse_names(m.group(2))) if m else set()


# A proposer docstring that ILLEGALLY compresses an unmastered survival link (place_torch) into
# Completed while sieging a gnome warrior on the pitch-black floor 2.
PROPOSER_DOC = """\
Objective: Defeat a gnome warrior on floor 2.
Description: Start on floor 2 (pitch black) with a diamond sword; kill the gnome warrior.
Relevant Achievements: DEFEAT_GNOME_WARRIOR
Completed Achievements: MAKE_DIAMOND_SWORD, COLLECT_DIAMOND, PLACE_TORCH
World:
- Player: floor 2, diamond sword in inventory.
"""


def test_full_chain_pulls_unmastered_link_into_relevant(tmp_path):
    nb = SiegeNotebook(str(tmp_path / "siege.json"))
    # Live held-out SR: student can make/collect diamond (mastered enablers) but CANNOT place torches
    # (dies in the dark) and cannot yet defeat the gnome warrior (the wall).
    profile = _mature({
        "defeat_gnome_warrior": 6.0,   # the wall
        "make_diamond_sword": 82.0,    # mastered enabler
        "collect_diamond": 80.0,       # mastered enabler
        "place_torch": 9.0,            # UNMASTERED survival link -> must be trained
        # v7fix4: the gnome's habitat entrances get AUTOFILLED into the chain — mastered here so
        # this test keeps pinning exactly one unmastered link (place_torch).
        "enter_dungeon": 90.0,
        "enter_gnomish_mines": 85.0,
    })
    # Modeler proposes the focus + its prereq chain (LLM side; code owns the mastery flags).
    modeler_proposal = {
        "focus": "defeat_gnome_warrior",
        "prereq_tree": [
            {"skill": "collect_diamond", "role": "reach diamonds on floor2/7"},
            {"skill": "make_diamond_sword", "role": "gear to hurt the warrior"},
            {"skill": "place_torch", "role": "survive the pitch-black floor2"},
        ],
    }
    nb.apply_llm_update(10, profile, modeler_proposal, num_snapshots=MATURITY_MIN_SNAPSHOTS)
    assert nb.focus == "defeat_gnome_warrior"

    # Live unmastered links = exactly the ones below MASTERED_SR (place_torch only here).
    unmastered = nb.unmastered_links(profile)
    assert unmastered == {"place_torch"}
    # sanity on the flag computation
    assert mastery_from_sr(profile["place_torch"]) == "UNMASTERED"
    assert mastery_from_sr(profile["make_diamond_sword"]) == "CONSOLIDATED"

    # The gate rewrites the proposer's docstring: place_torch out of Completed, into Relevant.
    fixed, moved = enforce_completed_gate(PROPOSER_DOC, unmastered)
    assert moved == ["place_torch"]
    assert "place_torch" not in parse_completed_achievements(fixed)
    assert "place_torch" in _relevant(fixed)
    # mastered enablers stay compressed (they DON'T sever the chain — the student can re-supply them)
    assert parse_completed_achievements(fixed) == {"make_diamond_sword", "collect_diamond"}
    # the wall itself is still the trained goal
    assert "defeat_gnome_warrior" in _relevant(fixed)


def test_full_chain_noop_when_no_focus(tmp_path):
    # Student too early / no focus set -> unmastered set empty -> gate is a strict no-op (baseline).
    nb = SiegeNotebook(str(tmp_path / "siege.json"))
    # too few snapshots => maturity gate refuses any focus
    nb.apply_llm_update(1, _mature({"defeat_gnome_warrior": 5.0}),
                        {"focus": "defeat_gnome_warrior"}, num_snapshots=1)
    assert nb.focus is None
    unmastered = nb.unmastered_links(_mature({"defeat_gnome_warrior": 5.0}))
    assert unmastered == set()
    fixed, moved = enforce_completed_gate(PROPOSER_DOC, unmastered)
    assert moved == [] and fixed == PROPOSER_DOC


def test_after_breakthrough_links_are_protected_and_no_longer_forbidden(tmp_path):
    # v6fix7 P1a (#8): a breakthrough is recorded as PROGRESS immediately, but protection requires a
    # VERIFIED CONQUEST — the wall must HOLD at mastered for conquest_consecutive (2) consecutive
    # snapshots. Then the focus retires gracefully (conquered ground is not a wall), target + links
    # enter the protected set, and the Completed gate forbids nothing.
    nb = SiegeNotebook(str(tmp_path / "siege.json"))
    profile = _mature({
        "defeat_gnome_warrior": 6.0, "collect_diamond": 80.0,
        "make_diamond_sword": 82.0, "place_torch": 9.0,
    })
    nb.apply_llm_update(10, profile, {
        "focus": "defeat_gnome_warrior",
        "prereq_tree": [
            {"skill": "collect_diamond", "role": "reach"},
            {"skill": "make_diamond_sword", "role": "gear"},
            {"skill": "place_torch", "role": "survive dark"},
        ],
    }, num_snapshots=MATURITY_MIN_SNAPSHOTS)
    # The student breaks through the wall AND has learned torches along the way.
    won = _mature({
        "defeat_gnome_warrior": 80.0, "collect_diamond": 80.0,
        "make_diamond_sword": 82.0, "place_torch": 78.0,
    })
    nb.apply_llm_update(11, won, None, num_snapshots=MATURITY_MIN_SNAPSHOTS)
    snap = nb.snapshot()
    # 1st mastered snapshot: recorded as PROGRESS; nothing protected yet; focus still active.
    assert nb.focus == "defeat_gnome_warrior"
    assert snap["verified_chains"] and snap["verified_chains"][0]["target"] == "defeat_gnome_warrior"
    assert snap["verified_chains"][0]["status"] == "progress"
    assert snap["protected_set"] == []
    # 2nd consecutive mastered snapshot: VERIFIED conquest.
    nb.apply_llm_update(12, won, None, num_snapshots=MATURITY_MIN_SNAPSHOTS)
    snap = nb.snapshot()
    assert nb.foci() == []  # conquered ground is not a wall — graceful retirement
    assert snap["verified_chains"][0]["status"] == "verified"
    for s in ("defeat_gnome_warrior", "collect_diamond", "make_diamond_sword", "place_torch"):
        assert s in snap["protected_set"]
    # no active foci -> the gate forbids nothing (links may be compressed as a conquered base).
    assert nb.unmastered_links(won) == set()
