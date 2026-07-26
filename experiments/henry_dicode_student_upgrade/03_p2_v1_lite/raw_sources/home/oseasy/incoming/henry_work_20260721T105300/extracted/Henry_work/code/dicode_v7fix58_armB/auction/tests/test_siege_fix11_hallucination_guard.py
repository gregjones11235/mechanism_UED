"""Offline tests for the HALLUCINATION GUARD, v7fix1 port of v6fix11 (r2 smelt_iron post-mortem).

r2 (job 3752823): the LLM wrote ``smelt_iron`` — a name NOT in the 67-achievement table — into
make_iron_sword's prereq_tree. The ChainOrderLog could never map it, so it sat at missing ~100%;
the door gate's rank-0 "unknown SR = closed" semantics then AUTO-OPENED it as a focus at s21,
where it could never be measured (held-out None forever), never graduate or conquer, and burned
the full 8/8 enabler budget until BUDGET-RETIRE (s37). v7's first run showed the same disease in
chain links (troll: eat_cow / collect_drink filler). Guard (defense in depth, both layers):
  parser  (Modeler._validate_siege): unknown names DROPPED from foci / prereq_tree / ranked_walls /
          key_missing_link, reported through the attrib_violations re-prompt channel;
  notebook (SiegeNotebook): _is_valid_focus requires table membership; _door_substitute skips
          non-member candidates at ANY rank; auto-open validates BEFORE the pending-track park.

No jax/craftax/LLM needed.
"""

import pytest

from auction.craftax_achievements import ACHIEVEMENT_TO_VALUE
from auction.modeler import MODELER_SIEGE_SYSTEM_PROMPT, Modeler
from auction.siege_notebook import (
    DOOR_MIN_SR,
    MATURITY_MIN_MASTERED,
    MATURITY_MIN_SNAPSHOTS,
    MATURITY_SKILL_SR,
    SiegeNotebook,
)

FAKE = "smelt_iron"  # the exact r2 hallucination — must stay outside the table


def _mature_profile(extra: dict | None = None) -> dict:
    prof = {f"filler_{i}": MATURITY_SKILL_SR + 10 for i in range(MATURITY_MIN_MASTERED)}
    if extra:
        prof.update(extra)
    return prof


@pytest.fixture
def nb_path(tmp_path):
    return str(tmp_path / "siege_notebook.json")


def _update(nb, session, profile, foci=None, ranked=None, forensics=None, incomplete=None):
    proposal = {}
    if foci is not None:
        proposal["foci"] = foci
    if ranked is not None:
        proposal["ranked_walls"] = ranked
    return nb.apply_llm_update(
        session, profile, proposal, num_snapshots=MATURITY_MIN_SNAPSHOTS,
        forensics=forensics, chain_incomplete=incomplete,
    )


def test_smelt_iron_is_really_not_an_achievement():
    assert FAKE not in ACHIEVEMENT_TO_VALUE
    assert len(ACHIEVEMENT_TO_VALUE) == 67


# ---- notebook layer: door gate -------------------------------------------------------------------

def test_door_substitute_skips_hallucinated_rank0(nb_path):
    """The exact r2 shape: hallucinated modal missing link at rank 0, real closed link at rank 1.
    The scan must skip the fake and substitute the REAL door instead."""
    nb = SiegeNotebook(nb_path)
    prof = _mature_profile({"make_iron_sword": 0.0, "collect_iron": 2.0})
    fx = {"missing_top": [(FAKE, 0.9), ("collect_iron", 0.5)]}
    _update(nb, 1, prof, foci=[{"skill": "make_iron_sword", "prereq_tree": []}],
            forensics={"make_iron_sword": fx})
    assert nb.focus_skills() == ["collect_iron"]
    assert nb.foci()[0]["gateway_for"] == "make_iron_sword"


def test_door_substitute_all_hallucinated_opens_wall_itself(nb_path):
    """Every missing link hallucinated -> no door exists -> the wall opens as its own focus
    (the r2 counterfactual: make_iron_sword under direct DEPTH pressure was breaking through)."""
    nb = SiegeNotebook(nb_path)
    prof = _mature_profile({"make_iron_sword": 0.0})
    fx = {"missing_top": [(FAKE, 0.9), ("craft_ingot", 0.4)]}
    _update(nb, 1, prof, foci=[{"skill": "make_iron_sword", "prereq_tree": []}],
            forensics={"make_iron_sword": fx})
    assert nb.focus_skills() == ["make_iron_sword"]
    assert nb.foci()[0]["gateway_for"] is None


# ---- notebook layer: focus validity --------------------------------------------------------------

def test_hallucinated_focus_rejected_even_with_forensics(nb_path):
    nb = SiegeNotebook(nb_path)
    prof = _mature_profile({})
    _update(nb, 1, prof, foci=[{"skill": FAKE, "prereq_tree": []}],
            forensics={FAKE: {"missing_top": []}})
    assert nb.focus_skills() == []
    assert f"scope_rejected({FAKE}" in nb.last_focus_decision


def test_hallucinated_relay_proposal_rejected_too(nb_path):
    """v7-specific: the relay path must not resurrect a hallucinated wall — _is_valid_focus rules
    before the relay branch."""
    nb = SiegeNotebook(nb_path)
    prof = _mature_profile({})
    _update(nb, 1, prof, foci=[{"skill": FAKE, "prereq_tree": [], "relay_r0_floor": 3}],
            forensics={})
    assert nb.focus_skills() == []
    assert f"scope_rejected({FAKE}" in nb.last_focus_decision


def test_auto_open_never_parks_or_opens_hallucinated_ranked_wall(nb_path):
    """Ordering fix: validity rules BEFORE the hazard-3a pending-track park, so a fake ranked wall
    neither opens NOR pollutes chain tracking; the next real candidate wins the slot."""
    nb = SiegeNotebook(nb_path)
    prof = _mature_profile({"defeat_gnome_warrior": 0.0, "enter_dungeon": DOOR_MIN_SR + 5})
    fx = {"missing_top": [("enter_dungeon", 0.7)]}
    _update(nb, 1, prof, foci=[], ranked=[{"skill": FAKE}, {"skill": "defeat_gnome_warrior"}],
            forensics={"defeat_gnome_warrior": fx})
    assert nb.focus_skills() == ["defeat_gnome_warrior"]
    assert FAKE not in (nb._nb.get("pending_track") or {})


# ---- notebook layer: ⑦ rejection message teaches the closed vocabulary ---------------------------

def test_chain_incomplete_message_names_the_vocabulary(nb_path):
    nb = SiegeNotebook(nb_path)
    prof = _mature_profile({"defeat_gnome_warrior": 0.0})
    _update(nb, 1, prof, foci=[{"skill": "defeat_gnome_warrior", "prereq_tree": []}],
            forensics={"defeat_gnome_warrior": {"missing_top": []}}, incomplete={"defeat_gnome_warrior"})
    assert "chain_incomplete(defeat_gnome_warrior" in nb.last_focus_decision
    assert "EXACT achievement names" in nb.last_focus_decision


# ---- parser layer: _validate_siege drops + reports ------------------------------------------------

def _parse(su, forensics=None):
    return Modeler._validate_siege({"siege_update": su}, forensics=forensics)


def test_parser_drops_hallucinated_focus_and_reports():
    out = _parse({"foci": [
        {"skill": FAKE, "prereq_tree": []},
        {"skill": "defeat_gnome_warrior", "prereq_tree": []},
    ]})
    assert [f["skill"] for f in out["foci"]] == ["defeat_gnome_warrior"]
    viols = " ".join(out["attrib_violations"])
    assert FAKE in viols and "achievement name" in viols


def test_parser_drops_hallucinated_tree_links_keeps_real_ones():
    out = _parse({"foci": [{"skill": "make_iron_sword", "prereq_tree": [
        {"skill": FAKE, "role": "smelt"},
        {"skill": "collect_iron", "role": "ore"},
    ]}]})
    links = [l["skill"] for l in out["foci"][0]["prereq_tree"]]
    assert links == ["collect_iron"]
    assert FAKE in " ".join(out["attrib_violations"])


def test_parser_drops_hallucinated_ranked_wall():
    out = _parse({"foci": [], "ranked_walls": [{"skill": FAKE}, {"skill": "defeat_gnome_warrior"}]})
    assert [r["skill"] for r in out["ranked_walls"]] == ["defeat_gnome_warrior"]


def test_parser_drops_hallucinated_legacy_focus():
    out = _parse({"focus": FAKE, "prereq_tree": []})
    assert out["foci"] == []
    assert FAKE in " ".join(out["attrib_violations"])


def test_key_missing_link_membership_beats_chain_incomplete_carveout():
    """The exact channel smelt_iron entered through: on a ⑦ chain-incomplete wall the containment
    check is waived (naming something NOT in the histogram is the point) — membership must still
    gate. The fake key is nulled + reported; the claim survives as chain_unreached without a key."""
    fx = {"defeat_gnome_warrior": {"chain_incomplete": True, "missing_top": [], "n_fail": 30}}
    out = _parse({"foci": [{
        "skill": "defeat_gnome_warrior", "prereq_tree": [],
        "failure_attribution": {"class": "chain_unreached", "key_missing_link": FAKE},
    }]}, forensics=fx)
    att = out["foci"][0]["failure_attribution"]
    assert att["key_missing_link"] is None
    assert FAKE in " ".join(out["attrib_violations"])


def test_key_missing_link_checked_even_without_forensics():
    out = _parse({"foci": [{
        "skill": "defeat_gnome_warrior", "prereq_tree": [],
        "failure_attribution": {"class": "chain_unreached", "key_missing_link": FAKE},
    }]}, forensics=None)
    assert out["foci"][0]["failure_attribution"]["key_missing_link"] is None
    assert FAKE in " ".join(out["attrib_violations"])


def test_real_key_missing_link_still_passes():
    out = _parse({"foci": [{
        "skill": "defeat_gnome_warrior", "prereq_tree": [],
        "failure_attribution": {"class": "chain_unreached", "key_missing_link": "enter_dungeon"},
    }]}, forensics=None)
    assert out["foci"][0]["failure_attribution"]["key_missing_link"] == "enter_dungeon"
    assert out["attrib_violations"] == []


# ---- prompt: the closed-vocabulary rule is actually in the shipped prompt -------------------------

def test_prompt_states_closed_vocabulary():
    assert "CLOSED VOCABULARY" in MODELER_SIEGE_SYSTEM_PROMPT
