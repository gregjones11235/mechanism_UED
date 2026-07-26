"""v7fix4.5: attribution-driven succession + expand-gate relay exclusion.

fix4.4-run post-mortem (job 3936082 @s114, 2026-07-13): the kobold relay ran at FULL dose with an
honest anchor and still crawled at 9-14% — and the modeler's own VERIFIED attribution had already
named why (chain_unreached, key_missing_link=enchant_sword; its tactic added learn/cast_fireball,
which relay episodes were casting 44% incidentally). But every road to ACT on the diagnosis was
locked: equipment/magic walls are deep-locked (relay-only) behind the busy single relay slot, and
the ordinary expand gate ("any focus >= 50%") was held hostage by the relay focus's
by-construction-zero held-out SR (s112: expand_refused(defeat_gnome_warrior)).

  P1 SUCCESSION — a relay retiring as relay_stalled WITH a verified chain_unreached attribution
     hands the trigger queue to its own chain's unmastered deep walls, in tree order (the run's
     own diagnosis picks the next campaign, equipment before re-fight). The directive explains it.
  P2 EXPAND EXCLUSION — the expand test (b) ignores ACTIVE relay foci; ordinary foci still gate
     each other (anti-sprawl semantics preserved).

Pure python (SiegeNotebook), no jax/LLM.
"""

import pytest

pytestmark = pytest.mark.relay_trigger  # the succession queue lives in the trigger

from auction.siege_notebook import (
    GAP_STALL_MIN_GAIN_PP,
    MATURITY_MIN_MASTERED,
    MATURITY_MIN_SNAPSHOTS,
    MATURITY_SKILL_SR,
    SiegeNotebook,
)

KOBOLD_TREE = [
    {"skill": "enter_sewers", "prereq_tree": []},
    {"skill": "make_iron_sword", "prereq_tree": []},
    {"skill": "cast_fireball", "prereq_tree": []},
    {"skill": "defeat_lizard", "prereq_tree": []},
]
ATTRIB = {"class": "chain_unreached", "key_missing_link": "enchant_sword", "verified": True}


def _mature_prof(**targets):
    prof = {f"f{i}": MATURITY_SKILL_SR + 10 for i in range(MATURITY_MIN_MASTERED)}
    prof.update(targets)
    return prof


def _apply(nb, session, prof, foci, **kw):
    return nb.apply_llm_update(
        session, prof, {"foci": foci}, num_snapshots=MATURITY_MIN_SNAPSHOTS,
        forensics=kw.pop("forensics", {}), **kw
    )


def _retire_kobold_relay(nb, prof):
    """Open the kobold relay, attach the verified attribution, stall it out via the REAL
    patience machinery (no hand-written registry entries). Returns the retirement session."""
    _apply(nb, 1, prof, [{"skill": "defeat_kobold", "prereq_tree": [], "relay_r0_floor": 3}])
    assert nb.relay_walls() == ["defeat_kobold"]
    _apply(nb, 3, prof, [{
        "skill": "defeat_kobold", "prereq_tree": KOBOLD_TREE,
        "failure_attribution": dict(ATTRIB),
    }])
    assert nb.foci()[0].get("failure_attribution", {}).get("key_missing_link") == "enchant_sword"
    # readings: first sets the anchor (new_high, patience 0); then flat sub-gain readings
    # burn patience through the real EARLY STOP branch until the campaign retires.
    nb.note_rung_reading("defeat_kobold", 12.0, session_idx=4)
    s = 5
    for _ in range(40):
        msg = nb.note_rung_reading(
            "defeat_kobold", 12.0 + GAP_STALL_MIN_GAIN_PP - 1.0, session_idx=s
        )
        if msg and "RELAY_RETIRED" in msg:
            return s
        s += 1
    raise AssertionError("kobold relay never retired through patience")


# ---- P1: attribution survives retirement --------------------------------------------------------

def test_attribution_survives_retirement(tmp_path):
    nb = SiegeNotebook(str(tmp_path / "nb.json"))
    prof = _mature_prof(defeat_kobold=0.0, cast_fireball=0.0)
    _retire_kobold_relay(nb, prof)
    reg = nb._nb["retired"]["defeat_kobold"]
    assert reg["last_event"] == "focus_retired_relay_stalled"
    fa = reg["failure_attribution_at_retirement"]
    assert fa["verified"] is True and fa["key_missing_link"] == "enchant_sword"
    assert "cast_fireball" in reg["links_at_retirement"]
    # resume-safe: a reload keeps it.
    nb2 = SiegeNotebook(str(tmp_path / "nb.json"))
    assert "failure_attribution_at_retirement" in nb2._nb["retired"]["defeat_kobold"]


# ---- P1: succession leads the trigger queue ------------------------------------------------------

def test_succession_reorders_trigger_candidates(tmp_path):
    nb = SiegeNotebook(str(tmp_path / "nb.json"))
    prof = _mature_prof(defeat_kobold=0.0, cast_fireball=0.0, defeat_lizard=0.0)
    s = _retire_kobold_relay(nb, prof)
    cands = nb._relay_trigger_candidates(prof, session_idx=s + 1)
    assert cands, "trigger has no candidates after retirement"
    # The DIAGNOSED link itself leads (rank -1), then the tree-mates in chain order, then fresh
    # fight walls; kobold itself is cooling down and absent; enter_sewers (also in the tree) is
    # NOT promoted — entrances fall as byproducts of any deep campaign.
    assert cands[0] == "enchant_sword", cands
    assert cands[1] == "cast_fireball", cands
    assert "defeat_kobold" not in cands
    li = cands.index("defeat_lizard") if "defeat_lizard" in cands else 10**6
    assert cands.index("cast_fireball") < li
    es = cands.index("enter_sewers") if "enter_sewers" in cands else None
    assert es is None or es > cands.index("defeat_lizard")


def test_no_verified_attribution_keeps_fight_first_order(tmp_path):
    """Baseline contract pinned: without a qualifying retirement the old deterministic order
    (defeat_* first, shallower floor, name) is byte-identical."""
    nb = SiegeNotebook(str(tmp_path / "nb.json"))
    prof = _mature_prof()
    cands = nb._relay_trigger_candidates(prof, session_idx=1)
    assert cands and cands[0].startswith("defeat_")


def test_succession_disabled_by_flag(tmp_path):
    nb = SiegeNotebook(str(tmp_path / "nb.json"))
    prof = _mature_prof(defeat_kobold=0.0, cast_fireball=0.0)
    s = _retire_kobold_relay(nb, prof)
    nb.th.relay_succession = False
    cands = nb._relay_trigger_candidates(prof, session_idx=s + 1)
    assert cands and cands[0].startswith("defeat_")  # fix4.2 semantics, pinned behind the flag


# ---- P1: the directive explains the succession ---------------------------------------------------

def test_directive_renders_succession_line(tmp_path):
    nb = SiegeNotebook(str(tmp_path / "nb.json"))
    prof = _mature_prof(defeat_kobold=0.0, cast_fireball=0.0)
    s = _retire_kobold_relay(nb, prof)
    _apply(nb, s + 1, prof, [])  # a mature decision arms the trigger
    rt = nb._nb["relay_trigger"]
    assert rt["armed"] and rt["candidates"][0] == "enchant_sword"
    assert rt["succession_from"] == "defeat_kobold"
    assert rt["succession_missing"] == "enchant_sword"
    journal = nb.render_for_prompt()
    assert "★RELAY TRIGGER" in journal
    assert "SUCCESSION:" in journal and "enchant_sword" in journal


def test_force_opens_the_succession_pick(tmp_path):
    nb = SiegeNotebook(str(tmp_path / "nb.json"))
    prof = _mature_prof(defeat_kobold=0.0, cast_fireball=0.0)
    s = _retire_kobold_relay(nb, prof)
    for i in range(1, 5):  # arm, then K ignored decisions force the top (succession) pick
        _apply(nb, s + i, prof, [])
        if nb.relay_walls():
            break
    assert nb.relay_walls() == ["enchant_sword"]  # the diagnosed link, not another fight
    assert any(
        h.get("event") == "relay_forced" and h.get("focus") == "enchant_sword"
        for h in nb._nb["history"]
    )


# ---- P2: expand-gate relay exclusion -------------------------------------------------------------

def _open_kobold_relay(tmp_path):
    nb = SiegeNotebook(str(tmp_path / "nb.json"))
    prof = _mature_prof(defeat_kobold=0.0, defeat_gnome_warrior=5.0)
    _apply(nb, 1, prof, [{"skill": "defeat_kobold", "prereq_tree": [], "relay_r0_floor": 3}])
    assert nb.relay_walls() == ["defeat_kobold"]
    return nb, prof


def test_relay_focus_no_longer_blocks_ordinary_expansion(tmp_path):
    nb, prof = _open_kobold_relay(tmp_path)
    assert nb._may_open_new_focus(prof) is True  # was False: relay held-out 0 < expand_sr
    _apply(nb, 3, prof, [
        {"skill": "defeat_kobold", "prereq_tree": []},
        {"skill": "defeat_gnome_warrior", "prereq_tree": []},
    ], forensics={"defeat_gnome_warrior": {"missing_top": []}})
    assert "defeat_gnome_warrior" in nb.focus_skills()


def test_old_contract_pinned_behind_flag(tmp_path):
    nb, prof = _open_kobold_relay(tmp_path)
    nb.th.relay_expand_excluded = False
    assert nb._may_open_new_focus(prof) is False  # fix4.2-era semantics preserved


def test_ordinary_foci_still_gate_each_other(tmp_path):
    """Anti-sprawl survives: with a relay + a 0% ORDINARY focus open, a THIRD focus still
    needs the >= expand_sr progress test over the ordinary foci."""
    nb, prof = _open_kobold_relay(tmp_path)
    _apply(nb, 3, prof, [
        {"skill": "defeat_kobold", "prereq_tree": []},
        {"skill": "defeat_gnome_warrior", "prereq_tree": []},
    ], forensics={"defeat_gnome_warrior": {"missing_top": []}})
    assert "defeat_gnome_warrior" in nb.focus_skills()
    assert nb._may_open_new_focus(prof) is False  # gnome at 5% blocks a third front
