"""Offline tests for the v6fix8 FOCUS ECONOMICS layer (user 2026-07-07).

fix7 post-mortem: the iron-pickaxe siege locked a single enabler focus for 17 sessions — the only
exits (conquest >=70x2, total-freeze retirement) were unreachable from the 44-61 held-out dead zone,
the second focus was prompt-soft and never came, the drill-transfer gap was computed but had no
teeth, and 22/24 generation seats were guaranteed away. These tests pin the five new hard gates:

  ① FAST-CLIMB GRADUATION: >= graduate_sr for graduate_consecutive snapshots -> maintenance
     (privileges withdraw, rehearsal holds it, re-siege only after a real collapse).
  ② MULTI-FOCUS HARD GATE: ranked_walls auto-open fills a free expand-gate slot the LLM left empty
     (combat candidates preferred while no combat focus is active).
  ③ GAP GATE: consecutive over-gap decisions force required_form=DEPTH; follows evidence both ways.
  ⑤ ENABLER BUDGET: a non-combat focus retires after enabler_max_sessions siege decisions through
     the normal retirement machinery; combat foci are exempt.

(④ the generation-seat cap lives in gen_manager._coop_select — tested in test_siege_ecosystem.py,
which needs the jax-importing gen_manager module.)

No jax/craftax/LLM needed.
"""

import pytest

from auction.siege_notebook import (
    ENABLER_MAX_SESSIONS,
    FOCUS_EXPAND_SR,
    GAP_FORCE_SESSIONS,
    GRADUATE_CONSECUTIVE,
    GRADUATE_SR,
    MAINT_RESIEGE_DROP_PP,
    MASTERED_SR,
    MATURITY_MIN_MASTERED,
    MATURITY_MIN_SNAPSHOTS,
    MATURITY_SKILL_SR,
    SiegeNotebook,
)


def _mature_profile(extra: dict | None = None) -> dict:
    prof = {f"filler_{i}": MATURITY_SKILL_SR + 10 for i in range(MATURITY_MIN_MASTERED)}
    if extra:
        prof.update(extra)
    return prof


@pytest.fixture
def nb_path(tmp_path):
    return str(tmp_path / "siege_notebook.json")


def _open(nb, session, skill, sr=8.0, tree=None):
    """Open ``skill`` as a focus at ``session`` with a low SR (a real wall)."""
    foci = [{"skill": skill, "prereq_tree": tree or []}]
    nb.apply_llm_update(session, _mature_profile({skill: sr}),
                        {"foci": foci}, num_snapshots=MATURITY_MIN_SNAPSHOTS)
    assert skill in nb.focus_skills()


# ---- ① fast-climb graduation --------------------------------------------------------------------

def test_fast_climb_graduates_to_maintenance(nb_path):
    nb = SiegeNotebook(nb_path)
    tree = [{"skill": "collect_iron", "role": "ore"}, {"skill": "place_furnace", "role": "smelt"}]
    _open(nb, 1, "defeat_gnome_warrior", tree=tree)
    # two consecutive snapshots at/above graduate_sr (but below mastered — NOT a conquest)...
    for i in range(GRADUATE_CONSECUTIVE):
        nb.apply_llm_update(
            2 + i, _mature_profile({"defeat_gnome_warrior": GRADUATE_SR + 2 + i}),
            {"foci": [{"skill": "defeat_gnome_warrior", "prereq_tree": tree}]},
            num_snapshots=MATURITY_MIN_SNAPSHOTS,
        )
    # ...graduates: leaves foci, enters maintenance, wall+links protected for rehearsal.
    assert nb.focus_skills() == []
    snap = nb.snapshot()
    assert "defeat_gnome_warrior" in snap["maintenance"]
    assert {"defeat_gnome_warrior", "collect_iron", "place_furnace"} <= set(snap["protected_set"])
    # graduation is NOT conquest: the experience entry stays 'progress' (#8 semantics preserved).
    entry = next(c for c in snap["verified_chains"] if c["target"] == "defeat_gnome_warrior")
    assert entry["status"] == "progress"
    assert any(h.get("event") == "focus_graduated_maintenance" for h in snap["history"])
    assert nb.last_graduation and "defeat_gnome_warrior" in nb.last_graduation


def test_conquest_outranks_graduation(nb_path):
    # Holding at/above mastered_sr exits as CONQUERED (verified), never as merely graduated.
    nb = SiegeNotebook(nb_path)
    _open(nb, 1, "defeat_gnome_warrior")
    for i in range(GRADUATE_CONSECUTIVE):
        nb.apply_llm_update(
            2 + i, _mature_profile({"defeat_gnome_warrior": MASTERED_SR + 5}),
            {"foci": [{"skill": "defeat_gnome_warrior"}]}, num_snapshots=MATURITY_MIN_SNAPSHOTS,
        )
    snap = nb.snapshot()
    assert "defeat_gnome_warrior" not in snap["maintenance"]
    entry = next(c for c in snap["verified_chains"] if c["target"] == "defeat_gnome_warrior")
    assert entry["status"] == "verified"


def test_maintained_wall_resiege_only_after_collapse(nb_path):
    nb = SiegeNotebook(nb_path)
    _open(nb, 1, "defeat_gnome_warrior")
    for i in range(GRADUATE_CONSECUTIVE):
        nb.apply_llm_update(2 + i, _mature_profile({"defeat_gnome_warrior": GRADUATE_SR + 3}),
                            {"foci": [{"skill": "defeat_gnome_warrior"}]},
                            num_snapshots=MATURITY_MIN_SNAPSHOTS)
    assert nb.focus_skills() == []
    # still healthy (above the collapse line) -> re-siege REFUSED.
    healthy = GRADUATE_SR - MAINT_RESIEGE_DROP_PP + 5
    nb.apply_llm_update(10, _mature_profile({"defeat_gnome_warrior": healthy}),
                        {"foci": [{"skill": "defeat_gnome_warrior"}]},
                        num_snapshots=MATURITY_MIN_SNAPSHOTS)
    assert nb.focus_skills() == []
    assert "maintained" in (nb.last_focus_decision or "")
    # truly collapsed (below the line) -> re-siege legal; maintenance entry is consumed.
    collapsed = GRADUATE_SR - MAINT_RESIEGE_DROP_PP - 5
    nb.apply_llm_update(11, _mature_profile({"defeat_gnome_warrior": collapsed}),
                        {"foci": [{"skill": "defeat_gnome_warrior"}]},
                        num_snapshots=MATURITY_MIN_SNAPSHOTS)
    assert nb.focus_skills() == ["defeat_gnome_warrior"]
    assert "defeat_gnome_warrior" not in nb.snapshot()["maintenance"]


# ---- ② multi-focus hard gate (ranked_walls auto-open) --------------------------------------------

def test_auto_open_fills_free_slot_from_ranked_walls(nb_path):
    nb = SiegeNotebook(nb_path)
    _open(nb, 1, "defeat_orc_mage")
    # expand gate satisfied (anchor at expand SR, first snapshot); LLM keeps only its one focus but
    # ranks the next walls -> the code opens the top viable candidate itself.
    nb.apply_llm_update(
        2, _mature_profile({"defeat_orc_mage": FOCUS_EXPAND_SR + 1, "collect_sapphire": 2.0}),
        {"foci": [{"skill": "defeat_orc_mage"}],
         "ranked_walls": [{"skill": "collect_sapphire", "why": "unlocks enchant"}]},
        num_snapshots=MATURITY_MIN_SNAPSHOTS,
    )
    assert nb.focus_skills() == ["defeat_orc_mage", "collect_sapphire"]
    snap = nb.snapshot()
    auto = next(f for f in snap["foci"] if f["skill"] == "collect_sapphire")
    assert auto["opened_by"] == "auto"
    assert nb.last_auto_open and "collect_sapphire" in nb.last_auto_open


def test_auto_open_prefers_combat_when_no_combat_focus(nb_path):
    nb = SiegeNotebook(nb_path)
    _open(nb, 1, "make_iron_pickaxe")  # CRAFT focus — no combat wall under siege
    nb.apply_llm_update(
        2, _mature_profile({"make_iron_pickaxe": FOCUS_EXPAND_SR + 1,
                            "collect_sapphire": 2.0, "defeat_gnome_warrior": 2.0}),
        {"foci": [{"skill": "make_iron_pickaxe"}],
         "ranked_walls": [{"skill": "collect_sapphire"}, {"skill": "defeat_gnome_warrior"}]},
        num_snapshots=MATURITY_MIN_SNAPSHOTS,
    )
    # the combat candidate wins the slot despite being ranked second.
    assert nb.focus_skills() == ["make_iron_pickaxe", "defeat_gnome_warrior"]


def test_no_auto_open_while_expand_gate_closed(nb_path):
    nb = SiegeNotebook(nb_path)
    _open(nb, 1, "defeat_orc_mage")
    nb.apply_llm_update(
        2, _mature_profile({"defeat_orc_mage": FOCUS_EXPAND_SR - 10, "defeat_gnome_warrior": 2.0}),
        {"foci": [{"skill": "defeat_orc_mage"}],
         "ranked_walls": [{"skill": "defeat_gnome_warrior"}]},
        num_snapshots=MATURITY_MIN_SNAPSHOTS,
    )
    assert nb.focus_skills() == ["defeat_orc_mage"]
    assert nb.last_auto_open is None


# ---- ③ gap gate ----------------------------------------------------------------------------------

def test_gap_gate_forces_depth_then_follows_evidence_back(nb_path):
    nb = SiegeNotebook(nb_path)
    _open(nb, 1, "make_iron_pickaxe")
    # consecutive over-gap decisions (drill won in sandbox, held-out lagging)...
    for i in range(GAP_FORCE_SESSIONS - 1):
        status = nb.note_transfer_gap("make_iron_pickaxe", 95.0, 40.0)
        assert status.startswith("over_gap")
        assert nb.required_form("make_iron_pickaxe") is None  # not forced yet
    # v6fix9 P3 appends the early-stop stall counter to the status string.
    assert nb.note_transfer_gap("make_iron_pickaxe", 95.0, 40.0).startswith("FORCED_DEPTH")
    assert nb.required_form("make_iron_pickaxe") == "DEPTH"
    # the gap closes -> the force lifts (the gate follows evidence in both directions).
    assert nb.note_transfer_gap("make_iron_pickaxe", 95.0, 75.0) == "ok"
    assert nb.required_form("make_iron_pickaxe") is None
    # no trained reading (no drills) -> no gap, counter resets.
    assert nb.note_transfer_gap("make_iron_pickaxe", None, 40.0) == "ok"
    # not an active focus -> None.
    assert nb.note_transfer_gap("defeat_gnome_warrior", 95.0, 40.0) is None


# ---- ⑤ enabler budget ----------------------------------------------------------------------------

def test_enabler_focus_retires_at_budget(nb_path):
    nb = SiegeNotebook(nb_path)
    _open(nb, 1, "make_iron_pickaxe")  # CRAFT -> enabler, budget-capped
    # rising steadily (never frozen, never >= graduate_sr) so ONLY the budget can end it.
    sr = 5.0
    for s in range(2, 2 + ENABLER_MAX_SESSIONS):
        sr = min(sr + 4.0, GRADUATE_SR - 5)
        nb.apply_llm_update(s, _mature_profile({"make_iron_pickaxe": sr}),
                            {"foci": [{"skill": "make_iron_pickaxe"}]},
                            num_snapshots=MATURITY_MIN_SNAPSHOTS)
        if nb.focus_skills() == []:
            break
    snap = nb.snapshot()
    assert nb.focus_skills() == []
    assert "make_iron_pickaxe" in snap["retired"]
    assert any(h.get("event") == "focus_retired_budget" for h in snap["history"])
    # normal retirement machinery applies: cooldown blocks an immediate reopen.
    nb.apply_llm_update(2 + ENABLER_MAX_SESSIONS,
                        _mature_profile({"make_iron_pickaxe": sr}),
                        {"foci": [{"skill": "make_iron_pickaxe"}]},
                        num_snapshots=MATURITY_MIN_SNAPSHOTS)
    assert nb.focus_skills() == []
    assert "cooldown_rejected" in (nb.last_focus_decision or "")


def test_combat_focus_exempt_from_budget(nb_path):
    nb = SiegeNotebook(nb_path)
    _open(nb, 1, "defeat_orc_mage")  # COMBAT -> unbounded patience while the tree moves
    sr = 2.0
    for s in range(2, 2 + ENABLER_MAX_SESSIONS + 4):
        sr += 4.0  # keeps rising -> never frozen; well past the enabler budget
        nb.apply_llm_update(s, _mature_profile({"defeat_orc_mage": min(sr, GRADUATE_SR - 5)}),
                            {"foci": [{"skill": "defeat_orc_mage"}]},
                            num_snapshots=MATURITY_MIN_SNAPSHOTS)
    assert nb.focus_skills() == ["defeat_orc_mage"]


# ---- ② schema: ranked_walls validation (modeler side) --------------------------------------------

def test_validate_siege_parses_ranked_walls():
    from auction.modeler import Modeler
    raw = {"siege_update": {
        "foci": [{"skill": "Defeat_Gnome_Warrior"}],
        "ranked_walls": [
            {"skill": "Defeat_Orc_Mage", "why": "stuck fight"},
            "collect_sapphire",              # bare-string form accepted
            {"skill": "defeat_orc_mage"},    # duplicate dropped
            {"skill": ""},                   # invalid dropped
        ],
    }}
    su = Modeler._validate_siege(raw)
    assert [f["skill"] for f in su["foci"]] == ["defeat_gnome_warrior"]
    assert su["ranked_walls"] == [
        {"skill": "defeat_orc_mage", "why": "stuck fight"},
        {"skill": "collect_sapphire", "why": ""},
    ]


def test_validate_siege_ranked_walls_absent_is_empty():
    from auction.modeler import Modeler
    su = Modeler._validate_siege({"siege_update": {"foci": [{"skill": "defeat_gnome_warrior"}]}})
    assert su["ranked_walls"] == []
