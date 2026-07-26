"""v7fix5.7 — graded pre-light anneal.

Design: the fix5.4 probe measured the entry-context cliff at ~25pp and the fix5.5 hypothesis
loop verified pre_light as its main axis (+25.4pp, compiled as the INSERT rung). Falling from a
graduated pre-lit insert straight back to the dark return stage re-imposes that whole cliff in
one step. fix5.7 splits it: (a) the builder's pre_light knob gains a middle value "ladder"
(ONLY the down ladder's 9x9 is torch-lit — dark start, lit destination); (b) a graduated
pre_light=True insert first descends to RUNG_INSERT_LIGHT_STAGE (49) with knobs annealed
True -> "ladder", and only the light leg's graduation pops the insert back to return_stage;
(c) the what-if pre_light axis becomes a 3-level ladder (one notch per step).
"""

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for p in (os.path.join(_REPO, "src"), _REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from auction.siege_notebook import (  # noqa: E402
    MATURITY_MIN_MASTERED,
    MATURITY_MIN_SNAPSHOTS,
    MATURITY_SKILL_SR,
    RUNG_INSERT_LIGHT_STAGE,
    RUNG_INSERT_STAGE,
    RUNG_WIN,
    SiegeNotebook,
)

WALL = "defeat_kobold"


def _mature_profile(extra=None):
    prof = {f"filler_{i}": MATURITY_SKILL_SR + 10 for i in range(MATURITY_MIN_MASTERED)}
    if extra:
        prof.update(extra)
    return prof


@pytest.fixture
def nb_path(tmp_path):
    return str(tmp_path / "siege_notebook.json")


def _open_relay(nb, wall=WALL, r0=3, session=1):
    prof = _mature_profile({wall: 0.0})
    nb.apply_llm_update(
        session, prof,
        {"foci": [{"skill": wall, "prereq_tree": [], "relay_r0_floor": r0}]},
        num_snapshots=MATURITY_MIN_SNAPSHOTS, forensics={},
    )
    return prof


def _set_relay(nb, floor=2, stage=4, readings=(12.0, 13.0, 12.5)):
    r = nb.foci()[0]["relay"]
    r["spawn_floor"] = floor
    r["sub_stage"] = int(stage)
    r["rung_trained"] = list(readings)
    r["readings_since_transition"] = 6
    r["gain_log"] = [0.0, 0.0]
    nb._save()
    return r


def _install_insert(nb, pre_light=True, return_stage=4):
    """Install a compiled insert the way the LIVE pre-5.7 notebooks carry it
    (knobs pre_light=True) or the way post-5.7 compiles produce it ("ladder")."""
    r = nb.foci()[0]["relay"]
    r["stage_insert"] = {
        "return_stage": int(return_stage), "floor": int(r["spawn_floor"]),
        "knobs": {"down_ladder_radius": None, "monster_credit": 8, "uplock": True,
                  "needs_multiplier": 0.3, "pre_light": pre_light},
        "axis": "pre_light", "direction": "easier",
        "hypothesis_id": "h57_test", "session": 10, "delta_pp": 25.4,
        "step_desc": "pre-light False -> True (spawn anchor unchanged)",
    }
    r["sub_stage"] = RUNG_INSERT_STAGE
    r["rung_trained"] = []
    r["rung_graduate_streak"] = 0
    r["rung_stall_streak"] = 0
    nb._nb.setdefault("hypothesis_log", []).append(
        {"id": "h57_test", "wall": WALL, "session": 10, "axis": "pre_light",
         "direction": "easier", "hypothesis": "x", "evidence": "y", "prediction": "z",
         "status": "verified_compiled"}
    )
    nb._save()
    return r


def _graduate(nb, r, base_session):
    # P2' (fix56设计 §3.2): graduation judges the last-RUNG_WIN window mean, so the window
    # must fill before the first judged reading — RUNG_WIN-1 extra above-bar readings.
    status = None
    for i in range(RUNG_WIN - 1 + nb.th.rung_substage_graduate_x):
        status = nb.note_rung_reading(
            WALL, nb.th.rung_graduate_sr + 5.0, session_idx=base_session + i
        )
    return status


# ---- the light-anneal leg --------------------------------------------------------------------------

def test_true_insert_graduates_via_light_anneal_leg(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    _set_relay(nb)
    r = _install_insert(nb, pre_light=True, return_stage=4)
    status = _graduate(nb, r, 12)
    assert "RUNG_INSERT_LIGHT_ANNEAL" in status
    assert r["sub_stage"] == RUNG_INSERT_LIGHT_STAGE
    ins = r["stage_insert"]                              # insert retained, knob annealed
    assert ins["knobs"]["pre_light"] == "ladder"
    assert r["rung_history"][-1]["event"] == "rung_insert_light_anneal"
    assert nb._nb["hypothesis_log"][-1]["status"] == "verified_compiled"  # not graduated yet
    sc = nb.relay_scaffold(WALL)                         # the leg serves the annealed knobs
    assert sc["sub_stage"] == RUNG_INSERT_LIGHT_STAGE and sc["pre_light"] == "ladder"


def test_light_leg_graduation_pops_back_to_return_stage(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    _set_relay(nb)
    r = _install_insert(nb, pre_light=True, return_stage=4)
    _graduate(nb, r, 12)
    status = _graduate(nb, r, 20)
    assert "RUNG_INSERT_GRADUATED" in status
    assert r["sub_stage"] == 4 and "stage_insert" not in r
    assert nb._nb["hypothesis_log"][-1]["status"] == "insert_graduated"


def test_ladder_insert_pops_in_one_graduation(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    _set_relay(nb)
    r = _install_insert(nb, pre_light="ladder", return_stage=4)
    status = _graduate(nb, r, 12)
    assert "RUNG_INSERT_GRADUATED" in status             # no second leg below "ladder"
    assert r["sub_stage"] == 4 and "stage_insert" not in r


def test_stall_on_light_leg_removes_insert_and_regresses(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    _set_relay(nb)
    r = _install_insert(nb, pre_light=True, return_stage=4)
    _graduate(nb, r, 12)
    assert r["sub_stage"] == RUNG_INSERT_LIGHT_STAGE
    status = None
    for i in range(RUNG_WIN - 1 + nb.th.rung_stall_readings):   # P2': window fills first
        status = nb.note_rung_reading(WALL, 1.0, session_idx=20 + i)
    assert "RUNG_INSERT_STALLED" in status
    assert "stage_insert" not in r                       # wrong leg self-heals as before
    assert r["sub_stage"] == 5                           # return_stage + 1 (one stage easier)


def test_light_leg_ratchet_is_fresh(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    _set_relay(nb)
    r = _install_insert(nb, pre_light=True, return_stage=4)
    key = f"{int(r['spawn_floor'])}:{RUNG_INSERT_LIGHT_STAGE}"
    r.setdefault("best_by_rung", {})[key] = 88.0         # a stale best from another life
    nb._save()
    _graduate(nb, r, 12)
    assert key not in (r.get("best_by_rung") or {})


# ---- disclosure ------------------------------------------------------------------------------------

def test_facts_clause_ladder_mode(nb_path):
    facts = SiegeNotebook._scaffold_fact_clauses(
        {"down_ladder_radius": None, "monster_credit": 8, "uplock": True,
         "needs_multiplier": 0.3, "pre_light": "ladder"}
    )
    assert "down ladder is torch-lit" in facts["pre-light"]
    assert "dark start" in facts["pre-light"]
    assert "entry" in facts["spawn"]


def test_render_marks_light_anneal_leg(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    _set_relay(nb)
    r = _install_insert(nb, pre_light=True, return_stage=4)
    _graduate(nb, r, 12)
    txt = nb.render_for_prompt()
    assert "LIGHT-ANNEAL leg" in txt
