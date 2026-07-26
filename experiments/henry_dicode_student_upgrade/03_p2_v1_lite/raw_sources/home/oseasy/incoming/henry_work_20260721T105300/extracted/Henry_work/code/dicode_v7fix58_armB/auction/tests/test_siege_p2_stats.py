"""v7fix5.7-P2' — judgment statistics repair (fix56设计 §3; E4/E5).

All rung judgments run on the last-3 window mean of the current rung's raw readings; every
eval is consumed at delivery (per session); DEFEND rising reads the raw series; the scientist
sees a feasible-axis menu and a refuted-strength unverifiable footnote.
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


def _set_relay(nb, floor=2, stage=4):
    r = nb.foci()[0]["relay"]
    r["spawn_floor"] = floor
    r["sub_stage"] = int(stage)
    r["rung_trained"] = []
    r["rung_graduate_streak"] = 0
    r["rung_stall_streak"] = 0
    r["stall_patience"] = 0
    nb._save()
    return r


def _feed(nb, readings, base_session=12):
    status = None
    for i, v in enumerate(readings):
        status = nb.note_rung_reading(WALL, float(v), session_idx=base_session + i)
    return status


# ---- T2: window-mean judgments ---------------------------------------------------------------------

def test_single_lucky_reading_does_not_graduate(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    r = _set_relay(nb, stage=4)
    bar = nb.th.rung_graduate_sr
    status = _feed(nb, [12.0, 13.0, bar + 5.0])          # lucky third read, win3 ~ low 30s
    assert "rung hold" in status and r["sub_stage"] == 4
    assert int(r.get("rung_graduate_streak", 0)) == 0


def test_win3_graduation_needs_a_full_window(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    r = _set_relay(nb, stage=4)
    bar = nb.th.rung_graduate_sr
    s = _feed(nb, [bar + 5.0])                            # window not full: hold
    assert "rung hold" in s and int(r.get("rung_graduate_streak", 0)) == 0
    s = _feed(nb, [bar + 5.0, bar + 5.0], base_session=13)
    assert "RUNG_SUBSTAGE_GRADUATED" in s                 # 3rd reading fills the window
    assert r["sub_stage"] == 3


def test_win3_stall_regress(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    r = _set_relay(nb, stage=4)
    n = int(nb.th.rung_stall_readings)
    # the first RUNG_WIN-1 readings judge nothing; then n consecutive window-lows regress.
    status = _feed(nb, [1.0] * (RUNG_WIN - 1 + n))
    assert "RUNG_SUBSTAGE_REGRESSED" in status or "RUNG_REGRESSED" in status
    assert r["sub_stage"] == 5                            # one stage easier than 4


def test_new_high_and_patience_ride_win3(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    r = _set_relay(nb, stage=4)
    # window fills (2 readings judge nothing, patience holds); the FIRST full window is a
    # new high (no prior win3 at this rung) -> patience 0
    _feed(nb, [20.0, 20.0, 20.0])
    assert int(r.get("stall_patience", 0)) == 0
    assert (r.get("best_win3_by_rung") or {}).get("2:4") == 20.0
    # flat windows burn patience one per judged reading
    _feed(nb, [20.0, 20.0], base_session=20)
    assert int(r.get("stall_patience", 0)) == 2
    # a win3 new-high (> best win3 + 2pp) resets it: [20,20,30] mean 23.3 > 22
    _feed(nb, [30.0], base_session=25)
    assert int(r.get("stall_patience", 0)) == 0
    # single lucky 30 never set the +3pp single-read anchor of the old law
    assert int(r.get("rung_graduate_streak", 0)) == 0


# ---- T3: rising reads the raw series ----------------------------------------------------------------

def test_rising_reads_raw_series(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    r = _set_relay(nb)
    r["rung_trained"] = [10.0, 11.0, 12.0, 13.0]
    assert nb._relay_ratchet_rising(r) is True
    r["rung_trained"] = [13.0, 12.0, 11.0, 10.0]
    assert nb._relay_ratchet_rising(r) is False
    r["rung_trained"] = [10.0, 11.0]                      # fewer than 4: not rising
    assert nb._relay_ratchet_rising(r) is False
    r["ratchet_log"] = [1, 1, 1]                          # the old input must be inert
    assert nb._relay_ratchet_rising(r) is False


# ---- T1: per-session consumption ---------------------------------------------------------------------

def test_consume_rung_eval_delivers_and_holds(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    r = _set_relay(nb, floor=2, stage=4)
    sc = nb.relay_scaffold(WALL)
    nb._nb.setdefault("rung_eval", {})[WALL] = {
        "session": 12, "spawn_floor": 2, "sub_stage": int(sc["sub_stage"]), "sr": 41.5,
    }
    nb._save()
    ev, st = nb.consume_rung_eval(WALL, 12)
    assert ev == 41.5 and "rung hold" in st
    assert (r.get("rung_trained") or [])[-1] == 41.5
    # stale eval -> counters hold (the no_fresh beat), nothing appended
    n0 = len(r.get("rung_trained") or [])
    ev, st = nb.consume_rung_eval(WALL, 20)
    assert ev is None and "no_fresh_rung_reading" in st
    assert len(r.get("rung_trained") or []) == n0
    # not a relay wall -> (None, None)
    assert nb.consume_rung_eval("defeat_zombie", 12) == (None, None)


# ---- T4/T5b: scientist information repair -------------------------------------------------------------

def test_axis_menu_and_unverifiable_footnote(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    _set_relay(nb, floor=2, stage=4)                      # entry stage: radius axis is dead
    txt = nb.render_for_prompt()
    assert "AXIS MENU" in txt
    assert "radius: EXHAUSTED" in txt
    assert "pre_light: easier" in txt                     # dark entry: easier still moves
    nb._nb.setdefault("hypothesis_log", []).append(
        {"id": "hx", "wall": WALL, "session": 10, "axis": "needs_clock",
         "direction": "easier", "hypothesis": "clocks", "evidence": "e", "prediction": "p",
         "status": "unverifiable", "note": "needs_clock_at_boundary"}
    )
    nb._save()
    txt = nb.render_for_prompt()
    assert "CANNOT be the binding constraint" in txt
    assert "Propose a DIFFERENT axis" in txt
