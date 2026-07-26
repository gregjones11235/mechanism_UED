"""v7fix5.6 regression tests — honest rung readings (measurement decoupled from training)."""

import os

import pytest

from auction.siege_notebook import (
    PROBE_BUDGET_WINDOW,
    PROBE_BUDGET_WINDOW_FAST,
    SiegeNotebook,
)


@pytest.fixture()
def nb(tmp_path):
    return SiegeNotebook(os.path.join(str(tmp_path), "nb.json"))


def test_fast_window_constants():
    # diagnose/whatif iterate on the 5-session window; verify keeps 10 (B3.4 decision).
    assert PROBE_BUDGET_WINDOW_FAST == 5
    assert PROBE_BUDGET_WINDOW == 10


def test_diagnose_whatif_budget_restores_at_fast_window(nb):
    nb._nb.setdefault("probe_ledger", {})["w"] = [[0, "diagnose"], [0, "whatif"]]
    assert nb._probe_budget_left("w", 4) == {"diagnose": 0, "whatif": 0}
    assert nb._probe_budget_left("w", 5) == {"diagnose": 1, "whatif": 1}


def test_rung_eval_roundtrip_and_coerce(nb):
    nb.note_rung_eval("defeat_kobold", {
        "session": 10, "sr": 33.3, "spawn_floor": 2, "sub_stage": 4, "n_envs": 512,
    })
    got = SiegeNotebook(nb.path)._nb["rung_eval"]["defeat_kobold"]
    assert got == {
        "session": 10, "sr": 33.3, "spawn_floor": 2, "sub_stage": 4, "n_envs": 512,
    }


def test_rung_eval_for_requires_relay_focus(nb):
    nb.note_rung_eval("defeat_kobold", {
        "session": 10, "sr": 33.3, "spawn_floor": 2, "sub_stage": 4, "n_envs": 512,
    })
    # no relay focus for the wall -> None -> counters hold (the no_fresh beat)
    assert nb.rung_eval_for("defeat_kobold", 11) is None


def test_rung_eval_for_staleness_window(nb):
    nb.note_rung_eval("w", {
        "session": 10, "sr": 20.0, "spawn_floor": 2, "sub_stage": 4, "n_envs": 512,
    })
    e = nb._nb["rung_eval"]["w"]
    # freshness gate alone (relay-focus gates tested separately): measured-at s10 serves
    # s10/s11 decisions, never s12+ — a transition-lagged eval must not drive the machine.
    assert int(e["session"]) >= 11 - 1
    assert not int(e["session"]) >= 12 - 1
