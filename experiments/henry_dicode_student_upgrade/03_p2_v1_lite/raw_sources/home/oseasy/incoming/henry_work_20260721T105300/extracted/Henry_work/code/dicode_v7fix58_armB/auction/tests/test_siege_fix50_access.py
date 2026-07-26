"""Offline tests for the v7fix5.0 ACCESS-WALL layer (frontier + cond + park/exempt machinery).

fix5.0 root cause (s213 forensics deep-dive, 2026-07-14): held-out defeat_gnome_warrior sat at
~15-20% for 20+ sessions and was STYLE_REJECTED, while the truth was upstream: 96% of failures
never entered the gnomish mines (reach 18.6%, the 95%->19% cliff at the floor1->2 descent), and
episodes that DID reach them completed the wall at cond=81% ~= the trained SR — combat had fully
transferred; ACCESS was the binding constraint. The LLM attribution meanwhile blamed the
diamond-gear chain (press-count fingerprints misread as crafting successes). These tests pin:

  A1  ChainOrderLog.access_frontier: shallowest-frontier rule, cond math on the gnome numbers,
      sample guards (episodes + reached), zero-win walls get a frontier but NO certificate.
  A2  forensics() carries the access pack; render_chain_hint prints the BINDING-ACCESS line.
  A3  Modeler._validate_siege: deterministic override to (chain_unreached, frontier, verified)
      with the LLM's original claim kept for audit; no override when the LLM already names the
      frontier; no reroll burned.
  A4  SiegeNotebook gap gate: a CERTIFIED cap parks the wall to WATCH with gap counters frozen;
      an uncertified cap (cond low = true style disease) leaves STYLE_REJECTED fully armed.
  A5  _process_watch: a certified-capped watcher is HELD (no park->resume oscillation); dropping
      the cap re-enables resume.
  A6  Expand gate: a named access frontier opens with a free slot even when every active focus
      is < focus_expand_sr; capacity still binds; access_frontiers() reads the fed caps.
  A7  Persistence: access_caps survives a notebook reload (_coerce whitelist).

No jax/craftax/LLM needed.
"""

import pytest

from auction.chain_order_log import (
    ACCESS_CAP_REACH,
    ACCESS_COND_TRANSFERRED,
    ACCESS_MIN_EPISODES,
    ACCESS_MIN_REACHED,
    ChainOrderLog,
)
from auction.modeler import Modeler
from auction.siege_notebook import (
    GAP_FORCE_SESSIONS,
    GAP_STALL_PATIENCE,
    SiegeNotebook,
)
from auction.tests.test_siege_fix8_economics import _mature_profile, _open


@pytest.fixture
def path(tmp_path):
    return str(tmp_path / "chain.json")


@pytest.fixture
def nb_path(tmp_path):
    return str(tmp_path / "siege_notebook.json")


GNOME_LINKS = ["enter_dungeon", "enter_gnomish_mines", "make_iron_pickaxe", "place_furnace"]


def _entry(target, links, n_fail, n_succ, missing, session=213):
    """A fail_hist entry in the stored schema (only the fields access_frontier reads + staples)."""
    return {
        "session": session, "target": target, "links": list(links),
        "n_fail": int(n_fail), "n_succ": int(n_succ), "mean_depth": 2.0,
        "last_link": {links[-1]: n_fail}, "missing": dict(missing),
        "died_frac": 1.0, "after_deepest_med": 140,
    }


def _gnome_log(path):
    """The real s213 gnome shape: 1024 episodes, 153 wins, mines missing in 834 failures."""
    log = ChainOrderLog(path)
    log._fail_hist.append(_entry(
        "defeat_gnome_warrior", GNOME_LINKS, n_fail=871, n_succ=153,
        missing={"enter_gnomish_mines": 834, "make_iron_pickaxe": 584, "enter_dungeon": 46},
    ))
    return log


def _kobold_log(path):
    """The real s213 kobold shape: zero-win wall, same mines frontier, cond ~0."""
    log = ChainOrderLog(path)
    log._fail_hist.append(_entry(
        "defeat_kobold",
        ["enter_dungeon", "enter_gnomish_mines", "enter_sewers", "make_iron_pickaxe"],
        n_fail=1023, n_succ=1,
        missing={"enter_gnomish_mines": 834, "enter_sewers": 1023, "enter_dungeon": 46},
    ))
    return log


# ---- A1: frontier + cond math --------------------------------------------------------------------

def test_gnome_frontier_and_cond_certified(path):
    ax = _gnome_log(path).access_frontier("defeat_gnome_warrior")
    assert ax is not None
    # shallowest binding link: dungeon reach = 1-46/1024 = 95.5% passes; mines 18.6% binds.
    assert ax["frontier"] == "enter_gnomish_mines"
    assert ax["reach_frac"] == pytest.approx(1 - 834 / 1024, abs=1e-3)
    # cond = 153/(1024-834) = 80.5% >= 0.60 with 190 reached episodes -> certified.
    assert ax["cond"] == pytest.approx(153 / 190, abs=1e-3)
    assert ax["certified"] is True
    assert ax["reached_n"] == 190


def test_kobold_frontier_is_mines_not_sewers_and_uncertified(path):
    ax = _kobold_log(path).access_frontier("defeat_kobold")
    # shallowest rule: sewers is missing in 100% but the mines (idx 1) bind FIRST.
    assert ax["frontier"] == "enter_gnomish_mines"
    # zero-win wall: cond = 1/190 ~ 0.5% -> frontier for attribution, NO park certificate.
    assert ax["cond"] < ACCESS_COND_TRANSFERRED
    assert ax["certified"] is False


def test_no_frontier_when_all_links_reached(path):
    log = ChainOrderLog(path)
    log._fail_hist.append(_entry(
        "make_iron_armour", ["collect_iron", "place_furnace"], n_fail=871, n_succ=153,
        missing={"collect_iron": 100},  # reach 90% — nothing below ACCESS_CAP_REACH
    ))
    assert log.access_frontier("make_iron_armour") is None


def test_sample_guards(path):
    # (a) too few episodes total -> None even with a glaring frontier.
    log = ChainOrderLog(path)
    log._fail_hist.append(_entry(
        "defeat_gnome_warrior", GNOME_LINKS,
        n_fail=ACCESS_MIN_EPISODES - 11, n_succ=10,
        missing={"enter_gnomish_mines": ACCESS_MIN_EPISODES - 12},
    ))
    assert log.access_frontier("defeat_gnome_warrior") is None
    # (b) frontier stands but the cond certificate needs ACCESS_MIN_REACHED past it.
    log2 = ChainOrderLog(path + "2")
    log2._fail_hist.append(_entry(
        "defeat_gnome_warrior", GNOME_LINKS, n_fail=980, n_succ=44,
        missing={"enter_gnomish_mines": 979},  # reached_n = 45 < ACCESS_MIN_REACHED
    ))
    ax = log2.access_frontier("defeat_gnome_warrior")
    assert ax["frontier"] == "enter_gnomish_mines"
    assert ax["cond"] > ACCESS_COND_TRANSFERRED  # 44/45 — high but under-sampled
    assert ax["certified"] is False


# ---- A2: forensics pack + chain-hint render ------------------------------------------------------

def test_forensics_carries_access_and_hint_renders_binding_line(path):
    log = _gnome_log(path)
    fx = log.forensics("defeat_gnome_warrior")
    assert fx and fx["access"]["frontier"] == "enter_gnomish_mines"
    hint = log.render_chain_hint("defeat_gnome_warrior")
    assert "BINDING-ACCESS=enter_gnomish_mines" in hint
    assert "TRANSFERRED" in hint  # the certificate sentence for the gnome shape
    hint_k = _kobold_log(path + "k").render_chain_hint("defeat_kobold")
    assert "BINDING-ACCESS=enter_gnomish_mines" in hint_k
    assert "TRANSFERRED" not in hint_k  # zero-win wall: no certificate language


# ---- A3: attribution override --------------------------------------------------------------------

def _fx_gnome(certified=True):
    access = {
        "frontier": "enter_gnomish_mines", "frontier_idx": 1,
        "reach_frac": 0.1855, "cond": 0.8053 if certified else 0.01,
        "certified": certified, "n_episodes": 1024, "reached_n": 190,
    }
    return {
        "defeat_gnome_warrior": {
            "n_fail": 871,
            "links": GNOME_LINKS,
            "missing_top": [("enter_gnomish_mines", 0.9575), ("make_iron_pickaxe", 0.6705)],
            "break_at_final": False,
            "died_frac": 1.0,
            "after_deepest_med": 140,
            "chain_incomplete": False,
            "inv_gaps": [("stone", 20, 14, 0.37)],
            "access": access,
        }
    }


def _su(attrib):
    return {"siege_update": {"foci": [{
        "skill": "defeat_gnome_warrior", "prereq_tree": [], "style_note": "x",
        "evidence_check": "no_evidence", "failure_attribution": attrib,
    }]}}


def test_attribution_downstream_claim_overridden_with_audit():
    # the literal s207 misdiagnosis: resource_shortfall on the diamond-gear chain.
    su = Modeler._validate_siege(
        _su({"class": "resource_shortfall", "key_missing_link": "make_iron_pickaxe"}),
        forensics=_fx_gnome(),
    )
    a = su["foci"][0]["failure_attribution"]
    assert a["class"] == "chain_unreached"
    assert a["key_missing_link"] == "enter_gnomish_mines"
    assert a["verified"] is True
    assert a["overridden"] == "ACCESS_CAPPED"
    assert a["llm_said_class"] == "resource_shortfall"
    assert a["llm_said_key"] == "make_iron_pickaxe"
    # deterministic override, not a reroll: no violation string burned for this.
    assert not any("make_iron_pickaxe" in v for v in su["attrib_violations"])


def test_attribution_no_override_when_llm_names_frontier():
    su = Modeler._validate_siege(
        _su({"class": "chain_unreached", "key_missing_link": "enter_gnomish_mines"}),
        forensics=_fx_gnome(),
    )
    a = su["foci"][0]["failure_attribution"]
    assert a["class"] == "chain_unreached" and a["key_missing_link"] == "enter_gnomish_mines"
    assert a["verified"] is True
    assert "overridden" not in a


def test_attribution_override_fires_even_uncertified():
    # kobold-shaped: frontier stands (attribution) though the park certificate does not.
    su = Modeler._validate_siege(
        _su({"class": "execution_failure", "key_missing_link": None}),
        forensics=_fx_gnome(certified=False),
    )
    a = su["foci"][0]["failure_attribution"]
    assert a["class"] == "chain_unreached"
    assert a["key_missing_link"] == "enter_gnomish_mines"
    assert a["access_certified"] is False


def test_attribution_untouched_without_access_pack():
    fx = _fx_gnome()
    fx["defeat_gnome_warrior"]["access"] = None
    su = Modeler._validate_siege(
        _su({"class": "resource_shortfall", "key_missing_link": "enter_gnomish_mines"}),
        forensics=fx,
    )
    a = su["foci"][0]["failure_attribution"]
    assert "overridden" not in a  # old behaviour byte-identical when no frontier


# ---- A4: gap gate park vs true style disease -----------------------------------------------------

CAP = {"frontier": "enter_gnomish_mines", "frontier_idx": 1, "reach_frac": 0.1855,
       "cond": 0.8053, "certified": True, "n_episodes": 1024, "reached_n": 190}


def test_certified_cap_parks_wall_to_watch(nb_path):
    nb = SiegeNotebook(nb_path)
    _open(nb, 1, "defeat_gnome_warrior")
    nb.note_access_caps({"defeat_gnome_warrior": dict(CAP)}, session_idx=10)
    st = nb.note_transfer_gap("defeat_gnome_warrior", 78.0, 18.0, session_idx=10)
    assert st.startswith("ACCESS_CAPPED(enter_gnomish_mines")
    assert "defeat_gnome_warrior" not in nb.focus_skills()
    w = nb.watch_registry()["defeat_gnome_warrior"]
    assert w["gap_stall"] == 0 and w["gap_forced"] is False  # counters frozen for a clean resume
    assert any(
        h.get("event") == "focus_parked_access_capped" for h in nb._nb["history"]
    )


def test_uncertified_cap_leaves_style_gate_armed(nb_path):
    # The gate still fires on an uncertified cap (no premature certified-park) — but v7fix5.2
    # revised the DEATH's destination: an uncertified cap whose reach < 35% is FRONTIER-STARVED
    # (the chain past the frontier was never assessable), so the decreed STYLE_REJECTED routes
    # to a PARK (watch, no registry strike), not a retirement. fix5.0 called this shape "true
    # style disease"; the fix51-run forensics falsified that reading — a REACHED-but-losing wall
    # has every link >= 35% and therefore NO cap at all (see the companion test below), which is
    # the path that still retires verbatim.
    nb = SiegeNotebook(nb_path)
    _open(nb, 1, "make_iron_armour")
    low = dict(CAP, cond=0.05, certified=False)
    nb.note_access_caps({"make_iron_armour": low}, session_idx=1)
    for i in range(GAP_FORCE_SESSIONS):
        st = nb.note_transfer_gap("make_iron_armour", 95.0, 15.0, session_idx=2 + i)
    assert st.startswith("FORCED_DEPTH")
    for i in range(GAP_STALL_PATIENCE - 1):
        nb.note_transfer_gap("make_iron_armour", 95.0, 16.0, session_idx=10 + i)
    st = nb.note_transfer_gap("make_iron_armour", 95.0, 16.0, session_idx=20)
    assert st == "STYLE_REJECTED->PARKED (frontier-starved)"
    snap = nb.snapshot()
    assert "make_iron_armour" in snap["watch"]
    assert "make_iron_armour" not in (snap.get("retired") or {})


def test_true_style_disease_no_cap_still_retires(nb_path):
    # reach HIGH = no frontier = no cap: the reached-but-losing wall retires exactly as before.
    nb = SiegeNotebook(nb_path)
    _open(nb, 1, "make_iron_armour")
    for i in range(GAP_FORCE_SESSIONS):
        st = nb.note_transfer_gap("make_iron_armour", 95.0, 15.0, session_idx=2 + i)
    assert st.startswith("FORCED_DEPTH")
    for i in range(GAP_STALL_PATIENCE - 1):
        nb.note_transfer_gap("make_iron_armour", 95.0, 16.0, session_idx=10 + i)
    st = nb.note_transfer_gap("make_iron_armour", 95.0, 16.0, session_idx=20)
    assert st == "STYLE_REJECTED"
    snap = nb.snapshot()
    assert "make_iron_armour" in (snap.get("retired") or {})
    assert "make_iron_armour" not in snap["watch"]


# ---- A5: watch-resume hold -----------------------------------------------------------------------

def test_capped_watcher_held_then_released(nb_path):
    nb = SiegeNotebook(nb_path)
    _open(nb, 1, "defeat_gnome_warrior")
    nb.note_access_caps({"defeat_gnome_warrior": dict(CAP)}, session_idx=10)
    nb.note_transfer_gap("defeat_gnome_warrior", 78.0, 18.0, session_idx=10)
    assert "defeat_gnome_warrior" in nb.watch_registry()
    # three flat readings = "stalled" — but the cap holds it in WATCH.
    prof = _mature_profile({"defeat_gnome_warrior": 18.0})
    for s in (11, 12, 13):
        nb._process_watch(s, prof)
    assert "defeat_gnome_warrior" in nb.watch_registry()
    assert "defeat_gnome_warrior" not in nb.focus_skills()
    # frontier opens -> cap disappears from the next feed -> resume re-engages.
    nb.note_access_caps({}, session_idx=14)
    nb._process_watch(14, prof)
    assert "defeat_gnome_warrior" in nb.focus_skills()
    assert "defeat_gnome_warrior" not in nb.watch_registry()


# ---- A6: expand-gate frontier exemption ----------------------------------------------------------

def test_expand_gate_waived_for_named_frontier(nb_path):
    nb = SiegeNotebook(nb_path)
    # a near-0% SHALLOW wall pinning condition (b) (kobold itself is deep-locked as ordinary).
    _open(nb, 1, "defeat_gnome_archer", sr=0.5)
    nb.note_access_caps(
        {"defeat_gnome_archer": dict(CAP, cond=0.005, certified=False)}, session_idx=2,
    )
    prof = _mature_profile({"defeat_gnome_archer": 0.5, "enter_gnomish_mines": 18.0})
    assert "enter_gnomish_mines" in nb.access_frontiers()
    assert nb._may_open_new_focus(prof, access_frontier=True) is True
    assert nb._may_open_new_focus(prof, access_frontier=False) is False  # old gate intact
    # capacity (a) still binds: force-fill every slot -> even a frontier is refused.
    while len(nb._nb["foci"]) < nb.th.max_focus:
        nb._nb["foci"].append({"skill": f"filler_wall_{len(nb._nb['foci'])}"})
    assert nb._may_open_new_focus(prof, access_frontier=True) is False


# ---- A7: persistence -----------------------------------------------------------------------------

def test_access_caps_survive_reload(nb_path):
    nb = SiegeNotebook(nb_path)
    nb.note_access_caps({"defeat_gnome_warrior": dict(CAP)}, session_idx=10)
    nb2 = SiegeNotebook(nb_path)
    assert nb2._access_cap("defeat_gnome_warrior")["frontier"] == "enter_gnomish_mines"
    assert nb2._nb["access_caps"]["session"] == 10
    assert "enter_gnomish_mines" in nb2.access_frontiers()
