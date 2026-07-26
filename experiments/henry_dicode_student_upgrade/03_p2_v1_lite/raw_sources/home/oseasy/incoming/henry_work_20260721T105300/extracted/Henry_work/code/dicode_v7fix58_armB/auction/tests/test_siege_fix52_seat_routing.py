"""Offline tests for the v7fix5.2 seat-routing layer (retire->park + access-root nomination).

fix5.2 root cause (fix51 run s228-243, 2026-07-15): fix5.0's therapy never started — watch stayed
empty (the cond>=0.6 certificate cannot certify multi-link deep walls, so their seats could not be
freed), the one free seat went to learn_fireball against the rendered BINDING-ACCESS instruction
(healthy modeler, 8/8 ignored), and the s238 budget retirement was gateway-refilled the same
boundary. These tests pin the two transmissions (NO new judgement was added):

  P0  _retire_or_park: a death already decreed by the existing judges routes to a PARK when the
      wall is frontier-STARVED (cap exists, not certified) — full campaign state preserved, no
      retired-registry strike, no cooldown; certified-cap and no-cap deaths retire byte-identically
      to fix5.1. A parked watcher is HELD until its park frontier moves +focus_improve_pp; the
      relay slot and the DISCOUNT share free structurally (the wall left ``foci``).
  P1  _access_auto_nominate: chase caps to the transitive ROOT (fireball->mines->armour shape),
      nominate it opened_by=access_auto THROUGH the existing admission gates (a blacklist veto
      stands down, never bypasses); an enter_* root opens as an ACCESS-LINK relay only while the
      relay slot is free, else as an ordinary DEPTH siege; a fresh run (no caps) is inert.

No jax/craftax world, no LLM needed.
"""

import pytest

from auction.siege_notebook import (
    ENABLER_MAX_SESSIONS,
    FOCUS_IMPROVE_PP,
    MATURITY_MIN_SNAPSHOTS,
    SiegeNotebook,
)
from auction.tests.test_siege_fix8_economics import _mature_profile, _open


@pytest.fixture
def nb_path(tmp_path):
    return str(tmp_path / "siege_notebook.json")


def _starved_cap(frontier="enter_gnomish_mines", reach=0.21, cond=0.01):
    return {"frontier": frontier, "frontier_idx": 1, "reach_frac": reach, "cond": cond,
            "certified": False, "n_episodes": 1024, "reached_n": 218}


def _certified_cap(frontier="enter_gnomish_mines"):
    return {"frontier": frontier, "frontier_idx": 1, "reach_frac": 0.21, "cond": 0.81,
            "certified": True, "n_episodes": 1024, "reached_n": 190}


ARMOUR_TREE = [{"skill": "enter_gnomish_mines", "role": "access"}]


def _burn_budget(nb, wall, profile_extra, first_session=2):
    """Run enough siege decisions to exhaust the enabler budget (or L4-stall) of ``wall``."""
    s = first_session
    for _ in range(ENABLER_MAX_SESSIONS + 4):
        if wall not in nb.focus_skills():
            break
        nb.apply_llm_update(
            s, _mature_profile(profile_extra),
            {"foci": [{"skill": wall, "prereq_tree": ARMOUR_TREE}]},
            num_snapshots=MATURITY_MIN_SNAPSHOTS,
        )
        s += 1
    return s


# ---- P0: routing ----------------------------------------------------------------------------------

def test_budget_death_of_starved_wall_parks(nb_path):
    nb = SiegeNotebook(nb_path)
    _open(nb, 1, "make_diamond_armour", sr=0.0, tree=ARMOUR_TREE)
    # uncertified cap = frontier-starved; frontier held saturated so P1 stands down (scope veto)
    nb.note_access_caps({"make_diamond_armour": _starved_cap()}, 1)
    _burn_budget(nb, "make_diamond_armour",
                 {"make_diamond_armour": 0.0, "enter_gnomish_mines": 85.0})
    snap = nb.snapshot()
    assert "make_diamond_armour" not in nb.focus_skills()
    # parked, not retired: no registry strike, no cooldown to outwait, full state in watch.
    assert "make_diamond_armour" in snap["watch"]
    assert "make_diamond_armour" not in (snap.get("retired") or {})
    w = snap["watch"]["make_diamond_armour"]
    assert w["park_event"] in ("focus_retired_budget", "focus_retired_stalled")
    assert w["park_frontier"] == "enter_gnomish_mines"
    assert any(h.get("event") == "focus_parked_frontier_starved"
               for h in snap["history"])


def test_no_cap_death_retires_exactly_as_before(nb_path):
    nb = SiegeNotebook(nb_path)
    _open(nb, 1, "make_diamond_armour", sr=0.0, tree=ARMOUR_TREE)
    _burn_budget(nb, "make_diamond_armour",
                 {"make_diamond_armour": 0.0, "enter_gnomish_mines": 85.0})
    snap = nb.snapshot()
    assert "make_diamond_armour" in (snap.get("retired") or {})
    assert "make_diamond_armour" not in snap["watch"]


def test_certified_cap_death_takes_the_old_path_too(nb_path):
    # park-on-death is for STARVED walls only; a certified cap that somehow reaches a death
    # verdict retires through the unchanged machinery (fix5.0 parks it earlier, at the gap gate).
    nb = SiegeNotebook(nb_path)
    _open(nb, 1, "make_diamond_armour", sr=0.0, tree=ARMOUR_TREE)
    nb.note_access_caps({"make_diamond_armour": _certified_cap()}, 1)
    _burn_budget(nb, "make_diamond_armour",
                 {"make_diamond_armour": 0.0, "enter_gnomish_mines": 85.0})
    snap = nb.snapshot()
    assert "make_diamond_armour" in (snap.get("retired") or {})
    assert "make_diamond_armour" not in snap["watch"]


def test_parked_wall_held_until_frontier_moves(nb_path):
    nb = SiegeNotebook(nb_path)
    _open(nb, 1, "make_diamond_armour", sr=0.0, tree=ARMOUR_TREE)
    nb.note_access_caps({"make_diamond_armour": _starved_cap()}, 1)
    # frontier live at 20% (recorded into link_best -> the park snapshot)
    s = _burn_budget(nb, "make_diamond_armour",
                     {"make_diamond_armour": 0.0, "enter_gnomish_mines": 20.0})
    assert "make_diamond_armour" in nb.snapshot()["watch"]
    base = nb.snapshot()["watch"]["make_diamond_armour"]["park_frontier_sr"]
    assert base == pytest.approx(20.0)
    # frontier crawling below the +focus_improve_pp bar -> HELD in watch every session
    for i in range(4):
        nb.apply_llm_update(
            s + i, _mature_profile({"make_diamond_armour": 0.0,
                                    "enter_gnomish_mines": 20.0 + FOCUS_IMPROVE_PP / 2}),
            {"foci": []}, num_snapshots=MATURITY_MIN_SNAPSHOTS,
        )
        assert "make_diamond_armour" in nb.snapshot()["watch"]
    # frontier MOVED (+focus_improve_pp and change) -> resumes into a free seat, bookkeeping gone
    nb.apply_llm_update(
        s + 4, _mature_profile({"make_diamond_armour": 0.0,
                                "enter_gnomish_mines": 20.0 + FOCUS_IMPROVE_PP + 2}),
        {"foci": []}, num_snapshots=MATURITY_MIN_SNAPSHOTS,
    )
    assert "make_diamond_armour" in nb.focus_skills()
    foc = [f for f in nb.snapshot()["foci"] if f["skill"] == "make_diamond_armour"][0]
    assert "park_event" not in foc and "park_frontier" not in foc


def test_relay_stall_death_parks_with_ladder_state(nb_path):
    nb = SiegeNotebook(nb_path)
    # an explicit relay proposal opens the (single) relay slot campaign
    nb.apply_llm_update(
        1, _mature_profile({"defeat_kobold": 0.0}),
        {"foci": [{"skill": "defeat_kobold", "prereq_tree": ARMOUR_TREE,
                   "relay_r0_floor": 3}]},
        num_snapshots=MATURITY_MIN_SNAPSHOTS,
    )
    foc = [f for f in nb._nb["foci"] if f["skill"] == "defeat_kobold"][0]
    assert isinstance(foc.get("relay"), dict)
    nb.note_access_caps({"defeat_kobold": _starved_cap()}, 1)
    # the rung machinery decrees a relay-stall death -> the router parks instead
    parked = nb._retire_or_park(
        foc, 2, "focus_retired_relay_stalled", relay_spawn_floor=2, relay_sub_stage=3,
    )
    nb._nb["foci"] = [f for f in nb._nb["foci"] if f is not foc]
    nb._save()
    assert parked is True
    snap = nb.snapshot()
    w = snap["watch"]["defeat_kobold"]
    # the FULL ladder state hibernates with the wall (this is what dissolves the
    # "how do we correctly get back to kobold" problem)
    assert isinstance(w.get("relay"), dict) and w["relay"].get("r0_floor") == 3
    assert "defeat_kobold" not in (snap.get("retired") or {})
    # DISCOUNT share freed structurally: zero-win scan covers active foci only
    assert "defeat_kobold" not in nb.zero_win_walls({"defeat_kobold": 0.0})
    # relay slot freed structurally: a new relay proposal opens (frontier saturated so the
    # access-root nomination stands down and cannot take the slot first)
    nb.apply_llm_update(
        3, _mature_profile({"defeat_lizard": 0.0, "enter_gnomish_mines": 85.0}),
        {"foci": [{"skill": "defeat_lizard", "prereq_tree": ARMOUR_TREE,
                   "relay_r0_floor": 3}]},
        num_snapshots=MATURITY_MIN_SNAPSHOTS,
    )
    lf = [f for f in nb._nb["foci"] if f["skill"] == "defeat_lizard"]
    assert lf and isinstance(lf[0].get("relay"), dict)


# ---- P1: root chase + nomination ------------------------------------------------------------------

def test_chase_reaches_transitive_root(nb_path):
    nb = SiegeNotebook(nb_path)
    nb.note_access_caps({
        "learn_fireball": _starved_cap(frontier="enter_gnomish_mines"),
        "enter_gnomish_mines": _certified_cap(frontier="make_iron_armour"),
    }, 1)
    # the fix51-run shape: fireball capped on mines, mines CERTIFIED-capped on armour ->
    # nominating mines would be re-parked next boundary; only the root is attackable.
    assert nb._access_root_of("learn_fireball") == "make_iron_armour"
    assert nb._access_root_of("enter_gnomish_mines") == "make_iron_armour"
    assert nb._access_root_of("make_iron_armour") is None  # a root has no frontier


def test_chase_cycle_guard(nb_path):
    nb = SiegeNotebook(nb_path)
    nb.note_access_caps({
        "a_wall": _starved_cap(frontier="b_wall"),
        "b_wall": _starved_cap(frontier="a_wall"),
    }, 1)
    # a malformed cyclic feed terminates (degrades to "deepest reached"), never hangs
    assert nb._access_root_of("a_wall") in ("a_wall", "b_wall")


def test_access_auto_opens_ordinary_root(nb_path):
    nb = SiegeNotebook(nb_path)
    nb.note_access_caps({
        "learn_fireball": _starved_cap(frontier="enter_gnomish_mines"),
        "enter_gnomish_mines": _certified_cap(frontier="make_iron_armour"),
    }, 4)
    nb.apply_llm_update(
        5, _mature_profile({"make_iron_armour": 15.0}), {"foci": []},
        num_snapshots=MATURITY_MIN_SNAPSHOTS,
    )
    assert "make_iron_armour" in nb.focus_skills()
    ev = [h for h in nb.snapshot()["history"] if h.get("event") == "access_auto_opened"]
    assert ev and ev[-1]["focus"] == "make_iron_armour" and ev[-1]["form"] == "ordinary"


def test_access_auto_opens_enter_root_as_relay_when_slot_free(nb_path):
    nb = SiegeNotebook(nb_path)
    nb.note_access_caps({"defeat_kobold": _starved_cap(frontier="enter_gnomish_mines")}, 4)
    nb.apply_llm_update(
        5, _mature_profile({"enter_gnomish_mines": 19.0}), {"foci": []},
        num_snapshots=MATURITY_MIN_SNAPSHOTS,
    )
    foc = [f for f in nb._nb["foci"] if f["skill"] == "enter_gnomish_mines"]
    assert foc, "the enter_* root must open"
    # ACCESS-LINK relay form: R0 = native floor - 1 (the descent IS the skill)
    assert isinstance(foc[0].get("relay"), dict) and foc[0]["relay"]["r0_floor"] == 1


def test_access_auto_enter_root_falls_back_to_ordinary_when_relay_slot_busy(nb_path):
    nb = SiegeNotebook(nb_path)
    # the single relay slot is held by an active campaign
    nb.apply_llm_update(
        1, _mature_profile({"defeat_kobold": 0.0}),
        {"foci": [{"skill": "defeat_kobold", "prereq_tree": ARMOUR_TREE,
                   "relay_r0_floor": 3}]},
        num_snapshots=MATURITY_MIN_SNAPSHOTS,
    )
    nb.note_access_caps({"defeat_kobold": _starved_cap(frontier="enter_gnomish_mines")}, 1)
    nb.apply_llm_update(
        2, _mature_profile({"defeat_kobold": 0.0, "enter_gnomish_mines": 19.0}),
        {"foci": [{"skill": "defeat_kobold", "prereq_tree": ARMOUR_TREE,
                   "relay_r0_floor": 3}]},
        num_snapshots=MATURITY_MIN_SNAPSHOTS,
    )
    foc = [f for f in nb._nb["foci"] if f["skill"] == "enter_gnomish_mines"]
    assert foc, "slot busy must NOT block the root — it opens as an ordinary DEPTH siege"
    assert foc[0].get("relay") is None  # never steals the active relay's slot (N3)
    kob = [f for f in nb._nb["foci"] if f["skill"] == "defeat_kobold"]
    assert kob and isinstance(kob[0].get("relay"), dict)  # the incumbent is untouched


def test_access_auto_respects_blacklist_veto(nb_path):
    nb = SiegeNotebook(nb_path)
    nb._nb["retired"]["make_iron_armour"] = {
        "count": 2, "failed_notes": [], "last_session": -100,
        "sr_at_retirement": 15.0, "link_sr_at_retirement": {"collect_iron": 80.0},
    }
    nb._save()
    nb.note_access_caps(
        {"enter_gnomish_mines": _certified_cap(frontier="make_iron_armour")}, 4)
    nb.apply_llm_update(
        5, _mature_profile({"make_iron_armour": 15.0, "collect_iron": 80.0}),
        {"foci": []}, num_snapshots=MATURITY_MIN_SNAPSHOTS,
    )
    # blacklisted with no new chain evidence -> stand down with an audit note; NO bypass.
    assert "make_iron_armour" not in nb.focus_skills()
    assert "access_auto_vetoed" in (nb.last_access_auto or "")
    # the escape hatch is the existing one: a link moving +focus_improve_pp re-admits it.
    nb.apply_llm_update(
        6, _mature_profile({"make_iron_armour": 15.0,
                            "collect_iron": 80.0 + FOCUS_IMPROVE_PP + 1}),
        {"foci": []}, num_snapshots=MATURITY_MIN_SNAPSHOTS,
    )
    assert "make_iron_armour" in nb.focus_skills()


def test_access_auto_never_evicts_and_stands_down_when_full(nb_path):
    from auction.siege_notebook import _empty_focus

    nb = SiegeNotebook(nb_path)
    # fill every seat directly (the expand gate is not under test here)
    for wall in ["defeat_zombie", "defeat_skeleton", "eat_bat"]:
        nb._nb["foci"].append(_empty_focus(wall, 1, 8.0, opened_by="llm"))
    nb._save()
    before = set(nb.focus_skills())
    assert len(before) == 3  # max_focus seats all taken
    nb.note_access_caps({"defeat_kobold": _starved_cap(frontier="enter_gnomish_mines")}, 3)
    nb.apply_llm_update(
        4, _mature_profile({w: 8.0 for w in before} | {"enter_gnomish_mines": 19.0}),
        {"foci": [{"skill": w} for w in before]},
        num_snapshots=MATURITY_MIN_SNAPSHOTS,
    )
    assert set(nb.focus_skills()) == before  # nobody was evicted
    assert "access_auto_vetoed" in (nb.last_access_auto or "")


def test_fresh_run_no_caps_is_inert(nb_path):
    nb = SiegeNotebook(nb_path)
    nb.apply_llm_update(
        1, _mature_profile({}), {"foci": []}, num_snapshots=MATURITY_MIN_SNAPSHOTS,
    )
    assert nb.focus_skills() == []
    assert nb.last_access_auto is None
    assert nb.last_park is None


def test_parked_state_survives_reload(nb_path):
    nb = SiegeNotebook(nb_path)
    _open(nb, 1, "make_diamond_armour", sr=0.0, tree=ARMOUR_TREE)
    nb.note_access_caps({"make_diamond_armour": _starved_cap()}, 1)
    _burn_budget(nb, "make_diamond_armour",
                 {"make_diamond_armour": 0.0, "enter_gnomish_mines": 85.0})
    assert "make_diamond_armour" in nb.snapshot()["watch"]
    nb2 = SiegeNotebook(nb_path)  # reload through _coerce
    w = nb2.snapshot()["watch"].get("make_diamond_armour")
    assert w and w.get("park_frontier") == "enter_gnomish_mines"
