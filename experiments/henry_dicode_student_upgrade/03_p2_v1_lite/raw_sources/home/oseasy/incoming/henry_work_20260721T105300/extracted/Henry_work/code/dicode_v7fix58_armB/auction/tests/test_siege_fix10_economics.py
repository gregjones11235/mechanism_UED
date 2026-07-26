"""Offline tests for the v6fix10 REACHABILITY ECONOMICS layer (user 2026-07-08).

fix9 post-mortem (job 3717871): defeat_gnome_warrior — a wall THREE closed doors deep (kobolds live in the
sewers, not the dungeon the LLM's chain named) — was auto-opened at s9, drilled at 95-100% trained /
0% held-out for 10 siege decisions, and monopolised the curriculum while its true door
(enter_dungeon, then the mines/sewers descent) was a naturally-learnable skill the baseline climbed
without any siege. The constitution these tests pin: the siege lifts skills OUT of the p~0 dead
zone into the learnability band; once in the band and moving, the normal curriculum owns them.

  ① DOOR GATE: a wall whose failures' top missing link is < door_min_sr held-out is opened AS the
     door (gateway_for=<wall>), on both the LLM-proposal and the auto-open path.
  ② YIELD-TO-MOMENTUM: two readings gaining >= yield_enter_pp combined inside the band -> WATCH
     (privileges withdraw, slot frees); stall -> resume with hysteresis; >= graduate_sr x2 -> maintenance.
  ③ ZERO-WIN DISCOUNT: zero_win_walls() flags foci with no held-out win in evidence.
  ④ ATTRIBUTION SHORTCUT: verified access-blocked attribution -> required_form == DEPTH.
  ⑤ HIGH-WATER: peaks >= highwater_sr are protected; a drop >= highwater_drop_pp flags FORGETTING.
  ⑥ relative interruption bound  ⑦ CHAIN-INCOMPLETE verdict + attribution rejection.

No jax/craftax/LLM needed.
"""

import pytest

from auction.chain_order_log import FRONTIER_MIN_FAILS, ChainOrderLog
from auction.modeler import Modeler
from auction.siege_notebook import (
    DOOR_MIN_SR,
    GATEWAY_RELEASE_SR,
    GRADUATE_CONSECUTIVE,
    GRADUATE_SR,
    HIGHWATER_DROP_PP,
    HIGHWATER_SR,
    MATURITY_MIN_MASTERED,
    MATURITY_MIN_SNAPSHOTS,
    MATURITY_SKILL_SR,
    RESUME_LOCK_READINGS,
    ZERO_WIN_MAX_SR,
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


_GNOME_FX = {"missing_top": [("enter_dungeon", 0.72), ("make_iron_sword", 0.4)]}


# ---- ① door gate ---------------------------------------------------------------------------------

def test_door_substitution_on_llm_open(nb_path):
    nb = SiegeNotebook(nb_path)
    prof = _mature_profile({"defeat_gnome_warrior": 0.0, "enter_dungeon": 0.0})
    _update(nb, 1, prof, foci=[{"skill": "defeat_gnome_warrior", "prereq_tree": []}],
            forensics={"defeat_gnome_warrior": _GNOME_FX})
    assert nb.focus_skills() == ["enter_dungeon"]
    foc = nb.foci()[0]
    assert foc["gateway_for"] == "defeat_gnome_warrior"
    assert "door_substituted(defeat_gnome_warrior->enter_dungeon" in nb.last_focus_decision
    assert nb.last_door_sub is not None


def test_wall_opens_itself_when_door_is_open(nb_path):
    nb = SiegeNotebook(nb_path)
    prof = _mature_profile({"defeat_gnome_warrior": 0.0, "enter_dungeon": DOOR_MIN_SR + 5})
    _update(nb, 1, prof, foci=[{"skill": "defeat_gnome_warrior", "prereq_tree": []}],
            forensics={"defeat_gnome_warrior": _GNOME_FX})
    assert nb.focus_skills() == ["defeat_gnome_warrior"]
    assert nb.foci()[0]["gateway_for"] is None


def test_door_scan_covers_top3_missing_links(nb_path):
    """v6fix10.1 hazard-3b: a closed door hiding at rank 2 behind an open rank-1 link (kobold s32:
    sword 39% at SR 57 over the sewers descent) must still substitute."""
    nb = SiegeNotebook(nb_path)
    prof = _mature_profile({
        "defeat_gnome_warrior": 0.0, "make_iron_sword": 57.0, "enter_gnomish_mines": 4.0,
    })
    fx = {"missing_top": [("make_iron_sword", 0.39), ("enter_gnomish_mines", 0.26)]}
    _update(nb, 1, prof, foci=[{"skill": "defeat_gnome_warrior", "prereq_tree": []}],
            forensics={"defeat_gnome_warrior": fx})
    assert nb.focus_skills() == ["enter_gnomish_mines"]
    assert nb.foci()[0]["gateway_for"] == "defeat_gnome_warrior"


# ---- v6fix10.1 hazard-3a: admission deferral (no forensics -> track before opening) --------------

def test_admission_deferred_without_forensics_then_ruled(nb_path):
    nb = SiegeNotebook(nb_path)
    prof = _mature_profile({"defeat_gnome_warrior": 0.0, "enter_dungeon": 0.0})
    tree = [{"skill": "enter_dungeon", "role": "descend"}]
    # forensics PROVIDED (dict) but the wall has none yet -> deferred, parked with its chain.
    _update(nb, 1, prof, foci=[{"skill": "defeat_gnome_warrior", "prereq_tree": tree}], forensics={})
    assert nb.focus_skills() == []
    assert "admission_deferred(defeat_gnome_warrior" in nb.last_focus_decision
    pending = nb.snapshot()["pending_track"]
    assert pending["defeat_gnome_warrior"]["links"] == ["enter_dungeon"]
    # the pending wall is chain-tracked (that IS the point of the waiting room)...
    assert nb.chain_targets()["defeat_gnome_warrior"] == ["enter_dungeon"]
    # ...and the journal says so.
    assert "PENDING ADMISSION" in nb.render_for_prompt()
    # next session forensics exist -> the door gate can rule (here: closed door -> substitution),
    # and the admitted wall leaves the waiting room only via _open_focus (gateway keeps it pending).
    _update(nb, 2, prof, foci=[{"skill": "defeat_gnome_warrior", "prereq_tree": tree}],
            forensics={"defeat_gnome_warrior": _GNOME_FX})
    assert nb.focus_skills() == ["enter_dungeon"]
    assert nb.foci()[0]["gateway_for"] == "defeat_gnome_warrior"
    assert "defeat_gnome_warrior" in nb.snapshot()["pending_track"]  # tracked through the gateway siege


def test_no_deferral_when_forensics_is_none(nb_path):
    """forensics=None = no chain log exists (old tests / degraded wiring): the deferral gate must
    stand down, else every proposal would wait forever for a tracker that cannot exist."""
    nb = SiegeNotebook(nb_path)
    prof = _mature_profile({"defeat_gnome_warrior": 0.0})
    _update(nb, 1, prof, foci=[{"skill": "defeat_gnome_warrior", "prereq_tree": []}])
    assert nb.focus_skills() == ["defeat_gnome_warrior"]


def test_auto_open_skips_unforensiced_candidate_and_parks_it(nb_path):
    nb = SiegeNotebook(nb_path)
    prof = _mature_profile({"defeat_gnome_warrior": 0.0, "defeat_skeleton": 5.0})
    # skeleton HAS forensics (open door) -> auto-open takes it; kobold (no forensics) is parked.
    _update(nb, 1, prof, ranked=[{"skill": "defeat_gnome_warrior"}, {"skill": "defeat_skeleton"}],
            forensics={"defeat_skeleton": {"missing_top": []}})
    assert nb.focus_skills() == ["defeat_skeleton"]
    pending = nb.snapshot()["pending_track"]
    assert "defeat_gnome_warrior" in pending and pending["defeat_gnome_warrior"]["links"] == []
    # a link-less pending entry cannot be chain-tracked; the journal demands a prereq_tree.
    assert "defeat_gnome_warrior" not in nb.chain_targets()
    assert "NO prereq_tree yet" in nb.render_for_prompt()


def test_auto_open_substitutes_door_and_gateway_blocks_wall(nb_path):
    nb = SiegeNotebook(nb_path)
    prof = _mature_profile({"defeat_gnome_warrior": 0.0, "enter_dungeon": 0.0})
    # no foci proposal -> auto-open path picks defeat_gnome_warrior (combat) but opens its DOOR.
    _update(nb, 1, prof, ranked=[{"skill": "defeat_gnome_warrior", "why": "stuck combat"}],
            forensics={"defeat_gnome_warrior": _GNOME_FX})
    assert nb.focus_skills() == ["enter_dungeon"]
    assert nb.foci()[0]["gateway_for"] == "defeat_gnome_warrior"
    assert "defeat_gnome_warrior" not in nb.focus_skills()


def test_gateway_blocks_wall_below_release_line(nb_path):
    """②'-4 (v6fix10.1): while the door sits below gateway_release_sr the wall stays OFF the
    auto-open menu, even with the door past door_min_sr and the expand gate satisfied."""
    nb = SiegeNotebook(nb_path)
    # s1: a decoy enabler focus opens (it will provide the expand condition later).
    _update(nb, 1, _mature_profile({"make_iron_armour": 5.0, "enter_dungeon": 0.0}),
            foci=[{"skill": "make_iron_armour", "prereq_tree": []}])
    # s2: decoy at 55 satisfies the expand gate -> auto-open picks kobold, substitutes its door.
    _update(nb, 2, _mature_profile({
        "make_iron_armour": GRADUATE_SR + 5, "defeat_gnome_warrior": 0.0, "enter_dungeon": 0.0,
    }), ranked=[{"skill": "defeat_gnome_warrior", "why": "stuck combat"}],
        forensics={"defeat_gnome_warrior": _GNOME_FX})
    assert "enter_dungeon" in nb.focus_skills()
    # s3: decoy dips below the line (resets its graduation counter), door creeps to 8.
    _update(nb, 3, _mature_profile({
        "make_iron_armour": GRADUATE_SR - 1, "defeat_gnome_warrior": 0.0, "enter_dungeon": 8.0,
    }))
    # s4: expand satisfied again, door at 17 — past door_min_sr but BELOW the release line:
    # the wall must stay gated.
    _update(nb, 4, _mature_profile({
        "make_iron_armour": GRADUATE_SR + 5, "defeat_gnome_warrior": 0.0,
        "enter_dungeon": GATEWAY_RELEASE_SR - 3,
    }), ranked=[{"skill": "defeat_gnome_warrior", "why": "door is opening"}],
        forensics={"defeat_gnome_warrior": {"missing_top": [("enter_dungeon", 0.7)]}})
    assert "defeat_gnome_warrior" not in nb.focus_skills()


def test_gateway_releases_wall_once_door_is_in_band(nb_path):
    """v6fix10.1 hazard-5: the auto-open block lifts at gateway_release_sr (default 20), NOT at
    full door graduation — a mid-band door plateau must not lock a gnome-type wall out forever."""
    nb = SiegeNotebook(nb_path)
    prof = _mature_profile({"defeat_gnome_warrior": 0.0, "enter_dungeon": 0.0})
    _update(nb, 1, prof, ranked=[{"skill": "defeat_gnome_warrior", "why": "stuck combat"}],
            forensics={"defeat_gnome_warrior": _GNOME_FX})
    assert nb.foci()[0]["gateway_for"] == "defeat_gnome_warrior"
    # drive the door into WATCH (2 -> 18 -> 35: strict double rise, sum >= 15, in band).
    for s, r in ((2, 18.0), (3, 35.0)):
        _update(nb, s, _mature_profile({"defeat_gnome_warrior": 0.0, "enter_dungeon": r}),
                forensics={"defeat_gnome_warrior": _GNOME_FX})
    assert "enter_dungeon" in nb.watch_registry()
    # door at 35 >= release line -> the wall is back on the menu, and its own door check now
    # passes (enter_dungeon 35 >= door_min_sr) -> kobold opens DIRECTLY.
    _update(nb, 4, _mature_profile({"defeat_gnome_warrior": 0.0, "enter_dungeon": 35.0}),
            ranked=[{"skill": "defeat_gnome_warrior", "why": "door cracked"}],
            forensics={"defeat_gnome_warrior": {"missing_top": [("enter_dungeon", 0.7)]}})
    assert "defeat_gnome_warrior" in nb.focus_skills()


# ---- ② yield-to-momentum -------------------------------------------------------------------------

def _drive_to_watch(nb, skill="enter_dungeon", readings=(2.0, 18.0, 35.0), start=1):
    """Open ``skill`` then feed ``readings``; the last one should trip the yield gate."""
    prof = _mature_profile({skill: readings[0]})
    _update(nb, start, prof, foci=[{"skill": skill, "prereq_tree": []}])
    assert skill in nb.focus_skills()
    s = start
    for r in readings[1:]:
        s += 1
        _update(nb, s, _mature_profile({skill: r}),
                foci=[{"skill": skill, "prereq_tree": []}])
    return s


def test_yield_on_fast_climb_in_band(nb_path):
    nb = SiegeNotebook(nb_path)
    last = _drive_to_watch(nb)  # 2 -> 18 -> 35: +33pp over 2 readings, SR in [20, 50)
    assert nb.focus_skills() == []
    assert "enter_dungeon" in nb.watch_registry()
    assert nb.last_yield is not None
    snap = nb.snapshot()
    assert any(h.get("event") == "focus_yielded_watch" for h in snap["history"])
    # ②'-2: the enabler budget froze with the watch (siege_sessions stops counting).
    spent = snap["watch"]["enter_dungeon"]["siege_sessions"]
    _update(nb, last + 1, _mature_profile({"enter_dungeon": 36.0}))
    assert nb.snapshot()["watch"]["enter_dungeon"]["siege_sessions"] == spent


def test_slow_true_wall_never_yields(nb_path):
    nb = SiegeNotebook(nb_path)
    # gnome_warrior-shaped climb (~1pp/reading) never sums to yield_enter_pp over 2 readings.
    prof = _mature_profile({"defeat_gnome_warrior": 20.0})
    _update(nb, 1, prof, foci=[{"skill": "defeat_gnome_warrior", "prereq_tree": []}])
    for i, r in enumerate((21.0, 22.0, 23.5, 24.0), start=2):
        _update(nb, i, _mature_profile({"defeat_gnome_warrior": r}),
                foci=[{"skill": "defeat_gnome_warrior", "prereq_tree": []}])
    assert nb.focus_skills() == ["defeat_gnome_warrior"]
    assert nb.watch_registry() == {}


def test_watch_resumes_on_stall_with_hysteresis(nb_path):
    nb = SiegeNotebook(nb_path)
    last = _drive_to_watch(nb)
    # two consecutive near-flat readings below graduate_sr -> resume into foci.
    _update(nb, last + 1, _mature_profile({"enter_dungeon": 36.0}))
    _update(nb, last + 2, _mature_profile({"enter_dungeon": 36.5}))
    assert nb.focus_skills() == ["enter_dungeon"]
    assert nb.last_resume is not None
    foc = nb.foci()[0]
    assert foc["resume_lock"] == RESUME_LOCK_READINGS
    assert foc["gap_forced"] is False and foc["gap_sessions"] == 0
    # ②'-5 hysteresis: a big jump during the lock does NOT re-yield.
    _update(nb, last + 3, _mature_profile({"enter_dungeon": 45.0}),
            foci=[{"skill": "enter_dungeon", "prereq_tree": []}])
    assert nb.focus_skills() == ["enter_dungeon"]


def test_watch_graduates_at_graduate_sr(nb_path):
    nb = SiegeNotebook(nb_path)
    last = _drive_to_watch(nb)
    for i in range(GRADUATE_CONSECUTIVE):
        _update(nb, last + 1 + i, _mature_profile({"enter_dungeon": GRADUATE_SR + 3 + i}))
    snap = nb.snapshot()
    assert "enter_dungeon" not in snap["watch"]
    assert "enter_dungeon" in snap["maintenance"]
    assert "enter_dungeon" in snap["protected_set"]


def test_watch_does_not_occupy_a_focus_slot(nb_path):
    nb = SiegeNotebook(nb_path)
    _drive_to_watch(nb)  # enter_dungeon parked in WATCH
    # a new (first-slot) focus opens freely: the watch entry holds no MAX_FOCUS slot.
    prof = _mature_profile({"make_iron_armour": 5.0})
    _update(nb, 10, prof, foci=[{"skill": "make_iron_armour", "prereq_tree": []}])
    assert nb.focus_skills() == ["make_iron_armour"]
    # the watched wall may not be re-proposed while momentum owns it.
    _update(nb, 11, _mature_profile({"make_iron_armour": 5.0, "enter_dungeon": 40.0}),
            foci=[{"skill": "make_iron_armour"}, {"skill": "enter_dungeon"}])
    assert "watching(enter_dungeon" in nb.last_focus_decision
    assert nb.focus_skills() == ["make_iron_armour"]


# ---- ③ zero-win discount -------------------------------------------------------------------------

def test_zero_win_walls_flags_only_zero_sr_foci(nb_path):
    nb = SiegeNotebook(nb_path)
    prof = _mature_profile({"defeat_gnome_warrior": 0.0, "defeat_gnome_archer": 4.0})
    _update(nb, 1, prof, foci=[{"skill": "defeat_gnome_warrior"}])
    zw = nb.zero_win_walls({k.lower(): v for k, v in prof.items()})
    assert zw == {"defeat_gnome_warrior"}
    # v6fix10.1 hazard-2: a FLUKE reading (one lucky episode out of 1024 = 0.098%) does NOT
    # ratchet the wall to full price — "no win" tolerates SR <= zero_win_max_sr.
    _update(nb, 2, _mature_profile({"defeat_gnome_warrior": ZERO_WIN_MAX_SR - 0.5}),
            foci=[{"skill": "defeat_gnome_warrior"}])
    assert nb.zero_win_walls({"defeat_gnome_warrior": 0.0}) == {"defeat_gnome_warrior"}
    # a REAL first breakthrough ratchets it out (best_sr records the win even if SR later reads 0).
    _update(nb, 3, _mature_profile({"defeat_gnome_warrior": 6.0}), foci=[{"skill": "defeat_gnome_warrior"}])
    assert nb.zero_win_walls({"defeat_gnome_warrior": 0.0}) == set()


# ---- ④ attribution -> forced DEPTH ---------------------------------------------------------------

def test_verified_access_blocked_attribution_forces_depth(nb_path):
    nb = SiegeNotebook(nb_path)
    prof = _mature_profile({"defeat_gnome_warrior": 0.0, "enter_dungeon": 0.0})
    attrib = {"class": "resource_shortfall", "key_missing_link": "enter_dungeon", "verified": True}
    foci = [{"skill": "defeat_gnome_warrior", "prereq_tree": [], "failure_attribution": attrib}]
    # forensics say the door is reachable enough to open the wall itself (no substitution) so the
    # shortcut is what forces DEPTH here.
    _update(nb, 1, prof, foci=foci)
    assert nb.focus_skills() == ["defeat_gnome_warrior"]
    # opening session = diagnostic drill allowed (siege_sessions == 0).
    assert nb.required_form("defeat_gnome_warrior") is None
    _update(nb, 2, prof, foci=foci)
    assert nb.required_form("defeat_gnome_warrior") == "DEPTH"
    # v6fix10.1 hazard-1: the operative shortcut LATCHES gap_forced, arming the P3 early-stop —
    # ④ stops drills from decision 2, so trained>=90 readings can never accumulate and the gap
    # gate would never latch on its own, leaving a ④-path COMBAT wall with no early-stop exit.
    foc = nb.foci()[0]
    assert foc["gap_forced"] is True
    assert any(h.get("event") == "attrib_shortcut_armed_earlystop"
               for h in nb.snapshot()["history"])
    assert nb.last_attrib_arm is not None


def test_attrib_shortcut_path_reaches_style_rejected(nb_path):
    """v6fix10.1 hazard-1 end-to-end: a ④-short-circuited COMBAT wall whose held-out stays flat
    under forced DEPTH is retired STYLE_REJECTED after the (doubled) COMBAT patience — the fuse
    that saved fix9's kobold at s29 now exists on the shortcut path too."""
    nb = SiegeNotebook(nb_path)
    prof = _mature_profile({"defeat_gnome_warrior": 0.0, "enter_dungeon": 0.0})
    attrib = {"class": "resource_shortfall", "key_missing_link": "enter_dungeon", "verified": True}
    foci = [{"skill": "defeat_gnome_warrior", "prereq_tree": [], "failure_attribution": attrib}]
    _update(nb, 1, prof, foci=foci)
    _update(nb, 2, prof, foci=foci)  # shortcut goes operative -> gap_forced latched
    assert nb.foci()[0]["gap_forced"] is True
    # flat held-out readings under forced DEPTH: first call sets the baseline, then each flat
    # decision counts stall; COMBAT patience = 2 x gap_stall_patience.
    from auction.siege_notebook import GAP_STALL_PATIENCE
    status = nb.note_transfer_gap("defeat_gnome_warrior", None, 0.0, session_idx=3)
    assert "FORCED_DEPTH" in status
    result = None
    for i in range(2 * GAP_STALL_PATIENCE):
        result = nb.note_transfer_gap("defeat_gnome_warrior", None, 0.0, session_idx=4 + i)
        if result == "STYLE_REJECTED":
            break
    assert result == "STYLE_REJECTED"
    assert nb.focus_skills() == []
    assert "defeat_gnome_warrior" in nb.snapshot()["retired"]


# ---- ⑤ high-water protection ---------------------------------------------------------------------

def test_highwater_ratchet_protects_and_flags_forgetting(nb_path):
    nb = SiegeNotebook(nb_path)
    peak = HIGHWATER_SR + 8.0
    # v6fix10.1 hazard-4: ONE reading >= the bar is a candidate, not a peak — confirmation
    # takes two consecutive readings (their MIN is what the skill actually HELD).
    _update(nb, 1, _mature_profile({"make_iron_sword": peak}))
    assert "make_iron_sword" not in nb.protected_set()
    _update(nb, 2, _mature_profile({"make_iron_sword": peak + 2}))
    assert "make_iron_sword" in nb.protected_set()
    assert nb.snapshot()["highwater"]["make_iron_sword"] == peak  # min of the confirming pair
    # a slide below peak - drop_pp flags FORGETTING; a smaller slide does not.
    small = {"make_iron_sword": peak - HIGHWATER_DROP_PP + 2}
    big = {"make_iron_sword": peak - HIGHWATER_DROP_PP - 1}
    assert nb.highwater_forgetting(_mature_profile(small)) == set()
    assert nb.highwater_forgetting(_mature_profile(big)) == {"make_iron_sword"}


def test_highwater_single_spike_does_not_poison_the_ratchet(nb_path):
    """v6fix10.1 hazard-4: a lockhole-burst overshoot (61 for one snapshot, settling at 45) must
    NOT enter the registry — else 45 <= 61-15 reads as permanent phantom FORGETTING and rehearsal
    pumps a level the student never sustained."""
    nb = SiegeNotebook(nb_path)
    _update(nb, 1, _mature_profile({"open_chest": HIGHWATER_SR + 1}))
    _update(nb, 2, _mature_profile({"open_chest": HIGHWATER_SR - 15}))
    assert "open_chest" not in nb.snapshot()["highwater"]
    assert nb.highwater_forgetting(_mature_profile({"open_chest": HIGHWATER_SR - 15})) == set()
    # the pending candidate was cleared by the sub-bar reading: a LATER genuine two-reading hold
    # still enters (the ratchet is delayed, never disabled).
    _update(nb, 3, _mature_profile({"open_chest": HIGHWATER_SR + 3}))
    _update(nb, 4, _mature_profile({"open_chest": HIGHWATER_SR + 5}))
    assert nb.snapshot()["highwater"]["open_chest"] == HIGHWATER_SR + 3


# ---- ⑦ chain-incomplete: notebook gate -----------------------------------------------------------

def test_chain_incomplete_blocks_open_and_auto_open(nb_path):
    nb = SiegeNotebook(nb_path)
    prof = _mature_profile({"defeat_gnome_warrior": 0.0})
    _update(nb, 1, prof, foci=[{"skill": "defeat_gnome_warrior"}], incomplete={"defeat_gnome_warrior"})
    assert nb.focus_skills() == []
    assert "chain_incomplete(defeat_gnome_warrior" in nb.last_focus_decision
    _update(nb, 2, prof, ranked=[{"skill": "defeat_gnome_warrior"}], incomplete={"defeat_gnome_warrior"})
    assert nb.focus_skills() == []


# ---- ⑦/⑥ chain log + attribution gate ------------------------------------------------------------

def _fabricate_entry(log, target, links, missing, n_fail=1024, inv=None, **extra):
    entry = {"session": 1, "target": target, "links": links, "n_fail": n_fail, "n_succ": 0,
             "mean_depth": float(len(links)), "last_link": {links[-1]: n_fail},
             "missing": missing}
    if inv is not None:
        entry["inv"] = inv
    entry.update(extra)
    log._fail_hist.append(entry)


def test_chain_incomplete_detection(tmp_path):
    log = ChainOrderLog(str(tmp_path / "chain.json"))
    links = ["enter_dungeon", "make_iron_sword"]
    # tail reached by ~99% of failures, zero wins, no inventory gap -> incomplete.
    _fabricate_entry(log, "defeat_gnome_warrior", links, {"make_iron_sword": 10})
    assert log.chain_incomplete("defeat_gnome_warrior") is True
    hint = log.render_chain_hint("defeat_gnome_warrior")
    assert "MISSING an unnamed prerequisite" in hint
    # an inventory gap explains the losses -> NOT chain-incomplete (resource disease instead).
    log2 = ChainOrderLog(str(tmp_path / "chain2.json"))
    _fabricate_entry(log2, "make_iron_armour", links, {"make_iron_sword": 10},
                     inv={"iron": {"succ_med": 3, "fail_med": 1, "ready_frac": 0.1}})
    assert log2.chain_incomplete("make_iron_armour") is False
    # v6fix10.1 hazard-2: a REAL win rate (> _CHAIN_INCOMPLETE_WIN_EPS) -> the chain CAN be
    # sufficient -> not incomplete; a FLUKE win (1-in-1024, rate ~0.1%) does NOT disable the
    # verdict — sewers-line walls sit at exactly that epsilon while still effectively unreached.
    log3 = ChainOrderLog(str(tmp_path / "chain3.json"))
    entry3 = {"session": 1, "target": "defeat_gnome_warrior", "links": links, "n_fail": 994,
              "n_succ": 30, "mean_depth": 2.0, "last_link": {links[-1]: 994},
              "missing": {"make_iron_sword": 10}}
    log3._fail_hist.append(entry3)
    assert log3.chain_incomplete("defeat_gnome_warrior") is False
    log3b = ChainOrderLog(str(tmp_path / "chain3b.json"))
    entry3b = dict(entry3, n_fail=1023, n_succ=1, last_link={links[-1]: 1023})
    log3b._fail_hist.append(entry3b)
    assert log3b.chain_incomplete("defeat_gnome_warrior") is True
    # the sample-size guard.
    log4 = ChainOrderLog(str(tmp_path / "chain4.json"))
    _fabricate_entry(log4, "defeat_gnome_warrior", links, {}, n_fail=FRONTIER_MIN_FAILS - 1)
    assert log4.chain_incomplete("defeat_gnome_warrior") is False


def test_forensics_reports_ambient_and_incomplete(tmp_path):
    log = ChainOrderLog(str(tmp_path / "chain.json"))
    links = ["enter_dungeon", "make_iron_sword"]
    _fabricate_entry(log, "defeat_gnome_warrior", links, {"make_iron_sword": 10},
                     died_frac=0.95, after_deepest_med=200)
    _fabricate_entry(log, "defeat_gnome_warrior", links, {"make_iron_sword": 12},
                     died_frac=0.96, after_deepest_med=400)
    fx = log.forensics("defeat_gnome_warrior")
    assert fx["chain_incomplete"] is True
    assert fx["after_deepest_ambient_med"] == 200  # _median_int takes the lower middle of [200, 400]


def _siege_raw(cls, key=None):
    return {"siege_update": {"foci": [{
        "skill": "defeat_gnome_warrior", "prereq_tree": [],
        "failure_attribution": {"class": cls, "key_missing_link": key},
    }]}}


def test_attribution_rejects_shortfall_on_incomplete_chain():
    fx = {"defeat_gnome_warrior": {
        "missing_top": [("make_iron_sword", 0.4)], "break_at_final": True,
        "chain_incomplete": True, "inv_gaps": [],
    }}
    su = Modeler._validate_siege(_siege_raw("resource_shortfall"), forensics=fx)
    a = su["foci"][0]["failure_attribution"]
    assert a["class"] == "unknown" and a["rejected"] == "resource_shortfall"
    assert su["attrib_violations"]
    # chain_unreached naming a NEW prerequisite (outside the histogram) is ACCEPTED — that is
    # exactly the answer the chain-incomplete verdict asks for.
    su2 = Modeler._validate_siege(_siege_raw("chain_unreached", key="enter_sewers"), forensics=fx)
    a2 = su2["foci"][0]["failure_attribution"]
    assert a2["class"] == "chain_unreached" and a2["key_missing_link"] == "enter_sewers"
    assert a2["verified"] is True


def test_attribution_interrupt_bound_is_relative():
    base = {"missing_top": [], "break_at_final": False, "chain_incomplete": False, "inv_gaps": []}
    # ambient 400 -> bound max(30, 120) = 120: a 200-step survival REFUTES interruption...
    fx = {"defeat_gnome_warrior": {**base, "after_deepest_med": 200, "after_deepest_ambient_med": 400}}
    su = Modeler._validate_siege(_siege_raw("interrupted_by_combat"), forensics=fx)
    assert su["foci"][0]["failure_attribution"]["class"] == "unknown"
    # ...while 100 steps (< 120) is admissible under the same ambient.
    fx2 = {"defeat_gnome_warrior": {**base, "after_deepest_med": 100, "after_deepest_ambient_med": 400}}
    su2 = Modeler._validate_siege(_siege_raw("interrupted_by_combat"), forensics=fx2)
    assert su2["foci"][0]["failure_attribution"]["class"] == "interrupted_by_combat"
