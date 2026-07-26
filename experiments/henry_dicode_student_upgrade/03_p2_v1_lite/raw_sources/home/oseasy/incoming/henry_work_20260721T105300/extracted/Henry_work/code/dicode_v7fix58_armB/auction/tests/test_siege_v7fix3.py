"""v7fix3 — relay unlock (P0-P3) + ecology economics (P4-P6). Offline tests.

Post-mortem being fixed (jobs 3813092 / 3812896, 2026-07-10):
  A) pigman (tier-4) opened as an ORDINARY focus and triple-locked the relay: no tier gate on the
     normal-open path (A1), `kept()` swallowed relay_r0_floor re-proposals for active walls (A2),
     and the expand gate's "any focus >= 50%" condition blocked new relay campaigns behind a 0%
     focus (A3).
  B) fix11's make_iron_armour focus (mid-band gear, never zero-win) monopolised generation AND
     training (24/24 tagged at s27, BREADTH 5/276, unthrottled force-activation) and starved the
     whole INTERMEDIATE tier; v7fix2's only effective breadth mechanism was 33 deep-spawn levels
     that slipped through the validator's foci-empty early-exit.

These tests pin: P1 tier-4 relay-only admission (LLM path + auto-open menu), P2 relay attach
(upgrade in place, zero-win/capacity/range guards, state resets), P3 relay exemption from the
expand gate, P0 the journal's zero-win upgrade hint, P5 the full-price force-activation cap, and
P6 the breadth spawn frontier (notebook state machine + always-on R6 with the BREADTH lane).
gen_manager wiring tests (@needs_jax) cover P4's ecology directive / role split / role quota and
P6's quota drop + frontier sweep.

No jax/craftax/LLM needed except the @needs_jax block (Oscar full suite only).
"""

import importlib.util
import os
import sys
import threading
import types

import networkx as nx
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for p in (os.path.join(_REPO, "src"), _REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from auction.craftax_achievements import (  # noqa: E402
    ALL_ACHIEVEMENTS,
    DEPTH_TIERS,
    MAX_DUNGEON_FLOOR,
    tier_of,
)
from auction.level_validator import RULE_SPAWN, validate_level  # noqa: E402
from auction.siege_notebook import (  # noqa: E402
    BREADTH_FRONTIER_SR,
    BREADTH_SPAWN_QUOTA,
    FOCUS_FORCE_CAP,
    MATURITY_MIN_MASTERED,
    MATURITY_MIN_SNAPSHOTS,
    MATURITY_SKILL_SR,
    SiegeNotebook,
    SiegeThresholds,
)

T4_WALL = "defeat_pigman"          # tier-4 (habitat floor 6): relay-only via the P1 tier gate
# v7fix4: kobold/lizard/troll are HABITAT floor-3+ walls — the fix4 deep lock makes them
# relay-only, so the "ordinary tier-3 wall" role in these tests moves to the gnome class
# (habitat floor 2 — fix8's actual winning walls, deliberately NOT deep-locked).
T3_WALL = "defeat_gnome_warrior"   # tier-3, floor 2: stays openable the ordinary way
T3_WALL2 = "defeat_gnome_archer"   # tier-3, floor 2: second proposal in expand-gate tests
T3_R0 = 2                          # the gnome walls' habitat floor (anchored relay R0)
DEEP_WALL = "defeat_kobold"        # tier-3 but habitat floor 3: fix4 deep-locked (relay-only)


def _mature_profile(extra: dict | None = None) -> dict:
    prof = {f"filler_{i}": MATURITY_SKILL_SR + 10 for i in range(MATURITY_MIN_MASTERED)}
    if extra:
        prof.update(extra)
    return prof


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


def _open_ordinary(nb, wall=T3_WALL, session=1, sr=0.0):
    """Open ``wall`` as a NORMAL focus (forensics supplied so the deferral doesn't park it;
    no missing link so the door gate stays quiet)."""
    prof = _mature_profile({wall: sr})
    _update(nb, session, prof, foci=[{"skill": wall, "prereq_tree": []}],
            forensics={wall: {"missing_top": []}})
    return prof


@pytest.fixture
def nb_path(tmp_path):
    return str(tmp_path / "siege_notebook.json")


# ---- tier_of (P1 substrate) -----------------------------------------------------------------------


def test_tier_of_covers_all_achievements_and_matches_depth_tiers():
    for tier, names in DEPTH_TIERS.items():
        for n in names:
            assert tier_of(n) == tier
    assert len(ALL_ACHIEVEMENTS) == 67
    assert tier_of("defeat_pigman") == 4 and tier_of("enter_fire_realm") == 4
    assert tier_of("defeat_kobold") == 3
    assert tier_of("smelt_iron") == 0        # hallucinated name -> 0, never gated as tier-4
    assert tier_of(None) == 0 and tier_of(123) == 0


# ---- P1: tier-4 walls are relay-only --------------------------------------------------------------


def test_tier4_ordinary_proposal_is_tier_locked(nb_path):
    nb = SiegeNotebook(nb_path)
    prof = _mature_profile({T4_WALL: 0.0})
    _update(nb, 1, prof, foci=[{"skill": T4_WALL, "prereq_tree": []}],
            forensics={T4_WALL: {"missing_top": []}})
    assert nb.focus_skills() == []
    assert f"tier_locked({T4_WALL}" in nb.last_focus_decision
    # the refusal must TEACH the exact re-proposal format (P0 contract).
    assert "relay_r0_floor" in nb.last_focus_decision


def test_tier4_relay_proposal_opens(nb_path):
    nb = SiegeNotebook(nb_path)
    prof = _mature_profile({T4_WALL: 0.0})
    _update(nb, 1, prof,
            foci=[{"skill": T4_WALL, "prereq_tree": [], "relay_r0_floor": 6}],
            forensics={})
    assert nb.focus_skills() == [T4_WALL]
    assert nb.relay_walls() == [T4_WALL]
    assert f"opened_relay({T4_WALL} @ R0 spawn_floor=6)" in nb.last_focus_decision


def test_tier3_ordinary_proposal_unaffected(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_ordinary(nb, T3_WALL)
    assert nb.focus_skills() == [T3_WALL]
    assert nb.foci()[0].get("relay") is None


def test_tier4_off_the_auto_open_menu(nb_path):
    nb = SiegeNotebook(nb_path)
    prof = _mature_profile({T4_WALL: 0.0, T3_WALL: 0.0})
    _update(nb, 1, prof, foci=[],
            ranked=[{"skill": T4_WALL, "why": "deep"}, {"skill": T3_WALL, "why": "stuck"}],
            forensics={T4_WALL: {"missing_top": []}, T3_WALL: {"missing_top": []}})
    # pigman skipped (relay-only, auto-open cannot carry an R0 floor); kobold auto-opens instead.
    assert nb.focus_skills() == [T3_WALL]


def test_tier_locked_is_mirrored_into_the_journal(nb_path):
    """The ⑦ lesson applied to P1: a refusal that lives only in the focus-decision LOG teaches the
    LLM nothing — the journal (the modeler's actual input) must carry it, and clear it next round."""
    nb = SiegeNotebook(nb_path)
    prof = _mature_profile({T4_WALL: 0.0})
    _update(nb, 1, prof, foci=[{"skill": T4_WALL, "prereq_tree": []}],
            forensics={T4_WALL: {"missing_top": []}})
    text = nb.render_for_prompt()
    assert "★TIER-LOCKED LAST SESSION" in text and T4_WALL in text
    assert "relay_r0_floor" in text
    # a session with no tier-locked proposal clears the mirror (no stale nagging).
    _update(nb, 3, prof, foci=[], forensics={})
    assert "TIER-LOCKED" not in nb.render_for_prompt()


def test_tier_gate_off_restores_fix2_behaviour(nb_path):
    nb = SiegeNotebook(nb_path)
    nb.th = SiegeThresholds(tier4_relay_only=False, wall_floor_anchor=False)
    prof = _mature_profile({T4_WALL: 0.0})
    _update(nb, 1, prof, foci=[{"skill": T4_WALL, "prereq_tree": []}],
            forensics={T4_WALL: {"missing_top": []}})
    assert nb.focus_skills() == [T4_WALL]  # fix2 equivalence when the knob is off


# ---- P2: relay attach (upgrade an active zero-win focus in place) ---------------------------------


def test_relay_attach_upgrades_active_zero_win_focus(nb_path):
    nb = SiegeNotebook(nb_path)
    prof = _open_ordinary(nb, T3_WALL, session=1, sr=0.0)
    foc = nb.foci()[0]
    # dirty the old attack-form state machines: the attach must reset them.
    foc["gap_forced"] = True
    foc["ladder_level"] = 2
    foc["frozen_sessions"] = 5
    foc["gap_sessions"] = 2
    _update(nb, 3, prof,
            foci=[{"skill": T3_WALL, "prereq_tree": [], "relay_r0_floor": T3_R0}],
            forensics={T3_WALL: {"missing_top": []}})
    assert f"relay_attached({T3_WALL} @ R0 spawn_floor={T3_R0}" in nb.last_focus_decision
    foc = nb.foci()[0]
    assert isinstance(foc["relay"], dict) and foc["relay"]["spawn_floor"] == T3_R0
    assert not foc["relay_sewn"]
    assert nb.relay_walls() == [T3_WALL]
    assert nb.required_spawn_floor(T3_WALL) == T3_R0
    # old form's state machines restart; held-out history is kept. (frozen_sessions may tick
    # once within the SAME apply call's stall pass — assert the reset, tolerate one tick.)
    assert foc["gap_forced"] is False and foc["ladder_level"] == 0
    assert foc["frozen_sessions"] <= 1 and foc["gap_sessions"] == 0
    assert foc.get("sr_history"), "held-out history must survive the upgrade"
    assert "UPGRADED" in (nb.last_relay_open or "")


def test_relay_attach_refused_when_wall_has_wins(nb_path):
    nb = SiegeNotebook(nb_path)
    prof = _open_ordinary(nb, T3_WALL, session=1, sr=30.0)
    _update(nb, 3, prof,
            foci=[{"skill": T3_WALL, "prereq_tree": [], "relay_r0_floor": T3_R0}],
            forensics={T3_WALL: {"missing_top": []}})
    assert f"relay_attach_refused({T3_WALL}" in nb.last_focus_decision
    assert "held-out wins" in nb.last_focus_decision
    assert nb.relay_walls() == []


def test_relay_attach_refused_at_relay_capacity(nb_path):
    nb = SiegeNotebook(nb_path)
    # campaign 1: a real relay on the tier-4 wall. (T3_WALL stays ZERO-win throughout — the
    # attach must be refused on CAPACITY, not on the has-wins guard.)
    prof = _mature_profile({T4_WALL: 0.0, T3_WALL: 0.0})
    _update(nb, 1, prof,
            foci=[{"skill": T4_WALL, "prereq_tree": [], "relay_r0_floor": 6}],
            forensics={})
    # a second, ORDINARY focus placed directly (the expand gate would refuse it behind the 0%
    # relay — its state is not what this test is about, so install a well-formed focus by hand).
    nb._open_focus(T3_WALL, 1, {k.lower(): v for k, v in prof.items()}, opened_by="llm")
    nb._save()
    prof2 = _mature_profile({T4_WALL: 0.0, T3_WALL: 0.0})
    _update(nb, 3, prof2,
            foci=[{"skill": T3_WALL, "prereq_tree": [], "relay_r0_floor": T3_R0}],
            forensics={})
    assert f"relay_attach_refused({T3_WALL}" in nb.last_focus_decision
    assert "campaign(s)" in nb.last_focus_decision
    assert nb.relay_walls() == [T4_WALL]


def test_relay_attach_range_guard(nb_path):
    nb = SiegeNotebook(nb_path)
    prof = _open_ordinary(nb, T3_WALL, session=1, sr=0.0)
    # bypass the modeler's parser range-check on purpose: the notebook must guard itself too.
    nb._reconcile_foci(3, {k.lower(): v for k, v in prof.items()},
                       [{"skill": T3_WALL, "prereq_tree": [],
                         "relay_r0_floor": MAX_DUNGEON_FLOOR + 3}])
    assert f"relay_attach_refused({T3_WALL}" in nb.last_focus_decision
    assert "outside 1.." in nb.last_focus_decision


def test_relay_attach_flag_off_keeps_fix2_kept(nb_path):
    nb = SiegeNotebook(nb_path)
    prof = _open_ordinary(nb, T3_WALL, session=1, sr=0.0)
    nb.th = SiegeThresholds(relay_attach=False)
    _update(nb, 3, prof,
            foci=[{"skill": T3_WALL, "prereq_tree": [], "relay_r0_floor": T3_R0}],
            forensics={T3_WALL: {"missing_top": []}})
    assert f"kept({T3_WALL})" in nb.last_focus_decision
    assert nb.relay_walls() == []


# ---- P3: relay opens are exempt from the expand gate's ">= 50%" condition -------------------------


def test_relay_open_exempt_from_expand_gate_zero_win_hostage(nb_path):
    """THE v7fix2 deadlock: a 0% ordinary focus holds the only slot; every new campaign gets
    expand_refused. A relay proposal must open anyway (free slot + relay capacity are enough)."""
    nb = SiegeNotebook(nb_path)
    _open_ordinary(nb, T3_WALL, session=1, sr=0.0)
    prof = _mature_profile({T3_WALL: 0.0, T3_WALL2: 0.0})
    _update(nb, 3, prof,
            foci=[{"skill": T3_WALL, "prereq_tree": []},
                  {"skill": T3_WALL2, "prereq_tree": [], "relay_r0_floor": T3_R0}],
            forensics={T3_WALL: {"missing_top": []}})
    assert f"opened_relay({T3_WALL2} @ R0 spawn_floor={T3_R0})" in nb.last_focus_decision
    assert nb.relay_walls() == [T3_WALL2]


def test_ordinary_open_still_expand_refused_behind_zero_win_focus(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_ordinary(nb, T3_WALL, session=1, sr=0.0)
    prof = _mature_profile({T3_WALL: 0.0, T3_WALL2: 0.0})
    _update(nb, 3, prof,
            foci=[{"skill": T3_WALL, "prereq_tree": []},
                  {"skill": T3_WALL2, "prereq_tree": []}],
            forensics={T3_WALL: {"missing_top": []}, T3_WALL2: {"missing_top": []}})
    assert f"expand_refused({T3_WALL2}" in nb.last_focus_decision
    assert nb.focus_skills() == [T3_WALL]


def test_relay_fallthrough_does_not_bypass_the_ordinary_expand_gate(nb_path):
    """v7fix4.5 UPDATE: the blocking focus here is a RELAY campaign, and P2 deliberately stops a
    relay's by-construction-zero held-out SR from holding ordinary expansion hostage (the s112
    expand_refused(defeat_gnome_warrior) lock-in) — so the fall-through now OPENS. The fix3-era
    contract ("relay_r0_floor is never a free expand pass") survives in two pinned forms: the
    flag-off branch below, and test_expand_gate_* above where the 0% blocker is an ORDINARY
    focus (anti-sprawl semantics unchanged there)."""
    nb = SiegeNotebook(nb_path)
    prof = _mature_profile({T4_WALL: 0.0, T3_WALL2: 0.0})
    _update(nb, 1, prof,
            foci=[{"skill": T4_WALL, "prereq_tree": [], "relay_r0_floor": 6}],
            forensics={})
    assert nb.relay_walls() == [T4_WALL]
    _update(nb, 3, prof,
            foci=[{"skill": T4_WALL, "prereq_tree": []},
                  {"skill": T3_WALL2, "prereq_tree": [], "relay_r0_floor": T3_R0}],
            forensics={T3_WALL2: {"missing_top": []}})
    assert f"relay_refused({T3_WALL2}" in nb.last_focus_decision
    assert f"opened({T3_WALL2}" in nb.last_focus_decision  # relay blocker no longer a hostage
    assert nb.focus_skills() == [T4_WALL, T3_WALL2]
    # fix3-era semantics pinned behind the flag: the same fall-through is expand_refused.
    nb2 = SiegeNotebook(nb_path + ".2")
    nb2.th.relay_expand_excluded = False
    _update(nb2, 1, prof,
            foci=[{"skill": T4_WALL, "prereq_tree": [], "relay_r0_floor": 6}],
            forensics={})
    _update(nb2, 3, prof,
            foci=[{"skill": T4_WALL, "prereq_tree": []},
                  {"skill": T3_WALL2, "prereq_tree": [], "relay_r0_floor": T3_R0}],
            forensics={T3_WALL2: {"missing_top": []}})
    assert f"expand_refused({T3_WALL2}" in nb2.last_focus_decision
    assert nb2.focus_skills() == [T4_WALL]


def test_relay_open_still_needs_a_free_slot(nb_path):
    nb = SiegeNotebook(nb_path)
    nb.th = SiegeThresholds(max_focus=1)
    _open_ordinary(nb, T3_WALL, session=1, sr=0.0)
    prof = _mature_profile({T3_WALL: 0.0, T3_WALL2: 0.0})
    _update(nb, 3, prof,
            foci=[{"skill": T3_WALL, "prereq_tree": []},
                  {"skill": T3_WALL2, "prereq_tree": [], "relay_r0_floor": T3_R0}],
            forensics={T3_WALL: {"missing_top": []}})
    assert f"expand_refused({T3_WALL2}: relay asked but no free focus slot" \
        in nb.last_focus_decision


# ---- P0: the journal teaches the upgrade path -----------------------------------------------------


def test_journal_zero_win_hint_names_the_upgrade(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_ordinary(nb, T3_WALL, session=1, sr=0.0)
    foc = nb.foci()[0]
    foc["sr_history"] = [0.0, 0.0, 0.5]  # 3 consecutive readings within zero_win_max_sr (1.0)
    text = nb.render_for_prompt()
    assert "★ZERO-WIN x3" in text
    assert "relay_r0_floor" in text and "UPGRADE" in text


def test_journal_hint_absent_with_wins_or_relay(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_ordinary(nb, T3_WALL, session=1, sr=0.0)
    nb.foci()[0]["sr_history"] = [0.0, 0.0, 12.0]  # last reading is a win
    assert "★ZERO-WIN" not in nb.render_for_prompt()
    # a live relay never shows the hint (it IS the upgrade).
    nb2 = SiegeNotebook(nb_path + ".2")
    _update(nb2, 1, _mature_profile({T4_WALL: 0.0}),
            foci=[{"skill": T4_WALL, "prereq_tree": [], "relay_r0_floor": 6}],
            forensics={})
    nb2.foci()[0]["relay"]["rung_trained"] = []
    nb2.foci()[0]["sr_history"] = [0.0, 0.0, 0.0, 0.0]
    assert "★ZERO-WIN" not in nb2.render_for_prompt()


# ---- P6: breadth spawn frontier (notebook state machine) ------------------------------------------


def test_frontier_defaults_advances_and_persists(nb_path):
    nb = SiegeNotebook(nb_path)
    assert nb.breadth_frontier() == 1
    # below the bar / wrong floor -> no advance.
    assert nb.note_breadth_frontier_reading(1, BREADTH_FRONTIER_SR - 5, session_idx=3) is None
    assert nb.note_breadth_frontier_reading(2, 95.0, session_idx=3) is None
    assert nb.breadth_frontier() == 1
    # at the frontier floor and above the bar -> floor 2 unlocks.
    msg = nb.note_breadth_frontier_reading(1, BREADTH_FRONTIER_SR + 5, session_idx=5)
    assert msg and "floor 2 unlocked" in msg
    assert nb.breadth_frontier() == 2
    # persisted: a reloaded notebook keeps the frontier (resume safety).
    nb2 = SiegeNotebook(nb_path)
    assert nb2.breadth_frontier() == 2


def test_frontier_caps_at_max_dungeon_floor(nb_path):
    nb = SiegeNotebook(nb_path)
    nb._nb["breadth_frontier"] = MAX_DUNGEON_FLOOR
    assert nb.note_breadth_frontier_reading(MAX_DUNGEON_FLOOR, 99.0, session_idx=3) is None
    assert nb.breadth_frontier() == MAX_DUNGEON_FLOOR


# ---- P6: R6 always-on + the BREADTH lane (validator) ----------------------------------------------

_DESC = "Objective: x\nRelevant Achievements: EAT_COW\nCompleted Achievements: NONE\nWorld: cow"


def _meta(type_="BREADTH", spawn_floor=0, siege_wall=None, drill_target=None):
    return {"type": type_, "drill_target": drill_target, "siege_wall": siege_wall,
            "spawn_floor": spawn_floor, "spawn_kit": None}


def test_r6_breadth_lane_legal_even_with_no_foci():
    v = validate_level(_DESC, _meta("BREADTH", 1), foci=[], breadth_frontier=1)
    assert v == []


def test_r6_breadth_beyond_frontier_rejected():
    v = validate_level(_DESC, _meta("BREADTH", 2), foci=[], breadth_frontier=1)
    assert [x.rule for x in v] == [RULE_SPAWN]
    assert "frontier" in v[0].message


def test_r6_depth_deep_spawn_rejected_no_foci():
    """The fix2-era hole: with foci=[] NOTHING was validated. Now the spawn contract is always-on
    and a non-BREADTH deep spawn is rejected with the two-lane teaching message."""
    v = validate_level(_DESC, _meta("DEPTH", 1), foci=[], breadth_frontier=3)
    assert [x.rule for x in v] == [RULE_SPAWN]
    assert "two lanes" in v[0].message or "BREADTH" in v[0].message


def test_r6_tagged_breadth_deep_spawn_rejected():
    """A siege-tagged 'BREADTH' level must not borrow the ecology lane as a scaffold (v5 lesson)."""
    v = validate_level(
        _DESC, _meta("BREADTH", 1, siege_wall="defeat_kobold"),
        foci=[{"skill": "defeat_kobold", "prereq_tree": []}], breadth_frontier=3,
    )
    assert RULE_SPAWN in [x.rule for x in v]


def test_r6_natural_spawn_stays_clean_no_foci():
    assert validate_level(_DESC, _meta("DEPTH", 0), foci=[], breadth_frontier=1) == []


def test_r6_relay_contract_untouched_by_the_frontier():
    """A relay wall's level obeys the RUNG floor exactly — the breadth frontier must not leak in."""
    foci = [{"skill": "defeat_kobold", "prereq_tree": []}]
    ok = validate_level(
        _DESC, _meta("DEPTH", 3, siege_wall="defeat_kobold"), foci=foci,
        required_spawn_floors={"defeat_kobold": 3}, breadth_frontier=1,
    )
    assert ok == []
    bad = validate_level(
        _DESC, _meta("DEPTH", 1, siege_wall="defeat_kobold"), foci=foci,
        required_spawn_floors={"defeat_kobold": 3}, breadth_frontier=8,
    )
    assert [x.rule for x in bad] == [RULE_SPAWN] and "rung" in bad[0].message


# ---- P5: full-price force-activation cap ----------------------------------------------------------
# evolution_efficient imports jax at module level (same boundary as test_siege_ecosystem): the P5
# tests run in the Oscar full suite and SKIP on the local non-jax subset.

_HAS_JAX_P5 = importlib.util.find_spec("jax") is not None
needs_jax_p5 = pytest.mark.skipif(
    not _HAS_JAX_P5, reason="evolution_efficient imports jax (Oscar full suite only)"
)


def _attempt_to_activate_task():
    _stub = types.ModuleType("dicode.dreaming.gen_manager")
    _stub.GenManager = object
    _stub.TaskArchive = object
    sys.modules.setdefault("dicode.dreaming.gen_manager", _stub)
    from dicode.evolution_efficient import attempt_to_activate_task

    return attempt_to_activate_task


class _Archive:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._lock = threading.Lock()

    @property
    def active_task_count(self):
        return sum(1 for _, d in self.graph.nodes(data=True) if d.get("is_active"))

    def set_task_active_status(self, task_id, status):
        self.graph.nodes[task_id]["is_active"] = bool(status)


class _PlogFake:
    def __init__(self, profile):
        self._p = profile

    def latest(self):
        return dict(self._p)


class _TGHolder:
    def __init__(self, notebook, profile):
        self._siege_notebook = notebook
        self._profile_log = _PlogFake(profile)
        self._siege_force_counts = {}


class _GM:
    def __init__(self, archive, holder):
        self.archive = archive
        self.task_generator = holder


class _DM:
    def __init__(self, capacity=100, min_entry=0.1):
        self.active_task_capacity = capacity
        self.min_entry_score_threshold = min_entry
        self.siege_focus_quota = 4


class _Cfg:
    def __init__(self, dm):
        self.dicode_manager = dm


def _activation_world(nb_path, wall, wall_sr):
    nb = SiegeNotebook(nb_path)
    _open_ordinary(nb, wall, session=1, sr=wall_sr)
    ar = _Archive()
    for i in range(FOCUS_FORCE_CAP + 3):
        ar.graph.add_node(f"d{i}", is_active=False, priority_score=0.0, siege_wall=wall)
    holder = _TGHolder(nb, _mature_profile({wall: wall_sr}))
    return _GM(ar, holder), _Cfg(_DM())


@needs_jax_p5
def test_full_price_wall_capped_at_focus_force_cap(nb_path, capsys):
    """fix11's exact monopoly: a mid-band (non-zero-win) wall used to force-activate UNLIMITED
    levels per session. Now the full-price lane stops at focus_force_cap (8, above fix8's healthy
    6-10 band mean) and the overflow falls back to the normal CAS competition."""
    attempt = _attempt_to_activate_task()
    gm, cfg = _activation_world(nb_path, T3_WALL, wall_sr=30.0)
    results = [attempt(gm, f"d{i}", 0.0, cfg) for i in range(FOCUS_FORCE_CAP + 2)]
    assert results[:FOCUS_FORCE_CAP] == [True] * FOCUS_FORCE_CAP
    # over the cap: score 0.0 loses the normal CAS entry bar -> not activated.
    assert results[FOCUS_FORCE_CAP:] == [False, False]
    out = capsys.readouterr().out
    assert "full-price force-activation(s)" in out


@needs_jax_p5
def test_zero_win_wall_still_capped_at_two(nb_path, capsys):
    attempt = _attempt_to_activate_task()
    gm, cfg = _activation_world(nb_path, T3_WALL, wall_sr=0.0)
    results = [attempt(gm, f"d{i}", 0.0, cfg) for i in range(4)]
    assert results == [True, True, False, False]
    assert "discounted force-activation(s)" in capsys.readouterr().out


# ---- P4/P6 gen_manager wiring (Oscar full suite only — gen_manager imports jax) -------------------

_HAS_JAX = importlib.util.find_spec("jax") is not None
needs_jax = pytest.mark.skipif(not _HAS_JAX, reason="gen_manager imports jax (Oscar full suite only)")

_gmmod = None


def _gen_manager_module():
    global _gmmod
    if _gmmod is None:
        gm_path = os.path.join(_REPO, "src", "dicode", "dreaming", "gen_manager.py")
        spec = importlib.util.spec_from_file_location("dicode_v7fix3_gen_manager_test", gm_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _gmmod = mod
    return _gmmod


class _HistPlog:
    """Profile-log fake with a real history (for peaks) and a latest()."""

    def __init__(self, snaps):
        self._snaps = [{"session": i, "profile": p} for i, p in enumerate(snaps)]

    def latest(self):
        return dict(self._snaps[-1]["profile"]) if self._snaps else {}

    def recent(self, k):
        return list(self._snaps[-k:])


@needs_jax
def test_ecology_directive_renders_three_sections(tmp_path):
    tg = object.__new__(_gen_manager_module().TaskGenerator)
    nb = SiegeNotebook(str(tmp_path / "nb.json"))
    tg._siege_notebook = nb
    early = {"find_bow": 50.0, "collect_iron": 40.0, "eat_cow": 90.0}
    now = {"find_bow": 20.0, "collect_iron": 42.0, "eat_cow": 91.0, "defeat_zombie": 5.0}
    tg._profile_log = _HistPlog([early, now])
    text = tg._render_ecology_directive(now)
    assert "STARVED FAMILIES" in text
    assert "DECLINING SKILLS" in text and "find_bow" in text  # 50 -> 20 = -30pp off a 50 peak
    assert "BREADTH SPAWN FRONTIER: floor 1" in text
    assert "spawn_floor" in text


@needs_jax
def test_ecology_proposer_idxs_from_personas():
    tg = object.__new__(_gen_manager_module().TaskGenerator)
    tg.config = {"personas": ["ambitious_coop", "ecology_coop"]}
    assert tg._ecology_proposer_idxs() == {1}
    tg.config = {"personas": ["ambitious_coop", "ambitious_coop"]}
    assert tg._ecology_proposer_idxs() == set()
    tg.config = {}
    assert tg._ecology_proposer_idxs() == set()


@needs_jax
def test_coop_select_role_quota_floors_the_ecology(tmp_path):
    """fix11's collapse: mid-band siege drills sweep every main seat 'on merit'. The role quota
    guarantees the ecology bucket its floor inside k, same bid scoring per bucket."""
    gmm = _gen_manager_module()
    tg = object.__new__(gmm.TaskGenerator)
    tg._siege_notebook = None  # no siege partition — exercise the quota branch directly
    tg._build_parent_learnability = lambda proposals: {}
    tg.config = types.SimpleNamespace(
        coop_role_quota=[2, 2], coop_w_amb=0.0, coop_w_lrn=1.0, coop_w_cov=0.0,
    )
    def _parsed(i, ridx):
        return {"description": f"Objective: t{i}\nRelevant Achievements: EAT_COW",
                "reasoning": "", "level_meta": None, "_proposer_idx": ridx}
    # 4 siege-arm candidates, 2 ecology candidates; open top-4 would be dominated by one role.
    all_parsed = [_parsed(i, 0) for i in range(4)] + [_parsed(i + 4, 1) for i in range(2)]
    all_parents = [["p"]] * 6
    all_examples = [[]] * 6
    sel_p, sel_pa, sel_e = tg._coop_select(
        all_parsed, all_parents, all_examples, 4, 1, {}, siege_partition=False,
    )
    assert len(sel_p) == 4
    roles = [p["_proposer_idx"] for p in sel_p]
    assert roles.count(0) == 2 and roles.count(1) == 2  # the quota floors each role


@needs_jax
def test_coop_select_role_quota_backfills_short_bucket(tmp_path):
    gmm = _gen_manager_module()
    tg = object.__new__(gmm.TaskGenerator)
    tg._siege_notebook = None
    tg._build_parent_learnability = lambda proposals: {}
    tg.config = types.SimpleNamespace(
        coop_role_quota=[2, 2], coop_w_amb=0.0, coop_w_lrn=1.0, coop_w_cov=0.0,
    )
    def _parsed(i, ridx):
        return {"description": f"Objective: t{i}\nRelevant Achievements: EAT_COW",
                "reasoning": "", "level_meta": None, "_proposer_idx": ridx}
    all_parsed = [_parsed(i, 0) for i in range(5)] + [_parsed(5, 1)]  # ecology has only 1
    sel_p, _, _ = tg._coop_select(
        all_parsed, [["p"]] * 6, [[]] * 6, 4, 1, {}, siege_partition=False,
    )
    roles = [p["_proposer_idx"] for p in sel_p]
    assert len(roles) == 4 and roles.count(1) == 1 and roles.count(0) == 3  # backfilled


@needs_jax
def test_breadth_frontier_sweep_advances_from_breadth_levels(tmp_path):
    gmm = _gen_manager_module()
    tg = object.__new__(gmm.TaskGenerator)
    nb = SiegeNotebook(str(tmp_path / "nb.json"))
    tg._siege_notebook = nb
    tg.config = types.SimpleNamespace()

    class _Ar:
        def __init__(self):
            self.graph = nx.DiGraph()
            self._lock = threading.Lock()

    ar = _Ar()
    hi = (BREADTH_FRONTIER_SR + 10) / 100.0
    ar.graph.add_node("b1", level_type="BREADTH", spawn_floor=1,
                      performance_history=[{"sr": hi, "session": 9}])
    # tagged / non-breadth / wrong-floor levels must NOT drive the frontier.
    ar.graph.add_node("d1", level_type="DEPTH", spawn_floor=1,
                      performance_history=[{"sr": 0.99, "session": 9}])
    ar.graph.add_node("b_tagged", level_type="BREADTH", spawn_floor=1, siege_wall="defeat_kobold",
                      performance_history=[{"sr": 0.99, "session": 9}])
    ar.graph.add_node("b_deep", level_type="BREADTH", spawn_floor=2,
                      performance_history=[{"sr": 0.99, "session": 9}])
    tg.archive = ar
    tg._note_breadth_frontier_readings(session_idx=9)
    assert nb.breadth_frontier() == 2  # advanced exactly one floor, driven by b1 only
    # stale readings (outside the recency window) must not advance it further.
    ar.graph.nodes["b_deep"]["performance_history"] = [{"sr": 0.99, "session": 2}]
    tg._note_breadth_frontier_readings(session_idx=9)
    assert nb.breadth_frontier() == 2


# ---- v7fix3.1 audit fixes (2026-07-10 post-launch audit) -------------------------------------------
# A1: the relay-capacity fall-through must not open a tier-4 wall as an ordinary siege.
# A2: the door gate must never open a tier-4 door (enter_fire_realm & co) as a gateway focus.
# A3: the cooldown waives for a relay re-proposal after an ORDINARY retirement (the ④ gap gate can
#     retire a zero-win wall in the very session the LLM answers the ★ZERO-WIN hint), but NOT after
#     the relay campaign itself stalled out.
# A7: the fix4-era stall ratchet is gone; only its best_sr high-water side effect survives.


def test_relay_capacity_fallthrough_tier4_refused(nb_path):
    nb = SiegeNotebook(nb_path)
    # s1: ordinary tier-3 focus, raised above expand_sr so the ordinary expand gate would PASS.
    prof = _mature_profile({T3_WALL2: 55.0, T3_WALL: 0.0, T4_WALL: 0.0})
    _update(nb, 1, prof, foci=[{"skill": T3_WALL2, "prereq_tree": []}],
            forensics={T3_WALL2: {"missing_top": []}})
    assert T3_WALL2 in nb.focus_skills()
    # s2: relay campaign occupies the single relay slot (P3: free slot is enough).
    _update(nb, 2, prof, foci=[{"skill": T3_WALL, "relay_r0_floor": T3_R0, "prereq_tree": []}])
    assert T3_WALL in nb.relay_walls()
    # s3: tier-4 relay ask while the slot is taken — without A1 this fell through to an
    # ordinary open (expand gate passes via the 55% focus): the pigman shape P1 exists to stop.
    _update(nb, 3, prof, foci=[{"skill": T4_WALL, "relay_r0_floor": 6, "prereq_tree": []}])
    assert "cannot open as a natural-spawn siege" in nb.last_focus_decision
    assert "treating this as a normal focus proposal" not in nb.last_focus_decision
    assert T4_WALL not in nb.focus_skills()


def test_door_substitute_skips_tier4_door(nb_path):
    nb = SiegeNotebook(nb_path)
    # kobold's failures name enter_fire_realm (tier-4, unknown SR = closed at rank 0): the old
    # code opened it as the gateway focus; now it is skipped and the wall opens plainly.
    prof = _mature_profile({T3_WALL: 0.0})
    _update(nb, 1, prof, foci=[{"skill": T3_WALL, "prereq_tree": []}],
            forensics={T3_WALL: {"missing_top": [["enter_fire_realm", 0.9]]}})
    assert f"opened({T3_WALL})" in nb.last_focus_decision
    assert "enter_fire_realm" not in nb.focus_skills()


def test_door_substitute_still_opens_lower_tier_door(nb_path):
    nb = SiegeNotebook(nb_path)
    # control: a tier-2 door below door_min_sr still substitutes as before.
    prof = _mature_profile({T3_WALL: 0.0, "enter_dungeon": 4.0})
    _update(nb, 1, prof, foci=[{"skill": T3_WALL, "prereq_tree": []}],
            forensics={T3_WALL: {"missing_top": [["enter_dungeon", 0.9]]}})
    assert "enter_dungeon" in nb.focus_skills()
    assert T3_WALL not in nb.focus_skills()


def _seed_retired(nb, wall, last_event, last_session=9):
    nb._nb.setdefault("retired", {})[wall] = {
        "count": 1, "last_session": last_session, "last_event": last_event,
        "failed_notes": ["hit it with a stone sword straight from natural spawn"],
        "sr_at_retirement": 0.0,
    }


def test_cooldown_waived_for_relay_reopen_after_ordinary_retirement(nb_path):
    nb = SiegeNotebook(nb_path)
    _seed_retired(nb, T4_WALL, "focus_retired_stalled")
    # the journal teaches the exemption for ordinary retirements
    assert "exempt from the cooldown" in nb.render_for_prompt()
    prof = _mature_profile({T4_WALL: 0.0})
    _update(nb, 10, prof, foci=[{
        "skill": T4_WALL, "relay_r0_floor": 6, "prereq_tree": [],
        "style_note": "spawn on the wall's own floor with an iron kit; learn the duel itself first",
    }])
    assert "cooldown_waived(" in nb.last_focus_decision
    assert f"opened_relay({T4_WALL}" in nb.last_focus_decision
    assert T4_WALL in nb.relay_walls()


def test_cooldown_not_waived_after_relay_stall_retirement(nb_path):
    nb = SiegeNotebook(nb_path)
    _seed_retired(nb, T4_WALL, "focus_retired_relay_stalled")
    assert "exempt from the cooldown" not in nb.render_for_prompt()
    prof = _mature_profile({T4_WALL: 0.0})
    _update(nb, 10, prof, foci=[{
        "skill": T4_WALL, "relay_r0_floor": 6, "prereq_tree": [],
        "style_note": "restart the ladder one floor shallower and clear with a bow this time",
    }])
    assert "cooldown_rejected(" in nb.last_focus_decision
    assert T4_WALL not in nb.focus_skills()


def test_ordinary_reopen_still_cooldown_rejected(nb_path):
    nb = SiegeNotebook(nb_path)
    _seed_retired(nb, T3_WALL, "focus_retired_stalled")
    prof = _mature_profile({T3_WALL: 0.0})
    _update(nb, 10, prof, foci=[{"skill": T3_WALL, "prereq_tree": [],
                                 "style_note": "a genuinely new tactic this time"}])
    assert "cooldown_rejected(" in nb.last_focus_decision
    assert T3_WALL not in nb.focus_skills()


def test_retirement_records_last_event(nb_path):
    nb = SiegeNotebook(nb_path)
    nb._archive_retirement({"skill": T3_WALL2, "best_sr": 3.0, "prereq_tree": []},
                           5, "focus_retired_budget")
    assert nb._nb["retired"][T3_WALL2]["last_event"] == "focus_retired_budget"


def test_stall_ratchet_removed_best_sr_survives(nb_path):
    nb = SiegeNotebook(nb_path)
    assert not hasattr(nb.th, "focus_min_stall_sessions")
    _open_ordinary(nb, T3_WALL, session=1, sr=8.0)
    assert "stall_sessions" not in nb.foci()[0]
    _update(nb, 2, _mature_profile({T3_WALL: 30.0}), foci=[{"skill": T3_WALL}])
    assert nb.foci()[0]["best_sr"] == 30.0
