"""v7 SPAWN-ANNEAL RELAY — offline tests (v7_design.md §2/§5.5).

The multiplication chain (spawn->FloorN exposure = product of per-floor descent SRs) makes
ADVANCED-tier walls structurally unreachable from natural-spawn levels (baseline s151: all 24
ADVANCED+ achievements at zero). A relay campaign attacks such a wall BACKWARD: R0 spawns the
student at the target floor with a winners'-median kit, then anneals the spawn point up one floor
per rung graduation until natural spawn ("sewn"). These tests pin:

  1. ADMISSION: an explicit LLM proposal with relay_r0_floor opens a relay (bypassing the door
     gate + admission deferral — a relay does not go THROUGH the door, it spawns behind it);
     capacity relay_max; auto-open never creates one.
  2. RUNG MACHINE: trained-SR-driven graduation (>=70 x2 fresh readings -> spawn up), regression
     (<20 x4 -> spawn back down, never past R0), sewing at floor 0, early stop (no transition +
     no trained new-high for relay_stall_patience readings, COMBAT x2) -> normal retirement.
  3. INTERFACE EXEMPTIONS while live: gap gate / ④ attribution forcing / required_form / ⑤
     enabler budget / ② yield all suspend; ③ discount reads RUNG PROGRESS instead of held-out
     wins; rung transitions feed the ladder's progress signal (c).
  4. GENERATION CONTRACT: level_meta spawn_floor/spawn_kit parsing; validator R6_SPAWN (wrong
     floor on a relay level rejected; deep spawn without a relay rejected); modeler parses the
     optional relay_r0_floor; gen_manager feeds current-rung-only trained readings into
     note_rung_reading (wrong-floor levels excluded) and renders the RELAY directive + kit hint.

No jax/craftax/LLM needed.
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

from auction.level_meta import parse_level_meta  # noqa: E402
from auction.level_validator import (  # noqa: E402
    RULE_SPAWN,
    reroll_worthy,
    validate_level,
)
from auction.modeler import Modeler  # noqa: E402
from auction.siege_notebook import (  # noqa: E402
    ENABLER_MAX_SESSIONS,
    MATURITY_MIN_MASTERED,
    MATURITY_MIN_SNAPSHOTS,
    MATURITY_SKILL_SR,
    RELAY_STALL_PATIENCE,
    RUNG_FLOOR_SR,
    RUNG_GRADUATE_CONSECUTIVE,
    RUNG_GRADUATE_SR,
    RUNG_STALL_READINGS,
    SiegeNotebook,
)

WALL = "defeat_kobold"          # COMBAT family -> relay patience x2
ENABLER_WALL = "enter_sewers"   # non-COMBAT family -> base patience, ⑤-budget test subject


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


def _open_relay(nb, wall=WALL, r0=3, session=1, profile=None):
    prof = profile or _mature_profile({wall: 0.0})
    _update(nb, session, prof,
            foci=[{"skill": wall, "prereq_tree": [], "relay_r0_floor": r0}],
            forensics={})
    return prof


# ---- 1. admission ---------------------------------------------------------------------------------

def test_relay_opens_bypassing_deferral_and_door_gate(nb_path):
    """forensics={} would normally DEFER a first-open wall (hazard-3a) and a closed missing link
    would door-substitute — a relay proposal bypasses both: R0 spawns behind the door."""
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    assert nb.focus_skills() == [WALL]
    foc = nb.foci()[0]
    assert isinstance(foc["relay"], dict)
    assert foc["relay"]["spawn_floor"] == 3 and foc["relay"]["r0_floor"] == 3
    assert not foc["relay_sewn"]
    assert f"opened_relay({WALL} @ R0 spawn_floor=3)" in nb.last_focus_decision
    assert nb.last_relay_open is not None
    assert nb.required_spawn_floor(WALL) == 3
    assert nb.relay_walls() == [WALL]


def test_relay_capacity_cap_second_campaign_refused(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    # expand gate satisfied artificially (one focus reads >= 50) so ONLY the relay cap can refuse.
    prof = _mature_profile({WALL: 55.0, ENABLER_WALL: 0.0})
    _update(nb, 3, prof,
            foci=[{"skill": WALL, "prereq_tree": []},
                  {"skill": ENABLER_WALL, "prereq_tree": [], "relay_r0_floor": 3}],
            forensics={})
    assert f"relay_refused({ENABLER_WALL}" in nb.last_focus_decision
    assert nb.relay_walls() == [WALL]  # only the first campaign is a relay


def test_auto_open_never_creates_a_relay(nb_path):
    """ranked_walls carries no relay channel: an auto-open focus must be a NORMAL siege focus
    (picking R0's floor is teacher knowledge — only an explicit LLM proposal may relay)."""
    nb = SiegeNotebook(nb_path)
    prof = _mature_profile({"defeat_orc_mage": 0.0})
    _update(nb, 1, prof, foci=[], ranked=[{"skill": "defeat_orc_mage", "why": "stuck"}],
            forensics={"defeat_orc_mage": {"missing_top": []}})
    assert nb.focus_skills() == ["defeat_orc_mage"]
    assert nb.foci()[0].get("relay") is None


# ---- 2. rung state machine ------------------------------------------------------------------------

def test_rung_graduation_climbs_and_sews(nb_path):
    """v7fix4 ladder walkthrough: R0 is habitat-anchored (kobold inhabits floor 3 — the proposed
    r0=2 is corrected, the exact v7fix3 lizard failure), each rung graduates upward, and reaching
    floor 0 enters the KIT_STRIP exam (natural spawn + EMPTY kit == held-out distribution) whose
    graduation alone SEWs."""
    nb = SiegeNotebook(nb_path)
    _open_relay(nb, r0=2)  # kobold inhabits floor 3 -> anchored: the ladder starts one deeper
    foc = nb.foci()[0]
    assert foc["relay"]["spawn_floor"] == 3 and foc["relay"]["r0_floor"] == 3
    hi = RUNG_GRADUATE_SR + 5
    # P2' (fix56设计 §3.2): x2 FULL graduation = window fill (2) + 2 judged passes = 4 HI.
    _sidx = [3]

    def _feed4(expect):
        st = None
        for _ in range(4):
            st = nb.note_rung_reading(WALL, hi, session_idx=_sidx[0])
            _sidx[0] += 2
            if expect in st:
                break
        assert expect in st
        return st

    assert "rung hold" in nb.note_rung_reading(WALL, hi, session_idx=_sidx[0])
    _sidx[0] += 2
    s = _feed4("RUNG_GRADUATED")
    assert "floor 2" in s
    foc = nb.foci()[0]
    assert foc["relay"]["spawn_floor"] == 2
    assert foc["relay"]["rung_trained"] == []          # fresh rung, fresh readings
    assert foc["relay"]["rung_graduate_streak"] == 0
    assert nb.required_spawn_floor(WALL) == 2
    # floor 2 -> floor 1.
    s = _feed4("RUNG_GRADUATED")
    assert "floor 1" in s
    # floor 1 graduates -> NOT sewn yet: the kit-strip exam comes first (v7fix4 P3).
    s = _feed4("KIT_STRIP")
    assert "EMPTY kit" in s
    foc = nb.foci()[0]
    assert foc["relay"]["spawn_floor"] == 0 and foc["relay"]["kit_strip"] is True
    assert not foc["relay_sewn"]
    assert nb.relay_kit_stripped(WALL) is True
    assert nb.required_spawn_floor(WALL) == 0          # natural spawn, still under contract
    # the kitless natural-spawn rung graduates -> SEWN (now a result certificate).
    s = _feed4("SEWN")
    assert "kitless" in s
    foc = nb.foci()[0]
    assert foc["relay_sewn"] is True
    assert nb.required_spawn_floor(WALL) is None       # natural spawn: no contract anymore
    assert nb.relay_walls() == []
    assert nb.relay_kit_stripped(WALL) is False        # no ACTIVE relay anymore
    assert nb.note_rung_reading(WALL, hi, session_idx=_sidx[0]) is None  # machine is done


def test_rung_regression_never_past_r0(nb_path):
    # v7fix4.6: cliff-split OFF here — this test pins the PURE floor semantics (never past r0);
    # with the scaffold ladder ON a stalled floor first steps through sub-stages (its own tests
    # live in test_siege_fix46_cliff_split.py).
    from auction.siege_notebook import SiegeThresholds

    nb = SiegeNotebook(nb_path, thresholds=SiegeThresholds(rung_cliff_split=False))
    _open_relay(nb, r0=3)
    hi, lo = RUNG_GRADUATE_SR + 5, RUNG_FLOOR_SR - 5
    for s in (3, 5, 7, 9):
        nb.note_rung_reading(WALL, hi, session_idx=s)  # graduate to floor 2 (P2': 4 readings)
    assert nb.required_spawn_floor(WALL) == 2
    status = None
    for i in range(2 + RUNG_STALL_READINGS):           # (P2': window fills first)
        status = nb.note_rung_reading(WALL, lo, session_idx=11 + 2 * i)
    assert "RUNG_REGRESSED" in status and "floor 3" in status
    assert nb.required_spawn_floor(WALL) == 3
    # at R0 there is nowhere deeper: more stall readings never move the floor past r0.
    for i in range(2 + RUNG_STALL_READINGS + 1):
        nb.note_rung_reading(WALL, lo, session_idx=31 + 2 * i)
        if not nb.relay_walls():
            break  # the early stop may retire it — that is the designed exit, not a regression
    if nb.relay_walls():
        assert nb.required_spawn_floor(WALL) == 3


def test_relay_early_stop_retires_flat_campaign(nb_path):
    """No rung transition + no trained new-high for relay_stall_patience readings (base patience:
    non-COMBAT wall) -> the campaign retires through the NORMAL machinery (cooldown/blacklist)."""
    nb = SiegeNotebook(nb_path)
    _open_relay(nb, wall=ENABLER_WALL, r0=3)
    mid = (RUNG_FLOOR_SR + RUNG_GRADUATE_SR) / 2  # neither graduates nor stalls the rung
    for i in range(3):    # P2': window fills (2 silent) + first judged window = new high
        nb.note_rung_reading(ENABLER_WALL, mid, session_idx=3 + 2 * i)
    status = None
    for i in range(RELAY_STALL_PATIENCE):
        status = nb.note_rung_reading(ENABLER_WALL, mid, session_idx=9 + 2 * i)
    assert "RELAY_RETIRED" in status
    assert nb.focus_skills() == []
    reg = nb.retired_registry()[ENABLER_WALL]
    assert reg["count"] == 1


def test_relay_combat_patience_doubles_and_new_high_resets(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb, wall=WALL, r0=3)  # COMBAT -> patience x2
    mid = (RUNG_FLOOR_SR + RUNG_GRADUATE_SR) / 2
    nb.note_rung_reading(WALL, mid, session_idx=3)
    for i in range(RELAY_STALL_PATIENCE):  # base patience worth of flat readings: must survive
        s = nb.note_rung_reading(WALL, mid, session_idx=5 + 2 * i)
        assert "RELAY_RETIRED" not in s
    # a real new high resets the patience entirely.
    s = nb.note_rung_reading(WALL, mid + 10, session_idx=15)
    assert "patience 0/" in s


def test_no_reading_holds_all_counters(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb, r0=2)
    mid = (RUNG_FLOOR_SR + RUNG_GRADUATE_SR) / 2
    nb.note_rung_reading(WALL, mid, session_idx=3)
    before = dict(nb.foci()[0]["relay"])
    s = nb.note_rung_reading(WALL, None, session_idx=5)
    assert "no_fresh_rung_reading" in s
    after = nb.foci()[0]["relay"]
    for k in ("rung_graduate_streak", "rung_stall_streak", "stall_patience", "spawn_floor"):
        assert after.get(k) == before.get(k)


def test_relay_state_survives_reload(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb, r0=3)
    for i in range(3):    # P2': the streak starts once the win3 window is full
        nb.note_rung_reading(WALL, RUNG_GRADUATE_SR + 5, session_idx=3 + 2 * i)
    nb2 = SiegeNotebook(nb_path)
    assert nb2.required_spawn_floor(WALL) == 3
    assert nb2.foci()[0]["relay"]["rung_graduate_streak"] == 1


# ---- 3. interface exemptions while a relay is live -------------------------------------------------

def test_relay_suspends_gap_gate_and_required_form(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    status = nb.note_transfer_gap(WALL, 100.0, 0.0, session_idx=3)
    assert status.startswith("relay(")
    assert nb.foci()[0].get("gap_sessions", 0) == 0      # the 100pp "gap" never counted
    assert nb.required_form(WALL) is None


def test_relay_suspends_attribution_forcing(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    # a verified access-blocked attribution on the relay wall must NOT force DEPTH / arm P3.
    prof = _mature_profile({WALL: 0.0, "enter_sewers": 0.0})
    _update(nb, 3, prof, foci=[{
        "skill": WALL, "prereq_tree": [],
        "failure_attribution": {"class": "chain_unreached",
                                "key_missing_link": "enter_sewers", "verified": True},
    }], forensics={WALL: {"missing_top": [("enter_sewers", 0.8)]}})
    foc = nb.foci()[0]
    assert foc["attrib_depth_required"] is False
    assert not foc.get("gap_forced")
    assert nb.required_form(WALL) is None


def test_relay_exempt_from_enabler_budget(nb_path):
    """enter_sewers is non-COMBAT: without the exemption ⑤ would kill the campaign at 8 siege
    decisions — a 3-rung anneal needs far more. The relay's own early stop is its budget."""
    nb = SiegeNotebook(nb_path)
    _open_relay(nb, wall=ENABLER_WALL, r0=3)
    prof = _mature_profile({ENABLER_WALL: 0.0})
    for s in range(3, 3 + 2 * (ENABLER_MAX_SESSIONS + 2), 2):
        _update(nb, s, prof, foci=[{"skill": ENABLER_WALL, "prereq_tree": []}], forensics={})
        # keep the rung machine visibly alive (a fresh new-high every reading, but below the
        # graduation line) so neither the early stop nor a full anneal fires during this test.
        nb.note_rung_reading(ENABLER_WALL, 20.0 + 1.5 * s, session_idx=s)
    assert nb.focus_skills() == [ENABLER_WALL]
    assert nb.foci()[0]["siege_sessions"] > ENABLER_MAX_SESSIONS


def test_relay_never_yields_to_momentum(nb_path):
    """Two fast-rising in-band readings would YIELD a normal focus to WATCH — a live relay must
    stay (its held-out is structurally 0; any apparent momentum is noise)."""
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    for s, sr in ((3, 25.0), (5, 40.0)):  # +15pp over two rising readings, inside the band
        _update(nb, s, _mature_profile({WALL: sr}),
                foci=[{"skill": WALL, "prereq_tree": []}], forensics={})
    assert nb.focus_skills() == [WALL]
    assert WALL not in nb.watch_registry()


def test_discount_reads_rung_progress_not_heldout_wins(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)  # COMBAT wall: patience 6 keeps it alive through the stall below
    prof = _mature_profile({WALL: 0.0})
    # freshly opened (a transition within the last 2 readings): progressing -> NOT discounted.
    assert WALL not in nb.zero_win_walls(prof)
    mid = (RUNG_FLOOR_SR + RUNG_GRADUATE_SR) / 2
    nb.note_rung_reading(WALL, mid, session_idx=3)
    for i in range(3):  # flat readings: past the grace window, no new highs
        nb.note_rung_reading(WALL, mid, session_idx=5 + 2 * i)
    assert WALL in nb.zero_win_walls(prof)  # stalled rung -> half price, held-out irrelevant
    # a climbing rung re-earns full price (new highs >= momentum threshold over 2 readings).
    nb.note_rung_reading(WALL, mid + 6, session_idx=13)
    assert WALL not in nb.zero_win_walls(prof)


def test_rung_transition_feeds_ladder_progress_signal(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb, r0=3)
    hi = RUNG_GRADUATE_SR + 5
    for i in range(4):    # P2': x2 FULL graduation = 4 HI readings
        nb.note_rung_reading(WALL, hi, session_idx=3 + 2 * i)
    assert nb.foci()[0].get("chain_frontier_advanced") is True
    # the next apply consumes it as progress signal (c): the freeze counter resets.
    _update(nb, 7, _mature_profile({WALL: 0.0}),
            foci=[{"skill": WALL, "prereq_tree": []}], forensics={})
    assert nb.foci()[0]["frozen_sessions"] == 0


def test_relay_journal_renders_rung_state(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb, r0=3)
    text = nb.render_for_prompt()
    assert "SPAWN-ANNEAL RELAY" in text
    assert "MUST spawn at floor 3" in text
    assert "Held-out SR staying 0 is EXPECTED" in text


# ---- 4a. level_meta spawn keys ---------------------------------------------------------------------

def test_level_meta_parses_spawn_keys():
    raw = ('x<level_meta>{"type": "DEPTH", "drill_target": null, "siege_wall": "defeat_kobold", '
           '"spawn_floor": 3, "spawn_kit": {"iron_sword": 1, "Torch": 4}}</level_meta>y')
    meta = parse_level_meta(raw)
    assert meta["spawn_floor"] == 3
    assert meta["spawn_kit"] == {"iron_sword": 1, "torch": 4}


def test_level_meta_old_three_key_block_defaults_to_natural_spawn():
    raw = '<level_meta>{"type": "DEPTH", "drill_target": null, "siege_wall": null}</level_meta>'
    meta = parse_level_meta(raw)
    assert meta["spawn_floor"] == 0
    assert meta["spawn_kit"] is None


def test_level_meta_malformed_spawn_values_degrade_to_defaults():
    raw = ('<level_meta>{"type": "DEPTH", "drill_target": null, "siege_wall": null, '
           '"spawn_floor": "deep", "spawn_kit": [1, 2]}</level_meta>')
    meta = parse_level_meta(raw)
    assert meta["spawn_floor"] == 0
    assert meta["spawn_kit"] is None


# ---- 4b. validator R6_SPAWN ------------------------------------------------------------------------

_FOCI = [{"skill": WALL, "prereq_tree": [{"skill": "enter_sewers", "role": "descend"}]}]
_DESC = "Relevant Achievements: DEFEAT_KOBOLD, ENTER_SEWERS\nCompleted Achievements: NONE"


def _meta(spawn_floor, wall=WALL):
    return {"type": "DEPTH", "drill_target": None, "siege_wall": wall,
            "spawn_floor": spawn_floor, "spawn_kit": None}


def test_r6_wrong_floor_on_relay_level_rejected():
    viols = validate_level(_DESC, _meta(0), _FOCI, required_spawn_floors={WALL: 3})
    assert any(v.rule == RULE_SPAWN for v in viols)
    assert reroll_worthy(viols)


def test_r6_correct_floor_passes():
    viols = validate_level(_DESC, _meta(3), _FOCI, required_spawn_floors={WALL: 3})
    assert not any(v.rule == RULE_SPAWN for v in viols)


def test_r6_deep_spawn_without_relay_rejected():
    """v5 lesson pinned in code: a deep spawn is a relay privilege, never a free scaffold."""
    viols = validate_level(_DESC, _meta(2), _FOCI, required_spawn_floors={})
    assert any(v.rule == RULE_SPAWN for v in viols)
    viols = validate_level(_DESC, _meta(2, wall=None), _FOCI, required_spawn_floors=None)
    assert any(v.rule == RULE_SPAWN for v in viols)


def test_r6_absent_meta_or_natural_spawn_is_clean():
    viols = validate_level(_DESC, None, _FOCI, required_spawn_floors={WALL: 3})
    assert not any(v.rule == RULE_SPAWN for v in viols)
    viols = validate_level(_DESC, _meta(0), _FOCI, required_spawn_floors={})
    assert not any(v.rule == RULE_SPAWN for v in viols)


# ---- 4c. modeler parses relay_r0_floor -------------------------------------------------------------

def test_validate_siege_parses_relay_r0_floor():
    raw = {"siege_update": {"foci": [
        {"skill": WALL, "prereq_tree": [], "relay_r0_floor": 3},
        {"skill": "defeat_orc_mage", "prereq_tree": []},
        {"skill": "defeat_troll", "prereq_tree": [], "relay_r0_floor": "not-a-floor"},
    ]}}
    su = Modeler._validate_siege(raw)
    by_skill = {f["skill"]: f for f in su["foci"]}
    assert by_skill[WALL]["relay_r0_floor"] == 3
    assert by_skill["defeat_orc_mage"]["relay_r0_floor"] is None
    assert by_skill["defeat_troll"]["relay_r0_floor"] is None


def test_validate_siege_range_checks_r0_floor():
    raw = {"siege_update": {"foci": [
        {"skill": WALL, "prereq_tree": [], "relay_r0_floor": 0},
        {"skill": "defeat_troll", "prereq_tree": [], "relay_r0_floor": 99},
    ]}}
    su = Modeler._validate_siege(raw)
    assert all(f["relay_r0_floor"] is None for f in su["foci"])


# ---- 4d. gen_manager wiring (real TaskGenerator, stubbed collaborators) ----------------------------
# gen_manager imports jax/craftax at module level, so — repo convention — these wiring tests run in
# the Oscar full suite and SKIP on the local non-jax subset (same boundary as test_siege_ecosystem).

_HAS_JAX = importlib.util.find_spec("jax") is not None
needs_jax = pytest.mark.skipif(not _HAS_JAX, reason="gen_manager imports jax (Oscar full suite only)")

_gmmod = None


def _gen_manager_module():
    global _gmmod
    if _gmmod is None:
        _stub = types.ModuleType("dicode.dreaming.gen_manager")
        _stub.GenManager = object
        _stub.TaskArchive = object
        sys.modules.setdefault("dicode.dreaming.gen_manager", _stub)
        _gm_path = os.path.join(_REPO, "src", "dicode", "dreaming", "gen_manager.py")
        _spec = importlib.util.spec_from_file_location("dicode_v7_gen_manager_relay_test", _gm_path)
        mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(mod)
        _gmmod = mod
    return _gmmod


class _ArchiveFake:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._lock = threading.Lock()


def _real_nb_with_relay(tmp_path, r0=3):
    nb = SiegeNotebook(str(tmp_path / "nb.json"))
    _open_relay(nb, r0=r0)
    return nb


@needs_jax
def test_gap_hint_routes_relay_wall_to_rung_reading_current_rung_only(tmp_path):
    """THE wiring test (fix-series rule: nothing ships 'written but unwired'): the relay wall's
    trained reading must (a) come only from SYSTEM-BUILT levels declared at the CURRENT rung's
    floor (v7fix4 quarantine — an FM-authored level is not reality-anchored), and (b) drive
    note_rung_reading, not the (suspended) gap gate."""
    tg = object.__new__(_gen_manager_module().TaskGenerator)
    tg.config = types.SimpleNamespace(siege_relay_worldgen="base")  # v7fix4 quarantine active
    nb = _real_nb_with_relay(tmp_path, r0=3)
    tg._siege_notebook = nb
    ar = _ArchiveFake()
    hi = RUNG_GRADUATE_SR + 15
    # current-rung, SYSTEM-BUILT level (floor 3) at hi = the only legitimate rung evidence.
    ar.graph.add_node("rung_lvl", siege_wall=WALL, spawn_floor=3, system_built=True,
                      performance_history=[{"sr": hi / 100.0, "session": 9}])
    # v7fix4 quarantine: an FM-authored floor-3 level scoring even HIGHER must be ignored.
    ar.graph.add_node("fm_lvl", siege_wall=WALL, spawn_floor=3,
                      performance_history=[{"sr": 0.99, "session": 9}])
    # a WRONG-floor level and a natural-spawn level are ignored as before.
    ar.graph.add_node("stale_lvl", siege_wall=WALL, spawn_floor=2, system_built=True,
                      performance_history=[{"sr": 0.99, "session": 9}])
    ar.graph.add_node("natural_lvl", siege_wall=WALL,
                      performance_history=[{"sr": 0.98, "session": 9}])
    tg.archive = ar
    # v7fix5.6: archive trained-SR is TELEMETRY only — without a fresh zero-shot eval the
    # ladder must read nothing (counters hold), even with 90%+ trained levels in the graph
    # (the within-session inflation lesson, probe 2026-07-18: entry trained 43 vs real 24).
    tg._render_siege_gap_hint({WALL: 0.0}, session_idx=9)
    r = nb.foci()[0]["relay"]
    assert r["rung_trained"] == []                       # the honest no_fresh beat
    # v7fix5.7-P2' T1: the decision-cadence site consumes NOTHING anymore — even with a
    # fresh eval stored, the gap hint is telemetry-only; the state machine eats the eval at
    # run_dicode Step 4d via consume_rung_eval, once per session.
    nb.note_rung_eval(WALL, {"session": 9, "sr": hi, "spawn_floor": 3, "sub_stage": 0,
                             "n_envs": 512})
    tg._render_siege_gap_hint({WALL: 0.0}, session_idx=9)
    r = nb.foci()[0]["relay"]
    assert r["rung_trained"] == []                       # the P2' pin: render never judges
    ev, st = nb.consume_rung_eval(WALL, 9)
    r = nb.foci()[0]["relay"]
    assert ev == pytest.approx(hi)
    assert r["rung_trained"] == [pytest.approx(hi)]      # the eval number drove the ladder
    assert nb.foci()[0].get("gap_sessions", 0) == 0      # gap gate untouched
    # win3 window fills across per-session consumptions, then x2 graduates end-to-end.
    for s_i in (11, 13, 15):
        nb.note_rung_eval(WALL, {"session": s_i, "sr": hi, "spawn_floor": 3, "sub_stage": 0,
                                 "n_envs": 512})
        ev, st = nb.consume_rung_eval(WALL, s_i)
    assert nb.required_spawn_floor(WALL) == 2


@needs_jax
def test_gap_hint_no_rung_level_reports_no_reading(tmp_path):
    tg = object.__new__(_gen_manager_module().TaskGenerator)
    nb = _real_nb_with_relay(tmp_path, r0=3)
    tg._siege_notebook = nb
    tg.archive = _ArchiveFake()
    tg._render_siege_gap_hint({WALL: 0.0}, session_idx=9)
    r = nb.foci()[0]["relay"]
    assert r["rung_trained"] == [] and r["rung_graduate_streak"] == 0


@needs_jax
def test_directive_renders_relay_contract_and_kit_hint(tmp_path):
    # v7fix4: on the DEFAULT (base) worldgen the relay levels are SYSTEM-BUILT, so the proposer
    # is told "do NOT author" and gets NO kit hint (see
    # test_directive_relay_wall_wording_shifts_under_system_worldgen). The winner-median kit-hint
    # contract — including the telemetry-label canonicalisation — still governs the FM ablation
    # arm (siege_relay_worldgen="fm"), which is what this test pins.
    class _ChainLogFake:
        def latest_fail_summary(self, target):
            if target == WALL:
                return {"n_fail": 50, "n_succ": 0}  # the relay wall has no winners yet
            if target == "enter_dungeon":
                # v7fix2: telemetry keys are the Inventory struct's flattened FIELD names
                # (sword/torches/armour_0..3), never achievement-style compounds — the kit hint
                # canonicalises them onto legal spawn_kit fields before showing the proposer.
                return {"n_fail": 20, "n_succ": 40,
                        "inv": {"sword": {"succ_med": 3, "fail_med": 0, "ready_frac": 0.2},
                                "torches": {"succ_med": 4, "fail_med": 1, "ready_frac": 0.1},
                                "armour_1": {"succ_med": 2, "fail_med": 0, "ready_frac": 0.1}}}
            return None

    tg = object.__new__(_gen_manager_module().TaskGenerator)
    tg.config = types.SimpleNamespace(siege_relay_worldgen="fm")  # v7fix4 FM ablation arm
    nb = _real_nb_with_relay(tmp_path, r0=3)
    # a tracked non-relay target supplies the winners'-median fallback kit evidence.
    nb._nb.setdefault("watch", {})["enter_dungeon"] = {"skill": "enter_dungeon", "prereq_tree": []}
    tg._siege_notebook = nb
    tg._chain_log = _ChainLogFake()
    text = tg._render_siege_directive(_mature_profile({WALL: 0.0}))
    assert "★SPAWN-ANNEAL RELAY" in text
    assert '"spawn_floor": 3' in text
    assert "WINNER-MEDIAN STOCKPILES" in text
    # canonical spawn_kit field names, ranked by median (armour_1 collapses to armour tier 2)
    assert "torches 4" in text and "sword 3" in text and "armour 2" in text
    # the illegal achievement-style compounds are gone — the hint only speaks legal kit fields
    assert "iron_sword" not in text and "torch 4" not in text


@needs_jax
def test_directive_no_relay_no_relay_block(tmp_path):
    tg = object.__new__(_gen_manager_module().TaskGenerator)
    nb = SiegeNotebook(str(tmp_path / "nb.json"))
    prof = _mature_profile({"defeat_orc_mage": 0.0})
    _update(nb, 1, prof, foci=[{"skill": "defeat_orc_mage", "prereq_tree": []}],
            forensics={"defeat_orc_mage": {"missing_top": []}})
    tg._siege_notebook = nb
    tg._chain_log = None
    text = tg._render_siege_directive(prof)
    assert "SPAWN-ANNEAL RELAY" not in text


@needs_jax
def test_set_level_meta_persists_spawn_floor_only_when_positive():
    ar = _gen_manager_module().TaskArchive.__new__(_gen_manager_module().TaskArchive)
    ar.graph = nx.DiGraph()
    ar._lock = threading.Lock()
    ar.graph.add_node("t1")
    ar.graph.add_node("t2")
    ar.set_level_meta("t1", {"type": "DEPTH", "siege_wall": WALL, "spawn_floor": 3})
    ar.set_level_meta("t2", {"type": "DEPTH", "siege_wall": WALL, "spawn_floor": 0})
    assert ar.graph.nodes["t1"]["spawn_floor"] == 3
    assert "spawn_floor" not in ar.graph.nodes["t2"]  # natural spawn: attr set unchanged (parity)
