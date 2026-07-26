"""Offline tests for the v6fix9 FAILURE FORENSICS + ATTRIBUTION GATE + GAP EARLY-STOP layer.

fix9 root cause (job 3691755 deep-dive, 2026-07-08): the modeler's causal narratives were never
grounded in the collected failure data — fail_hist's missing histogram was collected but not
rendered, the modal break line degenerated on enabler walls (place_table artifact), failures had
no death/inventory evidence at all, and nothing checked a claim against the data. These tests pin:

  P1  ChainOrderLog forensic fields: died_frac / after_deepest_med / winners-vs-failures inventory
      summary (all OPTIONAL inputs — old callers keep the old entry shape), their render lines,
      the inventory-readiness ratchet (inventory_advanced) and frontier saturation.
  P2  Modeler._validate_siege attribution gate: claims contradicted by forensics are coerced to
      "unknown" (+ violation strings for the reroll/log); absence of data never rejects.
  P3  SiegeNotebook gap early-stop: forced-DEPTH with no held-out movement for GAP_STALL_PATIENCE
      decisions -> STYLE_REJECTED through the normal retirement machinery; real movement
      re-baselines and re-earns patience.

No jax/craftax/LLM needed.
"""

import pytest

from auction.chain_order_log import (
    _INV_ADVANCE_FRAC,
    _SURVIVAL_LONG,
    ChainOrderLog,
)
from auction.craftax_achievements import ACHIEVEMENT_TO_VALUE, NUM_ACHIEVEMENTS
from auction.modeler import Modeler
from auction.siege_notebook import (
    GAP_FORCE_SESSIONS,
    GAP_STALL_PATIENCE,
    MATURITY_MIN_SNAPSHOTS,
    SiegeNotebook,
)
from auction.tests.test_siege_fix8_economics import _mature_profile, _open


@pytest.fixture
def path(tmp_path):
    return str(tmp_path / "chain.json")


@pytest.fixture
def nb_path(tmp_path):
    return str(tmp_path / "siege_notebook.json")


def _row(steps: dict[str, int]) -> list[int]:
    row = [-1] * NUM_ACHIEVEMENTS
    for name, step in steps.items():
        row[ACHIEVEMENT_TO_VALUE[name]] = int(step)
    return row


_ARMOUR_CHAIN = {"make_iron_armour": ["collect_iron", "place_table"]}


def _armour_session(log, session, ready_fails=0):
    """The armour shape: every episode places a table; 30/100 failures also touch iron; 10 win.

    ``ready_fails`` of the failing episodes carry iron at the winners' level (3) — the knob the
    inventory-readiness ratchet watches. Episode layout (indices matter for max_inv):
      rows 0..69   fails, table only, iron=0
      rows 70..99  fails, iron+table, iron=1 (except the first ``ready_fails`` of them: iron=3)
      rows 100..109 wins, iron=3
    All fails end at step 500 (died=True for 4 of them), wins at step 600.
    """
    episodes = (
        [{"place_table": 1}] * 70
        + [{"collect_iron": 3, "place_table": 1}] * 30
        + [{"collect_iron": 3, "place_table": 1, "make_iron_armour": 9}] * 10
    )
    rows = [_row(ep) for ep in episodes]
    finished = [True] * len(episodes)
    end_steps = [500] * 100 + [600] * 10
    died = [True] * 4 + [False] * 96 + [False] * 10
    max_inv = []
    for i in range(110):
        if i < 70:
            iron = 0
        elif i < 100:
            iron = 3 if (i - 70) < ready_fails else 1
        else:
            iron = 3
        max_inv.append([iron, 2])  # coal identical for winners and failures -> never a gap
    log.add_session(
        session, rows, finished, chain_targets=_ARMOUR_CHAIN,
        end_steps=end_steps, died=died, max_inv=max_inv, inv_names=["iron", "coal"],
    )


# ---- P1: forensic fields + renders ----------------------------------------------------------------

def test_forensic_fields_stored(path):
    log = ChainOrderLog(path)
    _armour_session(log, 1)
    entry = log.latest_fail_summary("make_iron_armour")
    assert entry["died_frac"] == pytest.approx(0.04)
    # deepest achieved link step is 3 (iron) or 1 (table-only); end=500 -> median ~499
    assert entry["after_deepest_med"] >= _SURVIVAL_LONG
    inv = entry["inv"]
    assert inv["iron"]["succ_med"] == 3 and inv["iron"]["fail_med"] <= 1
    assert inv["iron"]["ready_frac"] == pytest.approx(0.0)
    assert "coal" not in {c for c, d in inv.items() if d["succ_med"] > d["fail_med"]}


def test_old_callers_keep_old_entry_shape(path):
    log = ChainOrderLog(path)
    episodes = [{"place_table": 1}] * 100
    log.add_session(1, [_row(e) for e in episodes], [True] * 100, chain_targets=_ARMOUR_CHAIN)
    entry = log.latest_fail_summary("make_iron_armour")
    assert "died_frac" not in entry and "inv" not in entry and "after_deepest_med" not in entry


def test_render_death_and_inventory_lines(path):
    log = ChainOrderLog(path)
    _armour_session(log, 1)
    hint = log.render_chain_hint("make_iron_armour")
    assert "4% of failures end by DEATH" in hint
    assert "NOT an interruption problem" in hint      # median survival >= _SURVIVAL_LONG
    assert "winners' median 3 vs failures'" in hint   # the iron gap line
    assert "STILL fail the wall" in hint              # artifact guard still active (tail+universal)


def test_inventory_ratchet_advances_and_feeds_unlock(path):
    log = ChainOrderLog(path)
    _armour_session(log, 1, ready_fails=0)
    assert not log.inventory_advanced("make_iron_armour")  # single session: nothing to compare
    _armour_session(log, 2, ready_fails=int(100 * (_INV_ADVANCE_FRAC + 0.02)))
    assert log.inventory_advanced("make_iron_armour")
    _armour_session(log, 3, ready_fails=int(100 * (_INV_ADVANCE_FRAC + 0.03)))
    # only +1pp over the best previous (7%) — below the ratchet, no advance
    assert not log.inventory_advanced("make_iron_armour")


def test_frontier_saturated(path):
    log = ChainOrderLog(path)
    # every failure reaches BOTH links -> mean_depth 2.0 of chain length 2 -> saturated
    episodes = [{"collect_iron": 3, "place_table": 1}] * 100 \
        + [{"collect_iron": 3, "place_table": 1, "make_iron_armour": 9}] * 10
    log.add_session(1, [_row(e) for e in episodes], [True] * 110, chain_targets=_ARMOUR_CHAIN)
    assert log.frontier_saturated("make_iron_armour")
    # the armour shape (30% never touch iron, mean_depth 1.3 < 1.5) is NOT saturated
    log2 = ChainOrderLog(path + "2")
    _armour_session(log2, 1)
    assert not log2.frontier_saturated("make_iron_armour")


def test_forensics_summary(path):
    log = ChainOrderLog(path)
    _armour_session(log, 1)
    fx = log.forensics("make_iron_armour")
    assert fx["break_at_final"] is True                       # tail/universal artifact shape
    assert fx["missing_top"][0][0] == "collect_iron"          # 70% missing
    assert fx["missing_top"][0][1] == pytest.approx(0.7)
    assert fx["died_frac"] == pytest.approx(0.04)
    assert fx["inv_gaps"] and fx["inv_gaps"][0][0] == "iron"
    assert log.forensics("defeat_troll") is None              # untracked wall -> no sample


# ---- P2: the attribution gate ----------------------------------------------------------------------

def _fx_armour():
    return {
        "make_iron_armour": {
            "n_fail": 100,
            "links": ["collect_iron", "place_table"],
            "missing_top": [("collect_iron", 0.7)],
            "break_at_final": True,
            "died_frac": 0.04,
            "after_deepest_med": 499,
            "inv_gaps": [("iron", 3, 1, 0.0)],
        }
    }


def _su(attrib):
    return {"siege_update": {"foci": [{
        "skill": "make_iron_armour", "prereq_tree": [], "style_note": "x",
        "evidence_check": "no_evidence", "failure_attribution": attrib,
    }]}}


def test_attribution_verified_when_consistent():
    su = Modeler._validate_siege(
        _su({"class": "resource_shortfall", "key_missing_link": "collect_iron"}),
        forensics=_fx_armour(),
    )
    a = su["foci"][0]["failure_attribution"]
    assert a["class"] == "resource_shortfall" and a["verified"] is True
    assert a["key_missing_link"] == "collect_iron"
    assert su["attrib_violations"] == []


def test_attribution_combat_claim_rejected_by_death_timing():
    # the armour misdiagnosis shape: "cannot craft under pressure" while failures survive a median
    # 499 steps past the deepest link (death timing is the sole gate — an ambient died_frac floor
    # was calibrated away: with eval horizon == step cap, ~all finished episodes end by death).
    su = Modeler._validate_siege(
        _su({"class": "interrupted_by_combat", "key_missing_link": None}),
        forensics=_fx_armour(),
    )
    a = su["foci"][0]["failure_attribution"]
    assert a["class"] == "unknown" and a["rejected"] == "interrupted_by_combat"
    assert any("not dying at the frontier" in v for v in su["attrib_violations"])


def test_attribution_key_link_must_be_in_missing_top():
    su = Modeler._validate_siege(
        _su({"class": "resource_shortfall", "key_missing_link": "defeat_zombie"}),
        forensics=_fx_armour(),
    )
    a = su["foci"][0]["failure_attribution"]
    assert a["key_missing_link"] is None
    assert any("key_missing_link" in v for v in su["attrib_violations"])


def test_attribution_absent_forensics_never_rejects():
    su = Modeler._validate_siege(
        _su({"class": "interrupted_by_combat", "key_missing_link": None}), forensics={},
    )
    a = su["foci"][0]["failure_attribution"]
    assert a["class"] == "interrupted_by_combat" and a["verified"] is False
    assert su["attrib_violations"] == []


def test_attribution_invalid_class_coerces_unknown():
    su = Modeler._validate_siege(_su({"class": "bad_vibes"}), forensics=_fx_armour())
    assert su["foci"][0]["failure_attribution"]["class"] == "unknown"


# ---- P3: gap early-stop ------------------------------------------------------------------------------

def _force(nb, skill, held=15.0):
    for i in range(GAP_FORCE_SESSIONS):
        status = nb.note_transfer_gap(skill, 95.0, held, session_idx=2 + i)
    assert status.startswith("FORCED_DEPTH")
    return status


def test_gap_early_stop_style_rejected(nb_path):
    nb = SiegeNotebook(nb_path)
    _open(nb, 1, "make_iron_armour")
    _force(nb, "make_iron_armour", held=15.0)
    # held-out flat (below baseline+3pp) for GAP_STALL_PATIENCE forced decisions -> retirement.
    for i in range(GAP_STALL_PATIENCE - 1):
        st = nb.note_transfer_gap("make_iron_armour", 95.0, 16.0, session_idx=10 + i)
        assert st.startswith("FORCED_DEPTH")
    st = nb.note_transfer_gap("make_iron_armour", 95.0, 16.0, session_idx=20)
    assert st == "STYLE_REJECTED"
    assert nb.focus_skills() == []
    snap = nb.snapshot()
    assert "make_iron_armour" in snap["retired"]  # normal machinery: cooldown/blacklist apply
    assert any(h.get("event") == "focus_retired_style_rejected" for h in snap["history"])


def test_gap_early_stop_movement_rebaselines(nb_path):
    nb = SiegeNotebook(nb_path)
    _open(nb, 1, "make_iron_armour")
    _force(nb, "make_iron_armour", held=15.0)
    # two stalls...
    nb.note_transfer_gap("make_iron_armour", 95.0, 15.5, session_idx=10)
    nb.note_transfer_gap("make_iron_armour", 95.0, 16.0, session_idx=11)
    # ...then REAL movement (>= 15 + 3pp): patience re-earned, baseline ratchets to 19.
    st = nb.note_transfer_gap("make_iron_armour", 95.0, 19.0, session_idx=12)
    assert st.startswith("FORCED_DEPTH (stall 0/")
    # two flat readings against the NEW baseline still don't retire (counter was reset).
    nb.note_transfer_gap("make_iron_armour", 95.0, 19.5, session_idx=13)
    nb.note_transfer_gap("make_iron_armour", 95.0, 20.0, session_idx=14)
    assert nb.focus_skills() == ["make_iron_armour"]


def test_gap_close_clears_early_stop_state(nb_path):
    nb = SiegeNotebook(nb_path)
    _open(nb, 1, "make_iron_armour")
    _force(nb, "make_iron_armour", held=15.0)
    nb.note_transfer_gap("make_iron_armour", 95.0, 16.0, session_idx=10)
    # TRUE convergence (fresh >=90 sandbox reading, gap < 30pp) -> force lifts, state wiped.
    assert nb.note_transfer_gap("make_iron_armour", 95.0, 80.0, session_idx=11) == "ok"
    foc = next(f for f in nb.snapshot()["foci"] if f["skill"] == "make_iron_armour")
    assert "gap_force_baseline" not in foc and "gap_stall" not in foc


def test_forced_stays_latched_without_drill_readings(nb_path):
    """v6fix9 audit: after the force fires, drills STOP by design, so the trained reading vanishes
    within the #2 recency window. That must NOT unlatch the force (it would oscillate and starve the
    early-stop of its runway) — the early-stop keeps counting on held-out alone."""
    nb = SiegeNotebook(nb_path)
    _open(nb, 1, "make_iron_armour")
    _force(nb, "make_iron_armour", held=15.0)
    for i in range(GAP_STALL_PATIENCE - 1):
        st = nb.note_transfer_gap("make_iron_armour", None, 16.0, session_idx=10 + i)
        assert st.startswith("FORCED_DEPTH")           # latched despite trained=None
        assert nb.required_form("make_iron_armour") == "DEPTH"
    assert nb.note_transfer_gap("make_iron_armour", None, 16.0, session_idx=20) == "STYLE_REJECTED"
    assert nb.focus_skills() == []


def test_gap_early_stop_combat_double_patience(nb_path):
    """Calibrated on job 3691755: a tier4 COMBAT climber (gnome_warrior, ~0.5pp/session) crosses
    the +3pp ratchet only every ~5-6 decisions — COMBAT walls get 2x patience so real slow climbs
    survive; the enabler patience (3) is pinned by the armour tests above."""
    nb = SiegeNotebook(nb_path)
    _open(nb, 1, "defeat_gnome_warrior")
    _force(nb, "defeat_gnome_warrior", held=4.0)
    for i in range(2 * GAP_STALL_PATIENCE - 1):
        st = nb.note_transfer_gap("defeat_gnome_warrior", 98.0, 5.0, session_idx=10 + i)
        assert st.startswith("FORCED_DEPTH")  # patience 3 would have retired it at i == 2
    st = nb.note_transfer_gap("defeat_gnome_warrior", 98.0, 5.0, session_idx=30)
    assert st == "STYLE_REJECTED"
    assert nb.focus_skills() == []


def test_chain_progress_buys_early_stop_patience(nb_path):
    """v6fix9 audit: P1a constitution — measurable progress ANYWHERE (chain frontier / inventory
    ratchet via note_chain_progress) resets the early-stop counter instead of retiring the wall."""
    nb = SiegeNotebook(nb_path)
    _open(nb, 1, "make_iron_armour")
    _force(nb, "make_iron_armour", held=15.0)
    nb.note_transfer_gap("make_iron_armour", 95.0, 16.0, session_idx=10)
    nb.note_transfer_gap("make_iron_armour", 95.0, 16.0, session_idx=11)
    nb.note_chain_progress("make_iron_armour")          # failures dying deeper / stockpiling more
    st = nb.note_transfer_gap("make_iron_armour", 95.0, 16.0, session_idx=12)
    assert st.startswith("FORCED_DEPTH (stall 0/")      # patience re-earned, not retired
    assert nb.focus_skills() == ["make_iron_armour"]
