"""v6fix7 P1a — escalation ladder: adaptive frozen counter, L1-L4, cooldown/blacklist, conquest gate.

Pins the redesigned focus lifecycle:
  - frozen_sessions counts only WHOLE-TREE no-progress sessions (focus SR flat AND links flat AND no
    chain-frontier advance) — tier4 patience: foundations rising keeps the ladder at zero forever;
  - ladder levels at l1/l2/l3, retirement at l4 (never the legacy stall ratchet);
  - retirement archives failed tactics; reopening needs cooldown + a genuinely different tactic;
    two retirements blacklist the wall until new chain evidence;
  - conquest (#8 fix): verified_chains 'verified' + protected_set ONLY after holding mastered_sr for
    conquest_consecutive consecutive snapshots — a one-shot +delta record stays 'progress' and
    protects nothing (fix4 wrote make_iron_pickaxe into verified/protected at 44%).
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for p in (os.path.join(_REPO, "src"), _REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from auction.siege_notebook import (  # noqa: E402
    MATURITY_MIN_MASTERED,
    MATURITY_SKILL_SR,
    SiegeNotebook,
)

WALL = "defeat_gnome_warrior"
LINK = "collect_diamond"


def _mature(extra):
    prof = {f"solid_{i}": MATURITY_SKILL_SR + 10 for i in range(MATURITY_MIN_MASTERED)}
    prof.update(extra)
    return prof


def _nb(tmp_path):
    return SiegeNotebook(path=str(tmp_path / "nb.json"))


def _open_focus(nb, session, wall_sr=1.0, link_sr=10.0, note="initial tactic: rush the ladder"):
    prof = _mature({WALL: wall_sr, LINK: link_sr})
    nb.apply_llm_update(
        session_idx=session,
        latest_profile=prof,
        proposed={
            "foci": [{
                "skill": WALL,
                "prereq_tree": [{"skill": LINK, "role": "gear"}],
                "style_note": note,
            }]
        },
        num_snapshots=10,
    )
    return prof


def _tick(nb, session, wall_sr, link_sr, note=""):
    """One later session keeping the same focus."""
    prof = _mature({WALL: wall_sr, LINK: link_sr})
    nb.apply_llm_update(
        session_idx=session,
        latest_profile=prof,
        proposed={"foci": [{
            "skill": WALL,
            "prereq_tree": [{"skill": LINK, "role": "gear"}],
            "style_note": note,
        }]},
        num_snapshots=10,
    )


def _focus(nb):
    foci = nb.foci()
    return foci[0] if foci else None


# ---------- frozen counter + three progress signals ----------


def test_flat_tree_freezes_and_climbs_ladder(tmp_path):
    nb = _nb(tmp_path)
    _open_focus(nb, 1)
    for s in range(2, 6):  # 4 flat sessions
        _tick(nb, s, wall_sr=1.0, link_sr=10.0)
    foc = _focus(nb)
    assert foc["frozen_sessions"] == 4
    assert foc["ladder_level"] == 1  # >= LADDER_L1(3)


def test_link_progress_keeps_ladder_at_zero(tmp_path):
    """tier4 patience: wall SR stays 0 but the foundation climbs -> never frozen."""
    nb = _nb(tmp_path)
    _open_focus(nb, 1, wall_sr=0.0, link_sr=5.0)
    for s in range(2, 10):
        _tick(nb, s, wall_sr=0.0, link_sr=5.0 + 4.0 * (s - 1))  # link +4pp/session
    foc = _focus(nb)
    assert foc["frozen_sessions"] == 0
    assert foc["ladder_level"] == 0


def test_slope_progress_resets_without_new_best(tmp_path):
    """The fix4 ratchet fix: climbing back from a dip (below old best) is progress via slope."""
    nb = _nb(tmp_path)
    _open_focus(nb, 1, wall_sr=50.0)  # best = 50
    # dip then steady climb 20 -> 30, always below best+3 => no new best, but slope ~ +2pp/session
    for s, sr in enumerate([20.0, 22.0, 24.0, 26.0, 28.0, 30.0], start=2):
        _tick(nb, s, wall_sr=sr, link_sr=10.0)
    foc = _focus(nb)
    assert foc["frozen_sessions"] == 0  # slope keeps it live


def test_chain_frontier_signal_resets(tmp_path):
    nb = _nb(tmp_path)
    _open_focus(nb, 1)
    _tick(nb, 2, wall_sr=1.0, link_sr=10.0)
    assert _focus(nb)["frozen_sessions"] == 1
    nb.note_chain_progress(WALL)  # P2: failure episodes dying deeper
    _tick(nb, 3, wall_sr=1.0, link_sr=10.0)
    assert _focus(nb)["frozen_sessions"] == 0


# ---------- retirement, cooldown, new-tactic, blacklist ----------


def _freeze_until_retired(nb, start_session, sessions=13):
    for s in range(start_session, start_session + sessions):
        _tick(nb, s, wall_sr=1.0, link_sr=10.0)
    return start_session + sessions


def test_retirement_at_l4_archives_failed_tactic(tmp_path):
    nb = _nb(tmp_path)
    _open_focus(nb, 1, note="tactic A: rush with stone sword")
    end = _freeze_until_retired(nb, 2)
    assert nb.foci() == []
    reg = nb.retired_registry()[WALL]
    assert reg["count"] == 1
    assert any("tactic A" in n for n in reg["failed_notes"])
    events = [h["event"] for h in nb.snapshot()["history"]]
    assert "focus_retired_stalled" in events


def test_cooldown_blocks_immediate_reopen(tmp_path):
    nb = _nb(tmp_path)
    _open_focus(nb, 1, note="tactic A: rush with stone sword")
    end = _freeze_until_retired(nb, 2)
    # immediately re-propose the SAME wall with a NEW tactic -> cooldown must still reject
    _tick(nb, end, wall_sr=1.0, link_sr=10.0, note="tactic B: kite with arrows from range")
    assert nb.foci() == []
    assert "cooldown_rejected" in (nb.last_focus_decision or "")


def test_reopen_after_cooldown_needs_genuinely_new_tactic(tmp_path):
    nb = _nb(tmp_path)
    _open_focus(nb, 1, note="tactic A: rush the boss with stone sword and potions")
    end = _freeze_until_retired(nb, 2)
    later = end + nb.th.cooldown_sessions + 1
    # same tactic (high token overlap) -> rejected
    _tick(nb, later, wall_sr=1.0, link_sr=10.0,
          note="tactic A: rush the boss with stone sword and potions again")
    assert nb.foci() == []
    assert "reopen_needs_new_tactic" in (nb.last_focus_decision or "")
    # genuinely different tactic -> reopened
    _tick(nb, later + 1, wall_sr=1.0, link_sr=10.0,
          note="kite from range: arrows, pillar-jump, never melee, pull singles")
    assert nb.focus_skills() == [WALL]


def test_second_retirement_blacklists_until_new_evidence(tmp_path):
    nb = _nb(tmp_path)
    _open_focus(nb, 1, note="tactic A: rush with stone sword")
    end = _freeze_until_retired(nb, 2)
    later = end + nb.th.cooldown_sessions + 1
    _tick(nb, later, wall_sr=1.0, link_sr=10.0,
          note="kite from range with arrows and pillars, never melee")
    assert nb.focus_skills() == [WALL]  # reopened once
    end2 = _freeze_until_retired(nb, later + 1)
    assert nb.retired_registry()[WALL]["count"] == 2
    later2 = end2 + nb.th.cooldown_sessions + 1
    # no new evidence (link SR unchanged) -> blacklisted even with a fresh tactic
    _tick(nb, later2, wall_sr=1.0, link_sr=10.0,
          note="lava moat: dig trench, funnel enemies, torch spam everywhere")
    assert nb.foci() == []
    assert "blacklisted" in (nb.last_focus_decision or "")
    # new evidence: the link SR moved since retirement -> another attempt is earned
    _tick(nb, later2 + 1, wall_sr=1.0, link_sr=40.0,
          note="lava moat: dig trench, funnel enemies, torch spam everywhere")
    assert nb.focus_skills() == [WALL]


# ---------- conquest gate (#8) ----------


def test_delta_record_is_progress_not_conquest(tmp_path):
    nb = _nb(tmp_path)
    _open_focus(nb, 1, wall_sr=1.0)
    _tick(nb, 2, wall_sr=44.0, link_sr=10.0)  # +43pp jump: recorded, but NOT conquered
    chains = nb.verified_chains()
    assert len(chains) == 1
    assert chains[0]["status"] == "progress"
    # fix4 poisoning path is closed: no CONQUEST protection was granted (the raw stored set —
    # protected_set() additionally unions the v6fix10 high-water registry, out of scope here).
    assert nb.snapshot()["protected_set"] == []
    assert nb.focus_skills() == [WALL]       # focus stays active


def test_conquest_requires_consecutive_mastered_snapshots(tmp_path):
    nb = _nb(tmp_path)
    _open_focus(nb, 1, wall_sr=1.0)
    _tick(nb, 2, wall_sr=75.0, link_sr=10.0)   # 1st mastered snapshot — not yet
    assert nb.focus_skills() == [WALL]
    assert nb.snapshot()["protected_set"] == []  # raw stored set (see note above)
    _tick(nb, 3, wall_sr=78.0, link_sr=10.0)   # 2nd consecutive — conquered
    assert nb.foci() == []                     # graceful conquest retirement
    chains = {c["target"]: c for c in nb.verified_chains()}
    assert chains[WALL]["status"] == "verified"
    assert WALL in nb.protected_set() and LINK in nb.protected_set()
    events = [h["event"] for h in nb.snapshot()["history"]]
    assert "focus_conquered" in events and "focus_retired_stalled" not in events


def test_mastered_streak_broken_resets(tmp_path):
    nb = _nb(tmp_path)
    _open_focus(nb, 1, wall_sr=1.0)
    _tick(nb, 2, wall_sr=75.0, link_sr=10.0)
    _tick(nb, 3, wall_sr=40.0, link_sr=10.0)   # streak broken
    _tick(nb, 4, wall_sr=75.0, link_sr=10.0)   # streak restarts at 1
    assert nb.focus_skills() == [WALL]         # still not conquered
    assert WALL not in nb.snapshot()["protected_set"]  # raw stored set (see note above)


# ---------- L2 forced form + L3 tactic revision ----------


def test_required_form_flips_at_l2(tmp_path):
    nb = _nb(tmp_path)
    _open_focus(nb, 1)
    nb.note_siege_level_type(WALL, "CONSOLIDATE")
    for s in range(2, 9):  # 7 frozen sessions -> ladder level 2
        _tick(nb, s, wall_sr=1.0, link_sr=10.0)
    assert _focus(nb)["ladder_level"] == 2
    assert nb.required_form(WALL) == "DEPTH"
    nb.note_siege_level_type(WALL, "DEPTH")
    assert nb.required_form(WALL) == "CONSOLIDATE"


def test_required_form_none_below_l2_or_unknown_form(tmp_path):
    nb = _nb(tmp_path)
    _open_focus(nb, 1)
    assert nb.required_form(WALL) is None      # ladder 0
    for s in range(2, 9):
        _tick(nb, s, wall_sr=1.0, link_sr=10.0)
    assert nb.required_form(WALL) is None      # form never recorded -> cannot force a flip


def test_l3_rejects_rephrased_tactic_accepts_new_one(tmp_path):
    nb = _nb(tmp_path)
    base_note = "rush the boss with stone sword and speed potions through the gate"
    _open_focus(nb, 1, note=base_note)
    for s in range(2, 11):  # 9 frozen -> ladder 3 (leave headroom: retirement fires at 12 frozen)
        _tick(nb, s, wall_sr=1.0, link_sr=10.0)
    assert _focus(nb)["ladder_level"] == 3
    # near-identical note -> rejected, old kept (frozen now 10 — still short of L4)
    _tick(nb, 11, wall_sr=1.0, link_sr=10.0,
          note="rush the boss with stone sword and speed potions through the gate fast")
    assert _focus(nb)["style_note"] == base_note
    events = [h["event"] for h in nb.snapshot()["history"]]
    assert "tactic_revision_rejected" in events
    # materially different note -> accepted (frozen 11 — still short of L4)
    new_note = "abandon melee entirely: ranged arrows, pillar terrain, pull single enemies"
    _tick(nb, 12, wall_sr=1.0, link_sr=10.0, note=new_note)
    assert _focus(nb)["style_note"] == new_note


# ---------- P1c style_note lifecycle (AutoManual-lite) ----------


def _tick_ev(nb, session, note, evidence, wall_sr=1.0, link_sr=10.0):
    prof = _mature({WALL: wall_sr, LINK: link_sr})
    nb.apply_llm_update(
        session_idx=session,
        latest_profile=prof,
        proposed={"foci": [{
            "skill": WALL,
            "prereq_tree": [{"skill": LINK, "role": "gear"}],
            "style_note": note,
            "evidence_check": evidence,
        }]},
        num_snapshots=10,
    )


def test_supported_evidence_keeps_note_active(tmp_path):
    nb = _nb(tmp_path)
    _open_focus(nb, 1, note="mine iron beside furnace, craft immediately")
    _tick_ev(nb, 2, note="", evidence="supported")
    foc = _focus(nb)
    assert foc["note_status"] == "active"
    assert foc["note_last_supported_session"] == 2


def test_unsupported_note_goes_stale(tmp_path):
    nb = _nb(tmp_path)
    _open_focus(nb, 1, note="mine iron beside furnace, craft immediately")
    for s in range(2, 7):  # 5 sessions of no_evidence, > NOTE_STALE_SESSIONS(4) since start
        _tick_ev(nb, s, note="", evidence="no_evidence")
    assert _focus(nb)["note_status"] == "stale"


def test_contradicted_note_demands_material_rewrite(tmp_path):
    nb = _nb(tmp_path)
    base = "melee rush the wall with stone sword and shield"
    _open_focus(nb, 1, note=base)
    _tick_ev(nb, 2, note="", evidence="contradicted")
    assert _focus(nb)["note_status"] == "contradicted"
    # a rephrase of the same tactic is rejected while contradicted
    _tick_ev(nb, 3, note="melee rush the wall with stone sword and shield now", evidence="no_evidence")
    foc = _focus(nb)
    assert foc["style_note"] == base
    events = [h["event"] for h in nb.snapshot()["history"]]
    assert "tactic_revision_rejected" in events
    # a genuinely different tactic is accepted and the note returns to active
    _tick_ev(nb, 4, note="ranged only: arrows from pillars, kite singles, never engage close",
             evidence="no_evidence")
    foc = _focus(nb)
    assert "ranged only" in foc["style_note"]
    assert foc["note_status"] == "active"
