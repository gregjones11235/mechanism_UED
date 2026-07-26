"""v7fix4.7 — R0 scaffold + DEFEND-driven relay patience + blacklist exemption.

Zero-shot arbitration 2026-07-14 (jobs 3966171/3966348: golden ckpt-8100, lr=0, the exact roll-1
task_800 R0 level): TRUE zero-shot kobold SR = 0.28% — the 11-18% "first readings" were
within-session PPO bootstrap; the fix4.6 rolls (0.5-2%, x1.4/reading) were the same student whose
takeoff dice missed. Behavioural forensics (kit/floor pre-credits stripped): the student fights,
lights and sleeps underground fine, but drifts UP to its floor-2 comfort income instead of
engaging the R0 target. These tests pin the three repairs + the render/artifact fixes:

  Q1 R0 SCAFFOLD: cliff-split fires AT R0 too (LIT arena away from the entry); R0 skips the
     clear-gate stages 2/1 in BOTH directions (no descent leg) and its scaffold NEVER emits a
     monster credit (the R0 down-gate stays locked); rung_r0_scaffold=False restores fix4.6.
  Q2 DEFEND-DRIVEN PATIENCE: a micro-ratchet log marks strict new maxima of ANY size; at
     patience exhaustion with the ratchet rising and budget left, ONE defence window opens; a
     style_note that CITES the actual readings (verified numerically) resets patience; uncited,
     flat-noise, below-restored-best re-climbs and budget-spent cases all still retire.
     Fixture = roll-1's REAL recorded sequence (0.51 -> 0.74 -> 1.06 -> 1.58 -> 2.38, x1.4).
  Q3 BLACKLIST EXEMPTION: a retirement with the ratchet still rising archives + cooldowns but
     does not stack toward the 2-strikes blacklist (rising_retirements subtracts).
  Q4 RENDER/ARTIFACT: low-SR rung readings render at 1dp (citable); the defence window renders
     with the exact numbers; gen_manager strips reset-time kit/floor pre-credit rows (~100%
     inventory-derived / enter_* achievements) from the task-performance prompt block.

No jax/craftax/LLM needed except the @needs_jax block (Oscar full suite only).
"""

import importlib.util
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
    RELAY_DEFEND_BUDGET,
    RELAY_STALL_PATIENCE,
    RUNG_CLEAR_CREDIT_FULL,
    RUNG_CLIFF_SR,
    RUNG_FLOOR_SR,
    RUNG_GRADUATE_SR,
    RUNG_LADDER_RADII,
    RUNG_STALL_READINGS,
    SiegeNotebook,
    SiegeThresholds,
)


# v7fix5.3: this suite PINS the fix4.7 contract on the pre-5.3 6-stage ladder — the descent
# regime (9 stages) has its own suite in test_siege_fix53_descent_regime.py.
def _nb_old(path, **kw):
    kw.setdefault("rung_descent_regime", False)
    return SiegeNotebook(path, thresholds=SiegeThresholds(**kw))

WALL = "defeat_kobold"                     # COMBAT, native floor 3 -> patience is DOUBLE
HI = RUNG_GRADUATE_SR + 5
LO = RUNG_FLOOR_SR - 5                     # stalls the rung, NOT cliff evidence (> rung_cliff_sr)
ZERO = 0.0
MAX_STAGE = 2 + len(RUNG_LADDER_RADII)
COMBAT_PATIENCE = RELAY_STALL_PATIENCE * 2

# roll-1's REAL recorded R0 readings (v7fix4_s0 backup_fix46roll1, killed s92 with patience 4/6):
ROLL1 = [0.5069708491761723, 0.7374631268436578, 1.0551948051948052,
         1.5772870662460567, 2.380952380952381]


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


def _no_r0_scaffold(nb_path):
    """A notebook with the fix4.6 behaviour (R0 never splits) — the roll-1 reproduction rig."""
    return _nb_old(nb_path, rung_r0_scaffold=False)


def _rising_seq(start=0.51, factor=1.15, n=24):
    # P2': win3-mean new-highs (+2pp) would self-reset patience on a x1.4 tail; a x1.15
    # climb keeps every window delta under the margin while staying strictly ascending.
    v, out = start, []
    for _ in range(n):
        out.append(v)
        v *= factor
    return out


def _drive_to_window(nb, seq, s0=3):
    """Feed readings until the defence window opens; returns (status, next_session_idx, iter)."""
    s = s0
    for i, v in enumerate(seq):
        st = nb.note_rung_reading(WALL, v, session_idx=s)
        s += 2
        if "RELAY_DEFENCE_WINDOW" in st:
            return st, s, i
        assert nb.relay_walls(), f"campaign died before any window: {st}"
    raise AssertionError("no defence window opened")


# ---- Q1: R0 cliff-split ---------------------------------------------------------------------------

def test_r0_cliff_splits_to_lit_arena_with_locked_gate(nb_path):
    nb = _nb_old(nb_path)
    _open_relay(nb)                                       # R0 = floor 3 (habitat-anchored)
    st = nb.note_rung_reading(WALL, ZERO, session_idx=3)
    assert "rung hold" in st                              # 1 cliff reading is not yet a split
    st = nb.note_rung_reading(WALL, ZERO, session_idx=5)
    assert "RUNG_CLIFF_SPLIT" in st and "LIT arena" in st
    r = nb.foci()[0]["relay"]
    assert r["spawn_floor"] == 3 and r["sub_stage"] == MAX_STAGE
    knobs = nb.relay_scaffold(WALL)
    assert knobs["down_ladder_radius"] == RUNG_LADDER_RADII[0]
    assert knobs["monster_credit"] == 0                   # R0 down-gate stays LOCKED


def test_r0_scaffold_off_restores_fix46_never_splits(nb_path):
    nb = _no_r0_scaffold(nb_path)
    _open_relay(nb)
    for i in range(4):
        st = nb.note_rung_reading(WALL, ZERO, session_idx=3 + 2 * i)
        if not nb.relay_walls():
            break                                         # early stop may retire — designed exit
        assert "RUNG_CLIFF_SPLIT" not in st
        assert nb.foci()[0]["relay"]["sub_stage"] == 0


def test_r0_substage_graduation_skips_clear_gate_stages(nb_path):
    nb = _nb_old(nb_path)
    _open_relay(nb)
    nb.note_rung_reading(WALL, ZERO, session_idx=3)
    nb.note_rung_reading(WALL, ZERO, session_idx=5)       # -> split to stage 5
    seen = []
    s = 7
    for _ in range(3):                                    # scaffold stages graduate on x1
        st = None                                         # (P2': 3 window-judged HI per stage)
        for _k in range(4):
            st = nb.note_rung_reading(WALL, HI, session_idx=s)
            s += 2
            if "RUNG_SUBSTAGE_GRADUATED" in st:
                break
        assert "RUNG_SUBSTAGE_GRADUATED" in st
        seen.append(int(nb.foci()[0]["relay"]["sub_stage"]))
    assert seen == [4, 3, 0]                              # 5 -> 4 -> 3 -> FULL (2/1 skipped)
    # v7fix5.3 wording: the skip set is now named generically (descent-leg); the CONTRACT this
    # test pins is the transition path 5 -> 4 -> 3 -> 0, asserted above.
    assert "skips the descent-leg stages" in st


def test_r0_full_stall_regresses_onto_radius_ladder(nb_path):
    nb = _nb_old(nb_path)
    _open_relay(nb)
    s = 3
    for _ in range(2 + RUNG_STALL_READINGS):              # LO > cliff_sr: stall, not a cliff
        st = nb.note_rung_reading(WALL, LO, session_idx=s)   # (P2': window fills first)
        s += 2
    assert "RUNG_SUBSTAGE_REGRESSED" in st
    r = nb.foci()[0]["relay"]
    assert r["spawn_floor"] == 3 and r["sub_stage"] == 3  # FULL -> hardest RADIUS stage (not 1)


def test_below_r0_scaffold_credit_unchanged(nb_path):
    nb = _nb_old(nb_path)
    _open_relay(nb)
    r = nb.foci()[0]["relay"]
    r["spawn_floor"] = 2                                  # below R0: fix4.6 semantics intact
    r["sub_stage"] = 4
    nb._save()
    knobs = nb.relay_scaffold(WALL)
    assert knobs["monster_credit"] == RUNG_CLEAR_CREDIT_FULL
    assert knobs["down_ladder_radius"] == RUNG_LADDER_RADII[1]


# ---- Q2: DEFEND-driven patience -------------------------------------------------------------------

def test_roll1_sequence_opens_window_and_verified_citation_saves_it(nb_path):
    nb = _no_r0_scaffold(nb_path)                         # roll-1's exact rig: R0 FULL, no split
    _open_relay(nb)
    seq = ROLL1 + _rising_seq(ROLL1[-1] * 1.15, 1.15, 12)   # P2': gentle tail (see _rising_seq)
    st, s, _ = _drive_to_window(nb, seq)
    assert nb.relay_walls()                               # retirement HELD
    foc = nb.foci()[0]
    tail = [round(float(x), 1) for x in foc["relay"]["rung_trained"][-3:]]
    journal = nb.render_for_prompt()
    assert "★RELAY DEFENCE WINDOW" in journal
    for x in tail:
        assert str(x) in journal                          # the numbers to cite are shown
    foc["style_note"] = (
        f"slow true climb x1.4/reading: {tail[0]} -> {tail[1]} -> {tail[2]}; "
        f"keep the campaign, takeoff imminent"
    )
    nb._save()
    nxt = foc["relay"]["rung_trained"][-1] * 1.15         # still rising, under the win3 +2pp
    st = nb.note_rung_reading(WALL, nxt, session_idx=s)
    assert "RELAY_DEFENDED" in st
    r = nb.foci()[0]["relay"]
    assert r["stall_patience"] == 0 and r["defends_used"] == 1
    assert r.get("defend_pending") is None


def test_uncited_window_retires_with_blacklist_exemption(nb_path):
    nb = _no_r0_scaffold(nb_path)
    _open_relay(nb)
    st, s, _ = _drive_to_window(nb, _rising_seq())
    foc = nb.foci()[0]
    foc["style_note"] = "it is definitely learning, trust the process"   # narrative, no numbers
    nb._save()
    nxt = foc["relay"]["rung_trained"][-1] * 1.15
    st = nb.note_rung_reading(WALL, nxt, session_idx=s)
    assert "RELAY_RETIRED" in st and "blacklist EXEMPT" in st
    assert not nb.relay_walls()
    reg = nb.snapshot()["retired"][WALL]
    assert reg["count"] == 1 and reg["rising_retirements"] == 1
    assert SiegeNotebook._blacklist_count(reg) == 0       # Q3: does not stack toward blacklist


def test_wrong_numbers_fail_citation(nb_path):
    nb = _no_r0_scaffold(nb_path)
    _open_relay(nb)
    st, s, _ = _drive_to_window(nb, _rising_seq())
    foc = nb.foci()[0]
    foc["style_note"] = "readings 40.0 -> 55.0 -> 62.0 keep climbing"    # fabricated numbers
    nb._save()
    st = nb.note_rung_reading(WALL, foc["relay"]["rung_trained"][-1] * 1.15, session_idx=s)
    assert "RELAY_RETIRED" in st and "UNCITED" in st


def test_flat_noise_gets_no_window(nb_path):
    nb = _no_r0_scaffold(nb_path)
    _open_relay(nb)
    noise = [2.0, 1.9, 2.0, 1.8, 1.9, 2.0, 1.8, 1.9, 2.0, 1.9]
    s, st = 3, ""
    for v in noise:
        st = nb.note_rung_reading(WALL, v, session_idx=s)
        s += 2
        if not nb.relay_walls():
            break
    assert "RELAY_RETIRED" in st and "RELAY_DEFENCE_WINDOW" not in st
    assert "blacklist applies" in st                      # flat noise: no exemption
    reg = nb.snapshot()["retired"][WALL]
    assert reg.get("rising_retirements", 0) == 0


def test_reclimb_below_restored_best_never_defends(nb_path):
    """fix9 #2 / fix4.6 P2 oscillation family stays closed: a cheap re-climb under the rung's
    persisted best sets no strict maxima, so the ratchet never rises and no window opens."""
    nb = _no_r0_scaffold(nb_path)
    _open_relay(nb)
    r = nb.foci()[0]["relay"]
    r["best_rung_trained"] = 50.0                         # the restored (persisted) ratchet
    r["best_by_rung"] = {"3:0": 50.0}
    r["best_win3_by_rung"] = {"3:0": 50.0}                # P2': the win3 anchor restores too
    nb._save()
    s, st = 3, ""
    for v in [21.0, 24.0, 26.0, 28.0, 30.0, 33.0, 36.0, 39.0]:   # rising but all below 50
        st = nb.note_rung_reading(WALL, v, session_idx=s)
        s += 2
        if not nb.relay_walls():
            break
    assert "RELAY_RETIRED" in st and "RELAY_DEFENCE_WINDOW" not in st


def test_defence_budget_caps_the_windows(nb_path):
    nb = _no_r0_scaffold(nb_path)
    _open_relay(nb)
    nb.foci()[0]["relay"]["defends_used"] = RELAY_DEFEND_BUDGET
    nb._save()
    s, st = 3, ""
    for v in _rising_seq():
        st = nb.note_rung_reading(WALL, v, session_idx=s)
        s += 2
        if not nb.relay_walls():
            break
    assert "RELAY_RETIRED" in st and "defence budget spent" in st
    assert "blacklist EXEMPT" in st                       # rising at the cut -> still exempt


def test_new_high_clears_pending_window(nb_path):
    nb = _no_r0_scaffold(nb_path)
    _open_relay(nb)
    st, s, _ = _drive_to_window(nb, _rising_seq())
    r = nb.foci()[0]["relay"]
    big = float(r["best_rung_trained"]) + 10.0            # a REAL new high arrives instead
    st = nb.note_rung_reading(WALL, big, session_idx=s)
    r = nb.foci()[0]["relay"]
    assert r["stall_patience"] == 0 and r.get("defend_pending") is None
    assert int(r.get("defends_used", 0)) == 0             # the window was never consumed


# ---- Q4: render ------------------------------------------------------------------------------------

def test_low_sr_readings_render_one_decimal(nb_path):
    nb = _no_r0_scaffold(nb_path)
    _open_relay(nb)
    nb.note_rung_reading(WALL, ROLL1[0], session_idx=3)
    nb.note_rung_reading(WALL, ROLL1[1], session_idx=5)
    j = nb.render_for_prompt()
    assert "0.5" in j and "0.7" in j                      # not rounded to [0, 1]


# ---- gen_manager artifact strip (Oscar full suite only — gen_manager imports jax) ------------------

_HAS_JAX = importlib.util.find_spec("jax") is not None
needs_jax = pytest.mark.skipif(not _HAS_JAX, reason="gen_manager imports jax (Oscar full suite)")


def _gen_manager_module():
    gm_path = os.path.join(_REPO, "src", "dicode", "dreaming", "gen_manager.py")
    spec = importlib.util.spec_from_file_location("dicode_v7fix47_gm_test", gm_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@needs_jax
def test_system_relay_levels_r0_scaffold_renders_radius_without_credit(tmp_path):
    """An R0 scaffold level carries the radius dial but NEVER a set_monsters_killed line —
    the target floor's down-gate stays locked (v7fix4.7 Q1)."""
    import ast
    import types

    gmm = _gen_manager_module()
    tg = object.__new__(gmm.TaskGenerator)
    nb = _nb_old(str(tmp_path / "nb.json"))
    _open_relay(nb, wall="defeat_lizard")                 # habitat floor 3 = R0
    r = nb.foci()[0]["relay"]
    r["sub_stage"] = MAX_STAGE                            # R0 scaffold stage 5: r<=1, credit 0
    nb._save()
    tg._siege_notebook = nb
    tg._chain_log = type("_FL", (), {"latest_fail_summary": lambda self, t: {
        "n_succ": 4, "inv": {"torches": {"succ_med": 9}}}})()
    tg.config = types.SimpleNamespace(
        siege_relay_worldgen="base", siege_relay_levels_per_session=1,
    )
    out = tg._system_relay_levels(session_idx=7)
    assert len(out) == 1
    code = out[0]["_system_code"]
    assert f"down_ladder_radius={RUNG_LADDER_RADII[0]}" in code
    assert "set_monsters_killed" not in code              # locked down-gate at R0
    ast.parse(code)
    assert out[0]["level_meta"]["spawn_monster_credit"] == 0


@needs_jax
def test_task_performance_context_strips_precredit_artifacts():
    gmm = _gen_manager_module()
    tg = object.__new__(gmm.TaskGenerator)
    profile = {
        "sr": 0.0028,
        "achievement_srs": {
            "collect_wood": 100.0,        # kit pre-credit -> stripped
            "enter_sewers": 100.0,        # spawn-floor pre-credit -> stripped
            "make_iron_pickaxe": 99.9,    # kit pre-credit -> stripped
            "place_torch": 69.2,          # behaviour -> kept
            "collect_wood_2": 100.0,      # unknown name -> kept (only the known set strips)
            "defeat_gnome_warrior": 31.4, # behaviour -> kept
            "enter_dungeon": 88.4,        # below 99.5 -> kept (really entered, not pre-credited)
        },
    }
    out = tg._format_task_performance_context(profile)
    assert "place_torch: 69.20%" in out
    assert "defeat_gnome_warrior: 31.40%" in out
    assert "enter_dungeon: 88.40%" in out
    assert "collect_wood_2: 100.00%" in out
    assert "- collect_wood: 100.00%" not in out
    assert "- enter_sewers: 100.00%" not in out
    assert "- make_iron_pickaxe: 99.90%" not in out
    assert "pre-credit artifacts" in out
    assert "collect_wood" in out.split("pre-credit artifacts")[1]   # named in the omission note
