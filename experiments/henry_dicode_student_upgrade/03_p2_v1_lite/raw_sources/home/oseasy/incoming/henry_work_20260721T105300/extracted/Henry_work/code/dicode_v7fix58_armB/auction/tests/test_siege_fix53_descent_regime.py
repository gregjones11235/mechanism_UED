"""v7fix5.3 — descent-regime scaffold (uplock + needs-clock anneal + radius restart).

Death-forensics probe 2026-07-16 (armA ckpt-15500, jobs 4031672/4046511 — per-episode PHYSICAL
state telemetry, the first attribution of this wall with location data): the stage3->stage2
collapse was never "gnome exposure en route". 87% of stage-2 deaths happened ABOVE the rung
floor — the student abandons the descent, climbs the entry ladder into the uncleared 3x-spawn
floor above and dies there of sleep-kill (40%) / thirst (32%); winners cross in ~26 steps of a
~112-step speedrun. Paired what-if (512 shared worlds, zero training): base 14.1% / needs0.3x
19.1% / uplock 21.3% / BOTH 25.0%; uplock alone relocates wander-sleep-death to the rung floor,
so the lock ships WITH the slow clock. These tests pin:

  T1 STAGE TABLE: the 9-stage descent-regime ladder (8/7/6 = the old radius leg verbatim;
     5 = r[-1]+UPLOCK+slow clock; 4 = entry+UPLOCK+slow (the what-if D condition);
     3 = entry+UPLOCK+mid clock; 2/1/0 = the pre-5.3 entry/clear-gate/FULL stages verbatim).
  T2 ABLATION: rung_descent_regime=False renders the exact pre-5.3 6-stage table (uplock
     always False, needs always 1.0) — the fix46/fix47 suites stay pinned on it.
  T3 TRANSITIONS: cliff-split enters the new easiest stage (8); the ladder climbs 8..1 -> FULL
     one stage per graduate reading; R0 skips stages 5..1 in BOTH directions (descent-leg
     anneals are meaningless on the target floor).
  T4 RENDERING: gen_manager emits needs_depletion_multiplier via TaskParams and the UPLOCK
     post-build block (ladders_up + ItemType.NONE) ONLY at the lock stages; stages 2/1, FULL
     and the kit-strip exam render the exact pre-5.3 code (no ItemType import, plain build).
  T5 NO FM AUTHORITY: the knobs come from the notebook stage table only — the scaffold dict is
     code-computed; prompts render the regime as fact, never as a lever.
"""

import ast
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
    RUNG_FLOOR_SR,
    RUNG_GRADUATE_SR,
    RUNG_LADDER_RADII,
    RUNG_NEEDS_MID,
    RUNG_NEEDS_SLOW,
    SiegeNotebook,
    SiegeThresholds,
)

WALL = "defeat_kobold"        # COMBAT, native floor 3
HI = RUNG_GRADUATE_SR + 5
LO = RUNG_FLOOR_SR - 5
ZERO = 0.0
MAX_STAGE_NEW = 5 + len(RUNG_LADDER_RADII)   # 8 with the default radii
MAX_STAGE_OLD = 2 + len(RUNG_LADDER_RADII)   # 5 (the fix4.6 table)


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


def _graduate_to_floor2(nb, s0=3):
    # P2': x2 FULL graduation on win3 means = window fill (2) + 2 judged passes = 4 HI.
    s = None
    for i in range(4):
        s = nb.note_rung_reading(WALL, HI, session_idx=s0 + 2 * i)
    assert "RUNG_GRADUATED" in s and "floor 2" in s
    return s0 + 8


def _set_stage(nb, stage):
    r = nb.foci()[0]["relay"]
    r["sub_stage"] = stage
    nb._save()
    return r


# ---- T1: the stage table -------------------------------------------------------------------------

def test_stage_table_new_regime(nb_path):
    nb = SiegeNotebook(nb_path)                       # defaults: regime ON
    _open_relay(nb)
    r = nb.foci()[0]["relay"]
    r["spawn_floor"] = 2                              # mid-ladder (below r0=3)
    radii = list(RUNG_LADDER_RADII)
    expected = {
        8: (radii[0], 8, False, 1.0),
        7: (radii[1], 8, False, 1.0),
        6: (radii[2], 8, False, 1.0),
        5: (radii[2], 8, True, RUNG_NEEDS_SLOW),
        4: (None, 8, True, RUNG_NEEDS_SLOW),
        3: (None, 8, True, RUNG_NEEDS_MID),
        2: (None, 8, False, 1.0),
        1: (None, 4, False, 1.0),
    }
    for stage, (rad, cred, lock, needs) in expected.items():
        _set_stage(nb, stage)
        sc = nb.relay_scaffold(WALL)
        assert sc["sub_stage"] == stage
        assert sc["down_ladder_radius"] == rad, (stage, sc)
        assert sc["monster_credit"] == cred, (stage, sc)
        assert sc["uplock"] is lock, (stage, sc)
        assert sc["needs_multiplier"] == pytest.approx(needs), (stage, sc)
    _set_stage(nb, 0)
    assert nb.relay_scaffold(WALL) is None            # FULL renders the pre-4.6 level


def test_stage_table_r0_keeps_gate_locked_and_never_locks_up(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)                                   # spawn_floor == r0 == 3
    _set_stage(nb, 6)
    sc = nb.relay_scaffold(WALL)
    assert sc["monster_credit"] == 0                  # R0 down-gate stays LOCKED (fix4.7 Q1)
    assert sc["uplock"] is False and sc["needs_multiplier"] == 1.0


# ---- T2: ablation renders the exact pre-5.3 table -------------------------------------------------

def test_stage_table_old_regime_pinned(nb_path):
    nb = SiegeNotebook(nb_path, thresholds=SiegeThresholds(rung_descent_regime=False))
    _open_relay(nb)
    r = nb.foci()[0]["relay"]
    r["spawn_floor"] = 2
    radii = list(RUNG_LADDER_RADII)
    expected = {5: (radii[0], 8), 4: (radii[1], 8), 3: (radii[2], 8), 2: (None, 8), 1: (None, 4)}
    for stage, (rad, cred) in expected.items():
        _set_stage(nb, stage)
        sc = nb.relay_scaffold(WALL)
        assert sc["down_ladder_radius"] == rad and sc["monster_credit"] == cred
        assert sc["uplock"] is False and sc["needs_multiplier"] == 1.0
    # an over-max stale stage clamps to the OLD max under the ablation
    _set_stage(nb, 9)
    assert nb.relay_scaffold(WALL)["sub_stage"] == MAX_STAGE_OLD


# ---- T3: transitions on the new ladder ------------------------------------------------------------

def test_cliff_split_enters_stage_8(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    s = _graduate_to_floor2(nb)
    nb.note_rung_reading(WALL, ZERO, session_idx=s)
    st = nb.note_rung_reading(WALL, ZERO, session_idx=s + 2)
    assert "RUNG_CLIFF_SPLIT" in st and f"stage {MAX_STAGE_NEW}" in st
    assert nb.relay_sub_stage(WALL) == MAX_STAGE_NEW


def test_full_climb_passes_through_the_regime_stages(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    s = _graduate_to_floor2(nb)
    nb.note_rung_reading(WALL, ZERO, session_idx=s)
    nb.note_rung_reading(WALL, ZERO, session_idx=s + 2)  # cliff split -> 8
    seen = []
    sess = s + 4
    for _ in range(MAX_STAGE_NEW):                       # 8 -> 7 -> ... -> 1 -> 0 (FULL)
        st = None                                        # (P2': 3 window-judged HI per stage)
        for _k in range(4):
            st = nb.note_rung_reading(WALL, HI, session_idx=sess)
            sess += 2
            if "RUNG_SUBSTAGE_GRADUATED" in st:
                break
        assert "RUNG_SUBSTAGE_GRADUATED" in st, st
        seen.append(nb.relay_sub_stage(WALL))
    assert seen == [7, 6, 5, 4, 3, 2, 1, 0]
    # and the scaffold knobs flipped exactly where the table says (spot checks along the way
    # are covered by T1 — here we assert the ladder never skipped a stage below R0).


def test_r0_graduate_skips_descent_leg_stages(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)                                      # at R0 (floor 3)
    nb.note_rung_reading(WALL, ZERO, session_idx=3)
    st = nb.note_rung_reading(WALL, ZERO, session_idx=5)
    assert "RUNG_CLIFF_SPLIT" in st                      # R0 splits (fix4.7 Q1) -> stage 8
    sess = 7
    for _stage in (7, 6, 0):                             # 8 -> 7 -> 6 -> 0 (skips 5..1)
        st = None                                        # (P2': 3 window-judged HI per stage)
        for _k in range(4):
            st = nb.note_rung_reading(WALL, HI, session_idx=sess)
            sess += 2
            if "RUNG_SUBSTAGE_GRADUATED" in st:
                break
        assert "RUNG_SUBSTAGE_GRADUATED" in st, st
        assert nb.relay_sub_stage(WALL) == _stage
    assert "descent-leg stages 5..1" in st
    assert nb.relay_sub_stage(WALL) == 0


def test_r0_regress_skips_descent_leg_stages(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    r = nb.foci()[0]["relay"]
    # a previously-climbed R0 FULL that now stalls: LO x rung_stall_readings (NOT a fresh cliff —
    # give one healthy reading first so the cliff-split window has passed).
    nb.note_rung_reading(WALL, HI, session_idx=3)
    sess = 5
    st = ""
    for _ in range(10):
        st = nb.note_rung_reading(WALL, LO, session_idx=sess)
        sess += 2
        if "RUNG_SUBSTAGE_REGRESSED" in st:
            break
    assert "RUNG_SUBSTAGE_REGRESSED" in st, st
    assert nb.relay_sub_stage(WALL) == MAX_STAGE_NEW - 2  # 6: FULL stalls onto the radius leg
    del r


# ---- T4: gen_manager rendering --------------------------------------------------------------------

_HAS_JAX = importlib.util.find_spec("jax") is not None
needs_jax = pytest.mark.skipif(not _HAS_JAX, reason="gen_manager imports jax (Oscar full suite)")


def _template_text():
    gm_path = os.path.join(_REPO, "src", "dicode", "dreaming", "gen_manager.py")
    lines = open(gm_path, encoding="utf-8").read().splitlines(keepends=True)
    start = next(i for i, l in enumerate(lines) if "_RELAY_LEVEL_CODE = " in l)
    end = next(i for i in range(start + 1, len(lines)) if lines[i].rstrip() == "'''")
    return "".join(lines[start + 1:end])


def test_template_renders_lock_and_clock_variants_text_level():
    tpl = _template_text()
    lock_tail = (
        "        state = builder.build(rng)\n"
        "        up = builder.ladders_up[2]\n"
        "        state = state.replace(item_map=state.item_map.at[2, up[0], up[1]]"
        ".set(ItemType.NONE.value))\n"
        "        return state"
    )
    locked = tpl.format(
        docstring="Objective: regime rung", wall_enum="DEFEAT_KOBOLD", floor=2,
        radius_arg="", credit_line="        builder.set_monsters_killed(2, 8)\n",
        kit_line="", task_params_args="needs_depletion_multiplier=0.3",
        build_tail=lock_tail, constants_import=", ItemType",
    )
    ast.parse(locked)
    assert "TaskParams(needs_depletion_multiplier=0.3)" in locked
    assert "ladders_up[2]" in locked and "ItemType.NONE.value" in locked
    assert "from craftax.craftax.constants import Achievement, ItemType" in locked
    # the plain render stays byte-compatible with the pre-5.3 template
    plain = tpl.format(
        docstring="Objective: plain rung", wall_enum="DEFEAT_KOBOLD", floor=2,
        radius_arg="", credit_line="", kit_line="",
        task_params_args="", build_tail="        return builder.build(rng)",
        constants_import="",
    )
    ast.parse(plain)
    assert "TaskParams()" in plain
    assert "ItemType" not in plain and "ladders_up" not in plain
    assert plain.rstrip().endswith("return builder.build(rng)")


def _gen_manager_module():
    gm_path = os.path.join(_REPO, "src", "dicode", "dreaming", "gen_manager.py")
    spec = importlib.util.spec_from_file_location("dicode_v7fix53_gm_test", gm_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeChainLog:
    def latest_fail_summary(self, target):
        return {"n_succ": 4, "inv": {"sword": {"succ_med": 3}, "torches": {"succ_med": 9}}}


@needs_jax
def test_system_relay_levels_render_regime_knobs(tmp_path):
    import types

    gmm = _gen_manager_module()
    tg = object.__new__(gmm.TaskGenerator)
    nb = SiegeNotebook(str(tmp_path / "nb.json"))
    _open_relay(nb, wall="defeat_lizard")
    r = nb.foci()[0]["relay"]
    r["spawn_floor"] = 2
    r["sub_stage"] = 4                                    # entry + UPLOCK + slow clock
    nb._save()
    tg._siege_notebook = nb
    tg._chain_log = _FakeChainLog()
    tg.config = types.SimpleNamespace(
        siege_relay_worldgen="base", siege_relay_levels_per_session=1,
    )
    out = tg._system_relay_levels(session_idx=7)
    code = out[0]["_system_code"]
    ast.parse(code)
    assert f"TaskParams(needs_depletion_multiplier={RUNG_NEEDS_SLOW})" in code
    assert "ladders_up[2]" in code and "ItemType.NONE.value" in code
    assert "Achievement, ItemType" in code
    meta = out[0]["level_meta"]
    assert meta["spawn_uplock"] is True
    assert meta["spawn_needs_multiplier"] == pytest.approx(RUNG_NEEDS_SLOW)
    assert "up-ladder REMOVED" in out[0]["description"]
    assert f"survival clocks at {RUNG_NEEDS_SLOW:.1f}x" in out[0]["description"]
    # stage 2 (the old entry stage): the exact pre-5.3 render — no lock, no clock, plain build.
    r["sub_stage"] = 2
    nb._save()
    out = tg._system_relay_levels(session_idx=9)
    code = out[0]["_system_code"]
    ast.parse(code)
    assert "TaskParams()" in code
    assert "ItemType" not in code and "ladders_up" not in code
    assert out[0]["level_meta"]["spawn_uplock"] is False
    assert out[0]["level_meta"]["spawn_needs_multiplier"] is None
    assert "up-ladder REMOVED" not in out[0]["description"]


# ---- T5: prompts report the regime as FACT, never as a lever --------------------------------------

def test_modeler_render_reports_regime_without_granting_authority(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    r = nb.foci()[0]["relay"]
    r["spawn_floor"] = 2
    r["sub_stage"] = 4
    nb.note_rung_reading(WALL, 20.0, session_idx=3)
    txt = nb.render_for_prompt()
    assert "DESCENT REGIME" in txt
    assert "up-ladder is REMOVED" in txt
    assert "code-set, not yours to change" in txt
    assert "survival clocks" in txt and "0.3x" in txt
    # KNOWLEDGE-LEAK BOUNDARY (user 2026-07-16): world-rule FACTS only. No tactic dictation,
    # no researcher-probe numbers — the tactic must come from the modeler's own evidence.
    assert "COMMITTED DESCENT" not in txt
    assert "26 steps" not in txt and "probe-verified" not in txt
    assert "water/sleep" not in txt
    # at a non-regime stage the clause disappears
    r["sub_stage"] = 2
    nb._save()
    txt = nb.render_for_prompt()
    assert "DESCENT REGIME" not in txt
