"""v7fix4.6 — descent-wall cliff-split sub-rungs + oscillation liveness + succession widening.

Double post-mortem 2026-07-13 (fast 3941260 / 8100-replay 3940678, reproduced on BOTH arms):
kobold's R0 (floor-3 point-blank) graduated cleanly at 73%, then the floor-2 rung read 0% flat —
Craftax's descent is a compound gate (8 kills unlock the down ladder + a dark-floor traversal to a
random far tile), so one whole floor of descent is a zero-success cliff with no PPO gradient.
These tests pin the three repairs:

  P1 CLIFF SPLIT: a fresh FULL rung below R0 reading <= rung_cliff_sr for rung_cliff_readings
     consecutive readings splits to the easiest scaffold sub-stage (fires BEFORE the stall
     count); scaffold stages graduate on x1 back toward FULL (which keeps x2 = the floor
     graduation); a scaffold stall steps one stage easier; only the easiest stage regresses the
     floor; per-floor resume memory; the kit-strip exam never splits (R0 splits since v7fix4.7 —
     see test_siege_fix47_r0_scaffold_defend; the old pin moved to the ablation switch).
  P2 LIVENESS: best_by_rung persists each rung's new-high ratchet across transitions (a
     revisit's cheap re-climb is NOT a "new high" -> patience burns across oscillation), and
     relay_max_regressions bounds all regress-family moves — the move over budget retires
     through the normal machinery (attribution stays consumable by the succession).
  P3 SUCCESSION WIDENING: verified execution_failure with a named non-entrance key qualifies
     exactly like chain_unreached; an entrance key never re-enters via the -1 override.

No jax/craftax/LLM needed (the gen_manager rendering tests are Oscar-only, @needs_jax).
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
    RELAY_MAX_REGRESSIONS,
    RUNG_CLIFF_READINGS,
    RUNG_CLIFF_SR,
    RUNG_FLOOR_SR,
    RUNG_GRADUATE_SR,
    RUNG_LADDER_RADII,
    RUNG_STALL_READINGS,
    SiegeNotebook,
    SiegeThresholds,
    _new_relay,
)

WALL = "defeat_kobold"       # COMBAT, native floor 3
HI = RUNG_GRADUATE_SR + 5
LO = RUNG_FLOOR_SR - 5       # stalls the rung but is NOT cliff evidence (> rung_cliff_sr)
ZERO = 0.0                   # cliff evidence
MAX_STAGE = 2 + len(RUNG_LADDER_RADII)


# v7fix5.3: this whole suite PINS the pre-5.3 6-stage ladder (the fix4.6 contract) behind the
# ablation switch — the descent regime (9 stages, uplock + needs-clock anneal) has its own
# suite in test_siege_fix53_descent_regime.py.
def _nb_old(path, **kw):
    kw.setdefault("rung_descent_regime", False)
    return SiegeNotebook(path, thresholds=SiegeThresholds(**kw))


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
    # P2' (fix56设计 §3.2): judgments ride the last-3 window mean — the x2 FULL graduation
    # needs the window full (2 fill readings) + 2 judged passes = 4 HI readings.
    s = None
    for i in range(4):
        s = nb.note_rung_reading(WALL, HI, session_idx=s0 + 2 * i)
    assert "RUNG_GRADUATED" in s and "floor 2" in s
    return s0 + 8


# ---- P1: cliff split ------------------------------------------------------------------------------

def test_cliff_split_fires_before_stall_and_targets_easiest_stage(nb_path):
    nb = _nb_old(nb_path)
    _open_relay(nb)
    s = _graduate_to_floor2(nb)
    # first cliff reading: NOT yet a split (needs rung_cliff_readings consecutive).
    st = nb.note_rung_reading(WALL, ZERO, session_idx=s)
    assert "rung hold" in st
    st = nb.note_rung_reading(WALL, ZERO, session_idx=s + 2)
    assert "RUNG_CLIFF_SPLIT" in st                      # 2 readings, NOT the 4-reading stall
    r = nb.foci()[0]["relay"]
    assert r["spawn_floor"] == 2                          # same floor —
    assert r["sub_stage"] == MAX_STAGE                    # — easiest scaffold stage
    assert r["rung_trained"] == []                        # fresh rung
    assert nb.relay_sub_stage(WALL) == MAX_STAGE


def test_cliff_needs_all_readings_at_or_below_threshold(nb_path):
    nb = _nb_old(nb_path)
    _open_relay(nb)
    s = _graduate_to_floor2(nb)
    nb.note_rung_reading(WALL, RUNG_CLIFF_SR + 3, session_idx=s)   # 8% — above cliff line
    st = nb.note_rung_reading(WALL, ZERO, session_idx=s + 2)
    assert "RUNG_CLIFF_SPLIT" not in st                   # mixed readings: not a from-zero cliff
    assert nb.relay_sub_stage(WALL) == 0


def test_r0_never_splits_with_r0_scaffold_off(nb_path):
    # v7fix4.7 Q1 changed the default (R0 now splits — see test_siege_fix47_r0_scaffold_defend);
    # this pin moves to the ablation switch, which restores the exact fix4.6 behaviour.
    nb = _nb_old(nb_path, rung_r0_scaffold=False)
    _open_relay(nb)                                       # R0 = floor 3 (habitat-anchored)
    for i in range(RUNG_CLIFF_READINGS + RUNG_STALL_READINGS):
        st = nb.note_rung_reading(WALL, ZERO, session_idx=3 + 2 * i)
        if not nb.relay_walls():
            break                                         # early stop may retire — designed exit
        assert "RUNG_CLIFF_SPLIT" not in st
        assert nb.foci()[0]["relay"]["sub_stage"] == 0
        assert nb.foci()[0]["relay"]["spawn_floor"] == 3  # nowhere deeper than R0 either


def test_kit_strip_exam_never_splits(nb_path):
    nb = _nb_old(nb_path)
    _open_relay(nb)
    sidx = 3
    for _ in range(60):                                   # graduate the whole ladder to KIT_STRIP
        nb.note_rung_reading(WALL, HI, session_idx=sidx)  # (P2': ~4 window-judged HI per rung)
        sidx += 2
        if nb.relay_kit_stripped(WALL):
            break
    assert nb.relay_kit_stripped(WALL) is True
    st = nb.note_rung_reading(WALL, ZERO, session_idx=sidx)
    st = nb.note_rung_reading(WALL, ZERO, session_idx=sidx + 2)
    assert "RUNG_CLIFF_SPLIT" not in (st or "")
    assert nb.relay_kit_stripped(WALL) is True            # still the exam, not a scaffold


def test_substage_ladder_climbs_back_to_full_and_floor_graduates_on_x2(nb_path):
    nb = _nb_old(nb_path)
    _open_relay(nb)
    s = _graduate_to_floor2(nb)
    nb.note_rung_reading(WALL, ZERO, session_idx=s)
    nb.note_rung_reading(WALL, ZERO, session_idx=s + 2)   # -> split to MAX_STAGE
    sidx = s + 4
    for stage in range(MAX_STAGE, 0, -1):                 # x1 graduation per scaffold stage
        st = None                                         # (P2': 3 window-judged HI per stage)
        for _ in range(4):
            st = nb.note_rung_reading(WALL, HI, session_idx=sidx)
            sidx += 2
            if "RUNG_SUBSTAGE_GRADUATED" in st:
                break
        assert "RUNG_SUBSTAGE_GRADUATED" in st and f"stage {stage} -> stage {stage - 1}" in st
    r = nb.foci()[0]["relay"]
    assert r["sub_stage"] == 0 and r["spawn_floor"] == 2  # back at FULL, same floor
    # FULL keeps the x2 confirmation: a first window-judged HI must NOT graduate the floor.
    for _ in range(3):                                    # window fills, then ONE judged pass
        st = nb.note_rung_reading(WALL, HI, session_idx=sidx)
        sidx += 2
    assert "rung hold" in st
    st = nb.note_rung_reading(WALL, HI, session_idx=sidx)
    assert "RUNG_GRADUATED" in st and "floor 1" in st


def test_full_stall_steps_one_substage_easier_not_a_floor(nb_path):
    nb = _nb_old(nb_path)
    _open_relay(nb)
    s = _graduate_to_floor2(nb)
    st = None
    for i in range(2 + RUNG_STALL_READINGS):              # LO stalls but is not cliff evidence
        st = nb.note_rung_reading(WALL, LO, session_idx=s + 2 * i)   # (P2': window fills first)
    assert "RUNG_SUBSTAGE_REGRESSED" in st and "stage 0 -> easier scaffold stage 1" in st
    r = nb.foci()[0]["relay"]
    assert r["spawn_floor"] == 2 and r["sub_stage"] == 1  # within-floor, no floor regress


def test_easiest_stage_stall_regresses_floor_and_resume_memory_restores_stage(nb_path):
    nb = _nb_old(nb_path)
    _open_relay(nb)
    s = _graduate_to_floor2(nb)
    nb.note_rung_reading(WALL, ZERO, session_idx=s)
    nb.note_rung_reading(WALL, ZERO, session_idx=s + 2)   # split -> MAX_STAGE
    st = None
    for i in range(2 + RUNG_STALL_READINGS):              # (P2': window fills first)
        st = nb.note_rung_reading(WALL, ZERO, session_idx=s + 4 + 2 * i)
    assert "RUNG_REGRESSED" in st and "floor 3" in st     # easiest scaffold stalled -> floor down
    # re-graduating R0 must resume floor 2 at the REMEMBERED stage, not at FULL.
    sidx = s + 4 + 2 * (2 + RUNG_STALL_READINGS)
    for _ in range(4):                                    # (P2': x2 FULL = 4 HI readings)
        st = nb.note_rung_reading(WALL, HI, session_idx=sidx)
        sidx += 2
        if "RUNG_GRADUATED" in st:
            break
    assert "RUNG_GRADUATED" in st and f"resuming scaffold stage {MAX_STAGE}" in st
    r = nb.foci()[0]["relay"]
    assert r["spawn_floor"] == 2 and r["sub_stage"] == MAX_STAGE


# ---- P2: liveness ---------------------------------------------------------------------------------

def test_regress_budget_retires_an_orbiting_ladder(nb_path):
    """The 2026-07-13 oscillation, replayed: with the budget the campaign RETIRES (attribution
    stays consumable by the succession) instead of orbiting floor3<->floor2 to the wall clock."""
    nb = _nb_old(nb_path)
    _open_relay(nb)
    sidx = _graduate_to_floor2(nb)
    nb.note_rung_reading(WALL, ZERO, session_idx=sidx)
    st = nb.note_rung_reading(WALL, ZERO, session_idx=sidx + 2)
    assert "RUNG_CLIFF_SPLIT" in st                       # split itself is NOT a regress move
    sidx += 4
    for _cycle in range(RELAY_MAX_REGRESSIONS + 1):       # orbit until the budget bites
        st = None
        for _ in range(2 + RUNG_STALL_READINGS):          # stall the easiest scaffold stage
            st = nb.note_rung_reading(WALL, ZERO, session_idx=sidx)   # (P2': window fills first)
            sidx += 2
            if "RELAY_RETIRED" in (st or "") or "RUNG_REGRESSED" in (st or ""):
                break
        if "RELAY_RETIRED" in (st or ""):
            break
        assert "RUNG_REGRESSED" in st                     # floor regress consumed one budget move
        for _ in range(4):                                # cheap re-climb of the graduated R0
            st = nb.note_rung_reading(WALL, HI, session_idx=sidx)     # (P2': x2 = 4 readings)
            sidx += 2
            if "RUNG_GRADUATED" in (st or "") or "RELAY_RETIRED" in (st or ""):
                break
        if "RELAY_RETIRED" in (st or ""):
            break
    assert "RELAY_RETIRED" in st and "regress budget" in st
    assert nb.relay_walls() == []
    reg = nb.retired_registry()[WALL]
    assert reg["last_event"] == "focus_retired_relay_stalled"  # succession-consumable exit


def test_best_by_rung_blocks_the_fake_new_high_patience_reset(nb_path):
    """The liveness hole itself: pre-4.6, every transition None-reset the ratchet, so the first
    post-transition reading was always a 'new high' and patience restarted each oscillation
    cycle. Scaffold OFF isolates the pure floor oscillation."""
    nb = _nb_old(nb_path, rung_cliff_split=False)
    _open_relay(nb)
    _graduate_to_floor2(nb)                               # best_by_rung["3:0"] == HI now
    patience_pre = None
    for i in range(2 + RUNG_STALL_READINGS):              # stall floor 2 -> regress back to 3
        st = nb.note_rung_reading(WALL, LO, session_idx=7 + 2 * i)   # (P2': window fills first)
        if "patience" in st:
            patience_pre = st
    assert "RUNG_REGRESSED" in st
    # back at floor 3: patience keeps its count across the oscillation instead of resetting
    # to 0. (P2' trace: floor-2 readings 4/5 burnt 2 after the first full window's new-high;
    # the regress reading returns before the patience beat; the floor-3 reading below is
    # window-filling and judges nothing — the count stays.)
    st = nb.note_rung_reading(WALL, HI, session_idx=25)
    assert "patience 0/" not in st
    assert "patience 2/" in st
    del patience_pre


def test_new_relay_fields_and_state_survives_reload(nb_path):
    r = _new_relay(3, 1)
    for key, empty in (
        ("sub_stage", 0), ("sub_stage_by_floor", {}), ("best_by_rung", {}), ("regress_count", 0),
    ):
        assert r[key] == empty
    nb = _nb_old(nb_path)
    _open_relay(nb)
    s = _graduate_to_floor2(nb)
    nb.note_rung_reading(WALL, ZERO, session_idx=s)
    nb.note_rung_reading(WALL, ZERO, session_idx=s + 2)   # split
    nb2 = _nb_old(nb_path)                          # reload from disk
    r2 = nb2.foci()[0]["relay"]
    assert r2["sub_stage"] == MAX_STAGE
    assert r2["best_by_rung"].get("3:0") == HI            # per-rung ratchet survived
    assert r2["sub_stage_by_floor"].get("2") == MAX_STAGE


def test_old_notebook_without_fix46_keys_still_reads(nb_path):
    """A resumed pre-4.6 notebook (relay dict without the new keys) must not crash the machine."""
    nb = _nb_old(nb_path)
    _open_relay(nb)
    r = nb.foci()[0]["relay"]
    for k in ("sub_stage", "sub_stage_by_floor", "best_by_rung", "regress_count"):
        r.pop(k, None)                                    # simulate the old on-disk shape
    nb._save()
    nb2 = _nb_old(nb_path)
    st = nb2.note_rung_reading(WALL, HI, session_idx=3)
    assert "rung hold" in st                              # defaults kicked in, no KeyError


# ---- P1: scaffold knob mapping --------------------------------------------------------------------

def test_relay_scaffold_stage_to_knob_mapping(nb_path):
    nb = _nb_old(nb_path)
    _open_relay(nb)
    r = nb.foci()[0]["relay"]
    # the fix4.6 knob table is a BELOW-R0 (descent-rung) contract; at R0 the credit is forced
    # to 0 since v7fix4.7 Q1 (locked target-floor down-gate) — pinned in the fix4.7 tests.
    r["spawn_floor"] = 2
    radii = list(RUNG_LADDER_RADII)
    expect = {
        5: (radii[0], 8), 4: (radii[1], 8), 3: (radii[2], 8),
        2: (None, 8), 1: (None, 4),
    }
    for stage, (radius, credit) in expect.items():
        r["sub_stage"] = stage
        sc = nb.relay_scaffold(WALL)
        assert sc["sub_stage"] == stage
        assert sc["down_ladder_radius"] == radius
        assert sc["monster_credit"] == credit
    r["sub_stage"] = 0
    assert nb.relay_scaffold(WALL) is None                # FULL renders the exact pre-4.6 level
    r["sub_stage"] = 3
    r["kit_strip"] = True
    assert nb.relay_scaffold(WALL) is None                # the exam never scaffolds


# ---- P3: succession widening ----------------------------------------------------------------------

def _plant_retirement(nb, wall, cls, key, verified=True, session=50):
    nb._nb.setdefault("retired", {})[wall] = {
        "count": 1, "last_session": session, "last_event": "focus_retired_relay_stalled",
        "links_at_retirement": ["enter_gnomish_mines", "enter_sewers", "enchant_sword",
                                "defeat_lizard"],
        "failure_attribution_at_retirement": {
            "class": cls, "key_missing_link": key, "verified": verified,
        },
    }
    nb._save()


def test_succession_accepts_verified_execution_failure(nb_path):
    nb = _nb_old(nb_path)
    _plant_retirement(nb, WALL, "execution_failure", "enchant_sword")
    succ = nb._relay_succession()
    assert succ is not None
    rank, retired_wall, missing = succ
    assert retired_wall == WALL and missing == "enchant_sword"
    assert rank["enchant_sword"] == -1                    # the diagnosed link outranks the chain
    assert "enter_sewers" not in rank                     # entrances stay excluded


def test_succession_still_accepts_chain_unreached_and_rejects_others(nb_path):
    nb = _nb_old(nb_path)
    _plant_retirement(nb, WALL, "chain_unreached", "enchant_sword")
    assert nb._relay_succession() is not None
    _plant_retirement(nb, WALL, "resource_shortfall", "enchant_sword")
    assert nb._relay_succession() is None                 # class boundary holds elsewhere
    _plant_retirement(nb, WALL, "execution_failure", "enchant_sword", verified=False)
    assert nb._relay_succession() is None                 # unverified never qualifies


def test_succession_entrance_key_never_reenters_via_override(nb_path):
    nb = _nb_old(nb_path)
    _plant_retirement(nb, WALL, "execution_failure", "enter_sewers")
    succ = nb._relay_succession()
    assert succ is not None                               # the chain still succeeds the campaign
    rank, _, missing = succ
    assert missing == ""                                  # but the entrance key is dropped
    assert "enter_sewers" not in rank                     # and never re-enters through the -1


# ---- gen_manager rendering (Oscar full suite only — gen_manager imports jax) ----------------------

_HAS_JAX = importlib.util.find_spec("jax") is not None
needs_jax = pytest.mark.skipif(not _HAS_JAX, reason="gen_manager imports jax (Oscar full suite)")
_HAS_CRAFTAX = importlib.util.find_spec("craftax") is not None
needs_world = pytest.mark.skipif(
    not (_HAS_JAX and _HAS_CRAFTAX), reason="world_builder imports jax+craftax (Oscar full suite)"
)


@needs_world
def test_world_builder_scaffold_spawn_position_light_and_credit():
    """The e2e pin for the two scaffold dials on a REAL generated world: the player spawns
    within Manhattan radius of floor 2's down ladder (not at the up-ladder entry), the spawn
    neighbourhood is torch-lit (dark floors: an unlit scaffold spawn is a blind start), and the
    clear-gate pre-credit lands in monsters_killed."""
    import jax
    import numpy as np

    from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
    from minicraftax.world_builder import WorldBuilder

    rng = jax.random.PRNGKey(7)
    rng, b_rng, build_rng = jax.random.split(rng, 3)
    b = WorldBuilder(b_rng, StaticEnvParams(), EnvParams())
    b.set_starting_floor(2, down_ladder_radius=8)
    b.set_monsters_killed(2, 8)
    state = b.build(build_rng)
    pos = np.asarray(state.player_position)
    ladder = np.asarray(state.down_ladders[2])
    dist = abs(int(pos[0]) - int(ladder[0])) + abs(int(pos[1]) - int(ladder[1]))
    assert int(state.player_level) == 2
    assert 1 <= dist <= 8                                  # within the dial, never ON the ladder
    assert float(np.asarray(state.light_map)[2, pos[0], pos[1]]) > 0.2   # lit spawn
    assert float(np.asarray(state.light_map)[2, ladder[0], ladder[1]]) > 0.2  # lit ladder
    assert int(np.asarray(state.monsters_killed)[2]) == 8  # clear gate pre-credited
    # default path unchanged: no radius -> the up-ladder entry spawn.
    b2 = WorldBuilder(b_rng, StaticEnvParams(), EnvParams())
    b2.set_starting_floor(2)
    state2 = b2.build(build_rng)
    up = np.asarray(state2.up_ladders[2])
    assert (np.asarray(state2.player_position) == up).all()


def _gen_manager_module():
    gm_path = os.path.join(_REPO, "src", "dicode", "dreaming", "gen_manager.py")
    spec = importlib.util.spec_from_file_location("dicode_v7fix46_gm_test", gm_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeChainLog:
    def latest_fail_summary(self, target):
        return {"n_succ": 4, "inv": {"sword": {"succ_med": 3}, "torches": {"succ_med": 9}}}


@needs_jax
def test_system_relay_levels_render_scaffold_knobs(tmp_path):
    import types

    gmm = _gen_manager_module()
    tg = object.__new__(gmm.TaskGenerator)
    nb = _nb_old(str(tmp_path / "nb.json"))
    _open_relay(nb, wall="defeat_lizard")
    r = nb.foci()[0]["relay"]
    r["spawn_floor"] = 2                                  # mid-ladder,
    r["sub_stage"] = 4                                    # scaffold stage 4: r<=8, credit 8
    nb._save()
    tg._siege_notebook = nb
    tg._chain_log = _FakeChainLog()
    tg.config = types.SimpleNamespace(
        siege_relay_worldgen="base", siege_relay_levels_per_session=1,
    )
    out = tg._system_relay_levels(session_idx=7)
    assert len(out) == 1
    code = out[0]["_system_code"]
    assert "set_starting_floor(2, down_ladder_radius=8)" in code
    assert "set_monsters_killed(2, 8)" in code            # clear-gate pre-credit, floor >= 1
    assert "set_player_inventory" in code                 # winner-median kit still rides
    ast.parse(code)
    meta = out[0]["level_meta"]
    assert meta["spawn_sub_stage"] == 4
    assert meta["spawn_ladder_radius"] == 8
    assert meta["spawn_monster_credit"] == 8
    assert "scaffold stage 4" in out[0]["description"]
    # back at FULL: the exact pre-4.6 rendering (no radius arg, no credit line, stage 0).
    r["sub_stage"] = 0
    nb._save()
    out = tg._system_relay_levels(session_idx=9)
    code = out[0]["_system_code"]
    assert "set_starting_floor(2)" in code and "down_ladder_radius" not in code
    assert "set_monsters_killed" not in code
    assert out[0]["level_meta"]["spawn_sub_stage"] == 0
