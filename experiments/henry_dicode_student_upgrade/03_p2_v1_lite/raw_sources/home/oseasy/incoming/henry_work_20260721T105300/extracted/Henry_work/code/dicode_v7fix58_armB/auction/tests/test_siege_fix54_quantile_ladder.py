"""v7fix5.4 — QUANTILE ladder: radius rungs from the floor's MEASURED distance distribution.

Radius probe 2026-07-17 (armA ckpt~17300, 512 paired worlds, regime-on): entry D2D distribution
runs P50=22 / P90=40 while the fix5.3 static radii stop at 20; zero-shot SR across r in
{24,28,34,40} = 66/62/58/55% (all inside the >=50% fast-learning band) vs 30% at entry — and the
CLEAN paired d_old rerun (identical stage-2 code) moved only 25.0% -> 27.2% over ~17 sessions,
so the static entry step both under-samples the needed distances and learns at a crawl. Under
the quantile ladder the radii come from CODE-measured worlds of the actual floor (per-floor
adaptive; the modeler never sets a radius — the v7fix3 law). These tests pin:

  T1 QUANTILE TABLE: with resolved radii [q1..qQ] the ladder is (4+Q)..5 = radius rungs, EVERY
     one under the regime (uplock + slow clocks); 4/3/2/1 = the fix5.3 entry leg verbatim.
  T2 FALLBACKS: unresolved floor / flag off / R0 floor all render the fix5.3 shape byte-for-byte
     (rung_calibration_needed points at exactly the one floor that needs measuring).
  T3 TRANSITIONS: graduation walks (4+Q)..5..4 one stage per reading on the approach floor
     (no skip); the shape helper reports the widened stage count and the shrunken R0 skip set.
  T4 PERSISTENCE: floor_d2d_radii survives a reload (schema-whitelisted — the fix4.2 lesson).
  T5 PURE MATH: _quantile_radii is deterministic, sorted, deduped, >= 1.
  T6 RENDERING: gen_manager emits down_ladder_radius=<quantile radius> WITH the uplock block and
     the slow clock at a quantile radius stage (the probe measured this exact combination).
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
    RUNG_GRADUATE_SR,
    RUNG_LADDER_QUANTILES,
    RUNG_LADDER_RADII,
    RUNG_NEEDS_MID,
    RUNG_NEEDS_SLOW,
    RUNG_QUANTILE_LADDER,
    SiegeNotebook,
    SiegeThresholds,
)

try:  # T6 needs jax at import time (gen_manager module import)
    import jax  # noqa: F401
    _HAS_JAX = True
except Exception:  # noqa: BLE001
    _HAS_JAX = False
needs_jax = pytest.mark.skipif(not _HAS_JAX, reason="jax unavailable")

WALL = "defeat_kobold"
HI = RUNG_GRADUATE_SR + 5
QRADII = [10, 16, 22, 30, 40]      # the probe-era floor-2 quantiles, used as fixture values
MAX_Q = 4 + len(QRADII)            # 9


def _mature_profile(extra=None):
    prof = {f"filler_{i}": MATURITY_SKILL_SR + 10 for i in range(MATURITY_MIN_MASTERED)}
    if extra:
        prof.update(extra)
    return prof


@pytest.fixture
def nb_path(tmp_path):
    return str(tmp_path / "siege_notebook.json")


def _qnb(nb_path):
    return SiegeNotebook(nb_path, thresholds=SiegeThresholds(rung_quantile_ladder=True))


def _open_relay(nb, wall=WALL, r0=3, session=1):
    prof = _mature_profile({wall: 0.0})
    nb.apply_llm_update(
        session, prof,
        {"foci": [{"skill": wall, "prereq_tree": [], "relay_r0_floor": r0}]},
        num_snapshots=MATURITY_MIN_SNAPSHOTS, forensics={},
    )
    return prof


def _scaffold_at(nb, stage):
    r = nb.foci()[0]["relay"]
    r["sub_stage"] = stage
    nb._save()
    return nb.relay_scaffold(WALL)


# ---- T1: the quantile stage table ------------------------------------------------------------------

def test_quantile_table_regime_covers_every_radius_rung(nb_path):
    nb = _qnb(nb_path)
    _open_relay(nb)
    r = nb.foci()[0]["relay"]
    r["spawn_floor"] = 2
    nb.set_floor_radii(2, QRADII)
    # radius rungs (4+Q)..5: quantile radii easiest-first, ALL locked + slow-clocked
    for i, stage in enumerate(range(MAX_Q, 4, -1)):
        sc = _scaffold_at(nb, stage)
        assert sc["down_ladder_radius"] == QRADII[i], (stage, sc)
        assert sc["uplock"] is True and sc["needs_multiplier"] == pytest.approx(RUNG_NEEDS_SLOW)
        assert sc["monster_credit"] > 0
    # entry leg: fix5.3 verbatim
    sc = _scaffold_at(nb, 4)
    assert sc["down_ladder_radius"] is None and sc["uplock"] is True
    assert sc["needs_multiplier"] == pytest.approx(RUNG_NEEDS_SLOW)
    sc = _scaffold_at(nb, 3)
    assert sc["down_ladder_radius"] is None and sc["uplock"] is True
    assert sc["needs_multiplier"] == pytest.approx(RUNG_NEEDS_MID)
    sc = _scaffold_at(nb, 2)
    assert sc["down_ladder_radius"] is None and sc["uplock"] is False
    assert sc["needs_multiplier"] == pytest.approx(1.0)
    # stage clamp: anything above the table renders the easiest rung
    sc = _scaffold_at(nb, 99)
    assert sc["sub_stage"] == MAX_Q and sc["down_ladder_radius"] == QRADII[0]


# ---- T2: fallbacks ---------------------------------------------------------------------------------

def test_unresolved_floor_falls_back_to_fix53_table(nb_path):
    nb = _qnb(nb_path)
    _open_relay(nb)
    r = nb.foci()[0]["relay"]
    r["spawn_floor"] = 2                       # no set_floor_radii -> unresolved
    for stage in (8, 7, 6, 5, 4, 3, 2, 1):
        got = _scaffold_at(nb, stage)
        ref_nb = SiegeNotebook(nb_path + ".ref", thresholds=SiegeThresholds())
        _open_relay(ref_nb)
        ref_nb.foci()[0]["relay"]["spawn_floor"] = 2
        ref_nb.foci()[0]["relay"]["sub_stage"] = stage
        ref_nb._save()
        assert got == ref_nb.relay_scaffold(WALL), f"stage {stage} drifted from fix5.3"


def test_flag_defaults_off_and_r0_floor_ignores_quantiles(nb_path):
    assert RUNG_QUANTILE_LADDER is False       # armA behavior untouched by default
    nb = _qnb(nb_path)
    _open_relay(nb, r0=3)
    r = nb.foci()[0]["relay"]
    assert int(r["spawn_floor"]) == 3          # relay opens ON the R0 floor
    nb.set_floor_radii(3, QRADII)
    # R0 floor: quantile shape must NOT activate (fix5.3 shape, radius stages stay plain)
    max_stage, skip_hi, radii, quant = nb._ladder_shape(r)
    assert quant is False and max_stage == 5 + len(RUNG_LADDER_RADII) and skip_hi == 5
    assert nb.rung_calibration_needed(WALL) is None       # R0 floor never calibrates
    # approach floor: unresolved -> calibration needed; resolved -> not
    r["spawn_floor"] = 2
    nb._save()
    assert nb.rung_calibration_needed(WALL) == 2
    nb.set_floor_radii(2, QRADII)
    assert nb.rung_calibration_needed(WALL) is None
    # flag off: never needed
    off = SiegeNotebook(nb_path + ".off", thresholds=SiegeThresholds())
    _open_relay(off)
    off.foci()[0]["relay"]["spawn_floor"] = 2
    off._save()
    assert off.rung_calibration_needed(WALL) is None


# ---- T3: transitions under the widened shape --------------------------------------------------------

def test_graduation_walks_the_quantile_ladder_one_stage_per_reading(nb_path):
    nb = _qnb(nb_path)
    _open_relay(nb)
    r = nb.foci()[0]["relay"]
    r["spawn_floor"] = 2
    r["sub_stage"] = MAX_Q
    nb.set_floor_radii(2, QRADII)
    nb._save()
    max_stage, skip_hi, radii, quant = nb._ladder_shape(r)
    assert (max_stage, skip_hi, radii, quant) == (MAX_Q, 4, QRADII, True)
    sess = 3
    for expect in range(MAX_Q - 1, 3, -1):     # 8..4: one graduate per JUDGED WINDOW, NO skips
        s = None                               # (P2': 3 window-judged HI per stage)
        for _k in range(4):
            s = nb.note_rung_reading(WALL, HI, session_idx=sess)
            sess += 2
            if "RUNG_SUBSTAGE_GRADUATED" in s:
                break
        assert "RUNG_SUBSTAGE_GRADUATED" in s and f"-> stage {expect} " in s, s
    assert int(nb.foci()[0]["relay"]["sub_stage"]) == 4


# ---- T4: persistence --------------------------------------------------------------------------------

def test_floor_radii_survive_reload(nb_path):
    nb = _qnb(nb_path)
    _open_relay(nb)
    nb.set_floor_radii(2, QRADII)
    again = _qnb(nb_path)                      # fresh load from disk (through _coerce)
    assert again.floor_radii(2) == QRADII
    assert again.floor_radii(1) is None


# ---- T5: the pure quantile math ---------------------------------------------------------------------

@needs_jax
def test_quantile_radii_pure_math():
    gmm = _gen_manager_module()
    f = gmm.TaskGenerator._quantile_radii
    assert f([10] * 100, RUNG_LADDER_QUANTILES) == [10]           # dedupe collapses
    r = f(list(range(1, 101)), (0.05, 0.25, 0.50, 0.75, 0.90))
    assert r == sorted(r) and all(x >= 1 for x in r) and len(r) == 5
    assert f([0.0, 0.0], (0.5,)) == [1]                            # floor at 1


def _gen_manager_module():
    gm_path = os.path.join(_REPO, "src", "dicode", "dreaming", "gen_manager.py")
    spec = importlib.util.spec_from_file_location("dicode_v7fix54_gm_test", gm_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeChainLog:
    def latest_fail_summary(self, target):
        return {"n_succ": 4, "inv": {"sword": {"succ_med": 3}, "torches": {"succ_med": 9}}}


# ---- T6: gen_manager renders the quantile rung (radius + lock + slow clock together) ----------------

@needs_jax
def test_system_relay_levels_render_quantile_rung(tmp_path):
    import types

    gmm = _gen_manager_module()
    tg = object.__new__(gmm.TaskGenerator)
    nb = SiegeNotebook(
        str(tmp_path / "nb.json"),
        thresholds=SiegeThresholds(rung_quantile_ladder=True),
    )
    _open_relay(nb, wall="defeat_lizard")
    r = nb.foci()[0]["relay"]
    r["spawn_floor"] = 2
    r["sub_stage"] = MAX_Q - 3                 # the P75 rung (QRADII[3]) — the hotfix start stage
    nb._save()
    nb.set_floor_radii(2, QRADII)              # pre-resolved: the calib hook must be a no-op
    tg._siege_notebook = nb
    tg._chain_log = _FakeChainLog()
    tg.config = types.SimpleNamespace(
        siege_relay_worldgen="base", siege_relay_levels_per_session=1,
    )
    out = tg._system_relay_levels(session_idx=7)
    code = out[0]["_system_code"]
    ast.parse(code)
    assert f"down_ladder_radius={QRADII[3]}" in code
    assert f"TaskParams(needs_depletion_multiplier={RUNG_NEEDS_SLOW})" in code
    assert "ladders_up[2]" in code and "ItemType.NONE.value" in code
    meta = out[0]["level_meta"]
    assert meta["spawn_uplock"] is True
    assert meta["spawn_needs_multiplier"] == pytest.approx(RUNG_NEEDS_SLOW)
    assert meta["spawn_ladder_radius"] == QRADII[3]
    assert meta["spawn_sub_stage"] == MAX_Q - 3
    assert "up-ladder REMOVED" in out[0]["description"]
