"""v7fix5.4 designcheck — QUANTILE ladder (code-calibrated per-floor radius rungs).

Motivation pinned by the radius probe (2026-07-17, armA ckpt~17300, 512 paired worlds,
regime-on): entry D2D distribution P50=22 / P90=40 vs the static radii stopping at 20;
zero-shot SR 66/62/58/55% at r in {24,28,34,40} (all in the >=50% fast-learning band) vs 30%
at entry; the CLEAN paired d_old rerun moved 25.0% -> 27.2% in ~17 sessions (+0.13pp/session)
— the static entry step under-samples the needed distances AND learns at a crawl. fix5.4:
radius rungs resolved by CODE from measured worlds of the actual floor, EVERY radius rung
under the descent regime, R0 / unresolved / flag-off all fall back to fix5.3 byte-for-byte.
Run: python v7fix54_designcheck.py -> prints N/N PASS.
"""

import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "src"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)


def _read(*parts):
    with open(os.path.join(HERE, *parts), encoding="utf-8") as f:
        return f.read()


SNB = _read("auction", "siege_notebook.py")
GM = _read("src", "dicode", "dreaming", "gen_manager.py")
T54 = _read("auction", "tests", "test_siege_fix54_quantile_ladder.py")

from auction.siege_notebook import (  # noqa: E402
    MATURITY_MIN_MASTERED,
    MATURITY_MIN_SNAPSHOTS,
    MATURITY_SKILL_SR,
    RUNG_LADDER_QUANTILES,
    RUNG_LADDER_RADII,
    RUNG_NEEDS_MID,
    RUNG_NEEDS_SLOW,
    RUNG_QUANTILE_LADDER,
    SiegeNotebook,
    SiegeThresholds,
)

QR = [10, 16, 22, 30, 40]


def _mk_nb(tmp, **kw):
    th = SiegeThresholds(**kw) if kw else None
    nb = SiegeNotebook(os.path.join(tmp, f"nb_{len(os.listdir(tmp))}.json"),
                       **({"thresholds": th} if th else {}))
    prof = {f"filler_{i}": MATURITY_SKILL_SR + 10 for i in range(MATURITY_MIN_MASTERED)}
    prof["defeat_kobold"] = 0.0
    nb.apply_llm_update(
        1, prof,
        {"foci": [{"skill": "defeat_kobold", "prereq_tree": [], "relay_r0_floor": 3}]},
        num_snapshots=MATURITY_MIN_SNAPSHOTS, forensics={},
    )
    return nb


def _sc(nb, stage, floor=2):
    r = nb.foci()[0]["relay"]
    r["spawn_floor"] = floor
    r["sub_stage"] = stage
    return nb.relay_scaffold("defeat_kobold")


tmp = tempfile.mkdtemp()
q = _mk_nb(tmp, rung_quantile_ladder=True)
q.set_floor_radii(2, QR)
plain = _mk_nb(tmp, rung_quantile_ladder=True)          # unresolved floor
ref = _mk_nb(tmp)                                       # fix5.3 defaults

_quant_rungs_ok = all(
    (lambda s: s["down_ladder_radius"] == QR[i] and s["uplock"] and
     abs(s["needs_multiplier"] - RUNG_NEEDS_SLOW) < 1e-9)(_sc(q, 9 - i))
    for i in range(5)
)
_entry_leg_ok = (
    _sc(q, 4)["uplock"] and _sc(q, 4)["down_ladder_radius"] is None
    and abs(_sc(q, 3)["needs_multiplier"] - RUNG_NEEDS_MID) < 1e-9
    and not _sc(q, 2)["uplock"] and abs(_sc(q, 2)["needs_multiplier"] - 1.0) < 1e-9
)
_fallback_ok = all(_sc(plain, st) == _sc(ref, st) for st in range(1, 9))

_rq = q.foci()[0]["relay"]
_rq["spawn_floor"] = 2
_shape_q = q._ladder_shape(_rq)
_rq["spawn_floor"] = 3                                   # the R0 floor
_shape_r0 = q._ladder_shape(_rq)
_rq["spawn_floor"] = 2

CHECKS = [
    ("L.1 master switch defaults OFF (armA byte-compat) and all three knobs are config-plumbed "
     "(siege_rung_quantile_ladder / _ladder_quantiles / _calib_samples)",
     RUNG_QUANTILE_LADDER is False
     and 'g("siege_rung_quantile_ladder", RUNG_QUANTILE_LADDER)' in SNB
     and 'g("siege_rung_ladder_quantiles", RUNG_LADDER_QUANTILES)' in SNB
     and 'g("siege_rung_calib_samples", RUNG_CALIB_SAMPLES)' in SNB),
    ("L.2 floor_d2d_radii lives in the _empty_notebook schema (the fix4.2 _coerce lesson) and "
     "survives reload via floor_radii/set_floor_radii",
     '"floor_d2d_radii": {}' in SNB
     and SiegeNotebook(q.path).floor_radii(2) == QR),
    ("L.3 ONE ladder-shape source: relay_scaffold and note_rung_reading both derive the table "
     "from _ladder_shape (no drift-prone second copy)",
     SNB.count("self._ladder_shape(r)") >= 2
     and "max_stage = (5 if _regime else 2)" not in SNB),
    ("L.4 quantile table: 5 radius rungs (9..5), EVERY one uplock + slow clock (the probe's "
     "dose-response was measured regime-on)",
     _quant_rungs_ok and _shape_q == (9, 4, QR, True)),
    ("L.5 entry leg verbatim fix5.3 (4=lock+slow, 3=lock+mid, 2/1 plain)", _entry_leg_ok),
    ("L.6 fallbacks are fix5.3 byte-for-byte: unresolved floor AND the R0 floor (plain radius "
     "legs, skip set 5)",
     _fallback_ok and _shape_r0 == (5 + len(RUNG_LADDER_RADII), 5, list(RUNG_LADDER_RADII), False)),
    ("L.7 calibration is code-owned + deterministic: fixed PRNGKey, banner names the law, "
     "the modeler never writes a radius",
     "PRNGKey(54_000 + int(floor))" in GM
     and "[siege][RUNG-CALIB]" in GM
     and "never the modeler's to set" in GM
     and "rung_calibration_needed" in GM),
    ("L.8 calibration fires BEFORE the session's relay builds and only for unresolved approach "
     "floors (R0/resolved/off -> no-op)",
     GM.index("rung_calibration_needed") < GM.index('out: list[dict] = []')
     and q.rung_calibration_needed("defeat_kobold") is None
     and plain.rung_calibration_needed("defeat_kobold") == 2
     and ref.rung_calibration_needed("defeat_kobold") is None),
    ("L.9 quantile math is a pure, unit-tested staticmethod (sorted / deduped / >=1)",
     "def _quantile_radii(dists, quantiles)" in GM
     and "test_quantile_radii_pure_math" in T54),
    ("L.10 fix5.4 test suite pins table/fallback/transitions/persistence/rendering",
     all(t in T54 for t in (
         "test_quantile_table_regime_covers_every_radius_rung",
         "test_unresolved_floor_falls_back_to_fix53_table",
         "test_flag_defaults_off_and_r0_floor_ignores_quantiles",
         "test_graduation_walks_the_quantile_ladder_one_stage_per_reading",
         "test_floor_radii_survive_reload",
         "test_system_relay_levels_render_quantile_rung",
     ))),
]

n_ok = sum(bool(v) for _, v in CHECKS)
for label, v in CHECKS:
    print(("PASS " if v else "FAIL ") + " " + label)
print(f"{n_ok}/{len(CHECKS)} fix5.4 design points hold")
sys.exit(0 if n_ok == len(CHECKS) else 1)
