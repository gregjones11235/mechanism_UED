"""v7fix5.5 batch 2 — PROBE-AS-TOOL: the modeler's measurement tool behind code gates.

Design (fable_research_reports/v7fix55假设探针课程环设计.md, batch-2 施工设计 + §B hard rules):
the LLM's WHOLE freedom = 4 fields (kind / filter / axis+direction / justification), each behind
a code gate; every number is code-chosen; a probe only MEASURES (its output never writes training
state); budget = per wall <=1 diagnose + <=1 whatif per rolling 10 sessions; compile-failed
filters fall back to random sampling WITHOUT a reprompt (anti negotiation loop); reports STALE
after 5 sessions; modeler never calling the tool degrades to the fix5.4 static ladder exactly.

These tests pin:
  P1 state keys survive _coerce + reload (the fix4.2 lesson).
  P2 gates: non-relay wall, bad kind/axis/direction, budget window, one-pending-at-a-time,
     stall-or-Tier-1 trigger (citation handshake, _verify_relay_defence family).
  P3 filter compiler: whitelist pass, unknown-field/op/range fall back to random (NOT a reject).
  P4 variant table: one code-chosen step per axis, boundary asks refused, pre_light decoupling.
  P5 lifecycle: accept -> pending -> deliver -> report renders; STALE stamping; availability /
     pending / receipt journal lines; the rung reading stream is untouched by delivery.
  P6 (jax) level-code: _relay_level_build renders the variant from knobs (never string surgery)
     and emits the pre_light kwarg; extraction did not change _system_relay_levels output.
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
    PROBE_BUDGET_WINDOW,
    PROBE_SENSORS,
    RUNG_LADDER_RADII,
    RUNG_NEEDS_SLOW,
    SiegeNotebook,
)

WALL = "defeat_kobold"


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


def _set_relay(nb, floor=2, stage=6, stalled=True, readings=(12.0, 13.0, 12.5)):
    r = nb.foci()[0]["relay"]
    r["spawn_floor"] = floor
    r["sub_stage"] = stage
    r["rung_trained"] = list(readings)
    r["readings_since_transition"] = 6 if stalled else 1
    r["gain_log"] = [0.0, 0.0] if stalled else [5.0, 5.0]
    nb._save()
    return r


def _req(kind="diagnose", wall=WALL, **kw):
    base = {"wall": wall, "kind": kind, "justification": "flat at 12 -> 13 -> 12.5"}
    base.update(kw)
    return base


# ---- P1 state keys ---------------------------------------------------------------------------------

def test_probe_state_keys_survive_coerce_and_reload(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    _set_relay(nb)
    nb._admit_probe_request(_req(), 5)
    assert nb._nb["probe_pending"]["wall"] == WALL
    re = SiegeNotebook(nb_path)
    assert re._nb["probe_pending"]["wall"] == WALL
    assert re._nb["probe_ledger"][WALL] == [[5, "diagnose"]]
    assert re._nb["probe_receipt"].startswith("s5: probe_accepted")


# ---- P2 gates --------------------------------------------------------------------------------------

def test_rejects_non_relay_wall_and_bad_enums(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    _set_relay(nb)
    nb._admit_probe_request(_req(wall="defeat_zombie"), 5)
    assert "not_an_active_relay_wall" in nb._nb["probe_receipt"]
    assert nb._nb["probe_pending"] is None
    nb._admit_probe_request(_req(kind="teleport"), 5)
    assert "bad_kind" in nb._nb["probe_receipt"]
    nb._admit_probe_request(_req("whatif", axis="gravity", direction="easier"), 5)
    assert "bad_axis" in nb._nb["probe_receipt"]
    nb._admit_probe_request(_req("whatif", axis="uplock", direction="sideways"), 5)
    assert "bad_direction" in nb._nb["probe_receipt"]
    assert nb._nb["probe_pending"] is None


def test_trigger_needs_stall_or_verified_citation(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    _set_relay(nb, stalled=False)                      # progressing rung
    nb._admit_probe_request(_req(justification="because I have a theory"), 5)
    assert "no_trigger" in nb._nb["probe_receipt"]
    assert nb._nb["probe_pending"] is None
    nb._admit_probe_request(_req(justification="readings sit at 12.5 then 13"), 5)
    assert nb._nb["probe_pending"] is not None          # Tier-1 citation verified


def test_budget_window_and_single_pending(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    _set_relay(nb)
    nb._admit_probe_request(_req(), 10)
    assert nb._nb["probe_pending"]
    nb._admit_probe_request(_req("whatif", axis="uplock", direction="harder"), 10)
    assert "already_pending" in nb._nb["probe_receipt"]
    nb.deliver_probe_report({"wall": WALL, "kind": "diagnose"}, 10)
    assert nb._nb["probe_pending"] is None
    nb._admit_probe_request(_req(), 12)                 # same kind, inside the window
    assert "budget_exhausted" in nb._nb["probe_receipt"]
    nb._admit_probe_request(_req(), 10 + PROBE_BUDGET_WINDOW)   # window rolled past
    assert nb._nb["probe_pending"] is not None


# ---- P3 filter compiler ----------------------------------------------------------------------------

def test_filter_compiles_or_falls_back_random(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    _set_relay(nb)
    nb._admit_probe_request(
        _req(filter={"field": "sleeping", "op": "==", "value": 1}), 5
    )
    p = nb._nb["probe_pending"]
    assert p["filter"] == {"field": "sleeping", "op": "==", "value": 1.0}
    assert p["filter_error"] is None
    # a BAD filter is a FALLBACK, never a rejection (anti negotiation loop) — fresh notebook.
    nb2 = SiegeNotebook(nb_path + ".2")
    _open_relay(nb2)
    _set_relay(nb2)
    nb2._admit_probe_request(
        _req(filter={"field": "mana_level", "op": "==", "value": 3}), 5
    )
    p = nb2._nb["probe_pending"]
    assert p is not None and p["filter"] is None
    assert "unknown_sensor" in p["filter_error"]
    assert "uniform random" in nb2._nb["probe_receipt"]


# ---- P4 the variant table --------------------------------------------------------------------------

def test_variant_steps_are_code_chosen(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    _set_relay(nb, floor=2, stage=6)                    # radius leg: radii[2], no lock, 1.0x
    k, desc, err = nb.probe_variant_knobs(WALL, "uplock", "easier")
    assert err is None and k["uplock"] is True and "uplock" in desc
    k, desc, err = nb.probe_variant_knobs(WALL, "needs_clock", "easier")
    assert err is None and abs(k["needs_multiplier"] - nb.th.rung_needs_mid) < 1e-9
    k, desc, err = nb.probe_variant_knobs(WALL, "radius", "easier")
    assert err is None and k["down_ladder_radius"] == RUNG_LADDER_RADII[1]
    k, desc, err = nb.probe_variant_knobs(WALL, "spawn_anchor", "harder")
    assert err is None and k["down_ladder_radius"] is None
    _set_relay(nb, floor=2, stage=8)                    # easiest radius rung
    k, desc, err = nb.probe_variant_knobs(WALL, "radius", "easier")
    assert err == "radius_at_boundary" and k is None
    _set_relay(nb, floor=2, stage=4)                    # entry + lock + slow
    k, desc, err = nb.probe_variant_knobs(WALL, "spawn_anchor", "easier")
    assert err is None and k["down_ladder_radius"] == RUNG_LADDER_RADII[-1]
    k, desc, err = nb.probe_variant_knobs(WALL, "uplock", "easier")
    assert err == "no_change_on_axis"                   # already locked at stage 4


def test_pre_light_decouples_from_anchor(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    _set_relay(nb, floor=2, stage=6)                    # radius stage: lit by default
    k, desc, err = nb.probe_variant_knobs(WALL, "pre_light", "harder")
    # v7fix5.7 graded ladder: one notch per step — from lit, harder = "ladder" (down-ladder
    # stamp only), NOT straight to dark (the one-shot cliff the anneal exists to remove).
    assert err is None and k["pre_light"] == "ladder" and "ladder" in desc
    k, desc, err = nb.probe_variant_knobs(WALL, "pre_light", "easier")
    assert err == "pre_light_at_boundary"               # already at the brightest level
    _set_relay(nb, floor=2, stage=4)                    # entry stage: dark by default
    k, desc, err = nb.probe_variant_knobs(WALL, "pre_light", "easier")
    assert err is None and k["pre_light"] == "ladder"   # dark -> lit destination first
    assert k["down_ladder_radius"] is None              # anchor unchanged — that is the point
    k, desc, err = nb.probe_variant_knobs(WALL, "pre_light", "harder")
    assert err == "pre_light_at_boundary"               # already at the darkest level


# ---- P5 lifecycle + rendering ----------------------------------------------------------------------

def _report(**kw):
    rep = {
        "wall": WALL, "kind": "diagnose", "ckpt_step": 9000, "n_envs": 256,
        "success_pct": 21.5, "died_pct": 71.1, "timeout_pct": 7.4,
        "filter_used": None, "filter_error": None, "filter_matched": 182,
        "marginals": {"sleeping": {"rate_pct": 38.0},
                      "ladder_dist": {"p25": 4.0, "med": 11.0, "p75": 25.0}},
        "snapshots": [{"floor": 1, "sleeping": 1, "health": 0.0}],
    }
    rep.update(kw)
    return rep


def test_lifecycle_render_and_stale(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    _set_relay(nb)
    txt = nb.render_for_prompt()
    assert "PROBE TOOL available" in txt
    assert "sleeping" in txt and "ladder_dist" in txt   # sensor catalog rendered from code
    nb._admit_probe_request(_req(), 5)
    txt = nb.render_for_prompt()
    assert "PROBE PENDING" in txt and "PROBE RECEIPT" in txt
    nb.deliver_probe_report(_report(), 5)
    before = [dict(f) for f in nb.foci()]
    txt = nb.render_for_prompt()
    assert "PROBE REPORT (diagnose, measured s5" in txt
    assert "sleeping=38%" in txt
    assert "★STALE" not in txt
    # the delivery wrote NOTHING into the campaign state (probe only measures).
    assert [dict(f) for f in nb.foci()] == before
    nb._nb["last_session"] = 11
    nb._save()
    assert "★STALE" in nb.render_for_prompt()


def test_whatif_report_renders_paired_delta(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    _set_relay(nb)
    nb.deliver_probe_report({
        "wall": WALL, "kind": "whatif", "ckpt_step": 9000, "n_envs": 256,
        "step_desc": "uplock False -> True",
        "base_success_pct": 14.1, "variant_success_pct": 21.3, "delta_pp": 7.2,
    }, 5)
    txt = nb.render_for_prompt()
    assert "uplock False -> True" in txt
    assert "14.1% -> variant 21.3%" in txt and "7.2pp" in txt


def test_probe_request_flows_through_apply_llm_update(nb_path):
    nb = SiegeNotebook(nb_path)
    prof = _open_relay(nb)
    nb.apply_llm_update(
        2, prof,
        {"foci": [{"skill": WALL, "prereq_tree": [], "relay_r0_floor": 3}],
         "probe_request": _req()},
        num_snapshots=MATURITY_MIN_SNAPSHOTS, forensics={},
    )
    assert nb._nb["probe_pending"] is not None
    assert nb.last_probe_decision and "probe_accepted" in nb.last_probe_decision


# ---- P6 (jax) level code from knobs ----------------------------------------------------------------

_HAS_JAX = importlib.util.find_spec("jax") is not None
needs_jax = pytest.mark.skipif(not _HAS_JAX, reason="gen_manager imports jax (Oscar full suite)")


def _gen_manager_module():
    gm_path = os.path.join(_REPO, "src", "dicode", "dreaming", "gen_manager.py")
    spec = importlib.util.spec_from_file_location("dicode_v7fix55_gm_probe_test", gm_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeChainLog:
    def latest_fail_summary(self, target):
        return {"n_succ": 4, "inv": {"sword": {"succ_med": 3}, "torches": {"succ_med": 9}}}


@needs_jax
def test_relay_level_build_renders_variant_from_knobs(tmp_path):
    import ast
    import types

    gmm = _gen_manager_module()
    tg = object.__new__(gmm.TaskGenerator)
    nb = SiegeNotebook(str(tmp_path / "nb.json"))
    _open_relay(nb, wall="defeat_lizard")
    r = nb.foci()[0]["relay"]
    r["spawn_floor"] = 2
    r["sub_stage"] = 6
    nb._save()
    tg._siege_notebook = nb
    tg._chain_log = _FakeChainLog()
    tg.config = types.SimpleNamespace(
        siege_relay_worldgen="base", siege_relay_levels_per_session=1,
    )
    base = nb.relay_scaffold("defeat_lizard")
    # pre_light=False variant at a radius stage
    k = dict(base)
    k["pre_light"] = False
    _, code, _, stage = tg._relay_level_build("defeat_lizard", 2, k, False)
    ast.parse(code)
    assert f"down_ladder_radius={base['down_ladder_radius']}, pre_light=False" in code
    # pre_light=True at the entry: light WITHOUT a radius anchor
    ke = nb._stage_knobs(r, 4)
    ke["pre_light"] = True
    _, code2, _, _ = tg._relay_level_build("defeat_lizard", 2, ke, False)
    ast.parse(code2)
    assert "set_starting_floor(2, pre_light=True)" in code2
    # extraction parity: _system_relay_levels emits exactly _relay_level_build's output.
    out = tg._system_relay_levels(session_idx=7)
    _, code_direct, _, _ = tg._relay_level_build(
        "defeat_lizard", 2, nb.relay_scaffold("defeat_lizard"), False
    )
    assert out[0]["_system_code"] == code_direct


@needs_jax
def test_rung_probe_snap_catalog_single_source():
    src = open(os.path.join(_REPO, "src", "dicode", "evaluation", "rung_probe.py"),
               encoding="utf-8").read()
    assert "set(rec) == set(PROBE_SENSORS)" in src      # runtime drift assertion is in place
    assert "from auction.siege_notebook import PROBE_SENSORS" in src
