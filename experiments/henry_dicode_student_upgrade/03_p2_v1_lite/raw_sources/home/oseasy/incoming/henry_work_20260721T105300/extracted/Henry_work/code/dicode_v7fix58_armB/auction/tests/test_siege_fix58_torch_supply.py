"""v7fix5.8 arm-A supply axis: kit_torches knob + the pre_light render fidelity fix.

torch57 probe (2026-07-19, ckpt18300, 7 paired cells): the dark cliff is a torch SUPPLY
constraint (kit 10 -> 31.1%, kit 26 -> 52.5%, kit 0 -> 3.1%), and the fix5.7 light-anneal
leg was doubly broken — semantically empty at the entry anchor (S50 == S49 byte-for-byte:
entry spawns are naturally lit by the up-ladder stamp) AND never rendered ("ladder" hit
``bool(_pl55)`` -> pre_light=True, rebuilding the identical lit world). These tests pin:
  1. kit_torches overrides ONLY the kit's torch count (template-rendered, no string surgery);
  2. absent knob -> byte-identical winner-median kit line (regression guard);
  3. "ladder" renders as pre_light='ladder'; bool knobs keep their exact pre-5.8 rendering;
  4. KIT_STRIP exams ignore the knob (held-out semantics stay verbatim).
"""

import ast
import importlib.util
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for p in (os.path.join(_REPO, "src"), _REPO):
    if p not in sys.path:
        sys.path.insert(0, p)


def _gen_manager_module():
    gm_path = os.path.join(_REPO, "src", "dicode", "dreaming", "gen_manager.py")
    spec = importlib.util.spec_from_file_location("dicode_v7fix58_gm_supply_test", gm_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeChainLog:
    def latest_fail_summary(self, target):
        return {"n_succ": 4, "inv": {"sword": {"succ_med": 3}, "torches": {"succ_med": 9}}}


def _tg():
    gmm = _gen_manager_module()
    tg = object.__new__(gmm.TaskGenerator)
    tg._chain_log = _FakeChainLog()
    tg.config = types.SimpleNamespace(
        siege_relay_worldgen="base", siege_relay_levels_per_session=1,
    )
    return tg


_KNOBS = {
    "sub_stage": 50, "down_ladder_radius": None, "monster_credit": 8,
    "uplock": True, "needs_multiplier": 0.3,
}


def test_kit_torches_overrides_only_torches():
    tg = _tg()
    k = dict(_KNOBS, pre_light=False, kit_torches=26)
    _, code, meta, stage = tg._relay_level_build("defeat_kobold", 2, k, False)
    ast.parse(code)
    assert "'torches': 26" in code, code
    assert "'sword': 3" in code, "non-torch kit entries must stay winner-median"
    assert "set_starting_floor(2, pre_light=False)" in code
    assert "torch supply 26" in stage
    assert meta["spawn_kit_torches"] == 26


def test_absent_knob_is_byte_identical_to_pre_fix58():
    tg = _tg()
    base = dict(_KNOBS, pre_light=False)
    _, code_a, meta_a, stage_a = tg._relay_level_build("defeat_kobold", 2, base, False)
    assert "'torches': 9" in code_a, "winner-median kit must be untouched"
    assert "torch supply" not in stage_a
    assert meta_a["spawn_kit_torches"] is None
    # and the knob-present render differs ONLY in the kit line
    _, code_b, _, _ = tg._relay_level_build(
        "defeat_kobold", 2, dict(base, kit_torches=26), False
    )
    diff = [
        (la, lb) for la, lb in zip(code_a.splitlines(), code_b.splitlines()) if la != lb
    ]
    # exactly three lines may move, and every one is a DISCLOSED supply fact: the kit dict
    # renders twice (set_player_inventory + the docstring's kit fact) and the stage
    # disclosure ("torch supply 26") flows into the docstring's Objective line (fix9
    # attribution law: no silent difficulty edits).
    assert len(diff) == 3, diff
    assert sum(1 for la, _ in diff if "'torches':" in la) == 2, diff
    assert any("torch supply 26" in lb for _, lb in diff), diff


def test_pre_light_ladder_renders_faithfully():
    tg = _tg()
    _, code, _, _ = tg._relay_level_build(
        "defeat_kobold", 2, dict(_KNOBS, pre_light="ladder"), False
    )
    ast.parse(code)
    assert "set_starting_floor(2, pre_light='ladder')" in code, (
        "the fix5.7 bool() coercion bug: 'ladder' must not render as True"
    )
    # bool knobs keep their exact pre-5.8 rendering (fix55 designcheck compatibility)
    _, code_t, _, _ = tg._relay_level_build(
        "defeat_kobold", 2, dict(_KNOBS, pre_light=True), False
    )
    assert "set_starting_floor(2, pre_light=True)" in code_t
    _, code_f, _, _ = tg._relay_level_build(
        "defeat_kobold", 2, dict(_KNOBS, pre_light=False), False
    )
    assert "set_starting_floor(2, pre_light=False)" in code_f


def test_light_disclosure_respects_pre_light_override():
    """arm B': radius+dark must not LIE 'torch-lit' to the journal/docstring."""
    tg = _tg()
    dark = dict(_KNOBS, down_ladder_radius=16, pre_light=False)
    _, code, _, stage = tg._relay_level_build("defeat_kobold", 2, dark, False)
    assert "no scaffold pre-light" in stage and "torch-lit (9x9 each)" not in stage, stage
    assert "down_ladder_radius=16, pre_light=False" in code
    # None override keeps the coupled default disclosure byte-identical
    coupled = dict(_KNOBS, down_ladder_radius=16)
    _, _, _, stage_c = tg._relay_level_build("defeat_kobold", 2, coupled, False)
    assert "spawn & down ladder torch-lit (9x9 each)" in stage_c, stage_c
    _, _, _, stage_e = tg._relay_level_build("defeat_kobold", 2, dict(_KNOBS), False)
    assert "no scaffold pre-light" in stage_e, stage_e
    # the graded middle value discloses its asymmetry
    _, _, _, stage_l = tg._relay_level_build(
        "defeat_kobold", 2, dict(_KNOBS, pre_light="ladder"), False
    )
    assert "dark start, lit destination" in stage_l, stage_l
    # explicit True at entry discloses lit
    _, _, _, stage_t = tg._relay_level_build(
        "defeat_kobold", 2, dict(_KNOBS, pre_light=True), False
    )
    assert "torch-lit (9x9 each)" in stage_t, stage_t


def test_kit_strip_exam_ignores_the_knob():
    tg = _tg()
    _, code, meta, stage = tg._relay_level_build(
        "defeat_kobold", 2, dict(_KNOBS, pre_light=False, kit_torches=26), True
    )
    assert "set_player_inventory" not in code, "KIT_STRIP must stay an empty-kit exam"
    assert meta["spawn_kit_torches"] is None
    assert "KIT_STRIP" in stage
