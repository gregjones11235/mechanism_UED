"""v7fix4 — habitat fidelity (fable_research_reports/v7fix4真实世界接力与栖息地保真方案.md).

Post-mortem being fixed (job 3840016, killed @s61): lizard's relay ran its FULL lifecycle
(attach s25 -> graduated s55 -> SEWN s59) yet held-out stayed EXACTLY 0 (gap gate: trained=97%
held-out=0%). The FM-authored rung levels moved the lizard SHALLOWER along with the annealing
spawn (trained SR ROSE 76->94->97 as the spawn moved UP — a real ladder must lengthen the chain
and dip); the modeler (sole author of relay_r0_floor / prereq_tree) had no floor table and
anchored R0 at floor 2 while lizards inhabit floor 3, with no enter_sewers in the chain.

These tests pin: the deep lock (floor-3+ walls relay-only, on the LLM path, the capacity
fall-through AND the auto-open menu), R0 habitat anchoring (open + attach + the enter_* rule),
the entrance-chain autofill, the KIT_STRIP exam (regress semantics; the full ladder walkthrough
lives in test_siege_v7_relay), the post-SEWN sandbox_mismatch sentinel, and the system-built
relay level template (text-level here; the gen_manager wiring is @needs_jax, Oscar full suite).
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

from auction.craftax_achievements import (  # noqa: E402
    FLOOR_ENTRANCES,
    WALL_NATIVE_FLOOR,
    native_floor_of,
)
from auction.siege_notebook import (  # noqa: E402
    MATURITY_MIN_MASTERED,
    MATURITY_MIN_SNAPSHOTS,
    MATURITY_SKILL_SR,
    RUNG_FLOOR_SR,
    RUNG_GRADUATE_SR,
    RUNG_STALL_READINGS,
    SANDBOX_MISMATCH_READINGS,
    SiegeNotebook,
    SiegeThresholds,
)

DEEP = "defeat_lizard"      # habitat floor 3 (Sewers) — THE v7fix3 wall
DEEP5 = "defeat_troll"      # habitat floor 5
SHALLOW = "defeat_gnome_warrior"  # habitat floor 2 — fix8's winning class, NOT deep-locked


def _mature_profile(extra=None):
    prof = {f"filler_{i}": MATURITY_SKILL_SR + 10 for i in range(MATURITY_MIN_MASTERED)}
    if extra:
        prof.update(extra)
    return prof


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


@pytest.fixture
def nb_path(tmp_path):
    return str(tmp_path / "siege_notebook.json")


# ---- P1: the deep lock (habitat floor 3+ walls are relay-only) -------------------------------


def test_deep_wall_ordinary_proposal_is_deep_locked(nb_path):
    """v7fix4.2 UPDATE: with the autoconvert default ON, an ordinary deep proposal now CONVERTS
    into a relay campaign (the fix4-era unconditional refusal starved relay start for a whole
    run — s77, zero deep proposals ever). The original refusal contract survives, pinned
    behind the flag."""
    nb = SiegeNotebook(nb_path)
    nb.th.deep_wall_autoconvert = False  # the fix4 refusal semantics, preserved
    prof = _mature_profile({DEEP: 0.0})
    _update(nb, 1, prof, foci=[{"skill": DEEP, "prereq_tree": []}],
            forensics={DEEP: {"missing_top": []}})
    assert nb.focus_skills() == []
    assert f"deep_locked({DEEP}" in nb.last_focus_decision
    assert "floor 3" in nb.last_focus_decision          # names the habitat
    assert "relay_r0_floor" in nb.last_focus_decision   # teaches the re-proposal format
    # mirrored into the journal like tier_locked (the ⑦ lesson).
    assert DEEP in nb.render_for_prompt()
    # and the fix4.2 default on a fresh notebook: the same proposal converts instead.
    nb2 = SiegeNotebook(nb_path + ".2")
    _update(nb2, 1, prof, foci=[{"skill": DEEP, "prereq_tree": []}],
            forensics={DEEP: {"missing_top": []}})
    assert f"relay_converted({DEEP}" in nb2.last_focus_decision
    assert nb2.relay_walls() == [DEEP]


def test_deep_lock_fires_before_chain_incomplete(nb_path):
    """A deep wall that is ALSO ⑦-latched is handled by the DEEP branch, never ⑦: under fix4.2
    defaults it converts to a relay (⑦-exempt, the fix1 rule); with autoconvert off it gets the
    deep teaching. Either way ⑦'s expand-the-chain message stays for shallow walls only."""
    nb = SiegeNotebook(nb_path)
    prof = _mature_profile({DEEP: 0.0})
    _update(nb, 1, prof, foci=[{"skill": DEEP, "prereq_tree": []}],
            forensics={}, incomplete={DEEP})
    assert f"relay_converted({DEEP}" in nb.last_focus_decision
    assert "chain_incomplete" not in nb.last_focus_decision
    nb2 = SiegeNotebook(nb_path + ".2")
    nb2.th.deep_wall_autoconvert = False
    _update(nb2, 1, prof, foci=[{"skill": DEEP, "prereq_tree": []}],
            forensics={}, incomplete={DEEP})
    assert f"deep_locked({DEEP}" in nb2.last_focus_decision
    assert "chain_incomplete" not in nb2.last_focus_decision


def test_deep_wall_relay_proposal_opens_at_habitat_floor(nb_path):
    nb = SiegeNotebook(nb_path)
    prof = _mature_profile({DEEP: 0.0})
    _update(nb, 1, prof, foci=[{"skill": DEEP, "prereq_tree": [], "relay_r0_floor": 3}],
            forensics={})
    assert nb.relay_walls() == [DEEP]
    assert nb.required_spawn_floor(DEEP) == 3
    assert f"opened_relay({DEEP} @ R0 spawn_floor=3)" in nb.last_focus_decision
    assert "r0_corrected" not in nb.last_focus_decision  # 3 was already right


def test_deep_wall_off_the_auto_open_menu(nb_path):
    """The v7fix3 hole closed: the auto-open path had no deep gate — kobold/lizard could still
    auto-open as an ordinary siege behind the LLM path's lock."""
    nb = SiegeNotebook(nb_path)
    prof = _mature_profile({DEEP: 0.0, SHALLOW: 0.0})
    _update(nb, 1, prof, foci=[],
            ranked=[{"skill": DEEP, "why": "deep"}, {"skill": SHALLOW, "why": "stuck"}],
            forensics={DEEP: {"missing_top": []}, SHALLOW: {"missing_top": []}})
    assert nb.focus_skills() == [SHALLOW]  # lizard skipped, the gnome opens instead


def test_floor2_wall_not_deep_locked(nb_path):
    """fix8's winning gnome path must stay untouched (the deliberate floor-3 boundary)."""
    nb = SiegeNotebook(nb_path)
    prof = _mature_profile({SHALLOW: 0.0})
    _update(nb, 1, prof, foci=[{"skill": SHALLOW, "prereq_tree": []}],
            forensics={SHALLOW: {"missing_top": []}})
    assert nb.focus_skills() == [SHALLOW]


def test_anchor_off_restores_fix3_behaviour(nb_path):
    nb = SiegeNotebook(nb_path)
    nb.th = SiegeThresholds(wall_floor_anchor=False)
    prof = _mature_profile({DEEP: 0.0})
    _update(nb, 1, prof, foci=[{"skill": DEEP, "prereq_tree": []}],
            forensics={DEEP: {"missing_top": []}})
    assert nb.focus_skills() == [DEEP]  # fix3 equivalence when the knob is off


# ---- P1: R0 habitat anchoring -----------------------------------------------------------------


def test_r0_corrected_on_open(nb_path):
    """THE v7fix3 case: a relay for lizard proposed at floor 2 must anchor to floor 3."""
    nb = SiegeNotebook(nb_path)
    prof = _mature_profile({DEEP: 0.0})
    _update(nb, 1, prof, foci=[{"skill": DEEP, "prereq_tree": [], "relay_r0_floor": 2}],
            forensics={})
    assert f"r0_corrected({DEEP}" in nb.last_focus_decision
    assert "habitat is floor 3" in nb.last_focus_decision
    assert nb.required_spawn_floor(DEEP) == 3
    assert nb.foci()[0]["relay"]["r0_floor"] == 3


def test_r0_corrected_on_attach(nb_path):
    """An active zero-win gnome upgraded with a wrong floor anchors to its habitat (2)."""
    nb = SiegeNotebook(nb_path)
    prof = _mature_profile({SHALLOW: 0.0})
    _update(nb, 1, prof, foci=[{"skill": SHALLOW, "prereq_tree": []}],
            forensics={SHALLOW: {"missing_top": []}})
    _update(nb, 3, prof, foci=[{"skill": SHALLOW, "prereq_tree": [], "relay_r0_floor": 5}],
            forensics={SHALLOW: {"missing_top": []}})
    assert f"relay_attached({SHALLOW} @ R0 spawn_floor=2" in nb.last_focus_decision
    assert "r0_corrected" in nb.last_focus_decision
    assert nb.required_spawn_floor(SHALLOW) == 2


def test_enter_wall_r0_is_one_floor_above(nb_path):
    """An entrance wall spawns ONE floor above its own floor — descending INTO it IS the skill
    (spawning on it would pre-fire the enter_* achievement at reset)."""
    nb = SiegeNotebook(nb_path)
    prof = _mature_profile({"enter_sewers": 0.0})
    _update(nb, 1, prof,
            foci=[{"skill": "enter_sewers", "prereq_tree": [], "relay_r0_floor": 3}],
            forensics={})
    assert nb.required_spawn_floor("enter_sewers") == 2
    assert "r0_corrected(enter_sewers" in nb.last_focus_decision


def test_anchor_clamps_out_of_range_floor_for_unmapped_walls(nb_path):
    nb = SiegeNotebook(nb_path)
    r0, note = nb._anchored_r0("some_unmapped_wall", 99)
    assert r0 == 8 and note is None  # clamped to MAX_DUNGEON_FLOOR, no habitat to anchor to


# ---- P1: entrance-chain autofill ----------------------------------------------------------------


def test_autofill_inserts_missing_entrances_in_floor_order(nb_path):
    nb = SiegeNotebook(nb_path)
    prof = _mature_profile({DEEP: 0.0, "enter_dungeon": 85.0})
    _update(nb, 1, prof,
            foci=[{"skill": DEEP, "prereq_tree": [{"skill": "find_bow", "role": "gear"}],
                   "relay_r0_floor": 3}],
            forensics={})
    tree = [l["skill"] for l in nb.foci()[0]["prereq_tree"]]
    # every entrance up to floor 3 present, entrances first, floor order kept, original link kept
    assert tree[:3] == ["enter_dungeon", "enter_gnomish_mines", "enter_sewers"]
    assert "find_bow" in tree
    roles = {l["skill"]: l.get("role", "") for l in nb.foci()[0]["prereq_tree"]}
    assert "autofilled" in roles["enter_sewers"]
    assert nb.last_chain_autofill and "enter_sewers" in nb.last_chain_autofill
    # a second apply does NOT duplicate the entrances.
    _update(nb, 3, prof, foci=[{"skill": DEEP, "prereq_tree": [], "relay_r0_floor": 3}],
            forensics={})
    tree2 = [l["skill"] for l in nb.foci()[0]["prereq_tree"]]
    assert tree2.count("enter_sewers") == 1


def test_autofill_respects_supplied_entrances_and_enter_walls(nb_path):
    nb = SiegeNotebook(nb_path)
    prof = _mature_profile({"enter_sewers": 0.0})
    _update(nb, 1, prof,
            foci=[{"skill": "enter_sewers",
                   "prereq_tree": [{"skill": "enter_dungeon", "role": "descend"}],
                   "relay_r0_floor": 3}],
            forensics={})
    tree = [l["skill"] for l in nb.foci()[0]["prereq_tree"]]
    # an entrance wall needs the doors BELOW it (1..2), never itself; supplied link not duplicated
    assert "enter_sewers" not in tree
    assert tree.count("enter_dungeon") == 1 and "enter_gnomish_mines" in tree


def test_autofill_skips_shallow_walls(nb_path):
    nb = SiegeNotebook(nb_path)
    prof = _mature_profile({"make_iron_armour": 5.0})
    _update(nb, 1, prof,
            foci=[{"skill": "make_iron_armour",
                   "prereq_tree": [{"skill": "collect_iron", "role": "gear"}]}],
            forensics={"make_iron_armour": {"missing_top": []}})
    tree = [l["skill"] for l in nb.foci()[0]["prereq_tree"]]
    assert tree == ["collect_iron"]  # native floor 0: untouched
    assert nb.last_chain_autofill is None


# ---- P3: KIT_STRIP regress semantics (the full ladder lives in test_siege_v7_relay) -------------


def test_kit_strip_regress_restores_kit_at_floor_one(nb_path):
    nb = SiegeNotebook(nb_path)
    prof = _mature_profile({DEEP: 0.0})
    _update(nb, 1, prof, foci=[{"skill": DEEP, "prereq_tree": [], "relay_r0_floor": 3}],
            forensics={})
    hi = RUNG_GRADUATE_SR + 5
    sidx = 3
    for _ in range(60):                 # graduate 3 -> 2 -> 1 -> KIT_STRIP (P2': ~4 HI per rung)
        nb.note_rung_reading(DEEP, hi, session_idx=sidx)
        sidx += 2
        if nb.relay_kit_stripped(DEEP):
            break
    assert nb.relay_kit_stripped(DEEP) is True
    lo = RUNG_FLOOR_SR - 5
    msg = None
    for sidx in range(sidx, sidx + 2 * (2 + RUNG_STALL_READINGS), 2):   # P2': window fills first
        msg = nb.note_rung_reading(DEEP, lo, session_idx=sidx)
        if "RUNG_REGRESSED" in (msg or ""):
            break
    assert msg and "RUNG_REGRESSED" in msg and "WITH kit" in msg
    foc = nb.foci()[0]
    assert foc["relay"]["spawn_floor"] == 1
    assert not foc["relay"].get("kit_strip")
    assert nb.relay_kit_stripped(DEEP) is False


# ---- P4: post-SEWN sandbox_mismatch sentinel ----------------------------------------------------


def _sew(nb, wall, prof):
    _update(nb, 1, prof, foci=[{"skill": wall, "prereq_tree": [], "relay_r0_floor": 3}],
            forensics={})
    hi = RUNG_GRADUATE_SR + 5
    sidx = 3
    while nb.relay_walls():             # graduate every rung + the kit-strip exam
        nb.note_rung_reading(wall, hi, session_idx=sidx)
        sidx += 2
        assert sidx < 90, "ladder failed to sew"   # P2': ~4 window-judged HI per rung
    assert nb.foci()[0]["relay_sewn"] is True


def test_sandbox_mismatch_sentinel_retires_after_streak(nb_path):
    nb = SiegeNotebook(nb_path)
    prof = _mature_profile({DEEP: 0.0})
    _sew(nb, DEEP, prof)
    for i in range(SANDBOX_MISMATCH_READINGS - 1):
        msg = nb.note_transfer_gap(DEEP, trained_pct=95.0, held_pct=0.0, session_idx=40 + i)
        assert "sandbox_mismatch_watch" in msg
    msg = nb.note_transfer_gap(DEEP, trained_pct=95.0, held_pct=0.0, session_idx=50)
    assert "SANDBOX_MISMATCH" in msg
    assert nb.focus_skills() == []                        # retired, slot freed
    reg = nb._nb["retired"][DEEP]
    assert reg["last_event"] == "focus_retired_sandbox_mismatch"
    # a relay re-proposal during cooldown is NOT waived (the same broken ladder must not re-run).
    _update(nb, 51, _mature_profile({DEEP: 0.0}),
            foci=[{"skill": DEEP, "prereq_tree": [], "relay_r0_floor": 3,
                   "style_note": "a genuinely different tactic"}])
    assert "cooldown_rejected(" in nb.last_focus_decision


def test_sandbox_mismatch_counter_resets_on_real_held_out(nb_path):
    nb = SiegeNotebook(nb_path)
    prof = _mature_profile({DEEP: 0.0})
    _sew(nb, DEEP, prof)
    nb.note_transfer_gap(DEEP, trained_pct=95.0, held_pct=0.0, session_idx=40)
    assert nb.foci()[0]["sandbox_mismatch"] == 1
    # held-out moved: NOT a mismatch — the counter resets (normal gap machinery takes over).
    nb.note_transfer_gap(DEEP, trained_pct=95.0, held_pct=8.0, session_idx=42)
    assert nb.foci()[0].get("sandbox_mismatch") == 0
    assert nb.focus_skills() == [DEEP]


# ---- P2: the system-built relay level template (text-level; wiring is @needs_jax) ---------------


def _template_text():
    gm_path = os.path.join(_REPO, "src", "dicode", "dreaming", "gen_manager.py")
    lines = open(gm_path, encoding="utf-8").read().splitlines(keepends=True)
    start = next(i for i, l in enumerate(lines) if "_RELAY_LEVEL_CODE = " in l)
    end = next(i for i in range(start + 1, len(lines)) if lines[i].rstrip().endswith("'''"))
    return "".join(lines[start + 1:end])


def test_relay_level_template_renders_valid_code_both_stages():
    tpl = _template_text()
    kitted = tpl.format(
        docstring="Objective: rung", wall_enum="DEFEAT_LIZARD", floor=3,
        radius_arg="", credit_line="",
        kit_line="        builder.set_player_inventory({'sword': 3, 'torches': 9})\n",
        task_params_args="", build_tail="        return builder.build(rng)",
        constants_import="",
    )
    tree = ast.parse(kitted)
    assert "set_starting_floor(3)" in kitted and "set_player_inventory" in kitted
    assert "WorldBuilder" in kitted                       # the REAL world generator
    assert "Achievement.DEFEAT_LIZARD" in kitted
    kitless = tpl.format(
        docstring="Objective: kit-strip exam", wall_enum="DEFEAT_LIZARD", floor=0,
        radius_arg="", credit_line="", kit_line="",
        task_params_args="", build_tail="        return builder.build(rng)",
        constants_import="",
    )
    ast.parse(kitless)
    assert "set_player_inventory" not in kitless          # the exam carries NOTHING
    assert "set_starting_floor(0)" in kitless
    # generate_world consumes the RESET rng -> a fresh real world every episode (distribution,
    # not an instance — the held-out equivalence claim rests on this line).
    assert "def generate_world(self, rng: jax.Array)" in kitless
    # v7fix4.6: a scaffold sub-stage rides the same template — radius on set_starting_floor,
    # clear-gate pre-credit as its own line — and still parses.
    scaffold = tpl.format(
        docstring="Objective: scaffold rung", wall_enum="DEFEAT_KOBOLD", floor=2,
        radius_arg=", down_ladder_radius=8",
        credit_line="        builder.set_monsters_killed(2, 8)\n",
        kit_line="        builder.set_player_inventory({'sword': 3})\n",
        task_params_args="", build_tail="        return builder.build(rng)",
        constants_import="",
    )
    ast.parse(scaffold)
    assert "set_starting_floor(2, down_ladder_radius=8)" in scaffold
    assert "set_monsters_killed(2, 8)" in scaffold
    del tree


# ---- habitat map sanity used by the gates -------------------------------------------------------


def test_the_fix_targets_are_deep_locked_and_gnome_is_not():
    assert native_floor_of(DEEP) == 3 and native_floor_of("defeat_kobold") == 3
    assert native_floor_of(DEEP5) == 5 and native_floor_of("enter_sewers") == 3
    assert native_floor_of(SHALLOW) == 2          # below the lock line
    assert FLOOR_ENTRANCES[3] == "enter_sewers"   # the broken link of the v7fix3 run, pinned
    assert set(WALL_NATIVE_FLOOR).issuperset(
        {"defeat_lizard", "defeat_kobold", "enter_sewers"}
    )


# ---- modeler P0: the prompt carries the same knowledge the gates enforce ------------------------


def test_modeler_prompt_has_the_habitat_contract():
    from auction.modeler import MODELER_SIEGE_SYSTEM_PROMPT as P

    assert "HABITAT MAP" in P
    assert "enter_sewers" in P and "defeat_lizard" in P
    assert "chain_autofilled" in P and "r0_corrected" in P and "deep_locked" in P
    assert "KIT_STRIP exam" in P and "SYSTEM-BUILT" in P


# ---- gen_manager wiring (Oscar full suite only — gen_manager imports jax) -----------------------

_HAS_JAX = importlib.util.find_spec("jax") is not None
needs_jax = pytest.mark.skipif(not _HAS_JAX, reason="gen_manager imports jax (Oscar full suite)")


def _gen_manager_module():
    gm_path = os.path.join(_REPO, "src", "dicode", "dreaming", "gen_manager.py")
    spec = importlib.util.spec_from_file_location("dicode_v7fix4_gm_test", gm_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeChainLog:
    def latest_fail_summary(self, target):
        if target == "defeat_lizard":
            return {"n_succ": 4,
                    "inv": {"sword": {"succ_med": 3}, "torches": {"succ_med": 9},
                            "potions_2": {"succ_med": 2}, "potions_5": {"succ_med": 1}}}
        return None


@needs_jax
def test_system_relay_levels_build_and_respect_the_knobs(tmp_path):
    import types

    gmm = _gen_manager_module()
    tg = object.__new__(gmm.TaskGenerator)
    nb = SiegeNotebook(str(tmp_path / "nb.json"))
    prof = _mature_profile({"defeat_lizard": 0.0})
    _update(nb, 1, prof,
            foci=[{"skill": "defeat_lizard", "prereq_tree": [], "relay_r0_floor": 3}],
            forensics={})
    tg._siege_notebook = nb
    tg._chain_log = _FakeChainLog()
    tg.config = types.SimpleNamespace(
        siege_relay_worldgen="base", siege_relay_levels_per_session=2,
    )
    out = tg._system_relay_levels(session_idx=5)
    assert len(out) == 2
    p = out[0]
    assert p["level_meta"]["system_built"] is True
    assert p["level_meta"]["siege_wall"] == "defeat_lizard"
    assert p["level_meta"]["spawn_floor"] == 3
    assert "set_starting_floor(3)" in p["_system_code"]
    assert "set_player_inventory" in p["_system_code"]    # winner-median kit rode in
    assert "'sword': 3" in p["_system_code"] and "'torches': 9" in p["_system_code"]
    assert "'potions': 3" in p["_system_code"]            # colours summed onto the legal field
    ast.parse(p["_system_code"])
    assert "DEFEAT_LIZARD" in p["description"]
    # the fm ablation arm builds nothing.
    tg.config = types.SimpleNamespace(siege_relay_worldgen="fm")
    assert tg._system_relay_levels(session_idx=5) == []


@needs_jax
def test_system_relay_levels_kit_strip_stage_is_kitless(tmp_path):
    import types

    gmm = _gen_manager_module()
    tg = object.__new__(gmm.TaskGenerator)
    nb = SiegeNotebook(str(tmp_path / "nb.json"))
    prof = _mature_profile({"defeat_lizard": 0.0})
    _update(nb, 1, prof,
            foci=[{"skill": "defeat_lizard", "prereq_tree": [], "relay_r0_floor": 3}],
            forensics={})
    hi = RUNG_GRADUATE_SR + 5
    sidx = 3
    for _ in range(60):                 # graduate to KIT_STRIP (P2': ~4 HI per rung)
        nb.note_rung_reading("defeat_lizard", hi, session_idx=sidx)
        sidx += 2
        if nb.relay_kit_stripped("defeat_lizard"):
            break
    assert nb.relay_kit_stripped("defeat_lizard") is True
    tg._siege_notebook = nb
    tg._chain_log = _FakeChainLog()
    tg.config = types.SimpleNamespace(
        siege_relay_worldgen="base", siege_relay_levels_per_session=1,
    )
    out = tg._system_relay_levels(session_idx=15)
    assert len(out) == 1
    assert out[0]["level_meta"]["spawn_floor"] == 0
    assert "set_player_inventory" not in out[0]["_system_code"]
    assert "KIT_STRIP" in out[0]["description"]


@needs_jax
def test_rung_reading_filter_rejects_fm_levels(tmp_path):
    """The telemetry quarantine: only SYSTEM-BUILT levels feed a relay wall's rung readings."""
    import threading
    import types

    import networkx as nx

    gmm = _gen_manager_module()
    tg = object.__new__(gmm.TaskGenerator)
    nb = SiegeNotebook(str(tmp_path / "nb.json"))
    prof = _mature_profile({"defeat_lizard": 0.0})
    _update(nb, 1, prof,
            foci=[{"skill": "defeat_lizard", "prereq_tree": [], "relay_r0_floor": 3}],
            forensics={})
    tg._siege_notebook = nb
    tg.config = types.SimpleNamespace(
        siege_relay_worldgen="base", siege_gap_trained_recency=2,
    )

    class _Ar:
        def __init__(self):
            self.graph = nx.DiGraph()
            self._lock = threading.Lock()

    ar = _Ar()
    # an FM-authored level at the right floor scoring 0.99 — must NOT feed the ladder
    ar.graph.add_node("fm1", siege_wall="defeat_lizard", spawn_floor=3,
                      performance_history=[{"sr": 0.99, "session": 9}])
    # a system-built level at the right floor scoring 0.31 — the ONLY legitimate evidence
    ar.graph.add_node("sys1", siege_wall="defeat_lizard", spawn_floor=3, system_built=True,
                      performance_history=[{"sr": 0.31, "session": 9}])
    tg.archive = ar
    # v7fix5.6: the ladder reads the zero-shot eval, never archive levels (FM or otherwise);
    # the quarantine below still guards the trained-SR telemetry channel.
    nb.note_rung_eval("defeat_lizard", {"session": 9, "sr": 31.0, "spawn_floor": 3,
                                        "sub_stage": 0, "n_envs": 512})
    # v7fix5.7-P2' T1: the decision-cadence render is telemetry-only now — consumption
    # happens at run_dicode Step 4d via consume_rung_eval.
    tg._render_siege_gap_hint(_mature_profile({"defeat_lizard": 0.0}), session_idx=9)
    r = nb.foci()[0]["relay"]
    assert r["rung_trained"] == []      # render never judges (P2' pin)
    nb.consume_rung_eval("defeat_lizard", 9)
    r = nb.foci()[0]["relay"]
    assert r["rung_trained"] == [31.0]  # the FM 99% reading never reached the ladder


# ---- post-audit hardening: R6_SYSTEM_RELAY (system-built relay walls take no FM levels) ----------

from auction.level_validator import (  # noqa: E402
    RULE_SPAWN,
    RULE_SYS_RELAY,
    reroll_worthy,
    validate_level,
)

_V_FOCI = [{"skill": DEEP, "prereq_tree": [{"skill": "enter_sewers", "role": "descend"}]}]
_V_DESC = "Relevant Achievements: DEFEAT_LIZARD, ENTER_SEWERS\nCompleted Achievements: NONE"


def _v_meta(spawn_floor, wall=DEEP, drill=None):
    return {"type": "DEPTH", "drill_target": drill, "siege_wall": wall,
            "spawn_floor": spawn_floor, "spawn_kit": None}


def test_system_relay_wall_rejects_fm_level_even_at_correct_floor():
    """The directive promises 'any you propose will be rejected' — and it prints the rung floor,
    so a disobedient proposer could otherwise author a floor-correct level that eats one of the
    wall's 2 discounted force-activation slots while staying quarantined from rung evidence."""
    viols = validate_level(
        _V_DESC, _v_meta(3), _V_FOCI,
        required_spawn_floors={DEEP: 3}, system_relay_walls={DEEP},
    )
    assert any(v.rule == RULE_SYS_RELAY for v in viols)
    assert reroll_worthy(viols)
    # one actionable message, not a stack: no floor/scaffold violation muddying the feedback
    assert not any(v.rule == RULE_SPAWN for v in viols)
    msg = next(v.message for v in viols if v.rule == RULE_SYS_RELAY)
    assert "SYSTEM-BUILT" in msg and "tag" in msg


def test_system_relay_wall_rejects_fm_drill_too():
    viols = validate_level(
        _V_DESC, _v_meta(3, wall=None, drill=DEEP), _V_FOCI,
        required_spawn_floors={DEEP: 3}, system_relay_walls={DEEP},
    )
    assert any(v.rule == RULE_SYS_RELAY for v in viols)


def test_system_relay_walls_empty_restores_plain_r6():
    """The 'fm' ablation arm / siege-off path: byte-identical fix3 semantics."""
    viols = validate_level(
        _V_DESC, _v_meta(3), _V_FOCI, required_spawn_floors={DEEP: 3},
    )
    assert not viols
    viols = validate_level(
        _V_DESC, _v_meta(2), _V_FOCI, required_spawn_floors={DEEP: 3},
        system_relay_walls=set(),
    )
    assert any(v.rule == RULE_SPAWN for v in viols)
    assert not any(v.rule == RULE_SYS_RELAY for v in viols)


def test_untagged_levels_unaffected_by_system_relay_walls():
    viols = validate_level(
        _V_DESC, _v_meta(0, wall=None), _V_FOCI,
        required_spawn_floors={DEEP: 3}, system_relay_walls={DEEP},
    )
    assert not any(v.rule in (RULE_SYS_RELAY, RULE_SPAWN) for v in viols)


def test_modeler_prompt_carries_no_run_anecdotes():
    """User review 2026-07-11: run-specific war stories ('cost a prior run 40 sessions on a
    lizard focus') are knowledge leakage bait — the prompt may state aggregate world facts and
    mechanism rationale, never a named past run's trajectory."""
    import re

    from auction.modeler import MODELER_SIEGE_SYSTEM_PROMPT as P

    assert "prior run" not in P
    assert "40 sessions" not in P
    assert not re.search(r"wasted \d+ (siege )?decisions once", P)
    assert not re.search(r"cost a (whole )?(prior )?run", P)


@needs_jax
def test_directive_relay_wall_wording_shifts_under_system_worldgen(tmp_path):
    """Post-audit P2 wording: for a system-built relay wall the chain header / tactic line must
    not read as 'build levels toward this wall' (the FM's lane is the LINK levels); the fm
    ablation arm keeps the fix3 wording byte-for-byte."""
    import types

    gmm = _gen_manager_module()
    tg = object.__new__(gmm.TaskGenerator)
    nb = SiegeNotebook(str(tmp_path / "nb.json"))
    prof = _mature_profile({"defeat_lizard": 0.0, "enter_dungeon": 80.0})
    _update(nb, 1, prof,
            foci=[{"skill": "defeat_lizard", "prereq_tree": [], "relay_r0_floor": 3,
                   "style_note": "gear up on the way down, kite it in open water"}],
            forensics={})
    assert nb.foci() and nb.foci()[0].get("relay"), "relay must be attached for this test"
    tg._siege_notebook = nb
    tg._chain_log = None

    tg.config = types.SimpleNamespace(siege_relay_worldgen="base")
    text = tg._render_siege_directive(prof)
    assert "still-unmastered LINKS" in text
    assert "do NOT tag levels for the wall itself" in text
    assert "train the whole chain up to the wall" not in text
    if "ATTACK TACTIC" in text:
        assert "LINK levels" in text and "shape the level to enact it" not in text

    tg.config = types.SimpleNamespace(siege_relay_worldgen="fm")
    text_fm = tg._render_siege_directive(prof)
    assert "train the whole chain up to the wall" in text_fm
    assert "still-unmastered LINKS" not in text_fm
