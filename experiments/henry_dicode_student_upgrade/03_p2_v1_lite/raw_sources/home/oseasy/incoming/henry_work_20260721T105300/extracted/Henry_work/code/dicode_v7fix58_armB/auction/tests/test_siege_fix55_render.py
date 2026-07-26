"""v7fix5.5 batch 1 — P0 scaffold-facts disclosure + L1 render diet (R1/R2/R3).

Motivation (design doc fable_research_reports/v7fix55假设探针课程环设计.md):
  P0  The fix54 radius probe showed the entry cliff is a spawn-CONTEXT effect (same distance,
      r40=54.7% vs entry=30.1%), with lighting the prime mechanical suspect — yet the modeler
      was NEVER told the world rule that radius stages torch-light the spawn and the down
      ladder while entry stages pre-light nothing. Its evidence channels carry no light
      telemetry, so that cliff was structurally undiagnosable. fix55 P0 disclosures:
      every scaffold knob of THIS stage + the diff to the NEXT stage, rendered from the
      code's own stage table (never template constants), lighting included.
  L1  The 14k-char render suffered salience dilution, not overflow: RETIRED WALLS repeated
      3 lines of identical teaching prose per wall (5.2k, 37%), verified_chains rendered
      every style note in full (6.2k). R1: teach once in the header (exemption sentence only
      when a rendered wall qualifies — the fix3 tests pin its absence otherwise); latest
      failed note in full, older ones one line. R2: style/note prose only for entries that
      touch the CURRENT attack (target/links vs active foci skill/prereq trees). R3: a
      target that IS an active focus defers its prose to the focus's style-so-far line.
      Compression is render-only — every note stays in the notebook JSON (reversible).

These tests pin:
  F1 SCAFFOLD FACTS: rendered at scaffolded stages, absent at FULL/kit-strip; the clauses
     are computed from the stage table (radius value changes the sentence).
  F2 NEXT-STAGE DIFF: derived from _stage_knobs(stage-1) — the lock diff (6->5), the
     lighting/spawn diff (5->4), and the FULL boundary (1->0).
  F3 R1: latest note full, older one-lined; diagnosis tag; conditional [relay-exempt].
  F4 R2/R3: hot chains keep prose, cold chains one-line, active-focus targets defer;
     the suppressed prose survives in the JSON.
  F5 gen_manager: the RELAY-BUILD stage string carries the lighting fact both ways.
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
    RUNG_LADDER_RADII,
    RUNG_NEEDS_SLOW,
    SiegeNotebook,
)

WALL = "defeat_kobold"        # COMBAT, native floor 3


def _mature_profile(extra=None):
    prof = {f"filler_{i}": MATURITY_SKILL_SR + 10 for i in range(MATURITY_MIN_MASTERED)}
    if extra:
        prof.update(extra)
    return prof


@pytest.fixture
def nb_path(tmp_path):
    return str(tmp_path / "siege_notebook.json")


def _open_relay(nb, wall=WALL, r0=3, session=1, prereq_tree=None):
    prof = _mature_profile({wall: 0.0})
    nb.apply_llm_update(
        session, prof,
        {"foci": [{"skill": wall, "prereq_tree": prereq_tree or [],
                   "relay_r0_floor": r0}]},
        num_snapshots=MATURITY_MIN_SNAPSHOTS, forensics={},
    )
    return prof


def _set_relay(nb, floor=2, stage=6):
    r = nb.foci()[0]["relay"]
    r["spawn_floor"] = floor
    r["sub_stage"] = stage
    nb._save()
    return r


# ---- F1: SCAFFOLD FACTS presence + computed-not-constant ------------------------------------------

def test_scaffold_facts_rendered_at_scaffold_stage(nb_path):
    nb = SiegeNotebook(nb_path)                       # defaults: descent regime ON
    _open_relay(nb)
    _set_relay(nb, floor=2, stage=6)                  # radius leg: radii[2], no lock
    txt = nb.render_for_prompt()
    assert "SCAFFOLD FACTS" in txt
    assert f"within {RUNG_LADDER_RADII[2]} tiles of the down ladder" in txt
    assert "torch-lit (9x9)" in txt                   # the lighting fact, disclosed at last
    assert "never yours to change" in txt             # facts, not levers


def test_scaffold_facts_computed_from_stage_table(nb_path):
    # The disclosure must change when the stage changes — a template constant cannot.
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    _set_relay(nb, floor=2, stage=8)
    txt8 = nb.render_for_prompt()
    _set_relay(nb, floor=2, stage=7)
    txt7 = nb.render_for_prompt()
    assert f"within {RUNG_LADDER_RADII[0]} tiles" in txt8
    assert f"within {RUNG_LADDER_RADII[1]} tiles" in txt7
    assert txt8 != txt7


def test_scaffold_facts_absent_at_full_and_kit_strip(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    r = _set_relay(nb, floor=2, stage=0)              # FULL: the unscaffolded level
    assert "SCAFFOLD FACTS" not in nb.render_for_prompt()
    r["sub_stage"] = 6
    r["kit_strip"] = True
    nb._save()
    assert "SCAFFOLD FACTS" not in nb.render_for_prompt()


# ---- F2: the NEXT-stage diff ----------------------------------------------------------------------

def test_next_stage_diff_lock_and_clock(nb_path):
    # stage 6 -> 5: same radius, the regime arrives (uplock + slow clock). Spawn/light unchanged.
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    _set_relay(nb, floor=2, stage=6)
    txt = nb.render_for_prompt()
    assert "NEXT after graduating = stage 5" in txt
    assert "up-ladder -> REMOVED (no retreat upward)" in txt
    assert f"survival clocks -> {RUNG_NEEDS_SLOW:.1f}x" in txt
    assert "spawn -> " not in txt                     # unchanged knobs are not listed


def test_next_stage_diff_spawn_and_lighting(nb_path):
    # stage 5 -> 4: the radius leg ends — spawn moves to the entry and the pre-light is GONE.
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    _set_relay(nb, floor=2, stage=5)
    txt = nb.render_for_prompt()
    assert "NEXT after graduating = stage 4" in txt
    assert "spawn -> at the floor entry (up-ladder)" in txt
    assert "pre-light -> NONE" in txt


def test_next_stage_full_boundary(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    _set_relay(nb, floor=2, stage=1)
    txt = nb.render_for_prompt()
    assert "NEXT after graduating = FULL (the unscaffolded rung level)" in txt


# ---- F3: RETIRED WALLS diet -----------------------------------------------------------------------

_NOTES = [
    "First archived tactic prose. It rambles across several sentences of detail.",
    "Second archived tactic prose. Also long-winded with follow-up sentences.",
    "Third and LATEST tactic, kept in full for the modeler to differentiate against.",
]


def _retire_reg(last_event="focus_retired_stalled"):
    return {
        "count": 2, "last_session": 9, "sr_at_retirement": 12.0,
        "failed_notes": list(_NOTES), "last_event": last_event,
        "failure_attribution_at_retirement": {
            "class": "resource_shortfall", "verified": True,
        },
    }


def test_retired_latest_note_full_older_one_lined(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    nb._nb["retired"] = {"defeat_gnome_warrior": _retire_reg()}
    nb._save()
    txt = nb.render_for_prompt()
    assert f"failed tactic (latest): {_NOTES[2]}" in txt
    assert "earlier failed tactic 1: First archived tactic prose…" in txt
    assert "earlier failed tactic 2: Second archived tactic prose…" in txt
    assert "rambles across several sentences" not in txt      # older bodies are NOT rendered
    assert "diagnosis: resource_shortfall (verified)" in txt
    # reversible: the full notes still live in the JSON.
    assert SiegeNotebook(nb_path).retired_registry()["defeat_gnome_warrior"][
        "failed_notes"] == _NOTES


def test_retired_exemption_taught_once_and_only_when_earned(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    nb._nb["retired"] = {"defeat_gnome_warrior": _retire_reg()}
    nb._save()
    txt = nb.render_for_prompt()
    assert txt.count("exempt from the cooldown") == 1         # taught ONCE, in the header
    assert "[relay-exempt]" in txt
    # a relay-stalled retirement earns no exemption teaching at all (v7fix3.1 rule, fix3 tests).
    nb._nb["retired"] = {
        "defeat_gnome_warrior": _retire_reg("focus_retired_relay_stalled")
    }
    nb._save()
    txt = nb.render_for_prompt()
    assert "exempt from the cooldown" not in txt
    assert "[relay-exempt]" not in txt


# ---- F4: chains prose only for the current attack --------------------------------------------------

def _chains():
    return [
        {"target": "make_iron_sword", "links": ["collect_iron"], "category": "enabler",
         "status": "verified", "last_recorded_sr": 80, "last_recorded_session": 5,
         "style_note": "smelt two bars before descending"},
        {"target": "defeat_zombie", "links": ["make_wood_sword"],
         "category": "combat_milestone", "status": "verified", "last_recorded_sr": 90,
         "last_recorded_session": 4, "style_note": "kite it in daylight"},
    ]


def test_chain_prose_hot_kept_cold_one_lined(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    foc = nb.foci()[0]
    foc["prereq_tree"] = [{"skill": "make_iron_sword", "state": "wall", "sr": 30.0}]
    nb._nb["verified_chains"] = _chains()
    nb._save()
    txt = nb.render_for_prompt()
    assert "smelt two bars before descending" in txt          # touches the current attack
    assert "defeat_zombie" in txt                             # the record line stays
    assert "kite it in daylight" not in txt                   # ...but its prose does not
    # reversible: the suppressed prose still lives in the JSON.
    assert SiegeNotebook(nb_path).verified_chains()[1][
        "style_note"] == "kite it in daylight"


def test_chain_of_active_focus_defers_to_style_so_far(nb_path):
    nb = SiegeNotebook(nb_path)
    _open_relay(nb)
    foc = nb.foci()[0]
    foc["style_note"] = "my live tactic for the kobold"
    nb._nb["verified_chains"] = [{
        "target": WALL, "links": [], "category": "combat_milestone", "status": "progress",
        "last_recorded_sr": 5, "last_recorded_session": 6,
        "style_note": "an older recorded kobold tactic",
    }]
    nb._save()
    txt = nb.render_for_prompt()
    assert "my live tactic for the kobold" in txt             # ONE tactic text — the newest
    assert "an older recorded kobold tactic" not in txt


# ---- F5: RELAY-BUILD stage string carries the lighting fact (Oscar full suite — jax) ---------------

_HAS_JAX = importlib.util.find_spec("jax") is not None
needs_jax = pytest.mark.skipif(not _HAS_JAX, reason="gen_manager imports jax (Oscar full suite)")


def _gen_manager_module():
    gm_path = os.path.join(_REPO, "src", "dicode", "dreaming", "gen_manager.py")
    spec = importlib.util.spec_from_file_location("dicode_v7fix55_gm_test", gm_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeChainLog:
    def latest_fail_summary(self, target):
        return {"n_succ": 4, "inv": {"sword": {"succ_med": 3}, "torches": {"succ_med": 9}}}


@needs_jax
def test_relay_build_stage_string_carries_lighting(tmp_path):
    import types

    gmm = _gen_manager_module()
    tg = object.__new__(gmm.TaskGenerator)
    nb = SiegeNotebook(str(tmp_path / "nb.json"))
    _open_relay(nb, wall="defeat_lizard")
    r = nb.foci()[0]["relay"]
    r["spawn_floor"] = 2
    r["sub_stage"] = 6                                # radius stage -> lit
    nb._save()
    tg._siege_notebook = nb
    tg._chain_log = _FakeChainLog()
    tg.config = types.SimpleNamespace(
        siege_relay_worldgen="base", siege_relay_levels_per_session=1,
    )
    out = tg._system_relay_levels(session_idx=7)
    assert "spawn & down ladder torch-lit (9x9 each)" in out[0]["description"]
    r["sub_stage"] = 4                                # entry stage -> no pre-light
    nb._save()
    out = tg._system_relay_levels(session_idx=9)
    assert "no scaffold pre-light (the floor's own light only)" in out[0]["description"]
