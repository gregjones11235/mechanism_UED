"""_render_siege_directive must hand the proposer the focus's style_note as an ATTACK TACTIC.

Regression guard for the wiring bug found 2026-07-06: the modeler distilled a per-wall attack tactic
(style_note) into the siege notebook — e.g. "zero-mob clean drill, strip combat so the craft signal
isn't drowned" — but SIEGE_DIRECTIVE (the text injected into the PROPOSER's level-design prompt) only
ever contained the wall name + prereq links, never the style_note. So the proposer knew WHICH wall to
build toward but not HOW to shape the level, and kept adding combat/survival into a pure gear drill.
These tests pin that the tactic is now forwarded, and that siege-off still yields an empty directive
(baseline path byte-unchanged).
"""

import importlib.util
import os
import sys

# Load gen_manager.py BY ABSOLUTE PATH from THIS repo. Under pytest's import mode the conda env's
# editable `_editable_impl` .pth (registered on sys.meta_path, which outranks sys.path) can otherwise
# resolve `dicode` to a sibling dicode_auction where TaskGenerator differs/absent -> "unknown location"
# at collection. A direct spec-from-file load pins the exact file we changed, immune to that finder.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))          # dicode_v6/
for p in (os.path.join(_REPO, "src"), _REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from auction.siege_notebook import SiegeNotebook  # noqa: E402

_GM_PATH = os.path.join(_REPO, "src", "dicode", "dreaming", "gen_manager.py")
_spec = importlib.util.spec_from_file_location("dicode_v6_gen_manager_under_test", _GM_PATH)
_gm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gm)
TaskGenerator = _gm.TaskGenerator  # _render_siege_directive lives on TaskGenerator


def _bare_manager(nb):
    """A TaskGenerator with ONLY _siege_notebook set — enough to call _render_siege_directive, no LLM/env."""
    gm = object.__new__(TaskGenerator)
    gm._siege_notebook = nb
    return gm


def _mature(extra):
    """A held-out profile that passes the maturity gate (>= MATURITY_MIN_MASTERED skills at MATURITY_SKILL_SR),
    so a focus is actually allowed to open — same helper shape as test_siege_gate_integration."""
    from auction.siege_notebook import MATURITY_MIN_MASTERED, MATURITY_SKILL_SR
    prof = {f"solid_{i}": MATURITY_SKILL_SR + 10 for i in range(MATURITY_MIN_MASTERED)}
    prof.update(extra)
    return prof


def _notebook_with_focus(tmp_path, style_note):
    nb = SiegeNotebook(path=str(tmp_path / "nb.json"))
    prof = _mature({"make_iron_pickaxe": 29.0, "collect_iron": 88.0, "make_stone_pickaxe": 96.0})
    nb.apply_llm_update(
        session_idx=15,
        latest_profile=prof,
        proposed={
            "focus": "make_iron_pickaxe",
            "style_note": style_note,
            "prereq_tree": [
                {"skill": "collect_iron", "role": "mine iron ore"},
                {"skill": "make_stone_pickaxe", "role": "tool to mine iron"},
            ],
        },
        num_snapshots=999,  # force mature so the focus actually opens
    )
    return nb, prof


def test_style_note_forwarded_as_attack_tactic(tmp_path):
    tactic = "zero-mob clean drill: stations adjacent, strip combat until craft SR >70%"
    nb, prof = _notebook_with_focus(tmp_path, tactic)
    assert nb.focus_skills() == ["make_iron_pickaxe"], "focus must have opened for the test to be meaningful"
    text = _bare_manager(nb)._render_siege_directive(prof)
    assert "ATTACK TACTIC" in text
    assert tactic in text                      # the modeler's know-how reaches the proposer verbatim
    assert "make_iron_pickaxe" in text         # still names the wall + chain as before


def test_empty_style_note_adds_no_tactic_line(tmp_path):
    nb, prof = _notebook_with_focus(tmp_path, "")   # no tactic yet -> no ATTACK TACTIC line
    text = _bare_manager(nb)._render_siege_directive(prof)
    assert "ATTACK TACTIC" not in text
    assert "make_iron_pickaxe" in text              # wall + chain still rendered


def test_siege_off_directive_is_empty(tmp_path):
    # No notebook at all == siege off: directive stays empty so the baseline/v5 path is unchanged.
    gm = object.__new__(TaskGenerator)
    gm._siege_notebook = None
    assert gm._render_siege_directive({"make_iron_pickaxe": 29.0}) == ""
