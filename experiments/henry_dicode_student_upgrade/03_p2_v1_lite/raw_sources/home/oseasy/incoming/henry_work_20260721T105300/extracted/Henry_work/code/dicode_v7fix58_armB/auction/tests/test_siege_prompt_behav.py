"""Prompt-assembly tests for the v6 problem-2 behaviour-fingerprint hint (2026-07-05).

Verifies build_siege_modeler_user_prompt actually embeds the behav_hint block (and keeps the cooc
block), and that both degrade to nothing when empty — so a run with no winning-episode support gets a
byte-clean prompt rather than an empty-labelled block.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_AUCTION = os.path.dirname(_HERE)
if _AUCTION not in sys.path:
    sys.path.insert(0, _AUCTION)

from modeler import build_siege_modeler_user_prompt  # noqa: E402


def _prompt(**kw):
    return build_siege_modeler_user_prompt(
        session_idx=5,
        evidence={},
        parent_ids=["task_1"],
        parent_context={},
        notebook_text="focus: make_iron_pickaxe",
        combat_targets=["defeat_skeleton", "defeat_zombie"],
        **kw,
    )


# The section LABELS ("REAL-SUCCESS BEHAVIOUR" / "REAL-TRAJECTORY CO-OCCURRENCE") are added by the
# gen_manager render functions, not by the assembler — the assembler embeds the hint text verbatim.
# So these tests assert on the hint CONTENT going into the prompt (and its position), not the label.
_BEHAV = ("BEHAV_MARKER: won make_iron_pickaxe (40 winning eps, ~3% SR), ~84 steps, "
          "55% movement, used DO 30.0x, PLACE_STONE 2.1x.")
_COOC = "COOC_MARKER: succeeded defeat_skeleton, also reached place_torch 90%."


def test_behav_hint_embedded_when_present():
    p = _prompt(cooc_hint="", behav_hint=_BEHAV)
    assert "BEHAV_MARKER" in p
    assert "84 steps" in p and "PLACE_STONE 2.1x" in p


def test_both_cooc_and_behav_present():
    p = _prompt(cooc_hint=_COOC, behav_hint=_BEHAV)
    assert "COOC_MARKER" in p    # (c) block still there
    assert "BEHAV_MARKER" in p   # problem-2 block added
    # co-occurrence appears before behaviour (fixed order in the assembler)
    assert p.index("COOC_MARKER") < p.index("BEHAV_MARKER")


def test_empty_behav_hint_adds_nothing():
    p_empty = _prompt(cooc_hint="", behav_hint="")
    p_ws = _prompt(cooc_hint="", behav_hint="   \n  ")
    assert "BEHAV_MARKER" not in p_empty
    assert p_empty == p_ws  # whitespace-only hint is treated as empty (no stray block)


def test_default_behav_hint_is_optional():
    # calling without behav_hint at all (default "") must not raise and must omit the block.
    p = _prompt(cooc_hint="")
    assert "BEHAV_MARKER" not in p
