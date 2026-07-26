"""v7fix5.8 arm-A designcheck — torch-supply INSERT axis (torch57 verdict, 2026-07-19).

Design points pinned here (source-text checks, same style as the fix5.x stack):
  T57.1 the kit_torches knob exists in _relay_level_build, overrides ONLY the kit's torch
        count, and is scoped OUT of KIT_STRIP exams (held-out semantics stay verbatim).
  T57.2 the pre_light render is value-faithful: "ladder" renders as 'ladder' (the fix5.7
        bool() coercion rebuilt the identical lit world), bools keep their exact rendering.
  T57.3 the supply knob is disclosed (stage string "torch supply N" + meta spawn_kit_torches)
        — no silent difficulty edits (fix9 attribution law).
  T57.4 the graduation light-anneal branch keys on `is True`, so a pre_light=False insert
        (arm A) can NEVER be routed into the dead 49 light-leg; it pops normally.
  T57.5 world_builder is UNTOUCHED by fix5.8 (supply rides the kit line, not the builder) —
        the fix5.7 three-notch machinery stays byte-identical for radius-anchor inserts.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _read(*parts):
    with open(os.path.join(_HERE, *parts), encoding="utf-8") as f:
        return f.read()


GM = _read("src", "dicode", "dreaming", "gen_manager.py")
SNB = _read("auction", "siege_notebook.py")
WB = _read("src", "minicraftax", "world_builder.py")
TESTS = _read("auction", "tests", "test_siege_fix58_torch_supply.py")

CHECKS = [
    ("T57.1a kit_torches knob parsed from the scaffold dict, stripped-exam scoped out",
     '(scaffold or {}).get("kit_torches") if not stripped else None' in GM),
    ("T57.1b override touches ONLY the torch count of a COPY of the winner-median kit",
     'kit = dict(kit)' in GM and 'kit["torches"] = int(_kt58)' in GM),
    ("T57.2a 'ladder' renders faithfully (the bool() coercion bug is dead)",
     "\"'ladder'\" if _pl55 == \"ladder\" else repr(bool(_pl55))" in GM),
    ("T57.2b bool knobs keep the exact pre-5.8 rendering (repr(True)/'True' parity)",
     "radius_arg += f\", pre_light={_pl_repr}\"" in GM
     and "pre_light={bool(_pl55)}" not in GM),
    ("T57.3a stage disclosure carries the supply fact",
     'f", torch supply {int(_kt58)}" if _kt58 is not None else ""' in GM),
    ("T57.3b meta carries the forensic label (None = winner-median, attr skipped)",
     '"spawn_kit_torches": (int(_kt58) if _kt58 is not None else None)' in GM),
    ("T57.4 the light-anneal graduation branch requires pre_light is True — a dark "
     "supply insert pops normally, never into the dead 49 light-leg",
     '_k57.get("pre_light") is True' in SNB),
    ("T57.5 world_builder untouched by fix5.8 (supply rides the kit line)",
     "kit_torches" not in WB),
    ("T57.6 the fix5.8 test suite pins override/byte-parity/ladder-render/kit-strip",
     all(t in TESTS for t in (
         "test_kit_torches_overrides_only_torches",
         "test_absent_knob_is_byte_identical_to_pre_fix58",
         "test_pre_light_ladder_renders_faithfully",
         "test_kit_strip_exam_ignores_the_knob",
     ))),
    ("T57.7a the _light55 disclosure follows the ACTUAL build: an explicit pre_light "
     "override wins over the radius-coupling default (a radius+dark rung must not lie "
     "'torch-lit'; an entry+True insert must not lie 'no pre-light')",
     '_pl55d = scaffold.get("pre_light") if scaffold else None' in GM
     and 'if _pl55d is None:' in GM
     and '_light55 = _LIT_BOTH_55 if _pl55d else _LIT_NONE_55' in GM),
    ("T57.7b the graded middle value discloses its asymmetry (single-source with the "
     "notebook's _lit_clause semantics)",
     '"down ladder torch-lit (9x9), spawn NOT (dark start, lit destination)"' in GM
     and "dark start, " in SNB),
    ("T57.7c the disclosure test rides the suite",
     "test_light_disclosure_respects_pre_light_override" in TESTS),
]

failed = [name for name, ok in CHECKS if not ok]
for name, ok in CHECKS:
    print(("  PASS " if ok else "  FAIL ") + name)
if failed:
    print(f"v7fix5.8 arm-A designcheck: {len(failed)} FAILED")
    sys.exit(1)
print(f"v7fix5.8 arm-A designcheck: {len(CHECKS)}/{len(CHECKS)} ALL GREEN")
