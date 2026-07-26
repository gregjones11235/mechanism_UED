"""v7fix5.7 designcheck — graded pre-light anneal (both -> ladder -> none).

Why: the fix5.4 probe measured the entry-context cliff at ~25pp and fix5.5 verified pre_light
as its main axis (+25.4pp -> the compiled INSERT rung). Graduating that insert used to fall
straight back to the dark return stage — the whole verified cliff re-imposed in ONE step.
fix5.7 splits it into notches:
  (a) builder: pre_light gains the middle value "ladder" — ONLY the down ladder's 9x9 is
      torch-lit (dark start, lit destination); None/True/False byte-compatible.
  (b) state machine: a graduated pre_light=True insert descends to RUNG_INSERT_LIGHT_STAGE
      (49, distinct id -> fix4.6 exact-stage reading isolation + fresh best_by_rung key) with
      knobs annealed True -> "ladder"; the light leg's graduation pops the insert normally;
      a stall on the leg self-heals through the existing regress path.
  (c) what-if axis: 3-level ladder, one notch per step (no more lit<->dark cliff probes).
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "src"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _read(*parts):
    with open(os.path.join(_HERE, *parts), encoding="utf-8") as f:
        return f.read()


SNB = _read("auction", "siege_notebook.py")
WB = _read("src", "minicraftax", "world_builder.py")
TESTS = _read("auction", "tests", "test_siege_fix57_light_anneal.py")

from auction.siege_notebook import (  # noqa: E402
    RUNG_INSERT_LIGHT_STAGE,
    RUNG_INSERT_STAGE,
    SiegeNotebook,
)

_facts = SiegeNotebook._scaffold_fact_clauses(
    {"down_ladder_radius": None, "monster_credit": 8, "uplock": True,
     "needs_multiplier": 0.3, "pre_light": "ladder"}
)

CHECKS = [
    ("Y.1 builder accepts the middle value: signature is bool|str|None, 'ladder' resolves to "
     "a ladder-only stamp, and _stamp_scaffold_light takes ladder_only",
     "pre_light: bool | str | None = None" in WB
     and 'elif _pl == "ladder":' in WB
     and "def _stamp_scaffold_light(self, ladder_only: bool = False)" in WB
     and "((ladder,) if ladder_only else (ladder, self.player_position))" in WB),
    ("Y.2 None/True/False stay byte-compatible (coupled default + bool override literals "
     "unchanged in build())",
     "_stamp, _ladder_only = self._down_ladder_spawn_radius is not None, False" in WB
     and "_stamp, _ladder_only = bool(_pl), False" in WB),
    ("Y.3 the light leg id is distinct and outside every ladder range (same scheme as the "
     "insert id)",
     RUNG_INSERT_LIGHT_STAGE == 49 and RUNG_INSERT_STAGE == 50
     and RUNG_INSERT_LIGHT_STAGE != RUNG_INSERT_STAGE),
    ("Y.4 relay_scaffold serves BOTH insert ids from the frozen insert knobs",
     "stage in (RUNG_INSERT_STAGE, RUNG_INSERT_LIGHT_STAGE)" in SNB
     and 'knobs["sub_stage"] = stage' in SNB),
    ("Y.5 graduate path: a pre_light=True insert anneals (True -> \"ladder\", stage 49) "
     "BEFORE the pop; the pop still exists for the leg / non-light inserts",
     '_k57["pre_light"] = "ladder"' in SNB
     and "RUNG_INSERT_LIGHT_ANNEAL" in SNB
     and SNB.index("RUNG_INSERT_LIGHT_ANNEAL") < SNB.index("RUNG_INSERT_GRADUATED (trained")),
    ("Y.6 fresh ratchet on the light leg (fix53/54 surgery principle: new semantics = new "
     "best_by_rung key, stale best dropped)",
     'f"{_floor_i55}:{RUNG_INSERT_LIGHT_STAGE}"' in SNB),
    ("Y.7 the scaffold-facts disclosure has the third state (computed, not a template "
     "constant)",
     "dark start" in _facts["pre-light"] and "down ladder is torch-lit" in _facts["pre-light"]),
    ("Y.8 the what-if axis is a 3-level graded ladder with boundary receipts",
     '_levels = [False, "ladder", True]' in SNB and "pre_light_at_boundary" in SNB),
    ("Y.9 the journal render marks the leg for the LLM (facts wording, no tactics)",
     "LIGHT-ANNEAL leg" in SNB and "spawn stamp removed, down" in SNB),
    ("Y.10 the fix5.7 test suite pins the leg lifecycle",
     all(t in TESTS for t in (
         "test_true_insert_graduates_via_light_anneal_leg",
         "test_light_leg_graduation_pops_back_to_return_stage",
         "test_ladder_insert_pops_in_one_graduation",
         "test_stall_on_light_leg_removes_insert_and_regresses",
         "test_light_leg_ratchet_is_fresh",
         "test_facts_clause_ladder_mode",
         "test_render_marks_light_anneal_leg",
     ))),
]

n_pass = 0
for name, ok in CHECKS:
    print(("PASS  " if ok else "FAIL  ") + name)
    n_pass += bool(ok)
print()
print(f"v7fix5.7 light-anneal designcheck: {n_pass}/{len(CHECKS)}"
      + (" ALL GREEN" if n_pass == len(CHECKS) else " — FIX BEFORE LAUNCH"))
if n_pass != len(CHECKS):
    raise SystemExit(1)
