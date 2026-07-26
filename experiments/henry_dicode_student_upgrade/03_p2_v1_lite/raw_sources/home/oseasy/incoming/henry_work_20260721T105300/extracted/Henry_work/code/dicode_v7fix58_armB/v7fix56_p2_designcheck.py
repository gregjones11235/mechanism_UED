"""v7fix5.7-P2' designcheck — judgment statistics repair (fix56设计 §3, treats E4/E5).

LAW: judgment statistics must match the measurement's noise profile (±4pp heavy-tail per
honest zero-shot reading). fix5.6 swapped the measurement SOURCE but left the judgment layer
reading single values through the modeler-decision cadence. The five §3 points:
  1. full consumption — every eval judged, every session, at delivery (run_dicode Step 4d);
  2. window-mean judgments — new-high ratchet / patience / graduate / stall-regress all run
     on the last-3 window mean; a lone lucky reading moves nothing;
  3. DEFEND rising reads the RAW per-rung series (last-4 slope), not a consumption
     subsequence (the E4 root);
  4. park semantics untouched;
  5. scientist information repair — feasible-axis menu (dead axes say EXHAUSTED) + the
     unverifiable footnote hardened to refuted-strength (the E5 root).
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
GM = _read("src", "dicode", "dreaming", "gen_manager.py")
RD = _read("experiments", "training", "run_dicode.py")
TESTS = _read("auction", "tests", "test_siege_p2_stats.py")

import inspect  # noqa: E402

from auction.siege_notebook import (  # noqa: E402
    RUNG_WIN,
    RUNG_WIN_NEW_HIGH_PP,
    SiegeNotebook,
)

_rising_src = inspect.getsource(SiegeNotebook._relay_ratchet_rising)

CHECKS = [
    ("T1 full consumption: the state machine eats EVERY eval at delivery (run_dicode Step 4d "
     "calls consume_rung_eval per relay wall, per session); the decision-cadence site no "
     "longer judges anything (gen_manager has ZERO note_rung_reading calls left)",
     "consume_rung_eval" in RD and "[siege][RUNG]" in RD
     and RD.index("run_rung_eval(") < RD.index("consume_rung_eval")
     and "note_rung_reading" not in GM
     and "def consume_rung_eval" in SNB),
    ("T2 window-mean single source: ONE win-window derivation (hist[-RUNG_WIN:]) feeds "
     "new-high, both streaks AND the rung-hold render (same _w3 variable); window not full "
     "-> counters hold at 0",
     SNB.count("hist[-RUNG_WIN:]") == 1
     and RUNG_WIN == 3 and abs(RUNG_WIN_NEW_HIGH_PP - 2.0) < 1e-9
     and "_w3 is not None and _w3 >= self.th.rung_graduate_sr" in SNB
     and "_w3 is not None and _w3 < self.th.rung_floor_sr" in SNB
     and "(win3 {_w3_s})" in SNB),
    ("T3 DEFEND rising input = the RAW per-rung reading series (last-4 slope), never the "
     "consumption-time micro-ratchet log",
     "rung_trained" in _rising_src and "ratchet_log" not in _rising_src
     and "[-4:]" in _rising_src),
    ("T4 the feasible-axis menu renders from the SAME stepper that executes interventions "
     "(probe_variant_knobs), dead axes say EXHAUSTED",
     "AXIS MENU" in SNB and "EXHAUSTED" in SNB
     and SNB.index("AXIS MENU") > SNB.index("PROBE TOOL available")),
    ("T5 the old single-read judgment paths are DELETED (not switched): no single-reading "
     "graduate/stall/new-high comparisons survive; the +3pp single-read anchor is gone",
     "if reading >= self.th.rung_graduate_sr" not in SNB
     and "if reading < self.th.rung_floor_sr" not in SNB
     and "reading >= float(best) + self.th.gap_stall_min_gain_pp" not in SNB),
    ("T5b patience holds while the window fills (evidence, not absence, moves the machine) "
     "and the unverifiable footnote carries refuted-strength teaching (E5)",
     "elif _w3 is None:" in SNB
     and "CANNOT be the binding" in SNB
     and "Propose a DIFFERENT axis" in SNB),
    ("T6 the P2' test suite pins the semantics",
     all(t in TESTS for t in (
         "test_single_lucky_reading_does_not_graduate",
         "test_win3_graduation_needs_a_full_window",
         "test_win3_stall_regress",
         "test_new_high_and_patience_ride_win3",
         "test_rising_reads_raw_series",
         "test_consume_rung_eval_delivers_and_holds",
         "test_axis_menu_and_unverifiable_footnote",
     ))),
]

n_pass = 0
for name, ok in CHECKS:
    print(("PASS  " if ok else "FAIL  ") + name)
    n_pass += bool(ok)
print()
print(f"v7fix5.7-P2' judgment-stats designcheck: {n_pass}/{len(CHECKS)}"
      + (" ALL GREEN" if n_pass == len(CHECKS) else " — FIX BEFORE LAUNCH"))
if n_pass != len(CHECKS):
    raise SystemExit(1)
