"""v7fix5.6 designcheck — honest rung readings (measurement decoupled from training).

Run on a login node with PYTHONPATH pointing at the fix56 tree (export FIRST — the editable
.pth loads the v4 tree otherwise, fix55 P2.1 lesson), and inside the launcher gate.
"""

import os
import sys
import tempfile

BASE = os.path.expanduser("~/dicode_v7fix56")
FAILS = []
N = [0]


def check(name: str, ok: bool, detail: str = "") -> None:
    N[0] += 1
    tag = "PASS" if ok else "FAIL"
    print(f"{tag} {N[0]:2d} {name}" + (f" — {detail}" if (detail and not ok) else ""))
    if not ok:
        FAILS.append(name)


def read(rel_glob: str) -> str:
    import glob
    hits = [p for p in glob.glob(f"{BASE}/**/{rel_glob}", recursive=True)
            if "__pycache__" not in p and "/tests/" not in p]
    assert len(hits) == 1, f"{rel_glob}: {hits}"
    with open(hits[0], encoding="utf-8") as f:
        return f.read()


sn = read("siege_notebook.py")
gm = read("gen_manager.py")
rd = read("run_dicode.py")
rp = read("rung_probe.py")
md = read("modeler.py")

# ---- source pins ----
check("S1 budget: PROBE_BUDGET_WINDOW_FAST = 5 exists",
      "PROBE_BUDGET_WINDOW_FAST = 5" in sn)
check("S2a budget: diagnose/whatif ledger scan uses the FAST window",
      "- PROBE_BUDGET_WINDOW_FAST:\n                used[k] += 1" in sn)
check("S2b budget: the verify kind keeps the 10-session window",
      any(('k == "verify"' in ln and "PROBE_BUDGET_WINDOW:" in ln
           and "FAST" not in ln) for ln in sn.splitlines()))
check("S3 schema: rung_eval key in _empty_notebook (coerce-safe)",
      '"rung_eval": {},' in sn)
check("S4a accessor: note_rung_eval defined and self-persisting",
      "def note_rung_eval" in sn
      and "self._save()" in sn.split("def note_rung_eval", 1)[1][:1200])
check("S4b accessor: rung_eval_for stage identity derives from relay_scaffold (single source)",
      "def rung_eval_for" in sn
      and "self.relay_scaffold(wall)" in sn.split("def rung_eval_for", 1)[1][:1600])
check("S4c accessor: rung_eval_for freshness window is one session",
      'int(e.get("session", -1)) < int(session_idx) - 1' in sn)
check("S5a nudge: stale report voids the escape clause in so many words",
      "does NOT count as fresh evidence" in sn)
check("S5b nudge: patience >= 4 renders an unconditional DIRECTIVE",
      "_pat56 >= 4" in sn and "This is a DIRECTIVE" in sn)
check("S5c nudge: the unconditional escape clause survives ONLY in the fresh-report branch",
      sn.count("unless you can already cite FRESH") == 1)
check("S6a gen_manager: state machine consumes rung_eval_for, never the trained max",
      "rung_eval_for" in gm and "_nrr(wall, eval_v" in gm
      and "_nrr(wall, trained_v" not in gm)
check("S6b gen_manager: RUNG log line labels the honest source",
      "zero-shot={z_s} -> {rstatus}" in gm)
check("S7 gen_manager: wandb logs both the honest and the telemetry number",
      "rung_zeroshot_sr_" in gm and "rung_trained_sr_" in gm)
check("S8 run_dicode: Step 4d rung eval wired after the probe hook, guarded",
      "Step 4d" in rd and "run_rung_eval" in rd
      and "must never break training" in rd.split("Step 4d", 1)[1][:1200])
check("S9a rung_probe: run_rung_eval exists with certificate sizes 512x4096",
      "def run_rung_eval" in rp and "RUNG_EVAL_NUM_ENVS = 512" in rp
      and "RUNG_EVAL_NUM_STEPS = 4096" in rp)
check("S9b rung_probe: fix5.5 probe path sizes untouched (256x2048)",
      "PROBE_NUM_ENVS = 256" in rp and "PROBE_NUM_STEPS = 2048" in rp)
check("S9c rung_probe: eval seeds derive from session_idx + wall (deterministic, "
      "training-independent)",
      "7919 + 56" in rp and "zlib.crc32" in rp)
check("S9d rung_probe: eval writes ONLY the notebook rung_eval slot",
      "note_rung_eval(" in rp
      and "deliver_probe_report" not in rp.split("def run_rung_eval", 1)[1])
check("S9e rung_probe: eval level code comes from _relay_level_build (same template "
      "as training, never string surgery)",
      "_relay_level_build" in rp.split("def run_rung_eval", 1)[1])
check("S10 modeler: raw LLM responses persisted (both call sites)",
      "_dump_llm_call" in md and md.count("self._dump_llm_call(") >= 2
      and "llm_calls" in md)

# ---- behavioural ----
sys.path.insert(0, BASE)
try:
    from auction.siege_notebook import SiegeNotebook  # noqa: E402

    tmp = tempfile.mkdtemp(prefix="fix56dc_")

    nb = SiegeNotebook(os.path.join(tmp, "nb1.json"))
    nb._nb.setdefault("probe_ledger", {})["w"] = [[0, "diagnose"], [0, "whatif"]]
    b4 = nb._probe_budget_left("w", 4)
    b5 = nb._probe_budget_left("w", 5)
    check("B1 budget window: spent at s0 -> empty at s4, restored at s5 (FAST=5)",
          b4 == {"diagnose": 0, "whatif": 0} and b5 == {"diagnose": 1, "whatif": 1},
          f"b4={b4} b5={b5}")

    nb2 = SiegeNotebook(os.path.join(tmp, "nb2.json"))
    nb2.note_rung_eval("defeat_kobold", {
        "session": 10, "sr": 33.3, "spawn_floor": 2, "sub_stage": 4, "n_envs": 512,
    })
    re2 = SiegeNotebook(nb2.path)._nb.get("rung_eval", {}).get("defeat_kobold", {})
    check("B2 rung_eval survives save/reload/_coerce round-trip",
          re2.get("sr") == 33.3 and re2.get("sub_stage") == 4, f"got {re2}")

    check("B3 rung_eval_for: no relay focus -> None (counters hold)",
          nb2.rung_eval_for("defeat_kobold", 11) is None)
except Exception as e:  # noqa: BLE001
    check("B* behavioural block ran", False, f"{type(e).__name__}: {e}")

print(f"\n{N[0] - len(FAILS)}/{N[0]} fix5.6 design points hold")
if FAILS:
    print("FATAL: " + "; ".join(FAILS))
    sys.exit(1)
