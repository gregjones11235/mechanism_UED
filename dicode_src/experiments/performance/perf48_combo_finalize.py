#!/usr/bin/env python3
"""Final verdict for the BC combination experiment (Phase C).

Combines the deterministic-XLA semantic run (det_root) and the default-XLA
performance run (perf_root), each produced by perf48_combo_benchmark.py, and
applies the ten acceptance conditions:

  1  all six deterministic pairs are SEMANTIC_PASS
  2  all six default-XLA pairs are valid (both arms produced RESULT)
  3  mean session wall improvement > 0   (positive = BC_ON faster)
  4  median session wall improvement > 0 (positive = BC_ON faster)
  5  mean session throughput improvement > 0 (positive = BC_ON higher throughput)
  6  no single-pair session-wall regression > 1%
  7  C real cache hit (BC_ON compile once, first miss second hit; BC_OFF zero)
  8  B2 auditable task-reload elimination (reload absent in BC_ON, present in BC_OFF)
  9  no OOM/Xid/corruption/external interference/unresolved traceback
  10 memory gates (peak delta <= 512 MiB, min free >= 4096 MiB)

Conclusion (mutually exclusive, in priority order):
  REJECTED_SEMANTIC_MISMATCH    condition 1 fails
  REJECTED_RUNTIME_FAILURE      condition 2, 7, 8, 9 or 10 fails
  REJECTED_NO_SPEEDUP           condition 3, 4, 5 or 6 fails
  BC_COMBINATION_PASS_LIMITED_GAIN  0 < mean improvement < 3%
  BC_COMBINATION_PASS           mean improvement >= 3%
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Mapping

STAGES = ("early", "mid", "late")
RUNTIME_MARKERS = ("runtime_failure", "fatal_error", "oom", "xid", "checkpoint_error", "gpu_violation")


def load_pair(root: Path, stage: str, repeat: int) -> dict[str, Any]:
    path = Path(root) / "pairs" / f"{stage}_repeat_{repeat}" / "PAIR.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _arm(pair: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return pair.get(name, pair.get("e0" if name == "off" else "e3", {}))


def compute_verdict(det_root: Path, perf_root: Path) -> dict[str, Any]:
    det_pairs = [load_pair(det_root, s, r) for s in STAGES for r in (0, 1)]
    perf_pairs = [load_pair(perf_root, s, r) for s in STAGES for r in (0, 1)]

    conditions: dict[str, dict[str, Any]] = {}

    def cond(n: int, ok: bool, detail: Any = None) -> None:
        conditions[str(n)] = {"ok": bool(ok), "detail": detail}

    # 1: deterministic semantic
    det_statuses = [p.get("status") for p in det_pairs]
    cond(1, all(s == "SEMANTIC_PASS" for s in det_statuses), det_statuses)

    # 2: perf pairs valid
    cond(2, all(("off" in p and "on" in p and p.get("off") and p.get("on")) for p in perf_pairs))

    # 7: C cache
    c_ok = all(
        _arm(p, "off").get("eval_compile_span_count") == 0
        and _arm(p, "on").get("eval_compile_span_count") == 1
        and _arm(p, "on").get("eval_cache_hit_count") == 1
        and _arm(p, "on").get("eval_first_cache_miss") is True
        for p in perf_pairs)
    cond(7, c_ok)

    # 8: B2 reload elimination
    b2_ok = all(
        _arm(p, "on").get("preflight_task_reload_occurred") is False
        and _arm(p, "off").get("preflight_task_reload_occurred") is True
        for p in perf_pairs)
    cond(8, b2_ok)

    # 9: runtime markers / external / traceback
    runtime_ok = all(
        (not any(_arm(p, a).get(k) for k in RUNTIME_MARKERS))
        and _arm(p, a).get("checkpoint_loadable")
        for p in perf_pairs for a in ("off", "on"))
    cond(9, runtime_ok)

    # 10: memory gates
    peak_deltas = [float(_arm(p, "on").get("gpu_peak_memory_mib", 0)) - float(_arm(p, "off").get("gpu_peak_memory_mib", 0))
                   for p in perf_pairs]
    min_free = min(float(_arm(p, a).get("gpu_min_free_mib", 0)) for p in perf_pairs for a in ("off", "on"))
    cond(10, max(peak_deltas, default=0) <= 512 and min_free >= 4096,
         {"max_peak_delta_mib": max(peak_deltas, default=0), "min_free_mib": min_free})

    # performance from perf pairs
    d0 = [float(_arm(p, "off").get("session_wall_s", 0)) for p in perf_pairs]
    d3 = [float(_arm(p, "on").get("session_wall_s", 0)) for p in perf_pairs]
    st0 = [float(_arm(p, "off").get("session_throughput_env_s", 0)) for p in perf_pairs]
    st3 = [float(_arm(p, "on").get("session_throughput_env_s", 0)) for p in perf_pairs]
    mean_d0, mean_d3 = statistics.mean(d0), statistics.mean(d3)
    med_d0, med_d3 = statistics.median(d0), statistics.median(d3)
    # wall-clock improvement is POSITIVE when BC_ON is faster (lower wall)
    mean_imp = (mean_d0 - mean_d3) / mean_d0 if mean_d0 else 0.0
    median_imp = (med_d0 - med_d3) / med_d0 if med_d0 else 0.0
    # throughput improvement is POSITIVE when BC_ON has higher throughput
    mean_thr_imp = (statistics.mean(st3) - statistics.mean(st0)) / statistics.mean(st0) if statistics.mean(st0) else 0.0
    cond(3, mean_imp > 0, mean_imp)
    cond(4, median_imp > 0, median_imp)
    cond(5, mean_thr_imp > 0, mean_thr_imp)
    regressions = [i for i, (a, b) in enumerate(zip(d0, d3)) if b > a * 1.01]
    cond(6, len(regressions) == 0, regressions)

    # conclusion priority
    if not conditions["1"]["ok"]:
        conclusion = "REJECTED_SEMANTIC_MISMATCH"
    elif not all(conditions[k]["ok"] for k in ("2", "7", "8", "9", "10")):
        conclusion = "REJECTED_RUNTIME_FAILURE"
    elif not all(conditions[k]["ok"] for k in ("3", "4", "5", "6")):
        conclusion = "REJECTED_NO_SPEEDUP"
    elif mean_imp >= 0.03:
        conclusion = "BC_COMBINATION_PASS"
    else:
        conclusion = "BC_COMBINATION_PASS_LIMITED_GAIN"

    return {
        "conclusion": conclusion,
        "conditions": conditions,
        "det_root": str(det_root),
        "perf_root": str(perf_root),
        "det_statuses": det_statuses,
        "mean_session_off": mean_d0, "mean_session_on": mean_d3,
        "median_session_off": med_d0, "median_session_on": med_d3,
        "mean_session_improvement": mean_imp,
        "median_session_improvement": median_imp,
        "mean_session_throughput_improvement": mean_thr_imp,
        "per_pair_regressions": regressions,
        "per_pair_session_walls_off": d0,
        "per_pair_session_walls_on": d3,
        "max_peak_delta_mib": max(peak_deltas, default=0),
        "min_free_mib": min_free,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--det-root", required=True)
    p.add_argument("--perf-root", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    verdict = compute_verdict(Path(args.det_root), Path(args.perf_root))
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(verdict, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
