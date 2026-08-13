#!/usr/bin/env python3
"""D6 offline overlap/wait analysis from the archived Mason timing summary."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def p95(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[94]


def make_out_dir(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = root / f"timing_overlap_{stamp}"
    serial = 1
    while out.exists():
        out = root / f"timing_overlap_{stamp}_{serial}"
        serial += 1
    out.mkdir(parents=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=HERE / ".." / ".." / ".." / ".." / "d6_artifacts")
    args = ap.parse_args()

    mason_path = HERE / "MASON_LLM_BASELINE_AUDIT.json"
    mason = read_json(mason_path)
    phase = mason["phase_timing"]
    per_session = mason.get("evolution_per_session_s", [])
    evolution = [float(x["evolution_s"]) for x in per_session]
    compilation = [float(x["compilation_s"]) for x in per_session]
    evolution_total = float(phase["evolution_total_s"])
    compilation_total = float(phase["compilation_total_s"])
    training_total = float(phase["training_total_s"])
    wall = float(mason["wall_clock"]["duration_s_approx"])
    component_sum = evolution_total + compilation_total + training_total
    overlap = component_sum - wall
    overlap_fraction = overlap / component_sum if component_sum else None

    # No event-level critical path or aligned session boundaries are archived.
    # Therefore synchronized queue wait cannot be estimated, let alone gated.
    result = {
        "schema_version": 1,
        "classification": "D6_TIMING_OVERLAP_OFFLINE_AUDIT",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "INSUFFICIENT_SESSION_BOUNDARY_TRACE",
        "queue_decision": "NO_QUEUE_JUSTIFIED_INSUFFICIENT_TRACE",
        "conclusion": "NO_QUEUE_JUSTIFIED_INSUFFICIENT_TRACE",
        "network_calls": 0,
        "completion_requests": 0,
        "embedding_requests": 0,
        "gpu_processes_started": 0,
        "executed_count": 0,
        "input_paths": [str(mason_path)],
        "script_sha256": sha256(Path(__file__).resolve()),
        "source": {"mason_audit": {"path": mason_path.name, "sha256": sha256(mason_path)}},
        "wall_clock_s": wall,
        "component_seconds": {
            "evolution": evolution_total,
            "compilation": compilation_total,
            "training": training_total,
            "sum": component_sum,
        },
        "overlap": {
            "component_sum_minus_wall_s_upper_bound": overlap,
            "fraction_of_component_sum": overlap_fraction,
            "interpretation": "upper bound only: component durations exceed wall; background-thread overlap is documented, but actual overlap intervals are not observed",
        },
        "phase_duration_distribution_s": {
            "evolution": {"count": len(evolution), "median": statistics.median(evolution) if evolution else None, "p95": p95(evolution)},
            "compilation": {"count": len(compilation), "median": statistics.median(compilation) if compilation else None, "p95": p95(compilation)},
        },
        "synchronized_wait": {
            "median_s": None,
            "p95_s": None,
            "threshold_median_fraction": 0.05,
            "threshold_p95_s": 120.0,
            "reason_unavailable": "timings.csv summary has phase durations only; no critical_path/events or aligned producer-consumer wait spans, so the 5% relative median gate cannot be evaluated",
        },
        "gpu_idle_attribution": {
            "gpu3_training_is_historical_only": True,
            "idle_seconds_observed": None,
            "reason_unavailable": "no event-level GPU idle intervals in the archived summary",
        },
        "guardrails": [
            "Offline read of archived audit only.",
            "No fixed sequence queue experiment was launched because the gating median/P95 wait is not observable.",
            "NO_QUEUE_JUSTIFIED_INSUFFICIENT_TRACE means the threshold cannot be evaluated; it is not a claim that queue overhead is zero.",
        ],
    }

    out = make_out_dir(args.output_root.resolve())
    result_path = out / "D6_RESULT.json"
    report_path = out / "D6_REPORT.md"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = f"""# D6 timing/overlap audit (offline)

- Status: **INSUFFICIENT_SESSION_BOUNDARY_TRACE**
- Queue decision: **NO_QUEUE_JUSTIFIED_INSUFFICIENT_TRACE**
- Archived wall clock: `{wall:.2f}s`
- Evolution + compilation + training component sum: `{component_sum:.2f}s`
- Component-sum minus wall **upper bound**: `{overlap:.2f}s` (`{overlap_fraction:.2%}` of component sum; not measured interval overlap)

Evolution duration median/P95: `{statistics.median(evolution):.2f}s` / `{p95(evolution):.2f}s`;
compilation median/P95: `{statistics.median(compilation):.2f}s` / `{p95(compilation):.2f}s`.
These are phase-duration distributions, not synchronization waits. The archived run has no
critical-path/events file and no aligned producer-consumer/session wait spans, so synchronized
wait median/P95 and GPU-idle intervals are **not observable**. The requested fixed sequence
queue experiment is therefore not justified by evidence and was not launched; this is an
insufficiency-of-trace decision, not a claim that the threshold was below its gate.

No GPU, provider, or original timing artifact was touched.
    """
    report_path.write_text(report, encoding="utf-8")
    result["report_sha256"] = sha256(report_path)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sums = "\n".join(f"{sha256(p)}  {p.name}" for p in (result_path, report_path)) + "\n"
    (out / "SHA256SUMS").write_text(sums, encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
