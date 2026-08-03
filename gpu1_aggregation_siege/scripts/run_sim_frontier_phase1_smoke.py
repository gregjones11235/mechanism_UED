"""CC1 Phase 1 driver: real Craftax EnvState restore/dynamics parity (R4a).

Runs the G0-G8 check suite against the real MiniCraftaxTrain(survive.Env)
environment on CPU and writes:

- <out>/phase1_parity_report.json   (machine-readable evidence, <512KB)
- <out>/PHASE1_REPORT.md            (Chinese human-readable report)

Exit codes: 0 = PASS, 4 = FAIL (a truth check failed), 5 = BLOCKED (external
dependency missing, e.g. craftax/jax import failure).

Scope honesty: PASS here proves ONLY the R4a env-side restore/parity.  It is
not the R4c combined fresh-process proof and it is not a performance result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback


def _write_blocked_report(out_dir: str, where: str, reason: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    report = {
        "report": "simulator_frontier.cc1.phase1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "verdict": "BLOCKED",
        "blocking": {"blocked_at": where, "reason": reason, "checks_run": [],
                     "next_action": "resolve the blocker and rerun; never fake PASS"},
        "scope_note": "PASS here proves ONLY the R4a env-side restore/parity.",
    }
    with open(os.path.join(out_dir, "phase1_parity_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, sort_keys=True)
    with open(os.path.join(out_dir, "PHASE1_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("# Phase 1 BLOCKED\n\n"
                f"- blocked_at：{where}\n"
                f"- reason：{reason}\n\n"
                "未产生任何 PASS 结论；修复阻塞项后重跑本驱动。\n")


def _write_fail_report(out_dir: str, reason: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    report = {
        "report": "simulator_frontier.cc1.phase1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "verdict": "FAIL",
        "blocking": {"blocked_at": "driver", "reason": reason, "checks_run": [],
                     "next_action": "inspect traceback evidence below"},
        "traceback": reason,
        "scope_note": "PASS here proves ONLY the R4a env-side restore/parity.",
    }
    with open(os.path.join(out_dir, "phase1_parity_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, sort_keys=True)
    with open(os.path.join(out_dir, "PHASE1_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("# Phase 1 FAIL\n\n"
                f"- reason：驱动层异常，检查失败（fail-closed，不伪装 BLOCKED/PASS）。\n\n"
                "```\n" + reason + "\n```\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True,
                        help="output dir, e.g. gpu1_aggregation_siege/reports/simulator_frontier_cc1/phase1")
    args = parser.parse_args(argv)
    out_dir = args.out

    try:
        from dicode.simulator_frontier import craftax_checks as cc
    except Exception as exc:
        _write_blocked_report(out_dir, "import", f"{type(exc).__name__}: {exc}")
        print(f"[phase1] BLOCKED at import: {type(exc).__name__}: {exc}")
        return 5

    try:
        report = cc.run_all(out_dir)
    except Exception:
        evidence = traceback.format_exc()
        _write_fail_report(out_dir, evidence)
        print("[phase1] FAIL: unexpected exception inside checks (fail-closed)")
        print(evidence)
        return 4

    verdict = report.get("verdict")
    print(f"[phase1] verdict={verdict} wall_seconds={report.get('wall_seconds')}")
    for name, check in report.get("checks", {}).items():
        print(f"[phase1]   {name}: {'PASS' if check.get('pass') else 'FAIL'}")
    if report.get("blocking"):
        print(f"[phase1] blocking={report['blocking']}")
    print(f"[phase1] reports written to {out_dir}")
    print("[phase1] scope: R4a env-side restore/parity ONLY; not R4c joint proof; not performance.")
    if verdict == "PASS":
        return 0
    if verdict == "FAIL":
        return 4
    return 5


if __name__ == "__main__":
    sys.exit(main())
