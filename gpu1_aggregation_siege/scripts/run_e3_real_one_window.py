#!/usr/bin/env python3
"""E3 production driver: ONE real frontier window (P0-7 entrypoint).

Runs the full production chain for exactly one frontier window:

    standard-reset rollout -> production archive capture -> single-fresh-
    process joint restore -> real actual-N branch search -> 0-or-2 typed
    LLM calls -> deterministic evidence selector -> 12+4 composition ->
    mixed-start rollouts -> injected original loss -> EXACTLY ONE injected
    optimizer update -> checkpoint round trip -> replay -> NaN/Inf checks.

THIS ROUND: the driver is expected to stop at ``run_e3_preflight`` with
named blockers (R9 training surface, controller-signed RegistryBundle,
shared anchor manifest, frozen formal asset registry, saved-policy-memory
artifact, authorized LLM client, injected original loss/update).  It writes
an honest JSON report and exits 5 (BLOCKED).  It NEVER fills a gap with a
fake adapter, synthetic bundle or fake client, and never reports a blocked
window as executed.

Usage (venv python, PYTHONPATH=<repo>/gpu1_aggregation_siege/src,
JAX_PLATFORMS=cpu):

    python run_e3_real_one_window.py \
        student.profile=rmt16_persistent_98304 \
        student.checkpoint_path=<PATH> \
        [--run-id=<ID>] [--out=<DIR>]

Exit codes: 0 PASS, 4 FAIL, 5 BLOCKED.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "gpu1_aggregation_siege" / "src"
DEFAULT_OUT_DIR = REPO_ROOT / "gpu1_aggregation_siege" / "reports" / "simulator_frontier_foundation"

PASS, FAIL, BLOCKED = 0, 4, 5


def _log(msg: str) -> None:
    print(f"[e3-one-window] {msg}", flush=True)


def parse_args(argv):
    run_id = "e3w-unspecified"
    out_dir = str(DEFAULT_OUT_DIR)
    rest = []
    for arg in argv:
        if arg.startswith("--run-id="):
            run_id = arg.split("=", 1)[1]
        elif arg.startswith("--out="):
            out_dir = arg.split("=", 1)[1]
        else:
            rest.append(arg)
    return rest, run_id, out_dir


def _finish(report: dict, out_dir: str) -> None:
    try:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "e3_real_one_window.json"
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=False)
        os.replace(tmp, path)
        print(f"[e3-one-window] report: {path} ({path.stat().st_size} B)", flush=True)
    except Exception as exc:
        print(f"[e3-one-window] REPORT_WRITE_FAILED: {exc!r}", flush=True)
    print(f"[e3-one-window] verdict={report.get('verdict')} "
          f"reason={report.get('reason', '-')}", flush=True)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    started = time.time()
    report: dict = {
        "schema": "simulator_frontier.e3_real_one_window/v1",
        "REAL_ACTUAL_N_EXECUTED": False,
        "REAL_TWO_LLM_EXECUTED": False,
        "REAL_ONE_UPDATE_EXECUTED": False,
        "CHECKPOINT_RELOAD": False,
        "disclaimers": [
            "this entrypoint never substitutes fake adapters, synthetic registry "
            "bundles, fake LLM clients or fake branch outcomes",
            "a blocked preflight is reported as BLOCKED, never as executed",
            "the Student network/reward/action head/optimizer/original loss are "
            "never modified by this driver",
        ],
    }
    try:
        rest, run_id, out_dir = parse_args(argv)
    except Exception as exc:
        report["verdict"] = "BLOCKED"
        report["reason"] = f"argv parse error: {exc!r}"
        _finish(report, str(DEFAULT_OUT_DIR))
        return BLOCKED
    report["run_id"] = run_id

    try:
        from dicode.simulator_frontier.e3_window import (
            E3WindowConfig,
            one_window_pipeline,
            run_e3_preflight,
        )
        from dicode.simulator_frontier.errors import ProductionBlockedError
        from dicode.simulator_frontier.memory_modes import (
            MemoryRestoreMode,
            MemoryRestoreRequest,
        )
    except Exception as exc:
        report["verdict"] = "BLOCKED"
        report["reason"] = f"BLOCKED_ENVIRONMENT: {exc!r}"
        _finish(report, out_dir)
        return BLOCKED

    try:
        from dicode.student_adapters.registry import (
            default_profile_dir,
            load_student_profile,
            resolve_runtime_overrides,
        )
    except Exception as exc:
        report["verdict"] = "BLOCKED"
        report["reason"] = f"BLOCKED_ENVIRONMENT: {exc!r}"
        _finish(report, out_dir)
        return BLOCKED

    try:
        overrides = resolve_runtime_overrides(rest)
    except Exception as exc:
        report["verdict"] = "FAIL"
        report["reason"] = f"runtime override violation: {exc!r}"
        _finish(report, out_dir)
        return FAIL

    profile_name = overrides.get("student.profile")
    checkpoint_path = overrides.get("student.checkpoint_path")

    adapter = None
    profile = None
    loaded = None
    if not profile_name or not checkpoint_path:
        report["mount_status"] = ("BLOCKED_ARTIFACT: student.profile and "
                                  "student.checkpoint_path are both required "
                                  "(never defaulted, never guessed)")
    else:
        if not Path(checkpoint_path).is_file():
            report["verdict"] = "BLOCKED"
            report["reason"] = f"BLOCKED_ARTIFACT: checkpoint missing: {checkpoint_path}"
            _finish(report, out_dir)
            return BLOCKED
        try:
            profile = load_student_profile(default_profile_dir() / f"{profile_name}.yaml")
        except Exception as exc:
            report["verdict"] = "FAIL"
            report["reason"] = f"profile load failed (fail closed): {exc!r}"
            _finish(report, out_dir)
            return FAIL
        if profile.architecture_family != "RMT16":
            report["verdict"] = "BLOCKED"
            report["reason"] = (f"architecture_family {profile.architecture_family} has "
                                "no production adapter this round (RMT16 only)")
            _finish(report, out_dir)
            return BLOCKED
        try:
            from dicode.student_adapters.rmt16_adapter import RMT16StudentAdapter
            adapter = RMT16StudentAdapter(profile)
            loaded = adapter.load_full_state(checkpoint_path, profile.expected_identity())
            report["REAL_CHECKPOINT_LOADED"] = True
        except Exception as exc:
            report["verdict"] = "FAIL"
            report["REAL_CHECKPOINT_LOADED"] = False
            report["reason"] = f"checkpoint load gate chain failed: {exc!r}"
            _finish(report, out_dir)
            return FAIL
        report["candidate_id"] = profile.candidate_id

    # Official production memory mode for this window: SAVED_POLICY_MEMORY.
    # The CC2 checkpoint carries contains_memory=False, so NO artifact exists
    # this round — the preflight reports that honestly instead of faking one.
    memory_request = None
    if profile is not None:
        memory_request = MemoryRestoreRequest(
            mode=MemoryRestoreMode.SAVED_POLICY_MEMORY,
            policy_architecture_id=str(profile.architecture_family),
            checkpoint_id=str(profile.params_sha256),
        )

    config = E3WindowConfig(
        run_id=run_id,
        student=adapter,
        student_params=(loaded or {}).get("params"),
        loaded_state=loaded,
        memory_mode=MemoryRestoreMode.SAVED_POLICY_MEMORY.value,
        memory_request=memory_request,
        memory_artifact=None,     # no real artifact exists this round
        memory_loader=None,       # no production loader authorized this round
        capture_provenance=None,  # controller-provenance pending
        restore_request=None,     # controller-signed RegistryBundle pending
        two_llm_runtime=None,  # no authorized real LLM runtime this round
        anchor_manifest=None,     # controller-signed manifest pending
        retention=None,
        loss_fn=None,             # original loss injection pending audit
        optimizer_update_fn=None,  # optimizer update injection pending audit
    )

    pre = run_e3_preflight(config)
    report["preflight"] = {
        "ready": pre.ready,
        "gates": dict(pre.gates),
        "blockers": list(pre.blockers),
        "preflight_version": pre.preflight_version,
    }
    _log(f"preflight ready={pre.ready} blockers={list(pre.blockers)}")

    if not pre.ready:
        report["verdict"] = "BLOCKED"
        report["reason"] = ("E3_PREFLIGHT_BLOCKED: " + "; ".join(pre.blockers))
        report["elapsed_s"] = round(time.time() - started, 2)
        _finish(report, out_dir)
        return BLOCKED

    # Preflight green -> run the real window (reachable only after audit).
    try:
        result = one_window_pipeline(config)
    except ProductionBlockedError as exc:
        report["verdict"] = "BLOCKED"
        report["reason"] = str(exc)
        _finish(report, out_dir)
        return BLOCKED
    except Exception as exc:
        report["verdict"] = "FAIL"
        report["reason"] = f"one_window_pipeline failed: {exc!r}"
        _finish(report, out_dir)
        return FAIL

    report["pipeline_status"] = result["status"]
    report["REAL_ACTUAL_N_EXECUTED"] = bool(result.get("real_actual_n_executed"))
    report["REAL_TWO_LLM_EXECUTED"] = bool(result.get("real_two_llm_executed"))
    report["REAL_ONE_UPDATE_EXECUTED"] = bool(result.get("real_one_update_executed"))
    report["CHECKPOINT_RELOAD"] = bool(result.get("checkpoint_reload"))
    report["steps"] = result.get("steps", {})
    report["elapsed_s"] = round(time.time() - started, 2)
    if result["status"] == "PASS":
        report["verdict"] = "PASS"
        _finish(report, out_dir)
        return PASS
    if result["status"] == "SELECTOR_REJECTED":
        report["verdict"] = "BLOCKED"
        report["reason"] = "evidence selector rejected the plan (no update executed)"
        _finish(report, out_dir)
        return BLOCKED
    report["verdict"] = "FAIL"
    report["reason"] = f"pipeline status {result['status']}"
    _finish(report, out_dir)
    return FAIL


if __name__ == "__main__":
    raise SystemExit(main())
