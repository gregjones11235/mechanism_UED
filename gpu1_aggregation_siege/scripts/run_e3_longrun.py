#!/usr/bin/env python3
"""E3 long-run entrypoint (P0-8): config freeze ONLY, launch never happens here.

Freezes the complete long-run configuration — total_env_steps=98304, the
actual-N budget, search horizon, Student/Reference identities, the official
production memory modes (SAVED_POLICY_MEMORY / HISTORY_BURN_IN; ZERO_MEMORY
is ablation-only and can never be the official mode), the anchor manifest
reference, seed, the runtime Git SHA and the config hash — and writes it as
an immutable run config.

THIS ROUND PREPARES THE ENTRYPOINT ONLY.  The launch gate is DYNAMIC
(CC4 follow-up P0-17): the blocker list is derived from ACTUAL evidence —
the real E3 preflight result (carried over from ``--preflight-report``, or
``BLOCKED_E3_PREFLIGHT_NOT_EVALUATED`` if none is supplied), the Reference
designation state, the shared anchor manifest reference, the experiment
director's signed training-budget decision (``--budget-decision``) and the
external audit approval flag.  As production dependencies bind, the blocker
list shrinks mechanically; it never goes stale.

Budget semantics are one of exactly two director decisions:
TOTAL_FROM_COMMON_INITIALIZATION (the frozen total_env_steps is the whole
budget from common init) or ADDITIONAL_FROM_PRETRAINED_CHECKPOINT (the
frozen total_env_steps is additional training on top of the pretrained
checkpoint).  Without a minted decision the gate reports
BLOCKED_WAITING_DIRECTOR_BUDGET_DECISION — the budget is never assumed.

Usage (venv python):

    python run_e3_longrun.py [--out=<DIR>] [--audit-approved=true]
        [--preflight-report=<preflight JSON>] [--budget-decision=<decision JSON>]

Exit codes: 0 PASS (launch authorized and handed off — not this round),
4 FAIL, 5 BLOCKED.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "gpu1_aggregation_siege" / "reports" / "simulator_frontier_foundation"

PASS, FAIL, BLOCKED = 0, 4, 5

# ---------------------------------------------------------------------------
# FROZEN long-run parameters (changes require a new audit round)
# ---------------------------------------------------------------------------
TOTAL_ENV_STEPS = 98304
NUM_ENVS = 256
NUM_STEPS = 64
ENV_STEPS_PER_UPDATE = NUM_ENVS * NUM_STEPS        # 16384
NUM_UPDATES = TOTAL_ENV_STEPS // ENV_STEPS_PER_UPDATE  # 6
REQUESTED_N_PER_WINDOW = 12
SEARCH_HORIZON = 64
SEED = 20260803
STUDENT_CANDIDATE_ID = "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304"
# The Reference (stronger policy) identity is a controller designation —
# never self-invented.  It is pending this round and blocks the launch gate.
REFERENCE_CANDIDATE_ID = "PENDING_CONTROLLER_DESIGNATION"
OFFICIAL_MEMORY_MODES = ("SAVED_POLICY_MEMORY", "HISTORY_BURN_IN")
ABLATION_ONLY_MEMORY_MODE = "ZERO_MEMORY"
ANCHOR_MANIFEST_REF = "PENDING_CONTROLLER_SIGNED_SHARED_ANCHOR_MANIFEST"
CONFIG_SCHEMA = "simulator_frontier.e3-longrun-frozen-config/v1"
BUDGET_SEMANTICS_PENDING = "PENDING_DIRECTOR_BUDGET_DECISION"


def _log(msg: str) -> None:
    print(f"[e3-longrun] {msg}", flush=True)


def runtime_git_sha() -> str:
    """Read the worktree HEAD at runtime; UNAVAILABLE is honest, never faked."""
    try:
        proc = subprocess.run(
            ["git", "-c", f"safe.directory={REPO_ROOT}", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30)
        sha = (proc.stdout or "").strip()
        if proc.returncode == 0 and len(sha) == 40:
            return sha
    except Exception:
        pass
    return "UNAVAILABLE"


def frozen_config_payload(git_sha: str) -> dict:
    return {
        "schema": CONFIG_SCHEMA,
        "total_env_steps": TOTAL_ENV_STEPS,
        "num_envs": NUM_ENVS,
        "num_steps": NUM_STEPS,
        "env_steps_per_update": ENV_STEPS_PER_UPDATE,
        "num_updates": NUM_UPDATES,
        "requested_n_per_window": REQUESTED_N_PER_WINDOW,
        "search_horizon": SEARCH_HORIZON,
        "seed": SEED,
        "student_candidate_id": STUDENT_CANDIDATE_ID,
        "reference_candidate_id": REFERENCE_CANDIDATE_ID,
        "official_memory_modes": list(OFFICIAL_MEMORY_MODES),
        "ablation_only_memory_mode": ABLATION_ONLY_MEMORY_MODE,
        "anchor_manifest_ref": ANCHOR_MANIFEST_REF,
        # CC4 follow-up (P0-17): the training budget semantics are the
        # director's decision — never assumed.  Filled below from the minted
        # budget decision when one is bound.
        "training_budget_semantics": BUDGET_SEMANTICS_PENDING,
        "budget_decision_id": "",
        "budget_decision_hash": "",
        "longrun_gate_version": "",
        "git_sha": git_sha,
        "branch": "henry/simulator-frontier-foundation-codex",
        "launch_policy": ("PREPARE_ONLY_THIS_ROUND: launch requires a green "
                          "E3 preflight, the director's signed budget "
                          "decision and explicit external audit approval"),
    }


def config_hash_of(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    started = time.time()
    out_dir = str(DEFAULT_OUT_DIR)
    audit_approved = False
    budget_decision_path = None
    preflight_report_path = None
    for arg in argv:
        if arg.startswith("--out="):
            out_dir = arg.split("=", 1)[1]
        elif arg == "--audit-approved=true":
            audit_approved = True
        elif arg.startswith("--budget-decision="):
            budget_decision_path = arg.split("=", 1)[1]
        elif arg.startswith("--preflight-report="):
            preflight_report_path = arg.split("=", 1)[1]
        else:
            print(f"[e3-longrun] unknown argument: {arg!r}", flush=True)
            return FAIL

    try:
        from dicode.simulator_frontier.longrun_gate import (
            BUDGET_DECISION_SCHEMA,
            LONGRUN_GATE_VERSION,
            budget_decision_from_payload,
            evaluate_launch_blockers,
        )
        from dicode.simulator_frontier.errors import InvalidEvidenceError
    except Exception as exc:
        report = {
            "schema": "simulator_frontier.e3_longrun_entrypoint/v1",
            "REAL_LONG_RUN_STARTED": False,
            "frozen_config": {},
            "verdict": "BLOCKED",
            "reason": f"BLOCKED_ENVIRONMENT: {exc!r}",
        }
        _finish(report, out_dir)
        return BLOCKED

    # CC4 follow-up (P0-17): the launch gate is DYNAMIC — it is evaluated
    # from ACTUAL evidence: a real E3 preflight report (carried over
    # verbatim; a missing evaluation is itself a blocker), the Reference
    # designation state, the anchor manifest reference, the director's
    # signed budget decision and the audit approval flag.
    preflight_blockers = None
    if preflight_report_path:
        try:
            with open(preflight_report_path, "r", encoding="utf-8") as fh:
                pre_payload = json.load(fh)
            raw = pre_payload.get("blockers")
            if not isinstance(raw, list):
                raise ValueError("preflight report 'blockers' must be a list")
            preflight_blockers = tuple(str(b) for b in raw)
        except Exception as exc:
            report = {
                "schema": "simulator_frontier.e3_longrun_entrypoint/v1",
                "REAL_LONG_RUN_STARTED": False,
                "verdict": "FAIL",
                "reason": f"preflight report unreadable or invalid: {exc!r}",
            }
            _finish(report, out_dir)
            return FAIL

    budget_decision = None
    if budget_decision_path:
        try:
            with open(budget_decision_path, "r", encoding="utf-8") as fh:
                decision_payload = json.load(fh)
            budget_decision = budget_decision_from_payload(decision_payload)
        except (OSError, ValueError, InvalidEvidenceError) as exc:
            report = {
                "schema": "simulator_frontier.e3_longrun_entrypoint/v1",
                "REAL_LONG_RUN_STARTED": False,
                "verdict": "FAIL",
                "reason": f"budget decision invalid: {exc!r}",
            }
            _finish(report, out_dir)
            return FAIL

    git_sha = runtime_git_sha()
    payload = frozen_config_payload(git_sha)

    # The budget decision is cross-bound to the FROZEN total: a decision for
    # any other budget is a FAIL, not a silent re-interpretation.
    if budget_decision is not None:
        if int(budget_decision.total_env_steps) != TOTAL_ENV_STEPS:
            report = {
                "schema": "simulator_frontier.e3_longrun_entrypoint/v1",
                "frozen_config": payload,
                "REAL_LONG_RUN_STARTED": False,
                "verdict": "FAIL",
                "reason": (f"budget decision total_env_steps "
                           f"{int(budget_decision.total_env_steps)} != frozen "
                           f"TOTAL_ENV_STEPS {TOTAL_ENV_STEPS} (cross-binding "
                           "violation, fail closed)"),
            }
            _finish(report, out_dir)
            return FAIL
        payload["training_budget_semantics"] = budget_decision.budget_semantics
        payload["budget_decision_id"] = budget_decision.decision_id
        payload["budget_decision_hash"] = budget_decision.decision_hash
    payload["longrun_gate_version"] = LONGRUN_GATE_VERSION
    payload["config_hash"] = config_hash_of(payload)

    report = {
        "schema": "simulator_frontier.e3_longrun_entrypoint/v1",
        "frozen_config": payload,
        "audit_approved": bool(audit_approved),
        "budget_decision_bound": budget_decision is not None,
        "preflight_evaluated": preflight_blockers is not None,
        "REAL_LONG_RUN_STARTED": False,
        "disclaimers": [
            "this entrypoint only prepares the frozen run config",
            "the long run is NEVER started without external audit approval",
            "the launch blocker list is evaluated DYNAMICALLY from actual "
            "evidence (never a hardcoded list)",
            "ZERO_MEMORY is ablation-only and never the official memory mode",
        ],
    }

    # Fail-closed consistency check of the frozen arithmetic.
    if NUM_UPDATES * ENV_STEPS_PER_UPDATE != TOTAL_ENV_STEPS:
        report["verdict"] = "FAIL"
        report["reason"] = "frozen config arithmetic is inconsistent"
        _finish(report, out_dir)
        return FAIL

    blockers = evaluate_launch_blockers(
        preflight_blockers=preflight_blockers,
        reference_candidate_id=REFERENCE_CANDIDATE_ID,
        anchor_manifest_ref=ANCHOR_MANIFEST_REF,
        budget_decision=budget_decision,
        audit_approved=audit_approved,
    )
    report["launch_blockers"] = list(blockers)
    report["longrun_gate_version"] = LONGRUN_GATE_VERSION
    _log(f"frozen config_hash={payload['config_hash'][:16]}… "
         f"git_sha={git_sha[:12]}… blockers={len(blockers)}")

    if blockers:
        report["verdict"] = "BLOCKED"
        report["reason"] = "longrun launch blocked: " + "; ".join(blockers)
        _finish(report, out_dir)
        return BLOCKED

    # Reachable only when EVERY blocker is cleared AND audit approved.
    report["verdict"] = "PASS"
    report["reason"] = ("launch authorized — hand off to the window pipeline "
                        "driver (not executed by this entrypoint)")
    report["elapsed_s"] = round(time.time() - started, 2)
    _finish(report, out_dir)
    return PASS


def _finish(report: dict, out_dir: str) -> None:
    try:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        frozen_path = out / "e3_longrun_frozen_config.json"
        tmp = frozen_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(report["frozen_config"], fh, ensure_ascii=False, indent=2)
        os.replace(tmp, frozen_path)
        report_path = out / "e3_longrun_entrypoint.json"
        tmp2 = report_path.with_suffix(".json.tmp")
        with open(tmp2, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        os.replace(tmp2, report_path)
        print(f"[e3-longrun] frozen config: {frozen_path}", flush=True)
        print(f"[e3-longrun] report: {report_path}", flush=True)
    except Exception as exc:
        print(f"[e3-longrun] WRITE_FAILED: {exc!r}", flush=True)
    print(f"[e3-longrun] verdict={report.get('verdict')} "
          f"reason={report.get('reason', '-')}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
