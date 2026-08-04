#!/usr/bin/env python3
"""E3 long-run entrypoint (P0-8): config freeze ONLY, launch never happens here.

Freezes the complete long-run configuration — total_env_steps=98304, the
actual-N budget, search horizon, Student/Reference identities, the official
production memory modes (SAVED_POLICY_MEMORY / HISTORY_BURN_IN; ZERO_MEMORY
is ablation-only and can never be the official mode), the anchor manifest
reference, seed, the runtime Git SHA and the config hash — and writes it as
an immutable run config.

THIS ROUND PREPARES THE ENTRYPOINT ONLY: launching requires an explicit
external audit approval of every frozen parameter AND a green E3 preflight
(R9 training surface, controller-signed RegistryBundle, shared anchor
manifest, frozen formal asset registry, saved-policy-memory artifact,
authorized LLM client).  Without them this script exits 5 (BLOCKED) after
writing the frozen config — it NEVER starts the long run on its own.

Usage (venv python):

    python run_e3_longrun.py [--out=<DIR>] [--audit-approved=true]

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
        "git_sha": git_sha,
        "branch": "henry/simulator-frontier-foundation-codex",
        "launch_policy": ("PREPARE_ONLY_THIS_ROUND: launch requires explicit "
                          "external audit approval plus a green E3 preflight"),
    }


def config_hash_of(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def launch_blockers() -> list[str]:
    """Honest launch gate: every unresolved production dependency blocks."""
    blockers = [
        "BLOCKED_TRAINING_SURFACE_PENDING_R9",
        "BLOCKED_WAITING_CONTROLLER_SIGNED_REGISTRY_BUNDLE",
        "BLOCKED_SHARED_ANCHOR_MANIFEST",
        "BLOCKED_WAITING_FROZEN_FORMAL_ASSET_REGISTRY",
        "SAVED_POLICY_MEMORY_BLOCKED_NO_MEMORY_ARTIFACT",
        "REAL_TWO_LLM_BLOCKED_NO_AUTHORIZED_CLIENT",
        "BLOCKED_REFERENCE_IDENTITY_PENDING_CONTROLLER_DESIGNATION",
        "BLOCKED_AUDIT_APPROVAL_NOT_GRANTED",
    ]
    return blockers


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    started = time.time()
    out_dir = str(DEFAULT_OUT_DIR)
    audit_approved = False
    for arg in argv:
        if arg.startswith("--out="):
            out_dir = arg.split("=", 1)[1]
        elif arg == "--audit-approved=true":
            audit_approved = True
        else:
            print(f"[e3-longrun] unknown argument: {arg!r}", flush=True)
            return FAIL

    git_sha = runtime_git_sha()
    payload = frozen_config_payload(git_sha)
    payload["config_hash"] = config_hash_of(payload)

    report = {
        "schema": "simulator_frontier.e3_longrun_entrypoint/v1",
        "frozen_config": payload,
        "audit_approved": bool(audit_approved),
        "REAL_LONG_RUN_STARTED": False,
        "disclaimers": [
            "this entrypoint only prepares the frozen run config",
            "the long run is NEVER started without external audit approval",
            "ZERO_MEMORY is ablation-only and never the official memory mode",
        ],
    }

    # Fail-closed consistency check of the frozen arithmetic.
    if NUM_UPDATES * ENV_STEPS_PER_UPDATE != TOTAL_ENV_STEPS:
        report["verdict"] = "FAIL"
        report["reason"] = "frozen config arithmetic is inconsistent"
        _finish(report, out_dir)
        return FAIL

    blockers = launch_blockers()
    report["launch_blockers"] = blockers
    _log(f"frozen config_hash={payload['config_hash'][:16]}… "
         f"git_sha={git_sha[:12]}… blockers={len(blockers)}")

    if blockers or not audit_approved:
        report["verdict"] = "BLOCKED"
        report["reason"] = (
            "longrun launch blocked: " + "; ".join(blockers)
            + ("" if audit_approved else "; audit approval not granted"))
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
