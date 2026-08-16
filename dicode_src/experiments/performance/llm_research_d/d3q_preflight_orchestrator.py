"""D3Q GPU2 preflight orchestrator (local side).

``prepare``: aggregate each model/repeat arm's unique final-valid candidate
codes from the published phase-2 chunk artifacts into a staging directory
(fail closed on missing slots, except slots lost to incident 01 which are
recorded as reconciled losses).

``run``: deploy ``preflight_replay.py`` + ``d3q_preflight_remote.py`` + the
staged arms to a fresh remote exec root, execute the remote driver (strictly
sequential arms on GPU2 behind UUID/PID gates), collect every arm's evidence,
verify cleanup, and publish a local artifact directory with SHA256SUMS.

No LLM/API calls; no secret material is read or written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import d3q_slot_runner as runner_mod  # noqa: E402
from d3q_phase2_driver import (  # noqa: E402
    RECONCILIATION_FILENAME,
    _load_reconciliation,
    all_slots_ordered,
)

SSH_TARGET_DEFAULT = "oseasy@172.25.14.221"
SSH_KEY_DEFAULT = r"D:\Projects\dicode-codex-director\orchestration\control\ssh_oseasy_172_25_14_221_ed25519"
REMOTE_PYTHON = "/home/oseasy/miniconda3/envs/dicode310/bin/python"
# Source tree for preflight replay provenance + runtime imports: the FROZEN
# dicode_src snapshot captured with the B1 reference run (source_commit
# 4d1f54f; replay script sha256 7e431e8c...d53 identical to this repo's copy).
# The baseline worktree wt_d3q_mason_91a75e5 lacks
# skill_preflight/preflight_route.py -> all arms FAILED at 20260816T063130Z.
MASON_SRC = "/home/oseasy/e2_data_disk2/skill_preflight_runs/perf48_b1r2_gpu2_20260813T032611Z/dicode_src/src"
GPU2_UUID = "GPU-8df11537-ab79-722d-606f-411966196c4c"
OLLAMA_QWEN_DIGEST_PREFIX = "9ec8897f747e"
EXEC_ROOT_RE = re.compile(r"^/tmp/d3q_preflight_\d{8}T\d{6}Z$")
SSH_OPTS = ("-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=yes")
# incident-04 hardening: the remote driver runs detached (setsid) so an ssh
# disconnect can neither kill the run nor hide it; short ssh probes poll the
# state; cleanup never removes an exec root with live driver/replay processes.
POLL_INTERVAL_S = 60
MAX_RUN_S = 6 * 3600
MAX_CONSECUTIVE_POLL_FAILURES = 10
# incident-05 recovery (mirrors the phase-2 recover-completed-chunk precedent):
# only these post-run gate reasons may be recovered after the fact.
ALLOWED_RECOVERY_REASONS = ("gpu2_external_app",)
RECOVERY_EXECUTE_RATIO_LIMIT = 1.25


class OrchestratorError(RuntimeError):
    def __init__(self, reason: str, detail: Any = None):
        self.reason = reason
        self.detail = detail
        super().__init__(reason if detail is None else f"{reason}: {detail}")


def _ssh_argv(ssh_target: str, ssh_key: str, remote_cmd: Sequence[str]) -> List[str]:
    return ["ssh", "-i", ssh_key, *SSH_OPTS, ssh_target, " ".join(remote_cmd)]


def _run_local(argv: Sequence[str], timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(list(argv), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, check=False)


def _remote_or_fail(ssh_target: str, ssh_key: str, remote_cmd: Sequence[str], timeout: int = 600) -> str:
    proc = _run_local(_ssh_argv(ssh_target, ssh_key, remote_cmd), timeout=timeout)
    if proc.returncode != 0:
        raise OrchestratorError("remote_command_failed", {"cmd": list(remote_cmd), "rc": proc.returncode, "stderr_tail": proc.stderr[-500:]})
    return proc.stdout


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------


def _arm_id(arm: str, repeat: str) -> str:
    return f"{arm}_{repeat}"


def cmd_prepare(artifact_dirs: Sequence[Path], staging_dir: Path, artifacts_root: Path) -> Dict[str, Any]:
    artifact_dirs = [Path(p) for p in artifact_dirs]
    staging_dir = Path(staging_dir)
    if staging_dir.exists():
        raise OrchestratorError("staging_exists", str(staging_dir))
    reconciliation = _load_reconciliation(Path(artifacts_root))
    reconciled_slots = set(reconciliation["slot_consumed"])

    def locate(slot_id: str) -> Optional[Path]:
        for artifact in artifact_dirs:
            slot_dir = artifact / "slots" / slot_id
            if (slot_dir / f"{slot_id}.result.json").is_file():
                return slot_dir
        return None

    plan_arms = []
    for arm, repeat in runner_mod.ARM_ORDER:
        prefix = f"slot_{repeat}_{arm}_"
        slots = [s for s in all_slots_ordered() if s.startswith(prefix)]
        arm_dir = staging_dir / "arms" / _arm_id(arm, repeat)
        candidates_dir = arm_dir / "candidates"
        candidates: List[Dict[str, Any]] = []
        seen_sha: Dict[str, str] = {}
        slots_final_valid, slots_final_invalid, slots_lost = [], [], []
        scanned = []
        for slot_id in slots:
            if slot_id in reconciled_slots:
                slots_lost.append(slot_id)
                continue
            slot_dir = locate(slot_id)
            if slot_dir is None:
                raise OrchestratorError("slot_artifact_missing", slot_id)
            scanned.append(slot_id)
            result = json.loads((slot_dir / f"{slot_id}.result.json").read_text(encoding="utf-8"))
            if not result.get("final_valid"):
                slots_final_invalid.append(slot_id)
                continue
            code_path = slot_dir / "final_code.py"
            if not code_path.is_file():
                raise OrchestratorError("final_code_missing", slot_id)
            code_sha = _sha256_file(code_path)
            if code_sha in seen_sha:
                for cand in candidates:
                    if cand["code_sha256"] == code_sha:
                        cand["aliased_slots"].append(slot_id)
                slots_final_valid.append(slot_id)
                continue
            candidate_id = slot_id
            target = candidates_dir / f"{candidate_id}.py"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(code_path, target)
            candidates.append({
                "id": candidate_id,
                "path": f"candidates/{candidate_id}.py",
                "code_sha256": code_sha,
                "source_slot": slot_id,
                "aliased_slots": [],
            })
            seen_sha[code_sha] = candidate_id
            slots_final_valid.append(slot_id)
        provider, model, _url = runner_mod.arm_to_provider_model(arm)
        arm_meta = {
            "classification": "D3Q_PREFLIGHT_ARM_CANDIDATES",
            "schema_version": 1,
            "arm_id": _arm_id(arm, repeat),
            "arm": arm,
            "repeat": repeat,
            "provider": provider,
            "model": model,
            "slots_scanned": scanned,
            "slots_lost_reconciled": slots_lost,
            "slots_final_valid": slots_final_valid,
            "slots_final_invalid": slots_final_invalid,
            "candidates": candidates,
        }
        (arm_dir).mkdir(parents=True, exist_ok=True)
        (arm_dir / "ARM_CANDIDATES.json").write_text(json.dumps(arm_meta, indent=2) + "\n", encoding="utf-8")
        plan_arms.append({
            "arm_id": arm_meta["arm_id"],
            "candidate_count": len(candidates),
            "final_valid_slots": len(slots_final_valid),
            "final_invalid_slots": len(slots_final_invalid),
            "lost_slots": len(slots_lost),
        })
    staging_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "classification": "D3Q_PREFLIGHT_PLAN",
        "schema_version": 1,
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifact_dirs": [str(p) for p in artifact_dirs],
        "arms": plan_arms,
        "execution_order": [a["arm_id"] for a in plan_arms],
    }
    (staging_dir / "PREFLIGHT_PLAN.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return plan


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def _gpu_gate_remote(ssh_target: str, ssh_key: str) -> Dict[str, Any]:
    out = _remote_or_fail(ssh_target, ssh_key, [
        "nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader;",
        "nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader",
    ])
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    gpu2 = None
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and parts[0] == "2":
            gpu2 = {"uuid": parts[1], "memory_used_mib": parts[2]}
    external = [ln for ln in lines if ln.startswith(GPU2_UUID)]
    if gpu2 is None or gpu2["uuid"] != GPU2_UUID:
        raise OrchestratorError("gpu2_uuid_mismatch", gpu2)
    if external:
        raise OrchestratorError("gpu2_external_app", external)
    return {"gpu2": gpu2, "external": external}


def _ollama_gate_remote(ssh_target: str, ssh_key: str) -> Dict[str, Any]:
    out = _remote_or_fail(ssh_target, ssh_key, ["curl -s http://127.0.0.1:11434/api/ps"])
    data = json.loads(out)
    for model in data.get("models", []):
        if model.get("name", "").startswith(runner_mod.SMALL_MODEL):
            digest = model.get("details", {}).get("digest") or model.get("digest") or ""
            if digest.startswith(OLLAMA_QWEN_DIGEST_PREFIX):
                return {"qwen_digest": digest}
    raise OrchestratorError("ollama_digest_changed", out[:300])


# ---------------------------------------------------------------------------
# remote driver launch / poll (incident-04: survives ssh drops)
# ---------------------------------------------------------------------------


def _poll_probe_cmd(exec_root: str) -> str:
    # The [.] bracket trick keeps pgrep from matching the probe shell itself.
    return (
        f"if test -f {exec_root}/driver.rc; then echo \"DONE rc=$(cat {exec_root}/driver.rc)\"; "
        f"elif pgrep -f \"d3q_preflight_remote[.]py --exec-root {exec_root}\" > /dev/null; then echo RUNNING; "
        f"else echo DEAD; fi"
    )


def _classify_poll(probe: str) -> tuple[str, Optional[int]]:
    text = probe.strip()
    if text.startswith("DONE rc="):
        rc_text = text.split("=", 1)[1].strip()
        if rc_text.lstrip("-").isdigit():
            return "DONE", int(rc_text)
        return "UNKNOWN", None
    if text == "RUNNING":
        return "RUNNING", None
    if text == "DEAD":
        return "DEAD", None
    return "UNKNOWN", None


def _live_exec_root_procs(ssh_target: str, ssh_key: str, exec_root: str) -> str:
    probe = (
        f"pgrep -f \"d3q_preflight_remote[.]py --exec-root {exec_root}\"; "
        f"pgrep -f \"preflight_replay[.]py --spec {exec_root}\"; true"
    )
    return _run_local(_ssh_argv(ssh_target, ssh_key, [probe]), timeout=120).stdout.strip()


def _collect_remote_exec_root(ssh_target: str, ssh_key: str, exec_root: str, local_artifacts_dir: Path) -> None:
    tar_out = subprocess.Popen(_ssh_argv(ssh_target, ssh_key, [f"tar czf - -C {exec_root} ."]), stdout=subprocess.PIPE)
    untar = subprocess.Popen(["tar", "xzf", "-", "-C", str(local_artifacts_dir)], stdin=tar_out.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    tar_out.stdout.close()
    _o, e2 = untar.communicate(timeout=900)
    if untar.returncode != 0 or tar_out.wait() != 0:
        raise OrchestratorError("collect_failed", e2.decode("utf-8", "replace")[-500:])


def cmd_run(staging_dir: Path, ssh_target: str, ssh_key: str, local_artifacts_dir: Path) -> Dict[str, Any]:
    staging_dir = Path(staging_dir)
    plan = json.loads((staging_dir / "PREFLIGHT_PLAN.json").read_text(encoding="utf-8"))
    run_id = "d3q_preflight_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    exec_root = f"/tmp/{run_id}"
    if not EXEC_ROOT_RE.fullmatch(exec_root):
        raise OrchestratorError("exec_root_invalid", exec_root)
    local_artifacts_dir = Path(local_artifacts_dir)
    if local_artifacts_dir.exists():
        raise OrchestratorError("local_output_exists", str(local_artifacts_dir))

    gpu_pre = _gpu_gate_remote(ssh_target, ssh_key)
    ollama_pre = _ollama_gate_remote(ssh_target, ssh_key)
    _remote_or_fail(ssh_target, ssh_key, [f"test ! -e {exec_root} && mkdir -p {exec_root}"])
    try:
        # deploy scripts
        for name in ("preflight_replay.py", "d3q_preflight_remote.py"):
            source = (HERE.parent / name) if name == "preflight_replay.py" else (HERE / name)
            proc = _run_local(["scp", "-i", ssh_key, *SSH_OPTS, str(source), f"{ssh_target}:{exec_root}/{name}"], timeout=300)
            if proc.returncode != 0:
                raise OrchestratorError("deploy_failed", name)
        # deploy arms as tar stream
        tar = subprocess.Popen(["tar", "czf", "-", "-C", str(staging_dir), "arms"], stdout=subprocess.PIPE)
        remote_untar = subprocess.Popen(_ssh_argv(ssh_target, ssh_key, [f"tar xzf - -C {exec_root}"]), stdin=tar.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        tar.stdout.close()
        _out, err = remote_untar.communicate(timeout=600)
        if remote_untar.returncode != 0 or tar.wait() != 0:
            raise OrchestratorError("deploy_arms_failed", err.decode("utf-8", "replace")[-500:])
        verify = _remote_or_fail(ssh_target, ssh_key, [f"ls {exec_root}/arms | wc -l; test -f {exec_root}/preflight_replay.py && echo REPLAY_OK; test -f {exec_root}/d3q_preflight_remote.py && echo DRIVER_OK"])
        if "REPLAY_OK" not in verify or "DRIVER_OK" not in verify:
            raise OrchestratorError("deploy_verify_failed", verify)
        # launch the remote driver DETACHED: a dropped ssh connection must
        # neither kill the run nor blind us to it (incident-04). Short ssh
        # probes poll driver.rc / liveness; every probe is logged.
        local_artifacts_dir.mkdir(parents=True, exist_ok=False)
        poll_log = local_artifacts_dir / "poll_log.txt"
        launch_cmd = (
            f"cd {exec_root} && {{ setsid bash -c '{REMOTE_PYTHON} d3q_preflight_remote.py"
            f" --exec-root {exec_root} --mason-src {MASON_SRC}"
            f" --replay-script {exec_root}/preflight_replay.py"
            f" > remote_driver_stdout.txt 2> remote_driver_stderr.txt;"
            f" echo $? > driver.rc' < /dev/null > /dev/null 2>&1 & echo LAUNCHED; }}"
        )
        launched = _remote_or_fail(ssh_target, ssh_key, [launch_cmd], timeout=120)
        if "LAUNCHED" not in launched:
            raise OrchestratorError("driver_launch_failed", launched[-300:])
        deadline = time.monotonic() + MAX_RUN_S
        consecutive_failures = 0
        remote_rc: Optional[int] = None
        while True:
            if time.monotonic() >= deadline:
                raise OrchestratorError("run_timeout", {"exec_root": exec_root, "limit_s": MAX_RUN_S})
            time.sleep(POLL_INTERVAL_S)
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            try:
                probe_out = _remote_or_fail(ssh_target, ssh_key, [_poll_probe_cmd(exec_root)], timeout=120)
                state, rc = _classify_poll(probe_out)
            except OrchestratorError as exc:
                state, rc = "SSH_ERROR", None
                probe_out = f"{exc.reason}: {exc.detail}"
            with poll_log.open("a", encoding="utf-8") as fh:
                fh.write(f"{stamp} state={state} rc={rc} probe={probe_out.strip()[:200]!r}\n")
            if state == "DONE":
                remote_rc = rc
                break
            if state == "DEAD":
                _collect_remote_exec_root(ssh_target, ssh_key, exec_root, local_artifacts_dir)
                raise OrchestratorError("remote_driver_died_without_result", exec_root)
            if state in ("SSH_ERROR", "UNKNOWN"):
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_POLL_FAILURES:
                    raise OrchestratorError("poll_failures_exceeded", {"exec_root": exec_root, "consecutive": consecutive_failures})
                continue
            consecutive_failures = 0
        (local_artifacts_dir / "remote_run_rc.txt").write_text(str(remote_rc) + "\n", encoding="utf-8")
        # collect everything back
        _collect_remote_exec_root(ssh_target, ssh_key, exec_root, local_artifacts_dir)
        summary_path = local_artifacts_dir / "D3Q_PREFLIGHT_REMOTE_SUMMARY.json"
        if not summary_path.is_file():
            raise OrchestratorError("remote_summary_missing", str(summary_path))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        status = "PASS" if (remote_rc == 0 and summary.get("status") == "PASS") else "FAILED"
        if remote_rc == 2:
            status = "BLOCKED"
        gpu_post = _gpu_gate_remote(ssh_target, ssh_key)
        ollama_post = _ollama_gate_remote(ssh_target, ssh_key)
        result = {
            "classification": "D3Q_PREFLIGHT_ORCHESTRATOR",
            "schema_version": 1,
            "run_id": run_id,
            "status": status,
            "remote_rc": remote_rc,
            "plan_execution_order": plan["execution_order"],
            "gpu_pre": gpu_pre, "gpu_post": gpu_post,
            "ollama_pre": ollama_pre, "ollama_post": ollama_post,
            "arms": summary.get("arms", []),
        }
        (local_artifacts_dir / "D3Q_PREFLIGHT_ORCHESTRATOR_RESULT.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    finally:
        if EXEC_ROOT_RE.fullmatch(exec_root):
            # incident-04 guard: never delete an exec root whose driver/replay
            # processes are still alive; fail closed and preserve evidence.
            live = _live_exec_root_procs(ssh_target, ssh_key, exec_root)
            if live:
                raise OrchestratorError(
                    "cleanup_skipped_live_processes",
                    {"exec_root": exec_root, "pids": live, "note": "remote driver/replay still running; exec root preserved"},
                )
            _run_local(_ssh_argv(ssh_target, ssh_key, [f"rm -rf {exec_root}"]), timeout=300)
            check = _run_local(_ssh_argv(ssh_target, ssh_key, [f"test ! -e {exec_root} && echo ABSENT"]))
            if "ABSENT" not in check.stdout:
                raise OrchestratorError("cleanup_failed", exec_root)


def main(argv: Optional[Sequence[str]] = None) -> int:
    return _main_impl(argv)


# ---------------------------------------------------------------------------
# incident-05 recovery: post-run gate blocked AFTER all arms completed
# ---------------------------------------------------------------------------


def _arm_execute_metrics(arm_dir: Path) -> tuple[float, int]:
    critical = json.loads((arm_dir / "run" / "critical_path.json").read_text(encoding="utf-8"))
    phases = [p for p in critical.get("critical_path", []) if p.get("phase") == "preflight_eval_execute"]
    if len(phases) != 1:
        raise OrchestratorError("recovery_execute_phase_missing", arm_dir.name)
    meta = json.loads((arm_dir / "ARM_CANDIDATES.json").read_text(encoding="utf-8"))
    n = len(meta.get("candidates", []))
    if n <= 0:
        raise OrchestratorError("recovery_candidates_missing", arm_dir.name)
    return float(phases[0]["duration_s"]), n


def _interference_check(arms_root: Path, arm_ids: Sequence[str]) -> Dict[str, Any]:
    """Fail closed unless every arm's GPU execute time is within
    RECOVERY_EXECUTE_RATIO_LIMIT of the MEDIAN ABSOLUTE execute of the other
    arms. A co-resident GPU context that stole compute would inflate the
    overlapped arm(s). Absolute comparison is used because execute has a large
    fixed cost (~190s for the 40-update rollout): incident-05 data showed
    6-candidate arms taking the same absolute execute as 12-candidate arms,
    which invalidates per-candidate normalization."""
    metrics = {a: _arm_execute_metrics(arms_root / a) for a in arm_ids}
    checks: Dict[str, Any] = {}
    for arm in arm_ids:
        dur, n = metrics[arm]
        others = sorted(metrics[b][0] for b in arm_ids if b != arm)
        median = others[len(others) // 2]
        ratio = dur / median if median > 0 else float("inf")
        checks[arm] = {
            "execute_s": dur, "candidates": n,
            "median_others_execute_s": median, "ratio_vs_median": ratio,
            "ok": ratio <= RECOVERY_EXECUTE_RATIO_LIMIT,
        }
    if not all(c["ok"] for c in checks.values()):
        raise OrchestratorError("recovery_interference_check_failed", checks)
    return checks


def cmd_recover_completed_run(artifact_dir: Path, reason: str, external_pid: Optional[int], incident_detail: str, ssh_target: str, ssh_key: str) -> Dict[str, Any]:
    artifact_dir = Path(artifact_dir)
    if reason not in ALLOWED_RECOVERY_REASONS:
        raise OrchestratorError("recovery_reason_not_allowed", reason)
    result_path = artifact_dir / "D3Q_PREFLIGHT_ORCHESTRATOR_RESULT.json"
    if result_path.exists():
        raise OrchestratorError("recovery_result_exists", str(result_path))
    summary_path = artifact_dir / "D3Q_PREFLIGHT_REMOTE_SUMMARY.json"
    if not summary_path.is_file():
        raise OrchestratorError("recovery_summary_missing", str(artifact_dir))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS":
        raise OrchestratorError("recovery_summary_not_pass", summary.get("status"))
    rc_path = artifact_dir / "remote_run_rc.txt"
    driver_rc_path = artifact_dir / "driver.rc"
    rc_text = rc_path.read_text(encoding="utf-8").strip() if rc_path.is_file() else ""
    driver_rc = driver_rc_path.read_text(encoding="utf-8").strip() if driver_rc_path.is_file() else ""
    if rc_text != "0" or driver_rc != "0":
        raise OrchestratorError("recovery_rc_not_zero", {"remote_run_rc": rc_text, "driver_rc": driver_rc})
    arms_root = artifact_dir / "arms"
    evidence: Dict[str, str] = {}
    for arm in summary.get("arms", []):
        arm_id = arm.get("arm_id")
        if arm.get("status") != "PASS":
            raise OrchestratorError("recovery_arm_not_pass", arm_id)
        arm_dir = arms_root / arm_id
        for rel in ("run/RESULT.json", "run/replay_summary.json", "run/critical_path.json", "spec.json", "manifest.json"):
            p = arm_dir / rel
            if not p.is_file():
                raise OrchestratorError("recovery_evidence_missing", f"{arm_id}/{rel}")
            evidence[f"{arm_id}/{rel}"] = _sha256_file(p)
    arm_ids = [a["arm_id"] for a in summary.get("arms", [])]
    interference = _interference_check(arms_root, arm_ids)
    verified_gone = False
    if external_pid is not None:
        proc = _run_local(_ssh_argv(ssh_target, ssh_key, [f"ps -p {int(external_pid)} > /dev/null 2>&1 && echo ALIVE || echo GONE"]), timeout=120)
        if proc.returncode != 0 or "GONE" not in proc.stdout:
            raise OrchestratorError("recovery_external_process_still_alive", external_pid)
        verified_gone = True
    gpu_post = _gpu_gate_remote(ssh_target, ssh_key)
    ollama_post = _ollama_gate_remote(ssh_target, ssh_key)
    recovered_utc = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    recovery = {
        "classification": "D3Q_PREFLIGHT_RECOVERY",
        "schema_version": 1,
        "run_id": artifact_dir.name,
        "reason": reason,
        "allowed_reasons": list(ALLOWED_RECOVERY_REASONS),
        "precedent": "D3Q_PHASE2_INCIDENT_02 recover-completed-chunk (gpu2_external_app)",
        "external_process": {"pid": external_pid, "detail": incident_detail, "verified_gone": verified_gone},
        "interference_check": {"limit": RECOVERY_EXECUTE_RATIO_LIMIT, "arms": interference},
        "recovered_utc": recovered_utc,
        "note": "post-run GPU gate blocked after ALL arms completed and were collected; pre-run gates passed at launch (gpu/ollama pre-checks were not persisted because the block preceded result write)",
    }
    result = {
        "classification": "D3Q_PREFLIGHT_ORCHESTRATOR",
        "schema_version": 1,
        "run_id": artifact_dir.name,
        "status": "PASS",
        "remote_rc": 0,
        "plan_execution_order": arm_ids,
        "gpu_pre": {"note": "passed at launch; not persisted (incident-05 block preceded result write)"},
        "gpu_post": gpu_post,
        "ollama_pre": {"note": "passed at launch; not persisted (incident-05 block preceded result write)"},
        "ollama_post": ollama_post,
        "arms": summary.get("arms", []),
        "recovery": recovery,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (artifact_dir / "D3Q_PREFLIGHT_RECOVERY.json").write_text(
        json.dumps({"recovery": recovery, "evidence_sha256": evidence}, indent=2) + "\n", encoding="utf-8"
    )
    return result


def _main_impl(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="d3q_preflight_orchestrator")
    sub = parser.add_subparsers(dest="command", required=True)
    p_prep = sub.add_parser("prepare")
    p_prep.add_argument("--artifact-dir", action="append", required=True)
    p_prep.add_argument("--staging", required=True)
    p_prep.add_argument("--artifacts-root", default=str(HERE / "d3q_artifacts"))
    p_run = sub.add_parser("run")
    p_run.add_argument("--staging", required=True)
    p_run.add_argument("--ssh-target", default=SSH_TARGET_DEFAULT)
    p_run.add_argument("--ssh-key", default=SSH_KEY_DEFAULT)
    p_run.add_argument("--out", required=True)
    p_rec = sub.add_parser("recover-completed-run")
    p_rec.add_argument("--artifact-dir", required=True)
    p_rec.add_argument("--reason", required=True)
    p_rec.add_argument("--external-pid", type=int, default=None)
    p_rec.add_argument("--incident-detail", default="")
    p_rec.add_argument("--ssh-target", default=SSH_TARGET_DEFAULT)
    p_rec.add_argument("--ssh-key", default=SSH_KEY_DEFAULT)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "prepare":
            result = cmd_prepare([Path(p) for p in args.artifact_dir], Path(args.staging), Path(args.artifacts_root))
        elif args.command == "run":
            result = cmd_run(Path(args.staging), args.ssh_target, args.ssh_key, Path(args.out))
        else:
            result = cmd_recover_completed_run(
                Path(args.artifact_dir), args.reason, args.external_pid,
                args.incident_detail, args.ssh_target, args.ssh_key,
            )
    except OrchestratorError as exc:
        print(json.dumps({"status": "BLOCKED", "reason": exc.reason, "detail": exc.detail}, sort_keys=True, default=str))
        return 2
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
