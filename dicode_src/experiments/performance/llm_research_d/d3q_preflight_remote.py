"""D3Q GPU2 preflight remote driver (runs ON the server, inside one exec root).

For each arm directory (arms/<arm_id>/) prepared by the local orchestrator:

  1. GPU2 gate: UUID match + no external compute PIDs (fail closed).
  2. Copy the frozen Mason attempt06 step-2100 archive snapshot (read-only
     source) and inject the arm's candidate nodes (node attr ``code``).
  3. Write the deterministic conditioning table (zeros, [N+1, 67], float32;
     the frozen config uses ``conditioning_type: one_hot`` and the replay
     chain never consumes this table's values - it is evidence metadata).
  4. Write spec.json bound to the frozen checkpoint/config/rng and the mason
     worktree source files, then run the real production chain through
     ``preflight_replay.py`` (TaskArchive -> load_tasks_from_env_codes ->
     evaluate_new_tasks -> calculate_scores_from_snapshot -> preflight_route).
  5. Collect RESULT.json per arm; any failure writes FAILURE.json and the
     driver exits non-zero (fail closed).  Arms run strictly sequentially.

No network access, no LLM/API calls, no secret material.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

GPU2_UUID = "GPU-8df11537-ab79-722d-606f-411966196c4c"
# B1 fix commit that produced the frozen checkpoint/snapshot and was used by
# the original B1 replay (spec.json source_commit); matches the frozen
# dicode_src snapshot the orchestrator points MASON_SRC at.
SOURCE_COMMIT = "4d1f54fd32223ec0d51b38a64a3e6902d334c3c3"
CONDITIONING_DIM = 67
NUM_ENVS = 1024
NUM_STEPS = 128
ROLLOUT_UPDATES = 40
GLOBAL_STEP = 2100
SCORE_FUNCTION = "learnability"

FROZEN_ROOT = "/home/oseasy/e2_data_disk2/skill_preflight_runs/perf48_b1r2_gpu2_20260813T032611Z/frozen"
FROZEN_CHECKPOINT = FROZEN_ROOT + "/checkpoint/2100"
FROZEN_CONFIG = FROZEN_ROOT + "/config.yaml"
FROZEN_RNG = FROZEN_ROOT + "/rng.npy"
FROZEN_ARCHIVE = FROZEN_ROOT + "/archive_snapshot"

SOURCE_FILES = (
    ("gen_manager.py", "dreaming/gen_manager.py"),
    ("task_utils.py", "task_utils.py"),
    ("online_evaluation.py", "evaluation/online_evaluation.py"),
    ("ppo_tr.py", "ppo_tr.py"),
    ("scoring.py", "scoring.py"),
    ("setup.py", "setup.py"),
    ("preflight.py", "skill_preflight/preflight.py"),
    ("preflight_route.py", "skill_preflight/preflight_route.py"),
)


class PreflightDriverError(RuntimeError):
    pass


def gpu_gate() -> dict:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            gpus.append({"index": int(parts[0]), "uuid": parts[1], "memory_used_mib": int(parts[2].replace("MiB", "").strip())})
    gpu2 = next((g for g in gpus if g["index"] == 2), None)
    if gpu2 is None or gpu2["uuid"] != GPU2_UUID:
        raise PreflightDriverError("gpu2_uuid_mismatch")
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout
    external = []
    for line in apps.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] == GPU2_UUID:
            external.append(parts[1])
    if external:
        raise PreflightDriverError("gpu2_external_app")
    return {"gpu2": gpu2, "external_pids": external}


def inject_candidates(graphml_src: Path, archive_copy_dir: Path, candidates: list, arm_dir: Path) -> None:
    import networkx as nx

    graph = nx.read_graphml(graphml_src)
    for cand in candidates:
        cand_path = Path(cand["path"])
        if not cand_path.is_absolute():
            cand_path = Path(arm_dir) / cand_path
        code = cand_path.read_text(encoding="utf-8")
        graph.add_node(
            cand["id"],
            status="desc_generated",
            type="generated",
            description="D3Q candidate " + cand["id"],
            code=code,
            performance_history="[]",
            session_created="0",
            is_active="false",
            priority_score="0.0",
            session_last_trained="-1",
        )
    archive_copy_dir.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, archive_copy_dir / "task_graph.graphml")


def write_conditioning(arm_dir: Path, count: int) -> Path:
    import numpy as np

    table = np.zeros((count + 1, CONDITIONING_DIM), dtype=np.float32)
    path = arm_dir / "conditioning.npy"
    np.save(path, table, allow_pickle=False)
    return path


def run_arm(arm_dir: Path, mason_src: str, replay_script: Path) -> dict:
    arm_meta = json.loads((arm_dir / "ARM_CANDIDATES.json").read_text(encoding="utf-8"))
    candidates = arm_meta["candidates"]
    arm_id = arm_meta["arm_id"]
    if not candidates:
        return {"arm_id": arm_id, "status": "NO_CANDIDATES", "accepted": [], "note": "accepted=0_by_construction"}
    gate = gpu_gate()
    archive_copy = arm_dir / "archive_snapshot"
    if archive_copy.exists():
        raise PreflightDriverError("arm_dir_not_fresh")
    inject_candidates(Path(_find_graphml(FROZEN_ARCHIVE)), archive_copy, candidates, arm_dir)
    cond_path = write_conditioning(arm_dir, len(candidates))
    spec = {
        "classification": "PREFLIGHT_CANDIDATE_REPLAY",
        "global_step": GLOBAL_STEP,
        "rollout_updates": ROLLOUT_UPDATES,
        "score_function": SCORE_FUNCTION,
        "source_commit": SOURCE_COMMIT,
        "gpu_uuid": GPU2_UUID,
        "base_dir": str(arm_dir),
        "checkpoint": FROZEN_CHECKPOINT,
        "conditioning_path": str(cond_path),
        "archive_snapshot": str(archive_copy),
        "config_path": FROZEN_CONFIG,
        "rng_path": FROZEN_RNG,
        "candidate_codes": {c["id"]: c["path"] for c in candidates},
        "source_mapping": {name: f"{mason_src}/dicode/{rel}" for name, rel in SOURCE_FILES},
        "num_envs": NUM_ENVS,
        "num_steps": NUM_STEPS,
    }
    spec_path = arm_dir / "spec.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    manifest_path = arm_dir / "manifest.json"
    out_dir = arm_dir / "run"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = GPU2_UUID
    env["PYTHONPATH"] = mason_src
    env["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
    proc = subprocess.run(
        [
            sys.executable, str(replay_script),
            "--spec", str(spec_path),
            "--output", str(manifest_path),
            "--run",
            "--out-dir", str(out_dir),
        ],
        capture_output=True, text=True, env=env, cwd=str(arm_dir), check=False,
    )
    (arm_dir / "replay_stdout.txt").write_text(proc.stdout[-100000:], encoding="utf-8")
    (arm_dir / "replay_stderr.txt").write_text(proc.stderr[-100000:], encoding="utf-8")
    result_path = out_dir / "RESULT.json"
    if proc.returncode != 0 or not result_path.is_file():
        return {
            "arm_id": arm_id, "status": "FAILED", "rc": proc.returncode,
            "stderr_tail": proc.stderr[-2000:],
        }
    result = json.loads(result_path.read_text(encoding="utf-8"))
    return {
        "arm_id": arm_id,
        "status": "PASS",
        "gpu_gate": gate,
        "accepted": result.get("accepted_ids", []),
        "rejected": result.get("rejected_ids", []),
        "result_sha256": result.get("result_sha256"),
        "result_file": str(result_path),
    }


def _find_graphml(archive_dir: str) -> str:
    root = Path(archive_dir)
    matches = sorted(root.rglob("*.graphml"))
    if not matches:
        raise PreflightDriverError("frozen_archive_graphml_missing")
    return str(matches[0])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exec-root", required=True)
    parser.add_argument("--mason-src", required=True)
    parser.add_argument("--replay-script", required=True)
    args = parser.parse_args(argv)
    exec_root = Path(args.exec_root)
    arms_root = exec_root / "arms"
    arm_dirs = sorted(p for p in arms_root.iterdir() if p.is_dir())
    summary = {"classification": "D3Q_PREFLIGHT_REMOTE_SUMMARY", "arms": []}
    try:
        for arm_dir in arm_dirs:
            summary["arms"].append(run_arm(arm_dir, args.mason_src, Path(args.replay_script)))
    except PreflightDriverError as exc:
        summary["status"] = "BLOCKED"
        summary["reason"] = str(exc)
        (exec_root / "D3Q_PREFLIGHT_REMOTE_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}))
        return 2
    failed = [a for a in summary["arms"] if a["status"] == "FAILED"]
    summary["status"] = "PASS" if not failed else "FAILED"
    (exec_root / "D3Q_PREFLIGHT_REMOTE_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "arms": len(summary["arms"]), "failed": len(failed)}))
    return 0 if not failed else 3


if __name__ == "__main__":
    raise SystemExit(main())
