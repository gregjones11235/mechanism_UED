#!/usr/bin/env python3
"""Evaluate an E3 SlowGRU canonical RunState on Original Craftax only."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


EXPECTED_GPU_INDEX = "1"
EXPECTED_GPU_UUID = "GPU-3c7a2864-755b-7045-b293-6f80e748283f"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runstate-stem", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--num-steps", type=int, default=8192)
    parser.add_argument("--cuda-visible-devices", default=EXPECTED_GPU_INDEX)
    return parser.parse_args()


def _gpu_uuid(index: str) -> str:
    proc = subprocess.run(
        ["nvidia-smi", f"--id={index}", "--query-gpu=uuid",
         "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
    if proc.returncode != 0:
        raise RuntimeError(f"GPU query failed: {proc.stderr.strip()}")
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"GPU query returned {lines!r}")
    return lines[0]


def main() -> int:
    args = _parse_args()
    if args.cuda_visible_devices != EXPECTED_GPU_INDEX:
        raise RuntimeError("production evaluation is restricted to physical GPU1")
    already = os.environ.get("CUDA_VISIBLE_DEVICES")
    if already not in (None, "", EXPECTED_GPU_INDEX):
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={already!r} conflicts with required GPU1")
    actual_uuid = _gpu_uuid(EXPECTED_GPU_INDEX)
    if actual_uuid != EXPECTED_GPU_UUID:
        raise RuntimeError(
            f"physical GPU1 UUID {actual_uuid!r} != {EXPECTED_GPU_UUID!r}")

    os.environ["CUDA_VISIBLE_DEVICES"] = EXPECTED_GPU_INDEX
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_SILENT"] = "true"
    # No LLM client is imported on this path.  Remove provider selection to
    # make accidental provider routing fail closed if a future import drifts.
    os.environ.pop("E3_LLM_PROVIDER", None)

    script_dir = Path(__file__).resolve().parent
    siege_root = script_dir.parent
    src_dir = siege_root / "src"
    for path in (str(src_dir), str(script_dir)):
        if path not in sys.path:
            sys.path.insert(0, path)

    from dicode.simulator_frontier.e3_slowgru_original_eval import (
        SCHEMA,
        atomic_write_new_json,
        load_evaluation_runstate,
        original_task_protocol,
        pytree_sha256,
        run_original_task_rollout,
        sha256_file,
    )
    import run_e3_real_smoke as production_mount

    out = Path(args.out)
    if out.exists():
        raise RuntimeError(f"output already exists: {out}")
    restored = load_evaluation_runstate(args.runstate_stem)
    state, metadata = restored["state"], restored["metadata"]
    params_before = pytree_sha256(state["params"])
    optimizer_before = pytree_sha256(state["opt_state"])

    mount = production_mount.mount_student(state["candidate_id"])
    if mount["architecture_family"] != "SLOWGRU":
        raise RuntimeError("production mount is not SlowGRU")
    results = run_original_task_rollout(
        adapter=mount["adapter"], params=state["params"], seed=args.seed,
        num_envs=args.num_envs, num_steps=args.num_steps)

    params_after = pytree_sha256(state["params"])
    optimizer_after = pytree_sha256(state["opt_state"])
    if params_before != params_after or optimizer_before != optimizer_after:
        raise RuntimeError("evaluation mutated params or optimizer")

    module_path = Path(sys.modules[
        "dicode.simulator_frontier.e3_slowgru_original_eval"].__file__)
    payload = {
        "schema": SCHEMA,
        "status": "PASS",
        "source_commit": state["source_commit"],
        "evaluator_sha256": sha256_file(module_path),
        "entrypoint_sha256": sha256_file(__file__),
        "gpu": {"physical_index": 1, "uuid": actual_uuid},
        "runstate": {
            "stem": str(Path(args.runstate_stem).resolve()),
            "state_file_sha256": metadata["state_file_sha256"],
            "meta_file_sha256": sha256_file(args.runstate_stem + ".meta.json"),
            "checkpoint_hash": metadata["checkpoint_hash"],
            "params_sha256": params_before,
            "optimizer_sha256": optimizer_before,
            "global_update_step": int(state["global_update_step"]),
            "global_env_steps": int(state["global_env_steps"]),
            "current_session_idx": int(state["current_session_idx"]),
            "candidate_id": state["candidate_id"],
            "architecture_family": state["architecture_family"],
            "fresh_process_restore": restored["fresh_restore"],
        },
        "protocol": original_task_protocol(
            seed=args.seed, num_envs=args.num_envs, num_steps=args.num_steps),
        "metrics": results,
        "immutability": {
            "params_before_sha256": params_before,
            "params_after_sha256": params_after,
            "optimizer_before_sha256": optimizer_before,
            "optimizer_after_sha256": optimizer_after,
            "unchanged": True,
        },
        "wandb": "disabled",
        "llm_calls": 0,
        "training_updates": 0,
    }
    atomic_write_new_json(str(out), payload)
    print(f"E3_SLOWGRU_ORIGINAL_TASK_EVAL_PASS out={out}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"E3_SLOWGRU_ORIGINAL_TASK_EVAL_FAIL: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
