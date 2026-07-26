#!/usr/bin/env python3
"""Launcher for W512 × P2 Replay 2×2 experiment.

Usage:
  python launcher_w512_p2.py --arm persistent --steps 4096   # smoke on GPU0
  python launcher_w512_p2.py --arm reset128   --steps 4096   # smoke on GPU1
  python launcher_w512_p2.py --arm persistent --steps 24576  # full on GPU0
  python launcher_w512_p2.py --arm reset128   --steps 24576  # full on GPU1
"""
import os
import sys
import subprocess

BAKE = "/home/oseasy/experiments/bakeoff_phase1/w512_p2_replay"
PY = "/home/oseasy/miniconda3/envs/dicode310/bin/python3"

GPU_MAP = {
    "persistent": "GPU-e8c08612-c22a-c8a4-6df5-affb2dd1f9a6",  # GPU0
    "reset128":   "GPU-3c7a2864-755b-7045-b293-6f80e748283f",  # GPU1
}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["persistent", "reset128"])
    ap.add_argument("--steps", type=int, default=4096, choices=[4096, 24576])
    ap.add_argument("--gpu_uuid", default=None)
    args = ap.parse_args()

    gpu = args.gpu_uuid or GPU_MAP[args.arm]
    tag = f"w512_{args.arm}_p2replay"
    base = f"{BAKE}/{tag}_{args.steps}"
    run_dir = f"{base}/run"
    ckpt_dir = f"{base}/checkpoints"

    cmd = [
        PY, f"{BAKE}/src/run_w512_p2_levelB.py",
        "--steps", str(args.steps),
        "--run_dir", run_dir,
        "--ckpt_dir", ckpt_dir,
        "--carry_mode", args.arm,
        "--gpu_uuid", gpu,
    ]
    print(f"[launcher] {' '.join(cmd)}")
    os.execv(PY, cmd)


if __name__ == "__main__":
    main()
