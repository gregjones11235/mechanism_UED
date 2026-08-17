#!/usr/bin/env python3
"""REAL SlowGRU GPU throughput grid benchmark (G8 evidence).

Server-only: requires slowgru_runtime + the canonical 98304 checkpoint on disk.
Unlike benchmark_e3_litesim_grid.py (synthetic SliceStudentBackend), this runs
the REAL SlowGRUTrainingBackend and records full environment provenance:
device / GPU UUID / CUDA / git SHA / checkpoint hash / params hash / GPU memory
/ GPU utilization.

Grid: num_envs {64,256,1024} x horizon {128,256,512}; two modes per cell:
- full:        full rollouts from reset states (num_envs x horizon)
- state_start: short rollouts (horizon//4) from FrontierStateBank states

Each measured run is preceded by an unmeasured warmup call with identical
shapes, so JIT/XLA compile time is excluded from the reported wall times.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

os.environ.setdefault("E3_NO_LLM", "true")

import jax
import numpy as np
from craftax.craftax.craftax_state import EnvParams

from dicode.e3_litesim.data import lightweight_rollout as lr
from dicode.e3_litesim.data.state_bank import FrontierStateBank
from dicode.e3_litesim.measurement.capability_probe import run_capability_probe
from dicode.e3_litesim.measurement.frontier_locator import locate_frontier
from dicode.e3_litesim.measurement.tier_registry import TierRegistry, TierSpec
from dicode.e3_litesim.runtime.hashing import hash_pytree
from dicode.training_backend_slowgru import SlowGRUTrainingBackend

CANDIDATE_ID = "SLOWGRU_PERSISTENT_CANONICAL_98304"


def slice_registry() -> TierRegistry:
    return TierRegistry(tiers=(
        TierSpec("tier1_survive", "BASIC_SURVIVAL", 1, "survive", 64,
                 "survived_horizon"),
        TierSpec("tier2_combat", "THREAT_MANAGEMENT", 2, "combat", 64,
                 "monster_killed"),
        TierSpec("tier3_front", "DARK_NAVIGATION", 3, "original", 64,
                 "reached_floor2"),
    ))


# ----------------------------------------------------------------------
# environment provenance
# ----------------------------------------------------------------------
def _nvidia_smi(query: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            timeout=10, stderr=subprocess.DEVNULL)
        return out.decode().strip().splitlines()[0].strip()
    except Exception:
        return None


def gpu_mem_mb() -> float | None:
    try:
        stats = jax.devices()[0].memory_stats()
        return round(stats["bytes_in_use"] / 1e6, 1)
    except Exception:
        return None


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", os.path.join(HERE, ".."), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def cuda_version() -> str | None:
    try:
        from jax.lib import xla_bridge
        return str(xla_bridge.get_backend().platform_version)
    except Exception:
        return None


def _block(x):
    if hasattr(x, "block_until_ready"):
        x.block_until_ready()
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, nargs="+", default=[64, 256, 1024])
    ap.add_argument("--horizons", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--student-pool-cc3", default="/home/oseasy/student_pool_v1/cc3")
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(
        HERE, "..", "artifacts", "e3_litesim", "gpu1_slowgru_benchmark"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "throughput.csv")
    json_path = os.path.join(args.out, "grid_results.json")

    backend = SlowGRUTrainingBackend(
        candidate_id=CANDIDATE_ID,
        slowgru_runtime_path=os.path.join(args.student_pool_cc3, "slowgru_runtime"),
        checkpoint_contract_path=os.path.join(
            args.student_pool_cc3, CANDIDATE_ID, "checkpoint_contract.json"),
        checkpoint_path=os.path.join(
            args.student_pool_cc3, CANDIDATE_ID, "ckpt", "98304", "full_state.pkl"),
        action_dim=43, carry_mode="PERSISTENT")
    backend._ensure_loaded()
    params = backend._params
    params_hash = hash_pytree(params)

    registry = slice_registry()
    env_params = EnvParams(max_timesteps=4096)
    env = registry.get("tier1_survive").make_env()

    # state bank via canonical probe -> frontier -> capsule pipeline (real backend)
    print("[grid-slowgru] probing for frontier...", flush=True)
    meas = run_capability_probe(registry=registry, backend=backend,
                                params=params, env_params=env_params,
                                student_id=CANDIDATE_ID, checkpoint_step=98304,
                                seeds_per_tier=1, batch_envs=1,
                                rng_seed=args.rng_seed)
    frontier = locate_frontier(meas, registry)
    tier = registry.get(frontier.tier)
    fenv = tier.make_env()
    keys = jax.random.split(jax.random.PRNGKey(args.rng_seed + 5), 1)
    _o, state0 = lr.batched_reset(fenv, env_params, keys)
    mem = {k: np.asarray(v) for k, v in backend.init_runner_memory(1).items()}
    cap_batch = lr.collect_rollouts(
        env=fenv, env_params=env_params, backend=backend, params=params,
        start_states=[state0], start_memories=[mem], horizon=tier.horizon,
        rng=jax.random.PRNGKey(args.rng_seed + 5), deterministic=True,
        collect_trace=True, collect_memory_trace=True,
        architecture_family="slowgru", allow_memory_reset_experiment=True)
    from dicode.e3_litesim.measurement.failure_capsule import (
        capture_failure_capsule)
    t_cap = min(len(cap_batch.trace) - 1, tier.horizon // 2)
    capsule = capture_failure_capsule(
        env_state=cap_batch.trace[t_cap],
        memory=cap_batch.memory_trace[t_cap],
        params_hash=params_hash, rng_seed=args.rng_seed + 71, tier_id=tier.tier_id,
        probe_id=frontier.probe_id, episode_timestep=t_cap,
        task_params={"tier": tier.tier_id},
        observation=np.asarray(lr.batched_get_obs(fenv, cap_batch.trace[t_cap])))
    bank = FrontierStateBank(frontier.skill_family)
    bank.build_from_capsule(capsule, env=fenv, env_params=env_params,
                            backend=backend, params=params,
                            n_frozen=max(args.num_envs), prefix_steps=(2,),
                            architecture_family="slowgru",
                            student_version=0,
                            allow_memory_reset_experiment=True)
    print(f"[grid-slowgru] bank entries: {len(bank.entries)} "
          f"(frontier tier={frontier.tier})", flush=True)

    header = ["num_envs", "horizon", "mode", "horizon_used", "transitions",
              "warm_wall_s", "wall_s", "transitions_per_sec", "gpu_mem_mb"]
    new_csv = not os.path.isfile(csv_path)
    results = []

    for E in args.num_envs:
        for H in args.horizons:
            Hs = max(4, H // 4)
            rng = jax.random.PRNGKey(args.rng_seed)
            # full mode: warmup + measured
            t0 = time.time()
            _block(lr.collect_full_rollouts(env=env, env_params=env_params,
                                            backend=backend, params=params,
                                            num_envs=E, horizon=H, rng=rng))
            warm_full = round(time.time() - t0, 3)
            t0 = time.time()
            full = _block(lr.collect_full_rollouts(
                env=env, env_params=env_params, backend=backend, params=params,
                num_envs=E, horizon=H, rng=rng))
            wall_full = time.time() - t0
            row_full = [E, H, "full", H, full.num_transitions, warm_full,
                        round(wall_full, 4),
                        round(full.num_transitions / wall_full, 1), gpu_mem_mb()]
            # state-start mode: warmup + measured
            entries = bank.entries[:E]
            states, mems, ids = [], [], []
            for entry in entries:
                st, mm = bank.restore_entry(entry)
                states.append(st)
                mems.append(mm)
                ids.append(entry.state_id)
            t0 = time.time()
            _block(lr.collect_rollouts(env=fenv, env_params=env_params,
                                       backend=backend, params=params,
                                       start_states=states, start_memories=mems,
                                       horizon=Hs, rng=rng,
                                       start_state_ids=ids,
                                       architecture_family="slowgru",
                                       allow_memory_reset_experiment=True))
            warm_short = round(time.time() - t0, 3)
            t0 = time.time()
            short = _block(lr.collect_rollouts(
                env=fenv, env_params=env_params, backend=backend, params=params,
                start_states=states, start_memories=mems, horizon=Hs, rng=rng,
                start_state_ids=ids, architecture_family="slowgru",
                allow_memory_reset_experiment=True))
            wall_short = time.time() - t0
            row_short = [E, H, "state_start", Hs, short.num_transitions,
                         warm_short, round(wall_short, 4),
                         round(short.num_transitions / wall_short, 1),
                         gpu_mem_mb()]
            with open(csv_path, "a", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                if new_csv:
                    writer.writerow(header)
                    new_csv = False
                writer.writerow(row_full)
                writer.writerow(row_short)
            results.append({"num_envs": E, "horizon": H,
                            "full": dict(zip(header, row_full)),
                            "state_start": dict(zip(header, row_short))})
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump({
                    "student_id": CANDIDATE_ID,
                    "architecture_family": "SLOWGRU",
                    "carry_mode": "PERSISTENT",
                    "device": str(jax.devices()[0]),
                    "device_uuid": _nvidia_smi("uuid"),
                    "gpu_name": _nvidia_smi("name"),
                    "gpu_utilization_pct": _nvidia_smi("utilization.gpu"),
                    "jax_version": jax.__version__,
                    "cuda_platform": cuda_version(),
                    "git_sha": git_sha(),
                    "checkpoint_step": 98304,
                    "params_hash": params_hash,
                    "cells": results,
                }, fh, indent=2)
            print(f"[grid-slowgru] E={E} H={H}: full={row_full[7]} t/s "
                  f"(warm {warm_full}s), state_start={row_short[7]} t/s "
                  f"(warm {warm_short}s)", flush=True)
    print("[grid-slowgru] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
