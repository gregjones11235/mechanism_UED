#!/usr/bin/env python3
"""E3-litesim vertical slice with the REAL SlowGRU persistent student.

Server-only: requires slowgru_runtime + canonical 98304 checkpoint on disk.
Probe -> Frontier -> StateBank -> short on-policy rollouts -> PPO -> Reprobe,
fully LLM-free (E3_NO_LLM=true).

Gates exercised on the real backend:
  G1 student binding, G2 state restore (exact replay), G3 recurrent state
  (slowgru longstate entering-memory validation), G5 on-policy,
  G6 PPO bridge (params change), G7 accounting, G9 vertical slice.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

os.environ.setdefault("E3_NO_LLM", "true")

import jax

from dicode.e3_litesim.learning.ppo_bridge import PPOBridge, PPOConfig
from dicode.e3_litesim.measurement.tier_registry import TierRegistry, TierSpec
from dicode.e3_litesim.orchestration.e3_loop import E3Loop, E3LoopConfig
from dicode.e3_litesim.runtime.hashing import hash_pytree
from dicode.training_backend_slowgru import SlowGRUTrainingBackend

CANDIDATE_ID = "SLOWGRU_PERSISTENT_CANONICAL_98304"


def slice_registry() -> TierRegistry:
    """Reduced-horizon registry (same tiers as the local CPU slice)."""
    return TierRegistry(tiers=(
        TierSpec("tier1_survive", "BASIC_SURVIVAL", 1, "survive", 24,
                 "survived_horizon"),
        TierSpec("tier2_combat", "THREAT_MANAGEMENT", 2, "combat", 32,
                 "monster_killed"),
        TierSpec("tier3_front", "DARK_NAVIGATION", 3, "original", 48,
                 "reached_floor2"),
    ))


class SlowGRUE3Loop(E3Loop):
    """E3Loop with the real SlowGRU persistent backend bound in setup()."""

    def __init__(self, config: E3LoopConfig, registry: TierRegistry,
                 student_pool_cc3: str) -> None:
        super().__init__(config, registry=registry)
        self._student_pool_cc3 = student_pool_cc3

    def setup(self) -> None:
        pool = self._student_pool_cc3
        self.backend = SlowGRUTrainingBackend(
            candidate_id=CANDIDATE_ID,
            slowgru_runtime_path=os.path.join(pool, "slowgru_runtime"),
            checkpoint_contract_path=os.path.join(
                pool, CANDIDATE_ID, "checkpoint_contract.json"),
            checkpoint_path=os.path.join(
                pool, CANDIDATE_ID, "ckpt", "98304", "full_state.pkl"),
            action_dim=43, carry_mode="PERSISTENT")
        self.backend._ensure_loaded()
        params = self.backend._params
        self.bridge = PPOBridge(self.backend, self.config.ppo)
        self.train_state = self.bridge.create_train_state(
            params, jax.random.PRNGKey(self.config.rng_seed + 1))
        self.initial_params_hash = hash_pytree(params)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id",
                    default="vertical_slice_slowgru_" + time.strftime("%Y%m%d"))
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--student-pool-cc3",
                    default="/home/oseasy/student_pool_v1/cc3")
    ap.add_argument("--artifacts", default=os.path.join(
        HERE, "..", "artifacts", "e3_litesim"))
    args = ap.parse_args()

    t0 = time.time()
    loop = SlowGRUE3Loop(
        E3LoopConfig(iterations=args.iterations, num_envs=args.num_envs,
                     rollout_horizon=args.horizon, seeds_per_tier=1,
                     batch_envs=2,
                     ppo=PPOConfig(update_epochs=2, num_minibatches=1),
                     artifacts_dir=args.artifacts, run_id=args.run_id),
        registry=slice_registry(),
        student_pool_cc3=args.student_pool_cc3)
    summary = loop.run()
    out_dir = summary["artifacts_dir"]

    final_hash = hash_pytree(loop.train_state.params)
    meta = {
        "schema": "e3_litesim.slowgru_vertical_slice/v1",
        "candidate_id": CANDIDATE_ID,
        "carry_mode": "PERSISTENT",
        "backend": "SlowGRUTrainingBackend",
        "architecture_family": loop._arch_family(),
        "student_id_note": "probe labels hardcoded to slice_student in "
                           "E3Loop.run(); real student is " + CANDIDATE_ID,
        "initial_params_hash": loop.initial_params_hash,
        "final_params_hash": final_hash,
        "params_changed": bool(final_hash != loop.initial_params_hash),
        "wall_s": round(time.time() - t0, 2),
        "gates": summary["gates"],
        "accounting": summary["accounting"],
    }
    with open(os.path.join(out_dir, "slowgru_slice_meta.json"), "w",
              encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True, default=str)

    print("=" * 60)
    print("E3 LITESIM VERTICAL SLICE - REAL SLOWGRU (server)")
    print("=" * 60)
    gates = summary["gates"]
    print("STATUS:", "PASS" if all(gates.values()) else "PARTIAL")
    for k in sorted(gates):
        print(f"  {k}: {gates[k]}")
    print("params_changed:", meta["params_changed"])
    print("final_params_hash:", final_hash[:16])
    print("accounting:", json.dumps(summary["accounting"], default=str))
    print("wall_s:", meta["wall_s"])
    print(f"ARTIFACTS: {os.path.abspath(out_dir)}")
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
