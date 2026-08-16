"""Standalone throughput benchmark for the litesim data plane."""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import jax
import numpy as np
from craftax.craftax.craftax_state import EnvParams

from dicode.e3_litesim.diagnostics.throughput import benchmark_throughput
from dicode.e3_litesim.measurement.tier_registry import TierRegistry
from dicode.e3_litesim.runtime.slice_student import SliceStudentBackend


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--horizon-full", type=int, default=64)
    ap.add_argument("--horizon-short", type=int, default=16)
    args = ap.parse_args()
    env_params = EnvParams(max_timesteps=4096)
    registry = TierRegistry()
    env = registry.get("tier1_survive").make_env()
    obs0, _ = env.reset(jax.random.PRNGKey(0), env_params)
    backend = SliceStudentBackend(int(np.asarray(obs0).shape[-1]),
                                  int(env.action_space(env_params).n))
    params = backend.initial_params(jax.random.PRNGKey(0))
    bench = benchmark_throughput(env=env, env_params=env_params,
                                 backend=backend, params=params, bank=None,
                                 num_envs=args.num_envs,
                                 horizon_full=args.horizon_full,
                                 horizon_short=args.horizon_short)
    print(json.dumps(bench, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())