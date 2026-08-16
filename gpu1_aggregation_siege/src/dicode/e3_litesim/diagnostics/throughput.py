"""Throughput benchmark: canonical full rollout vs state-start short rollout."""
from __future__ import annotations

import time

import jax
import numpy as np

from ..data import lightweight_rollout as lr


def benchmark_throughput(*, env, env_params, backend, params, bank=None,
                         num_envs: int = 8, horizon_full: int = 64,
                         horizon_short: int = 16, rng_seed: int = 0) -> dict:
    rng = jax.random.PRNGKey(rng_seed)

    t0 = time.time()
    full = lr.collect_full_rollouts(env=env, env_params=env_params,
                                    backend=backend, params=params,
                                    num_envs=num_envs, horizon=horizon_full,
                                    rng=rng)
    full.block_until_ready() if hasattr(full, "block_until_ready") else None
    full_wall = time.time() - t0
    full_n = full.num_transitions

    short_wall = None
    short_n = 0
    if bank is not None and bank.entries:
        states, mems, ids = [], [], []
        for entry in bank.entries[:num_envs]:
            st, mm = bank.restore_entry(entry)
            states.append(st)
            mems.append(mm)
            ids.append(entry.state_id)
        t0 = time.time()
        short = lr.collect_rollouts(env=env, env_params=env_params,
                                    backend=backend, params=params,
                                    start_states=states, start_memories=mems,
                                    horizon=horizon_short, rng=rng,
                                    start_state_ids=ids)
        short_wall = time.time() - t0
        short_n = short.num_transitions

    return {
        "num_envs": num_envs,
        "full_horizon": horizon_full,
        "full_transitions": full_n,
        "full_wall_s": round(full_wall, 4),
        "full_transitions_per_sec": round(full_n / full_wall, 1) if full_wall else 0.0,
        "short_horizon": horizon_short,
        "short_transitions": short_n,
        "short_wall_s": round(short_wall, 4) if short_wall else None,
        "short_transitions_per_sec": (round(short_n / short_wall, 1)
                                      if short_wall else None),
    }