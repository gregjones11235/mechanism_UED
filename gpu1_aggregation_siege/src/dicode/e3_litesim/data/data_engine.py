"""LightweightSimulatorDataEngine: frontier distribution -> on-policy batches."""
from __future__ import annotations

from typing import Any, Mapping

import jax
import numpy as np

from . import lightweight_rollout as lr
from .state_bank import FrontierStateBank
from .state_sampler import sample_entries


class DataEngineError(RuntimeError):
    pass


class LightweightSimulatorDataEngine:
    """generate_batch(student, distribution, num_envs, horizon, rng).

    One batch per distribution family (envs differ across task families, so
    batches are kept separate and concatenated by PPOBridge).  State-start
    families draw from the FrontierStateBank; 'original'/'anchors' use
    canonical resets of their tier env.
    """

    def __init__(self, *, registry, bank: FrontierStateBank,
                 bank_env, bank_env_params, backend,
                 architecture_family: str = "slice") -> None:
        self.registry = registry
        self.bank = bank
        self.bank_env = bank_env
        self.bank_env_params = bank_env_params
        self.backend = backend
        self.architecture_family = architecture_family

    def generate_batch(self, *, params, student_version: str,
                       distribution: Mapping[str, float],
                       num_envs: int, horizon: int, rng,
                       deterministic: bool = False,
                       allow_memory_reset_experiment: bool = False) -> dict:
        total_w = float(sum(distribution.values()))
        if total_w <= 0:
            raise DataEngineError("empty distribution")
        batches = {}
        accounting = {}
        seed_base = int(np.asarray(rng).reshape(-1)[0]) % (2 ** 31 - 1)
        for family, weight in sorted(distribution.items()):
            n = int(round(num_envs * float(weight) / total_w))
            if n <= 0:
                continue
            if family in ("original", "anchors"):
                tier_id = ("tier1_survive" if family == "anchors"
                           else "tier3_front")
                env = self.registry.get(tier_id).make_env()
                keys = jax.random.split(jax.random.PRNGKey(seed_base + 91), n)
                _o, state = lr.batched_reset(env, self.bank_env_params, keys)
                mem = {k: np.asarray(v) for k, v in
                       self.backend.init_runner_memory(n).items()}
                batch = lr.collect_rollouts(
                    env=env, env_params=self.bank_env_params,
                    backend=self.backend, params=params,
                    start_states=[state], start_memories=[mem],
                    horizon=horizon, rng=jax.random.PRNGKey(seed_base + 92),
                    deterministic=deterministic,
                    student_version=student_version,
                    frontier_family=family,
                    start_state_ids=[f"{family}_reset"],
                    architecture_family=self.architecture_family,
                    allow_memory_reset_experiment=allow_memory_reset_experiment)
            else:
                entries = sample_entries(self.bank.entries, n,
                                         seed=seed_base + hash(family) % 997)
                states, mems, ids = [], [], []
                for entry in entries:
                    st, mm = self.bank.restore_entry(entry)
                    states.append(st)
                    mems.append(mm)
                    ids.append(entry.state_id)
                batch = lr.collect_rollouts(
                    env=self.bank_env, env_params=self.bank_env_params,
                    backend=self.backend, params=params,
                    start_states=states, start_memories=mems,
                    horizon=horizon, rng=jax.random.PRNGKey(seed_base + 93),
                    deterministic=deterministic,
                    student_version=student_version,
                    frontier_family=family, start_state_ids=ids,
                    architecture_family=self.architecture_family,
                    allow_memory_reset_experiment=allow_memory_reset_experiment)
            batches[family] = batch
            accounting[family] = batch.num_transitions
        return {"batches": batches, "accounting": accounting,
                "distribution": dict(distribution)}