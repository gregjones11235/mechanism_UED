"""Capability Measurement Plane: read-only tier probes, comprehensive metrics.

Answers "what can the current Student do / not do" with simulator evidence
only.  Never trains, never mutates the student (G4 read-only probe).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict, field
from typing import Any, List

import jax
import numpy as np

from ..data import lightweight_rollout as lr
from ..runtime.hashing import hash_payload, hash_pytree
from . import behavior_metrics
from .tier_registry import TierRegistry


def classify_success_rate(sr: float, n: int) -> str:
    if n <= 0:
        return "UNKNOWN"
    if sr >= 0.8:
        return "MASTERED"
    if sr >= 0.4:
        return "FRONTIER"
    if sr >= 0.1:
        return "UNSTABLE"
    return "FAILED"


@dataclass
class TierProbeResult:
    tier_id: str
    skill_family: str
    probe_id: str
    n_episodes: int
    horizon: int
    success_rate: float
    ci_low: float
    ci_high: float
    status: str
    metrics_aggregate: dict
    params_hash: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CapabilityMeasurementResult:
    student_id: str
    checkpoint_step: int
    params_hash: str
    tier_results: List[TierProbeResult]
    read_only_verified: bool
    probe_wall_s: float

    def to_dict(self) -> dict:
        return {
            "student_id": self.student_id,
            "checkpoint_step": self.checkpoint_step,
            "params_hash": self.params_hash,
            "read_only_verified": self.read_only_verified,
            "probe_wall_s": self.probe_wall_s,
            "tier_results": [t.to_dict() for t in self.tier_results],
        }


def run_capability_probe(*, registry: TierRegistry, backend, params,
                         env_params, student_id: str,
                         checkpoint_step: int = 0, seeds_per_tier: int = 2,
                         batch_envs: int = 4, rng_seed: int = 0,
                         deterministic: bool = True) -> CapabilityMeasurementResult:
    t0 = time.time()
    before = hash_pytree(params)
    tier_results: List[TierProbeResult] = []
    for tier_id in registry.ids():
        tier = registry.get(tier_id)
        env = tier.make_env()
        pred = registry.predicate(tier_id)
        successes = []
        aggs = []
        for seed in range(seeds_per_tier):
            rng = jax.random.PRNGKey(rng_seed * 100003 + seed * 977 + tier.order * 131)
            keys = jax.random.split(rng, batch_envs)
            _obs0, state0 = lr.batched_reset(env, env_params, keys)
            mem = {k: np.asarray(v) for k, v in
                   backend.init_runner_memory(batch_envs).items()}
            batch = lr.collect_rollouts(
                env=env, env_params=env_params, backend=backend, params=params,
                start_states=[state0], start_memories=[mem],
                horizon=tier.horizon, rng=rng, deterministic=deterministic,
                student_version=student_id, frontier_family=tier.skill_family,
                start_state_ids=[f"{tier_id}#seed{seed}"])
            success = np.asarray(pred(batch.trace[-1])).astype(bool)
            metrics = behavior_metrics.trace_metrics(
                batch.trace, batch.actions, batch.rewards, batch.dones, success)
            successes.append(success)
            aggs.append(behavior_metrics.aggregate(metrics))
        succ = np.concatenate(successes)
        n = int(succ.size)
        sr = float(succ.mean()) if n else 0.0
        ci = 1.96 * float(np.sqrt(sr * (1.0 - sr) / n)) if n else 1.0
        merged: dict = {}
        for key in aggs[0]:
            vals = [a[key] for a in aggs if a[key] is not None]
            merged[key] = float(np.mean(vals)) if vals else None
        params_hash = hash_pytree(params)
        tier_results.append(TierProbeResult(
            tier_id=tier_id, skill_family=tier.skill_family,
            probe_id=f"{tier_id}#{params_hash[:8]}", n_episodes=n,
            horizon=tier.horizon, success_rate=sr,
            ci_low=max(0.0, sr - ci), ci_high=min(1.0, sr + ci),
            status=classify_success_rate(sr, n), metrics_aggregate=merged,
            params_hash=params_hash))
    after = hash_pytree(params)
    return CapabilityMeasurementResult(
        student_id=student_id, checkpoint_step=int(checkpoint_step),
        params_hash=before, tier_results=tier_results,
        read_only_verified=before == after,
        probe_wall_s=time.time() - t0)