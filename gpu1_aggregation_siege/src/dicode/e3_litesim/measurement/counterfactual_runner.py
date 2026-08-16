"""Lightweight single-factor counterfactual branches (G2 isolation).

Each intervention may change ONLY its whitelisted state paths; the diff guard
rejects any branch that leaks outside the whitelist.  Evidence insufficient
=> UNKNOWN (never invent a cause).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, List, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from dicode.simulator_frontier import env_restore

from ..data import lightweight_rollout as lr
from ..measurement import behavior_metrics
from .causal_evidence import CauseRecord, aggregate_causal_evidence, CausalEvidence
from .failure_capsule import FailureCapsule, restore_capsule


class InterventionIsolationError(RuntimeError):
    """A counterfactual branch changed non-whitelisted state (fail closed)."""


@dataclass(frozen=True)
class InterventionSpec:
    name: str
    cause_family: str
    whitelist: tuple
    patch: Callable[[Any], Any]


def _patch_visibility(state):
    return state.replace(light_map=jnp.full_like(state.light_map, 9))


def _patch_enemy_removed(state):
    return state.replace(mob_map=jnp.zeros_like(state.mob_map))


def _patch_health_boost(state):
    return state.replace(
        player_health=jnp.full_like(state.player_health, 9.0))


def _patch_torch_preloaded(state):
    inv = state.inventory
    if hasattr(inv, "replace"):
        new_inv = inv.replace(torch=inv.torch + 5)
    else:
        new_inv = dataclasses.replace(inv, torch=inv.torch + 5)
    return state.replace(inventory=new_inv)


def default_interventions() -> List[InterventionSpec]:
    return [
        InterventionSpec("visibility_plus", "PERCEPTION", ("light_map",),
                         _patch_visibility),
        InterventionSpec("enemy_removed", "THREAT_MANAGEMENT", ("mob_map",),
                         _patch_enemy_removed),
        InterventionSpec("health_boost", "SURVIVABILITY", ("player_health",),
                         _patch_health_boost),
        InterventionSpec("torch_preloaded", "TOOL_USE", ("inventory.torch",),
                         _patch_torch_preloaded),
    ]


def assert_intervention_isolation(before_state, after_state,
                                  whitelist: Sequence[str]) -> dict:
    flat_a = env_restore.flatten_env_state(before_state)["leaves"]
    flat_b = env_restore.flatten_env_state(after_state)["leaves"]
    changed = []
    for key in set(flat_a) | set(flat_b):
        va, vb = flat_a.get(key), flat_b.get(key)
        same = (va is None and vb is None) or (
            va is not None and vb is not None and
            np.asarray(va).shape == np.asarray(vb).shape and
            bool(np.array_equal(np.asarray(va), np.asarray(vb))))
        if not same:
            changed.append(key)
    bad = [k for k in changed
           if not any(k == w or k.startswith(w + ".") or k.startswith(w + "[")
                      for w in whitelist)]
    if bad:
        raise InterventionIsolationError(
            f"intervention leaked outside whitelist {sorted(whitelist)}: {sorted(bad)}")
    return {"changed_paths": sorted(changed), "whitelist": sorted(whitelist)}


def _batched(state, memory, seeds: int):
    states = [state] * seeds
    mems = [memory] * seeds
    if seeds == 1:
        return state, memory
    stacked = jax.tree_util.tree_map(
        lambda *xs: np.concatenate([np.asarray(x) for x in xs], axis=0), *states)
    mem_stack = {k: np.concatenate([np.asarray(m[k]) for m in mems], axis=0)
                 for k in memory}
    return stacked, mem_stack


def run_counterfactual_diagnosis(*, capsule: FailureCapsule, env, env_params,
                                 backend, params, success_fn,
                                 interventions: List[InterventionSpec] = None,
                                 seeds: int = 2, horizon: int = 24,
                                 rng_seed: int = 0,
                                 architecture_family: str = "slice",
                                 allow_memory_reset_experiment: bool = False
                                 ) -> CausalEvidence:
    interventions = interventions if interventions is not None else default_interventions()

    def _run(state, memory, tag):
        bstate, bmem = _batched(state, memory, seeds)
        batch = lr.collect_rollouts(
            env=env, env_params=env_params, backend=backend, params=params,
            start_states=[bstate], start_memories=[bmem], horizon=horizon,
            rng=jax.random.PRNGKey(rng_seed + hash(tag) % 10000),
            deterministic=True, frontier_family=f"cf_{tag}",
            architecture_family=architecture_family,
            allow_memory_reset_experiment=allow_memory_reset_experiment,
            collect_trace=True)
        success = np.asarray(success_fn(batch.trace[-1])).astype(float)
        prog = np.asarray(batch.rewards).sum(axis=0)
        return float(success.mean()), float(prog.mean())

    base_state, base_memory = restore_capsule(capsule)
    base_s, base_p = _run(base_state, base_memory, "baseline")

    records: List[CauseRecord] = []
    for spec in interventions:
        try:
            patched = spec.patch(base_state)
        except Exception:
            continue  # intervention not constructible for this world: skip
        assert_intervention_isolation(base_state, patched, spec.whitelist)
        s, p = _run(patched, base_memory, spec.name)
        records.append(CauseRecord(
            name=spec.name, cause_family=spec.cause_family,
            delta_success=s - base_s, delta_progress=p - base_p,
            support={"baseline_success": base_s, "branch_success": s}))
    return aggregate_causal_evidence(records)