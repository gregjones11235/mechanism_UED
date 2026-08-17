"""FrontierStateBank: frozen + simulator-valid prefix-variant start states.

Every entry carries full provenance (source capsule, variation seed,
environment patch, task params, state hash).  Invalid / impossible EnvStates
are rejected by the restore-and-step validity guard.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, List

import jax
import numpy as np

from dicode.simulator_frontier import env_restore

from ..runtime import recurrent_state
from ..runtime.hashing import hash_payload, hash_pytree
from ..runtime.provenance import GENESIS_HASH, ProvenanceRecord
from . import lightweight_rollout as lr
from ..measurement.failure_capsule import (FailureCapsule, get_template,
                                           restore_capsule)


class StateBankError(RuntimeError):
    pass


@dataclass
class BankEntry:
    state_id: str
    source_capsule_hash: str
    frontier_family: str
    tier: str
    variation_seed: int
    environment_patch: str
    task_params: dict
    state_hash: str
    template_fingerprint: str
    env_state_flat: dict
    memory: dict
    memory_hash: str
    provenance_hash: str
    student_version: int
    params_hash: str
    architecture_family: str

    def to_dict(self) -> dict:
        return asdict(self)


class FrontierStateBank:
    def __init__(self, frontier_family: str) -> None:
        self.frontier_family = frontier_family
        self.entries: List[BankEntry] = []
        self._prev_hash = GENESIS_HASH

    def _add(self, *, state, memory, capsule, variation_seed, patch_name,
             task_params, tier: str = "", student_version: int = 0,
             params_hash: str = "", architecture_family: str = "slice"
             ) -> BankEntry:
        template = env_restore.build_template(state)
        flat = env_restore.flatten_env_state(state)
        flat_json = {k: (None if v is None else
                         (v if isinstance(v, (bool, int, float, str))
                          else np.asarray(v).tolist()))
                     for k, v in flat["leaves"].items()}
        state_hash = hash_payload(flat_json)
        mem = recurrent_state.capture_memory(memory,
                                             architecture_family=architecture_family)
        prov = ProvenanceRecord(
            "frontier_state", {
                "source_capsule": capsule.capsule_hash,
                "frontier_family": self.frontier_family,
                "tier": tier,
                "variation_seed": int(variation_seed),
                "environment_patch": patch_name,
                "task_params": task_params or {},
                "state_hash": state_hash,
                "student_version": int(student_version),
                "params_hash": params_hash,
                "architecture_family": architecture_family,
            }, prev_hash=self._prev_hash).finalized()
        self._prev_hash = prov.record_hash
        entry = BankEntry(
            state_id=f"st_{len(self.entries):06d}",
            source_capsule_hash=capsule.capsule_hash,
            frontier_family=self.frontier_family,
            tier=tier,
            variation_seed=int(variation_seed),
            environment_patch=patch_name,
            task_params=task_params or {},
            state_hash=state_hash,
            template_fingerprint=template.treedef_fingerprint,
            env_state_flat=flat_json,
            memory={k: v.tolist() for k, v in mem.items()},
            memory_hash=recurrent_state.memory_hash(
                mem, architecture_family=architecture_family),
            provenance_hash=prov.record_hash,
            student_version=int(student_version),
            params_hash=params_hash,
            architecture_family=architecture_family)
        self.entries.append(entry)
        return entry

    def restore_entry(self, entry: BankEntry):
        from ..measurement.failure_capsule import _from_json

        template = get_template(entry.template_fingerprint)
        specs = dict(zip(template.leaf_paths, template.leaf_specs))
        leaves = {k: _from_json(v, specs[k]) for k, v in entry.env_state_flat.items()}
        envelope = {
            "flat_version": env_restore.FLAT_ENV_STATE_VERSION,
            "env_state_type": template.env_state_type,
            "treedef_fingerprint": template.treedef_fingerprint,
            "leaf_paths": list(template.leaf_paths),
            "leaves": leaves,
        }
        state = env_restore.unflatten_env_state(envelope, template)
        memory = {k: np.asarray(v) for k, v in entry.memory.items()}
        return state, memory

    def validate_entry(self, entry: BankEntry, *, env, env_params, backend,
                       params) -> dict:
        """Restore-and-step guard: state must be simulator-valid.

        Performs one simulator step per restored env (accounted as
        ``state_bank_validation`` transitions by the caller).
        """
        state, memory = self.restore_entry(entry)
        obs = lr.batched_get_obs(env, state)
        if not bool(np.isfinite(np.asarray(obs)).all()):
            raise StateBankError(f"entry {entry.state_id}: non-finite obs")
        rng = jax.random.PRNGKey(entry.variation_seed)
        pi, value, _mo, _nm = backend.policy_forward_eval(params, memory, obs)
        action = pi.mode()
        batch = int(np.asarray(action).shape[0])
        step_keys = jax.random.split(rng, batch)
        _o, _s, _r, _d, _i = lr.batched_step(env, step_keys, state, action, env_params)
        if not bool(np.isfinite(np.asarray(_o)).all()):
            raise StateBankError(f"entry {entry.state_id}: step produced NaN")
        return {"state_id": entry.state_id, "valid": True, "transitions": batch}

    def build_from_capsule(self, capsule: FailureCapsule, *, env, env_params,
                           backend, params, n_frozen: int = 2,
                           prefix_steps=(0, 2, 4), rng_seed: int = 0,
                           architecture_family: str = "slice",
                           student_version: int = 0,
                           allow_memory_reset_experiment: bool = False) -> tuple:
        """Build bank entries from a failure capsule.

        Returns ``(manifest, transitions)`` where ``transitions`` is the number
        of simulator steps consumed by the prefix-variant rollouts (accounted
        by the caller as ``state_bank_build``).
        """
        base_state, base_memory = restore_capsule(capsule)
        params_hash = hash_pytree(params)
        tier = getattr(capsule, "tier_id", "")
        # 1) frozen states: exact capsule state, distinct variation seeds
        for i in range(n_frozen):
            self._add(state=base_state, memory=base_memory, capsule=capsule,
                      variation_seed=rng_seed + i, patch_name="frozen",
                      task_params=capsule.task_params, tier=tier,
                      student_version=student_version, params_hash=params_hash,
                      architecture_family=architecture_family)
        # 2) simulator-valid prefix variants: continue the episode k steps
        #    under the CURRENT policy and freeze intermediate (state, memory)
        transitions = 0
        for k in prefix_steps:
            if k == 0:
                continue
            batch = lr.collect_rollouts(
                env=env, env_params=env_params, backend=backend, params=params,
                start_states=[base_state], start_memories=[base_memory],
                horizon=int(k), rng=jax.random.PRNGKey(rng_seed + 5000 + k),
                deterministic=True, architecture_family=architecture_family,
                allow_memory_reset_experiment=allow_memory_reset_experiment,
                collect_trace=False, collect_memory_trace=True)
            state_k = batch.trace[-1] if batch.trace else base_state
            mem_k = batch.memory_trace[-1]
            transitions += int(batch.num_transitions)
            self._add(state=state_k, memory=mem_k, capsule=capsule,
                      variation_seed=rng_seed + 100 + k,
                      patch_name=f"prefix_k{k}",
                      task_params=capsule.task_params, tier=tier,
                      student_version=student_version, params_hash=params_hash,
                      architecture_family=architecture_family)
        return self.manifest(), transitions

    def manifest(self) -> dict:
        return {
            "frontier_family": self.frontier_family,
            "n_entries": len(self.entries),
            "entries": [e.to_dict() for e in self.entries],
            "provenance_head": self._prev_hash,
            "manifest_hash": hash_payload(
                [e.state_hash for e in self.entries] + [self._prev_hash]),
        }