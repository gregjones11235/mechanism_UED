"""FailureCapsule: freeze EnvState + recurrent state + RNG + TaskParams (G2/G3).

Includes the exact-replay contract: restoring a capsule and re-running the
same student with the same RNG must reproduce the identical trajectory.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Optional

import jax
import numpy as np

from dicode.simulator_frontier import env_restore

from ..data import lightweight_rollout as lr
from ..runtime import recurrent_state
from ..runtime.hashing import hash_payload

_TEMPLATE_REGISTRY: dict = {}


class CapsuleError(RuntimeError):
    pass


def register_template(template) -> str:
    _TEMPLATE_REGISTRY[template.treedef_fingerprint] = template
    return template.treedef_fingerprint


def get_template(fingerprint: str):
    if fingerprint not in _TEMPLATE_REGISTRY:
        raise CapsuleError(f"unknown state template {fingerprint[:12]}")
    return _TEMPLATE_REGISTRY[fingerprint]


def _to_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    return np.asarray(value).tolist()


def _from_json(value: Any, spec) -> Any:
    if spec.kind == "none" or value is None:
        return None
    if spec.kind == "array":
        return np.asarray(value, dtype=spec.dtype)
    if spec.scalar_type == "bool":
        return bool(value)
    if spec.scalar_type == "int":
        return int(value)
    if spec.scalar_type == "float":
        return float(value)
    return str(value)


@dataclass
class FailureCapsule:
    tier_id: str
    probe_id: str
    episode_timestep: int
    template_fingerprint: str
    env_state_flat: dict
    observation: list
    params_hash: str
    memory: dict
    memory_hash: str
    rng_seed: int
    task_params: dict
    snapshot: dict
    base_state_hash: str
    capsule_hash: str

    def to_dict(self) -> dict:
        return asdict(self)


def capture_failure_capsule(*, env_state, memory, params_hash: str,
                            rng_seed: int, tier_id: str, probe_id: str,
                            episode_timestep: int,
                            task_params: Optional[dict] = None,
                            observation=None) -> FailureCapsule:
    template = env_restore.build_template(env_state)
    fp = register_template(template)
    flat = env_restore.flatten_env_state(env_state)
    flat_json = {k: _to_json(v) for k, v in flat["leaves"].items()}
    base_state_hash = hash_payload(flat_json)
    mem = recurrent_state.capture_memory(memory)
    mem_hash = recurrent_state.memory_hash(memory)
    pos = np.asarray(env_state.player_position)
    snapshot = {
        "player_health": np.asarray(env_state.player_health).tolist(),
        "player_position": pos.tolist(),
        "floor": np.asarray(env_state.player_level).tolist(),
        "timestep": int(episode_timestep),
    }
    body = {
        "tier_id": tier_id, "probe_id": probe_id,
        "episode_timestep": int(episode_timestep),
        "template_fingerprint": fp, "base_state_hash": base_state_hash,
        "params_hash": params_hash, "memory_hash": mem_hash,
        "rng_seed": int(rng_seed), "task_params": task_params or {},
    }
    return FailureCapsule(
        tier_id=tier_id, probe_id=probe_id,
        episode_timestep=int(episode_timestep), template_fingerprint=fp,
        env_state_flat=flat_json,
        observation=np.asarray(observation).tolist() if observation is not None else [],
        params_hash=params_hash,
        memory={k: _to_json(v) for k, v in mem.items()},
        memory_hash=mem_hash, rng_seed=int(rng_seed),
        task_params=task_params or {}, snapshot=snapshot,
        base_state_hash=base_state_hash,
        capsule_hash=hash_payload(body))


def restore_capsule(capsule: FailureCapsule):
    """Returns (env_state, memory_dict_np)."""
    template = get_template(capsule.template_fingerprint)
    specs = dict(zip(template.leaf_paths, template.leaf_specs))
    leaves = {k: _from_json(v, specs[k]) for k, v in capsule.env_state_flat.items()}
    envelope = {
        "flat_version": env_restore.FLAT_ENV_STATE_VERSION,
        "env_state_type": template.env_state_type,
        "treedef_fingerprint": template.treedef_fingerprint,
        "leaf_paths": list(template.leaf_paths),
        "leaves": leaves,
    }
    env_state = env_restore.unflatten_env_state(envelope, template)
    memory = {k: np.asarray(v) for k, v in capsule.memory.items()}
    return env_state, memory


def exact_replay_check(capsule: FailureCapsule, backend, params, *,
                       env, env_params, horizon: int = 8) -> dict:
    """G3/G2 evidence: two independent restores -> identical greedy trace."""
    hashes = []
    for _arm in range(2):
        env_state, memory = restore_capsule(capsule)
        obs = lr.batched_get_obs(env, env_state)
        mem = jax.tree_util.tree_map(np.asarray, memory)
        parts = []
        for _t in range(horizon):
            pi, value, _mo, new_mem = backend.policy_forward_eval(params, mem, obs)
            action = pi.mode()
            parts.append(np.asarray(action).tobytes())
            parts.append(np.asarray(obs).tobytes())
            rng = jax.random.PRNGKey(capsule.rng_seed)
            step_keys = jax.random.split(rng, int(np.asarray(action).shape[0]))
            _o, env_state, _r, _d, _i = lr.batched_step(
                env, step_keys, env_state, action, env_params)
            obs = _o
            mem = new_mem
        import hashlib
        hashes.append(hashlib.sha256(b"".join(parts)).hexdigest())
    return {"ok": hashes[0] == hashes[1], "trajectory_hash_a": hashes[0],
            "trajectory_hash_b": hashes[1]}