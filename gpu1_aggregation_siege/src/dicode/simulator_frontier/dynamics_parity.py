"""Leaf-level state comparison and dual-track dynamics parity runners.

Parity protocol (fail-closed):

- Two rollouts are ALWAYS separate tracks: the original state and the restored
  state each run their own trajectory with the same key stream and the same
  action sequence.  They are never mixed into one batch (which would couple
  their RNG streams through per-env key splitting and could hide divergence).
- Key stream convention: runner key ``r``; per step ``r, step_key = split(r)``
  and both tracks are stepped with the identical ``step_key``.
- Comparison is leaf-by-leaf sha256 over flattened states plus obs/reward/done
  equality per step.  ``first_divergence is None`` is the pass condition.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Mapping, Sequence

import jax
import numpy as np

from .env_restore import flatten_env_state


def leaf_sha(value: Any) -> str:
    """Deterministic content hash for one leaf (array/None/python scalar)."""
    if value is None:
        payload = b"none:"
    elif isinstance(value, np.ndarray) or (hasattr(value, "shape") and hasattr(value, "dtype")):
        arr = np.asarray(value)
        payload = f"array:{arr.dtype}:{arr.shape}:".encode("utf-8") + np.ascontiguousarray(arr).tobytes()
    elif isinstance(value, (bool, int, float, str)):
        payload = f"scalar:{type(value).__name__}:{value!r}".encode("utf-8")
    else:
        raise TypeError(f"unsupported leaf type for parity hashing: {type(value).__name__}")
    return hashlib.sha256(payload).hexdigest()


def compare_flat_states(flat_a: Mapping[str, Any], flat_b: Mapping[str, Any]) -> dict:
    """Compare two flattened env states leaf by leaf (JSON-serializable result)."""
    paths_a = list(flat_a.get("leaf_paths", []))
    paths_b = list(flat_b.get("leaf_paths", []))
    leaves_a = flat_a.get("leaves", {}) or {}
    leaves_b = flat_b.get("leaves", {}) or {}
    order = paths_a + [p for p in paths_b if p not in set(paths_a)]
    records = []
    mismatched = []
    missing_in_b = []
    extra_in_b = []
    for path in order:
        present_a = path in leaves_a
        present_b = path in leaves_b
        if present_a and not present_b:
            missing_in_b.append(path)
        if present_b and not present_a:
            extra_in_b.append(path)
        value_a = leaves_a.get(path)
        value_b = leaves_b.get(path)
        sha_a = leaf_sha(value_a) if present_a else None
        sha_b = leaf_sha(value_b) if present_b else None
        equal = bool(present_a and present_b and sha_a == sha_b)
        if not equal:
            mismatched.append(path)
        shape = None
        dtype = None
        for value in (value_a, value_b):
            if value is not None and (isinstance(value, np.ndarray) or (hasattr(value, "shape") and hasattr(value, "dtype"))):
                arr = np.asarray(value)
                shape = [int(x) for x in arr.shape]
                dtype = str(arr.dtype)
                break
        records.append({"path": path, "shape": shape, "dtype": dtype,
                        "sha256_a": sha_a, "sha256_b": sha_b, "equal": equal})
    ok = not mismatched and not missing_in_b and not extra_in_b
    return {"ok": ok, "n_leaves": len(order), "mismatched": mismatched,
            "missing_in_b": missing_in_b, "extra_in_b": extra_in_b, "leaves": records}


def compare_env_states(state_a: Any, state_b: Any) -> dict:
    """Flatten both states and compare leaf by leaf."""
    return compare_flat_states(flatten_env_state(state_a), flatten_env_state(state_b))


def run_parity_rollout(step_fn: Callable[[Any, Any, int], tuple[Any, Any, Any, Any, Any]],
                       state_a: Any, state_b: Any, *, actions: Sequence[int], key: Any,
                       flatten_fn: Callable[[Any], dict] = flatten_env_state) -> tuple[dict, Any, Any]:
    """Step two independent tracks with the same key stream and compare each step.

    ``step_fn(step_key, state, action) -> (obs, state, reward, done, info)``.
    Returns ``(report, final_state_a, final_state_b)``; the report dict is
    JSON-serializable and carries ``first_divergence=None`` when fully equal.
    """
    steps = []
    first_divergence = None
    s_a, s_b = state_a, state_b
    for i, action in enumerate(actions):
        key, step_key = jax.random.split(key)
        obs_a, s_a, r_a, d_a, _ = step_fn(step_key, s_a, int(action))
        obs_b, s_b, r_b, d_b, _ = step_fn(step_key, s_b, int(action))
        obs_equal = bool(np.array_equal(np.asarray(obs_a), np.asarray(obs_b)))
        reward_equal = bool(float(np.asarray(r_a)) == float(np.asarray(r_b)))
        done_equal = bool(bool(np.asarray(d_a)) == bool(np.asarray(d_b)))
        state_cmp = compare_flat_states(flatten_fn(s_a), flatten_fn(s_b))
        step_ok = obs_equal and reward_equal and done_equal and state_cmp["ok"]
        steps.append({"step": i, "action": int(action), "obs_equal": obs_equal,
                      "reward_equal": reward_equal, "done_equal": done_equal,
                      "state_ok": state_cmp["ok"]})
        if not step_ok and first_divergence is None:
            if not obs_equal:
                kind, path = "obs", None
            elif not reward_equal:
                kind, path = "reward", None
            elif not done_equal:
                kind, path = "done", None
            else:
                paths = state_cmp["mismatched"] or state_cmp["missing_in_b"] or state_cmp["extra_in_b"]
                kind, path = "state", (paths[0] if paths else None)
            first_divergence = {"step": i, "kind": kind, "path": path}
    final_cmp = compare_flat_states(flatten_fn(s_a), flatten_fn(s_b))
    report = {"ok": first_divergence is None, "n_steps": len(steps),
              "first_divergence": first_divergence, "steps": steps,
              "final_state_comparison": final_cmp}
    return report, s_a, s_b
