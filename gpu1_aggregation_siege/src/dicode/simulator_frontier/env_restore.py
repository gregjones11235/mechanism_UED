"""Flatten/restore bridge between real Craftax EnvState pytrees and StateCodec.

Design notes (all fail-closed via SchemaMismatchError):

- None leaves are first-class.  ``EnvState.fractal_noise_angles`` defaults to
  ``(None, None, None, None)`` and jax treats ``None`` as an empty pytree node,
  so flattening passes ``is_leaf=lambda x: x is None`` to keep them explicit.
- Template lineage: a template may only restore states from the same lineage
  (plain eager reset state vs jit-stepped state vs batched/sliced state).
  ``stack``/``slice`` turn python scalars into arrays, which changes leaf
  kinds; each lineage is self-consistent and checked strictly, never coerced.
- No implicit dtype conversion anywhere: shape/dtype/kind mismatches raise.
  In particular ``astype`` is never used to "fix" a mismatch.
- The foundation ``StateCodec`` is reused unchanged: the flattened env state is
  a plain dict of arrays/scalars/None, which is exactly the kind set the
  codec already supports.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .errors import SchemaMismatchError
from .state_codec import EncodedState, StateBundle, StateCodec


FLAT_ENV_STATE_VERSION = "simulator_frontier.flat-env-state/v1"

_NONE_LEAF = lambda x: x is None  # noqa: E731 - documented is_leaf predicate

# FlattenKey only exists in newer jax versions; resolve defensively.
_FLATTEN_KEY = getattr(jax.tree_util, "FlattenKey", None)


@dataclass(frozen=True)
class LeafSpec:
    """Structural identity of one flattened leaf (no values)."""

    kind: str  # "array" | "scalar" | "none"
    shape: tuple[int, ...] | None = None  # arrays only (0-dim stays ())
    dtype: str | None = None  # arrays only, str(np.dtype)
    scalar_type: str | None = None  # "bool" | "int" | "float" | "str"


@dataclass(frozen=True)
class EnvStateTemplate:
    """Lineage-bound restore key for flattened env states.

    ``leaf_paths`` is in treedef order; ``leaf_specs`` is aligned with it.
    """

    env_state_type: str  # e.g. "minicraftax.craftax_state.EnvState"
    treedef_fingerprint: str
    leaf_paths: tuple[str, ...]
    leaf_specs: tuple[LeafSpec, ...]
    treedef: Any


def _type_name(state: Any) -> str:
    return f"{type(state).__module__}.{type(state).__qualname__}"


def _flatten_with_paths(state: Any) -> tuple[list, list]:
    """Version-tolerant tree_flatten_with_path.

    jax 0.4.35 returns ``(list[(keypath, leaf)], treedef)``; newer jax returns
    ``(list[keypath], list[leaf])``.  Detect by the second element's type.
    """
    first, second = jax.tree_util.tree_flatten_with_path(state, is_leaf=_NONE_LEAF)
    if isinstance(second, list):
        return list(first), list(second)
    keypaths = [pair[0] for pair in first]
    leaves = [pair[1] for pair in first]
    return keypaths, leaves


def _keypath_str(keypath: Sequence[Any]) -> str:
    parts: list[str] = []
    for key in keypath:
        if isinstance(key, jax.tree_util.GetAttrKey):
            parts.append("." + key.name)
        elif isinstance(key, jax.tree_util.SequenceKey):
            parts.append(f"[{key.idx}]")
        elif isinstance(key, jax.tree_util.DictKey):
            parts.append("." + str(key.key))
        elif _FLATTEN_KEY is not None and isinstance(key, _FLATTEN_KEY):
            parts.append("." + str(key.key))
        else:
            raise SchemaMismatchError(f"unsupported pytree key type: {type(key).__name__}")
    text = "".join(parts)
    return text[1:] if text.startswith(".") else text


def _is_array_like(value: Any) -> bool:
    return isinstance(value, np.ndarray) or (hasattr(value, "shape") and hasattr(value, "dtype"))


def _leaf_spec(value: Any) -> LeafSpec:
    if value is None:
        return LeafSpec(kind="none")
    if _is_array_like(value):
        arr = np.asarray(value)
        return LeafSpec(kind="array", shape=tuple(int(x) for x in arr.shape), dtype=str(arr.dtype))
    if isinstance(value, bool):  # bool before int: bool is an int subclass
        return LeafSpec(kind="scalar", scalar_type="bool")
    if isinstance(value, int):
        return LeafSpec(kind="scalar", scalar_type="int")
    if isinstance(value, float):
        return LeafSpec(kind="scalar", scalar_type="float")
    if isinstance(value, str):
        return LeafSpec(kind="scalar", scalar_type="str")
    raise SchemaMismatchError(f"unsupported env state leaf type: {type(value).__name__}")


def _normalize_leaf(value: Any) -> Any:
    if value is None:
        return None
    if _is_array_like(value):
        return np.asarray(value)  # dtype preserved; 0-dim stays 0-dim
    if isinstance(value, (bool, int, float, str)):
        return value
    raise SchemaMismatchError(f"unsupported env state leaf type: {type(value).__name__}")


def _treedef_fingerprint(treedef: Any, paths: Sequence[str], specs: Mapping[str, LeafSpec]) -> str:
    canonical = {
        "treedef_repr": repr(treedef),
        "leaf_paths": list(paths),
        "leaf_specs": [
            {"kind": specs[p].kind, "shape": list(specs[p].shape) if specs[p].shape is not None else None,
             "dtype": specs[p].dtype, "scalar_type": specs[p].scalar_type}
            for p in paths
        ],
    }
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_template(reference_state: Any, *, env_state_type: str | None = None) -> EnvStateTemplate:
    """Build a lineage template from a reference state of the same lineage."""
    treedef = jax.tree_util.tree_structure(reference_state, is_leaf=_NONE_LEAF)
    keypaths, leaves = _flatten_with_paths(reference_state)
    paths = tuple(_keypath_str(kp) for kp in keypaths)
    if len(set(paths)) != len(paths):
        raise SchemaMismatchError("duplicate flattened leaf paths in reference state")
    specs = tuple(_leaf_spec(v) for v in leaves)
    spec_map = dict(zip(paths, specs))
    fingerprint = _treedef_fingerprint(treedef, paths, spec_map)
    return EnvStateTemplate(
        env_state_type=env_state_type or _type_name(reference_state),
        treedef_fingerprint=fingerprint,
        leaf_paths=paths,
        leaf_specs=specs,
        treedef=treedef,
    )


def flatten_env_state(state: Any) -> dict:
    """Flatten an env state pytree into a JSON-friendly dict (None leaves kept)."""
    keypaths, leaves = _flatten_with_paths(state)
    paths = tuple(_keypath_str(kp) for kp in keypaths)
    if len(set(paths)) != len(paths):
        raise SchemaMismatchError("duplicate flattened leaf paths")
    treedef = jax.tree_util.tree_structure(state, is_leaf=_NONE_LEAF)
    specs = {p: _leaf_spec(v) for p, v in zip(paths, leaves)}
    return {
        "flat_version": FLAT_ENV_STATE_VERSION,
        "env_state_type": _type_name(state),
        "treedef_fingerprint": _treedef_fingerprint(treedef, paths, specs),
        "leaf_paths": list(paths),
        "leaves": {p: _normalize_leaf(v) for p, v in zip(paths, leaves)},
    }


def _restore_leaf(path: str, value: Any, spec: LeafSpec) -> Any:
    if spec.kind == "none":
        if value is not None:
            raise SchemaMismatchError(f"leaf {path}: expected None, got {type(value).__name__}")
        return None
    if spec.kind == "array":
        if value is None or not _is_array_like(value):
            raise SchemaMismatchError(f"leaf {path}: expected array, got {type(value).__name__}")
        arr = np.asarray(value)
        if str(arr.dtype) != spec.dtype:
            raise SchemaMismatchError(f"leaf {path}: dtype mismatch {arr.dtype} != {spec.dtype} (no astype)")
        if tuple(int(x) for x in arr.shape) != spec.shape:
            raise SchemaMismatchError(f"leaf {path}: shape mismatch {arr.shape} != {spec.shape}")
        return jnp.asarray(arr)
    # scalar kinds
    if value is None or _is_array_like(value):
        raise SchemaMismatchError(f"leaf {path}: expected scalar, got {type(value).__name__}")
    expected = spec.scalar_type
    if expected == "bool" and not isinstance(value, bool):
        raise SchemaMismatchError(f"leaf {path}: expected bool scalar, got {type(value).__name__}")
    if expected == "int" and (not isinstance(value, int) or isinstance(value, bool)):
        raise SchemaMismatchError(f"leaf {path}: expected int scalar, got {type(value).__name__}")
    if expected == "float" and (not isinstance(value, float) or isinstance(value, bool)):
        raise SchemaMismatchError(f"leaf {path}: expected float scalar, got {type(value).__name__}")
    if expected == "str" and not isinstance(value, str):
        raise SchemaMismatchError(f"leaf {path}: expected str scalar, got {type(value).__name__}")
    return value


def unflatten_env_state(flat: Mapping[str, Any], template: EnvStateTemplate) -> Any:
    """Strictly rebuild the pytree; every mismatch raises SchemaMismatchError."""
    if not isinstance(flat, Mapping):
        raise SchemaMismatchError("flat env state must be a mapping")
    if flat.get("flat_version") != FLAT_ENV_STATE_VERSION:
        raise SchemaMismatchError(
            f"flat_version mismatch: {flat.get('flat_version')!r} != {FLAT_ENV_STATE_VERSION!r}")
    if flat.get("env_state_type") != template.env_state_type:
        raise SchemaMismatchError(
            f"env_state_type mismatch: {flat.get('env_state_type')!r} != {template.env_state_type!r}")
    if flat.get("treedef_fingerprint") != template.treedef_fingerprint:
        raise SchemaMismatchError("treedef_fingerprint mismatch (env state class/layout drift)")
    leaves_map = flat.get("leaves")
    if not isinstance(leaves_map, Mapping):
        raise SchemaMismatchError("flat env state missing leaves mapping")
    declared = flat.get("leaf_paths")
    if not isinstance(declared, (list, tuple)):
        raise SchemaMismatchError("flat env state missing leaf_paths")
    template_paths = list(template.leaf_paths)
    if sorted(declared) != sorted(template_paths) or len(declared) != len(template_paths):
        missing = sorted(set(template_paths) - set(declared))
        extra = sorted(set(declared) - set(template_paths))
        raise SchemaMismatchError(f"leaf path set mismatch: missing={missing} extra={extra}")
    leaves = [_restore_leaf(p, leaves_map.get(p), spec) for p, spec in zip(template_paths, template.leaf_specs)]
    return jax.tree_util.tree_unflatten(template.treedef, leaves)


def make_state_bundle(state: Any, *, next_step_key: Any, previous_action: Any, previous_reward: Any,
                      wrapper_state: Mapping[str, Any] | None = None,
                      policy_memory: Any = None, history_reference: Any = None) -> StateBundle:
    """Bundle a captured env state with the runner key that continues stepping.

    ``next_step_key`` is the runner key r used as ``r, step_key = split(r)`` at
    the next step (NOT the engine-written ``state.state_rng``).
    """
    return StateBundle(
        env_state=flatten_env_state(state),
        env_rng=np.asarray(next_step_key),
        wrapper_state=dict(wrapper_state) if wrapper_state is not None else {"wrapper": "none", "autoreset": False},
        previous_action=previous_action,
        previous_reward=previous_reward,
        policy_memory=policy_memory,
        history_reference=history_reference,
    )


def encode_env_state(state: Any, *, next_step_key: Any, previous_action: Any, previous_reward: Any,
                     wrapper_state: Mapping[str, Any] | None = None,
                     policy_memory: Any = None, history_reference: Any = None,
                     codec: StateCodec | None = None) -> tuple[EncodedState, StateBundle]:
    """Encode a captured state, forwarding the mode-conditional memory fields.

    CC4 follow-up (P0-3): a capture that omits ``policy_memory`` /
    ``history_reference`` cannot back a SAVED_POLICY_MEMORY / HISTORY_BURN_IN
    production entry — callers MUST forward the live rollout memory here; the
    archive guard chain independently re-verifies the mode-conditional
    presence inside the encoded bundle before any production write.
    """
    codec = codec or StateCodec()
    bundle = make_state_bundle(state, next_step_key=next_step_key, previous_action=previous_action,
                               previous_reward=previous_reward, wrapper_state=wrapper_state,
                               policy_memory=policy_memory,
                               history_reference=history_reference)
    return codec.encode(bundle), bundle


def restore_env_state(encoded: EncodedState | Mapping[str, Any], template: EnvStateTemplate,
                      codec: StateCodec | None = None) -> StateBundle:
    """Decode then strictly unflatten; returns a bundle whose env_state is a pytree."""
    codec = codec or StateCodec()
    bundle = codec.decode(encoded)
    env_state = unflatten_env_state(bundle.env_state, template)
    return dataclasses.replace(bundle, env_state=env_state)


def slice_env_state(batched_state: Any, index: int) -> Any:
    """Select one env from a batched pytree (None leaves pass through)."""
    return jax.tree.map(lambda x: None if x is None else x[index], batched_state, is_leaf=_NONE_LEAF)


def stack_env_states(states: Sequence[Any]) -> Any:
    """Stack same-lineage env states into a batched pytree.

    Python scalars become 0-dim/1-dim arrays here by construction: that is the
    documented lineage change of batched states.
    """
    def _stack(*xs: Any) -> Any:
        nones = [x is None for x in xs]
        if any(nones):
            if not all(nones):
                raise SchemaMismatchError("cannot stack mixed None/array leaves across states")
            return None
        return jnp.stack([jnp.asarray(x) for x in xs])

    return jax.tree.map(_stack, *states, is_leaf=_NONE_LEAF)
