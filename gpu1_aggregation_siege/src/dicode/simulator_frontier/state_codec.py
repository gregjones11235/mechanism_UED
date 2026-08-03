"""Deterministic, pickle-free state bundle encoding."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .errors import SchemaMismatchError


SCHEMA_VERSION = "simulator_frontier.state/v1"


@dataclass(frozen=True)
class StateBundle:
    env_state: Any
    env_rng: Any
    wrapper_state: Any
    previous_action: Any
    previous_reward: Any
    policy_memory: Any = None
    history_reference: Any = None


@dataclass(frozen=True)
class EncodedState:
    schema_version: str
    tree_definition: Any
    arrays: tuple[Mapping[str, Any], ...]
    scalar_metadata: Mapping[str, Any]
    payload: Mapping[str, Any]
    payload_hash: str
    codec_version: str = "state-codec-v1"


def _encode(value: Any) -> Any:
    if isinstance(value, np.ndarray) or (hasattr(value, "shape") and hasattr(value, "dtype")):
        arr = np.asarray(value)
        return {"kind": "array", "dtype": str(arr.dtype), "shape": list(arr.shape),
                "data": base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")}
    if isinstance(value, np.generic):
        return {"kind": "scalar", "dtype": str(value.dtype), "value": value.item()}
    if isinstance(value, Mapping):
        return {"kind": "mapping", "items": [[str(k), _encode(v)] for k, v in sorted(value.items(), key=lambda x: str(x[0]))]}
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_encode(v) for v in value]}
    if isinstance(value, list):
        return {"kind": "list", "items": [_encode(v) for v in value]}
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"kind": "scalar", "dtype": type(value).__name__, "value": value}
    raise TypeError(f"unsupported state value type: {type(value).__name__}")


def _decode(value: Any) -> Any:
    kind = value.get("kind")
    if kind == "array":
        raw = base64.b64decode(value["data"].encode("ascii"), validate=True)
        arr = np.frombuffer(raw, dtype=np.dtype(value["dtype"]))
        expected = tuple(int(x) for x in value["shape"])
        if int(arr.size) != int(np.prod(expected, dtype=np.int64)):
            raise SchemaMismatchError("array payload size does not match shape")
        return arr.reshape(expected).copy()
    if kind == "scalar":
        return value.get("value")
    if kind == "mapping":
        return {k: _decode(v) for k, v in value["items"]}
    if kind == "tuple":
        return tuple(_decode(v) for v in value["items"])
    if kind == "list":
        return [_decode(v) for v in value["items"]]
    raise SchemaMismatchError(f"unknown encoded value kind: {kind!r}")


class StateCodec:
    def encode(self, state_bundle: StateBundle) -> EncodedState:
        if not isinstance(state_bundle, StateBundle):
            raise TypeError("encode expects StateBundle")
        payload = {name: _encode(getattr(state_bundle, name)) for name in (
            "env_state", "env_rng", "wrapper_state", "previous_action", "previous_reward",
            "policy_memory", "history_reference")}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        arrays = []
        def collect(node: Any, path: str = "") -> None:
            if isinstance(node, Mapping) and node.get("kind") == "array":
                arrays.append({"path": path, "shape": node["shape"], "dtype": node["dtype"]})
            elif isinstance(node, Mapping):
                for k, v in node.items(): collect(v, f"{path}.{k}" if path else str(k))
            elif isinstance(node, list):
                for i, v in enumerate(node): collect(v, f"{path}[{i}]")
        collect(payload)
        return EncodedState(SCHEMA_VERSION, "StateBundle/v1", tuple(arrays),
                            {"field_names": list(payload)}, payload,
                            hashlib.sha256(canonical.encode()).hexdigest())

    def decode(self, encoded_state: EncodedState | Mapping[str, Any]) -> StateBundle:
        data = encoded_state
        if isinstance(data, Mapping):
            data = EncodedState(**data)
        if not isinstance(data, EncodedState) or data.schema_version != SCHEMA_VERSION:
            raise SchemaMismatchError("unsupported or malformed state schema")
        canonical = json.dumps(data.payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        if hashlib.sha256(canonical.encode()).hexdigest() != data.payload_hash:
            raise SchemaMismatchError("state payload hash mismatch")
        vals = {name: _decode(data.payload[name]) for name in data.scalar_metadata.get("field_names", [])}
        required = {"env_state", "env_rng", "wrapper_state", "previous_action", "previous_reward",
                    "policy_memory", "history_reference"}
        if set(vals) != required:
            raise SchemaMismatchError("state field set mismatch")
        return StateBundle(**vals)
