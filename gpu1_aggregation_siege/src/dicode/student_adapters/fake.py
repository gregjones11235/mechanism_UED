"""FakeStudentAdapter: a numpy-only adapter implementing the full protocol.

For CONTRACT TESTS ONLY — explicitly labelled, never scientific content, no
jax dependency.  It provides a deterministic fake policy + fake memory tree so
the shared contract (identity gates, binding, save/restore round-trip, memory
validation) can be tested without any real network or checkpoint.
"""

from __future__ import annotations

import hashlib
import pickle
from typing import Any, Mapping

import numpy as np

from .identity import StudentIdentity, validate_identity
from .protocol import ActionSpec, CheckpointSpec, MemoryFieldSpec, MemorySpec, ObsSpec


FAKE_FORMAT = "FAKE_CONTRACT_ONLY"
_FAKE_SAVE_SCHEMA = "student_adapters.fake_full_state/v1"

OBS_DIM = 4
ACTION_COUNT = 3
MEMORY_DIM = 8


def fake_params_sha256(params: Mapping[str, Any]) -> str:
    """Numpy-only params hash: leaves concatenated in sorted-key order."""
    h = hashlib.sha256()
    for key in sorted(params):
        leaf = np.ascontiguousarray(np.asarray(params[key]))
        h.update(key.encode("utf-8"))
        h.update(b"|")
        h.update(str(leaf.dtype).encode("utf-8"))
        h.update(b"|")
        h.update(str(leaf.shape).encode("utf-8"))
        h.update(b"|")
        h.update(leaf.tobytes())
    return h.hexdigest()


class FakeStudentAdapter:
    """Implements StudentAdapter with a tiny deterministic numpy policy."""

    def __init__(self, candidate_id: str = "FAKE_STUDENT_CONTRACT_ONLY",
                 seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self._candidate_id = candidate_id
        self._params: dict[str, np.ndarray] = {
            "w": rng.standard_normal((OBS_DIM, ACTION_COUNT)).astype(np.float32),
            "b": np.zeros(ACTION_COUNT, dtype=np.float32),
        }
        self._params_sha = fake_params_sha256(self._params)

    # --- specs -------------------------------------------------------------
    def observation_spec(self) -> ObsSpec:
        return ObsSpec(shape=(OBS_DIM,), dtype="float32")

    def action_spec(self) -> ActionSpec:
        return ActionSpec(count=ACTION_COUNT, source="fake_contract")

    def memory_spec(self) -> MemorySpec:
        return MemorySpec(
            fields={"h": MemoryFieldSpec(shape=(None, MEMORY_DIM), dtype="float32")},
            mode="PERSISTENT",
        )

    def checkpoint_spec(self) -> CheckpointSpec:
        return CheckpointSpec(
            format=FAKE_FORMAT,
            params_sha256=self._params_sha,
            source_commit="fake-contract",
            contains_optimizer=True,
            contains_rng=True,
            contains_memory=True,
            notes={"schema": _FAKE_SAVE_SCHEMA},
        )

    def identity(self) -> StudentIdentity:
        return validate_identity(StudentIdentity(
            candidate_id=self._candidate_id,
            architecture_family="FAKE",
            checkpoint_format=FAKE_FORMAT,
            global_step=0,
            total_env_steps=0,
            params_sha256=self._params_sha,
            source_commit="fake-contract",
            observation_shape=(OBS_DIM,),
            action_count=ACTION_COUNT,
            memory_spec_hash=self.memory_spec().spec_hash(),
            extras={"label": "SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT"},
        ))

    # --- memory --------------------------------------------------------------
    def initial_memory(self, batch_size: int) -> dict[str, np.ndarray]:
        if int(batch_size) <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        return {"h": np.zeros((int(batch_size), MEMORY_DIM), dtype=np.float32)}

    def validate_memory(self, memory: Any, batch_size: int) -> Mapping[str, Any]:
        reasons: list[str] = []
        if not isinstance(memory, dict):
            return {"ok": False, "reasons": ["memory is not a dict"]}
        if set(memory.keys()) != {"h"}:
            reasons.append(f"memory keys {sorted(memory.keys())} != ['h']")
        else:
            h = np.asarray(memory["h"])
            if h.ndim != 2 or h.shape[0] != int(batch_size) or h.shape[1] != MEMORY_DIM:
                reasons.append(f"memory h shape {h.shape} != ({int(batch_size)}, {MEMORY_DIM})")
            if h.dtype != np.float32:
                reasons.append(f"memory h dtype {h.dtype} != float32")
        return {"ok": not reasons, "reasons": reasons}

    # --- policy --------------------------------------------------------------
    def policy_step(self, params: Any, observation: Any, memory: Any,
                    previous_action: Any, previous_reward: Any, rng: Any,
                    deterministic: bool) -> Mapping[str, Any]:
        check = self.validate_memory(memory, batch_size=np.asarray(observation).shape[0]
                                     if np.asarray(observation).ndim > 1 else 1)
        if not check["ok"]:
            raise ValueError(f"invalid memory for policy_step: {check['reasons']}")
        obs = np.asarray(observation, dtype=np.float32)
        single = obs.ndim == 1
        if single:
            obs = obs[None, :]
        logits = obs @ params["w"] + params["b"]
        if deterministic:
            action = np.argmax(logits, axis=-1).astype(np.int32)
        else:
            if rng is None:
                raise ValueError("stochastic policy_step requires rng (numpy Generator)")
            action = np.array([rng.choice(ACTION_COUNT, p=_softmax_row(row))
                               for row in logits], dtype=np.int32)
        h = np.asarray(memory["h"])
        obs_mean = np.broadcast_to(obs.mean(axis=-1, keepdims=True), h.shape)
        new_memory = {"h": np.tanh(h + 0.01 * obs_mean).astype(np.float32)}
        return {"action": action[0] if single else action,
                "new_memory": new_memory,
                "logits": logits[0] if single else logits}

    # --- full state ----------------------------------------------------------
    def save_full_state(self, output_path: str, train_state: Any, metadata: Mapping[str, Any]) -> str:
        if not isinstance(train_state, dict):
            raise ValueError("fake train_state must be a dict")
        for key in ("params", "optimizer", "global_step", "rng_state", "memory"):
            if key not in train_state:
                raise ValueError(f"fake train_state missing {key!r}")
        payload = {"schema": _FAKE_SAVE_SCHEMA,
                   "train_state": dict(train_state),
                   "metadata": dict(metadata),
                   "params_sha256": fake_params_sha256(train_state["params"])}
        with open(output_path, "wb") as fh:
            pickle.dump(payload, fh, protocol=4)
        return output_path

    def restore_full_state(self, checkpoint_path: str) -> Mapping[str, Any]:
        with open(checkpoint_path, "rb") as fh:
            payload = pickle.load(fh)
        if not isinstance(payload, dict) or payload.get("schema") != _FAKE_SAVE_SCHEMA:
            raise ValueError(f"{checkpoint_path!r} is not a fake full-state file")
        train_state = payload["train_state"]
        if fake_params_sha256(train_state["params"]) != payload["params_sha256"]:
            raise ValueError("fake full-state params hash mismatch (tampered)")
        return {"train_state": train_state, "metadata": payload["metadata"],
                "params_sha256": payload["params_sha256"]}

    def load_full_state(self, checkpoint_path: str, expected_identity: StudentIdentity) -> Mapping[str, Any]:
        restored = self.restore_full_state(checkpoint_path)
        if restored["params_sha256"] != expected_identity.params_sha256:
            raise ValueError(
                f"loaded params sha {restored['params_sha256'][:16]}… != expected identity "
                f"{expected_identity.params_sha256[:16]}… (never guess)")
        return restored


def _softmax_row(row: np.ndarray) -> np.ndarray:
    x = row - row.max()
    e = np.exp(x)
    return e / e.sum()
