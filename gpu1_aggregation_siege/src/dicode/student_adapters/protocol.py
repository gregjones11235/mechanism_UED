"""StudentAdapter protocol: the single shared contract for mounting Students.

This is the ONLY adapter/loader/registry contract in the repo.  CC2/CC3 are
consumers of this contract; they must not fork a parallel framework.  The
protocol is deliberately framework-thin: implementations may use jax/numpy,
but the contract types here are plain python so that importing this module
never pulls in jax.

A StudentAdapter exposes a high-capability Student (policy + memory) to the
frontier machinery: read-only mounting, policy forward, identity gating, and
full-state save/restore for the R4c combined fresh-process proof.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from .identity import StudentIdentity


@dataclass(frozen=True)
class ObsSpec:
    """Observation contract: flat vector length as seen by the network."""

    shape: tuple[int, ...]  # e.g. (8335,)
    dtype: str = "float32"


@dataclass(frozen=True)
class ActionSpec:
    """Discrete action contract."""

    count: int  # e.g. 43 for Craftax
    source: str = "craftax.craftax.constants.Action"


@dataclass(frozen=True)
class MemoryFieldSpec:
    """One named memory tensor spec (shape may use None for the batch dim)."""

    shape: tuple[int | None, ...]  # None marks the batch dimension
    dtype: str = "float32"


@dataclass(frozen=True)
class MemorySpec:
    """The full policy-memory contract for a Student architecture family."""

    fields: Mapping[str, MemoryFieldSpec]
    mode: str = "PERSISTENT"  # PERSISTENT | RESET128 | WINDOW | NONE

    def field_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.fields))

    def spec_hash(self) -> str:
        """Canonical sha256 of the memory contract (fields + mode)."""
        payload = {
            "mode": self.mode,
            "fields": {
                name: {"shape": [None if d is None else int(d) for d in spec.shape],
                       "dtype": spec.dtype}
                for name, spec in sorted(self.fields.items())
            },
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CheckpointSpec:
    """What a checkpoint file/directory is expected to contain."""

    format: str  # one of checkpoint_codec.FORMAT_*
    params_sha256: str  # 64-hex expected params tree hash
    source_commit: str
    contains_optimizer: bool = False
    contains_rng: bool = False
    contains_memory: bool = False
    notes: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class StudentAdapter(Protocol):
    """Contract every Student mount must satisfy.  All methods are mandatory."""

    def identity(self) -> StudentIdentity:
        """Return the validated Student identity (raises if unavailable)."""
        ...

    def observation_spec(self) -> ObsSpec:
        ...

    def action_spec(self) -> ActionSpec:
        ...

    def memory_spec(self) -> MemorySpec:
        ...

    def checkpoint_spec(self) -> CheckpointSpec:
        ...

    def load_full_state(self, checkpoint_path: str, expected_identity: StudentIdentity) -> Mapping[str, Any]:
        """Load params/optimizer/rng/memory read-only; verify against expected_identity.

        Must raise (never guess) on any hash/format/identity mismatch.
        """
        ...

    def policy_step(self, params: Any, observation: Any, memory: Any,
                    previous_action: Any, previous_reward: Any, rng: Any,
                    deterministic: bool) -> Mapping[str, Any]:
        """One forward pass returning {action, new_memory, logits?}.  No updates."""
        ...

    def initial_memory(self, batch_size: int) -> Any:
        ...

    def validate_memory(self, memory: Any, batch_size: int) -> Mapping[str, Any]:
        """Return {'ok': bool, 'reasons': [...]}; never coerce a wrong memory."""
        ...

    def save_full_state(self, output_path: str, train_state: Any, metadata: Mapping[str, Any]) -> str:
        """Persist a full training state (params+optimizer+step+rrng+memory)."""
        ...

    def restore_full_state(self, checkpoint_path: str) -> Mapping[str, Any]:
        """Inverse of save_full_state; raises on malformed/foreign states."""
        ...
