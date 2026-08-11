"""Trajectory provenance contract (task §七).

Every trajectory admitted to the BA-CWM world model carries FULL provenance so
its origin is auditable and so formal-evaluation state can never be smuggled in:

    trajectory_id / trajectory_source / student_checkpoint_sha256 /
    taskparams_sha256 / environment_descriptor_sha256 /
    environment_lock_sha256 / rollout_runner_sha256 /
    symbolic_adapter_sha256 / action_registry_sha256 / event_schema_version /
    world_seed / episode_id / generator_round / created_at

All SHA fields are full 64-char lowercase hex digests (``validate_sha256_hex``);
placeholders are rejected. ``provenance_hash`` is the canonical-JSON sha256 over
the record minus itself (tamper detection, mirrors symbolic_behavior_clip).

This layer is CONTRACT ONLY (pydantic): validation, serialization, hashing,
provenance (amendment §4). Source admissibility is enforced by
``source_policy.CwmSourcePolicy`` — a forbidden/unknown source fails closed.
"""
from __future__ import annotations

import hashlib
from typing import Optional

from pydantic import Field, field_validator, model_validator

from d052.bagr_ued.hashing import canonical_sha256
from d052.ba_cwm_ued import constants as C
from d052.schemas.common import CanonicalModel, validate_sha256_hex

PROVENANCE_SCHEMA_VERSION = "ba_cwm_ued.provenance.v1"

_HEX_SHA_FIELDS = (
    "student_checkpoint_sha256",
    "taskparams_sha256",
    "environment_descriptor_sha256",
    "environment_lock_sha256",
    "rollout_runner_sha256",
    "symbolic_adapter_sha256",
    "action_registry_sha256",
)


class ProvenanceError(Exception):
    SOURCE_NOT_ADMISSIBLE = "SOURCE_NOT_ADMISSIBLE"
    PROVENANCE_HASH_MISMATCH = "PROVENANCE_HASH_MISMATCH"

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"[{code}] {message}")


class TrajectoryProvenance(CanonicalModel):
    """Full, auditable provenance for one world-model training trajectory."""

    trajectory_id: str = Field(min_length=1)
    trajectory_source: str = Field(min_length=1)
    student_checkpoint_sha256: str
    taskparams_sha256: str
    environment_descriptor_sha256: str
    environment_lock_sha256: str
    rollout_runner_sha256: str
    symbolic_adapter_sha256: str
    action_registry_sha256: str
    event_schema_version: str = Field(min_length=1)
    world_seed: int = Field(ge=0)
    episode_id: str = Field(min_length=1)
    generator_round: int = Field(default=0, ge=0)
    #: data timestamp (trajectory data, NOT a certificate timestamp)
    created_at: str = Field(min_length=1)
    schema_version: str = PROVENANCE_SCHEMA_VERSION
    provenance_hash: str = ""

    @field_validator(*_HEX_SHA_FIELDS)
    @classmethod
    def _sha_fields(cls, v, info):
        return validate_sha256_hex(v, info.field_name)

    @model_validator(mode="after")
    def _hash(self) -> "TrajectoryProvenance":
        if not self.provenance_hash:
            object.__setattr__(self, "provenance_hash", provenance_hash(self))
        return self


def provenance_hash(p: TrajectoryProvenance) -> str:
    """Content hash over the provenance record minus the hash field itself."""
    dump = p.model_dump()
    dump.pop("provenance_hash", None)
    return canonical_sha256(dump)


def assert_provenance_hash_valid(p: TrajectoryProvenance) -> None:
    if provenance_hash(p) != p.provenance_hash:
        raise ProvenanceError(
            ProvenanceError.PROVENANCE_HASH_MISMATCH,
            f"trajectory {p.trajectory_id}: recorded provenance_hash does not "
            f"match content (tampering or stale hash)")


# ---------------------------------------------------------------------------
# Deterministic MOCK provenance (synthetic tests / dry run ONLY). Labeled mock;
# not real checkpoint / runner identity. Mirrors symbolic_behavior_clip's
# mock_clip_provenance convention.
# ---------------------------------------------------------------------------
def mock_provenance(*, trajectory_id: str = "synthetic_traj_0",
                    trajectory_source: str = "SYNTHETIC_TEST_TRACE",
                    world_seed: int = 0, episode_id: str = "ep_0",
                    generator_round: int = 0,
                    created_at: str = "1970-01-01T00:00:00Z"
                    ) -> TrajectoryProvenance:
    def _h(label: str) -> str:
        return hashlib.sha256(
            f"ba_cwm_ued.mock_provenance.{label}".encode("utf-8")).hexdigest()
    return TrajectoryProvenance(
        trajectory_id=trajectory_id,
        trajectory_source=trajectory_source,
        student_checkpoint_sha256=_h("student_checkpoint_sha256"),
        taskparams_sha256=_h("taskparams_sha256"),
        environment_descriptor_sha256=_h("environment_descriptor_sha256"),
        environment_lock_sha256=_h("environment_lock_sha256"),
        rollout_runner_sha256=_h("rollout_runner_sha256"),
        symbolic_adapter_sha256=_h("symbolic_adapter_sha256"),
        action_registry_sha256=_h("action_registry_sha256"),
        event_schema_version="cwm.event.v1",
        world_seed=world_seed,
        episode_id=episode_id,
        generator_round=generator_round,
        created_at=created_at)
