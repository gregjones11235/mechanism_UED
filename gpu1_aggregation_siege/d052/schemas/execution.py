"""ExecutionMappingCertificate — proof that a selected candidate was mapped to a
real Craftax training goal exactly as intended.

Encodes the full chain (task §Candidate->真实训练目标执行映射):

  candidate_id -> canonical name -> canonical id -> goal-vector index
              -> task spec -> 67-dim multi-hot -> Student obs(8335)
              -> training task id

``executed_as_intended`` may be True ONLY if every hard gate in ``gates`` passed;
the schema enforces this, so a certificate cannot claim success while any gate
failed (NO_RAW_DATA_NO_STRONG_CLAIM at the schema level).
"""
from __future__ import annotations

from typing import Dict, List

from pydantic import Field, model_validator

from d052.legacy.canonical_constants import (
    CONDITIONING_DIMENSION,
    CONDITIONING_TYPE,
    MAX_ACHIEVEMENT_VALUE,
    STUDENT_OBS_DIM,
)
from d052.schemas.common import CanonicalModel, validate_sha256_hex

#: The hard gates that must ALL pass for executed_as_intended=True.
REQUIRED_GATES = (
    "target_is_canonical",        # every target resolves to one of the 67
    "goal_vector_dim_67",         # multi-hot length == 67
    "goal_vector_index_aligned",  # index == canonical_id == enum value
    "student_obs_dim_8335",       # obs space matches the Student
    "no_silent_fallback",         # no default-goal / substitution occurred
    "task_compiled",              # candidate compiled to a real env/task spec
)


class ExecutionMappingCertificate(CanonicalModel):
    candidate_id: str = Field(min_length=1)
    chash: str
    canonical_names: List[str] = Field(min_length=1)
    canonical_ids: List[int] = Field(min_length=1)
    goal_vector_indices: List[int] = Field(min_length=1)
    goal_vector_dim: int
    goal_vector_ones: int = Field(ge=1)     # number of 1 bits == #distinct targets
    student_obs_dim: int
    conditioning_type: str
    conditioning_dimension: int
    task_spec_hash: str
    training_task_id: str = Field(min_length=1)
    gates: Dict[str, bool]
    executed_as_intended: bool
    notes: str = ""

    @model_validator(mode="after")
    def _validate_chain(self) -> "ExecutionMappingCertificate":
        validate_sha256_hex(self.chash, "chash")
        validate_sha256_hex(self.task_spec_hash, "task_spec_hash")

        n = len(self.canonical_names)
        if len(self.canonical_ids) != n or len(self.goal_vector_indices) != n:
            raise ValueError(
                "CHAIN_LENGTH_MISMATCH: names/ids/indices differ in length")
        if self.canonical_ids != self.goal_vector_indices:
            raise ValueError(
                "INDEX_MISALIGNED: goal_vector_indices != canonical_ids "
                "(canonical_id must equal goal_vector_index)")
        for cid in self.canonical_ids:
            if not (0 <= cid <= MAX_ACHIEVEMENT_VALUE):
                raise ValueError(f"CANONICAL_ID_OUT_OF_RANGE: {cid}")
        if len(set(self.canonical_ids)) != n:
            raise ValueError("DUPLICATE_CANONICAL_ID")

        if self.goal_vector_dim != CONDITIONING_DIMENSION:
            raise ValueError(
                f"GOAL_VECTOR_DIM: expected {CONDITIONING_DIMENSION}, "
                f"got {self.goal_vector_dim}")
        if self.goal_vector_ones != n:
            raise ValueError(
                f"GOAL_VECTOR_ONES: expected {n}, got {self.goal_vector_ones}")
        if self.student_obs_dim != STUDENT_OBS_DIM:
            raise ValueError(
                f"STUDENT_OBS_DIM: expected {STUDENT_OBS_DIM}, "
                f"got {self.student_obs_dim}")
        if self.conditioning_type != CONDITIONING_TYPE:
            raise ValueError(
                f"CONDITIONING_TYPE: expected {CONDITIONING_TYPE!r}, "
                f"got {self.conditioning_type!r}")
        if self.conditioning_dimension != CONDITIONING_DIMENSION:
            raise ValueError(
                f"CONDITIONING_DIMENSION: expected {CONDITIONING_DIMENSION}, "
                f"got {self.conditioning_dimension}")

        # all required gates present
        missing = [g for g in REQUIRED_GATES if g not in self.gates]
        if missing:
            raise ValueError(f"MISSING_GATES: {missing}")

        all_pass = all(self.gates[g] for g in REQUIRED_GATES)
        if self.executed_as_intended and not all_pass:
            failed = [g for g in REQUIRED_GATES if not self.gates[g]]
            raise ValueError(
                f"EXECUTED_AS_INTENDED_REQUIRES_ALL_GATES: failed gates {failed}")
        if (not self.executed_as_intended) and all_pass:
            raise ValueError(
                "INCONSISTENT_CERTIFICATE: all gates pass but "
                "executed_as_intended=False")
        return self
