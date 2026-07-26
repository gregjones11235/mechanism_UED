"""Deterministic candidate -> real-training-goal execution mapping.

This module turns a validated, selected ``Candidate`` plus a ``CompiledTaskSpec``
(what the real compiler/environment reports) into an audit-grade
``ExecutionMappingCertificate``. It is the enforcement point for
§Candidate->真实训练目标执行映射:

  candidate_id -> canonical name -> canonical id -> goal-vector index
              -> 67-dim multi-hot -> Student obs(8335) -> training task id

Discipline:
  * The certificate's stored dimensions/conditioning are the CANONICAL constants;
    the ``gates`` record whether the supplied compiled spec actually CONFORMED. So
    a non-conforming spec yields executed_as_intended=False (never a silent pass),
    while the certificate itself still validates against canonical_v2.
  * NO silent fallback: if a fallback/substitution is flagged, or the spec's
    conditioning/obs/goal-vector disagree with canonical, the relevant gate fails.
  * The candidate chash is RE-VERIFIED against its contents (defence in depth).
  * No training is launched here. This builds evidence only.
"""
from __future__ import annotations

import hashlib
import json
from typing import List, Optional

from pydantic import Field, field_validator, model_validator

from d052.achievements import REGISTRY
from d052.legacy.canonical_constants import (
    CONDITIONING_DIMENSION,
    CONDITIONING_TYPE,
    NUM_ACHIEVEMENTS,
    STUDENT_OBS_DIM,
)
from d052.schemas.candidate import Candidate, compute_candidate_chash
from d052.schemas.common import CanonicalModel, validate_sha256_hex
from d052.schemas.execution import REQUIRED_GATES, ExecutionMappingCertificate


class ExecutionMappingError(Exception):
    """Fail-closed mapping violation with a stable ``code``."""

    TASK_NOT_COMPILED = "TASK_NOT_COMPILED"
    CHASH_MISMATCH = "CHASH_MISMATCH"
    GOAL_VECTOR_NOT_MULTIHOT = "GOAL_VECTOR_NOT_MULTIHOT"

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"[{code}] {message}")


class CompiledTaskSpec(CanonicalModel):
    """What a real compiler/environment must report for one compiled candidate.

    ``compiled_goal_vector`` is the actual multi-hot the task was compiled with;
    the mapper checks it element-wise against the candidate's canonical vector.
    """

    training_task_id: str = Field(min_length=1)
    task_spec_hash: str
    student_obs_dim: int
    conditioning_type: str
    goal_vector_dim: int
    compiled_goal_vector: List[float]
    compiled: bool = True

    @field_validator("task_spec_hash")
    @classmethod
    def _hash_format(cls, v: str) -> str:
        return validate_sha256_hex(v, "task_spec_hash")

    @model_validator(mode="after")
    def _check_vector(self) -> "CompiledTaskSpec":
        if len(self.compiled_goal_vector) != self.goal_vector_dim:
            raise ValueError(
                f"GOAL_VECTOR_LEN: len(compiled_goal_vector)="
                f"{len(self.compiled_goal_vector)} != goal_vector_dim="
                f"{self.goal_vector_dim}")
        for i, v in enumerate(self.compiled_goal_vector):
            if float(v) not in (0.0, 1.0):
                raise ValueError(
                    f"NOT_MULTIHOT: compiled_goal_vector[{i}]={v} not in {{0,1}}")
        return self


def candidate_goal_vector(candidate: Candidate) -> List[float]:
    """The canonical 67-dim multi-hot for a candidate (deterministic)."""
    return REGISTRY.to_goal_vector(candidate.canonical_target_names)


def compute_task_spec_hash(candidate_chash: str, training_task_id: str,
                           goal_vector: List[float], student_obs_dim: int,
                           conditioning_type: str) -> str:
    """Deterministic content hash binding a compiled spec to its candidate."""
    payload = {
        "candidate_chash": candidate_chash,
        "training_task_id": training_task_id,
        "compiled_goal_vector": [float(x) for x in goal_vector],
        "student_obs_dim": student_obs_dim,
        "conditioning_type": conditioning_type,
    }
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def canonical_compiled_spec(candidate: Candidate,
                            training_task_id: str) -> CompiledTaskSpec:
    """Build a CONFORMING compiled spec for a candidate (used by tests and the
    authorization-gated training adapter). Deterministic task_spec_hash."""
    vec = candidate_goal_vector(candidate)
    return CompiledTaskSpec(
        training_task_id=training_task_id,
        task_spec_hash=compute_task_spec_hash(
            candidate.chash, training_task_id, vec, STUDENT_OBS_DIM,
            CONDITIONING_TYPE),
        student_obs_dim=STUDENT_OBS_DIM,
        conditioning_type=CONDITIONING_TYPE,
        goal_vector_dim=NUM_ACHIEVEMENTS,
        compiled_goal_vector=vec,
        compiled=True,
    )


def build_execution_certificate(
    candidate: Candidate,
    spec: Optional[CompiledTaskSpec],
    *,
    silent_fallback_occurred: bool = False,
    notes: str = "",
) -> ExecutionMappingCertificate:
    """Map a selected candidate to its real training goal and certify the chain.

    The certificate stores CANONICAL dimensions/conditioning; ``gates`` record
    whether ``spec`` conformed. executed_as_intended is True iff every gate passes.
    """
    if spec is None or not spec.compiled:
        raise ExecutionMappingError(
            ExecutionMappingError.TASK_NOT_COMPILED,
            f"candidate {candidate.task_id} has no compiled task spec; cannot "
            f"certify execution mapping (no silent substitution)")

    # re-verify candidate identity against its contents (defence in depth)
    expected_chash = compute_candidate_chash(
        candidate.task_id, candidate.canonical_target_names,
        candidate.task_params.model_dump())
    if candidate.chash != expected_chash:
        raise ExecutionMappingError(
            ExecutionMappingError.CHASH_MISMATCH,
            f"candidate {candidate.task_id} chash != recomputed content hash")

    # canonical chain (unknown target -> AchievementError, never silent)
    names = REGISTRY.canonicalize_targets(candidate.canonical_target_names)
    ids = [REGISTRY.canonical_id(n) for n in names]
    indices = list(ids)                       # canonical_id == goal_vector_index
    expected_vec = REGISTRY.to_goal_vector(names)

    gates = {
        "target_is_canonical": True,  # resolved via registry (else raised above)
        "goal_vector_dim_67": (spec.goal_vector_dim == NUM_ACHIEVEMENTS
                               and len(spec.compiled_goal_vector)
                               == NUM_ACHIEVEMENTS),
        "goal_vector_index_aligned": spec.compiled_goal_vector == expected_vec,
        "student_obs_dim_8335": spec.student_obs_dim == STUDENT_OBS_DIM,
        "no_silent_fallback": ((not silent_fallback_occurred)
                               and spec.conditioning_type == CONDITIONING_TYPE
                               and spec.goal_vector_dim == CONDITIONING_DIMENSION),
        "task_compiled": bool(spec.training_task_id) and spec.compiled,
    }
    # sanity: gate keys must be exactly the required gate set
    assert set(gates) == set(REQUIRED_GATES), "gate set drifted from REQUIRED_GATES"

    return ExecutionMappingCertificate(
        candidate_id=candidate.task_id,
        chash=candidate.chash,
        canonical_names=names,
        canonical_ids=ids,
        goal_vector_indices=indices,
        goal_vector_dim=NUM_ACHIEVEMENTS,
        goal_vector_ones=len(names),
        student_obs_dim=STUDENT_OBS_DIM,
        conditioning_type=CONDITIONING_TYPE,
        conditioning_dimension=CONDITIONING_DIMENSION,
        task_spec_hash=spec.task_spec_hash,
        training_task_id=spec.training_task_id,
        gates=gates,
        executed_as_intended=all(gates.values()),
        notes=notes,
    )
