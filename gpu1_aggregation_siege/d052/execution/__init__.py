"""Execution mapping: certify that a selected candidate maps to a real Craftax
training goal exactly as intended (canonical name -> id -> goal-vector index ->
67-dim multi-hot -> Student obs 8335 -> training task id).

Public surface:
    CompiledTaskSpec            contract a real compiler/env must report
    ExecutionMappingError       fail-closed mapping error
    build_execution_certificate candidate + spec -> ExecutionMappingCertificate
    canonical_compiled_spec     build a conforming spec (tests / training adapter)
    candidate_goal_vector       the canonical 67-dim multi-hot for a candidate
    compute_task_spec_hash      deterministic spec<->candidate binding hash
    REQUIRED_GATES              the gate set that gates executed_as_intended
"""
from d052.execution.mapper import (
    CompiledTaskSpec,
    ExecutionMappingError,
    build_execution_certificate,
    candidate_goal_vector,
    canonical_compiled_spec,
    compute_task_spec_hash,
)
from d052.schemas.execution import REQUIRED_GATES, ExecutionMappingCertificate

__all__ = [
    "CompiledTaskSpec",
    "ExecutionMappingError",
    "ExecutionMappingCertificate",
    "REQUIRED_GATES",
    "build_execution_certificate",
    "candidate_goal_vector",
    "canonical_compiled_spec",
    "compute_task_spec_hash",
]
