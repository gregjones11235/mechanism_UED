"""Candidate generation: validator + shared frozen pool.

Public surface:
    canonicalize_candidate     raw salvaged dict -> strict content-hashed Candidate
    validate_target_names      validate/canonicalize a target-name list
    CandidateValidationError   fail-closed candidate error
    build_pool                 assemble a frozen CandidatePool from raw candidates
    SharedFrozenPoolStore      no-overwrite persist/load/verify of shared pools
    PoolError                  fail-closed pool error
"""
from d052.generation.pool import (
    PoolError,
    SharedFrozenPoolStore,
    build_pool,
)
from d052.generation.validator import (
    CandidateValidationError,
    canonicalize_candidate,
    validate_target_names,
)

__all__ = [
    "PoolError",
    "SharedFrozenPoolStore",
    "build_pool",
    "CandidateValidationError",
    "canonicalize_candidate",
    "validate_target_names",
]
