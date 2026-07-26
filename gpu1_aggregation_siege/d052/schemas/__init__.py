"""Canonical_v2 D052 data schemas (pydantic, extra=forbid, fail-closed).

Inventory (10 core schemas; CellSpec/CellAuthorization land with the cell
registry in Commit 7):

  AchievementRef                 validated canonical achievement reference
  TaskParams                     the four D052 task knobs (positive, finite)
  Candidate                      salvaged D052 field set, strict + content-hashed
  CandidatePool                  shared frozen pool (pool_hash tamper-evident)
  RoleJudgment                   per-candidate scoring-role output (tutor/critic/explorer)
  NormalizedEntry / NormalizedRoleScores   rank_percentile_v1 output shape
  SelectorConfig                 unified selector configuration
  SelectionResult                replayable selection manifest
  ExecutionMappingCertificate    candidate -> real training goal proof
  RunConfig                      top-level run config (protocol_version REQUIRED)
"""
from d052.schemas.achievements import AchievementRef
from d052.schemas.candidate import (
    MAX_TARGET_ACHIEVEMENTS,
    Candidate,
    CandidatePool,
    TaskParams,
    compute_candidate_chash,
    compute_legacy_short_id,
)
from d052.schemas.execution import (
    REQUIRED_GATES,
    ExecutionMappingCertificate,
)
from d052.schemas.roles import (
    HEADLINE_SCORE_KEY,
    NormalizedEntry,
    NormalizedRoleScores,
    RoleName,
    RoleJudgment,
    ScoringRole,
)
from d052.schemas.run_config import RunConfig
from d052.schemas.selector import (
    CriticPolicy,
    SelectionStatus,
    SelectorConfig,
    SelectionResult,
    SelectorType,
    compute_selection_hash,
)

__all__ = [
    "AchievementRef",
    "MAX_TARGET_ACHIEVEMENTS",
    "Candidate",
    "CandidatePool",
    "TaskParams",
    "compute_candidate_chash",
    "compute_legacy_short_id",
    "REQUIRED_GATES",
    "ExecutionMappingCertificate",
    "HEADLINE_SCORE_KEY",
    "NormalizedEntry",
    "NormalizedRoleScores",
    "RoleName",
    "RoleJudgment",
    "ScoringRole",
    "RunConfig",
    "CriticPolicy",
    "SelectionStatus",
    "SelectorConfig",
    "SelectionResult",
    "SelectorType",
    "compute_selection_hash",
]
