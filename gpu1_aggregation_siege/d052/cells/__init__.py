"""Cell register / validate / prepare / authorize / on-demand launch.

A cell is the unit of per-cell authorization & launch. Lifecycle:
    DRAFT -> VALIDATED -> READY -> AUTHORIZED -> RUNNING -> COMPLETE
    (BLOCKED / FAILED on gate failure; FAILED & COMPLETE terminal).

Public surface:
    CellState / ALLOWED_TRANSITIONS / can_transition    the state machine
    CellSpec / IDENTITY_FIELDS                          content-addressed identity
    CellAuthorization / make_authorization              explicit per-cell grant
    CellRegistry / CellRecord / validate_cell_spec      lifecycle orchestration
    no_op_runner                                        the only no-training runner
    CellError                                           fail-closed errors

Registry root convention: ``<repo>/gpu1_aggregation_siege/configs/d052/cells/``
for committed templates/index (data), separate from this ``d052/cells/`` package
(code). Tests use a temp root. prepare/validate/status never launch; launch is
authorization-gated and structurally incapable of training under a no-training
authorization (D052_LONG_TRAINING_RUNS=0).
"""
from d052.cells.authorization import (
    LEGAL_SCOPES,
    SCOPE_NO_TRAINING,
    SCOPE_TRAINING,
    CellAuthorization,
    compute_authorization_hash,
    make_authorization,
)
from d052.cells.registry import (
    DENY_LEGACY_OUTPUT_PREFIXES,
    CellError,
    CellRecord,
    CellRegistry,
    HistoryEntry,
    no_op_runner,
    validate_cell_spec,
)
from d052.cells.spec import ENVIRONMENT_VERSION, IDENTITY_FIELDS, CellSpec
from d052.cells.states import (
    ACTIVE_STATES,
    ALLOWED_TRANSITIONS,
    TERMINAL_STATES,
    CellState,
    assert_transition,
    can_transition,
)

__all__ = [
    "ACTIVE_STATES",
    "ALLOWED_TRANSITIONS",
    "CellAuthorization",
    "CellError",
    "CellRecord",
    "CellRegistry",
    "CellSpec",
    "CellState",
    "DENY_LEGACY_OUTPUT_PREFIXES",
    "ENVIRONMENT_VERSION",
    "HistoryEntry",
    "IDENTITY_FIELDS",
    "LEGAL_SCOPES",
    "SCOPE_NO_TRAINING",
    "SCOPE_TRAINING",
    "TERMINAL_STATES",
    "assert_transition",
    "can_transition",
    "compute_authorization_hash",
    "make_authorization",
    "no_op_runner",
    "validate_cell_spec",
]
