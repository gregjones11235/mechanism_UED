"""Cell lifecycle state machine (deterministic, fail-closed).

States:
    DRAFT -> VALIDATED -> READY -> AUTHORIZED -> RUNNING -> COMPLETE
    any active state may go to BLOCKED (gate fail) or FAILED (error);
    BLOCKED -> DRAFT (revise + resubmit). FAILED and COMPLETE are TERMINAL.

A failed cell is preserved as evidence and never mutated; to retry, register a NEW
cell with a new cell_id. This protects audit integrity (no rewriting a failed
record). Every transition is checked against ALLOWED_TRANSITIONS; an illegal move
is a hard error, never coerced.
"""
from __future__ import annotations

from enum import Enum


class CellState(str, Enum):
    DRAFT = "DRAFT"
    VALIDATED = "VALIDATED"
    READY = "READY"
    AUTHORIZED = "AUTHORIZED"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


#: Active (non-terminal) states.
ACTIVE_STATES = frozenset({
    CellState.DRAFT, CellState.VALIDATED, CellState.READY,
    CellState.AUTHORIZED, CellState.RUNNING, CellState.BLOCKED,
})

#: Terminal states (no outgoing transitions).
TERMINAL_STATES = frozenset({CellState.COMPLETE, CellState.FAILED})

ALLOWED_TRANSITIONS = {
    CellState.DRAFT: frozenset({CellState.VALIDATED, CellState.BLOCKED,
                                CellState.FAILED}),
    CellState.VALIDATED: frozenset({CellState.READY, CellState.BLOCKED,
                                    CellState.FAILED}),
    CellState.READY: frozenset({CellState.AUTHORIZED, CellState.BLOCKED,
                                CellState.FAILED}),
    CellState.AUTHORIZED: frozenset({CellState.RUNNING, CellState.BLOCKED,
                                     CellState.FAILED}),
    CellState.RUNNING: frozenset({CellState.COMPLETE, CellState.FAILED}),
    CellState.BLOCKED: frozenset({CellState.DRAFT}),   # revise + resubmit
    CellState.FAILED: frozenset(),                     # terminal
    CellState.COMPLETE: frozenset(),                   # terminal
}


def can_transition(src: CellState, dst: CellState) -> bool:
    return dst in ALLOWED_TRANSITIONS[src]


def assert_transition(src: CellState, dst: CellState) -> None:
    if not can_transition(src, dst):
        raise ValueError(
            f"ILLEGAL_TRANSITION: {src.value} -> {dst.value} not permitted; "
            f"allowed from {src.value}: "
            f"{sorted(s.value for s in ALLOWED_TRANSITIONS[src])}")
