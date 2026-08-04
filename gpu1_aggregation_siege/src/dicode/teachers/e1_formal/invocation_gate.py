"""Stage 2a: deterministic review-window invocation gate.

A review window (with its full six-role board) is opened only when at
least one trigger condition holds; otherwise the teacher REUSEs the
previous tasks and records ZERO LLM calls. Evaluation is a pure
function of the gate state; the trigger codes are checked in a fixed
priority order so the outcome is deterministic.

Trigger conditions (priority order):

1. FIRST_WINDOW               — no previous review window this run;
2. CAPABILITY_SHIFT           — the Student capability profile moved;
3. NEW_FAILURE_PATTERN        — a failure pattern not seen before;
4. INTERVENTIONS_EXHAUSTED    — prior interventions ran out of steam;
5. STAGNATION                 — window metrics stagnated;
6. FORGETTING_REGRESSION      — a previously-held skill regressed;
7. EXPLORATION_SLOT_AVAILABLE — a scheduled exploration slot is due;
8. CURRICULUM_DRIFT           — batch composition drifted off-plan.

Anything else => REUSE (no window, no LLM calls).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

from .schemas import E1SchemaError

FIRST_WINDOW = "FIRST_WINDOW"
CAPABILITY_SHIFT = "CAPABILITY_SHIFT"
NEW_FAILURE_PATTERN = "NEW_FAILURE_PATTERN"
INTERVENTIONS_EXHAUSTED = "INTERVENTIONS_EXHAUSTED"
STAGNATION = "STAGNATION"
FORGETTING_REGRESSION = "FORGETTING_REGRESSION"
EXPLORATION_SLOT_AVAILABLE = "EXPLORATION_SLOT_AVAILABLE"
CURRICULUM_DRIFT = "CURRICULUM_DRIFT"
REUSE = "REUSE"

#: Fixed evaluation order; the first true condition wins.
GATE_TRIGGER_ORDER = (
    FIRST_WINDOW,
    CAPABILITY_SHIFT,
    NEW_FAILURE_PATTERN,
    INTERVENTIONS_EXHAUSTED,
    STAGNATION,
    FORGETTING_REGRESSION,
    EXPLORATION_SLOT_AVAILABLE,
    CURRICULUM_DRIFT,
)

#: trigger code -> gate-state boolean field name
_TRIGGER_FIELD_BY_CODE = {
    FIRST_WINDOW: "is_first_window",
    CAPABILITY_SHIFT: "capability_shift",
    NEW_FAILURE_PATTERN: "new_failure_pattern",
    INTERVENTIONS_EXHAUSTED: "interventions_exhausted",
    STAGNATION: "stagnation",
    FORGETTING_REGRESSION: "forgetting_regression",
    EXPLORATION_SLOT_AVAILABLE: "exploration_slot_available",
    CURRICULUM_DRIFT: "curriculum_drift",
}


class InvocationGateError(E1SchemaError):
    """Fail-closed gate-state violation; ``code`` is greppable."""


class _GateCode:
    BAD_TYPE = "INVOCATION_GATE_BAD_TYPE"
    MISSING_FIELD = "INVOCATION_GATE_MISSING_FIELD"
    UNKNOWN_FIELD = "INVOCATION_GATE_UNKNOWN_FIELD"
    BAD_STEP = "INVOCATION_GATE_BAD_STEP"
    BAD_BINDING = "INVOCATION_GATE_BAD_BINDING"

#: 64 lowercase hex chars (sha256 hex digest)
_BINDING_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class GateState:
    """Immutable snapshot of every trigger condition for one session.

    ``signals_binding_hash`` binds the eight boolean signals to the
    inputs that produced them (session/cycle counters, evidence hash,
    previous window hash, reuse counter, threshold version) — see
    ``gate_signals.GateSignalReport.binding_hash``. It is REQUIRED:
    a gate state without its signal provenance is rejected.
    """

    session_idx: int
    is_first_window: bool
    capability_shift: bool
    new_failure_pattern: bool
    interventions_exhausted: bool
    stagnation: bool
    forgetting_regression: bool
    exploration_slot_available: bool
    curriculum_drift: bool
    signals_binding_hash: str


@dataclass(frozen=True)
class GateDecision:
    """Outcome of the gate: which condition triggered (or REUSE)."""

    triggered: bool
    code: str
    session_idx: int


def build_gate_state(raw: Mapping[str, object], context: str) -> GateState:
    """Consume a raw gate-state mapping fail-closed (no defaults)."""
    if not isinstance(raw, Mapping):
        raise InvocationGateError(
            _GateCode.BAD_TYPE,
            f"{context}: gate state must be a mapping, got "
            f"{type(raw).__name__}",
        )
    required = ("session_idx", "signals_binding_hash") + tuple(
        sorted(_TRIGGER_FIELD_BY_CODE.values())
    )
    for key in raw:
        if key not in required:
            raise InvocationGateError(
                _GateCode.UNKNOWN_FIELD,
                f"{context}: unknown gate-state field {key!r} (fail-closed)",
            )
    for key in required:
        if key not in raw:
            raise InvocationGateError(
                _GateCode.MISSING_FIELD,
                f"{context}: gate state missing field {key!r}",
            )
    session_idx = raw["session_idx"]
    if isinstance(session_idx, bool) or not isinstance(session_idx, int):
        raise InvocationGateError(
            _GateCode.BAD_STEP,
            f"{context}: session_idx must be an int, got {session_idx!r}",
        )
    if session_idx < 0:
        raise InvocationGateError(
            _GateCode.BAD_STEP,
            f"{context}: session_idx must be >= 0, got {session_idx}",
        )
    binding = raw["signals_binding_hash"]
    if not isinstance(binding, str) or not _BINDING_RE.fullmatch(binding):
        raise InvocationGateError(
            _GateCode.BAD_BINDING,
            f"{context}: signals_binding_hash must be a 64-char lowercase "
            f"hex string, got {binding!r} (no defaults)",
        )
    values = {}
    for code, field in _TRIGGER_FIELD_BY_CODE.items():
        value = raw[field]
        if not isinstance(value, bool):
            raise InvocationGateError(
                _GateCode.BAD_TYPE,
                f"{context}: gate field {field!r} must be a bool, got "
                f"{value!r} (no coercion)",
            )
        values[field] = value
    return GateState(
        session_idx=session_idx, signals_binding_hash=binding, **values
    )


def evaluate_invocation_gate(state: GateState) -> GateDecision:
    """Pure, deterministic gate evaluation; first true condition wins."""
    if not isinstance(state, GateState):
        raise InvocationGateError(
            _GateCode.BAD_TYPE,
            f"gate evaluation requires a GateState, got "
            f"{type(state).__name__}",
        )
    for code in GATE_TRIGGER_ORDER:
        field = _TRIGGER_FIELD_BY_CODE[code]
        if getattr(state, field):
            return GateDecision(
                triggered=True, code=code, session_idx=state.session_idx
            )
    return GateDecision(
        triggered=False, code=REUSE, session_idx=state.session_idx
    )
