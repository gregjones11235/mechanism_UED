"""Stage 2c: review-cycle controller (gate -> six-role board).

Pure orchestration: decides via the invocation gate whether a window
opens, then runs the full board. Every LLM call goes through the
accounting ledger; a REUSE decision records ZERO calls. A void window
(incomplete outputs or all families vetoed) also means REUSE for the
downstream curriculum.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .board import ReviewWindow, WINDOW_STATUS_VOID, run_review_board
from .evidence import EvidenceSnapshot
from .invocation_gate import (
    GateDecision,
    GateState,
    evaluate_invocation_gate,
)


@dataclass(frozen=True)
class CycleOutcome:
    """Result of one review cycle."""

    decision: GateDecision
    window: Optional[ReviewWindow]
    reuse: bool  # True when no usable window was produced
    void_code: str  # "" or the window's void code


def run_review_cycle(
    llm: Any,
    *,
    window_id: str,
    gate_state: GateState,
    evidence: EvidenceSnapshot,
    ledger: Any,
) -> CycleOutcome:
    """Gate first; only a triggered gate opens a (fully accounted) window."""
    decision = evaluate_invocation_gate(gate_state)
    if not decision.triggered:
        return CycleOutcome(
            decision=decision, window=None, reuse=True, void_code=""
        )
    window = run_review_board(
        llm,
        window_id=window_id,
        session_idx=gate_state.session_idx,
        trigger_code=decision.code,
        evidence=evidence,
        ledger=ledger,
    )
    return CycleOutcome(
        decision=decision,
        window=window,
        reuse=(window.status == WINDOW_STATUS_VOID),
        void_code=window.void_code,
    )
