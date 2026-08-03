"""G5: LLMCallLedger — N1 = 6*G1 + T1 + K1 + F1, fail-closed reconcile.

Counting rules (supervisor gate G5):

* G1 — number of TRIGGERED review windows. Each triggered window runs
  exactly the six board roles once; board calls are recorded for all
  six roles even when the window is later invalidated (the calls were
  made; their outputs are discarded);
* T1 — TaskGenerator calls. E1 has no TaskGenerator role; this round
  reconcile fails closed if T1 != 0;
* K1 — EnvCoder calls counted per ACTUAL UNIQUE artifact
  (artifact_id = spec x variant); duplicate productions of the same
  artifact are never double-counted;
* F1 — repair calls (re-invocations after a gate failure), counted
  SEPARATELY from K1. This round is single-pass (F1 == 0) but the
  counter slot exists and never mixes with K1.

No fixed per-window total is claimed anywhere: board calls belong to
windows, EnvCoder calls belong to artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .manifest import BOARD_ROLE_ORDER, ENVCODER_ROLE
from .schemas import E1SchemaError

LLM_ACCOUNTING_MISMATCH = "LLM_ACCOUNTING_MISMATCH"
LLM_BUDGET_EXCEEDED = "LLM_BUDGET_EXCEEDED"

KIND_BOARD = "BOARD"
KIND_ENVCODER = "ENVCODER"
KIND_REPAIR = "REPAIR"


@dataclass(frozen=True)
class LLMCallRecord:
    """One ledger entry (append-only)."""

    seq: int
    kind: str
    role: str
    window_id: str
    artifact_id: str  # "" for board calls


class LLMCallLedger:
    """Append-only ledger with a fail-closed reconcile against N1."""

    def __init__(self) -> None:
        self._records: List[LLMCallRecord] = []
        self._triggered_windows: List[str] = []
        self._board_calls: Dict[str, List[str]] = {}
        self._artifacts_produced: List[str] = []  # unique artifact ids, in order
        self._artifact_seen: set = set()
        self._repair_count: int = 0
        self._task_generator_calls: int = 0

    # -- recording ---------------------------------------------------------
    def record_window_open(self, window_id: str) -> None:
        if not isinstance(window_id, str) or not window_id:
            raise ValueError("window_id must be a non-empty str")
        if window_id in self._board_calls:
            raise E1SchemaError(
                LLM_ACCOUNTING_MISMATCH,
                f"window {window_id!r} opened twice",
            )
        self._triggered_windows.append(window_id)
        self._board_calls[window_id] = []

    def record_board_call(self, window_id: str, role: str) -> None:
        if window_id not in self._board_calls:
            raise E1SchemaError(
                LLM_ACCOUNTING_MISMATCH,
                f"board call for unopened window {window_id!r}",
            )
        if role not in BOARD_ROLE_ORDER:
            raise E1SchemaError(
                LLM_ACCOUNTING_MISMATCH,
                f"unknown board role {role!r}",
            )
        if role in self._board_calls[window_id]:
            raise E1SchemaError(
                LLM_ACCOUNTING_MISMATCH,
                f"board role {role!r} called twice in window {window_id!r}",
            )
        self._board_calls[window_id].append(role)
        self._records.append(
            LLMCallRecord(
                seq=len(self._records),
                kind=KIND_BOARD,
                role=role,
                window_id=window_id,
                artifact_id="",
            )
        )

    def record_envcoder_call(self, window_id: str, artifact_id: str) -> bool:
        """Record an EnvCoder artifact production; returns True if unique.

        Duplicate productions of the same artifact are recorded in the
        audit trail but NEVER double-counted toward K1.
        """
        if not isinstance(artifact_id, str) or not artifact_id:
            raise ValueError("artifact_id must be a non-empty str")
        unique = artifact_id not in self._artifact_seen
        if unique:
            self._artifact_seen.add(artifact_id)
            self._artifacts_produced.append(artifact_id)
        self._records.append(
            LLMCallRecord(
                seq=len(self._records),
                kind=KIND_ENVCODER,
                role=ENVCODER_ROLE,
                window_id=window_id,
                artifact_id=artifact_id,
            )
        )
        return unique

    def record_repair_call(self, window_id: str, artifact_id: str) -> None:
        """Repair calls are counted separately (F1), never merged into K1."""
        self._repair_count += 1
        self._records.append(
            LLMCallRecord(
                seq=len(self._records),
                kind=KIND_REPAIR,
                role=ENVCODER_ROLE,
                window_id=window_id,
                artifact_id=artifact_id,
            )
        )

    def record_task_generator_call(self, window_id: str) -> None:
        """E1 has NO TaskGenerator; recorded only so reconcile can reject it."""
        self._task_generator_calls += 1
        self._records.append(
            LLMCallRecord(
                seq=len(self._records),
                kind="TASK_GENERATOR",
                role="task_generator",
                window_id=window_id,
                artifact_id="",
            )
        )

    # -- reporting ---------------------------------------------------------
    def counts(self) -> Dict[str, int]:
        board_calls = sum(len(v) for v in self._board_calls.values())
        g1 = len(self._triggered_windows)
        t1 = self._task_generator_calls
        k1 = len(self._artifacts_produced)
        f1 = self._repair_count
        return {
            "G1": g1,
            "T1": t1,
            "K1": k1,
            "F1": f1,
            "board_calls": board_calls,
            "N1": 6 * g1 + t1 + k1 + f1,
        }

    def reconcile(self, expect_no_task_generator: bool = True) -> Dict[str, int]:
        """Fail-closed check that the ledger obeys N1 = 6*G1 + T1 + K1 + F1."""
        c = self.counts()
        if c["board_calls"] != 6 * c["G1"]:
            raise E1SchemaError(
                LLM_ACCOUNTING_MISMATCH,
                f"board_calls={c['board_calls']} != 6*G1={6 * c['G1']}; "
                "every triggered window must run exactly the six board roles",
            )
        if expect_no_task_generator and c["T1"] != 0:
            raise E1SchemaError(
                LLM_ACCOUNTING_MISMATCH,
                f"T1={c['T1']} != 0 but E1 has no TaskGenerator this round",
            )
        # N1 identity (defensive; the formula is structural)
        if c["N1"] != 6 * c["G1"] + c["T1"] + c["K1"] + c["F1"]:
            raise E1SchemaError(
                LLM_ACCOUNTING_MISMATCH, "N1 identity violated"
            )
        return c

    def to_records(self) -> Tuple[Dict[str, object], ...]:
        """JSONL-ready audit records."""
        return tuple(
            {
                "seq": r.seq,
                "kind": r.kind,
                "role": r.role,
                "window_id": r.window_id,
                "artifact_id": r.artifact_id,
            }
            for r in self._records
        )
