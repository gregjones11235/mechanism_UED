"""C6 tests: review-cycle controller (gate -> board, fully accounted)."""
from dicode.teachers.e1_formal import board as B
from dicode.teachers.e1_formal import invocation_gate as G
from dicode.teachers.e1_formal.accounting import LLMCallLedger
from dicode.teachers.e1_formal.controller import run_review_cycle

# reuse the board test fixtures (same directory; pytest prepend import)
from test_board import _build_store, _evidence  # noqa: E402

import dicode.teachers.e1_formal.llm_client as LC


def _gate_state(**over):
    base = {
        "session_idx": 5,
        "is_first_window": False,
        "capability_shift": False,
        "new_failure_pattern": False,
        "interventions_exhausted": False,
        "stagnation": False,
        "forgetting_regression": False,
        "exploration_slot_available": False,
        "curriculum_drift": False,
        # round-3 P0-3: signal provenance binding (64 lowercase hex)
        "signals_binding_hash": "0" * 64,
    }
    base.update(over)
    return G.build_gate_state(base, "t")


class TestReusePath:
    def test_reuse_records_zero_calls(self):
        ledger = LLMCallLedger()
        outcome = run_review_cycle(
            LC.ReplayLLMClient({}, "t"),
            window_id="w01",
            gate_state=_gate_state(),
            evidence=_evidence(),
            ledger=ledger,
        )
        assert outcome.reuse is True
        assert outcome.window is None
        assert outcome.decision.triggered is False
        assert outcome.decision.code == G.REUSE
        counts = ledger.reconcile()
        assert counts["G1"] == 0
        assert counts["board_calls"] == 0
        assert ledger.to_records() == ()


class TestTriggeredPath:
    def test_triggered_window_runs_full_board(self):
        evidence = _evidence()
        # the controller feeds the gate state's session_idx and the
        # decision code into the board context (round-3 P0-1 binding)
        store = _build_store(evidence, session_idx=5)
        ledger = LLMCallLedger()
        outcome = run_review_cycle(
            LC.ReplayLLMClient(store, "t"),
            window_id="w01",
            gate_state=_gate_state(is_first_window=True),
            evidence=evidence,
            ledger=ledger,
        )
        assert outcome.decision.triggered is True
        assert outcome.decision.code == G.FIRST_WINDOW
        assert outcome.reuse is False
        assert outcome.window is not None
        assert outcome.window.status == B.WINDOW_STATUS_COMPLETE
        counts = ledger.reconcile()
        assert counts["G1"] == 1
        assert counts["board_calls"] == 6

    def test_void_window_yields_reuse(self):
        evidence = _evidence()
        store = _build_store(
            evidence,
            overrides={"behavior_auditor": {"bad": 1}},
            session_idx=5,
            trigger_code="STAGNATION",
        )
        ledger = LLMCallLedger()
        outcome = run_review_cycle(
            LC.ReplayLLMClient(store, "t"),
            window_id="w01",
            gate_state=_gate_state(stagnation=True),
            evidence=evidence,
            ledger=ledger,
        )
        assert outcome.decision.code == G.STAGNATION
        assert outcome.reuse is True
        assert outcome.void_code == "INCOMPLETE_REVIEW_WINDOW"
        assert outcome.window.status == B.WINDOW_STATUS_VOID

    def test_double_run_equality(self):
        evidence = _evidence()
        store = _build_store(evidence, session_idx=5)
        o1 = run_review_cycle(
            LC.ReplayLLMClient(store, "t"),
            window_id="w01",
            gate_state=_gate_state(is_first_window=True),
            evidence=evidence,
            ledger=LLMCallLedger(),
        )
        o2 = run_review_cycle(
            LC.ReplayLLMClient(store, "t"),
            window_id="w01",
            gate_state=_gate_state(is_first_window=True),
            evidence=evidence,
            ledger=LLMCallLedger(),
        )
        assert o1 == o2
        assert o1.window.window_hash == o2.window.window_hash
