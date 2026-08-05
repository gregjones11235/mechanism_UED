"""P0-13 / P0-14 (§19 seam coverage): the post-run RealComputeLedger and
the training-budget semantics gate.

Contract under test:

* the ledger gathers the ACTUAL executed compute of a finished run (board
  calls, EnvCoder calls, LLM calls, probes, transitions, feedback records,
  anchors, training steps, checkpoint saves, verified round-trip passes);
* ``computation_match_status`` is COMPUTE_MATCH_PASS ONLY for a run that
  completed the full expected horizon — a run cut short (REQUEST_CONTROL
  stop or an early halt) is COMPUTE_MATCH_EXECUTION_INCOMPLETE and can
  never attest compute match;
* ``verify_against_config`` compares the actuals against the frozen
  compute-match budget field by field (COMPUTE_MATCH_BROKEN on drift);
* ledgers of different completed modes agree on every compute field
  (cross-mode equality);
* the training-budget SEMANTICS is a DIRECTOR decision — the two legal
  values (TOTAL_FROM_COMMON_INITIALIZATION /
  ADDITIONAL_FROM_PRETRAINED_CHECKPOINT) are not the default, and the
  longrun launcher refuses a launch while it is
  BLOCKED_WAITING_DIRECTOR_BUDGET_DECISION.

Fixtures are TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION: mock backends
and the deterministic symbolic runner — NO real LLM call, NO simulator
episode, and NO passing test flips a REAL_* flag.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.controller import FeedbackUEDController
from d052.feedback_llm_ued.real_compute_ledger import (
    COMPUTE_MATCH_EXECUTION_INCOMPLETE,
    COMPUTE_MATCH_PASS,
    TRAINING_BUDGET_SEMANTICS,
    RealComputeLedgerBlocked,
    build_real_compute_ledger,
)


def _load_longrun():
    path = (Path(__file__).resolve().parents[2] / "scripts"
            / "run_e2_longrun.py")
    spec = importlib.util.spec_from_file_location(
        "run_e2_longrun_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LONGRUN = _load_longrun()


def _run(mode: str, windows: int):
    ctl = FeedbackUEDController(mode)
    ctl.run(max_windows=windows)
    return ctl


class TestLedgerBuild:
    def test_positive_two_window_ledger(self):
        ctl = _run(C.MODE_NORMAL_FEEDBACK, 2)
        ledger = build_real_compute_ledger(ctl, expected_windows=2)
        assert ledger.mode == C.MODE_NORMAL_FEEDBACK
        assert ledger.windows_executed == 2
        assert ledger.board_calls_total == 12
        assert ledger.envcoder_calls_total == 2
        assert ledger.llm_calls_total == 14
        assert ledger.probe_calls_total > 0
        assert ledger.simulator_transitions_total > 0
        assert ledger.feedback_records_total == 128
        assert ledger.anchors_per_window == C.GLOBAL_ANCHOR_SLOTS
        assert ledger.training_steps_total == 2
        #: no training authorized this round: no checkpoint save/round-trip
        assert ledger.checkpoint_saves_total == 0
        assert ledger.checkpoint_round_trip_passes == 0
        assert ledger.computation_match_status == COMPUTE_MATCH_PASS

    def test_requires_a_finished_run(self):
        ctl = FeedbackUEDController(C.MODE_NORMAL_FEEDBACK)
        with pytest.raises(RealComputeLedgerBlocked,
                           match="REAL_COMPUTE_LEDGER_REQUIRES_RUN"):
            build_real_compute_ledger(ctl, expected_windows=2)

    def test_cut_short_run_is_execution_incomplete(self):
        #: a run halted after ONE of the expected two windows can never
        #: attest compute match
        ctl = _run(C.MODE_NORMAL_FEEDBACK, 1)
        ledger = build_real_compute_ledger(ctl, expected_windows=2)
        assert ledger.windows_executed == 1
        assert ledger.computation_match_status \
            == COMPUTE_MATCH_EXECUTION_INCOMPLETE


class TestVerifyAgainstConfig:
    def test_completed_run_verifies_against_frozen_budget(self):
        ctl = _run(C.MODE_NORMAL_FEEDBACK, C.MAX_WINDOWS)
        ledger = build_real_compute_ledger(ctl, expected_windows=C.MAX_WINDOWS)
        config = LONGRUN.longrun_config("normal_feedback")
        assert ledger.computation_match_status == COMPUTE_MATCH_PASS
        assert ledger.verify_against_config(config) == []

    def test_cut_short_run_reports_execution_incomplete(self):
        ctl = _run(C.MODE_NORMAL_FEEDBACK, 2)
        ledger = build_real_compute_ledger(ctl, expected_windows=C.MAX_WINDOWS)
        config = LONGRUN.longrun_config("normal_feedback")
        problems = ledger.verify_against_config(config)
        assert any("COMPUTE_MATCH_EXECUTION_INCOMPLETE" in p
                   for p in problems)

    def test_config_drift_reports_compute_match_broken(self):
        ctl = _run(C.MODE_NORMAL_FEEDBACK, C.MAX_WINDOWS)
        ledger = build_real_compute_ledger(ctl, expected_windows=C.MAX_WINDOWS)
        config = LONGRUN.longrun_config("normal_feedback")
        config["probe_transitions_total"] = \
            int(config["probe_transitions_total"]) + 1
        problems = ledger.verify_against_config(config)
        assert any("COMPUTE_MATCH_BROKEN" in p for p in problems)

    def test_cross_mode_ledgers_agree(self):
        ledgers = {
            mode: build_real_compute_ledger(
                _run(mode, C.MAX_WINDOWS), expected_windows=C.MAX_WINDOWS)
            for mode in (C.MODE_NORMAL_FEEDBACK, C.MODE_STATIC_LLM,
                         C.MODE_SHUFFLED_FEEDBACK)}
        for mode, ledger in ledgers.items():
            assert ledger.computation_match_status == COMPUTE_MATCH_PASS, mode
        first = ledgers[C.MODE_NORMAL_FEEDBACK]
        for mode, ledger in ledgers.items():
            assert first.matches(ledger), mode


class TestBudgetSemantics:
    def test_legal_semantics_and_blocked_default(self):
        assert C.BLOCKED_WAITING_DIRECTOR_BUDGET_DECISION not in \
            TRAINING_BUDGET_SEMANTICS
        assert TRAINING_BUDGET_SEMANTICS == {
            C.TRAINING_BUDGET_TOTAL_FROM_COMMON_INITIALIZATION,
            C.TRAINING_BUDGET_ADDITIONAL_FROM_PRETRAINED_CHECKPOINT}
        #: the launcher's selected semantics is the blocked default
        assert LONGRUN.TRAINING_BUDGET_SEMANTICS_SELECTED \
            == C.BLOCKED_WAITING_DIRECTOR_BUDGET_DECISION

    def test_check_only_reports_blocked_semantics(self, capsys):
        code = LONGRUN.main(["--mode", "normal_feedback", "--check-only"])
        assert code == 0
        out = capsys.readouterr().out
        assert C.BLOCKED_WAITING_DIRECTOR_BUDGET_DECISION in out
        assert "TOTAL_FROM_COMMON_INITIALIZATION" in out

    def test_launch_refused_while_budget_semantics_blocked(self, monkeypatch,
                                                           capsys):
        #: with the pilot flag granted, the P0-14 gate still refuses the
        #: launch: the 98304-step budget's meaning is undecided
        monkeypatch.setattr(C, "E2_PILOT_AUTHORIZED", True)
        code = LONGRUN.main(["--mode", "normal_feedback"])
        assert code == 1
        err = capsys.readouterr().err
        assert "training_budget_semantics" in err
        assert "BLOCKED_WAITING_DIRECTOR_BUDGET_DECISION" in err


class TestPosture:
    def test_real_capability_flags_stay_false(self):
        _run(C.MODE_NORMAL_FEEDBACK, 2)
        for name in C.NEVER_TRUE_REAL_CAPABILITY_FLAGS:
            assert getattr(C, name) is False, name
