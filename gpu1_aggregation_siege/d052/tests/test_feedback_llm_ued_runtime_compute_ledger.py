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


def _budget_config(windows, longrun_cfg):
    """A per-window audit budget for ``windows`` executed windows, built
    from the E2 launcher's shared per-window fields (the formal timeline
    is the DiCode clock — the audit compares the executed compute)."""
    return dict(
        windows=windows,
        board_llm_calls_per_window=C.BOARD_CALLS_PER_WINDOW,
        envcoder_calls_per_window=1,
        llm_calls_per_window=longrun_cfg["llm_calls_per_window"],
        probe_transitions_total=(
            int(longrun_cfg["probe_transitions_per_window"]) * windows),
        anchor_slots=C.GLOBAL_ANCHOR_SLOTS)


class TestVerifyAgainstConfig:
    def test_completed_run_verifies_against_audit_budget(self):
        windows = 2
        ctl = _run(C.MODE_NORMAL_FEEDBACK, windows)
        ledger = build_real_compute_ledger(ctl, expected_windows=windows)
        config = _budget_config(windows, LONGRUN.dicode_launch_config(
            "normal_feedback", LONGRUN.load_dicode_resolved_config()))
        assert ledger.computation_match_status == COMPUTE_MATCH_PASS
        assert ledger.verify_against_config(config) == []

    def test_cut_short_run_reports_execution_incomplete(self):
        #: a run halted after TWO of the expected THREE windows can never
        #: attest compute match
        ctl = _run(C.MODE_NORMAL_FEEDBACK, 2)
        ledger = build_real_compute_ledger(ctl, expected_windows=3)
        config = _budget_config(3, LONGRUN.dicode_launch_config(
            "normal_feedback", LONGRUN.load_dicode_resolved_config()))
        problems = ledger.verify_against_config(config)
        assert any("COMPUTE_MATCH_EXECUTION_INCOMPLETE" in p
                   for p in problems)

    def test_config_drift_reports_compute_match_broken(self):
        windows = 2
        ctl = _run(C.MODE_NORMAL_FEEDBACK, windows)
        ledger = build_real_compute_ledger(ctl, expected_windows=windows)
        config = _budget_config(windows, LONGRUN.dicode_launch_config(
            "normal_feedback", LONGRUN.load_dicode_resolved_config()))
        config["probe_transitions_total"] = \
            int(config["probe_transitions_total"]) + 1
        problems = ledger.verify_against_config(config)
        assert any("COMPUTE_MATCH_BROKEN" in p for p in problems)

    def test_cross_mode_ledgers_agree(self):
        windows = 2
        ledgers = {
            mode: build_real_compute_ledger(
                _run(mode, windows), expected_windows=windows)
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

    def test_check_only_reports_dicode_clock(self, capsys):
        code = LONGRUN.main(["--mode", "normal_feedback", "--check-only"])
        assert code == 0
        out = capsys.readouterr().out
        #: the formal timeline is the frozen DiCode clock
        assert '"clock_field": "global_env_steps"' in out
        assert '"total_timesteps"' in out
        assert LONGRUN.load_dicode_resolved_config()["training"][
            "total_timesteps"] > 0

    def test_launch_refused_until_human_approval(self, capsys):
        #: the formal launch gate is FORMAL_EXPERIMENT_AUTHORIZED=false —
        #: the formal experiment start waits for a human-approved Manifest
        code = LONGRUN.main(["--mode", "normal_feedback"])
        assert code == 1
        err = capsys.readouterr().err
        assert "FORMAL_EXPERIMENT_AUTHORIZED=false" in err


class TestPosture:
    def test_real_capability_flags_stay_false(self):
        _run(C.MODE_NORMAL_FEEDBACK, 2)
        for name in C.NEVER_TRUE_REAL_CAPABILITY_FLAGS:
            assert getattr(C, name) is False, name
