"""P0-13 / P0-14: the post-run RealComputeLedger and the training-budget
semantics gate.

P0-13: a compute-matched comparison is only as honest as the ACTUAL
compute a run performed — the declared budget alone proves nothing.
:func:`build_real_compute_ledger` gathers the executed totals from a
finished controller (windows, board calls, EnvCoder calls, LLM calls,
probes, simulator transitions, feedback records, anchors, training steps,
checkpoint saves and verified round-trip passes), and
``computation_match_status`` is ``COMPUTE_MATCH_PASS`` ONLY when the run
completed the full horizon. Any shorter run — a REQUEST_CONTROL stop or
any other early halt — is ``COMPUTE_MATCH_EXECUTION_INCOMPLETE``: a
truncated run can never attest compute match. ``verify_against_config``
then compares the actuals against the frozen compute-match budget field
by field (fail closed on every deviation).

P0-14: the long-run training budget's SEMANTICS is a DIRECTOR decision —
direction two consumes the shared runtime and cannot decide it itself:
``TOTAL_FROM_COMMON_INITIALIZATION`` (the 98304 environment steps are the
whole training budget from one common initialization) or
``ADDITIONAL_FROM_PRETRAINED_CHECKPOINT`` (additional fine-tuning on top
of a pretrained checkpoint). Until the director decides, the budget is
``BLOCKED_WAITING_DIRECTOR_BUDGET_DECISION`` and no long run may launch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.student_binding import EXECUTED_ONE_UPDATE_STATUS

COMPUTE_MATCH_PASS = "COMPUTE_MATCH_PASS"
COMPUTE_MATCH_EXECUTION_INCOMPLETE = "COMPUTE_MATCH_EXECUTION_INCOMPLETE"


class RealComputeLedgerBlocked(RuntimeError):
    """Fail-closed refusal of the compute-ledger seam."""


@dataclass(frozen=True)
class RealComputeLedger:
    """Actual executed compute of one finished run, per mode.

    ``computation_match_status``: COMPUTE_MATCH_PASS only when the run
    completed the full expected horizon; anything shorter is
    COMPUTE_MATCH_EXECUTION_INCOMPLETE.
    """

    mode: str
    windows_executed: int
    board_calls_total: int
    envcoder_calls_total: int
    llm_calls_total: int
    probe_calls_total: int
    simulator_transitions_total: int
    feedback_records_total: int
    anchors_per_window: int
    training_steps_total: int
    checkpoint_saves_total: int
    checkpoint_round_trip_passes: int
    computation_match_status: str

    def verify_against_config(self, config: Dict[str, object]
                              ) -> List[str]:
        """Compare the ACTUAL ledger against the declared compute-match
        budget, fail closed on every deviation. An empty list is the only
        passing state."""
        problems: List[str] = []
        windows = int(config["windows"])
        if self.windows_executed != windows:
            problems.append(
                f"COMPUTE_MATCH_EXECUTION_INCOMPLETE: {self.mode} executed "
                f"{self.windows_executed}/{windows} windows — a truncated "
                "run cannot attest compute match")
        if self.computation_match_status != COMPUTE_MATCH_PASS:
            problems.append(
                f"COMPUTE_MATCH_EXECUTION_INCOMPLETE: {self.mode} ledger "
                f"status {self.computation_match_status!r}")
        if self.board_calls_total != \
                int(config["board_llm_calls_per_window"]) * windows:
            problems.append(
                f"COMPUTE_MATCH_BROKEN: {self.mode} board calls "
                f"{self.board_calls_total} != "
                f"{config['board_llm_calls_per_window']} * {windows}")
        if self.envcoder_calls_total != \
                int(config["envcoder_calls_per_window"]) * windows:
            problems.append(
                f"COMPUTE_MATCH_BROKEN: {self.mode} EnvCoder calls "
                f"{self.envcoder_calls_total} != "
                f"{config['envcoder_calls_per_window']} * {windows}")
        if self.llm_calls_total != \
                int(config["llm_calls_per_window"]) * windows:
            problems.append(
                f"COMPUTE_MATCH_BROKEN: {self.mode} LLM calls "
                f"{self.llm_calls_total} != "
                f"{config['llm_calls_per_window']} * {windows}")
        if self.simulator_transitions_total != \
                int(config["probe_transitions_total"]):
            problems.append(
                f"COMPUTE_MATCH_BROKEN: {self.mode} simulator transitions "
                f"{self.simulator_transitions_total} != "
                f"{config['probe_transitions_total']}")
        if self.anchors_per_window != int(config["anchor_slots"]):
            problems.append(
                f"COMPUTE_MATCH_BROKEN: {self.mode} anchors/window "
                f"{self.anchors_per_window} != {config['anchor_slots']}")
        if self.training_steps_total != windows:
            problems.append(
                f"COMPUTE_MATCH_BROKEN: {self.mode} training steps "
                f"{self.training_steps_total} != {windows}")
        return problems

    def matches(self, other: "RealComputeLedger") -> bool:
        """P0-13 cross-mode equality: two completed ledgers must agree on
        every compute field (only the mode label may differ)."""
        if self.computation_match_status != other.computation_match_status:
            return False
        for field in ("windows_executed", "board_calls_total",
                      "envcoder_calls_total", "llm_calls_total",
                      "probe_calls_total", "simulator_transitions_total",
                      "feedback_records_total", "anchors_per_window",
                      "training_steps_total", "checkpoint_saves_total",
                      "checkpoint_round_trip_passes"):
            if getattr(self, field) != getattr(other, field):
                return False
        return True


def build_real_compute_ledger(controller, *, expected_windows: int
                              ) -> RealComputeLedger:
    """Gather the executed compute of one finished controller run.

    Fail-closed: the controller must have completed a run (the summary
    exists) and the total LLM-call count must reconcile with the sum of
    the per-window board + EnvCoder calls.
    """
    summary = getattr(controller, "_summary", None)
    if summary is None:
        raise RealComputeLedgerBlocked(
            "REAL_COMPUTE_LEDGER_REQUIRES_RUN: build_real_compute_ledger "
            "needs a controller whose run() has completed")
    windows = list(summary.windows)
    board_calls_total = sum(int(w["board_call_count"]) for w in windows)
    envcoder_calls_total = sum(int(w["env_coder_call_count"])
                               for w in windows)
    llm_calls_total = int(summary.n_llm_calls)
    if llm_calls_total != board_calls_total + envcoder_calls_total:
        raise RealComputeLedgerBlocked(
            "COMPUTE_LEDGER_CALL_MISMATCH: summary n_llm_calls="
            f"{llm_calls_total} != board {board_calls_total} + envcoder "
            f"{envcoder_calls_total}")
    anchors_per_window = 0
    if windows:
        anchors_per_window = int(windows[0]["funnel_stats"]["anchors"])
    executed_updates = sum(
        1 for t in controller.training_log
        if t.status == EXECUTED_ONE_UPDATE_STATUS)
    return RealComputeLedger(
        mode=controller.mode,
        windows_executed=int(summary.n_windows),
        board_calls_total=board_calls_total,
        envcoder_calls_total=envcoder_calls_total,
        llm_calls_total=llm_calls_total,
        probe_calls_total=int(controller.runner.probe_calls),
        simulator_transitions_total=int(summary.total_simulator_transitions),
        feedback_records_total=len(list(controller.store.ids())),
        anchors_per_window=anchors_per_window,
        training_steps_total=len(controller.training_log),
        #: the seam saves pre + post around every executed update
        checkpoint_saves_total=2 * executed_updates,
        checkpoint_round_trip_passes=sum(
            1 for t in controller.training_log
            if t.checkpoint_round_trip_pass),
        computation_match_status=(
            COMPUTE_MATCH_PASS
            if int(summary.n_windows) == expected_windows
            else COMPUTE_MATCH_EXECUTION_INCOMPLETE))


#: the two legal training-budget semantics (P0-14), for the report
TRAINING_BUDGET_SEMANTICS = frozenset({
    C.TRAINING_BUDGET_TOTAL_FROM_COMMON_INITIALIZATION,
    C.TRAINING_BUDGET_ADDITIONAL_FROM_PRETRAINED_CHECKPOINT,
})


__all__ = [
    "COMPUTE_MATCH_PASS", "COMPUTE_MATCH_EXECUTION_INCOMPLETE",
    "RealComputeLedgerBlocked", "RealComputeLedger",
    "build_real_compute_ledger", "TRAINING_BUDGET_SEMANTICS",
]
