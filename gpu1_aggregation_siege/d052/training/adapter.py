"""Authorization-gated training adapter (canonical_v2).

This is the seam where a real Craftax training run would be launched. In THIS
phase it is a NO-OP by design: D052_LONG_TRAINING_RUNS=0. The adapter exists so
that:
  * the cell registry has a concrete, authorized runner to hand to ``launch``;
  * the contract a future real runner must satisfy is written down and tested;
  * nothing can train without a training-scope authorization (NO_UNAUTHORIZED_
    TRAINING), and a no-training authorization can only ever produce 0 timesteps.

A real runner is deferred to a later, explicitly-authorized phase. Anyone wiring
one in must keep the gate: check authorization scope, refuse on mismatch, report
honest timesteps_run (NO_RAW_DATA_NO_STRONG_CLAIM).
"""
from __future__ import annotations

from d052.cells.authorization import SCOPE_TRAINING
from d052.cells.registry import CellRecord, no_op_runner


class TrainingAdapterError(Exception):
    UNAUTHORIZED_SCOPE = "UNAUTHORIZED_SCOPE"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"[{code}] {message}")


def canonical_training_runner(record: CellRecord) -> dict:
    """The authorization-gated runner the cell registry hands to ``launch``.

    Behaviour by authorization scope:
      * single_cell_no_training -> the no-op runner (0 timesteps). This is the
        ONLY outcome this phase ever produces.
      * single_cell_training    -> NOT implemented in this phase; refuses with a
        hard error rather than silently doing nothing or starting a run. A real
        runner is deferred to an explicitly-authorized later phase.
    """
    auth = record.authorization
    scope = auth.scope if auth is not None else None
    if scope == SCOPE_TRAINING:
        raise TrainingAdapterError(
            TrainingAdapterError.NOT_IMPLEMENTED,
            "a training-scope authorization was presented, but real training is "
            "NOT implemented in this phase (D052_LONG_TRAINING_RUNS=0); defer to "
            "an explicitly-authorized later phase")
    # no-training (or absent) scope -> guaranteed 0 timesteps
    return no_op_runner(record)


def assert_no_training_phase() -> dict:
    """Frozen label bundle asserting this phase performs no training."""
    return {
        "D052_LONG_TRAINING_RUNS": 0,
        "D052_4096_SMOKE_AUTHORIZED": False,
        "D052_24576_AUTHORIZED": False,
        "D052_98304_AUTHORIZED": False,
        "NO_UNAUTHORIZED_TRAINING": True,
    }
