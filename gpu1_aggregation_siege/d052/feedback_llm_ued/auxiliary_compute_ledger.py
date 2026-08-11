"""P0-16 (director smoke handoff, section 5): the AuxiliaryComputeLedger.

The DiCode training CLOCK is ``global_env_steps`` against the frozen
DiCode config's ``training.total_timesteps`` — E2 mode only changes the
Feedback View, never the training clock. The E2 OVERHEAD — LLM calls
(board + EnvCoder) and probe simulator transitions — is tracked SEPARATELY
in the AuxiliaryComputeLedger (a director-declared Runtime Bundle asset):
auxiliary compute is never mixed into the training timestep budget.

This ledger is consume-only accounting: direction two records the
auxiliary costs it incurs; the DiCode runtime owns the training clock.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class AuxiliaryComputeLedger:
    """Auxiliary (non-training-clock) compute of the E2 loop.

    ``llm_calls``       — every six-role board + EnvCoder logical call;
    ``probe_calls``     — every staged candidate probe;
    ``probe_transitions`` — simulator transitions consumed by probes
    (NEVER counted as training timesteps — the training clock is the
    DiCode runtime's ``global_env_steps`` / ``total_timesteps``).
    """

    llm_calls: int = 0
    probe_calls: int = 0
    probe_transitions: int = 0
    recorded_events: list = field(default_factory=list)

    def record_llm_calls(self, n: int = 1) -> None:
        if n < 0:
            raise ValueError(f"ILLEGAL_LLM_CALL_COUNT: {n!r}")
        self.llm_calls += n
        self.recorded_events.append({"kind": "llm", "n": n})

    def record_probe(self, *, transitions: int) -> None:
        if transitions <= 0:
            raise ValueError(
                f"ILLEGAL_PROBE_TRANSITIONS: {transitions!r}")
        self.probe_calls += 1
        self.probe_transitions += transitions
        self.recorded_events.append(
            {"kind": "probe", "transitions": transitions})

    def to_dict(self) -> Dict[str, int]:
        return dict(llm_calls=self.llm_calls, probe_calls=self.probe_calls,
                    probe_transitions=self.probe_transitions)

    def merge(self, other: "AuxiliaryComputeLedger") -> None:
        self.llm_calls += other.llm_calls
        self.probe_calls += other.probe_calls
        self.probe_transitions += other.probe_transitions


__all__ = ["AuxiliaryComputeLedger"]
