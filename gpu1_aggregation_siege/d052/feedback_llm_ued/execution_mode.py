"""Execution mode + strongly-typed launch gate for the feedback-adaptive loop.

P0-1 fix: authorization decisions live HERE, once, fail-closed — the
controller no longer reaches into mock-only attributes (``mock_calls`` /
``assert_no_real_calls``) and no call site decides for itself whether real
LLM calls, real simulator probes, training steps or formal runs are allowed.

Every decision derives from the round-constant flags in
``d052.feedback_llm_ued.constants`` (all False this round), the declared
:class:`ExecutionMode`, and — on the production path only — an explicit
runtime grant set (:class:`~d052.feedback_llm_ued.runtime_authorization.
RealRuntimeAuthorization`). The round constants are NEVER hand-flipped;
runtime grants are the sole channel through which a REAL run may be
authorized, and they take effect ONLY in ``EXECUTION_MODE_REAL``. With no
grants injected (the default) the gate's behavior is byte-identical to the
constants-only evaluation. Anything not explicitly allowed is refused with
``LaunchGateBlocked`` — never silently downgraded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.runtime_authorization import (
    RealRuntimeAuthorization,
    empty_authorization,
)

# ---------------------------------------------------------------------------
# Execution modes for the loop driver
# ---------------------------------------------------------------------------
EXECUTION_MODE_MOCK_DRY_RUN = "MOCK_DRY_RUN"   # mock backend + symbolic probe
EXECUTION_MODE_REPLAY = "REPLAY"               # replay a recorded corpus
EXECUTION_MODE_REAL = "REAL"                   # real backends (blocked unless
#                                                  the round flags are flipped)
EXECUTION_MODES = frozenset({
    EXECUTION_MODE_MOCK_DRY_RUN, EXECUTION_MODE_REPLAY, EXECUTION_MODE_REAL,
})


class LaunchGateBlocked(RuntimeError):
    """Fail-closed refusal from the FeedbackLaunchGate."""


@dataclass(frozen=True)
class LaunchDecision:
    """Strongly-typed outcome of one gate evaluation (audit-friendly)."""

    execution_mode: str
    backend_kinds_allowed: Tuple[str, ...]
    real_llm_calls_allowed: bool
    real_simulator_probe_allowed: bool
    training_allowed: bool
    final_formal_run_allowed: bool
    reason: str


@dataclass(frozen=True)
class FinalBatchDecision:
    """C11: the final-batch verdict for a finished (or stopped) loop.

    An execution batch may ship as FINAL only if the loop ran to completion
    (no REQUEST_CONTROL stop) AND training is authorized this round. A
    REQUEST_CONTROL stop can never produce a final batch — the artifact
    awaits human review.
    """

    final: bool
    loop_completed: bool
    request_control_stopped: bool
    training_allowed: bool
    reason: str


class FeedbackLaunchGate:
    """Strongly-typed gate over the round's authorization constants.

    Imitates the frozen-decision pattern of ``bagr_ued.launch_gate``:
    evaluate first, then assert per capability; every assert re-evaluates so a
    flag change between evaluate() and use is always observed.
    """

    def __init__(self,
                 execution_mode: str = EXECUTION_MODE_MOCK_DRY_RUN,
                 runtime_grants: Optional[RealRuntimeAuthorization] = None
                 ) -> None:
        if execution_mode not in EXECUTION_MODES:
            raise ValueError(f"UNKNOWN_EXECUTION_MODE: {execution_mode!r}")
        self.execution_mode = execution_mode
        #: production-path grants; default = no grants (constants-only
        #: evaluation, byte-identical to the pre-grant gate)
        self.runtime_grants = runtime_grants or empty_authorization()

    # ------------------------------------------------------------- evaluate
    def evaluate(self) -> LaunchDecision:
        g = self.runtime_grants
        real_mode = self.execution_mode == EXECUTION_MODE_REAL
        #: the round constants stay False and are never hand-flipped; on the
        #: production path an explicit runtime grant is the ONLY way a real
        #: capability becomes allowed (and only in EXECUTION_MODE_REAL)
        real_llm = real_mode and (C.REAL_LLM_CALLS_AUTHORIZED
                                  or g.real_llm_backend)
        allowed = (C.BACKEND_KIND_MOCK, C.BACKEND_KIND_REPLAY)
        if real_llm:
            allowed = allowed + (C.BACKEND_KIND_REAL,)
        reason = ("fail-closed evaluation of "
                  "d052.feedback_llm_ued.constants authorization flags")
        if g.any_grant():
            reason += f" + runtime grants: {g.describe()}"
        return LaunchDecision(
            execution_mode=self.execution_mode,
            backend_kinds_allowed=allowed,
            real_llm_calls_allowed=real_llm,
            real_simulator_probe_allowed=(
                real_mode and (C.REAL_SIMULATOR_PROBE_AUTHORIZED
                               or g.real_probe)),
            training_allowed=(real_mode and (C.TRAINING_AUTHORIZED
                                             or g.real_training)),
            final_formal_run_allowed=(
                real_mode and C.FORMAL_EVALUATION_AUTHORIZED),
            reason=reason)

    # ------------------------------------------------------- C11 final batch
    def evaluate_final_batch(self, *, loop_completed: bool,
                             request_control_stopped: bool
                             ) -> FinalBatchDecision:
        """Fail-closed verdict on whether the loop's last batch is FINAL.

        ``final`` requires ALL of: the loop ran to completion, no
        REQUEST_CONTROL stop, and training authorized this round. The reason
        string names every failed condition (REQUEST_CONTROL first — a human
        stop outranks every other consideration).
        """
        decision = self.evaluate()
        reasons = []
        if request_control_stopped:
            reasons.append("REQUEST_CONTROL_STOPPED: the board requested "
                           "human control; the stopped window produced a "
                           "HumanDecisionArtifact and NO execution batch "
                           "awaiting autonomous continuation")
        if not loop_completed:
            reasons.append("LOOP_NOT_COMPLETED")
        if not decision.training_allowed:
            reasons.append(f"TRAINING_NOT_ALLOWED: TRAINING_AUTHORIZED="
                           f"{C.TRAINING_AUTHORIZED} this round")
        final = (loop_completed and not request_control_stopped
                 and decision.training_allowed)
        return FinalBatchDecision(
            final=final,
            loop_completed=loop_completed,
            request_control_stopped=request_control_stopped,
            training_allowed=decision.training_allowed,
            reason=("; ".join(reasons) if reasons
                    else "loop completed, no REQUEST_CONTROL stop, training "
                         "authorized"))

    # -------------------------------------------------------------- asserts
    def assert_backend_allowed(self, backend_kind: str) -> LaunchDecision:
        decision = self.evaluate()
        if backend_kind not in C.BACKEND_KINDS:
            raise ValueError(f"UNKNOWN_BACKEND_KIND: {backend_kind!r}")
        if backend_kind not in decision.backend_kinds_allowed:
            raise LaunchGateBlocked(
                f"BACKEND_KIND_NOT_ALLOWED: kind={backend_kind!r} "
                f"mode={self.execution_mode} "
                f"(REAL_LLM_CALLS_AUTHORIZED="
                f"{C.REAL_LLM_CALLS_AUTHORIZED})")
        return decision

    def assert_real_probe_allowed(self) -> LaunchDecision:
        decision = self.evaluate()
        if not decision.real_simulator_probe_allowed:
            raise LaunchGateBlocked(
                "REAL_SIMULATOR_PROBE_NOT_ALLOWED: mode="
                f"{self.execution_mode} (REAL_SIMULATOR_PROBE_AUTHORIZED="
                f"{C.REAL_SIMULATOR_PROBE_AUTHORIZED}, status="
                f"{C.REAL_SIMULATOR_PROBE_STATUS})")
        return decision

    def assert_training_allowed(self) -> LaunchDecision:
        decision = self.evaluate()
        if not decision.training_allowed:
            raise LaunchGateBlocked(
                "TRAINING_NOT_ALLOWED: TRAINING_AUTHORIZED="
                f"{C.TRAINING_AUTHORIZED} this round")
        return decision

    def assert_final_formal_run_allowed(self) -> LaunchDecision:
        decision = self.evaluate()
        if not decision.final_formal_run_allowed:
            raise LaunchGateBlocked(
                "FINAL_FORMAL_RUN_NOT_ALLOWED: FORMAL_EVALUATION_AUTHORIZED="
                f"{C.FORMAL_EVALUATION_AUTHORIZED} this round")
        return decision
