"""CC4 shared StudentAdapter thin binding (direction-two, consume-only seam).

Director rule: the strong Student is fixed to
``PERSISTENT_RMT16_ORIGINAL_VTRACE_98304`` and direction two ONLY consumes the
CC4 shared StudentInitContract/StudentAdapter. This module deliberately
contains NO loader, NO registry and NO checkpoint codec:

* :func:`resolve_student_binding` validates an EXPLICITLY injected contract
  and fails closed on missing contract / identity mismatch / incomplete
  identity — it never guesses a checkpoint path or loads anything itself.
* :func:`local_symbolic_binding` is the honest stand-in while the CC4
  contract is absent from this worktree (verified): weights status
  ``NOT_LOADED_LOCAL``, label ``ENGINEERING_SCAFFOLD``,
  ``REAL_CHECKPOINT_LOADED`` stays False.
* :class:`StudentTrainingSeam` is the single place the window loop may touch
  Student training; this round it only records SKIPPED_UNAUTHORIZED.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

from d052.bagr_ued.hashing import canonical_sha256
from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.execution_mode import (
    FeedbackLaunchGate,
    LaunchGateBlocked,
)
from d052.schemas.common import is_sha256_hex

WEIGHTS_NOT_LOADED_LOCAL = "NOT_LOADED_LOCAL"
WEIGHTS_REAL_CHECKPOINT = "REAL_CHECKPOINT"


class StudentBindingBlocked(RuntimeError):
    """Fail-closed refusal of the Student binding seam."""


@runtime_checkable
class StudentInitContract(Protocol):
    """Read-only shape of the CC4 shared contract this direction consumes."""

    candidate_id: str
    parameter_tree_hash: str
    checkpoint_global_step: int


@dataclass(frozen=True)
class StudentBindingIdentity:
    """Identity stamped onto every feedback record this loop produces."""

    candidate_id: str
    architecture_family: str
    memory_family: str
    carry_mode: str
    parameter_tree_hash: str
    checkpoint_global_step: int
    weights_status: str
    provenance_label: str

    @property
    def identity_hash(self) -> str:
        return canonical_sha256(dict(
            candidate_id=self.candidate_id,
            architecture_family=self.architecture_family,
            memory_family=self.memory_family,
            carry_mode=self.carry_mode,
            parameter_tree_hash=self.parameter_tree_hash,
            checkpoint_global_step=self.checkpoint_global_step,
            weights_status=self.weights_status,
            provenance_label=self.provenance_label))


def resolve_student_binding(
        contract: Optional[StudentInitContract]) -> StudentBindingIdentity:
    """Validate an explicitly injected CC4 contract; never guess, never load.

    Fail-closed ladder:
      * no contract          -> STUDENT_INIT_CONTRACT_MISSING
      * wrong candidate      -> STUDENT_IDENTITY_MISMATCH
      * incomplete identity  -> STUDENT_IDENTITY_INCOMPLETE
    """
    if contract is None:
        raise StudentBindingBlocked(
            "STUDENT_INIT_CONTRACT_MISSING: the CC4 shared "
            "StudentInitContract/StudentAdapter is not present in this "
            "worktree; direction two consumes it only — no local loader, no "
            "guessing (REAL_CHECKPOINT_LOADED=false)")
    candidate_id = getattr(contract, "candidate_id", None)
    if candidate_id != C.STRONG_STUDENT_CANDIDATE_ID:
        raise StudentBindingBlocked(
            f"STUDENT_IDENTITY_MISMATCH: contract candidate_id="
            f"{candidate_id!r} but direction two is bound to "
            f"{C.STRONG_STUDENT_CANDIDATE_ID!r}")
    param_hash = getattr(contract, "parameter_tree_hash", None) or ""
    step = getattr(contract, "checkpoint_global_step", None)
    if not is_sha256_hex(param_hash):
        raise StudentBindingBlocked(
            f"STUDENT_IDENTITY_INCOMPLETE: parameter_tree_hash must be a "
            f"sha256 hex string, got {param_hash!r}")
    if not isinstance(step, int) or step < 0:
        raise StudentBindingBlocked(
            f"STUDENT_IDENTITY_INCOMPLETE: checkpoint_global_step must be a "
            f"non-negative int, got {step!r}")
    return StudentBindingIdentity(
        candidate_id=candidate_id,
        architecture_family=str(getattr(contract, "architecture_family",
                                        "UNKNOWN")),
        memory_family=str(getattr(contract, "memory_family", "UNKNOWN")),
        carry_mode=str(getattr(contract, "carry_mode", "UNKNOWN")),
        parameter_tree_hash=param_hash,
        checkpoint_global_step=step,
        weights_status=WEIGHTS_REAL_CHECKPOINT,
        provenance_label="CC4_SHARED_STUDENT_INIT_CONTRACT")


def local_symbolic_binding() -> StudentBindingIdentity:
    """Honest stand-in while no CC4 contract exists (NOT_LOADED_LOCAL).

    The parameter hash is derived from the identity description itself and is
    explicitly labelled — it is NOT a checkpoint hash and must never be cited
    as evidence a real checkpoint was loaded.
    """
    payload = dict(candidate_id=C.STRONG_STUDENT_CANDIDATE_ID,
                   weights_status=WEIGHTS_NOT_LOADED_LOCAL,
                   label="symbolic-binding-no-weights")
    return StudentBindingIdentity(
        candidate_id=C.STRONG_STUDENT_CANDIDATE_ID,
        architecture_family="RMT16",
        memory_family="RMT16_ORIGINAL",
        carry_mode="PERSISTENT",
        parameter_tree_hash=canonical_sha256(payload),
        checkpoint_global_step=0,
        weights_status=WEIGHTS_NOT_LOADED_LOCAL,
        provenance_label=C.ENGINEERING_SCAFFOLD)


@dataclass(frozen=True)
class TrainingStepRecord:
    """Per-window training-seam bookkeeping (honest no-op this round)."""

    status: str
    student_training_transitions: int
    reason: str


class StudentTrainingSeam:
    """The ONLY place the loop may touch Student training.

    Fail-closed twice: the launch gate refuses training this round, and even
    with training authorized the seam still requires real CC4 adapter evidence
    (``REAL_CHECKPOINT_LOADED``) before any update could exist.
    """

    def __init__(self, gate: FeedbackLaunchGate,
                 identity: StudentBindingIdentity) -> None:
        self._gate = gate
        self.identity = identity

    def execute_training_step(self, window: int) -> TrainingStepRecord:
        try:
            self._gate.assert_training_allowed()
        except LaunchGateBlocked as exc:
            return TrainingStepRecord(
                status="SKIPPED_UNAUTHORIZED",
                student_training_transitions=0,
                reason=f"window={window}: {exc}")
        if not C.REAL_CHECKPOINT_LOADED:
            raise StudentBindingBlocked(
                "REAL_TRAINING_SEAM_NOT_IMPLEMENTED: training is authorized "
                "by the gate but REAL_CHECKPOINT_LOADED=false — CC4 adapter "
                "evidence is required before any real update can exist")
        raise StudentBindingBlocked(
            "REAL_TRAINING_SEAM_NOT_IMPLEMENTED: direction two consumes the "
            "CC4 shared adapter only; no local optimizer path exists")
