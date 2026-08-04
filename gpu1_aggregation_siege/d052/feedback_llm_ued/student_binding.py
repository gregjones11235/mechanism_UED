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

    Production path (P0-4): with training authorized AND an explicitly
    injected shared training contract (consume-only — the shared runtime's
    ``run_one_optimizer_update`` / ``save_checkpoint`` / ``load_checkpoint``
    surface; direction two never implements an optimizer or a checkpoint
    codec), the seam executes EXACTLY ONE optimizer update over the window's
    final probe-selected batch and verifies the checkpoint save/load
    round-trip around it. The directive ordering inside window k+1 is
    probe -> select -> update -> checkpoint, so the window loop calls
    ``execute_training_step(window, defer_real_update=True)`` in the
    REVISION phase and ``execute_real_window_update`` only AFTER the probe
    funnel has selected the final batch. Without a contract the historical
    fail-closed behavior is preserved byte for byte.
    """

    def __init__(self, gate: FeedbackLaunchGate,
                 identity: StudentBindingIdentity,
                 training_contract=None) -> None:
        self._gate = gate
        self.identity = identity
        self._training_contract = training_contract

    def execute_training_step(self, window: int, *,
                              defer_real_update: bool = False
                              ) -> TrainingStepRecord:
        try:
            self._gate.assert_training_allowed()
        except LaunchGateBlocked as exc:
            return TrainingStepRecord(
                status="SKIPPED_UNAUTHORIZED",
                student_training_transitions=0,
                reason=f"window={window}: {exc}")
        if self._training_contract is None:
            if not C.REAL_CHECKPOINT_LOADED:
                raise StudentBindingBlocked(
                    "REAL_TRAINING_SEAM_NOT_IMPLEMENTED: training is "
                    "authorized by the gate but REAL_CHECKPOINT_LOADED="
                    "false — CC4 adapter evidence is required before any "
                    "real update can exist")
            raise StudentBindingBlocked(
                "REAL_TRAINING_SEAM_NOT_IMPLEMENTED: direction two consumes "
                "the CC4 shared adapter only; no local optimizer path "
                "exists")
        if defer_real_update:
            #: directive ordering: the real update consumes the window's
            #: probe-selected final batch, which does not exist until AFTER
            #: the PROBING phase — the update is deferred there, never run
            #: on a stale or empty batch
            return TrainingStepRecord(
                status="DEFERRED_TO_POST_SELECTION",
                student_training_transitions=0,
                reason=(f"window={window}: the single real optimizer update "
                        "is deferred until the probe funnel has selected the "
                        "final batch (probe -> select -> exactly one "
                        "optimizer update)"))
        return self._execute_one_update(window, batch_candidate_ids=())

    def execute_real_window_update(self, window: int, *,
                                   batch_candidate_ids
                                   ) -> TrainingStepRecord:
        """Window k+1's single real optimizer update over the final batch
        (12 dynamic + 4 anchors), wrapped in a checkpoint save/load
        round-trip. Re-checks the gate; refuses without a contract."""
        self._gate.assert_training_allowed()
        if self._training_contract is None:
            raise StudentBindingBlocked(
                "REAL_TRAINING_SEAM_NOT_IMPLEMENTED: a real window update "
                "requires the explicitly injected shared training contract")
        if not batch_candidate_ids:
            raise StudentBindingBlocked(
                f"REAL_TRAINING_BATCH_EMPTY: window={window} — the update "
                "must consume the probe-selected final batch; an empty "
                "batch is refused (NO_SILENT_FALLBACK)")
        return self._execute_one_update(window,
                                        batch_candidate_ids
                                        =tuple(batch_candidate_ids))

    def _execute_one_update(self, window: int, *,
                            batch_candidate_ids) -> TrainingStepRecord:
        """Exactly one optimizer update + checkpoint round-trip, fail closed
        on any deviation (wrong step count, unbound or unchanged checkpoint
        hash, failed reload)."""
        contract = self._training_contract
        hash_before = contract.save_checkpoint(
            tag=f"window-{window:02d}-pre-update")
        result = contract.run_one_optimizer_update(
            window=window, batch_candidate_ids=list(batch_candidate_ids))
        if int(getattr(result, "optimizer_steps", 0)) != 1:
            raise StudentBindingBlocked(
                "REAL_TRAINING_STEP_COUNT_MISMATCH: window="
                f"{window} requires EXACTLY ONE optimizer update, got "
                f"{getattr(result, 'optimizer_steps', None)!r}")
        if getattr(result, "window", None) != window:
            raise StudentBindingBlocked(
                f"REAL_TRAINING_WINDOW_MISMATCH: update result window="
                f"{getattr(result, 'window', None)!r} != {window}")
        if getattr(result, "checkpoint_hash_before", "") != hash_before:
            raise StudentBindingBlocked(
                "REAL_TRAINING_CHECKPOINT_BINDING_MISMATCH: the update "
                "result's checkpoint_hash_before does not match the "
                f"pre-update save ({hash_before[:16]}...)")
        hash_after = contract.save_checkpoint(
            tag=f"window-{window:02d}-post-update")
        if hash_after == hash_before:
            raise StudentBindingBlocked(
                "REAL_TRAINING_CHECKPOINT_UNCHANGED: exactly one optimizer "
                "update must change the full-state checkpoint hash")
        #: round-trip: the post-update checkpoint must reload cleanly
        contract.load_checkpoint(checkpoint_hash=hash_after)
        return TrainingStepRecord(
            status="EXECUTED_ONE_UPDATE_CHECKPOINT_ROUNDTRIP",
            student_training_transitions=int(
                getattr(result, "env_steps", 0)),
            reason=(f"window={window}: exactly one optimizer update over "
                    f"{len(batch_candidate_ids)} final-batch candidates; "
                    f"checkpoint {hash_before[:16]} -> {hash_after[:16]}; "
                    "reload round-trip completed"))
