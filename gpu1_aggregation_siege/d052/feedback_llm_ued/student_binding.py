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
from typing import Mapping, Optional, Protocol, runtime_checkable

from pydantic import ConfigDict, Field, model_validator

from d052.bagr_ued.hashing import canonical_sha256, verify_content_hash
from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.execution_mode import (
    FeedbackLaunchGate,
    LaunchGateBlocked,
)
from d052.schemas.common import CanonicalModel, is_sha256_hex

WEIGHTS_NOT_LOADED_LOCAL = "NOT_LOADED_LOCAL"
WEIGHTS_REAL_CHECKPOINT = "REAL_CHECKPOINT"


class StudentBindingBlocked(RuntimeError):
    """Fail-closed refusal of the Student binding seam."""


@runtime_checkable
class StudentInitContract(Protocol):
    """Read-only shape of the CC4 shared contract this direction consumes.

    P0-16 (dual student): the contract carries the FULL Student identity —
    candidate_id (director-selected, one of the allowed set), architecture_
    family=RMT16, parameter_tree_hash, checkpoint_global_step, profile_hash,
    memory_mode, memory_spec_hash, carry_mode, adapter_identity_hash,
    runtime_bundle_hash. There is NO default candidate.
    """

    candidate_id: str
    architecture_family: str
    memory_family: str
    parameter_tree_hash: str
    checkpoint_global_step: int
    profile_hash: str
    memory_mode: str
    memory_spec_hash: str
    carry_mode: str
    adapter_identity_hash: str
    runtime_bundle_hash: str


@dataclass(frozen=True)
class StudentBindingIdentity:
    """Identity stamped onto every feedback record this loop produces.

    P0-16 (dual student): the full director-selected Student identity —
    memory_mode / memory_spec_hash / profile_hash / adapter_identity_hash /
    runtime_bundle_hash all participate in the identity hash.
    """

    candidate_id: str
    architecture_family: str
    memory_family: str
    carry_mode: str
    parameter_tree_hash: str
    checkpoint_global_step: int
    profile_hash: str
    memory_mode: str
    memory_spec_hash: str
    adapter_identity_hash: str
    runtime_bundle_hash: str
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
            profile_hash=self.profile_hash,
            memory_mode=self.memory_mode,
            memory_spec_hash=self.memory_spec_hash,
            adapter_identity_hash=self.adapter_identity_hash,
            runtime_bundle_hash=self.runtime_bundle_hash,
            weights_status=self.weights_status,
            provenance_label=self.provenance_label))


def _require_sha256(value, code: str, what: str) -> str:
    value = str(value or "")
    if not is_sha256_hex(value):
        raise StudentBindingBlocked(
            f"{code}: {what} must be a sha256 hex string, got {value!r}")
    return value


def resolve_student_binding(
        contract: Optional[StudentInitContract], *,
        director_selected_candidate_id: str) -> StudentBindingIdentity:
    """Validate an explicitly injected CC4 contract against the DIRECTOR-
    selected Student; never guess, never load, NO default candidate.

    Fail-closed ladder:
      * no contract               -> STUDENT_INIT_CONTRACT_MISSING
      * no director selection     -> E2_STUDENT_NO_DIRECTOR_SELECTION
      * unknown selected candidate-> E2_STUDENT_UNKNOWN_CANDIDATE
      * contract candidate != selection -> E2_STUDENT_PROFILE_MISMATCH
      * non-RMT16 / bad hashes     -> E2_STUDENT_PROFILE_MISMATCH
      * memory/carry not the legal mapping -> E2_STUDENT_MEMORY_MODE_MISMATCH
      * bad memory-spec hash       -> E2_STUDENT_MEMORY_MODE_MISMATCH
      * bad adapter identity       -> E2_STUDENT_ADAPTER_IDENTITY_MISMATCH
    """
    if contract is None:
        raise StudentBindingBlocked(
            "STUDENT_INIT_CONTRACT_MISSING: the CC4 shared "
            "StudentInitContract/StudentAdapter is not present in this "
            "worktree; direction two consumes it only — no local loader, no "
            "guessing (REAL_CHECKPOINT_LOADED=false)")
    if not director_selected_candidate_id:
        raise StudentBindingBlocked(
            "E2_STUDENT_NO_DIRECTOR_SELECTION: there is NO default Student "
            "candidate — the director must select one of "
            f"{sorted(C.ALLOWED_STUDENT_CANDIDATE_IDS)}")
    if director_selected_candidate_id not in C.ALLOWED_STUDENT_CANDIDATE_IDS:
        raise StudentBindingBlocked(
            f"E2_STUDENT_UNKNOWN_CANDIDATE: "
            f"{director_selected_candidate_id!r} is not in "
            "ALLOWED_STUDENT_CANDIDATE_IDS="
            f"{sorted(C.ALLOWED_STUDENT_CANDIDATE_IDS)}")
    candidate_id = getattr(contract, "candidate_id", None)
    if candidate_id != director_selected_candidate_id:
        raise StudentBindingBlocked(
            f"E2_STUDENT_PROFILE_MISMATCH: contract candidate_id="
            f"{candidate_id!r} != director-selected "
            f"{director_selected_candidate_id!r}")
    if str(getattr(contract, "architecture_family", "")) != "RMT16":
        raise StudentBindingBlocked(
            f"E2_STUDENT_PROFILE_MISMATCH: architecture_family must be "
            "RMT16, got "
            f"{getattr(contract, 'architecture_family', None)!r}")
    param_hash = _require_sha256(
        getattr(contract, "parameter_tree_hash", ""),
        "E2_STUDENT_PROFILE_MISMATCH", "parameter_tree_hash")
    step = getattr(contract, "checkpoint_global_step", None)
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise StudentBindingBlocked(
            f"E2_STUDENT_PROFILE_MISMATCH: checkpoint_global_step must be "
            f"a non-negative int, got {step!r}")
    profile_hash = _require_sha256(
        getattr(contract, "profile_hash", ""),
        "E2_STUDENT_PROFILE_MISMATCH", "profile_hash")
    runtime_bundle_hash = _require_sha256(
        getattr(contract, "runtime_bundle_hash", ""),
        "E2_STUDENT_PROFILE_MISMATCH", "runtime_bundle_hash")
    memory_mode = str(getattr(contract, "memory_mode", "") or "")
    carry_mode = str(getattr(contract, "carry_mode", "") or "")
    expected_mem, expected_carry = C.STUDENT_PROFILE_MEMORY_MAP[
        director_selected_candidate_id]
    if memory_mode != expected_mem or carry_mode != expected_carry:
        raise StudentBindingBlocked(
            f"E2_STUDENT_MEMORY_MODE_MISMATCH: candidate "
            f"{candidate_id!r} requires memory_mode={expected_mem!r} / "
            f"carry_mode={expected_carry!r}, got {memory_mode!r} / "
            f"{carry_mode!r}")
    memory_spec_hash = _require_sha256(
        getattr(contract, "memory_spec_hash", ""),
        "E2_STUDENT_MEMORY_MODE_MISMATCH", "memory_spec_hash")
    adapter_identity_hash = _require_sha256(
        getattr(contract, "adapter_identity_hash", ""),
        "E2_STUDENT_ADAPTER_IDENTITY_MISMATCH", "adapter_identity_hash")
    return StudentBindingIdentity(
        candidate_id=candidate_id,
        architecture_family="RMT16",
        memory_family=str(getattr(contract, "memory_family", "UNKNOWN")),
        carry_mode=carry_mode,
        parameter_tree_hash=param_hash,
        checkpoint_global_step=step,
        profile_hash=profile_hash,
        memory_mode=memory_mode,
        memory_spec_hash=memory_spec_hash,
        adapter_identity_hash=adapter_identity_hash,
        runtime_bundle_hash=runtime_bundle_hash,
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
        carry_mode=C.STUDENT_CARRY_MODE_PERSISTENT,
        parameter_tree_hash=canonical_sha256(payload),
        checkpoint_global_step=0,
        #: P0-16: the symbolic stand-in carries NO real profile/memory/
        #: adapter/bundle identity (empty = NOT_LOADED_LOCAL, honest)
        profile_hash="",
        memory_mode="",
        memory_spec_hash="",
        adapter_identity_hash="",
        runtime_bundle_hash="",
        weights_status=WEIGHTS_NOT_LOADED_LOCAL,
        provenance_label=C.ENGINEERING_SCAFFOLD)


@dataclass(frozen=True)
class TrainingStepRecord:
    """Per-window training-seam bookkeeping (honest no-op this round).

    P0-11: ``checkpoint_round_trip_pass`` is True ONLY for the record of
    an update whose checkpoint save/load round-trip was ATTESTED by the
    director-verifier's FullStateRoundTripResult — every skipped /
    deferred / unverified record carries False (never implied).
    """

    status: str
    student_training_transitions: int
    reason: str
    checkpoint_round_trip_pass: bool = False


#: status a seam record carries when exactly one update + checkpoint
#: round-trip executed (the ONLY status counted as an executed update)
EXECUTED_ONE_UPDATE_STATUS = "EXECUTED_ONE_UPDATE_CHECKPOINT_ROUNDTRIP"


@dataclass(frozen=True)
class RealTwoWindowSmokePolicy:
    """P0-10: the update-count contract of the two-window real smoke.

    Window k FREEZES feedback_k; the first window that may CONSUME
    feedback is window k+1 (exactly one window of lag — CC3 C9 gate),
    i.e. plan_{k+1} is built from feedback_k. The single real optimizer
    update therefore belongs to ``update_window_index`` (default 1):

    * window 0 (and any non-update window) trains NOTHING (Δ=0) — there
      is no prior feedback to train on;
    * the update window executes EXACTLY ONE optimizer update (Δ=1) over
      the probe-selected final batch, wrapped in the checkpoint
      save/load round-trip;
    * a COMPLETED run (no REQUEST_CONTROL stop) must end with exactly
      ``updates_expected_total`` executed updates — anything else fails
      closed at the end of ``run()`` (TWO_WINDOW_SMOKE_UPDATE_COUNT_
      MISMATCH).
    """

    updates_expected_total: int = 1
    update_window_index: int = 1

    def __post_init__(self) -> None:
        if (not isinstance(self.updates_expected_total, int)
                or isinstance(self.updates_expected_total, bool)
                or self.updates_expected_total < 0):
            raise ValueError(
                "ILLEGAL_SMOKE_POLICY_UPDATE_COUNT: "
                f"updates_expected_total={self.updates_expected_total!r}")
        if (not isinstance(self.update_window_index, int)
                or isinstance(self.update_window_index, bool)
                or self.update_window_index < 0):
            raise ValueError(
                "ILLEGAL_SMOKE_POLICY_UPDATE_WINDOW: "
                f"update_window_index={self.update_window_index!r}")


# ---------------------------------------------------------------------------
# P0-11/P0-16: the director-runtime's UNFORGEABLE round-trip attestation
# ---------------------------------------------------------------------------
class FullStateRoundTripResult(CanonicalModel):
    """TEST_ONLY legacy shape: a LOCALLY-SIGNED round-trip attestation.

    The production seam REFUSES this shape (see
    :func:`consume_director_verified_round_trip`) — it exists only so the
    rejection tests can prove that "save hash differs + load called" and
    local self-signatures are never accepted as a round-trip. The signing
    helper lives in the tests directory (e2_test_sign_helpers), never in
    production.
    """

    model_config = ConfigDict(frozen=True)

    window: int = Field(ge=0)
    checkpoint_hash: str = Field(min_length=1)
    state_hash_before_save: str = Field(min_length=1)
    state_hash_after_reload: str = Field(min_length=1)
    verifier_id: str = Field(min_length=1)
    verified: bool = False
    round_trip_hash: str = ""

    @model_validator(mode="after")
    def _validate_and_hash(self) -> "FullStateRoundTripResult":
        for field_name in ("checkpoint_hash", "state_hash_before_save",
                           "state_hash_after_reload"):
            value = getattr(self, field_name)
            if not is_sha256_hex(value):
                raise ValueError(
                    "FULL_STATE_ROUND_TRIP_HASH_NOT_SHA256: "
                    f"{field_name}={value!r}")
        if not is_sha256_hex(self.verifier_id):
            raise ValueError(
                "FULL_STATE_ROUND_TRIP_VERIFIER_IDENTITY_INVALID: "
                f"verifier_id={self.verifier_id!r}")
        if not self.verified:
            raise ValueError(
                "FULL_STATE_ROUND_TRIP_NOT_VERIFIED: only a PASSING "
                "attestation may be consumed")
        if self.state_hash_before_save != self.state_hash_after_reload:
            raise ValueError(
                "FULL_STATE_ROUND_TRIP_STATE_MISMATCH: reloaded full-state "
                f"hash {self.state_hash_after_reload!r} does not reproduce "
                f"the pre-save full-state hash "
                f"{self.state_hash_before_save!r}")
        if not self.round_trip_hash:
            raise ValueError(
                "FULL_STATE_ROUND_TRIP_UNSIGNED: the attestation must carry "
                "the recomputable round_trip_hash")
        computed = verify_content_hash(self.model_dump(),
                                       hash_field="round_trip_hash",
                                       carried=self.round_trip_hash,
                                       kind="FullStateRoundTripResult")
        object.__setattr__(self, "round_trip_hash", computed)
        return self


class DirectorVerifiedRunStateRoundTrip(CanonicalModel):
    """P0-16 (section 6): the director-runtime's UNFORGEABLE full-state
    round-trip attestation — the ONLY proof the production seam accepts.

    The director-runtime (CanonicalDiCodeRunStateCheckpoint) verifies that
    the reloaded FULL training state reproduces the pre-update state and
    attests:

    * ``verifier_id`` — a verifier REGISTERED in the FormalAssetRegistry,
      plus its ``verifier_implementation_hash``;
    * ``runtime_bundle_hash`` — the signed Runtime Bundle this run
      consumed;
    * the Student checkpoint and the optimizer state hashes;
    * the DiCode training clock (``global_update_step`` /
      ``global_env_steps``);
    * the RNG state hash;
    * the controller / feedback store identity hash;
    * ``next_policy_step_equivalent`` — the reloaded policy reproduces the
      next update's step.

    Direction two NEVER signs this (no local signer exists); plain
    mappings and locally-signed shapes are REFUSED by the consumer.
    """

    model_config = ConfigDict(frozen=True)

    window: int = Field(ge=0)
    checkpoint_hash: str = Field(min_length=1)
    verifier_id: str = Field(min_length=1)
    verifier_implementation_hash: str = Field(min_length=1)
    runtime_bundle_hash: str = Field(min_length=1)
    student_checkpoint_hash: str = Field(min_length=1)
    optimizer_state_hash: str = Field(min_length=1)
    global_update_step: int = Field(ge=0)
    global_env_steps: int = Field(ge=0)
    rng_state_hash: str = Field(min_length=1)
    controller_store_hash: str = Field(min_length=1)
    next_policy_step_equivalent: bool = False
    verified: bool = False
    attestation_hash: str = ""

    @model_validator(mode="after")
    def _validate_and_hash(self) -> "DirectorVerifiedRunStateRoundTrip":
        for field_name in ("checkpoint_hash", "verifier_id",
                           "verifier_implementation_hash",
                           "runtime_bundle_hash", "student_checkpoint_hash",
                           "optimizer_state_hash", "rng_state_hash",
                           "controller_store_hash"):
            value = getattr(self, field_name)
            if not is_sha256_hex(value):
                raise ValueError(
                    "DIRECTOR_ROUND_TRIP_HASH_NOT_SHA256: "
                    f"{field_name}={value!r}")
        if not self.verified:
            raise ValueError(
                "DIRECTOR_ROUND_TRIP_NOT_VERIFIED: only a PASSING "
                "director-runtime attestation may be consumed")
        if not self.next_policy_step_equivalent:
            raise ValueError(
                "DIRECTOR_ROUND_TRIP_NEXT_STEP_NOT_EQUIVALENT: the reloaded "
                "policy must reproduce the next update's step")
        if not self.attestation_hash:
            raise ValueError(
                "DIRECTOR_ROUND_TRIP_UNSIGNED: the director-runtime must "
                "sign the attestation (attestation_hash is mandatory)")
        computed = verify_content_hash(
            self.model_dump(), hash_field="attestation_hash",
            carried=self.attestation_hash,
            kind="DirectorVerifiedRunStateRoundTrip")
        object.__setattr__(self, "attestation_hash", computed)
        return self


def consume_full_state_round_trip(raw: object, *, window: int,
                                  checkpoint_hash: str
                                  ) -> FullStateRoundTripResult:
    """P0-11 consume-only gate: accept ONLY the immutable signed
    attestation, bound to THIS window and THIS checkpoint. Fail-closed:

      * not a FullStateRoundTripResult / mapping ->
            REAL_TRAINING_ROUND_TRIP_NOT_SIGNED
      * mapping failing the attestation contract ->
            REAL_TRAINING_ROUND_TRIP_ILLEGAL
      * attestation for another window / checkpoint ->
            REAL_TRAINING_ROUND_TRIP_WINDOW_MISMATCH /
            REAL_TRAINING_ROUND_TRIP_CHECKPOINT_MISMATCH
    """
    if isinstance(raw, FullStateRoundTripResult):
        attestation = raw
    elif isinstance(raw, Mapping):
        try:
            attestation = FullStateRoundTripResult(**dict(raw))
        except Exception as exc:
            raise StudentBindingBlocked(
                f"REAL_TRAINING_ROUND_TRIP_ILLEGAL: mapping payload failed "
                f"the attestation contract: {type(exc).__name__}: {exc}"
            ) from exc
    else:
        raise StudentBindingBlocked(
            "REAL_TRAINING_ROUND_TRIP_NOT_SIGNED: the training seam "
            "consumes ONLY the immutable director-verifier "
            f"FullStateRoundTripResult, got {type(raw).__name__} — 'save "
            "hash differs + load called' is NOT a round-trip")
    if attestation.window != window:
        raise StudentBindingBlocked(
            "REAL_TRAINING_ROUND_TRIP_WINDOW_MISMATCH: attestation window="
            f"{attestation.window} but this update ran in window={window}")
    if attestation.checkpoint_hash != checkpoint_hash:
        raise StudentBindingBlocked(
            "REAL_TRAINING_ROUND_TRIP_CHECKPOINT_MISMATCH: attestation "
            f"binds checkpoint {attestation.checkpoint_hash!r} but this "
            f"update reloaded {checkpoint_hash!r}")
    return attestation


def consume_director_verified_round_trip(
        raw: object, *, window: int, checkpoint_hash: str,
        expected_runtime_bundle_hash: str
) -> DirectorVerifiedRunStateRoundTrip:
    """P0-16 (section 6): the ONLY production round-trip consumer.

    Accepts ONLY the director-runtime's immutable
    DirectorVerifiedRunStateRoundTrip instance, bound to THIS window /
    checkpoint / runtime bundle. Fail-closed:

      * a plain Mapping -> REAL_TRAINING_ROUND_TRIP_PLAIN_MAPPING_REJECTED
        (a mapping is not a signed attestation);
      * any other object (incl. the TEST_ONLY locally-signed
        FullStateRoundTripResult) ->
        REAL_TRAINING_ROUND_TRIP_NOT_DIRECTOR_VERIFIED;
      * wrong window / checkpoint / runtime bundle ->
        WINDOW_MISMATCH / CHECKPOINT_MISMATCH / RUNTIME_BUNDLE_MISMATCH.
    """
    if isinstance(raw, Mapping):
        raise StudentBindingBlocked(
            "REAL_TRAINING_ROUND_TRIP_PLAIN_MAPPING_REJECTED: a plain "
            "Mapping may NOT enter the production round-trip consumption "
            "surface — only the director-runtime's signed "
            "DirectorVerifiedRunStateRoundTrip is accepted")
    if not isinstance(raw, DirectorVerifiedRunStateRoundTrip):
        raise StudentBindingBlocked(
            "REAL_TRAINING_ROUND_TRIP_NOT_DIRECTOR_VERIFIED: the training "
            "seam consumes ONLY the director-runtime's unforgeable "
            f"DirectorVerifiedRunStateRoundTrip, got {type(raw).__name__} — "
            "locally-signed shapes and 'save hash differs + load called' "
            "are NOT a round-trip")
    if raw.window != window:
        raise StudentBindingBlocked(
            "REAL_TRAINING_ROUND_TRIP_WINDOW_MISMATCH: attestation window="
            f"{raw.window} but this update ran in window={window}")
    if raw.checkpoint_hash != checkpoint_hash:
        raise StudentBindingBlocked(
            "REAL_TRAINING_ROUND_TRIP_CHECKPOINT_MISMATCH: attestation "
            f"binds checkpoint {raw.checkpoint_hash!r} but this update "
            f"reloaded {checkpoint_hash!r}")
    if raw.runtime_bundle_hash != expected_runtime_bundle_hash:
        raise StudentBindingBlocked(
            "REAL_TRAINING_ROUND_TRIP_RUNTIME_BUNDLE_MISMATCH: attestation "
            f"binds runtime bundle {raw.runtime_bundle_hash!r} but this run "
            f"consumed {expected_runtime_bundle_hash!r}")
    return raw


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
                 training_contract=None,
                 runtime_bundle_hash: str = "") -> None:
        self._gate = gate
        self.identity = identity
        self._training_contract = training_contract
        #: P0-16 (section 6): the signed Runtime Bundle hash this run
        #: consumes — the director's round-trip attestation must bind it
        self._runtime_bundle_hash = runtime_bundle_hash

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
                                   batch_candidate_ids,
                                   batch_plan=None,
                                   test_only=False) -> TrainingStepRecord:
        """Window k+1's single real optimizer update over the batch.

        P0-16 (DiCode 15+1): with a :class:`CanonicalDiCodeTrainingBatchPlan`
        the update is executed EXCLUSIVELY by the director-shared
        CanonicalDiCodeOneUpdateRuntime over the 15 curriculum task ids
        (the OriginalTask is appended internally once by that runtime —
        direction two never implements a second optimizer).

        Request-changes (section 6): the PRODUCTION path requires the
        canonical batch plan — without it the update fails
        (REAL_DICODE_BATCH_PLAN_REQUIRED); the legacy optimizer surface
        (run_one_optimizer_update / save_checkpoint / load_checkpoint) is
        reachable ONLY through the explicit TEST_ONLY_LEGACY_ADAPTER
        (``test_only=True``).
        """
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
        if batch_plan is None:
            if not test_only:
                raise StudentBindingBlocked(
                    "REAL_DICODE_BATCH_PLAN_REQUIRED: window="
                    f"{window} — the production training path consumes "
                    "ONLY the canonical DiCode 15+1 batch plan; the legacy "
                    "optimizer surface is TEST_ONLY_LEGACY_ADAPTER "
                    "(test_only=True) and never enters the production "
                    "path")
            return self._execute_one_update(
                window, batch_candidate_ids=tuple(batch_candidate_ids))
        return self._execute_one_dicode_update(window, batch_plan)

    def _execute_one_dicode_update(self, window: int, batch_plan
                                   ) -> TrainingStepRecord:
        """EXACTLY ONE CanonicalDiCode optimizer update over the plan."""
        contract = self._training_contract
        run = getattr(contract, "run_one_dicode_update", None)
        if not callable(run):
            raise StudentBindingBlocked(
                "REAL_DICODE_RUNTIME_MISSING: a DiCode batch plan requires "
                "the director-shared CanonicalDiCodeOneUpdateRuntime "
                "surface (run_one_dicode_update) — direction two never "
                "implements a second PPO/optimizer")
        result = run(batch_plan=batch_plan)
        if int(getattr(result, "optimizer_steps", 0)) != 1:
            raise StudentBindingBlocked(
                "REAL_DICODE_STEP_COUNT_MISMATCH: window="
                f"{window} requires EXACTLY ONE DiCode optimizer update, "
                f"got {getattr(result, 'optimizer_steps', None)!r}")
        if getattr(result, "window", None) != window:
            raise StudentBindingBlocked(
                f"REAL_DICODE_WINDOW_MISMATCH: update result window="
                f"{getattr(result, 'window', None)!r} != {window}")
        curriculum = list(getattr(batch_plan, "curriculum_task_ids", []))
        if len(curriculum) != C.DICODE_CURRICULUM_TASK_COUNT:
            raise StudentBindingBlocked(
                "REAL_DICODE_CURRICULUM_COUNT_MISMATCH: the plan must carry "
                f"15 curriculum task ids, got {len(curriculum)}")
        checkpoint_hash_after = str(
            getattr(result, "checkpoint_hash_after", "") or "")
        attestation = self._verify_director_round_trip(
            contract, window=window, checkpoint_hash=checkpoint_hash_after)
        return TrainingStepRecord(
            status=EXECUTED_ONE_UPDATE_STATUS,
            student_training_transitions=int(
                getattr(result, "env_steps", 0)),
            reason=(f"window={window}: EXACTLY ONE CanonicalDiCode optimizer "
                    f"update over {len(curriculum)} curriculum task ids "
                    "(12 dynamic + 3 non-target anchors); the OriginalTask "
                    f"{getattr(batch_plan, 'original_task_id', '')!r} is "
                    "appended ONCE internally by the shared DiCode runtime "
                    "and never enters batch_candidate_ids; round-trip "
                    "VERIFIED by director-verifier "
                    f"{attestation.verifier_id[:16]}"),
            checkpoint_round_trip_pass=True)

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
        attestation = self._verify_director_round_trip(
            contract, window=window, checkpoint_hash=hash_after)
        return TrainingStepRecord(
            status=EXECUTED_ONE_UPDATE_STATUS,
            student_training_transitions=int(
                getattr(result, "env_steps", 0)),
            reason=(f"window={window}: exactly one optimizer update over "
                    f"{len(batch_candidate_ids)} final-batch candidates; "
                    f"checkpoint {hash_before[:16]} -> {hash_after[:16]}; "
                    "full-state round-trip VERIFIED by director-verifier "
                    f"{attestation.verifier_id[:16]} (runtime bundle "
                    f"{attestation.runtime_bundle_hash[:16]})"),
            checkpoint_round_trip_pass=True)

    def _verify_director_round_trip(self, contract, *, window: int,
                                    checkpoint_hash: str
                                    ) -> DirectorVerifiedRunStateRoundTrip:
        """P0-16 (section 6): the ONLY acceptable round-trip proof is the
        director-runtime's unforgeable DirectorVerifiedRunStateRoundTrip —
        locally-signed shapes and plain mappings are REFUSED (the consumer
        rejects them); 'save hash differs + load called' is NOT a
        round-trip."""
        verify = getattr(contract, "verify_director_round_trip", None)
        if not callable(verify):
            raise StudentBindingBlocked(
                "REAL_TRAINING_ROUND_TRIP_NOT_ATTESTED: the shared training "
                "contract must expose verify_director_round_trip (the "
                "director-runtime's unforgeable attestation)")
        return consume_director_verified_round_trip(
            verify(window=window, checkpoint_hash=checkpoint_hash),
            window=window, checkpoint_hash=checkpoint_hash,
            expected_runtime_bundle_hash=self._runtime_bundle_hash)
