"""P0-11 / P0-16 (§19 seam coverage): the training seam consumes ONLY the
director-runtime's UNFORGEABLE DirectorVerifiedRunStateRoundTrip.

Contract under test:

* "the save hash differs and load_checkpoint was called" is NOT a
  round-trip — the ONLY acceptable proof is the director-runtime's signed
  DirectorVerifiedRunStateRoundTrip (verifier registered in the Formal
  Asset Registry + implementation hash, runtime bundle hash, Student
  checkpoint, optimizer state, global_update_step/global_env_steps, RNG,
  controller/feedback store, next-policy-step equivalence);
* a plain Mapping NEVER enters the production consumption surface
  (REAL_TRAINING_ROUND_TRIP_PLAIN_MAPPING_REJECTED); locally-signed
  shapes (FullStateRoundTripResult) and duck-typed objects are refused
  (REAL_TRAINING_ROUND_TRIP_NOT_DIRECTOR_VERIFIED);
* the legacy FullStateRoundTripResult remains as the TEST_ONLY local
  self-signature shape — its model + consume_full_state_round_trip tests
  document the rejected shape;
* CHECKPOINT_ROUND_TRIP_PASS semantics: True only on the executed record
  with a verified director attestation — every skipped/deferred record
  carries False;
* the shared training slot refuses contracts lacking the
  verify_director_round_trip surface (SHARED_TRAINING_CONTRACT_INCOMPLETE).

Fixtures are TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION: the scripted
contract emulates the director-runtime and signs attestations over
pre-built hashes — NO checkpoint is saved, NO optimizer runs, and NO
passing test flips a REAL_* flag.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from d052.bagr_ued.hashing import text_sha256
from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.execution_mode import (
    EXECUTION_MODE_REAL,
    FeedbackLaunchGate,
)
from d052.feedback_llm_ued.runtime_authorization import (
    RealRuntimeAuthorization,
)
from d052.feedback_llm_ued.shared_runtime_binding import (
    SharedTrainingSlot,
)
from d052.feedback_llm_ued.student_binding import (
    EXECUTED_ONE_UPDATE_STATUS,
    DirectorVerifiedRunStateRoundTrip,
    FullStateRoundTripResult,
    StudentBindingBlocked,
    StudentTrainingSeam,
    TrainingStepRecord,
    consume_director_verified_round_trip,
    consume_full_state_round_trip,
    resolve_student_binding,
)

from e2_test_sign_helpers import (
    director_round_trip_payload,
    sign_director_verified_round_trip,
    sign_full_state_round_trip,
)

WINDOW = 1
CHECKPOINT = text_sha256("TEST_ONLY_POST_UPDATE_CHECKPOINT")
STATE_HASH = text_sha256("TEST_ONLY_FULL_STATE_IDENTITY")
VERIFIER_ID = text_sha256("TEST_ONLY_DIRECTOR_VERIFIER_IDENTITY")
TRAINEE_REGISTRY_ID = text_sha256("TEST_ONLY_TRAINING_CONTRACT_IDENTITY")
RUNTIME_BUNDLE_HASH = text_sha256("TEST_ONLY_RUNTIME_BUNDLE")


def attestation_payload(*, window=WINDOW, checkpoint_hash=CHECKPOINT,
                        state_before=STATE_HASH, state_after=STATE_HASH,
                        verifier_id=VERIFIER_ID, verified=True):
    return dict(window=window, checkpoint_hash=checkpoint_hash,
                state_hash_before_save=state_before,
                state_hash_after_reload=state_after,
                verifier_id=verifier_id, verified=verified)


class AttestingTrainingContract:
    """TEST_ONLY / SYNTHETIC training surface that signs a PASSING
    full-state round-trip attestation (plays the director-verifier)."""

    registry_identity = TRAINEE_REGISTRY_ID

    def __init__(self, *, attest=True, result=None):
        self._attest = attest
        self._result = result
        self.update_calls = []
        self.save_calls = []
        self.load_calls = []
        self.round_trip_verifications = []
        self._version = 0
        self._last_hash = text_sha256("TEST_ONLY_GENESIS_STATE")

    def save_checkpoint(self, *, tag):
        self._version += 1
        checkpoint_hash = text_sha256(
            f"TEST_ONLY_CHECKPOINT_STATE_{self._version}")
        self.save_calls.append((tag, checkpoint_hash))
        self._last_hash = checkpoint_hash
        return checkpoint_hash

    def run_one_optimizer_update(self, *, window, batch_candidate_ids):
        self.update_calls.append((window, tuple(batch_candidate_ids)))
        return SimpleNamespace(
            window=window, optimizer_steps=1,
            env_steps=len(list(batch_candidate_ids)),
            checkpoint_hash_before=self._last_hash,
            checkpoint_hash_after=text_sha256("TEST_ONLY_POST_UPDATE"))

    def load_checkpoint(self, *, checkpoint_hash):
        self.load_calls.append(checkpoint_hash)

    def verify_director_round_trip(self, *, window, checkpoint_hash):
        self.round_trip_verifications.append((window, checkpoint_hash))
        if not self._attest:
            raise AssertionError("test fixture must not be called")
        if self._result is not None:
            return self._result
        return sign_director_verified_round_trip(
            director_round_trip_payload(
                window, checkpoint_hash, RUNTIME_BUNDLE_HASH,
                verifier_id=VERIFIER_ID))


class ContractWithoutVerifier(AttestingTrainingContract):
    """A training surface with NO director-runtime attestation surface —
    the seam and the shared slot must both refuse it."""

    verify_director_round_trip = None


def make_seam(contract):
    authorization = RealRuntimeAuthorization(
        real_llm_backend=True, real_envcoder=True, real_probe=True,
        real_training=True)
    gate = FeedbackLaunchGate(EXECUTION_MODE_REAL,
                              runtime_grants=authorization)
    from e2_test_sign_helpers import student_contract as _sc
    identity = resolve_student_binding(
        _sc(C.STRONG_STUDENT_CANDIDATE_ID,
            runtime_bundle_hash=RUNTIME_BUNDLE_HASH),
        director_selected_candidate_id=C.STRONG_STUDENT_CANDIDATE_ID)
    return StudentTrainingSeam(gate, identity, training_contract=contract,
                               runtime_bundle_hash=RUNTIME_BUNDLE_HASH)


class TestAttestationContract:
    def test_positive_sign_and_fields(self):
        attestation = sign_full_state_round_trip(attestation_payload())
        assert attestation.window == WINDOW
        assert attestation.checkpoint_hash == CHECKPOINT
        assert attestation.verified is True
        assert (attestation.state_hash_before_save
                == attestation.state_hash_after_reload == STATE_HASH)
        assert len(attestation.round_trip_hash) == 64

    def test_signature_is_deterministic(self):
        assert (sign_full_state_round_trip(attestation_payload())
                .round_trip_hash
                == sign_full_state_round_trip(attestation_payload())
                .round_trip_hash)

    def test_frozen_model_refuses_mutation(self):
        attestation = sign_full_state_round_trip(attestation_payload())
        with pytest.raises(ValidationError, match="frozen"):
            attestation.verified = False

    def test_unsigned_attestation_refused(self):
        with pytest.raises(
                ValidationError,
                match="FULL_STATE_ROUND_TRIP_UNSIGNED"):
            FullStateRoundTripResult(**attestation_payload(),
                                     round_trip_hash="")

    def test_tampered_attestation_refused(self):
        attestation = sign_full_state_round_trip(attestation_payload())
        dump = attestation.model_dump()
        dump["state_hash_after_reload"] = text_sha256("TAMPERED_STATE")
        dump["round_trip_hash"] = attestation.round_trip_hash
        with pytest.raises(ValidationError):
            #: the tamper surfaces as either the state-mismatch or the
            #: content-hash ladder — both are fail-closed refusals
            FullStateRoundTripResult(**dump)

    def test_state_mismatch_refused(self):
        with pytest.raises(
                ValidationError,
                match="FULL_STATE_ROUND_TRIP_STATE_MISMATCH"):
            sign_full_state_round_trip(attestation_payload(
                state_after=text_sha256("TEST_ONLY_DIFFERENT_STATE")))

    def test_not_verified_refused(self):
        with pytest.raises(ValidationError,
                           match="FULL_STATE_ROUND_TRIP_NOT_VERIFIED"):
            sign_full_state_round_trip(attestation_payload(verified=False))

    @pytest.mark.parametrize("field", ["checkpoint_hash",
                                       "state_hash_before_save",
                                       "state_hash_after_reload"])
    @pytest.mark.parametrize("bad", ["z" * 64, "AB" * 32, "short"])
    def test_hashes_must_be_sha256(self, field, bad):
        payload = attestation_payload()
        payload[field] = bad
        with pytest.raises(
                ValidationError,
                match="FULL_STATE_ROUND_TRIP_HASH_NOT_SHA256"):
            sign_full_state_round_trip(payload)

    @pytest.mark.parametrize("bad", ["not-a-verifier", "Z" * 64])
    def test_verifier_identity_must_be_registry_issued_sha256(self, bad):
        with pytest.raises(
                ValidationError,
                match="FULL_STATE_ROUND_TRIP_VERIFIER_IDENTITY_INVALID"):
            sign_full_state_round_trip(attestation_payload(verifier_id=bad))

    def test_empty_verifier_identity_refused(self):
        #: empty trips the min_length ladder first — still fail-closed
        with pytest.raises(ValidationError):
            sign_full_state_round_trip(attestation_payload(verifier_id=""))

    def test_extra_field_refused(self):
        payload = attestation_payload()
        payload["optimistic_note"] = "looks fine"
        with pytest.raises(ValidationError, match="Extra inputs"):
            sign_full_state_round_trip(payload)


class TestConsumeGate:
    def test_positive_consume_binds_window_and_checkpoint(self):
        attestation = sign_full_state_round_trip(attestation_payload())
        consumed = consume_full_state_round_trip(
            attestation, window=WINDOW, checkpoint_hash=CHECKPOINT)
        assert consumed is attestation

    @pytest.mark.parametrize("raw", [
        SimpleNamespace(verified=True), "an-attestation-string", None, 7,
    ], ids=["simplenamespace", "string", "none", "int"])
    def test_duck_typed_attestations_refused(self, raw):
        with pytest.raises(StudentBindingBlocked,
                           match="REAL_TRAINING_ROUND_TRIP_NOT_SIGNED"):
            consume_full_state_round_trip(raw, window=WINDOW,
                                          checkpoint_hash=CHECKPOINT)

    def test_wrong_window_refused(self):
        attestation = sign_full_state_round_trip(attestation_payload())
        with pytest.raises(
                StudentBindingBlocked,
                match="REAL_TRAINING_ROUND_TRIP_WINDOW_MISMATCH"):
            consume_full_state_round_trip(attestation, window=WINDOW + 1,
                                          checkpoint_hash=CHECKPOINT)

    def test_wrong_checkpoint_refused(self):
        attestation = sign_full_state_round_trip(attestation_payload())
        with pytest.raises(
                StudentBindingBlocked,
                match="REAL_TRAINING_ROUND_TRIP_CHECKPOINT_MISMATCH"):
            consume_full_state_round_trip(
                attestation, window=WINDOW,
                checkpoint_hash=text_sha256("TEST_ONLY_OTHER_CHECKPOINT"))

    def test_mapping_path_validates_and_wraps(self):
        attestation = sign_full_state_round_trip(attestation_payload())
        consumed = consume_full_state_round_trip(
            dict(attestation.model_dump()), window=WINDOW,
            checkpoint_hash=CHECKPOINT)
        assert consumed.round_trip_hash == attestation.round_trip_hash
        with pytest.raises(StudentBindingBlocked,
                           match="REAL_TRAINING_ROUND_TRIP_ILLEGAL") as exc:
            consume_full_state_round_trip(
                attestation_payload(
                    state_after=text_sha256("TEST_ONLY_DIFFERENT_STATE")),
                window=WINDOW, checkpoint_hash=CHECKPOINT)
        assert "FULL_STATE_ROUND_TRIP_STATE_MISMATCH" in str(exc.value)


class TestDirectorVerifiedConsumeGate:
    """P0-16 (section 6): the PRODUCTION round-trip consumer accepts ONLY
    the director-runtime's DirectorVerifiedRunStateRoundTrip — plain
    mappings and locally-signed shapes are refused."""

    def _valid(self):
        return sign_director_verified_round_trip(
            director_round_trip_payload(WINDOW, CHECKPOINT,
                                        RUNTIME_BUNDLE_HASH))

    def test_accepts_director_verified_attestation(self):
        attestation = self._valid()
        consumed = consume_director_verified_round_trip(
            attestation, window=WINDOW, checkpoint_hash=CHECKPOINT,
            expected_runtime_bundle_hash=RUNTIME_BUNDLE_HASH)
        assert consumed is attestation
        assert consumed.verifier_implementation_hash
        assert consumed.global_env_steps > 0
        assert consumed.next_policy_step_equivalent is True

    def test_plain_mapping_rejected(self):
        payload = director_round_trip_payload(
            WINDOW, CHECKPOINT, RUNTIME_BUNDLE_HASH)
        with pytest.raises(
                StudentBindingBlocked,
                match="REAL_TRAINING_ROUND_TRIP_PLAIN_MAPPING_REJECTED"):
            consume_director_verified_round_trip(
                payload, window=WINDOW, checkpoint_hash=CHECKPOINT,
                expected_runtime_bundle_hash=RUNTIME_BUNDLE_HASH)

    def test_locally_signed_shape_rejected(self):
        local = sign_full_state_round_trip(attestation_payload())
        with pytest.raises(
                StudentBindingBlocked,
                match="REAL_TRAINING_ROUND_TRIP_NOT_DIRECTOR_VERIFIED"):
            consume_director_verified_round_trip(
                local, window=WINDOW, checkpoint_hash=CHECKPOINT,
                expected_runtime_bundle_hash=RUNTIME_BUNDLE_HASH)

    def test_duck_typed_object_rejected(self):
        with pytest.raises(
                StudentBindingBlocked,
                match="REAL_TRAINING_ROUND_TRIP_NOT_DIRECTOR_VERIFIED"):
            consume_director_verified_round_trip(
                SimpleNamespace(verified=True), window=WINDOW,
                checkpoint_hash=CHECKPOINT,
                expected_runtime_bundle_hash=RUNTIME_BUNDLE_HASH)

    def test_wrong_runtime_bundle_rejected(self):
        attestation = self._valid()
        with pytest.raises(
                StudentBindingBlocked,
                match="REAL_TRAINING_ROUND_TRIP_RUNTIME_BUNDLE_MISMATCH"):
            consume_director_verified_round_trip(
                attestation, window=WINDOW, checkpoint_hash=CHECKPOINT,
                expected_runtime_bundle_hash=text_sha256("OTHER_BUNDLE"))

    def test_wrong_window_rejected(self):
        attestation = self._valid()
        with pytest.raises(
                StudentBindingBlocked,
                match="REAL_TRAINING_ROUND_TRIP_WINDOW_MISMATCH"):
            consume_director_verified_round_trip(
                attestation, window=WINDOW + 1, checkpoint_hash=CHECKPOINT,
                expected_runtime_bundle_hash=RUNTIME_BUNDLE_HASH)


class TestSeamConsumesAttestationOnly:
    def test_executed_update_carries_verified_round_trip_pass(self):
        contract = AttestingTrainingContract()
        seam = make_seam(contract)
        record = seam.execute_real_window_update(
            WINDOW, batch_candidate_ids=["cand-a", "cand-b"])
        assert record.status == EXECUTED_ONE_UPDATE_STATUS
        assert record.checkpoint_round_trip_pass is True
        assert "VERIFIED" in record.reason
        #: the attestation was requested for the exact reloaded checkpoint
        _tag, hash_after = contract.save_calls[-1]
        assert contract.round_trip_verifications == [(WINDOW, hash_after)]
        assert contract.load_calls == [hash_after]

    def test_contract_without_verifier_surface_refused(self):
        contract = ContractWithoutVerifier()
        seam = make_seam(contract)
        with pytest.raises(
                StudentBindingBlocked,
                match="REAL_TRAINING_ROUND_TRIP_NOT_ATTESTED"):
            seam.execute_real_window_update(
                WINDOW, batch_candidate_ids=["cand-a"])

    def test_duck_typed_attestation_refused_by_seam(self):
        contract = AttestingTrainingContract(
            result=SimpleNamespace(verified=True))
        seam = make_seam(contract)
        with pytest.raises(
                StudentBindingBlocked,
                match="REAL_TRAINING_ROUND_TRIP_NOT_DIRECTOR_VERIFIED"):
            seam.execute_real_window_update(
                WINDOW, batch_candidate_ids=["cand-a"])

    def test_attestation_for_wrong_checkpoint_refused_by_seam(self):
        foreign = sign_director_verified_round_trip(
            director_round_trip_payload(
                WINDOW, text_sha256("TEST_ONLY_FOREIGN_CHECKPOINT"),
                RUNTIME_BUNDLE_HASH))
        contract = AttestingTrainingContract(result=foreign)
        seam = make_seam(contract)
        with pytest.raises(
                StudentBindingBlocked,
                match="REAL_TRAINING_ROUND_TRIP_CHECKPOINT_MISMATCH"):
            seam.execute_real_window_update(
                WINDOW, batch_candidate_ids=["cand-a"])

    def test_plain_mapping_attestation_refused_by_seam(self):
        #: a plain Mapping may never enter the production consumption
        #: surface (REAL_TRAINING_ROUND_TRIP_PLAIN_MAPPING_REJECTED)
        contract = AttestingTrainingContract(result=dict(verified=True))
        seam = make_seam(contract)
        with pytest.raises(
                StudentBindingBlocked,
                match="REAL_TRAINING_ROUND_TRIP_PLAIN_MAPPING_REJECTED"):
            seam.execute_real_window_update(
                WINDOW, batch_candidate_ids=["cand-a"])

    def test_locally_signed_attestation_refused_by_seam(self):
        #: a locally-signed FullStateRoundTripResult is NOT the director's
        #: unforgeable attestation — refused
        local = sign_full_state_round_trip(attestation_payload())
        contract = AttestingTrainingContract(result=local)
        seam = make_seam(contract)
        with pytest.raises(
                StudentBindingBlocked,
                match="REAL_TRAINING_ROUND_TRIP_NOT_DIRECTOR_VERIFIED"):
            seam.execute_real_window_update(
                WINDOW, batch_candidate_ids=["cand-a"])

    def test_non_executed_records_carry_round_trip_pass_false(self):
        #: CHECKPOINT_ROUND_TRIP_PASS is false by default — a skipped or
        #: deferred record can never imply a verified round-trip
        skipped = TrainingStepRecord(status="SKIPPED_UNAUTHORIZED",
                                     student_training_transitions=0,
                                     reason="test")
        assert skipped.checkpoint_round_trip_pass is False
        #: a REAL-mode gate WITHOUT runtime grants never authorizes
        #: training — the seam records the skip with pass=False
        from e2_test_sign_helpers import student_contract as _sc
        seam = StudentTrainingSeam(
            FeedbackLaunchGate(EXECUTION_MODE_REAL),
            resolve_student_binding(
                _sc(C.STRONG_STUDENT_CANDIDATE_ID, checkpoint_global_step=0),
                director_selected_candidate_id=C.STRONG_STUDENT_CANDIDATE_ID))
        record = seam.execute_training_step(0)
        assert record.status == "SKIPPED_UNAUTHORIZED"
        assert record.checkpoint_round_trip_pass is False


class TestTrainingSlotRequiresVerifierSurface:
    def test_slot_binds_contract_with_verifier_surface(self):
        contract = AttestingTrainingContract()
        slot = SharedTrainingSlot().bind(contract)
        assert slot.status == "BOUND"
        assert slot.registry_identity == TRAINEE_REGISTRY_ID

    def test_slot_refuses_contract_without_verifier_surface(self):
        from d052.feedback_llm_ued.shared_runtime_binding import (
            SharedBindingRejected,
        )
        with pytest.raises(SharedBindingRejected,
                           match="SHARED_TRAINING_CONTRACT_INCOMPLETE.*"
                                 "verify_director_round_trip"):
            SharedTrainingSlot().bind(ContractWithoutVerifier())


class TestPosture:
    def test_real_capability_flags_stay_false(self):
        contract = AttestingTrainingContract()
        seam = make_seam(contract)
        seam.execute_real_window_update(WINDOW,
                                        batch_candidate_ids=["cand-a"])
        for name in C.NEVER_TRUE_REAL_CAPABILITY_FLAGS:
            assert getattr(C, name) is False, name
