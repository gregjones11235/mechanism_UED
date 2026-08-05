"""§八 (director smoke handoff, section 6): the production round-trip
consumer accepts ONLY the director-runtime's unforgeable
DirectorVerifiedRunStateRoundTrip.

Contract under test:

* the attestation verifies the FULL director surface: verifier registered
  in the FormalAssetRegistry (verifier_id) + implementation hash, runtime
  bundle hash, Student checkpoint, optimizer state,
  global_update_step/global_env_steps, RNG state, controller/feedback
  store, next-policy-step equivalence;
* the production consumer rejects plain Mappings
  (REAL_TRAINING_ROUND_TRIP_PLAIN_MAPPING_REJECTED) and locally-signed /
  duck-typed shapes (REAL_TRAINING_ROUND_TRIP_NOT_DIRECTOR_VERIFIED);
* a locally-signed FullStateRoundTripResult is NOT a round-trip;
* the production modules hold NO signer for the director attestation
  (the signing helper lives only in the tests directory).

Fixtures are TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.student_binding import (
    DirectorVerifiedRunStateRoundTrip,
    StudentBindingBlocked,
    consume_director_verified_round_trip,
)

from e2_test_sign_helpers import (
    director_round_trip_payload,
    sign_director_verified_round_trip,
    sign_full_state_round_trip,
)

WINDOW = 1
CHECKPOINT = "ab" * 32
BUNDLE = "cd" * 32


class TestDirectorAttestationContract:
    def test_positive_sign_and_fields(self):
        att = sign_director_verified_round_trip(
            director_round_trip_payload(WINDOW, CHECKPOINT, BUNDLE))
        assert att.window == WINDOW
        assert att.checkpoint_hash == CHECKPOINT
        assert att.runtime_bundle_hash == BUNDLE
        assert att.verified is True
        assert att.next_policy_step_equivalent is True
        assert att.verifier_implementation_hash
        assert att.global_env_steps > 0
        assert att.controller_store_hash
        assert len(att.attestation_hash) == 64

    def test_not_verified_refused(self):
        with pytest.raises(ValidationError,
                           match="DIRECTOR_ROUND_TRIP_NOT_VERIFIED"):
            sign_director_verified_round_trip(
                director_round_trip_payload(
                    WINDOW, CHECKPOINT, BUNDLE, verified=False))

    def test_next_step_not_equivalent_refused(self):
        with pytest.raises(
                ValidationError,
                match="DIRECTOR_ROUND_TRIP_NEXT_STEP_NOT_EQUIVALENT"):
            sign_director_verified_round_trip(
                director_round_trip_payload(
                    WINDOW, CHECKPOINT, BUNDLE,
                    next_policy_step_equivalent=False))

    def test_bad_hash_field_refused(self):
        payload = director_round_trip_payload(WINDOW, CHECKPOINT, BUNDLE)
        payload["optimizer_state_hash"] = "not-a-hash"
        with pytest.raises(ValidationError,
                           match="DIRECTOR_ROUND_TRIP_HASH_NOT_SHA256"):
            sign_director_verified_round_trip(payload)


class TestProductionConsumer:
    def test_accepts_director_verified_only(self):
        att = sign_director_verified_round_trip(
            director_round_trip_payload(WINDOW, CHECKPOINT, BUNDLE))
        assert consume_director_verified_round_trip(
            att, window=WINDOW, checkpoint_hash=CHECKPOINT,
            expected_runtime_bundle_hash=BUNDLE) is att

    def test_plain_mapping_rejected(self):
        with pytest.raises(
                StudentBindingBlocked,
                match="REAL_TRAINING_ROUND_TRIP_PLAIN_MAPPING_REJECTED"):
            consume_director_verified_round_trip(
                director_round_trip_payload(WINDOW, CHECKPOINT, BUNDLE),
                window=WINDOW, checkpoint_hash=CHECKPOINT,
                expected_runtime_bundle_hash=BUNDLE)

    def test_locally_signed_shape_rejected(self):
        #: a locally-signed FullStateRoundTripResult is NOT the director's
        #: unforgeable attestation
        local = sign_full_state_round_trip(dict(
            window=WINDOW, checkpoint_hash=CHECKPOINT,
            state_hash_before_save="ef" * 32,
            state_hash_after_reload="ef" * 32,
            verifier_id="11" * 32, verified=True))
        with pytest.raises(
                StudentBindingBlocked,
                match="REAL_TRAINING_ROUND_TRIP_NOT_DIRECTOR_VERIFIED"):
            consume_director_verified_round_trip(
                local, window=WINDOW, checkpoint_hash=CHECKPOINT,
                expected_runtime_bundle_hash=BUNDLE)

    def test_duck_typed_object_rejected(self):
        with pytest.raises(
                StudentBindingBlocked,
                match="REAL_TRAINING_ROUND_TRIP_NOT_DIRECTOR_VERIFIED"):
            consume_director_verified_round_trip(
                SimpleNamespace(verified=True), window=WINDOW,
                checkpoint_hash=CHECKPOINT,
                expected_runtime_bundle_hash=BUNDLE)

    def test_runtime_bundle_binding(self):
        att = sign_director_verified_round_trip(
            director_round_trip_payload(WINDOW, CHECKPOINT, BUNDLE))
        with pytest.raises(
                StudentBindingBlocked,
                match="REAL_TRAINING_ROUND_TRIP_RUNTIME_BUNDLE_MISMATCH"):
            consume_director_verified_round_trip(
                att, window=WINDOW, checkpoint_hash=CHECKPOINT,
                expected_runtime_bundle_hash="99" * 32)


class TestNoProductionSigner:
    def test_student_binding_has_no_local_signer(self):
        #: the production module must NOT expose sign_director_verified_
        #: round_trip / sign_full_state_round_trip
        import d052.feedback_llm_ued.student_binding as sb
        assert not hasattr(sb, "sign_full_state_round_trip")
        assert not hasattr(sb, "sign_director_verified_round_trip")

    def test_signing_helpers_live_in_tests_only(self):
        #: the tests-directory helper module holds every signer
        from e2_test_sign_helpers import (
            sign_director_verified_round_trip as d,
            sign_full_state_round_trip as f,
        )
        assert callable(d) and callable(f)


class TestPosture:
    def test_no_capability_flag_flips(self):
        for name in C.NEVER_TRUE_REAL_CAPABILITY_FLAGS:
            assert getattr(C, name) is False, name
