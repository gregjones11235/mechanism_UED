"""CC2 follow-up P0-12 tests: full-state round-trip attestation.

TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION / NOT_SCIENTIFIC_EVIDENCE:
no real checkpoint restore happens here; the production round-trip
signer whitelist is EMPTY, so production attestation fails closed.

Covered negative matrix:
* no fresh subprocess restore              -> ROUNDTRIP_NO_FRESH_RESTORE
* next-policy-step replay differs          -> ROUNDTRIP_REPLAY_MISMATCH
* leaf comparison != restored state        -> ROUNDTRIP_LEAF_MISMATCH
* attestation tamper                       -> ROUNDTRIP_HASH_MISMATCH
* identity swap                            -> ROUNDTRIP_HASH_MISMATCH
* TEST_ONLY signer on production surface   -> ROUNDTRIP_SIGNER_UNAUTHORIZED
* bad types / hashes                       -> ROUNDTRIP_BAD_TYPE
* full state list completeness (18 fields)
"""
from dataclasses import replace

import pytest

from dicode.teachers.e1_formal import roundtrip_attestation as RA

_IDENTITY = dict(
    params_hash="41" * 32,
    optimizer_state_hash="42" * 32,
    global_env_steps=8192,
    update_step=8,
    optimizer_step=43,
    training_rng_hash="43" * 32,
    env_rng_hash="44" * 32,
    env_state_hash="45" * 32,
    wrapper_state_hash="46" * 32,
    prev_action_reward_hash="47" * 32,
    policy_memory_history_hash="48" * 32,
    student_identity_hash="11" * 32,
    anchor_manifest_hash="51" * 32,
    formal_asset_registry_hash="52" * 32,
    window_hash="53" * 32,
    selection_hash="54" * 32,
    verified_batch_hash="55" * 32,
    source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
)
_RESTORED = "61" * 32
_LEAF = _RESTORED
_NEXT_STEP = "62" * 32
_SIGNER = RA.SYNTHETIC_TEST_ONLY_ROUNDTRIP_SIGNER


def _identity(**overrides):
    kwargs = dict(_IDENTITY)
    kwargs.update(overrides)
    return RA.build_full_state_checkpoint_identity(**kwargs)


def _attest(identity=None, **overrides):
    kwargs = dict(
        identity=identity or _identity(),
        restored_state_hash=_RESTORED,
        leaf_comparison_hash=_LEAF,
        next_policy_step_hash=_NEXT_STEP,
        fresh_process_restored=True,
        replay_identical=True,
        signer_id=_SIGNER,
        test_only=True,
        ctx="test",
    )
    kwargs.update(overrides)
    return RA.attest_full_state_round_trip(**kwargs)


class TestFullStateIdentity:
    def test_state_list_is_complete(self):
        assert len(RA.FULL_STATE_FIELDS) == 18
        for field in (
            "params_hash",
            "optimizer_state_hash",
            "global_env_steps",
            "update_step",
            "optimizer_step",
            "training_rng_hash",
            "env_rng_hash",
            "env_state_hash",
            "wrapper_state_hash",
            "prev_action_reward_hash",
            "policy_memory_history_hash",
            "student_identity_hash",
            "anchor_manifest_hash",
            "formal_asset_registry_hash",
            "window_hash",
            "selection_hash",
            "verified_batch_hash",
            "source_commit",
        ):
            assert field in RA.FULL_STATE_FIELDS

    def test_builds_the_checkpoint_identity(self):
        identity = _identity()
        assert len(identity.checkpoint_hash) == 64
        assert identity.params_hash == _IDENTITY["params_hash"]
        assert identity.global_env_steps == 8192
        assert identity.student_identity_hash == (
            _IDENTITY["student_identity_hash"]
        )

    def test_bad_hash_field_refused(self):
        with pytest.raises(RA.RoundtripAttestationError) as excinfo:
            _identity(params_hash="short")
        assert excinfo.value.code == RA.ROUNDTRIP_BAD_TYPE

    def test_bad_step_refused(self):
        with pytest.raises(RA.RoundtripAttestationError) as excinfo:
            _identity(global_env_steps=-1)
        assert excinfo.value.code == RA.ROUNDTRIP_BAD_TYPE


class TestRoundTripAttestation:
    def test_test_only_attestation_binds_restore_evidence(self):
        attested = _attest()
        assert attested.fresh_process_restored is True
        assert attested.replay_identical is True
        assert attested.restored_state_hash == _RESTORED
        assert attested.test_only is True
        assert len(attested.attestation_hash) == 64

    def test_verification_passes_untampered(self):
        identity = _identity()
        attested = _attest(identity)
        RA.verify_full_state_round_trip(attested, identity)

    def test_no_fresh_restore_refused(self):
        with pytest.raises(RA.RoundtripAttestationError) as excinfo:
            _attest(fresh_process_restored=False)
        assert excinfo.value.code == RA.ROUNDTRIP_NO_FRESH_RESTORE

    def test_replay_mismatch_refused(self):
        with pytest.raises(RA.RoundtripAttestationError) as excinfo:
            _attest(replay_identical=False)
        assert excinfo.value.code == RA.ROUNDTRIP_REPLAY_MISMATCH

    def test_leaf_mismatch_refused(self):
        with pytest.raises(RA.RoundtripAttestationError) as excinfo:
            _attest(leaf_comparison_hash="63" * 32)
        assert excinfo.value.code == RA.ROUNDTRIP_LEAF_MISMATCH

    def test_attestation_tamper_detected(self):
        identity = _identity()
        attested = _attest(identity)
        tampered = replace(attested, replay_identical=False)
        with pytest.raises(RA.RoundtripAttestationError) as excinfo:
            RA.verify_full_state_round_trip(tampered, identity)
        assert excinfo.value.code == RA.ROUNDTRIP_HASH_MISMATCH

    def test_identity_swap_detected(self):
        attested = _attest(_identity())
        other = _identity(student_identity_hash="ff" * 32)
        with pytest.raises(RA.RoundtripAttestationError) as excinfo:
            RA.verify_full_state_round_trip(attested, other)
        assert excinfo.value.code == RA.ROUNDTRIP_HASH_MISMATCH

    def test_production_signer_unauthorized_this_round(self):
        with pytest.raises(RA.RoundtripAttestationError) as excinfo:
            _attest(
                signer_id="would-be-roundtrip-signer",
                test_only=False,
            )
        assert excinfo.value.code == RA.ROUNDTRIP_SIGNER_UNAUTHORIZED

    def test_wrong_test_only_signer_refused(self):
        with pytest.raises(RA.RoundtripAttestationError) as excinfo:
            _attest(signer_id="attacker-roundtrip-signer")
        assert excinfo.value.code == RA.ROUNDTRIP_TEST_ONLY_REJECTED

    def test_bad_types_refused(self):
        with pytest.raises(RA.RoundtripAttestationError) as excinfo:
            _attest(identity={"identity": "summary"})
        assert excinfo.value.code == RA.ROUNDTRIP_BAD_TYPE
