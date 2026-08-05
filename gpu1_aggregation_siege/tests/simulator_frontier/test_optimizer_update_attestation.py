# -*- coding: utf-8 -*-
"""TEST_ONLY / SYNTHETIC fixtures.  NOT_REAL_EXECUTION.

CC4 follow-up (P0-13): self-reported update_count / grad_norm are never
evidence.  The OptimizerUpdateAttestation is minted from pipeline-measured
facts only — before/after params hashes, the loaded-state step baseline
(increment before -> before+1), structural finiteness and the digest of the
exact batch — and carries NO self-reported counters.
"""

import dataclasses

import numpy as np
import pytest

from dicode.simulator_frontier.errors import (
    InvalidEvidenceError,
    ProductionBlockedError,
)
from dicode.simulator_frontier.optimizer_attestation import (
    OPTIMIZER_ATTESTATION_VERSION,
    OptimizerUpdateAttestation,
    attestation_fields,
    mint_optimizer_update_attestation,
    verify_optimizer_update_attestation,
)

SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT = True

SHA_A = "a" * 64
SHA_B = "b" * 64


def _batch():
    return {
        "observations": np.zeros((4, 3), dtype=np.float32),
        "actions": np.asarray([0, 1, 2, 1], dtype=np.int64),
        "rewards": np.asarray([0.0, 1.0, 0.5, 0.0], dtype=np.float32),
        "dones": np.asarray([False, False, False, True], dtype=np.bool_),
    }


def _params():
    return {"w": np.asarray([1.0, 2.0], dtype=np.float32)}


class TestMintAndVerify:
    def test_positive_mint_attests_real_update(self):
        attestation = mint_optimizer_update_attestation(
            params_sha256_before=SHA_A, params_sha256_after=SHA_B,
            params_after=_params(), optimizer_step_before=17, batch=_batch())
        verify_optimizer_update_attestation(attestation)
        assert attestation.attestation_version == OPTIMIZER_ATTESTATION_VERSION
        assert attestation.optimizer_step_after == 18
        assert attestation.transitions == 4
        # NO self-reported counters enter the evidence.
        fields = attestation_fields(attestation)
        assert "update_count" not in fields and "grad_norm" not in fields

    def test_batch_digest_binds_the_exact_arrays(self):
        attestation = mint_optimizer_update_attestation(
            params_sha256_before=SHA_A, params_sha256_after=SHA_B,
            params_after=_params(), optimizer_step_before=17, batch=_batch())
        changed = {**_batch(),
                   "rewards": np.asarray([0.0, 2.0, 0.5, 0.0], dtype=np.float32)}
        other = mint_optimizer_update_attestation(
            params_sha256_before=SHA_A, params_sha256_after=SHA_B,
            params_after=_params(), optimizer_step_before=17, batch=changed)
        assert other.batch_digest != attestation.batch_digest

    def test_mint_only_attestation_hash(self):
        with pytest.raises(TypeError):
            OptimizerUpdateAttestation(
                params_sha256_before=SHA_A, params_sha256_after=SHA_B,
                optimizer_step_before=0, optimizer_step_after=1,
                batch_digest="c" * 64, transitions=1,
                params_changed=True, params_finite_after=True,
                attestation_hash="f" * 64)


class TestNeverAttestFakes:
    def test_bit_identical_update_never_attested(self):
        with pytest.raises(ProductionBlockedError):
            mint_optimizer_update_attestation(
                params_sha256_before=SHA_A, params_sha256_after=SHA_A,
                params_after=_params(), optimizer_step_before=0, batch=_batch())

    def test_non_finite_params_never_attested(self):
        bad = {"w": np.asarray([float("nan")])}
        with pytest.raises(ProductionBlockedError):
            mint_optimizer_update_attestation(
                params_sha256_before=SHA_A, params_sha256_after=SHA_B,
                params_after=bad, optimizer_step_before=0, batch=_batch())

    def test_negative_or_bool_step_baseline_refused(self):
        with pytest.raises(ProductionBlockedError):
            mint_optimizer_update_attestation(
                params_sha256_before=SHA_A, params_sha256_after=SHA_B,
                params_after=_params(), optimizer_step_before=-1, batch=_batch())
        with pytest.raises(ProductionBlockedError):
            mint_optimizer_update_attestation(
                params_sha256_before=SHA_A, params_sha256_after=SHA_B,
                params_after=_params(), optimizer_step_before=True, batch=_batch())

    def test_malformed_digest_refused(self):
        with pytest.raises(ProductionBlockedError):
            mint_optimizer_update_attestation(
                params_sha256_before="nothex", params_sha256_after=SHA_B,
                params_after=_params(), optimizer_step_before=0, batch=_batch())

    def test_none_params_refused(self):
        with pytest.raises(ProductionBlockedError):
            mint_optimizer_update_attestation(
                params_sha256_before=SHA_A, params_sha256_after=SHA_B,
                params_after=None, optimizer_step_before=0, batch=_batch())

    def test_incomplete_or_empty_batch_refused(self):
        with pytest.raises(ProductionBlockedError):
            mint_optimizer_update_attestation(
                params_sha256_before=SHA_A, params_sha256_after=SHA_B,
                params_after=_params(), optimizer_step_before=0,
                batch={"observations": _batch()["observations"]})
        empty = {**_batch(), "actions": np.zeros((0,), dtype=np.int64)}
        with pytest.raises(ProductionBlockedError):
            mint_optimizer_update_attestation(
                params_sha256_before=SHA_A, params_sha256_after=SHA_B,
                params_after=_params(), optimizer_step_before=0, batch=empty)


class TestStructuralInvariants:
    def test_invalid_step_increment_unconstructible(self):
        with pytest.raises(InvalidEvidenceError):
            OptimizerUpdateAttestation(
                params_sha256_before=SHA_A, params_sha256_after=SHA_B,
                optimizer_step_before=5, optimizer_step_after=7,
                batch_digest="c" * 64, transitions=1,
                params_changed=True, params_finite_after=True)

    def test_false_flags_unconstructible(self):
        for flag in ("params_changed", "params_finite_after"):
            with pytest.raises(InvalidEvidenceError):
                OptimizerUpdateAttestation(
                    params_sha256_before=SHA_A, params_sha256_after=SHA_B,
                    optimizer_step_before=0, optimizer_step_after=1,
                    batch_digest="c" * 64, transitions=1,
                    params_changed=(flag != "params_changed"),
                    params_finite_after=(flag != "params_finite_after"))


class TestVerifyFailClosed:
    def test_mapping_and_foreign_refused(self):
        with pytest.raises(InvalidEvidenceError):
            verify_optimizer_update_attestation({"attestation_hash": "f" * 64})
        with pytest.raises(InvalidEvidenceError):
            verify_optimizer_update_attestation("att")

    def test_tampered_step_or_digest_rejected(self):
        attestation = mint_optimizer_update_attestation(
            params_sha256_before=SHA_A, params_sha256_after=SHA_B,
            params_after=_params(), optimizer_step_before=17, batch=_batch())
        tampered = dataclasses.replace(attestation)
        object.__setattr__(tampered, "optimizer_step_after", 99)
        with pytest.raises(InvalidEvidenceError):
            verify_optimizer_update_attestation(tampered)
        tampered2 = dataclasses.replace(attestation)
        object.__setattr__(tampered2, "batch_digest", "d" * 64)
        with pytest.raises(InvalidEvidenceError):
            verify_optimizer_update_attestation(tampered2)
