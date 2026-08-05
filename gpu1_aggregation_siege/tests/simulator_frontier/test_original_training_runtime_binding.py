# -*- coding: utf-8 -*-
"""TEST_ONLY / SYNTHETIC fixtures.  NOT_REAL_EXECUTION.

CC4 follow-up (P0-12): the original loss and the optimizer update are never
plain callables on the production path.  They arrive through ONE minted
OriginalTrainingRuntime whose per-callable source hashes and recomputed
runtime hash make substitution, tampering and self-reporting structurally
impossible.
"""

import dataclasses

import pytest

from dicode.simulator_frontier.errors import InvalidEvidenceError
from dicode.simulator_frontier.training_runtime import (
    TRAINING_RUNTIME_VERSION,
    OriginalTrainingRuntime,
    mint_original_training_runtime,
    runtime_binding_summary,
    verify_original_training_runtime,
)

SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT = True


def original_loss(batch, params):
    """SYNTHETIC stand-in for the ORIGINAL PPO/V-trace loss (test only)."""
    return 0.5


def original_update(params, batch):
    """SYNTHETIC stand-in for the ORIGINAL optimizer update (test only)."""
    return {"params": params, "update_count": 1, "grad_norm": 0.1}


def imposter_loss(batch, params):
    """A REIMPLEMENTATION — different source text, never the original."""
    return 0.5


def _mint():
    return mint_original_training_runtime(
        loss_fn=original_loss, optimizer_update_fn=original_update,
        runtime_id="rt-fixture", loss_name="PPO_ORIGINAL_VTRACE",
        optimizer_name="ADAMW_ORIGINAL", contract_ref="controller-shared/cc2")


class TestMintAndVerify:
    def test_positive_mint_verifies(self):
        runtime = _mint()
        assert runtime.runtime_version == TRAINING_RUNTIME_VERSION
        verify_original_training_runtime(runtime)
        assert runtime.loss_source_sha256 != runtime.optimizer_source_sha256

    def test_deterministic_remint(self):
        assert _mint().runtime_hash == _mint().runtime_hash

    def test_summary_excludes_callables(self):
        summary = runtime_binding_summary(_mint())
        assert summary["bound"] is True
        assert "loss_fn" not in summary and "optimizer_update_fn" not in summary

    def test_mint_only_runtime_hash(self):
        with pytest.raises(TypeError):
            OriginalTrainingRuntime(
                runtime_id="x", loss_name="l", optimizer_name="o",
                contract_ref="c", loss_fn=original_loss,
                optimizer_update_fn=original_update,
                loss_source_sha256="a" * 64,
                optimizer_source_sha256="b" * 64,
                runtime_hash="f" * 64)


class TestSubstitutionAndTamper:
    def test_substituted_loss_detected(self):
        runtime = _mint()
        swapped = dataclasses.replace(runtime, loss_fn=imposter_loss)
        with pytest.raises(InvalidEvidenceError):
            verify_original_training_runtime(swapped)

    def test_substituted_update_detected(self):
        runtime = _mint()
        swapped = dataclasses.replace(runtime, optimizer_update_fn=imposter_loss)
        with pytest.raises(InvalidEvidenceError):
            verify_original_training_runtime(swapped)

    def test_tampered_runtime_hash_detected(self):
        runtime = _mint()
        tampered = dataclasses.replace(runtime)
        object.__setattr__(tampered, "runtime_hash", "0" * 64)
        with pytest.raises(InvalidEvidenceError):
            verify_original_training_runtime(tampered)

    def test_tampered_descriptor_detected(self):
        runtime = _mint()
        tampered = dataclasses.replace(runtime)
        object.__setattr__(tampered, "contract_ref", "other-contract")
        with pytest.raises(InvalidEvidenceError):
            verify_original_training_runtime(tampered)


class TestFailClosed:
    def test_mapping_runtime_refused(self):
        with pytest.raises(InvalidEvidenceError):
            verify_original_training_runtime({"runtime_hash": "f" * 64})

    def test_foreign_runtime_refused(self):
        with pytest.raises(InvalidEvidenceError):
            verify_original_training_runtime("runtime")

    def test_source_less_builtin_refused(self):
        with pytest.raises(InvalidEvidenceError):
            mint_original_training_runtime(
                loss_fn=len, optimizer_update_fn=original_update,
                runtime_id="x", loss_name="l", optimizer_name="o",
                contract_ref="c")

    def test_mapping_loss_refused(self):
        with pytest.raises(InvalidEvidenceError):
            mint_original_training_runtime(
                loss_fn={"call": 1}, optimizer_update_fn=original_update,
                runtime_id="x", loss_name="l", optimizer_name="o",
                contract_ref="c")

    def test_non_callable_update_refused(self):
        with pytest.raises(InvalidEvidenceError):
            mint_original_training_runtime(
                loss_fn=original_loss, optimizer_update_fn=None,
                runtime_id="x", loss_name="l", optimizer_name="o",
                contract_ref="c")

    def test_empty_descriptors_refused(self):
        with pytest.raises(InvalidEvidenceError):
            mint_original_training_runtime(
                loss_fn=original_loss, optimizer_update_fn=original_update,
                runtime_id="  ", loss_name="l", optimizer_name="o",
                contract_ref="c")
        with pytest.raises(InvalidEvidenceError):
            mint_original_training_runtime(
                loss_fn=original_loss, optimizer_update_fn=original_update,
                runtime_id="x", loss_name="", optimizer_name="o",
                contract_ref="c")
