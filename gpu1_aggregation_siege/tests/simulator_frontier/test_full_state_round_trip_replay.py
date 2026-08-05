# -*- coding: utf-8 -*-
"""TEST_ONLY / SYNTHETIC fixtures.  NOT_REAL_EXECUTION.

CC4 follow-up (P0-14): a checkpoint round trip is never verified by the
params hash alone.  CheckpointRoundTripEvidence requires the FULL state
(params + global step) to survive the trip AND the reloaded parameters to
behave IDENTICALLY to the updated ones on one deterministic next-policy step
(action/logits/value/new-memory).  A params-only comparison or an
action-range-only replay is never accepted.
"""

import dataclasses

import numpy as np
import pytest

from dicode.simulator_frontier.errors import (
    InvalidEvidenceError,
    ProductionBlockedError,
)
from dicode.simulator_frontier.round_trip_evidence import (
    RESTORE_DRIVER_IN_PROCESS_ADAPTER,
    ROUND_TRIP_EVIDENCE_VERSION,
    CheckpointRoundTripEvidence,
    measure_replay_equivalence,
    mint_checkpoint_round_trip_evidence,
    verify_checkpoint_round_trip_evidence,
)

SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT = True

SHA_A = "a" * 64
SHA_B = "b" * 64


def _mint_kwargs(**overrides):
    kwargs = dict(
        checkpoint_path="/tmp/ckpt",
        restore_driver=RESTORE_DRIVER_IN_PROCESS_ADAPTER,
        params_sha256_saved=SHA_A,
        params_sha256_reloaded=SHA_A,
        global_step_saved=18,
        global_step_reloaded=18,
        replay_action_equal=True,
        replay_logits_equal=True,
        replay_value_equal=True,
        replay_memory_equal=True,
    )
    kwargs.update(overrides)
    return kwargs


class SyntheticStudent:
    """SYNTHETIC policy surface (NOT_REAL_EXECUTION): deterministic step."""

    def __init__(self, *, logits, value, memory, with_logits=True, with_value=True):
        self._logits = np.asarray(logits, dtype=np.float32)
        self._value = np.asarray(value, dtype=np.float32)
        self._memory = {"h": np.asarray(memory, dtype=np.float32)}
        self._with_logits = with_logits
        self._with_value = with_value

    def policy_step(self, params, observation, memory, previous_action,
                    previous_reward, rng, deterministic):
        out = {"action": int(np.argmax(self._logits)),
               "memory": dict(self._memory)}
        if self._with_logits:
            out["logits"] = self._logits
        if self._with_value:
            out["value"] = self._value
        return out


class TestRoundTripEvidence:
    def test_positive_mint_verifies_full_state(self):
        evidence = mint_checkpoint_round_trip_evidence(**_mint_kwargs())
        verify_checkpoint_round_trip_evidence(evidence)
        assert evidence.evidence_version == ROUND_TRIP_EVIDENCE_VERSION
        assert evidence.params_sha256_saved == evidence.params_sha256_reloaded
        assert evidence.global_step_saved == evidence.global_step_reloaded

    def test_mint_only_evidence_hash(self):
        with pytest.raises(TypeError):
            CheckpointRoundTripEvidence(
                checkpoint_path="/tmp/x", restore_driver="D",
                params_sha256_saved=SHA_A, params_sha256_reloaded=SHA_A,
                global_step_saved=1, global_step_reloaded=1,
                replay_action_equal=True, replay_logits_equal=True,
                replay_value_equal=True, replay_memory_equal=True,
                evidence_hash="f" * 64)


class TestLossyRoundTripNeverAttested:
    def test_params_changed_by_trip_refused(self):
        with pytest.raises(ProductionBlockedError):
            mint_checkpoint_round_trip_evidence(**_mint_kwargs(
                params_sha256_reloaded=SHA_B))

    def test_step_changed_by_trip_refused(self):
        with pytest.raises(ProductionBlockedError):
            mint_checkpoint_round_trip_evidence(**_mint_kwargs(
                global_step_reloaded=19))

    def test_invalid_step_or_digest_refused(self):
        with pytest.raises(ProductionBlockedError):
            mint_checkpoint_round_trip_evidence(**_mint_kwargs(global_step_saved=-1))
        with pytest.raises(ProductionBlockedError):
            mint_checkpoint_round_trip_evidence(**_mint_kwargs(global_step_saved=True))
        with pytest.raises(ProductionBlockedError):
            mint_checkpoint_round_trip_evidence(**_mint_kwargs(params_sha256_saved="zz"))

    def test_anonymous_or_empty_driver_refused(self):
        with pytest.raises(ProductionBlockedError):
            mint_checkpoint_round_trip_evidence(**_mint_kwargs(restore_driver="  "))

    def test_non_equivalent_replay_never_attested(self):
        with pytest.raises(ProductionBlockedError):
            mint_checkpoint_round_trip_evidence(**_mint_kwargs(replay_action_equal=False))
        with pytest.raises(ProductionBlockedError):
            mint_checkpoint_round_trip_evidence(**_mint_kwargs(replay_logits_equal=False))
        with pytest.raises(ProductionBlockedError):
            mint_checkpoint_round_trip_evidence(**_mint_kwargs(replay_memory_equal=False))


class TestReplayEquivalence:
    def test_identical_replay_measures_all_equal(self):
        student = SyntheticStudent(logits=[0.1, 0.9, 0.3], value=0.7, memory=[1.0, 2.0])
        eq = measure_replay_equivalence(
            student, params_saved=None, params_reloaded=None,
            observation=np.zeros((1, 4), dtype=np.float32),
            memory={"h": np.zeros(2, dtype=np.float32)})
        assert eq["action_equal"] and eq["logits_equal"]
        assert eq["value_equal"] and eq["memory_equal"]

    def test_drifted_reload_measures_not_equivalent(self):
        class TwoFace(SyntheticStudent):
            def __init__(self):
                super().__init__(logits=[0.1, 0.9, 0.3], value=0.7, memory=[1.0, 2.0])
                self._calls = 0

            def policy_step(self, *a, **k):
                self._calls += 1
                if self._calls >= 2:
                    self._logits = np.asarray([0.9, 0.1, 0.3], dtype=np.float32)
                return super().policy_step(*a, **k)

        eq = measure_replay_equivalence(
            TwoFace(), params_saved=None, params_reloaded=None,
            observation=np.zeros((1, 4), dtype=np.float32),
            memory={"h": np.zeros(2, dtype=np.float32)})
        assert eq["action_equal"] is False and eq["logits_equal"] is False

    def test_missing_logits_fail_closed(self):
        student = SyntheticStudent(logits=[0.2, 0.8], value=0.1, memory=[0.0],
                                   with_logits=False)
        with pytest.raises(ProductionBlockedError):
            measure_replay_equivalence(
                student, params_saved=None, params_reloaded=None,
                observation=np.zeros((1, 4), dtype=np.float32),
                memory={"h": np.zeros(1, dtype=np.float32)})

    def test_one_sided_value_is_not_equivalent(self):
        class ValueDrop(SyntheticStudent):
            def __init__(self):
                super().__init__(logits=[0.2, 0.8], value=0.1, memory=[0.0])
                self._calls = 0

            def policy_step(self, *a, **k):
                self._calls += 1
                out = super().policy_step(*a, **k)
                if self._calls >= 2:
                    out.pop("value", None)
                return out

        eq = measure_replay_equivalence(
            ValueDrop(), params_saved=None, params_reloaded=None,
            observation=np.zeros((1, 4), dtype=np.float32),
            memory={"h": np.zeros(1, dtype=np.float32)})
        assert eq["value_equal"] is False


class TestVerifyFailClosed:
    def test_mapping_and_foreign_refused(self):
        with pytest.raises(InvalidEvidenceError):
            verify_checkpoint_round_trip_evidence({"evidence_hash": "f" * 64})
        with pytest.raises(InvalidEvidenceError):
            verify_checkpoint_round_trip_evidence("evidence")

    def test_tampered_evidence_rejected(self):
        evidence = mint_checkpoint_round_trip_evidence(**_mint_kwargs())
        tampered = dataclasses.replace(evidence)
        object.__setattr__(tampered, "global_step_reloaded", 999)
        with pytest.raises(InvalidEvidenceError):
            verify_checkpoint_round_trip_evidence(tampered)
        tampered2 = dataclasses.replace(evidence)
        object.__setattr__(tampered2, "restore_driver", "OTHER_DRIVER")
        with pytest.raises(InvalidEvidenceError):
            verify_checkpoint_round_trip_evidence(tampered2)
