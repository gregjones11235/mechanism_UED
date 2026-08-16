"""Contract-layer tests: enums, canonical hashing, capture validation,
mechanical isolation receipts (tamper -> FailClosed)."""
from __future__ import annotations

import numpy as np
import pytest

from dicode.memory_study.ho_contract import (
    FailClosed,
    HOMode,
    HistoryCapture,
    IsolationContext,
    IsolationReceipt,
    canonical_json_bytes,
    canonical_obs_dim,
    hash_pytree,
    sha256_hex,
    structural_form,
)


def test_ho_mode_values():
    assert HOMode.BASE.value == "base"
    assert HOMode.HO_ZERO.value == "ho_zero"
    assert HOMode.HO_REAL.value == "ho_real"
    assert {m for m in HOMode} == {HOMode.BASE, HOMode.HO_ZERO, HOMode.HO_REAL}


def test_canonical_obs_dim_is_frozen_8335():
    assert canonical_obs_dim() == 8335


def test_canonical_json_key_order_invariant():
    a = canonical_json_bytes({"b": 1, "a": [1, 2]})
    b = canonical_json_bytes({"a": [1, 2], "b": 1})
    assert a == b


def test_sha256_known_vector():
    assert sha256_hex(b"") == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")


def test_hash_pytree_order_invariant_and_sensitive():
    p1 = {"a": [1.0, 2.0], "b": {"x": 3}}
    p2 = {"b": {"x": 3}, "a": [1.0, 2.0]}
    assert hash_pytree(p1) == hash_pytree(p2)
    p3 = {"a": [1.0, 2.0], "b": {"x": 4}}
    assert hash_pytree(p1) != hash_pytree(p3)


def test_hash_pytree_numpy_leaves():
    arr = np.zeros((2, 3), dtype=np.float32)
    h1 = hash_pytree({"w": arr})
    h2 = hash_pytree({"w": np.zeros((2, 3), dtype=np.float32)})
    assert h1 == h2
    h3 = hash_pytree({"w": np.ones((2, 3), dtype=np.float32)})
    assert h1 != h3


def test_structural_form_rejects_unknown_nodes():
    class Weird:
        pass
    with pytest.raises(FailClosed):
        structural_form(Weird())


def _capture(segment=((0.1, 0.2), (0.3, 0.4)), policy="P1", seed=5):
    from dicode.memory_study.ho_capture_bank import _payload_sha
    sha = _payload_sha("C0", segment, seed, policy, 2, 0)
    return HistoryCapture(capture_id="C0", obs_segment=segment,
                          source_seed=seed, capture_policy_id=policy,
                          bank_hash="BANK", payload_sha256=sha)


def test_capture_validate_ok():
    _capture().validate()


def test_capture_validate_rejects_tampered_payload():
    cap = _capture()
    bad = HistoryCapture(capture_id=cap.capture_id,
                         obs_segment=((0.1, 0.2), (0.3, 0.999)),
                         source_seed=cap.source_seed,
                         capture_policy_id=cap.capture_policy_id,
                         bank_hash=cap.bank_hash,
                         payload_sha256=cap.payload_sha256)
    with pytest.raises(FailClosed, match="CAPTURE_PAYLOAD_HASH_MISMATCH"):
        bad.validate()


def test_capture_validate_rejects_ragged_segment():
    from dicode.memory_study.ho_capture_bank import _payload_sha
    seg = ((0.1, 0.2), (0.3,))
    sha = _payload_sha("C0", ((0.1, 0.2), (0.3, 0.4)), 5, "P1", 2, 0)
    bad = HistoryCapture(capture_id="C0", obs_segment=seg, source_seed=5,
                         capture_policy_id="P1", bank_hash="B",
                         payload_sha256=sha)
    with pytest.raises(FailClosed, match="RAGGED_OBS_SEGMENT"):
        bad.validate()


def _ctx(**over):
    base = dict(params_sha_before="P", env_state_payload_hash=None,
                rng_stream_id="S", task_embedding_hash="T", timestep=0,
                inventory_hash="I", position_hash="X", entities_hash="E")
    base.update(over)
    return IsolationContext(**base)


def test_receipt_issues_pass_when_all_checks_hold():
    r = IsolationReceipt.issue("ho_real", "P", "P", _ctx(), burnin_steps=3)
    assert r.verdict == "PASS"
    assert r.burnin_steps == 3
    assert dict(r.checks)["params_invariant"] is True


def test_receipt_fails_closed_on_params_mutation():
    with pytest.raises(FailClosed, match="params_invariant"):
        IsolationReceipt.issue("ho_real", "P", "P_MUTATED", _ctx(), 3)


def test_receipt_fails_closed_when_env_state_present():
    with pytest.raises(FailClosed, match="env_state_structurally_absent"):
        IsolationReceipt.issue("ho_real", "P", "P",
                               _ctx(env_state_payload_hash="ENV"), 3)


def test_receipt_fails_closed_on_missing_declared_hashes():
    with pytest.raises(FailClosed, match="rng_stream_declared"):
        IsolationReceipt.issue("base", "P", "P", _ctx(rng_stream_id=""), 0)