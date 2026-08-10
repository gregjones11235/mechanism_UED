# -*- coding: utf-8 -*-
"""P0-6/7/8 contract tests.

P0-6: trusted_signer must come from a controller-signed authorization manifest
      — hardcoding director/cc4 is forbidden.
P0-7: no zero-filled asset-registry hash; HEAD / authorization mismatch fails
      closed before output / LLM / GPU.
P0-8: output directories use an atomic unique claim; existing dirs / duplicate
      run ids are rejected.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

import e3_authorization as auth_mod


def _make_manifest(tmpdir, *, commit="a" * 40, candidate="PERSISTENT_RMT16_ORIGINAL_VTRACE_98304",
                   sig="controller-alpha-20260810", registry="b" * 64,
                   auth_id="auth-001"):
    payload = {
        "authorization_id": auth_id,
        "controller_signature_ref": sig,
        "source_commit": commit,
        "candidate_id": candidate,
        "formal_asset_registry_hash": registry,
        "allowed_heads": [],
    }
    payload["manifest_hash"] = auth_mod._sha256_text(
        json.dumps({k: v for k, v in payload.items() if k != "manifest_hash"},
                   sort_keys=True, default=str))
    p = Path(tmpdir) / "auth.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return str(p)


def test_p06_signer_from_manifest_not_hardcoded():
    """The formal runner must not hardcode director/cc4."""
    src = Path(__file__).resolve().parents[2] / "scripts" / "run_e3_formal_longrun.py"
    text = src.read_text(encoding="utf-8")
    assert "director/cc4" not in text
    assert "trusted_signer=trusted_signer" in text or "trusted_signer" in text
    assert "--auth-manifest" in text


def test_p06_rejects_placeholder_signer():
    with tempfile.TemporaryDirectory() as td:
        for bad in ("director/cc4", "synthetic", "none", "pending", "e3-smoke"):
            m = _make_manifest(td, sig=bad)
            with pytest.raises(ValueError):
                auth_mod.load_authorization(m)


def test_p07_rejects_zero_registry_hash():
    with tempfile.TemporaryDirectory() as td:
        m = _make_manifest(td, registry="0" * 64)
        with pytest.raises(ValueError):
            auth_mod.load_authorization(m)


def test_p07_rejects_missing_manifest():
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(ValueError):
            auth_mod.load_authorization(str(Path(td) / "missing.json"))


def test_p07_rejects_head_mismatch():
    with tempfile.TemporaryDirectory() as td:
        m = _make_manifest(td, commit="c" * 40)
        auth = auth_mod.load_authorization(m)
        with pytest.raises(ValueError):
            auth_mod.verify_runtime_authorization(auth, "d" * 40,
                                                  "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304")


def test_p07_rejects_candidate_mismatch():
    with tempfile.TemporaryDirectory() as td:
        m = _make_manifest(td, commit="a" * 40, candidate="SLOWGRU_PERSISTENT_CANONICAL_98304")
        auth = auth_mod.load_authorization(m)
        with pytest.raises(ValueError):
            auth_mod.verify_runtime_authorization(auth, "a" * 40,
                                                  "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304")


def test_p08_atomic_claim_ok():
    with tempfile.TemporaryDirectory() as td:
        out = str(Path(td) / "run1")
        auth_mod.claim_output_dir(out, "e3-run-1")
        assert (Path(out) / "CLAIM.json").is_file()


def test_p08_rejects_existing_dir():
    with tempfile.TemporaryDirectory() as td:
        out = str(Path(td) / "run1")
        auth_mod.claim_output_dir(out, "e3-run-1")
        with pytest.raises(ValueError):
            auth_mod.claim_output_dir(out, "e3-run-1")


def test_p08_rejects_duplicate_run_id_via_existing():
    with tempfile.TemporaryDirectory() as td:
        out = str(Path(td) / "run1")
        auth_mod.claim_output_dir(out, "e3-run-dup")
        with pytest.raises(ValueError):
            auth_mod.claim_output_dir(out, "e3-run-dup")


def test_auth_manifest_hash_drift_rejected():
    with tempfile.TemporaryDirectory() as td:
        m = _make_manifest(td, commit="a" * 40)
        # tamper the manifest content without updating manifest_hash
        p = Path(m)
        payload = json.loads(p.read_text(encoding="utf-8"))
        payload["formal_asset_registry_hash"] = "c" * 64
        p.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            auth_mod.load_authorization(m)
