# -*- coding: utf-8 -*-
"""E3 authorization REAL-signature tests (audit item 2).

  * controller_signature_ref strings are gone: the manifest carries an Ed25519
    signature produced by the controlled local controller's private key; the
    runner holds ONLY the public key and verifies.
  * manifest binds source_commit / candidate / runner SHA / checkpoint SHA /
    student profile SHA / task-asset manifest SHA / registry hash.
  * formal_asset_registry_hash is RECOMPUTED from the actual registry file and
    every asset's file SHA is re-verified per-asset.
  * any tamper / mismatch / missing material fails closed.
"""

import base64
import hashlib
import json
import os
import tempfile

import pytest

import e3_authorization as am
from dicode.simulator_frontier.ed25519_pure import generate_keypair_bytes, sign_bytes


def _make_registry(tmpdir):
    asset = os.path.join(tmpdir, "asset.bin")
    with open(asset, "wb") as fh:
        fh.write(b"real-asset-bytes-v1")
    reg = {
        "schema": "e3_formal_asset_registry/v2",
        "assets": {
            "test_asset": {
                "path": asset,
                "kind": "student_checkpoint",
                "sha256": hashlib.sha256(b"real-asset-bytes-v1").hexdigest(),
            },
        },
    }
    reg_path = os.path.join(tmpdir, "formal_asset_registry.json")
    canonical = json.dumps(reg, sort_keys=True, indent=2)
    with open(reg_path, "w", encoding="utf-8") as fh:
        fh.write(canonical)
    return reg_path, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _canonical(payload):
    base = {k: v for k, v in payload.items()
            if k not in ("signature", "manifest_hash")}
    return json.dumps(base, sort_keys=True, default=str, ensure_ascii=False,
                      separators=(",", ":"))


def _build_signed_manifest(tmpdir, sk, pk, *, tamper_commit=False,
                           wrong_key=False):
    reg_path, reg_hash = _make_registry(tmpdir)
    pubkey_path = os.path.join(tmpdir, "pubkey.bin")
    with open(pubkey_path, "wb") as fh:
        fh.write(pk if not wrong_key else bytes(31) + bytes([pk[-1] ^ 1]))
    payload = {
        "authorization_id": "auth-test-1",
        "source_commit": "a" * 40,
        "candidate_id": "CAND_TEST",
        "runner_sha256": "b" * 64,
        "checkpoint_sha256": hashlib.sha256(b"real-asset-bytes-v1").hexdigest(),
        "student_profile_sha256": "c" * 64,
        "task_asset_manifest_sha256": "d" * 64,
        "formal_asset_registry_hash": reg_hash,
        "allowed_heads": [],
        "issued_at_utc": "2026-08-10T00:00:00Z",
        "scope": "test",
        "signer_public_key_fingerprint": hashlib.sha256(pk).hexdigest(),
    }
    canonical = _canonical(payload)
    sig = sign_bytes(canonical.encode("utf-8"), sk)
    payload["signature"] = base64.b64encode(sig).decode("ascii")
    payload["manifest_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    manifest_path = os.path.join(tmpdir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    if tamper_commit:
        # tamper the manifest FILE after signing -> signature no longer valid.
        m = json.load(open(manifest_path, encoding="utf-8"))
        m["source_commit"] = "f" * 40
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(m, fh)
    return manifest_path, pubkey_path, reg_path


def test_ed25519_roundtrip():
    sk, pk = generate_keypair_bytes()
    sig = sign_bytes(b"payload", sk)
    am.verify_bytes  # ensure import surface exists
    from dicode.simulator_frontier.ed25519_pure import verify_bytes
    verify_bytes(b"payload", sig, pk)  # no raise
    with pytest.raises(ValueError):
        verify_bytes(b"tampered", sig, pk)


def test_load_authorization_verifies_signature_registry_per_asset():
    sk, pk = generate_keypair_bytes()
    with tempfile.TemporaryDirectory() as td:
        m, pubk, reg = _build_signed_manifest(td, sk, pk)
        auth = am.load_authorization(
            m, public_key_path=pubk, registry_path=reg)
        assert auth.source_commit == "a" * 40
        assert auth.manifest_hash


def test_tampered_manifest_rejected():
    sk, pk = generate_keypair_bytes()
    with tempfile.TemporaryDirectory() as td:
        m, pubk, reg = _build_signed_manifest(td, sk, pk, tamper_commit=True)
        with pytest.raises(ValueError):
            am.load_authorization(m, public_key_path=pubk, registry_path=reg)


def test_wrong_public_key_rejected():
    sk, pk = generate_keypair_bytes()
    with tempfile.TemporaryDirectory() as td:
        m, pubk, reg = _build_signed_manifest(td, sk, pk, wrong_key=True)
        with pytest.raises(ValueError):
            am.load_authorization(m, public_key_path=pubk, registry_path=reg)


def test_registry_asset_drift_detected():
    sk, pk = generate_keypair_bytes()
    with tempfile.TemporaryDirectory() as td:
        m, pubk, reg = _build_signed_manifest(td, sk, pk)
        # corrupt the asset AFTER signing — per-asset verification must fail.
        with open(os.path.join(td, "asset.bin"), "wb") as fh:
            fh.write(b"tampered-asset-bytes")
        with pytest.raises(ValueError):
            am.load_authorization(m, public_key_path=pubk, registry_path=reg)


def test_runtime_binding_mismatch_rejected():
    sk, pk = generate_keypair_bytes()
    with tempfile.TemporaryDirectory() as td:
        m, pubk, reg = _build_signed_manifest(td, sk, pk)
        auth = am.load_authorization(m, public_key_path=pubk, registry_path=reg)
        with pytest.raises(ValueError):
            am.verify_runtime_authorization(
                auth, auth.source_commit, auth.candidate_id,
                runner_sha256="z" * 64,   # runner mismatch
                checkpoint_sha256=auth.checkpoint_sha256,
                student_profile_sha256=auth.student_profile_sha256,
                registry_path=reg)
