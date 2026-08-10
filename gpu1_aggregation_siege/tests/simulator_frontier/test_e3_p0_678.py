# -*- coding: utf-8 -*-
"""P0-6/7/8 contract tests — audit-hardened (Ed25519 signature auth).

P0-6: the runner's trusted authorization is a REAL Ed25519-signed manifest
      (never a hardcoded signer string).  The runner holds only the public
      key.
P0-7: formal_asset_registry_hash is recomputed from the actual registry file
      and every asset's SHA is re-verified; HEAD / candidate / runner /
      checkpoint / profile mismatch fails closed before output/LLM/GPU.
P0-8: output directories use an atomic unique claim; existing dirs /
      duplicate run ids are rejected.
The signature mechanics themselves are covered by
test_e3_authorization_signature.py; this file tests the RUNNER integration.
"""

import base64
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

import e3_authorization as auth_mod

generate_keypair_bytes = auth_mod._ED_MODULE.generate_keypair_bytes
sign_bytes = auth_mod._ED_MODULE.sign_bytes


def _canonical(payload):
    base = {k: v for k, v in payload.items()
            if k not in ("signature", "manifest_hash")}
    return json.dumps(base, sort_keys=True, default=str, ensure_ascii=False,
                      separators=(",", ":"))


def _make_signed_env(tmpdir, *, commit="a" * 40, candidate="CAND_TEST",
                     tamper=False):
    sk, pk = generate_keypair_bytes()
    pubk = Path(tmpdir) / "pubkey.bin"
    pubk.write_bytes(pk)
    asset = Path(tmpdir) / "asset.bin"
    asset.write_bytes(b"asset-bytes-v1")
    reg = {"schema": "e3_formal_asset_registry/v2", "assets": {
        "test_asset": {"path": str(asset), "kind": "student_checkpoint",
                       "sha256": hashlib.sha256(b"asset-bytes-v1").hexdigest()}}}
    reg_path = Path(tmpdir) / "registry.json"
    canonical = json.dumps(reg, sort_keys=True, indent=2)
    reg_path.write_text(canonical, encoding="utf-8")
    reg_hash = auth_mod._sha256_file(str(reg_path))
    payload = {
        "authorization_id": "auth-p0-678",
        "source_commit": commit,
        "candidate_id": candidate,
        "runner_sha256": "b" * 64,
        "checkpoint_sha256": "c" * 64,
        "student_profile_sha256": "d" * 64,
        "task_asset_manifest_sha256": "e" * 64,
        "formal_asset_registry_hash": reg_hash,
        "allowed_heads": [],
        "issued_at_utc": "2026-08-10T00:00:00Z",
        "scope": "test",
        "signer_public_key_fingerprint": hashlib.sha256(pk).hexdigest(),
    }
    canonical_p = _canonical(payload)
    sig = sign_bytes(canonical_p.encode("utf-8"), sk)
    payload["signature"] = base64.b64encode(sig).decode("ascii")
    payload["manifest_hash"] = hashlib.sha256(
        canonical_p.encode("utf-8")).hexdigest()
    manifest = Path(tmpdir) / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    if tamper:
        m = json.loads(manifest.read_text(encoding="utf-8"))
        m["source_commit"] = "f" * 40
        manifest.write_text(json.dumps(m), encoding="utf-8")
    return str(manifest), str(pubk), str(reg_path)


def test_runner_requires_signed_auth_manifest():
    """The formal runner must use the signed-auth flow, not a hardcoded
    signer string and not the legacy controller_signature_ref."""
    src = Path(__file__).resolve().parents[2] / "scripts" / "run_e3_formal_longrun.py"
    text = src.read_text(encoding="utf-8")
    assert "--auth-manifest" in text
    assert "load_authorization(" in text
    assert "public_key_path=" in text and "registry_path=" in text
    assert "verify_runtime_authorization(" in text
    assert "E3_FORMAL_LONGRUN_AUTHORIZED = False" in text


def test_full_budget_blocked_when_not_authorized():
    import run_e3_formal_longrun as ctrl
    assert ctrl.E3_FORMAL_LONGRUN_AUTHORIZED is False
    assert ctrl.VERIFICATION_SESSIONS_MAX == 3


def test_missing_authorization_blocks_before_production_import_and_output():
    import run_e3_formal_longrun as ctrl
    with tempfile.TemporaryDirectory() as td:
        out = str(Path(td) / "must-not-exist")
        sys.modules.pop("run_e3_real_smoke", None)
        rc = ctrl.main([
            "--student=PERSISTENT_RMT16_ORIGINAL_VTRACE_98304",
            "--sessions=1", f"--out={out}",
            f"--auth-manifest={Path(td) / 'missing.json'}",
        ])
        assert rc == ctrl.BLOCKED
        assert not Path(out).exists()
        assert "run_e3_real_smoke" not in sys.modules


def test_manifest_missing_fails_closed():
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(ValueError):
            auth_mod.load_authorization(
                str(Path(td) / "missing.json"),
                public_key_path=str(Path(td) / "nokey"),
                registry_path=str(Path(td) / "noreg"))


def test_tampered_manifest_fails_closed():
    with tempfile.TemporaryDirectory() as td:
        m, pubk, reg = _make_signed_env(td, tamper=True)
        with pytest.raises(ValueError):
            auth_mod.load_authorization(m, public_key_path=pubk,
                                        registry_path=reg)


def test_registry_asset_drift_fails_closed():
    with tempfile.TemporaryDirectory() as td:
        m, pubk, reg = _make_signed_env(td)
        Path(td, "asset.bin").write_bytes(b"tampered")
        with pytest.raises(ValueError):
            auth_mod.load_authorization(m, public_key_path=pubk,
                                        registry_path=reg)


def test_head_mismatch_fails_closed():
    with tempfile.TemporaryDirectory() as td:
        m, pubk, reg = _make_signed_env(td)
        auth = auth_mod.load_authorization(m, public_key_path=pubk,
                                           registry_path=reg)
        with pytest.raises(ValueError):
            auth_mod.verify_runtime_authorization(
                auth, "d" * 40, auth.candidate_id,
                auth.runner_sha256, auth.checkpoint_sha256,
                auth.student_profile_sha256, reg)


def test_candidate_mismatch_fails_closed():
    with tempfile.TemporaryDirectory() as td:
        m, pubk, reg = _make_signed_env(td, candidate="OTHER")
        auth = auth_mod.load_authorization(m, public_key_path=pubk,
                                           registry_path=reg)
        with pytest.raises(ValueError):
            auth_mod.verify_runtime_authorization(
                auth, auth.source_commit, "CAND_TEST",
                auth.runner_sha256, auth.checkpoint_sha256,
                auth.student_profile_sha256, reg)


def test_runner_sha_mismatch_fails_closed():
    with tempfile.TemporaryDirectory() as td:
        m, pubk, reg = _make_signed_env(td)
        auth = auth_mod.load_authorization(m, public_key_path=pubk,
                                           registry_path=reg)
        with pytest.raises(ValueError):
            auth_mod.verify_runtime_authorization(
                auth, auth.source_commit, auth.candidate_id,
                "9" * 64, auth.checkpoint_sha256,
                auth.student_profile_sha256, reg)


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
