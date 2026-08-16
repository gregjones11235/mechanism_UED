"""Capture bank tests: result-blind discipline, hash recomputation, tamper
rejection, determinism."""
from __future__ import annotations

import json

import pytest

from dicode.memory_study.ho_capture_bank import (
    FORBIDDEN_BANK_TOKENS,
    GENERATOR_SYNTHETIC,
    SCHEMA_ID_BANK,
    assign_capture,
    generate_synthetic_capture_bank,
    load_capture_bank,
    write_capture_bank,
)
from dicode.memory_study.ho_contract import FailClosed


def test_generate_and_load_roundtrip(tmp_path):
    manifest, captures = generate_synthetic_capture_bank(
        num_captures=3, segment_len=4, obs_dim=6, seed=42)
    assert manifest["schema_id"] == SCHEMA_ID_BANK
    assert manifest["generator"] == GENERATOR_SYNTHETIC
    assert len(captures) == 3
    for cap in captures:
        assert cap.bank_hash == manifest["bank_sha256"]
        cap.validate()
    write_capture_bank(tmp_path, manifest, captures)
    m2, caps2 = load_capture_bank(tmp_path / "manifest.json")
    assert m2["bank_sha256"] == manifest["bank_sha256"]
    assert [c.capture_id for c in caps2] == [c.capture_id for c in captures]
    for orig, loaded in zip(captures, caps2):
        assert orig.obs_segment == loaded.obs_segment
        assert orig.payload_sha256 == loaded.payload_sha256


def test_manifest_is_result_blind(tmp_path):
    manifest, captures = generate_synthetic_capture_bank(2, 3, 5, seed=7)
    text = json.dumps(manifest, sort_keys=True).lower()
    for tok in FORBIDDEN_BANK_TOKENS:
        assert tok not in text
    write_capture_bank(tmp_path, manifest, captures)
    for f in (tmp_path / "captures").iterdir():
        body = f.read_text(encoding="utf-8").lower()
        for tok in FORBIDDEN_BANK_TOKENS:
            assert tok not in body


def test_determinism_same_seed_same_bank():
    m1, _ = generate_synthetic_capture_bank(3, 4, 6, seed=99)
    m2, _ = generate_synthetic_capture_bank(3, 4, 6, seed=99)
    assert m1["bank_sha256"] == m2["bank_sha256"]
    m3, _ = generate_synthetic_capture_bank(3, 4, 6, seed=100)
    assert m1["bank_sha256"] != m3["bank_sha256"]


def test_bank_geometry_validated():
    with pytest.raises(FailClosed, match="BANK_GEOMETRY_INVALID"):
        generate_synthetic_capture_bank(0, 4, 6, seed=1)


def test_tampered_capture_payload_rejected(tmp_path):
    manifest, captures = generate_synthetic_capture_bank(2, 3, 5, seed=3)
    write_capture_bank(tmp_path, manifest, captures)
    victim = tmp_path / "captures" / (captures[0].capture_id + ".json")
    form = json.loads(victim.read_text(encoding="utf-8"))
    form["obs_segment"][0][0] = 0.999999
    victim.write_text(json.dumps(form), encoding="utf-8")
    with pytest.raises(FailClosed, match="CAPTURE_PAYLOAD_HASH_MISMATCH"):
        load_capture_bank(tmp_path / "manifest.json")


def test_tampered_bank_hash_rejected(tmp_path):
    manifest, captures = generate_synthetic_capture_bank(2, 3, 5, seed=3)
    write_capture_bank(tmp_path, manifest, captures)
    mpath = tmp_path / "manifest.json"
    m = json.loads(mpath.read_text(encoding="utf-8"))
    m["bank_sha256"] = "0" * 64
    mpath.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(FailClosed, match="CAPTURE_BANK_HASH_MISMATCH"):
        load_capture_bank(mpath)


def test_missing_capture_file_rejected(tmp_path):
    manifest, captures = generate_synthetic_capture_bank(2, 3, 5, seed=3)
    write_capture_bank(tmp_path, manifest, captures)
    (tmp_path / "captures" / (captures[0].capture_id + ".json")).unlink()
    with pytest.raises(FailClosed, match="CAPTURE_PAYLOAD_MISSING"):
        load_capture_bank(tmp_path / "manifest.json")


def test_missing_manifest_rejected(tmp_path):
    with pytest.raises(FailClosed, match="CAPTURE_BANK_MANIFEST_MISSING"):
        load_capture_bank(tmp_path / "nope.json")


def test_assign_capture_deterministic_and_result_blind():
    _, captures = generate_synthetic_capture_bank(5, 2, 4, seed=1)
    a1 = assign_capture(captures, "SYNSTATE0001")
    a2 = assign_capture(captures, "SYNSTATE0001")
    assert a1.capture_id == a2.capture_id
    # index depends only on the key, never on results; different keys may map
    # anywhere, but the mapping must stay in range for many keys
    for i in range(50):
        c = assign_capture(captures, "KEY%03d" % i)
        assert c in captures