# -*- coding: utf-8 -*-
"""TEST_ONLY / SYNTHETIC fixtures.  NOT_REAL_EXECUTION.

CC4 follow-up (P0-16): the signed runtime bundle is the ONE real asset
injection channel.  Manifest validation is exact-key fail-closed, every
referenced file is exist + sha256 checked, synthetic signatures are rejected
everywhere, and the entrypoint refuses ad-hoc overrides.  These tests pin
the manifest contract and the entrypoint's fail-closed exit behaviour.
"""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from dicode.simulator_frontier import runtime_bundle as rb
from dicode.simulator_frontier.discovery_provenance import (
    REGISTRY_USAGE_TEST_ONLY,
    AssetKind,
    DiscoveryAssetRecord,
    DiscoveryProvenanceRegistry,
    FormalAssetIdentity,
    registry_hash_of,
)
from dicode.simulator_frontier.errors import InvalidEvidenceError

SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT = True

IDENT = "ab" * 32
SHA = "c" * 64


def _manifest() -> dict:
    manifest = {
        "schema": rb.RUNTIME_BUNDLE_SCHEMA,
        "bundle_id": "bundle-001",
        "run_id": "run-001",
        "controller_identity": "director/cc4",
        "controller_signature_ref": "controller-signature/abc",
        "student": {
            "selected_candidate_id": "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304",
            "profile_name": "rmt16_persistent_98304",
            "profile_hash": SHA,
            "checkpoint_path": "/tmp/ckpt",
            "checkpoint_file_sha256": SHA,
            "params_sha256": SHA,
            "source_commit": "src-sha256:453bd1ecc8d9671c741c4462214bd7699c74611a52ec157ff30cd68653b4bafc",
            "adapter_entrypoint": "dicode.student_adapters.rmt16_adapter:RMT16StudentAdapter",
            "adapter_implementation_hash": SHA,
            "adapter_identity_hash": IDENT,
            "memory_mode": "PERSISTENT",
            "memory_spec_hash": SHA,
            "carry_mode": "PERSISTENT",
            "driver_source_path": "/tmp/driver.py",
            "driver_source_sha256": SHA,
        },
        "reference": {"profile": "rmt16_reset128_98304",
                      "checkpoint_path": "/tmp/ref_ckpt",
                      "checkpoint_file_sha256": SHA,
                      "abi_identity_hash": "dd" * 32,
                      "adapter_entrypoint": "dicode.student_adapters.rmt16_adapter:RMT16StudentAdapter",
                      "adapter_hash": SHA,
                      "memory_mode": "SAVED_POLICY_MEMORY",
                      "memory_artifact_path": "/tmp/ref_mem.npz",
                      "memory_artifact_sha256": SHA,
                      "memory_spec_hash": SHA,
                      "memory_loader_entrypoint": "dicode.loaders:load_ref_memory",
                      "burn_in_executor_entrypoint": "dicode.loaders:burn_in_ref",
                      "history_artifact_ref": "controller-history/ref-001",
                      "reset_protocol_hash": "ee" * 32},
        "training_runtime": {"runtime_id": "rt-001",
                             "loss_name": "PPO_ORIGINAL_VTRACE",
                             "optimizer_name": "ADAMW_ORIGINAL",
                             "contract_ref": "controller-shared/cc2",
                             "loss_entrypoint": "dicode.ppo_tr:original_loss",
                             "update_entrypoint": "dicode.ppo_tr:original_update"},
        "training_surface_capability": {"descriptor_id": "cap-001",
                                        "verifier_id": "controller-audit/cc4",
                                        "signature_ref": "controller-signature/cap",
                                        "save_full_state_capable": True,
                                        "restore_full_state_capable": True},
        "memory": {"mode": "SAVED_POLICY_MEMORY",
                   "artifact_path": "/tmp/mem.npz",
                   "artifact_sha256": SHA,
                   "memory_spec_hash": SHA,
                   "student_identity_hash": IDENT,
                   "loader_entrypoint": "dicode.loaders:load_memory"},
        "capture_provenance": {"provenance": "TRAINING_DISCOVERY",
                               "rollout_protocol_id": "standard-reset/v1",
                               "world_set_hash": SHA,
                               "world_set_id": "ws-001",
                               "bank_refs": []},
        "formal_asset_registry_payload_path": "/tmp/registry.json",
        "restore_request_payload_path": "/tmp/request.json",
        "anchor_manifest_payload_path": "/tmp/anchors.json",
        "retention": {"dynamic_distribution_count": 12, "anchor_slot_count": 4,
                      "anchor_ratio": 0.25,
                      "formal_banks_in_online_curriculum": False},
        "taskparam_apply_entrypoint": "dicode.apply:taskparam_apply",
        "predicates": {"success_entrypoint": "dicode.pred:success",
                       "progress_entrypoint": "dicode.pred:progress"},
        "two_llm_runtime": {
            "descriptor_id": "llm-desc-001",
            "authorization_id": "auth-001",
            "provider": "test-provider",
            "model": "test-model",
            "client_factory_entrypoint": "dicode.clients:factory",
            "client_factory_implementation_hash": SHA,
            "token_cap": 0,
            "retry_cap": 0,
            "journal_sink": "controller-audit/llm-journal",
            "trusted_signer": "director/cc4",
        },
        "search": {"requested_n": 6, "horizon": 16, "seed_base": 7,
                   "mixed_episodes": 4, "episode_horizon": 8,
                   "max_timesteps": 256, "reset_seed": 3, "capture_at_step": 5},
        "paths": {"archive_path": "/tmp/archive.json",
                  "checkpoint_dir": "/tmp/ckpts",
                  "scratch_dir": "/tmp/scratch"},
    }
    manifest["manifest_hash"] = rb.manifest_canonical_hash(manifest)
    return manifest


class TestManifestValidation:
    def test_positive_manifest_validates(self):
        rb.validate_runtime_bundle_manifest(_manifest())

    def test_missing_and_unknown_keys_fail_closed(self):
        manifest = _manifest()
        manifest.pop("memory")
        with pytest.raises(InvalidEvidenceError):
            rb.validate_runtime_bundle_manifest(manifest)
        manifest = _manifest()
        manifest["extra"] = 1
        with pytest.raises(InvalidEvidenceError):
            rb.validate_runtime_bundle_manifest(manifest)

    def test_synthetic_signatures_rejected_everywhere(self):
        manifest = _manifest()
        manifest["controller_signature_ref"] = "SYNTHETIC_SIGNATURE_x"
        with pytest.raises(InvalidEvidenceError):
            rb.validate_runtime_bundle_manifest(manifest)
        manifest = _manifest()
        manifest["training_surface_capability"]["signature_ref"] = "SYNTHETIC_SIGNATURE_x"
        with pytest.raises(InvalidEvidenceError):
            rb.validate_runtime_bundle_manifest(manifest)

    def test_zero_memory_and_llm_runtime_rejected(self):
        manifest = _manifest()
        manifest["memory"]["mode"] = "ZERO_MEMORY"
        with pytest.raises(InvalidEvidenceError):
            rb.validate_runtime_bundle_manifest(manifest)
        manifest = _manifest()
        manifest["two_llm_runtime"] = {"client": "x"}
        with pytest.raises(InvalidEvidenceError):
            rb.validate_runtime_bundle_manifest(manifest)

    def test_bad_budget_values_rejected(self):
        manifest = _manifest()
        manifest["search"]["requested_n"] = 0
        with pytest.raises(InvalidEvidenceError):
            rb.validate_runtime_bundle_manifest(manifest)
        manifest = _manifest()
        manifest["search"]["seed_base"] = -1
        with pytest.raises(InvalidEvidenceError):
            rb.validate_runtime_bundle_manifest(manifest)

    def test_non_training_discovery_provenance_rejected(self):
        manifest = _manifest()
        manifest["capture_provenance"]["provenance"] = "FORMAL_EVAL"
        with pytest.raises(InvalidEvidenceError):
            rb.validate_runtime_bundle_manifest(manifest)


class TestAssetFileResolution:
    def test_missing_file_fails_closed(self, tmp_path):
        manifest = _manifest()
        manifest["student"]["checkpoint_path"] = str(tmp_path / "nope.ckpt")
        with pytest.raises(InvalidEvidenceError):
            rb.resolve_bundle_asset_files(manifest)

    def test_sha256_mismatch_fails_closed(self, tmp_path):
        path = tmp_path / "ckpt.bin"
        path.write_bytes(b"SYNTHETIC_CONTENT")
        manifest = _manifest()
        manifest["student"]["checkpoint_path"] = str(path)
        with pytest.raises(InvalidEvidenceError):
            rb.resolve_bundle_asset_files(manifest)

    def test_matching_file_resolves(self, tmp_path):
        path = tmp_path / "ckpt.bin"
        path.write_bytes(b"SYNTHETIC_CONTENT")
        sha = hashlib.sha256(b"SYNTHETIC_CONTENT").hexdigest()
        manifest = _manifest()
        manifest["student"]["checkpoint_path"] = str(path)
        manifest["student"]["checkpoint_file_sha256"] = sha
        # Every referenced file must exist (the memory artifact has a sha
        # check; the payloads are existence-checked).
        mem = tmp_path / "mem.npz"
        mem.write_bytes(b"SYNTHETIC_MEMORY")
        manifest["memory"]["artifact_path"] = str(mem)
        manifest["memory"]["artifact_sha256"] = hashlib.sha256(b"SYNTHETIC_MEMORY").hexdigest()
        ref_ckpt = tmp_path / "ref_ckpt.bin"
        ref_ckpt.write_bytes(b"SYNTHETIC_REFERENCE")
        manifest["reference"]["checkpoint_path"] = str(ref_ckpt)
        manifest["reference"]["checkpoint_file_sha256"] = hashlib.sha256(
            b"SYNTHETIC_REFERENCE").hexdigest()
        ref_mem = tmp_path / "ref_mem.npz"
        ref_mem.write_bytes(b"SYNTHETIC_REFERENCE_MEMORY")
        manifest["reference"]["memory_artifact_path"] = str(ref_mem)
        manifest["reference"]["memory_artifact_sha256"] = hashlib.sha256(
            b"SYNTHETIC_REFERENCE_MEMORY").hexdigest()
        for key in ("formal_asset_registry_payload_path",
                    "restore_request_payload_path",
                    "anchor_manifest_payload_path"):
            f = tmp_path / f"{key}.json"
            f.write_text("{}", encoding="utf-8")
            manifest[key] = str(f)
        resolved = rb.resolve_bundle_asset_files(manifest)
        assert resolved["student.checkpoint"] == str(path)
        assert resolved["reference.checkpoint"] == str(ref_ckpt)


class TestEntryPointResolution:
    def test_unimportable_entrypoint_fails_closed(self):
        with pytest.raises(InvalidEvidenceError):
            rb.import_entrypoint("no.such.module_c18:attr", "test")

    def test_wrong_schema_payloads_fail_closed(self):
        with pytest.raises(InvalidEvidenceError):
            rb.restore_request_from_payload({"schema": "x"})
        with pytest.raises(InvalidEvidenceError):
            rb.anchor_manifest_from_payload({"schema": "x"})
        with pytest.raises(InvalidEvidenceError):
            rb.discovery_registry_from_payload({"usage": "TEST_ONLY"})


def _test_only_registry_payload():
    forbidden = (
        FormalAssetIdentity(asset_kind=AssetKind.BANK,
                            canonical_id="formal_bank_fixture",
                            sha256="b" * 64),
        FormalAssetIdentity(asset_kind=AssetKind.WORLD_SET,
                            canonical_id="formal_world_fixture",
                            sha256="c" * 64),
    )
    allowed = (
        DiscoveryAssetRecord(asset_id="discovery_a", asset_kind=AssetKind.WORLD_SET,
                             world_set_hash="d" * 64),
        DiscoveryAssetRecord(asset_id="discovery_b", asset_kind=AssetKind.BANK,
                             content_sha256="e" * 64),
    )
    registry_hash = registry_hash_of("reg", "controller-signature/reg",
                                     forbidden, allowed,
                                     usage=REGISTRY_USAGE_TEST_ONLY)
    return {
        "registry_id": "reg",
        "controller_signature_ref": "controller-signature/reg",
        "frozen": True,
        "forbidden_formal_identities": [
            {"asset_kind": ident.asset_kind.value, "canonical_id": ident.canonical_id,
             "sha256": ident.sha256} for ident in forbidden],
        "allowed_discovery_assets": [
            {"asset_id": rec.asset_id, "asset_kind": rec.asset_kind.value,
             "world_set_hash": rec.world_set_hash, "content_sha256": rec.content_sha256}
            for rec in allowed],
        "registry_hash": registry_hash,
        "usage": REGISTRY_USAGE_TEST_ONLY,
    }


class TestRegistryPayloadRebuild:
    def test_test_only_registry_never_enters_production_slot(self):
        with pytest.raises(InvalidEvidenceError):
            rb.discovery_registry_from_payload(_test_only_registry_payload())

    def test_registry_hash_mismatch_fails_closed(self):
        payload = _test_only_registry_payload()
        payload["registry_hash"] = "0" * 64
        with pytest.raises(InvalidEvidenceError):
            rb.discovery_registry_from_payload(payload)


def _load_script(name: str):
    path = Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestEntrypointExitBehaviour:
    def test_missing_bundle_argument_fails(self, tmp_path):
        script = _load_script("run_e3_runtime_bundle")
        assert script.main([f"--out={tmp_path}"]) == script.FAIL

    def test_unknown_override_argument_fails(self, tmp_path):
        script = _load_script("run_e3_runtime_bundle")
        assert script.main(["--runtime-bundle=/tmp/x.json",
                            "student.profile=oops", f"--out={tmp_path}"]) == script.FAIL

    def test_nonexistent_bundle_fails(self, tmp_path):
        script = _load_script("run_e3_runtime_bundle")
        assert script.main([f"--runtime-bundle={tmp_path}/missing.json",
                            "--check-only", f"--out={tmp_path}"]) == script.FAIL
