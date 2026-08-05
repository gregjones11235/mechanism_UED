# -*- coding: utf-8 -*-
"""TEST_ONLY / SYNTHETIC fixtures.  NOT_REAL_EXECUTION.

E3-DS: a Persistent runtime bundle (selected_candidate_id
PERSISTENT_RMT16_ORIGINAL_VTRACE_98304, memory_mode/carry_mode PERSISTENT)
validates end-to-end and its carry semantics are classified as
PERSISTENT_CARRY.
"""

import pytest

from dicode.simulator_frontier import runtime_bundle as rb
from dicode.simulator_frontier.dual_student import (
    carry_semantics_snapshot,
    memory_carry_rule,
)
from dicode.simulator_frontier.errors import InvalidEvidenceError

SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT = True

PERSISTENT = "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304"
SHA = "a" * 64


def _manifest():
    manifest = {
        "schema": rb.RUNTIME_BUNDLE_SCHEMA,
        "bundle_id": "bundle-001", "run_id": "run-001",
        "controller_signature_ref": "controller-signature/abc",
        "student": {
            "selected_candidate_id": PERSISTENT,
            "profile_name": "rmt16_persistent_98304",
            "profile_hash": SHA, "checkpoint_path": "/tmp/ckpt",
            "checkpoint_file_sha256": SHA, "params_sha256": SHA,
            "source_commit": "src-sha256:" + "a" * 56,
            "adapter_entrypoint": "dicode.student_adapters.rmt16_adapter:RMT16StudentAdapter",
            "adapter_implementation_hash": SHA, "adapter_identity_hash": SHA,
            "memory_mode": "PERSISTENT", "memory_spec_hash": SHA,
            "carry_mode": "PERSISTENT",
            "driver_source_path": "/tmp/driver.py", "driver_source_sha256": SHA},
        "reference": {"profile": "rmt16_reset128_98304",
                      "checkpoint_path": "/tmp/ref", "checkpoint_file_sha256": SHA,
                      "abi_identity_hash": "dd" * 32,
                      "adapter_entrypoint": "dicode.student_adapters.rmt16_adapter:RMT16StudentAdapter",
                      "adapter_hash": SHA, "memory_mode": "SAVED_POLICY_MEMORY",
                      "memory_artifact_path": "/tmp/refm", "memory_artifact_sha256": SHA,
                      "memory_spec_hash": SHA,
                      "memory_loader_entrypoint": "dicode.l:l",
                      "burn_in_executor_entrypoint": "dicode.l:b",
                      "history_artifact_ref": "h", "reset_protocol_hash": SHA},
        "training_runtime": {"runtime_id": "r", "loss_name": "l", "optimizer_name": "o",
                             "contract_ref": "c", "loss_entrypoint": "d.m:a",
                             "update_entrypoint": "d.m:b"},
        "training_surface_capability": {"descriptor_id": "c", "verifier_id": "v",
                                        "signature_ref": "controller-signature/s",
                                        "save_full_state_capable": True,
                                        "restore_full_state_capable": True},
        "memory": {"mode": "SAVED_POLICY_MEMORY", "artifact_path": "/tmp/mem",
                   "artifact_sha256": SHA, "memory_spec_hash": SHA,
                   "student_identity_hash": SHA,
                   "loader_entrypoint": "d.l:l"},
        "capture_provenance": {"provenance": "TRAINING_DISCOVERY",
                               "rollout_protocol_id": "p", "world_set_hash": SHA,
                               "world_set_id": "w", "bank_refs": []},
        "formal_asset_registry_payload_path": "/tmp/r.json",
        "restore_request_payload_path": "/tmp/q.json",
        "anchor_manifest_payload_path": "/tmp/a.json",
        "retention": {"dynamic_distribution_count": 12, "anchor_slot_count": 4,
                      "anchor_ratio": 0.25, "formal_banks_in_online_curriculum": False},
        "taskparam_apply_entrypoint": "d.m:c",
        "predicates": {"success_entrypoint": "d.m:d", "progress_entrypoint": "d.m:e"},
        "two_llm_runtime": {"descriptor_id": "d", "authorization_id": "a",
                            "provider": "p", "model": "m",
                            "client_factory_entrypoint": "d.m:f",
                            "client_factory_implementation_hash": SHA,
                            "token_cap": 0, "retry_cap": 0,
                            "journal_sink": "j", "trusted_signer": "director/cc4"},
        "search": {"requested_n": 6, "horizon": 16, "seed_base": 7,
                   "mixed_episodes": 4, "episode_horizon": 8,
                   "max_timesteps": 256, "reset_seed": 3, "capture_at_step": 5},
        "paths": {"archive_path": "/tmp/ar", "checkpoint_dir": "/tmp/ck",
                  "scratch_dir": "/tmp/sc"},
    }
    manifest["manifest_hash"] = rb.manifest_canonical_hash(manifest)
    return manifest


def test_persistent_bundle_validates():
    rb.validate_runtime_bundle_manifest(_manifest())


def test_persistent_carry_semantics():
    snapshot = carry_semantics_snapshot(PERSISTENT)
    assert snapshot["memory_mode"] == "PERSISTENT"
    assert snapshot["carry_mode"] == "PERSISTENT"
    assert snapshot["carry_rule"] == "PERSISTENT_CARRY"
    assert snapshot["network_family"] == "RMT16"
    assert memory_carry_rule("PERSISTENT") == "PERSISTENT_CARRY"


def test_persistent_bundle_unknown_candidate_rejected():
    manifest = _manifest()
    manifest["student"]["selected_candidate_id"] = "UNKNOWN_CANDIDATE"
    with pytest.raises(InvalidEvidenceError):
        rb.validate_runtime_bundle_manifest(manifest)
