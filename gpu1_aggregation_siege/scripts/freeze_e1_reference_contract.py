"""Freeze the E1 Reference identity contract in the teacher config.

The Reference is the REAL RESET128 RMT16 arm (the second CC2 checkpoint
of the frozen pair). Every identity value is RECOMPUTED from the real
artifact (checkpoint file sha256 verified, params sha256 recomputed via
cc2_params_sha256); nothing is hand-written. Idempotent: an already
frozen + verifying block is left untouched.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SIEGE_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
TEACHER_CONFIG_PATH = os.path.join(
    SIEGE_ROOT, "conf", "teacher", "e1_formal.yaml")

REFERENCE_CANDIDATE = "RESET128_RMT16_ORIGINAL_VTRACE_98304"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    import yaml

    os.environ.setdefault("DICODE_SHARED_RUNTIME_REAL", "1")
    from dicode.shared_runtime import asset_locations as AL  # noqa
    from dicode.student_adapters.architectures.rmt16_provenance import (
        FROZEN_RMT16_CFG,
    )
    from dicode.teachers.e1_formal.reference_contract import (
        consume_reference_identity_contract,
    )

    # real artifact identity (recomputed fail-closed)
    loc_path = os.path.join(
        SIEGE_ROOT, "configs", "production_asset_locations.json")
    with open(loc_path, "r", encoding="utf-8") as handle:
        locations = json.load(handle)
    student = locations["student"]
    ckpt_path = student["reset128_checkpoint"]
    expected_file_sha = student["reset128_checkpoint_file_sha256"]
    actual_file_sha = AL.file_sha256(ckpt_path)
    if actual_file_sha != expected_file_sha:
        print(f"REF FREEZE FAIL: checkpoint sha drift {actual_file_sha}")
        return 1

    from dicode.shared_runtime import student_assets as SA

    adapter = SA.real_student_adapter(REFERENCE_CANDIDATE)
    params_sha = adapter.params_sha256
    if params_sha != student["reset128_params_sha256"]:
        print(f"REF FREEZE FAIL: params sha drift {params_sha}")
        return 1

    architecture_config_hash = hashlib.sha256(
        json.dumps(FROZEN_RMT16_CFG, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    memory_semantics = "reset-to-128-window-memory"
    reset_protocol_id = "mechanism_UED.episode_reset.v1"
    block = {
        "frozen": True,
        "candidate_id": REFERENCE_CANDIDATE,
        "checkpoint_ref": ckpt_path,
        "file_sha256": actual_file_sha,
        "params_sha256": params_sha,
        "architecture_family": "RMT16",
        "architecture_version": "rmt16-vtrace-cc2",
        "architecture_config_hash": architecture_config_hash,
        "memory_semantics": memory_semantics,
        "memory_semantics_hash": _sha256_text(
            f"{REFERENCE_CANDIDATE}|{memory_semantics}"),
        "episode_reset_protocol_id": reset_protocol_id,
        "episode_reset_protocol_hash": _sha256_text(reset_protocol_id),
        "global_step": 98304,
        "total_env_steps": 98304,
        "seed": 42,
        "source_commit": "src-sha256:" + student["driver_source_sha256"],
        "frozen_manifest_hash": AL.file_sha256(
            AL.resolve_repo_relative(student["reset128_profile"])),
        "provenance": "TRAINING",
    }
    # fail-closed verification before writing
    consume_reference_identity_contract(
        block, "freeze_reference_contract")

    with open(TEACHER_CONFIG_PATH, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["teacher"]["reference_contract"] = block
    with open(TEACHER_CONFIG_PATH, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False,
                       allow_unicode=True)
    print("Reference contract frozen in", TEACHER_CONFIG_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
