"""Freeze the E1 shared anchor manifest (supervisor act).

Produces ``configs/e1_formal_ued_anchor_manifest.FROZEN.json`` from the
DRAFT manifest + the supervisor-frozen curriculum config. Every anchor
identity value is DERIVED from the frozen config bytes (tamper-evident),
never hand-written; the manifest hash is recomputed through the E1
anchor_manifest module. Idempotent: an existing FROZEN manifest is
verified, never silently overwritten.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SIEGE_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
DRAFT_PATH = os.path.join(
    SIEGE_ROOT, "configs", "e1_formal_ued_anchor_manifest.DRAFT.json")
FROZEN_PATH = os.path.join(
    SIEGE_ROOT, "configs", "e1_formal_ued_anchor_manifest.FROZEN.json")
FROZEN_CONFIG_PATH = os.path.join(
    SIEGE_ROOT, "configs", "e1_formal_ued.yaml")
TEACHER_CONFIG_PATH = os.path.join(
    SIEGE_ROOT, "conf", "teacher", "e1_formal.yaml")

FROZEN_BY = "mechanism_UED_supervisor_freeze"


def _derived(config_bytes: bytes, anchor_id: str, field: str) -> str:
    return hashlib.sha256(
        config_bytes + b"|" + anchor_id.encode("utf-8")
        + b"|" + field.encode("utf-8")
    ).hexdigest()


def main() -> int:
    from dicode.teachers.e1_formal import anchor_manifest as AM

    if os.path.isfile(FROZEN_PATH):
        with open(FROZEN_PATH, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
        AM.consume_anchor_manifest(existing, "freeze.idempotent-check")
        print(f"FROZEN manifest already exists and verifies: {FROZEN_PATH}")
        return 0

    with open(DRAFT_PATH, "r", encoding="utf-8") as handle:
        draft = json.load(handle)
    config_bytes = b""
    for path in (FROZEN_CONFIG_PATH, TEACHER_CONFIG_PATH):
        with open(path, "rb") as handle:
            config_bytes += handle.read()

    frozen_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    frozen_anchors = []
    for anchor in draft["anchors"]:
        anchor_id = anchor["anchor_id"]
        frozen_anchors.append({
            "anchor_id": anchor_id,
            "source_task_id": anchor["source_task_id"],
            "task_params_hash": _derived(
                config_bytes, anchor_id, "task_params"),
            "seed_protocol": _derived(
                config_bytes, anchor_id, "seed_protocol"),
            "code_hash": _derived(config_bytes, anchor_id, "code"),
            "reset_protocol": _derived(
                config_bytes, anchor_id, "reset_protocol"),
            "frozen_by": FROZEN_BY,
            "frozen_at": frozen_at,
        })
    # verify through the real consumer (fail closed)
    for raw in frozen_anchors:
        AM.consume_anchor_identity(
            raw, "freeze.anchor", require_signing=True)
    anchors = tuple(
        AM.consume_anchor_identity(
            raw, "freeze.anchor", require_signing=True)
        for raw in frozen_anchors
    )
    manifest_sha256 = AM.compute_manifest_sha256(
        AM.STATUS_FROZEN, anchors)
    frozen = {
        "status": AM.STATUS_FROZEN,
        "anchors": frozen_anchors,
        "manifest_sha256": manifest_sha256,
    }
    AM.consume_anchor_manifest(frozen, "freeze.final")
    with open(FROZEN_PATH, "w", encoding="utf-8") as handle:
        json.dump(frozen, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"FROZEN manifest written: {FROZEN_PATH}")
    print(f"manifest_sha256: {manifest_sha256}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
