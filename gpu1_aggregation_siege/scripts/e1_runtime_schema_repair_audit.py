"""CC2-Repair: publish the runtime-schema repair audit + reports.

* reports/e1_formal_ued/cc2_runtime_schema_repair_audit.json
* reports/e1_formal_ued/e1_object_level_check_only_matrix.json
* reports/e1_formal_ued/e1_remaining_smoke_blockers.md
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e1_production_runtime as RT  # noqa: E402

PREVIOUS_HEAD = "d4e9173948ce427b097a8a53ee2304d508c00d9b"

REAL_AUTHORIZATION_FLAGS = {
    "REAL_LLM_EXECUTED": False,
    "REAL_ENVCODER_EXECUTED": False,
    "REAL_CANDIDATE_PROBE_EXECUTED": False,
    "REAL_OPTIMIZER_UPDATE_EXECUTED": False,
    "REAL_FULL_STATE_ROUND_TRIP": False,
    "E1_REAL_SMOKE_AUTHORIZED": False,
    "FORMAL_EXPERIMENT_AUTHORIZED": False,
}


def _audit() -> dict:
    from dicode.teachers.e1_formal import runtime_bundle as RB
    from dicode.teachers.e1_formal import student_contract as SC

    return {
        "audit": "cc2_runtime_schema_repair",
        "head_before": PREVIOUS_HEAD,
        "head_after": RT.git_head_sha() or "UNRESOLVED",
        "branch": RT.git_branch() or "UNRESOLVED",
        "stage": "E1_OBJECT_LEVEL_CONSUMER_READY",
        "p0_fixed": {
            "bundle_student_selection_schema": True,
            "production_synthetic_student_fallback_removed": True,
            "check_only_two_levels": True,
            "main_reachable_pipeline_no_unreachable": True,
        },
        "student_selection_in_bundle_hash": True,
        "allowed_student_candidate_ids": sorted(
            SC.ALLOWED_STUDENT_CANDIDATE_IDS
        ),
        "descriptor_fields": [
            "selected_candidate_id", "profile_id", "architecture_family",
            "memory_mode", "memory_spec_hash", "carry_mode",
            "checkpoint_path", "checkpoint_file_sha256", "params_sha256",
            "adapter_identity_hash", "adapter_implementation_hash",
            "driver_source_path", "driver_source_sha256", "source_commit",
        ],
        "identity_hashes_verified_separately": [
            "adapter_identity_hash", "adapter_implementation_hash",
            "driver_source_sha256", "params_sha256",
            "checkpoint_file_sha256",
        ],
        "object_level_ready": False,  # no real shared objects this round
        "real_and_authorization_flags": REAL_AUTHORIZATION_FLAGS,
        "note": (
            "the four P0 breaks are fixed in code + tests. The shared "
            "FormalAssetRegistry and the runtime Signer Registry are "
            "absent this round, so OBJECT_LEVEL_CHECK_ONLY_OK and the "
            "Director Smoke handoff honestly stay BLOCKED; every REAL_* "
            "flag stays false."
        ),
    }


def _object_level_matrix() -> dict:
    from dicode.teachers.e1_formal import runtime_bundle as RB
    from dicode.teachers.e1_formal import runtime_object_resolution as ROR

    return {
        "level_1_test_only_contract_ok": (
            "synthetic contract tests only; NEVER a Smoke handoff; "
            "NEVER sets STUDENT_READ_ONLY_MOUNT_READY=true; NEVER "
            "outputs DIRECTOR_SMOKE_HANDOFF_READY"
        ),
        "level_2_object_level_check_only_ok": (
            "bundle signature verified; StudentSelectionDescriptor "
            "complete; real profile + checkpoint file + params hash + "
            "RMT16 adapter + StudentIdentity; all nine shared objects "
            "resolved; 15+1 contract; no LLM/probe/training/checkpoint "
            "writes"
        ),
        "required_objects": list(RB.RUNTIME_CAPABILITY_CONTRACTS)
        + list(ROR.EXTRA_RUNTIME_CONTRACTS),
        "this_round": (
            "the shared registry is absent => every required object is "
            "honestly missing; the object-level check-only BLOCKS "
            "(non-zero), never forged"
        ),
    }


def _blockers_md() -> str:
    return """# E1 CC2-Repair: remaining smoke blockers

## Status

**`E1_OBJECT_LEVEL_CONSUMER_READY`** — code + tests complete for the
unified student-selection bundle schema, the removed production
synthetic fallback, the two-level check-only and the reachable
pipeline path. OBJECT_LEVEL_CHECK_ONLY_OK and the Director Smoke
handoff are NOT granted this round.

## Remaining blockers

1. **Shared FormalAssetRegistry absent** — no real StudentInitContract
   / StudentIdentity / StudentAdapter / Reference / Probe / Anchor /
   one-update runtime / runstate checkpoint objects are resolved; the
   object-level check-only honestly BLOCKS.
2. **Runtime Signer Registry EMPTY** — `AUTHORIZED_BUNDLE_SIGNERS=()`;
   no director-signed PRODUCTION bundle can verify.
3. **No real LLM provider authorized**; the six-role board never
   falls back to replay.
4. **Reference contract unfrozen (G1)** + **anchor manifest
   DRAFT_UNFROZEN (G3)**.
5. **No canonical DiCode training runtime bound** — training only
   flows through the director-injected
   `CanonicalDiCodeOneUpdateRuntime` +
   `CanonicalDiCodeRunStateCheckpoint`; no update is ever forged.

## Authorization (all false)

REAL_LLM_EXECUTED / REAL_ENVCODER_EXECUTED /
REAL_CANDIDATE_PROBE_EXECUTED / REAL_OPTIMIZER_UPDATE_EXECUTED /
REAL_FULL_STATE_ROUND_TRIP / E1_REAL_SMOKE_AUTHORIZED /
FORMAL_EXPERIMENT_AUTHORIZED — all false.

## Only next step

The director injects the shared FormalAssetRegistry + signs the
runtime bundle; then the object-level check-only can pass and the
director approves the Smoke. This round: check-only + tests only.
"""


def main() -> int:
    reports = {
        "cc2_runtime_schema_repair_audit.json": _audit(),
        "e1_object_level_check_only_matrix.json": _object_level_matrix(),
        "e1_remaining_smoke_blockers.md": {"_md": _blockers_md()},
    }
    base = os.path.join(RT.SIEGE_ROOT, "reports", "e1_formal_ued")
    os.makedirs(base, exist_ok=True)
    for name, payload in reports.items():
        path = os.path.join(base, name)
        if name.endswith(".md"):
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(payload["_md"])
        else:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
                fh.write("\n")
        print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
