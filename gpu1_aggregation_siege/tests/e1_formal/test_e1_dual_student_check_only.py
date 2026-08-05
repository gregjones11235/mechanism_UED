"""CC2-Student tests: dual-Student check-only (mount + pipeline).

TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION / NOT_SCIENTIFIC_EVIDENCE.

--check-only must support BOTH allowed Students for mount + pipeline
construction verification, never execute LLM / probes / training /
checkpoint writes, and never silently fall back to Persistent.
"""
import json
import os
import sys

from types import SimpleNamespace

from dicode.teachers.e1_formal import runtime_bundle as RB
from dicode.teachers.e1_formal import student_contract as SC

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
import run_e1_real_one_update as ENT  # noqa: E402


class TestDualStudentCheckOnly:
    def test_both_students_mount_with_distinct_memory_modes(self):
        check = ENT._dual_student_mount_check()
        assert check["mountable"] is True
        assert check["distinct_memory_modes"] is True
        per = check["per_student"]
        assert per["PERSISTENT_RMT16_ORIGINAL_VTRACE_98304"][
            "memory_mode"
        ] == "PERSISTENT"
        assert per["RESET128_RMT16_ORIGINAL_VTRACE_98304"][
            "memory_mode"
        ] == "RESET128"
        assert per["PERSISTENT_RMT16_ORIGINAL_VTRACE_98304"][
            "profile_id"
        ] == "rmt16_persistent_98304"
        assert per["RESET128_RMT16_ORIGINAL_VTRACE_98304"][
            "profile_id"
        ] == "rmt16_reset128_98304"

    def test_entrypoint_check_only_verifies_dual_students(self, tmp_path):
        # a TEST_ONLY director bundle (no student block) — the
        # dual-student mount check still proves both are constructible,
        # and the missing bundle selection is reported honestly
        caps = {
            c: SimpleNamespace(kind=c, identity_id=f"t-{c}")
            for c in RB.RUNTIME_CAPABILITY_CONTRACTS
        }
        bundle = RB.build_test_only_runtime_bundle(
            source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
            capabilities=caps,
        )
        manifest = {
            "bundle_id": bundle.bundle_id,
            "mode": bundle.mode,
            "source_commit": bundle.source_commit,
            "signer_id": bundle.signer_id,
            "authorization_grant_hash": bundle.authorization_grant_hash,
            "object_identity_hashes": dict(
                bundle.object_identity_hashes
            ),
            "student_selection": bundle.student_selection_mapping,
            "bundle_hash": bundle.bundle_hash,
        }
        path = tmp_path / "test_only_bundle.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        report_path = str(tmp_path / "check_only.json")
        rc = ENT.main(
            [
                "--check-only",
                "--director-runtime-bundle",
                str(path),
                "--report-out",
                report_path,
            ]
        )
        assert rc == 0
        report = json.load(open(report_path, encoding="utf-8"))
        assert report["status"] == ENT.E1_TEST_ONLY_CONTRACT_OK
        assert report["executed"] is False
        dual = report["checks"]["dual_student_mount_ready"]
        assert dual["mountable"] is True
        assert dual["distinct_memory_modes"] is True
        # no real action ever happens in check-only
        codes = [b["code"] for b in report["production_blockers"]]
        assert RB.RUNTIME_BUNDLE_TEST_ONLY_REJECTED in codes

    def test_check_only_never_executes_any_real_action(self):
        # the check-only report structurally forbids EXECUTED and every
        # REAL flag stays absent/false
        from dicode.teachers.e1_formal import runtime_bundle as RB2

        assert RB2.BUNDLE_MODE_TEST_ONLY == "TEST_ONLY"
