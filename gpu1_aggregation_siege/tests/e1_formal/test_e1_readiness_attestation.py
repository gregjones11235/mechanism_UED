"""CC2 follow-up P0-13 tests: signed readiness attestation replaces
plain-JSON trust.

TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION / NOT_SCIENTIFIC_EVIDENCE:
the production smoke-signer whitelist is EMPTY; only the
conspicuously-marked TEST_ONLY surface mints here. Plain JSON status
reports are parse-level evidence only — without a verified signed
attestation they NEVER grant readiness.

Covered negative matrix:
* status != EXECUTED                        -> SMOKE_STATUS
* production signer (whitelist EMPTY)       -> SMOKE_SIGNER_UNAUTHORIZED
* wrong TEST_ONLY signer                    -> SMOKE_TEST_ONLY_REJECTED
* attestation hash tamper                   -> SMOKE_HASH_MISMATCH
* bound-hash drift vs live expected         -> SMOKE_BOUND_HASH_MISMATCH
* claimed hash with no live counterpart     -> SMOKE_BOUND_HASH_MISMATCH
* plain EXECUTED JSON without attestation   -> readiness invalid
* unparseable / missing report              -> readiness invalid
* verified attestation + matching expected  -> readiness valid
"""
import json
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
import e1_formal_readiness as RD  # noqa: E402

from dicode.teachers.e1_formal import smoke_attestation as SM  # noqa: E402


def _expected():
    return {
        "run_id": "TEST_ONLY_SYNTHETIC_RUN",
        "branch": "test-branch",
        "git_sha": "ab" * 32,
        "runtime_bundle_hash": "01" * 32,
        "student_identity_hash": "02" * 32,
        "student_checkpoint_hash": "03" * 32,
        "reference_identity_hash": "04" * 32,
        "reference_checkpoint_hash": "05" * 32,
        "board_journal_hash": "06" * 32,
        "envcoder_artifact_pool_hash": "07" * 32,
        "probe_pool_hash": "08" * 32,
        "selection_attestation_hash": "09" * 32,
        "verified_batch_hash": "10" * 32,
        "update_attestation_hash": "11" * 32,
        "roundtrip_attestation_hash": "12" * 32,
        "formal_asset_registry_hash": "13" * 32,
        "anchor_manifest_hash": "14" * 32,
    }


def _issue(expected=None):
    expected = expected or _expected()
    return SM.issue_e1_real_smoke_attestation(
        run_id=expected["run_id"],
        branch=expected["branch"],
        git_sha=expected["git_sha"],
        runtime_bundle_hash=expected["runtime_bundle_hash"],
        student_identity_hash=expected["student_identity_hash"],
        student_checkpoint_hash=expected["student_checkpoint_hash"],
        reference_identity_hash=expected["reference_identity_hash"],
        reference_checkpoint_hash=expected["reference_checkpoint_hash"],
        board_journal_hash=expected["board_journal_hash"],
        envcoder_artifact_pool_hash=expected["envcoder_artifact_pool_hash"],
        probe_pool_hash=expected["probe_pool_hash"],
        selection_attestation_hash=expected["selection_attestation_hash"],
        verified_batch_hash=expected["verified_batch_hash"],
        update_attestation_hash=expected["update_attestation_hash"],
        roundtrip_attestation_hash=expected["roundtrip_attestation_hash"],
        formal_asset_registry_hash=expected["formal_asset_registry_hash"],
        anchor_manifest_hash=expected["anchor_manifest_hash"],
        status="EXECUTED",
        signer_id=SM.SYNTHETIC_TEST_ONLY_SMOKE_SIGNER,
        test_only=True,
        ctx="test",
    )


def _report_with(attestation_block=None, status="EXECUTED"):
    report = {"status": status, "real_one_update_executed": True,
              "flags": {
                  "real_envcoder_used": True,
                  "real_student_reference_eval": True,
                  "real_training_update_executed": True,
              }}
    if attestation_block is not None:
        report["e1_real_smoke_attestation"] = attestation_block
    return report


def _write(tmp_path, report) -> str:
    path = tmp_path / "real_one_update_status.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# minting
# ---------------------------------------------------------------------------
class TestMinting:
    def test_production_whitelist_is_empty_this_round(self):
        assert SM.AUTHORIZED_SMOKE_SIGNERS == ()

    def test_non_executed_status_refused(self):
        with pytest.raises(SM.SmokeAttestationError) as excinfo:
            SM.issue_e1_real_smoke_attestation(
                run_id="TEST_ONLY_SYNTHETIC_RUN",
                branch="test-branch",
                git_sha="ab" * 32,
                runtime_bundle_hash="01" * 32,
                student_identity_hash="02" * 32,
                student_checkpoint_hash="03" * 32,
                reference_identity_hash="04" * 32,
                reference_checkpoint_hash="05" * 32,
                board_journal_hash="06" * 32,
                envcoder_artifact_pool_hash="07" * 32,
                probe_pool_hash="08" * 32,
                selection_attestation_hash="09" * 32,
                verified_batch_hash="10" * 32,
                update_attestation_hash="11" * 32,
                roundtrip_attestation_hash="12" * 32,
                formal_asset_registry_hash="13" * 32,
                anchor_manifest_hash="14" * 32,
                status="BLOCKED",
                signer_id=SM.SYNTHETIC_TEST_ONLY_SMOKE_SIGNER,
                test_only=True,
                ctx="test",
            )
        assert excinfo.value.code == SM.SMOKE_STATUS

    def test_production_signer_unauthorized(self):
        with pytest.raises(SM.SmokeAttestationError) as excinfo:
            SM.issue_e1_real_smoke_attestation(
                run_id="TEST_ONLY_SYNTHETIC_RUN",
                branch="test-branch",
                git_sha="ab" * 32,
                runtime_bundle_hash="01" * 32,
                student_identity_hash="02" * 32,
                student_checkpoint_hash="03" * 32,
                reference_identity_hash="04" * 32,
                reference_checkpoint_hash="05" * 32,
                board_journal_hash="06" * 32,
                envcoder_artifact_pool_hash="07" * 32,
                probe_pool_hash="08" * 32,
                selection_attestation_hash="09" * 32,
                verified_batch_hash="10" * 32,
                update_attestation_hash="11" * 32,
                roundtrip_attestation_hash="12" * 32,
                formal_asset_registry_hash="13" * 32,
                anchor_manifest_hash="14" * 32,
                status="EXECUTED",
                signer_id="would-be-smoke-signer",
                test_only=False,
                ctx="test",
            )
        assert excinfo.value.code == SM.SMOKE_SIGNER_UNAUTHORIZED

    def test_wrong_test_only_signer_refused(self):
        with pytest.raises(SM.SmokeAttestationError) as excinfo:
            SM.issue_e1_real_smoke_attestation(
                run_id="TEST_ONLY_SYNTHETIC_RUN",
                branch="test-branch",
                git_sha="ab" * 32,
                runtime_bundle_hash="01" * 32,
                student_identity_hash="02" * 32,
                student_checkpoint_hash="03" * 32,
                reference_identity_hash="04" * 32,
                reference_checkpoint_hash="05" * 32,
                board_journal_hash="06" * 32,
                envcoder_artifact_pool_hash="07" * 32,
                probe_pool_hash="08" * 32,
                selection_attestation_hash="09" * 32,
                verified_batch_hash="10" * 32,
                update_attestation_hash="11" * 32,
                roundtrip_attestation_hash="12" * 32,
                formal_asset_registry_hash="13" * 32,
                anchor_manifest_hash="14" * 32,
                status="EXECUTED",
                signer_id="attacker-smoke-signer",
                test_only=True,
                ctx="test",
            )
        assert excinfo.value.code == SM.SMOKE_TEST_ONLY_REJECTED

    def test_valid_test_only_attestation_assembles(self):
        attested = _issue()
        assert attested.status == SM.SMOKE_STATUS_EXECUTED
        assert attested.test_only is True
        assert len(attested.attestation_hash) == 64
        assert len(attested.verifier_id) == 64


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------
class TestVerification:
    def test_untampered_verifies_against_matching_expected(self):
        expected = _expected()
        attested = _issue(expected)
        SM.verify_e1_real_smoke_attestation(
            attested, expected=expected
        )

    def test_hash_tamper_detected(self):
        from dataclasses import replace

        expected = _expected()
        attested = replace(_issue(expected), git_sha="ff" * 32)
        with pytest.raises(SM.SmokeAttestationError) as excinfo:
            SM.verify_e1_real_smoke_attestation(
                attested, expected=expected
            )
        assert excinfo.value.code == SM.SMOKE_HASH_MISMATCH

    def test_bound_hash_drift_detected(self):
        expected = _expected()
        attested = _issue(expected)
        drifted = dict(expected)
        drifted["probe_pool_hash"] = "ff" * 32
        with pytest.raises(SM.SmokeAttestationError) as excinfo:
            SM.verify_e1_real_smoke_attestation(
                attested, expected=drifted
            )
        assert excinfo.value.code == SM.SMOKE_BOUND_HASH_MISMATCH

    def test_claimed_hash_without_live_counterpart_refused(self):
        expected = _expected()
        expected_without_anchor = dict(expected)
        expected_without_anchor["anchor_manifest_hash"] = ""
        attested = _issue(expected)
        with pytest.raises(SM.SmokeAttestationError) as excinfo:
            SM.verify_e1_real_smoke_attestation(
                attested, expected=expected_without_anchor
            )
        assert excinfo.value.code == SM.SMOKE_BOUND_HASH_MISMATCH


# ---------------------------------------------------------------------------
# readiness evidence: plain JSON is NEVER enough
# ---------------------------------------------------------------------------
class TestReadinessEvidence:
    def test_missing_report_invalid(self, tmp_path):
        smoke = RD._compute_real_smoke_evidence(
            str(tmp_path / "no_such.json"), expected={}
        )
        assert smoke["valid"] is False

    def test_unparseable_report_invalid(self, tmp_path):
        path = tmp_path / "report.json"
        path.write_text("{not json", encoding="utf-8")
        smoke = RD._compute_real_smoke_evidence(str(path), expected={})
        assert smoke["valid"] is False
        assert smoke["detail"] == "unparseable report"

    def test_plain_executed_json_invalid(self, tmp_path):
        # STRENGTHENED pin (P0-13): EXECUTED + forged flags, NO
        # attestation block -> parse-level only, readiness stays false
        smoke = RD._compute_real_smoke_evidence(
            _write(tmp_path, _report_with()), expected={}
        )
        assert smoke["valid"] is False
        assert "e1_real_smoke_attestation" in smoke["detail"]

    def test_non_executed_status_invalid(self, tmp_path):
        smoke = RD._compute_real_smoke_evidence(
            _write(tmp_path, _report_with(status="BLOCKED")), expected={}
        )
        assert smoke["valid"] is False
        assert smoke["detail"] == "status is not EXECUTED"

    def test_verified_attestation_with_matching_expected_valid(self, tmp_path):
        expected = _expected()
        attested = _issue(expected)
        report = _report_with(attestation_block=dict(attested.__dict__))
        smoke = RD._compute_real_smoke_evidence(
            _write(tmp_path, report), expected=expected
        )
        assert smoke["valid"] is True
        assert (smoke["probe_executed"], smoke["update_executed"]) == (
            True,
            True,
        )
        assert smoke["attestation_signer"] == (
            SM.SYNTHETIC_TEST_ONLY_SMOKE_SIGNER
        )

    def test_tampered_attestation_invalid(self, tmp_path):
        expected = _expected()
        attested = _issue(expected)
        block = dict(attested.__dict__)
        block["anchor_manifest_hash"] = "ff" * 32
        report = _report_with(attestation_block=block)
        smoke = RD._compute_real_smoke_evidence(
            _write(tmp_path, report), expected=expected
        )
        assert smoke["valid"] is False

    def test_bound_hash_drift_invalid(self, tmp_path):
        expected = _expected()
        attested = _issue(expected)
        report = _report_with(attestation_block=dict(attested.__dict__))
        drifted = dict(expected)
        drifted["runtime_bundle_hash"] = "ff" * 32
        smoke = RD._compute_real_smoke_evidence(
            _write(tmp_path, report), expected=drifted
        )
        assert smoke["valid"] is False
        assert SM.SMOKE_BOUND_HASH_MISMATCH in smoke["detail"]
        assert "runtime_bundle_hash" in smoke["detail"]
