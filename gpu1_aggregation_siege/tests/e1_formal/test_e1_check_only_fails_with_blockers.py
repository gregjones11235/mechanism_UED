"""CC2-Repair: any blocker => check-only returns non-zero."""
import json, os, sys
from dataclasses import replace
from types import SimpleNamespace
from dicode.teachers.e1_formal import runtime_bundle as RB
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)
import run_e1_real_one_update as ENT


class TestCheckOnlyFailsWithBlockers:
    def test_object_level_check_only_blocks_without_real_objects(self, tmp_path):
        caps = {c: SimpleNamespace(kind=c, identity_id=f"t-{c}")
                for c in RB.RUNTIME_CAPABILITY_CONTRACTS}
        b = RB.build_test_only_runtime_bundle(
            source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
            capabilities=caps)
        forged = replace(b, mode="PRODUCTION")
        m = {"bundle_id": forged.bundle_id, "mode": forged.mode,
             "source_commit": forged.source_commit, "signer_id": forged.signer_id,
             "authorization_grant_hash": forged.authorization_grant_hash,
             "object_identity_hashes": dict(forged.object_identity_hashes),
             "student_selection": forged.student_selection_mapping,
             "bundle_hash": forged.bundle_hash}
        p = tmp_path / "b.json"; p.write_text(json.dumps(m), encoding="utf-8")
        rp = str(tmp_path / "c.json")
        rc = ENT.main(["--check-only", "--director-runtime-bundle", str(p),
                       "--report-out", rp])
        assert rc != 0
        rep = json.load(open(rp, encoding="utf-8"))
        assert rep["status"] in (ENT.E1_OBJECT_LEVEL_CHECK_ONLY_BLOCKED,
                                 ENT.E1_CHECK_ONLY_BLOCKED)
