"""CC2-Repair: TEST_ONLY_CONTRACT_OK never forms a Smoke handoff."""
import json, os, sys
from types import SimpleNamespace
from dicode.teachers.e1_formal import runtime_bundle as RB
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)
import run_e1_real_one_update as ENT


class TestTestOnlyContractNotSmokeHandoff:
    def test_test_only_contract_ok_is_not_a_smoke_handoff(self, tmp_path):
        caps = {c: SimpleNamespace(kind=c, identity_id=f"t-{c}")
                for c in RB.RUNTIME_CAPABILITY_CONTRACTS}
        b = RB.build_test_only_runtime_bundle(
            source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
            capabilities=caps)
        m = {"bundle_id": b.bundle_id, "mode": b.mode,
             "source_commit": b.source_commit, "signer_id": b.signer_id,
             "authorization_grant_hash": b.authorization_grant_hash,
             "object_identity_hashes": dict(b.object_identity_hashes),
             "student_selection": b.student_selection_mapping,
             "bundle_hash": b.bundle_hash}
        p = tmp_path / "b.json"; p.write_text(json.dumps(m), encoding="utf-8")
        rp = str(tmp_path / "c.json")
        rc = ENT.main(["--check-only", "--director-runtime-bundle", str(p),
                       "--report-out", rp])
        assert rc == 0
        rep = json.load(open(rp, encoding="utf-8"))
        assert rep["status"] == ENT.E1_TEST_ONLY_CONTRACT_OK
        assert rep["smoke_handoff"] is False
        assert rep["level"] == "TEST_ONLY_CONTRACT"
