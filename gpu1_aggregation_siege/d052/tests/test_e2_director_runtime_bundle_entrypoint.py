"""§八 (director smoke handoff): the two-window entrypoint consumes a
DIRECTOR-provided Runtime Bundle via --director-runtime-bundle.

Contract under test:

* with NO bundle, the entrypoint keeps the honest empty-bundle block
  (10 blockers — the director has provided nothing yet);
* with a VALID TEST_ONLY signed bundle, the entrypoint NO LONGER blocks
  because of the empty bundle: the five BLOCKED_WAITING_SHARED_RUNTIME
  entries, STUDENT_INIT_CONTRACT_NOT_INJECTED and
  REAL_BACKEND_IDENTITY_UNDECLARED all disappear; only the genuine local
  runtime-module blockers remain;
* --check-only validates the bindings and data flow WITHOUT executing
  any real call (no backend, no LLM, no probe, no training — the smoke
  itself is the director's job);
* an INVALID / tampered bundle fails closed (DIRECTOR_RUNTIME_BUNDLE_
  INVALID, exit 1); a bundle whose binding contract is wrong is reported
  as blockers;
* --report-out writes the JSON report to a file.

All fixtures are TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION; no REAL_*
flag is flipped.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from d052.feedback_llm_ued import constants as C

from e2_test_sign_helpers import (
    valid_director_bundle,
    valid_director_bundle_payload,
    sign_director_runtime_bundle,
)

NO_MANIFEST_EXPECTED = {
    "STUDENT_INIT_CONTRACT_NOT_INJECTED": 1,
    C.REAL_MODE_BLOCKED_NO_LLM_BACKEND: 1,
    C.BLOCKED_WAITING_SHARED_RUNTIME: 5,
    "LOCAL_RUNTIME_MODULE_MISSING": 2,
    "REAL_BACKEND_IDENTITY_UNDECLARED": 1,
}


def _load_entrypoint():
    path = (Path(__file__).resolve().parents[2] / "scripts"
            / "run_e2_real_two_window.py")
    spec = importlib.util.spec_from_file_location(
        "run_e2_real_two_window_e2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENTRYPOINT = _load_entrypoint()


def _write_bundle(manifest, tmp_path) -> str:
    path = tmp_path / "director_bundle.json"
    path.write_text(json.dumps(manifest.model_dump(), ensure_ascii=False,
                               default=str), encoding="utf-8")
    return str(path)


class TestNoManifestPreservesHonestBlock:
    def test_without_bundle_keeps_ten_blockers(self, capsys):
        code = ENTRYPOINT.main(["--check-only"])
        assert code == 1
        report = json.loads(capsys.readouterr().out)
        counts: dict = {}
        for b in report["blockers"]:
            counts[b["code"]] = counts.get(b["code"], 0) + 1
        assert counts == NO_MANIFEST_EXPECTED
        assert report["director_runtime_bundle"] is None
        assert report["e2_real_smoke_authorized"] is False
        assert report["formal_experiment_authorized"] is False


class TestValidBundleRemovesEmptyBundleBlock:
    def test_empty_bundle_block_is_gone_with_valid_bundle(self, tmp_path,
                                                          capsys):
        manifest = valid_director_bundle()
        path = _write_bundle(manifest, tmp_path)
        code = ENTRYPOINT.main(
            ["--check-only", "--director-runtime-bundle", path])
        assert code == 1          # still blocked by jax/craftax only
        report = json.loads(capsys.readouterr().out)
        codes = [b["code"] for b in report["blockers"]]
        #: the empty-bundle block is GONE
        assert C.BLOCKED_WAITING_SHARED_RUNTIME not in codes
        assert "STUDENT_INIT_CONTRACT_NOT_INJECTED" not in codes
        assert "REAL_BACKEND_IDENTITY_UNDECLARED" not in codes
        assert C.REAL_MODE_BLOCKED_NO_LLM_BACKEND not in codes
        #: only the genuine local runtime-module blockers remain
        assert set(codes) == {"LOCAL_RUNTIME_MODULE_MISSING"}
        assert len(codes) == 2
        #: the bundle identity is surfaced
        assert report["director_runtime_bundle"]["registry_identity"] \
            == manifest.registry_identity
        #: the five slots are all BOUND (director-declared / data-bound)
        for entry in report["shared_runtime_status"].values():
            assert entry["status"] == "BOUND", entry

    def test_check_only_never_invokes_a_callable(self, tmp_path, capsys):
        #: check-only must not call the LLM, the probe or training: the
        #: report carries no execution fields and no journal was persisted
        manifest = valid_director_bundle()
        path = _write_bundle(manifest, tmp_path)
        ENTRYPOINT.main(["--check-only", "--director-runtime-bundle", path])
        report = json.loads(capsys.readouterr().out)
        assert "journal_entries" not in report
        assert "optimizer_updates_executed" not in report
        assert report["blockers"]

    def test_report_out_writes_the_report(self, tmp_path, capsys):
        manifest = valid_director_bundle()
        path = _write_bundle(manifest, tmp_path)
        out_path = tmp_path / "report_out.json"
        ENTRYPOINT.main(["--check-only", "--director-runtime-bundle", path,
                         "--report-out", str(out_path)])
        written = json.loads(out_path.read_text(encoding="utf-8"))
        assert written["entrypoint"] == \
            "scripts/run_e2_real_two_window.py"
        assert "blockers" in written


class TestInvalidBundleFailsClosed:
    def test_tampered_bundle_rejected(self, tmp_path, capsys):
        manifest = valid_director_bundle()
        dump = manifest.model_dump()
        dump["transport_closure"] = "f" * 64      # tampered identity
        bad_path = tmp_path / "tampered.json"
        bad_path.write_text(json.dumps(dump, ensure_ascii=False,
                                       default=str), encoding="utf-8")
        code = ENTRYPOINT.main(
            ["--check-only", "--director-runtime-bundle", str(bad_path)])
        assert code == 1
        err = capsys.readouterr().err
        assert C.DIRECTOR_RUNTIME_BUNDLE_INVALID in err

    def test_wrong_batch_binding_reported_as_blocker(self, tmp_path, capsys):
        payload = valid_director_bundle_payload()
        payload["batch_binding"]["total_task_count"] = 17
        manifest = sign_director_runtime_bundle(payload)
        path = _write_bundle(manifest, tmp_path)
        ENTRYPOINT.main(["--check-only", "--director-runtime-bundle", path])
        report = json.loads(capsys.readouterr().out)
        codes = [b["code"] for b in report["blockers"]]
        assert C.DIRECTOR_RUNTIME_BUNDLE_INVALID in codes
        assert any("DICODE_TOTAL_COUNT_MISMATCH" in b["detail"]
                   for b in report["blockers"])

    def test_missing_bundle_path_rejected(self, tmp_path, capsys):
        code = ENTRYPOINT.main(
            ["--check-only", "--director-runtime-bundle",
             str(tmp_path / "does_not_exist.json")])
        assert code == 1
        err = capsys.readouterr().err
        assert C.DIRECTOR_RUNTIME_BUNDLE_INVALID in err


class TestPosture:
    def test_real_capability_flags_stay_false(self):
        for name in C.NEVER_TRUE_REAL_CAPABILITY_FLAGS:
            assert getattr(C, name) is False, name
        assert C.E2_REAL_SMOKE_AUTHORIZED is False
        assert C.FORMAL_EXPERIMENT_AUTHORIZED is False
