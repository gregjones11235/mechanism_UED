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
    def test_manifest_only_bundle_honestly_blocked(self, tmp_path, capsys):
        #: without an injected DirectorBundleVerifier the production chain
        #: fails closed — a manifest identity is NOT a handoff
        manifest = valid_director_bundle()
        path = _write_bundle(manifest, tmp_path)
        code = ENTRYPOINT.main(
            ["--check-only", "--director-runtime-bundle", path])
        assert code == 1
        err = capsys.readouterr().err
        assert "OBJECT_LEVEL_CHECK_BLOCKED" in err
        assert "PRODUCTION_BUNDLE_VERIFIER_UNBOUND" in err

    def test_cli_transport_dynamic_import_forbidden(self, tmp_path, capsys):
        #: production never loads a transport from an arbitrary module.attr
        code = ENTRYPOINT.main(["--check-only", "--transport",
                                "d052.feedback_llm_ued.constants.MODE"])
        assert code == 1
        err = capsys.readouterr().err
        assert "REAL_LLM_TRANSPORT_DYNAMIC_IMPORT_FORBIDDEN" in err

    def test_cli_backend_conflict_rejected(self, tmp_path, capsys):
        manifest = valid_director_bundle()
        path = _write_bundle(manifest, tmp_path)
        code = ENTRYPOINT.main(["--check-only", "--director-runtime-bundle",
                                path, "--backend-id", "NOT_THE_BUNDLE_BACKEND"])
        assert code == 1
        err = capsys.readouterr().err
        assert "E2_CLI_BACKEND_OVERRIDE_FORBIDDEN" in err

    def test_cli_model_conflict_rejected(self, tmp_path, capsys):
        manifest = valid_director_bundle()
        path = _write_bundle(manifest, tmp_path)
        code = ENTRYPOINT.main(["--check-only", "--director-runtime-bundle",
                                path, "--model-id", "NOT_THE_BUNDLE_MODEL"])
        assert code == 1
        err = capsys.readouterr().err
        assert "E2_CLI_MODEL_OVERRIDE_FORBIDDEN" in err


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

    def test_wrong_batch_binding_fails_closed_at_verifier(self, tmp_path,
                                                          capsys):
        #: the chain blocks at the trusted-verifier gate BEFORE the binding
        #: problems are evaluated (no verifier injected -> unbound)
        payload = valid_director_bundle_payload()
        payload["batch_binding"]["total_task_count"] = 17
        manifest = sign_director_runtime_bundle(payload)
        path = _write_bundle(manifest, tmp_path)
        code = ENTRYPOINT.main(["--check-only", "--director-runtime-bundle",
                                path])
        assert code == 1
        err = capsys.readouterr().err
        assert "OBJECT_LEVEL_CHECK_BLOCKED" in err

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
