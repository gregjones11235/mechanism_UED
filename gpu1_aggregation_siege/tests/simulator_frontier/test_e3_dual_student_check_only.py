# -*- coding: utf-8 -*-
"""TEST_ONLY / SYNTHETIC fixtures.  NOT_REAL_EXECUTION.

E3-DS (section 7): run_e3_runtime_bundle.py supports --runtime-bundle /
--student-candidate-id / --check-only / --report-out.  A missing or
mismatched candidate is FAIL — never defaulted to the first candidate —
and check-only never executes actual-N / LLM / mixed-start / update /
checkpoint write.
"""

import importlib.util
import json
from pathlib import Path


def _load_script():
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_e3_runtime_bundle.py"
    spec = importlib.util.spec_from_file_location("run_e3_runtime_bundle", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _persistent_manifest():
    p = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "tpers", p / "test_e3_persistent_runtime_bundle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._manifest()


def test_missing_bundle_fails(tmp_path):
    script = _load_script()
    assert script.main(["--runtime-bundle=/tmp/nope.json", "--check-only",
                        f"--report-out={tmp_path}"]) == script.FAIL


def test_student_arg_mismatch_fails(tmp_path):
    manifest = _persistent_manifest()
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    script = _load_script()
    # Reset128 arg vs a Persistent bundle -> FAIL (never mixes two Students).
    assert script.main([f"--runtime-bundle={path}", "--check-only",
                        "--student-candidate-id=RESET128_RMT16_ORIGINAL_VTRACE_98304",
                        f"--report-out={tmp_path}"]) == script.FAIL


def test_no_candidate_arg_is_not_defaulted(tmp_path):
    # A bundle missing selected_candidate_id is structurally invalid (schema
    # requires it); an empty --student-candidate-id never falls back to the
    # first candidate.
    manifest = _persistent_manifest()
    manifest["student"]["selected_candidate_id"] = "UNKNOWN_CANDIDATE"
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    script = _load_script()
    assert script.main([f"--runtime-bundle={path}", "--check-only",
                        f"--report-out={tmp_path}"]) == script.FAIL
