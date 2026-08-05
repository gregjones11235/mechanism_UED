"""§七 (dual student): the PERSISTENT_RMT16_ORIGINAL_VTRACE_98304 Student
is accepted by the two-window check-only path.

Contract under test:

* a valid director Runtime Bundle selecting PERSISTENT is consumed; the
  entrypoint no longer blocks on the empty bundle;
* the binding's memory/carry are the legal PERSISTENT profile.

Fixtures are TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.student_binding import resolve_student_binding

from e2_test_sign_helpers import (
    student_contract,
    valid_director_bundle,
)


def _load_entrypoint():
    path = (Path(__file__).resolve().parents[2] / "scripts"
            / "run_e2_real_two_window.py")
    spec = importlib.util.spec_from_file_location(
        "run_e2_real_two_window_persistent", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENTRYPOINT = _load_entrypoint()


class TestAcceptsPersistentStudent:
    def test_bundle_resolves_persistent_profile(self):
        manifest = valid_director_bundle(
            candidate_id=C.STRONG_STUDENT_CANDIDATE_ID)
        contract = student_contract(C.STRONG_STUDENT_CANDIDATE_ID)
        identity = resolve_student_binding(
            contract, director_selected_candidate_id=(
                C.STRONG_STUDENT_CANDIDATE_ID))
        assert identity.candidate_id == \
            C.STRONG_STUDENT_CANDIDATE_ID
        assert identity.memory_mode == C.STUDENT_MEMORY_MODE_PERSISTENT
        assert identity.carry_mode == C.STUDENT_CARRY_MODE_PERSISTENT
        assert identity.architecture_family == "RMT16"
        assert manifest.student_init_contract.candidate_id == \
            C.STRONG_STUDENT_CANDIDATE_ID

    def test_two_window_check_only_not_blocked_by_empty_bundle(
            self, tmp_path, capsys):
        manifest = valid_director_bundle(
            candidate_id=C.STRONG_STUDENT_CANDIDATE_ID)
        path = tmp_path / "bundle.json"
        path.write_text(json.dumps(manifest.model_dump(), ensure_ascii=False,
                                   default=str), encoding="utf-8")
        code = ENTRYPOINT.main(
            ["--check-only", "--director-runtime-bundle", str(path),
             "--student-candidate-id", C.STRONG_STUDENT_CANDIDATE_ID])
        assert code == 1
        report = json.loads(capsys.readouterr().out)
        codes = [b["code"] for b in report["blockers"]]
        #: the empty-bundle / identity / transport blocks are gone
        assert "STUDENT_INIT_CONTRACT_NOT_INJECTED" not in codes
        assert "REAL_BACKEND_IDENTITY_UNDECLARED" not in codes
        assert C.REAL_MODE_BLOCKED_NO_LLM_BACKEND not in codes
        #: MANIFEST_ONLY is NOT a handoff: the object slots are still
        #: DECLARED_NOT_RESOLVED (blocked) plus the local modules
        assert "LOCAL_RUNTIME_MODULE_MISSING" in codes
        assert C.BLOCKED_WAITING_SHARED_RUNTIME in codes
        assert report["student_read_only_mount_ready"] is True


class TestPosture:
    def test_no_capability_flag_flips(self):
        for name in C.NEVER_TRUE_REAL_CAPABILITY_FLAGS:
            assert getattr(C, name) is False, name
