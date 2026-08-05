"""§七 (dual student): the RESET128_RMT16_ORIGINAL_VTRACE_98304 Student is
accepted by the two-window check-only path with a DIFFERENT memory
identity.

Contract under test:

* a valid director Runtime Bundle selecting RESET128 is consumed; the
  entrypoint no longer blocks on the empty bundle;
* the binding's memory/carry are the legal RESET128 profile — a DIFFERENT
  memory identity from PERSISTENT.

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

RESET128 = C.RESET128_STUDENT_CANDIDATE_ID


def _load_entrypoint():
    path = (Path(__file__).resolve().parents[2] / "scripts"
            / "run_e2_real_two_window.py")
    spec = importlib.util.spec_from_file_location(
        "run_e2_real_two_window_reset128", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENTRYPOINT = _load_entrypoint()


class TestAcceptsReset128Student:
    def test_bundle_resolves_reset128_profile(self):
        manifest = valid_director_bundle(candidate_id=RESET128)
        identity = resolve_student_binding(
            student_contract(RESET128),
            director_selected_candidate_id=RESET128)
        assert identity.candidate_id == RESET128
        assert identity.memory_mode == C.STUDENT_MEMORY_MODE_RESET128
        assert identity.carry_mode == C.STUDENT_CARRY_MODE_RESET128
        assert identity.architecture_family == "RMT16"
        assert manifest.student_init_contract.candidate_id == RESET128

    def test_distinct_memory_identity_from_persistent(self):
        #: the two Students produce DIFFERENT memory identities
        persistent = resolve_student_binding(
            student_contract(C.STRONG_STUDENT_CANDIDATE_ID),
            director_selected_candidate_id=(
                C.STRONG_STUDENT_CANDIDATE_ID))
        reset128 = resolve_student_binding(
            student_contract(RESET128),
            director_selected_candidate_id=RESET128)
        assert persistent.memory_mode != reset128.memory_mode
        assert persistent.identity_hash != reset128.identity_hash

    def test_two_window_check_only_not_blocked_by_empty_bundle(
            self, tmp_path, capsys):
        manifest = valid_director_bundle(candidate_id=RESET128)
        path = tmp_path / "bundle.json"
        path.write_text(json.dumps(manifest.model_dump(), ensure_ascii=False,
                                   default=str), encoding="utf-8")
        code = ENTRYPOINT.main(
            ["--check-only", "--director-runtime-bundle", str(path),
             "--student-candidate-id", RESET128])
        assert code == 1
        report = json.loads(capsys.readouterr().out)
        codes = [b["code"] for b in report["blockers"]]
        assert C.BLOCKED_WAITING_SHARED_RUNTIME not in codes
        assert "STUDENT_INIT_CONTRACT_NOT_INJECTED" not in codes
        assert codes == ["LOCAL_RUNTIME_MODULE_MISSING"] * 2


class TestPosture:
    def test_no_capability_flag_flips(self):
        for name in C.NEVER_TRUE_REAL_CAPABILITY_FLAGS:
            assert getattr(C, name) is False, name
