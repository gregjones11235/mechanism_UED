"""§19 seam coverage: the real two-window entrypoint's fail-closed blocker
discovery (scripts/run_e2_real_two_window.py).

Contract under test:

* ``discover_blockers`` enumerates EVERY reason a real two-window run
  cannot start, fail-closed codes first: the injected StudentInitContract
  (STUDENT_INIT_CONTRACT_NOT_INJECTED), the real LLM transport
  (REAL_MODE_BLOCKED_NO_LLM_BACKEND), every missing shared asset
  (BLOCKED_WAITING_SHARED_RUNTIME), the local jax/craftax modules
  (LOCAL_RUNTIME_MODULE_MISSING) and the declared backend/model identity
  (REAL_BACKEND_IDENTITY_UNDECLARED);
* even a FULLY bound TEST_ONLY bundle still blocks on the absent local
  runtime modules — the entrypoint never reaches execution;
* the transport loader refuses a non-callable transport
  (REAL_LLM_TRANSPORT_NOT_CALLABLE) and an absent one returns None;
* ``main(--check-only)`` prints the report and exits 1 with the exact 10
  blockers, all REAL_* flags False and the pilot False — the entrypoint
  NEVER falls back and claims to be real (NO_SILENT_FALLBACK).

Fixtures are TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from d052.feedback_llm_ued import constants as C

from test_feedback_llm_ued_shared_runtime_identity import (
    fully_bound_bundle,
)


def _load_entrypoint():
    path = (Path(__file__).resolve().parents[2] / "scripts"
            / "run_e2_real_two_window.py")
    spec = importlib.util.spec_from_file_location(
        "run_e2_real_two_window_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENTRYPOINT = _load_entrypoint()

EXPECTED_DEFAULT_CODES = {
    "STUDENT_INIT_CONTRACT_NOT_INJECTED": 1,
    C.REAL_MODE_BLOCKED_NO_LLM_BACKEND: 1,
    C.BLOCKED_WAITING_SHARED_RUNTIME: 5,
    "LOCAL_RUNTIME_MODULE_MISSING": 2,
    "REAL_BACKEND_IDENTITY_UNDECLARED": 1,
}


class TestDiscoverBlockers:
    def test_default_bundle_enumerates_every_blocker(self):
        blockers = ENTRYPOINT.discover_blockers(
            bundle=ENTRYPOINT.SharedRuntimeBundle(),
            llm_transport=None, backend_id="", model_id="",
            student_init_contract=None)
        counts: dict = {}
        for code, _detail in blockers:
            counts[code] = counts.get(code, 0) + 1
        assert counts == EXPECTED_DEFAULT_CODES
        assert len(blockers) == 10

    def test_fully_bound_bundle_still_blocks_on_local_modules(self):
        #: even every shared asset bound with TEST_ONLY contracts, the
        #: absent jax/craftax runtime keeps the path blocked
        bundle = fully_bound_bundle()
        transport = lambda role, prompt: None  # noqa: E731
        blockers = ENTRYPOINT.discover_blockers(
            bundle=bundle, llm_transport=transport,
            backend_id="test.backend.v1", model_id="test-model.v1",
            student_init_contract=object())
        codes = {code for code, _ in blockers}
        assert codes == {"LOCAL_RUNTIME_MODULE_MISSING"}
        assert len(blockers) == 2

    def test_missing_student_contract_reported(self):
        blockers = ENTRYPOINT.discover_blockers(
            bundle=ENTRYPOINT.SharedRuntimeBundle(),
            llm_transport=lambda role, prompt: None,
            backend_id="b", model_id="m", student_init_contract=None)
        codes = {code for code, _ in blockers}
        assert "STUDENT_INIT_CONTRACT_NOT_INJECTED" in codes

    def test_missing_transport_reported(self):
        blockers = ENTRYPOINT.discover_blockers(
            bundle=ENTRYPOINT.SharedRuntimeBundle(),
            llm_transport=None, backend_id="b", model_id="m",
            student_init_contract=object())
        codes = {code for code, _ in blockers}
        assert C.REAL_MODE_BLOCKED_NO_LLM_BACKEND in codes

    def test_undeclared_backend_identity_reported(self):
        blockers = ENTRYPOINT.discover_blockers(
            bundle=ENTRYPOINT.SharedRuntimeBundle(),
            llm_transport=lambda role, prompt: None,
            backend_id="", model_id="", student_init_contract=object())
        codes = {code for code, _ in blockers}
        assert "REAL_BACKEND_IDENTITY_UNDECLARED" in codes


class TestTransportLoader:
    def test_empty_transport_is_none(self):
        assert ENTRYPOINT._load_transport("") is None

    def test_non_callable_transport_refused(self):
        #: mock-impersonating-real: a dotted path to a NON-callable object
        #: is refused, never silently treated as a transport
        with pytest.raises(ENTRYPOINT.RealTwoWindowBlocked,
                           match="REAL_LLM_TRANSPORT_NOT_CALLABLE"):
            ENTRYPOINT._load_transport(
                "d052.feedback_llm_ued.constants.MODE_NORMAL_FEEDBACK")


class TestEntrypointMain:
    def test_check_only_reports_exactly_ten_blockers(self, capsys):
        code = ENTRYPOINT.main(["--check-only"])
        assert code == 1
        out = capsys.readouterr().out
        report = json.loads(out)
        assert report["entrypoint"] == \
            "scripts/run_e2_real_two_window.py"
        assert report["target"] == "TWO_REAL_WINDOWS_READY_FOR_AUDIT"
        counts: dict = {}
        for b in report["blockers"]:
            counts[b["code"]] = counts.get(b["code"], 0) + 1
        assert counts == EXPECTED_DEFAULT_CODES
        assert len(report["blockers"]) == 10
        #: the five shared-runtime slots are reported EMPTY
        for entry in report["shared_runtime_status"].values():
            assert entry["status"] == C.BLOCKED_WAITING_SHARED_RUNTIME
        #: no capability flag flips, pilot stays off
        assert all(v is False for v in report["real_capability_flags"].values())
        assert report["e2_pilot_authorized"] is False

    def test_check_only_exits_1_and_never_falls_back(self, capsys):
        #: the refusal message names every code — NO_SILENT_FALLBACK
        code = ENTRYPOINT.main(["--check-only"])
        assert code == 1
        err = capsys.readouterr().err
        assert "REAL TWO-WINDOW RUN BLOCKED" in err
        for code_name in sorted(EXPECTED_DEFAULT_CODES):
            assert code_name in err

    def test_default_main_is_equally_blocked(self, capsys):
        #: without --check-only the entrypoint still refuses (never runs)
        code = ENTRYPOINT.main([])
        assert code == 1


class TestPosture:
    def test_real_capability_flags_stay_false(self):
        for name in C.NEVER_TRUE_REAL_CAPABILITY_FLAGS:
            assert getattr(C, name) is False, name
