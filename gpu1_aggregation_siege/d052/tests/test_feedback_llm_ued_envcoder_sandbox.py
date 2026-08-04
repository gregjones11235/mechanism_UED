"""P0-3 (CC3 follow-up audit, §19 mandated): EnvArtifactSandboxRunner.

Isolation contract for EnvCoder-produced Python artifacts:

* static AST scan rejects forbidden imports / attribute chains / calls
  WITHOUT spawning a subprocess;
* a clean source runs in a fresh ``[sys.executable, "-I", driver]`` child
  inside a fresh temp dir with a whitelisted credential-free environment,
  an in-child import guard, wall-clock timeout, and guaranteed temp-dir
  destruction;
* the immutable SandboxReport records the outcome honestly, including the
  two attestation caveats (no CPU/memory caps on Windows this round;
  network isolation = import allowlist only).

ALL fixture sources in this module are TEST_ONLY / SYNTHETIC /
NOT_REAL_EXECUTION: no real (LLM-produced) untrusted code is executed
anywhere in this worktree this round. Passing tests here flip NO REAL_*
capability flag (verified in TestPosture).
"""
from __future__ import annotations

import os

import pytest

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.env_artifact_sandbox import (
    ALLOWED_IMPORT_ROOTS,
    NETWORK_ISOLATION_IMPORT_ALLOWLIST_ONLY,
    RESOURCE_LIMITS_STATUS_WINDOWS_THIS_ROUND,
    SANDBOX_FAILED,
    SANDBOX_PASSED,
    SANDBOX_STATIC_REJECTED,
    SANDBOX_TIMEOUT,
    EnvArtifactSandboxRunner,
    SandboxBlocked,
    SandboxReport,
    sandbox_environment,
    static_scan_source,
)
from d052.bagr_ued.hashing import text_sha256

#: TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION fixture sources — never real
#: LLM output.
GOOD_SOURCE = '''\
"""TEST_ONLY SYNTHETIC NOT_REAL_EXECUTION fixture."""
import math


def reset(seed):
    return {"seed": int(seed), "x": float(int(seed) % 3)}


def step(state, action):
    next_x = state["x"] + 0.25
    reward = float(math.isfinite(next_x))
    return (dict(state, x=next_x), reward, next_x > 10.0, {})
'''

RESET_RAISE_SOURCE = '''\
def reset(seed):
    raise ValueError("synthetic reset failure")


def step(state, action):
    return (state, 0.0, False, {})
'''

STEP_ARITY_SOURCE = '''\
def reset(seed):
    return 0


def step(state, action):
    return (state, 0.0, False)
'''

MISSING_STEP_SOURCE = '''\
def reset(seed):
    return 0
'''

IMPORT_RAISE_SOURCE = '''\
def _boom():
    raise ValueError("synthetic import failure")


_boom()


def reset(seed):
    return 0


def step(state, action):
    return (state, 0.0, False, {})
'''

SYSTEM_EXIT_SOURCE = '''\
raise SystemExit(0)


def reset(seed):
    return 0


def step(state, action):
    return (state, 0.0, False, {})
'''

INFINITE_RESET_SOURCE = '''\
def reset(seed):
    while True:
        pass


def step(state, action):
    return (state, 0.0, False, {})
'''

ENV_PROBE_SOURCE = '''\
import os

print("SANDBOX_ENV_KEYS:" + ",".join(sorted(os.environ)))


def reset(seed):
    return 0


def step(state, action):
    return (state, 0.0, False, {})
'''


def _runner(**kwargs) -> EnvArtifactSandboxRunner:
    return EnvArtifactSandboxRunner(**kwargs)


# ---------------------------------------------------------------------------
# static scan
# ---------------------------------------------------------------------------
class TestStaticScan:
    def test_clean_source_has_no_blockers(self):
        assert static_scan_source(GOOD_SOURCE) == []

    @pytest.mark.parametrize("source, expected", [
        ("import socket\n", "STATIC_SCAN_FORBIDDEN_IMPORT: socket"),
        ("import subprocess\n", "STATIC_SCAN_FORBIDDEN_IMPORT: subprocess"),
        ("import requests\n", "STATIC_SCAN_FORBIDDEN_IMPORT: requests"),
        ("import ctypes\n", "STATIC_SCAN_FORBIDDEN_IMPORT: ctypes"),
        ("import multiprocessing\n",
         "STATIC_SCAN_FORBIDDEN_IMPORT: multiprocessing"),
        ("import urllib.request\n",
         "STATIC_SCAN_FORBIDDEN_IMPORT: urllib.request"),
        ("from http.client import HTTPConnection\n",
         "STATIC_SCAN_FORBIDDEN_IMPORT: from http.client"),
        # legal stdlib, but NOT on the sandbox allowlist
        ("import decimal\n", "STATIC_SCAN_FORBIDDEN_IMPORT: decimal"),
        ("from socket import socket\n",
         "STATIC_SCAN_FORBIDDEN_IMPORT: from socket"),
        # dynamic-download / interpreter-escape machinery
        ("import ssl\n", "STATIC_SCAN_FORBIDDEN_IMPORT: ssl"),
        ("import importlib\n",
         "STATIC_SCAN_FORBIDDEN_IMPORT: importlib"),
        ("x = __import__('os')\n",
         "STATIC_SCAN_FORBIDDEN_CALL: __import__"),
        # dangerous os attribute chains / families
        ("import os\nos.system('dir')\n",
         "STATIC_SCAN_FORBIDDEN_ATTRIBUTE: os.system"),
        ("import os\nos.spawnl(os.P_WAIT, 'x', 'x')\n", "os.spawnl"),
        ("import os\nos.execl('x', 'x')\n", "os.execl"),
        ("import os\nos.popen('dir')\n", "os.popen"),
        ("import os\nos.kill(1, 9)\n", "os.kill"),
        ("import os\nos._exit(0)\n", "os._exit"),
        ("import os\nos.startfile('x.txt')\n", "os.startfile"),
        # external writes / filesystem mutation
        ("import os\nos.remove('f')\n", "os.remove"),
        ("import os\nos.unlink('f')\n", "os.unlink"),
        ("import os\nos.rmdir('d')\n", "os.rmdir"),
        ("import os\nos.rename('a', 'b')\n", "os.rename"),
        ("import os\nos.truncate('f', 0)\n", "os.truncate"),
        ("import os\nos.makedirs('d')\n", "os.makedirs"),
        ("import os\nos.chmod('f', 0)\n", "os.chmod"),
        ("import os\nos.open('f', 0)\n", "os.open"),
        ("import os\nos.putenv('A', 'B')\n", "os.putenv"),
        # interpreter escape
        ("import sys\nsys.exit(1)\n",
         "STATIC_SCAN_FORBIDDEN_ATTRIBUTE: sys.exit"),
        # banned bare calls
        ("y = eval('1')\n", "STATIC_SCAN_FORBIDDEN_CALL: eval"),
        ("exec('x = 1')\n", "STATIC_SCAN_FORBIDDEN_CALL: exec"),
        ("compile('x', 'f', 'exec')\n",
         "STATIC_SCAN_FORBIDDEN_CALL: compile"),
        ("f = open('x', 'w')\n", "STATIC_SCAN_FORBIDDEN_CALL: open"),
        ("s = input()\n", "STATIC_SCAN_FORBIDDEN_CALL: input"),
    ])
    def test_forbidden_pattern_rejected(self, source, expected):
        blockers = static_scan_source(source)
        assert blockers, f"expected rejection for {source!r}"
        assert any(expected in b for b in blockers), blockers

    def test_syntax_error_is_statically_rejected(self):
        blockers = static_scan_source("def reset(:\n")
        assert any(b.startswith("STATIC_SCAN_SYNTAX_ERROR")
                   for b in blockers), blockers

    def test_allowlisted_roots_are_consistent(self):
        assert ALLOWED_IMPORT_ROOTS == frozenset({
            "math", "random", "typing", "dataclasses", "json",
            "itertools", "collections", "functools", "operator", "enum",
            "abc", "copy", "numbers", "os", "sys"})


# ---------------------------------------------------------------------------
# runner: sandboxed execution contract
# ---------------------------------------------------------------------------
class TestSandboxExecution:
    def test_clean_artifact_passes(self):
        runner = _runner()
        report = runner.run(artifact_id="e2_test_artifact_clean",
                            python_source=GOOD_SOURCE)
        assert report.status == SANDBOX_PASSED
        assert report.exit_code == 0
        assert report.signal_number is None
        assert report.blockers == []
        assert report.temp_dir_removed is True
        assert runner.subprocess_runs == 1
        assert "SANDBOX_MARKER:" in report.stdout_tail
        assert '"status": "PASSED"' in report.stdout_tail
        assert report.python_source_hash == text_sha256(GOOD_SOURCE)
        assert report.wall_clock_ms > 0.0

    def test_static_rejection_never_spawns_subprocess(self):
        runner = _runner()
        report = runner.run(artifact_id="e2_test_artifact_socket",
                            python_source="import socket\n")
        assert report.status == SANDBOX_STATIC_REJECTED
        assert report.exit_code is None
        assert report.temp_dir_removed is False
        assert runner.subprocess_runs == 0
        assert any(b.startswith("STATIC_SCAN_FORBIDDEN_IMPORT")
                   for b in report.blockers)

    def test_reset_failure_captured(self):
        runner = _runner()
        report = runner.run(artifact_id="e2_test_artifact_reset_raise",
                            python_source=RESET_RAISE_SOURCE)
        assert report.status == SANDBOX_FAILED
        assert report.exit_code == 3
        assert "ARTIFACT_STAGE:reset" in report.blockers
        assert "ValueError" in report.stderr_tail
        assert report.temp_dir_removed is True

    def test_step_arity_mismatch_captured(self):
        runner = _runner()
        report = runner.run(artifact_id="e2_test_artifact_step_arity",
                            python_source=STEP_ARITY_SOURCE)
        assert report.status == SANDBOX_FAILED
        assert report.exit_code == 3
        assert "ARTIFACT_STAGE:step_arity" in report.blockers

    def test_missing_step_callable_captured(self):
        runner = _runner()
        report = runner.run(artifact_id="e2_test_artifact_missing_step",
                            python_source=MISSING_STEP_SOURCE)
        assert report.status == SANDBOX_FAILED
        assert report.exit_code == 3
        assert "ARTIFACT_STAGE:missing_step_callable" in report.blockers

    def test_import_time_failure_captured(self):
        runner = _runner()
        report = runner.run(artifact_id="e2_test_artifact_import_raise",
                            python_source=IMPORT_RAISE_SOURCE)
        assert report.status == SANDBOX_FAILED
        assert report.exit_code == 3
        assert "ARTIFACT_STAGE:import" in report.blockers
        assert "synthetic import failure" in report.stderr_tail

    def test_system_exit_without_marker_is_failed(self):
        #: exit code 0 alone is NOT trust — the PASSED marker is mandatory
        runner = _runner()
        report = runner.run(artifact_id="e2_test_artifact_system_exit",
                            python_source=SYSTEM_EXIT_SOURCE)
        assert report.status == SANDBOX_FAILED
        assert report.exit_code == 0
        assert any(b.startswith("NO_SANDBOX_MARKER")
                   for b in report.blockers)

    def test_timeout_enforced(self):
        runner = _runner()
        report = runner.run(artifact_id="e2_test_artifact_infinite",
                            python_source=INFINITE_RESET_SOURCE,
                            timeout_seconds=1.0)
        assert report.status == SANDBOX_TIMEOUT
        assert report.exit_code is None
        assert report.timeout_seconds == 1.0
        assert report.wall_clock_ms >= 900.0
        assert any(b.startswith("SANDBOX_TIMEOUT_AFTER_SECONDS")
                   for b in report.blockers)
        assert report.temp_dir_removed is True
        assert runner.subprocess_runs == 1

    def test_subprocess_counter_counts_only_spawned_runs(self):
        runner = _runner()
        runner.run(artifact_id="a1", python_source="import socket\n")
        runner.run(artifact_id="a2", python_source="import ctypes\n")
        assert runner.subprocess_runs == 0
        runner.run(artifact_id="a3", python_source=GOOD_SOURCE)
        assert runner.subprocess_runs == 1


# ---------------------------------------------------------------------------
# environment / credential isolation
# ---------------------------------------------------------------------------
class TestEnvironmentIsolation:
    def test_sandbox_environment_drops_secrets(self, monkeypatch):
        monkeypatch.setitem(os.environ, "E2_FAKE_API_KEY", "s3cret-token")
        monkeypatch.setitem(os.environ, "ANTHROPIC_API_KEY", "sk-fake")
        env = sandbox_environment()
        assert "E2_FAKE_API_KEY" not in env
        assert "ANTHROPIC_API_KEY" not in env
        assert "PYTHONPATH" not in env
        assert set(env) <= {"PATH", "SYSTEMROOT", "SystemRoot",
                            "TEMP", "TMP"}

    def test_child_never_sees_parent_credentials(self, monkeypatch):
        monkeypatch.setitem(os.environ, "E2_FAKE_API_KEY", "s3cret-token")
        runner = _runner()
        report = runner.run(artifact_id="e2_test_artifact_env_probe",
                            python_source=ENV_PROBE_SOURCE)
        assert report.status == SANDBOX_PASSED
        assert "s3cret-token" not in report.stdout_tail
        assert "E2_FAKE_API_KEY" not in report.stdout_tail
        #: the child sees EXACTLY the whitelist intersection (nothing more)
        marker_lines = [line for line in report.stdout_tail.splitlines()
                        if line.startswith("SANDBOX_ENV_KEYS:")]
        assert len(marker_lines) == 1
        seen = set(marker_lines[0].split(":", 1)[1].split(",")) - {""}
        assert seen <= {"PATH", "SYSTEMROOT", "SystemRoot", "TEMP", "TMP"}
        expected = {key for key in ("PATH", "SYSTEMROOT", "SystemRoot",
                                    "TEMP", "TMP") if key in os.environ}
        #: Windows environment blocks are case-INSENSITIVE: the pair
        #: SYSTEMROOT/SystemRoot may collapse to one entry in the child,
        #: so compare case-insensitively
        assert {k.lower() for k in seen} == {k.lower() for k in expected}


# ---------------------------------------------------------------------------
# report integrity (hash tamper / fabrication ladder)
# ---------------------------------------------------------------------------
class TestReportIntegrity:
    def _passed_report(self) -> SandboxReport:
        return _runner().run(artifact_id="e2_test_artifact_clean",
                             python_source=GOOD_SOURCE)

    def test_report_round_trip_is_stable(self):
        report = self._passed_report()
        clone = SandboxReport(**report.model_dump())
        assert clone.report_hash == report.report_hash
        assert clone.model_dump() == report.model_dump()

    @pytest.mark.parametrize("field, value", [
        ("stdout_tail", "tampered output"),
        ("status", SANDBOX_FAILED),
        ("python_source_hash", "0" * 64),
        ("exit_code", 3),
        ("temp_dir_removed", False),
    ])
    def test_tamper_breaks_content_hash(self, field, value):
        report = self._passed_report()
        dump = report.model_dump()
        dump[field] = value
        if field in ("status", "exit_code", "temp_dir_removed"):
            #: cross-field validators may refuse the tamper outright —
            #: equally fail-closed; otherwise the hash must mismatch
            try:
                clone = SandboxReport(**dump)
            except Exception as exc:
                assert "SANDBOX" in str(exc) or "CONTENT_HASH" in str(exc)
                return
            pytest.fail(f"tampered field {field!r} accepted: {clone}")
        else:
            with pytest.raises(Exception) as exc_info:
                SandboxReport(**dump)
            assert "CONTENT_HASH_MISMATCH" in str(exc_info.value)

    def test_resource_status_fabrication_refused(self):
        dump = self._passed_report().model_dump()
        dump["resource_limits_status"] = "FULL_CPU_MEMORY_CAPS_ENFORCED"
        with pytest.raises(Exception) as exc_info:
            SandboxReport(**dump)
        assert "SANDBOX_RESOURCE_STATUS_FABRICATED" in str(exc_info.value)

    def test_network_status_fabrication_refused(self):
        dump = self._passed_report().model_dump()
        dump["network_isolation_status"] = "FULL_NETWORK_NAMESPACE"
        with pytest.raises(Exception) as exc_info:
            SandboxReport(**dump)
        assert "SANDBOX_NETWORK_STATUS_FABRICATED" in str(exc_info.value)

    def test_illegal_status_refused(self):
        dump = self._passed_report().model_dump()
        dump["status"] = "MOSTLY_PASSED"
        with pytest.raises(Exception) as exc_info:
            SandboxReport(**dump)
        assert "ILLEGAL_SANDBOX_STATUS" in str(exc_info.value)

    def test_passed_with_blockers_refused(self):
        dump = self._passed_report().model_dump()
        dump["blockers"] = ["INJECTED_BLOCKER"]
        with pytest.raises(Exception) as exc_info:
            SandboxReport(**dump)
        assert "SANDBOX_PASSED_CONTRACT_VIOLATION" in str(exc_info.value)

    def test_failed_without_reason_refused(self):
        dump = self._passed_report().model_dump()
        dump.update(status=SANDBOX_FAILED, blockers=[])
        with pytest.raises(Exception) as exc_info:
            SandboxReport(**dump)
        assert "SANDBOX_FAILED_WITHOUT_REASON" in str(exc_info.value)

    def test_timeout_with_exit_code_refused(self):
        dump = self._passed_report().model_dump()
        dump.update(status=SANDBOX_TIMEOUT, exit_code=0, blockers=[
            "SANDBOX_TIMEOUT_AFTER_SECONDS:1.0"])
        with pytest.raises(Exception) as exc_info:
            SandboxReport(**dump)
        assert "SANDBOX_EXIT_CODE_CONFLICT" in str(exc_info.value)

    def test_invalid_source_hash_refused(self):
        dump = self._passed_report().model_dump()
        dump["python_source_hash"] = "not-a-hash"
        with pytest.raises(Exception) as exc_info:
            SandboxReport(**dump)
        assert "INVALID_HASH" in str(exc_info.value)

    def test_honest_attestation_fields_pinned(self):
        report = self._passed_report()
        assert report.resource_limits_status == \
            RESOURCE_LIMITS_STATUS_WINDOWS_THIS_ROUND
        assert report.network_isolation_status == \
            NETWORK_ISOLATION_IMPORT_ALLOWLIST_ONLY


# ---------------------------------------------------------------------------
# usage errors
# ---------------------------------------------------------------------------
class TestUsageErrors:
    def test_empty_artifact_id_blocked(self):
        with pytest.raises(SandboxBlocked) as exc_info:
            _runner().run(artifact_id="", python_source=GOOD_SOURCE)
        assert "SANDBOX_ARTIFACT_ID_MISSING" in str(exc_info.value)

    def test_empty_source_blocked(self):
        with pytest.raises(SandboxBlocked) as exc_info:
            _runner().run(artifact_id="a", python_source="")
        assert "SANDBOX_PYTHON_SOURCE_MISSING" in str(exc_info.value)

    def test_illegal_timeout_blocked(self):
        with pytest.raises(SandboxBlocked) as exc_info:
            _runner().run(artifact_id="a", python_source=GOOD_SOURCE,
                          timeout_seconds=0.0)
        assert "SANDBOX_TIMEOUT_ILLEGAL" in str(exc_info.value)

    def test_constructor_validation(self):
        with pytest.raises(SandboxBlocked):
            EnvArtifactSandboxRunner(default_timeout_seconds=0.0)
        with pytest.raises(SandboxBlocked):
            EnvArtifactSandboxRunner(max_output_bytes=0)


# ---------------------------------------------------------------------------
# posture: nothing here flips a capability flag
# ---------------------------------------------------------------------------
class TestPosture:
    def test_never_true_flags_stay_false(self):
        for name in C.NEVER_TRUE_REAL_CAPABILITY_FLAGS:
            assert getattr(C, name) is False, name
        assert C.E2_PILOT_AUTHORIZED is False
