"""P0-3 (CC3 follow-up audit): isolation contract for EnvCoder artifacts.

The real EnvCoder emits Python module source per AxisDirective. Executing
LLM-produced code in-process is unacceptable; this module defines the
isolation contract every executable env artifact must pass BEFORE the
four-link verification chain (``real_env_coder.verify_directive_artifact``)
may trust it in a future fully-wired production path:

* STATIC AST scan (in-process, cheap, fail-closed): import allowlist,
  banned attribute chains (``os.system`` / ``os.spawn*`` / ``os.exec*`` /
  ``os.popen`` / ``os.fork`` / ``os.kill`` / external-write calls /
  ``sys.exit``), banned calls (``eval`` / ``exec`` / ``compile`` /
  ``__import__`` / ``open`` / ``input`` / ``breakpoint``). ANY hit rejects
  the artifact WITHOUT spawning a subprocess;
* SANDBOXED EXECUTION (only if the static scan is clean): a fresh
  ``[sys.executable, "-I", driver]`` subprocess in a freshly created temp
  directory with a WHITELISTED environment (no credentials, no PYTHONPATH,
  no user site), an ``sys.meta_path`` import guard installed INSIDE the
  child (the full allowlist import closure is pre-resolved while still
  trusted; afterwards any non-allowlisted root module raises
  ``SANDBOX_FORBIDDEN_IMPORT``), a wall-clock timeout enforced by the
  parent, stdout/stderr/exit-code/signal capture, and guaranteed
  destruction of the temp directory after validation;
* IMMUTABLE REPORT: ``SandboxReport`` (canonical_v2, content-hash bound)
  records the outcome honestly — including what this round does NOT
  enforce (see the two attestation fields below).

Honest scope (audit requirement — Windows caveats stated, not hidden):

* ``resource_limits_status`` is pinned to
  ``CPU_MEMORY_CAPS_NOT_ENFORCED_ON_WINDOWS_THIS_ROUND``: reliable
  per-subprocess CPU/memory caps require OS facilities (job objects /
  resource.setrlimit semantics) that this round does NOT wire; a runaway
  artifact is bounded only by the wall-clock timeout (the parent kills the
  child on ``TimeoutExpired``).
* ``network_isolation_status`` is pinned to ``IMPORT_ALLOWLIST_ONLY``: the
  network boundary is the import allowlist (no ``socket`` / ``http`` /
  ``urllib`` / ``requests`` / ``ssl`` importable) — there is NO OS-level
  network namespace / firewall rule this round.
* The static scan is defense-in-depth and is NOT complete against
  obfuscation (``getattr(os, "system")``, aliasing ``from os import open
  as o``, string-built imports). The hard boundaries are the subprocess
  isolation, the import guard and the credential-free environment;
  OS-level syscall filtering is out of scope on Windows this round.

THIS ROUND: the contract is implemented and tested against TEST_ONLY /
SYNTHETIC / NOT_REAL_EXECUTION sources ONLY. No real (LLM-produced)
untrusted code is executed anywhere in this worktree.
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional

from pydantic import Field, model_validator

from d052.bagr_ued.hashing import text_sha256, verify_content_hash
from d052.schemas.common import CanonicalModel, is_sha256_hex

# ---------------------------------------------------------------------------
# statuses + honest attestation constants
# ---------------------------------------------------------------------------
SANDBOX_PASSED = "PASSED"
SANDBOX_FAILED = "FAILED"
SANDBOX_TIMEOUT = "TIMEOUT"
SANDBOX_STATIC_REJECTED = "STATIC_REJECTED"
SANDBOX_STATUSES = frozenset({SANDBOX_PASSED, SANDBOX_FAILED,
                              SANDBOX_TIMEOUT, SANDBOX_STATIC_REJECTED})

#: honest attestation: what this round does NOT enforce (audit-visible).
RESOURCE_LIMITS_STATUS_WINDOWS_THIS_ROUND = \
    "CPU_MEMORY_CAPS_NOT_ENFORCED_ON_WINDOWS_THIS_ROUND"
NETWORK_ISOLATION_IMPORT_ALLOWLIST_ONLY = "IMPORT_ALLOWLIST_ONLY"

#: stdout marker protocol between the child driver and the parent
MARKER_PREFIX = "SANDBOX_MARKER:"

#: driver exit-code convention (documented; the parent additionally
#: requires the PASSED marker before trusting exit code 0)
DRIVER_EXIT_OK = 0
DRIVER_EXIT_CAUGHT_FAILURE = 3

#: environment whitelist for the sandboxed child — PATH (locate the
#: interpreter), SYSTEMROOT/SystemRoot + TEMP/TMP (Windows runtime needs).
#: NOTHING else: no credentials, no PYTHONPATH, no user overrides.
_ENV_PASSTHROUGH = ("PATH", "SYSTEMROOT", "SystemRoot", "TEMP", "TMP")

#: import allowlist (static scan AND the child's runtime import guard).
#: os / sys are importable, but their dangerous attributes are banned by
#: the static scan (the artifact may compute with paths/env reads only).
ALLOWED_IMPORT_ROOTS = frozenset({
    "math", "random", "typing", "dataclasses", "json", "itertools",
    "collections", "functools", "operator", "enum", "abc", "copy",
    "numbers", "os", "sys",
})

#: banned dotted attribute chains (exact matches)
BANNED_ATTRIBUTE_CHAINS = frozenset({
    # process control / external command execution
    "os.system", "os.popen", "os.popen2", "os.popen3", "os.popen4",
    "os.fork", "os.forkpty", "os.kill", "os.killpg", "os.abort",
    "os._exit", "os.posix_spawn", "os.posix_spawnp",
    "os.register_at_fork", "os.startfile",
    # env mutation / privilege
    "os.putenv", "os.unsetenv", "os.setuid", "os.setgid", "os.seteuid",
    "os.setegid", "os.setgroups",
    # external writes / filesystem mutation
    "os.open", "os.fdopen", "os.remove", "os.unlink", "os.rmdir",
    "os.removedirs", "os.rename", "os.renames", "os.replace",
    "os.truncate", "os.mkdir", "os.makedirs", "os.chmod", "os.chown",
    "os.lchmod", "os.lchown", "os.symlink", "os.link", "os.dup",
    "os.dup2", "os.openpty",
    # interpreter escape
    "sys.exit",
})

#: banned dotted prefixes (family bans: os.exec* / os.spawn*)
BANNED_ATTRIBUTE_PREFIXES = ("os.exec", "os.spawn")

#: banned bare call names (Name-id calls)
BANNED_CALL_NAMES = frozenset({
    "eval", "exec", "compile", "open", "input", "__import__",
    "breakpoint",
})


class SandboxBlocked(RuntimeError):
    """The sandbox cannot even be attempted — fail closed (usage errors,
    spawn failures). Artifact-level failures are REPORTED, not raised."""


def sandbox_environment(environ: Optional[Dict[str, str]] = None
                        ) -> Dict[str, str]:
    """The child's complete environment: the whitelist intersection only.
    Credentials, PYTHONPATH, PYTHONHOME and every other variable are
    dropped unconditionally."""
    source = os.environ if environ is None else environ
    return {key: source[key] for key in _ENV_PASSTHROUGH if key in source}


# ---------------------------------------------------------------------------
# static AST scan (fail closed; no subprocess spawned on rejection)
# ---------------------------------------------------------------------------
def _dotted_name(node: ast.AST) -> Optional[str]:
    """Best-effort dotted name of an Attribute chain (``os.path.join`` ->
    ``"os.path.join"``); None when the base is not a plain Name."""
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def static_scan_source(python_source: str) -> List[str]:
    """Fail-closed static rejection reasons ([] = statically clean).

    Codes: STATIC_SCAN_SYNTAX_ERROR / STATIC_SCAN_FORBIDDEN_IMPORT /
    STATIC_SCAN_FORBIDDEN_ATTRIBUTE / STATIC_SCAN_FORBIDDEN_CALL. This is
    defense-in-depth (see module docstring for the honesty statement about
    obfuscation); a clean scan only earns a SANDBOXED run, never trust."""
    blockers: List[str] = []
    try:
        tree = ast.parse(python_source)
    except SyntaxError as exc:
        blockers.append(
            f"STATIC_SCAN_SYNTAX_ERROR: {exc.msg} (line {exc.lineno})")
        return blockers
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORT_ROOTS:
                    blockers.append(
                        f"STATIC_SCAN_FORBIDDEN_IMPORT: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.split(".")[0] not in ALLOWED_IMPORT_ROOTS:
                blockers.append(
                    f"STATIC_SCAN_FORBIDDEN_IMPORT: from {module or '.'}")
        elif isinstance(node, ast.Attribute):
            dotted = _dotted_name(node)
            if dotted is None:
                continue
            if (dotted in BANNED_ATTRIBUTE_CHAINS
                    or dotted.startswith(BANNED_ATTRIBUTE_PREFIXES)):
                blockers.append(
                    f"STATIC_SCAN_FORBIDDEN_ATTRIBUTE: {dotted}")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in BANNED_CALL_NAMES:
                blockers.append(f"STATIC_SCAN_FORBIDDEN_CALL: {func.id}")
    return blockers


# ---------------------------------------------------------------------------
# immutable sandbox report
# ---------------------------------------------------------------------------
class SandboxReport(CanonicalModel):
    """Audit-grade record of one sandbox validation run.

    Cross-field invariants are enforced fail-closed: a PASSED report MUST
    have exit_code 0, no blockers and a removed temp dir; TIMEOUT and
    STATIC_REJECTED reports carry no exit code; the two attestation fields
    may only carry this round's honest constants (any other value is a
    fabricated capability claim)."""

    artifact_id: str = Field(min_length=1)
    python_source_hash: str = Field(min_length=1)
    status: str = SANDBOX_FAILED
    exit_code: Optional[int] = None
    signal_number: Optional[int] = Field(default=None, ge=1)
    wall_clock_ms: float = Field(default=0.0, ge=0.0)
    stdout_tail: str = ""
    stderr_tail: str = ""
    timeout_seconds: float = Field(gt=0.0)
    resource_limits_status: str = RESOURCE_LIMITS_STATUS_WINDOWS_THIS_ROUND
    network_isolation_status: str = NETWORK_ISOLATION_IMPORT_ALLOWLIST_ONLY
    temp_dir_removed: bool = False
    blockers: List[str] = Field(default_factory=list)
    report_hash: str = ""

    @model_validator(mode="after")
    def _validate_and_hash(self) -> "SandboxReport":
        if self.status not in SANDBOX_STATUSES:
            raise ValueError(
                f"ILLEGAL_SANDBOX_STATUS: {self.status!r} not in "
                f"{sorted(SANDBOX_STATUSES)}")
        if not is_sha256_hex(self.python_source_hash):
            raise ValueError(
                "INVALID_HASH: python_source_hash must be a lowercase "
                f"64-char sha256 hex digest, got {self.python_source_hash!r}")
        if self.resource_limits_status \
                != RESOURCE_LIMITS_STATUS_WINDOWS_THIS_ROUND:
            raise ValueError(
                "SANDBOX_RESOURCE_STATUS_FABRICATED: resource_limits_status "
                f"must be {RESOURCE_LIMITS_STATUS_WINDOWS_THIS_ROUND!r} "
                f"this round, got {self.resource_limits_status!r}")
        if self.network_isolation_status \
                != NETWORK_ISOLATION_IMPORT_ALLOWLIST_ONLY:
            raise ValueError(
                "SANDBOX_NETWORK_STATUS_FABRICATED: "
                "network_isolation_status must be "
                f"{NETWORK_ISOLATION_IMPORT_ALLOWLIST_ONLY!r} this round, "
                f"got {self.network_isolation_status!r}")
        if self.status == SANDBOX_PASSED:
            if self.exit_code != DRIVER_EXIT_OK or self.blockers \
                    or not self.temp_dir_removed:
                raise ValueError(
                    "SANDBOX_PASSED_CONTRACT_VIOLATION: a PASSED report "
                    "requires exit_code=0, no blockers and "
                    f"temp_dir_removed=True; got exit_code={self.exit_code} "
                    f"blockers={self.blockers!r} "
                    f"temp_dir_removed={self.temp_dir_removed}")
        if self.status in (SANDBOX_TIMEOUT, SANDBOX_STATIC_REJECTED) \
                and self.exit_code is not None:
            raise ValueError(
                "SANDBOX_EXIT_CODE_CONFLICT: status "
                f"{self.status!r} cannot carry exit_code="
                f"{self.exit_code!r}")
        if self.status == SANDBOX_FAILED and not self.blockers:
            raise ValueError(
                "SANDBOX_FAILED_WITHOUT_REASON: a FAILED report must "
                "carry at least one blocker explaining the failure")
        computed = verify_content_hash(self.model_dump(),
                                       hash_field="report_hash",
                                       carried=self.report_hash,
                                       kind="SandboxReport")
        object.__setattr__(self, "report_hash", computed)
        return self


# ---------------------------------------------------------------------------
# child driver (trusted code written into the fresh temp dir)
# ---------------------------------------------------------------------------
_DRIVER_TEMPLATE = '''\
# TEST_ONLY sandbox driver generated by d052.feedback_llm_ued.
# env_artifact_sandbox (P0-3). This file is TRUSTED harness code; the
# artifact it executes is NOT. It may use open/compile/exec/sys.exit —
# those bans apply to the artifact source, which is statically scanned
# before any subprocess is spawned.
import importlib
import json
import sys
import traceback

ALLOWED_ROOTS = frozenset(__ALLOWED_ROOTS_REPR__)

# Resolve the FULL import closure of the allowlist while still trusted:
# modules already in sys.modules bypass meta_path afterwards, so the guard
# below only ever sees NEW (artifact-requested) imports.
for _root in sorted(ALLOWED_ROOTS):
    importlib.import_module(_root)


class _ImportGuard:
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root not in ALLOWED_ROOTS:
            raise ImportError(
                "SANDBOX_FORBIDDEN_IMPORT: " + fullname)
        return None


sys.meta_path.insert(0, _ImportGuard())

_STATUS = {"status": "FAILED", "stage": "import"}


def _emit():
    print(__MARKER_PREFIX_REPR__ + json.dumps(_STATUS, sort_keys=True))


_ARTIFACT_PATH = __ARTIFACT_PATH_REPR__
try:
    with open(_ARTIFACT_PATH, "r", encoding="utf-8") as _fh:
        _source = _fh.read()
    _code = compile(_source, "<envcoder_artifact>", "exec")
    _namespace = {}
    exec(_code, _namespace)
except Exception:
    traceback.print_exc()
    _emit()
    sys.exit(3)

_reset = _namespace.get("reset")
_step = _namespace.get("step")
if not callable(_reset):
    _STATUS["stage"] = "missing_reset_callable"
    _emit()
    sys.exit(3)
if not callable(_step):
    _STATUS["stage"] = "missing_step_callable"
    _emit()
    sys.exit(3)

try:
    _state = _reset(0)
except Exception:
    traceback.print_exc()
    _STATUS["stage"] = "reset"
    _emit()
    sys.exit(3)

try:
    _out = _step(_state, 0)
except Exception:
    traceback.print_exc()
    _STATUS["stage"] = "step"
    _emit()
    sys.exit(3)

if not isinstance(_out, (tuple, list)) or len(_out) != 4:
    _STATUS["stage"] = "step_arity"
    _emit()
    sys.exit(3)

_STATUS.update(status="PASSED", stage="complete")
_emit()
'''


def _render_driver(artifact_path: str) -> str:
    return (_DRIVER_TEMPLATE
            .replace("__ALLOWED_ROOTS_REPR__",
                     repr(sorted(ALLOWED_IMPORT_ROOTS)))
            .replace("__MARKER_PREFIX_REPR__", repr(MARKER_PREFIX))
            .replace("__ARTIFACT_PATH_REPR__", repr(artifact_path)))


def _parse_marker(stdout_text: str) -> Optional[dict]:
    for line in stdout_text.splitlines():
        if line.startswith(MARKER_PREFIX):
            try:
                payload = json.loads(line[len(MARKER_PREFIX):])
            except json.JSONDecodeError:
                return None
            return payload if isinstance(payload, dict) else None
    return None


def _tail(data: Optional[bytes], max_bytes: int) -> str:
    if not data:
        return ""
    text = data.decode("utf-8", errors="replace")
    return text[-max_bytes:]


# ---------------------------------------------------------------------------
# the runner
# ---------------------------------------------------------------------------
class EnvArtifactSandboxRunner:
    """Validates one env artifact source under the isolation contract.

    Static rejection never spawns a subprocess (``subprocess_runs`` stays
    untouched); a clean source runs in a fresh ``-I`` interpreter inside a
    fresh temp dir with the whitelisted environment, bounded by the
    wall-clock timeout, and the temp dir is destroyed afterwards. The
    runner holds no credentials and never reads environment secrets — the
    child receives only ``sandbox_environment()``.
    """

    def __init__(self, *, default_timeout_seconds: float = 10.0,
                 max_output_bytes: int = 4096):
        if default_timeout_seconds <= 0:
            raise SandboxBlocked(
                "SANDBOX_TIMEOUT_ILLEGAL: default_timeout_seconds="
                f"{default_timeout_seconds!r}")
        if max_output_bytes <= 0:
            raise SandboxBlocked(
                "SANDBOX_MAX_OUTPUT_ILLEGAL: max_output_bytes="
                f"{max_output_bytes!r}")
        self.default_timeout_seconds = float(default_timeout_seconds)
        self.max_output_bytes = int(max_output_bytes)
        #: count of subprocesses actually spawned (static rejections must
        #: never increment this)
        self.subprocess_runs = 0

    # -- internal helpers --------------------------------------------------
    def _report(self, *, artifact_id: str, source_hash: str, status: str,
                exit_code: Optional[int], signal_number: Optional[int],
                wall_clock_ms: float, stdout_tail: str, stderr_tail: str,
                timeout_seconds: float, temp_dir_removed: bool,
                blockers: List[str]) -> SandboxReport:
        return SandboxReport(
            artifact_id=artifact_id, python_source_hash=source_hash,
            status=status, exit_code=exit_code,
            signal_number=signal_number, wall_clock_ms=wall_clock_ms,
            stdout_tail=stdout_tail, stderr_tail=stderr_tail,
            timeout_seconds=float(timeout_seconds),
            resource_limits_status=RESOURCE_LIMITS_STATUS_WINDOWS_THIS_ROUND,
            network_isolation_status=NETWORK_ISOLATION_IMPORT_ALLOWLIST_ONLY,
            temp_dir_removed=temp_dir_removed, blockers=blockers)

    # -- public contract ---------------------------------------------------
    def run(self, *, artifact_id: str, python_source: str,
            timeout_seconds: Optional[float] = None) -> SandboxReport:
        """Validate one artifact source; ALWAYS returns a SandboxReport for
        artifact-level outcomes (never raises for a bad artifact). Raises
        SandboxBlocked only for usage/spawn errors."""
        if not isinstance(artifact_id, str) or not artifact_id:
            raise SandboxBlocked(
                "SANDBOX_ARTIFACT_ID_MISSING: artifact_id must be a "
                f"non-empty string, got {artifact_id!r}")
        if not isinstance(python_source, str) or not python_source:
            raise SandboxBlocked(
                "SANDBOX_PYTHON_SOURCE_MISSING: python_source must be a "
                f"non-empty string, got {type(python_source).__name__}")
        timeout = (self.default_timeout_seconds if timeout_seconds is None
                   else float(timeout_seconds))
        if timeout <= 0:
            raise SandboxBlocked(
                f"SANDBOX_TIMEOUT_ILLEGAL: timeout_seconds={timeout!r}")

        source_hash = text_sha256(python_source)
        started = time.perf_counter()

        #: link 1 — static fail-closed scan (no subprocess on rejection)
        blockers = static_scan_source(python_source)
        if blockers:
            return self._report(
                artifact_id=artifact_id, source_hash=source_hash,
                status=SANDBOX_STATIC_REJECTED, exit_code=None,
                signal_number=None,
                wall_clock_ms=(time.perf_counter() - started) * 1000.0,
                stdout_tail="", stderr_tail="", timeout_seconds=timeout,
                temp_dir_removed=False, blockers=blockers)

        #: link 2 — sandboxed execution in a fresh child + temp dir
        tempdir = tempfile.mkdtemp(prefix="e2_envcoder_sandbox_")
        temp_dir_removed = False
        exit_code: Optional[int] = None
        signal_number: Optional[int] = None
        stdout_text = stderr_text = ""
        timed_out = False
        try:
            artifact_path = os.path.join(tempdir, "envcoder_artifact.py")
            driver_path = os.path.join(tempdir, "_sandbox_driver.py")
            with open(artifact_path, "w", encoding="utf-8") as handle:
                handle.write(python_source)
            with open(driver_path, "w", encoding="utf-8") as handle:
                handle.write(_render_driver(artifact_path))
            command = [sys.executable, "-I", driver_path]
            self.subprocess_runs += 1
            try:
                proc = subprocess.run(
                    command, cwd=tempdir, env=sandbox_environment(),
                    timeout=timeout, capture_output=True)
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                stdout_text = _tail(exc.stdout, self.max_output_bytes)
                stderr_text = _tail(exc.stderr, self.max_output_bytes)
            except OSError as exc:
                raise SandboxBlocked(
                    f"SANDBOX_SUBPROCESS_SPAWN_FAILED: {exc}") from exc
            if not timed_out:
                exit_code = proc.returncode
                if exit_code is not None and exit_code < 0:
                    signal_number = -exit_code
                    exit_code = None
                stdout_text = _tail(proc.stdout, self.max_output_bytes)
                stderr_text = _tail(proc.stderr, self.max_output_bytes)
        finally:
            shutil.rmtree(tempdir, ignore_errors=True)
            temp_dir_removed = not os.path.isdir(tempdir)
            wall_clock_ms = (time.perf_counter() - started) * 1000.0

        blockers = []
        marker = None if timed_out else _parse_marker(stdout_text)
        if timed_out:
            status = SANDBOX_TIMEOUT
            blockers.append(
                f"SANDBOX_TIMEOUT_AFTER_SECONDS:{timeout}")
        elif (exit_code == DRIVER_EXIT_OK and marker is not None
                and marker.get("status") == SANDBOX_PASSED):
            status = SANDBOX_PASSED
        else:
            #: FAILED — every failure must state its reason (the report
            #: schema refuses a FAILED status with empty blockers)
            status = SANDBOX_FAILED
            if marker is None:
                blockers.append(
                    f"NO_SANDBOX_MARKER: exit_code={exit_code} "
                    f"signal_number={signal_number}")
            else:
                blockers.append(
                    f"ARTIFACT_STAGE:{marker.get('stage', 'unknown')}")
                if exit_code != DRIVER_EXIT_OK:
                    blockers.append(
                        f"EXIT_CODE_CONFLICT: exit_code={exit_code} "
                        f"marker_status={marker.get('status')!r}")
        return self._report(
            artifact_id=artifact_id, source_hash=source_hash, status=status,
            exit_code=exit_code, signal_number=signal_number,
            wall_clock_ms=wall_clock_ms, stdout_tail=stdout_text,
            stderr_tail=stderr_text, timeout_seconds=timeout,
            temp_dir_removed=temp_dir_removed, blockers=blockers)


__all__ = [
    "SANDBOX_PASSED", "SANDBOX_FAILED", "SANDBOX_TIMEOUT",
    "SANDBOX_STATIC_REJECTED", "SANDBOX_STATUSES",
    "RESOURCE_LIMITS_STATUS_WINDOWS_THIS_ROUND",
    "NETWORK_ISOLATION_IMPORT_ALLOWLIST_ONLY", "MARKER_PREFIX",
    "DRIVER_EXIT_OK", "DRIVER_EXIT_CAUGHT_FAILURE",
    "ALLOWED_IMPORT_ROOTS", "BANNED_ATTRIBUTE_CHAINS",
    "BANNED_ATTRIBUTE_PREFIXES", "BANNED_CALL_NAMES", "SandboxBlocked",
    "SandboxReport", "EnvArtifactSandboxRunner", "sandbox_environment",
    "static_scan_source",
]
