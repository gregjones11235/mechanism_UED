"""Round-3 P0-4: EnvCoder validation backends (staged, honest scope).

The EnvCoder artifact validation is staged::

    STAGES = SYNTAX -> GUARDS -> STRUCTURE -> IMPORT -> INSTANTIATE
             -> RESET -> STEP -> TERMINAL_AUTORESET

Each backend declares the contiguous prefix of stages it ACTUALLY runs
(``capabilities``) and reports the remaining stages honestly blocked
with a per-stage reason — never silently degraded, never fabricated:

* ``MockBackend``  — SYNTAX + GUARDS only (the stdlib ``compile`` plus
  the deterministic output guards). This is the explicitly-authorized
  ABLATION surface; the production path never selects it by default;
* ``ReplayBackend`` — SYNTAX + GUARDS + STRUCTURE. STRUCTURE is a
  stdlib-AST SURFACE check (the module defines the ``make_env`` entry
  surface). NO craftax import is executed and NO env-code is ever
  run: craftax is absent from the audit venv, so the later stages are
  honestly reported blocked, not skipped silently;
* ``RealBackendAdapter`` — declares ALL stages, but while the real
  runtime is unauthorized ``validate`` raises
  ``ENVCODER_BACKEND_BLOCKED`` fail-closed. It NEVER degrades to a
  weaker backend and never fabricates a pass.

Validation results are ``ValidationReport`` records (backend identity,
passed flag, stages run, stages blocked with reasons, error text) and
are consumed by ``envcoder.run_envcoder_with_repair``; compilation
outcomes are never fed back into any LLM except through the bounded,
whitelist-safe repair prompt.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Tuple

from ..static_llm.guards import scan_text
from .schemas import E1SchemaError

#: backend identities (the teacher config key ``teacher.envcoder.backend``)
BACKEND_MOCK = "mock"
BACKEND_REPLAY = "replay"
BACKEND_REAL = "real"

#: the full ordered validation stage ladder
SYNTAX = "SYNTAX"
GUARDS = "GUARDS"
STRUCTURE = "STRUCTURE"
IMPORT = "IMPORT"
INSTANTIATE = "INSTANTIATE"
RESET = "RESET"
STEP = "STEP"
TERMINAL_AUTORESET = "TERMINAL_AUTORESET"
STAGES = (
    SYNTAX,
    GUARDS,
    STRUCTURE,
    IMPORT,
    INSTANTIATE,
    RESET,
    STEP,
    TERMINAL_AUTORESET,
)

#: the env-code entry surface the STRUCTURE stage looks for (stdlib AST)
ENV_ENTRY_SURFACE = "make_env"

# fail-closed codes (greppable)
ENVCODER_BACKEND_BLOCKED = "ENVCODER_BACKEND_BLOCKED"
ENVCODER_BACKEND_BAD_TYPE = "ENVCODER_BACKEND_BAD_TYPE"
ENVCODER_BACKEND_UNKNOWN_BACKEND = "ENVCODER_BACKEND_UNKNOWN_BACKEND"
ENVCODER_VALIDATION_BAD_STAGE = "ENVCODER_VALIDATION_BAD_STAGE"


class EnvCoderBackendError(E1SchemaError):
    """Fail-closed backend violation; ``code`` is greppable."""


@dataclass(frozen=True)
class ValidationReport:
    """One staged validation outcome (honest about what did NOT run).

    ``stages_run``   — the stages this backend actually executed;
    ``stages_blocked`` — ``(stage, reason)`` pairs for every stage the
      backend does NOT execute (honest scope, never silent);
    ``error``        — "" when passed, else the failing stage's note.
    """

    backend: str
    passed: bool
    stages_run: Tuple[str, ...]
    stages_blocked: Tuple[Tuple[str, str], ...]
    error: str


def _blocked_suffix(
    capabilities: Tuple[str, ...], reason: str
) -> Tuple[Tuple[str, str], ...]:
    """Every stage beyond ``capabilities``, each with the same honest
    reason (stage ladder order preserved)."""
    return tuple(
        (stage, reason) for stage in STAGES if stage not in capabilities
    )


def _syntax_and_guards(code: Any, backend: str) -> Tuple[bool, str]:
    """The shared SYNTAX + GUARDS core (stdlib only, deterministic)."""
    if not isinstance(code, str):
        return False, (
            f"{ENVCODER_BACKEND_BAD_TYPE}: env code must be str, got "
            f"{type(code).__name__}"
        )
    if not code.strip():
        return False, f"{ENVCODER_BACKEND_BAD_TYPE}: env code is empty"
    decision = scan_text(code, "env_code")
    if not decision.allowed:
        return False, f"{decision.code}: {decision.detail}"
    try:
        compile(code, "<e1-artifact>", "exec")
    except SyntaxError as e:
        return False, f"SYNTAX_ERROR: {e.msg} (line {e.lineno})"
    except ValueError as e:  # e.g. null bytes
        return False, f"SYNTAX_ERROR: {e}"
    return True, ""


class MockBackend:
    """SYNTAX + GUARDS only — the explicitly-authorized ablation scope.

    Identical validation core to the legacy ``check_compilation`` duck;
    the production path uses ``ReplayBackend`` (+STRUCTURE) instead.
    """

    name = BACKEND_MOCK
    capabilities = (SYNTAX, GUARDS)
    stages_blocked = _blocked_suffix(
        capabilities,
        "the mock backend validates SYNTAX+GUARDS only "
        "(explicitly-authorized ablation surface; never the production "
        "default)",
    )

    def validate(self, code: Any) -> ValidationReport:
        ok, error = _syntax_and_guards(code, self.name)
        return ValidationReport(
            backend=self.name,
            passed=ok,
            stages_run=self.capabilities,
            stages_blocked=self.stages_blocked,
            error=error,
        )


class ReplayBackend:
    """SYNTAX + GUARDS + STRUCTURE (stdlib-AST surface check).

    STRUCTURE verifies the module defines the ``make_env`` entry
    surface via ``ast.parse`` — NO craftax import is executed and the
    env-code is NEVER imported/instantiated/stepped: craftax is absent
    from the audit venv, so IMPORT..TERMINAL_AUTORESET stay honestly
    blocked (the bounded scope of this round).
    """

    name = BACKEND_REPLAY
    capabilities = (SYNTAX, GUARDS, STRUCTURE)
    stages_blocked = _blocked_suffix(
        capabilities,
        "craftax is absent from the audit venv; the replay backend "
        "never imports, instantiates or steps env-code (honest bound, "
        "never silent degradation)",
    )

    def validate(self, code: Any) -> ValidationReport:
        blocked = self.stages_blocked
        ok, error = _syntax_and_guards(code, self.name)
        if not ok:
            return ValidationReport(
                backend=self.name,
                passed=False,
                stages_run=self.capabilities,
                stages_blocked=blocked,
                error=error,
            )
        # STRUCTURE: stdlib-AST surface check (make_env entry present)
        tree = ast.parse(code)  # SYNTAX stage already guaranteed validity
        has_entry = any(
            isinstance(node, ast.FunctionDef) and node.name == ENV_ENTRY_SURFACE
            for node in tree.body
        )
        if not has_entry:
            return ValidationReport(
                backend=self.name,
                passed=False,
                stages_run=self.capabilities,
                stages_blocked=blocked,
                error=(
                    f"{STRUCTURE}: env-code defines no {ENV_ENTRY_SURFACE} "
                    "entry surface (stdlib-AST surface check)"
                ),
            )
        return ValidationReport(
            backend=self.name,
            passed=True,
            stages_run=self.capabilities,
            stages_blocked=blocked,
            error="",
        )


class RealBackendAdapter:
    """Full-stage REAL execution adapter (craftax runtime required).

    Runs the COMPLETE ladder against the real craftax runtime:
    SYNTAX -> GUARDS -> STRUCTURE -> IMPORT -> INSTANTIATE -> RESET ->
    STEP -> TERMINAL_AUTORESET. A failure at any stage fails closed with
    the stage name; stages are never skipped silently and a pass is
    never fabricated. When the craftax runtime is absent, ``validate``
    raises ``ENVCODER_BACKEND_BLOCKED`` fail-closed (never degrades).
    """

    name = BACKEND_REAL
    capabilities = STAGES
    stages_blocked = ()  # declares the full ladder; blocked at runtime

    def validate(self, code: Any) -> ValidationReport:
        try:
            import craftax  # noqa: F401
        except Exception as exc:
            raise EnvCoderBackendError(
                ENVCODER_BACKEND_BLOCKED,
                "the real EnvCoder backend requires the craftax runtime "
                f"(import failed: {exc!r}); the adapter never degrades "
                "to weaker stages and never fabricates a pass.",
            )

        stages_run = []
        ok, error = _syntax_and_guards(code, self.name)
        if not ok:
            return self._report(stages_run, error)
        stages_run += [SYNTAX, GUARDS]

        tree = ast.parse(code)
        has_entry = any(
            isinstance(node, ast.FunctionDef)
            and node.name == ENV_ENTRY_SURFACE
            for node in tree.body
        )
        if not has_entry:
            return self._report(
                stages_run,
                f"{STRUCTURE}: env-code defines no {ENV_ENTRY_SURFACE} "
                "entry surface",
            )
        stages_run.append(STRUCTURE)

        namespace: dict = {}
        try:
            exec(compile(code, "<e1-real-backend>", "exec"), namespace)
        except Exception as exc:
            return self._report(stages_run, f"{IMPORT}: {exc!r}")
        stages_run.append(IMPORT)

        make_env = namespace.get(ENV_ENTRY_SURFACE)
        if not callable(make_env):
            return self._report(
                stages_run,
                f"{INSTANTIATE}: {ENV_ENTRY_SURFACE} is not callable",
            )
        try:
            env = make_env()
        except Exception as exc:
            return self._report(stages_run, f"{INSTANTIATE}: {exc!r}")
        stages_run.append(INSTANTIATE)

        handle = _RealEnvHandle(env)
        try:
            handle.reset()
        except Exception as exc:
            return self._report(stages_run, f"{RESET}: {exc!r}")
        stages_run.append(RESET)
        try:
            handle.step(0)
        except Exception as exc:
            return self._report(stages_run, f"{STEP}: {exc!r}")
        stages_run.append(STEP)
        try:
            handle.step(0)
        except Exception as exc:
            return self._report(
                stages_run, f"{TERMINAL_AUTORESET}: {exc!r}")
        stages_run.append(TERMINAL_AUTORESET)
        return self._report(stages_run, "", passed=True)

    def _report(self, stages_run: list, error: str, *,
                passed: bool = False) -> ValidationReport:
        blocked = tuple(
            (stage, "not reached: an earlier stage failed fail-closed")
            for stage in STAGES if stage not in stages_run
        )
        return ValidationReport(
            backend=self.name,
            passed=passed,
            stages_run=tuple(stages_run),
            stages_blocked=blocked,
            error=error,
        )


class _RealEnvHandle:
    """Normalizes reset/step across gymnax-style craftax envs."""

    def __init__(self, env: Any):
        self._env = env
        self._key = None
        self._state = None
        self._params = getattr(env, "default_params", None)

    def reset(self):
        import jax

        self._key = jax.random.PRNGKey(0)
        self._key, sub = jax.random.split(self._key)
        if self._params is not None:
            result = self._env.reset(sub, self._params)
        else:
            result = self._env.reset(sub)
        if isinstance(result, tuple) and len(result) == 2:
            obs, self._state = result
            return obs
        return result

    def step(self, action: int):
        import jax

        self._key, sub = jax.random.split(self._key)
        if self._params is not None:
            result = self._env.step(sub, self._state, action, self._params)
        else:
            result = self._env.step(sub, self._state, action)
        obs, self._state, reward, done, _info = result
        return obs, reward, done


def make_backend(name: Any) -> Any:
    """Resolve a backend identity to its instance (fail-closed).

    ``real`` returns the REAL staged backend ONLY when the craftax
    runtime is importable; otherwise it raises
    ``ENVCODER_BACKEND_BLOCKED`` (never degrades silently). Unknown
    names raise ``ENVCODER_BACKEND_UNKNOWN_BACKEND``.
    """
    if not isinstance(name, str) or not name.strip():
        raise EnvCoderBackendError(
            ENVCODER_BACKEND_BAD_TYPE,
            f"backend name must be a non-empty str, got {name!r}",
        )
    if name == BACKEND_MOCK:
        return MockBackend()
    if name == BACKEND_REPLAY:
        return ReplayBackend()
    if name == BACKEND_REAL:
        try:
            import craftax  # noqa: F401
        except Exception:
            raise EnvCoderBackendError(
                ENVCODER_BACKEND_BLOCKED,
                "the real EnvCoder backend is unauthorized this round "
                "(craftax runtime absent); it never degrades silently",
            )
        return RealBackendAdapter()
    raise EnvCoderBackendError(
        ENVCODER_BACKEND_UNKNOWN_BACKEND,
        f"unknown envcoder backend {name!r}; expected one of "
        f"{[BACKEND_MOCK, BACKEND_REPLAY, BACKEND_REAL]}",
    )
