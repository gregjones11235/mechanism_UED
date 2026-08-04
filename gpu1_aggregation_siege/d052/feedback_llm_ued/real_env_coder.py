"""P0-2: the REAL EnvCoder execution chain (TWO_REAL_WINDOWS_READY_FOR_AUDIT).

Production contract (master directive)::

    AxisDirective batch -> CanonicalTaskSpec (UNIQUE template, ONE call per
    window batch) -> real backend -> RealEnvCoderOutput (Python artifacts)
    -> compile -> import -> reset -> step verification -> BOUNDED repair ->
    fail-closed block on budget exhaustion.

Hard rules encoded here:

* the coder is called under EXACTLY ONE template
  (``C.ENVCODER_UNIQUE_TEMPLATE_ID``) once per window directive batch —
  never per candidate, never under ad-hoc prompt variants (a repair re-call
  appends the failure report to the SAME unique template context);
* every produced Python artifact passes a four-link verification chain
  (syntax compile -> import/instantiate -> ``reset(seed)`` -> ``step(state,
  action)``) mirroring the symbolic ``env_coder_gate`` semantics;
* repair is BOUNDED by ``C.ENVCODER_MAX_REPAIR_ATTEMPTS``; once exhausted
  the execution raises ``RealEnvCoderBlocked`` with
  ``REAL_ENVCODER_REPAIR_BUDGET_EXHAUSTED`` — there is NO fallback to the
  symbolic coder on the production path;
* this module never imports the symbolic code-symbol machinery or the mock
  backend as a data source; the backend must be kind="real".

Honesty: ``REAL_ENVCODER_USED`` stays False until this chain has actually
executed against a real backend; no real LLM transport exists in this
worktree, so locally the chain can only be READY, never executed.
"""
from __future__ import annotations

import json
from typing import Callable, Dict, List, Optional

from pydantic import Field, model_validator

from d052.bagr_ued.hashing import canonical_sha256, text_sha256, \
    verify_content_hash
from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.axis_directive import AxisDirective
from d052.feedback_llm_ued.env_coder import RealEnvCoderBlocked
from d052.feedback_llm_ued.feedback_contracts import (
    CONTEXT_CLOSE,
    CONTEXT_OPEN,
    FeedbackRoleEnvelope,
)
from d052.feedback_llm_ued.real_call_journal import (
    OUTPUT_SCHEMA_FAILED,
    OUTPUT_SCHEMA_PARSED,
    RealCallJournal,
    default_logical_call_id,
)
from d052.feedback_llm_ued.runtime_authorization import (
    RealRuntimeAuthorization,
)
from d052.schemas.common import CanonicalModel

ROLE = C.ROLE_ENV_CODER
PROMPT_VERSION = f"{C.ROLE_PROMPT_VERSION}.{ROLE}.real.v1"

#: per-link verification statuses
STATUS_PASSED = "PASSED"
STATUS_FAILED = "FAILED"
STATUS_NOT_RUN = "NOT_RUN"
VERIFICATION_STATUSES = frozenset({STATUS_PASSED, STATUS_FAILED,
                                   STATUS_NOT_RUN})

#: the unique template's artifact contract: each directive binding must ship
#: one Python module source defining module-level ``reset(seed) -> state``
#: and ``step(state, action) -> (state, reward, terminal, info)`` callables.
RESET_CALLABLE = "reset"
STEP_CALLABLE = "step"
STEP_OUTPUT_ARITY = 4
#: the seed used by the reset smoke check (a smoke check only: real probe
#: seeds come from the probe's banked seed schedule, never from here)
RESET_SMOKE_SEED = 0
RESET_SMOKE_ACTION = 0

REAL_PROMPT_TEMPLATE = f"""\
You are the independent EnvCoder of the simulator-grounded
feedback-adaptive LLM-UED loop, running under the single authorized
template {C.ENVCODER_UNIQUE_TEMPLATE_ID}. You receive the review board's
CanonicalTaskSpec (the window's AxisDirectives, controlled environment
specifications) and NOTHING else — no Student data, no feedback, no probe
results. Emit ONE JSON object of schema RealEnvCoderOutput: for EVERY
directive, one directive_binding whose python_source is a self-contained
Python module realizing exactly the requested axis setting while holding
the declared axes constant. Each module MUST define module-level
callables reset(seed) -> state and step(state, action) ->
(state, reward, terminal, info). Do not invent axes; do not touch action,
reward, or policy knobs.
Prompt version: {{prompt_version}}
{CONTEXT_OPEN}
{{context_json}}
{CONTEXT_CLOSE}
Respond with a single JSON object matching the RealEnvCoderOutput schema.
"""


# ---------------------------------------------------------------------------
# spec + output schemas
# ---------------------------------------------------------------------------
class CanonicalTaskSpec(CanonicalModel):
    """The EnvCoder's entire world for one window: the directive batch.

    ``template_id`` is pinned to the unique template; the directive batch
    hash is recomputed from the carried directive dumps and compared
    verbatim (a substituted batch cannot be coded silently).
    """

    window: int = Field(ge=0)
    plan_id: str = Field(min_length=1)
    template_id: str = C.ENVCODER_UNIQUE_TEMPLATE_ID
    directives: List[Dict[str, object]] = Field(default_factory=list)
    directive_batch_hash: str = ""
    spec_hash: str = ""

    @model_validator(mode="after")
    def _validate_and_hash(self) -> "CanonicalTaskSpec":
        if self.template_id != C.ENVCODER_UNIQUE_TEMPLATE_ID:
            raise ValueError(
                f"ENVCODER_TEMPLATE_NOT_UNIQUE: template_id="
                f"{self.template_id!r} — the real EnvCoder may only be "
                f"called under {C.ENVCODER_UNIQUE_TEMPLATE_ID!r}")
        if not self.directives:
            raise ValueError(
                "EMPTY_REAL_ENVCODER_SPEC: a real EnvCoder call requires at "
                "least one AxisDirective")
        computed_batch = canonical_sha256(self.directives)
        if self.directive_batch_hash and \
                self.directive_batch_hash != computed_batch:
            raise ValueError(
                "CONTENT_HASH_MISMATCH: CanonicalTaskSpec carried "
                f"directive_batch_hash={self.directive_batch_hash!r} but "
                f"the carried directives recompute to {computed_batch!r}")
        object.__setattr__(self, "directive_batch_hash", computed_batch)
        computed = verify_content_hash(self.model_dump(),
                                       hash_field="spec_hash",
                                       carried=self.spec_hash,
                                       kind="CanonicalTaskSpec")
        object.__setattr__(self, "spec_hash", computed)
        return self


class RealDirectiveBinding(CanonicalModel):
    """One directive's real coded artifact: Python source + contracts."""

    directive_id: str = Field(min_length=1)
    directive_hash: str = Field(min_length=1)
    environment_family: str = Field(min_length=1)
    python_source: str = Field(min_length=1)
    reset_contract: str = Field(min_length=1)
    step_contract: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate(self) -> "RealDirectiveBinding":
        if self.environment_family not in C.ENVIRONMENT_FAMILIES:
            raise ValueError(
                f"UNKNOWN_ENVIRONMENT_FAMILY: {self.environment_family!r}")
        return self


class RealEnvCoderOutput(CanonicalModel):
    """The real EnvCoder's response schema (distinct from the symbolic
    ``EnvCoderOutput``: the production path parses its own contract)."""

    window: int = Field(ge=0)
    plan_id: str = Field(min_length=1)
    directive_bindings: List[RealDirectiveBinding] = \
        Field(default_factory=list)
    directive_batch_hash: str = ""
    coder_summary: str = ""

    @model_validator(mode="after")
    def _validate(self) -> "RealEnvCoderOutput":
        if not self.directive_bindings:
            raise ValueError(
                "EMPTY_REAL_ENVCODER_OUTPUT: at least one directive "
                "binding is required")
        ids = [b.directive_id for b in self.directive_bindings]
        if len(set(ids)) != len(ids):
            raise ValueError(
                f"DUPLICATE_REAL_DIRECTIVE_BINDING: {sorted(ids)}")
        return self


class RealDirectiveArtifact(CanonicalModel):
    """Per-directive verification-chain verdict (audit-grade)."""

    directive_id: str = Field(min_length=1)
    directive_hash: str = Field(min_length=1)
    python_source_hash: str = Field(min_length=1)
    compile_status: str = STATUS_NOT_RUN
    import_status: str = STATUS_NOT_RUN
    reset_status: str = STATUS_NOT_RUN
    step_status: str = STATUS_NOT_RUN
    blockers: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> "RealDirectiveArtifact":
        for field_name in ("compile_status", "import_status",
                           "reset_status", "step_status"):
            value = getattr(self, field_name)
            if value not in VERIFICATION_STATUSES:
                raise ValueError(
                    f"ILLEGAL_VERIFICATION_STATUS: {field_name}={value!r}")
        return self

    @property
    def passed(self) -> bool:
        return (self.compile_status == STATUS_PASSED
                and self.import_status == STATUS_PASSED
                and self.reset_status == STATUS_PASSED
                and self.step_status == STATUS_PASSED)


class RealEnvCoderArtifact(CanonicalModel):
    """The complete window-level audit record of one real EnvCoder run."""

    window: int = Field(ge=0)
    plan_id: str = Field(min_length=1)
    template_id: str = C.ENVCODER_UNIQUE_TEMPLATE_ID
    spec_hash: str = Field(min_length=1)
    backend_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    n_calls: int = Field(default=0, ge=1)
    repair_attempts: int = Field(default=0, ge=0)
    logical_call_ids: List[str] = Field(default_factory=list)
    envelope_request_hashes: List[str] = Field(default_factory=list)
    directive_artifacts: List[RealDirectiveArtifact] = \
        Field(default_factory=list)
    overall_status: str = STATUS_FAILED
    blockers: List[str] = Field(default_factory=list)
    artifact_hash: str = ""

    @model_validator(mode="after")
    def _validate_and_hash(self) -> "RealEnvCoderArtifact":
        if self.template_id != C.ENVCODER_UNIQUE_TEMPLATE_ID:
            raise ValueError(
                f"ENVCODER_TEMPLATE_NOT_UNIQUE: {self.template_id!r}")
        if self.overall_status not in (STATUS_PASSED, STATUS_FAILED):
            raise ValueError(
                f"ILLEGAL_OVERALL_STATUS: {self.overall_status!r}")
        computed = verify_content_hash(self.model_dump(),
                                       hash_field="artifact_hash",
                                       carried=self.artifact_hash,
                                       kind="RealEnvCoderArtifact")
        object.__setattr__(self, "artifact_hash", computed)
        return self


# ---------------------------------------------------------------------------
# prompt / parse
# ---------------------------------------------------------------------------
def build_real_env_coder_prompt(spec: CanonicalTaskSpec, *,
                                repair_context: Optional[dict] = None
                                ) -> str:
    """The UNIQUE template prompt; a repair re-call only appends the failure
    report to the context block (the template itself never changes)."""
    context: Dict[str, object] = dict(
        template_id=spec.template_id,
        window=spec.window,
        plan_id=spec.plan_id,
        directives=spec.directives,
        directive_batch_hash=spec.directive_batch_hash,
        spec_hash=spec.spec_hash)
    if repair_context is not None:
        context["repair_context"] = repair_context
    return REAL_PROMPT_TEMPLATE.format(
        prompt_version=PROMPT_VERSION,
        context_json=json.dumps(context, sort_keys=True, ensure_ascii=False,
                                default=str))


def parse_real_env_coder(raw: str) -> RealEnvCoderOutput:
    return RealEnvCoderOutput.model_validate_json(raw)


# ---------------------------------------------------------------------------
# verification chain (compile -> import -> reset -> step)
# ---------------------------------------------------------------------------
def verify_directive_artifact(binding: RealDirectiveBinding
                              ) -> RealDirectiveArtifact:
    """Run the four-link verification chain on one Python artifact.

    Runs inside a FRESH namespace per directive; any exception at a link
    stops the chain (later links stay NOT_RUN) and records the blocker.
    This executes coder-produced code under an explicitly authorized
    production path only — locally it is never reached.
    """
    source_hash = text_sha256(binding.python_source)
    filename = f"<real_envcoder:{binding.directive_id}>"
    blockers: List[str] = []
    compile_status = import_status = STATUS_NOT_RUN
    reset_status = step_status = STATUS_NOT_RUN
    namespace: Dict[str, object] = {}
    state: object = None

    try:
        code = compile(binding.python_source, filename, "exec")
        compile_status = STATUS_PASSED
    except SyntaxError as exc:
        compile_status = STATUS_FAILED
        blockers.append(f"COMPILE_FAILED: {exc.msg} "
                        f"(line {exc.lineno})")
        code = None

    if code is not None:
        try:
            exec(code, namespace)  # noqa: S102 — authorized production path
            if not callable(namespace.get(RESET_CALLABLE)):
                raise ValueError("MISSING_RESET_CALLABLE: module defines no "
                                 "callable reset(seed)")
            if not callable(namespace.get(STEP_CALLABLE)):
                raise ValueError("MISSING_STEP_CALLABLE: module defines no "
                                 "callable step(state, action)")
            import_status = STATUS_PASSED
        except Exception as exc:
            import_status = STATUS_FAILED
            blockers.append(f"IMPORT_FAILED: {type(exc).__name__}: {exc}")

    if import_status == STATUS_PASSED:
        try:
            state = namespace[RESET_CALLABLE](RESET_SMOKE_SEED)  # type: ignore[operator]
            reset_status = STATUS_PASSED
        except Exception as exc:
            reset_status = STATUS_FAILED
            blockers.append(f"RESET_FAILED: {type(exc).__name__}: {exc}")

    if reset_status == STATUS_PASSED:
        try:
            out = namespace[STEP_CALLABLE](state, RESET_SMOKE_ACTION)  # type: ignore[operator]
            if not isinstance(out, (tuple, list)) or \
                    len(out) != STEP_OUTPUT_ARITY:
                raise ValueError(
                    f"STEP_OUTPUT_ARITY_MISMATCH: step(state, action) must "
                    f"return (state, reward, terminal, info); got "
                    f"{type(out).__name__} of arity "
                    f"{len(out) if isinstance(out, (tuple, list)) else 'NA'}")
            step_status = STATUS_PASSED
        except Exception as exc:
            step_status = STATUS_FAILED
            blockers.append(f"STEP_FAILED: {type(exc).__name__}: {exc}")

    return RealDirectiveArtifact(
        directive_id=binding.directive_id,
        directive_hash=binding.directive_hash,
        python_source_hash=source_hash,
        compile_status=compile_status,
        import_status=import_status,
        reset_status=reset_status,
        step_status=step_status,
        blockers=blockers)


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------
def execute_real_env_coder(*, window: int, plan_id: str,
                           directives: List[AxisDirective],
                           backend,
                           authorization: RealRuntimeAuthorization,
                           sequence: int,
                           journal: Optional[RealCallJournal] = None,
                           max_repair_attempts: int
                           = C.ENVCODER_MAX_REPAIR_ATTEMPTS,
                           run_sink: Optional[Callable[..., None]] = None
                           ) -> RealEnvCoderArtifact:
    """The window's real 7th LLM-family call with bounded repair.

    Returns the audit artifact on success; raises ``RealEnvCoderBlocked``
    (REAL_ENVCODER_REPAIR_BUDGET_EXHAUSTED / authorization / backend-kind
    codes) otherwise. NO fallback exists — a blocked real EnvCoder blocks
    the window.

    ``sequence`` is the base sequence number; repair attempt ``i`` consumes
    ``sequence + i``. A journal, when given, MUST be the same object the
    real backend journals into (schema outcomes are refused for calls the
    journal never transported).

    ``run_sink`` (P0-2, CC3 follow-up audit): called EXACTLY ONCE on
    success, BEFORE the artifact is returned, with the complete run
    evidence — ``run_sink(spec=..., parsed=..., artifact=...)`` — so the
    production path can derive the immutable ExecutableEnvironmentArtifacts
    from the SAME objects that passed verification (never re-derived from
    disk or memory copies). A sink failure propagates (the window is
    blocked; there is no artifact without a reachable probe binding).
    """
    if not authorization.real_envcoder:
        raise RealEnvCoderBlocked(
            "REAL_ENVCODER_NOT_AUTHORIZED: runtime grant real_envcoder="
            "false; the real EnvCoder chain may not execute")
    if getattr(backend, "kind", None) != C.BACKEND_KIND_REAL:
        raise RealEnvCoderBlocked(
            "REAL_ENVCODER_BACKEND_NOT_REAL: the real EnvCoder chain "
            f"requires a kind={C.BACKEND_KIND_REAL!r} backend, got "
            f"{getattr(backend, 'kind', None)!r}")
    if max_repair_attempts < 0:
        raise ValueError(
            f"NEGATIVE_MAX_REPAIR_ATTEMPTS: {max_repair_attempts}")

    spec = CanonicalTaskSpec(
        window=window, plan_id=plan_id,
        directives=[d.model_dump() for d in directives])

    logical_call_ids: List[str] = []
    envelope_request_hashes: List[str] = []
    repair_context: Optional[dict] = None
    attempt = 0
    while True:
        prompt = build_real_env_coder_prompt(spec,
                                             repair_context=repair_context)
        raw = backend.complete(ROLE, prompt)
        logical_call_id = default_logical_call_id(ROLE, prompt,
                                                  backend.backend_id)
        logical_call_ids.append(logical_call_id)
        try:
            parsed: Optional[RealEnvCoderOutput] = parse_real_env_coder(raw)
            schema_status = OUTPUT_SCHEMA_PARSED
        except Exception:
            parsed = None
            schema_status = OUTPUT_SCHEMA_FAILED
        if journal is not None:
            journal.record_schema_outcome(logical_call_id,
                                          status=schema_status,
                                          window=window,
                                          sequence=sequence + attempt)
        envelope = FeedbackRoleEnvelope.make(
            role=ROLE, prompt_version=PROMPT_VERSION,
            backend_id=backend.backend_id, model_id=backend.model_id,
            window=window, sequence=sequence + attempt, prompt=prompt,
            raw_response=raw,
            parsed_dump=parsed.model_dump() if parsed is not None else {})
        envelope_request_hashes.append(envelope.request_hash)

        blockers: List[str] = []
        directive_artifacts: List[RealDirectiveArtifact] = []
        if parsed is None:
            blockers.append(
                f"SCHEMA_FAILED: attempt {attempt} response did not parse "
                "against RealEnvCoderOutput")
        else:
            if parsed.window != window:
                raise RealEnvCoderBlocked(
                    f"ENVCODER_WINDOW_MISMATCH: output window="
                    f"{parsed.window} but the spec window is {window}")
            if parsed.plan_id != plan_id:
                raise RealEnvCoderBlocked(
                    f"ENVCODER_PLAN_MISMATCH: output plan_id="
                    f"{parsed.plan_id!r} but the spec plan_id is "
                    f"{plan_id!r}")
            if spec.directive_batch_hash and parsed.directive_batch_hash \
                    and parsed.directive_batch_hash \
                    != spec.directive_batch_hash:
                raise RealEnvCoderBlocked(
                    "CONTENT_HASH_MISMATCH: RealEnvCoderOutput carried "
                    f"directive_batch_hash={parsed.directive_batch_hash!r} "
                    f"but the spec recomputes to "
                    f"{spec.directive_batch_hash!r}")
            bound_ids = {b.directive_id for b in parsed.directive_bindings}
            spec_ids = {d.directive_id for d in directives}
            if bound_ids != spec_ids:
                blockers.append(
                    f"DIRECTIVE_BINDING_COVERAGE_MISMATCH: spec directives "
                    f"{sorted(spec_ids)} vs output bindings "
                    f"{sorted(bound_ids)}")
            for binding in parsed.directive_bindings:
                directive_artifacts.append(verify_directive_artifact(binding))
            blockers.extend(
                f"{a.directive_id}:{b}"
                for a in directive_artifacts for b in a.blockers)

        all_passed = (parsed is not None and not blockers
                      and all(a.passed for a in directive_artifacts))
        if all_passed:
            artifact = RealEnvCoderArtifact(
                window=window, plan_id=plan_id, template_id=spec.template_id,
                spec_hash=spec.spec_hash, backend_id=backend.backend_id,
                model_id=backend.model_id, n_calls=attempt + 1,
                repair_attempts=attempt,
                logical_call_ids=logical_call_ids,
                envelope_request_hashes=envelope_request_hashes,
                directive_artifacts=directive_artifacts,
                overall_status=STATUS_PASSED, blockers=[])
            if run_sink is not None:
                #: P0-2: hand the EXACT verified run (spec + parsed output
                #: + artifact) to the production path so the executable
                #: artifacts are derived from the same objects that passed
                #: the four-link chain
                run_sink(spec=spec, parsed=parsed, artifact=artifact)
            return artifact
        if attempt >= max_repair_attempts:
            raise RealEnvCoderBlocked(
                "REAL_ENVCODER_REPAIR_BUDGET_EXHAUSTED: window="
                f"{window} plan={plan_id} attempts={attempt + 1} "
                f"(repair cap {max_repair_attempts}); last blockers: "
                f"{blockers[:8]} — no fallback exists; the window is blocked")
        repair_context = dict(attempt=attempt, blockers=blockers[:16],
                              instruction=("Repair the directive bindings "
                                           "so every verification link "
                                           "(compile/import/reset/step) "
                                           "passes. Keep the same "
                                           "RealEnvCoderOutput schema."))
        attempt += 1


__all__ = [
    "ROLE", "PROMPT_VERSION", "STATUS_PASSED", "STATUS_FAILED",
    "STATUS_NOT_RUN", "VERIFICATION_STATUSES", "RESET_CALLABLE",
    "STEP_CALLABLE", "STEP_OUTPUT_ARITY", "REAL_PROMPT_TEMPLATE",
    "CanonicalTaskSpec", "RealDirectiveBinding", "RealEnvCoderOutput",
    "RealDirectiveArtifact", "RealEnvCoderArtifact",
    "build_real_env_coder_prompt", "parse_real_env_coder",
    "verify_directive_artifact", "execute_real_env_coder",
]
