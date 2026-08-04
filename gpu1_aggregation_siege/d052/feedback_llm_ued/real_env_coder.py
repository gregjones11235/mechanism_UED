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
from d052.feedback_llm_ued.axis_directive import (
    AxisDirective,
    DIRECTIONS,
    EXPERIMENT_ROLES,
    OLD_LEVELS,
)
from d052.feedback_llm_ued.environment_generator import AXIS_LEVELS
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
#: v2 (P0-4, CC3 follow-up audit): the coder must echo every directive's
#: controlled-experiment content (changed axis triple, held axes, predicted
#: signature, treatment/control role) so the strict content binding can be
#: verified verbatim — a v1-format response fails the schema and enters the
#: bounded repair loop.
PROMPT_VERSION = f"{C.ROLE_PROMPT_VERSION}.{ROLE}.real.v2"

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
STRICT CONTENT BINDING (P0-4): every directive_binding MUST echo its
directive's controlled-experiment content VERBATIM — directive_id,
directive_hash, environment_family, changed_axis, old_level, new_level,
direction, experiment_control_role, held_constant_axes and
expected_next_signature. Any divergence (extra, missing, duplicate or
altered echo) is rejected fail-closed.
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
    """One directive's real coded artifact: Python source + contracts +
    the directive's FULL controlled-experiment content echoed verbatim.

    P0-4 (CC3 follow-up audit): the coder must ECHO every directive's
    content — changed axis triple, held axes, predicted signature,
    treatment/control role — so the strict content binding can be checked
    verbatim against the board's AxisDirective. A missing echo field fails
    the schema outright (NO_SILENT_SCHEMA_COERCION); a WRONG echo is
    refused by ``directive_content_binding_blockers``. Vocabulary legality
    is checked here; consistency with the directive is the content
    checker's job (an echo of legal-but-wrong values must be constructible
    so the mismatch can be reported with its exact code).
    """

    directive_id: str = Field(min_length=1)
    directive_hash: str = Field(min_length=1)
    environment_family: str = Field(min_length=1)
    changed_axis: str = Field(min_length=1)
    old_level: str = Field(min_length=1)
    new_level: str = Field(min_length=1)
    direction: str = Field(min_length=1)
    experiment_control_role: str = Field(min_length=1)
    held_constant_axes: Dict[str, str] = Field(default_factory=dict)
    expected_next_signature: Dict[str, float] = Field(default_factory=dict)
    python_source: str = Field(min_length=1)
    reset_contract: str = Field(min_length=1)
    step_contract: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate(self) -> "RealDirectiveBinding":
        if self.environment_family not in C.ENVIRONMENT_FAMILIES:
            raise ValueError(
                f"UNKNOWN_ENVIRONMENT_FAMILY: {self.environment_family!r}")
        if self.changed_axis not in C.MUTATION_AXES:
            raise ValueError(
                f"ILLEGAL_BINDING_AXIS: {self.changed_axis!r}")
        if self.old_level not in OLD_LEVELS:
            raise ValueError(
                f"ILLEGAL_BINDING_OLD_LEVEL: {self.old_level!r}")
        if self.new_level not in AXIS_LEVELS:
            raise ValueError(
                f"ILLEGAL_BINDING_NEW_LEVEL: {self.new_level!r}")
        if self.direction not in DIRECTIONS:
            raise ValueError(
                f"ILLEGAL_BINDING_DIRECTION: {self.direction!r}")
        if self.experiment_control_role not in EXPERIMENT_ROLES:
            raise ValueError(
                f"ILLEGAL_BINDING_EXPERIMENT_ROLE: "
                f"{self.experiment_control_role!r}")
        if not self.expected_next_signature:
            raise ValueError(
                "EMPTY_BINDING_EXPECTED_SIGNATURE: the coder must echo "
                "the directive's predicted signature")
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
# P0-4: strict per-directive content binding (spec vs echoed output)
# ---------------------------------------------------------------------------
def directive_content_binding_blockers(*,
                                       spec: CanonicalTaskSpec,
                                       directives: List[AxisDirective],
                                       parsed: RealEnvCoderOutput
                                       ) -> List[str]:
    """Strict directive-to-artifact content binding: fail-closed reasons
    ([] = bound exactly).

    Every binding must echo its directive's content VERBATIM — id, hash,
    family, window, changed-axis triple, held axes, predicted signature,
    treatment/control role — and the batch hash must bind the exact
    directive batch. Extra / missing bindings are rejected; duplicates are
    refused by the output schema itself. Codes (audit-mandated):

    * DIRECTIVE_BATCH_HASH_MISSING / TASK_SPEC_HASH_MISMATCH
    * DIRECTIVE_BINDING_MISSING / DIRECTIVE_BINDING_EXTRA
    * DIRECTIVE_SOURCE_WINDOW_MISMATCH
    * DIRECTIVE_HASH_MISMATCH / FAMILY_BINDING_MISMATCH
    * CHANGED_AXES_MISMATCH / HELD_AXES_MISMATCH
    * EXPECTED_SIGNATURE_MISMATCH / EXPERIMENT_ROLE_MISMATCH
    """
    blockers: List[str] = []
    recomputed_batch = canonical_sha256([d.model_dump() for d in directives])
    if not parsed.directive_batch_hash:
        blockers.append(
            "DIRECTIVE_BATCH_HASH_MISSING: the parsed EnvCoder output "
            "carries no directive_batch_hash — the batch binding is "
            "mandatory, never defaulted")
    elif parsed.directive_batch_hash != recomputed_batch:
        blockers.append(
            "TASK_SPEC_HASH_MISMATCH: parsed directive_batch_hash="
            f"{parsed.directive_batch_hash!r} but the carried directive "
            f"batch recomputes to {recomputed_batch!r} (spec binds "
            f"{spec.directive_batch_hash!r})")

    directive_by_id = {d.directive_id: d for d in directives}
    binding_by_id = {b.directive_id: b for b in parsed.directive_bindings}
    for did in sorted(set(directive_by_id) - set(binding_by_id)):
        blockers.append(
            f"DIRECTIVE_BINDING_MISSING: directive {did!r} has no binding "
            "in the EnvCoder output")
    for did in sorted(set(binding_by_id) - set(directive_by_id)):
        blockers.append(
            f"DIRECTIVE_BINDING_EXTRA: binding {did!r} does not correspond "
            "to any directive in the spec batch")

    for did in sorted(set(directive_by_id) & set(binding_by_id)):
        directive = directive_by_id[did]
        binding = binding_by_id[did]
        if directive.source_window != spec.window:
            blockers.append(
                f"DIRECTIVE_SOURCE_WINDOW_MISMATCH: directive {did!r} is "
                f"from window {directive.source_window} but the spec "
                f"window is {spec.window}")
        if binding.directive_hash != directive.directive_hash:
            blockers.append(
                f"DIRECTIVE_HASH_MISMATCH: binding {did!r} carries "
                f"directive_hash={binding.directive_hash!r}, the board "
                f"directive recomputes to {directive.directive_hash!r}")
        if binding.environment_family != directive.environment_family:
            blockers.append(
                f"FAMILY_BINDING_MISMATCH: binding {did!r} claims family "
                f"{binding.environment_family!r}, the directive belongs "
                f"to {directive.environment_family!r}")
        echoed_changed = (binding.changed_axis, binding.old_level,
                          binding.new_level, binding.direction)
        directive_changed = (directive.axis, directive.old_level,
                             directive.new_level, directive.direction)
        if echoed_changed != directive_changed:
            blockers.append(
                f"CHANGED_AXES_MISMATCH: binding {did!r} echoes "
                f"{echoed_changed!r}, the directive is {directive_changed!r}")
        if dict(binding.held_constant_axes) != dict(
                directive.held_constant_axes):
            blockers.append(
                f"HELD_AXES_MISMATCH: binding {did!r} holds "
                f"{dict(binding.held_constant_axes)!r}, the directive "
                f"holds {dict(directive.held_constant_axes)!r}")
        if canonical_sha256(dict(binding.expected_next_signature)) != \
                canonical_sha256(dict(directive.expected_next_signature)):
            blockers.append(
                f"EXPECTED_SIGNATURE_MISMATCH: binding {did!r} echoes "
                f"signature {dict(binding.expected_next_signature)!r}, "
                "the directive predicts "
                f"{dict(directive.expected_next_signature)!r}")
        if binding.experiment_control_role != \
                directive.experiment_control_role:
            blockers.append(
                f"EXPERIMENT_ROLE_MISMATCH: binding {did!r} echoes role "
                f"{binding.experiment_control_role!r}, the directive is "
                f"{directive.experiment_control_role!r}")
    return blockers


def assert_directive_content_binding(*,
                                     spec: CanonicalTaskSpec,
                                     directives: List[AxisDirective],
                                     parsed: RealEnvCoderOutput) -> None:
    """Fail closed on ANY strict content-binding violation."""
    blockers = directive_content_binding_blockers(
        spec=spec, directives=directives, parsed=parsed)
    if blockers:
        raise RealEnvCoderBlocked(
            "DIRECTIVE_CONTENT_BINDING_MISMATCH: " + "; ".join(blockers[:8]))


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
            #: P0-4: the strict per-directive content binding — every echo
            #: (hash, family, window, changed/held axes, predicted
            #: signature, role, batch hash) must match the board's
            #: directives VERBATIM. Violations are repair-eligible
            #: blockers; a mismatch is never silently accepted.
            blockers.extend(directive_content_binding_blockers(
                spec=spec, directives=directives, parsed=parsed))
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
    "directive_content_binding_blockers", "assert_directive_content_binding",
    "verify_directive_artifact", "execute_real_env_coder",
]
