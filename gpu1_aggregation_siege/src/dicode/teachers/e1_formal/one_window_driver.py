"""CC2 follow-up P0-1: the one-window driver — REAL OBJECT flow.

The former entrypoint passed ``teacher.evolve()`` (nonexistent),
``stage_real_probe(())`` (empty candidate set), ``gen_manager=None``
and ``rl_train_state=None`` — i.e. stage boundaries carried summaries
or nothing at all. This module replaces that with an explicit object
flow: EVERY stage consumes the real object produced by the previous
stage and returns the real object the next stage consumes. Plain
``None`` placeholders, string contract names and caller-shaped
summary dicts standing in for stage objects fail closed at every
boundary.

Stage chain (this module orchestrates the SAME primitives
``evolve_tasks`` uses, against ONE teacher instance — P0-10 same
GenManager)::

    execute_real_review_window(teacher, runtime)
        -> E1WindowResult (ReviewWindow + evidence + gate signals)
    execute_real_envcoder_and_compile(teacher, window_result, runtime)
        -> E1CandidateMaterials (CompileResult + per-template
           EnvCoderArtifacts, hash-bound to the window result)

Later CC2 follow-up commits extend this module with the executable
candidate pool, the signed probe pool, the signed criterion signals,
the attested selection, the certified 12+4 batch, the exactly-one
optimizer update attestation and the full-state round-trip
attestation — each stage returning the object the next stage consumes.

``E1OneWindowArtifacts`` is the frozen record of the WHOLE window;
``validate_one_window_artifacts`` refuses any field that is missing,
None or a mere placeholder. Every field is REQUIRED.
"""
from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields
from typing import Any, Tuple

from .board import ReviewWindow, WINDOW_STATUS_COMPLETE
from .canonical import canonical_sha256
from .controller import CycleOutcome, run_review_cycle
from .envcoder import EnvCoderArtifact, RepairRecord, run_envcoder_with_repair
from .evidence import EvidenceSnapshot, build_evidence_snapshot
from .gate_signals import GateSignalReport, compute_gate_signals
from .invocation_gate import build_gate_state
from .runtime_bundle import (
    E1RuntimeBundle,
    RUNTIME_CAPABILITY_CONTRACTS,
    RuntimeBundleError,
)
from .schemas import E1SchemaError
from .task_specs import CompileResult, compile_task_specs

#: the dynamic slot count (kept local to avoid a layout import cycle)
_NUM_DYNAMIC_SLOTS = 12

# fail-closed driver codes (greppable)
E1_DRIVER_BAD_TYPE = "E1_DRIVER_BAD_TYPE"
E1_DRIVER_MISSING_OBJECT = "E1_DRIVER_MISSING_OBJECT"
E1_DRIVER_SUMMARY_REJECTED = "E1_DRIVER_SUMMARY_DICT_REJECTED"
E1_DRIVER_RUNTIME_UNBOUND = "E1_DRIVER_RUNTIME_UNBOUND"
E1_DRIVER_NO_EVIDENCE = "E1_DRIVER_NO_ADMISSIBLE_EVIDENCE"
E1_DRIVER_GATE_NOT_TRIGGERED = "E1_DRIVER_GATE_NOT_TRIGGERED"
E1_DRIVER_WINDOW_VOID = "E1_DRIVER_WINDOW_VOID"
E1_DRIVER_INSUFFICIENT_ARTIFACTS = "INSUFFICIENT_DYNAMIC_ARTIFACTS"
E1_DRIVER_ARTIFACTS_INCOMPLETE = "E1_ONE_WINDOW_ARTIFACTS_INCOMPLETE"


class DriverError(E1SchemaError):
    """Fail-closed driver violation; ``code`` is greppable."""


def require_real_object(value: Any, name: str, ctx: str) -> Any:
    """Stage boundaries carry REAL objects: never None, never a bare
    string placeholder (a contract name is not an object)."""
    if value is None:
        raise DriverError(
            E1_DRIVER_MISSING_OBJECT,
            f"{ctx}: stage input {name!r} is None — every stage must "
            "receive the real object the previous stage produced",
        )
    if isinstance(value, str):
        raise DriverError(
            E1_DRIVER_SUMMARY_REJECTED,
            f"{ctx}: stage input {name!r} is a bare string ({value!r}); "
            "string placeholders never stand in for runtime objects",
        )
    return value


def _require_gen_manager(teacher: Any, ctx: str) -> Any:
    """The SAME E1FormalGenManager instance flows through the whole
    window (P0-10); importing here avoids a module cycle at import
    time."""
    from .gen_manager import E1FormalGenManager

    require_real_object(teacher, "teacher", ctx)
    if not isinstance(teacher, E1FormalGenManager):
        raise DriverError(
            E1_DRIVER_BAD_TYPE,
            f"{ctx}: teacher must be the window's E1FormalGenManager "
            f"instance, got {type(teacher).__name__}",
        )
    return teacher


def validate_runtime_surface(runtime: Any, ctx: str) -> E1RuntimeBundle:
    """The runtime carrier must be a real bundle with ALL capability
    objects bound (string contract names are refused)."""
    require_real_object(runtime, "runtime", ctx)
    if not isinstance(runtime, E1RuntimeBundle):
        raise DriverError(
            E1_DRIVER_BAD_TYPE,
            f"{ctx}: runtime must be an E1RuntimeBundle, got "
            f"{type(runtime).__name__}",
        )
    for contract in RUNTIME_CAPABILITY_CONTRACTS:
        try:
            obj = runtime.capability(contract)
        except RuntimeBundleError as e:
            raise DriverError(
                E1_DRIVER_RUNTIME_UNBOUND,
                f"{ctx}: {e}",
            ) from e
        require_real_object(obj, f"runtime.{contract}", ctx)
    return runtime


@dataclass(frozen=True)
class E1WindowResult:
    """The REAL review-window stage output (object, not summary)."""

    window: ReviewWindow
    evidence: EvidenceSnapshot
    gate_signals: GateSignalReport
    cycle: CycleOutcome
    window_result_hash: str


@dataclass(frozen=True)
class E1CandidateMaterials:
    """The REAL compile+EnvCoder stage output, hash-bound to the
    window result that produced it."""

    window_result_hash: str
    compile_result: CompileResult
    #: (template_hash, EnvCoderArtifact, repair records) per template
    template_artifacts: Tuple[
        Tuple[str, EnvCoderArtifact, Tuple[RepairRecord, ...]], ...
    ]
    materials_hash: str


@dataclass(frozen=True)
class E1OneWindowArtifacts:
    """The frozen record of ONE complete window (ALL fields required).

    Fields whose object types land in later CC2 follow-up commits are
    carried as opaque objects here and mechanically re-validated by
    the commit that introduces each type; ``None`` is refused for
    every field. ``run_id`` and ``source_commit`` bind the record to
    one execution attempt and one git SHA.
    """

    runtime_bundle_hash: str
    student_identity: Any
    reference_identity: Any
    student_adapter: Any
    reference_adapter: Any
    student_checkpoint_identity: str
    reference_checkpoint_identity: str
    gen_manager: Any  # the ONE E1FormalGenManager of this window
    review_window: Any  # E1WindowResult
    candidate_materials: Any  # E1CandidateMaterials
    executable_candidate_pool: Tuple[Any, ...]
    probe_result_pool: Tuple[Any, ...]
    criterion_signals_pool: Tuple[Any, ...]
    selection_outcome: Any
    verified_batch: Any
    update_attestation: Any
    roundtrip_attestation: Any
    run_id: str
    source_commit: str


def validate_one_window_artifacts(
    artifacts: Any, ctx: str = "e1_driver.artifacts"
) -> E1OneWindowArtifacts:
    """Refuse any missing/None/placeholder field in the window record.

    Type-level checks available now are enforced here; every pool must
    be a tuple (possibly empty only where the owning commit allows it
    — the closed loop requires them all non-empty).
    """
    if not isinstance(artifacts, E1OneWindowArtifacts):
        raise DriverError(
            E1_DRIVER_BAD_TYPE,
            f"{ctx}: expected E1OneWindowArtifacts, got "
            f"{type(artifacts).__name__}",
        )
    for field in dataclass_fields(artifacts):
        value = getattr(artifacts, field.name)
        if value is None:
            raise DriverError(
                E1_DRIVER_ARTIFACTS_INCOMPLETE,
                f"{ctx}: field {field.name!r} is None — the window "
                "record requires every stage's real object",
            )
        if field.name.endswith("_pool"):
            if not isinstance(value, tuple):
                raise DriverError(
                    E1_DRIVER_BAD_TYPE,
                    f"{ctx}: field {field.name!r} must be a tuple, got "
                    f"{type(value).__name__}",
                )
            for i, item in enumerate(value):
                require_real_object(item, f"{field.name}[{i}]", ctx)
    for name in ("runtime_bundle_hash", "run_id", "source_commit",
                 "student_checkpoint_identity",
                 "reference_checkpoint_identity"):
        value = getattr(artifacts, name)
        if not isinstance(value, str) or not value.strip():
            raise DriverError(
                E1_DRIVER_ARTIFACTS_INCOMPLETE,
                f"{ctx}: field {name!r} must be a non-empty str, got "
                f"{value!r}",
            )
    if not isinstance(artifacts.review_window, E1WindowResult):
        raise DriverError(
            E1_DRIVER_BAD_TYPE,
            f"{ctx}: review_window must be the E1WindowResult object, "
            f"got {type(artifacts.review_window).__name__}",
        )
    if not isinstance(artifacts.candidate_materials, E1CandidateMaterials):
        raise DriverError(
            E1_DRIVER_BAD_TYPE,
            f"{ctx}: candidate_materials must be the "
            f"E1CandidateMaterials object, got "
            f"{type(artifacts.candidate_materials).__name__}",
        )
    _require_gen_manager(artifacts.gen_manager, ctx)
    return artifacts


# ---------------------------------------------------------------------------
# stage 1: the real review window (gate signals -> board -> ReviewWindow)
# ---------------------------------------------------------------------------
def execute_real_review_window(teacher: Any, runtime: Any) -> E1WindowResult:
    """Open ONE real review window through the teacher's own client.

    Returns the REAL ``E1WindowResult`` (window + evidence + gate
    signals + cycle outcome), hash-bound for downstream stages. Any
    REUSE/void outcome fails closed — the one-window pipeline never
    continues on a window that did not COMPLETE.
    """
    ctx = "e1_driver.review_window"
    validate_runtime_surface(runtime, ctx)
    _require_gen_manager(teacher, ctx)

    raw_items = teacher.collect_evidence_raw_items()
    if len(raw_items) == 0:
        raise DriverError(
            E1_DRIVER_NO_EVIDENCE,
            f"{ctx}: no admissible evidence items; the review window "
            "cannot open (the gate never fires on nothing)",
        )
    evidence = build_evidence_snapshot(raw_items, ctx)
    signals = compute_gate_signals(
        session_idx=teacher.session_idx,
        cycles_run=teacher.cycles_run,
        evidence=evidence,
        raw_items=raw_items,
        prev_window=teacher.last_review_window,
        thresholds=teacher.invocation_thresholds,
        threshold_version=teacher.invocation_threshold_version,
        consecutive_reuses=teacher.consecutive_reuses,
    )
    gate_state = build_gate_state(
        {
            "session_idx": teacher.session_idx,
            **{name: value for name, value in signals.signals},
            "signals_binding_hash": signals.binding_hash,
        },
        ctx,
    )
    window_id = f"e1-w{teacher.session_idx:06d}"
    outcome = run_review_cycle(
        teacher.llm_client,
        window_id=window_id,
        gate_state=gate_state,
        evidence=evidence,
        ledger=teacher.ledger,
    )
    teacher.record_driver_cycle(outcome)
    if outcome.window is None:
        raise DriverError(
            E1_DRIVER_GATE_NOT_TRIGGERED,
            f"{ctx}: the invocation gate did not trigger a window "
            f"(decision {outcome.decision.code}); the pipeline stops "
            "honestly instead of fabricating a window",
        )
    if outcome.reuse or outcome.window.status != WINDOW_STATUS_COMPLETE:
        raise DriverError(
            E1_DRIVER_WINDOW_VOID,
            f"{ctx}: window {outcome.window.window_id} is VOID "
            f"({outcome.window.void_code}); a void window can never "
            "feed the pipeline (and is never relabelled COMPLETE)",
        )
    window_result_hash = canonical_sha256(
        {
            "window_hash": outcome.window.window_hash,
            "evidence_hash": evidence.evidence_hash,
            "signals_binding_hash": signals.binding_hash,
            "decision_code": outcome.decision.code,
        }
    )
    return E1WindowResult(
        window=outcome.window,
        evidence=evidence,
        gate_signals=signals,
        cycle=outcome,
        window_result_hash=window_result_hash,
    )


# ---------------------------------------------------------------------------
# stage 2: real EnvCoder + compile (templates -> artifacts -> materials)
# ---------------------------------------------------------------------------
def execute_real_envcoder_and_compile(
    teacher: Any, window_result: Any, runtime: Any
) -> E1CandidateMaterials:
    """Compile the COMPLETE window and run ONE EnvCoder call per unique
    template through the teacher's staged backend (bounded repair).

    ``compile_task_specs`` is a pure re-derivation over the SAME
    window (deterministic; zero extra LLM calls); the EnvCoder calls
    are the window's ONLY K1/F1 accounting. Returns the REAL
    ``E1CandidateMaterials``; fewer than 12 compiled dynamic artifacts
    refuses the whole window (no stub padding).
    """
    ctx = "e1_driver.envcoder_compile"
    validate_runtime_surface(runtime, ctx)
    _require_gen_manager(teacher, ctx)
    require_real_object(window_result, "window_result", ctx)
    if not isinstance(window_result, E1WindowResult):
        raise DriverError(
            E1_DRIVER_BAD_TYPE,
            f"{ctx}: window_result must be the E1WindowResult object "
            f"the board stage produced, got {type(window_result).__name__}",
        )
    compile_result = compile_task_specs(window_result.window)
    template_artifacts = []
    for template in compile_result.templates:
        representative = next(
            spec
            for spec in compile_result.specs
            if spec.template_hash == template.template_hash
        )
        artifact, repairs = run_envcoder_with_repair(
            teacher.llm_client,
            spec=representative,
            seed_examples=teacher.seed_examples,
            backend=teacher.envcoder_backend,
            max_repairs=teacher.max_repairs,
            ledger=teacher.ledger,
            window_id=window_result.window.window_id,
        )
        require_real_object(
            artifact, f"envcoder_artifact[{template.template_hash}]", ctx
        )
        template_artifacts.append(
            (template.template_hash, artifact, tuple(repairs))
        )
    succeeded = {t for t, _a, _r in template_artifacts}
    compiled_count = sum(
        1
        for spec in compile_result.specs
        if spec.template_hash in succeeded
    )
    teacher.record_driver_compiled_pool(compiled_count)
    if compiled_count < _NUM_DYNAMIC_SLOTS:
        raise DriverError(
            E1_DRIVER_INSUFFICIENT_ARTIFACTS,
            f"{ctx}: window produced {compiled_count} compiled dynamic "
            f"artifact(s) < {_NUM_DYNAMIC_SLOTS}; the whole window is "
            "refused (no stub/placeholder padding)",
        )
    materials_hash = canonical_sha256(
        {
            "window_result_hash": window_result.window_result_hash,
            "templates": [
                [template_hash, artifact.artifact_id, artifact.env_code]
                for template_hash, artifact, _repairs in template_artifacts
            ],
            "spec_hashes": [
                spec.spec_hash for spec in compile_result.specs
            ],
        }
    )
    return E1CandidateMaterials(
        window_result_hash=window_result.window_result_hash,
        compile_result=compile_result,
        template_artifacts=tuple(template_artifacts),
        materials_hash=materials_hash,
    )
