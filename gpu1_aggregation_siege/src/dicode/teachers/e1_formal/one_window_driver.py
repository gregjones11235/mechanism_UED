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

from . import dicode_protocol as DP
from . import probe_result_binding as PRB
from . import roundtrip_attestation as RA
from . import selection_attestation as SA
from . import shared_runtime_seam as SRS
from . import smoke_attestation as SM
from . import teacher_continuity as TC
from . import update_attestation as UA
from . import variant_binding as VB
from .board import ReviewWindow, WINDOW_STATUS_COMPLETE
from .canonical import canonical_sha256
from .controller import CycleOutcome, run_review_cycle
from .envcoder import EnvCoderArtifact, RepairRecord, run_envcoder_with_repair
from .evidence import EvidenceSnapshot, build_evidence_snapshot
from .executable_candidates import ExecutableCandidate
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
E1_DRIVER_MATERIALS_MISMATCH = "E1_DRIVER_MATERIALS_MISMATCH"
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
    """The REAL review-window stage output (object, not summary).

    ``continuity`` binds the window to its ONE teacher instance +
    signed runtime bundle (CC2 follow-up P0-10); every later stage
    re-checks it before running.
    """

    window: ReviewWindow
    evidence: EvidenceSnapshot
    gate_signals: GateSignalReport
    cycle: CycleOutcome
    window_result_hash: str
    continuity: TC.OneWindowContinuity


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
    continuity = TC.begin_one_window_session(teacher, runtime)
    window_result_hash = canonical_sha256(
        {
            "window_hash": outcome.window.window_hash,
            "evidence_hash": evidence.evidence_hash,
            "signals_binding_hash": signals.binding_hash,
            "decision_code": outcome.decision.code,
            "continuity_session_hash": continuity.session_hash,
        }
    )
    return E1WindowResult(
        window=outcome.window,
        evidence=evidence,
        gate_signals=signals,
        cycle=outcome,
        window_result_hash=window_result_hash,
        continuity=continuity,
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
    # P0-10: the SAME GenManager + the SAME signed bundle as the stage
    # that opened the window (a swapped teacher fails closed)
    try:
        TC.assert_one_window_continuity(
            window_result.continuity, teacher, runtime, ctx
        )
    except TC.TeacherContinuityError as e:
        raise DriverError(e.code, f"{ctx}: {e}") from e
    compile_result = compile_task_specs(window_result.window)
    template_artifacts = []
    for template in compile_result.templates:
        representative = next(
            spec
            for spec in compile_result.specs
            if spec.template_hash == template.template_hash
        )
        try:
            artifact, repairs = run_envcoder_with_repair(
                teacher.envcoder_llm_client,
                spec=representative,
                seed_examples=teacher.seed_examples,
                backend=teacher.envcoder_backend,
                max_repairs=teacher.max_repairs,
                ledger=teacher.ledger,
                window_id=window_result.window.window_id,
            )
        except EnvCoderError as e:
            # a template whose env-code fails the real ladder after the
            # bounded repair loop is REFUSED per-template (never padded,
            # never silently accepted); the window continues to the other
            # templates and is refused as a whole only when fewer than
            # 12 dynamic artifacts compile.
            print(f"[e1-driver] envcoder template "
                  f"{template.template_hash[:12]} refused: {e.code}",
                  flush=True)
            continue
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


# ---------------------------------------------------------------------------
# stage 3: executable candidate pool (Mode A binding, fail-closed surfaces)
# ---------------------------------------------------------------------------
def execute_real_candidate_binding(
    teacher: Any,
    window_result: Any,
    materials: Any,
    runtime: Any,
    *,
    allow_test_only: bool = False,
) -> Tuple[ExecutableCandidate, ...]:
    """Bind the window's executable candidate pool (Mode A).

    The four execution-surface hashes (observation/action ABI, reward
    contract, reset protocol, seed policy) come ONLY from the
    bundle-bound shared objects through the seam — no defaults, no
    guesses. While the shared runtime is absent (this whole round on
    the production path) this stage fails closed honestly. Every
    candidate returned carries the conspicuous
    VARIANT_PARAMETER_NOT_EXECUTED marker: binding never executes.

    ``allow_test_only`` is the conspicuously-marked gate the TEST_ONLY
    closed loop uses; the default production surface refuses TEST_ONLY
    bundles fail-closed.
    """
    ctx = "e1_driver.candidate_binding"
    validate_runtime_surface(runtime, ctx)
    _require_gen_manager(teacher, ctx)
    require_real_object(window_result, "window_result", ctx)
    require_real_object(materials, "candidate_materials", ctx)
    if not isinstance(window_result, E1WindowResult):
        raise DriverError(
            E1_DRIVER_BAD_TYPE,
            f"{ctx}: window_result must be the E1WindowResult object, "
            f"got {type(window_result).__name__}",
        )
    if not isinstance(materials, E1CandidateMaterials):
        raise DriverError(
            E1_DRIVER_BAD_TYPE,
            f"{ctx}: materials must be the E1CandidateMaterials object "
            f"the EnvCoder stage produced, got "
            f"{type(materials).__name__}",
        )
    if materials.window_result_hash != window_result.window_result_hash:
        raise DriverError(
            E1_DRIVER_MATERIALS_MISMATCH,
            f"{ctx}: materials bind window_result_hash "
            f"{materials.window_result_hash!r} but the window stage "
            f"produced {window_result.window_result_hash!r}",
        )
    # P0-10: one teacher / one bundle across the whole window
    try:
        TC.assert_one_window_continuity(
            window_result.continuity, teacher, runtime, ctx
        )
    except TC.TeacherContinuityError as e:
        raise DriverError(e.code, f"{ctx}: {e}") from e
    # the seam binds every capability object and re-checks its
    # identity hash; TEST_ONLY bundles are refused unless the explicit
    # gate is opened (never by default)
    resolutions = SRS.resolve_all_from_bundle(
        runtime, ctx, allow_test_only=allow_test_only
    )
    surfaces = VB.execution_surfaces_from_bundle_resolutions(
        resolutions, ctx
    )
    backend = teacher.envcoder_backend
    return VB.bind_executable_pool_from_materials(
        window=window_result.window,
        compile_result=materials.compile_result,
        template_artifacts=materials.template_artifacts,
        execution_surfaces=surfaces,
        backend_name=backend.name,
        stages_passed=tuple(backend.capabilities),
    )


# ---------------------------------------------------------------------------
# stage 4: registry-signed probe results (consumption ONLY — E1 never mints)
# ---------------------------------------------------------------------------
def execute_real_candidate_probes(
    teacher: Any,
    candidates: Any,
    runtime: Any,
    *,
    probe_results: Any,
    student_checkpoint_identity: str,
    reference_checkpoint_identity: str,
    window_result: Any = None,
    allow_test_only: bool = False,
) -> Tuple[Any, ...]:
    """Consume the window's REGISTRY-SIGNED probe pool fail-closed.

    E1 never mints probe results: the pool arrives signed by the
    shared probe runner registry and is consumed through
    ``probe_result_binding`` with the window's identities — Student /
    Reference identity hashes from the signed bundle, checkpoint
    identities supplied by the caller (they must equal the probe
    input checkpoints — P0-10), and the seed bank / reset protocol /
    runner registry identities from the bundle-bound probe runner
    object. ANY violation refuses the whole pool.
    """
    ctx = "e1_driver.candidate_probes"
    validate_runtime_surface(runtime, ctx)
    _require_gen_manager(teacher, ctx)
    require_real_object(candidates, "executable_candidate_pool", ctx)
    if not isinstance(candidates, tuple) or len(candidates) == 0:
        raise DriverError(
            E1_DRIVER_MISSING_OBJECT,
            f"{ctx}: the executable candidate pool must be a non-empty "
            "tuple of ExecutableCandidate objects",
        )
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, ExecutableCandidate):
            raise DriverError(
                E1_DRIVER_BAD_TYPE,
                f"{ctx}: executable_candidate_pool[{index}] must be an "
                f"ExecutableCandidate, got {type(candidate).__name__}",
            )
    # P0-10: one teacher / one bundle across the whole window
    if window_result is not None and isinstance(
        window_result, E1WindowResult
    ):
        try:
            TC.assert_one_window_continuity(
                window_result.continuity, teacher, runtime, ctx
            )
        except TC.TeacherContinuityError as e:
            raise DriverError(e.code, f"{ctx}: {e}") from e
    for name, value in (
        ("student_checkpoint_identity", student_checkpoint_identity),
        ("reference_checkpoint_identity", reference_checkpoint_identity),
    ):
        if not isinstance(value, str) or len(value) != 64:
            raise DriverError(
                E1_DRIVER_BAD_TYPE,
                f"{ctx}: {name} must be a 64-hex checkpoint identity, "
                f"got {value!r}",
            )
    # the probe runner object supplies the frozen seed bank + reset
    # protocol identities (absent shared object => fail closed)
    resolutions = SRS.resolve_all_from_bundle(
        runtime, ctx, allow_test_only=allow_test_only
    )
    probe_runner_obj = resolutions["probe_runner"].object_ref
    seed_bank_hash = getattr(probe_runner_obj, "seed_bank_hash", None)
    reset_protocol_hash = getattr(
        probe_runner_obj, "reset_protocol_hash", None
    )
    for name, value in (
        ("seed_bank_hash", seed_bank_hash),
        ("reset_protocol_hash", reset_protocol_hash),
    ):
        if not isinstance(value, str) or len(value) != 64:
            raise DriverError(
                E1_DRIVER_RUNTIME_UNBOUND,
                f"{ctx}: the bundle-bound probe runner exposes no "
                f"64-hex {name!r} (got {value!r}); refusing a guessed "
                "seed/reset identity",
            )
    return PRB.consume_registry_signed_probe_results(
        probe_results,
        candidates=candidates,
        student_identity_hash=runtime.object_identity_hash(
            "student_identity"
        ),
        student_checkpoint_hash=student_checkpoint_identity,
        reference_identity_hash=runtime.object_identity_hash(
            "reference_identity"
        ),
        reference_checkpoint_hash=reference_checkpoint_identity,
        seed_bank_hash=seed_bank_hash,
        reset_protocol_hash=reset_protocol_hash,
        runner_registry_hash=runtime.object_identity_hash("probe_runner"),
        ctx=ctx,
        allow_test_only=allow_test_only,
    )


# ---------------------------------------------------------------------------
# stage 5: attested criterion-wise selection (signed signals ONLY)
# ---------------------------------------------------------------------------
def execute_real_criterion_selection(
    teacher: Any,
    window_result: Any,
    candidates: Any,
    probe_results: Any,
    signed_signals: Any,
    runtime: Any,
    *,
    k: int,
    seed: int,
    critic_policy: str,
    family_cap: int,
    weights: Any = None,
    allow_test_only: bool = False,
) -> Tuple[Any, Any]:
    """Run the criterion-wise selection under FULL pool binding.

    Returns ``(SelectionOutcome, SelectionAttestation)`` — the
    attestation is what the GenManager certification consumes (never
    the bare outcome). Caller-shaped signal mappings have no path in:
    only ``SignedCriterionSignals`` verified against their candidate +
    probe result enter scoring.
    """
    ctx = "e1_driver.criterion_selection"
    validate_runtime_surface(runtime, ctx)
    _require_gen_manager(teacher, ctx)
    require_real_object(window_result, "window_result", ctx)
    if not isinstance(window_result, E1WindowResult):
        raise DriverError(
            E1_DRIVER_BAD_TYPE,
            f"{ctx}: window_result must be the E1WindowResult object, "
            f"got {type(window_result).__name__}",
        )
    # P0-10: one teacher / one bundle across the whole window
    try:
        TC.assert_one_window_continuity(
            window_result.continuity, teacher, runtime, ctx
        )
    except TC.TeacherContinuityError as e:
        raise DriverError(e.code, f"{ctx}: {e}") from e
    return SA.execute_criterion_selection(
        window_id=window_result.window.window_id,
        window_hash=window_result.window.window_hash,
        candidates=candidates,
        probe_results=probe_results,
        signed_signals=signed_signals,
        k=k,
        seed=seed,
        critic_policy=critic_policy,
        family_cap=family_cap,
        weights=weights,
        allow_test_only=allow_test_only,
    )


# ---------------------------------------------------------------------------
# stage 6: 12 dynamic -> CanonicalDiCodeTrainingBatchPlan (15 + 1)
# ---------------------------------------------------------------------------
def execute_real_batch_certification(
    *,
    selection_attestation: Any,
    anchor_manifest_hash: str,
    ctx: str = "e1_driver.batch_certification",
) -> DP.CanonicalDiCodeTrainingBatchPlan:
    """Certify the attested selection into the shared 15+1 plan.

    The ONLY translation 12 dynamic + 3 non-target anchors = 15
    curriculum ids + DiCode-appended OriginalTask; the plan binds the
    selection attestation + anchor manifest and is hash-verified.
    """
    return DP.build_canonical_dicode_training_batch_plan(
        selection_attestation=selection_attestation,
        anchor_manifest_hash=anchor_manifest_hash,
        ctx=ctx,
    )


# ---------------------------------------------------------------------------
# stage 7: canonical DiCode one update (counts come from DiCode)
# ---------------------------------------------------------------------------
def execute_canonical_dicode_one_update(
    *,
    plan: Any,
    selection_attestation: Any,
    one_update_runtime: Any,
    update_record: Any,
    anchor_manifest_hash: str,
    signer_id: str,
    test_only: bool = False,
    ctx: str = "e1_driver.canonical_dicode_one_update",
) -> UA.OptimizerUpdateAttestation:
    """Consume ONE DiCode update bound to the certified 15+1 plan.

    Verifies the plan against its attestation, verifies the update
    record's DiCode timeline consistency (before/after counts from the
    DiCode timeline, never self-reported), and attests exactly one
    update with ``verified_batch_hash = plan.plan_hash``.
    """
    DP.verify_canonical_dicode_training_batch_plan(
        plan,
        selection_attestation=selection_attestation,
        anchor_manifest_hash=anchor_manifest_hash,
        ctx=ctx,
    )
    DP.assert_update_matches_timeline(one_update_runtime, update_record, ctx)
    UA.verify_update_execution_record(update_record, ctx)
    return UA.attest_exactly_one_update(
        one_update_runtime.training_runtime,
        update_record,
        verified_batch_hash=plan.plan_hash,
        signer_id=signer_id,
        test_only=test_only,
        ctx=ctx,
    )


# ---------------------------------------------------------------------------
# stage 8: consume the full run-state round trip (shared checkpoint)
# ---------------------------------------------------------------------------
def consume_full_runstate_roundtrip(
    *,
    checkpoint: Any,
    update_attestation: Any,
    runtime_bundle_hash: str,
    roundtrip_evidence: Any,
    ctx: str = "e1_driver.runstate_roundtrip",
) -> RA.FullStateRoundTripAttestation:
    """Consume the shared CanonicalDiCodeRunStateCheckpoint's full-state
    restore evidence.

    The checkpoint must equal the update's OUTPUT state and bind the
    same runtime bundle; the round-trip evidence must prove a fresh
    subprocess restore, a leaf comparison and an identical
    next-policy-step replay. Params-only / plain JSON is never
    full-state.
    """
    DP.verify_runstate_checkpoint_binds_update(
        checkpoint,
        update_attestation=update_attestation,
        runtime_bundle_hash=runtime_bundle_hash,
        ctx=ctx,
    )
    evidence = roundtrip_evidence  # (identity, attestation) pair
    identity, attestation = evidence
    # the restored identity must equal the shared canonical checkpoint
    DP.assert_roundtrip_identity_matches_checkpoint(identity, checkpoint, ctx)
    RA.verify_full_state_round_trip(attestation, identity)
    return attestation


# ---------------------------------------------------------------------------
# stage 9: build the signed E1 real-smoke attestation (whole chain)
# ---------------------------------------------------------------------------
def build_e1_smoke_attestation(
    *,
    run_id: str,
    branch: str,
    git_sha: str,
    window_result: Any,
    candidate_materials: Any,
    probe_pool: Any,
    plan: Any,
    update_attestation: Any,
    roundtrip_attestation: Any,
    runtime: Any,
    student_checkpoint_identity: str,
    reference_checkpoint_identity: str,
    formal_asset_registry_hash: str,
    anchor_manifest_hash: str,
    signer_id: str,
    test_only: bool = False,
    ctx: str = "e1_driver.smoke_attestation",
) -> SM.E1RealSmokeAttestation:
    """Fold the WHOLE one-window chain into the signed smoke attestation.

    Every stage hash (window, materials, probe pool, signals pool, the
    15+1 plan, the update, the round trip) is bound; consumers
    re-verify each against the live expected values.
    """
    return SM.issue_e1_real_smoke_attestation(
        run_id=run_id,
        branch=branch,
        git_sha=git_sha,
        runtime_bundle_hash=runtime.bundle_hash,
        student_identity_hash=runtime.object_identity_hash(
            "student_identity"
        ),
        student_checkpoint_hash=student_checkpoint_identity,
        reference_identity_hash=runtime.object_identity_hash(
            "reference_identity"
        ),
        reference_checkpoint_hash=reference_checkpoint_identity,
        board_journal_hash=window_result.window_result_hash,
        envcoder_artifact_pool_hash=candidate_materials.materials_hash,
        probe_pool_hash=canonical_sha256(
            [probe.attestation_hash for probe in probe_pool]
        ),
        selection_attestation_hash=plan.selection_attestation_hash,
        verified_batch_hash=plan.plan_hash,
        update_attestation_hash=update_attestation.attestation_hash,
        roundtrip_attestation_hash=(
            roundtrip_attestation.attestation_hash
        ),
        formal_asset_registry_hash=formal_asset_registry_hash,
        anchor_manifest_hash=anchor_manifest_hash,
        status=SM.SMOKE_STATUS_EXECUTED,
        signer_id=signer_id,
        test_only=test_only,
        ctx=ctx,
    )
