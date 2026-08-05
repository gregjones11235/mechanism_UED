"""E3 production pipeline: one REAL frontier window (P0-7) + preflight gates.

This module wires the frozen production surfaces into ONE end-to-end window:

  standard-reset Student rollout -> FrontierEntry capture (production archive
  write path) -> single-fresh-process joint full-state restore
  (controller-signed RegistryBundle only) -> same-state actual-N branch
  search -> 0-or-2 typed LLM calls -> deterministic evidence selector
  (official final authority) -> 12 dynamic frontier distributions + 4 shared
  anchors -> mixed-start rollouts -> INJECTED original PPO / Original
  V-trace loss -> EXACTLY ONE injected optimizer update -> checkpoint
  save/load round trip -> next-policy-step replay -> NaN/Inf checks.

Honesty rules enforced structurally:

* ``run_e3_preflight`` is fail-closed: any missing production dependency is a
  named blocker, and ``one_window_pipeline`` raises ``ProductionBlockedError``
  instead of running on fakes.
* The Student network, reward, action head, optimizer and the original loss
  definition are NEVER modified or redefined here: the loss and the optimizer
  update arrive through ONE minted ``OriginalTrainingRuntime`` binding
  (CC4 follow-up P0-12 — no plain callables) and the pipeline only verifies
  their outputs.
* ZERO_MEMORY is never accepted as a production memory mode.
* REAL_* execution flags in the report reflect ONLY what actually ran.
"""

from __future__ import annotations

import base64
import hashlib
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .archive_schema import FrontierArchiveEntry
from .branch_search_runner import (
    BranchSearchRunConfig,
    BranchSearchRunner,
    MemoryArtifactRef,
    SEARCH_SOURCE_REFERENCE_POLICY,
    SEARCH_SOURCE_STUDENT_DETERMINISTIC,
    SEARCH_SOURCE_STUDENT_STOCHASTIC,
)
from .discovery_provenance import (
    BLOCKED_WAITING_FROZEN_FORMAL_ASSET_REGISTRY,
    DiscoveryProvenance,
    production_registry_bound,
)
from .distribution_runtime import (
    DISTRIBUTION_RUNTIME_VERSION,
    resolve_distribution_binding,
    verify_distribution_binding,
)
from .env_restore import build_template, encode_env_state
from .errors import InvalidEvidenceError, ProductionBlockedError
from .evidence_selector import (
    evidence_based_select,
    mint_selection_evidence_from_outcomes,
)
from .feasibility_classifier import classify_frontier
from .frontier_archive import FrontierArchive
from .frontier_distributions import (
    compile_planner_to_frontier_distributions,
    compose_12_plus_4,
)
from .fresh_process_restore import (
    BLOCKED_WAITING_CONTROLLER_SIGNED_REGISTRY_BUNDLE,
    SYNTHETIC_SIGNATURE_PREFIX,
    production_joint_pass,
    run_fresh_process_restore_production,
    verdict_from_evidence,
)
from .invocation_gate import (
    build_aggregate_evidence,
    evidence_hash_of,
)
from .llm_contracts import (
    REAL_TWO_LLM_BLOCKED_NO_AUTHORIZED_CLIENT,
    AuthorizedTwoLLMRuntime,
    PlannerOutput,
    assert_planner_output_bound,
    derive_invocation_from_evidence,
    run_two_llm_production,
)
from .memory_modes import MemoryRestoreMode, MemoryRestoreRequest
from .optimizer_attestation import (
    BLOCKED_OPTIMIZER_STEP_BASELINE_UNMEASURED,
    OPTIMIZER_ATTESTATION_VERSION,
    attestation_fields,
    mint_optimizer_update_attestation,
    verify_optimizer_update_attestation,
)
from .provenance import DataSource
from .round_trip_evidence import (
    RESTORE_DRIVER_IN_PROCESS_ADAPTER,
    measure_replay_equivalence,
    mint_checkpoint_round_trip_evidence,
    verify_checkpoint_round_trip_evidence,
)
from .search_statistics import (
    BranchOutcome,
    estimate_feasibility,
    estimate_feasibility_by_source,
)
from .anchor_manifest import (
    BLOCKED_SHARED_ANCHOR_MANIFEST,
    AnchorManifest,
    RetentionContract,
    validate_anchor_manifest,
)
from .student_binding import bind_capture_entry
from .surface_capability import (
    BLOCKED_NO_SIGNED_TRAINING_SURFACE_CAPABILITY,
    TrainingSurfaceCapability,
    verify_training_surface_capability,
)
from .training_runtime import (
    BLOCKED_NO_BOUND_ORIGINAL_TRAINING_RUNTIME,
    OriginalTrainingRuntime,
    runtime_binding_summary,
    verify_original_training_runtime,
)
from .verified_restore_context import mint_verified_restore_context
from dicode.student_adapters.protocol import StudentAdapter

E3_WINDOW_SCHEMA = "simulator_frontier.e3-window/v1"
PREFLIGHT_VERSION = "e3-preflight/v1"

# This round ships ENTRYPOINTS ONLY: the real window is blocked at preflight
# (R9 training surface + controller-signed artifacts pending).  These flags
# mirror the script reports and never self-upgrade.
REAL_ACTUAL_N_EXECUTED = False
REAL_TWO_LLM_EXECUTED = False
REAL_ONE_UPDATE_EXECUTED = False
CHECKPOINT_RELOAD = False

BLOCKED_TRAINING_SURFACE_PENDING_R9 = "BLOCKED_TRAINING_SURFACE_PENDING_R9"
BLOCKED_NO_CAPTURE_PROVENANCE = "BLOCKED_NO_CAPTURE_PROVENANCE"
BLOCKED_NO_INJECTED_TASKPARAM_APPLY_FN = "BLOCKED_NO_INJECTED_TASKPARAM_APPLY_FN"
BLOCKED_NO_INJECTED_PREDICATES = "BLOCKED_NO_INJECTED_PREDICATES"
BLOCKED_NO_OBSERVE_FN = "BLOCKED_NO_OBSERVE_FN"
ZERO_MEMORY_NOT_A_PRODUCTION_MODE = "ZERO_MEMORY_NOT_A_PRODUCTION_MODE"
SAVED_POLICY_MEMORY_BLOCKED_NO_MEMORY_ARTIFACT = (
    "SAVED_POLICY_MEMORY_BLOCKED_NO_MEMORY_ARTIFACT")
HISTORY_BURN_IN_BLOCKED_NO_BURN_IN_EXECUTOR = "HISTORY_BURN_IN_BLOCKED_NO_BURN_IN_EXECUTOR"

E3_WINDOW_STEPS = (
    "STEP01_STANDARD_RESET_ROLLOUT",
    "STEP02_FRONTIER_CAPTURE_PRODUCTION_WRITE",
    "STEP03_COMBINED_FRESH_PROCESS_RESTORE",
    "STEP04_REAL_ACTUAL_N_BRANCH_SEARCH",
    "STEP05_TWO_LLM_TYPED_PRODUCTION_GATE",
    "STEP06_EVIDENCE_SELECTOR_FINAL_AUTHORITY",
    "STEP07_FRONTIER_12_PLUS_4_COMPOSITION",
    "STEP08_MIXED_START_ROLLOUT",
    "STEP09_INJECTED_ORIGINAL_LOSS",
    "STEP10_EXACTLY_ONE_OPTIMIZER_UPDATE",
    "STEP11_CHECKPOINT_SAVE_LOAD_ROUND_TRIP",
    "STEP12_NEXT_POLICY_STEP_REPLAY",
    "STEP13_FINITE_CHECK",
)

_UNMEASURED = "UNMEASURED"


@dataclass(frozen=True, kw_only=True)
class E3WindowConfig:
    """Everything one real frontier window needs (fail-closed at preflight).

    Every production dependency is INJECTED by the caller/controller; the
    pipeline never constructs fakes to fill a gap.  Fields left unbound stay
    unbound and surface as named preflight blockers.
    """

    run_id: str = ""
    # --- Student mount (never modified: network/reward/action head intact) ---
    student: Any = None                       # StudentAdapter
    student_params: Any = None
    loaded_state: Mapping[str, Any] | None = None  # adapter.load_full_state output
    # CC4 follow-up (P0-15): training-surface capability arrives as a signed
    # descriptor (never inferred from exception probes).  Unbound/invalid/
    # self-signed/foreign-identity descriptors block the preflight.
    training_surface_capability: TrainingSurfaceCapability | None = None
    reference_student: Any = None
    reference_params: Any = None
    # CC4 follow-up (P0-5): the Reference identity/checkpoint/memory binding.
    # A mounted Reference without these stays blocked — Student identity or
    # Student memory is never substituted for the Reference's own binding.
    reference_checkpoint_id: str = ""
    reference_memory_artifact: Any = None         # MemoryArtifactRef
    reference_memory_loader: Callable[[Any], Any] | None = None
    reference_history_artifact_ref: str = ""
    reference_burn_in_executor: Callable[[Any], Any] | None = None
    # --- capture rollout ---------------------------------------------------
    max_timesteps: int = 0
    reset_seed: int = 0
    capture_at_step: int = 0
    capture_provenance: Any = None            # CaptureProvenance (controller-provided)
    memory_mode: str = ""
    memory_request: MemoryRestoreRequest | None = None
    memory_artifact: MemoryArtifactRef | None = None
    memory_loader: Callable[[MemoryArtifactRef], Any] | None = None
    history_artifact_ref: str = ""
    burn_in_executor: Callable[[Any], Any] | None = None
    success_predicate: Callable[[Mapping[str, Any]], bool] | None = None
    progress_fn: Callable[[Mapping[str, Any]], float] | None = None
    # --- search -------------------------------------------------------------
    requested_n: int = 0
    horizon: int = 0
    seed_base: int = 0
    # --- joint restore --------------------------------------------------------
    restore_request: Any = None               # FreshProcessRestoreRequest
    scratch_dir: str = ""
    # --- two LLM --------------------------------------------------------------
    # CC4 follow-up (P0-8): production requires an AuthorizedTwoLLMRuntime
    # (mint-only authorization + call journal); a bare client factory is no
    # longer accepted on the official path.
    two_llm_runtime: AuthorizedTwoLLMRuntime | None = None
    # CC4 follow-up (P0-9): the 0-or-2 decision is DERIVED from measured
    # evidence change; a reuse is only possible with the full typed previous
    # plan AND the evidence hash that plan was bound to.
    reuse_plan: PlannerOutput | None = None
    previous_evidence_hash: str = ""
    # --- selector / distributions ----------------------------------------------
    # CC4 follow-up (P0-10): the 12 dynamic frontier distributions are no
    # longer a caller-supplied field.  They are COMPILED deterministically
    # from the typed planner output in STEP07
    # (``compile_planner_to_frontier_distributions``), so the planner — not an
    # arbitrary hand-built list — actually drives what gets executed.
    anchor_manifest: AnchorManifest | None = None
    retention: RetentionContract | None = None
    # --- mixed-start update -----------------------------------------------------
    mixed_episodes: int = 0
    episode_horizon: int = 0
    # CC4 follow-up (P0-12): the original loss and the optimizer update are
    # never plain callables on the production path.  They arrive through ONE
    # minted OriginalTrainingRuntime binding (descriptors + per-callable
    # source hashes + recomputed runtime hash); an unbound or drifted runtime
    # is a named preflight blocker (no second loss, no second optimizer).
    training_runtime: OriginalTrainingRuntime | None = None
    # CC4 follow-up (P0-11): the compiled distributions' TaskParams must
    # EXECUTE: every frontier episode applies its binding's taskparams to the
    # base env params through this injected surface
    # (taskparam_apply_fn(params_env, taskparams) -> episode env params).
    # TaskParam application is environment/training-runtime specific, so it is
    # injected and audited exactly like the original loss / optimizer update;
    # an unbound surface is a named preflight blocker (fail closed).
    taskparam_apply_fn: Callable[..., Any] | None = None
    archive_path: str = ""
    checkpoint_dir: str = ""


@dataclass(frozen=True)
class E3PreflightResult:
    ready: bool
    gates: Mapping[str, bool]
    blockers: tuple[str, ...]
    preflight_version: str = PREFLIGHT_VERSION


def run_e3_preflight(config: E3WindowConfig) -> E3PreflightResult:
    """Fail-closed production preflight: every gap is a named blocker."""
    gates: dict[str, bool] = {}
    blockers: list[str] = []

    student = config.student
    mounted = isinstance(student, StudentAdapter) and config.student_params is not None
    gates["STUDENT_MOUNTED"] = bool(mounted)
    if not mounted:
        blockers.append("BLOCKED_STUDENT_NOT_MOUNTED")

    # CC4 follow-up (P0-15): training-surface capability is never inferred
    # from exception behaviour — the old empty-path probes were spoofable
    # (any adapter raising a generic error got certified) and are DELETED.
    # Capability is now evidence: a signed TrainingSurfaceCapability
    # descriptor, verified here, bound to the MOUNTED adapter's identity,
    # and carrying a non-synthetic controller signature.
    capability = config.training_surface_capability
    capability_verified = isinstance(capability, TrainingSurfaceCapability)
    if capability_verified:
        try:
            verify_training_surface_capability(capability)
        except InvalidEvidenceError:
            capability_verified = False
    if capability_verified and str(capability.signature_ref).startswith(
            SYNTHETIC_SIGNATURE_PREFIX):
        # A self-signed capability descriptor is never production evidence.
        capability_verified = False
    if capability_verified and mounted and str(
            capability.adapter_identity_hash) != str(
                student.identity().identity_hash()):
        # A descriptor for ANOTHER adapter never certifies this mount.
        capability_verified = False
    gates["TRAINING_SURFACE_CAPABILITY_SIGNED"] = bool(capability_verified)
    if not capability_verified:
        blockers.append(BLOCKED_NO_SIGNED_TRAINING_SURFACE_CAPABILITY)

    surface = (mounted and capability_verified
               and bool(capability.save_full_state_capable)
               and bool(capability.restore_full_state_capable))
    gates["STUDENT_TRAINING_SURFACE"] = bool(surface)
    if mounted and capability_verified and not surface:
        # The signed descriptor itself declares the surface absent (R9).
        blockers.append(BLOCKED_TRAINING_SURFACE_PENDING_R9)

    # A round trip requires BOTH save and restore, so it is green exactly
    # when the signed descriptor certifies the full surface.
    round_trip = surface
    gates["CHECKPOINT_ROUND_TRIP_CAPABILITY"] = bool(round_trip)

    # Director handoff (P0-b2): the Reference runtime is a COMPLETE mount —
    # adapter + params + checkpoint identity + its OWN memory surface.  The
    # three gates below verify it mechanically:
    #   * REFERENCE_RUNTIME_BOUND            — adapter mounted, params bound,
    #     checkpoint identity non-empty;
    #   * REFERENCE_IDENTITY_DISTINCT_FROM_STUDENT — a Reference is never the
    #     Student under another name;
    #   * REFERENCE_MEMORY_ISOLATED          — Reference memory is bound to the
    #     REFERENCE adapter's own identity/spec (or its own burn-in), never
    #     the Student memory surface.
    ref = config.reference_student
    ref_mounted = (isinstance(ref, StudentAdapter)
                   and config.reference_params is not None
                   and str(config.reference_checkpoint_id).strip() != "")
    gates["REFERENCE_RUNTIME_BOUND"] = bool(ref_mounted)
    if not ref_mounted:
        blockers.append("BLOCKED_REFERENCE_RUNTIME_NOT_BOUND")

    ref_distinct = (ref_mounted and mounted
                    and str(ref.identity().identity_hash())
                    != str(student.identity().identity_hash()))
    gates["REFERENCE_IDENTITY_DISTINCT_FROM_STUDENT"] = bool(ref_distinct)
    if ref_mounted and not ref_distinct:
        blockers.append("BLOCKED_REFERENCE_IDENTITY_EQUALS_STUDENT")

    ref_memory_ok = False
    if ref_mounted:
        try:
            ref_mode = MemoryRestoreMode(str(config.memory_mode))
        except ValueError:
            ref_mode = None
        if ref_mode is MemoryRestoreMode.SAVED_POLICY_MEMORY:
            ref_artifact = config.reference_memory_artifact
            ref_memory_ok = (
                isinstance(ref_artifact, MemoryArtifactRef)
                and str(ref_artifact.student_identity_hash)
                == str(ref.identity().identity_hash())
                and str(ref_artifact.memory_spec_hash)
                == str(ref.memory_spec().spec_hash())
                and callable(config.reference_memory_loader))
        elif ref_mode is MemoryRestoreMode.HISTORY_BURN_IN:
            ref_memory_ok = (
                str(config.reference_history_artifact_ref).strip() != ""
                and callable(config.reference_burn_in_executor))
    gates["REFERENCE_MEMORY_ISOLATED"] = bool(ref_memory_ok)
    if ref_mounted and not ref_memory_ok:
        blockers.append("BLOCKED_REFERENCE_MEMORY_NOT_ISOLATED")

    request = config.restore_request
    bundle_ok = (
        request is not None
        and getattr(request, "registry_bundle", None) is not None
        and not str(request.registry_bundle.controller_signature_ref)
        .startswith(SYNTHETIC_SIGNATURE_PREFIX))
    gates["CONTROLLER_REGISTRY_BUNDLE"] = bool(bundle_ok)
    if not bundle_ok:
        blockers.append(BLOCKED_WAITING_CONTROLLER_SIGNED_REGISTRY_BUNDLE)

    manifest = config.anchor_manifest
    manifest_ok = manifest is not None
    if manifest_ok:
        try:
            validate_anchor_manifest(manifest)
        except InvalidEvidenceError:
            manifest_ok = False
    gates["SHARED_ANCHOR_MANIFEST"] = bool(manifest_ok)
    if not manifest_ok:
        blockers.append(BLOCKED_SHARED_ANCHOR_MANIFEST)

    gates["FROZEN_FORMAL_ASSET_REGISTRY"] = bool(production_registry_bound())
    if not production_registry_bound():
        blockers.append(BLOCKED_WAITING_FROZEN_FORMAL_ASSET_REGISTRY)

    if config.capture_provenance is None:
        gates["CAPTURE_PROVENANCE_BOUND"] = False
        blockers.append(BLOCKED_NO_CAPTURE_PROVENANCE)
    else:
        gates["CAPTURE_PROVENANCE_BOUND"] = True

    try:
        mode = MemoryRestoreMode(str(config.memory_mode))
    except ValueError:
        mode = None
    memory_ok = False
    if mode is MemoryRestoreMode.SAVED_POLICY_MEMORY:
        memory_ok = config.memory_artifact is not None and config.memory_loader is not None
        if not memory_ok:
            blockers.append(SAVED_POLICY_MEMORY_BLOCKED_NO_MEMORY_ARTIFACT)
    elif mode is MemoryRestoreMode.HISTORY_BURN_IN:
        memory_ok = (str(config.history_artifact_ref).strip() != ""
                     and config.burn_in_executor is not None)
        if not memory_ok:
            blockers.append(HISTORY_BURN_IN_BLOCKED_NO_BURN_IN_EXECUTOR)
    elif mode is MemoryRestoreMode.ZERO_MEMORY:
        blockers.append(ZERO_MEMORY_NOT_A_PRODUCTION_MODE)
    else:
        blockers.append("BLOCKED_UNKNOWN_OR_MISSING_MEMORY_MODE")
    gates["MEMORY_PRODUCTION_MODE"] = bool(memory_ok)

    predicates_ok = (config.success_predicate is not None
                     and callable(config.success_predicate)
                     and config.progress_fn is not None
                     and callable(config.progress_fn))
    gates["FEASIBILITY_PREDICATES_INJECTED"] = bool(predicates_ok)
    if not predicates_ok:
        blockers.append(BLOCKED_NO_INJECTED_PREDICATES)

    gates["TWO_LLM_AUTHORIZED_RUNTIME"] = isinstance(
        config.two_llm_runtime, AuthorizedTwoLLMRuntime)
    if not isinstance(config.two_llm_runtime, AuthorizedTwoLLMRuntime):
        blockers.append(REAL_TWO_LLM_BLOCKED_NO_AUTHORIZED_CLIENT)

    # CC4 follow-up (P0-12): the original loss + optimizer update must arrive
    # as ONE minted, verified OriginalTrainingRuntime binding — plain
    # callables are no longer accepted on the production path.
    runtime = config.training_runtime
    runtime_ok = isinstance(runtime, OriginalTrainingRuntime)
    if runtime_ok:
        try:
            verify_original_training_runtime(runtime)
        except InvalidEvidenceError:
            runtime_ok = False
    gates["ORIGINAL_TRAINING_RUNTIME_BOUND"] = bool(runtime_ok)
    if not runtime_ok:
        blockers.append(BLOCKED_NO_BOUND_ORIGINAL_TRAINING_RUNTIME)
    # CC4 follow-up (P0-13): the optimizer-step attestation needs a
    # mechanically measurable step baseline (loaded_state["global_step"]);
    # without it the step increment would be self-reportable.
    loaded = config.loaded_state
    step_baseline = loaded.get("global_step", None) if isinstance(loaded, Mapping) else None
    step_baseline_ok = (not isinstance(step_baseline, bool)
                        and isinstance(step_baseline, int)
                        and step_baseline >= 0)
    gates["OPTIMIZER_STEP_BASELINE_MEASURABLE"] = bool(step_baseline_ok)
    if not step_baseline_ok:
        blockers.append(BLOCKED_OPTIMIZER_STEP_BASELINE_UNMEASURED)
    # CC4 follow-up (P0-11): every compiled distribution carries non-empty
    # taskparam_ranges (the compiler enforces it), so the mixed-start step
    # ALWAYS needs the injected TaskParams application surface — an unbound
    # surface means the taskparam distribution could never execute.
    gates["TASKPARAM_APPLY_FN_INJECTED"] = bool(callable(config.taskparam_apply_fn))
    if not callable(config.taskparam_apply_fn):
        blockers.append(BLOCKED_NO_INJECTED_TASKPARAM_APPLY_FN)

    return E3PreflightResult(ready=not blockers, gates=gates, blockers=tuple(blockers))


# ---------------------------------------------------------------------------
# measurement helpers (never record actions/routes/logits)
# ---------------------------------------------------------------------------

def _band(value: Any, low: float, high: float) -> str:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)):
        return _UNMEASURED
    if float(value) <= low:
        return "LOW"
    if float(value) >= high:
        return "HIGH"
    return "MID"


def _measure_entry_facts(state: Any, terminal: bool) -> dict[str, Any]:
    """Aggregate, non-action-guiding facts measured from the real state.

    Anything not mechanically derivable is recorded as UNMEASURED/-1 — never
    invented.
    """
    health = getattr(state, "health", None)
    max_health = getattr(state, "max_health", None)
    ratio = None
    if health is not None and max_health not in (None, 0):
        try:
            ratio = float(health) / float(max_health)
        except (TypeError, ValueError, ZeroDivisionError):
            ratio = None
    achievements = getattr(state, "achievements", None)
    snapshot: dict[str, Any] = {}
    if achievements is not None:
        try:
            snapshot["achievements_done"] = int(sum(bool(v) for v in achievements))
        except TypeError:
            snapshot = {}
    return {
        "floor": int(getattr(state, "floor_number", -1)),
        "gate_progress": float(getattr(state, "gate_progress", 0.0) or 0.0),
        "health_band": _band(ratio, 0.34, 0.67),
        "threat_band": _UNMEASURED,
        "resource_band": _UNMEASURED,
        "inventory_stage": _UNMEASURED,
        "achievement_snapshot": snapshot,
        "terminal": bool(terminal),
    }


def _params_sha256(params: Any) -> str:
    from dicode.student_adapters.checkpoint_codec import cc2_params_sha256
    return cc2_params_sha256(params)


def _optimizer_step_before(config: E3WindowConfig) -> int:
    """The mechanically measurable optimizer step baseline (P0-13).

    Comes from the loaded full state (``loaded_state["global_step"]``) —
    never from the update callable, so the attested increment
    (before -> before+1) cannot be self-reported.
    """
    loaded = config.loaded_state
    step = loaded.get("global_step", None) if isinstance(loaded, Mapping) else None
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ProductionBlockedError(
            f"{BLOCKED_OPTIMIZER_STEP_BASELINE_UNMEASURED}: loaded_state.global_step "
            f"must be a non-negative int to attest the optimizer step increment, "
            f"got {step!r} (fail closed)")
    return int(step)


def _verify_window_memory_source(config: E3WindowConfig, student: Any,
                                 mode: MemoryRestoreMode) -> str:
    """Verify the mixed-start memory source ONCE (fail closed per mode).

    CC4 follow-up (P0-11): the mode-conditional hash/spec/identity checks run
    here, before any episode executes.  Per-episode memory instances are then
    produced FRESH by ``_fresh_window_memory`` — the window never shares one
    memory object across episodes.
    """
    if mode is MemoryRestoreMode.ZERO_MEMORY:
        raise ProductionBlockedError(
            f"{ZERO_MEMORY_NOT_A_PRODUCTION_MODE}: mixed-start rollouts never run "
            "zero-memory")
    if mode is MemoryRestoreMode.SAVED_POLICY_MEMORY:
        artifact = config.memory_artifact
        if artifact is None or config.memory_loader is None:
            raise ProductionBlockedError(
                f"{SAVED_POLICY_MEMORY_BLOCKED_NO_MEMORY_ARTIFACT} (mixed-start)")
        hasher = hashlib.sha256()
        with open(artifact.path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                hasher.update(chunk)
        if hasher.hexdigest() != artifact.sha256:
            raise ProductionBlockedError(
                "memory artifact sha256 mismatch on mixed-start load (fail closed)")
        if artifact.memory_spec_hash != student.memory_spec().spec_hash():
            raise ProductionBlockedError(
                "memory artifact spec hash != search Student memory spec hash")
        if artifact.student_identity_hash != student.identity().identity_hash():
            raise ProductionBlockedError(
                "memory artifact identity hash != search Student identity hash")
        return "SAVED_POLICY_MEMORY_VERIFIED"
    if mode is MemoryRestoreMode.HISTORY_BURN_IN:
        if config.burn_in_executor is None:
            raise ProductionBlockedError(
                f"{HISTORY_BURN_IN_BLOCKED_NO_BURN_IN_EXECUTOR} (mixed-start)")
        return "HISTORY_BURN_IN_VERIFIED"
    raise ProductionBlockedError(f"unhandled memory mode: {mode!r} (fail closed)")


def _fresh_window_memory(config: E3WindowConfig, student: Any, bundle: Any,
                         mode: MemoryRestoreMode) -> Any:
    """A FRESH memory instance for ONE mixed-start episode.

    CC4 follow-up (P0-11): per-state memory restore.  SAVED_POLICY_MEMORY
    re-reads the artifact through the injected loader on every call;
    HISTORY_BURN_IN re-executes burn-in on the RESTORED bundle's own history
    reference (the history that belongs to that very state).  Every instance
    is validated before use; no memory object is ever shared across episodes.
    """
    if mode is MemoryRestoreMode.SAVED_POLICY_MEMORY:
        memory = config.memory_loader(config.memory_artifact)
    elif mode is MemoryRestoreMode.HISTORY_BURN_IN:
        reference = getattr(bundle, "history_reference", None) \
            if bundle is not None else None
        if reference is None:
            reference = config.history_artifact_ref
        if not reference:
            raise ProductionBlockedError(
                "HISTORY_BURN_IN mixed-start blocked: the restored bundle carries "
                "no history reference (per-state memory is never faked)")
        memory = config.burn_in_executor(reference)
    else:
        raise ProductionBlockedError(f"unhandled memory mode: {mode!r} (fail closed)")
    check = student.validate_memory(memory, 1)
    if not bool(check.get("ok")):
        raise ProductionBlockedError(
            f"mixed-start episode memory failed validate_memory: "
            f"{tuple(check.get('reasons', ()))}")
    return memory


def _load_production_memory(config: E3WindowConfig, student: Any,
                            bundle: Any) -> tuple[Any, str]:
    """One-shot convenience wrapper: verify source once + one fresh memory."""
    mode = MemoryRestoreMode(str(config.memory_mode))
    status = _verify_window_memory_source(config, student, mode)
    memory = _fresh_window_memory(config, student, bundle, mode)
    return memory, status


def _binding_action_seed(binding: Any) -> int:
    """Deterministic numpy seed for the episode's action-level RNG stream."""
    digest = hashlib.sha256(
        f"mixed-start-action-stream:{binding.binding_hash}".encode("utf-8")
    ).hexdigest()
    return int(digest[:12], 16)


def _select_mixed_action(*, student: Any, params: Any, obs: Any, memory: Any,
                         prev_action: Any, prev_reward: Any, step_rng: Any,
                         binding: Any, gen: Any, action_count: int) -> tuple[int, Any]:
    """One mixed-start step with the distribution's stochasticity EXECUTING.

    Mirrors the branch runner's action semantics exactly: epsilon-greedy
    override plus optional temperature re-sampling; temperature without
    adapter logits fails closed rather than silently sampling at T=1.  The
    Student network is never modified.
    """
    import numpy as np
    deterministic = float(binding.epsilon) == 0.0 and float(binding.temperature) == 1.0
    out = student.policy_step(params, obs, memory, prev_action, prev_reward,
                              step_rng, deterministic)
    new_memory = out.get("new_memory", out.get("memory"))
    action = int(np.asarray(out["action"]).reshape(-1)[0])
    if deterministic:
        return action, new_memory
    if float(gen.random()) < float(binding.epsilon):
        action = int(gen.integers(0, action_count))
        return action, new_memory
    if float(binding.temperature) != 1.0:
        if "logits" not in out:
            raise ProductionBlockedError(
                "distribution temperature != 1.0 requires logits from the adapter; "
                "none returned (fail closed rather than silently sample at T=1)")
        logits = np.asarray(out["logits"], dtype=np.float64).reshape(-1)
        if logits.shape[0] != action_count:
            raise ProductionBlockedError(
                f"logits length {logits.shape[0]} != action count {action_count} "
                "(fail closed)")
        scaled = logits / float(binding.temperature)
        scaled = scaled - float(np.max(scaled))
        weights = np.exp(scaled)
        probs = weights / float(np.sum(weights))
        action = int(gen.choice(action_count, p=probs))
    return action, new_memory


# ---------------------------------------------------------------------------
# the 13-step pipeline
# ---------------------------------------------------------------------------

def one_window_pipeline(config: E3WindowConfig) -> dict[str, Any]:
    """Run ONE real frontier window end-to-end (never on fakes).

    Raises ``ProductionBlockedError`` when preflight is not green.  Returns a
    report whose REAL_* flags reflect only what actually executed.
    """
    import jax
    import numpy as np

    pre = run_e3_preflight(config)
    if not pre.ready:
        raise ProductionBlockedError(
            "E3_PREFLIGHT_BLOCKED: " + "; ".join(pre.blockers))

    student = config.student
    identity = student.identity()
    params_sha = _params_sha256(config.student_params)
    memory_spec_hash = student.memory_spec().spec_hash()
    capture_student_id = str(identity.candidate_id)

    steps: dict[str, Any] = {}
    report: dict[str, Any] = {
        "schema": E3_WINDOW_SCHEMA,
        "run_id": config.run_id,
        "window_steps": list(E3_WINDOW_STEPS),
        "preflight": {"gates": dict(pre.gates), "blockers": list(pre.blockers)},
        "status": "IN_PROGRESS",
        "real_actual_n_executed": False,
        "real_two_llm_executed": False,
        "real_one_update_executed": False,
        "checkpoint_reload": False,
    }

    # STEP 1 — standard-reset real rollout up to the capture point.
    # CC4 follow-up (P0-3): the capture records the ACTUAL executed steps and,
    # per memory mode, the policy memory / history reference that make the
    # captured state a faithful branch point.  ZERO_MEMORY is ablation-only and
    # can never back a production capture.
    from .craftax_checks import build_core_setup
    try:
        capture_mode = MemoryRestoreMode(str(config.memory_mode))
    except ValueError as exc:
        raise ProductionBlockedError(
            f"unknown memory mode {config.memory_mode!r}: capture refuses to "
            "construct any entry with an unresolvable memory mode (fail closed)") from exc
    if capture_mode is MemoryRestoreMode.ZERO_MEMORY:
        raise ProductionBlockedError(
            f"{ZERO_MEMORY_NOT_A_PRODUCTION_MODE}: ZERO_MEMORY is an ablation-only "
            "mode and can never back a production frontier capture (the formal "
            "archive write is refused before any entry is constructed)")
    record_history = capture_mode is MemoryRestoreMode.HISTORY_BURN_IN
    setup = build_core_setup(max_timesteps=config.max_timesteps,
                             reset_seed=config.reset_seed)
    env, params_env = setup["env"], setup["params"]
    observe_fn = getattr(env, "get_obs", None)
    if observe_fn is None:
        raise ProductionBlockedError(
            f"{BLOCKED_NO_OBSERVE_FN}: the mounted env exposes no get_obs(state)")
    runner_key = jax.random.PRNGKey(config.reset_seed + 1)
    state = setup["state0"]
    obs = observe_fn(state)
    memory = student.initial_memory(1)
    prev_action, prev_reward = 0, 0.0
    done = False
    steps_executed = 0
    rollout_history: list[dict[str, Any]] = []
    for _ in range(int(config.capture_at_step)):
        obs_batch = np.asarray(obs).reshape(1, -1)
        out = student.policy_step(config.student_params, obs_batch, memory,
                                  prev_action, prev_reward, None, True)
        memory = out.get("new_memory", out.get("memory"))
        action = int(np.asarray(out["action"]).reshape(-1)[0])
        if record_history:
            obs_arr = np.ascontiguousarray(np.asarray(obs_batch))
            rollout_history.append({
                "timestep": steps_executed,
                "obs_b64": base64.b64encode(obs_arr.tobytes()).decode("ascii"),
                "obs_dtype": str(obs_arr.dtype),
                "obs_shape": [int(x) for x in obs_arr.shape],
                "action": action,
                "prev_action": int(prev_action),
                "prev_reward": float(prev_reward),
            })
        runner_key, step_key = jax.random.split(runner_key)
        obs, state, reward, done, _info = setup["step_fn"](step_key, state,
                                                           action, params_env)
        prev_action, prev_reward = action, float(np.asarray(reward))
        steps_executed += 1
        if bool(np.asarray(done)):
            break
    terminal_before_capture = bool(np.asarray(done))
    steps["STEP01_STANDARD_RESET_ROLLOUT"] = {
        "capture_at_step": int(config.capture_at_step),
        "steps_executed": steps_executed,
        "terminal_before_capture": terminal_before_capture,
    }

    # CC4 follow-up (P0-3): a capture taken AFTER a terminal transition is not
    # a live branch point — the formal archive write is refused BEFORE any
    # entry is constructed (fail closed; terminal states never enter the
    # production archive as frontier captures).
    if terminal_before_capture:
        raise ProductionBlockedError(
            "CAPTURE_REFUSED_TERMINAL_BEFORE_CAPTURE: the rollout reached a "
            f"terminal state after {steps_executed} executed step(s), before the "
            f"planned capture point {int(config.capture_at_step)}; a terminal "
            "capture is never a frontier branch point, so the formal archive "
            "write is refused (fail closed)")

    # STEP 2 — frontier capture through the PRODUCTION archive write path.
    # CC4 follow-up (P0-3) memory binding: SAVED_POLICY_MEMORY entries carry
    # the LIVE rollout policy memory; HISTORY_BURN_IN entries carry the
    # recorded rollout history that a burn-in executor can replay into memory.
    # The guard chain re-verifies mode-conditional presence independently.
    policy_memory_payload: Any = None
    history_payload: Any = None
    if capture_mode is MemoryRestoreMode.SAVED_POLICY_MEMORY:
        if memory is None:
            raise ProductionBlockedError(
                "SAVED_POLICY_MEMORY capture blocked: the rollout produced no "
                "policy memory at the capture point (fail closed)")
        policy_memory_payload = memory
    elif capture_mode is MemoryRestoreMode.HISTORY_BURN_IN:
        if not rollout_history:
            raise ProductionBlockedError(
                "HISTORY_BURN_IN capture blocked: no rollout steps were executed "
                "before the capture point (history_length must be positive)")
        history_payload = {
            "mode": capture_mode.value,
            "history_length": len(rollout_history),
            "source_episode": f"{config.run_id}:standard-reset:{config.reset_seed}",
            "steps": rollout_history,
        }
    encoded, bundle = encode_env_state(state, next_step_key=runner_key,
                                       previous_action=prev_action,
                                       previous_reward=prev_reward,
                                       policy_memory=policy_memory_payload,
                                       history_reference=history_payload)
    if capture_mode is MemoryRestoreMode.SAVED_POLICY_MEMORY \
            and bundle.policy_memory is None:
        raise ProductionBlockedError(
            "capture memory binding invariant violated: the encoded bundle lost "
            "the SAVED_POLICY_MEMORY policy memory (fail closed)")
    if capture_mode is MemoryRestoreMode.HISTORY_BURN_IN \
            and bundle.history_reference is None:
        raise ProductionBlockedError(
            "capture memory binding invariant violated: the encoded bundle lost "
            "the HISTORY_BURN_IN history reference (fail closed)")
    state_id = encoded.payload_hash
    facts = _measure_entry_facts(state, terminal=bool(np.asarray(done)))
    entry = FrontierArchiveEntry(
        state_id=state_id,
        source_checkpoint_id=str(identity.params_sha256),
        source_episode_id=f"{config.run_id}:standard-reset:{config.reset_seed}",
        source_seed=int(config.reset_seed),
        # CC4 follow-up (P0-3): the EXECUTED step count, never the planned
        # capture_at_step — the entry must record what actually happened.
        source_timestep=int(steps_executed),
        capture_reason="E3_WINDOW_STANDARD_RESET_CAPTURE",
        floor=facts["floor"],
        gate_progress=facts["gate_progress"],
        health_band=facts["health_band"],
        threat_band=facts["threat_band"],
        resource_band=facts["resource_band"],
        inventory_stage=facts["inventory_stage"],
        achievement_snapshot=facts["achievement_snapshot"],
        terminal=facts["terminal"],
        memory_mode=str(config.memory_mode),
        encoded_state_ref=encoded.payload_hash,
        state_hash=encoded.payload_hash,
        provenance_hash="",
        created_at=f"{config.run_id}:window-step2",
    )
    entry = bind_capture_entry(
        entry,
        student_identity_hash=identity.identity_hash(),
        parameter_hash=params_sha,
        memory_spec_hash=memory_spec_hash,
        capture_student_id=capture_student_id,
        discovery_provenance=DiscoveryProvenance.TRAINING_DISCOVERY.value,
    )
    archive = FrontierArchive()
    # The production registry comes ONLY from the controller injection slot,
    # resolved inside the guard chain (P0-1: no caller-supplied registry).
    added, finalized = archive.add_production_entry(
        entry, encoded,
        capture_provenance=config.capture_provenance,
        student_identity=identity,
        expected_parameter_hash=params_sha,
        memory_request=config.memory_request,
    )
    if not added:
        raise ProductionBlockedError(
            "production archive refused the capture entry (dup/quota/capacity)")
    archive.save_production(config.archive_path)
    steps["STEP02_FRONTIER_CAPTURE_PRODUCTION_WRITE"] = {
        "state_id": state_id,
        "archive_size": len(archive),
        "provenance_hash_bound": bool(finalized.provenance_hash),
        "archive_path": config.archive_path,
        "source_timestep": int(steps_executed),
        "memory_mode": capture_mode.value,
        "policy_memory_bound": policy_memory_payload is not None,
        "history_length": len(rollout_history),
    }

    # STEP 3 — joint full-state restore in ONE fresh process (P0-3).
    outcome = run_fresh_process_restore_production(
        config.restore_request, scratch_dir=config.scratch_dir)
    if not outcome.accepted:
        raise ProductionBlockedError(
            "joint fresh-process restore rejected: " + "; ".join(outcome.violations))
    verdict = verdict_from_evidence(outcome.evidence)
    joint = production_joint_pass(verdict, outcome.evidence)
    if not joint:
        raise ProductionBlockedError(
            "production_joint_pass is not green for the fresh-process evidence")
    # P0-2: mint the immutable, evidence-bound restore context.  The minter
    # RECOMPUTES the joint pass internally — no self-reported context exists.
    validate_anchor_manifest(config.anchor_manifest)
    restore_context = mint_verified_restore_context(
        restore_request=config.restore_request,
        outcome=outcome,
        verdict=verdict,
        student_identity_hash=identity.identity_hash(),
        anchor_manifest_hash=config.anchor_manifest.manifest_hash,
        state_id=state_id,
        state_hash=finalized.state_hash,
        archive_hash=archive.archive_hash(),
        source_checkpoint_id=finalized.source_checkpoint_id,
        source_memory_spec_hash=finalized.source_memory_spec_hash,
    )
    steps["STEP03_COMBINED_FRESH_PROCESS_RESTORE"] = {
        "child_pid": outcome.child_pid,
        "joint_proof_status": outcome.joint_proof_status,
        "production_joint_pass": True,
        "context_hash": restore_context.context_hash,
    }

    # STEP 4 — real actual-N branch search from the exact restored state.
    template = build_template(setup["state0"])
    runner = BranchSearchRunner(
        student=student,
        student_params=config.student_params,
        step_fn=setup["step_fn"],
        env_params=params_env,
        template=template,
        observe_fn=observe_fn,
        capture_student_id=capture_student_id,
        search_student_id=capture_student_id,
        train_student_id=capture_student_id,
        reference_student=config.reference_student,
        reference_params=config.reference_params,
        reference_checkpoint_id=config.reference_checkpoint_id,
    )
    search_config = BranchSearchRunConfig(
        state_id=state_id,
        horizon=config.horizon,
        requested_n=config.requested_n,
        memory_mode=config.memory_mode,
        memory_request=config.memory_request,
        success_predicate=config.success_predicate,
        progress_fn=config.progress_fn,
        memory_artifact=config.memory_artifact,
        memory_loader=config.memory_loader,
        history_artifact_ref=config.history_artifact_ref,
        burn_in_executor=config.burn_in_executor,
        # CC4 follow-up (P0-4/P0-5): Reference branches consume ONLY the
        # Reference memory surface — Student memory is never substituted.
        reference_memory_artifact=config.reference_memory_artifact,
        reference_memory_loader=config.reference_memory_loader,
        reference_history_artifact_ref=config.reference_history_artifact_ref,
        reference_burn_in_executor=config.reference_burn_in_executor,
    )
    outcomes: tuple[BranchOutcome, ...] = runner.run_actual_n(
        archive, search_config, seed_base=config.seed_base,
        restore_context=restore_context)
    if len(outcomes) != int(config.requested_n):
        raise ProductionBlockedError(
            f"actual_N {len(outcomes)} != requested_N {config.requested_n} "
            "(never report a partial run as complete)")
    # CC4 follow-up (P0-6): source-specific feasibility.  Student and
    # Reference evidence is NEVER mixed into one success rate: each source
    # keeps its own Wilson estimate, and the frontier classification consumes
    # ONLY the Student branches (the training policy's own evidence).
    by_source = estimate_feasibility_by_source(outcomes)
    student_outcomes = tuple(
        o for o in outcomes
        if o.search_source in (SEARCH_SOURCE_STUDENT_DETERMINISTIC,
                               SEARCH_SOURCE_STUDENT_STOCHASTIC))
    if not student_outcomes:
        raise ProductionBlockedError(
            "source-specific feasibility: no attested Student branches "
            "(a training frontier classification cannot be backed by "
            "Reference-only evidence; fail closed)")
    estimate = estimate_feasibility(student_outcomes)
    classification = classify_frontier(estimate, outcomes=student_outcomes)
    report["real_actual_n_executed"] = True
    steps["STEP04_REAL_ACTUAL_N_BRANCH_SEARCH"] = {
        "actual_n": len(outcomes),
        "requested_n": int(config.requested_n),
        "successes": int(sum(bool(o.success) for o in outcomes)),
        "student_branches": len(student_outcomes),
        "student_successes": int(estimate.successes),
        "classification_source": "STUDENT_ONLY",
        "frontier_class": classification.frontier_class.value,
        "reason_codes": list(classification.reason_codes),
        "source_estimates": {
            source: {
                "actual_branches": est.actual_branches,
                "successes": est.successes,
                "success_rate": est.success_rate,
                "confidence_interval": list(est.confidence_interval),
                "mean_progress": est.mean_progress,
                "max_progress": est.max_progress,
                "transition_cost": est.transition_cost,
                "uncertainty": est.uncertainty,
            }
            for source, est in sorted(by_source.items())
        },
    }

    # STEP 5 — 0-or-2 typed LLM calls (production path, never faked).
    archive_summary = {
        "entry_count": len(archive),
        "bucket_diversity": len({e.bucket() for e in archive.list()}),
        "archive_hash_state_count": len(archive),
    }
    evidence = build_aggregate_evidence(
        estimate, archive_summary, data_source=DataSource.TRAINING_FRONTIER_CAPTURE)
    # CC4 follow-up (P0-9): the 0-or-2 decision is DERIVED from measured
    # evidence change — never from the mere presence of a previous plan
    # reference.  A revision (2 calls) is forced whenever the aggregate
    # evidence, the search budget or the memory mode drifted since the
    # previous typed plan was issued.
    decision, invocation_reasons = derive_invocation_from_evidence(
        current_evidence_hash=evidence_hash_of(evidence),
        previous_plan=config.reuse_plan,
        previous_evidence_hash=config.previous_evidence_hash,
        requested_n=int(config.requested_n),
        horizon=int(config.horizon),
        memory_mode=str(MemoryRestoreMode(str(config.memory_mode)).value),
    )
    llm_result = run_two_llm_production(
        decision, evidence, runtime=config.two_llm_runtime,
        expected_state_id=state_id)
    if llm_result["llm_calls"] == 2:
        plan = llm_result["planner"]
        report["real_two_llm_executed"] = True
    else:
        if config.reuse_plan is None \
                or config.reuse_plan.plan_id != llm_result["reuse_plan_ref"]:
            raise ProductionBlockedError(
                "0-call reuse requires the explicit typed previous plan "
                "(reuse is never implied)")
        # The FULL typed plan must genuinely bind the previous evidence hash
        # before it may be reused (defense in depth: the derivation step
        # already re-verified it).
        assert_planner_output_bound(config.reuse_plan,
                                    evidence_hash=config.previous_evidence_hash)
        plan = config.reuse_plan
    steps["STEP05_TWO_LLM_TYPED_PRODUCTION_GATE"] = {
        "llm_calls": llm_result["llm_calls"],
        "role_order": list(llm_result["role_order"]),
        "evidence_hash": llm_result["evidence_hash"],
        "plan_id": plan.plan_id,
        "invocation_reasons": list(invocation_reasons),
        # CC4 follow-up (P0-8): authorization + call journal audit trail.
        "authorization_id": llm_result["authorization_id"],
        "journal_entries": len(llm_result["journal"]["entries"]),
        "journal_hash": llm_result["journal"]["journal_hash"],
    }

    # STEP 6 — deterministic evidence selector: the OFFICIAL final authority.
    # CC4 follow-up (P0-7): the evidence vector is MINTED straight from the
    # attested branch outcomes — rates, progress, per-source counts, gap and
    # transition cost are recomputed inside the minter, and the evidence hash
    # is derived, never supplied.  Self-reported evidence is structurally
    # impossible on the production path.
    retention_ok = False
    try:
        config.retention.validate()
        retention_ok = True
    except InvalidEvidenceError:
        retention_ok = False
    selection_evidence = mint_selection_evidence_from_outcomes(
        state_id=state_id,
        frontier_class=classification.frontier_class,
        outcomes=outcomes,
        retention_ok=retention_ok,
        anchor_coverage_ok=bool(pre.gates.get("SHARED_ANCHOR_MANIFEST")),
        bucket_diversity=int(archive_summary["bucket_diversity"]),
    )
    selection = evidence_based_select(plan, evidence=selection_evidence)
    steps["STEP06_EVIDENCE_SELECTOR_FINAL_AUTHORITY"] = {
        "accepted": selection.accepted,
        "plan_id": selection.plan_id,
        "frontier_start_weight": selection.frontier_start_weight,
        "reason_codes": list(selection.reason_codes),
        "selection_hash": selection.selection_hash,
    }
    if not selection.accepted:
        report["status"] = "SELECTOR_REJECTED"
        report["steps"] = steps
        return report

    # STEP 7 — 12 dynamic frontier distributions + 4 shared anchors.
    # CC4 follow-up (P0-10): the 12 dynamic distributions are COMPILED from
    # the typed planner output — never a caller-supplied list.  The compiler
    # re-binds the plan to the measured aggregate evidence hash, verifies the
    # minted selection evidence it stamps onto every distribution, and checks
    # every eligible state against the archive (fail closed).
    compilation = compile_planner_to_frontier_distributions(
        plan,
        plan_evidence_hash=llm_result["evidence_hash"],
        selection_evidence=selection_evidence,
        archive=archive,
    )
    frontier_plan = compose_12_plus_4(
        compilation.distributions,
        manifest=config.anchor_manifest,
        retention=config.retention,
        archive=archive,
        evidence_hashes=(selection_evidence.evidence_hash,),
    )
    steps["STEP07_FRONTIER_12_PLUS_4_COMPOSITION"] = {
        "distribution_source": "COMPILED_FROM_TYPED_PLANNER_OUTPUT",
        "compiler_version": compilation.compiler_version,
        "compilation_hash": compilation.compilation_hash,
        "plan_id": plan.plan_id,
        "distributions": len(frontier_plan.distributions),
        "anchors_bound": len(frontier_plan.anchor_binding.get("anchor_ids", ())),
        "plan_hash": frontier_plan.plan_hash,
    }

    # STEP 8 — mixed-start rollouts (restored-frontier + standard-reset).
    # CC4 follow-up (P0-11): the compiled distribution fields EXECUTE here.
    # Every frontier episode samples ONE distribution uniformly, then a start
    # state within it by weight, and resolves an immutable runtime binding
    # whose episode_seed / epsilon / temperature / taskparams mechanically
    # drive the rollout (seeded env continuation, applied action-level
    # stochasticity, per-episode env params).  Memory is prepared FRESH per
    # episode from the restored state's own surface — never one shared
    # window_memory object.
    memory_mode = MemoryRestoreMode(str(config.memory_mode))
    memory_status = _verify_window_memory_source(config, student, memory_mode)
    start_rng = np.random.default_rng(config.seed_base + 2)
    distributions = tuple(frontier_plan.distributions)
    if not distributions:
        raise ProductionBlockedError(
            "no frontier distributions to execute in mixed-start (fail closed)")
    frontier_weight = float(selection.frontier_start_weight)
    action_count = int(student.action_spec().count)
    obs_dim = int(np.asarray(obs).reshape(-1).shape[0])
    observations = np.zeros((0, obs_dim), dtype=np.float32)
    actions: list[int] = []
    rewards: list[float] = []
    dones: list[bool] = []
    start_kinds: list[str] = []
    binding_hashes: list[str] = []
    executed_distribution_ids: list[str] = []
    frontier_starts = standard_starts = 0
    stochastic_episodes = 0
    taskparams_episodes = 0
    frontier_episode_index = 0
    for episode in range(int(config.mixed_episodes)):
        kind = "FRONTIER" if float(start_rng.random()) < frontier_weight \
            else "STANDARD_RESET"
        if kind == "FRONTIER":
            distribution = distributions[
                int(start_rng.integers(0, len(distributions)))]
            states = tuple(distribution.eligible_states)
            state_probs = np.asarray(
                [float(distribution.start_state_weights[s]) for s in states],
                dtype=np.float64)
            state_probs = state_probs / float(state_probs.sum())
            sid = str(start_rng.choice(np.asarray(states), p=state_probs))
            _entry, ep_bundle = runner.restore_entry(archive, sid)
            ep_state = ep_bundle.env_state
            binding = resolve_distribution_binding(
                distribution,
                episode_index=frontier_episode_index,
                seed_base=int(config.seed_base),
            )
            verify_distribution_binding(binding)
            binding_hashes.append(binding.binding_hash)
            executed_distribution_ids.append(binding.distribution_id)
            # Per-state memory restore: a FRESH memory instance for this very
            # episode (artifact re-read / burn-in of the restored bundle's own
            # history reference).
            ep_memory = _fresh_window_memory(config, student, ep_bundle, memory_mode)
            ep_prev_action = ep_bundle.previous_action
            ep_prev_reward = ep_bundle.previous_reward
            # Seed distribution executes: the env continuation RNG is derived
            # from the binding's episode seed (never unseeded, never shared).
            ep_key = jax.random.PRNGKey(int(binding.episode_seed))
            # TaskParams distribution executes: per-episode env params via the
            # injected application surface (fail closed if unbound).
            if binding.taskparams:
                if config.taskparam_apply_fn is None:
                    raise ProductionBlockedError(
                        f"{BLOCKED_NO_INJECTED_TASKPARAM_APPLY_FN}: mixed-start "
                        "cannot apply the distribution taskparams without the "
                        "injected surface (fail closed)")
                ep_params = config.taskparam_apply_fn(
                    params_env, dict(binding.taskparams))
                if ep_params is None:
                    raise ProductionBlockedError(
                        "taskparam_apply_fn returned None (fail closed)")
                taskparams_episodes += 1
            else:
                ep_params = params_env
            if float(binding.epsilon) > 0.0 or float(binding.temperature) != 1.0:
                stochastic_episodes += 1
            gen = np.random.default_rng(_binding_action_seed(binding))
            frontier_starts += 1
            frontier_episode_index += 1
        else:
            ep_key = jax.random.PRNGKey(int(start_rng.integers(0, 2 ** 31)))
            _ep_obs, ep_state = env.reset_env(ep_key, params_env)
            ep_memory = student.initial_memory(1)
            ep_prev_action, ep_prev_reward = 0, 0.0
            ep_params = params_env
            binding = None
            gen = None
            standard_starts += 1
        ep_obs_now = observe_fn(ep_state)
        for _t in range(int(config.episode_horizon)):
            obs_batch = np.asarray(ep_obs_now).reshape(1, -1)
            if binding is None:
                out = student.policy_step(config.student_params, obs_batch,
                                          ep_memory, ep_prev_action,
                                          ep_prev_reward, None, True)
                ep_memory = out.get("new_memory", out.get("memory"))
                action = int(np.asarray(out["action"]).reshape(-1)[0])
            else:
                policy_seed_step = int(gen.integers(0, 2 ** 31))
                step_rng = jax.random.PRNGKey(policy_seed_step)
                action, ep_memory = _select_mixed_action(
                    student=student, params=config.student_params, obs=obs_batch,
                    memory=ep_memory, prev_action=ep_prev_action,
                    prev_reward=ep_prev_reward, step_rng=step_rng,
                    binding=binding, gen=gen, action_count=action_count)
            ep_key, step_key = jax.random.split(ep_key)
            ep_obs_now, ep_state, reward, ep_done, _info = setup["step_fn"](
                step_key, ep_state, action, ep_params)
            observations = np.concatenate(
                [observations, obs_batch.astype(np.float32)], axis=0)
            actions.append(action)
            rewards.append(float(np.asarray(reward)))
            dones.append(bool(np.asarray(ep_done)))
            start_kinds.append(kind)
            ep_prev_action, ep_prev_reward = action, float(np.asarray(reward))
            if bool(np.asarray(ep_done)):
                break
    batch = {
        "observations": observations,
        "actions": np.asarray(actions, dtype=np.int64),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.bool_),
        "start_kinds": tuple(start_kinds),
    }
    steps["STEP08_MIXED_START_ROLLOUT"] = {
        "episodes": int(config.mixed_episodes),
        "frontier_starts": frontier_starts,
        "standard_starts": standard_starts,
        "frontier_start_weight": frontier_weight,
        "transitions": len(actions),
        "memory_status": memory_status,
        "memory_scope": "PER_EPISODE_FRESH",
        "distribution_sampling": "PER_DISTRIBUTION_UNIFORM_THEN_STATE_WEIGHTED",
        "distribution_runtime_version": DISTRIBUTION_RUNTIME_VERSION,
        "distribution_fields_executed": ("seed", "stochasticity", "taskparams"),
        "stochastic_episodes": stochastic_episodes,
        "taskparams_episodes": taskparams_episodes,
        "binding_hashes": tuple(binding_hashes),
        "executed_distribution_ids": tuple(executed_distribution_ids),
    }

    # STEP 9 — ORIGINAL loss through the bound training runtime (P0-12).
    # Defense in depth: preflight already verified the binding; re-verify
    # here so a drifted runtime object can never reach the loss call.
    training_runtime = config.training_runtime
    if not isinstance(training_runtime, OriginalTrainingRuntime):
        raise ProductionBlockedError(
            f"{BLOCKED_NO_BOUND_ORIGINAL_TRAINING_RUNTIME}: STEP09 requires the "
            "minted OriginalTrainingRuntime binding (plain callables are never "
            "accepted)")
    verify_original_training_runtime(training_runtime)
    loss_value = training_runtime.loss_fn(batch, config.student_params)
    loss_float = float(np.asarray(loss_value))
    if not math.isfinite(loss_float):
        raise ProductionBlockedError(
            f"injected original loss returned a non-finite value: {loss_float}")
    steps["STEP09_INJECTED_ORIGINAL_LOSS"] = {
        "loss": loss_float,
        "runtime_binding": runtime_binding_summary(training_runtime),
    }

    # STEP 10 — EXACTLY ONE optimizer update from the SAME bound runtime.
    # CC4 follow-up (P0-13): self-reported update_count / grad_norm values
    # are NEVER read — only the "params" key is consumed.  The evidence of
    # the single real update is the minted OptimizerUpdateAttestation,
    # derived from pipeline-measured facts: before/after params hashes, the
    # loaded-state step baseline (increment before -> before+1), structural
    # finiteness and the digest of the exact batch that was updated.
    update_out = training_runtime.optimizer_update_fn(config.student_params, batch)
    if not isinstance(update_out, Mapping) or "params" not in update_out:
        raise ProductionBlockedError(
            "optimizer_update_fn must return a mapping containing 'params'")
    new_params = update_out["params"]
    new_sha = _params_sha256(new_params)
    attestation = mint_optimizer_update_attestation(
        params_sha256_before=params_sha,
        params_sha256_after=new_sha,
        params_after=new_params,
        optimizer_step_before=_optimizer_step_before(config),
        batch=batch,
    )
    verify_optimizer_update_attestation(attestation)
    report["real_one_update_executed"] = True
    steps["STEP10_EXACTLY_ONE_OPTIMIZER_UPDATE"] = {
        "attestation": attestation_fields(attestation),
        "attestation_version": OPTIMIZER_ATTESTATION_VERSION,
        "runtime_hash": training_runtime.runtime_hash,
    }

    # STEP 11 — checkpoint save/load round trip via the adapter.
    import os
    ckpt_dir = config.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = student.save_full_state(
        str(os.path.join(ckpt_dir, "e3_window_checkpoint")),
        {"params": new_params,
         # CC4 follow-up (P0-13): the saved step is the ATTESTED increment,
         # never a value reported by the update callable.
         "global_step": int(attestation.optimizer_step_after)},
        {"schema": E3_WINDOW_SCHEMA, "run_id": config.run_id,
         "memory_mode": str(config.memory_mode)})
    reloaded = student.restore_full_state(str(ckpt_path))
    if not isinstance(reloaded, Mapping) or "params" not in reloaded:
        raise ProductionBlockedError(
            "restore_full_state must return a mapping containing 'params' "
            "(a reload without params is never accepted; fail closed)")
    reloaded_sha = _params_sha256(reloaded["params"])
    # CC4 follow-up (P0-14): a FULL-state round trip must preserve more than
    # the params — the reloaded global step must equal the attested step.
    reloaded_step = reloaded.get("global_step", None)
    if isinstance(reloaded_step, bool) or not isinstance(reloaded_step, int) \
            or reloaded_step < 0:
        raise ProductionBlockedError(
            f"restored full state carries no valid global_step: "
            f"{reloaded_step!r} (a params-only reload is never accepted as a "
            "full-state round trip; fail closed)")
    if reloaded_sha != new_sha:
        raise ProductionBlockedError(
            "checkpoint round trip changed params: saved "
            f"{new_sha[:16]}… != reloaded {reloaded_sha[:16]}…")
    if int(reloaded_step) != int(attestation.optimizer_step_after):
        raise ProductionBlockedError(
            f"checkpoint round trip changed the global step: saved "
            f"{int(attestation.optimizer_step_after)} != reloaded "
            f"{int(reloaded_step)} (fail closed)")
    report["checkpoint_reload"] = True
    steps["STEP11_CHECKPOINT_SAVE_LOAD_ROUND_TRIP"] = {
        "checkpoint_path": str(ckpt_path),
        "params_sha256_round_trip": reloaded_sha,
        "global_step_round_trip": int(reloaded_step),
        "restore_driver": RESTORE_DRIVER_IN_PROCESS_ADAPTER,
    }

    # STEP 12 — TRUE replay equivalence (CC4 follow-up P0-14): one identical
    # deterministic next-policy step through the UPDATED and the RELOADED
    # parameters must agree exactly on action/logits/value/new-memory.  The
    # measured equivalences are minted into immutable
    # CheckpointRoundTripEvidence together with the round-trip facts; a
    # params-only comparison is never accepted again.
    equivalence = measure_replay_equivalence(
        student,
        params_saved=new_params,
        params_reloaded=reloaded["params"],
        observation=observations[:1],
        memory=student.initial_memory(1),
    )
    replay_range_ok = (
        0 <= int(equivalence["action_saved"]) < action_count
        and 0 <= int(equivalence["action_reloaded"]) < action_count)
    if not replay_range_ok:
        raise ProductionBlockedError(
            f"replay action out of range [0, {action_count}): "
            f"saved={equivalence['action_saved']} "
            f"reloaded={equivalence['action_reloaded']}")
    round_trip_evidence = mint_checkpoint_round_trip_evidence(
        checkpoint_path=str(ckpt_path),
        restore_driver=RESTORE_DRIVER_IN_PROCESS_ADAPTER,
        params_sha256_saved=new_sha,
        params_sha256_reloaded=reloaded_sha,
        global_step_saved=int(attestation.optimizer_step_after),
        global_step_reloaded=int(reloaded_step),
        replay_action_equal=bool(equivalence["action_equal"]),
        replay_logits_equal=bool(equivalence["logits_equal"]),
        replay_value_equal=bool(equivalence["value_equal"]),
        replay_memory_equal=bool(equivalence["memory_equal"]),
    )
    verify_checkpoint_round_trip_evidence(round_trip_evidence)
    steps["STEP12_NEXT_POLICY_STEP_REPLAY"] = {
        "replay_equivalence": {
            "action_equal": bool(equivalence["action_equal"]),
            "logits_equal": bool(equivalence["logits_equal"]),
            "value_equal": bool(equivalence["value_equal"]),
            "memory_equal": bool(equivalence["memory_equal"]),
        },
        "round_trip_evidence_hash": round_trip_evidence.evidence_hash,
        "round_trip_evidence_version": round_trip_evidence.evidence_version,
    }

    # STEP 13 — NaN/Inf sweep over the updated parameter tree and scalars.
    leaves = jax.tree_util.tree_leaves(new_params)
    finite = all(bool(np.isfinite(np.asarray(leaf, dtype=np.float64)).all())
                 for leaf in leaves if np.asarray(leaf).size > 0
                 and np.issubdtype(np.asarray(leaf).dtype, np.number))
    # CC4 follow-up (P0-13): grad_norm is never self-reported; finiteness of
    # the updated params is attested structurally in STEP10.
    scalar_ok = math.isfinite(loss_float)
    steps["STEP13_FINITE_CHECK"] = {
        "params_finite": bool(finite),
        "scalars_finite": bool(scalar_ok),
        "num_leaves_checked": len(leaves),
    }
    if not (finite and scalar_ok):
        report["status"] = "FAIL_FINITE_CHECK"
        report["steps"] = steps
        return report

    report["status"] = "PASS"
    report["steps"] = steps
    return report
