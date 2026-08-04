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
  update are injected callables (``loss_fn`` / ``optimizer_update_fn``) and
  the pipeline only verifies their outputs.
* ZERO_MEMORY is never accepted as a production memory mode.
* REAL_* execution flags in the report reflect ONLY what actually ran.
"""

from __future__ import annotations

import base64
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
from .env_restore import build_template, encode_env_state
from .errors import InvalidEvidenceError, ProductionBlockedError
from .evidence_selector import (
    evidence_based_select,
    mint_selection_evidence_from_outcomes,
)
from .feasibility_classifier import classify_frontier
from .frontier_archive import FrontierArchive
from .frontier_distributions import compose_12_plus_4
from .fresh_process_restore import (
    BLOCKED_WAITING_CONTROLLER_SIGNED_REGISTRY_BUNDLE,
    SYNTHETIC_SIGNATURE_PREFIX,
    production_joint_pass,
    run_fresh_process_restore_production,
    verdict_from_evidence,
)
from .invocation_gate import (
    InvocationReason,
    build_aggregate_evidence,
    decide_invocation,
)
from .llm_contracts import (
    REAL_TWO_LLM_BLOCKED_NO_AUTHORIZED_CLIENT,
    AuthorizedTwoLLMRuntime,
    PlannerOutput,
    run_two_llm_production,
)
from .memory_modes import MemoryRestoreMode, MemoryRestoreRequest
from .provenance import DataSource
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
BLOCKED_NO_INJECTED_ORIGINAL_LOSS = "BLOCKED_NO_INJECTED_ORIGINAL_LOSS"
BLOCKED_NO_INJECTED_OPTIMIZER_UPDATE = "BLOCKED_NO_INJECTED_OPTIMIZER_UPDATE"
BLOCKED_NO_INJECTED_PREDICATES = "BLOCKED_NO_INJECTED_PREDICATES"
BLOCKED_FRONTIER_DISTRIBUTIONS_UNPROVIDED = "BLOCKED_FRONTIER_DISTRIBUTIONS_UNPROVIDED"
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
    previous_plan_ref: str | None = None      # None -> REVISION_REQUIRED (2 calls)
    reuse_plan: PlannerOutput | None = None   # required when 0 calls reuse it
    # --- selector / distributions ----------------------------------------------
    anchor_manifest: AnchorManifest | None = None
    retention: RetentionContract | None = None
    frontier_distributions: tuple[Any, ...] = ()
    # --- mixed-start update -----------------------------------------------------
    mixed_episodes: int = 0
    episode_horizon: int = 0
    loss_fn: Callable[..., Any] | None = None            # ORIGINAL loss, injected
    optimizer_update_fn: Callable[..., Any] | None = None  # exactly one update
    archive_path: str = ""
    checkpoint_dir: str = ""


@dataclass(frozen=True)
class E3PreflightResult:
    ready: bool
    gates: Mapping[str, bool]
    blockers: tuple[str, ...]
    preflight_version: str = PREFLIGHT_VERSION


def _probe_training_surface(student: Any, method_name: str) -> bool:
    """True unless the adapter's method raises NotImplementedError.

    The probe argument is an empty path: a real implementation fails with a
    file/schema error (surface exists); the R9-pending read-only mount raises
    NotImplementedError (surface absent).  Nothing is written either way.
    """
    method = getattr(student, method_name, None)
    if method is None:
        return False
    try:
        if method_name == "restore_full_state":
            method("")
        else:  # save_full_state: empty path can only fail, never write
            method("", None, {})
    except NotImplementedError:
        return False
    except Exception:
        return True
    return True


def run_e3_preflight(config: E3WindowConfig) -> E3PreflightResult:
    """Fail-closed production preflight: every gap is a named blocker."""
    gates: dict[str, bool] = {}
    blockers: list[str] = []

    student = config.student
    mounted = isinstance(student, StudentAdapter) and config.student_params is not None
    gates["STUDENT_MOUNTED"] = bool(mounted)
    if not mounted:
        blockers.append("BLOCKED_STUDENT_NOT_MOUNTED")

    surface = mounted and _probe_training_surface(student, "restore_full_state")
    gates["STUDENT_TRAINING_SURFACE"] = bool(surface)
    if not surface:
        blockers.append(BLOCKED_TRAINING_SURFACE_PENDING_R9)

    round_trip = mounted and _probe_training_surface(student, "save_full_state")
    gates["CHECKPOINT_ROUND_TRIP_CAPABILITY"] = bool(round_trip)
    if not round_trip:
        blockers.append(f"{BLOCKED_TRAINING_SURFACE_PENDING_R9}:save_full_state")

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

    gates["ORIGINAL_LOSS_INJECTED"] = bool(callable(config.loss_fn))
    if not callable(config.loss_fn):
        blockers.append(BLOCKED_NO_INJECTED_ORIGINAL_LOSS)
    gates["ORIGINAL_OPTIMIZER_UPDATE_INJECTED"] = bool(callable(config.optimizer_update_fn))
    if not callable(config.optimizer_update_fn):
        blockers.append(BLOCKED_NO_INJECTED_OPTIMIZER_UPDATE)

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


def _load_production_memory(config: E3WindowConfig, student: Any,
                            bundle: Any) -> tuple[Any, str]:
    """Memory for mixed-start rollouts, fail-closed with the runner semantics."""
    mode = MemoryRestoreMode(str(config.memory_mode))
    if mode is MemoryRestoreMode.ZERO_MEMORY:
        raise ProductionBlockedError(
            f"{ZERO_MEMORY_NOT_A_PRODUCTION_MODE}: mixed-start rollouts never run "
            "zero-memory")
    if mode is MemoryRestoreMode.SAVED_POLICY_MEMORY:
        artifact = config.memory_artifact
        if artifact is None or config.memory_loader is None:
            raise ProductionBlockedError(
                f"{SAVED_POLICY_MEMORY_BLOCKED_NO_MEMORY_ARTIFACT} (mixed-start)")
        import hashlib
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
        memory = config.memory_loader(artifact)
        check = student.validate_memory(memory, 1)
        if not bool(check.get("ok")):
            raise ProductionBlockedError(
                f"mixed-start policy memory failed validate_memory: "
                f"{tuple(check.get('reasons', ()))}")
        return memory, "SAVED_POLICY_MEMORY_VERIFIED"
    if mode is MemoryRestoreMode.HISTORY_BURN_IN:
        if config.burn_in_executor is None:
            raise ProductionBlockedError(
                f"{HISTORY_BURN_IN_BLOCKED_NO_BURN_IN_EXECUTOR} (mixed-start)")
        reference = getattr(bundle, "history_reference", None) \
            if bundle is not None else None
        if reference is None:
            reference = config.history_artifact_ref
        if not reference:
            raise ProductionBlockedError(
                "HISTORY_BURN_IN mixed-start blocked: no history reference")
        memory = config.burn_in_executor(reference)
        check = student.validate_memory(memory, 1)
        if not bool(check.get("ok")):
            raise ProductionBlockedError(
                f"mixed-start burn-in memory failed validate_memory: "
                f"{tuple(check.get('reasons', ()))}")
        return memory, "HISTORY_BURN_IN_VERIFIED"
    raise ProductionBlockedError(f"unhandled memory mode: {mode!r} (fail closed)")


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
    if config.previous_plan_ref:
        decision = decide_invocation(InvocationReason.NO_SIGNIFICANT_CHANGE,
                                     reuse_plan_ref=config.previous_plan_ref)
    else:
        decision = decide_invocation(InvocationReason.REVISION_REQUIRED)
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
        plan = config.reuse_plan
    steps["STEP05_TWO_LLM_TYPED_PRODUCTION_GATE"] = {
        "llm_calls": llm_result["llm_calls"],
        "role_order": list(llm_result["role_order"]),
        "evidence_hash": llm_result["evidence_hash"],
        "plan_id": plan.plan_id,
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
    if len(config.frontier_distributions) == 0:
        raise ProductionBlockedError(
            f"{BLOCKED_FRONTIER_DISTRIBUTIONS_UNPROVIDED}: the caller must supply "
            "the measured 12 distributions (never self-invented)")
    frontier_plan = compose_12_plus_4(
        config.frontier_distributions,
        manifest=config.anchor_manifest,
        retention=config.retention,
        archive=archive,
        evidence_hashes=(selection_evidence.evidence_hash,),
    )
    steps["STEP07_FRONTIER_12_PLUS_4_COMPOSITION"] = {
        "distributions": len(frontier_plan.distributions),
        "anchors_bound": len(frontier_plan.anchor_binding.get("anchor_ids", ())),
        "plan_hash": frontier_plan.plan_hash,
    }

    # STEP 8 — mixed-start rollouts (restored-frontier + standard-reset).
    window_memory, memory_status = _load_production_memory(config, student, bundle)
    start_rng = np.random.default_rng(config.seed_base + 2)
    eligible: list[str] = []
    weights: list[float] = []
    for distribution in frontier_plan.distributions:
        for sid in distribution.eligible_states:
            eligible.append(sid)
            weights.append(float(distribution.start_state_weights[sid]))
    total_weight = float(sum(weights))
    probs = [w / total_weight for w in weights]
    frontier_weight = float(selection.frontier_start_weight)
    obs_dim = int(np.asarray(obs).reshape(-1).shape[0])
    observations = np.zeros((0, obs_dim), dtype=np.float32)
    actions: list[int] = []
    rewards: list[float] = []
    dones: list[bool] = []
    start_kinds: list[str] = []
    frontier_starts = standard_starts = 0
    for episode in range(int(config.mixed_episodes)):
        kind = "FRONTIER" if float(start_rng.random()) < frontier_weight \
            else "STANDARD_RESET"
        if kind == "FRONTIER":
            sid = str(start_rng.choice(np.asarray(eligible), p=np.asarray(probs)))
            _entry, ep_bundle = runner.restore_entry(archive, sid)
            ep_state = ep_bundle.env_state
            ep_memory = window_memory
            ep_prev_action = ep_bundle.previous_action
            ep_prev_reward = ep_bundle.previous_reward
            frontier_starts += 1
        else:
            ep_key = jax.random.PRNGKey(int(start_rng.integers(0, 2 ** 31)))
            _ep_obs, ep_state = env.reset_env(ep_key, params_env)
            ep_memory = student.initial_memory(1)
            ep_prev_action, ep_prev_reward = 0, 0.0
            standard_starts += 1
        ep_obs_now = observe_fn(ep_state)
        ep_key = jax.random.PRNGKey(int(start_rng.integers(0, 2 ** 31)))
        for _t in range(int(config.episode_horizon)):
            obs_batch = np.asarray(ep_obs_now).reshape(1, -1)
            out = student.policy_step(config.student_params, obs_batch, ep_memory,
                                      ep_prev_action, ep_prev_reward, None, True)
            ep_memory = out.get("new_memory", out.get("memory"))
            action = int(np.asarray(out["action"]).reshape(-1)[0])
            ep_key, step_key = jax.random.split(ep_key)
            ep_obs_now, ep_state, reward, ep_done, _info = setup["step_fn"](
                step_key, ep_state, action, params_env)
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
    }

    # STEP 9 — injected ORIGINAL loss (never redefined here).
    loss_value = config.loss_fn(batch, config.student_params)
    loss_float = float(np.asarray(loss_value))
    if not math.isfinite(loss_float):
        raise ProductionBlockedError(
            f"injected original loss returned a non-finite value: {loss_float}")
    steps["STEP09_INJECTED_ORIGINAL_LOSS"] = {"loss": loss_float}

    # STEP 10 — EXACTLY ONE injected optimizer update.
    update_out = config.optimizer_update_fn(config.student_params, batch)
    if not isinstance(update_out, Mapping) or "params" not in update_out:
        raise ProductionBlockedError(
            "optimizer_update_fn must return a mapping containing 'params'")
    if int(update_out.get("update_count", -1)) != 1:
        raise ProductionBlockedError(
            f"exactly one optimizer update is required, got "
            f"update_count={update_out.get('update_count')!r}")
    new_params = update_out["params"]
    new_sha = _params_sha256(new_params)
    if new_sha == params_sha:
        raise ProductionBlockedError(
            "optimizer update left params bit-identical (no real update happened)")
    grad_norm = float(update_out.get("grad_norm", float("nan")))
    if not math.isfinite(grad_norm):
        raise ProductionBlockedError(f"grad_norm is not finite: {grad_norm}")
    report["real_one_update_executed"] = True
    steps["STEP10_EXACTLY_ONE_OPTIMIZER_UPDATE"] = {
        "update_count": 1,
        "params_sha256_before": params_sha,
        "params_sha256_after": new_sha,
        "grad_norm": grad_norm,
    }

    # STEP 11 — checkpoint save/load round trip via the adapter.
    import os
    ckpt_dir = config.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = student.save_full_state(
        str(os.path.join(ckpt_dir, "e3_window_checkpoint")),
        {"params": new_params,
         "global_step": int(update_out.get("global_step", 1))},
        {"schema": E3_WINDOW_SCHEMA, "run_id": config.run_id,
         "memory_mode": str(config.memory_mode)})
    reloaded = student.restore_full_state(str(ckpt_path))
    reloaded_sha = _params_sha256(reloaded["params"])
    if reloaded_sha != new_sha:
        raise ProductionBlockedError(
            "checkpoint round trip changed params: saved "
            f"{new_sha[:16]}… != reloaded {reloaded_sha[:16]}…")
    report["checkpoint_reload"] = True
    steps["STEP11_CHECKPOINT_SAVE_LOAD_ROUND_TRIP"] = {
        "checkpoint_path": str(ckpt_path),
        "params_sha256_round_trip": reloaded_sha,
    }

    # STEP 12 — next-policy-step replay with the reloaded params.
    replay_out = student.policy_step(reloaded["params"], observations[:1],
                                     student.initial_memory(1), None, None, None, True)
    replay_action = int(np.asarray(replay_out["action"]).reshape(-1)[0])
    action_count = student.action_spec().count
    if not (0 <= replay_action < action_count):
        raise ProductionBlockedError(
            f"replay action {replay_action} out of range [0, {action_count})")
    replay_ok = True
    if "logits" in replay_out:
        replay_ok = bool(np.isfinite(np.asarray(replay_out["logits"])).all())
    steps["STEP12_NEXT_POLICY_STEP_REPLAY"] = {
        "replay_action_in_range": True,
        "replay_logits_finite": replay_ok,
    }

    # STEP 13 — NaN/Inf sweep over the updated parameter tree and scalars.
    leaves = jax.tree_util.tree_leaves(new_params)
    finite = all(bool(np.isfinite(np.asarray(leaf, dtype=np.float64)).all())
                 for leaf in leaves if np.asarray(leaf).size > 0
                 and np.issubdtype(np.asarray(leaf).dtype, np.number))
    scalar_ok = math.isfinite(loss_float) and math.isfinite(grad_norm)
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
