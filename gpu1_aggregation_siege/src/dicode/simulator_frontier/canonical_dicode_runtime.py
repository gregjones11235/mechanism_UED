"""Canonical DiCode training runtime + batch plan + env adapter (P0-b3).

Director handoff: direction 三 must NOT execute a second PPO.  The old
production path built a custom observations/actions/rewards batch and called
``loss_fn(batch, params)`` / ``optimizer_update_fn(params, batch)`` — a
parallel training loop that is now FORBIDDEN as the official path.

The official structure is:

    Frontier Evidence
      -> 12 dynamic frontier distributions + 3 non-target anchors
      -> CanonicalDiCodeTrainingBatchPlan   (the CURRICULUM plan, 15+1)
      -> FrontierDistributionEnvironmentAdapter
      -> CanonicalDiCodeOneUpdateRuntime     (director-shared; ONE update via
         DiCode's original run_session_training -> run_training_session ->
         original PPO-GTrXL)

Direction 三 only GENERATES the start-state distribution, the TaskParams
distribution, seed/stochasticity, the memory restore binding and the
curriculum environment adapter.  It never updates params itself: the ONE
update is delegated to the bound ``CanonicalDiCodeOneUpdateRuntime`` and its
receipt is verified at the contract level.  The DiCode runtime, the PPO loss,
the optimizer, update_epochs / num_minibatches and the reward/action head are
never modified or redefined here.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Mapping

from .errors import InvalidEvidenceError, ProductionBlockedError

CANONICAL_DICODE_RUNTIME_VERSION = "canonical-dicode-one-update-runtime/v1"
# E3_DICODE_SESSION_ALIGNED_CONSERVATIVE_16x128: the formal session runtime
# delegates ONE invocation to DiCode's run_session_training with the configured
# max_updates_per_session (100) — never a for-loop of one-update calls.
CANONICAL_DICODE_SESSION_RUNTIME_VERSION = "canonical-dicode-session-runtime/v1"
CANONICAL_BATCH_PLAN_SCHEMA = "simulator_frontier.canonical-dicode-batch-plan/v1"
ENV_ADAPTER_SCHEMA = "simulator_frontier.frontier-distribution-env-adapter/v1"

BLOCKED_NO_BOUND_CANONICAL_DICODE_RUNTIME = (
    "BLOCKED_NO_BOUND_CANONICAL_DICODE_RUNTIME")

# 15 curriculum slots = 12 dynamic frontier distributions + 3 non-target
# standard-reset anchors.  DiCode appends ORIGINAL_TASK exactly once -> 16.
CURRICULUM_SLOT_COUNT = 15
ORIGINAL_TASK_SLOT = "ORIGINAL_TASK"
ORIGINAL_TASK_PROPORTION = 0.20
CURRICULUM_PROPORTION_TOTAL = 1.0 - ORIGINAL_TASK_PROPORTION  # 0.80

_SYNTHETIC_SIGNATURE_PREFIX = "SYNTHETIC_SIGNATURE_"


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _require_sha256(name: str, value: Any) -> str:
    text = str(value)
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise InvalidEvidenceError(
            f"{name} is not a lowercase sha256 hex digest: {text[:24]!r}…")
    return text


def _require_nonempty_str(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidEvidenceError(
            f"{name} must be a non-empty string, got {value!r}")
    return value


def _import_entrypoint(entrypoint: str, purpose: str) -> Any:
    import importlib
    if not isinstance(entrypoint, str) or entrypoint.count(":") != 1 \
            or not all(part.strip() for part in entrypoint.split(":")):
        raise InvalidEvidenceError(
            f"{purpose} entry point must be 'module:attr', got {entrypoint!r}")
    module_name, attr_name = entrypoint.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise InvalidEvidenceError(
            f"cannot import {purpose} entry point module {module_name!r}: {exc!r}") from exc
    try:
        target = getattr(module, attr_name)
    except AttributeError as exc:
        raise InvalidEvidenceError(
            f"{purpose} entry point attribute {attr_name!r} not found in "
            f"{module_name!r}") from exc
    if not callable(target):
        raise InvalidEvidenceError(
            f"{purpose} entry point resolved to a non-callable "
            f"({type(target).__name__})")
    return target


def callable_source_sha256(name: str, fn: Any) -> str:
    """sha256 of a callable's source file + text (EOL-normalized), fail-closed."""
    if isinstance(fn, Mapping) or not callable(fn):
        raise InvalidEvidenceError(
            f"{name}: expected a callable, got {type(fn).__name__}")
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError) as exc:
        raise InvalidEvidenceError(
            f"{name}: cannot bind the callable — its source text is unavailable "
            f"({exc!r}); fail closed") from exc
    try:
        source_file = str(inspect.getsourcefile(fn) or "<unknown>")
    except TypeError:
        source_file = "<unknown>"
    normalized = source.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(
        f"{source_file}\n::\n{normalized}".encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CanonicalDiCodeTrainingBatchPlan:
    """The 15+1 curriculum plan direction 三 GENERATES for the DiCode runtime.

    Mint-only: ``plan_hash`` is computed in ``__post_init__``.  Structural
    invariants: exactly 15 curriculum slots (no ORIGINAL_TASK among them),
    OriginalTask present exactly once at proportion 0.20, and the 15
    curriculum weights summing to 0.80 (the fourth shared anchor maps to
    OriginalTask semantically, never duplicated as a regular distribution).
    """

    plan_id: str
    curriculum_slots: tuple[str, ...]
    curriculum_weights: Mapping[str, float]
    original_task_included: bool
    original_task_proportion: float
    curriculum_proportion_total: float
    slot_distributions: Mapping[str, Any]
    memory_bindings: Mapping[str, Any]
    env_adapter_id: str
    plan_hash: str = field(init=False)
    plan_schema: str = CANONICAL_BATCH_PLAN_SCHEMA

    def __post_init__(self) -> None:
        _require_nonempty_str("plan_id", self.plan_id)
        _require_nonempty_str("env_adapter_id", self.env_adapter_id)
        slots = tuple(self.curriculum_slots)
        if len(slots) != CURRICULUM_SLOT_COUNT:
            raise InvalidEvidenceError(
                f"canonical batch plan requires exactly {CURRICULUM_SLOT_COUNT} "
                f"curriculum slots, got {len(slots)}")
        if len(set(slots)) != len(slots):
            raise InvalidEvidenceError("duplicate curriculum slot ids")
        if ORIGINAL_TASK_SLOT in slots:
            raise InvalidEvidenceError(
                f"{ORIGINAL_TASK_SLOT} is appended by DiCode exactly once and is "
                "never one of the 15 curriculum slots")
        if not isinstance(self.original_task_included, bool) \
                or not self.original_task_included:
            raise InvalidEvidenceError(
                "OriginalTask must be included exactly once "
                "(original_task_included=True)")
        if isinstance(self.original_task_proportion, bool) \
                or not isinstance(self.original_task_proportion, (int, float)) \
                or abs(float(self.original_task_proportion)
                       - ORIGINAL_TASK_PROPORTION) > 1e-9:
            raise InvalidEvidenceError(
                f"original_task_proportion must be {ORIGINAL_TASK_PROPORTION}, "
                f"got {self.original_task_proportion!r}")
        if isinstance(self.curriculum_proportion_total, bool) \
                or not isinstance(self.curriculum_proportion_total, (int, float)) \
                or abs(float(self.curriculum_proportion_total)
                       - CURRICULUM_PROPORTION_TOTAL) > 1e-9:
            raise InvalidEvidenceError(
                f"the {CURRICULUM_SLOT_COUNT} curriculum slots must share "
                f"{CURRICULUM_PROPORTION_TOTAL}, got "
                f"{self.curriculum_proportion_total!r}")
        weights = dict(self.curriculum_weights)
        if set(weights) != set(slots):
            raise InvalidEvidenceError(
                "curriculum_weights keys must equal curriculum_slots exactly")
        total = 0.0
        for slot, weight in weights.items():
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) \
                    or not 0.0 < float(weight) or not float(weight) < 1.0:
                raise InvalidEvidenceError(
                    f"curriculum weight for {slot!r} must be in (0, 1)")
            total += float(weight)
        if abs(total - CURRICULUM_PROPORTION_TOTAL) > 1e-9:
            raise InvalidEvidenceError(
                f"curriculum weights must sum to {CURRICULUM_PROPORTION_TOTAL}, "
                f"got {total}")
        for name, mapping in (("slot_distributions", self.slot_distributions),
                              ("memory_bindings", self.memory_bindings)):
            if not isinstance(mapping, Mapping):
                raise InvalidEvidenceError(f"{name} must be a mapping")
            if set(mapping) != set(slots):
                raise InvalidEvidenceError(
                    f"{name} keys must equal curriculum_slots exactly "
                    "(per-slot execution, never skipped)")
        payload = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name != "plan_hash"
        }
        payload["curriculum_slots"] = list(slots)
        payload["curriculum_weights"] = dict(weights)
        object.__setattr__(self, "plan_hash", _canonical_sha256(payload))


@dataclass(frozen=True)
class DiCodeOneUpdateContext:
    """The immutable context DiCode's ``run_session_training`` consumes.

    E3-P0-2: the real DiCode entry point is
    ``dicode.training.run_session_training(config, rng, rl_train_state,
    gen_manager, global_update_step, global_env_steps, current_session_idx,
    sampled_task_ids, original_return_prev_session)``.  Direction 三 never
    invents a parallel loop — it assembles THIS context (plus the sampled
    task ids from the batch plan) and delegates.  Mint-only: ``context_hash``
    is derived in ``__post_init__``.
    """

    config: Any
    rng: Any
    rl_train_state: Any
    gen_manager: Any
    global_update_step: int
    global_env_steps: int
    current_session_idx: int
    original_return_prev_session: Any
    selected_candidate_id: str
    runtime_bundle_hash: str
    formal_asset_registry_hash: str
    context_hash: str = field(init=False)
    context_schema: str = "simulator_frontier.dicode-one-update-context/v1"

    def __post_init__(self) -> None:
        for label, value in (("global_update_step", self.global_update_step),
                             ("global_env_steps", self.global_env_steps),
                             ("current_session_idx", self.current_session_idx)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise InvalidEvidenceError(
                    f"DiCodeOneUpdateContext.{label} must be a non-negative int, "
                    f"got {value!r}")
        _require_nonempty_str("selected_candidate_id", self.selected_candidate_id)
        _require_sha256("runtime_bundle_hash", self.runtime_bundle_hash)
        _require_sha256("formal_asset_registry_hash", self.formal_asset_registry_hash)
        payload = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name != "context_hash"
        }
        payload["config"] = "BOUND"
        payload["rng"] = "BOUND"
        payload["rl_train_state"] = "BOUND"
        payload["gen_manager"] = "BOUND"
        payload["original_return_prev_session"] = "BOUND"
        object.__setattr__(self, "context_hash", _canonical_sha256(payload))


@dataclass(frozen=True)
class FrontierDistributionEnvironmentAdapter:
    """The environment/task adapter direction 三 builds for the DiCode runtime.

    Mint-only: resolves and source-hash binds the env factory and the
    taskparam application surface; ``adapter_hash`` is computed in
    ``__post_init__``.  Direction 三 never updates params — it only adapts
    the environment so the DiCode runtime can execute the mixed-start
    curriculum.
    """

    adapter_id: str
    env_entrypoint: str
    env_implementation_hash: str
    taskparam_apply_entrypoint: str
    taskparam_implementation_hash: str
    adapter_hash: str = field(init=False)
    adapter_schema: str = ENV_ADAPTER_SCHEMA

    def __post_init__(self) -> None:
        _require_nonempty_str("adapter_id", self.adapter_id)
        _require_sha256("env_implementation_hash", self.env_implementation_hash)
        _require_sha256("taskparam_implementation_hash",
                        self.taskparam_implementation_hash)
        if self.env_entrypoint.count(":") != 1 \
                or self.taskparam_apply_entrypoint.count(":") != 1:
            raise InvalidEvidenceError(
                "env and taskparam entry points must be 'module:attr'")
        payload = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name != "adapter_hash"
        }
        object.__setattr__(self, "adapter_hash", _canonical_sha256(payload))


def mint_frontier_distribution_environment_adapter(
        *, adapter_id: Any, env_entrypoint: Any, env_implementation_hash: Any,
        taskparam_apply_entrypoint: Any,
        taskparam_implementation_hash: Any) -> FrontierDistributionEnvironmentAdapter:
    env_factory = _import_entrypoint(str(env_entrypoint),
                                     "frontier environment factory")
    if callable_source_sha256("env factory", env_factory) \
            != _require_sha256("env_implementation_hash", env_implementation_hash):
        raise InvalidEvidenceError(
            "env factory implementation hash drift (substitution rejected)")
    taskparam_apply = _import_entrypoint(str(taskparam_apply_entrypoint),
                                         "taskparam application surface")
    if callable_source_sha256("taskparam application", taskparam_apply) \
            != _require_sha256("taskparam_implementation_hash",
                               taskparam_implementation_hash):
        raise InvalidEvidenceError(
            "taskparam application implementation hash drift (substitution "
            "rejected)")
    return FrontierDistributionEnvironmentAdapter(
        adapter_id=str(adapter_id),
        env_entrypoint=str(env_entrypoint),
        env_implementation_hash=str(env_implementation_hash),
        taskparam_apply_entrypoint=str(taskparam_apply_entrypoint),
        taskparam_implementation_hash=str(taskparam_implementation_hash),
    )


def _env_module_source(entrypoint: str, purpose: str = "frontier env") -> str:
    """Read the FULL module source named by a 'module:attr' entry point.

    BUG-E3-10: the loadable task code must be the whole module source (the
    imports + the ``Env`` class) so DiCode's ``load_tasks_from_env_codes``
    can exec it in a fresh module namespace.  ``inspect.getsource`` of a
    class/function alone yields only the class/function body, which is NOT
    loadable.  The entry point is still resolved (identity-bound) by the
    adapter verification; the registered code is the module FILE source.
    """
    module_name, _attr = entrypoint.split(":", 1)
    module = importlib.import_module(module_name)
    try:
        source_file = str(inspect.getsourcefile(module) or "")
    except TypeError:
        source_file = ""
    if not source_file:
        raise InvalidEvidenceError(
            f"{purpose} entry point module {module_name!r} has no retrievable "
            "source file — its code can never be registered as a loadable task "
            "(fail closed)")
    try:
        with open(source_file, "r", encoding="utf-8") as fh:
            return fh.read()
    except OSError as exc:
        raise InvalidEvidenceError(
            f"cannot read {purpose} entry point module source "
            f"{source_file!r}: {exc!r} (fail closed)") from exc


def materialize_and_register(adapter: Any, plan: Any,
                             gen_manager_archive: Any,
                             session_idx: int = 0) -> tuple[str, ...]:
    """Register the 15 curriculum tasks into the GenManager TaskArchive.

    E3-P0-3: ``run_session_training`` only receives ``sampled_task_ids`` and
    loads tasks through the GenManager archive.  This contract method
    materializes the 12 dynamic frontier distributions + 3 non-target anchors
    as 15 real loadable tasks (consumable by DiCode's
    ``load_tasks_from_env_codes``), each bound to the distribution hash,
    source frontier, Student identity and memory binding.  Registration
    failure fails closed; no second training loop is constructed.  Returns
    the 15 registered task ids (== plan.curriculum_slots).

    BUG-E3-05: the real ``TaskArchive`` has NO ``register_task`` — tasks are
    registered through ``record_new_task(child_task, parent_tasks,
    description, session_id)``, then the distribution metadata is stamped
    as node attributes and the environment-factory source as the loadable
    ``code`` (so ``load_tasks_from_env_codes`` reads back the SAME 15 ids).
    """
    verify_frontier_distribution_environment_adapter(adapter)
    if not isinstance(plan, CanonicalDiCodeTrainingBatchPlan):
        raise InvalidEvidenceError(
            "materialize_and_register requires a minted "
            "CanonicalDiCodeTrainingBatchPlan")
    if gen_manager_archive is None:
        raise ProductionBlockedError(
            "materialize_and_register requires the GenManager archive — "
            "frontier tasks are never invented outside the archive "
            "(fail closed)")
    # BUG-E3-10: the registered code is the FULL module source (loadable),
    # not the class/function body returned by inspect.getsource.
    env_code = _env_module_source(str(adapter.env_entrypoint))
    registered = []
    for slot in plan.curriculum_slots:
        distribution = plan.slot_distributions.get(slot)
        if distribution is None:
            raise ProductionBlockedError(
                f"task registration failed: slot {slot!r} has no distribution "
                "binding (fail closed)")
        try:
            #: the REAL TaskArchive registration surface (record_new_task),
            #: never a nonexistent register_task
            gen_manager_archive.record_new_task(
                child_task=str(slot),
                parent_tasks=[],
                description=str(slot),
                session_id=int(session_idx),
            )
            if gen_manager_archive.graph.has_node(str(slot)):
                gen_manager_archive.graph.nodes[str(slot)].update({
                    "distribution_hash": str(plan.plan_hash),
                    "source_frontier": (
                        "FRONTIER_DYNAMIC" if "::" in slot
                        else "NON_TARGET_ANCHOR"),
                    "student_identity": str(plan.env_adapter_id),
                    "memory_binding": json.dumps(
                        plan.memory_bindings.get(slot, {}),
                        sort_keys=True),
                    "code": env_code,
                })
        except Exception as exc:
            raise ProductionBlockedError(
                f"GenManager TaskArchive refused task registration for "
                f"{slot!r}: {exc!r} (fail closed)") from exc
        registered.append(str(slot))
    return tuple(registered)


def verify_frontier_distribution_environment_adapter(adapter: Any) -> None:
    if isinstance(adapter, Mapping) or not isinstance(
            adapter, FrontierDistributionEnvironmentAdapter):
        raise InvalidEvidenceError(
            "expected a minted FrontierDistributionEnvironmentAdapter")
    env_factory = _import_entrypoint(adapter.env_entrypoint,
                                     "frontier environment factory")
    if callable_source_sha256("env factory", env_factory) \
            != adapter.env_implementation_hash:
        raise InvalidEvidenceError(
            "env factory implementation hash drift (substitution rejected)")
    taskparam_apply = _import_entrypoint(adapter.taskparam_apply_entrypoint,
                                         "taskparam application surface")
    if callable_source_sha256("taskparam application", taskparam_apply) \
            != adapter.taskparam_implementation_hash:
        raise InvalidEvidenceError(
            "taskparam application implementation hash drift (substitution "
            "rejected)")
    payload = {
        f.name: getattr(adapter, f.name)
        for f in fields(adapter)
        if f.name != "adapter_hash"
    }
    if _canonical_sha256(payload) != adapter.adapter_hash:
        raise InvalidEvidenceError(
            "adapter_hash mismatch: the environment adapter was tampered with "
            "or self-reported (fail closed)")


@dataclass(frozen=True)
class CanonicalDiCodeOneUpdateRuntime:
    """The director-shared ONE-update runtime bound to DiCode's original chain.

    Mint-only: resolves and source-hash binds BOTH DiCode entry points
    (``run_session_training`` and ``run_training_session``), so direction 三
    can verify the original training chain exists and delegate to it — it
    never executes a second PPO.  ``runtime_hash`` is computed in
    ``__post_init__``; a synthetic trusted signer is rejected.
    """

    runtime_id: str
    selected_candidate_id: str
    run_session_training_entrypoint: str
    run_session_implementation_hash: str
    run_training_session_entrypoint: str
    run_training_implementation_hash: str
    trusted_signer: str
    runtime_hash: str = field(init=False)
    runtime_version: str = CANONICAL_DICODE_RUNTIME_VERSION

    def __post_init__(self) -> None:
        _require_nonempty_str("runtime_id", self.runtime_id)
        _require_nonempty_str("selected_candidate_id", self.selected_candidate_id)
        _require_nonempty_str("trusted_signer", self.trusted_signer)
        if str(self.trusted_signer).startswith(_SYNTHETIC_SIGNATURE_PREFIX):
            raise InvalidEvidenceError(
                "trusted_signer must be a real director signer id — a synthetic "
                "self-signature can never bind the DiCode training runtime")
        _require_sha256("run_session_implementation_hash",
                        self.run_session_implementation_hash)
        _require_sha256("run_training_implementation_hash",
                        self.run_training_implementation_hash)
        if self.run_session_training_entrypoint.count(":") != 1 \
                or self.run_training_session_entrypoint.count(":") != 1:
            raise InvalidEvidenceError(
                "run_session_training and run_training_session entry points "
                "must be 'module:attr'")
        payload = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name != "runtime_hash"
        }
        object.__setattr__(self, "runtime_hash", _canonical_sha256(payload))


def mint_canonical_dicode_one_update_runtime(
        *, runtime_id: Any, selected_candidate_id: Any,
        run_session_training_entrypoint: Any,
        run_session_implementation_hash: Any,
        run_training_session_entrypoint: Any,
        run_training_implementation_hash: Any,
        trusted_signer: Any) -> CanonicalDiCodeOneUpdateRuntime:
    session = _import_entrypoint(str(run_session_training_entrypoint),
                                 "run_session_training")
    if callable_source_sha256("run_session_training", session) \
            != _require_sha256("run_session_implementation_hash",
                               run_session_implementation_hash):
        raise InvalidEvidenceError(
            "run_session_training implementation hash drift (substitution "
            "rejected)")
    training = _import_entrypoint(str(run_training_session_entrypoint),
                                  "run_training_session")
    if callable_source_sha256("run_training_session", training) \
            != _require_sha256("run_training_implementation_hash",
                               run_training_implementation_hash):
        raise InvalidEvidenceError(
            "run_training_session implementation hash drift (substitution "
            "rejected)")
    return CanonicalDiCodeOneUpdateRuntime(
        runtime_id=str(runtime_id),
        selected_candidate_id=str(selected_candidate_id),
        run_session_training_entrypoint=str(run_session_training_entrypoint),
        run_session_implementation_hash=str(run_session_implementation_hash),
        run_training_session_entrypoint=str(run_training_session_entrypoint),
        run_training_implementation_hash=str(run_training_implementation_hash),
        trusted_signer=str(trusted_signer),
    )


def verify_canonical_dicode_one_update_runtime(runtime: Any) -> None:
    if isinstance(runtime, Mapping) or not isinstance(
            runtime, CanonicalDiCodeOneUpdateRuntime):
        raise InvalidEvidenceError(
            "expected a minted CanonicalDiCodeOneUpdateRuntime")
    session = _import_entrypoint(runtime.run_session_training_entrypoint,
                                 "run_session_training")
    if callable_source_sha256("run_session_training", session) \
            != runtime.run_session_implementation_hash:
        raise InvalidEvidenceError(
            "run_session_training implementation hash drift (substitution "
            "rejected)")
    training = _import_entrypoint(runtime.run_training_session_entrypoint,
                                  "run_training_session")
    if callable_source_sha256("run_training_session", training) \
            != runtime.run_training_implementation_hash:
        raise InvalidEvidenceError(
            "run_training_session implementation hash drift (substitution "
            "rejected)")
    payload = {
        f.name: getattr(runtime, f.name)
        for f in fields(runtime)
        if f.name != "runtime_hash"
    }
    if _canonical_sha256(payload) != runtime.runtime_hash:
        raise InvalidEvidenceError(
            "runtime_hash mismatch: the canonical DiCode runtime was tampered "
            "with or self-reported (fail closed)")


@dataclass(frozen=True)
class CanonicalDiCodeSessionRuntime:
    """The formal SESSION runtime bound to DiCode's original training chain.

    E3_DICODE_SESSION_ALIGNED_CONSERVATIVE_16x128: one invocation == one native
    ``run_session_training`` == ``config.dicode_manager.max_updates_per_session``
    (100) outer updates.  This is the formal replacement for the one-update
    smoke runtime; it NEVER builds a second training loop or a for-loop of
    single updates — DiCode's own ``run_training_session`` runs the full
    session.

    Mint-only: resolves and source-hash binds BOTH DiCode entry points
    (``run_session_training`` and ``run_training_session``); ``runtime_hash``
    is computed in ``__post_init__``; a synthetic trusted signer is rejected.
    """

    runtime_id: str
    selected_candidate_id: str
    run_session_training_entrypoint: str
    run_session_implementation_hash: str
    run_training_session_entrypoint: str
    run_training_implementation_hash: str
    trusted_signer: str
    runtime_hash: str = field(init=False)
    runtime_version: str = CANONICAL_DICODE_SESSION_RUNTIME_VERSION

    def __post_init__(self) -> None:
        _require_nonempty_str("runtime_id", self.runtime_id)
        _require_nonempty_str("selected_candidate_id", self.selected_candidate_id)
        _require_nonempty_str("trusted_signer", self.trusted_signer)
        if str(self.trusted_signer).startswith(_SYNTHETIC_SIGNATURE_PREFIX):
            raise InvalidEvidenceError(
                "trusted_signer must be a real director signer id — a synthetic "
                "self-signature can never bind the DiCode training runtime")
        _require_sha256("run_session_implementation_hash",
                        self.run_session_implementation_hash)
        _require_sha256("run_training_implementation_hash",
                        self.run_training_implementation_hash)
        if self.run_session_training_entrypoint.count(":") != 1 \
                or self.run_training_session_entrypoint.count(":") != 1:
            raise InvalidEvidenceError(
                "run_session_training and run_training_session entry points "
                "must be 'module:attr'")
        payload = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name != "runtime_hash"
        }
        object.__setattr__(self, "runtime_hash", _canonical_sha256(payload))


def mint_canonical_dicode_session_runtime(
        *, runtime_id: Any, selected_candidate_id: Any,
        run_session_training_entrypoint: Any,
        run_session_implementation_hash: Any,
        run_training_session_entrypoint: Any,
        run_training_implementation_hash: Any,
        trusted_signer: Any) -> CanonicalDiCodeSessionRuntime:
    session = _import_entrypoint(str(run_session_training_entrypoint),
                                 "run_session_training")
    if callable_source_sha256("run_session_training", session) \
            != _require_sha256("run_session_implementation_hash",
                               run_session_implementation_hash):
        raise InvalidEvidenceError(
            "run_session_training implementation hash drift (substitution "
            "rejected)")
    training = _import_entrypoint(str(run_training_session_entrypoint),
                                  "run_training_session")
    if callable_source_sha256("run_training_session", training) \
            != _require_sha256("run_training_implementation_hash",
                               run_training_implementation_hash):
        raise InvalidEvidenceError(
            "run_training_session implementation hash drift (substitution "
            "rejected)")
    return CanonicalDiCodeSessionRuntime(
        runtime_id=str(runtime_id),
        selected_candidate_id=str(selected_candidate_id),
        run_session_training_entrypoint=str(run_session_training_entrypoint),
        run_session_implementation_hash=str(run_session_implementation_hash),
        run_training_session_entrypoint=str(run_training_session_entrypoint),
        run_training_implementation_hash=str(run_training_implementation_hash),
        trusted_signer=str(trusted_signer),
    )


def verify_canonical_dicode_session_runtime(runtime: Any) -> None:
    if isinstance(runtime, Mapping) or not isinstance(
            runtime, CanonicalDiCodeSessionRuntime):
        raise InvalidEvidenceError(
            "expected a minted CanonicalDiCodeSessionRuntime")
    session = _import_entrypoint(runtime.run_session_training_entrypoint,
                                 "run_session_training")
    if callable_source_sha256("run_session_training", session) \
            != runtime.run_session_implementation_hash:
        raise InvalidEvidenceError(
            "run_session_training implementation hash drift (substitution "
            "rejected)")
    training = _import_entrypoint(runtime.run_training_session_entrypoint,
                                  "run_training_session")
    if callable_source_sha256("run_training_session", training) \
            != runtime.run_training_implementation_hash:
        raise InvalidEvidenceError(
            "run_training_session implementation hash drift (substitution "
            "rejected)")
    payload = {
        f.name: getattr(runtime, f.name)
        for f in fields(runtime)
        if f.name != "runtime_hash"
    }
    if _canonical_sha256(payload) != runtime.runtime_hash:
        raise InvalidEvidenceError(
            "runtime_hash mismatch: the canonical DiCode session runtime was "
            "tampered with or self-reported (fail closed)")


def compile_canonical_15_plus_1(*, plan_id: Any, distributions: Any,
                                non_target_anchor_ids: Any,
                                original_task_anchor_id: Any,
                                original_task_id: Any,
                                env_adapter_id: Any,
                                memory_bindings: Any,
                                anchor_memory_binding: Any = None
                                ) -> CanonicalDiCodeTrainingBatchPlan:
    """Compile the 15+1 curriculum plan with EXPLICIT anchor identities.

    E3-P0-3: anchor semantics are identity-based, never position-based.  The
    Anchor Manifest must explicitly declare the three non-target anchors, the
    single OriginalTask anchor and the OriginalTask id (``original_craftax``).
    Arbitrary four unique strings never pass automatically.

    * Exactly 12 dynamic ``FrontierDistribution`` objects (unique ids).
    * ``non_target_anchor_ids`` must be EXACTLY 3 unique ids.
    * ``original_task_anchor_id`` must be EXACTLY 1 id and
      ``original_task_id`` must be ``original_craftax`` (the OriginalTask is
      appended by DiCode exactly once and NEVER enters ``sampled_task_ids``).
    * The curriculum slots are the 12 dynamic ids + the 3 explicit
      non-target anchors = 15; they share 0.80 (equal share per slot); the
      OriginalTask occupies the remaining 0.20.
    * Anchor ORDER never changes the semantics — the three non-target ids and
      the single OriginalTask id are named, not sliced.
    """
    from .frontier_distributions import FrontierDistribution
    if isinstance(distributions, Mapping) or not isinstance(
            distributions, (list, tuple)):
        raise InvalidEvidenceError(
            "compile_canonical_15_plus_1 requires a sequence of "
            "FrontierDistribution objects")
    dists = tuple(distributions)
    if len(dists) != 12 or len({d.distribution_id for d in dists}) != 12 \
            or any(not isinstance(d, FrontierDistribution) for d in dists):
        raise InvalidEvidenceError(
            "exactly 12 unique typed FrontierDistribution objects are required "
            "(hand-built mappings are never accepted)")
    non_targets = tuple(str(a) for a in non_target_anchor_ids)
    if len(non_targets) != 3 or len(set(non_targets)) != 3:
        raise InvalidEvidenceError(
            "exactly 3 unique non_target_anchor_ids are required (anchor "
            "identity is explicit, never position-based)")
    original_anchor = str(original_task_anchor_id)
    if not original_anchor:
        raise InvalidEvidenceError(
            "original_task_anchor_id must be explicitly declared")
    if original_anchor in non_targets:
        raise InvalidEvidenceError(
            "original_task_anchor_id must differ from the non-target anchors")
    if str(original_task_id) != "original_craftax":
        raise InvalidEvidenceError(
            f"original_task_id must be 'original_craftax', got "
            f"{original_task_id!r} — the OriginalTask identity is explicit")
    curriculum_slots = tuple(d.distribution_id for d in dists) + non_targets
    if len(curriculum_slots) != CURRICULUM_SLOT_COUNT:
        raise InvalidEvidenceError(
            f"expected {CURRICULUM_SLOT_COUNT} curriculum slots, got "
            f"{len(curriculum_slots)}")
    per_slot = CURRICULUM_PROPORTION_TOTAL / CURRICULUM_SLOT_COUNT
    curriculum_weights = {slot: per_slot for slot in curriculum_slots}
    slot_distributions = {d.distribution_id: dict(asdict(d)) for d in dists}
    for anchor in non_targets:
        slot_distributions[anchor] = {"anchor_id": anchor,
                                      "kind": "NON_TARGET_STANDARD_RESET_ANCHOR",
                                      "original_task_anchor_id": original_anchor,
                                      "original_task_id": str(original_task_id)}
    if isinstance(memory_bindings, Mapping):
        bindings = dict(memory_bindings)
    else:
        raise InvalidEvidenceError("memory_bindings must be a mapping")
    for slot in curriculum_slots:
        if slot not in bindings:
            bindings[slot] = (dict(anchor_memory_binding)
                              if anchor_memory_binding is not None
                              else {"memory_mode": "SAVED_POLICY_MEMORY"})
    return CanonicalDiCodeTrainingBatchPlan(
        plan_id=str(plan_id),
        curriculum_slots=curriculum_slots,
        curriculum_weights=curriculum_weights,
        original_task_included=True,
        original_task_proportion=ORIGINAL_TASK_PROPORTION,
        curriculum_proportion_total=CURRICULUM_PROPORTION_TOTAL,
        slot_distributions=slot_distributions,
        memory_bindings=bindings,
        env_adapter_id=str(env_adapter_id),
    )


def _delegate_canonical_session(runtime: Any, *, context: Any, plan: Any,
                                adapter: Any, backend: Any = None,
                                checkpoint_params: Any = None,
                                initial_memory_dict: Any = None,
                                purpose: str = "session") -> Mapping[str, Any]:
    """Common delegate: call DiCode's ``run_session_training`` with the EXACT
    8-tuple ABI and validate every field fail-closed.

    Shared by ``execute_one_update`` (smoke/unit, config max_updates_per_session
    forced to 1) and ``execute_session`` (formal, config
    max_updates_per_session = 100).  Direction 三 NEVER updates params itself
    and NEVER builds a second training loop; ``sampled_task_ids`` come from the
    minted plan and the OriginalTask is appended by DiCode exactly once.

    P0-4: ``initial_memory_dict`` (resumed architecture memory from the previous
    session's RunState) is forwarded to run_session_training so the PPO initial
    runner state starts from the trained hidden state, never a zero re-init.
    """
    if not isinstance(context, DiCodeOneUpdateContext):
        raise InvalidEvidenceError(
            f"execute_{purpose} requires a minted DiCodeOneUpdateContext")
    if str(context.selected_candidate_id) != str(runtime.selected_candidate_id):
        raise InvalidEvidenceError(
            f"execute_{purpose}: the context selected_candidate_id must equal "
            "the runtime selected_candidate_id (training binds the SELECTED "
            "Student; fail closed)")
    if not isinstance(plan, CanonicalDiCodeTrainingBatchPlan):
        raise InvalidEvidenceError(
            f"execute_{purpose} requires a minted CanonicalDiCodeTrainingBatchPlan")
    if not isinstance(adapter, FrontierDistributionEnvironmentAdapter):
        raise InvalidEvidenceError(
            f"execute_{purpose} requires a minted "
            "FrontierDistributionEnvironmentAdapter")
    session = _import_entrypoint(runtime.run_session_training_entrypoint,
                                 "run_session_training")
    sampled_task_ids = list(plan.curriculum_slots)
    if ORIGINAL_TASK_SLOT in sampled_task_ids:
        raise InvalidEvidenceError(
            "OriginalTask must never enter sampled_task_ids — DiCode appends "
            "it exactly once (fail closed)")
    try:
        receipt_tuple = session(
            context.config,
            context.rng,
            context.rl_train_state,
            context.gen_manager,
            int(context.global_update_step),
            int(context.global_env_steps),
            int(context.current_session_idx),
            sampled_task_ids,
            context.original_return_prev_session,
            backend=backend,
            checkpoint_params=checkpoint_params,
            initial_memory_dict=initial_memory_dict,
        )
    except Exception as exc:
        raise ProductionBlockedError(
            f"run_session_training rejected the {purpose} request: {exc!r}") from exc
    # BUG-E3-06: run_session_training returns the canonical EIGHT-tuple
    # (rng, rl_train_state, global_update_step, global_env_steps,
    #  training_metrics, num_updates_in_session, categorized_tasks,
    #  evaluation_metrics) — NEVER a Mapping. Unpack it EXPLICITLY and
    # validate every field's semantics fail-closed; a wrong structure is
    # refused, never silently accepted by a compatibility fallback.
    if not isinstance(receipt_tuple, tuple) or len(receipt_tuple) != 8:
        raise ProductionBlockedError(
            "run_session_training must return the canonical 8-tuple "
            "(rng, rl_train_state, global_update_step, global_env_steps, "
            "training_metrics, num_updates_in_session, categorized_tasks, "
            "evaluation_metrics); got "
            f"{type(receipt_tuple).__name__} of length "
            f"{len(receipt_tuple) if isinstance(receipt_tuple, tuple) else 'n/a'} "
            "(fail closed)")
    (rng, rl_train_state, global_update_step, global_env_steps,
     training_metrics, num_updates_in_session, categorized_tasks,
     evaluation_metrics) = receipt_tuple
    if isinstance(num_updates_in_session, bool) \
            or not isinstance(num_updates_in_session, int) \
            or num_updates_in_session < 0:
        raise ProductionBlockedError(
            "run_session_training returned an invalid "
            f"num_updates_in_session={num_updates_in_session!r} "
            "(fail closed)")
    if isinstance(global_update_step, bool) \
            or not isinstance(global_update_step, int) \
            or global_update_step < 0:
        raise ProductionBlockedError(
            "run_session_training returned an invalid "
            f"global_update_step={global_update_step!r} (fail closed)")
    if not isinstance(training_metrics, Mapping):
        raise ProductionBlockedError(
            "run_session_training returned non-mapping training_metrics "
            f"({type(training_metrics).__name__}); fail closed")
    # BLOCKER-5: the REAL post-session architecture runner memory.  When a
    # backend is bound the RunState MUST carry real memory values — a session
    # that trained an architecture but reports no memory is a lie (fail closed).
    final_memory = None
    if backend is not None:
        try:
            from dicode.training import get_session_final_memory
            final_memory = get_session_final_memory()
        except Exception:  # pragma: no cover - defensive import guard
            final_memory = None
        if final_memory is None:
            raise ProductionBlockedError(
                "ARCHITECTURE_MEMORY_MISSING: a backend was bound for this "
                f"{purpose} session but no final architecture memory was "
                "captured (the RunState must carry REAL post-session memory "
                "values; fail closed)")
    return {
        "rng": rng,
        "rl_train_state": rl_train_state,
        "global_update_step": int(global_update_step),
        "global_env_steps": int(global_env_steps),
        "training_metrics": training_metrics,
        "num_updates_in_session": int(num_updates_in_session),
        "categorized_tasks": categorized_tasks,
        "evaluation_metrics": evaluation_metrics,
        "sampled_task_ids": tuple(sampled_task_ids),
        "architecture_memory": final_memory,
    }


def execute_one_update(runtime: Any, *, context: Any, plan: Any,
                       adapter: Any,
                       backend: Any = None,
                       checkpoint_params: Any = None,
                       initial_memory_dict: Any = None) -> Mapping[str, Any]:
    """Delegate ONE canonical DiCode update (smoke / unit test contract).

    E3-P0-2: this is the single-update runtime used by unit tests, object
    checks and single-update smoke.  ``context.config`` must force
    ``dicode_manager.max_updates_per_session == 1`` (build_hydra_config
    default).  The formal path (100 updates/session) uses ``execute_session``.
    """
    verify_canonical_dicode_one_update_runtime(runtime)
    return _delegate_canonical_session(
        runtime, context=context, plan=plan, adapter=adapter,
        backend=backend, checkpoint_params=checkpoint_params,
        initial_memory_dict=initial_memory_dict,
        purpose="one-update")


def execute_session(runtime: Any, *, context: Any, plan: Any,
                    adapter: Any,
                    backend: Any = None,
                    checkpoint_params: Any = None,
                    initial_memory_dict: Any = None) -> Mapping[str, Any]:
    """Delegate ONE complete native DiCode curriculum session (formal path).

    E3_DICODE_SESSION_ALIGNED_CONSERVATIVE_16x128: one invocation == one
    ``run_session_training`` == ``config.dicode_manager.max_updates_per_session``
    (100) outer updates.  This is NEVER a for-loop of ``execute_one_update``
    calls — DiCode's own ``run_training_session`` runs the full session.

    P0-4: ``initial_memory_dict`` forwards the resumed architecture memory to
    the PPO initial runner state.
    """
    verify_canonical_dicode_session_runtime(runtime)
    receipt = _delegate_canonical_session(
        runtime, context=context, plan=plan, adapter=adapter,
        backend=backend, checkpoint_params=checkpoint_params,
        initial_memory_dict=initial_memory_dict,
        purpose="session")
    expected = int(context.config.dicode_manager.max_updates_per_session)
    if int(receipt["num_updates_in_session"]) != expected:
        raise ProductionBlockedError(
            f"formal session must execute exactly {expected} updates "
            f"(max_updates_per_session), got "
            f"{int(receipt['num_updates_in_session'])} (fail closed)")
    return receipt
