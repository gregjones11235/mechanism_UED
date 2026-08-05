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
import inspect
import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Mapping

from .errors import InvalidEvidenceError, ProductionBlockedError

CANONICAL_DICODE_RUNTIME_VERSION = "canonical-dicode-one-update-runtime/v1"
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


def compile_canonical_15_plus_1(*, plan_id: Any, distributions: Any,
                                anchor_ids: Any, env_adapter_id: Any,
                                memory_bindings: Any,
                                anchor_memory_binding: Any = None
                                ) -> CanonicalDiCodeTrainingBatchPlan:
    """Compile the 15+1 curriculum plan from 12 dynamic distributions + 4 anchors.

    * Exactly 12 dynamic ``FrontierDistribution`` objects (unique ids) and
      exactly 4 anchor ids are required.
    * The curriculum slots are the 12 dynamic ids + the FIRST 3 anchor ids
      (non-target standard-reset anchors) = 15.
    * The FOURTH anchor is the semantic counterpart of DiCode's
      ORIGINAL_TASK: it is NOT duplicated as a regular distribution — it is
      mapped to the single OriginalTask slot at proportion 0.20.  The 15
      curriculum slots share the remaining 0.80 (equal share per slot).
    * ``compose_12_plus_4`` may keep its scientific-report semantics; this
      conversion is what enters the training runtime.
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
    anchors = tuple(str(a) for a in anchor_ids)
    if len(anchors) != 4 or len(set(anchors)) != 4:
        raise InvalidEvidenceError("exactly 4 unique anchor ids are required")
    curriculum_slots = tuple(d.distribution_id for d in dists) + anchors[:3]
    if len(curriculum_slots) != CURRICULUM_SLOT_COUNT:
        raise InvalidEvidenceError(
            f"expected {CURRICULUM_SLOT_COUNT} curriculum slots, got "
            f"{len(curriculum_slots)}")
    per_slot = CURRICULUM_PROPORTION_TOTAL / CURRICULUM_SLOT_COUNT
    curriculum_weights = {slot: per_slot for slot in curriculum_slots}
    slot_distributions = {d.distribution_id: dict(asdict(d))
                          for d in dists}
    for anchor in anchors[:3]:
        slot_distributions[anchor] = {"anchor_id": anchor,
                                      "kind": "NON_TARGET_STANDARD_RESET_ANCHOR"}
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


def execute_one_update(runtime: Any, *, plan: Any, adapter: Any, params: Any,
                       run_state: Mapping[str, Any] | None,
                       budget: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Delegate ONE update to the bound DiCode run_session_training chain.

    Direction 三 NEVER updates params itself: the update is executed by the
    director-shared DiCode runtime (original PPO-GTrXL).  The returned
    receipt must be a mapping carrying new ``params``; direction 三 only
    verifies the contract-level invariants (a real params change, no
    self-invented counters).
    """
    verify_canonical_dicode_one_update_runtime(runtime)
    if not isinstance(plan, CanonicalDiCodeTrainingBatchPlan):
        raise InvalidEvidenceError(
            "execute_one_update requires a minted CanonicalDiCodeTrainingBatchPlan")
    if not isinstance(adapter, FrontierDistributionEnvironmentAdapter):
        raise InvalidEvidenceError(
            "execute_one_update requires a minted "
            "FrontierDistributionEnvironmentAdapter")
    session = _import_entrypoint(runtime.run_session_training_entrypoint,
                                 "run_session_training")
    try:
        receipt = session(plan, adapter, params, run_state, budget)
    except Exception as exc:
        raise ProductionBlockedError(
            f"CanonicalDiCodeOneUpdateRuntime rejected the batch plan: {exc!r}") from exc
    if not isinstance(receipt, Mapping) or "params" not in receipt:
        raise ProductionBlockedError(
            "CanonicalDiCodeOneUpdateRuntime receipt must be a mapping carrying "
            "'params' (fail closed)")
    return receipt
