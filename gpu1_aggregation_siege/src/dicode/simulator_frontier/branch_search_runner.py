"""Real actual-N branch search runner (P0-2) with P0-3 wiring.

Executes REAL branches from the EXACT same restored state — Student
deterministic, Student stochastic, stronger Reference — and records only
aggregate feasibility evidence.  Honesty rules enforced structurally:

- ``actual_N`` is ALWAYS the number of branches actually executed: it is the
  length of the returned tuple; no best-of-N extrapolation anywhere.
- No action sequence, successful route, logits, hidden state or Reference
  memory is ever recorded or returned: outcomes carry aggregates only, and the
  runner never persists per-step actions.
- Memory modes are fail closed: SAVED_POLICY_MEMORY requires a real artifact
  (file sha256 recomputed + ``validate_memory`` + spec/identity hashes exactly
  equal to the search Student's); HISTORY_BURN_IN requires a real history
  reference plus a burn-in executor; a missing piece raises
  ``BranchSearchBlockedError`` — never faked, never downgraded silently.
  ZERO_MEMORY is labelled ablation-only and can never pose as a production
  memory mode.

P0-3 wiring: a production run may only start from a joint full-state restore
executed by ``fresh_process_restore.run_fresh_process_restore_production``
(exactly one fresh child process, controller-signed ProductionRegistryBundle,
atomic evidence bound to the single child PID and every component digest).
``require_production_restore_context`` mechanically rejects anything else —
plain Mappings (self-reported contexts), parent-process global-registry
fallbacks and callback self-asserted restores included.  ``run_actual_n``
accepts ONLY a minted ``verified_restore_context.VerifiedRestoreContext`` and
re-binds it to the run before the first branch: state id/hash, capture
checkpoint id, memory spec hash and the search Student identity hash must all
agree with the archive entry and the mounted Student.  Without a verified
context ``run_actual_n`` stays blocked; the blocking interface is explicit,
never papered over.

Per-branch isolation (CC4 follow-up, P0-4, Option B — the verifier-signed
restore context is consumed in-process with per-branch mechanical
verification):

- EVERY branch re-restores its own state from the archive (fresh decode,
  payload hash recomputed) and re-prepares its own memory (artifact re-read /
  burn-in re-execution).  No state object and no memory object is ever shared
  across branches; a mutating adapter cannot let one branch leak into another.
- every branch's start state leaf digest is recomputed and must equal every
  other branch's — any divergence fails the whole run closed; each outcome's
  provenance attests its start digest, executing-policy identity hash, memory
  status and the restore context hash that authorized the run.
- Reference branches NEVER run on Student memory: the REFERENCE_POLICY source
  consumes ONLY the Reference-specific memory surface
  (``reference_memory_artifact`` / ``reference_burn_in_executor``); an unbound
  Reference memory surface blocks the run instead of silently substituting
  Student memory.
- ``actual_N`` counts ONLY completed AND attested branches.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import jax
import numpy as np

from .combined_restore_contract import REQUIRED_COMPONENTS
from .discovery_provenance import DiscoveryProvenance
from .env_restore import flatten_env_state, restore_env_state
from .errors import BranchSearchBlockedError, InvalidEvidenceError, ProvenanceViolationError
from .fresh_process_restore import (
    BLOCKED_WAITING_CONTROLLER_SIGNED_REGISTRY_BUNDLE,
    leaves_digest_of,
    tree_leaf_records,
)
from .memory_modes import MemoryRestoreMode
from .provenance import SearchActionLeakageGuard
from .search_statistics import BranchOutcome
from .state_codec import StateCodec
from .verified_restore_context import (
    RESTORE_CONTEXT_DRIVER,
    VerifiedRestoreContext,
    verify_verified_restore_context,
)
from .student_binding import (
    assert_entry_bound,
    assert_outcome_bound,
    bind_branch_outcome,
    check_bound_entry_memory_request,
)
# Shared contract layer ONLY (jax-free protocol module): the runner depends on
# the StudentAdapter contract, never on any concrete/fake adapter class.
from dicode.student_adapters.protocol import StudentAdapter

RUNNER_VERSION = "branch-search-runner/v1"

SEARCH_SOURCE_STUDENT_DETERMINISTIC = "STUDENT_DETERMINISTIC"
SEARCH_SOURCE_STUDENT_STOCHASTIC = "STUDENT_STOCHASTIC"
SEARCH_SOURCE_REFERENCE_POLICY = "REFERENCE_POLICY"
SEARCH_SOURCES = (
    SEARCH_SOURCE_STUDENT_DETERMINISTIC,
    SEARCH_SOURCE_STUDENT_STOCHASTIC,
    SEARCH_SOURCE_REFERENCE_POLICY,
)

_HEX64 = "^[0-9a-f]{64}$"


def _require_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or not re.match(_HEX64, value):
        raise BranchSearchBlockedError(f"{name} must be a 64-hex sha256, got {value!r} (fail closed)")
    return value


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _file_sha256(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass(frozen=True)
class MemoryArtifactRef:
    """Identity of a real saved-policy-memory artifact (never self-described)."""

    path: str
    sha256: str
    memory_spec_hash: str
    student_identity_hash: str

    def __post_init__(self) -> None:
        if not str(self.path).strip():
            raise BranchSearchBlockedError("memory artifact path is empty (fail closed)")
        _require_sha256("MemoryArtifactRef.sha256", self.sha256)
        _require_sha256("MemoryArtifactRef.memory_spec_hash", self.memory_spec_hash)
        _require_sha256("MemoryArtifactRef.student_identity_hash", self.student_identity_hash)


@dataclass(frozen=True, kw_only=True)
class BranchSearchRunConfig:
    """One actual-N run against one archived state (fail-closed fields)."""

    state_id: str
    horizon: int
    requested_n: int
    memory_mode: MemoryRestoreMode | str
    memory_request: Any  # MemoryRestoreRequest
    success_predicate: Callable[[Mapping[str, Any]], bool]
    progress_fn: Callable[[Mapping[str, Any]], float]
    memory_artifact: MemoryArtifactRef | None = None
    memory_loader: Callable[[MemoryArtifactRef], Any] | None = None
    history_artifact_ref: str = ""
    burn_in_executor: Callable[[Any], Any] | None = None
    # CC4 follow-up (P0-4): the Reference memory surface is SEPARATE from the
    # Student's.  Reference branches consume ONLY these bindings; an unbound
    # Reference surface blocks the run — Student memory is never substituted.
    reference_memory_artifact: MemoryArtifactRef | None = None
    reference_memory_loader: Callable[[MemoryArtifactRef], Any] | None = None
    reference_history_artifact_ref: str = ""
    reference_burn_in_executor: Callable[[Any], Any] | None = None
    epsilon: float = 0.0
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if not str(self.state_id).strip():
            raise BranchSearchBlockedError("BranchSearchRunConfig.state_id is empty (fail closed)")
        if int(self.horizon) <= 0:
            raise BranchSearchBlockedError("BranchSearchRunConfig.horizon must be > 0")
        if int(self.requested_n) <= 0:
            raise BranchSearchBlockedError("BranchSearchRunConfig.requested_n must be > 0")
        try:
            mode = MemoryRestoreMode(str(self.memory_mode))
        except ValueError as exc:
            raise BranchSearchBlockedError(f"unknown memory mode: {self.memory_mode!r}") from exc
        object.__setattr__(self, "memory_mode", mode)
        object.__setattr__(self, "horizon", int(self.horizon))
        object.__setattr__(self, "requested_n", int(self.requested_n))
        if not (0.0 <= float(self.epsilon) <= 1.0) or not math.isfinite(float(self.epsilon)):
            raise BranchSearchBlockedError("epsilon must be a finite number in [0, 1]")
        if not (float(self.temperature) > 0.0) or not math.isfinite(float(self.temperature)):
            raise BranchSearchBlockedError("temperature must be a finite number > 0")
        object.__setattr__(self, "epsilon", float(self.epsilon))
        object.__setattr__(self, "temperature", float(self.temperature))


def require_production_restore_context(context: Any) -> VerifiedRestoreContext:
    """Mechanical P0-2/P0-3 gate: only a MINTED verified context counts.

    The context must be a ``VerifiedRestoreContext`` minted by
    ``verified_restore_context.mint_verified_restore_context`` from
    mechanically verified ``run_fresh_process_restore_production`` evidence.
    Plain Mappings are rejected outright — a self-reported
    ``production_joint_pass`` field can never be production evidence (the
    mint-only constructor does not accept it in the first place).  Anything
    else — no context, parent-process execution, callback self-asserted
    restores, tampered fields — raises ``BranchSearchBlockedError`` (never
    faked, never silently accepted).
    """
    if isinstance(context, Mapping):
        raise BranchSearchBlockedError(
            "plain Mapping restore contexts are rejected: self-reported "
            "production_joint_pass is never production evidence; run_actual_n "
            "requires a minted VerifiedRestoreContext bound to "
            f"{RESTORE_CONTEXT_DRIVER} evidence (fail closed)")
    if not isinstance(context, VerifiedRestoreContext):
        raise BranchSearchBlockedError(
            f"{BLOCKED_WAITING_CONTROLLER_SIGNED_REGISTRY_BUNDLE}: production branch "
            "search requires a minted VerifiedRestoreContext from verified "
            f"fresh-process evidence ({RESTORE_CONTEXT_DRIVER}); none supplied")
    try:
        return verify_verified_restore_context(context)
    except InvalidEvidenceError as exc:
        raise BranchSearchBlockedError(
            f"verified restore context rejected fail closed: {exc}") from exc


def derive_branch_seeds(*, seed_base: int, state_id: str, source: str,
                        branch_index: int) -> tuple[int, int]:
    """Canonical, replayable per-branch seeds (env stream + policy stream)."""
    digest = _canonical_sha256({
        "schema": RUNNER_VERSION,
        "seed_base": int(seed_base),
        "state_id": state_id,
        "search_source": source,
        "branch_index": int(branch_index),
    })
    env_seed = int(digest[:16], 16) % (2 ** 32)
    policy_seed = int(digest[16:32], 16) % (2 ** 32)
    return env_seed, policy_seed


class BranchSearchRunner:
    """Executes real actual-N branch searches from a jointly restored state.

    The runner never rebuilds the Student, never touches reward/action heads,
    and never records per-step actions: it only steps the real env from the
    restored bundle and reduces each branch to aggregate evidence.
    """

    def __init__(self, *, student: Any, student_params: Any, step_fn: Any,
                 env_params: Any, template: Any, observe_fn: Any,
                 capture_student_id: str, search_student_id: str,
                 train_student_id: str, codec: StateCodec | None = None,
                 reference_student: Any = None, reference_params: Any = None):
        if not isinstance(student, StudentAdapter):
            raise BranchSearchBlockedError(
                "search student does not satisfy the StudentAdapter protocol (fail closed)")
        if reference_student is not None and not isinstance(reference_student, StudentAdapter):
            raise BranchSearchBlockedError(
                "reference student does not satisfy the StudentAdapter protocol (fail closed)")
        for label, value in (("capture_student_id", capture_student_id),
                             ("search_student_id", search_student_id),
                             ("train_student_id", train_student_id)):
            if not str(value).strip() or str(value).upper() in {"NONE", "PENDING", "UNKNOWN", "TODO"}:
                raise BranchSearchBlockedError(f"{label} is unbound ({value!r}); bind real identity first")
        self._student = student
        self._student_params = student_params
        self._step_fn = step_fn
        self._env_params = env_params
        self._template = template
        self._observe_fn = observe_fn
        self._codec = codec or StateCodec()
        self._capture_student_id = str(capture_student_id)
        self._search_student_id = str(search_student_id)
        self._train_student_id = str(train_student_id)
        self._reference_student = reference_student
        self._reference_params = reference_params

    # ------------------------------------------------------------------
    # restore
    # ------------------------------------------------------------------

    def restore_entry(self, archive: Any, state_id: str) -> tuple[Any, Any]:
        """Restore one archived entry to a pytree bundle (exact same state)."""
        entry, encoded = archive.get(state_id)
        assert_entry_bound(entry)
        if entry.discovery_provenance != DiscoveryProvenance.TRAINING_DISCOVERY.value:
            raise ProvenanceViolationError(
                f"branch search may only start from TRAINING_DISCOVERY captures, got "
                f"{entry.discovery_provenance!r} ({state_id})")
        bundle = restore_env_state(encoded, self._template, codec=self._codec)
        return entry, bundle

    # ------------------------------------------------------------------
    # memory preparation (fail closed per mode)
    # ------------------------------------------------------------------

    def _prepare_memory(self, entry: Any, bundle: Any,
                        config: BranchSearchRunConfig, *,
                        source: str) -> tuple[Any, str]:
        """Memory for ONE branch of ONE source (CC4 follow-up, P0-4).

        The memory surface is chosen by the EXECUTING policy: Student sources
        consume the Student memory bindings; REFERENCE_POLICY consumes ONLY
        the Reference bindings.  Student memory is NEVER substituted into a
        Reference branch — an unbound Reference surface blocks the run.
        Every call prepares a FRESH memory instance (artifact re-read /
        burn-in re-execution) so branches never share a memory object.
        """
        if source == SEARCH_SOURCE_REFERENCE_POLICY:
            adapter = self._reference_student
            artifact = config.reference_memory_artifact
            loader = config.reference_memory_loader
            history_ref = config.reference_history_artifact_ref
            burn_in = config.reference_burn_in_executor
            who = "Reference"
        else:
            adapter = self._student
            artifact = config.memory_artifact
            loader = config.memory_loader
            history_ref = config.history_artifact_ref
            burn_in = config.burn_in_executor
            who = "Student"
        if adapter is None:
            raise BranchSearchBlockedError(
                f"{source} branch blocked: no executing adapter mounted for the "
                f"{who} memory surface (fail closed)")

        mode = config.memory_mode
        request_mode = MemoryRestoreMode(str(config.memory_request.mode))
        if request_mode is not mode:
            raise BranchSearchBlockedError(
                f"memory mode mismatch: run config carries {mode.value}, restore request "
                f"carries {request_mode.value} (fail closed)")
        report = check_bound_entry_memory_request(entry, config.memory_request)
        if not report.compatible:
            raise BranchSearchBlockedError(
                f"memory compatibility guard rejected the restore request: {tuple(report.reasons)}")

        if mode is MemoryRestoreMode.ZERO_MEMORY:
            # Ablation only: never a production memory mode, always labelled.
            memory = adapter.initial_memory(1)
            check = adapter.validate_memory(memory, 1)
            if not bool(check.get("ok")):
                raise BranchSearchBlockedError(
                    f"ZERO_MEMORY initial memory failed validate_memory: {tuple(check.get('reasons', ()))}")
            return memory, "ZERO_MEMORY_ABLATION_ONLY"

        if mode is MemoryRestoreMode.SAVED_POLICY_MEMORY:
            if artifact is None:
                raise BranchSearchBlockedError(
                    f"SAVED_POLICY_MEMORY_BLOCKED_NO_MEMORY_ARTIFACT: no {who} "
                    "saved-policy-memory artifact is bound for this source (an empty "
                    f"reference is never accepted; Student memory is never substituted "
                    f"into {who} branches)")
            if loader is None:
                raise BranchSearchBlockedError(
                    f"SAVED_POLICY_MEMORY_BLOCKED_NO_MEMORY_LOADER: no {who} memory "
                    "loader was injected (never fabricate memory)")
            actual_sha = _file_sha256(artifact.path)
            if actual_sha != artifact.sha256:
                raise BranchSearchBlockedError(
                    f"memory artifact sha256 mismatch: file recomputes to {actual_sha[:16]}…, "
                    f"ref declares {artifact.sha256[:16]}… (fail closed)")
            spec_hash = adapter.memory_spec().spec_hash()
            if artifact.memory_spec_hash != spec_hash:
                raise BranchSearchBlockedError(
                    f"memory artifact spec hash does not equal the executing {who} "
                    "adapter memory spec hash (cross-policy memory rejected)")
            identity_hash = adapter.identity().identity_hash()
            if artifact.student_identity_hash != identity_hash:
                raise BranchSearchBlockedError(
                    f"memory artifact identity hash does not equal the executing {who} "
                    "adapter identity hash (cross-policy memory rejected)")
            memory = loader(artifact)
            check = adapter.validate_memory(memory, 1)
            if not bool(check.get("ok")):
                raise BranchSearchBlockedError(
                    f"loaded {who} policy memory failed validate_memory: "
                    f"{tuple(check.get('reasons', ()))}")
            return memory, "SAVED_POLICY_MEMORY_VERIFIED"

        if mode is MemoryRestoreMode.HISTORY_BURN_IN:
            if not str(history_ref).strip():
                raise BranchSearchBlockedError(
                    f"HISTORY_BURN_IN_BLOCKED_NO_HISTORY_REFERENCE: no {who} history "
                    f"artifact reference is bound for this source (never guess)")
            if burn_in is None:
                raise BranchSearchBlockedError(
                    f"HISTORY_BURN_IN_BLOCKED_NO_BURN_IN_EXECUTOR: no {who} burn-in "
                    "executor is bound (never fabricate burn-in memory)")
            if bundle.history_reference is None:
                raise BranchSearchBlockedError(
                    "HISTORY_BURN_IN blocked: the archived bundle carries no history_reference")
            memory = burn_in(bundle.history_reference)
            check = adapter.validate_memory(memory, 1)
            if not bool(check.get("ok")):
                raise BranchSearchBlockedError(
                    f"burn-in {who} memory failed validate_memory: "
                    f"{tuple(check.get('reasons', ()))}")
            return memory, "HISTORY_BURN_IN_VERIFIED"

        raise BranchSearchBlockedError(f"unhandled memory mode: {mode!r} (fail closed)")

    # ------------------------------------------------------------------
    # single branch execution
    # ------------------------------------------------------------------

    def _select_action(self, *, source: str, student: Any, params: Any, obs: Any,
                       memory: Any, prev_action: Any, prev_reward: Any,
                       gen: np.random.Generator, step_rng: Any,
                       config: BranchSearchRunConfig) -> tuple[int, Any]:
        """Choose the branch action; NEVER retains the action anywhere."""
        deterministic = source is not SEARCH_SOURCE_STUDENT_STOCHASTIC
        step_out = student.policy_step(params, obs, memory, prev_action, prev_reward,
                                       step_rng, deterministic)
        new_memory = step_out.get("new_memory", step_out.get("memory"))
        action = int(np.asarray(step_out["action"]).reshape(-1)[0])
        if deterministic:
            return action, new_memory
        # Student stochastic: epsilon-greedy override + optional temperature.
        if float(gen.random()) < config.epsilon:
            action = int(gen.integers(0, self._student.action_spec().count))
            return action, new_memory
        if config.temperature != 1.0:
            if "logits" not in step_out:
                raise BranchSearchBlockedError(
                    "temperature != 1.0 requires logits from the adapter; none returned "
                    "(fail closed rather than silently sampling at T=1)")
            logits = np.asarray(step_out["logits"], dtype=np.float64).reshape(-1)
            n_actions = self._student.action_spec().count
            if logits.shape[0] != n_actions:
                raise BranchSearchBlockedError(
                    f"logits length {logits.shape[0]} != action count {n_actions} (fail closed)")
            scaled = logits / float(config.temperature)
            scaled -= float(np.max(scaled))
            weights = np.exp(scaled)
            probs = weights / float(np.sum(weights))
            action = int(gen.choice(n_actions, p=probs))
        return action, new_memory

    def run_branch(self, restored: Any, *, config: BranchSearchRunConfig, source: str,
                   branch_id: str, seed_base: int, branch_index: int, env_seed: int,
                   policy_seed: int, memory: Any, memory_status: str,
                   start_digest: str, policy_identity_hash: str,
                   context_hash: str) -> BranchOutcome:
        """Execute ONE real branch from ITS OWN restored state (bare env steps).

        Bare stepping (no AutoResetEnvWrapper) means terminal-before-autoreset
        holds by construction: the branch stops at the real terminal state.
        CC4 follow-up (P0-4): the outcome attests the per-branch start-state
        digest, the executing policy's identity hash, the memory status and
        the restore context hash — only completed, attested branches count.
        """
        if source not in SEARCH_SOURCES:
            raise BranchSearchBlockedError(f"unknown search source: {source!r}")
        if source == SEARCH_SOURCE_REFERENCE_POLICY:
            if self._reference_student is None or self._reference_params is None:
                raise BranchSearchBlockedError(
                    "REFERENCE_POLICY branch requested but no reference student/params mounted")
            student, params = self._reference_student, self._reference_params
        else:
            student, params = self._student, self._student_params
        if source != SEARCH_SOURCE_STUDENT_STOCHASTIC and (
                config.epsilon != 0.0 or config.temperature != 1.0):
            raise BranchSearchBlockedError(
                f"{source} branches must run with epsilon=0 and temperature=1 (fail closed)")
        if self._observe_fn is None:
            raise BranchSearchBlockedError("runner has no observe_fn; cannot derive observations")

        gen = np.random.default_rng(policy_seed)
        runner_key = jax.random.PRNGKey(env_seed)
        state = restored.env_state
        obs = self._observe_fn(state)
        prev_action = restored.previous_action
        prev_reward = restored.previous_reward
        branch_memory = memory
        transitions_used = 0
        terminal_event: str | None = None
        for _ in range(config.horizon):
            policy_seed_step = int(gen.integers(0, 2 ** 31))
            step_rng = jax.random.PRNGKey(policy_seed_step)
            action, branch_memory = self._select_action(
                source=source, student=student, params=params, obs=obs, memory=branch_memory,
                prev_action=prev_action, prev_reward=prev_reward, gen=gen,
                step_rng=step_rng, config=config)
            runner_key, step_key = jax.random.split(runner_key)
            obs, state, reward, done, _info = self._step_fn(step_key, state, int(action),
                                                            self._env_params)
            prev_action = int(action)
            prev_reward = float(np.asarray(reward))
            transitions_used += 1
            if bool(np.asarray(done)):
                terminal_event = "ENV_TERMINAL"
                break

        final_flat = flatten_env_state(state)
        success = bool(config.success_predicate(final_flat))
        progress = float(config.progress_fn(final_flat))
        if not math.isfinite(progress):
            raise BranchSearchBlockedError("progress_fn returned a non-finite value (fail closed)")
        if success:
            failure_category: str | None = None
        elif terminal_event is not None:
            failure_category = "TERMINAL_BEFORE_SUCCESS"
        else:
            failure_category = "HORIZON_EXHAUSTED"

        _require_sha256("branch attestation start_digest", start_digest)
        _require_sha256("branch attestation policy_identity_hash", policy_identity_hash)
        _require_sha256("branch attestation context_hash", context_hash)
        aggregate = {
            "branch_id": branch_id,
            "state_id": config.state_id,
            "search_source": source,
            "rng_seed": env_seed,
            "horizon": config.horizon,
            "transitions_used": transitions_used,
            "success": success,
            "progress": progress,
            "terminal_event": terminal_event,
            "failure_category": failure_category,
            "memory_mode": config.memory_mode.value,
        }
        outcome_hash = _canonical_sha256({"schema": RUNNER_VERSION, "aggregate": aggregate})
        provenance = {
            "schema": RUNNER_VERSION,
            "requested_n": config.requested_n,
            "actual_n_context": {
                "search_source": source,
                "branch_index": branch_index,
                "requested_n": config.requested_n,
            },
            "seed_base": int(seed_base),
            "env_seed": env_seed,
            "policy_seed": policy_seed,
            "epsilon": config.epsilon,
            "temperature": config.temperature,
            "memory_status": memory_status,
            # CC4 follow-up (P0-4): mechanical attestation that THIS branch
            # completed from ITS OWN isolated state + memory under the minted
            # restore context.  actual_N counts only attested branches.
            "branch_attestation": {
                "completed": True,
                "start_state_digest": start_digest,
                "policy_identity_hash": policy_identity_hash,
                "memory_status": memory_status,
                "restore_context_hash": context_hash,
            },
        }
        SearchActionLeakageGuard.validate_aggregate(
            {"provenance": provenance, "aggregate": aggregate})
        outcome = BranchOutcome(
            branch_id=branch_id,
            state_id=config.state_id,
            search_source=source,
            rng_seed=env_seed,
            horizon=config.horizon,
            transitions_used=transitions_used,
            success=success,
            progress=progress,
            terminal_event=terminal_event,
            failure_category=failure_category,
            memory_mode=config.memory_mode.value,
            outcome_hash=outcome_hash,
            provenance=provenance,
        )
        outcome = bind_branch_outcome(
            outcome,
            capture_student_id=self._capture_student_id,
            search_student_id=self._search_student_id,
            train_student_id=self._train_student_id,
            memory_compatibility_status=memory_status,
        )
        assert_outcome_bound(outcome)
        return outcome

    # ------------------------------------------------------------------
    # actual-N orchestration
    # ------------------------------------------------------------------

    def run_actual_n(self, archive: Any, config: BranchSearchRunConfig, *,
                     seed_base: int, sources: Sequence[str] = SEARCH_SOURCES,
                     restore_context: Any = None) -> tuple[BranchOutcome, ...]:
        """Execute requested_n REAL branches; actual_N == len(returned tuple).

        Sources rotate in fixed order.  Every precondition is verified BEFORE
        the first branch executes (restore context, source availability,
        memory mode) so a run either honestly completes requested_n branches
        or raises ``BranchSearchBlockedError`` — it never reports a partial
        run as complete.
        """
        context = require_production_restore_context(restore_context)
        source_tuple = tuple(sources)
        if not source_tuple:
            raise BranchSearchBlockedError("run_actual_n requires at least one search source")
        for source in source_tuple:
            if source not in SEARCH_SOURCES:
                raise BranchSearchBlockedError(f"unknown search source: {source!r}")
        if SEARCH_SOURCE_REFERENCE_POLICY in source_tuple and (
                self._reference_student is None or self._reference_params is None):
            raise BranchSearchBlockedError(
                "REFERENCE_POLICY requested but no reference student/params mounted (fail closed)")

        # Context <-> run/state binding (P0-2): the minted context must name
        # EXACTLY the state, checkpoint, memory binding and Student this run
        # consumes.  Any mismatch is a wrong-state/wrong-checkpoint attempt.
        if context.state_id != config.state_id:
            raise BranchSearchBlockedError(
                f"restore context state_id {context.state_id!r} != run config state_id "
                f"{config.state_id!r} (search must start from the captured state)")
        bound_entry, bound_encoded = archive.get(config.state_id)
        if context.state_hash != bound_encoded.payload_hash \
                or context.state_hash != bound_entry.state_hash:
            raise BranchSearchBlockedError(
                "restore context state_hash does not equal the archive-encoded state "
                "hash for this run (wrong-state search rejected fail closed)")
        if context.source_checkpoint_id != bound_entry.source_checkpoint_id:
            raise BranchSearchBlockedError(
                f"restore context checkpoint {context.source_checkpoint_id!r} != capture "
                f"checkpoint {bound_entry.source_checkpoint_id!r} (wrong-checkpoint "
                "search rejected fail closed)")
        if context.source_memory_spec_hash != bound_entry.source_memory_spec_hash:
            raise BranchSearchBlockedError(
                "restore context memory spec hash does not equal the capture entry "
                "memory binding (memory-substitution search rejected fail closed)")
        if set(context.component_digests) != set(REQUIRED_COMPONENTS):
            raise BranchSearchBlockedError(
                "restore context must bind EXACTLY the nine required restored "
                f"components {tuple(REQUIRED_COMPONENTS)}")
        runner_identity_hash = self._student.identity().identity_hash()
        if context.student_identity_hash != runner_identity_hash:
            raise BranchSearchBlockedError(
                "restore context student identity hash does not equal the mounted "
                "search Student identity (identity-substitution search rejected "
                "fail closed)")

        outcomes: list[BranchOutcome] = []
        branch_index = 0
        first_start_digest: str | None = None
        while len(outcomes) < config.requested_n:
            source = source_tuple[branch_index % len(source_tuple)]
            # CC4 follow-up (P0-4): EVERY branch re-restores its OWN state
            # (fresh decode, payload hash recomputed) and prepares its OWN
            # memory — no state object or memory object crosses branches.
            entry, restored = self.restore_entry(archive, config.state_id)
            if entry.state_hash != context.state_hash:
                raise BranchSearchBlockedError(
                    f"branch {branch_index}: per-branch restore drifted from the "
                    "verified restore context state hash (wrong-state branch start "
                    "rejected fail closed)")
            start_digest = leaves_digest_of(
                tree_leaf_records(flatten_env_state(restored.env_state)))
            if first_start_digest is None:
                first_start_digest = start_digest
            elif start_digest != first_start_digest:
                raise BranchSearchBlockedError(
                    f"branch {branch_index}: per-branch state isolation violated — "
                    f"start digest {start_digest[:16]}… differs from the first "
                    f"branch {first_start_digest[:16]}… (all branches must start "
                    "from the identical restored state; fail closed)")
            memory, memory_status = self._prepare_memory(entry, restored, config,
                                                         source=source)
            executing = self._reference_student \
                if source == SEARCH_SOURCE_REFERENCE_POLICY else self._student
            policy_identity_hash = executing.identity().identity_hash()
            env_seed, policy_seed = derive_branch_seeds(
                seed_base=seed_base, state_id=config.state_id, source=source,
                branch_index=branch_index)
            branch_id = f"{config.state_id}:{source}:{branch_index:04d}"
            outcome = self.run_branch(
                restored, config=config, source=source, branch_id=branch_id,
                seed_base=seed_base, branch_index=branch_index, env_seed=env_seed,
                policy_seed=policy_seed, memory=memory, memory_status=memory_status,
                start_digest=start_digest, policy_identity_hash=policy_identity_hash,
                context_hash=context.context_hash)
            # Attestation gate: only COMPLETED AND ATTESTED branches count
            # toward actual_N — anything less fails the whole run closed.
            attestation = outcome.provenance.get("branch_attestation", {})
            if not (attestation.get("completed") is True
                    and attestation.get("start_state_digest") == start_digest
                    and attestation.get("policy_identity_hash") == policy_identity_hash
                    and attestation.get("memory_status") == memory_status
                    and attestation.get("restore_context_hash") == context.context_hash
                    and str(outcome.memory_compatibility_status) == memory_status):
                raise BranchSearchBlockedError(
                    f"branch {branch_index} outcome attestation is incomplete or "
                    "inconsistent (unattested branches never count toward actual_N; "
                    "fail closed)")
            outcomes.append(outcome)
            branch_index += 1
        return tuple(outcomes)


def actual_n_summary(outcomes: Sequence[BranchOutcome]) -> Mapping[str, Any]:
    """Aggregate summary of a real run: actual_N is the executed branch count."""
    rows = list(outcomes)
    by_source: dict[str, int] = {}
    for row in rows:
        by_source[row.search_source] = by_source.get(row.search_source, 0) + 1
    requested = int(rows[0].provenance.get("requested_n", -1)) if rows else 0
    return {
        "actual_n": len(rows),
        "requested_n": requested,
        "actual_equals_requested": bool(rows) and len(rows) == requested,
        "by_source": by_source,
        "successes": sum(bool(r.success) for r in rows),
        "runner_version": RUNNER_VERSION,
    }
