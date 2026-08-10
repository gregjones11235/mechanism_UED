#!/usr/bin/env python3
"""E3 real frontier-capsule actual-N (P0-1/2/3 + audit-hardened predicates).

Implements the REAL same-state actual-N for the E3 formal longrun:

  P0-1  capture ONE complete frontier capsule from a real Student rollout:
        env state, observation, entering architecture memory, previous
        action/reward, Student params identity, task identity, capture RNG.
        Every branch restores from that SAME capsule, changing ONLY the branch
        RNG.  NO per-branch env.reset_env(), NO per-branch initial_memory().
  P0-2  success is decided by a TASK-BASED predicate derived from the concrete
        Task class (is_success == all relevant achievements done), NEVER from
        fake flattened leaves (gate_progress / floor_number / health /
        max_health do not exist on the real EnvState).  death / timeout /
        plain done is NOT auto success.  terminal_event, failure_category and
        the RAW success basis (per-achievement done flags) are recorded.
  P0-3  no fake fixed state_id; state/capsule hash comes from the REAL
        serialized bytes (encode_env_state payload_hash).

Audit-hardening (sole-controller 2026-08-10 directive):
  * success/progress are built per Task class from is_success() /
    relevant_achievements (frozen achievement predicate).  A predicate that
    cannot be constructed for the task, or a state missing the required real
    fields (achievements / player_health / player_level / timestep), FAILS
    CLOSED — never a silent default of 0.
  * predicate applicability is verified by constructing ONE positive example
    (all relevant achievements done -> success) and ONE negative example
    (none done -> not success) from a real state; failure to distinguish them
    is a hard error.
  * actual-N branches are checked for non-degeneracy: every branch executes
    >= 1 transition, branch RNG seeds are pairwise distinct, and progress is
    finite in [0,1] computed from real achievement bytes.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _params_hash(params) -> str:
    leaves = jax.tree_util.tree_leaves(params)
    digest = hashlib.sha256()
    for leaf in leaves:
        digest.update(jnp.asarray(leaf).astype(jnp.float32).tobytes())
    return digest.hexdigest()


def _slice_batch_memory(memory: Mapping[str, Any], index: int) -> dict:
    """Slice a batch>=1 architecture memory dict to a single env (index)."""
    import numpy as _np
    out = {}
    for k, v in memory.items():
        arr = _np.asarray(v)
        if arr.ndim >= 1 and arr.shape[0] >= 1:
            out[k] = arr[index:index + 1]
        else:
            out[k] = arr
    return out


def _resolve_branch_memory(saved_memory, memory_mode: str,
                           initial_memory_factory):
    """Resolve branch memory without ever disguising a fresh reset as saved."""
    if memory_mode == "SAVED_POLICY_MEMORY":
        if saved_memory is None:
            raise RuntimeError(
                "SAVED_POLICY_MEMORY requested but the restored capsule "
                "contains no policy_memory (fail closed; never silently "
                "initialize fresh memory)")
        return saved_memory
    if memory_mode == "ZERO_MEMORY_ABLATION":
        return initial_memory_factory(1)
    raise RuntimeError(
        f"unsupported actual-N memory_mode {memory_mode!r}; only "
        "SAVED_POLICY_MEMORY or explicit ZERO_MEMORY_ABLATION are accepted "
        "(fail closed)")


def _build_multitask_setup(*, max_timesteps: int, reset_seed: int) -> dict:
    """Build the MultiTaskMiniCraftaxEnv setup with 8335-dim observations
    (8268 symbolic + 67 task embedding) — the SAME env the RMT16 / SlowGRU
    Student adapters expect (obs_dim 8335).

    Exposes ``task`` (the initialized concrete Task the capsule uses) so
    the success/progress predicates are derived from the REAL task interface.
    """
    import jax
    import jax.numpy as jnp
    from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
    from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv
    from minicraftax.tasks.seed_tasks import survive

    sp = StaticEnvParams()
    env_params = EnvParams(max_timesteps=max_timesteps)
    env = MultiTaskMiniCraftaxEnv(
        task_classes=[survive.Env], static_env_params=sp, params=env_params,
        condition_on_task=True, conditioning_type="embedding",
        embedding_size=67)
    task_embeddings = jnp.zeros((1, 67))
    step_fn = jax.jit(env.step_env)
    obs, state = env.reset_env(jax.random.PRNGKey(reset_seed), env_params,
                               0, task_embeddings)
    return {
        "env": env,
        "params": env_params,
        "step_fn": step_fn,
        "task_embeddings": task_embeddings,
        "state0": state,
        "obs0": obs,
        "max_timesteps": max_timesteps,
        "reset_seed": reset_seed,
        # relevant_achievements is populated on the instantiated task, not on
        # the class.  Capture and verification must use this exact env-owned
        # instance so their task interface cannot diverge.
        "task": env.tasks[0],
    }


# ---------------------------------------------------------------------------
# Task-based success / progress predicates (fail closed, real fields only)
# ---------------------------------------------------------------------------

def task_relevant_achievement_indices(task) -> list[int]:
    """Extract the relevant achievement indices for a concrete Task class.

    Fails closed if the task does not expose the real interface
    (is_success / relevant_achievements with .value indices).  There is no
    default; a missing interface is a hard error.
    """
    if isinstance(task, type):
        raise RuntimeError(
            f"task {task!r} is a class, not the initialized task instance "
            "owned by the environment (fail closed)")
    if not hasattr(task, "is_success"):
        raise RuntimeError(
            f"task {task!r} lacks is_success(state) — cannot build a "
            "task success predicate (fail closed)")
    rel = getattr(task, "relevant_achievements", None)
    if not rel or not isinstance(rel, (list, tuple)):
        raise RuntimeError(
            f"task {task!r} lacks a non-empty relevant_achievements "
            "list — cannot build a task success predicate (fail closed)")
    indices: list[int] = []
    for ach in rel:
        idx = getattr(ach, "value", None)
        if idx is None:
            raise RuntimeError(
                f"achievement {ach!r} lacks .value (fail closed)")
        indices.append(int(idx))
    if len(set(indices)) != len(indices):
        raise RuntimeError(
            "duplicate relevant achievement indices (fail closed)")
    return indices


def _read_achievements(state) -> np.ndarray:
    """Read the REAL state.achievements array; fail closed if absent."""
    ach = getattr(state, "achievements", None)
    if ach is None:
        raise RuntimeError(
            "state lacks the real field 'achievements' (fail closed)")
    try:
        return np.asarray(ach)
    except Exception as exc:
        raise RuntimeError(
            f"cannot read state.achievements: {exc!r} (fail closed)") from exc


def build_task_success_predicate(task) -> tuple[Callable[[Any], bool], dict]:
    """Build a success predicate from the Task class's is_success().

    success == all relevant achievements done (the task's true objective).
    Returns (predicate(state)->bool, meta) where meta records the frozen
    achievement indices + task identity (the RAW success basis definition).
    """
    indices = task_relevant_achievement_indices(task)
    task_name = type(task).__name__

    def _pred(state) -> bool:
        ach = _read_achievements(state)
        try:
            done = [bool(ach[i]) for i in indices]
        except IndexError as exc:
            raise RuntimeError(
                f"achievement index out of range for task {task_name}: {exc!r} "
                "(fail closed)") from exc
        return all(done)

    meta = {
        "predicate_kind": "TASK_ACHIEVEMENT_ALL_RELEVANT_DONE",
        "task": task_name,
        "achievement_indices": indices,
        "success_threshold": None,   # achievement predicate is threshold-free
        "basis": "task.is_success(state): all relevant achievements done",
    }
    return _pred, meta


def build_task_progress_fn(task) -> Callable[[Any], float]:
    """Progress = fraction of relevant achievements completed (0..1).

    Computed from the REAL state.achievements bytes at the relevant indices.
    Fails closed on missing fields / non-finite results.
    """
    indices = task_relevant_achievement_indices(task)
    task_name = type(task).__name__
    n = float(len(indices))

    def _prog(state) -> float:
        ach = _read_achievements(state)
        try:
            done = [bool(ach[i]) for i in indices]
        except IndexError as exc:
            raise RuntimeError(
                f"achievement index out of range for task {task_name}: {exc!r} "
                "(fail closed)") from exc
        p = float(sum(done)) / n
        if not np.isfinite(p) or not (0.0 <= p <= 1.0):
            raise RuntimeError(
                f"progress {p} for task {task_name} not finite in [0,1] "
                "(fail closed)")
        return p

    return _prog


def record_success_basis(state, indices: list[int], task_name: str) -> dict:
    """Raw success basis: per-achievement done flags + counts.

    This is the ORIGINAL evidence the success/progress values derive from.
    """
    ach = _read_achievements(state)
    per: dict[str, bool] = {}
    for i in indices:
        try:
            per[str(i)] = bool(ach[i])
        except IndexError as exc:
            raise RuntimeError(
                f"achievement index {i} out of range for task {task_name} "
                f"(fail closed)") from exc
    return {
        "task": task_name,
        "achievement_indices": indices,
        "achievements_done": per,
        "achievements_completed": sum(per.values()),
        "achievements_total": len(indices),
    }


def verify_predicate_applicability(state, pred, indices: list[int],
                                   task_name: str) -> dict:
    """Predicate applicability: construct ONE positive and ONE negative
    example from a real state and require the predicate to distinguish them.

    positive: every relevant achievement set to True  -> must be success.
    negative: every relevant achievement set to False -> must NOT be success.
    Any failure is a hard error (the predicate is degenerate or the state is
    not usable).
    """
    if not dataclasses.is_dataclass(state):
        raise RuntimeError(
            "cannot construct predicate examples: state is not a replaceable "
            "dataclass (fail closed)")
    try:
        pos_ach = _read_achievements(state).copy()
        neg_ach = _read_achievements(state).copy()
    except Exception:
        raise
    for i in indices:
        pos_ach[i] = True
        neg_ach[i] = False
    pos_state = dataclasses.replace(state, achievements=jnp.asarray(pos_ach))
    neg_state = dataclasses.replace(state, achievements=jnp.asarray(neg_ach))
    pos_ok = bool(pred(pos_state))
    neg_ok = not bool(pred(neg_state))
    if not (pos_ok and neg_ok):
        raise RuntimeError(
            f"task {task_name} predicate degeneracy: positive_example={pos_ok} "
            f"negative_example={neg_ok} — the predicate does not distinguish "
            "the real task objective (fail closed)")
    return {
        "task": task_name,
        "positive_example_success": pos_ok,
        "negative_example_not_success": neg_ok,
        "applicable": True,
    }


def build_state_facts(state, indices: list[int], task_name: str) -> dict:
    """Real EnvState facts (fail closed on missing fields).

    Only REAL fields are read: player_level, player_health, achievements,
    timestep.  gate_progress / floor_number / health / max_health do NOT
    exist on the real EnvState and are never read.
    """
    facts: dict[str, Any] = {}
    for field in ("player_level", "player_health", "timestep"):
        if not hasattr(state, field):
            raise RuntimeError(
                f"state lacks the real field {field!r} (fail closed)")
        try:
            facts[field] = float(np.asarray(getattr(state, field)))
        except Exception as exc:
            raise RuntimeError(
                f"cannot read real field {field!r}: {exc!r} (fail closed)") from exc
    # health band: coarse, from the REAL player_health against the frozen
    # Craftax default reference (player_health at episode start is 9.0).
    ph = facts["player_health"]
    ref = 9.0
    ratio = ph / ref if ref not in (0, None) else None
    facts["health_band"] = _band(ratio, 0.34, 0.67)
    basis = record_success_basis(state, indices, task_name)
    facts["achievement_snapshot"] = {
        "relevant_done": basis["achievements_completed"],
        "relevant_total": basis["achievements_total"],
        "per_achievement": basis["achievements_done"],
    }
    facts["threat_band"] = "UNMEASURED"
    facts["resource_band"] = "UNMEASURED"
    facts["inventory_stage"] = "UNMEASURED"
    return facts


# ---------------------------------------------------------------------------
# capture + actual-N
# ---------------------------------------------------------------------------

def capture_frontier_capsule(*, student, student_params, run_id: str,
                             reset_seed: int, capture_at_step: int,
                             max_timesteps: int, success_threshold: float,
                             memory_mode: str = "SAVED_POLICY_MEMORY",
                             initial_memory=None) -> dict:
    """P0-1: capture ONE real frontier capsule from a real Student rollout.

    Returns a dict with archive, entry, encoded, bundle, template, setup,
    state_id, facts, the live capture memory, and the TASK-BASED success /
    progress predicate (bound to the capsule so actual-N reuses them).

    ``success_threshold`` is retained for API compatibility; the task
    achievement predicate is threshold-free.
    """
    from dicode.simulator_frontier.env_restore import (
        build_template, encode_env_state,
    )
    from dicode.simulator_frontier.archive_schema import FrontierArchiveEntry
    from dicode.simulator_frontier.student_binding import bind_capture_entry
    from dicode.simulator_frontier.frontier_archive import FrontierArchive
    from dicode.simulator_frontier.discovery_provenance import (
        DiscoveryProvenance,
    )

    # P0-1: use the SAME 8335-dim multitask env the Student adapters expect.
    setup = _build_multitask_setup(max_timesteps=max_timesteps,
                                   reset_seed=reset_seed)
    env = setup["env"]
    params_env = setup["params"]
    task_embeddings = setup["task_embeddings"]
    observe_fn = getattr(env, "get_obs", None)
    if observe_fn is None:
        raise RuntimeError("capture: env exposes no get_obs(state)")
    task = setup["task"]
    task_name = type(task).__name__

    identity = student.identity()
    params_sha = _params_hash(student_params)
    memory_spec_hash = student.memory_spec().spec_hash()
    capture_student_id = str(identity.candidate_id)

    runner_key = jax.random.PRNGKey(reset_seed + 1)
    state = setup["state0"]
    obs = observe_fn(state, task_embeddings)
    if initial_memory is not None:
        memory = _slice_batch_memory(initial_memory, 0)
    else:
        memory = student.initial_memory(1)
    prev_action, prev_reward = 0, 0.0
    done = False
    steps_executed = 0
    for _ in range(int(capture_at_step)):
        obs_batch = np.asarray(obs).reshape(1, -1)
        out = student.policy_step(student_params, obs_batch, memory,
                                  prev_action, prev_reward, None, True)
        memory = out.get("new_memory", out.get("memory"))
        action = int(np.asarray(out["action"]).reshape(-1)[0])
        runner_key, step_key = jax.random.split(runner_key)
        obs, state, reward, done, _info = setup["step_fn"](
            step_key, state, action, params_env, task_embeddings)
        prev_action, prev_reward = action, float(np.asarray(reward))
        steps_executed += 1
        if bool(np.asarray(done)):
            break
    terminal_before_capture = bool(np.asarray(done))
    if terminal_before_capture:
        raise RuntimeError(
            f"capture refused: rollout reached terminal after {steps_executed} "
            f"steps before capture point {int(capture_at_step)} (a terminal "
            "state is never a frontier branch point)")

    # Task-based predicate (frozen achievement semantics), validated.
    pred, pred_meta = build_task_success_predicate(task)
    prog = build_task_progress_fn(task)
    indices = pred_meta["achievement_indices"]
    applicability = verify_predicate_applicability(state, pred, indices,
                                                   task_name)
    facts = build_state_facts(state, indices, task_name)
    facts["terminal"] = terminal_before_capture
    facts["predicate"] = pred_meta
    facts["predicate_applicability"] = applicability
    success_at_capture = bool(pred(state))
    progress_at_capture = float(prog(state))
    raw_basis = record_success_basis(state, indices, task_name)

    # Encode the REAL captured state (P0-3: state_id = real payload hash).
    encoded, bundle = encode_env_state(
        state, next_step_key=runner_key,
        previous_action=prev_action, previous_reward=prev_reward,
        policy_memory=memory, history_reference=None,
    )
    state_id = encoded.payload_hash

    entry = FrontierArchiveEntry(
        state_id=state_id,
        source_checkpoint_id=str(identity.params_sha256),
        source_episode_id=f"{run_id}:standard-reset:{reset_seed}",
        source_seed=int(reset_seed),
        source_timestep=int(steps_executed),
        capture_reason="E3_WINDOW_STANDARD_RESET_CAPTURE",
        floor=int(facts["player_level"]),
        gate_progress=progress_at_capture,
        health_band=facts["health_band"],
        threat_band=facts["threat_band"],
        resource_band=facts["resource_band"],
        inventory_stage=facts["inventory_stage"],
        achievement_snapshot=facts["achievement_snapshot"],
        terminal=facts["terminal"],
        memory_mode=str(memory_mode),
        encoded_state_ref=encoded.payload_hash,
        state_hash=encoded.payload_hash,
        provenance_hash="",
        created_at=f"{run_id}:window-capture",
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
    archive.add(entry, encoded)  # non-production local add (P0-6 gates are separate)
    # P0-1: the template MUST be built from the CAPTURE-point state (same
    # lineage as the restored branch states) — reset state0 uses Python scalars
    # for is_resting/is_sleeping/task_id while post-step states use 0-dim arrays,
    # so a state0 template would fail restore's treedef fingerprint check.
    template = build_template(state)

    return {
        "archive": archive,
        "entry": entry,
        "encoded": encoded,
        "bundle": bundle,
        "template": template,
        "setup": setup,
        "state_id": state_id,
        "facts": facts,
        "memory": memory,
        "identity": identity,
        "params_sha": params_sha,
        "memory_spec_hash": memory_spec_hash,
        "capture_student_id": capture_student_id,
        "success_at_capture": success_at_capture,
        "progress_at_capture": progress_at_capture,
        "success_basis": raw_basis,
        "predicate": pred,
        "progress_fn": prog,
        "predicate_meta": pred_meta,
        "task": task,
        "task_name": task_name,
        "steps_executed": steps_executed,
        "student": student,
        "student_params": student_params,
    }


def run_same_state_actual_n(*, capsule: dict, n: int, horizon: int,
                            seed_base: int,
                            success_predicate=None,
                            progress_fn=None,
                            memory_mode: str = "SAVED_POLICY_MEMORY") -> dict:
    """P0-1/2/3: N branches from the SAME capsule, only branch RNG differs.

    Each branch re-restores the capsule (restore_env_state) — never a fresh
    env reset, never initial_memory per branch.  success uses the TASK-BASED
    predicate bound at capture time (or an explicit override).  Non-degeneracy
    is enforced: every branch executes >= 1 transition, branch seeds are
    pairwise distinct, and progress is finite in [0,1] from real bytes.
    Returns outcomes + feasibility estimate.
    """
    from dicode.simulator_frontier.env_restore import restore_env_state
    from dicode.simulator_frontier.branch_search_runner import derive_branch_seeds
    from dicode.simulator_frontier.search_statistics import (
        BranchOutcome, estimate_feasibility,
    )
    from dicode.simulator_frontier.discovery_provenance import (
        DiscoveryProvenance,
    )

    pred = success_predicate or capsule.get("predicate")
    if pred is None:
        raise RuntimeError(
            "run_same_state_actual_n: no success predicate bound to capsule "
            "and none provided (fail closed)")
    prog = progress_fn or capsule.get("progress_fn")
    if prog is None:
        raise RuntimeError(
            "run_same_state_actual_n: no progress_fn bound to capsule and "
            "none provided (fail closed)")
    pred_meta = capsule.get("predicate_meta") or {}
    indices = list(pred_meta.get("achievement_indices", []))
    task_name = pred_meta.get("task", capsule.get("task_name", "UNKNOWN"))

    student = capsule["student"]
    student_params = capsule["student_params"]
    setup = capsule["setup"]
    archive = capsule["archive"]
    state_id = capsule["state_id"]
    identity = capsule["identity"]
    params_sha = capsule["params_sha"]
    memory_spec_hash = capsule["memory_spec_hash"]
    capture_student_id = capsule["capture_student_id"]
    observe_fn = getattr(setup["env"], "get_obs", None)
    task_embeddings = setup["task_embeddings"]
    source = "STUDENT_DETERMINISTIC"

    outcomes: list[BranchOutcome] = []
    branch_seeds: list[int] = []
    for i in range(int(n)):
        branch_id = f"e3-capture-{state_id[:12]}-b{i:02d}"
        env_seed, policy_seed = derive_branch_seeds(
            seed_base=seed_base, state_id=state_id, source=source,
            branch_index=i)
        branch_seeds.append(int(env_seed))
        # Restore the SAME capsule (fresh decode per branch) — never reset.
        entry, encoded = archive.get(state_id)
        bundle = restore_env_state(encoded, capsule["template"])
        state = bundle.env_state
        obs = observe_fn(state, task_embeddings)
        memory = _resolve_branch_memory(
            bundle.policy_memory, memory_mode, student.initial_memory)
        prev_action = bundle.previous_action if hasattr(bundle, "previous_action") else 0
        prev_reward = bundle.previous_reward if hasattr(bundle, "previous_reward") else 0.0
        transitions_used = 0
        terminal_event = None
        rng = jax.random.PRNGKey(env_seed)
        gen = np.random.default_rng(policy_seed)
        for _t in range(int(horizon)):
            policy_seed_step = int(gen.integers(0, 2 ** 31))
            step_rng = jax.random.PRNGKey(policy_seed_step)
            obs_batch = np.asarray(obs).reshape(1, -1)
            out = student.policy_step(student_params, obs_batch, memory,
                                      prev_action, prev_reward, step_rng, True)
            memory = out.get("new_memory", out.get("memory"))
            action = int(np.asarray(out["action"]).reshape(-1)[0])
            rng, step_key = jax.random.split(rng)
            obs, state, reward, done, _info = setup["step_fn"](
                step_key, state, action, setup["params"], task_embeddings)
            prev_action, prev_reward = int(action), float(np.asarray(reward))
            transitions_used += 1
            if bool(np.asarray(done)):
                terminal_event = "ENV_TERMINAL"
                break
        if transitions_used < 1:
            raise RuntimeError(
                f"branch {branch_id} executed 0 transitions — degenerate "
                "branch (fail closed)")
        success = bool(pred(state))
        progress = float(prog(state))
        basis = record_success_basis(state, indices, task_name) \
            if indices else {"achievements_done": {}, "achievements_completed": 0,
                             "achievements_total": 0, "task": task_name}
        if success:
            failure_category = None
        elif terminal_event is not None:
            failure_category = "TERMINAL_BEFORE_SUCCESS"
        else:
            failure_category = "HORIZON_EXHAUSTED"
        outcome = BranchOutcome(
            branch_id=branch_id,
            state_id=state_id,
            search_source=source,
            rng_seed=int(env_seed),
            horizon=int(horizon),
            transitions_used=transitions_used,
            success=success,
            progress=progress,
            terminal_event=terminal_event,
            failure_category=failure_category,
            memory_mode=memory_mode,
            outcome_hash=_sha256_text(json.dumps({
                "branch_id": branch_id, "state_id": state_id,
                "source": source, "env_seed": int(env_seed),
                "horizon": int(horizon),
                "transitions_used": transitions_used,
                "success": success, "progress": progress,
                "terminal_event": terminal_event,
                "failure_category": failure_category,
                "success_basis": basis,
            }, sort_keys=True, default=str)),
            capture_student_id=capture_student_id,
            search_student_id=capture_student_id,
            train_student_id=capture_student_id,
            executing_policy_identity_hash=identity.identity_hash(),
        )
        outcomes.append(outcome)

    # Non-degeneracy: branch seeds pairwise distinct.
    if len(set(branch_seeds)) != len(branch_seeds):
        raise RuntimeError(
            "actual-N branch seeds are not pairwise distinct (fail closed)")

    estimate = estimate_feasibility(outcomes, state_id=state_id)
    return {
        "outcomes": outcomes,
        "n": len(outcomes),
        "successes": int(estimate.successes),
        "estimate": estimate,
        "predicate_meta": pred_meta,
        "branch_seeds": branch_seeds,
    }


def _band(value, lo, hi):
    if value is None:
        return "UNMEASURED"
    if value < lo:
        return "LOW"
    if value > hi:
        return "HIGH"
    return "MID"
