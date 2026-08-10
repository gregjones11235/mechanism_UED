#!/usr/bin/env python3
"""E3 real frontier-capsule actual-N (P0-1/2/3).

Implements the REAL same-state actual-N for the E3 formal longrun:

  P0-1  capture ONE complete frontier capsule from a real Student rollout:
        env state, observation, entering architecture memory, previous
        action/reward, Student params identity, task identity, capture RNG.
        Every branch restores from that SAME capsule, changing ONLY the branch
        RNG.  NO per-branch env.reset_env(), NO per-branch initial_memory().
  P0-2  success is decided by an explicit frontier/task success predicate
        (gate_progress >= threshold); death/timeout/plain done is NEVER auto
        success.  terminal_event and failure_category are recorded.
  P0-3  no fake fixed state_id; state/capsule hash comes from the REAL
        serialized bytes (encode_env_state payload_hash).  No hand-written
        TRAINING_FRONTIER_CAPTURE string as a substitute for measured facts.

Reuses the mature, locally-constructible mechanisms:
  build_core_setup, encode_env_state, restore_env_state, build_template,
  FrontierArchive, FrontierArchiveEntry, bind_capture_entry, BranchOutcome,
  estimate_feasibility, derive_branch_seeds.

This module is TEST_ONLY-independent: it performs real rollouts but does NOT
call LLMs, does NOT train, and does NOT depend on a controller-signed restore
bundle (that is the P0-6 scope handled separately).
"""

from __future__ import annotations

import hashlib
import json
import math
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


def _build_multitask_setup(*, max_timesteps: int, reset_seed: int) -> dict:
    """Build the MultiTaskMiniCraftaxEnv setup with 8335-dim observations
    (8268 symbolic + 67 task embedding) — the SAME env the RMT16 / SlowGRU
    Student adapters expect (obs_dim 8335)."""
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
    }


def default_success_predicate(threshold: float = 0.50):
    """Frontier success predicate: gate_progress >= threshold.

    death / timeout / plain done is NOT success unless the frontier reached
    the gate threshold.  The flattened state exposes gate_progress as a leaf.
    """
    def _pred(final_flat: Mapping[str, Any]) -> bool:
        leaves = final_flat.get("leaves", {})
        val = leaves.get("gate_progress")
        if val is None:
            # fall back to a top-level gate_progress if present
            val = final_flat.get("gate_progress")
        if val is None:
            return False
        try:
            return float(np.asarray(val)) >= float(threshold)
        except (TypeError, ValueError):
            return False
    return _pred


def default_progress_fn() -> Callable[[Mapping[str, Any]], float]:
    """Progress: gate_progress leaf (0..1), clamped finite."""
    def _fn(final_flat: Mapping[str, Any]) -> float:
        leaves = final_flat.get("leaves", {})
        val = leaves.get("gate_progress")
        if val is None:
            val = final_flat.get("gate_progress")
        if val is None:
            return 0.0
        try:
            p = float(np.asarray(val))
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(p):
            return 0.0
        return max(0.0, min(1.0, p))
    return _fn


def capture_frontier_capsule(*, student, student_params, run_id: str,
                             reset_seed: int, capture_at_step: int,
                             max_timesteps: int, success_threshold: float,
                             memory_mode: str = "SAVED_POLICY_MEMORY",
                             initial_memory=None) -> dict:
    """P0-1: capture ONE real frontier capsule from a real Student rollout.

    P0-4: when ``initial_memory`` is provided (the resumed architecture memory
    from the previous session's RunState), the capture rollout STARTS from that
    memory instead of a zero initial_memory — the frontier / actual-N for
    session k>1 begin from the trained hidden state, never a reset.

    Returns a dict with archive, entry, encoded, bundle, template, setup,
    state_id, facts, and the live capture memory.
    """
    from dicode.simulator_frontier.env_restore import (
        build_template, encode_env_state, flatten_env_state,
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

    identity = student.identity()
    params_sha = _params_hash(student_params)
    memory_spec_hash = student.memory_spec().spec_hash()
    capture_student_id = str(identity.candidate_id)

    runner_key = jax.random.PRNGKey(reset_seed + 1)
    state = setup["state0"]
    obs = observe_fn(state, task_embeddings)
    # P0-4: start the capture rollout from the resumed architecture memory
    # (session k>1) instead of a fresh initial_memory.  The memory is a
    # batch>=1 dict; we slice env 0 to the single capture env.
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

    # Encode the REAL captured state (P0-3: state_id = real payload hash).
    encoded, bundle = encode_env_state(
        state, next_step_key=runner_key,
        previous_action=prev_action, previous_reward=prev_reward,
        policy_memory=memory, history_reference=None,
    )
    state_id = encoded.payload_hash
    final_flat = flatten_env_state(state)
    pred = default_success_predicate(success_threshold)
    success_at_capture = bool(pred(final_flat))
    progress_at_capture = float(default_progress_fn()(final_flat))

    # Real measured facts (aggregate, non-action-guiding).
    health = getattr(state, "health", None)
    max_health = getattr(state, "max_health", None)
    ratio = None
    if health is not None and max_health not in (None, 0):
        try:
            ratio = float(health) / float(max_health)
        except (TypeError, ValueError, ZeroDivisionError):
            ratio = None
    achievements = getattr(state, "achievements", None)
    snapshot: dict = {}
    if achievements is not None:
        try:
            snapshot["achievements_done"] = int(sum(bool(v) for v in achievements))
        except TypeError:
            snapshot = {}
    facts = {
        "floor": int(getattr(state, "floor_number", -1)),
        "gate_progress": float(getattr(state, "gate_progress", 0.0) or 0.0),
        "health_band": _band(ratio, 0.34, 0.67),
        "threat_band": "UNMEASURED",
        "resource_band": "UNMEASURED",
        "inventory_stage": "UNMEASURED",
        "achievement_snapshot": snapshot,
        "terminal": terminal_before_capture,
    }

    entry = FrontierArchiveEntry(
        state_id=state_id,
        source_checkpoint_id=str(identity.params_sha256),
        source_episode_id=f"{run_id}:standard-reset:{reset_seed}",
        source_seed=int(reset_seed),
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
    env reset, never initial_memory per branch.  success uses the explicit
    predicate.  Returns outcomes + feasibility estimate.
    """
    from dicode.simulator_frontier.env_restore import (
        restore_env_state, flatten_env_state,
    )
    from dicode.simulator_frontier.branch_search_runner import derive_branch_seeds
    from dicode.simulator_frontier.search_statistics import (
        BranchOutcome, estimate_feasibility,
    )
    from dicode.simulator_frontier.discovery_provenance import (
        DiscoveryProvenance,
    )

    pred = success_predicate or default_success_predicate(0.50)
    prog = progress_fn or default_progress_fn()
    student = capsule["student"] if "student" in capsule else None
    # capsule may carry student ref via callers; here we require it via param
    if student is None:
        raise RuntimeError("run_same_state_actual_n: capsule has no student; "
                           "pass student through the capsule dict")

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
    for i in range(int(n)):
        branch_id = f"e3-capture-{state_id[:12]}-b{i:02d}"
        env_seed, policy_seed = derive_branch_seeds(
            seed_base=seed_base, state_id=state_id, source=source,
            branch_index=i)
        # Restore the SAME capsule (fresh decode per branch) — never reset.
        entry, encoded = archive.get(state_id)
        bundle = restore_env_state(encoded, capsule["template"])
        state = bundle.env_state
        obs = observe_fn(state, task_embeddings)
        memory = bundle.policy_memory if memory_mode == "SAVED_POLICY_MEMORY" \
            else student.initial_memory(1)
        prev_action = bundle.previous_action if hasattr(bundle, "previous_action") else 0
        prev_reward = bundle.previous_reward if hasattr(bundle, "previous_reward") else 0.0
        if memory is None:
            memory = student.initial_memory(1)
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
        final_flat = flatten_env_state(state)
        success = bool(pred(final_flat))
        progress = float(prog(final_flat))
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
            }, sort_keys=True, default=str)),
            capture_student_id=capture_student_id,
            search_student_id=capture_student_id,
            train_student_id=capture_student_id,
            executing_policy_identity_hash=identity.identity_hash(),
        )
        outcomes.append(outcome)

    estimate = estimate_feasibility(outcomes, state_id=state_id)
    return {
        "outcomes": outcomes,
        "n": len(outcomes),
        "successes": int(estimate.successes),
        "estimate": estimate,
    }


def _band(value, lo, hi):
    if value is None:
        return "UNMEASURED"
    if value < lo:
        return "LOW"
    if value > hi:
        return "HIGH"
    return "MID"
