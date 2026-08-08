#!/usr/bin/env python3
"""E3 FORMAL LONGRUN controller (SESSION-ALIGNED CONSERVATIVE 16x128) — THIN
orchestration over the production chain.

Protocol: E3_DICODE_SESSION_ALIGNED_CONSERVATIVE_16x128.
  1 E3 window == 1 complete native DiCode curriculum session ==
  max_updates_per_session (100) outer updates.

Layout (FROZEN, conservative):
  num_envs = 1024, num_steps = 128  -> env_steps/update = 131072
  15 sampled curriculum tasks = 12 dynamic + 3 non-target anchors
  DiCode internally appends OriginalTask exactly once -> 16 total classes.
  original_task_proportion = 0.20.

Reuses the production E3 chain functions from ``run_e3_real_smoke`` (mount /
actual-N / two-LLM / canonical session runtime / compile+register / runstate
codec) and threads the selected Student forward across sessions:

    session 0: canonical checkpoint -> mount -> actual-N -> two-LLM ->
               compile 15+1 -> canonical DiCode SESSION (100 updates) ->
               full RunState
    session k: restore RunState(k-1) -> train_state + counters + memory ->
               mount (updated student) -> actual-N -> two-LLM ->
               compile 15+1 -> canonical DiCode SESSION (100 updates) ->
               full RunState

The frontier / actual-N / two-LLM / curriculum are regenerated ONCE per
session — never per update inside a session.  The OriginalTask is appended by
DiCode exactly once and is never in sampled_task_ids.

NO E3 algorithm is re-implemented here.  Every E3 step is delegated to the
production modules.  This file ONLY owns the session loop + counter threading
+ per-session evidence bookkeeping.

Usage (on server, in tmux):

    python run_e3_formal_longrun.py \
        --student=PERSISTENT_RMT16_ORIGINAL_VTRACE_98304 \
        --sessions=<N> \
        --out=<RUN_DIR>

Env: source ~/.qwen_env; WANDB_MODE=offline; XLA_PYTHON_CLIENT_PREALLOCATE=false;
     CUDA_VISIBLE_DEVICES=<GPU>; PYTHONPATH=<repo>/gpu1_aggregation_siege/src
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIEGE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(SIEGE_ROOT, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, SCRIPT_DIR)

PASS, FAIL, BLOCKED = 0, 4, 5

PERSISTENT = "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304"
RESET128 = "RESET128_RMT16_ORIGINAL_VTRACE_98304"
SLOWGRU = "SLOWGRU_PERSISTENT_CANONICAL_98304"

FORMAL_SOURCE_COMMIT = "0eebbf751b9fa3128928bff5817bd4bbc2bf5aad"
FORMAL_BRANCH = "henry/simulator-frontier-foundation-codex"

# E3_DICODE_SESSION_ALIGNED_CONSERVATIVE_16x128 resolved budget.
# num_envs = 1024, num_steps = 128 -> env_steps/update = 131072.
NUM_ENVS = 1024
NUM_STEPS = 128
ENV_STEPS_PER_UPDATE = NUM_ENVS * NUM_STEPS          # 131072
TOTAL_TIMESTEPS = 2_005_401_600
NATIVE_TOTAL_UPDATES = TOTAL_TIMESTEPS // ENV_STEPS_PER_UPDATE   # 15300
MAX_UPDATES_PER_SESSION_NATIVE = 100
# One formal window == one complete native DiCode curriculum session ==
# max_updates_per_session (100) outer updates.
UPDATES_PER_SESSION = MAX_UPDATES_PER_SESSION_NATIVE  # 100
# Reference experiment budget: selected Student is already past seed stage, so
# E3 starts from the curriculum part = 151 sessions x 100 = 15100 updates.
# Each session executes the full native session (never a for-loop of one-update
# calls).
REFERENCE_EXPERIMENT_UPDATES = 15100
REFERENCE_EXPERIMENT_SESSIONS = REFERENCE_EXPERIMENT_UPDATES // UPDATES_PER_SESSION  # 151
REFERENCE_EXPERIMENT_ENV_STEPS = REFERENCE_EXPERIMENT_UPDATES * ENV_STEPS_PER_UPDATE

# 15 sampled curriculum slots = 12 dynamic + 3 non-target anchors.
CURRICULUM_SLOT_COUNT_CONSERVATIVE = 15

# Session constants used by the smoke driver / E3 chain.
ACTUAL_N = 4
SEARCH_HORIZON = 16
REQUESTED_N_PER_SESSION = 12
SEED = 42


def _log(msg: str) -> None:
    print(f"[e3-longrun-ctrl] {msg}", flush=True)


def _git_head() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=SIEGE_ROOT, timeout=30)
        return (proc.stdout or "").strip()
    except Exception:
        return "UNAVAILABLE"


def _gpu_uuid() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        for line in out.splitlines():
            idx, uuid, name = [p.strip() for p in line.split(",")]
            if idx == cvd:
                return uuid
        return out.splitlines()[0].split(",")[1].strip() if out.strip() else "UNKNOWN"
    except Exception as exc:
        return f"UNKNOWN:{exc!r}"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _params_hash(params) -> str:
    import jax
    leaves = jax.tree_util.tree_leaves(params)
    digest = hashlib.sha256()
    for leaf in leaves:
        arr = jax.numpy.asarray(leaf)
        digest.update(arr.astype(jax.numpy.float32).tobytes())
    return digest.hexdigest()


def _write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True,
                  default=str)
    os.replace(tmp, path)


def _mount_student(candidate_id: str, checkpoint_params=None) -> dict:
    """Mount the selected student via the production adapter."""
    import run_e3_real_smoke as prod
    mount = prod.mount_student(candidate_id)
    if checkpoint_params is not None:
        # Session k>0: mount the adapter identity (identity gates still run on
        # the canonical checkpoint) but inject the UPDATED params — the
        # resumed student — so the frontier/actual-N use the trained student.
        mount["params"] = checkpoint_params
        mount["params_sha256"] = _params_hash(checkpoint_params)
    return mount


def _resolved_config_hash() -> str:
    """Canonical hash of the frozen E3 formal resolved config (16x128)."""
    payload = {
        "protocol": "E3_DICODE_SESSION_ALIGNED_CONSERVATIVE_16x128",
        "num_envs": NUM_ENVS,
        "num_steps": NUM_STEPS,
        "env_steps_per_update": ENV_STEPS_PER_UPDATE,
        "max_updates_per_session": UPDATES_PER_SESSION,
        "curriculum_slot_count": CURRICULUM_SLOT_COUNT_CONSERVATIVE,
    }
    return _sha256_text(json.dumps(payload, sort_keys=True, default=str))


def run_one_session(*, candidate_id: str, session_idx: int, run_dir: str,
                    prev_runstate: str | None, source_commit: str) -> dict:
    """Run ONE production E3 session (100 outer updates) — delegates to the
    production chain.  frontier/actual-N/LLM/curriculum run ONCE per session."""
    import jax
    from dicode.dreaming.gen_manager import GenManager
    from dicode.simulator_frontier.canonical_dicode_runtime import (
        DiCodeOneUpdateContext,
        mint_canonical_dicode_session_runtime,
        execute_session,
        callable_source_sha256,
    )
    from dicode.simulator_frontier.runstate_codec import (
        RunStateCheckpointManager,
        build_full_run_state,
        fresh_process_restore,
        runstate_content_hash,
    )
    import run_e3_real_smoke as prod

    started = time.time()

    # ---- 0. Resolve starting state -------------------------------------------
    start_global_update = 0
    start_global_env_steps = 0
    current_session_idx = 1
    training_rng = jax.random.PRNGKey(SEED)
    student_params = None

    if prev_runstate is not None:
        manager = RunStateCheckpointManager()
        restored = manager.restore(prev_runstate)
        prev_state = restored["run_state"]
        start_global_update = int(prev_state["global_update_step"])
        start_global_env_steps = int(prev_state["global_env_steps"])
        current_session_idx = int(prev_state["current_session_idx"]) + 1
        training_rng = prev_state["training_rng"]
        student_params = prev_state["params"]
        _log(f"session {session_idx}: resume from {prev_runstate} "
             f"global_update={start_global_update} "
             f"global_env={start_global_env_steps} session={current_session_idx}")

    # ---- 1. mount student (real) ---------------------------------------------
    mount = _mount_student(candidate_id, checkpoint_params=student_params)
    _log(f"session {session_idx}: student mounted "
         f"params_sha={mount['params_sha256'][:16]}... "
         f"arch={mount['architecture_family']}")

    # ---- 2. real actual-N from real rollouts (once per session) --------------
    actual_n = prod.run_real_actual_n(mount, n=ACTUAL_N, horizon=SEARCH_HORIZON)
    est = actual_n["estimate"]
    _log(f"session {session_idx}: actual-N={est.total_actual_branches} "
         f"successes={est.successes} sr={est.success_rate:.3f}")

    evidence = {
        "feasibility": {
            "state_id": est.state_id,
            "total_actual_branches": int(est.total_actual_branches),
            "actual_branches_by_source": dict(est.actual_branches_by_source),
            "successes": int(est.successes),
            "success_rate": float(est.success_rate),
            "confidence_interval": [float(est.confidence_interval[0]),
                                    float(est.confidence_interval[1])],
            "mean_progress": float(est.mean_progress),
            "max_progress": float(est.max_progress),
            "transition_cost": int(est.transition_cost),
            "uncertainty": float(est.uncertainty),
            "estimate_version": est.estimate_version,
        },
        "archive_summary": {
            "entry_count": 1,
            "bucket_diversity": 1,
            "evidence_ids": [f"e3-capture-s{session_idx}"],
            "bucket_id": "bucket:real",
        },
        "data_source": "TRAINING_FRONTIER_CAPTURE",
    }

    # ---- 3. two REAL LLM roles (once per session) ----------------------------
    two_llm = prod.build_two_llm_runtime()
    os.environ["E3_FRONTIER_STATE_ID"] = str(est.state_id)
    os.environ["E3_FRONTIER_BUCKET_ID"] = "bucket:real"
    os.environ["E3_ACTUAL_N"] = str(int(est.total_actual_branches))
    os.environ["E3_HORIZON"] = str(SEARCH_HORIZON)
    llm_result = prod.run_two_real_llm_roles(two_llm, evidence)
    plan = llm_result["planner"]
    _log(f"session {session_idx}: two-LLM {llm_result['llm_calls']} calls "
         f"plan_id={plan.plan_id}")

    # ---- 4. hydra config (100 updates/session) + GenManager ------------------
    work_dir = os.path.join(run_dir, "canonical_update", f"s{session_idx:03d}")
    os.makedirs(work_dir, exist_ok=True)
    config = prod.build_hydra_config(
        work_dir, max_updates_per_session=UPDATES_PER_SESSION)
    gen_manager = GenManager(config)

    # ---- 5. train_state (session 0: from checkpoint; k>0: resumed) ----------
    if prev_runstate is None:
        selected = prod.build_train_state_from_selected_student(
            config, mount, candidate_id)
        train_state = selected["train_state"]
        backend = selected["backend"]
        checkpoint_params = selected["checkpoint_params"]
        initial_params_sha = selected["checkpoint_params_sha256"]
    else:
        # Rebuild the backend for the candidate, then reattach the resumed
        # params + opt_state so the optimizer continues (never a fresh reset).
        selected = prod.build_train_state_from_selected_student(
            config, mount, candidate_id)
        backend = selected["backend"]
        checkpoint_params = student_params
        # Recreate TrainState from restored params + opt_state via the backend.
        tx = selected["train_state"].tx
        apply_fn = selected["train_state"].apply_fn
        from flax.training.train_state import TrainState
        opt_state = prev_state["opt_state"]
        train_step = int(prev_state["train_step"])
        train_state = TrainState(
            apply_fn=apply_fn, params=student_params, tx=tx,
            opt_state=opt_state, step=train_step)
        initial_params_sha = _params_hash(student_params)
        _log(f"session {session_idx}: resumed TrainState params="
             f"{initial_params_sha[:16]}... opt_step={train_step}")

    # ---- 6. canonical 15+1 plan + real TaskArchive ---------------------------
    register_result = prod.compile_and_register(
        {"planner": plan, "evidence_hash": llm_result["evidence_hash"]},
        run_id=f"e3-longrun-s{session_idx}",
        state_id=str(est.state_id), memory_mode="SAVED_POLICY_MEMORY",
        gen_manager=gen_manager, session_idx=current_session_idx)
    canonical_plan = register_result["canonical_plan"]
    _log(f"session {session_idx}: canonical plan "
         f"{len(canonical_plan.curriculum_slots)} slots; "
         f"{len(register_result['registered_ids'])} registered")

    # ---- 7. canonical SESSION (100 updates, threaded counters) ---------------
    # One invocation == one native run_session_training == 100 outer updates.
    # NEVER a for-loop of execute_one_update calls.
    session_runtime = mint_canonical_dicode_session_runtime(
        runtime_id=f"e3-formal-session-{candidate_id[:12]}-{session_idx}",
        selected_candidate_id=candidate_id,
        run_session_training_entrypoint="dicode.training:run_session_training",
        run_session_implementation_hash=callable_source_sha256(
            "run_session_training",
            __import__("dicode.training", fromlist=["run_session_training"])
            .run_session_training),
        run_training_session_entrypoint="dicode.ppo_tr:run_training_session",
        run_training_implementation_hash=callable_source_sha256(
            "run_training_session",
            __import__("dicode.ppo_tr", fromlist=["run_training_session"])
            .run_training_session),
        trusted_signer="director/cc4",
    )
    context = DiCodeOneUpdateContext(
        config=config,
        rng=training_rng,
        rl_train_state=train_state,
        gen_manager=gen_manager,
        global_update_step=start_global_update,
        global_env_steps=start_global_env_steps,
        current_session_idx=current_session_idx,
        original_return_prev_session=0.0,
        selected_candidate_id=candidate_id,
        runtime_bundle_hash=session_runtime.runtime_hash,
        formal_asset_registry_hash="0" * 64,
    )
    receipt = execute_session(
        session_runtime, context=context,
        plan=register_result["canonical_plan"],
        adapter=register_result["env_adapter"],
        backend=backend,
        checkpoint_params=checkpoint_params)
    if int(receipt["num_updates_in_session"]) != UPDATES_PER_SESSION:
        raise RuntimeError(
            f"session {session_idx}: expected {UPDATES_PER_SESSION} updates, "
            f"got {int(receipt['num_updates_in_session'])} (fail closed)")
    _log(f"session {session_idx}: canonical session "
         f"{receipt['num_updates_in_session']} updates -> global_update="
         f"{int(receipt['global_update_step'])} env="
         f"{int(receipt['global_env_steps'])}")

    # ---- 8. full RunState + fresh-process restore -----------------------------
    new_state = receipt["rl_train_state"]
    env_rng = jax.random.split(receipt["rng"])[1]
    archive_parts = []
    for tid in sorted(register_result["registered_ids"]):
        archive_parts.append(tid)
    archive_identity = _sha256_text("|".join(archive_parts))
    extra = {}
    if backend is not None:
        arch_memory = receipt.get("architecture_memory")
        if arch_memory is None:
            raise RuntimeError(
                f"session {session_idx}: backend bound but no architecture "
                "memory in receipt (fail closed)")
        extra["architecture_memory"] = backend.serialize_memory_state(arch_memory)
    run_state = build_full_run_state(
        rl_train_state=new_state,
        training_rng=receipt["rng"],
        env_rng=env_rng,
        global_update_step=int(receipt["global_update_step"]),
        global_env_steps=int(receipt["global_env_steps"]),
        current_session_idx=current_session_idx + 1,
        task_archive_identity=archive_identity,
        plan_hash=canonical_plan.plan_hash,
        runtime_bundle_hash=session_runtime.runtime_hash,
        config_hash=_resolved_config_hash(),
        source_commit=source_commit,
        candidate_id=candidate_id,
        architecture_family=mount["architecture_family"],
        extra=extra,
    )
    ckpt_dir = os.path.join(run_dir, "runstate")
    os.makedirs(ckpt_dir, exist_ok=True)
    manager = RunStateCheckpointManager()
    ckpt_path = os.path.join(ckpt_dir,
                             f"e3_canonical_runstate_s{session_idx:03d}")
    save_report = manager.save(run_state, ckpt_path,
                               idempotency_token=f"e3-longrun-s{session_idx}")
    local_content_hash = runstate_content_hash(run_state)
    restored = fresh_process_restore(ckpt_path, extra_pythonpath=SRC_DIR)
    equivalent = bool(restored.get("content_hash") == local_content_hash)
    if not equivalent:
        raise RuntimeError(
            f"session {session_idx}: FRESH_PROCESS_RESTORE mismatch "
            f"(parent={local_content_hash[:16]} child="
            f"{restored.get('content_hash', '')[:16]}) (fail closed)")
    _log(f"session {session_idx}: RunState saved + fresh-restore OK "
         f"sha={save_report['state_file_sha256'][:16]}...")

    # ---- 9. evidence ----------------------------------------------------------
    session_report = {
        "schema": "simulator_frontier.e3_formal_longrun_session/v1",
        "protocol": "E3_DICODE_SESSION_ALIGNED_CONSERVATIVE_16x128",
        "session_idx": session_idx,
        "run_id": f"e3-longrun-s{session_idx}",
        "candidate_id": candidate_id,
        "architecture_family": mount["architecture_family"],
        "params_sha256": mount["params_sha256"],
        "initial_trainstate_params_sha256": initial_params_sha,
        "checkpoint_params_sha256": checkpoint_params and _params_hash(checkpoint_params),
        "initial_equals_checkpoint": bool(initial_params_sha == (checkpoint_params and _params_hash(checkpoint_params))),
        "start_global_update": start_global_update,
        "start_global_env_steps": start_global_env_steps,
        "current_session_idx": current_session_idx,
        "global_update_step": int(receipt["global_update_step"]),
        "global_env_steps": int(receipt["global_env_steps"]),
        "num_updates_in_session": int(receipt["num_updates_in_session"]),
        "expected_updates_per_session": UPDATES_PER_SESSION,
        "update_shape": [16, 128],
        "num_envs": NUM_ENVS,
        "num_steps": NUM_STEPS,
        "env_steps_per_update": ENV_STEPS_PER_UPDATE,
        "curriculum_slots": len(canonical_plan.curriculum_slots),
        "registered_ids": list(register_result["registered_ids"]),
        "plan_hash": canonical_plan.plan_hash,
        "llm_calls": int(llm_result["llm_calls"]),
        "evidence_hash": llm_result["evidence_hash"],
        "actual_n_branches": int(est.total_actual_branches),
        "actual_n_successes": int(est.successes),
        "fresh_process_restore_equivalent": equivalent,
        "checkpoint_path": ckpt_path,
        "checkpoint_state_sha256": save_report["state_file_sha256"],
        "checkpoint_hash": save_report["checkpoint_hash"],
        "elapsed_s": round(time.time() - started, 2),
    }
    _write_json(os.path.join(run_dir, "evidence",
                             f"session_{session_idx:03d}.json"), session_report)
    return session_report


def _wandb_offline_init(run_id: str) -> None:
    """The canonical DiCode PPO loop (ppo_tr) logs through a jax.debug
    callback that calls wandb.log() unconditionally (even with
    use_wandb=false).  Initialize wandb OFFLINE so those calls succeed
    (never a network run, never a fake) — mirrors run_e3_real_smoke."""
    try:
        import wandb
        os.environ.setdefault("WANDB_MODE", "offline")
        wandb.init(mode="offline", project="e3_formal_longrun",
                   entity="e3", name=run_id, reinit=True)
    except Exception as exc:
        _log(f"wandb offline init warning: {exc!r}")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    candidate_id = None
    sessions = 2  # default = integration (2 sessions x 100); full-budget passes 151
    run_dir = None
    for arg in argv:
        if arg.startswith("--student="):
            candidate_id = arg.split("=", 1)[1]
        elif arg.startswith("--sessions="):
            sessions = int(arg.split("=", 1)[1])
        elif arg.startswith("--out="):
            run_dir = arg.split("=", 1)[1]
        else:
            print(f"[e3-longrun-ctrl] unknown argument {arg!r}", flush=True)
            return FAIL
    if candidate_id not in (PERSISTENT, RESET128, SLOWGRU):
        print(f"[e3-longrun-ctrl] invalid --student {candidate_id!r}", flush=True)
        return FAIL
    if not run_dir:
        print("[e3-longrun-ctrl] --out=<RUN_DIR> required", flush=True)
        return FAIL
    os.makedirs(run_dir, exist_ok=True)
    source_commit = _git_head()
    _wandb_offline_init(f"e3-formal-{candidate_id[:12]}")

    _write_json(os.path.join(run_dir, "RUN_METADATA.json"), {
        "schema": "simulator_frontier.e3_formal_longrun/v2",
        "protocol": "E3_DICODE_SESSION_ALIGNED_CONSERVATIVE_16x128",
        "branch": FORMAL_BRANCH,
        "source_commit": source_commit,
        "candidate_id": candidate_id,
        "sessions": sessions,
        "updates_per_session": UPDATES_PER_SESSION,
        "update_shape": [16, 128],
        "num_envs": NUM_ENVS,
        "num_steps": NUM_STEPS,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": _gpu_uuid(),
        "gpu_device": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "pid": os.getpid(),
        "reference_experiment_updates": REFERENCE_EXPERIMENT_UPDATES,
        "reference_experiment_env_steps": REFERENCE_EXPERIMENT_ENV_STEPS,
    })
    _write_json(os.path.join(run_dir, "RESOLVED_CONFIG.json"), {
        "protocol": "E3_DICODE_SESSION_ALIGNED_CONSERVATIVE_16x128",
        "num_envs": NUM_ENVS,
        "num_steps": NUM_STEPS,
        "env_steps_per_update": ENV_STEPS_PER_UPDATE,
        "env_steps_per_update_changed": False,
        "max_updates_per_session_native": MAX_UPDATES_PER_SESSION_NATIVE,
        "updates_per_session": UPDATES_PER_SESSION,
        "curriculum_slot_count": CURRICULUM_SLOT_COUNT_CONSERVATIVE,
        "total_timesteps_native": TOTAL_TIMESTEPS,
        "native_total_updates": NATIVE_TOTAL_UPDATES,
        "reference_experiment_sessions": REFERENCE_EXPERIMENT_SESSIONS,
        "reference_experiment_updates": REFERENCE_EXPERIMENT_UPDATES,
        "reference_experiment_env_steps": REFERENCE_EXPERIMENT_ENV_STEPS,
        "seed": SEED,
        "actual_n": ACTUAL_N,
        "search_horizon": SEARCH_HORIZON,
        "requested_n_per_session": REQUESTED_N_PER_SESSION,
    })
    _write_json(os.path.join(run_dir, "GIT_BINDING.json"), {
        "branch": FORMAL_BRANCH,
        "head": source_commit,
        "formal_source_commit": FORMAL_SOURCE_COMMIT,
        "head_matches_formal": source_commit == FORMAL_SOURCE_COMMIT,
    })

    _log(f"START candidate={candidate_id} sessions={sessions} run_dir={run_dir} "
         f"head={source_commit[:12]} update_shape=[16,128] "
         f"updates_per_session={UPDATES_PER_SESSION}")
    prev_runstate = None
    results = []
    for s in range(1, sessions + 1):
        report = run_one_session(
            candidate_id=candidate_id, session_idx=s, run_dir=run_dir,
            prev_runstate=prev_runstate, source_commit=source_commit)
        results.append(report)
        prev_runstate = report["checkpoint_path"]

    final = {
        "schema": "simulator_frontier.e3_formal_longrun_final/v2",
        "protocol": "E3_DICODE_SESSION_ALIGNED_CONSERVATIVE_16x128",
        "candidate_id": candidate_id,
        "sessions_completed": len(results),
        "updates_per_session": UPDATES_PER_SESSION,
        "final_global_update": results[-1]["global_update_step"] if results else 0,
        "final_global_env_steps": results[-1]["global_env_steps"] if results else 0,
        "latest_checkpoint": prev_runstate,
        "all_fresh_restore_ok": all(r["fresh_process_restore_equivalent"]
                                    for r in results),
        "experiment_updates": (results[-1]["global_update_step"] - 0) if results else 0,
        "experiment_env_steps": (results[-1]["global_env_steps"] - 0) if results else 0,
        "source_commit": source_commit,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(os.path.join(run_dir, "FINAL_STATUS.json"), final)
    _log(f"DONE candidate={candidate_id} sessions={len(results)} "
         f"final_global_update={final['final_global_update']} "
         f"final_global_env={final['final_global_env_steps']}")
    return PASS


if __name__ == "__main__":
    raise SystemExit(main())
