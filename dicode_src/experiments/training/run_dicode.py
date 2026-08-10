"""DiCode: Main training script for online curriculum learning.

This script orchestrates the DiCode training loop, which interleaves:
1. Task evolution (LLM-based generation of new tasks)
2. Training (PPO on a sampled batch of tasks)
3. Task activation (compare-and-swap for the active set)
"""

# --- Standard Library ---
import concurrent.futures
import gc
import os
import pickle
import random
import time

# --- Third-Party ---
import hydra
import jax
import wandb
from omegaconf import DictConfig

# --- Local Modules ---
from dicode.evaluation import run_session_evaluation
from dicode.evolution_efficient import (
    attempt_to_activate_task,
    dispatch_evolution_worker,
)
from dicode.logging_utils import log_session_summary
from dicode.runtime_analysis import tracker
from dicode.selection import sample_tasks_for_training
from dicode.session_boundary import BoundaryStore, sha256_bytes, sha256_path
from dicode.setup import run_initial_seed_training, setup_experiment
from dicode.training import run_session_training


# --- Constants ---
MAX_JAX_CACHE_CLEAR_RETRIES = 10


@hydra.main(version_base="1.2", config_path="../../conf/", config_name="config")
def main(config: DictConfig):
    """Main entry point for the DiCode training loop."""

    # =========================================================================
    # Phase 1: Experiment Setup
    # =========================================================================
    (
        rng,
        gen_manager,
        rl_ckpt_manager,
        rl_train_state,
        global_update_step,
        global_env_steps,
        latest_step,
        cumulative_compiled,
        cumulative_activated,
    ) = setup_experiment(config)

    rl_ckpt_path = os.path.join(os.getcwd(), config.checkpoint_dir)
    last_known_original_return = 0.0

    runtime_cfg = config.get("runtime", {})
    boundary_mode = bool(runtime_cfg.get("session_isolated", False))
    max_sessions_per_process = runtime_cfg.get("max_sessions_per_process")
    if max_sessions_per_process is not None:
        max_sessions_per_process = int(max_sessions_per_process)
        if max_sessions_per_process <= 0:
            raise ValueError("runtime.max_sessions_per_process must be positive")
    boundary_store = BoundaryStore(
        runtime_cfg.get("boundary_dir", os.path.join(os.getcwd(), "session_boundaries"))
    )
    restored_pending_worker_results = None
    restored_sessions_since_evolution = None
    restored_boundary = boundary_store.latest() if boundary_mode else None
    if restored_boundary is not None:
        boundary_manifest, boundary_state = restored_boundary
        if latest_step is None or boundary_manifest.global_update_step != int(latest_step):
            raise RuntimeError(
                "SESSION_BOUNDARY_CHECKPOINT_MISMATCH: boundary and TrainState steps differ"
            )
        rng = boundary_state["rng"]
        global_update_step = int(boundary_state["global_update_step"])
        global_env_steps = int(boundary_state["global_env_steps"])
        last_known_original_return = float(boundary_state["last_known_original_return"])
        cumulative_compiled = int(boundary_state["cumulative_compiled"])
        cumulative_activated = int(boundary_state["cumulative_activated"])
        restored_pending_worker_results = boundary_state.get("pending_worker_results")
        restored_sessions_since_evolution = int(
            boundary_state.get("sessions_since_evolution", 1)
        )
        print(
            f"[Boundary] restored session={boundary_manifest.session_idx} "
            f"global_update_step={global_update_step} global_env_steps={global_env_steps}"
        )

    # =========================================================================
    # Phase 1.5: Initial Seed Training (only if starting from scratch)
    # =========================================================================
    needs_seed_training = gen_manager.session_idx <= 1 and latest_step is None

    if needs_seed_training:
        (
            rng,
            rl_train_state,
            global_update_step,
            global_env_steps,
            evaluation_metrics,
        ) = run_initial_seed_training(
            config,
            gen_manager,
            rng,
            rl_train_state,
            global_update_step,
            global_env_steps,
        )
        global_env_steps = int(global_env_steps)
        last_known_original_return = evaluation_metrics.get("mean_return", 0.0)

    elif latest_step is not None:
        print("Agent state loaded from checkpoint, skipping initial seed training.")
        # Run one-off evaluation to prime metrics for the first evolution step
        rng, evaluation_metrics = run_session_evaluation(
            config,
            rng,
            rl_train_state,
            gen_manager,
            gen_manager.session_idx,
            global_env_steps,
        )
        last_known_original_return = evaluation_metrics.get("mean_return", 0.0)
        print(f"Original return: {last_known_original_return}")

    # =========================================================================
    # Phase 2: Initialize Background Evolution Worker
    # =========================================================================
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    evolve_future = None
    worker_start_time = 0.0
    print("Initialized ThreadPoolExecutor for background evolution.")

    # Evolution interval: how many sessions to wait before syncing with worker.
    # If k=2: Train -> Hold -> Sync & Train with new tasks
    evolution_interval = config.dicode_manager.get("evolution_interval", 2)
    sessions_since_evolution = evolution_interval  # Start with sync on first loop
    if restored_sessions_since_evolution is not None:
        sessions_since_evolution = restored_sessions_since_evolution
    sessions_completed_in_process = 0

    # =========================================================================
    # Phase 3: Main Curriculum Loop
    # =========================================================================
    while global_env_steps < config.training.total_timesteps:
        current_session_idx = gen_manager.session_idx
        print(f"\n{'=' * 60}")
        print(f"--- Starting Session {current_session_idx} ---")
        print(f"{'=' * 60}")

        # --- Step 1: Check if we should sync with evolution worker ---
        new_task_ids = []
        compiled_count = 0
        generation_table = None
        current_worker_wait_time = 0.0
        current_worker_total_time = 0.0

        if restored_pending_worker_results is not None:
            print("  [Boundary] Applying pending evolution results from prior process.")
            new_task_ids, compiled_count = _process_worker_results(
                restored_pending_worker_results, gen_manager, config
            )
            restored_pending_worker_results = None
            sessions_since_evolution = 1
            should_sync = False
        else:
            should_sync = sessions_since_evolution >= evolution_interval

        if should_sync:
            print(f"  [Sync] Iteration {evolution_interval} reached. Waiting for worker...")

            if evolve_future is not None:
                wait_start = time.time()
                worker_timeout = config.dicode_manager.get("worker_sync_timeout_s", 600)
                try:
                    worker_results = evolve_future.result(timeout=worker_timeout)
                    wait_end = time.time()

                    current_worker_wait_time = wait_end - wait_start
                    current_worker_total_time = wait_end - worker_start_time
                    evolve_future = None

                    print(f"  [Timing] Waited: {current_worker_wait_time:.2f}s | "
                          f"Total: {current_worker_total_time:.2f}s")

                    new_task_ids, compiled_count = _process_worker_results(
                        worker_results, gen_manager, config
                    )
                except concurrent.futures.TimeoutError:
                    wait_end = time.time()
                    current_worker_wait_time = wait_end - wait_start
                    current_worker_total_time = wait_end - worker_start_time
                    print(
                        f"  [Sync] WARNING: evolution worker did not finish within "
                        f"{worker_timeout}s (waited {current_worker_wait_time:.2f}s). "
                        f"Skipping new tasks this session and training on the existing "
                        f"archive. Worker keeps running in the background and is re-polled "
                        f"on the next sync."
                    )
                    # Keep evolve_future alive so we do NOT dispatch a duplicate job;
                    # new_task_ids/compiled_count stay at their empty defaults so
                    # training proceeds on the archive instead of blocking.

            sessions_since_evolution = 1
        else:
            print(f"  [Hold] Iteration {sessions_since_evolution}/{evolution_interval}. "
                  "Skipping new tasks.")
            sessions_since_evolution += 1

        # [A] Skill Graph Scheduler: locate the learning frontier from last eval's
        #     per-achievement SR; stash on task_generator for the design prompt.
        #     Flag-gated (default off) -> baseline behaviour unchanged.
        _sched = None
        gen_manager.task_generator.current_skill_target = None
        gen_manager.env_generator.scaffold_rules_block = None  # [C2lite §2] reset per session
        if config.get("skill_preflight", {}).get("use_scheduler", False) and evaluation_metrics:
            try:
                from dicode.skill_preflight.skill_scheduler import (
                    pick_target,
                    format_target_for_prompt,
                    format_target_for_prompt_one_step,
                    format_scaffold_rules_for_coder,
                )
                _sp = config.get("skill_preflight", {})
                _frontier_mode = _sp.get("frontier_mode", "tier")
                _sched = pick_target(
                    evaluation_metrics,
                    threshold=_sp.get("mastery_threshold", 0.60),
                    frontier_mode=_frontier_mode,
                    prereq_threshold=_sp.get("prereq_threshold", 0.3),
                    max_target_achievements=int(_sp.get("max_target_achievements", 6)),
                )
                # [C2lite §2] one-step scaffold prompt rides the prereq criterion by default
                # (scaffold_prompt: "auto"); force "one_step"/"legacy" for single-layer arms.
                _sp_prompt = _sp.get("scaffold_prompt", "auto")
                _one_step = (
                    _sp_prompt == "one_step"
                    or (_sp_prompt == "auto" and _frontier_mode == "prereq")
                )
                _r3_exempt = bool(_sp.get("r3_mastered_exemption", False))
                _r3_v2 = bool(_sp.get("r3_v2", False))
                if _r3_v2 and not _r3_exempt:
                    print("  [r3-v2] IGNORED: requires r3_mastered_exemption=true")
                    _r3_v2 = False
                elif _r3_v2:
                    print("  [r3-v2] ON: rule3 split (pre-mark vs provision) + tool example")
                _tier_cap = bool(_sp.get("r3_tier_cap", False))
                if _tier_cap and not _r3_v2:
                    print("  [r3-cap] IGNORED: requires r3_v2=true")
                    _tier_cap = False
                elif _tier_cap:
                    print("  [r3-cap] ON: lowest sufficient tier, no downstream tools")
                if _one_step:
                    gen_manager.task_generator.current_skill_target = (
                        format_target_for_prompt_one_step(
                            _sched, mastered_exemption=_r3_exempt, r3_v2=_r3_v2,
                            tier_cap=_tier_cap,
                        )
                    )
                    gen_manager.env_generator.scaffold_rules_block = (
                        format_scaffold_rules_for_coder(
                            _sched.sr_snapshot, mastered_exemption=_r3_exempt, r3_v2=_r3_v2,
                            tier_cap=_tier_cap,
                        )
                    )
                else:
                    gen_manager.task_generator.current_skill_target = (
                        format_target_for_prompt(_sched)
                    )
                print(f"  [SkillGraph] mode={_sched.mode} frontier tier {_sched.tier}, "
                      f"targets: {_sched.target_achievements}"
                      + (" (one-step scaffold prompt)" if _one_step else ""))
            except Exception as e:
                print(f"  [SkillGraph] scheduler skipped: {e}")
        # --- Step 2: Dispatch new evolution worker if needed ---
        if evolve_future is None:
            worker_start_time = time.time()
            evolve_future = dispatch_evolution_worker(
                executor, evolve_future, gen_manager, config, evaluation_metrics
            )

        # --- Step 3: Sample tasks for training ---
        print("Sampling tasks for training...")
        target_batch_size = config.dicode_manager.training_sample_size_n
        num_new_to_use = len(new_task_ids)
        num_to_sample_from_archive = max(0, (target_batch_size - 1) - num_new_to_use)

        print(f"  New tasks: {num_new_to_use}. "
              f"Sampling {num_to_sample_from_archive} from archive.")

        sampled_from_archive = sample_tasks_for_training(
            gen_manager, config, num_to_sample_from_archive
        )
        # [B] Preflight Gate (v2): score new tasks with the CURRENT policy and keep
        #     only learnable ones. Reuses the codebase's own machinery end-to-end:
        #     load_tasks_from_env_codes (env from archived code) -> evaluate_new_tasks
        #     (batched frozen-policy rollouts, embeddings, masks) ->
        #     calculate_scores_from_snapshot (per-task SR) -> route() (accept/reject).
        #     Flag-gated (default off) -> baseline behaviour unchanged.
        if config.get("skill_preflight", {}).get("use_preflight", False) and new_task_ids:
            try:
                from dicode.evaluation import evaluate_new_tasks
                from dicode.scoring import calculate_scores_from_snapshot
                from dicode.task_utils import load_tasks_from_env_codes
                from dicode.skill_preflight.preflight import route

                if "validation" not in config:
                    raise RuntimeError(
                        "config.validation missing - add `+validation=default` to the "
                        "run command (preflight needs validation.rollout_updates)")

                _pf_t0 = time.time()
                # Resolve which ids actually load, in order (index-aligns with scores)
                _pf_classes, _pf_ok_ids = load_tasks_from_env_codes(
                    gen_manager.archive, new_task_ids)
                # Ids whose code failed to load: keep (same as baseline; they will be
                # skipped again by the training loader anyway)
                _kept = [t for t in new_task_ids if t not in _pf_ok_ids]

                if _pf_ok_ids:
                    rng, _pf_rng = jax.random.split(rng)
                    _pf_raw = evaluate_new_tasks(
                        config, _pf_rng, rl_train_state, _pf_ok_ids,
                        gen_manager.archive, gen_manager.selector.embedding_model,
                    )
                    _pf_swd = _pf_raw.get("scoring_window_data")
                    if _pf_swd is None:
                        print("  [Preflight] WARNING: rollouts returned no scoring data; "
                              "keeping all new tasks")
                        _kept = list(new_task_ids)
                    else:
                        _pf_scores = calculate_scores_from_snapshot(
                            _pf_swd, len(_pf_ok_ids),
                            _pf_raw["task_achievement_mask"],
                            _pf_raw["task_completed_mask"],
                            config,
                        )
                        for _pf_i, _tid in enumerate(_pf_ok_ids):
                            _sr = float(_pf_scores.get(str(_pf_i), {}).get("sr", -1.0))
                            # sr < 0 => no episode finished => no partial progress
                            _d = route(max(_sr, 0.0), any_partial_progress=(_sr >= 0.0))
                            if _d.action == "accept":
                                _kept.append(_tid)
                                _clip = min(max(_sr, 0.0), 1.0)
                                gen_manager.archive.update_node_learnability(
                                    _tid, _clip * (1.0 - _clip))
                            else:
                                gen_manager.archive.update_node_status(
                                    _tid, f"preflight_{_d.reason}")
                                gen_manager.archive.set_task_active_status(_tid, False)
                                print(f"  [Preflight] reject {_tid}: {_d.reason} "
                                      f"(sr={_sr:.2f})")
                print(f"  [Preflight] kept {len(_kept)}/{len(new_task_ids)} new tasks "
                      f"({time.time() - _pf_t0:.1f}s)")
                new_task_ids = _kept
            except Exception as e:
                print(f"  [Preflight] ERROR (kept all, gate inactive!): {e}")
        _bb = bool(config.get("skill_preflight", {}).get("batch_backfill", False))
        if _bb:
            _shortfall = (target_batch_size - 1) - len(new_task_ids) - len(sampled_from_archive)
            if _shortfall > 0:
                _extra = sample_tasks_for_training(gen_manager, config, _shortfall)
                _seen = set(sampled_from_archive) | set(new_task_ids)
                _add = [t for t in _extra if t not in _seen]
                sampled_from_archive = sampled_from_archive + _add
                print(f"  [batch-fill] ON: +{len(_add)}/{_shortfall} archive top-up after preflight")
        sampled_task_ids = new_task_ids + sampled_from_archive

        if not sampled_task_ids:
            print("  No tasks sampled. Skipping to next session.")
            gen_manager.session_idx += 1
            continue

        # --- Step 4: Run Training ---
        tracker.start_timer("Training")
        (
            rng,
            rl_train_state,
            global_update_step,
            global_env_steps,
            training_metrics,
            num_updates_in_session,
            categorized_tasks,
            evaluation_metrics,
            session_context,
        ) = run_session_training(
            config,
            rng,
            rl_train_state,
            gen_manager,
            global_update_step,
            global_env_steps,
            current_session_idx,
            sampled_task_ids,
            original_return_prev_session=last_known_original_return,
        )
        global_env_steps = int(global_env_steps)

        if "mean_return" in evaluation_metrics:
            last_known_original_return = evaluation_metrics["mean_return"]
            print(f"  Updated Original Task Return: {last_known_original_return:.2f}")

        # [LEAK FIX 2026-07-18] The per-session "evaluation" numbers were never an
        # independent eval: they are extracted from the original_craftax slot of the
        # TRAINING env -- which is wrapped by reward-shaping wrappers, so shaped
        # bounties leaked into evaluation/mean_return (achievements stayed clean;
        # gap == bounty arithmetic; priming eval was the only true eval).
        # Fix: run the real held-out evaluation (same function priming uses) each
        # session -> posts clean evaluation/* internally; keep the training-slot
        # numbers under evaluation_shaped/* (per-session leak-model verification).
        rng, _clean_eval_metrics = run_session_evaluation(
            config,
            rng,
            rl_train_state,
            gen_manager,
            current_session_idx,
            global_env_steps,
        )
        if "mean_return" in _clean_eval_metrics:
            last_known_original_return = _clean_eval_metrics["mean_return"]

        # Log training-slot (shaped) metrics under a truthful name
        if config.use_wandb and evaluation_metrics:
            eval_log_data = {
                "session": current_session_idx,
                "global_env_steps": global_env_steps,
            }
            for key, value in evaluation_metrics.items():
                eval_log_data[f"evaluation_shaped/{key}"] = value
            wandb.log(eval_log_data)

        # --- Step 5: Post-Training Activation (Compare-and-Swap) ---
        real_activated_count = 0
        if new_task_ids:
            print(f"  Attempting to activate {len(new_task_ids)} new tasks...")
            for new_task_id in new_task_ids:
                with gen_manager.archive._lock:
                    if gen_manager.archive.graph.has_node(new_task_id):
                        new_score = gen_manager.archive.graph.nodes[new_task_id].get(
                            "priority_score", 0.0
                        )
                    else:
                        print(f"    Warning: Task {new_task_id} not found. Skipping.")
                        continue

                if attempt_to_activate_task(gen_manager, new_task_id, new_score, config):
                    real_activated_count += 1

        cumulative_compiled += compiled_count
        cumulative_activated += real_activated_count
        tracker.stop_timer("Training", current_session_idx)

        # Handle training failure
        if num_updates_in_session == 0 and not training_metrics and sampled_task_ids:
            print("  Error: Training session failed. Skipping to next session.")
            gen_manager.session_idx += 1
            continue

        # --- Step 6: Cleanup & Checkpointing ---
        print("Forcing garbage collection...")
        gc.collect()

        for i in range(MAX_JAX_CACHE_CLEAR_RETRIES):
            try:
                jax.clear_caches()
                break
            except RuntimeError as e:
                if "Set changed size" in str(e) and i < MAX_JAX_CACHE_CLEAR_RETRIES - 1:
                    time.sleep(0.1)
                    continue
                raise

        print("Checkpointing agent state and saving task graph...")
        rl_ckpt_manager.save(global_update_step, rl_train_state)
        gen_manager.archive.save_graph()

        sessions_completed_in_process += 1
        if boundary_mode and (
            max_sessions_per_process is None
            or sessions_completed_in_process >= max_sessions_per_process
        ):
            pending_worker_results = None
            if evolve_future is not None:
                if not evolve_future.done():
                    raise RuntimeError(
                        "SESSION_BOUNDARY_BLOCKED_IN_FLIGHT_EVOLUTION: "
                        "cannot publish a process boundary while the generation worker is active"
                    )
                pending_worker_results = evolve_future.result()
                evolve_future = None
            graph_path = os.path.join(os.getcwd(), "task_graph.graphml")
            checkpoint_path = os.path.join(rl_ckpt_path, str(global_update_step))
            references = {
                "train_state": sha256_path(checkpoint_path),
                "task_graph": sha256_path(graph_path),
                "sampled_task_ids": sha256_bytes(
                    pickle.dumps(sampled_task_ids, protocol=pickle.HIGHEST_PROTOCOL)
                ),
                "session_context": sha256_bytes(
                    pickle.dumps(session_context, protocol=pickle.HIGHEST_PROTOCOL)
                ),
            }
            boundary_store.write(
                session_idx=current_session_idx,
                global_update_step=global_update_step,
                global_env_steps=global_env_steps,
                state={
                    "rng": rng,
                    "global_update_step": global_update_step,
                    "global_env_steps": global_env_steps,
                    "last_known_original_return": last_known_original_return,
                    "cumulative_compiled": cumulative_compiled,
                    "cumulative_activated": cumulative_activated,
                    "sessions_since_evolution": sessions_since_evolution,
                    "sampled_task_ids": sampled_task_ids,
                    "session_context": session_context,
                    "pending_worker_results": pending_worker_results,
                    "python_random_state": random.getstate(),
                },
                references=references,
                provenance={
                    "git_head": os.environ.get("DICODE_GIT_HEAD", "UNDECLARED"),
                    "gpu_uuid": os.environ.get("DICODE_GPU_UUID", "UNDECLARED"),
                },
            )
            print("[Boundary] committed atomically; stopping process for fresh restore.")
            break

        # --- Step 7: Logging ---
        log_session_summary(
            config,
            current_session_idx,
            global_env_steps,
            global_update_step,
            gen_manager,
            sampled_task_ids,
            num_updates_in_session,
            training_metrics,
            categorized_tasks,
            generation_table,
            rl_ckpt_path,
            current_worker_wait_time,
            current_worker_total_time,
            cumulative_compiled=cumulative_compiled,
            cumulative_activated=cumulative_activated,
        )

        tracker.save_data()
        tracker.plot_results()

        gen_manager.session_idx += 1

    # =========================================================================
    # Phase 4: Final Cleanup
    # =========================================================================
    if config.use_wandb:
        print("\n--- Run complete. Closing W&B run. ---")
        wandb.finish()


def _process_worker_results(
    worker_results: list[dict] | None,
    gen_manager,
    config: DictConfig,
) -> tuple[list[str], int]:
    """Processes results from the evolution worker.

    Args:
        worker_results: List of task generation results from the worker.
        gen_manager: The GenManager instance managing the task archive.
        config: Hydra configuration.

    Returns:
        A tuple of (new_task_ids, compiled_count).
    """
    if not worker_results:
        return [], 0

    compiled_tasks = [res for res in worker_results if res.get("compiled")]
    failed_tasks = [res for res in worker_results if not res.get("compiled")]

    compiled_count = len(compiled_tasks)
    compile_fail_count = len(failed_tasks)

    print(f"  Worker returned {len(worker_results)} tasks: "
          f"{compiled_count} compiled, {compile_fail_count} failed.")

    # Selection: limit to configured number of tasks
    limit = config.dicode_manager.num_generation_tasks
    if compiled_count > limit:
        selected_new_tasks = random.sample(compiled_tasks, limit)
    else:
        selected_new_tasks = compiled_tasks

    # Register selected tasks in archive
    new_task_ids = []
    for res in selected_new_tasks:
        task_id = res.get("generated_task_id")
        code = res.get("code_string")
        reasoning = res.get("reasoning")

        if task_id and code:
            gen_manager.archive.update_node_status(task_id, "compiled")
            gen_manager.archive.set_task_active_status(task_id, False)
            gen_manager.archive.update_node_priority_score(task_id, 0.0)
            if reasoning:
                gen_manager.archive.update_node_reasoning(task_id, reasoning)
            gen_manager.archive.update_node_code(task_id, code)
            new_task_ids.append(task_id)

    return new_task_ids, compiled_count


if __name__ == "__main__":
    main()
