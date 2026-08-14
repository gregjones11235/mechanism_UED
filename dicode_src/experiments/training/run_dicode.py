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
from dicode.skill_preflight.contract import (
    PreflightOptimizationContractError,
    handle_preflight_gate_error,
)
from dicode.ppo_tr import clear_train_compile_cache
from dicode.craftax_evaluation import clear_compiled_evaluator_cache
from dicode.selection import sample_tasks_for_training
from dicode.setup import run_initial_seed_training, setup_experiment
from dicode.training import run_session_training


# --- Constants ---
MAX_JAX_CACHE_CLEAR_RETRIES = 10


def _preflight_route(scores, ok_ids, kept, archive, route_fn):
    """Preflight accept/reject loop + archive mutation (shared production helper).

    Delegates to ``dicode.skill_preflight.preflight_route.preflight_route`` so the
    production driver and the frozen preflight replay share one implementation;
    profiling spans fire only when ``tracker.enabled``.
    """
    from dicode.skill_preflight.preflight_route import preflight_route as _impl
    _impl(scores, ok_ids, kept, archive, route_fn, tracker=tracker)


def _learnability_scores_from_counts(finished_counts, success_counts, num_tasks):
    """Pure conversion from fused counters to the route's minimal score map."""
    finished = [int(value) for value in finished_counts]
    successes = [int(value) for value in success_counts]
    if len(finished) != num_tasks or len(successes) != num_tasks:
        raise PreflightOptimizationContractError(
            "fused learnability counter length does not match loaded task count"
        )

    scores = {}
    for task_idx, (num_finished, num_successes) in enumerate(zip(finished, successes)):
        if num_finished < 0 or num_successes < 0 or num_successes > num_finished:
            raise PreflightOptimizationContractError(
                "invalid fused learnability counters: expected "
                "0 <= successes <= finished"
            )
        sr = -1.0 if num_finished == 0 else num_successes / num_finished
        clipped_sr = min(1.0, max(0.0, sr)) if sr >= 0.0 else 0.0
        scores[str(task_idx)] = {
            "sr": float(sr),
            "priority_score": float(clipped_sr * (1.0 - clipped_sr)),
        }
    return scores


@hydra.main(version_base="1.2", config_path="../../conf/", config_name="config")
def main(config: DictConfig):
    """Main entry point for the DiCode training loop."""
    tracker.configure(config, reset=True)
    clear_compiled_evaluator_cache()
    clear_train_compile_cache()

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

    # =========================================================================
    # Phase 3: Main Curriculum Loop
    # =========================================================================
    while global_env_steps < config.training.total_timesteps:
        current_session_idx = gen_manager.session_idx
        tracker.set_session(current_session_idx)
        session_start_ns = time.monotonic_ns()
        print(f"\n{'=' * 60}")
        print(f"--- Starting Session {current_session_idx} ---")
        print(f"{'=' * 60}")

        # --- Step 1: Check if we should sync with evolution worker ---
        new_task_ids = []
        compiled_count = 0
        generation_table = None
        current_worker_wait_time = 0.0
        current_worker_total_time = 0.0

        should_sync = sessions_since_evolution >= evolution_interval

        if should_sync:
            print(f"  [Sync] Iteration {evolution_interval} reached. Waiting for worker...")

            if evolve_future is not None:
                wait_start = time.time()
                wait_start_ns = time.monotonic_ns()
                worker_timeout = config.dicode_manager.get("worker_sync_timeout_s", 600)
                try:
                    worker_results = evolve_future.result(timeout=worker_timeout)
                    wait_end = time.time()

                    current_worker_wait_time = wait_end - wait_start
                    tracker.record("evolution_sync_wait", wait_start_ns, session=current_session_idx)
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
                    tracker.record("evolution_sync_wait", wait_start_ns, session=current_session_idx,
                                   status="timeout")
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

        task_sample_start = time.monotonic_ns()
        sampled_from_archive = sample_tasks_for_training(gen_manager, config, num_to_sample_from_archive)
        tracker.record("task_sampling", task_sample_start, session=current_session_idx)
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
                # [B1] preflight_wall start time: only captured when profiling is
                # enabled (zero instrumentation otherwise).
                _pf_t0_ns = time.monotonic_ns() if tracker.enabled else None
                # Resolve which ids actually load, in order (index-aligns with scores)
                if tracker.enabled:
                    with tracker.span("preflight_task_reload"):
                        _pf_classes, _pf_ok_ids = load_tasks_from_env_codes(
                            gen_manager.archive, new_task_ids)
                else:
                    _pf_classes, _pf_ok_ids = load_tasks_from_env_codes(
                        gen_manager.archive, new_task_ids)
                # Ids whose code failed to load: keep (same as baseline; they will be
                # skipped again by the training loader anyway)
                _kept = [t for t in new_task_ids if t not in _pf_ok_ids]

                if _pf_ok_ids:
                    rng, _pf_rng = jax.random.split(rng)
                    # [B2] performance.preflight_reuse_loaded_tasks (default off):
                    # pass the first load's classes/ids so evaluate_new_tasks skips
                    # its second load_tasks_from_env_codes. Off -> both None -> the
                    # historical second load happens, byte-identical.
                    _reuse_tasks = bool((config.get("performance", {})
                                         if hasattr(config, "get") else {})
                                        .get("preflight_reuse_loaded_tasks", False))
                    # [R2] fail fast before the 40-update rollout: if B3 compact
                    # payload is enabled, the score function must be one that
                    # compact supports. Unknown -> PreflightOptimizationContractError
                    # propagates (fail-closed) through the outer gate catch.
                    _compact_payload = bool((config.get("performance", {})
                                             if hasattr(config, "get") else {})
                                            .get("compact_preflight_payload", False))
                    if _compact_payload:
                        from dicode.skill_preflight.scoring_contract import compact_field_decisions
                        compact_field_decisions(config.dicode_manager.score_function)
                    _fused_summary = bool((config.get("performance", {})
                                           if hasattr(config, "get") else {})
                                          .get("learnability_fused_preflight_summary", False))
                    if _fused_summary:
                        from dicode.skill_preflight.learnability_summary import (
                            require_learnability_fused_contract,
                        )
                        # Contract validation intentionally precedes rollout
                        # construction/execution. Invalid PVL/MaxMC use is fatal.
                        require_learnability_fused_contract(
                            config.dicode_manager.score_function
                        )
                    _pf_raw = evaluate_new_tasks(
                        config, _pf_rng, rl_train_state, _pf_ok_ids,
                        gen_manager.archive, gen_manager.selector.embedding_model,
                        preloaded_task_classes=(_pf_classes if _reuse_tasks else None),
                        preloaded_task_ids=(_pf_ok_ids if _reuse_tasks else None),
                    )
                    if _fused_summary:
                        _pf_summary = _pf_raw.get("learnability_summary")
                        if _pf_summary is None:
                            raise PreflightOptimizationContractError(
                                "fused learnability evaluation returned no summary"
                            )
                        if tracker.enabled:
                            with tracker.span("scoring_transfer"):
                                _pf_finished, _pf_successes = jax.device_get((
                                    _pf_summary.get("finished_counts"),
                                    _pf_summary.get("success_counts"),
                                ))
                        else:
                            _pf_finished, _pf_successes = jax.device_get((
                                _pf_summary.get("finished_counts"),
                                _pf_summary.get("success_counts"),
                            ))
                        if _pf_finished is None or _pf_successes is None:
                            raise PreflightOptimizationContractError(
                                "fused learnability summary is missing counters"
                            )
                        if tracker.enabled:
                            with tracker.span("scoring_cpu"):
                                _pf_scores = _learnability_scores_from_counts(
                                    _pf_finished, _pf_successes, len(_pf_ok_ids)
                                )
                        else:
                            _pf_scores = _learnability_scores_from_counts(
                                _pf_finished, _pf_successes, len(_pf_ok_ids)
                            )
                    else:
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
                    if _fused_summary or _pf_swd is not None:
                        if tracker.enabled:
                            with tracker.span("route"):
                                _preflight_route(_pf_scores, _pf_ok_ids, _kept,
                                                 gen_manager.archive, route)
                        else:
                            _preflight_route(_pf_scores, _pf_ok_ids, _kept,
                                             gen_manager.archive, route)
                print(f"  [Preflight] kept {len(_kept)}/{len(new_task_ids)} new tasks "
                      f"({time.time() - _pf_t0:.1f}s)")
                if tracker.enabled:
                    tracker.record("preflight_wall", _pf_t0_ns, session=current_session_idx)
                new_task_ids = _kept
            except Exception as e:
                # [R2] fail-closed: an enabled preflight optimization (B2/B3)
                # contract violation must terminate the run, never degrade to
                # "keep all". Ordinary preflight errors still degrade historically.
                handle_preflight_gate_error(e)
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
        heldout_start = time.monotonic_ns()
        rng, _clean_eval_metrics = run_session_evaluation(
            config, rng, rl_train_state, gen_manager, current_session_idx, global_env_steps,
        )
        tracker.record("heldout_eval", heldout_start, session=current_session_idx)
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
            _wandb_start = time.monotonic_ns()
            try:
                wandb.log(eval_log_data)
            except Exception:
                tracker.record("wandb_log_eval", _wandb_start, session=current_session_idx, status="error")
                raise
            else:
                tracker.record("wandb_log_eval", _wandb_start, session=current_session_idx)

        # --- Step 5: Post-Training Activation (Compare-and-Swap) ---
        activation_start = time.monotonic_ns()
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
        tracker.record("task_activation", activation_start, session=current_session_idx)

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
        _ckpt_start = time.monotonic_ns()
        try:
            rl_ckpt_manager.save(global_update_step, rl_train_state)
            if tracker.enabled and hasattr(rl_ckpt_manager, "wait_until_finished"):
                rl_ckpt_manager.wait_until_finished()
        except Exception:
            tracker.record("checkpoint_save", _ckpt_start, session=current_session_idx, status="error")
            raise
        else:
            tracker.record("checkpoint_save", _ckpt_start, session=current_session_idx)

        _graph_start = time.monotonic_ns()
        try:
            gen_manager.archive.save_graph()
        except Exception:
            tracker.record("task_graph_save", _graph_start, session=current_session_idx, status="error")
            raise
        else:
            tracker.record("task_graph_save", _graph_start, session=current_session_idx)

        # --- Step 7: Logging ---
        _summary_start = time.monotonic_ns()
        try:
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
        except Exception:
            tracker.record("session_summary", _summary_start, session=current_session_idx, status="error")
            raise
        else:
            tracker.record("session_summary", _summary_start, session=current_session_idx)

        tracker.record("session_wall", session_start_ns, session=current_session_idx)
        tracker.save_data()
        tracker.plot_results()
        if tracker.enabled:
            tracker.derive_reports()

        gen_manager.session_idx += 1

    # =========================================================================
    # Phase 4: Final Cleanup
    # =========================================================================
    if config.use_wandb:
        print("\n--- Run complete. Closing W&B run. ---")
        _finish_start = time.monotonic_ns()
        try:
            wandb.finish()
        except Exception:
            tracker.record("wandb_finish", _finish_start, status="error")
            raise
        else:
            tracker.record("wandb_finish", _finish_start)


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
