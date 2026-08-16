# E3 Current Architecture Audit (P0)

Date: 2026-08-16. Read-only audit of `mechanism_UED` (branch `Henry-branch`,
worktree `mechanism_UED_Henry_worktree`, code root `gpu1_aggregation_siege/`).
Purpose: ground the E3-litesim refactor ("Capability Measurement + Lightweight
Simulator Data Engine + PPO Student Learning") in the real data flow before
changing anything.

## 1. Current flow (as found, not as documented)

Long-run entry: `scripts/run_e3_formal_longrun.py`
  -> per-session window: `src/dicode/simulator_frontier/e3_window.py::one_window_pipeline`
  -> Actual-N probe: `src/dicode/simulator_frontier/branch_search_runner.py::BranchSearchRunner.run_actual_n`
     (restore_entry from `frontier_archive`, `_prepare_memory`, `_select_action`,
      `run_branch` on the vendored simulator)
  -> curriculum: `frontier_distributions.py::compile_planner_to_frontier_distributions`
     (12 dynamic + 4 anchor slots) + `production_task_materializer.py`
  -> PPO: `src/dicode/ppo_tr.py::make_train(config, task_classes, ..., backend=,
     checkpoint_params=, initial_memory_dict=)` -> `train(rng, train_state, current_original_return)`
  -> persistence: `src/dicode/shared_runtime/runstate.py::RunStateCheckpointManager`
     (REQUIRED_RUNSTATE_FIELDS: params, opt_state, train_step, training_rng,
      env_rng, global_update_step, global_env_steps, current_session_idx,
      task_archive_identity, mechanism_state_identity, plan_hash,
      runtime_bundle_hash, config_hash, source_commit)

## 2. Major classes / functions and source files

| Module | File | Core symbols | Role |
|---|---|---|---|
| simulator | `src/minicraftax/**` (vendored craftax fork) | `MiniCraftaxTrain` (envs/base.py:38), `EnvState` (craftax_state.py:42), seed tasks `survive/combat/original` | JAX vectorizable env; NO auto-reset in Train variant |
| env state codec | `src/dicode/simulator_frontier/env_restore.py` | `build_template`, `flatten_env_state`, `unflatten_env_state`, `slice_env_state`, `stack_env_states` | freeze/restore/stack EnvState |
| student adapters | `src/dicode/student_adapters/*` | `StudentAdapter` protocol, `SlowGRUStudentAdapter.policy_step` (slowgru_adapter.py:541) | read-only policy forward; binds caller-owned params to ephemeral handle |
| binding | `src/dicode/simulator_frontier/student_binding.py` | `bind_capture_entry`, `bind_branch_outcome`, `assert_*_bound` | per-entry/per-outcome student binding (E3-CF era) |
| probe | `branch_search_runner.py` | `BranchSearchRunner`, `run_actual_n`, `actual_n_summary` | state-restored Actual-N branches |
| distributions | `frontier_distributions.py`, `distribution_runtime.py` | `FrontierDistribution`, `compose_12_plus_4`, `resolve_distribution_binding` | 12+4 slot semantics |
| PPO | `src/dicode/ppo_tr.py` | `make_train`, `train`, `_update_step`, `_env_step_backend` | canonical PPO core; rollout collected INTERNALLY |
| backend ABC | `src/dicode/training_backend.py` | `StudentTrainingBackend` (policy_forward_eval/train, init/reset_runner_memory, create_train_state_from_checkpoint) | architecture-specific surface; "PPO core NEVER modified" |
| SlowGRU backend | `src/dicode/training_backend_slowgru.py` | `SlowGRUTrainingBackend` | longstate-correct forwards |
| runstate | `src/dicode/shared_runtime/runstate.py` | `RunStateCheckpointManager` | full-state checkpoint with hash chain |

## 3. State ownership

- PPO TrainState owns params+opt_state inside `train()`; E3 continuation re-enters
  via `checkpoint_params` + `initial_memory_dict` (P0-4 path, ppo_tr.py:196-214).
- SlowGRU adapter keeps an authenticated source handle; `policy_step` validates
  caller params on every call and shallow-copies the handle (slowgru_adapter.py:560-575).
- Recurrent state: GTrXL memories/mask/idx (legacy path) or backend memory_dict
  (`rmt.mem_tokens`, `longstate.h/buf/count`); `_env_step_backend` captures the
  PRE-ACTION longstate so the loss recomputes from entering state (ppo_tr.py:415+).

## 4. PPO data path

Rollouts are collected inside `train()` by `jax.lax.scan(_env_step_backend)`;
`Transition` comes from `dicode.network` (ppo_tr.py:15). There is NO external
transition-injection surface: make_train hardcodes env construction
(MultiTaskMiniCraftaxEnv + DistributedMultiTaskOptimisticLogWrapper) and the
loss/GAE/optimizer closures live inside `train()`.

## 5. Simulator path

minicraftax env is fully JAX and batchable; `env_restore.stack_env_states` /
`slice_env_state` already support batched freeze/restore; `BranchSearchRunner`
already restores archive states and runs branches with a bound student.

## 6. Major blockers for the litesim refactor

1. No external-batch injection in ppo_tr -> PPOBridge must reuse the backend ABC
   (`policy_forward_eval/train`) and mirror the canonical objective OUTSIDE
   ppo_tr, leaving ppo_tr byte-identical (baseline compatibility).
2. `slowgru_runtime` is NOT importable in the local venv (server-only); local
   vertical slice needs a labeled slice-student implementing the same backend
   surface; SlowGRU validation deferred to server commands.
3. Tier3 dark-corridor world generator not registered locally; tier registry
   must be config-driven with a Tier3-front entry validated on the server.
4. LLM clients exist (`_e3_real_llm_clients.py`) but must stay out of the hot
   loop (E3_NO_LLM).

## 7. Recommended minimal modification points

- NEW package `src/dicode/e3_litesim/` (measurement/data/learning/runtime/
  scheduler/orchestration/diagnostics) wrapping: backend ABC, env_restore,
  student_adapters, minicraftax envs, RunStateCheckpointManager.
- NO changes to ppo_tr.py, training_backend*.py, student_adapters, minicraftax.
- Tests under `tests/e3_litesim/`; benchmark `tools/benchmark_e3_litesim.py`;
  artifacts `artifacts/e3_litesim/<RUN_ID>/`.