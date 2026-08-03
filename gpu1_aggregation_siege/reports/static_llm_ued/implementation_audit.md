# Static-LLM-UED V1 — Implementation Audit (audit-first, pre-code)

- Branch: `henry/static-llm-ued-v1` @ base `9eca2de914068a33e500e2ad90d50f48e6e4e632` (`Henry-branch` local tip; 10 ahead / 0 behind origin at audit time)
- Worktree: `mechanism_UED_static_llm_ued_worktree` (independent, clean; cut from the base SHA, NOT from the previously checked-out HEAD)
- Audit date: 2026-08-03. All line numbers below were re-verified in THIS worktree on the audit date (not copied from another branch).
- Scope root: `gpu1_aggregation_siege/` (all training/generation code lives here).

## 1. Research contract this implementation must satisfy

Research question: WITHOUT simulator probe feedback, WITHOUT state save/restore,
WITHOUT a world model, can an LLM design effective UED curricula using ONLY the
Student's normal training performance, behavior summaries, capability profiles,
and Craftax rule knowledge?

Hard constraints (must hold for every artifact on this branch):

| # | Constraint | Enforcement point |
|---|------------|-------------------|
| 1 | LLM only analyzes Student and authors env designs | `teachers/static_llm/controller.py` is the only LLM caller |
| 2 | NO per-candidate multi-episode simulator probe | legality = `EnvGenerator.check_compilation` only (single jit reset+step, §4) |
| 3 | NO candidate success-rate / regret / partial-progress probe feedback to LLM | compile pass/fail never re-enters any prompt; controller has no probe API |
| 4 | NO Frontier Archive | static teacher never reads/writes frontier structures; no import of simulator-frontier code |
| 5 | NO intermediate EnvState save/restore | no `save_state`/`restore_state` calls anywhere in the new package |
| 6 | NO multi-branch simulator search | single `generate_code_only` pass per candidate; no search loop |
| 7 | NO world model / imagined rollout | no dynamics model in the package |
| 8 | LLM must NOT give action sequences/routes/waypoints, next-step instructions, reward edits, expert trajectories, hidden states/logits, or policy edits | deterministic scanners in `teachers/static_llm/guards.py`, fail-closed |
| 9 | Formal-evaluation data never enters teacher/selector/archive-priority/optimizer | `Provenance` enum + leakage guard (`schemas.py`/`guards.py`); `evolve_tasks` metrics arg ignored |

## 2. Branch topology and isolation statement

- This branch modifies ONLY: new package `src/dicode/teachers/static_llm/`, four
  integration files (`setup.py`, `training.py`, `evolution_efficient.py`,
  `run_dicode.py`) behind optional kwargs / duck hooks / `teacher_type` branches,
  new config files (`conf/teacher/`, `configs/static_llm_ued.yaml`), new tests
  (`tests/static_llm/`), and `reports/static_llm_ued/`.
- NOT touched: CC3 branches (`henry/ba-bagr-ued-review-board-v2`), CC1 branches
  (`henry/simulator-frontier-foundation-codex`), `henry/ba-cwm-ued-shadow-v1`,
  tier3 evaluation branches, or any of their output directories.
- `d052/bagr_ued` and `d052/ba_cwm_ued` do NOT exist on the base commit
  (verified: `ls` fails); nothing is imported from them. `d052/` on this branch
  contains only `achievements`, `cells`, `docs`, `evaluation`.
- Outputs are isolated under `outputs/static_llm_ued/<run_id>/` via
  `hydra.run.dir` override (`hydra.job.chdir: true` ⇒ cwd = run dir ⇒
  `rl_checkpoints/` and `task_graph.graphml` land inside the run dir).

## 3. Reuse anchors (verified line numbers, base commit)

### 3.1 `src/dicode/dreaming/gen_manager.py`
| Anchor | Line | Notes |
|---|---|---|
| `class Task` | 51 | loads env file, wraps `MiniCraftaxTrain` |
| `class TaskArchive` | 97 | NetworkX DiGraph + `threading.Lock` (`._lock`, :111) |
| `TaskArchive.load_graph` | 113 | reads `config.graph_path`; seeds from `config.example_paths` as `task_{i+1}` (:162-179) with status `seed` |
| `TaskArchive.save_graph` | 186 | **hardcodes `"task_graph.graphml"` in cwd** (:188) — see hazard H3 |
| `TaskArchive.record_new_task` | 215 | adds node `status="desc_generated"` + parent edges — registration used by static controller |
| `TaskArchive.update_node_performance` | 254 | appends `{"session": idx, **metrics}` to `performance_history` (lock-guarded) — primary Student-evidence source |
| `TaskArchive.get_max_session_idx` | 352 | resume counter |
| `TaskArchive.set_task_active_status` | 461 | atomic active-count bookkeeping |
| `class TaskSelector` | 509 | `.embedding_model` attr (:523); similarity selection calls embedding API (:605) — NOT used by static teacher |
| `class TaskGenerator` | 621 | `evolve_mastered` :711 (LLM call 1; consumes `global_agent_profile` — provenance trap, see H6); `_organize_data` :950 registers nodes and builds worker-result dicts; `task_num_counter` :654/656 |
| `class EnvGenerator` | 1014 | `__init__` imports prompt modules from config (:1031-1041) |
| `EnvGenerator.generate` | 1043 | DEAD/HAZARD — see H1; never called by static teacher |
| `EnvGenerator.generate_code_only` | 1184 | single LLM pass, no reflection, thread-safe — the ONLY code-gen entry the static teacher uses; expects `[{"task","description","examples"}]`, returns `{task_id: code|None}`; extracts `<code>...</code>` via `_extract_file` (:1319) |
| `EnvGenerator.check_compilation` | 1252 | THE static legality gate: temp file → CPU-pinned `Task(...)` → `MiniCraftaxTrain` → jit reset+step + inventory int32 dtype check (:1290-1305). Returns `(bool, str)`. No candidate-value metrics |
| `class GenManager` | 1336 | `__init__` :1343 builds `archive` (:1380), `selector` (:1381), `task_generator` (:1382), `env_generator` (:1383), `session_idx = archive.get_max_session_idx()+1` (:1385). Constructs `LLM` objects with `local` provider — `AsyncOpenAI` client creation performs NO network I/O (llm.py:42), so `GenManager.__init__` is offline-safe |
| `GenManager.evolve_tasks` | 1387 | signature `(dict_of_tasks, global_agent_profile) -> list[dict]`; static subclass overrides this |

### 3.2 Worker-result dict schema (contract, unchanged)
Produced by `TaskGenerator._organize_data` + `GenManager.evolve_tasks`, consumed by
`run_dicode._process_worker_results` (:295) and `evolution_efficient.evolve_and_validate_tasks` (:149):
`{generated_task_id, parent_task_id, parent_task, evolution_type, reasoning, docstring, examples, code_string, compiled(None at handoff), code, error}`.

### 3.3 Training / orchestration anchors
| Anchor | Line | Notes |
|---|---|---|
| `setup.setup_experiment` | 44 | **teacher injection point: `gen_manager = GenManager(config)` :64** |
| `setup._load_agent_state` | 155 | reads `config.gen_manager.embedding_model.embedding_size` even in one_hot mode (:160-161) ⇒ `gen_manager` config group must stay complete |
| `setup.run_initial_seed_training` | 172 | runs `max_updates_per_session*2` updates (:241); trains seeds + OriginalTask; has its OWN copy of `_calculate_task_distribution` (:350) — left untouched by this work |
| `training.run_session_training` | 39 | loads sampled tasks (:78), appends `OriginalTask` LAST (:87-88), `_create_achievement_masks` (:92), embeddings (:97), distribution (:102), `run_training_session` (:113), metrics with `force_include_achievements_indices=[num_tasks_in_session-1]` (:155) ⇒ original must stay last |
| `training._calculate_task_distribution` | 328 | single original anchor at `original_task_proportion` (default 0.2), rest uniform |
| `training._update_archive_with_metrics` | 375 | `update_node_priority_score` + `update_node_performance(session, {task_id: metrics})` + `session_last_trained` — fills the Student-evidence history |
| `evolution_efficient.dispatch_evolution_worker` | 95 | `num_to_evolve = num_generation_tasks + additional_num_parents` (:125-127); `select_tasks_for_evolution` (:128); returns None if no tasks (:132-134) — starvation risk, see H7 |
| `evolution_efficient.evolve_and_validate_tasks` | 149 | `gen_manager.evolve_tasks(tasks, metrics)` (:172) then parallel `check_compilation` in a ThreadPool (:182-199), sets `res["compiled"]/res["error"]` |
| `run_dicode.main` | 40 | seed gate :64; resume branch :84-96 (`run_session_evaluation` → `evaluation_metrics`); Step1 sync :127-152; Step2 dispatch :155-159; Step3 batch :161-178 (`target=training_sample_size_n`, `num_to_sample=max(0,(target-1)-new)`, `sampled = new + archive_sample`, skip-if-empty :175-178); Step4 :191; Step5 activation :218-233; checkpoint+`save_graph` :260-261; `session_idx += 1` :285 |
| `run_dicode._process_worker_results` | 295 | filters `compiled`, caps at `num_generation_tasks` (:323) with `random.sample` if over limit (:325) — see H5; registers nodes via `update_node_*` (:337-342) |
| `selection.sample_tasks_for_training` | 20 | aggregation duck hook precedent :41-49; PLR sampling from active set; `np.random.choice` :96 |
| `selection.select_tasks_for_evolution` | 149 | `strict` ⇒ frontier A/B + children<5 + no viable child (:173-181); fallback status "A" only (:243-247); `_replicate_to_fill` :289 |
| `task_utils.load_tasks_from_env_codes` | — | exec code strings → Env classes (used by `run_session_training` and seed phase) |

### 3.4 Env code contract (candidates must satisfy)
BaseTask subclass named `Env` + docstring (Objective/Description/Relevant
Achievements/Completed Achievements/World) + `relevant_achievements` /
`completed_achievements` / `label` / `get_task_params` / `generate_world`.
Templates: `src/minicraftax/tasks/seed_tasks/{collecting,combat,crafting,survive,original}.py`.
Seed archive ids (from `conf/gen_manager/default.yaml example_paths` order):
`task_1=collecting, task_2=combat, task_3=crafting, task_4=survive`.

### 3.5 Deterministic-replay precedents to mirror
- `src/dicode/immutable_cache.py` — content-addressed immutable cache keys.
- `src/dicode/model_manifest.py` — pinned role/model/temperature-0/prompt_version/schema_version manifest (already has a `generator` role).
- `src/dicode/mechanisms/llm_cache.py` — JSONL LLM cache.
- `src/dicode/siege/production_dispatcher.py` — frozen pool artifact + pool hash, hard-fail on cache miss.
- `src/dicode/mechanisms/llm_costs.py::LLMCostTracker.can_call` — budget gate precedent.
- `src/dicode/mechanisms/llm_roles.py` — strict-JSON + repair parser precedent.
- Canonical schemas: `d052/schemas/common.py::CanonicalModel` (pydantic v2, `extra="forbid"`), `validate_sha256_hex`, `validate_finite`.

### 3.6 LLM client surface (`src/dicode/dreaming/llm.py`)
`LLM.query(system_prompt, user_prompts) -> list[{"content": str, ...}]` — the duck
surface the static teacher's offline clients must replicate. `query` supports only
`local`/`gemini` providers (:221-227); NO mock/cache/replay/seed exists.
**This file is NOT modified**; the static teacher swaps in its own client objects
(duck-typed `query`) after `GenManager.__init__`.

## 4. Static legality (constraint #2/#3 enforcement)

The ONLY legality gate is `EnvGenerator.check_compilation(code) -> (bool, str)`
(:1252): temp-file import + `Task` construction + `MiniCraftaxTrain` wrap +
jit-compiled reset+step on CPU + inventory int32 dtype assertion. It returns no
performance value of any kind; its boolean outcome is never fed back to any LLM
prompt. No multi-episode probe, no success-rate estimation, no candidate scoring.

## 5. Hazards found in baseline (documented; NOT fixed unless required)

- **H1 — `EnvGenerator.generate` is dead code with a latent crash**: references
  `self.config.num_reflections_max` (:1158) which is absent from
  `conf/gen_manager/default.yaml` (no `num_reflections_max` anywhere in `conf/`)
  ⇒ `ConfigAttributeError` if ever called. Additionally its reflection loop feeds
  compile errors back to the LLM, which would violate the static contract.
  Disposition: NEVER call; static teacher uses `generate_code_only` only. Not fixed.
- **H2 — `LLM.query` raises for `openai`/`openrouter` providers** (llm.py:227) and
  `get_embedding` raises `NotImplementedError` for gemini (:235). Disposition:
  static teacher uses its own offline client; config keeps `provider: local` so
  `GenManager.__init__` stays constructible. `llm.py` untouched.
- **H3 — `TaskArchive.save_graph` hardcodes `"task_graph.graphml"` in cwd** (:188)
  while `load_graph` reads `config.graph_path` (:121). With `hydra.job.chdir: true`
  both resolve inside the run dir. Disposition: rely on `hydra.chdir`; tests use
  tmp-path graph files. Not fixed (shared file; fixing is out of scope and risky).
- **H4 — conditioning branch quirk**: `training._generate_embeddings_for_session`
  tests `config.training.condition_on_task == "embedding"` (:311) but
  `condition_on_task` is a boolean ⇒ always takes the one_hot path; `setup.py`
  (:329) correctly tests `conditioning_type`. Defaults (`condition_on_task: true`,
  `conditioning_type: one_hot`) hit one_hot either way. Disposition: keep one_hot,
  document, do not fix. One-hot width = `len(Achievement)` = 67.
- **H5 — nondeterministic cap**: `_process_worker_results` uses `random.sample`
  when compiled count exceeds `num_generation_tasks` (run_dicode.py:325).
  Disposition: static teacher emits exactly `num_generation_tasks` (=12) candidates
  so the cap never triggers.
- **H6 — provenance trap on the `evolve_tasks` metrics argument**: the resume path
  (run_dicode.py:84-96) runs `run_session_evaluation` (formal 1024×8192 evaluation)
  and passes its output as `evaluation_metrics` → `dispatch_evolution_worker`
  (:157-159) → `evolve_and_validate_tasks(metrics)` → `gen_manager.evolve_tasks(tasks, metrics)`.
  The same parameter also carries benign training-window metrics on the fresh-start
  path, so the two provenances are indistinguishable by shape. Disposition: the
  static teacher's `evolve_tasks` IGNORES this argument entirely (fail-closed) and
  sources evidence only from `observe_session_feedback` + archive
  `performance_history`. A guard test simulates the resume-path injection.
- **H7 — frontier starvation for the static teacher**: `dispatch_evolution_worker`
  with `parent_selection: strict` needs status-A/B frontier tasks; if seeds all
  categorize C/D, `select_tasks_for_evolution` returns `[]` and the teacher never
  runs. Disposition: duck hook `select_context_tasks` on the gen_manager
  (`getattr`-fallback in `dispatch_evolution_worker`; legacy path unchanged).
- **H8 — embedding config read in one_hot mode**: `setup._load_agent_state` reads
  `config.gen_manager.embedding_model.embedding_size` (:160-161) regardless of
  conditioning type. Disposition: keep the full `gen_manager` config group for the
  static teacher (no config deletion).
- **H9 — nondeterministic archive sampling**: `sample_tasks_for_training` uses
  `np.random.choice` (selection.py:96). Disposition: the static batch path uses
  deterministic quota+dedup for dynamic slots and never routes anchors through
  archive sampling.
- **H10 — external `craftax` (pip) is a hard import dependency**: `training.py:14-15`,
  `setup.py:17-18`, `gen_manager.py:29`, `minicraftax/envs/*`, `ppo_tr.py`,
  `scoring.py`, `evaluation/*`. Declared in `gpu1_aggregation_siege/pyproject.toml`
  dependencies. Local isolated venv has jax/jaxlib 0.4.35 CPU stack but craftax is
  NOT yet installed ⇒ smoke preflight gate required before any real Student update.

## 6. Missing pieces (implemented by this branch)

1. No teacher selection mechanism: `conf/config.yaml` has no `teacher` group; no
   `teacher_type` dispatch in `setup_experiment`.
2. No 12+4 (dynamic + anchor) batch split: `run_session_training` only appends the
   single `original_craftax` anchor; `_calculate_task_distribution` supports only
   one anchor proportion.
3. No provenance enum / formal-evaluation leakage guard anywhere in `src/dicode`.
4. No offline/replay LLM client (`llm.py` has no mock/cache/replay mode).
5. No invocation gate: legacy pipeline re-invokes the LLM every evolution window
   unconditionally.
6. `hydra.run.dir` default (`outputs/${now:...}`) is not teacher-scoped.

## 7. Configuration facts (defaults on base commit)

- `conf/config.yaml`: defaults = dicode_manager/gen_manager/training/evaluation/
  validation/visualize/aggregation + `_self_`; `hydra.job.chdir: true`;
  `hydra.run.dir: outputs/${now:%Y-%m-%d_%H%M%S_%f}`; `use_wandb: true`; `seed: 42`;
  `ablation: false`; `checkpoint_dir: rl_checkpoints/`. NO teacher group.
- `conf/dicode_manager/default.yaml`: `max_updates_per_session: 100`,
  `evolution_interval: 2`, `training_sample_size_n: 16`, `active_task_capacity: 100`,
  `score_function: learnability`, `original_task_proportion: 0.2`,
  `num_generation_tasks: 10`, `additional_num_parents: 2`, `parent_selection: strict`,
  `mode: None`.
- `conf/gen_manager/default.yaml`: `graph_path: task_graph.graphml`;
  `example_paths`: collecting/combat/crafting/survive (⇒ `task_1..task_4`);
  `num_examples: 6`; prompts under `dicode.dreaming.prompts.cl_.*`;
  llm sub-groups at `conf/gen_manager/llm/{local_gen,local_embed,...}.yaml`
  (provider `local`, temperature 0.6 — static teacher pins its OWN temp-0 manifest).
- `conf/training/default.yaml`: PPO `num_envs: 1024`, `num_steps: 128`,
  `update_epochs: 4`, `num_minibatches: 8`; GTrXL `num_layers: 2`, `embed_size: 256`;
  `condition_on_task: true`, `conditioning_type: one_hot`.
- `conf/evaluation/default.yaml`: `num_steps: 8192`, `num_envs: 1024` (formal eval
  scale — never a teacher input).
- Packaging: `pyproject.toml` `[tool.hatch.build.targets.wheel] packages =
  ["src/dicode", "src/minicraftax"]` ⇒ `src/dicode/teachers/` is included
  automatically; no packaging change needed.

## 8. Test conventions (base commit)

Flat files `gpu1_aggregation_siege/tests/test_*.py`, each starting with
`sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))`;
no `tests/` conftest. New tests live under `tests/static_llm/` with their own
conftest applying the same sys.path insertion, and run via
`<venv>/Scripts/python -m pytest tests/static_llm -k static_llm` with
`JAX_PLATFORMS=cpu`. No test may call a real external API.

## 9. Architecture decision (from validated plan)

Injection seam = `StaticLLMGenManager(GenManager)` reusing `GenManager.__init__`,
overriding `evolve_tasks` (worker schema unchanged, metrics arg ignored), adding
`select_context_tasks`, `observe_session_feedback`, `anchor_seed_ids`,
`select_replay_fill`. Default path stays byte-identical: every shared-file change
is an optional kwarg, a `getattr` duck hook with verbatim legacy fallback, a
`teacher_type` branch, or a literally-preserved legacy code branch.
