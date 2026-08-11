# Static-LLM-UED V1 — Design Contract and Constraint Manifest

Status: FROZEN for branch `henry/static-llm-ued-v1` (base `9eca2de`).
Any deviation requires a new recorded decision; this file is the audit baseline.

## 1. Research question

WITHOUT simulator probe feedback, WITHOUT environment-state save/restore, and
WITHOUT a world model, can an LLM design effective UED curricula using ONLY:

1. the Student's normal training performance (per-task success rates,
   achievement success rates, episode lengths, learning-progress records stored
   in the task archive `performance_history` by the normal training loop);
2. behavior summaries captured from normal training windows;
3. capability profiles derived only from (1) and (2);
4. static Craftax / minicraftax rule and API knowledge baked into prompts?

"Effective" is ultimately judged by the frozen formal evaluation tooling, which
this branch never touches and whose outputs never flow back into the teacher.

## 2. Definition of "static / pure-LLM" (invariants I1–I7)

- **I1** The LLM performs exactly two functions: analyze the Student, and author
  environment designs / environment code. Nothing else.
- **I2** NO per-candidate multi-episode simulator probe. Legality of a candidate
  is established ONLY by `EnvGenerator.check_compilation` (single CPU jit
  reset+step + dtype check).
- **I3** NO candidate success-rate, regret, or partial-progress probe feedback is
  ever given to the LLM. Compilation pass/fail is never included in any prompt.
- **I4** NO Frontier Archive is used or consulted.
- **I5** NO intermediate EnvState save/restore.
- **I6** NO multi-branch simulator search over environment states.
- **I7** NO world model or imagined rollout of any kind.

Student training and formal evaluation still run in the real Craftax simulator;
only the TEACHER is static.

## 3. Forbidden LLM content (guard-enforced, fail-closed)

The LLM (any role) must NOT, in any output that enters the system:

- F1. prescribe action sequences, routes, or waypoints for the Student;
- F2. tell the Student what to do next step-by-step;
- F3. modify or specify reward functions (env code reward tampering);
- F4. provide expert trajectories or demonstrations;
- F5. consume or reference formal-evaluation trajectories;
- F6. emit hidden states, logits, or policy parameters;
- F7. directly modify the Student policy / optimizer.

`teachers/static_llm/guards.py` implements deterministic scanners for F1–F7 and
returns fail-closed `GuardDecision` objects with greppable codes.

## 4. Provenance enum (data-admissibility contract)

`teachers/static_llm/schemas.py::Provenance`:

| Member | Admissible as teacher evidence? | Meaning |
|---|---|---|
| `TRAINING` | YES | metrics produced by the normal PPO training loop on curriculum tasks |
| `NORMAL_TRAINING_FEEDBACK` | YES | per-session training-window metrics + original-task `skill_*` success rates extracted from the training window (NOT a formal evaluation) |
| `FORMAL_FRONT` | NO — fail-closed reject | formal FRONT evaluation output |
| `FORMAL_BACK` | NO — fail-closed reject | formal BACK evaluation output |
| `FORMAL_FULL` | NO — fail-closed reject | formal FULL evaluation output |

Rules:
- Formal evaluation data never enters BehaviorDiagnostician, CurriculumDesigner,
  TaskGenerator, TaskSelector, archive priority updates, or the Student optimizer.
- Formal evaluation is read-only and used only as the final external judgement.
- `StaticLLMGenManager.evolve_tasks` IGNORES its `global_agent_profile`/metrics
  argument entirely, because on the resume path that argument carries formal
  `run_session_evaluation` output shaped identically to training metrics
  (see audit hazard H6). Evidence enters only through
  `observe_session_feedback` (stamped `NORMAL_TRAINING_FEEDBACK`) and the archive
  `performance_history` (stamped `TRAINING`).
- Insufficient or ambiguous provenance ⇒ fail-closed: the data is rejected and
  the previous diagnosis/intervention is reused.

## 5. Roles

Default (always on):
- **BehaviorDiagnostician** — pure analysis of the Student evidence snapshot;
  outputs ≤3 prioritized weaknesses, ≤3 hypotheses per weakness (≤6 hypotheses
  total), a reuse-previous-direction flag, and an overall confidence.
- **CurriculumDesigner** — maps a diagnosis to ≤8 intervention families (≤3
  mutated axes each) plus ≤2 exploration proposals; each family names the target
  achievement/skill chain, the bottleneck to expose, allowed scaffolding, what the
  Student must accomplish unaided, axes held constant vs mutated, and env-code
  constraints.

Conditional (default OFF, `adversarial_reviewer: false`):
- **AdversarialReviewer** — reviews proposed designs for contract violations.

All roles run temperature 0 through a pinned manifest (role, model id,
prompt_version, schema_version) and a content-addressed plan cache with
replay-only mode for tests and smoke.

## 6. LLM invocation gate

`LLMInvocationGate.should_reinvoke(state) -> (bool, reason)`. Re-invoke ONLY on
one of eight deterministic, documented conditions; otherwise REUSE the previous
diagnosis/intervention/templates without any LLM call; insufficient/ambiguous
evidence fails closed to reuse with a recorded reason.

| Code | Condition |
|---|---|
| `FIRST_WINDOW` | no existing plan |
| `CAPABILITY_SHIFT` | capability profile moved beyond threshold since last invocation |
| `NEW_FAILURE_PATTERN` | repeated failure/behavior pattern not covered by existing hypotheses |
| `INTERVENTIONS_EXHAUSTED` | all current families applied, no untried axes remain |
| `STAGNATION` | K consecutive windows without learning-progress improvement |
| `FORGETTING_REGRESSION` | previously mastered skill regressed beyond threshold |
| `EXPLORATION_SLOT_AVAILABLE` | untried exploration proposal and budget permit it |
| `CURRICULUM_DRIFT` | active curriculum drifted away from the templates in force |

## 7. Curriculum shape (12 + 4)

- Batch = 12 LLM-authored dynamic tasks + 4 global anchors.
- Anchors = `original_craftax` (always appended last by `run_session_training`,
  required by `force_include_achievements_indices=[num_tasks-1]`) + 3 frozen seed
  anchors (default `task_1, task_2, task_3` = collecting/combat/crafting).
- Anchor weight is ALWAYS > 0 (structural positivity + runtime assertion).
- Selection of dynamic tasks: transparent deterministic quota + dedup. NO Soft
  Copeland this round (no compliant candidate-value inputs exist by design); no
  import of `d052/bagr_ued`.

## 8. Minimal smoke disclaimers (mandatory in every smoke report)

1. The smoke uses a MOCK / REPLAY LLM client: outputs are pre-authored fixtures,
   NOT real LLM generations. No claim about real-LLM capability is made.
2. Passing `check_compilation` proves only that an env compiles, resets, and
   steps once with valid dtypes; it does NOT prove the task is meaningful,
   solvable, or well-formed as a curriculum element.
3. A single (or few) real Student update(s) at minimal scale demonstrates only
   pipeline viability; it does NOT demonstrate performance improvement.
4. The smoke is NOT a SOTA, course-value, or curriculum-effectiveness claim.

## 9. Isolation statement

This branch does not modify, run, read from, or write to CC3
(simulator-feedback LLM-UED), CC1 (simulator-led Frontier-UED), BA-CWM shadow,
or tier3 formal-evaluation branches/artifacts. Its outputs live exclusively
under `outputs/static_llm_ued/<run_id>/`. No API keys, `.env` files,
checkpoints, large logs, caches, training datasets, or formal-evaluation
trajectories are committed.

## 10. Verification gates before merge-ready claim

- 15 tests green with NO real external API calls.
- Legacy default path byte-identical (compose equality, distribution equality,
  fake-GenManager legacy-branch equality).
- Formal-eval injection rejected (including simulated resume path).
- Smoke executed at whatever tier the environment allows, reported honestly with
  `student_update_skipped` status and the four disclaimers.
- `INDEPENDENT_AUDIT_REQUIRED=true`: nothing on this branch is self-certified for
  research claims.
