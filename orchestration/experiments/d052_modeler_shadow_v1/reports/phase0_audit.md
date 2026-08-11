# D052-V2 Modeler Shadow — Phase 0 Read-Only Audit

**STATUS: PHASE_0 = PASS** (with documented raw-data boundaries)
Date (server local): 2026-07-26 · Operator: henry/d052-modeler-shadow-v1

## A. Server & Git
- host: `i-00000226`, user `oseasy`
- GPUs: 4× RTX A6000, all IDLE (1 MiB used, 0% util) — no training running; none touched.
- tmux sessions observed (NOT touched): cc_gpu0, director, p2_director.
- git repo root: `/home/oseasy/experiments/mechanism_UED_continuation_20260715/mechanism_UED`
- base branch: `Henry-branch` @ `e6765fb26d2cca8b7a49bd3b1ce6a2626d72f69b` (clean tree)
- work branch created: `henry/d052-modeler-shadow-v1`
- remote: github gregjones11235/mechanism_UED (no push performed)
- NOTE: the D052 *pipeline* code lives OUTSIDE the git repo, under the project root
  `workers/` and `orchestration/` (not version-pinned by the repo).

## B. Environment
- Documented conda env `sfl` does NOT exist. Actual project env: **`dicode310`**
  (Python 3.10.20, jax 0.6.0, flax 0.10.7, optax 0.2.8, 4 CUDA devices).
- `anthropic` module absent — irrelevant: D052 LLM calls use a direct-HTTP client
  (`api()` in the launcher) hitting DashScope/DeepSeek/BigModel (OpenAI-compatible).
- Phase 0–2 are offline; no GPU consumed.

## C. Chosen cell / round / checkpoint
- cell: **soft_copeland_x_original** (Soft Copeland aggregation × Original training)
- seed run: **seed0_1784462982** (4 rounds COMPLETED, gate_passed, gate_violations=0,
  no_api_ppo=true, llm_judgments=true, steps_per_round=24576, total=98304)
- **shadow target round: 4** — the ONLY round with real per-episode Student evaluation.
- checkpoint↔round map: round_N ↔ step N×24576 → ckpts {24576,49152,73728,98304}.
- round-4: snapshot_hash `96fc8a200b1ce01f`, prev `17341295acbd51be` (round-3).
- manifest commit `e6765fb` == current git HEAD (code/artifact aligned).

## D. Recomputation proof (deterministic, original code)
Using `workers/gpu0_original/.../scripts/launch_d052_pure_dynamic_enhanced.py`
formulas + `dicode.mechanisms.aggregation._aggregate_soft_copeland`:
- **pool_hash** = sha256(json.dumps(sorted spec_payload[id,tp,achs,prov]))[:16]
  → round1 `46ee15406386af83` ✓ and round4 `1902b71a5d86fa00` ✓ reproduce EXACTLY.
- **Soft Copeland selected-8** recomputed from the 96 frozen judgments reproduces the
  recorded selected-8 EXACTLY (round1 and round4).
- **selection_hash** = sha256(json.dumps(sorted(selected_ids)))[:16]
  → round1 `5ae0dc9fa74e797e` ✓ and round4 `6a285e8fe13731d5` ✓ reproduce EXACTLY.
- Weights: progression 0.34 / retention 0.33 / novelty 0.33 / critic 0.01 / monopoly 0.01;
  role fields tutor→progression_score, critic→critic_penalty, explorer→novelty_score.
- Per-role models (from archive): tutor=qwen-flash-2025-07-28(qw),
  critic=deepseek-v4-pro(ds), explorer=glm-4-flash(gl). temperature=0.

## E. Salted-hash hazard (located)
- `single_director_20260722/d052_eval/d052_eval_pilot.py` **L139**:
  `ALL_A[hash("%s_%s"%(cid,a)) % len(ALL_A)]` — Python built-in `hash()` (per-process
  salted) maps enhanced-cell `target_achievements` strings → achievement index.
  UNRECOVERABLE across processes ⇒ `success_mode = "UNDEFINED"` for enhanced cells;
  intended-target success rate cannot be computed. Confirmed in eval records
  (`success:"UNDEFINED"`, `target_achievement:["A","B"]` placeholders).
- Canonical achievement IDs ARE recoverable from `craftax.craftax.constants.Achievement`
  (67 achievements). RELIABLE success uses fixed `ALL_A[0]`=COLLECT_WOOD only.

## F. Available data (round 4)
- frozen_candidate_pool.json (32 candidates, canonical target_achievements strings)
- judgment_cache.jsonl (96 = 32 tutor + 32 critic + 32 explorer), gen_cache.jsonl (32)
- d052_manifest.json (selected-8, pool_hash, selection_hash, snapshot chain)
- per-episode eval: `single_director_20260722/d052_eval/soft_copeland_x_original/round4_selected8.jsonl`
  = 64 records (round-4 selected-8 tasks × 8 episodes) with REAL fields:
  return, episode_length, death, timeout, achievements (canonical names), achievement_count,
  provenance (checkpoint_sha256, evaluator_sha256). Eval task_ids == round-4 selected-8 EXACTLY.
- deterministic base extracted → `outputs/student_evidence_base.json`:
  64 eps, mean_return 1.147, death_rate 1.0, timeout_rate 0.0, deepest reliable
  achievement WAKE_UP; breakpoints PLACE_TABLE/COLLECT_DRINK/MAKE_WOOD_PICKAXE/PLACE_PLANT.

## G. Missing / boundary data
- intended-target SR: UNDEFINED (salted hash). Only empirical per-achievement COMPLETION
  rate is a non-fabricated SR proxy.
- best_sr / recent_delta / retention / trajectory status (IMPROVING/STALLED/FORGETTING):
  INSUFFICIENT_EVIDENCE — single cross-sectional snapshot; rounds 1–3 have NO per-episode eval.
- eval coverage: round-4 selected-8 tasks only (8×8). The 24 non-selected candidates are
  architecturally NOT evaluable (enhanced eye(8) one-hot conditioning; all-32 eval N/A).
  (This does NOT block the shadow: roles re-JUDGE all 32 via LLM; only Student *policy*
  eval is selected-8-only.)
- in-run snapshots carry only aggregate transition_reward_mean/std + param_leaves (sparse).

## H. Artifact SHA256 (round 4)
- frozen_candidate_pool.json eff6aafab50a242c961b1c7267ebfcf62004d2cc11318091acfbc08d48339385
- judgment_cache.jsonl      362b9119a7fa647091c8900770d0ab4cef2d5f9b378220594482bc5031aaad9a
- d052_manifest.json        afd4f2ca738db4293232be9076fde7a582863f29a4058ed587bedc7c696680fc
- gen_cache.jsonl           405939751bf359aec93bee406b0511829e44134d94305ac46e112c01750e0299
- eval round4_selected8     8963f377fef058aa59d6680068c1651e7fdade68e22aafea20783ffef3285451
- eval round4_summary       edf0c153f601a8cbd11058c48c8b003f10b7752eb76bd49044b84bd6c5d12215

## I. LLM credential availability (blind-sourced; values never read/printed)
- DASHSCOPE_API_KEY (tutor/qwen): SET · DEEPSEEK_API_KEY (critic): SET
- ZHIPUAI_API_KEY (explorer/glm): provided via EXP_GLM_API_KEY → mapped; base URLs
  all match launcher providers (dashscope / deepseek / open.bigmodel.cn).
- Source loaders: ~/.qwen_env, ~/.claude_deepseek_env, ~/.config/dicode/experiment_llm.env.

## J. Verdict
PHASE_0 = **PASS**. Proceed to Phase 1 (8-candidate pilot), bounded by:
Modeler uses ONLY round-4 real evidence; marks intended-target SR, retention, delta,
best_sr and trajectory statuses as INSUFFICIENT_EVIDENCE/UNDEFINED; fabricates nothing.
