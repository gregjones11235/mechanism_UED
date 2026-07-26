# Base GTrXL matched-Replay freeze matrix (CC4 premerge hardening -- eight)

- UTC: `2026-07-26T13:33:22Z`
- **BASE_GTRXL_MATCHED_REPLAY_CONTROL = READY_NOT_AUTHORIZED** (max allowed status)
- match_verdict: **SPEC_SINGLE_DIFFERENCE_CONFIRMED + FIELD_VERIFICATION_PARTIAL**
- authoritative field counts: IDENTICAL=23, UNVERIFIED_MUST_FREEZE=5, DIFFERENT=0
- is NOT_MATCHED? **False** (different=0 => no major mismatch => stays READY_NOT_AUTHORIZED)
- Config audit only; NO training launched; NO auto-modify+run.

## Five must-freeze identity fields (before ANY run)
- replay.L_SEQ (129 vs 512 conflict; NOT auto-picked; must freeze on server, MISS-6)
- ppo.total_environment_steps (pin 24576 screen vs 98304 longrun; match controlled P2 arm)
- evaluation.evaluation_seed (pin which P2 line; seed42 vs seed100000 are distinct world sets, never pooled)
- evaluation.evaluator_sha256 (must equal canonical frozen evaluator eval_phase2_unified.py#22451402... at run)
- evaluation.world_set_hash (materialize on a JAX+craftax host; currently BLOCKED_SOURCE_UNVERIFIED)

## L_SEQ -- NOT auto-picked
- NOT_AUTO_PICKED -- conflict 129 (run_p2_full_smoke.py:66 'formal run uses 512') vs 512 (RMT16/P2-Full-A v2.1 frozen); MUST be frozen on server and recorded in provenance.

## Authoritative field table (28 rows)

| field | matched-control | real P2-Full-A | verdict | evidence |
|---|---|---|---|---|
| `network.class` | ActorCriticTransformer (Base GTrXL) | ActorCriticTransformer (Base GTrXL trunk) | **IDENTICAL** | GIT_PROVENANCE |
| `network.long_memory_module` | NONE | NONE (P2-Full-A has no long-memory module; W512/RMT ADD one) | **IDENTICAL** | GIT_PROVENANCE |
| `init.checkpoint` | teacher17500 | base_ckpt_v7fix55_armA_s0/rl_checkpoints/17500 | **IDENTICAL** | RAW_FILE (P2 summary base_checkpoint) |
| `init.params_sha256` | d4e85af58b7f87d6 | teacher_sha=d4e85af58b7f87d6 (LC smoke logs); P2 base = ckpt17500 | **IDENTICAL** | RAW_FILE (LC logs teacher_sha) |
| `ppo.optimizer` | Adam | Adam (ppo) | **IDENTICAL** | GIT_PROVENANCE |
| `ppo.learning_rate` | 2.0e-5 | 2e-05 | **IDENTICAL** | RAW_FILE (P2 manifest lr) |
| `ppo.adam_eps` | 1.0e-5 | 1e-05 | **IDENTICAL** | RAW_FILE (P2 manifest config) |
| `ppo.gamma` | 0.999 | 0.999 | **IDENTICAL** | RAW_FILE |
| `ppo.grad_clip_global_norm` | 1.0 | 1.0 | **IDENTICAL** | RAW_FILE |
| `ppo.num_envs` | 16 | 16 | **IDENTICAL** | RAW_FILE |
| `ppo.rollout_steps` | 128 | 128 | **IDENTICAL** | RAW_FILE |
| `ppo.transitions_per_update` | 2048 | 2048 | **IDENTICAL** | RAW_FILE |
| `replay.enabled` | true | true (replay_samples_drawn>0; format p2_full_a_pure_pickle_v1) | **IDENTICAL** | RAW_FILE (P2 manifest counters) |
| `replay.capacity` | 64 | 64 | **IDENTICAL** | RAW_FILE |
| `replay.vtrace` | true | w_vtrace=0.5 vt_clip[-50.0,300.0] | **IDENTICAL** | RAW_FILE |
| `replay.awr_hindsight` | true | w_awr=0.5; hindsight_eligible=188 | **IDENTICAL** | RAW_FILE |
| `replay.ema.tau` | 0.995 | 0.995 | **IDENTICAL** | RAW_FILE |
| `replay.policy_lag_gate.max_lag` | 16 | 16 | **IDENTICAL** | RAW_FILE |
| `replay.transactional_kl_gate` | true | kl_max=0.05 kl_replay_max=0.05 lambda_kl=0.01 | **IDENTICAL** | RAW_FILE |
| `arch.embed` | 256 | 256 | **IDENTICAL** | RAW_FILE |
| `arch.num_heads` | 8 | 8 | **IDENTICAL** | RAW_FILE |
| `arch.num_layers` | 2 | 2 | **IDENTICAL** | RAW_FILE |
| `arch.window_mem (GTrXL window)` | 128 | 128 (frozen design: 128-step UNCHANGED from ckpt17500) | **IDENTICAL** | RAW_FILE |
| `replay.L_SEQ` | 512 | CONFLICT: W512 repro used 129 (run_p2_full_smoke.py:66 'formal run uses 512'); RMT16/P2-Full-A v2.1 frozen=512; P2-Full-A actual run value NOT in synced manifests | **UNVERIFIED_MUST_FREEZE** | REPORT_ONLY |
| `ppo.total_environment_steps` | 24576 (screen) / 98304 (longrun) | P2 resume ran to 98304; matched control must pin ONE and match | **UNVERIFIED_MUST_FREEZE** | REPORT_ONLY |
| `evaluation.evaluation_seed` | 42 | P2-Full-A levelB 24576 eval summary seed_base=100000 (a DIFFERENT P2 line); multiple P2 eval lines exist | **UNVERIFIED_MUST_FREEZE** | RAW_FILE (P2 summary seed_base) |
| `evaluation.evaluator_sha256` | REQUIRED (at run time) | P2 levelB eval evaluator_sha256=51c37c27...; canonical frozen evaluator = eval_phase2_unified.py#22451402... | **UNVERIFIED_MUST_FREEZE** | RAW_FILE |
| `evaluation.world_set_hash` | REQUIRED (materialized on JAX host) | NOT materialized (JAX absent) => GLOBAL_WORLD_SET_HASH=BLOCKED_SOURCE_UNVERIFIED | **UNVERIFIED_MUST_FREEZE** | RECOMPUTED (blocked) |

## Additional task-requested checks

| field | verdict | detail | evidence |
|---|---|---|---|
| `initialization_params_sha` | **IDENTICAL** | VERIFIED identical (== init.params_sha256 row) | RAW_FILE (LC logs teacher_sha + P2 base_checkpoint) |
| `base_checkpoint` | **IDENTICAL** | == init.checkpoint row | RAW_FILE (P2 summary base_checkpoint) |
| `optimizer` | **IDENTICAL** | == ppo.optimizer row | GIT_PROVENANCE |
| `learning_rate` | **IDENTICAL** | == ppo.learning_rate row | RAW_FILE |
| `rollout_length` | **IDENTICAL** | == ppo.rollout_steps row | RAW_FILE |
| `replay_bundle` | **IDENTICAL** | == replay.* rows (enabled/capacity/vtrace/awr/ema/kl) | RAW_FILE (P2 manifest counters) |
| `vtrace` | **IDENTICAL** | == replay.vtrace row | RAW_FILE |
| `awr` | **IDENTICAL** | == replay.awr_hindsight row | RAW_FILE |
| `ema` | **IDENTICAL** | == replay.ema.tau row | RAW_FILE |
| `kl` | **IDENTICAL** | == replay.transactional_kl_gate row | RAW_FILE |
| `replay_ratio` | **CHECKED_DERIVED** | derivable from frozen PPO config; explicit replay ratio not separately listed -> minor, record at run | RAW_FILE (derived) |
| `batch` | **CHECKED_DERIVED** | transitions/update IDENTICAL; explicit minibatch count not separately listed -> minor, record at run | RAW_FILE |
| `action_mode` | **FROZEN_BY_CANONICAL_SPEC** | CANONICAL_EVALUATOR_V1 pins eval action mode; matched replay must match controlled arm; not among the 5 must-freeze identity fields | CANONICAL_SPEC |
| `checkpoint_schema` | **DECLARED** | P2 checkpoint format declared; binary not synced locally (see section seven) | GIT_PROVENANCE (manifest) |
| `resume_policy` | **CLARIFIED** | the matched control itself does not resume; if any resume comparison is needed, Exact Resume harness is READY but execution NOT_RUN | REPORT_ONLY |

## Architecture single axis
- long_memory_module=NONE removes EXACTLY this delta and nothing else => single-difference holds at architecture level
- init identity: checkpoint `teacher17500 = base_ckpt_v7fix55_armA_s0/rl_checkpoints/17500`, params_sha256 `d4e85af58b7f87d6` -- VERIFIED identical (matches LC smoke logs teacher_sha and P2 base_checkpoint)

## Conclusions
- 23 fields IDENTICAL to real P2-Full-A; 5 identity fields UNVERIFIED and must be frozen on a connected host; 0 extra main difference.
- Architecture single-difference (long_memory_module=NONE) holds; PPO/Replay/init verified identical.
- Because 5 identity fields are unresolved (incl. world_set_hash BLOCKED), status = READY_NOT_AUTHORIZED; L_SEQ NOT auto-picked.

## Discipline
- config audit only; NO training launched; NO auto-modify+run
- max status READY_NOT_AUTHORIZED
- did NOT auto-pick L_SEQ
- NOT_MATCHED only on major mismatch (none here)
- MISSING/BLOCKED never relabeled FAIL
