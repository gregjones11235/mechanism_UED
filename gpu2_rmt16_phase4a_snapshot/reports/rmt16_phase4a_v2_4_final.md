# RMT16 Phase4A V2.4 — Actual CLI Binding & Executed Protocol Certificate Closure (Final Report)

**Task:** `RMT16_PHASE4A_V2_4_ACTUAL_CLI_AND_EXECUTED_PROTOCOL_CERTIFICATE_CLOSURE`
**Branch:** `henry/rmt16-phase4a-v2-original-vtrace`
**Parent / remote HEAD at start:** `edb93bdee3ba6f2ee65539c1221876d89503e765` (V2.3; chain
`f2b7aead… → 7905e754… → edb93bde…`; base `87d1e552…`)
**Review baseline:** `d3c8c7d6abd2df3d0ba69dc2c1f326f8668798e5`
**Scope:** evidence-chain closure ONLY — actual-CLI pre-JAX binding, staged-checkpoint
failure finalization, certificate disk-object/payload-SHA binding, executed-protocol
binding before certificate PASS, effective-protocol cross-arm comparison, strict sidecar
validation, accurate formal-path labels. No algorithm / design / hyperparameter / network /
task / evaluator / seed / budget / CC3 / CC4 / Henry-branch changes. No training, no smoke,
no merge; no push was performed before this commit's creation.

> **Publication-status discipline (carried from v2.3):** every publication label in this
> round is **creation-time evidence** (`V2_4_..._AT_COMMIT_CREATION` /
> `..._BEFORE_COMMIT`). V2.3 was later pushed by 总控 (remote HEAD became `edb93bde…`);
> this round records that as `V2_3_ERRATUM_REMOTE_PUBLICATION_STATUS=PUSHED` /
> `V2_3_ERRATUM_REMOTE_HEAD=edb93bde…`. V2.3 files are NOT rewritten — the v2.3 labels
> used creation-time keys, so no statement in them was falsified (unlike the v2.2 errata
> case, which is itself carried forward as `V2_2_ERRATUM_*`). This report makes NO claim
> about the CURRENT remote state of the branch.

---

## 1. Director's V2.4 items (§三–§十二) → resolution

| § | Item | Resolution | Gates |
| --- | --- | --- | --- |
| 三 | Pre-JAX binding used defaults, not the ACTUAL CLI | `_prejax_kwargs = FSPEC.build_kwargs(args.carry_mode)` then `.update(...)` with the seven ACTUAL CLI values (`carry_mode`, `replay_mode`, `allow_full_p2_legacy`, `sequence_length`, `seed`, `total_updates`, `save_every`) BEFORE `build_runtime_scientific_config` and BEFORE `import jax`. Wrong `--seed 43` / `--total_updates 13` / `--save_every 3` / `--sequence_length 130` / `--allow-full-p2-legacy` ⇒ `FORMAL_CONFIG_RUNTIME_MISMATCH` pre-JAX; wrong `--replay_mode` / `--carry_mode` ⇒ `FORMAL_CONFIG_ARM_MISMATCH`. | GATE51 (5 mismatch + 2 arm NEG), GATE52 |
| 三.3 | No proof the refusal precedes `import jax` | SUBPROCESS test with a fake blocking `jax` package writing an import sentinel: correct CLI ⇒ sentinel written (jax imported) and no mismatch; wrong CLI ⇒ nonzero exit, message present, sentinel ABSENT ⇒ `import jax` never ran. | GATE52 |
| 四 | Checkpoint-flow failures could skip finalization | Unified staged try: `CHECKPOINT_MANAGER_INIT → RESTORE → STRUCTURE → PARAMS_EXTRACTION → PARAMS_HASH → SHA_COMPARE`, `except Exception as exc: _CHECKPOINT_ERROR=...`, `finalize_certificate(..., checkpoint_error=..., checkpoint_failure_stage=...)` + unconditional atomic final write. New certificate field `checkpoint_failure_stage` ∈ `RTC.CHECKPOINT_FAILURE_STAGES` (invalid label ⇒ `CHECKPOINT_FAILURE_STAGE_INVALID`). Every failure mode leaves a finalized FAIL certificate ON DISK. | GATE53 (6 stages + invalid-label NEG) |
| 五 | Payload SHA did not enter the manifest; disk object not re-read | `write_certificate_atomic()` returns a 5-tuple incl. the written artifact; the caller adopts it (`RUNTIME_CONFIG_CERTIFICATE = written_certificate`) and calls `verify_certificate_artifact(...)` after EVERY write (4 sites). On final PASS the driver re-reads the disk certificate and requires equality (`CERTIFICATE_DISK_OBJECT_MISMATCH`) + non-null payload SHA. Manifest / summary / launch-status bind `runtime_config_certificate_payload_sha256`, `runtime_config_certificate_file_sha256`, `base_checkpoint_params_sha256` — all length-64 non-null. | GATE54, GATE55 |
| 六 | Executed protocol not bound before certificate PASS | Driver order: `import jax` → imported-constants binding → learner source binding → sampler source binding → replay-RNG construction (`np.random.RandomState(args.seed + 7)`; type-verify only, NO state consumed — training reuses the same instance) + RNG identity binding → effective protocol build → checkpoint load/identity → certificate final PASS → env/optimizer/training. Any executed-protocol failure ⇒ finalize FAIL + atomic write + nonzero exit (`EXECUTED_PROTOCOL_BINDING_FAILURE`). | GATE36 (rewritten), GATE58 |
| 七 | Source identity too thin; impostors possible | `_source_identity(fn)` records `module / qualname / name / module_realpath / module_file_sha256 / function_source_sha256 / source_lines`; fails closed `EXECUTED_PROTOCOL_SOURCE_UNAVAILABLE`. Negatives: same-name impostor, same-name-different-source, different module-file SHA, missing realpath, deleted source file — all rejected. | GATE48 (rewritten; 6 NEG), GATE56, GATE57 |
| 八 | No effective protocol identity | `effective_protocol_definition = {declared_protocol, executed_learner(5 fields), executed_sampler(5 fields), executed_rng(class_module, class_name, numpy_version, seed_derivation="run_seed_plus_7", hidden_buffer_rng_used=false)}`; `effective_protocol_sha256` over canonical JSON (key-order invariant). Both arms' summaries emit declared + executed + effective + SHA. | GATE59 (3 NEG) |
| 九 | Cross-arm validator compared declared only | `PROTOCOL_MATCH=PASS` requires declared complete+identical AND executed identity complete AND effective keysets + learner/sampler module/file/function SHAs + RNG class + numpy version + effective SHA all identical. Missing executed identity ⇒ `EXECUTED_PROTOCOL_IDENTITY_REQUIRED` (NO declared-only fallback). New outputs: `DECLARED_PROTOCOL_MATCH`, `EXECUTED_PROTOCOL_MATCH`, `EFFECTIVE_PROTOCOL_MATCH`, `EFFECTIVE_PROTOCOL_SHA256_ARM_A/_ARM_B`, `EXECUTED_PROTOCOL_DIFFERING_FIELDS`. `MATCHED_REPLAY_EXPOSURE=PASS` requires EFFECTIVE + EXPOSURE PASS. | GATE57, GATE60; validator self-test 25/25 |
| 十 | Sidecar basename not validated | Sidecar must be EXACTLY `<sha256>  <certificate basename>\n` (2 tokens; token0 == file SHA; token1 == basename; trailing newline). Wrong basename / extra tokens / empty / lone token / truncation ⇒ `RUNTIME_CONFIG_CERTIFICATE_TAMPERED`. | GATE61 (5 attacks) |
| 十一 | Formal-path labels overclaimed (NO_COPY) | Whole-snapshot relocation IS legitimate; `NO_COPY` wording dropped. Labels: `FORMAL_CONFIG_PATH_IDENTITY=CANONICAL_RELATIVE_PATH_UNDER_EXECUTING_SNAPSHOT_ROOT`, `FORMAL_CONFIG_SNAPSHOT_RELOCATION=LAYOUT_AND_CONTENT_BOUND`. `derived_snapshot_root = realpath(dirname(__file__)/../..)`; `realpath(args.snapshot_root)` must equal it (fail closed). | GATE62 |
| 十二 | Gate coverage | GATE51–GATE62 added (≥62 total). | this round |

## 2. Driver gate order (v2.4; statically asserted by GATE36, behaviorally by GATE52)

```
ARGPARSE < preflight < FORMAL_IDENTITY < RUNTIME_ASSIGNMENT
< ACTUAL_CLI_PREJAX_SCIENTIFIC_BINDING < PRECHECK(PENDING) < PRE_JAX_REFUSAL
< IMPORT_JAX < IMPORTED_CONSTANTS_BINDING (+drift finalize/refusal)
< REPLAY_RNG_CONSTRUCTION (seed+7, no state consumed)
< EXECUTED_LEARNER/SAMPLER_SOURCE_BINDING < EXECUTED_RNG_BINDING < EFFECTIVE_PROTOCOL_BUILD
< EXECUTED_PROTOCOL finalize/refusal
< STAGED_CHECKPOINT_LOAD (6 failure stages) < CHECKPOINT_VERIFY/FINALIZE
< POST_WRITE_VERIFY + DISK_REREAD < FINAL_REFUSAL < ENV_BUILD < TRAINING LOOP
```

## 3. Gates (62 total; §十二)

- **New:** GATE51–GATE62 (12 gates).
- **Rewritten (reason+diff):** `_v23_prejax_chain` helper (must mirror the driver's
  ACTUAL-CLI override of the frozen kwargs — the old helper built from defaults only, which
  no longer models the driver); GATE36 (v2.4 order: actual-CLI override < RNG construction
  < executed bind < staged checkpoint < disk re-read; 3 finalize sites classified
  drift/executed/checkpoint); GATE46 (5-tuple adoption: written artifact == disk);
  GATE47 (5-tuple unpacks at both write sites); GATE48 (extended 7-field source identity +
  RNG no-state-consumption + deleted-source negative).
- **Unchanged & green:** GATE01–GATE35, GATE37–GATE45, GATE49–GATE50 (zero regressions).
- Local (Anaconda py3.12.4 + numpy 1.26.4, no jax): 59 PASS / 0 FAIL / 3 SKIP
  (GATE08/GATE56/GATE59 import the real frozen replay modules, which require jax) →
  `PASS_LOCAL`.
- Server CPU (`JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""`, py3.10.20, jax 0.6.0,
  numpy 2.2.6, yaml 6.0.3; deploy `phase4a_v2_4_deploy/`): **62 PASS / 0 FAIL / 0 SKIP** →
  `GATES_RESULT=PASS` (exit 0).

Module self-tests: runtime_config 76/76, frozen_spec 5/5, formal_identity 13/13,
exposure validator 25/25. compileall: PASS.

## 4. Frozen evidence parity (§十三)

`evidence/raw_probe/` byte-unchanged vs `edb93bde…` (`git diff --exit-code` clean;
SHA256SUMS unchanged). Both formal YAMLs byte-unchanged (same check). `first_ge512=8979`
and `L512_REACHABILITY=BOTH` re-derived by GATE26 (recomputed, not hardcoded). The two
arms differ ONLY in `carry_mode` (GATE14).

## 5. Operational compliance (§十五/§十六)

Changed files (protection scope respected — `git diff --name-status edb93bde…` shows ONLY):
`runtime/experiment_src/{phase4a_v2_contract,phase4a_v2_formal_identity,phase4a_v2_runtime_config,train_rmt16_p2replay}.py`,
`tests/{phase4a_v2_exposure_validator,test_phase4a_v2_gates}.py`, plus NEW
`reports/rmt16_phase4a_v2_4_{labels.json,final.md}`. `git diff --check` clean. No training /
smoke / evaluation launched; no GPU touched (GPU0/GPU1 forbidden); no parameter updates; no
algorithm/hyperparameter/architecture/task/evaluator/seed/budget changes; CC3/CC4/Henry
files untouched; no merge/rebase/amend/force-push/reset/clean; no `git add .` (path-limited
add only); single local commit `fix(rmt16): bind actual cli and executed protocol before
certificate pass`; no push performed before commit creation
(`V2_4_PUBLICATION_STATUS_AT_COMMIT_CREATION=NOT_PUSHED`,
`V2_4_PUSH_PERFORMED_BEFORE_COMMIT=false` — creation-time facts; NO current-remote claim);
no secrets printed or requested; every fix recorded with reason+diff (driver
`verify_certificate_artifact` keyword-argument fix: RTC's signature is keyword-only for
`expected_file_sha256` / `expected_payload_sha256`; all four driver call sites updated;
detected by GATE52's subprocess run).

## 6. §十七 frozen labels (final)

```
RMT16_PHASE4A_V2_4_IMPLEMENTATION            = PASS
ACTUAL_CLI_FULL_BINDING_BEFORE_JAX           = PASS
WRONG_CLI_IMPORTS_JAX                        = false
CHECKPOINT_MANAGER_INIT_FAILURE_CERTIFICATE  = FINAL_FAIL
CHECKPOINT_RESTORE_FAILURE_CERTIFICATE       = FINAL_FAIL
CHECKPOINT_STRUCTURE_FAILURE_CERTIFICATE     = FINAL_FAIL
CHECKPOINT_PARAMS_FAILURE_CERTIFICATE        = FINAL_FAIL
CERTIFICATE_DISK_OBJECT_IDENTITY             = PASS
CERTIFICATE_PAYLOAD_SHA_IN_CHECKPOINT_MANIFEST = PASS_NON_NULL
CERTIFICATE_PAYLOAD_SHA_IN_SUMMARY           = PASS_NON_NULL
CERTIFICATE_FILE_SHA_BINDING                 = PASS
CERTIFICATE_SIDECAR_BASENAME_BINDING         = PASS
EXECUTED_PROTOCOL_BOUND_BEFORE_CERTIFICATE_PASS = PASS
EXECUTED_LEARNER_MODULE_REALPATH             = BOUND
EXECUTED_LEARNER_MODULE_FILE_SHA             = BOUND
EXECUTED_LEARNER_FUNCTION_SOURCE_SHA         = BOUND
EXECUTED_SAMPLER_MODULE_REALPATH             = BOUND
EXECUTED_SAMPLER_MODULE_FILE_SHA             = BOUND
EXECUTED_SAMPLER_FUNCTION_SOURCE_SHA         = BOUND
EXECUTED_RNG_NUMPY_VERSION_BINDING           = PASS
EFFECTIVE_PROTOCOL_DEFINITION                = IMPLEMENTED
EFFECTIVE_PROTOCOL_CROSS_ARM_MATCH_GATE      = PASS
SAME_NAME_DIFFERENT_SOURCE_NEGATIVE_TEST     = PASS
FORMAL_CONFIG_PATH_IDENTITY                  = CANONICAL_RELATIVE_PATH_UNDER_EXECUTING_SNAPSHOT_ROOT
FORMAL_CONFIG_SNAPSHOT_RELOCATION            = LAYOUT_AND_CONTENT_BOUND
RAW_PROBE_EVIDENCE_REMOTE_RECOMPUTABILITY    = PASS
FIRST_GE512_RESOLVED_ENV_STEP                = 8979
L512_REACHABILITY                            = BOTH
SMOKE_4096                                   = NOT_RUN
FORMAL_TWO_ARM_LAUNCH                        = NOT_AUTHORIZED
NEW_TRAINING_RUNS                            = 0
GPU_TRAINING_RUNS                            = 0
CC3_FILES_TOUCHED                            = false
CC4_FILES_TOUCHED                            = false
HENRY_BRANCH_TOUCHED                         = false
```
