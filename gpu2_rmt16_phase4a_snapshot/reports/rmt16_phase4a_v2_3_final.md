# RMT16 Phase4A V2.3 — Formal Identity & Runtime Certificate Finalization (Final Report)

**Task:** `RMT16_PHASE4A_V2_3_FORMAL_IDENTITY_AND_CERTIFICATE_FINALIZATION`
**Branch:** `henry/rmt16-phase4a-v2-original-vtrace`
**Base / remote HEAD at start:** `f2b7aead44426825f905fa8b82c5f66c29ee167a` (V2.2; parent `87d1e552...`)
**Review baseline:** `d3c8c7d6abd2df3d0ba69dc2c1f326f8668798e5`
**Scope:** evidence-chain integrity + fail-closed enforcement ONLY. No algorithm, design,
hyperparameter, network, task, evaluator, seed, budget, CC3/CC4/Henry-branch changes. No
training, no smoke, no push, no merge.

---

## 1. Director's 8 residual items → resolution

| # | Item | Resolution | Gates |
| --- | --- | --- | --- |
| 1 | Formal YAML values bound, but the canonical pre-registered FILE identity not frozen | `phase4a_v2_formal_identity.py` freezes per-arm canonical `relative_path` + `file_sha256` + `scientific_config_sha256`, computed from the real files (self-test re-derives). Path identity = realpath equality vs `snapshot_root/relative_path` (no copies, no symlink escape, no `..`). Content identity = frozen file SHA + frozen scientific SHA. Runs BEFORE `import jax`. | GATE39 (PASS both arms), GATE40 (6 NEG) |
| 2 | `runtime_assignment` may fail-open on missing fields | `resolve_runtime_assignment` raises `RUNTIME_ASSIGNMENT_INCOMPLETE` on missing/null/empty/non-string `arm`/`gpu_uuid`/`out_dir` — no default, no bypass. `validate_runtime_assignment` adds four-way ARM equality, exact GPU equality, and strict out_dir (relative-only, no `..`, realpath-equal under `--run_root`; v2.2 suffix match deleted). | GATE41 (6 NEG), GATE42 (3 NEG), GATE43 (5 NEG) |
| 3 | Certificate lacks an independent final file SHA | `write_certificate_atomic`: canonical JSON → embedded `certificate_payload_sha256` (§七.1) → temp file + flush + fsync + `os.replace` (§六.4) → final FILE SHA over exact written bytes → detached `<name>.sha256` sidecar `"<sha>  <basename>"`, fsynced (§七.2). Returns `(path, sidecar, file_sha, payload_sha)`. | GATE46 |
| 4 | Checkpoint/summary not bound to the final certificate file SHA | `certificate_shas_record` is now a 13-key superset: v2.2 keys + `runtime_config_certificate_{version,finalized,payload_sha256,file_sha256,sidecar_path}` + `base_checkpoint_params_sha256` + `base_checkpoint_match`. The driver's checkpoint manifest AND the summary embed this record with the final file SHA + sidecar. | GATE37 (rewritten), GATE46 |
| 5 | Checkpoint-SHA failure may leave a stale PASS certificate on disk | Explicit state machine: `build_precheck_certificate` → `PENDING_CHECKPOINT_IDENTITY` (pre-JAX, `certificate_finalized=False`, checkpoint match relabeled `PENDING`); `finalize_certificate` → `PASS` only from PENDING + checkpoint PASS/NOT_FROZEN, else `FAIL` + reasons. A FAIL precheck can NEVER finalize to PASS (`CERTIFICATE_NOT_PENDING_AT_FINALIZE` stale-PASS guard). Every FAIL path writes the finalized FAIL certificate over the on-disk file BEFORE exiting nonzero — no stale PASS/PENDING survives. | GATE44, GATE45 (4 NEG), GATE47 (6 tamper NEG) |
| 6 | Full scientific binding happens AFTER `import jax` | `phase4a_v2_frozen_spec.py` (STDLIB ONLY) holds the ~50 frozen values (`FROZEN_SPEC_SHA256=722b99…`). The driver imports RTC/FSPEC/FID BEFORE `import jax`, builds the FULL runtime scientific config from the frozen spec, diffs it against the formal YAML, and refuses pre-JAX (`FORMAL_CONFIG_RUNTIME_MISMATCH: prejax precheck certificate_status=…`). AFTER `import jax`, the REAL imported objects (Cfg/FullP2Config/K_BATCH/ANCHOR_INTERVAL/MIN_SEQUENCE_LENGTH/RL.W_ORIGINAL_VTRACE/…) are diffed against the frozen spec (`IMPORTED_RUNTIME_CONSTANTS_MISMATCH` on drift, finalized FAIL certificate written, exit). Transitivity: formal YAML == frozen spec (pre-JAX) AND frozen spec == REAL objects (post-import) ⟹ formal YAML == executing constants. | GATE36 (rewritten; §十四 static order), GATE44 |
| 7 | Protocol learner/sampler still string declarations | §八 two-phase: (2a) `executed_function_source_identity` binds the ACTUALLY EXECUTING learner/sampler via `inspect` (module/qualname/source SHA256/line count), fail-closed `EXECUTED_PROTOCOL_SOURCE_UNAVAILABLE`/`_MISMATCH`; (2b) `verify_rng_instance_identity` binds the sampler RNG instance class (must be `numpy.random.RandomState`). Then `verify_executed_protocol_matches_declared` reconciles the bound identity with `replay_protocol_labels(...).protocol_definition` (sampler label `eligible_only` ↔ function `sample_eligible`). The driver binds all three into the certificate + summary. | GATE48 (5 NEG) |
| 8 | V2.2 report says NOT_PUSHED but `f2b7aead` IS remote HEAD | Errata only: `reports/rmt16_phase4a_v2_2_publication_errata.md` records V2.2 = PUSHED @ `f2b7aead…`; V2.2 files NOT rewritten (GATE38 + GATE50 assert the v2.2 label file is byte-unchanged); V2.3 labels carry `V2_2_ERRATUM_*` forward. | GATE49, GATE50 |

## 2. Driver gate order (§十四, statically asserted by GATE36)

```
ARGPARSE < preflight < FORMAL_IDENTITY < RUNTIME_ASSIGNMENT < FULL_SCIENTIFIC_BINDING(pre-JAX)
< PRECHECK(PENDING) < pre-JAX refusal < IMPORT_JAX < IMPORTED_CONSTANTS_BINDING (+drift
finalize/refusal) < ENV_BUILD < CHECKPOINT_LOAD < checkpoint verify/FINALIZE < final refusal
< TRAINING LOOP
```

New driver args: `--snapshot_root` (pins the canonical formal-config path) and `--run_root`
(pins strict out_dir realpath); both required for `original_vtrace` (fail closed in modules).

## 3. Gates (50 total; §十/§十一)

- **New:** GATE39–GATE50 (12 gates; ≥34 fail-closed negatives; §十一 requires ≥25).
- **Rewritten:** GATE36 (reason+diff: v2.3 two-phase certificate; order semantics changed per
  §十四 — the v2.2 single-write needle no longer exists), GATE37 needle 2 (reason+diff: §七.3
  binds the final certificate file SHA + sidecar into the summary; record = 13-key superset).
- **Unchanged & green:** GATE01–GATE35, GATE38 (zero regressions).
- Local (Anaconda py3.12, no jax): 49 PASS / 0 FAIL / 1 SKIP (GATE08 JAX) → `PASS_LOCAL`.
- Server CPU (`JAX_PLATFORMS=cpu`): 50 PASS / 0 FAIL / 0 SKIP → `GATES_RESULT=PASS`.

Module self-tests: runtime_config 64/64 (63 fail-closed negatives), frozen_spec 5/5,
formal_identity 11/11, contract §八 smoke positive + 4 negatives. Exposure validator +
config_diff_validator + compileall: PASS (local + server).

## 4. Frozen evidence parity

`evidence/raw_probe/` and `configs/` are byte-unchanged (SHA256SUMS unchanged; both formal
YAML file SHAs still equal the frozen constants — asserted by GATE39). Base checkpoint SHA
`d4e85af5…` (`ckpt17500`) unchanged.

## 5. Operational compliance

No training/smoke/evaluation launched; no GPU touched (GPU0/GPU1 forbidden); no parameter
updates; no hyperparameter/architecture/task/evaluator/seed/budget changes; CC3/CC4/Henry
files untouched; no merge/rebase/amend/force-push/reset/clean; no `git add .` (path-limited
add only); single local commit `fix(rmt16): finalize formal identity and runtime certificate
chain`; no push performed (V2.3 = NOT_PUSHED, pending 总控复审); no secrets printed or
requested; every fix recorded with reason+diff.

## 6. §十七 frozen labels (final)

```
RMT16_PHASE4A_V2_3_IMPLEMENTATION          = PASS
CERTIFICATE_VERSION                        = phase4a_v2.3
FROZEN_SPEC_SHA256                         = 722b99716215c6393d25c40c3baba93f73cb4fc7e84974de343ea9dbbc769bf8
FORMAL_PERSISTENT_FILE_SHA256              = 3ac2c9bbc3f1a9b53b8c2c58df820874ffe02c35177901fc2cee88b096643138
FORMAL_PERSISTENT_SCIENTIFIC_SHA256        = 078ecdc072fad59fdbcd8ce004675d612021eaafb171227b8671c031de43ecde
FORMAL_RESET128_FILE_SHA256                = c2672e735098ced69849e0c3a6d4f5be38702f3efe1572726f6cc16ce2cee80c
FORMAL_RESET128_SCIENTIFIC_SHA256          = 71c3cce5e92217d5a32a616c7d33d285f327fb2e424f574f38ba5b6ab5922047
BASE_CHECKPOINT_PARAMS_SHA256              = d4e85af58b7f87d689fadea12eec70c852fa098a09f5ea8907448684b3bf60f5
DECLARED_PROTOCOL_LEARNER                  = original_vtrace_update_rmt
DECLARED_PROTOCOL_SAMPLER_FUNCTION         = sample_eligible
DECLARED_PROTOCOL_SAMPLER_LABEL            = eligible_only
DECLARED_PROTOCOL_RNG_ENGINE               = np.random.RandomState
TOTAL_GATE_COUNT                           = 50
V2_3_REMOTE_PUBLICATION_STATUS             = NOT_PUSHED
V2_2_ERRATUM_REMOTE_PUBLICATION_STATUS     = PUSHED (f2b7aead44426825f905fa8b82c5f66c29ee167a)
```
