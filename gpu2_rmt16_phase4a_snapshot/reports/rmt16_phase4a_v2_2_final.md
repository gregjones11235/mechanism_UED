# RMT16 Phase4A-v2.2 — Final Report (runtime binding + protocol completeness)

Task: `RMT16_PHASE4A_V2_2_RUNTIME_BINDING_AND_PROTOCOL_COMPLETENESS`
Branch: `henry/rmt16-phase4a-v2-original-vtrace`
Parent commit (== origin HEAD, base publication PASS): `87d1e552415d292417dcb6e6f9f6b16b97a6d135`
Review baseline: `d3c8c7d6abd2df3d0ba69dc2c1f326f8668798e5`
Status: **IMPLEMENTATION COMPLETE — LOCAL COMMIT ONLY, NOT PUSHED. Awaiting 总控复审.**

---

## 0. Headline

`RMT16_PHASE4A_V2_2_IMPLEMENTATION=PASS`. V2.1 defined the replay exposure contract and the
policy-version provenance; V2.2 (a) upgrades the protocol comparison to a **full canonical
dictionary identity** (incl. `learner`/`rng_rule`), (b) removes the residual active-scope
`max_policy_lag` leak, (c) records the episode policy-version range in the JSONL, (d) adds
trajectory/sample range invariant validators, (e) **binds the pre-registered YAML to the real
CLI/runtime fail-closed** and emits a runtime-config certificate, and (f) replaces the
time-unscoped publication flag with **layered, time-scoped** status labels. No scientific
threshold / network / task / evaluator / seed / budget was changed.

## 1. Six items — disposition

| § | Item | Result |
|---|---|---|
| 二/八 | protocol comparison covers learner, rng_rule, COMPLETE identity | DONE — `PROTOCOL_MATCH_FIELDS` whitelist deleted; `compare_protocols` does required-field completeness + keyset identity + full diff + canonical-JSON + SHA256. `PROTOCOL_REQUIRED_FIELDS_COMPLETE=true`, `PROTOCOL_LEARNER_IDENTITY=COMPARED`, `PROTOCOL_RNG_RULE_IDENTITY=COMPARED`. |
| 三 | clear residual `max_policy_lag=16` from original_vtrace ACTIVE scope | DONE — launcher `p2_frozen` now `policy_lag_gate_active=False`/`max_policy_lag=None`; legacy 16 only under `legacy_full_p2_only{active:false}`. `ORIGINAL_VTRACE_ACTIVE_POLICY_LAG_LEAK=NONE`. |
| 四 | episode JSONL records policy-version start/end/span | DONE — `policy_version_start/end/span` written; old `policy_version` = deprecated alias of `end`. `POLICY_VERSION_EPISODE_LOG_RANGE=RECORDED`. |
| 五 | trajectory/sample range invariant validation | DONE — `validate_policy_version_range_fields` + trajectory method (enforced on insert) + read-only sample validator. `TRAJECTORY_POLICY_RANGE_VALIDATOR=IMPLEMENTED_FAIL_CLOSED`, `SAMPLE_POLICY_RANGE_VALIDATOR=IMPLEMENTED_READ_ONLY`. |
| 六 | bind pre-registered YAML to real CLI/runtime fail-closed (critical gate) | DONE — `--formal_config` required for original_vtrace before `import jax`; arm binding; canonical scientific-config diff + SHA; runtime-assignment check; frozen base-checkpoint SHA comparison; certificate emitted + embedded. `RUNTIME_CONFIG_CERTIFICATE=IMPLEMENTED`, `BASE_CHECKPOINT_SHA_EXPECTATION=PASS`. |
| 七 | fix remote publication status labels | DONE — layered labels; no time-unscoped `PUSH_PERFORMED`. `BASE_REMOTE_PUBLICATION_STATUS=PASS`, `V2_2_REMOTE_PUBLICATION_STATUS=NOT_PUSHED`. |

## 2. Binding order (honest statement)

The launcher runs the gates in this source order (line numbers in `train_rmt16_p2replay.py`):

1. `argparse` (incl. `--formal_config`) — line 66
2. **pre-JAX preflight**: `RTC.preflight_require_formal_config(REPLAY_MODE, args.formal_config)`
   (missing under original_vtrace → `FORMAL_CONFIG_REQUIRED_FOR_ORIGINAL_VTRACE`) + arm binding —
   line 84 **< `import jax` line 102**. No CUDA init before this (CUDA_VISIBLE_DEVICES set via env).
3. `import jax` — line 102
4. build REAL runtime scientific config + `validate_runtime_against_formal_config` →
   certificate; `certificate_status != PASS` → `SystemExit(FORMAL_CONFIG_RUNTIME_MISMATCH)` —
   binding line 237 / refusal **< env build line 320**
5. env build — line 320
6. checkpoint load — line 341; base params SHA verified vs frozen expectation, certificate rewritten

So §六.2's explicit pre-JAX requirement (the REQUIRED + ARM gates) is satisfied; the full
scientific certificate necessarily runs after `import jax` (the frozen `FullP2Config` constants it
binds live in a jax-importing module) but still before any env build, network init, or checkpoint
load. This is a two-phase certificate by design, not a bypass.

## 3. Frozen base-checkpoint expectation

`EXPECTED_BASE_CHECKPOINT_SHA256 = d4e85af58b7f87d689fadea12eec70c852fa098a09f5ea8907448684b3bf60f5`,
taken verbatim from BOTH frozen probe summaries' `base_sha256` (GATE33 re-verifies this against the
evidence in-repo; GATE37 verifies the comparison is fail-closed and returns `NOT_FROZEN` when no
expectation exists — never fabricated). `BASE_CHECKPOINT_SHA_EXPECTATION=PASS`.

## 4. Tests / gates

* Gates total **38** (was 26). New GATE27–38 are all non-JAX → local 1-skip (GATE08), server 0-skip.
  * Local (Anaconda py3.12 + numpy): **37 PASS / 0 FAIL / 1 SKIP** (`GATES_RESULT=PASS_LOCAL`).
  * Server CPU (JAX_PLATFORMS=cpu, CUDA_VISIBLE_DEVICES=""): **38 PASS / 0 FAIL / 0 SKIP**
    (`GATES_RESULT=PASS`).
* exposure validator self-test: **18/18** (was 11).
* runtime_config self-test: **29/29**; `FAIL_CLOSED_NEGATIVE_CASES=28` (≥19 required).
* config_diff_validator: `GATE14_CONFIG_DIFF_UNIVARIATE=PASS`, differing paths still `['carry_mode']`.
* compileall: runtime + tests + configs clean, local + server.

## 5. What was NOT done (forbidden this round — all hold)

`SMOKE_4096=NOT_RUN`; no 24576/98304; `FORMAL_TWO_ARM_LAUNCH=NOT_AUTHORIZED`; no GPU
(`GPU_TRAINING_RUNS=0`); `NEW_TRAINING_RUNS=0`; no parameter updates; no old-probe rerun; no
Hindsight/AWR/full_p2_legacy; no CC3/CC4/Henry-branch changes; no merge/rebase/amend/force-push/
reset/clean. `MATCHED_REPLAY_EXPOSURE=NOT_RUN`, `MATCHED_REPLAY_CONTENT=NOT_CLAIMED`,
`ENDOGENOUS_REPLAY_SCREENING=READY_AFTER_SMOKE`. `GATE13_NUMERIC_PARAMETER_UPDATE_HASH_RERUN=NOT_RUN`.

## 6. Publication / push state

`BASE_REMOTE_PUBLICATION_STATUS=PASS` @ `BASE_REMOTE_HEAD=87d1e55…`. This round produced ONE local
commit on the worktree branch; **no push**:
`IMPLEMENTATION_ROUND_PUSH_PERFORMED=false`, `V2_2_REMOTE_PUBLICATION_STATUS=NOT_PUSHED`,
`V2_2_PUSH_PERFORMED=false`. The time-unscoped `PUSH_PERFORMED` key is forbidden and absent.
Stopping here to await 总控复审 — no 4096 smoke, no push, no merge.

## 7. CC2 outputs (§十三)

See `rmt16_phase4a_v2_2_labels.json` for the full §十四 label set. Key outputs:
`CC2_PARENT_SHA=87d1e552415d292417dcb6e6f9f6b16b97a6d135`; `CC2_TOTAL_GATE_COUNT=38` (≥38);
`CC2_TEST_RESULT=PASS (server CPU 38/0/0)`; `CC2_RAW_PROBE_FILES_MODIFIED=false`;
`IMPLEMENTATION_ROUND_PUSH_PERFORMED=false`; `V2_2_PUSH_PERFORMED=false`.
