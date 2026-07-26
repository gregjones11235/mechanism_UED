# RMT16 Phase4A-v2.1 — Replay Exposure Contract (§四/§五)

Status: **CONTRACT DEFINED; TWO-ARM EXPOSURE NOT_RUN.** This document is the normative spec
implemented by `runtime/experiment_src/phase4a_v2_contract.py`, emitted by the launcher
(checkpoint manifest + final summary), and enforced by `tests/phase4a_v2_exposure_validator.py`.

## 1. The four-way label split (§四)

The previously single `matched_replay_protocol_ready=true` flag is **removed**. It conflated
three distinct claims. The launcher now emits four mutually-distinct labels
(`summary.phase4a_v2.replay_labels`):

| Label | This round | Meaning |
|---|---|---|
| `SAME_REPLAY_PROTOCOL` | **READY** | Both arms share an IDENTICAL protocol definition (sequence_length, batch_size, replay_mode, sampler=eligible_only, learner=original_vtrace_update_rmt, loss=vtrace_original_goal, rng rule `RandomState(seed+7)`). Says NOTHING about exposure or content. |
| `MATCHED_REPLAY_EXPOSURE` | **NOT_RUN** | Both arms consumed identical replay exposure. Requires a real two-arm run + validator PASS. No formal two-arm run this round. |
| `MATCHED_REPLAY_CONTENT` | **NOT_CLAIMED** | Both arms replayed identical trajectory CONTENT. **Unclaimable** for endogenous per-arm buffers (no shared trajectory identity). |
| `ENDOGENOUS_REPLAY_SCREENING` | **READY_AFTER_SMOKE** | The pair is usable as an endogenous screening comparison even if exposure is not perfectly matched. |

Writing a single `MATCHED_REPLAY_PROTOCOL_READY=true` (or any one label standing in for all
four) is forbidden (GATE22).

## 2. Per-arm exposure certificate (§五)

Each arm's final summary MUST emit `exposure_certificate` with all 14 fields
(`EXPOSURE_CERTIFICATE_FIELDS`):

```
outer_update_count                     # number of outer updates recorded
replay_attempt_mask                    # bool per outer update: replay sample attempted?
replay_attempt_outer_updates           # indices where attempted
replay_not_ready_outer_updates         # indices where buffer NOT eligible (explicit NOT_READY)
replay_update_outer_updates            # indices where a replay gradient update RAN
replay_update_count                    # executed replay updates
accepted_replay_policy_update_count    # replay updates committed through the KL gate
kl_rejected_replay_update_count        # replay updates KL-rolled-back (NEW in v2.1; counted, no policy advance)
replay_sequences_consumed              # total sequences consumed
replay_batch_sizes                     # per executed update: batch size (== K_BATCH)
replay_sequence_lengths                # per executed update: sampled sequence lengths
eligible_count_by_outer_update         # eligible trajectory count at each outer update
sample_ids_by_outer_update             # INTERNAL per-arm provenance — NOT compared cross-arm
start_offsets_by_outer_update          # INTERNAL per-arm provenance — NOT compared cross-arm
```

`sample_ids_by_outer_update` / `start_offsets_by_outer_update` identify trajectories inside ONE
arm's endogenous buffer; those ids have no cross-arm meaning and are excluded from every
cross-arm comparison.

## 3. Three-level two-arm comparison

`phase4a_v2_exposure_validator.py` (or `phase4a_v2_contract.compare_exposure`) adjudicates:

* **Level 1 PROTOCOL_MATCH** — `PROTOCOL_MATCH_FIELDS` (sequence_length, batch_size, replay_mode,
  sampler, loss) equal across arms.
* **Level 2 EXPOSURE_COUNT_MATCH** — all six `EXPOSURE_MATCH_FIELDS` equal:
  `replay_attempt_mask`, `replay_update_outer_updates`, `replay_update_count`,
  `replay_sequences_consumed`, `replay_batch_sizes`, `replay_sequence_lengths`.
  `MATCHED_REPLAY_EXPOSURE=PASS` ⇔ Level 1 AND Level 2 PASS.
* **Level 3 CONTENT_MATCH** — always `NOT_APPLICABLE_ENDOGENOUS_BUFFERS`; never PASS.

If one arm is NOT_READY (zero replay updates) while the other DID replay, the comparison is FAIL,
flagged `one_arm_not_ready_vs_replayed=true`, but the pair remains valid
`ENDOGENOUS_REPLAY_SCREENING=READY_AFTER_SMOKE` — the runs are NOT discarded and NOT silently
rerun.

## 4. Fail-closed gates

| Gate | Condition |
|---|---|
| `MATCHED_REPLAY_CERTIFICATE_REQUIRED` | `MATCHED_REPLAY_EXPOSURE=PASS` claimed while either arm's certificate is missing/incomplete, or the comparison is not PASS. |
| `ENDOGENOUS_REPLAY_CONTENT_CANNOT_BE_CLAIMED_MATCHED` | Any claim of `MATCHED_REPLAY_CONTENT=PASS` with endogenous buffers (input certificate or report). |

## 5. Usage

```sh
# synthetic self-test (11 checks, no data, no training)
python tests/phase4a_v2_exposure_validator.py --self-test

# real two-arm comparison (after a formal two-arm run exists)
python tests/phase4a_v2_exposure_validator.py \
    --persistent <persistent>_train_summary.json \
    --reset128   <reset128>_train_summary.json \
    --out reports/rmt16_phase4a_v2_1_exposure_report.json
```

This round: `MATCHED_REPLAY_EXPOSURE=NOT_RUN` — no formal two-arm run has been launched
(`FORMAL_TWO_ARM_LAUNCH=NOT_AUTHORIZED`, `NEW_TRAINING_RUNS=0`).
