# RMT16 Phase4A-v2.1 (§六) — Frozen raw L512 probe evidence

Status: **FROZEN, remotely recomputable.** These are the ORIGINAL probe outputs from the frozen
16384-step L512 reachability probe, copied byte-identically into Git so the first_ge512 resolved
step can be recomputed by anyone WITHOUT server access and WITHOUT rerunning the probe.

## Provenance

| File | Server source (read-only; NOT modified) | SHA256 |
|---|---|---|
| `persistent_probe_episodes.jsonl` | `runs/RMT16-PERSISTENT-PROBE-L512-16384/out/RMT16-Persistent-PPO_probe_episodes.jsonl` | `23d00d2e…0b0d276` |
| `persistent_probe_updates.jsonl`  | `runs/RMT16-PERSISTENT-PROBE-L512-16384/out/RMT16-Persistent-PPO_probe_updates.jsonl`  | `33e456d3…f2efbcf05` |
| `persistent_probe_summary.json`   | `runs/RMT16-PERSISTENT-PROBE-L512-16384/out/RMT16-Persistent-PPO_probe_summary.json`   | `2dbc5378…448c7e1b` |
| `reset128_probe_episodes.jsonl`   | `runs/RMT16-RESET128-PROBE-L512-16384/out/RMT16-Reset128-PPO_probe_episodes.jsonl`     | `644d4ccc…42c6effa` |
| `reset128_probe_updates.jsonl`    | `runs/RMT16-RESET128-PROBE-L512-16384/out/RMT16-Reset128-PPO_probe_updates.jsonl`      | `c765e366…49fd70` |
| `reset128_probe_summary.json`     | `runs/RMT16-RESET128-PROBE-L512-16384/out/RMT16-Reset128-PPO_probe_summary.json`       | `da30cb50…485a43` |

(Full digests in `SHA256SUMS`.)

Freeze protocol (CC2 §六):
1. Server-side SHA256 of all six sources verified against the expected manifest BEFORE copy
   (`RAW_PROBE_SOURCE_HASH_VERIFIED=PASS`; a mismatch would have been
   `RAW_PROBE_SOURCE_HASH_MISMATCH=BLOCKED` with the copy aborted).
2. Copied byte-identically (scp) into this directory; local SHA256 re-verified equal to the
   server sources (`RAW_PROBE_SHA_VERIFIED=true`).
3. Security scan of all six files: NO api_key/token/secret/password/authorization/bearer/private
   key material, NO absolute home paths, NO hostnames/usernames, NO GPU UUIDs.

## Recomputation (no rerun, no hardcoded values)

```sh
python tests/recompute_probe_step.py \
    --persistent evidence/raw_probe/persistent_probe_episodes.jsonl \
    --reset128   evidence/raw_probe/reset128_probe_episodes.jsonl \
    --sha256sums evidence/raw_probe/SHA256SUMS \
    --out reports/rmt16_l512_probe_recomputed.json
```

The script verifies the inputs against `SHA256SUMS` (fail closed: mismatch =>
`RAW_PROBE_SOURCE_HASH_MISMATCH=BLOCKED`; missing source =>
`RAW_PROBE_EVIDENCE_REMOTE_RECOMPUTABILITY=BLOCKED_SOURCE_UNAVAILABLE`) and then re-derives every
reported number from the episode records:

    resolved = update_index * num_envs * rollout_steps + rollout_step * num_envs + env_id + 1
    (num_envs=16, rollout_steps=128)

Recomputed result (`reports/rmt16_l512_probe_recomputed.json`):

* Persistent: completed=20, count_ge512=6, first_ge512 episode_id=2 length=562 update_index=4
  rollout_step=49 env_id=2 → **resolved env step 8979** (deprecated formula gave 8241; Δ=738).
* Reset128: completed=21, count_ge512=5, identical first_ge512 record → **resolved env step 8979**.
* `cross_arm_resolved_step_agree=true`, `L512_REACHABILITY=BOTH`,
  `RAW_PROBE_EVIDENCE_REMOTE_RECOMPUTABILITY=PASS`.

## Scope

Probe evidence ONLY — `not_for_formal_science=true`. These runs predate the v2.1 provenance
fields (episode `policy_version_start/end/span`); the probe records carry the older
`update_index`/`rollout_step`/`env_id` tuple, which is exactly what the recompute formula needs.
No formal two-arm run has been launched this round (`FORMAL_TWO_ARM_LAUNCH=NOT_AUTHORIZED`,
`NEW_TRAINING_RUNS=0`).
