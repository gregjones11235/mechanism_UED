# Henry DiCode / Craftax Student Upgrade Archive

This archive preserves the research evolution from D052 and P2-lite through the current Long-Memory × Replay Phase4A work. It is an archival and analysis submission only: no new training, no new 256-world evaluation, and no checkpoint mutation was performed during this Git task.

## Scope

Covered stages:

1. D052
2. P2-v0 invalid exploratory implementation
3. P2-v1-lite
4. P2-Full-A-v1
5. P2 posthoc attribution
6. P7 EgoMap
7. P8 Summary LongMemory
8. P9 Authentic Reset
9. Long-Memory Bakeoff Phase1
10. Long-Memory Carry Phase2
11. SlowGRU Reset128 Sustainability Phase3
12. W512 × P2 Replay Phase4A
13. RMT16 × P2 Replay Phase4A current engineering state

## Important separation

D052 and P2 are different research lines.

- D052 studies task aggregation and Student training-mechanism combinations under native end-to-end evaluation.
- P2 studies Student temporal structure, long-context behavior, whole-episode replay, and replay/memory interactions.

D052 results must not be mixed into P2 causal tables.

## Artifact policy

This archive includes small source/config/report/evaluation artifacts and excludes checkpoint entities, replay buffers, binary snapshots, caches, secrets, raw long logs, and large files. Checkpoints are represented only by paths, steps, hashes, and metadata when those appear in reports or manifests.

See:

- `EXPERIMENT_TIMELINE.md`
- `SCIENTIFIC_STATUS.md`
- `TERMINOLOGY.md`
- `EVALUATION_PROTOCOL.md`
- `inventory/henry_experiment_inventory.md`
- `ARTIFACT_MANIFEST.md`

