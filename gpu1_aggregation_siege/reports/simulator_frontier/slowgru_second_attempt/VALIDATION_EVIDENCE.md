# SlowGRU second attempt — validation-only evidence

This record covers only production materialization and dependency/JAX smoke checks. It is not a formal-run PASS and did not call a paid API or start training.

- GPU1 physical UUID: `GPU-3c7a2864-755b-7045-b293-6f80e748283f` (`CUDA_VISIBLE_DEVICES=1`).
- 15-slot JAX smoke: exit `0`; 12 dynamic modules plus collecting/combat/crafting anchors; stacked TaskParams length 15; resets for task IDs 0, 1, 2, and 3. Result SHA256: `b75d5ad8d4dd2278c1e145ba890b9f74ca9768d8c52d4a61c2416fb6826a9b84`.
- Server hardening suite (pre-Repair2): `28 passed, 0 skipped`, exit `0`; Repair2 rerun: `31 passed, 0 skipped`, exit `0`.
- Server runstate atomicity suite (Repair2): `10 passed, 0 skipped`, exit `0`.
- Server authorization suite: `20 passed, 0 skipped`, exit `0`.
- Evidence roots: `/media/数据磁盘2/e3_slowgru_anchor_validation_20260810T2` and `/media/数据磁盘2/e3_slowgru_server_tests_20260810T`.

No 151-session process was started. The full-budget controller signature remains external to this validation artifact.
