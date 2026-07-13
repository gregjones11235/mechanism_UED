# GPU0 DiCode DeepSeek-V4-Pro Substitution Reproduction

This directory is a migration-ready snapshot of the completed GPU0 experiment.

## Scope

- Experiment: DiCode DeepSeek-V4-Pro substitution baseline
- Run: `dspro-gpu0-g3-seed0-20260712T041233Z`
- Source branch: `exp/dicode-dspro`
- Source commit: `88bec3aad1bd7343bf36c9ccced17d050281ba0c`
- Seed: `0`
- Training: `15,300` PPO updates, `2,005,401,600` environment steps
- Final checkpoint: `results/checkpoint_15300`
- Final mean episode return: `46.32880593643708`

This is not an exact original-model reproduction. The DiCode paper used Qwen3-235B and reported five-seed statistics; this snapshot uses DeepSeek-V4-Pro and one seed.

## Layout

- `source/`: exact GPU0 source/config/test subset and dependency manifests.
- `results/`: final checkpoint, resolved Hydra configuration, logs, task graph, timing artifacts, SHA256 inventory and machine-readable result.
- `governance/`: approval and independent-review evidence.

## Integrity

Use `results/artifact_inventory.sha256` to verify the exported result artifacts. The checkpoint reached the full configured training horizon. The process exited without a final overall-completion marker after entering post-target generated-task validation, so the independent verdict is `PASS_WITH_MINOR_ISSUES`.

No API credentials are included. Configure required provider credentials through environment variables; never commit them.

## Paper correspondence

See `governance/20260713T224800+0800_gpu0_final_training_review.md` and `results/final_result.json` for the metric-by-metric comparison with the DiCode paper.
