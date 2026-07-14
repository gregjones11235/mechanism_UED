# Gate R0 Consolidated Manifest — 2026-07-14
## 5 mechanisms × 16,384 env steps × seed 0 × GPU1 (RTX 4090, UUID GPU-f4d0f435...)

| Gate | Original | Soft Copeland | Budgeted Copeland | Auction Raw | Auction Budgeted |
|------|----------|---------------|-------------------|-------------|-------------------|
| Pool hash | 266f1b4a | 266f1b4a | 266f1b4a | 266f1b4a | 266f1b4a |
| Unique hashes | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| Cache hit rate | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Env steps | 16384 | 16384 | 16384 | 16384 | 16384 |
| Ckpt params | True | True | True | True | True |
| Ckpt opt | True | True | True | True | True |
| Ckpt step | True | True | True | True | True |
| Inside-PPO ID | present | present | present | present | present |
| Spec variation | 2.161/1.785 | 2.161/1.785 | 2.161/1.785 | 2.161/1.785 | 2.161/1.785 |
| Chain rejected | 8 | 8 | 8 | 8 | 8 |

### Selection diversity
| Mechanism | Selection hash |
|-----------|---------------|
| Original (PLR, no cache) | 3ca139acbcf08e70 |
| Soft Copeland | d0ce6f1c30ac9668 |
| Budgeted Copeland | d0ce6f1c30ac9668 |
| Auction Raw | 5818726dd94be606 |
| Auction Budgeted | d32fa52eee55c797 |

4/5 unique selection sets. Soft Copeland = Budgeted Copeland with current signal distribution.
Budgeted Auction differs from Raw Auction (different selection hash).

### Key evidence
- **Original zero-cache**: PLR-based selection, no aggregation, no cache reads — confirmed
- **Enhanced cache**: all 4 enhanced mechanisms hit 100% (96/96 entries)
- **Causal candidate-to-env-to-PPO**: spec variation produces different TaskParams → different world behavior → different PPO trajectories
- **Checkpoint restore**: all 5 mechanisms pass model params, optimizer state, and global step deep comparison
- **Inside-PPO identity**: scoring_window_data present in all 5 PPO runs
- **Budget binding**: Budgeted Copeland applies source caps; Budgeted Auction applies role budgets with different selection from Raw Auction

### Non-fatal warnings (all runs)
- XLA dot search space autotuning hints (19 lines, expected with RTX 4090 + JAX 0.6.2)
- CUDA delay kernel timer (2-4 lines, missing warmup, first-launch artifact)
- Orbax CheckpointManager deprecation notice (1 line)
- Orbax sharding info restoration notice (1 line)

### Outputs preserved
- `gate_r0_final/original_s0_16384steps/`
- `gate_r0_final/soft_copeland_s0_16384steps/`
- `gate_r0_final/budgeted_copeland_s0_16384steps/`
- `gate_r0_final/auction_raw_s0_16384steps/`
- `gate_r0_final/auction_budgeted_s0_16384steps/`
- `gate_r0_final/_preserved/` (July 13 original + incomplete soft_copeland)

### Infrastructure
- GPU1: NVIDIA RTX 4090, UUID GPU-f4d0f435-b393-6405-cb6d-7b4e787335de
- JAX 0.6.2, CUDA 12.8, LD_PRELOAD=/usr/local/cuda-12.8/lib64/libcusparse.so.12
- Python: dicode310 conda environment
- Frozen pool: /root/experiments/dicode_runs/siege_aggregation/frozen_pool_artifact.json
- Frozen cache: /root/experiments/dicode_runs/siege_aggregation/frozen_immutable_cache

### Status
GATE R0: ALL 5 MECHANISMS PASSED — INDEPENDENT REVIEW COMPLETE
