# 50k Aggregation Screening Matrix — 2026-07-14
## 4 mechanisms × 49,152 env steps (3 PPO updates × 16,384) × seed 0 × GPU1

| Gate | Original | Soft Copeland | Budgeted Copeland | Budgeted Auction |
|------|----------|---------------|-------------------|-------------------|
| Selector | original_plr | soft_copeland | budgeted_copeland | auction_budgeted |
| Pool hash | 266f1b4a | 266f1b4a | 266f1b4a | 266f1b4a |
| Unique hashes | 8/8 | 8/8 | 8/8 | 8/8 |
| Cache hit rate | N/A (zero) | 1.0000 | 1.0000 | 1.0000 |
| Env steps | 49,152 | 49,152 | 49,152 | 49,152 |
| Train time | 182.5s | 181.3s | 174.7s | 176.3s |
| Ckpt params | True | True | True | True |
| Ckpt opt | True | True | True | True |
| Ckpt step | True | True | True | True |
| Inside-PPO ID | present | present | present | present |
| Selection hash | 5befb4f8 | d0ce6f1c | d0ce6f1c | d32fa52e |

### Selection diversity: 3/4 unique selection sets
- Original (PLR, zero-cache) differs from all enhanced mechanisms
- Soft Copeland = Budgeted Copeland (current signal distribution yields identical selections with max_source_share=0.25)
- Budgeted Auction differs from all others (role budgets bind)

### Gate evidence
- **Original zero-cache influence**: PLR-based selection, no cache reads — confirmed
- **Enhanced cache**: all 3 enhanced mechanisms hit 100% (96/96 entries)
- **Budget binding**: Budgeted Copeland applies source caps; Budgeted Auction applies role budgets with different selection from Soft Copeland
- **Checkpoint restore**: all 4 mechanisms pass params/opt/step deep comparison
- **Inside-PPO candidate identity**: scoring_window_data present in all 4 PPO runs
- **No API calls, no fallback**: confirmed via code path audit
- **No hard-gate failures**: all 4 runs completed without error

### Outputs
- `gate_r0_final/original_s0_50000steps/`
- `gate_r0_final/soft_copeland_s0_50000steps/`
- `gate_r0_final/budgeted_copeland_s0_50000steps/`
- `gate_r0_final/auction_budgeted_s0_50000steps/`

### Infrastructure
- GPU1: NVIDIA RTX 4090 (UUID GPU-f4d0f435...)
- JAX 0.6.2, CUDA 12.8, LD_PRELOAD libcusparse.so.12
- Frozen pool: frozen_pool_artifact.json (hash 266f1b4a...)
- Frozen cache: frozen_immutable_cache (96 entries, read-only)

### Status
50K AGGREGATION SCREENING MATRIX: ALL 4 MECHANISMS PASSED
