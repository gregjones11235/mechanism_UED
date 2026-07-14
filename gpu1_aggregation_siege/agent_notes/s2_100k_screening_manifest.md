# S2 Aggregation Screening Manifest — 2026-07-14
## 3 mechanisms × 98,304 env steps (6 PPO updates × 16,384) × seed 0 × GPU1

### S2 Screening Matrix

| Gate | Original | Soft Copeland | Budgeted Auction |
|------|----------|---------------|-------------------|
| Requested steps | 100,000 | 100,000 | 100,000 |
| Actual steps | 98,304 | 98,304 | 98,304 |
| Step formula | nu=6 × 256×64 | nu=6 × 256×64 | nu=6 × 256×64 |
| Selector | original_plr | soft_copeland | auction_budgeted |
| Pool hash | 266f1b4a | 266f1b4a | 266f1b4a |
| Unique hashes | 8/8 | 8/8 | 8/8 |
| Cache hit rate | N/A (zero) | 1.0000 | 1.0000 |
| Train time | 177.3s | 174.7s | 175.4s |
| Ckpt params | True | True | True |
| Ckpt opt | True | True | True |
| Ckpt step | True | True | True |
| Inside-PPO ID | present | present | present |
| Selection hash | 5befb4f8 | d0ce6f1c | d32fa52e |

### Selection diversity: 3/3 unique
All three mechanisms produce different selection sets. Auction Budgeted differs from Soft Copeland, providing a genuine aggregation comparison point.

### Budgeted Copeland exclusion diagnosis (CPU-side)
**Finding:** Budgeted Copeland produces identical selections to Soft Copeland on the frozen pool/cache.

**Root cause:** All 32 candidates share source_id `"hold"`. The `apply_budget_caps` function with `max_source_share=0.25` applies source caps to 24/32 candidates, but since every candidate shares the same source, the relative score ordering is preserved. With a single source, per-source capping reduces all scores proportionally and cannot reorder the top-8.

**Evidence:**
- Source distribution: `hold: 32/32` (1 unique source)
- Soft Copeland top-8: `{6, 10, 14, 17, 19, 21, 25, 29}`
- Budgeted top-8: `{6, 10, 14, 17, 19, 21, 25, 29}` (identical)
- `source_caps_applied: 24`, `signal_caps_applied: 0`
- Overlap: 8/8, Changed selection: False

**Re-admission condition:** Budgeted Copeland requires a pool with ≥2 distinct source_ids to demonstrate budget-induced selection changes. With the current frozen cache (uniform "hold" attribution), source-based budgeting is mathematically unable to differentiate.

### Gate evidence (all mechanisms)
- **Original zero-cache**: PLR-based, no cache reads
- **Enhanced cache**: 100% hit rate (96/96)
- **Checkpoint restore**: All params/opt/step deep comparisons pass
- **Inside-PPO identity**: scoring_window_data present
- **No API/fallback**: confirmed via code audit
- **No hard-gate failures**: all 3 runs clean

### Outputs
- `gate_r0_final/original_s0_100000steps/`
- `gate_r0_final/soft_copeland_s0_100000steps/`
- `gate_r0_final/auction_budgeted_s0_100000steps/`

### Infrastructure (unchanged)
- GPU1: NVIDIA RTX 4090, JAX 0.6.2, CUDA 12.8, LD_PRELOAD libcusparse.so.12
- Frozen pool: frozen_pool_artifact.json (hash 266f1b4a...)
- Frozen cache: frozen_immutable_cache (96 entries)

### Full screening summary (16k → 50k → 100k)

| Mechanism | 16k | 50k | 100k | Notes |
|-----------|-----|-----|------|-------|
| Original | ✅ | ✅ | ✅ | Zero-cache PLR baseline |
| Soft Copeland | ✅ | ✅ | ✅ | Aggregation finalist |
| Budgeted Copeland | ✅ | ✅ | — | Excluded: identical to SC on uniform-source pool |
| Auction Raw | ✅ | — | — | 16k preflight only |
| Budgeted Auction | ✅ | ✅ | ✅ | Aggregation finalist, budget binding confirmed |

### Status
S2 AGGREGATION SCREENING: ALL 3 MECHANISMS PASSED AT 98,304 STEPS
