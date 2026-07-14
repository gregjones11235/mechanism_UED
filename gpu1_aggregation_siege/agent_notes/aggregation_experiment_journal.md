# Performance-Aware Aggregation Experiment Journal

## Research Objective
Design, implement, and evaluate performance-aware aggregation mechanisms for DiCode/Craftax curriculum selection, including multi-LLM role collaboration.

## Final Status: 2026-07-06

### Completed Stages

| Stage | Description | Status | Key Result |
|-------|-------------|--------|------------|
| Phase 0 | Repository inspection | ✅ | Hook: `selection.py:sample_tasks_for_training()` |
| Phase 1 | Implementation (6 aggregation modes) | ✅ | `src/dicode/mechanisms/aggregation.py` |
| Phase 2 | Unit tests | ✅ | 16/16 passed |
| Phase 3 | Original behavior smoke | ✅ | aggregation.enabled=false works |
| Phase 4 | Aggregation smoke (3 modes) | ✅ | All modes load correctly |
| Phase 5 | Rule-based sweep (3 modes × 200K) | ✅ | soft_copeland best (entropy 1.30) |
| Phase 6 | Summarization | ✅ | CSV + MD reports generated |
| Phase 7 | Recommendation | ✅ | soft_copeland recommended |
| Stage L1 | LLM fake role test | ✅ | 9/9 API calls, $0.0008 cost |
| Stage L2 | Real candidate export | ✅ | 4 seed tasks exported |
| Stage L3 | Real LLM pilot (12 calls) | ✅ | $0.0013, 21 cache entries |
| Stage L4 | LLM-cache aggregation smoke | ✅ | Config loads, no crashes |
| Stage L5 | Cost-aware comparison | 🟢 | Running (l5c tmux, ~3h) |
| Stage L6 | LLM collaboration summary | ✅ | Written on existing data |

### Key Findings

1. **soft_copeland** is the best rule-based aggregation mode:
   - Highest curriculum entropy (1.30 vs 1.18 for robust_weighted)
   - Composite score: 2.30
   - Zero failures

2. **Multi-LLM role architecture works**:
   - Qwen/Tutor: $0.00018/call — progression/learnability
   - DeepSeek/Critic: $0.00009/call — failure risk
   - GLM/Explorer: $0.00004/call — novelty/diversity
   - Cache mechanism prevents redundant API calls
   - Total cost < $0.01 for all testing

3. **Embedding crash fixed**: Backported safe wrapper to `dreaming/utils.py`

4. **LLM API bottleneck**: OpenRouter rate-limiting blocks evolution. Workaround: `num_generation_tasks=0`

### Files Created

```
src/dicode/mechanisms/__init__.py         — Package exports
src/dicode/mechanisms/aggregation.py      — 6 aggregation strategies + LLM cache
src/dicode/mechanisms/diagnostics.py      — JSONL reader + summary
src/dicode/mechanisms/llm_providers.py    — Qwen/DeepSeek/GLM API configs
src/dicode/mechanisms/llm_roles.py        — Tutor/Critic/Explorer role definitions
src/dicode/mechanisms/llm_cache.py        — JSONL caching layer
src/dicode/mechanisms/llm_costs.py        — Cost tracking
conf/aggregation/default.yaml             — Aggregation + LLM config
scripts/test_aggregation_selector.py      — 16 unit tests
scripts/test_cached_llm_roles.py          — L1 fake role test
scripts/export_pending_llm_tasks.py       — L2 real candidate export
scripts/generate_llm_judgments.py         — L3 LLM judgment generation
scripts/run_fast_sweep.sh                 — Phase 5 sweep launcher
scripts/run_l5_comparison.sh              — L5 comparison sweep
scripts/summarize_aggregation_runs.py     — Phase 6 summary
scripts/summarize_llm_collaboration_runs.py — L6 LLM summary
scripts/monitor_sweep.sh                  — Progress monitor
agent_notes/aggregation_experiment_journal.md — This journal
```

### Files Modified

```
conf/config.yaml              — Added aggregation default
src/dicode/selection.py       — Aggregation hook
src/dicode/dreaming/utils.py  — Safe embedding distance fix
```

### Running Experiment

- **l5c tmux session**: L5 comparison sweep (B4, B5, B5b)
- **Log**: `/root/experiments/dicode_runs/aggregation/logs/l5_comparison/`
- **ETA**: ~3 hours from launch (23:52 UTC → ~03:00 UTC)

### Output Locations

- Summary: `/root/experiments/dicode_runs/aggregation/summary.md`
- CSV: `/root/experiments/dicode_runs/aggregation/summary.csv`
- Recommendation: `/root/experiments/dicode_runs/aggregation/recommendation.md`
- LLM Summary: `/root/experiments/dicode_runs/aggregation/llm_summary.md`
- LLM CSV: `/root/experiments/dicode_runs/aggregation/llm_summary.csv`
- JSONL: `/root/experiments/dicode_runs/aggregation/mechanism_logs/aggregation_selector.jsonl`
- LLM Cache: `mechanism_logs/llm_judgments_cache.jsonl`
- Sweep Logs: `/root/experiments/dicode_runs/aggregation/logs/sweep/`
- L5 Logs: `/root/experiments/dicode_runs/aggregation/logs/l5_comparison/`

### Next Recommended Action

After L5 completes:
1. Run `python scripts/summarize_llm_collaboration_runs.py` to update L6 with L5 data
2. Compare B0/B1 (rule-based) vs B4/B5 (LLM-enhanced) entropy and composite scores
3. If LLM enhancement shows improvement, recommend for longer 5M-step run
