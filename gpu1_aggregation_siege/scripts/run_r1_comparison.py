#!/usr/bin/env python3
"""R1 Aggregation Comparison — real frozen pool, all 5 mechanisms, full diagnostics.

Uses ProductionDispatcher with real frozen pool (32→8) and immutable cache.
Computes: Jaccard agreement, rank correlation, source diversity, budget utilization,
selection overlap matrix, mechanism differentiation.

GPU1 Item 4: Proceed directly from R0 PASS to R1 comparisons.
CPU-only analysis. No training.
"""
import sys, os, json, hashlib, time
import numpy as np

_siege = os.path.join(os.path.dirname(__file__), "..", "src")
_agg = "/root/experiments/dicode-aggregation-v2/src"
for p in [_siege, _agg]:
    if p in sys.path: sys.path.remove(p)
sys.path.insert(0, _siege); sys.path.insert(1, _agg)

from dicode.siege.production_dispatcher import (
    ProductionDispatcher, ALL_MECHANISMS, make_test_defaults, build_runtime_adapter,
)

OUTPUT_DIR = "/root/experiments/dicode_runs/siege_aggregation/r1_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("R1 AGGREGATION COMPARISON — Real Frozen Pool + All 5 Mechanisms")
print("=" * 70)

# 1. Initialize dispatcher with real pool
print("\n1. Loading real frozen pool via ProductionDispatcher")
d = ProductionDispatcher()
print(f"   Pool hash: {d.pool_hash}")
print(f"   Candidates: {d.candidate_count}, Select: {d.selected_count}")

# 2. Dispatch all 5 mechanisms
print("\n2. Dispatching all 5 mechanisms")
gm, cfg = make_test_defaults(d.pool)
results = {}
selection_sets = {}
score_vectors = {}

for mech in ALL_MECHANISMS:
    if mech == "original":
        r = d.dispatch(mech, gen_manager=gm, config=cfg)
    else:
        r = d.dispatch(mech)
    results[mech] = r
    selection_sets[mech] = set(r["selected_ids"])
    print(f"   {mech}: {len(r['selected_ids'])} selected")

# 3. Build score vectors for rank correlation
print("\n3. Building score vectors")
# For Original: rank-based scores (lower rank = higher score)
original_order = {tid: i for i, tid in enumerate(d.pool)}
for mech in ALL_MECHANISMS:
    sel = results[mech]["selected_ids"]
    # Score: 8 for selected in position 0, 7 for position 1, ..., 1 for position 7, 0 for unselected
    scores = np.zeros(32, dtype=np.float64)
    for rank, tid in enumerate(sel):
        if tid in original_order:
            scores[original_order[tid]] = float(8 - rank)
    score_vectors[mech] = scores

# 4. Jaccard agreement matrix
print("\n4. Pairwise Jaccard agreement")
for i, m1 in enumerate(ALL_MECHANISMS):
    for j, m2 in enumerate(ALL_MECHANISMS):
        if i >= j:
            continue
        s1, s2 = selection_sets[m1], selection_sets[m2]
        inter = len(s1 & s2)
        union = len(s1 | s2)
        jac = inter / max(1, union)
        overlap_pct = inter / 8.0
        print(f"   {m1[:20]:<20} vs {m2[:20]:<20}: Jaccard={jac:.3f} overlap={overlap_pct:.1%}")

# 5. Rank correlation (Spearman)
print("\n5. Spearman rank correlation")
try:
    from scipy.stats import spearmanr
    for i, m1 in enumerate(ALL_MECHANISMS):
        for j, m2 in enumerate(ALL_MECHANISMS):
            if i >= j:
                continue
            rho, pval = spearmanr(score_vectors[m1], score_vectors[m2])
            print(f"   {m1[:20]:<20} vs {m2[:20]:<20}: rho={rho:+.3f} p={pval:.4f}")
except ImportError:
    print("   scipy not available — computing manual rank correlation")
    def spearman_r(a, b):
        n = len(a)
        ra = np.argsort(np.argsort(a)).astype(np.float64)
        rb = np.argsort(np.argsort(b)).astype(np.float64)
        ra_mean, rb_mean = ra.mean(), rb.mean()
        num = ((ra - ra_mean) * (rb - rb_mean)).sum()
        den = np.sqrt(((ra - ra_mean)**2).sum() * ((rb - rb_mean)**2).sum())
        return num / max(1e-10, den)
    for i, m1 in enumerate(ALL_MECHANISMS):
        for j, m2 in enumerate(ALL_MECHANISMS):
            if i >= j:
                continue
            rho = spearman_r(score_vectors[m1], score_vectors[m2])
            print(f"   {m1[:20]:<20} vs {m2[:20]:<20}: rho={rho:+.3f}")

# 6. Source diversity
print("\n6. Source diversity (from cache source_ids)")
if d._cache_loaded and hasattr(d, 'source_ids_arr'):
    sources = d.source_ids_arr
    for mech in ALL_MECHANISMS:
        sel = results[mech]["selected_ids"]
        sel_sources = []
        for tid in sel:
            idx = original_order.get(tid)
            if idx is not None and idx < len(sources):
                sel_sources.append(sources[idx])
        unique = len(set(sel_sources))
        print(f"   {mech}: {unique} unique sources from {len(sel_sources)} selected")
else:
    print("   Cache signals not loaded (enhanced only)")

# 7. Mechanism differentiation
print("\n7. Mechanism differentiation")
all_sets = [frozenset(selection_sets[m]) for m in ALL_MECHANISMS]
unique_selection_sets = len(set(all_sets))
print(f"   Unique selection sets: {unique_selection_sets} / {len(ALL_MECHANISMS)}")
if unique_selection_sets >= 2:
    print("   Mechanisms produce DIFFERENT selections — aggregation is non-trivial")
else:
    print("   WARNING: All mechanisms select identical sets — aggregation has no effect")

# 8. Budget effect
print("\n8. Budget effect")
budget_mechanisms = ["soft_copeland", "budgeted_copeland", "auction_raw", "auction_budgeted"]
for mech in budget_mechanisms:
    if mech not in results:
        continue
    r = results[mech]
    trace = r.get("trace", {})
    if "budget_info" in trace:
        bi = trace["budget_info"]
        print(f"   {mech}: budget_info keys={list(bi.keys())[:5]}")
    elif "auction" in trace:
        auc = trace["auction"]
        print(f"   {mech}: total_utility={auc.get('total_utility', 'N/A')}, "
              f"budget_changed={auc.get('budget_changed_selection', 'N/A')}")

# 9. Runtime adapter compatibility
print("\n9. Runtime adapter validation")
for mech in ALL_MECHANISMS:
    adp = build_runtime_adapter(results[mech])
    n_hashes = len(set(adp["candidate_hashes"]))
    print(f"   {mech}: {adp['n_tasks']} classes, {n_hashes} unique hashes, dist sum={sum(adp['distribution']):.3f}")

# 10. Evidence manifest
evidence = {
    "r1_timestamp": time.time(),
    "pool_hash": d.pool_hash,
    "cache_hit_rate": d.cache_hit_rate,
    "original_zero_cache": results["original"]["trace"]["cache_reads"] == 0,
    "original_injected": results["original"]["trace"]["gen_manager_injected"],
    "mechanisms": list(ALL_MECHANISMS),
    "unique_selection_sets": unique_selection_sets,
    "jaccard": {},
    "spearman": {},
    "selections": {m: sorted(list(selection_sets[m])) for m in ALL_MECHANISMS},
    "status": "R1_COMPARISON_COMPLETE",
}

# Fill Jaccard
for i, m1 in enumerate(ALL_MECHANISMS):
    for j, m2 in enumerate(ALL_MECHANISMS):
        if i >= j:
            continue
        s1, s2 = selection_sets[m1], selection_sets[m2]
        inter = len(s1 & s2)
        evidence["jaccard"][f"{m1}_vs_{m2}"] = round(inter / max(1, len(s1 | s2)), 4)

with open(os.path.join(OUTPUT_DIR, "r1_evidence.json"), "w") as f:
    json.dump(evidence, f, indent=2, default=str)

print(f"\n{'=' * 70}")
print(f"R1 COMPARISON COMPLETE — evidence at {OUTPUT_DIR}/r1_evidence.json")
print(f"Unique selection sets: {unique_selection_sets}/5")
print(f"Cache hit rate: {d.cache_hit_rate}")
print(f"Original zero-cache: {results['original']['trace']['cache_reads'] == 0}")
print(f"{'=' * 70}")
