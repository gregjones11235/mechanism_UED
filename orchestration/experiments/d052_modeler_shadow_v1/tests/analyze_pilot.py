"""Phase 1 controlled comparison.
Same 8-candidate sub-pool, SAME original Soft Copeland code. Only the judgments differ:
  OLD = round-4 original judgment_cache (sparse snapshot_str conditioning)
  NEW = shadow judgments conditioned on the Modeler StudentProfile
Outputs ranking tables, judgment/ranking deltas, Spearman rank corr, decision flips,
per-role mean-abs signal deltas, and a determinism self-check (recompute twice -> identical).
"""
import sys, json, os, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import soft_copeland_recompute as SC

ARCHIVE = sys.argv[1]      # .../seed0_1784462982
REMOTE_B = sys.argv[2]
OUT = os.path.join(REMOTE_B, "outputs")
AN = os.path.join(REMOTE_B, "analysis")
os.makedirs(AN, exist_ok=True)

pool = json.load(open(os.path.join(ARCHIVE, "round_4/frozen_candidate_pool.json")))
cand = {c["task_id"]: c for c in pool["candidates"]}
old_all = [json.loads(l) for l in open(os.path.join(ARCHIVE, "round_4/judgment_cache.jsonl")) if l.strip()]
pilot = json.load(open(os.path.join(REMOTE_B, "manifests/pilot8_manifest.json")))
chosen = pilot["chosen"]

def load_new(role):
    return [json.loads(l) for l in open(os.path.join(OUT, role + "_judgments.jsonl")) if l.strip()]
new_all = load_new("tutor") + load_new("critic") + load_new("explorer")

cands8 = [cand[t] for t in chosen]
old8 = [e for e in old_all if e["task_id"] in set(chosen)]
new8 = new_all

# integrity: each set must have exactly 24 judgments (8 x 3 roles), all roles per task
def check_coverage(jents, tag):
    seen = {}
    for e in jents:
        seen.setdefault(e["task_id"], set()).add(e["role"])
    bad = [t for t in chosen if seen.get(t) != {"tutor", "critic", "explorer"}]
    assert not bad, "%s missing role coverage for %s" % (tag, bad)
    assert len(jents) == 24, "%s expected 24 got %d" % (tag, len(jents))
check_coverage(old8, "OLD")
check_coverage(new8, "NEW")

# ---- determinism self-check (original code, same inputs -> identical) ----
selA, scA, _ = SC.recompute(cands8, old8, k=8)
selB, scB, _ = SC.recompute(cands8, old8, k=8)
assert selA == selB and scA == scB, "DETERMINISM FAIL on OLD recompute"
selN, scN, rawN = SC.recompute(cands8, new8, k=8)
selN2, scN2, _ = SC.recompute(cands8, new8, k=8)
assert selN == selN2 and scN == scN2, "DETERMINISM FAIL on NEW recompute"

_, scO, rawO = SC.recompute(cands8, old8, k=8)

# ---- per-task table ----
def ranks(scores):
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    r = [0] * len(scores)
    for rank, i in enumerate(order, 1):
        r[i] = rank
    return r

ids = [c["task_id"] for c in cands8]
oR = ranks(scO); nR = ranks(scN)
# decisions per role
def dec_map(jents):
    m = {}
    for e in jents:
        m[(e["task_id"], e["role"])] = e["judgment"].get("decision")
    return m
od, nd = dec_map(old8), dec_map(new8)
# original full-32 rank context
full = json.load(open(os.path.join(AN, "round4_full_ranking.json")))
full_rank = {row["task_id"]: row["rank"] for row in full} if isinstance(full, list) else {}
full_in8 = {row["task_id"]: row.get("in_original_selected8") for row in full} if isinstance(full, list) else {}

rows = []
for i, t in enumerate(ids):
    rows.append({
        "task_id": t,
        "tier": cand[t].get("difficulty_tier"),
        "old_prog": round(rawO["progression"][i], 3), "new_prog": round(rawN["progression"][i], 3),
        "d_prog": round(rawN["progression"][i] - rawO["progression"][i], 3),
        "old_crit": round(rawO["critic_penalty"][i], 3), "new_crit": round(rawN["critic_penalty"][i], 3),
        "d_crit": round(rawN["critic_penalty"][i] - rawO["critic_penalty"][i], 3),
        "old_nov": round(rawO["novelty"][i], 3), "new_nov": round(rawN["novelty"][i], 3),
        "d_nov": round(rawN["novelty"][i] - rawO["novelty"][i], 3),
        "old_score": round(scO[i], 4), "new_score": round(scN[i], 4),
        "old_rank_sub8": oR[i], "new_rank_sub8": nR[i], "rank_delta": oR[i] - nR[i],
        "orig_full32_rank": full_rank.get(t),
        "orig_in_selected8": full_in8.get(t),
        "old_dec_T/C/E": "%s/%s/%s" % (od[(t,"tutor")], od[(t,"critic")], od[(t,"explorer")]),
        "new_dec_T/C/E": "%s/%s/%s" % (nd[(t,"tutor")], nd[(t,"critic")], nd[(t,"explorer")]),
    })

# ---- stats ----
def mean_abs(xs): return round(sum(abs(x) for x in xs) / len(xs), 4)
def spearman(a, b):
    ra = ranks(a); rb = ranks(b)
    n = len(a); ma = sum(ra)/n; mb = sum(rb)/n
    num = sum((ra[i]-ma)*(rb[i]-mb) for i in range(n))
    da = sum((x-ma)**2 for x in ra) ** 0.5; db = sum((x-mb)**2 for x in rb) ** 0.5
    return round(num/(da*db), 4) if da and db else None

decision_flips = sum(1 for t in chosen for r in ("tutor","critic","explorer") if od[(t,r)] != nd[(t,r)])
top4_old = set(sorted(ids, key=lambda t: -scO[ids.index(t)])[:4])
top4_new = set(sorted(ids, key=lambda t: -scN[ids.index(t)])[:4])

stats = {
    "n_candidates": 8,
    "determinism_self_check": "PASS (OLD & NEW recompute identical across 2 runs)",
    "spearman_old_vs_new_rank": spearman(scO, scN),
    "mean_abs_delta": {"progression": mean_abs([r["d_prog"] for r in rows]),
                        "critic_penalty": mean_abs([r["d_crit"] for r in rows]),
                        "novelty": mean_abs([r["d_nov"] for r in rows])},
    "decision_flips_of_24": decision_flips,
    "top4_subpool_old": sorted(top4_old),
    "top4_subpool_new": sorted(top4_new),
    "top4_overlap": len(top4_old & top4_new),
    "old_selection_hash_sub8": SC.selection_hash(selA),
    "new_selection_hash_sub8": SC.selection_hash(selN),
}

# ---- write ----
json.dump({"ordering": ids, "scores": [round(s,4) for s in scO],
           "selection_order": selA, "raw": {k:[round(float(x),4) for x in v] for k,v in rawO.items()}},
          open(os.path.join(AN, "pilot8_old_ranking.json"), "w"), indent=2)
json.dump({"ordering": ids, "scores": [round(s,4) for s in scN],
           "selection_order": selN, "raw": {k:[round(float(x),4) for x in v] for k,v in rawN.items()}},
          open(os.path.join(AN, "pilot8_new_ranking.json"), "w"), indent=2)
json.dump(stats, open(os.path.join(AN, "pilot8_stats.json"), "w"), indent=2)

with open(os.path.join(AN, "ranking_delta.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["task_id","tier","old_score","new_score","old_rank_sub8","new_rank_sub8","rank_delta","orig_full32_rank","orig_in_selected8"])
    w.writeheader()
    for r in sorted(rows, key=lambda x: x["old_rank_sub8"]):
        w.writerow({k: r[k] for k in w.fieldnames})
with open(os.path.join(AN, "judgment_delta.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["task_id","old_prog","new_prog","d_prog","old_crit","new_crit","d_crit","old_nov","new_nov","d_nov","old_dec_T/C/E","new_dec_T/C/E"])
    w.writeheader()
    for r in sorted(rows, key=lambda x: x["task_id"]):
        w.writerow({k: r[k] for k in w.fieldnames})

print(json.dumps(stats, indent=2))
print("\nRANKING (by old_rank):")
for r in sorted(rows, key=lambda x: x["old_rank_sub8"]):
    print("  %-14s tier=%-7s old#%d(new#%d) score %.3f->%.3f  full32#%s in8=%s  dec %s -> %s" % (
        r["task_id"], str(r["tier"]), r["old_rank_sub8"], r["new_rank_sub8"],
        r["old_score"], r["new_score"], r["orig_full32_rank"], r["orig_in_selected8"],
        r["old_dec_T/C/E"], r["new_dec_T/C/E"]))
print("\nPILOT_ANALYSIS_OK")
