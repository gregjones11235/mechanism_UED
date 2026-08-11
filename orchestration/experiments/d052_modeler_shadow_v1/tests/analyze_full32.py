"""Phase 2 full-32 shadow analysis.
OLD = round-4 original judgment_cache (96). NEW = shadow judgments conditioned on the
frozen Modeler StudentProfile (96). Same original Soft Copeland code, same frozen pool.
Determinism anchors, selected-8 delta, full-rank correlation, per-role deltas, decision
flips, role ablation, and mapping of replacements to real Student gaps.
"""
import sys, json, os, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import soft_copeland_recompute as SC

ARCHIVE = sys.argv[1]; REMOTE_B = sys.argv[2]
OUT = os.path.join(REMOTE_B, "outputs"); AN = os.path.join(REMOTE_B, "analysis")
os.makedirs(AN, exist_ok=True)

pool = json.load(open(os.path.join(ARCHIVE, "round_4/frozen_candidate_pool.json")))
cands = pool["candidates"]; cand = {c["task_id"]: c for c in cands}
ids = sorted(cand.keys()); assert len(ids) == 32
old_all = [json.loads(l) for l in open(os.path.join(ARCHIVE, "round_4/judgment_cache.jsonl")) if l.strip()]
def load(role): return [json.loads(l) for l in open(os.path.join(OUT, "full32_%s_judgments.jsonl" % role)) if l.strip()]
new_all = load("tutor") + load("critic") + load("explorer")

# integrity: 96 each, full role coverage
def coverage(jents, tag):
    seen = {}
    for e in jents: seen.setdefault(e["task_id"], set()).add(e["role"])
    bad = [t for t in ids if seen.get(t) != {"tutor","critic","explorer"}]
    assert not bad, "%s missing coverage %s" % (tag, bad)
    assert len(jents) == 96, "%s expected 96 got %d" % (tag, len(jents))
coverage(old_all, "OLD"); coverage(new_all, "NEW")

# ---- determinism anchors ----
ph = SC.pool_hash(cands)
selO, scO, rawO = SC.recompute(cands, old_all, k=8)
selO2, scO2, _ = SC.recompute(cands, old_all, k=8)
assert selO == selO2 and scO == scO2, "OLD recompute not deterministic"
selN, scN, rawN = SC.recompute(cands, new_all, k=8)
selN2, scN2, _ = SC.recompute(cands, new_all, k=8)
assert selN == selN2 and scN == scN2, "NEW recompute not deterministic"

# cross-check OLD selected-8 == original round-4 selected-8 (Phase 0 anchor)
full = json.load(open(os.path.join(AN, "round4_full_ranking.json")))
orig_in8 = {row["task_id"]: bool(row.get("in_original_selected8")) for row in full}
orig_sel8 = sorted([t for t in ids if orig_in8[t]])
old_sel8_sorted = sorted(selO)
anchor_match = (old_sel8_sorted == orig_sel8)

# ---- helpers ----
def ranks(scores):
    order = sorted(range(len(scores)), key=lambda i: -scores[i]); r = [0]*len(scores)
    for rank, i in enumerate(order, 1): r[i] = rank
    return r
def spearman(a, b):
    ra, rb = ranks(a), ranks(b); n = len(a); ma = sum(ra)/n; mb = sum(rb)/n
    num = sum((ra[i]-ma)*(rb[i]-mb) for i in range(n))
    da = sum((x-ma)**2 for x in ra)**0.5; db = sum((x-mb)**2 for x in rb)**0.5
    return round(num/(da*db), 4) if da and db else None
def mean_abs(xs): return round(sum(abs(x) for x in xs)/len(xs), 4)

oR, nR = ranks(scO), ranks(scN)
def dec_map(jents):
    return {(e["task_id"], e["role"]): e["judgment"].get("decision") for e in jents}
od, nd = dec_map(old_all), dec_map(new_all)

old_set, new_set = set(selO), set(selN)
entered = sorted(new_set - old_set); exited = sorted(old_set - new_set)
overlap = sorted(old_set & new_set)
jaccard = round(len(old_set & new_set) / len(old_set | new_set), 4)

# ---- per-task rows ----
rows = []
for i, t in enumerate(ids):
    rows.append({
        "task_id": t, "tier": cand[t].get("difficulty_tier"),
        "target_achievements": ",".join(cand[t].get("target_achievements", [])),
        "old_prog": round(rawO["progression"][i],3), "new_prog": round(rawN["progression"][i],3),
        "d_prog": round(rawN["progression"][i]-rawO["progression"][i],3),
        "old_crit": round(rawO["critic_penalty"][i],3), "new_crit": round(rawN["critic_penalty"][i],3),
        "d_crit": round(rawN["critic_penalty"][i]-rawO["critic_penalty"][i],3),
        "old_nov": round(rawO["novelty"][i],3), "new_nov": round(rawN["novelty"][i],3),
        "d_nov": round(rawN["novelty"][i]-rawO["novelty"][i],3),
        "old_score": round(scO[i],4), "new_score": round(scN[i],4),
        "old_rank": oR[i], "new_rank": nR[i], "rank_delta": oR[i]-nR[i],
        "in_old_sel8": t in old_set, "in_new_sel8": t in new_set,
    })

# ---- role ablation: revert one role to OLD, keep other two NEW ----
old_by = {(e["task_id"], e["role"]): e for e in old_all}
new_by = {(e["task_id"], e["role"]): e for e in new_all}
def mixed(revert_role):
    jents = []
    for t in ids:
        for r in ("tutor","critic","explorer"):
            jents.append(old_by[(t,r)] if r == revert_role else new_by[(t,r)])
    return jents
ablation = {}
for rr in ("tutor","critic","explorer"):
    sel, sc, _ = SC.recompute(cands, mixed(rr), k=8)
    s = set(sel)
    ablation["revert_"+rr] = {
        "selected8": sorted(sel),
        "overlap_with_OLD": len(s & old_set),
        "overlap_with_NEW_all": len(s & new_set),
        "selection_hash": SC.selection_hash(sel),
    }

# ---- map replacements to real Student gaps ----
profile = json.load(open(os.path.join(OUT, "student_profile.json")))
skill_sr = {s["achievement_id"]: s["current_sr"] for s in profile["skills"]}
weak = sorted([a for a, sr in skill_sr.items() if sr is not None and sr < 0.2], key=lambda a: skill_sr[a])
def gap_note(t):
    achs = cand[t].get("target_achievements", [])
    canon = [a for a in achs if a in skill_sr]
    if not canon:
        return "UNDEFINED (salted-hash placeholders: %s)" % ",".join(achs) if achs else "UNDEFINED (no targets)"
    return "targets weak skill(s): " + ", ".join("%s(sr=%.3f)" % (a, skill_sr[a]) for a in canon if skill_sr[a] is not None and skill_sr[a] < 0.5) or "targets mastered/strong skill(s): " + ",".join(canon)
enter_map = {t: gap_note(t) for t in entered}

# ---- stats ----
decision_flips = sum(1 for t in ids for r in ("tutor","critic","explorer") if od[(t,r)] != nd[(t,r)])
stats = {
    "n_candidates": 32,
    "pool_hash": ph,
    "determinism_self_check": "PASS (OLD & NEW each identical across 2 recomputes)",
    "old_recompute_matches_original_round4_selected8": anchor_match,
    "old_selected8": old_sel8_sorted,
    "new_selected8": sorted(selN),
    "old_selection_hash": SC.selection_hash(selO),
    "new_selection_hash": SC.selection_hash(selN),
    "selection_hash_changed": SC.selection_hash(selO) != SC.selection_hash(selN),
    "selected8_overlap": len(old_set & new_set),
    "selected8_jaccard": jaccard,
    "entered_new_sel8": entered,
    "exited_old_sel8": exited,
    "spearman_full32_old_vs_new": spearman(scO, scN),
    "mean_abs_delta": {"progression": mean_abs([r["d_prog"] for r in rows]),
                        "critic_penalty": mean_abs([r["d_crit"] for r in rows]),
                        "novelty": mean_abs([r["d_nov"] for r in rows])},
    "decision_flips_of_96": decision_flips,
    "role_ablation": ablation,
    "student_weak_skills_sr_lt_0.2": weak,
    "entered_gap_mapping": enter_map,
}

# ---- write ----
json.dump(stats, open(os.path.join(AN, "full32_stats.json"), "w"), indent=2, ensure_ascii=False)
def ranking_json(sel, sc, raw):
    order = sorted(range(32), key=lambda i: -sc[i])
    out = []
    for rank, i in enumerate(order, 1):
        t = ids[i]
        out.append({"rank": rank, "task_id": t, "progression": round(raw["progression"][i],3),
                    "critic_penalty": round(raw["critic_penalty"][i],3), "novelty": round(raw["novelty"][i],3),
                    "soft_copeland_score": round(sc[i],4), "difficulty_tier": cand[t].get("difficulty_tier"),
                    "target_achievements": cand[t].get("target_achievements", []),
                    "in_old_selected8": t in old_set, "in_new_selected8": t in new_set})
    return out
json.dump(ranking_json(selO, scO, rawO), open(os.path.join(AN, "full32_old_ranking.json"), "w"), indent=2, ensure_ascii=False)
json.dump(ranking_json(selN, scN, rawN), open(os.path.join(AN, "full32_new_ranking.json"), "w"), indent=2, ensure_ascii=False)
with open(os.path.join(AN, "full32_ranking_delta.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["task_id","tier","old_score","new_score","old_rank","new_rank","rank_delta","in_old_sel8","in_new_sel8"])
    w.writeheader()
    for r in sorted(rows, key=lambda x: x["old_rank"]):
        w.writerow({k: r[k] for k in w.fieldnames})
with open(os.path.join(AN, "full32_judgment_delta.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["task_id","old_prog","new_prog","d_prog","old_crit","new_crit","d_crit","old_nov","new_nov","d_nov","in_old_sel8","in_new_sel8"])
    w.writeheader()
    for r in sorted(rows, key=lambda x: x["task_id"]):
        w.writerow({k: r[k] for k in w.fieldnames})

print(json.dumps(stats, indent=2, ensure_ascii=False))
print("\nFULL32_ANALYSIS_OK")
