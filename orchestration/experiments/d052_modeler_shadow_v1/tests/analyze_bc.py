"""Phase 2.5 matched-counterfactual analysis: B_NO_MODELER vs C_WITH_MODELER.
Same 32 pool/order/anon IDs, same templates/models/providers/temp/timeout/schema, same
Soft Copeland code. The ONLY difference between arms is whether the frozen Modeler
StudentProfile is appended after an identical deterministic raw numerical summary.
Computes: per-role B/C score & rank correlation, judgment flip rate, full-rank Spearman,
selected-8 overlap & Jaccard, replaced tasks, per-role ablation contribution, B/C
selection_hash, tokens & retries.
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
PH = SC.pool_hash(cands)

def load(arm, role):
    return [json.loads(l) for l in open(os.path.join(OUT, "bc_%s_%s_judgments.jsonl" % (arm, role))) if l.strip()]
def arm_jents(arm): return load(arm, "tutor") + load(arm, "critic") + load(arm, "explorer")
B = arm_jents("B"); C = arm_jents("C")

def coverage(jents, tag):
    seen = {}
    for e in jents: seen.setdefault(e["task_id"], set()).add(e["role"])
    bad = [t for t in ids if seen.get(t) != {"tutor", "critic", "explorer"}]
    assert not bad, "%s missing coverage %s" % (tag, bad)
    assert len(jents) == 96, "%s expected 96 got %d" % (tag, len(jents))
coverage(B, "B"); coverage(C, "C")

# determinism: each arm identical across 2 recomputes
selB, scB, rawB = SC.recompute(cands, B, k=8)
selB2, scB2, _ = SC.recompute(cands, B, k=8)
assert selB == selB2 and scB == scB2, "B recompute not deterministic"
selC, scC, rawC = SC.recompute(cands, C, k=8)
selC2, scC2, _ = SC.recompute(cands, C, k=8)
assert selC == selC2 and scC == scC2, "C recompute not deterministic"

def ranks(scores):
    order = sorted(range(len(scores)), key=lambda i: -scores[i]); r = [0]*len(scores)
    for rank, i in enumerate(order, 1): r[i] = rank
    return r
def spearman(a, b):
    ra, rb = ranks(a), ranks(b); n = len(a); ma = sum(ra)/n; mb = sum(rb)/n
    num = sum((ra[i]-ma)*(rb[i]-mb) for i in range(n))
    da = sum((x-ma)**2 for x in ra)**0.5; db = sum((x-mb)**2 for x in rb)**0.5
    return round(num/(da*db), 4) if da and db else None
def pearson(a, b):
    n = len(a); ma = sum(a)/n; mb = sum(b)/n
    num = sum((a[i]-ma)*(b[i]-mb) for i in range(n))
    da = sum((x-ma)**2 for x in a)**0.5; db = sum((x-mb)**2 for x in b)**0.5
    return round(num/(da*db), 4) if da and db else None
def mean_abs(xs): return round(sum(abs(x) for x in xs)/len(xs), 4)

# ---- per-role signal correlation (B vs C) ----
def sig(raw, key): return [round(float(raw[key][i]), 4) for i in range(32)]
role_corr = {}
for role, key in [("tutor", "progression"), ("critic", "critic_penalty"), ("explorer", "novelty")]:
    b = sig(rawB, key); c = sig(rawC, key)
    role_corr[role] = {"signal": key,
                       "spearman_rank": spearman(b, c), "pearson_score": pearson(b, c),
                       "mean_abs_delta": mean_abs([c[i]-b[i] for i in range(32)]),
                       "mean_B": round(sum(b)/32, 4), "mean_C": round(sum(c)/32, 4)}

# ---- judgment decision flips /96 ----
def dec_map(jents): return {(e["task_id"], e["role"]): e["judgment"].get("decision") for e in jents}
bd, cd = dec_map(B), dec_map(C)
flips = [(t, r) for t in ids for r in ("tutor", "critic", "explorer") if bd[(t, r)] != cd[(t, r)]]
flip_by_role = {r: sum(1 for (t, rr) in flips if rr == r) for r in ("tutor", "critic", "explorer")}

# ---- selected-8 comparison ----
setB, setC = set(selB), set(selC)
entered = sorted(setC - setB); exited = sorted(setB - setC)
overlap = sorted(setB & setC)
jaccard = round(len(setB & setC) / len(setB | setC), 4)
n_changed = len(entered)  # == len(exited)
hashB, hashC = SC.selection_hash(selB), SC.selection_hash(selC)

# ---- full ranking ----
rB, rC = ranks(scB), ranks(scC)
full_spearman = spearman(scB, scC)

# ---- per-role ablation: C with one role reverted to B -> contribution of that role ----
B_by = {(e["task_id"], e["role"]): e for e in B}
C_by = {(e["task_id"], e["role"]): e for e in C}
def mixed(revert_role):
    return [B_by[(t, r)] if r == revert_role else C_by[(t, r)] for t in ids for r in ("tutor", "critic", "explorer")]
ablation = {}
for rr in ("tutor", "critic", "explorer"):
    sel, sc, _ = SC.recompute(cands, mixed(rr), k=8)
    s = set(sel)
    ablation["C_revert_" + rr + "_to_B"] = {
        "selected8": sorted(sel),
        "overlap_with_B": len(s & setB),
        "overlap_with_C": len(s & setC),
        "selection_hash": SC.selection_hash(sel),
        "interpretation": ("reverting %s to B moves selection toward B by %d (overlap_with_B)"
                           % (rr, len(s & setB)))
    }

# ---- per-task rows ----
rows = []
for i, t in enumerate(ids):
    rows.append({
        "task_id": t, "anon": "C%03d" % (ids.index(t) + 1), "tier": cand[t].get("difficulty_tier"),
        "target_achievements": ",".join(cand[t].get("target_achievements", [])),
        "B_prog": round(float(rawB["progression"][i]), 3), "C_prog": round(float(rawC["progression"][i]), 3),
        "d_prog": round(float(rawC["progression"][i]-rawB["progression"][i]), 3),
        "B_crit": round(float(rawB["critic_penalty"][i]), 3), "C_crit": round(float(rawC["critic_penalty"][i]), 3),
        "d_crit": round(float(rawC["critic_penalty"][i]-rawB["critic_penalty"][i]), 3),
        "B_nov": round(float(rawB["novelty"][i]), 3), "C_nov": round(float(rawC["novelty"][i]), 3),
        "d_nov": round(float(rawC["novelty"][i]-rawB["novelty"][i]), 3),
        "B_score": round(scB[i], 4), "C_score": round(scC[i], 4),
        "B_rank": rB[i], "C_rank": rC[i], "rank_delta": rB[i]-rC[i],
        "in_B_sel8": t in setB, "in_C_sel8": t in setC,
        "B_dec_t": bd[(t, "tutor")], "C_dec_t": cd[(t, "tutor")],
        "B_dec_c": bd[(t, "critic")], "C_dec_c": cd[(t, "critic")],
        "B_dec_e": bd[(t, "explorer")], "C_dec_e": cd[(t, "explorer")],
    })

# ---- tokens & retries ----
cost = json.load(open(os.path.join(OUT, "llm_cost_phase25.json")))
tok = {}
for k, v in cost.items():
    tok[k] = {"attempts": v.get("attempts"), "itok": v.get("itok"), "otok": v.get("otok"), "err": v.get("err")}
tot_itok = sum(v.get("itok") or 0 for v in cost.values())
tot_otok = sum(v.get("otok") or 0 for v in cost.values())
tot_attempts = sum(v.get("attempts") or 0 for v in cost.values())

stats = {
    "design": "B_NO_MODELER vs C_WITH_MODELER; identical deterministic raw summary; C adds frozen StudentProfile; "
              "same 32 pool/order/anon IDs, same templates/models(qw/ds/gl)/temperature(0)/timeout/schema, "
              "same Soft Copeland code; ONLY profile-append differs.",
    "n_candidates": 32, "pool_hash": PH,
    "determinism_self_check": "PASS (B & C each identical across 2 recomputes)",
    "per_role_B_vs_C": role_corr,
    "decision_flips_of_96": len(flips),
    "decision_flips_by_role": flip_by_role,
    "B_selected8": sorted(selB), "C_selected8": sorted(selC),
    "B_selection_hash": hashB, "C_selection_hash": hashC,
    "selection_hash_changed": hashB != hashC,
    "selected8_n_changed": n_changed,
    "selected8_overlap": len(setB & setC), "selected8_jaccard": jaccard,
    "entered_C_only": entered, "exited_B_only": exited, "shared": overlap,
    "full_rank_spearman_B_vs_C": full_spearman,
    "role_ablation_contribution": ablation,
    "tokens_retries": {"per_call": tok, "total_itok": tot_itok, "total_otok": tot_otok,
                       "total_attempts": tot_attempts, "n_calls": len(cost)},
    "verdict_rule_inputs": {
        "selected8_changed_ge_2_of_8": n_changed >= 2,
        "salted_hash_affects_training_semantics": True,  # from Phase 2.5a audit
        "note": "Per rule: salted hash affecting training semantics => STOP_AND_FIX_CANONICAL_TARGETS "
                "takes precedence regardless of B/C magnitude."
    },
}

json.dump(stats, open(os.path.join(AN, "bc_stats.json"), "w"), indent=2, ensure_ascii=False)

def ranking_json(sel, sc, raw, tag):
    order = sorted(range(32), key=lambda i: -sc[i]); out = []
    for rank, i in enumerate(order, 1):
        t = ids[i]
        out.append({"rank": rank, "task_id": t, "progression": round(float(raw["progression"][i]), 3),
                    "critic_penalty": round(float(raw["critic_penalty"][i]), 3),
                    "novelty": round(float(raw["novelty"][i]), 3),
                    "soft_copeland_score": round(sc[i], 4), "tier": cand[t].get("difficulty_tier"),
                    "target_achievements": cand[t].get("target_achievements", []),
                    "in_B_sel8": t in setB, "in_C_sel8": t in setC})
    return out
json.dump(ranking_json(selB, scB, rawB, "B"), open(os.path.join(AN, "bc_B_ranking.json"), "w"), indent=2, ensure_ascii=False)
json.dump(ranking_json(selC, scC, rawC, "C"), open(os.path.join(AN, "bc_C_ranking.json"), "w"), indent=2, ensure_ascii=False)

with open(os.path.join(AN, "bc_judgment_delta.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["task_id", "anon", "tier", "B_prog", "C_prog", "d_prog",
        "B_crit", "C_crit", "d_crit", "B_nov", "C_nov", "d_nov",
        "B_dec_t", "C_dec_t", "B_dec_c", "C_dec_c", "B_dec_e", "C_dec_e", "in_B_sel8", "in_C_sel8"])
    w.writeheader()
    for r in sorted(rows, key=lambda x: x["task_id"]):
        w.writerow({k: r[k] for k in w.fieldnames})
with open(os.path.join(AN, "bc_ranking_delta.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["task_id", "anon", "tier", "B_score", "C_score", "B_rank", "C_rank",
        "rank_delta", "in_B_sel8", "in_C_sel8"])
    w.writeheader()
    for r in sorted(rows, key=lambda x: x["B_rank"]):
        w.writerow({k: r[k] for k in w.fieldnames})

print(json.dumps(stats, indent=2, ensure_ascii=False))
print("\nBC_ANALYSIS_OK")
