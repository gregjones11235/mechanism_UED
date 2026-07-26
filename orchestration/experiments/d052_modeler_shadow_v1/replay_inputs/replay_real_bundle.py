"""OFFLINE, DETERMINISTIC, NO-LLM replay of the REAL D052 Phase 2.5 bundle.

Recomputes the Soft-Copeland selected-8 for arms B and C from the bundle's
judgments_{B,C}.jsonl using the ORIGINAL selector source (SHA 27492e8a...) that
Modeler CC used, and checks every historical anchor in expected_behavior.json.

Faithful to tests/soft_copeland_recompute.py:
  - candidate order = sorted(task_id)
  - progression = tutor.progression_score ; critic_penalty = critic.critic_penalty
    novelty = explorer.novelty_score ; retention = 1 - critic_penalty
    monopoly_penalty = 0 ; source_ids="d052" ; skill_counts=1
  - scores = _aggregate_soft_copeland(sig, WEIGHTS, 1.0)   (WEIGHTS 0.34/0.33/0.33/0.01/0.01)
  - selected = task_ids[np.argsort(-scores)[:8]]
  - selection_hash = sha256(json.dumps(sorted(selected)))[:16]

If ANY anchor mismatches the script exits non-zero and does NOT mutate anything.
"""
import importlib.util, json, hashlib, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.normpath(os.path.join(HERE, "..", "artifacts", "d052_phase25_canonical_migration"))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "replay_result.json")

# --- load the ORIGINAL selector source (exact bytes that produced the anchors) ---
AGG_PATH = os.path.join(HERE, "aggregation_original.py")
AGG_SHA = hashlib.sha256(open(AGG_PATH, "rb").read()).hexdigest()
spec = importlib.util.spec_from_file_location("agg_original", AGG_PATH)
AGG = importlib.util.module_from_spec(spec); spec.loader.exec_module(AGG)

WEIGHTS = {"w_progression": 0.34, "w_retention": 0.33, "w_novelty": 0.33,
           "w_critic": 0.01, "w_monopoly": 0.01}

def sha16(x):
    d = x.encode() if isinstance(x, str) else x
    return hashlib.sha256(d).hexdigest()[:16]

def load(arm):
    p = os.path.join(BUNDLE, "judgments_%s.jsonl" % arm)
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def build_signals(recs):
    ids = sorted({r["task_id"] for r in recs})          # candidate order = sorted(task_id)
    n = len(ids); idx = {t: i for i, t in enumerate(ids)}
    by_role = {role: {} for role in ("tutor", "critic", "explorer")}
    for r in recs:
        by_role[r["role"]][r["task_id"]] = r["raw_scores"]
    prog = np.full(n, 0.5); nov = np.full(n, 0.5); crit = np.zeros(n)
    for t, i in idx.items():
        prog[i] = float(by_role["tutor"].get(t, {}).get("progression_score", 0.5))
        crit[i] = float(by_role["critic"].get(t, {}).get("critic_penalty", 0.0))
        nov[i] = float(by_role["explorer"].get(t, {}).get("novelty_score", 0.5))
    ret = 1.0 - crit
    sig = {"progression": prog, "retention": ret, "novelty": nov,
           "critic_penalty": crit, "monopoly_penalty": np.zeros(n),
           "source_ids": np.array(["d052"] * n), "skill_counts": np.ones(n)}
    return ids, sig

def run_arm(arm):
    recs = load(arm)
    ids, sig = build_signals(recs)
    scores = AGG._aggregate_soft_copeland(sig, WEIGHTS, 1.0)
    sel_idx = np.argsort(-scores)[:8]
    sel = [ids[i] for i in sel_idx]
    return {"n_candidates": len(ids), "selected8_score_order": sel,
            "selected8_sorted": sorted(sel),
            "selection_hash": sha16(json.dumps(sorted(sel))),
            "scores": [float(x) for x in scores]}

eb = json.load(open(os.path.join(BUNDLE, "expected_behavior.json"), encoding="utf-8"))
sc = json.load(open(os.path.join(BUNDLE, "selector_config.json"), encoding="utf-8"))

B1, C1 = run_arm("B"), run_arm("C")
B2, C2 = run_arm("B"), run_arm("C")     # second pass -> determinism

# pool_hash check (legacy pool anchor) using the fetched round_4 pool
pool_path = os.path.join(HERE, "frozen_candidate_pool_round4.json")
pool_hash = None
if os.path.exists(pool_path):
    pool = json.load(open(pool_path, encoding="utf-8"))["candidates"]
    spec_list = sorted([{"id": c["task_id"], "tp": c.get("task_params", {}),
                         "achs": sorted(c.get("target_achievements", [])),
                         "prov": c.get("_prov", {})} for c in pool], key=lambda x: x["id"])
    pool_hash = sha16(json.dumps(spec_list, sort_keys=True))

overlap = set(eb["B_selected8"]) & set(eb["C_selected8"])
union = set(eb["B_selected8"]) | set(eb["C_selected8"])
change = 8 - len(overlap)
jaccard = len(overlap) / len(union)

checks = {
    "agg_source_sha_matches_bundle": AGG_SHA == sc["selector_source_sha256"],
    "B_selection_hash_match": B1["selection_hash"] == eb["B_selection_hash"],
    "C_selection_hash_match": C1["selection_hash"] == eb["C_selection_hash"],
    "B_selected8_exact_set": B1["selected8_sorted"] == sorted(eb["B_selected8"]),
    "C_selected8_exact_set": C1["selected8_sorted"] == sorted(eb["C_selected8"]),
    "candidate_count_32": B1["n_candidates"] == 32 and C1["n_candidates"] == 32,
    "change_is_4_of_8": change == 4 and eb["selected_set_change"] == "4/8",
    "jaccard_match": abs(jaccard - eb["jaccard"]) < 1e-3,
    "overlap_4": len(overlap) == 4,
    "B_determinism_bitidentical": (B1["selection_hash"] == B2["selection_hash"]
                                   and B1["scores"] == B2["scores"]
                                   and B1["selected8_score_order"] == B2["selected8_score_order"]),
    "C_determinism_bitidentical": (C1["selection_hash"] == C2["selection_hash"]
                                   and C1["scores"] == C2["scores"]
                                   and C1["selected8_score_order"] == C2["selected8_score_order"]),
    "selector_rng_seed_null": sc["rng_seed"] is None,
    "pool_hash_match": pool_hash == eb["legacy_pool_hash"],
}

result = {
    "replay_mode": "OFFLINE_DETERMINISTIC_NO_LLM",
    "aggregation_source_sha256": AGG_SHA,
    "aggregation_source_note": "27492e8a... = bundle selector_source_sha256 (server workers/gpu0_original); "
                               "robust_normalize & _aggregate_soft_copeland are byte-identical to the "
                               "worktree-frozen gpu1_aggregation_siege aggregation.py (590fcef4).",
    "weights": WEIGHTS, "temperature": 1.0, "k": 8,
    "expected_anchors": {"B_selection_hash": eb["B_selection_hash"],
                         "C_selection_hash": eb["C_selection_hash"],
                         "legacy_pool_hash": eb["legacy_pool_hash"],
                         "B_selected8": eb["B_selected8"], "C_selected8": eb["C_selected8"]},
    "recomputed": {
        "B_selection_hash": B1["selection_hash"], "C_selection_hash": C1["selection_hash"],
        "B_selected8_score_order": B1["selected8_score_order"],
        "C_selected8_score_order": C1["selected8_score_order"],
        "pool_hash": pool_hash,
        "overlap": sorted(overlap), "entered_C_only": sorted(set(eb["C_selected8"]) - set(eb["B_selected8"])),
        "exited_B_only": sorted(set(eb["B_selected8"]) - set(eb["C_selected8"])),
        "change": "%d/8" % change, "jaccard": round(jaccard, 4),
    },
    "checks": checks,
}
result["ALL_ANCHORS_PASS"] = all(checks.values())
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print("aggregation_source_sha256 :", AGG_SHA)
print("recomputed B_selection_hash:", B1["selection_hash"], "| expected:", eb["B_selection_hash"])
print("recomputed C_selection_hash:", C1["selection_hash"], "| expected:", eb["C_selection_hash"])
print("recomputed pool_hash       :", pool_hash, "| expected:", eb["legacy_pool_hash"])
print("B_selected8 (score order)  :", B1["selected8_score_order"])
print("C_selected8 (score order)  :", C1["selected8_score_order"])
print("change=%s jaccard=%.4f overlap=%d" % (result["recomputed"]["change"], jaccard, len(overlap)))
print("--- checks ---")
for k, v in checks.items():
    print(("  [PASS] " if v else "  [FAIL] ") + k)
print("ALL_ANCHORS_PASS =", result["ALL_ANCHORS_PASS"])
sys.exit(0 if result["ALL_ANCHORS_PASS"] else 1)
