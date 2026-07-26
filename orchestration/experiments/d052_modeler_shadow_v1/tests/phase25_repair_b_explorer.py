"""TRANSPARENT REPAIR (0 new LLM calls).
glm-4-flash returned a valid 32-item explorer array for arm B, but self-mislabeled the
echoed `role` field as 'builder'/'survivor' on 9 entries. The substantive explorer signal
(novelty_score, diversity_score, decision, short_reason) is intact and complete; only the
cosmetic self-label is wrong, and `role` is NOT consumed by Soft Copeland (signal is read
by score keys). We normalize ONLY the echoed role string to 'explorer' (the authoritative
role of this call), preserve the original label in `original_role_echo`, re-run the exact
hard validation gate, and write the judgments file. No scores/decisions are altered.
"""
import sys, json, os
OUTDIR = sys.argv[1]; POOL = sys.argv[2]
RAW = os.path.join(OUTDIR, "bc_B_explorer_FAILED_raw.json")
OUT = os.path.join(OUTDIR, "bc_B_explorer_judgments.jsonl")
LOG = os.path.join(OUTDIR, "bc_B_explorer_normalization_log.json")

pool = json.load(open(POOL)); cands = pool["candidates"]
real_ids = sorted(c["task_id"] for c in cands); assert len(real_ids) == 32
anon_to_real = {"C%03d" % (i + 1): tid for i, tid in enumerate(real_ids)}

d = json.load(open(RAW)); raw = d["raw"]; meta = d["meta"]
REQ = ["novelty_score", "diversity_score"]; DEC = {"accept", "hold", "reject"}
assert isinstance(raw, list) and len(raw) == 32, "raw not a 32-list"

norm_log = []
entries = []
errs = []
got = set()
for j in raw:
    aid = j.get("task_id")
    if aid not in anon_to_real: errs.append("unmapped id %r" % aid); continue
    got.add(aid)
    scc = j.get("scores", {})
    for k in REQ:
        if not isinstance(scc.get(k), (int, float)):
            errs.append("%s non-numeric %s=%r" % (aid, k, scc.get(k)))
    if j.get("decision") not in DEC: errs.append("%s bad decision %r" % (aid, j.get("decision")))
    orig_role = j.get("role")
    fixed = dict(j); fixed["role"] = "explorer"  # normalize echoed label only
    if orig_role != "explorer":
        norm_log.append({"anon_id": aid, "task_id": anon_to_real[aid],
                         "original_role_echo": orig_role, "normalized_to": "explorer"})
    entries.append({"anon_id": aid, "task_id": anon_to_real[aid], "role": "explorer", "arm": "B",
                    "judgment": fixed, "original_role_echo": orig_role,
                    "provider": "gl", "model_rq": meta.get("model_rq"), "model_rt": meta.get("model_rt")})

miss = set(anon_to_real.keys()) - got
if miss: errs.append("MISSING %s" % sorted(miss))
if errs:
    print("REPAIR_VALIDATION_FAILED:")
    for e in errs: print("  - " + e)
    sys.exit(2)

with open(OUT, "w") as f:
    for e in entries:
        f.write(json.dumps(e, sort_keys=True, ensure_ascii=False) + "\n")
json.dump({"arm": "B", "role": "explorer", "n_entries": len(entries),
           "n_role_echo_normalized": len(norm_log), "normalizations": norm_log,
           "note": "Only the echoed `role` string was normalized builder/survivor->explorer; "
                   "all scores/decisions/short_reason are the model's verbatim output. "
                   "`role` is not consumed by Soft Copeland. 0 new LLM calls.",
           "source_meta": meta}, open(LOG, "w"), indent=2, ensure_ascii=False)
print("REPAIR_OK entries=%d role_echo_normalized=%d" % (len(entries), len(norm_log)))
print("normalized ids:", [n["anon_id"] for n in norm_log])
