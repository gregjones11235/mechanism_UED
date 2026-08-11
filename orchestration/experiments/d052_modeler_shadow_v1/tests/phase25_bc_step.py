"""Phase 2.5 matched counterfactual: B_NO_MODELER vs C_WITH_MODELER.
Two arms, 3 roles each = 6 batched calls. The ONLY difference between arms is whether
the frozen Modeler StudentProfile is appended after an IDENTICAL deterministic raw
numerical summary. Same 32 pool, same sorted order, same anonymized IDs (C001..C032),
same prompt template, same models/providers/temperature/timeout/output schema.
Hard-fail validation, idempotent skip, no silent fallback.
"""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_client as L

POOL = sys.argv[1]; PROFILE = sys.argv[2]; BASE = sys.argv[3]; OUTDIR = sys.argv[4]; COST = sys.argv[5]

pool = json.load(open(POOL)); cands = pool["candidates"]
cand = {c["task_id"]: c for c in cands}
real_ids = sorted(cand.keys()); assert len(real_ids) == 32
# anonymized IDs, identical across arms (sorted task_id -> C001..C032)
anon = {tid: "C%03d" % (i + 1) for i, tid in enumerate(real_ids)}
anon_to_real = {v: k for k, v in anon.items()}
profile = json.load(open(PROFILE)); base = json.load(open(BASE))

# ---- deterministic raw numerical summary (NO Modeler interpretation) ----
el = base["episode_level"]
pac = base["per_achievement_completion"]
sc = base.get("skill_chain", {})
bps = base.get("dominant_breakpoints", [])
eb = base.get("evidence_boundaries", {})
lines = []
lines.append("Deterministic Student statistics (computed by code; round-4 checkpoint, 64 real per-episode evals):")
lines.append("mean_return=%.4f; death_rate=%.2f; timeout_rate=%.2f; mean_episode_length=%.2f; mean_achievement_count=%.3f; max_return=%.2f; min_return=%.2f."
             % (el["mean_return"], el["death_rate"], el["timeout_rate"], el["mean_episode_length"],
                el["mean_achievement_count"], el["max_return"], el["min_return"]))
lines.append("Per-achievement empirical completion rates: " +
             ", ".join("%s=%.4f(n=%d)" % (a["achievement_id"], a["completion_rate"], a["episodes_completed"]) for a in pac) + ".")
lines.append("Canonical skill-chain frontier ever achieved (deterministic): %s (depth %s of %s)."
             % (sc.get("frontier_ever_achieved"), sc.get("frontier_ever_depth"), sc.get("n_canonical_achievements")))
lines.append("Deterministic low-completion breakpoints: " +
             ", ".join("%s=%.4f" % (b["achievement_id"], b["completion_rate"]) for b in bps) + ".")
lines.append("Evidence boundaries (deterministic): " + json.dumps(eb, ensure_ascii=False) + ".")
RAW_SUMMARY = "\n".join(lines)

PROFILE_JSON = json.dumps(profile, ensure_ascii=False)

# ---- anonymized candidate block (identical across arms) ----
def cb(tid):
    c = cand[tid]; tp = c.get("task_params", {})
    return ("Task:%s Desc:%s Achs:%s Tier:%s Spawn:%s Health:%s Damage:%s" % (
        anon[tid], c.get("description", "Task " + anon[tid]), ",".join(c.get("target_achievements", [])),
        c.get("difficulty_tier", "medium"), tp.get("passive_spawn_multiplier", "?"),
        tp.get("mob_health_multiplier", "?"), tp.get("mob_damage_multiplier", "?")))
CANDS_TXT = "\n".join(cb(t) for t in real_ids)

ROLE_INSTR = {
    "tutor": ("Evaluate Craftax task progression/teaching value for a curriculum. For each task judge current "
        "learnability, scaffolding soundness, downstream skill-chain value, and whether it merely repeats an "
        "already-mastered ability."),
    "critic": ("Evaluate Craftax task failure risk for this student. For each task judge whether it is too far "
        "ahead, invites shortcuts, gives trivial success, uses wrong scaffolding, risks negative transfer, or "
        "wrongly assumes the student state."),
    "explorer": ("Evaluate Craftax task novelty and curriculum diversity for this student. For each task judge "
        "skill coverage, novelty vs redundant candidates, and alternative skill paths."),
}
SCHEMA = {
    "tutor": '{"task_id":"...","role":"tutor","scores":{"progression_score":X.XX,"learnability_score":X.XX},"decision":"accept|hold|reject","short_reason":"..."} . Scores 0-10.',
    "critic": '{"task_id":"...","role":"critic","scores":{"critic_penalty":X.XX},"flags":{"too_hard":bool,"already_mastered":bool},"decision":"accept|hold|reject","short_reason":"..."} . critic_penalty 0-1.',
    "explorer": '{"task_id":"...","role":"explorer","scores":{"novelty_score":X.XX,"diversity_score":X.XX},"decision":"accept|hold|reject","short_reason":"..."} . Scores 0-10.',
}
REQ = {"tutor": ["progression_score", "learnability_score"], "critic": ["critic_penalty"],
       "explorer": ["novelty_score", "diversity_score"]}
DEC = {"accept", "hold", "reject"}
MTOK = {"tutor": 6000, "critic": 9000, "explorer": 6000}  # IDENTICAL across arms per role

def build_prompt(arm, role):
    student = "Student information (deterministic, computed by code):\n" + RAW_SUMMARY
    if arm == "C":
        student += "\n\nModeler StudentProfile (frozen interpretation of the above evidence):\n" + PROFILE_JSON
    return (ROLE_INSTR[role] + "\n\n" + student +
            "\n\nCandidates to evaluate (ALL 32, anonymized IDs C001..C032):\n" + CANDS_TXT +
            "\n\nEvaluate ALL 32 candidates. Return ONLY a JSON array of 32 objects, one per anonymized task_id "
            '(e.g. "C007"), no prose, no markdown. Keep each short_reason under 12 words. Each object: ' + SCHEMA[role])

def outpath(arm, role): return os.path.join(OUTDIR, "bc_%s_%s_judgments.jsonl" % (arm, role))

def existing_valid(arm, role):
    p = outpath(arm, role)
    if not os.path.exists(p): return False
    try: rows = [json.loads(l) for l in open(p) if l.strip()]
    except Exception: return False
    ids = set()
    for e in rows:
        j = e.get("judgment", {}); scc = j.get("scores", {})
        aid = j.get("task_id")
        if aid not in anon_to_real: return False
        if not all(isinstance(scc.get(k), (int, float)) for k in REQ[role]): return False
        if j.get("decision") not in DEC: return False
        ids.add(aid)
    return len(rows) == 32 and ids == set(anon_to_real.keys())

cost = {}
if os.path.exists(COST):
    try: cost = json.load(open(COST))
    except Exception: cost = {}

all_ok = True
for arm in ["B", "C"]:
    for role in ["tutor", "critic", "explorer"]:
        if existing_valid(arm, role):
            print("ARM %s ROLE %s SKIP (valid 32 present)" % (arm, role)); continue
        prov = {"tutor": "qw", "critic": "ds", "explorer": "gl"}[role]
        model = L.ROLE_MODEL_MAP[role]
        parsed, meta = L.call_json(prov, model, build_prompt(arm, role), mtok=MTOK[role], retries=3)
        cost["%s_%s" % (arm, role)] = {"provider": prov, "model_rq": model, "model_rt": meta.get("model_rt"),
                                        "attempts": meta.get("attempts"), "itok": meta.get("itok"),
                                        "otok": meta.get("otok"), "err": meta.get("err")}
        errs = []
        if not isinstance(parsed, list):
            errs.append("not a JSON array (got %s): %s" % (type(parsed).__name__, meta.get("err")))
        else:
            got = set()
            for j in parsed:
                aid = j.get("task_id")
                if aid not in anon_to_real: errs.append("bad/unmapped task_id %r" % aid); continue
                got.add(aid)
                if j.get("role") != role: errs.append("%s role=%r" % (aid, j.get("role")))
                scc = j.get("scores", {})
                for k in REQ[role]:
                    if not isinstance(scc.get(k), (int, float)):
                        errs.append("%s missing/non-numeric %s=%r" % (aid, k, scc.get(k)))
                if j.get("decision") not in DEC: errs.append("%s bad decision=%r" % (aid, j.get("decision")))
            miss = set(anon_to_real.keys()) - got
            if miss: errs.append("MISSING %d: %s" % (len(miss), sorted(miss)))
            if len(parsed) != 32: errs.append("expected 32 got %d" % len(parsed))
        if errs:
            all_ok = False
            print("ARM %s ROLE %s VALIDATION_FAILED:" % (arm, role))
            for e in errs[:40]: print("   - " + e)
            print("   META:", json.dumps(meta, ensure_ascii=False)[:300])
            json.dump({"arm": arm, "role": role, "raw": parsed, "meta": meta},
                      open(os.path.join(OUTDIR, "bc_%s_%s_FAILED_raw.json" % (arm, role)), "w"), indent=2, ensure_ascii=False)
            continue
        with open(outpath(arm, role), "w") as f:
            for j in parsed:
                entry = {"anon_id": j["task_id"], "task_id": anon_to_real[j["task_id"]], "role": role, "arm": arm,
                         "judgment": j, "provider": prov, "model_rq": model, "model_rt": meta.get("model_rt")}
                f.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")
        print("ARM %s ROLE %s OK: 32 judgments, attempts=%s otok=%s" % (arm, role, meta.get("attempts"), meta.get("otok")))
        json.dump(cost, open(COST, "w"), indent=2, ensure_ascii=False)  # persist incrementally

json.dump(cost, open(COST, "w"), indent=2, ensure_ascii=False)
if not all_ok: sys.exit(2)
print("ALL_BC_OK")
