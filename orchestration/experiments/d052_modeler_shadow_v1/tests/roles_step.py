"""Phase 1 roles step: Tutor / Critic / Explorer each make ONE batched call over the
8 pilot candidates, conditioned on the Modeler StudentProfile (replacing the original
sparse snapshot_str). Output schema mirrors the original judgment_cache entries.
Hard-fail validation, no silent fallback.
"""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_client as L

POOL = sys.argv[1]        # round_4/frozen_candidate_pool.json
PROFILE = sys.argv[2]     # outputs/student_profile.json
BASE = sys.argv[3]        # outputs/student_evidence_base.json
PILOT = sys.argv[4]       # manifests/pilot8_manifest.json
OUTDIR = sys.argv[5]      # outputs/
COST = sys.argv[6]        # outputs/llm_cost.json

pool = json.load(open(POOL))
cand = {c["task_id"]: c for c in pool["candidates"]}
profile = json.load(open(PROFILE))
base = json.load(open(BASE))
pilot = json.load(open(PILOT))
chosen = pilot["chosen"]

# ---- compact Modeler profile summary (faithful to profile + base numbers) ----
el = base["episode_level"]
sk = {s["achievement_id"]: s for s in profile["skills"]}
lines = ["Modeler StudentProfile (round-4 checkpoint, 64 episodes, real per-episode evidence):"]
lines.append("Episode stats: mean_return=%.3f, death_rate=%.2f, timeout_rate=%.2f, mean_episode_length=%.0f, mean_achievement_count=%.2f."
             % (el["mean_return"], el["death_rate"], el["timeout_rate"], el["mean_episode_length"], el["mean_achievement_count"]))
lines.append("Skill chain frontier: %s. Dominant breakpoints: %s."
             % (profile["chain_frontier"], ", ".join(profile["dominant_breakpoints"])))
sr_parts = []
for aid in ["WAKE_UP", "COLLECT_WOOD", "COLLECT_SAPLING", "PLACE_PLANT", "COLLECT_DRINK", "PLACE_TABLE", "MAKE_WOOD_PICKAXE"]:
    s = sk.get(aid)
    if s:
        sr_parts.append("%s=%.3f(%s)" % (aid, s["current_sr"], s["status"]))
lines.append("Empirical achievement completion rates: " + ", ".join(sr_parts) + ".")
lines.append("Curriculum priorities: " + "; ".join(profile["curriculum_priorities"]) + ".")
lines.append("Uncertainties: " + "; ".join(profile["uncertainties"]) + ".")
lines.append("NOTE: only cross-sectional evidence (no trends); intended-target success rate is UNDEFINED.")
PROFILE_SUMMARY = " ".join(lines)

# ---- candidate block (identical info per role) ----
def cand_block(tid):
    c = cand[tid]; tp = c.get("task_params", {})
    return ("Task:%s Desc:%s Achs:%s Tier:%s Spawn:%s Health:%s Damage:%s" % (
        tid, c.get("description", "Task " + tid), ",".join(c.get("target_achievements", [])),
        c.get("difficulty_tier", "medium"), tp.get("passive_spawn_multiplier", "?"),
        tp.get("mob_health_multiplier", "?"), tp.get("mob_damage_multiplier", "?")))

CANDS_TXT = "\n".join(cand_block(t) for t in chosen)
IDS_JSON = json.dumps(chosen)

COMMON = ("Student profile (Modeler): " + PROFILE_SUMMARY + "\n\nCandidates to evaluate:\n" + CANDS_TXT +
          "\n\nEvaluate ALL %d candidates. Return ONLY a JSON array of %d objects, one per task_id, "
          "no prose, no markdown." % (len(chosen), len(chosen)))

PROMPTS = {
    "tutor": ("Evaluate Craftax task progression/teaching value for a curriculum. "
        "For each task judge current learnability, scaffolding soundness, downstream skill-chain value, "
        "and whether it merely repeats an already-mastered ability (WAKE_UP is mastered). " + COMMON +
        ' Each object: {"task_id":"...","role":"tutor","scores":{"progression_score":X.XX,"learnability_score":X.XX},'
        '"decision":"accept|hold|reject","short_reason":"..."} . Scores on 0-10 scale.'),
    "critic": ("Evaluate Craftax task failure risk for this student. "
        "For each task judge whether it is too far ahead, invites shortcuts, gives trivial success, "
        "uses wrong scaffolding, risks negative transfer, or wrongly assumes the student state "
        "(note death_rate=1.00 and low tool/plant/drink completion). " + COMMON +
        ' Each object: {"task_id":"...","role":"critic","scores":{"critic_penalty":X.XX},'
        '"flags":{"too_hard":bool,"already_mastered":bool},"decision":"accept|hold|reject","short_reason":"..."} . '
        'critic_penalty on 0-1 scale (0=safe, 1=high risk).'),
    "explorer": ("Evaluate Craftax task novelty and curriculum diversity for this student. "
        "For each task judge skill coverage, novelty vs redundant candidates, and alternative skill paths, "
        "given the achieved set (WAKE_UP mastered; wood/sapling partial; tools/plant/drink weak). " + COMMON +
        ' Each object: {"task_id":"...","role":"explorer","scores":{"novelty_score":X.XX,"diversity_score":X.XX},'
        '"decision":"accept|hold|reject","short_reason":"..."} . Scores on 0-10 scale.'),
}

REQ_SCORES = {"tutor": ["progression_score", "learnability_score"],
              "critic": ["critic_penalty"],
              "explorer": ["novelty_score", "diversity_score"]}
DECISIONS = {"accept", "hold", "reject"}

cost = {}
if os.path.exists(COST):
    try: cost = json.load(open(COST))
    except Exception: cost = {}

MTOK = {"tutor": 2500, "critic": 4000, "explorer": 2500}  # deepseek reasoning needs more output budget

def existing_valid(role):
    """Idempotency: skip a role whose output jsonl already has 8 valid judgments."""
    p = os.path.join(OUTDIR, role + "_judgments.jsonl")
    if not os.path.exists(p):
        return False
    try:
        rows = [json.loads(l) for l in open(p) if l.strip()]
    except Exception:
        return False
    ids = set()
    for e in rows:
        j = e.get("judgment", {})
        sc = j.get("scores", {})
        if j.get("task_id") not in chosen: return False
        if not all(isinstance(sc.get(k), (int, float)) for k in REQ_SCORES[role]): return False
        if j.get("decision") not in DECISIONS: return False
        ids.add(j["task_id"])
    return ids == set(chosen)

all_ok = True
for role in ["tutor", "critic", "explorer"]:
    if existing_valid(role):
        print("ROLE %s SKIP (valid output already present)" % role)
        continue
    prov = {"tutor": "qw", "critic": "ds", "explorer": "gl"}[role]
    model = L.ROLE_MODEL_MAP[role]
    parsed, meta = L.call_json(prov, model, PROMPTS[role], mtok=MTOK[role], retries=3)
    cost[role] = {"provider": prov, "model_rq": model, "model_rt": meta.get("model_rt"),
                  "attempts": meta.get("attempts"), "itok": meta.get("itok"),
                  "otok": meta.get("otok"), "err": meta.get("err")}
    errs = []
    if not isinstance(parsed, list):
        errs.append("not a JSON array (got %s): %s" % (type(parsed).__name__, meta.get("err")))
    else:
        got_ids = set()
        for j in parsed:
            tid = j.get("task_id")
            if tid not in chosen:
                errs.append("unexpected/missing task_id %r" % tid); continue
            got_ids.add(tid)
            if j.get("role") != role:
                errs.append("%s role field=%r" % (tid, j.get("role")))
            sc = j.get("scores", {})
            for k in REQ_SCORES[role]:
                v = sc.get(k)
                if not isinstance(v, (int, float)):
                    errs.append("%s missing/non-numeric scores.%s=%r" % (tid, k, v))
            if j.get("decision") not in DECISIONS:
                errs.append("%s bad decision=%r" % (tid, j.get("decision")))
        missing = set(chosen) - got_ids
        if missing:
            errs.append("missing judgments for: %s" % sorted(missing))
        if len(parsed) != len(chosen):
            errs.append("expected %d judgments got %d" % (len(chosen), len(parsed)))
    if errs:
        all_ok = False
        print("ROLE %s VALIDATION_FAILED:" % role)
        for e in errs:
            print("   - " + e)
        print("   META:", json.dumps(meta, ensure_ascii=False)[:300])
        # write raw for debugging but mark failure
        json.dump({"role": role, "raw": parsed, "meta": meta},
                  open(os.path.join(OUTDIR, role + "_FAILED_raw.json"), "w"), indent=2, ensure_ascii=False)
        continue
    # write judgments jsonl in original cache-entry shape
    outp = os.path.join(OUTDIR, role + "_judgments.jsonl")
    with open(outp, "w") as f:
        for j in parsed:
            entry = {"task_id": j["task_id"], "role": role, "judgment": j,
                     "provider": prov, "model_rq": model, "model_rt": meta.get("model_rt"),
                     "snapshot_hash": "MODELER_SHADOW_V1",
                     "cache_key": "shadow_v1_%s_%s_%s" % (role, j["task_id"], model)}
            f.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")
    print("ROLE %s OK: %d judgments, attempts=%s otok=%s" % (role, len(parsed), meta.get("attempts"), meta.get("otok")))

json.dump(cost, open(COST, "w"), indent=2, ensure_ascii=False)
if not all_ok:
    sys.exit(2)
print("ALL_ROLES_OK")
