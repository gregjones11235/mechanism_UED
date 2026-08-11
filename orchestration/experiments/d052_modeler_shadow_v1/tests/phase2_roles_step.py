"""Phase 2 roles step: Tutor / Critic / Explorer each make ONE batched call over the
FULL 32 candidate pool, conditioned on the SAME frozen Modeler StudentProfile (reused
from Phase 1 -- the profile is pool-independent, built from round-4 Student eval evidence).
Output schema mirrors the original judgment_cache entries. Hard-fail, no silent fallback.
Higher mtok + concise-reason instruction to avoid truncation over 32 judgments.
"""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_client as L

POOL = sys.argv[1]        # round_4/frozen_candidate_pool.json
PROFILE = sys.argv[2]     # outputs/student_profile.json
BASE = sys.argv[3]        # outputs/student_evidence_base.json
OUTDIR = sys.argv[4]      # outputs/
COST = sys.argv[5]        # outputs/llm_cost_phase2.json

pool = json.load(open(POOL))
cands = pool["candidates"]
cand = {c["task_id"]: c for c in cands}
chosen = sorted(cand.keys())            # deterministic full-32 order
assert len(chosen) == 32, "expected 32 candidates, got %d" % len(chosen)
profile = json.load(open(PROFILE))
base = json.load(open(BASE))

# ---- compact Modeler profile summary (identical to Phase 1) ----
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

def cand_block(tid):
    c = cand[tid]; tp = c.get("task_params", {})
    return ("Task:%s Desc:%s Achs:%s Tier:%s Spawn:%s Health:%s Damage:%s" % (
        tid, c.get("description", "Task " + tid), ",".join(c.get("target_achievements", [])),
        c.get("difficulty_tier", "medium"), tp.get("passive_spawn_multiplier", "?"),
        tp.get("mob_health_multiplier", "?"), tp.get("mob_damage_multiplier", "?")))

CANDS_TXT = "\n".join(cand_block(t) for t in chosen)
COMMON = ("Student profile (Modeler): " + PROFILE_SUMMARY + "\n\nCandidates to evaluate (ALL %d):\n" % len(chosen)
          + CANDS_TXT +
          "\n\nEvaluate ALL %d candidates. Return ONLY a JSON array of %d objects, one per task_id, "
          "no prose, no markdown. Keep each short_reason under 12 words." % (len(chosen), len(chosen)))

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
# deepseek reasoning over 32 items needs large output budget
MTOK = {"tutor": 6000, "critic": 9000, "explorer": 6000}

def outpath(role): return os.path.join(OUTDIR, "full32_%s_judgments.jsonl" % role)

def existing_valid(role):
    p = outpath(role)
    if not os.path.exists(p): return False
    try: rows = [json.loads(l) for l in open(p) if l.strip()]
    except Exception: return False
    ids = set()
    for e in rows:
        j = e.get("judgment", {}); sc = j.get("scores", {})
        if j.get("task_id") not in chosen: return False
        if not all(isinstance(sc.get(k), (int, float)) for k in REQ_SCORES[role]): return False
        if j.get("decision") not in DECISIONS: return False
        ids.add(j["task_id"])
    return ids == set(chosen) and len(rows) == 32

cost = {}
if os.path.exists(COST):
    try: cost = json.load(open(COST))
    except Exception: cost = {}

all_ok = True
for role in ["tutor", "critic", "explorer"]:
    if existing_valid(role):
        print("ROLE %s SKIP (valid 32-judgment output present)" % role); continue
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
        got = set()
        for j in parsed:
            tid = j.get("task_id")
            if tid not in chosen: errs.append("unexpected task_id %r" % tid); continue
            got.add(tid)
            if j.get("role") != role: errs.append("%s role=%r" % (tid, j.get("role")))
            sc = j.get("scores", {})
            for k in REQ_SCORES[role]:
                if not isinstance(sc.get(k), (int, float)):
                    errs.append("%s missing/non-numeric scores.%s=%r" % (tid, k, sc.get(k)))
            if j.get("decision") not in DECISIONS: errs.append("%s bad decision=%r" % (tid, j.get("decision")))
        miss = set(chosen) - got
        if miss: errs.append("MISSING %d judgments: %s" % (len(miss), sorted(miss)))
        if len(parsed) != 32: errs.append("expected 32 got %d (likely truncation)" % len(parsed))
    if errs:
        all_ok = False
        print("ROLE %s VALIDATION_FAILED:" % role)
        for e in errs[:40]: print("   - " + e)
        print("   META:", json.dumps(meta, ensure_ascii=False)[:300])
        json.dump({"role": role, "raw": parsed, "meta": meta},
                  open(os.path.join(OUTDIR, "full32_%s_FAILED_raw.json" % role), "w"), indent=2, ensure_ascii=False)
        continue
    with open(outpath(role), "w") as f:
        for j in parsed:
            entry = {"task_id": j["task_id"], "role": role, "judgment": j,
                     "provider": prov, "model_rq": model, "model_rt": meta.get("model_rt"),
                     "snapshot_hash": "MODELER_SHADOW_V1_FULL32",
                     "cache_key": "shadow_v1_full32_%s_%s_%s" % (role, j["task_id"], model)}
            f.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")
    print("ROLE %s OK: %d judgments, attempts=%s otok=%s" % (role, len(parsed), meta.get("attempts"), meta.get("otok")))

json.dump(cost, open(COST, "w"), indent=2, ensure_ascii=False)
if not all_ok: sys.exit(2)
print("ALL_ROLES_FULL32_OK")
