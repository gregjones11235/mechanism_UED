"""Bounded diagnostic: ONE glm-4-flash attempt on the full-32 explorer prompt.
Reveals the exact failure mode (timeout vs truncation vs parse) without writing artifacts.
Run with D052_MAX_ATTEMPTS=1.
"""
import sys, json, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_client as L

POOL = sys.argv[1]; PROFILE = sys.argv[2]; BASE = sys.argv[3]
pool = json.load(open(POOL)); cands = pool["candidates"]
cand = {c["task_id"]: c for c in cands}; chosen = sorted(cand.keys())
profile = json.load(open(PROFILE)); base = json.load(open(BASE))

el = base["episode_level"]
sk = {s["achievement_id"]: s for s in profile["skills"]}
lines = ["Modeler StudentProfile (round-4 checkpoint, 64 episodes, real per-episode evidence):"]
lines.append("Episode stats: mean_return=%.3f, death_rate=%.2f, timeout_rate=%.2f." % (el["mean_return"], el["death_rate"], el["timeout_rate"]))
lines.append("Skill chain frontier: %s. Breakpoints: %s." % (profile["chain_frontier"], ", ".join(profile["dominant_breakpoints"])))
sr = ", ".join("%s=%.3f(%s)" % (a, sk[a]["current_sr"], sk[a]["status"]) for a in
               ["WAKE_UP","COLLECT_WOOD","COLLECT_SAPLING","PLACE_PLANT","COLLECT_DRINK","PLACE_TABLE","MAKE_WOOD_PICKAXE"] if a in sk)
lines.append("Empirical completion rates: " + sr + ".")
lines.append("Priorities: " + "; ".join(profile["curriculum_priorities"]) + ".")
lines.append("Uncertainties: " + "; ".join(profile["uncertainties"]) + ". Target SR UNDEFINED.")
PS = " ".join(lines)

def cb(t):
    c = cand[t]; tp = c.get("task_params", {})
    return "Task:%s Desc:%s Achs:%s Tier:%s Spawn:%s Health:%s Damage:%s" % (
        t, c.get("description","Task "+t), ",".join(c.get("target_achievements",[])),
        c.get("difficulty_tier","medium"), tp.get("passive_spawn_multiplier","?"),
        tp.get("mob_health_multiplier","?"), tp.get("mob_damage_multiplier","?"))
CT = "\n".join(cb(t) for t in chosen)
PROMPT = ("Evaluate Craftax task novelty and curriculum diversity for this student. "
    "For each task judge skill coverage, novelty vs redundant candidates, and alternative skill paths, "
    "given the achieved set (WAKE_UP mastered; wood/sapling partial; tools/plant/drink weak). "
    "Student profile (Modeler): " + PS + "\n\nCandidates to evaluate (ALL %d):\n" % len(chosen) + CT +
    "\n\nEvaluate ALL %d candidates. Return ONLY a JSON array of %d objects, one per task_id, "
    "no prose, no markdown. Keep each short_reason under 12 words." % (len(chosen), len(chosen)) +
    ' Each object: {"task_id":"...","role":"explorer","scores":{"novelty_score":X.XX,"diversity_score":X.XX},'
    '"decision":"accept|hold|reject","short_reason":"..."} . Scores on 0-10 scale.')

print("PROMPT_CHARS=%d  n_candidates=%d" % (len(PROMPT), len(chosen)))
t0 = time.time()
r = L.api("gl", "glm-4-flash", [{"role":"user","content":PROMPT}], mtok=6000)
dt = time.time() - t0
print("WALL=%.1fs  ok=%s  err=%s  itok=%s  otok=%s  mrt=%s" % (dt, r.get("ok"), r.get("err"), r.get("itok"), r.get("otok"), r.get("mrt")))
c = r.get("content") or ""
print("CONTENT_LEN=%d" % len(c))
print("HEAD<<<%s>>>" % c[:300])
print("TAIL<<<%s>>>" % c[-300:])
pj = L.extj(c)
print("EXTJ_TYPE=%s" % type(pj).__name__)
if isinstance(pj, list):
    print("EXTJ_LIST_LEN=%d" % len(pj))
    if pj: print("EXTJ_ELEM0=%s" % json.dumps(pj[0], ensure_ascii=False)[:200])
