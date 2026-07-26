#!/usr/bin/env python3
"""D052-V2 Modeler shadow — deterministic Student-evidence extraction.

Computes BASE NUMBERS only (no LLM). The Modeler may interpret but NOT alter
these. Source: round-4 selected-8 per-episode evaluation (the only round with
real per-episode evidence for cell soft_copeland_x_original / seed0_1784462982).

Hard honesty rules baked in:
  - intended-target success is UNDEFINED (enhanced cells map target_achievements
    via salted Python hash(); see d052_eval_pilot.py L139). We expose empirical
    per-achievement COMPLETION rate as the only non-fabricated SR proxy.
  - best_sr / recent_delta / retention require longitudinal eval that does not
    exist -> marked INSUFFICIENT_EVIDENCE.
"""
import json, hashlib, collections, os, sys

EVAL = sys.argv[1]
CANON_ORDER = sys.argv[2]  # json list of canonical achievement names in order
OUT = sys.argv[3]

def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
canon = json.load(open(CANON_ORDER))
depth = {name: i for i, name in enumerate(canon)}

recs = [json.loads(l) for l in open(EVAL) if l.strip()]
n = len(recs)
assert n > 0, "empty eval"

# ---- per-episode aggregates ----
returns = [float(r["return"]) for r in recs]
lens = [int(r["episode_length"]) for r in recs]
deaths = [int(r["death"]) for r in recs]
timeouts = [int(r["timeout"]) for r in recs]
ach_counts = [int(r["achievement_count"]) for r in recs]

# ---- per-achievement empirical completion (across episodes) ----
ach_done = collections.Counter()
for r in recs:
    for a in set(r.get("achievements", [])):  # per episode, count once
        ach_done[a] += 1
per_ach = []
for a in sorted(ach_done, key=lambda x: depth.get(x, 999)):
    per_ach.append({
        "achievement_id": a,
        "canonical_depth": depth.get(a),
        "episodes_completed": ach_done[a],
        "completion_rate": round(ach_done[a] / n, 4),
    })

# ---- per-task aggregates ----
by_task = collections.defaultdict(list)
for r in recs:
    by_task[r["task_id"]].append(r)
per_task = []
for tid in sorted(by_task):
    rs = by_task[tid]
    per_task.append({
        "task_id": tid,
        "target_achievement_raw": rs[0].get("target_achievement"),  # placeholder A/B (salted)
        "n_episodes": len(rs),
        "mean_return": round(sum(float(x["return"]) for x in rs) / len(rs), 5),
        "mean_episode_length": round(sum(int(x["episode_length"]) for x in rs) / len(rs), 2),
        "death_rate": round(sum(int(x["death"]) for x in rs) / len(rs), 4),
        "timeout_rate": round(sum(int(x["timeout"]) for x in rs) / len(rs), 4),
        "mean_achievement_count": round(sum(int(x["achievement_count"]) for x in rs) / len(rs), 3),
        "distinct_achievements": sorted({a for x in rs for a in x.get("achievements", [])}),
    })

# ---- skill-chain frontier (deepest canonical achievement ever achieved) ----
achieved_depths = [(depth.get(a, -1), a) for a in ach_done]
frontier_depth, frontier_ach = max(achieved_depths) if achieved_depths else (-1, None)
# most-deep achievement reached in >=50% of episodes (reliable frontier)
reliable = [(depth.get(a, -1), a) for a, c in ach_done.items() if c / n >= 0.5]
rel_depth, rel_ach = max(reliable) if reliable else (-1, None)

# ---- dominant breakpoints: shallow canonical ach with low completion while
#      shallower ones are high (gaps in the chain) ----
sorted_ach = sorted(per_ach, key=lambda x: x["canonical_depth"])
breakpoints = []
for i, a in enumerate(sorted_ach):
    if a["canonical_depth"] is None: continue
    if a["completion_rate"] <= 0.25:
        # a prerequisite-ish achievement rarely completed
        breakpoints.append({"achievement_id": a["achievement_id"],
                            "canonical_depth": a["canonical_depth"],
                            "completion_rate": a["completion_rate"],
                            "note": "low_completion_relative_to_chain"})

base = {
    "schema_version": "d052_modeler_evidence_base_v1",
    "cell": "soft_copeland_x_original",
    "seed": "seed0_1784462982",
    "round": 4,
    "checkpoint_step": 98304,
    "provenance": {
        "eval_source": EVAL,
        "eval_sha256": sha(EVAL),
        "checkpoint_sha256": recs[0]["checkpoint_sha256"],
        "evaluator_sha256": recs[0]["evaluator_sha256"],
        "n_episodes": n,
        "n_tasks_evaluated": len(by_task),
        "tasks_are": "round_4_selected8_only",
        "architecture_note": "enhanced_selected8_eye8: all-32 eval architecturally UNDEFINED; only selected-8 evaluable",
    },
    "episode_level": {
        "mean_return": round(sum(returns) / n, 5),
        "min_return": round(min(returns), 5),
        "max_return": round(max(returns), 5),
        "mean_episode_length": round(sum(lens) / n, 2),
        "min_episode_length": min(lens),
        "max_episode_length": max(lens),
        "death_rate": round(sum(deaths) / n, 4),
        "timeout_rate": round(timeouts and sum(timeouts) / n or 0.0, 4),
        "mean_achievement_count": round(sum(ach_counts) / n, 3),
        "max_achievement_count": max(ach_counts),
    },
    "per_achievement_completion": per_ach,
    "per_task": per_task,
    "skill_chain": {
        "canonical_order_source": "craftax.craftax.constants.Achievement (enum order)",
        "n_canonical_achievements": len(canon),
        "frontier_ever_achieved": frontier_ach,
        "frontier_ever_depth": frontier_depth,
        "reliable_frontier_ach": rel_ach,
        "reliable_frontier_depth": rel_depth,
    },
    "dominant_breakpoints": breakpoints,
    "evidence_boundaries": {
        "intended_target_success_rate": "UNDEFINED (salted hash mapping unrecoverable; d052_eval_pilot.py L139)",
        "best_sr": "INSUFFICIENT_EVIDENCE (single snapshot; no longitudinal eval)",
        "recent_delta": "INSUFFICIENT_EVIDENCE (rounds 1-3 have no per-episode eval)",
        "retention": "INSUFFICIENT_EVIDENCE (no repeated measurement)",
        "trajectory_status_improving_stalled_forgetting": "INSUFFICIENT_EVIDENCE (single cross-sectional snapshot)",
        "eval_coverage": "round_4 selected-8 tasks only (8 tasks x 8 episodes); 24 non-selected candidates NOT evaluated",
    },
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(base, f, indent=2, sort_keys=True, ensure_ascii=False)
print("WROTE", OUT)
print("n_episodes", n, "mean_return", base["episode_level"]["mean_return"],
      "death_rate", base["episode_level"]["death_rate"],
      "timeout_rate", base["episode_level"]["timeout_rate"])
print("frontier_ever", frontier_ach, frontier_depth, "| reliable", rel_ach, rel_depth)
print("n_ach_observed", len(per_ach), "breakpoints", [b["achievement_id"] for b in breakpoints])
