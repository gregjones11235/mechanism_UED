"""Phase 1/2 Modeler step: build a frozen StudentProfile from DETERMINISTIC base
evidence (computed by code). The Modeler interprets only; it must not alter numbers.
One LLM call. Hard-fail validation, no silent fallback.
"""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_client as L

BASE = sys.argv[1]            # outputs/student_evidence_base.json
CANON = sys.argv[2]           # manifests/canonical_achievement_order.json
OUT_PROFILE = sys.argv[3]     # outputs/student_profile.json
OUT_COST = sys.argv[4]        # outputs/llm_cost.json

base = json.load(open(BASE))
canon = json.load(open(CANON))
canon_set = set(canon)

# Compact deterministic evidence block handed to the Modeler (numbers are FIXED).
ev = {
    "episode_level": base["episode_level"],
    "per_achievement_completion": base["per_achievement_completion"],
    "skill_chain": base["skill_chain"],
    "dominant_breakpoints": base["dominant_breakpoints"],
    "evidence_boundaries": base["evidence_boundaries"],
    "provenance": base["provenance"],
    "round": base["round"], "cell": base["cell"], "checkpoint_step": base["checkpoint_step"],
}

SCHEMA = '''{
  "schema_version": "d052_modeler_v1",
  "round_id": "round_4",
  "checkpoint_id": "soft_copeland_x_original/seed0_1784462982/checkpoints/98304",
  "data_window": {"rounds_with_per_episode_eval": [4], "n_episodes": 64, "tasks_evaluated": "round_4_selected8 (8 tasks)"},
  "skills": [
    {"achievement_id": "<canonical>", "current_sr": <number|null>, "best_sr": <number|null>,
     "recent_delta": <number|null>, "status": "IMPROVING|STALLED|FORGETTING|NOISY|MASTERED|NORMAL_EARLY|INSUFFICIENT_EVIDENCE",
     "confidence": <0..1>, "evidence_ids": ["<base field path>"], "missing_evidence": ["..."]}
  ],
  "chain_frontier": "<canonical achievement id>",
  "dominant_breakpoints": ["<canonical>", ...],
  "curriculum_priorities": ["<short phrase>", ...],
  "uncertainties": ["<short phrase>", ...]
}'''

PROMPT = """You are the MODELER role in a curriculum-learning shadow experiment (Craftax).
Your ONLY job: build a frozen StudentProfile that EXPLAINS deterministic evidence already
computed by code. You must NOT invent, alter, or recompute any number. You must NOT evaluate
candidates, select tasks, or comment on training weights.

HARD RULES (machine-checked; violation = hard fail):
1. achievement_id MUST be a canonical Craftax achievement name from this list (exact spelling):
%s
2. Every skill conclusion MUST include non-empty evidence_ids referencing a field path in the
   EVIDENCE json below (e.g. "per_achievement_completion[0].completion_rate", "episode_level.death_rate").
3. For any achievement present in EVIDENCE.per_achievement_completion, set current_sr EXACTLY to
   its completion_rate value (do not round differently, do not change it). best_sr and recent_delta
   MUST be null (no longitudinal data) and listed in missing_evidence.
4. There is only a SINGLE cross-sectional snapshot (round-4 checkpoint). Therefore trajectory
   statuses IMPROVING/STALLED/FORGETTING/NOISY are NOT determinable: use MASTERED only if
   completion_rate >= 0.8; NORMAL_EARLY if 0 < completion_rate < 0.8; INSUFFICIENT_EVIDENCE if the
   achievement was never observed or has no evidence. Never claim a trend.
5. Intended-target success rate is UNDEFINED (salted-hash mapping unrecoverable). Never claim a
   target success rate. The only legitimate rate is empirical achievement completion_rate.
6. Output ONLY a single JSON object matching this schema, no prose, no markdown:
%s

EVIDENCE (deterministic, fixed — interpret only):
%s

Produce the StudentProfile JSON now.""" % (
    "[" + ", ".join(canon) + "]",
    SCHEMA,
    json.dumps(ev, ensure_ascii=False, indent=1),
)

profile, meta = L.call_json(L.MODELER_PROVIDER, L.MODELER_MODEL, PROMPT, mtok=3000, retries=3)

# ---- machine validation (hard gates) ----
errors = []
if profile is None:
    errors.append("MODELER_RETURNED_NO_VALID_JSON: %s" % meta.get("err"))
else:
    if profile.get("schema_version") != "d052_modeler_v1":
        errors.append("bad schema_version: %r" % profile.get("schema_version"))
    skills = profile.get("skills", [])
    if not isinstance(skills, list) or not skills:
        errors.append("skills missing/empty")
    base_sr = {a["achievement_id"]: a["completion_rate"] for a in base["per_achievement_completion"]}
    seen = set()
    ALLOWED = {"IMPROVING", "STALLED", "FORGETTING", "NOISY", "MASTERED", "NORMAL_EARLY", "INSUFFICIENT_EVIDENCE"}
    for i, s in enumerate(skills):
        aid = s.get("achievement_id")
        if aid not in canon_set:
            errors.append("skill[%d] non-canonical achievement_id: %r" % (i, aid))
        if not s.get("evidence_ids"):
            errors.append("skill[%d] (%s) missing evidence_ids" % (i, aid))
        st = s.get("status")
        if st not in ALLOWED:
            errors.append("skill[%d] (%s) bad status: %r" % (i, aid, st))
        # anti-fabrication: current_sr must match base completion_rate when observed
        if aid in base_sr:
            csr = s.get("current_sr")
            if csr is None or abs(float(csr) - base_sr[aid]) > 1e-6:
                errors.append("skill[%d] (%s) current_sr %r != base completion_rate %r (fabrication/mismatch)"
                              % (i, aid, csr, base_sr[aid]))
            seen.add(aid)
        else:
            if s.get("current_sr") is not None:
                errors.append("skill[%d] (%s) current_sr must be null (not in evidence)" % (i, aid))
        cf = s.get("confidence")
        if not (isinstance(cf, (int, float)) and 0.0 <= cf <= 1.0):
            errors.append("skill[%d] (%s) confidence out of range: %r" % (i, aid, cf))
    # every observed achievement must be represented
    for aid in base_sr:
        if aid not in seen:
            errors.append("observed achievement %s not represented in skills" % aid)
    if profile.get("chain_frontier") not in canon_set:
        errors.append("chain_frontier non-canonical: %r" % profile.get("chain_frontier"))

# ---- persist cost meta regardless ----
cost = {}
if os.path.exists(OUT_COST):
    try: cost = json.load(open(OUT_COST))
    except Exception: cost = {}
cost["modeler"] = {"provider": meta.get("provider"), "model_rq": meta.get("model_rq"),
                   "model_rt": meta.get("model_rt"), "attempts": meta.get("attempts"),
                   "itok": meta.get("itok"), "otok": meta.get("otok"), "err": meta.get("err")}
json.dump(cost, open(OUT_COST, "w"), indent=2, ensure_ascii=False)

if errors:
    print("MODELER_VALIDATION_FAILED")
    for e in errors:
        print("  - " + e)
    print("META:", json.dumps(meta, ensure_ascii=False)[:400])
    sys.exit(2)

json.dump(profile, open(OUT_PROFILE, "w"), indent=2, ensure_ascii=False)
print("MODELER_OK skills=%d frontier=%s breakpoints=%s" % (
    len(profile.get("skills", [])), profile.get("chain_frontier"), profile.get("dominant_breakpoints")))
print("META attempts=%s itok=%s otok=%s model_rt=%s" % (
    meta.get("attempts"), meta.get("itok"), meta.get("otok"), meta.get("model_rt")))
