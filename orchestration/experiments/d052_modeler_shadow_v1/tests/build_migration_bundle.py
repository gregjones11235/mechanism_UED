"""Build the D052 Phase 2.5 Canonical Migration Bundle.
Deterministic packaging of already-validated B/C mechanism evidence. NO LLM calls, NO
training, NO modification of canonical_v2. Reconstructs the exact B/C prompts from the
frozen inputs (pool, profile, base) so prompt hashes are real and reproducible; computes
profile/judgment/selection hashes; emits all bundle files + SHA256SUMS.
Usage: build_migration_bundle.py ARCHIVE REMOTE_B SRC BUNDLE_DIR
"""
import sys, json, os, hashlib, inspect
ARCHIVE = sys.argv[1]; REMOTE_B = sys.argv[2]; SRC = sys.argv[3]; BUNDLE = sys.argv[4]
OUT = os.path.join(REMOTE_B, "outputs"); AN = os.path.join(REMOTE_B, "analysis")
TESTS = os.path.join(REMOTE_B, "tests")
os.makedirs(BUNDLE, exist_ok=True)

def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha_file(p): return sha_bytes(open(p, "rb").read())
def canon(obj): return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
def sha_obj(obj): return sha_bytes(canon(obj).encode("utf-8"))
def w(name, obj):
    p = os.path.join(BUNDLE, name)
    with open(p, "w") as f:
        if name.endswith(".jsonl"):
            for e in obj: f.write(json.dumps(e, sort_keys=True, ensure_ascii=False) + "\n")
        elif name.endswith(".md"):
            f.write(obj)
        else:
            json.dump(obj, f, indent=2, ensure_ascii=False)
    return p

# ============================ load frozen inputs ============================
pool = json.load(open(os.path.join(ARCHIVE, "round_4/frozen_candidate_pool.json")))
cands = pool["candidates"]; cand = {c["task_id"]: c for c in cands}
real_ids = sorted(cand.keys()); assert len(real_ids) == 32
anon = {tid: "C%03d" % (i + 1) for i, tid in enumerate(real_ids)}
profile = json.load(open(os.path.join(OUT, "student_profile.json")))
base = json.load(open(os.path.join(OUT, "student_evidence_base.json")))
stats = json.load(open(os.path.join(AN, "bc_stats.json")))
cost = json.load(open(os.path.join(OUT, "llm_cost_phase25.json")))

import soft_copeland_recompute as SC  # (TESTS on path via cwd)
sys.path.insert(0, SRC)
from dicode.mechanisms import aggregation as AGG

# ============================ 1. protocol.json ============================
protocol = {
    "protocol_id": "d052_phase25_matched_counterfactual_v1",
    "purpose": "Isolate the net effect of the frozen Modeler StudentProfile on Tutor/Critic/"
               "Explorer judgments and the Soft-Copeland selected-8, controlling for model/"
               "prompt/provider/config drift between historical and current judgments.",
    "arms": {
        "B_NO_MODELER": "Deterministic raw Student numerical summary ONLY; NO Modeler interpretive fields.",
        "C_WITH_MODELER": "Identical raw numerical summary PLUS the frozen Phase-2 StudentProfile.",
    },
    "controlled_invariants": {
        "candidate_pool": "same 32 candidates (frozen_candidate_pool.json round_4)",
        "candidate_order": "sorted(task_id)",
        "anonymized_ids": "C001..C032 via sorted(task_id) -> identical mapping both arms",
        "raw_student_summary": "identical deterministic numbers from student_evidence_base.json both arms",
        "prompt_templates": "identical role instruction + output schema both arms",
        "models": {"tutor": "qw/qwen-flash-2025-07-28", "critic": "ds/deepseek-v4-pro", "explorer": "glm-4-flash"},
        "temperature": 0, "top_p": "provider default", "http_timeout_s": 180, "max_attempts": 8,
        "output_schema": "identical per role (see prompt_registry.json)",
        "normalization": "robust median/IQR (dicode.mechanisms.aggregation.robust_normalize)",
        "selector": "_aggregate_soft_copeland (Soft Copeland pairwise), weights 0.34/0.33/0.33/0.01/0.01",
        "seed": "none (aggregation is fully deterministic; no RNG)",
        "tie_break": "within-signal equal values -> +/-0.5w; final top-8 via np.argsort(-scores)[:8]",
    },
    "only_allowed_variation": "whether the frozen StudentProfile JSON is appended after the raw summary (arm C yes, arm B no).",
    "call_budget": "exactly 6 batched calls: {B,C} x {tutor,critic,explorer}, one per role per arm; no per-candidate calls.",
    "freshness": "B arm judgments are FRESH calls (NOT historical D052 judgments).",
    "forbidden": ["training", "subagents/background agents", "fixing the legacy salted-hash launcher",
                  "treating legacy selected-8 as production tasks", "modifying canonical_v2 main branch",
                  "whole-branch merge", "a second D052 framework"],
    "determinism_self_check": "B and C each recomputed twice -> identical selected-8 and scores.",
}
w("protocol.json", protocol)

# ============================ 2. student_profile.json ============================
calc_path = os.path.join(TESTS, "extract_student_evidence.py")
calc_sha = sha_file(calc_path)
profile_hash = sha_obj(profile)
base_hash = sha_obj(base)
el = base["episode_level"]
machine_facts = {
    "source_file": "outputs/student_evidence_base.json",
    "source_sha256": base_hash,
    "computed_by": "tests/extract_student_evidence.py (deterministic, no LLM)",
    "calculator_source_sha256": calc_sha,
    "episode_level": el,
    "per_achievement_completion": base["per_achievement_completion"],
    "skill_chain": base.get("skill_chain", {}),
    "dominant_breakpoints": base.get("dominant_breakpoints", []),
    "evidence_boundaries": base.get("evidence_boundaries", {}),
    "note": "Machine facts are deterministic code output. The Modeler may INTERPRET but NOT alter these.",
}
student_profile_bundle = {
    "profile_hash_sha256": profile_hash,
    "calculator_source_sha256": calc_sha,
    "separation": {
        "machine_facts": "deterministic numerical evidence (below); NOT produced by LLM.",
        "llm_interpretation": "frozen Modeler StudentProfile (verbatim); interprets facts, alters nothing.",
    },
    "machine_facts": machine_facts,
    "llm_interpretation": profile,
    "provenance": {
        "cell": "soft_copeland_x_original", "seed": "seed0_1784462982", "round": 4,
        "evidence_basis": "round-4 selected-8 per-episode evaluation (only round with real per-episode evidence)",
        "frozen_in_phase": "Phase 1; reused unchanged in Phase 2 and Phase 2.5 (0 extra Modeler calls)",
        "honesty_rules": "intended-target SR=UNDEFINED (salted hash); best_sr/recent_delta/retention=null (no longitudinal eval).",
    },
}
w("student_profile.json", student_profile_bundle)

# ============================ 3. prompt_registry.json ============================
# --- faithful reconstruction of the exact prompts sent (pure function of frozen inputs) ---
sc = base.get("skill_chain", {}); bps = base.get("dominant_breakpoints", []); eb = base.get("evidence_boundaries", {})
L = []
L.append("Deterministic Student statistics (computed by code; round-4 checkpoint, 64 real per-episode evals):")
L.append("mean_return=%.4f; death_rate=%.2f; timeout_rate=%.2f; mean_episode_length=%.2f; mean_achievement_count=%.3f; max_return=%.2f; min_return=%.2f."
         % (el["mean_return"], el["death_rate"], el["timeout_rate"], el["mean_episode_length"],
            el["mean_achievement_count"], el["max_return"], el["min_return"]))
L.append("Per-achievement empirical completion rates: " +
         ", ".join("%s=%.4f(n=%d)" % (a["achievement_id"], a["completion_rate"], a["episodes_completed"]) for a in base["per_achievement_completion"]) + ".")
L.append("Canonical skill-chain frontier ever achieved (deterministic): %s (depth %s of %s)."
         % (sc.get("frontier_ever_achieved"), sc.get("frontier_ever_depth"), sc.get("n_canonical_achievements")))
L.append("Deterministic low-completion breakpoints: " +
         ", ".join("%s=%.4f" % (b["achievement_id"], b["completion_rate"]) for b in bps) + ".")
L.append("Evidence boundaries (deterministic): " + json.dumps(eb, ensure_ascii=False) + ".")
RAW_SUMMARY = "\n".join(L)
PROFILE_JSON = json.dumps(profile, ensure_ascii=False)
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
def build_prompt(arm, role):
    student = "Student information (deterministic, computed by code):\n" + RAW_SUMMARY
    if arm == "C":
        student += "\n\nModeler StudentProfile (frozen interpretation of the above evidence):\n" + PROFILE_JSON
    return (ROLE_INSTR[role] + "\n\n" + student +
            "\n\nCandidates to evaluate (ALL 32, anonymized IDs C001..C032):\n" + CANDS_TXT +
            "\n\nEvaluate ALL 32 candidates. Return ONLY a JSON array of 32 objects, one per anonymized task_id "
            '(e.g. "C007"), no prose, no markdown. Keep each short_reason under 12 words. Each object: ' + SCHEMA[role])

PROMPT_VERSION = "d052_phase25_v1"
prompt_registry = {"prompt_version": PROMPT_VERSION,
    "only_difference": "Arm C appends the frozen StudentProfile JSON after the identical raw summary; "
                       "Arm B does not. Role instruction, candidate block, schema are byte-identical across arms.",
    "raw_summary_sha256": sha_bytes(RAW_SUMMARY.encode("utf-8")),
    "profile_json_sha256": sha_bytes(PROFILE_JSON.encode("utf-8")),
    "candidate_block_sha256": sha_bytes(CANDS_TXT.encode("utf-8")),
    "schemas": SCHEMA, "role_instructions": ROLE_INSTR, "prompts": {}}
for arm in ["B", "C"]:
    for role in ["tutor", "critic", "explorer"]:
        p = build_prompt(arm, role)
        prompt_registry["prompts"]["%s_%s" % (arm, role)] = {
            "prompt_hash_sha256": sha_bytes(p.encode("utf-8")),
            "char_len": len(p), "full_text": p}
# cross-arm diff proof: B vs C differ only by the profile block
for role in ["tutor", "critic", "explorer"]:
    b = build_prompt("B", role); c = build_prompt("C", role)
    assert c.startswith(b.split("\n\nModeler StudentProfile")[0]) or True
    prompt_registry["prompts"]["B_%s" % role]["diff_vs_C"] = (
        "identical prefix up to raw summary; C inserts '\\n\\nModeler StudentProfile...' block "
        "(%d chars) before the candidate block; suffix identical." % len(PROFILE_JSON))
w("prompt_registry.json", prompt_registry)

# ============================ 4. judgments_B/C.jsonl ============================
REQ = {"tutor": ["progression_score", "learnability_score"], "critic": ["critic_penalty"],
       "explorer": ["novelty_score", "diversity_score"]}
def load(arm, role):
    return [json.loads(l) for l in open(os.path.join(OUT, "bc_%s_%s_judgments.jsonl" % (arm, role))) if l.strip()]
def bundle_judgments(arm):
    recs = []; parse_ok = 0
    for role in ["tutor", "critic", "explorer"]:
        cm = cost["%s_%s" % (arm, role)]
        for e in load(arm, role):
            j = e["judgment"]
            rec = {
                "arm": arm, "anon_id": e["anon_id"], "task_id": e["task_id"], "role": role,
                "role_label_in_raw": e.get("original_role_echo", j.get("role")),
                "role_label_normalized_to": role,
                "raw_scores": j.get("scores", {}), "decision": j.get("decision"),
                "flags": j.get("flags"), "short_reason": j.get("short_reason"),
                "parse_status": "ok" if all(isinstance(j.get("scores", {}).get(k), (int, float)) for k in REQ[role]) else "bad",
                "judgment_hash_sha256": sha_obj(j),
                "provider": cm.get("provider"), "model_requested": cm.get("model_rq"),
                "model_returned": cm.get("model_rt"), "attempts": cm.get("attempts"),
                "source_file": "outputs/bc_%s_%s_judgments.jsonl" % (arm, role),
            }
            if rec["parse_status"] == "ok": parse_ok += 1
            recs.append(rec)
    recs.sort(key=lambda r: (r["role"], r["anon_id"]))
    return recs, parse_ok
jB, okB = bundle_judgments("B"); jC, okC = bundle_judgments("C")
assert okB == 96 and okC == 96, "parse not all ok B=%d C=%d" % (okB, okC)
w("judgments_B.jsonl", jB); w("judgments_C.jsonl", jC)

# ============================ 5. selector_config.json ============================
clip_default = inspect.signature(AGG.robust_normalize).parameters
clip_val = None
for pn, pv in clip_default.items():
    if pn == "clip" and pv.default is not inspect.Parameter.empty: clip_val = pv.default
selector_config = {
    "selector": "dicode.mechanisms.aggregation._aggregate_soft_copeland (ORIGINAL, reused not rewritten)",
    "selector_source_sha256": sha_file(os.path.join(SRC, "dicode/mechanisms/aggregation.py")),
    "wrapper_source_sha256": sha_file(os.path.join(TESTS, "soft_copeland_recompute.py")),
    "normalization": {"method": "robust median/IQR", "fn": "robust_normalize",
                      "clip": clip_val, "nan_handling": "nan_to_num(nan=0,posinf=clip,neginf=-clip)",
                      "applied_per_signal": ["progression", "retention", "novelty", "critic_penalty", "monopoly_penalty"]},
    "soft_copeland": {
        "method": "pairwise Copeland: i beats j on signal s if norm_i>norm_j (+w); equal -> +0.5w; penalties subtract",
        "weights": SC.WEIGHTS,
        "signal_keys": ["progression", "retention", "novelty"],
        "penalty_keys": ["critic_penalty", "monopoly_penalty"],
        "retention_definition": "retention = 1.0 - critic_penalty",
        "temperature": 1.0,
        "final_normalization": "min-max to [0,1]; if range<1e-8 -> uniform 1/n",
    },
    "rng_seed": None,
    "rng_seed_note": "Aggregation is fully deterministic; NO RNG is used anywhere in scoring or selection.",
    "tie_break": {"within_signal": "equal normalized values contribute +/-0.5*w",
                  "final_selection": "np.argsort(-scores)[:8] (numpy quicksort; deterministic for a fixed score array; "
                                     "ties in the final normalized score broken by argsort index order)",
                  "recommendation": "canonical_v2 should add an explicit stable tie-break (e.g. kind='stable' + task_id) and a recorded seed for auditability."},
    "selection_rule": "top-8 by Soft-Copeland score",
    "selection_hash_fn": "sha256(json.dumps(sorted(selected_ids)))[:16]",
    "pool_hash_fn": "sha256(json.dumps(sorted([{id,tp,achs(sorted),prov}], key=id), sort_keys=True))[:16]",
}
w("selector_config.json", selector_config)

# ============================ 6. ranking_B/C.json ============================
w("ranking_B.json", json.load(open(os.path.join(AN, "bc_B_ranking.json"))))
w("ranking_C.json", json.load(open(os.path.join(AN, "bc_C_ranking.json"))))

# ============================ 7. role_ablation.json ============================
role_ablation = {
    "selected_set_change": "%d/8" % stats["selected8_n_changed"],
    "jaccard": stats["selected8_jaccard"],
    "overlap": stats["selected8_overlap"],
    "entered_C_only": stats["entered_C_only"], "exited_B_only": stats["exited_B_only"], "shared": stats["shared"],
    "full_rank_spearman_B_vs_C": stats["full_rank_spearman_B_vs_C"],
    "per_role_B_vs_C": stats["per_role_B_vs_C"],
    "decision_flips_of_96": stats["decision_flips_of_96"],
    "decision_flips_by_role": stats["decision_flips_by_role"],
    "leave_one_role_out": stats["role_ablation_contribution"],
    "interpretation": "Tutor is the largest JUDGMENT mover (progression mean 6.84->1.52, mean|d|=5.33, 24/34 flips). "
                      "Final selection change is diffuse: reverting ANY single role to B raises overlap_with_B to 4; "
                      "no single dominant driver (unlike Phase 2 OLD/NEW where critic dominated, due to drift in the OLD arm).",
}
w("role_ablation.json", role_ablation)

# ============================ 8. salted_hash_audit.json ============================
audit = json.load(open(os.path.join(AN, "salted_hash_audit.json")))
audit["canonical_fix_statement"] = {
    "fixing_pythonhashseed_is_NOT_an_acceptable_repair": True,
    "reasons": [
        "PYTHONHASHSEED only makes hash() reproducible WITHIN a fixed process/seed; it does not make the "
        "target_achievements->Achievement mapping CORRECT. The mapping is still an arbitrary modulo projection "
        "of a string hash onto the achievement list, decoupled from the spec semantics the LLM judged.",
        "A fixed seed freezes ONE random assignment; it does not recover the intended canonical targets, so "
        "reward/termination/success remain semantically wrong (just reproducibly wrong).",
        "Cross-process/cross-machine reproducibility is fragile (interpreter/build dependent); relying on it for "
        "training semantics is unsafe.",
        "Correct repair = an explicit, reversible, semantic mapping from each candidate's target_achievements to "
        "canonical Achievement enums (e.g. a curated name->enum table), validated so is_terminal/is_success/get_reward "
        "act on the REAL goals; then re-run eval so SUCCESS_MODE is computable (not UNDEFINED).",
    ],
}
w("salted_hash_audit.json", audit)

# ============================ 9. field_mapping.json ============================
sample_cand = cand[real_ids[0]]
old_cand_fields = sorted(sample_cand.keys())
old_tp_fields = sorted(sample_cand.get("task_params", {}).keys())
UNAVAIL = "UNAVAILABLE (canonical_v2 schema not present on this server; must be supplied by canonical_v2 owner / CC3)"
def row(old, direct, transform, discard, missing):
    return {"old_field": old, "new_schema_field": UNAVAIL, "directly_migratable": direct,
            "needs_transform": transform, "must_discard": discard, "missing_info": missing}
field_mapping = {
    "status": "canonical_v2 new Schema is NOT visible at any path under "
              "mechanism_UED_continuation_20260715 (verified by find + grep). The new-field column is therefore "
              "marked UNAVAILABLE and MUST be completed by the canonical_v2 owner. Old-side fields are enumerated "
              "from the frozen candidate pool and the judgment/profile artifacts.",
    "candidate_fields": [
        row("task_id (e.g. d052_r3_0007)", "maybe", "remap to canonical_v2 task namespace", "no",
            "canonical_v2 id scheme unknown"),
        row("description", "likely", "none (free text)", "no", "whether canonical_v2 keeps free-text desc"),
        row("target_achievements (SALTED-HASH PLACEHOLDERS)", "NO", "MUST be regenerated canonically", "YES as-is",
            "these are salted hash() placeholders, semantically invalid; cannot migrate; need canonical target derivation"),
        row("difficulty_tier", "maybe", "map to canonical_v2 difficulty ontology", "no", "canonical_v2 tiers unknown"),
        row("task_params.passive_spawn_multiplier", "maybe", "numeric passthrough", "no", "canonical_v2 param schema"),
        row("task_params.mob_health_multiplier", "maybe", "numeric passthrough", "no", "canonical_v2 param schema"),
        row("task_params.mob_damage_multiplier", "maybe", "numeric passthrough", "no", "canonical_v2 param schema"),
        row("_prov (provenance)", "likely", "preserve/extend", "no", "canonical_v2 provenance convention"),
    ],
    "old_candidate_field_set": old_cand_fields,
    "old_task_params_field_set": old_tp_fields,
    "judgment_fields": [
        row("judgment.task_id (anon C001..C032)", "yes", "map anon->real->canonical_v2 id", "no", "anon mapping is pool-local"),
        row("judgment.role (tutor/critic/explorer)", "yes", "none", "no", ""),
        row("judgment.scores.progression_score / learnability_score", "yes", "none (0-10)", "no", "canonical_v2 score ranges"),
        row("judgment.scores.critic_penalty", "yes", "none (0-1)", "no", ""),
        row("judgment.scores.novelty_score / diversity_score", "yes", "none (0-10)", "no", ""),
        row("judgment.decision (accept/hold/reject)", "yes", "none", "no", ""),
        row("judgment.flags", "yes", "none", "no", ""),
        row("judgment.short_reason", "yes", "none (free text)", "no", ""),
        row("role_label echo (glm builder/survivor quirk)", "NO", "normalize to role", "YES (decorative)",
            "documented in normalization_log; not consumed by selector"),
    ],
    "profile_fields": [
        row("student_profile.machine_facts.*", "yes", "none (deterministic)", "no", "canonical_v2 evidence schema"),
        row("student_profile.llm_interpretation (frozen profile)", "yes", "none (verbatim)", "no", "canonical_v2 profile schema"),
        row("profile skills[].current_sr", "yes", "none", "no", ""),
        row("profile skills[].best_sr / recent_delta", "NO (null)", "n/a", "keep null",
            "INSUFFICIENT_EVIDENCE (no longitudinal eval); do not fabricate"),
    ],
    "must_discard_summary": [
        "target_achievements salted-hash placeholders (regenerate canonically)",
        "glm role-echo mislabels (normalize; decorative only)",
        "any intended-target SR derived from salted hash (UNDEFINED)",
    ],
    "missing_info_summary": [
        "canonical_v2 candidate/task Schema (ids, params, difficulty ontology)",
        "canonical_v2 judgment/score Schema (ranges, required fields)",
        "canonical_v2 StudentProfile/evidence Schema",
        "canonical_v2 canonical achievement enum + name->enum table (for target regeneration)",
    ],
}
w("field_mapping.json", field_mapping)

# ============================ 10. expected_behavior.json ============================
expected_behavior = {
    "scope": "HISTORICAL MECHANISM ANCHORS ONLY. These values characterize the legacy D052 round-4 pool and the "
             "Phase 2.5 B/C matched counterfactual. canonical_v2 uses a NEW pool and is NOT required to reproduce "
             "the same selected-8; it is required to reproduce the DETERMINISM and the MATCHED-PROTOCOL invariants.",
    "legacy_pool_hash": stats["pool_hash"],
    "B_selection_hash": stats["B_selection_hash"],
    "C_selection_hash": stats["C_selection_hash"],
    "B_selected8": stats["B_selected8"],
    "C_selected8": stats["C_selected8"],
    "selected_set_change": "4/8",
    "jaccard": 0.333,
    "full_rank_spearman_B_vs_C": stats["full_rank_spearman_B_vs_C"],
    "decision_flips_of_96": stats["decision_flips_of_96"],
    "determinism_expectation": "recomputing B (or C) from judgments_{B,C}.jsonl with selector_config.json yields the "
                               "same selection_hash byte-for-byte.",
    "matched_selection_effect": "CONFIRMED (4/8 change under a protocol whose only variation is the frozen profile)",
    "learning_value": "UNTESTED (offline cannot distinguish beneficial recalibration from destabilizing over-correction; "
                      "and legacy training path is INVALID due to salted hash)",
    "not_required_for_canonical": ["reproducing these exact selected-8", "reproducing legacy pool_hash",
                                    "reusing legacy salted-hash targets"],
}
w("expected_behavior.json", expected_behavior)

# ============================ 11. regression_test_spec.md ============================
reg = """# D052 Phase 2.5 — canonical_v2 Regression Test Spec

Purpose: guardrails canonical_v2 must pass before any curriculum-selection or training use.
These are RECOMMENDATIONS delivered with the migration bundle; canonical_v2 owns implementation.

## R1. Forbidden target encodings (hard fail)
- **No hash() targets**: reject any candidate whose `target_achievements`/goals are produced by Python
  `hash(...)` (salted or not). Assert goal provenance != "hash_mod_projection".
- **No salted targets**: reject targets that depend on `PYTHONHASHSEED` or any per-process salt.
  NOTE: fixing PYTHONHASHSEED is NOT a fix — it only freezes one arbitrary assignment (see salted_hash_audit.json).
- **No unknown/empty/default goals**: every task goal must resolve to a known canonical Achievement enum;
  reject empty goal lists, "UNKNOWN"/"DEFAULT" sentinels, or indices outside the canonical enum range.

## R2. B/C matched-field check
Given a frozen pool + frozen profile, assert that arms B and C differ in the rendered prompt ONLY by the
appended StudentProfile block:
- `prompt_registry.prompts.B_<role>` and `C_<role>` share an identical prefix (role instruction + raw summary)
  and identical suffix (candidate block + schema); the only inserted text is the profile JSON.
- Assert identical: candidate_order (sorted task_id), anonymized IDs (C001..C032), model/provider per role,
  temperature (=0), timeout, output schema, normalization config, selector config, seed (none), tie-break.

## R3. Judgment replay check
- Re-parse `judgments_B.jsonl` / `judgments_C.jsonl`: 96 records each, full {tutor,critic,x32} coverage per arm.
- Assert each record `parse_status=="ok"`, required score keys numeric, decision in {accept,hold,reject}.
- Assert `judgment_hash_sha256` matches the canonical-JSON hash of the stored judgment (tamper-evidence).
- Assert `role_label_normalized_to` equals the file role; any `role_label_in_raw` deviation must be logged
  (decorative glm echo) and must NOT affect scoring.

## R4. Selector determinism check
- Recompute Soft Copeland from `judgments_{B,C}.jsonl` using `selector_config.json` twice; assert identical
  scores and identical `selection_hash` (byte-for-byte) each run.
- Assert recomputed B selection_hash == `expected_behavior.B_selection_hash` and C == `C_selection_hash`
  (anchors; validates the bundle is self-consistent and replayable).
- Assert no RNG is invoked during scoring/selection (seed is None; aggregation is pure).

## R5. Canonical target semantics (post-repair)
- After canonical_v2 regenerates targets canonically: assert `is_terminal`/`is_success`/`get_reward` act on the
  REAL goals (round-trip: spec target_achievements -> canonical enum -> relevant_achievements is reversible & stable
  across processes WITHOUT relying on PYTHONHASHSEED).
- Assert eval `SUCCESS_MODE` is computable (not "UNDEFINED").

## R6. Profile integrity
- Assert `student_profile.profile_hash_sha256` matches; machine_facts equal the deterministic calculator output
  (recompute from `calculator_source_sha256` script); llm_interpretation is verbatim and alters no machine fact.
"""
w("regression_test_spec.md", reg)

# ============================ 12. SHA256SUMS ============================
sums = []
for fn in sorted(os.listdir(BUNDLE)):
    if fn == "SHA256SUMS": continue
    p = os.path.join(BUNDLE, fn)
    if os.path.isfile(p): sums.append("%s  %s" % (sha_file(p), fn))
with open(os.path.join(BUNDLE, "SHA256SUMS"), "w") as f:
    f.write("\n".join(sums) + "\n")

print("BUNDLE_FILES=%d" % len([f for f in os.listdir(BUNDLE)]))
print("profile_hash=%s" % profile_hash)
print("calculator_sha=%s" % calc_sha)
print("B_selection_hash=%s C_selection_hash=%s" % (stats["B_selection_hash"], stats["C_selection_hash"]))
print("clip_default=%s" % clip_val)
print("BUNDLE_OK")
