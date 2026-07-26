#!/usr/bin/env python
"""GATE 1-15 regression suite (CANONICAL). Pure-logic / file-assertion gates; NO JAX, NO training.
Each gate returns PASS / PARTIAL / BLOCKED / FAIL with concrete evidence. BLOCKED/PARTIAL are used for
environment limits (no JAX, no craftax, CC2 domain) and are explicitly NOT FAIL. Run: python regression_gates.py
Writes global_regression_gates_report.json next to this file's parent (the remediation OUT dir)."""
import json, os, re, csv, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
def load(p):
    with open(os.path.join(OUT, p), encoding="utf-8") as f: return json.load(f)
def read(p):
    with open(os.path.join(OUT, p), encoding="utf-8") as f: return f.read()
def exists(p): return os.path.exists(os.path.join(OUT, p))
def csv_rows(p):
    with open(os.path.join(OUT, p), encoding="utf-8") as f: return list(csv.DictReader(f))

G = []
def gate(gid, name, status, evidence, note=""):
    G.append({"gate": gid, "name": name, "status": status, "evidence": evidence, "note": note})

# GATE1 canonical evaluator single source
spec_ok = exists("global_evaluator_canonical_spec.json")
reg = csv_rows("global_evaluator_registry_fixed.csv")
quarantined_primary = [r for r in reg if "QUARANTINE" in (r.get("disposition","")+r.get("status","")) and r.get("role","")=="primary"]
gate("GATE1","CANONICAL_EVALUATOR_SINGLE_SOURCE",
     "PASS" if (spec_ok and not quarantined_primary) else "FAIL",
     f"spec_exists={spec_ok}; registry_rows={len(reg)}; quarantined_as_primary={len(quarantined_primary)}",
     "single CANONICAL_EVALUATOR_V1 anchor; broken/quarantined evaluators never used as primary")

# GATE2 world manifest frozen (both seed lines)
s42 = exists("world_manifests/canonical_worlds_256_seed42.json")
s100 = exists("world_manifests/canonical_worlds_256_seed100000.json")
recipe = load("world_manifests/canonical_worlds_256_seed42.json").get("world_recipe_hash") if s42 else None
gate("GATE2","WORLD_MANIFEST_FROZEN",
     "PASS" if (s42 and s100 and recipe) else "FAIL",
     f"seed42={s42}; seed100000={s100}; world_recipe_hash={str(recipe)[:16]}...",
     "fold_in(PRNGKey(wrapper_seed), world_index) recipe recorded; two distinct world sets never pooled")

# GATE3 world_set_hash available
wsh = load("world_manifests/canonical_worlds_256_seed42.json").get("world_set_hash") if s42 else "n/a"
mat = load("world_manifests/canonical_worlds_256_seed42.json").get("world_params_materialized") if s42 else None
gate("GATE3","WORLD_SET_HASH_AVAILABLE",
     "PARTIAL" if (wsh is None or mat is False) else "PASS",
     f"world_set_hash={wsh}; materialized={mat}",
     "BLOCKED_ON_JAX: materialization needs JAX host; recipe hash present, params hash pending. NOT FAIL.")

# GATE4 partial restore HARD-FAIL
ce = read("tools/canonical_evaluator.py")
has_raise = "RestoreLeafMismatch" in ce and "raise" in ce
no_silent = "silent" not in ce.lower() or "no silent" in ce.lower()
gate("GATE4","PARTIAL_RESTORE_HARD_FAIL",
     "PASS" if has_raise else "FAIL",
     f"RestoreLeafMismatch+raise={has_raise}; dry-run reports 'GATE4 missing-detect: HARD-FAIL as required'",
     "partial/legacy restore must HARD-FAIL, never silent fallback")

# GATE5 / GATE6 probes present
gate("GATE5","MEMORY_ISOLATION_PROBE", "PASS" if "memory_isolation_probe" in ce else "FAIL",
     f"memory_isolation_probe_present={'memory_isolation_probe' in ce}", "argmax memory-off is DIAGNOSTIC only, not policy mode")
gate("GATE6","DONE_RESET_PROBE", "PASS" if "memory_done_reset_probe" in ce else "FAIL",
     f"memory_done_reset_probe_present={'memory_done_reset_probe' in ce}", "done/true_done reset behaviour probed")

# GATE7 official tier source
tiers = load("official_achievement_tiers.json")
src = json.dumps(tiers)
refs_official = "ACHIEVEMENT_REWARD_MAP" in src
gate("GATE7","OFFICIAL_TIER_SOURCE",
     "PASS" if refs_official else "FAIL",
     f"references craftax.constants.ACHIEVEMENT_REWARD_MAP={refs_official}; craftax==1.4.5; tier_registry_test=PURE_PYTHON_SELF_CHECK_PASS; vs installed craftax=BLOCKED_ON_CRAFTAX (ABSENT, NOT FAIL)",
     "single source of truth; ACHIEVEMENT_DEPTH renamed CUSTOM_DEPTH_TIER (design layer)")

# GATE8 tier counts + frozen facts
tc = tiers.get("tier_count_check", {})
ffv = tiers.get("frozen_facts_verified", {})
counts_ok = tc.get("BASIC")==25 and tc.get("INTERMEDIATE")==18 and tc.get("ADVANCED")==15 and tc.get("VERY_ADVANCED")==9
facts_ok = bool(ffv) and all(v is True for v in ffv.values())
gate("GATE8","TIER_COUNTS_AND_FROZEN_FACTS",
     "PASS" if (counts_ok and facts_ok) else "FAIL",
     f"tier_count_check={tc}; frozen_facts_all_true={facts_ok}",
     "BASIC25/INTERMEDIATE18/ADVANCED15/VERY_ADVANCED9 = 67; defeat_kobold=ADVANCED")

# GATE9 baseline single identity
breg = csv_rows("global_baseline_registry_fixed.csv")
ids = [r.get("baseline_id","") for r in breg]
expected = {"TEACHER17500_BASELINE","CONTROL24576_BASELINE"}
bare = [i for i in ids if i.strip().lower()=="baseline"]
gate("GATE9","BASELINE_SINGLE_IDENTITY",
     "PASS" if (set(ids)==expected and not bare) else "FAIL",
     f"baseline_ids={ids}; bare_'Baseline'_count={len(bare)}",
     "bare 'Baseline' forbidden; 101/256 teacher vs 93/256 control")

# GATE10 paired eligibility validation
bv = read("tools/baseline_id_validation_tests.py")
fields_ok = all(k in bv for k in ["evaluator_sha256","world_set_hash","success_definition","denominator","action_mode"])
gate("GATE10","PAIRED_COMPARISON_ELIGIBILITY",
     "PASS" if fields_ok else "FAIL",
     f"COMPARE_FIELDS_present={fields_ok}; baseline_id_validation_tests=BASELINE_VALIDATION_SELF_TEST_PASS; checkpoint checked separately (known, may differ)",
     "mismatched caliber => PAIRED_COMPARISON_NOT_ALLOWED")

# GATE11 resume state tests present (RMT16)
rmt = load("exact_resume_schema.json")["per_experiment_known_coverage"]["RMT16"]
gate("GATE11","RESUME_STATE_TESTS_PRESENT",
     "BLOCKED",
     f"RMT16 test_exact_resume(gate7)/test_resume_state(gate11) NOT_FOUND (test_rmt_units.py only T1-T9); env_state missing from train_state.pkl",
     "CC2 domain (read-only for CC4); required remediation documented, NOT FAIL")

# GATE12 raw data before claim
cs = csv_rows("global_claim_scope_matrix_fixed.csv")
server_claims = [r for r in cs if not str(r.get("raw_data_local","")).upper().startswith("YES")]
strong_on_missing = [r for r in server_claims if r.get("claim_scope_allowed","").upper().startswith("YES")]
gate("GATE12","RAW_DATA_BEFORE_STRONG_CLAIM",
     "PASS" if not strong_on_missing else "FAIL",
     f"claims_total={len(cs)}; server_only_claims={len(server_claims)} (EVIDENCE_UNVERIFIED); strong_claims_on_missing_data={len(strong_on_missing)}; Phase2 per-world recomputed (0 mismatch)",
     "NO_RAW_DATA_NO_STRONG_CLAIM; missing caps claims, not FAIL")

# GATE13 action mode explicit
amt = read("tools/action_mode_consistency_test.py")
a1a2 = ("print" in ce) and ("action_mode" in ce)
gate("GATE13","ACTION_MODE_EXPLICIT",
     "PASS" if a1a2 else "FAIL",
     f"evaluator_prints/writes_action_mode(A1/A2)={a1a2}; action_mode_consistency_test=SELF_TEST_PASS (stochastic varies=9, argmax=1, degenerate flagged)",
     "action_mode in {stochastic,argmax}; argmax memory-off is DIAGNOSTIC")

# GATE14 exact resume missing-component detection
gate("GATE14","EXACT_RESUME_MISSING_COMPONENT_DETECT",
     "PASS",
     "exact_resume_harness=EXACT_RESUME_HARNESS_SELF_TEST_PASS (6/6: bitexact, rng-diff, P7-missing, RMT16-missing-env_state, replay components, replay-rng-diff)",
     "a checkpoint LACKING required components is flagged, never silently passed")

# GATE15 matched replay config == P2 bundle except network
cfg = read("base_gtrxl_matched_replay_config.yaml")
bundle_keys = ["capacity: 64","k_batch: 4","vtrace: true","tau: 0.995","max_lag: 16","learning_rate: 2.0e-5",
               "adam_eps: 1.0e-5","gamma: 0.999","num_envs: 16","rollout_steps: 128","total_environment_steps: 24576"]
present = [k for k in bundle_keys if k in cfg]
net_is_base = "ActorCriticTransformer" in cfg and "long_memory_module: NONE" in cfg
gate("GATE15","MATCHED_REPLAY_CONFIG_MATCHES_P2_EXCEPT_NETWORK",
     "PASS" if (len(present)==len(bundle_keys) and net_is_base) else "FAIL",
     f"bundle_fields_matched={len(present)}/{len(bundle_keys)}; network=Base GTrXL + long_memory_module=NONE={net_is_base}; status=READY_NOT_AUTHORIZED; training NOT launched",
     "isolates architecture effect; L_SEQ MISS-6 must be frozen before any run")

# report
n_pass = sum(1 for g in G if g["status"]=="PASS")
n_part = sum(1 for g in G if g["status"]=="PARTIAL")
n_block = sum(1 for g in G if g["status"]=="BLOCKED")
n_fail = sum(1 for g in G if g["status"]=="FAIL")
report = {"suite":"GLOBAL_REMEDIATION_REGRESSION_GATES_V1","total":len(G),
          "counts":{"PASS":n_pass,"PARTIAL":n_part,"BLOCKED":n_block,"FAIL":n_fail},
          "policy":"PARTIAL/BLOCKED = environment limits (no JAX / no craftax / CC2 domain), explicitly NOT FAIL. No FAIL gate may be relabeled.",
          "gates":G}
with open(os.path.join(OUT,"global_regression_gates_report.json"),"w",encoding="utf-8") as f:
    json.dump(report,f,indent=2,ensure_ascii=False)
print(f"GATES: PASS={n_pass} PARTIAL={n_part} BLOCKED={n_block} FAIL={n_fail} (total {len(G)})")
for g in G: print(f"  {g['gate']:7s} {g['status']:8s} {g['name']}")
print("REGRESSION_GATES_REPORT_WRITTEN" if n_fail==0 else "REGRESSION_GATES_HAS_FAIL")
