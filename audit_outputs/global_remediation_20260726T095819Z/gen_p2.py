#!/usr/bin/env python
# CC4 remediation [2/6]: CANONICAL_EVALUATOR_V1 spec + fixed registry + diff + world recipe manifest.
# Grounded in eval_phase2_unified.py (anchor) lines extracted during freeze. Read-only on sources.
import csv, json, os, hashlib, datetime
BASE=os.getcwd()
OUT=open(os.path.join(BASE,"audit_outputs","_remediation_outdir.txt")).read().strip()
WM=os.path.join(OUT,"world_manifests")
def J(p,o):
    with open(p,"w",encoding="utf-8") as f: json.dump(o,f,indent=2,ensure_ascii=False)
def Wcsv(p,rows,f):
    with open(p,"w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=f); w.writeheader(); w.writerows(rows)

# ===================== CANONICAL_EVALUATOR_V1 SPEC =====================
spec={
 "protocol_id":"CANONICAL_EVALUATOR_V1",
 "status":"FROZEN_SPEC (reference implementation requires JAX/Craftax; runtime BLOCKED in this env: JAX ABSENT)",
 "anchor_reference":"student_upgrade_wave1_4gpu/eval_phase2_unified.py",
 "anchor_sha256":"224514026aefd273a6647e055fc2e1a434760dc5f4b6b9acd0624bbba57035a1",
 "derivation_note":"Fields transcribed from the bit-identity-verified Phase2 anchor evaluator; NOT a blind copy. Each field has a source line citation.",
 "frozen_fields":{
   "evaluator_source_sha":{"value":"MUST equal the running evaluator file SHA256; recorded at startup and written to output","source":"eval_phase2_unified.py:80 EVAL_SHA256=sha256(self); :417 written to output","required":True},
   "environment_task_definition":{"value":"Craftax Stage4 / S4_dark native start, target DEFEAT_KOBOLD","source":"docstring :2-5; wrapper :104","required":True},
   "EnvParams":{"value":"EnvParams(max_timesteps=4096)","source":":82 ctor=EnvParams(max_timesteps=NUM_STEPS)","required":True},
   "StaticEnvParams":{"value":"Craftax default StaticEnvParams for S4_dark unless explicitly overridden; MUST be recorded if overridden","source":"not overridden in anchor","required":True},
   "goal_conditioning":{"value":"condition_on_task=True","source":":104","required":True},
   "max_timesteps":{"value":4096,"source":":82 NUM_STEPS","required":True},
   "optimistic_reset":{"value":"optimistic_reset_ratio=16, mode='score', bonus_type='none'","source":":104","required":True},
   "spawn_floor":{"value":2,"source":":200 spawn_floor=2","required":True},
   "action_mode":{"value":"stochastic","allowed":["stochastic","argmax"],
      "source":":146 action=pi.sample(seed=a_rng); :200 policy_mode='stochastic'",
      "rule":"MUST be explicit; printed at startup AND written to output. NEVER inferred from a dead branch or default. The argmax used at :277 is a MEMORY-OFF DIAGNOSTIC PROBE on logits, NOT the policy action_mode, and MUST NOT be reported as the policy mode.",
      "required":True},
   "action_rng":{"value":"per-step a_rng derived deterministically from the eval RNG stream; stochastic pi.sample(seed=a_rng)","source":":146","required":True},
   "evaluation_seed":{"value":42,"allowed_note":"seed=100000 (P7/LC line) is a DIFFERENT world set and MUST be flagged separately, never paired with seed42","source":":77 EVAL_SEED=42; :169 rng=PRNGKey(EVAL_SEED)","required":True},
   "world_generator":{"value":"DistributedMultiTaskOptimisticLogWrapper(s4_base, PRNGKey(0), ...) with NUM_ENVS=256 parallel envs, paired by world index","source":":121; :104","required":True},
   "world_manifest":{"value":"MUST reference canonical_worlds_256_seed42.json and record world_set_hash","required":True},
   "world_hash":{"value":"MUST be written per-run; identical hash required for any paired comparison","required":True},
   "memory_init":{"value":"zero-initialized short-term GTrXL memory + per-network long state at reset","source":":264 reset0=zeros bool; forward_eval","required":True},
   "memory_reset_on_done":{"value":"short-term mem reset on done; long state reset on true_done","source":"anchor memory machinery (verified in prior audit)","required":True},
   "memory_env_isolation":{"value":"per-world memory rows must not cross-contaminate in the batched wrapper; verified by GATE5","required":True},
   "success_definition":{"value":"success = seen_target OR (info_accuracy > 0), ever-set across episode (once True stays True); denominator = num worlds","source":":190 success_np = seen_np | (info_acc_np>0)","required":True},
   "death_definition":{"value":"died = finished AND NOT success AND NOT timeout","source":":192","required":True},
   "timeout_definition":{"value":"timeout = finished AND (ep_len>=max_timesteps) AND NOT success; if any not_finished, fold not_finished into timeout","source":":191,194","required":True},
   "floor_reach_definition":{"value":"floor3_reach = (max_floor >= 3) per world","source":":196","required":True},
   "achievement_statistics":{"value":"official Craftax 67-achievement registry, 4 official tiers (BASIC1/INTERMEDIATE3/ADVANCED5/VERY_ADVANCED8); see official_achievement_tiers.json. Custom design tiers MUST be labelled CUSTOM_DEPTH_TIER and never mixed.","required":True},
   "output_schema":{"value":"per-world bool arrays (success_per_world, floor3_per_world, died_per_world) + aggregates (SR, n_success, floor3_reach_rate, n_floor3, conditional_kill_rate, n_died, n_timeout, n_not_finished, death_rate, timeout_rate) + provenance (evaluator_sha256, world_set_hash, action_mode, evaluation_seed, spawn_floor, policy_mode)","source":":199-208,416-417","required":True}
 },
 "mandatory_runtime_assertions":[
   "A1 print actual action_mode at startup",
   "A2 write action_mode into output JSON",
   "A3 test that actual sampling behavior matches declared action_mode (stochastic => >1 distinct action over repeated draws at fixed state; argmax => identical)",
   "A4 checkpoint param set mismatch (missing OR extra leaves) => HARD-FAIL; no silent partial restore; no compat fallback",
   "A5 batched-env memory cross-contamination probe => must be zero (GATE5)",
   "A6 after episode done, memory returns to zero on next episode start (GATE6)"
 ],
 "action_mode_test_logic":"stochastic: sample K times at fixed (params,state,rng-stream-with-distinct-keys) => expect >1 unique action with high prob; argmax: sample K times => expect exactly 1 unique action. Test implemented in tools/action_mode_consistency_test.py (pure-logic, JAX-optional).",
 "partial_restore_policy":"DEFAULT HARD-FAIL. Restore compares leaf-set(ckpt) vs leaf-set(model). If ckpt ⊋ model or ckpt ⊊ model => raise RestoreShapeMismatch/RestoreLeafMismatch. Compat fallback is FORBIDDEN unless an explicit, SHA-recorded migration map is supplied and audited.",
 "disciplines":["action_mode never guessed from dead code/default","seed100000 results flagged separate from seed42","no cross-evaluator/world-set paired comparison without identical world_set_hash + evaluator SHA"]
}
J(os.path.join(OUT,"global_evaluator_canonical_spec.json"),spec)

# ===================== FIXED EVALUATOR REGISTRY =====================
reg_fields=["evaluator_id","script_path","sha256","action_mode_declared","action_mode_actual","action_mode_consistent",
 "worlds","seed","world_set_hash_recorded","checkpoint_load_policy","memory_isolation_checked","canonical_v1_compliant",
 "required_fix","disposition"]
reg=[
 dict(evaluator_id="PHASE2_UNIFIED",script_path="student_upgrade_wave1_4gpu/eval_phase2_unified.py",sha256="224514026aefd273...",action_mode_declared="stochastic",action_mode_actual="stochastic (pi.sample :146)",action_mode_consistent="YES",worlds="256",seed="42",world_set_hash_recorded="NO (implicit only)",checkpoint_load_policy="strict pickle->{'params':..}, no fallback",memory_isolation_checked="partial (forward_eval resets)",canonical_v1_compliant="NEAR (add explicit world_set_hash + startup action_mode print)",required_fix="add world_set_hash + A1/A2 explicit action_mode emission",disposition="ADOPT AS ANCHOR; upgrade to V1"),
 dict(evaluator_id="P7_PAIRED_256",script_path="gpu1_p7_egomap/eval_p7_egomap_paired_256.py",sha256="protocol claims 51c37c27",action_mode_declared="stochastic",action_mode_actual="stochastic (pi.sample)",action_mode_consistent="YES (functional path)",worlds="256",seed="100000",world_set_hash_recorded="NO",checkpoint_load_policy="strict, no fallback",memory_isolation_checked="unknown",canonical_v1_compliant="PARTIAL (seed differs; no world hash)",required_fix="flag seed=100000 separate from seed42; record params SHA (hashlib imported but unused); add world_set_hash",disposition="KEEP but mark seed100000 line; not pairable with seed42"),
 dict(evaluator_id="P7_BROKEN",script_path="gpu1_p7_egomap/eval_p7.py",sha256="n/a",action_mode_declared="n/a",action_mode_actual="ARGMAX (pi.mode) - CONTRADICTS paired_256",action_mode_consistent="N/A (dead code: build_eval_fn never called; TypeError/NameError)",worlds="n/a",seed="n/a",world_set_hash_recorded="n/a",checkpoint_load_policy="n/a",memory_isolation_checked="n/a",canonical_v1_compliant="NO",required_fix="REMOVE or QUARANTINE: dead argmax path + unreachable compare_arms(McNemar/bootstrap). Must not be cited as protocol.",disposition="QUARANTINE (dead code); argmax must not represent P7 protocol"),
 dict(evaluator_id="P8_FINAL",script_path="gpu2_p8_longmemory/src/eval_p8_final.py",sha256="self-hash A",action_mode_declared="stochastic",action_mode_actual="stochastic",action_mode_consistent="YES",worlds="256",seed="42",world_set_hash_recorded="NO",checkpoint_load_policy="load_params tree_unflatten, raise on missing (strict)",memory_isolation_checked="short-on-done/long-on-true_done",canonical_v1_compliant="PARTIAL",required_fix="add world_set_hash; sync per-world arrays to audit",disposition="KEEP; upgrade to V1"),
 dict(evaluator_id="P8_MIGRATION",script_path="gpu2_p8_longmemory/src/eval_p8_migration.py",sha256="self-hash B (!=A)",action_mode_declared="stochastic",action_mode_actual="stochastic",action_mode_consistent="YES",worlds="64",seed="42",world_set_hash_recorded="NO",checkpoint_load_policy="orbax load_weights_only nested->flat->raw SILENT FALLBACK CHAIN",memory_isolation_checked="same as final",canonical_v1_compliant="NO (silent fallback violates A4; 64 worlds != 256)",required_fix="HARD-FAIL the silent fallback chain (GATE4); mark 64-world gate distinct from 256-world; baseline=teacher17500 differs from final's Control",disposition="FIX load policy; keep as 64w migration gate only, never pooled with 256w"),
 dict(evaluator_id="P9_FINAL",script_path="gpu3_p9_authentic_reset/src/eval_p9_final.py",sha256="self-hash",action_mode_declared="stochastic",action_mode_actual="stochastic",action_mode_consistent="YES",worlds="256",seed="42",world_set_hash_recorded="NO",checkpoint_load_policy="strict",memory_isolation_checked="GTrXL on done",canonical_v1_compliant="PARTIAL",required_fix="add world_set_hash; sync per-world + compare_resume artifact",disposition="KEEP; upgrade to V1"),
 dict(evaluator_id="W512_A_SIDE_UNIFIED",script_path="bakeoff_phase1/shared/eval_a_side_unified.py",sha256="dcf7fe20",action_mode_declared="stochastic",action_mode_actual="stochastic",action_mode_consistent="YES",worlds="256",seed="42",world_set_hash_recorded="NO",checkpoint_load_policy="strict",memory_isolation_checked="GTrXL on done",canonical_v1_compliant="PARTIAL",required_fix="UNIFY with W512_P2REPLAY evaluator (single file); add world_set_hash",disposition="MERGE into single W512 canonical evaluator (drift fix)"),
 dict(evaluator_id="W512_P2REPLAY",script_path="bakeoff_phase1/eval_w512_p2replay.py",sha256="f76bb53c (!=dcf7fe20)",action_mode_declared="stochastic",action_mode_actual="stochastic",action_mode_consistent="YES",worlds="256",seed="42",world_set_hash_recorded="NO",checkpoint_load_policy="strict",memory_isolation_checked="long_buf only true_done reset",canonical_v1_compliant="PARTIAL",required_fix="UNIFY: two W512 evaluator files cause caliber drift across arms1-4 vs arms5-6; collapse to one file under CANONICAL_EVALUATOR_V1",disposition="MERGE into single W512 canonical evaluator (drift fix)"),
 dict(evaluator_id="G6_GATE_ARGMAX",script_path="(G6 / memory-off gate code)",sha256="n/a",action_mode_declared="argmax (gate)",action_mode_actual="argmax",action_mode_consistent="INCONSISTENT vs main stochastic protocol",worlds="n/a",seed="n/a",world_set_hash_recorded="n/a",checkpoint_load_policy="n/a",memory_isolation_checked="n/a",canonical_v1_compliant="NO",required_fix="RELABEL: G6 argmax is a MEMORY-OFF LOGIT DIAGNOSTIC (cf anchor :277), NOT the policy action_mode; document explicitly and never report as eval policy mode",disposition="RELABEL as diagnostic probe"),
 dict(evaluator_id="LC_LINE",script_path="student_long_context_wave1/shared",sha256="self",action_mode_declared="stochastic",action_mode_actual="stochastic",action_mode_consistent="YES",worlds="256",seed="100000",world_set_hash_recorded="NO",checkpoint_load_policy="n/a",memory_isolation_checked="LC summary chunk",canonical_v1_compliant="PARTIAL (seed 100000)",required_fix="flag seed100000 separate; no pooled comparison with seed42",disposition="KEEP separate seed line"),
]
Wcsv(os.path.join(OUT,"global_evaluator_registry_fixed.csv"),reg,reg_fields)

# ===================== EVALUATOR DIFF =====================
diff="""# Global Evaluator Diff — CANONICAL_EVALUATOR_V1 remediation

Anchor: `eval_phase2_unified.py` (SHA `224514026aefd273...`, bit-identity verified == local file).
All fixes below are realized as NEW canonical files under this remediation dir; **no existing evaluator or
old result is overwritten or silently changed** (per discipline).

## The 7 mandated fixes

### FIX-E1 — P7 argmax dead code
- **Finding:** `gpu1_p7_egomap/eval_p7.py` uses `pi.mode` (ARGMAX) and contains `build_eval_fn` that is
  never called (TypeError/NameError on invocation); its `compare_arms` (McNemar/bootstrap) is therefore
  unreachable. The functional P7 evaluator `eval_p7_egomap_paired_256.py` is STOCHASTIC (`pi.sample`).
- **Risk:** a dead argmax path could be mis-cited as the P7 protocol, contradicting the stochastic main path.
- **Fix:** QUARANTINE `eval_p7.py`; CANONICAL_EVALUATOR_V1 forbids reporting argmax unless it is the actual
  executed mode. The canonical reference evaluator prints + records `action_mode` (assertions A1/A2) so a dead
  branch can never silently define the mode.

### FIX-E2 — G6 / gate argmax-stochastic inconsistency
- **Finding:** the anchor itself uses `jnp.argmax(logits)` at `:277` — but ONLY as a memory-off diagnostic probe
  (do on/off networks pick different argmax actions?), NOT as the policy. `policy_mode="stochastic"` at `:200`.
- **Risk:** the argmax diagnostic could be mis-read as the evaluation policy.
- **Fix:** RELABEL all such argmax usages as `memory_off_diagnostic_probe`; CANONICAL_EVALUATOR_V1 separates
  `action_mode` (policy) from any diagnostic probe and requires the probe to be named as such in output.

### FIX-E3 — W512 multi-evaluator caliber drift
- **Finding:** W512 arms1-4 use `eval_a_side_unified.py` (SHA dcf7fe20) while arms5-6 use
  `eval_w512_p2replay.py` (SHA f76bb53c). Two files, "functionally equivalent" claimed but NOT byte-identical.
- **Risk:** cross-arm W512 comparisons carry an evaluator-file difference; violates "single canonical protocol".
- **Fix:** MERGE to a single W512 evaluator instance under CANONICAL_EVALUATOR_V1 (one SHA, one world_set_hash),
  re-run all 6 arms through it (future, authorized eval). Until then, W512 cross-file deltas are flagged
  `EVALUATOR_FILE_DRIFT`.

### FIX-E4 — checkpoint partial restore / silent fallback
- **Finding:** `eval_p8_migration.py` uses orbax `load_weights_only` with a nested→flat→raw **silent fallback
  chain**. Silent partial restore can mask missing/extra params.
- **Fix:** CANONICAL_EVALUATOR_V1 mandates HARD-FAIL on any leaf-set mismatch (missing OR extra), GATE4. Compat
  fallback FORBIDDEN unless an explicit SHA-recorded migration map is supplied + audited.

### FIX-E5 — memory-off ablation truly only disables memory
- **Finding:** memory-off must zero the long-state input while keeping the short-term policy path identical;
  verified by the argmax-logit divergence probe (`:277`) which should show off≠on ONLY where memory matters.
- **Fix:** CANONICAL_EVALUATOR_V1 records the memory-off probe result explicitly and requires GATE5/GATE6
  (isolation + done-reset) so "memory-off" is provably only the memory channel.

### FIX-E6 — batched-env state cross-contamination
- **Risk:** in the distributed/batched wrapper, per-world memory rows could leak across the batch.
- **Fix:** GATE5 cross-contamination probe (perturb world i's memory, assert worlds j≠i unchanged). Required
  before any per-world paired statistic is trusted.

### FIX-E7 — episode-done memory reset
- **Risk:** memory not zeroed at episode boundary leaks previous-episode state into the next world's episode.
- **Fix:** GATE6 asserts memory == zero at the start of each episode after a done. Anchor resets short mem on
  done and long state on true_done; canonical evaluator records the reset hook + asserts it.

## action_mode handling (global)
- Only two legal values: `stochastic`, `argmax`.
- MUST be printed at startup (A1) and written to output (A2).
- A behavioral test (A3, `tools/action_mode_consistency_test.py`) confirms sampling matches declaration.
- NEVER inferred from a dead branch or default value.

## seed/world discipline
- seed42 (Phase2/P8/P9/W512) and seed100000 (P7/LC) are DIFFERENT world sets — never pooled.
- No paired comparison without identical `world_set_hash` AND evaluator SHA (GATE11).
"""
open(os.path.join(OUT,"global_evaluator_diff.md"),"w",encoding="utf-8").write(diff)

# ===================== WORLD MANIFEST BUILDER (tool) =====================
builder='''#!/usr/bin/env python
"""CANONICAL_EVALUATOR_V1 world-manifest builder.

Builds canonical_worlds_256_seed<SEED>.json binding the FULL world-generation recipe so that an identical
manifest is bit-reproducible under an identical (env_version, world_generator_sha, evaluator_sha).

TWO modes:
  --recipe-only   (default in a JAX-less env): emit the deterministic INPUTS (world_index, base_seed,
                  fold_in rule, full RNG-input derivation, generator/evaluator SHA, env_version). The
                  world_set_hash is left null and a world_recipe_hash (over the recipe) is recorded.
                  world_params_materialized=False. This is HONEST: worlds are not materialized here.
  --materialize   (requires JAX+Craftax): actually run env.reset for all 256 worlds, record per-world
                  task/params, and compute world_set_hash = sha256(canonical_json(materialized worlds)).
                  GATE2 (reproducible) and GATE3 (order-sensitivity) become checkable.

This file is a reference tool. In this audit env JAX is ABSENT, so only --recipe-only is runnable; the
materialized manifest + world_set_hash are produced on a JAX/Craftax host (future authorized run).
"""
import argparse, hashlib, json, os, sys

NUM_WORLDS = 256

def fold_in_rule(key, i):
    """Documented RNG fold rule (matches jax.random.fold_in semantics)."""
    return f"fold_in(base_key, world_index={i})"

def build_recipe(base_seed, wrapper_key_seed, generator_sha, evaluator_sha, env_version):
    worlds = []
    for i in range(NUM_WORLDS):
        worlds.append({
            "world_index": i,
            "base_seed": base_seed,
            "wrapper_prng_seed": wrapper_key_seed,
            "fold_in": fold_in_rule("PRNGKey(%d)" % wrapper_key_seed, i),
            "rng_input": "reset_rng = split(PRNGKey(base_seed)); obsv,log = env.reset(reset_rng, ctor) [world %d]" % i,
            "world_params": None,   # populated only in --materialize
            "task_params": None,     # populated only in --materialize
        })
    recipe = {
        "protocol": "CANONICAL_EVALUATOR_V1",
        "num_worlds": NUM_WORLDS,
        "base_seed": base_seed,
        "wrapper_prng_seed": wrapper_key_seed,
        "fold_in_rule": "jax.random.fold_in(PRNGKey(wrapper_prng_seed), world_index)",
        "world_generator_sha256": generator_sha,
        "evaluator_sha256": evaluator_sha,
        "env_version": env_version,
        "paired_by": "world_index (identical index across arms => paired)",
        "worlds": worlds,
    }
    return recipe

def canonical_json_bytes(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wrapper-key-seed", type=int, default=0)
    ap.add_argument("--generator-sha", default="UNVERIFIED")
    ap.add_argument("--evaluator-sha", default="224514026aefd273a6647e055fc2e1a434760dc5f4b6b9acd0624bbba57035a1")
    ap.add_argument("--env-version", default="craftax==1.4.5(EXPECTED; verify on host)")
    ap.add_argument("--materialize", action="store_true",
                    help="requires JAX+Craftax; materializes worlds and computes world_set_hash")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    recipe = build_recipe(a.seed, a.wrapper_key_seed, a.generator_sha, a.evaluator_sha, a.env_version)
    recipe["world_recipe_hash"] = hashlib.sha256(canonical_json_bytes(recipe)).hexdigest()

    if a.materialize:
        try:
            import jax, craftax  # noqa
        except Exception as e:
            print("MATERIALIZE BLOCKED: JAX/Craftax absent (%s). Emitting recipe-only." % e)
            a.materialize = False
    if a.materialize:
        # Real materialization would run the wrapper here and fill world_params/task_params, then:
        # recipe["world_set_hash"] = sha256(canonical_json(materialized))
        raise SystemExit("materialization path must be completed on a JAX/Craftax host; not implemented in audit env")
    else:
        recipe["world_set_hash"] = None
        recipe["world_params_materialized"] = False
        recipe["status"] = "RECIPE_ONLY (JAX/Craftax absent in this env; world_set_hash requires materialization on host)"
        recipe["gate2_reproducible"] = "NOT_VERIFIED (requires materialization)"
        recipe["gate3_order_sensitive"] = "NOT_VERIFIED (requires materialization)"

    out = a.out or ("canonical_worlds_%d_seed%d.json" % (NUM_WORLDS, a.seed))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(recipe, f, indent=2, ensure_ascii=False)
    sh = hashlib.sha256(open(out, "rb").read()).hexdigest()
    with open(out + ".sha256", "w", encoding="utf-8") as f:
        f.write(sh + "  " + os.path.basename(out) + "\\n")
    print("wrote", out, "recipe_hash", recipe["world_recipe_hash"][:16], "world_set_hash", recipe["world_set_hash"])

if __name__ == "__main__":
    main()
'''
open(os.path.join(OUT,"tools","build_world_manifest.py"),"w",encoding="utf-8").write(builder)

# Run recipe-only for seed42 (and note seed100000 separate)
import subprocess
for seed in (42,100000):
    out=os.path.join(WM,f"canonical_worlds_256_seed{seed}.json")
    subprocess.run([sys.executable if False else "python",
                    os.path.join(OUT,"tools","build_world_manifest.py"),
                    "--seed",str(seed),"--out",out],check=True)
print("WROTE spec/registry/diff/builder + recipe manifests for seed42 & seed100000")
