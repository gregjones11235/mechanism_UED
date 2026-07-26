#!/usr/bin/env python
# CC4 remediation [3/6]: official achievement tiers + tier diff + metric defs fixed +
# baseline registry fixed. Official tier table parsed from the craftax==1.4.5 reward table
# (read from CC3 constants.py docstring, which mirrors craftax.constants.ACHIEVEMENT_REWARD_MAP).
# Canonical IDs cross-referenced from CC3-produced canonical_craftax_achievements.json (read-only).
import csv, json, os, hashlib
BASE=os.getcwd()
OUT=open(os.path.join(BASE,"audit_outputs","_remediation_outdir.txt")).read().strip()
def J(p,o):
    with open(p,"w",encoding="utf-8") as f: json.dump(o,f,indent=2,ensure_ascii=False)
def Wcsv(p,rows,f):
    with open(p,"w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=f); w.writeheader(); w.writerows(rows)

# canonical_id from CC3-produced canonical_craftax_achievements.json (read-only reference)
CANON=json.load(open(os.path.join(BASE,"audit_outputs/d052_readonly_audit_20260726T043613Z/canonical_craftax_achievements.json"),encoding="utf-8"))
NAME2ID=CANON["achievements"]
assert len(NAME2ID)==67 and sorted(NAME2ID.values())==list(range(67))

# OFFICIAL tier table (craftax==1.4.5 ACHIEVEMENT_REWARD_MAP), transcribed from constants.py TABLE 1
# name(UPPER) -> (official_tier, reward)
T={
"COLLECT_WOOD":("BASIC",1),"PLACE_TABLE":("BASIC",1),"EAT_COW":("BASIC",1),"COLLECT_SAPLING":("BASIC",1),
"COLLECT_DRINK":("BASIC",1),"MAKE_WOOD_PICKAXE":("BASIC",1),"MAKE_WOOD_SWORD":("BASIC",1),"PLACE_PLANT":("BASIC",1),
"DEFEAT_ZOMBIE":("BASIC",1),"COLLECT_STONE":("BASIC",1),"PLACE_STONE":("BASIC",1),"EAT_PLANT":("BASIC",1),
"DEFEAT_SKELETON":("BASIC",1),"MAKE_STONE_PICKAXE":("BASIC",1),"MAKE_STONE_SWORD":("BASIC",1),"WAKE_UP":("BASIC",1),
"PLACE_FURNACE":("BASIC",1),"COLLECT_COAL":("BASIC",1),"COLLECT_IRON":("BASIC",1),"COLLECT_DIAMOND":("BASIC",1),
"MAKE_IRON_PICKAXE":("BASIC",1),"MAKE_IRON_SWORD":("BASIC",1),"MAKE_ARROW":("BASIC",1),"MAKE_TORCH":("BASIC",1),
"PLACE_TORCH":("BASIC",1),
"MAKE_DIAMOND_SWORD":("INTERMEDIATE",3),"MAKE_IRON_ARMOUR":("INTERMEDIATE",3),"MAKE_DIAMOND_ARMOUR":("INTERMEDIATE",3),
"ENTER_GNOMISH_MINES":("INTERMEDIATE",3),"ENTER_DUNGEON":("INTERMEDIATE",3),
"DEFEAT_GNOME_WARRIOR":("INTERMEDIATE",3),"DEFEAT_GNOME_ARCHER":("INTERMEDIATE",3),"DEFEAT_ORC_SOLIDER":("INTERMEDIATE",3),
"DEFEAT_ORC_MAGE":("INTERMEDIATE",3),"EAT_BAT":("INTERMEDIATE",3),"EAT_SNAIL":("INTERMEDIATE",3),
"FIND_BOW":("INTERMEDIATE",3),"FIRE_BOW":("INTERMEDIATE",3),"COLLECT_SAPPHIRE":("INTERMEDIATE",3),
"COLLECT_RUBY":("INTERMEDIATE",3),"MAKE_DIAMOND_PICKAXE":("INTERMEDIATE",3),"OPEN_CHEST":("INTERMEDIATE",3),
"DRINK_POTION":("INTERMEDIATE",3),
"ENTER_SEWERS":("ADVANCED",5),"ENTER_VAULT":("ADVANCED",5),"ENTER_TROLL_MINES":("ADVANCED",5),
"DEFEAT_LIZARD":("ADVANCED",5),"DEFEAT_KOBOLD":("ADVANCED",5),"DEFEAT_TROLL":("ADVANCED",5),"DEFEAT_DEEP_THING":("ADVANCED",5),
"LEARN_FIREBALL":("ADVANCED",5),"CAST_FIREBALL":("ADVANCED",5),"LEARN_ICEBALL":("ADVANCED",5),"CAST_ICEBALL":("ADVANCED",5),
"ENCHANT_SWORD":("ADVANCED",5),"ENCHANT_ARMOUR":("ADVANCED",5),"DEFEAT_KNIGHT":("ADVANCED",5),"DEFEAT_ARCHER":("ADVANCED",5),
"ENTER_FIRE_REALM":("VERY_ADVANCED",8),"ENTER_ICE_REALM":("VERY_ADVANCED",8),"ENTER_GRAVEYARD":("VERY_ADVANCED",8),
"DEFEAT_PIGMAN":("VERY_ADVANCED",8),"DEFEAT_FIRE_ELEMENTAL":("VERY_ADVANCED",8),"DEFEAT_FROST_TROLL":("VERY_ADVANCED",8),
"DEFEAT_ICE_ELEMENTAL":("VERY_ADVANCED",8),"DAMAGE_NECROMANCER":("VERY_ADVANCED",8),"DEFEAT_NECROMANCER":("VERY_ADVANCED",8),
}
assert len(T)==67, len(T)
# every table name must map to a canonical id
for nm in T:
    assert nm.lower() in NAME2ID, nm

counts={}
for nm,(tier,rw) in T.items(): counts[tier]=counts.get(tier,0)+1
# frozen-fact assertions
assert counts=={"BASIC":25,"INTERMEDIATE":18,"ADVANCED":15,"VERY_ADVANCED":9}, counts
assert T["MAKE_IRON_PICKAXE"]==("BASIC",1)
assert T["MAKE_DIAMOND_SWORD"][0]==T["MAKE_DIAMOND_ARMOUR"][0]==T["MAKE_DIAMOND_PICKAXE"][0]=="INTERMEDIATE"
assert T["LEARN_FIREBALL"][0]==T["CAST_FIREBALL"][0]==T["LEARN_ICEBALL"][0]==T["CAST_ICEBALL"][0]=="ADVANCED"
assert T["ENTER_FIRE_REALM"][0]==T["ENTER_ICE_REALM"][0]==T["DEFEAT_NECROMANCER"][0]=="VERY_ADVANCED"

TIER_REWARD={"BASIC":1,"INTERMEDIATE":3,"ADVANCED":5,"VERY_ADVANCED":8}
achievements=[]
for nm,(tier,rw) in sorted(T.items(), key=lambda kv: NAME2ID[kv[0].lower()]):
    achievements.append({"canonical_id":NAME2ID[nm.lower()],"name":nm.lower(),"name_upper":nm,
        "official_tier_name":tier,"official_reward_weight":rw,"goal_vector_index":NAME2ID[nm.lower()]})
reg={
 "schema":"mechanism_UED.official_achievement_tiers/v1",
 "single_source_of_truth":"craftax.craftax.constants.ACHIEVEMENT_REWARD_MAP (craftax==1.4.5)",
 "import_for_verification":"from craftax.craftax.constants import ACHIEVEMENT_REWARD_MAP, Achievement",
 "num_achievements":67,"max_value":66,
 "official_tiers":{t:{"reward_weight":TIER_REWARD[t],"count":counts[t]} for t in ["BASIC","INTERMEDIATE","ADVANCED","VERY_ADVANCED"]},
 "tier_count_check":{"BASIC":25,"INTERMEDIATE":18,"ADVANCED":15,"VERY_ADVANCED":9,"total":67},
 "frozen_facts_verified":{
   "ADVANCED_has_15_items":counts["ADVANCED"]==15,
   "make_iron_pickaxe_is_BASIC":T["MAKE_IRON_PICKAXE"]==("BASIC",1),
   "make_diamond_sword_armour_pickaxe_are_INTERMEDIATE":all(T[k][0]=="INTERMEDIATE" for k in ["MAKE_DIAMOND_SWORD","MAKE_DIAMOND_ARMOUR","MAKE_DIAMOND_PICKAXE"]),
   "learn_cast_fireball_iceball_are_ADVANCED":all(T[k][0]=="ADVANCED" for k in ["LEARN_FIREBALL","CAST_FIREBALL","LEARN_ICEBALL","CAST_ICEBALL"]),
   "realm_necromancer_are_VERY_ADVANCED":all(T[k][0]=="VERY_ADVANCED" for k in ["ENTER_FIRE_REALM","ENTER_ICE_REALM","DEFEAT_NECROMANCER","DAMAGE_NECROMANCER"]),
 },
 "tier3_semantics_rule":"If a report uses 'tier3' for baseline cross-comparison it MUST equal official ADVANCED (reward 5, the 15 items above). 'tier3' MUST NOT be used for any other set.",
 "custom_tier_rule":"The design-layer ACHIEVEMENT_DEPTH (dicode_src/auction/craftax_achievements.py) is NOT authoritative; it MUST be renamed CUSTOM_DEPTH_TIER and NEVER mixed into official-tier tables. New reports MUST emit official_tier_name + official_reward_weight + achievement list + mapping source SHA.",
 "gate7":"registry must equal installed craftax Achievement enum (67, ids 0-66) - auto-checked by tools/tier_registry_test.py in a craftax-present env (BLOCKED here: craftax ABSENT)",
 "gate8":"official 4 tiers must match ACHIEVEMENT_REWARD_MAP reward weights 1/3/5/8 - auto-checked by tools/tier_registry_test.py",
 "verification_status_in_this_env":"craftax ABSENT => runtime GATE7/8 BLOCKED; registry built from craftax==1.4.5 reward table (CC3 constants.py TABLE1) cross-referenced to CC3 canonical_craftax_achievements.json IDs; all frozen-fact assertions PASS in pure python",
 "mapping_source_sha256":{"note":"SHA of this registry recorded in SHA256SUMS; upstream craftax_achievements.py git blob SHA = server-side (NOT_FOUND locally)"},
 "achievements":achievements,
}
J(os.path.join(OUT,"official_achievement_tiers.json"),reg)

# ===================== TIER MAPPING DIFF =====================
# design-layer ACHIEVEMENT_DEPTH lives in server dicode_src/auction/craftax_achievements.py (NOT local).
# We record official tier (filled) vs design-layer CUSTOM_DEPTH_TIER (server, not synced) + divergence flags.
df=["canonical_id","name","official_tier_name","official_reward_weight","custom_depth_tier_design_layer",
    "custom_source","rename_required","divergence_flag","notes"]
rows=[]
for a in achievements:
    nm=a["name_upper"]
    # documented divergence anchors from the task's frozen facts:
    note=""
    if nm=="MAKE_IRON_PICKAXE": note="FROZEN FACT: official BASIC(+1). Any old 'tier3/ADVANCED' label for iron pickaxe is WRONG."
    elif nm in ("LEARN_FIREBALL","CAST_FIREBALL","LEARN_ICEBALL","CAST_ICEBALL"): note="FROZEN FACT: official ADVANCED(+5)."
    elif nm in ("ENTER_FIRE_REALM","ENTER_ICE_REALM","DEFEAT_NECROMANCER","DAMAGE_NECROMANCER"): note="FROZEN FACT: official VERY_ADVANCED(+8)."
    elif nm in ("MAKE_DIAMOND_SWORD","MAKE_DIAMOND_ARMOUR","MAKE_DIAMOND_PICKAXE"): note="FROZEN FACT: official INTERMEDIATE(+3)."
    elif nm=="DEFEAT_KOBOLD": note="target achievement for Stage4 eval; official ADVANCED(+5)."
    rows.append(dict(canonical_id=a["canonical_id"],name=a["name"],official_tier_name=a["official_tier_name"],
        official_reward_weight=a["official_reward_weight"],
        custom_depth_tier_design_layer="NOT_AVAILABLE_LOCALLY (server: dicode_src/auction/craftax_achievements.py ACHIEVEMENT_DEPTH)",
        custom_source="ACHIEVEMENT_DEPTH (design-layer)",rename_required="YES -> CUSTOM_DEPTH_TIER",
        divergence_flag="POSSIBLE (design-layer != official; verify on server)",notes=note))
Wcsv(os.path.join(OUT,"global_tier_mapping_diff.csv"),rows,df)

# ===================== METRIC DEFINITIONS FIXED =====================
# read seed42 recipe hash for reference
seed42=json.load(open(os.path.join(OUT,"world_manifests","canonical_worlds_256_seed42.json"),encoding="utf-8"))
recipe_hash=seed42["world_recipe_hash"]
mdf=f"""# Global Metric Definitions (FIXED) — CANONICAL_EVALUATOR_V1

Supersedes the audit-era metric definitions for all NEW reports. Old reports unchanged (read-only).

## Outcome metrics (unchanged, from Phase2 anchor)
- success = seen_target OR (info_accuracy>0), ever-set; denominator = num worlds (256).
- SR = n_success / N (pp = SR*100). floor3_reach = (max_floor>=3). died/timeout/not_finished partition (sum==N).
- conditional_kill = n_success / n_floor3.

## Achievement / tier metrics (FIXED)
- Single source of truth: `craftax.craftax.constants.ACHIEVEMENT_REWARD_MAP` (craftax==1.4.5), 67 achievements, ids 0-66.
- Official tiers (reward weight): BASIC=1 (25), INTERMEDIATE=3 (18), ADVANCED=5 (15), VERY_ADVANCED=8 (9). See `official_achievement_tiers.json`.
- New reports MUST emit: official_tier_name, official_reward_weight, achievement list, mapping source SHA.
- The design-layer `ACHIEVEMENT_DEPTH` is NOT official. It is retained ONLY as `CUSTOM_DEPTH_TIER` and MUST NOT be mixed into official-tier tables or used as 'tier3'.
- 'tier3' in any baseline cross-comparison MUST mean official ADVANCED (reward 5, 15 items). Other uses are invalid.
- New vs old tier results MUST NOT be placed in the same table.

## Baseline identity (FIXED)
- Always cite a `baseline_id` (TEACHER17500_BASELINE | CONTROL24576_BASELINE). Never bare "Baseline".
- Any percentage-point delta MUST verify identical: checkpoint, evaluator SHA, world_set_hash, success definition, denominator, action_mode. Mismatch => PAIRED_COMPARISON_NOT_ALLOWED.

## World identity (FIXED)
- world_set_hash MUST be recorded per run; paired comparison requires identical world_set_hash AND evaluator SHA (GATE11).
- seed42 (Phase2/P8/P9/W512) and seed100000 (P7/LC) are distinct world sets — never pooled.
- This env is JAX-less: the canonical world manifest is RECIPE-ONLY (world_recipe_hash={recipe_hash[:16]}...); materialized world_set_hash requires a JAX/Craftax host (GATE2/3 NOT_VERIFIED here).

## Statistics (unchanged)
- Primary: paired McNemar (discordant counts, same 256 worlds) + paired bootstrap 95% CI (fixed seed).
- Signal: p<0.05 AND bootstrap CI not crossing 0. Wilson + Clopper-Pearson per arm. Collapsed-regime comparisons INVALID regardless of p.
"""
open(os.path.join(OUT,"global_metric_definitions_fixed.md"),"w",encoding="utf-8").write(mdf)

# ===================== BASELINE REGISTRY FIXED =====================
bl_fields=["baseline_id","label","checkpoint_path","params_sha256","evaluator_sha256","world_set_hash",
 "world_recipe_hash","worlds","seed","action_mode","n_success","SR_pp","task_caliber","success_definition","denominator","notes"]
bl=[
 dict(baseline_id="TEACHER17500_BASELINE",label="Healthy teacher ckpt17500",checkpoint_path="teacher17500 (server)",
   params_sha256="d4e85af58b7f87d6",evaluator_sha256="224514026aefd273a6647e055fc2e1a434760dc5f4b6b9acd0624bbba57035a1",
   world_set_hash="REQUIRED (materialized; JAX-blocked here)",world_recipe_hash=recipe_hash,worlds="256",seed="42",
   action_mode="stochastic",n_success="101",SR_pp="39.453125",task_caliber="S4_dark native start, DEFEAT_KOBOLD, spawn_floor2, optimistic_reset_ratio16, max4096",
   success_definition="seen|(info_acc>0) ever-set",denominator="256 worlds",notes="anchor A/B; vs CONTROL24576 gap +3.125pp NOT significant (p=0.428)"),
 dict(baseline_id="CONTROL24576_BASELINE",label="Control PPO RUN2 @24576",checkpoint_path="control_RUN2/ckpt/24576/full_state.pkl (server)",
   params_sha256="ece6fa9962e815123ce947577a93040057bc9df0b1e686dd28424cb2bbdabf55",evaluator_sha256="224514026aefd273a6647e055fc2e1a434760dc5f4b6b9acd0624bbba57035a1",
   world_set_hash="REQUIRED (materialized; JAX-blocked here)",world_recipe_hash=recipe_hash,worlds="256",seed="42",
   action_mode="stochastic",n_success="93",SR_pp="36.328125",task_caliber="identical to TEACHER17500_BASELINE",
   success_definition="seen|(info_acc>0) ever-set",denominator="256 worlds",notes="anchor C; reproduces canonical 36.33; 24576 hard-gate params_sha; GPU0 trained"),
]
Wcsv(os.path.join(OUT,"global_baseline_registry_fixed.csv"),bl,bl_fields)
print("tier counts:",counts,"| total",sum(counts.values()))
print("WROTE official_achievement_tiers.json, global_tier_mapping_diff.csv, global_metric_definitions_fixed.md, global_baseline_registry_fixed.csv")
