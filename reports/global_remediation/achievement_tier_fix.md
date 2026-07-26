> GLOBAL_EVALUATION_REMEDIATION (CC4) · repo mechanism_UED @ Henry-branch · HEAD=UNVERIFIED (git access BLOCKED)
> Environment: JAX/Craftax ABSENT, experiment server UNREACHABLE, GPU idle (not used), no CC2/CC3 processes.
> Discipline: read-only w.r.t. CC2/CC3 & old audit; NEW_TRAINING_RUNS=0; MISSING/UNVERIFIED never relabeled FAIL.

# Achievement Tier Fix

Single source of truth = `craftax.craftax.constants.ACHIEVEMENT_REWARD_MAP` (craftax==1.4.5); 67 achievements
(IDs 0–66). Server `ACHIEVEMENT_DEPTH` is a DESIGN-layer field → renamed `CUSTOM_DEPTH_TIER`, never mixed with
or used to impersonate official tiers.

## Tier counts (GATE8 PASS)
{"BASIC": 25, "INTERMEDIATE": 18, "ADVANCED": 15, "VERY_ADVANCED": 9, "total": 67}  (sum=67)

## Frozen facts (all verified true)
make_iron_pickaxe=BASIC(20); make_diamond_sword(25)/armour(27)/pickaxe(60)=INTERMEDIATE;
learn/cast fireball(55,56)/iceball(57,58)=ADVANCED; fire/ice realm(33,34)/graveyard(35)/
damage+defeat_necromancer(48,49)=VERY_ADVANCED; defeat_kobold=ADVANCED(41).
For baseline cross-compare, 'tier3' MUST == official ADVANCED.

tools/tier_registry_test.py: PURE_PYTHON_SELF_CHECK_PASS; GATE7/8 vs installed craftax = BLOCKED_ON_CRAFTAX
(craftax ABSENT) — NOT FAIL. Per-achievement official-vs-CUSTOM_DEPTH_TIER diff in global_tier_mapping_diff.csv.
