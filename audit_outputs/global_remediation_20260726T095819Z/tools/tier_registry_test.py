#!/usr/bin/env python
"""GATE7 + GATE8: official achievement registry must match installed craftax source.

GATE7: the 67 achievements (names + ids 0-66) equal craftax.craftax.constants.Achievement enum.
GATE8: the official 4 tiers + reward weights (BASIC1/INTERMEDIATE3/ADVANCED5/VERY_ADVANCED8) equal
       craftax.craftax.constants.ACHIEVEMENT_REWARD_MAP, and tier counts are 25/18/15/9.

In a craftax-present env this imports the real source and asserts equality (authoritative).
In THIS audit env craftax is ABSENT, so the test runs the PURE-PYTHON frozen-fact self-check against
official_achievement_tiers.json and reports GATE7/8 as BLOCKED_ON_CRAFTAX (NOT FAIL).
"""
import argparse, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
REG_PATH = os.path.join(os.path.dirname(HERE), "official_achievement_tiers.json")
EXPECTED_COUNTS = {"BASIC": 25, "INTERMEDIATE": 18, "ADVANCED": 15, "VERY_ADVANCED": 9}
EXPECTED_REWARD = {"BASIC": 1, "INTERMEDIATE": 3, "ADVANCED": 5, "VERY_ADVANCED": 8}

def load_registry():
    return json.load(open(REG_PATH, encoding="utf-8"))

def pure_python_checks(reg):
    achs = reg["achievements"]
    ids = sorted(a["canonical_id"] for a in achs)
    c7_ids = (len(achs) == 67 and ids == list(range(67)))
    counts = {}
    reward_ok = True
    for a in achs:
        counts[a["official_tier_name"]] = counts.get(a["official_tier_name"], 0) + 1
        if a["official_reward_weight"] != EXPECTED_REWARD[a["official_tier_name"]]:
            reward_ok = False
    c8 = (counts == EXPECTED_COUNTS and reward_ok)
    ff = reg["frozen_facts_verified"]
    ff_ok = all(ff.values())
    return {"gate7_ids_0_66_count_67": c7_ids, "gate8_tier_counts_25_18_15_9": counts == EXPECTED_COUNTS,
            "gate8_reward_weights_1_3_5_8": reward_ok, "frozen_facts": ff, "frozen_facts_all_pass": ff_ok,
            "counts": counts}

def craftax_checks(reg):
    """Authoritative check against installed craftax (runs only when craftax importable)."""
    from craftax.craftax.constants import ACHIEVEMENT_REWARD_MAP, Achievement  # noqa
    achs = reg["achievements"]
    # GATE7: enum membership + ids
    enum_names = {a.name.lower(): a.value for a in Achievement}
    g7 = (len(enum_names) == 67 and
          all(a["name"] in enum_names and enum_names[a["name"]] == a["canonical_id"] for a in achs))
    # GATE8: reward map -> tier
    rev = {v: k for k, v in EXPECTED_REWARD.items()}
    g8 = True
    for a in achs:
        enum_member = Achievement[a["name"].upper()]
        rw = ACHIEVEMENT_REWARD_MAP[enum_member]
        if rw != a["official_reward_weight"] or rev.get(rw) != a["official_tier_name"]:
            g8 = False
    return {"gate7_vs_craftax_enum": g7, "gate8_vs_ACHIEVEMENT_REWARD_MAP": g8}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verbose", action="store_true"); a = ap.parse_args()
    reg = load_registry()
    pp = pure_python_checks(reg)
    print("PURE-PYTHON FROZEN-FACT CHECKS:", json.dumps(pp, indent=2))
    try:
        import craftax  # noqa
        cx = craftax_checks(reg)
        print("CRAFTAX SOURCE CHECKS:", json.dumps(cx, indent=2))
        ok = pp["frozen_facts_all_pass"] and all(cx.values())
        print("TIER_REGISTRY_TEST_PASS" if ok else "TIER_REGISTRY_TEST_FAIL")
        raise SystemExit(0 if ok else 1)
    except ImportError:
        print("GATE7/GATE8 vs installed craftax: BLOCKED_ON_CRAFTAX (craftax ABSENT) - NOT FAIL")
        ok = pp["gate7_ids_0_66_count_67"] and pp["gate8_tier_counts_25_18_15_9"] and pp["gate8_reward_weights_1_3_5_8"] and pp["frozen_facts_all_pass"]
        print("PURE_PYTHON_SELF_CHECK_PASS" if ok else "PURE_PYTHON_SELF_CHECK_FAIL")
        raise SystemExit(0 if ok else 2)

if __name__ == "__main__":
    main()
