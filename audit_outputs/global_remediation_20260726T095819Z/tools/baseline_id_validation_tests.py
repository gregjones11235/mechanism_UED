#!/usr/bin/env python
"""Baseline single-identity validator (GATE10, GATE11).

A paired percentage-point comparison is ALLOWED only if BOTH endpoints share identical:
  checkpoint, evaluator_sha, world_set_hash, success_definition, denominator, action_mode.
Otherwise => PAIRED_COMPARISON_NOT_ALLOWED (refuse to put the delta in a global comparison table).

Baseline IDs must be explicit (TEACHER17500_BASELINE | CONTROL24576_BASELINE); bare "Baseline" is rejected.
Pure logic => runs without JAX. Importable as a library and runnable as a self-test.
"""
import argparse, json

LEGAL_BASELINE_IDS = {"TEACHER17500_BASELINE", "CONTROL24576_BASELINE"}
# Fields that MUST be IDENTICAL for a paired comparison to be valid. The checkpoint is the VARIED factor
# (the thing being compared) so it is NOT here; it is checked separately for known identity only.
COMPARE_FIELDS = ["evaluator_sha256", "world_set_hash",
                  "success_definition", "denominator", "action_mode"]

def validate_baseline_id(bid):
    if bid is None or bid.strip().lower() == "baseline":
        return {"ok": False, "reason": "bare 'Baseline' is forbidden; cite an explicit baseline_id"}
    if bid not in LEGAL_BASELINE_IDS:
        return {"ok": False, "reason": f"unknown baseline_id {bid!r}; legal={sorted(LEGAL_BASELINE_IDS)}"}
    return {"ok": True, "baseline_id": bid}

def validate_paired_comparison(endpoint_a, endpoint_b):
    """endpoint_a/b: dicts with baseline_id + COMPARE_FIELDS + checkpoint_path.

    Valid iff: both baseline_ids legal; checkpoint_path known for both (may DIFFER — it is the varied
    factor); and all COMPARE_FIELDS identical. world_set_hash None/REQUIRED => not materialized => deny.
    """
    ida = validate_baseline_id(endpoint_a.get("baseline_id"))
    idb = validate_baseline_id(endpoint_b.get("baseline_id"))
    mismatches = []
    # checkpoint identity must be KNOWN (recorded), but is allowed to differ between the two endpoints
    cka, ckb = endpoint_a.get("checkpoint_path"), endpoint_b.get("checkpoint_path")
    if not cka or cka in ("UNKNOWN", None) or not ckb or ckb in ("UNKNOWN", None):
        mismatches.append({"field": "checkpoint_path", "a": cka, "b": ckb, "reason": "checkpoint identity not recorded"})
    for f in COMPARE_FIELDS:
        va, vb = endpoint_a.get(f), endpoint_b.get(f)
        # world_set_hash None/REQUIRED means not yet materialized => cannot confirm equality => deny
        if f == "world_set_hash" and (va in (None, "REQUIRED") or vb in (None, "REQUIRED")):
            mismatches.append({"field": f, "a": va, "b": vb, "reason": "world_set_hash not materialized"})
        elif va != vb:
            mismatches.append({"field": f, "a": va, "b": vb, "reason": "mismatch"})
    allowed = ida["ok"] and idb["ok"] and not mismatches
    verdict = "PAIRED_COMPARISON_ALLOWED" if allowed else "PAIRED_COMPARISON_NOT_ALLOWED"
    return {"verdict": verdict, "baseline_id_checks": {"a": ida, "b": idb},
            "checkpoint_varied": cka != ckb, "mismatches": mismatches}

def self_test():
    teacher = {"baseline_id": "TEACHER17500_BASELINE", "checkpoint_path": "teacher17500",
               "evaluator_sha256": "224514026a", "world_set_hash": "HASH_X",
               "success_definition": "seen|(info_acc>0)", "denominator": "256", "action_mode": "stochastic"}
    control = dict(teacher); control["baseline_id"] = "CONTROL24576_BASELINE"; control["checkpoint_path"] = "control_RUN2/24576"
    # 1: same protocol, only checkpoint differs => ALLOWED
    r1 = validate_paired_comparison(teacher, control)
    # 2: different evaluator sha => DENIED
    bad_eval = dict(control); bad_eval["evaluator_sha256"] = "dcf7fe20"
    r2 = validate_paired_comparison(teacher, bad_eval)
    # 3: world hash not materialized => DENIED
    no_hash = dict(control); no_hash["world_set_hash"] = "REQUIRED"
    r3 = validate_paired_comparison(teacher, no_hash)
    # 4: seed100000 line (different world hash) vs seed42 => DENIED
    p7line = dict(control); p7line["world_set_hash"] = "HASH_SEED100000"
    r4 = validate_paired_comparison(teacher, p7line)
    # 5: bare 'Baseline' => DENIED
    bare = dict(control); bare["baseline_id"] = "Baseline"
    r5 = validate_paired_comparison(teacher, bare)
    # 6: action_mode mismatch (argmax vs stochastic) => DENIED
    am = dict(control); am["action_mode"] = "argmax"
    r6 = validate_paired_comparison(teacher, am)
    results = {"same_protocol_only_ckpt_differs": r1, "diff_evaluator_sha": r2, "world_hash_unmaterialized": r3,
               "seed100000_vs_seed42": r4, "bare_baseline": r5, "action_mode_mismatch": r6}
    expect = {"same_protocol_only_ckpt_differs": "PAIRED_COMPARISON_ALLOWED",
              "diff_evaluator_sha": "PAIRED_COMPARISON_NOT_ALLOWED",
              "world_hash_unmaterialized": "PAIRED_COMPARISON_NOT_ALLOWED",
              "seed100000_vs_seed42": "PAIRED_COMPARISON_NOT_ALLOWED",
              "bare_baseline": "PAIRED_COMPARISON_NOT_ALLOWED",
              "action_mode_mismatch": "PAIRED_COMPARISON_NOT_ALLOWED"}
    ok = all(results[k]["verdict"] == expect[k] for k in expect)
    print(json.dumps(results, indent=2))
    print("BASELINE_VALIDATION_SELF_TEST_PASS" if ok else "BASELINE_VALIDATION_SELF_TEST_FAIL")
    return ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--a", help="JSON file for endpoint A")
    ap.add_argument("--b", help="JSON file for endpoint B")
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(0 if self_test() else 1)
    if a.a and a.b:
        ea = json.load(open(a.a)); eb = json.load(open(a.b))
        print(json.dumps(validate_paired_comparison(ea, eb), indent=2))
    else:
        print("usage: --self-test  |  --a A.json --b B.json")

if __name__ == "__main__":
    main()
