#!/usr/bin/env python
"""Exact Resume audit harness (CANONICAL). Implements the A/B continuation comparison and GATE14
(missing-component detection). The state-diff engine is PURE LOGIC and runs without JAX (--self-test).

The training-type continuation (run A continuously to 8192; run B to 4096 -> save -> exit -> new process
restore -> continue to 8192; compare FULL state) requires JAX + a training step and is NOT executed in
this round: --run-continuation raises NOT_AUTHORIZED until explicit re-authorization.

State A and B are represented as {component_id: hash_or_value}. compare_full_state reports per-component
MATCH/DIFF/MISSING. detect_missing_components implements GATE14 against the required set (feature-gated).
"""
import argparse, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = json.load(open(os.path.join(os.path.dirname(HERE), "exact_resume_schema.json"), encoding="utf-8"))

def required_components(features):
    """features: dict of booleans -> which conditionally-required components apply."""
    out = []
    for c in SCHEMA["required_components"]:
        req = c["required"]
        if req is True:
            out.append(c["id"]); continue
        # conditional requirements
        if req == "if replay/EMA enabled" and features.get("replay"): out.append(c["id"])
        elif req == "if separately carried" and features.get("separate_action_rng"): out.append(c["id"])
        elif req == "if replay enabled" and features.get("replay"): out.append(c["id"])
        elif req == "if replay/hindsight enabled" and (features.get("replay") or features.get("hindsight")): out.append(c["id"])
        elif req == "if policy-lag gate enabled" and features.get("policy_lag"): out.append(c["id"])
        elif req == "for RMT16" and features.get("rmt"): out.append(c["id"])
        elif req == "for W512" and features.get("w512"): out.append(c["id"])
        elif req == "recommended": out.append(c["id"])  # always compared, but reported as recommended
    return out

def detect_missing_components(ckpt_components, features):
    """GATE14: which REQUIRED components are absent from the checkpoint."""
    req = set(required_components(features))
    present = set(ckpt_components)
    missing = sorted(req - present)
    return {"required": sorted(req), "present": sorted(present), "missing": missing,
            "complete": len(missing) == 0}

def compare_full_state(stateA, stateB, features):
    """leaf-by-leaf compare. Returns per-component verdict + overall bit-exact bool."""
    req = required_components(features)
    per = {}
    bitexact = True
    for c in req:
        a = stateA.get(c, "<MISSING>"); b = stateB.get(c, "<MISSING>")
        if a == "<MISSING>" or b == "<MISSING>":
            v = "MISSING"; bitexact = False
        elif a == b:
            v = "MATCH"
        else:
            v = "DIFF"; bitexact = False
        per[c] = {"A": a, "B": b, "verdict": v}
    return {"bit_exact_continuation": bitexact, "components": per,
            "n_required": len(req),
            "n_match": sum(1 for c in req if per[c]["verdict"] == "MATCH"),
            "n_diff": sum(1 for c in req if per[c]["verdict"] == "DIFF"),
            "n_missing": sum(1 for c in req if per[c]["verdict"] == "MISSING")}

def self_test():
    base_features = {"replay": False, "rmt": False, "w512": False, "separate_action_rng": False,
                     "hindsight": False, "policy_lag": False}
    # 1: identical full states => bit-exact PASS
    full = {"params": "h1", "optimizer_state": "h2", "global_step": "8192", "update_count": "4",
            "jax_rng": "h3", "env_state": "h4", "observation": "h5", "gtrxl_memory": "h6",
            "gtrxl_mask": "h7", "gtrxl_index": "h8", "per_update_metrics": "h9"}
    r1 = compare_full_state(full, dict(full), base_features)
    # 2: a single RNG diff => bit-exact FAIL (this is what params-only checks MISS)
    rng_diff = dict(full); rng_diff["jax_rng"] = "DIFFERENT"
    r2 = compare_full_state(full, rng_diff, base_features)
    # 3: GATE14 P7-like ckpt (params+carry only) => missing optimizer/global_step/env_state/rng
    p7_ckpt = ["params", "gtrxl_memory"]
    g3 = detect_missing_components(p7_ckpt, base_features)
    # 4: GATE14 RMT16-like ckpt missing env_state, replay enabled => missing env_state + restore-relevant
    rmt_features = dict(base_features); rmt_features.update({"replay": True, "rmt": True, "separate_action_rng": True, "policy_lag": True, "hindsight": True})
    rmt_ckpt = ["params", "optimizer_state", "target_ema_params", "global_step", "update_count", "jax_rng",
                "action_rng", "gtrxl_memory", "gtrxl_mask", "gtrxl_index", "replay_buffer", "replay_sampling_rng",
                "pending_episodes", "policy_version", "rmt_extra_state", "next_trajectory_ids", "per_update_metrics"]  # NOTE: no env_state, no observation
    g4 = detect_missing_components(rmt_ckpt, rmt_features)
    # 5: replay-enabled full compare must include replay components
    replay_features = dict(base_features); replay_features["replay"] = True
    full_replay = dict(full); full_replay.update({"target_ema_params": "e1", "replay_buffer": "rb",
                      "replay_sampling_rng": "rs", "pending_episodes": "pe", "next_trajectory_ids": "ti"})
    r5 = compare_full_state(full_replay, dict(full_replay), replay_features)
    rep_replay = dict(full_replay); rep_replay["replay_sampling_rng"] = "DIFFERENT"
    r6 = compare_full_state(full_replay, rep_replay, replay_features)

    results = {"ident_full_bitexact": r1["bit_exact_continuation"],
               "rng_diff_detected": (not r2["bit_exact_continuation"]) and r2["components"]["jax_rng"]["verdict"] == "DIFF",
               "p7_missing_detected": (not g3["complete"]) and set(["optimizer_state", "global_step", "env_state", "jax_rng"]).issubset(set(g3["missing"])),
               "rmt_missing_env_state_detected": (not g4["complete"]) and "env_state" in g4["missing"],
               "replay_components_compared": r5["bit_exact_continuation"],
               "replay_rng_diff_detected": not r6["bit_exact_continuation"]}
    print(json.dumps({"detail": {"r1": r1, "r2_rng": r2["components"]["jax_rng"], "g3_p7": g3, "g4_rmt": g4,
                                  "r6_replay_rng": r6["components"]["replay_sampling_rng"]},
                      "checks": results}, indent=2))
    ok = all(results.values())
    print("EXACT_RESUME_HARNESS_SELF_TEST_PASS" if ok else "EXACT_RESUME_HARNESS_SELF_TEST_FAIL")
    return ok

def run_continuation():
    raise SystemExit("NOT_AUTHORIZED: training-type Exact Resume continuation (A continuous vs B new-process "
                     "restore to 8192) requires JAX + a training step and explicit re-authorization. "
                     "This round delivers the harness + schema + plan ONLY; no training is run.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--run-continuation", action="store_true")
    ap.add_argument("--state-a", help="JSON file: {component: hash}")
    ap.add_argument("--state-b", help="JSON file: {component: hash}")
    ap.add_argument("--features", help="JSON file: feature flags")
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(0 if self_test() else 1)
    if a.run_continuation:
        run_continuation()
    if a.state_a and a.state_b:
        sa = json.load(open(a.state_a)); sb = json.load(open(a.state_b))
        feat = json.load(open(a.features)) if a.features else {}
        print(json.dumps(compare_full_state(sa, sb, feat), indent=2))
    else:
        print("usage: --self-test | --run-continuation | --state-a A.json --state-b B.json [--features f.json]")

if __name__ == "__main__":
    main()
