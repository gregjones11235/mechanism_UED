#!/usr/bin/env python3
"""Unified 3-model Stage4-native (S4_dark) 64-episode comparison (CPU only).

Models (same frozen evaluator protocol, parent SHA 06221187; same 64 world seeds,
seed 42; success = DEFEAT_KOBOLD ever-set, achievement read before terminal
auto-reset; stochastic policy; 4096 max steps):
  - session175 baseline (healthy start, no P2 training): 25/64 frozen reference
  - P2-v1 Full @24576 (方案2 isolated critic-only replay-aux + PPO)
  - Original PPO @24576 (pure Henry native PPO control, replay/hindsight OFF)

Computes per-model: DK SR + Wilson95 CI, floor3 reach, conditional kill (DK|floor3),
ENTER_SEWERS, death, timeout, episode-length stats, max_floor dist. Paired seed
flips + McNemar for each pair. Writes evidence JSON + per-seed CSV, prints table.
"""
import json, os, math, csv
import numpy as np

FILES = {
    "session175_baseline": "/home/oseasy/experiments/session175_dual_caliber_pilot_20260722/results/STAGE4_NATIVE_episodes.jsonl",
    "p2_v1_full_24576":    "/home/oseasy/experiments/p2_v1_full_24576_64ep_stage4_eval_20260723/results/STAGE4_NATIVE_episodes.jsonl",
    "original_ppo_24576":  "/home/oseasy/experiments/original_ppo_24576_64ep_stage4_eval_20260723/results/STAGE4_NATIVE_episodes.jsonl",
}
def load(path):
    # Pairing key = episode_idx (0..63): the evaluator derives each env's world
    # seed deterministically from the master evaluation_seed=42, so episode_idx i
    # is the SAME world seed across all three models (verified identical mapping).
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return {int(r["episode_idx"]): r for r in rows}

def truthy(v): return bool(v) if not isinstance(v, str) else v.lower() in ("true", "1", "yes")

def wilson(k, n, z=1.959963985):
    if n == 0: return (0.0, 0.0)
    p = k / n; den = 1 + z*z/n
    c = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return ((p + z*z/(2*n) - c)/den, (p + z*z/(2*n) + c)/den)

data = {name: load(p) for name, p in FILES.items()}
# paired-design validity: identical seed sets
seedsets = {name: set(d.keys()) for name, d in data.items()}
common = set.intersection(*seedsets.values())
assert all(len(d) == 64 for d in data.values()), "each model must have 64 episodes"
seeds_ok = all(s == seedsets["session175_baseline"] for s in seedsets.values())

def metrics(d):
    n = len(d); rows = list(d.values())
    dk = sum(truthy(r["DEFEAT_KOBOLD"]) for r in rows)
    f3 = sum(truthy(r["floor3_reach"]) for r in rows)
    es = sum(truthy(r["ENTER_SEWERS"]) for r in rows)
    death = sum(truthy(r["death"]) for r in rows)
    timeout = sum(truthy(r["timeout"]) for r in rows)
    dk_and_f3 = sum(truthy(r["DEFEAT_KOBOLD"]) and truthy(r["floor3_reach"]) for r in rows)
    lens = [int(r["episode_length"]) for r in rows]
    mf = {}
    for r in rows: mf[int(r["max_floor"])] = mf.get(int(r["max_floor"]), 0) + 1
    lo, hi = wilson(dk, n)
    return {"n": n, "DEFEAT_KOBOLD": dk, "SR_pct": round(100*dk/n, 2),
            "wilson95_pct": [round(100*lo, 2), round(100*hi, 2)],
            "floor3_reach": f3, "floor3_pct": round(100*f3/n, 2),
            "conditional_kill_DK_given_floor3": dk_and_f3,
            "conditional_kill_pct": round(100*dk_and_f3/f3, 2) if f3 else None,
            "ENTER_SEWERS": es, "death": death, "timeout": timeout,
            "ep_len_mean": round(float(np.mean(lens)), 1), "ep_len_median": int(np.median(lens)),
            "ep_len_min": int(min(lens)), "ep_len_max": int(max(lens)),
            "max_floor_dist": dict(sorted(mf.items()))}

per_model = {name: metrics(d) for name, d in data.items()}

def succ(d, seed): return truthy(d[seed]["DEFEAT_KOBOLD"])
def mcnemar(a, b, seeds):
    # b = a-only success (gained by a), c = b-only success (gained by b)
    bonly = sum(1 for s in seeds if succ(a, s) and not succ(b, s))
    conly = sum(1 for s in seeds if not succ(a, s) and succ(b, s))
    both = sum(1 for s in seeds if succ(a, s) and succ(b, s))
    neither = sum(1 for s in seeds if not succ(a, s) and not succ(b, s))
    chi2 = ((bonly-conly)**2 / (bonly+conly)) if (bonly+conly) > 0 else 0.0
    z = ((bonly-conly) / math.sqrt(bonly+conly)) if (bonly+conly) > 0 else 0.0
    from math import erfc, sqrt
    p = erfc(abs(z)/sqrt(2))  # two-sided normal approx
    return {"a_only_gained": bonly, "b_only_gained": conly, "both_success": both,
            "neither": neither, "mcnemar_chi2": round(chi2, 4), "z": round(z, 4),
            "p_two_sided": round(p, 4)}

seeds = sorted(common)
pairs = {
    "p2full_vs_original_ppo": mcnemar(data["p2_v1_full_24576"], data["original_ppo_24576"], seeds),
    "p2full_vs_baseline":     mcnemar(data["p2_v1_full_24576"], data["session175_baseline"], seeds),
    "original_ppo_vs_baseline": mcnemar(data["original_ppo_24576"], data["session175_baseline"], seeds),
}

report = {"protocol": "Stage4-native S4_dark, 64 world seeds (seed 42), stochastic, "
                       "4096 max steps, success=DEFEAT_KOBOLD ever-set, evaluator parent SHA 06221187",
          "evaluator_sha": {"p2_v1_full_24576": "b67d991d9ba7967f498b7daddd85fea2031fa2d281f15dae3baab4dc7d065600",
                            "original_ppo_24576": "b4dfacab40c5aacfacc1cd9ab9f4eac25e47b191e8b27f63af82e5e61aa27bfc",
                            "session175_baseline_parent": "06221187ac06d7da59dac64e6273abfc865b3baafdc75615e0808fc5065d26e2"},
          "paired_seeds_identical": seeds_ok, "n_seeds": len(common),
          "per_model": per_model, "paired_mcnemar": pairs,
          "files": FILES}
ev = "/home/oseasy/experiments/single_director_20260722/evidence"
os.makedirs(ev, exist_ok=True)
with open(os.path.join(ev, "p2_v1_three_model_unified_eval.json"), "w") as f:
    json.dump(report, f, indent=2, sort_keys=True, default=str); f.write("\n")

# per-seed CSV
csv_path = os.path.join(ev, "p2_v1_three_model_per_seed.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["episode_idx", "baseline_DK", "baseline_floor3", "baseline_len",
                "p2full_DK", "p2full_floor3", "p2full_len",
                "origppo_DK", "origppo_floor3", "origppo_len"])
    for s in seeds:
        b, p2, op = data["session175_baseline"][s], data["p2_v1_full_24576"][s], data["original_ppo_24576"][s]
        w.writerow([s, int(truthy(b["DEFEAT_KOBOLD"])), int(truthy(b["floor3_reach"])), b["episode_length"],
                    int(truthy(p2["DEFEAT_KOBOLD"])), int(truthy(p2["floor3_reach"])), p2["episode_length"],
                    int(truthy(op["DEFEAT_KOBOLD"])), int(truthy(op["floor3_reach"])), op["episode_length"]])

# terse print
print("paired_seeds_identical =", seeds_ok, "| n_seeds =", len(common))
print(f"{'model':24s} {'DK/64':>7s} {'SR%':>7s} {'Wilson95%':>16s} {'floor3':>7s} {'DK|f3':>7s} {'sewers':>7s} {'death':>6s} {'TO':>4s} {'lenMean':>8s}")
for name in ["session175_baseline", "original_ppo_24576", "p2_v1_full_24576"]:
    m = per_model[name]
    print(f"{name:24s} {m['DEFEAT_KOBOLD']:>3d}/64 {m['SR_pct']:>6.2f} "
          f"{str(m['wilson95_pct']):>16s} {m['floor3_reach']:>7d} "
          f"{str(m['conditional_kill_DK_given_floor3']):>7s} {m['ENTER_SEWERS']:>7d} "
          f"{m['death']:>6d} {m['timeout']:>4d} {m['ep_len_mean']:>8.1f}")
print("--- paired McNemar (a_only_gained / b_only_gained / both / chi2 / z / p) ---")
for k, v in pairs.items():
    print(f"  {k:28s} a={v['a_only_gained']:2d} b={v['b_only_gained']:2d} both={v['both_success']:2d} "
          f"chi2={v['mcnemar_chi2']:.3f} z={v['z']:.3f} p={v['p_two_sided']:.4f}")
print("EVIDENCE_JSON =", os.path.join(ev, "p2_v1_three_model_unified_eval.json"))
print("PER_SEED_CSV  =", csv_path)
