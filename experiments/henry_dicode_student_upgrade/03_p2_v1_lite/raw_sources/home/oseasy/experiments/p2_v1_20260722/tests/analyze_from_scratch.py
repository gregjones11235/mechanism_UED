#!/usr/bin/env python3
"""Unified from-scratch fair-comparison Stage4-native (S4_dark) 64-episode
learning-curve analysis (CPU only).

Models (same frozen evaluator protocol, parent SHA 06221187; same 64 world seeds,
seed 42; success = DEFEAT_KOBOLD ever-set, achievement read before terminal
auto-reset; stochastic policy; 4096 max steps):
  - step0_common_init : the shared random start (params SHA e78426c8), evaluated once
  - original_ppo @ 24576/49152/73728/98304 (pure Henry native PPO, replay/hindsight OFF)
  - p2_v1_full   @ 24576/49152/73728/98304 (Plan B + hindsight + isolated critic-only replay-aux)

Both groups loaded the SAME bit-exact random params at step0 and were trained with
identical PPO hyperparams / gamma=0.999 / gae=0.8 / num_envs=16 / rollout=128 / seed0.
Only difference: replay+hindsight ON (P2) vs OFF (Original PPO).

Per model: DK SR + Wilson95 CI, floor3 reach, conditional kill (DK|floor3),
ENTER_SEWERS, death, timeout, ep-length stats, max_floor dist.
Learning-curve table (Step | OP SR | P2 SR). Paired McNemar: OP vs P2 at each
matched step, and each model vs the common step0. Writes evidence JSON + per-seed
CSV + learning-curve CSV. seed0 is engineering screening ONLY (no single-seed
significance claim).
"""
import json, os, math, csv
import numpy as np

E = "/home/oseasy/experiments"
D = "20260723"
STEPS = [0, 24576, 49152, 73728, 98304]
def outdir(name): return f"{E}/fs_{name}_64ep_stage4_eval_{D}/results/STAGE4_NATIVE_episodes.jsonl"
NAMES = ["step0_common_init"] + [f"original_ppo_{s}" for s in (24576,49152,73728,98304)] \
                              + [f"p2_v1_full_{s}" for s in (24576,49152,73728,98304)]
FILES = {name: outdir(name) for name in NAMES}
EVAL_SHA = {
  "step0_common_init": "39add2fe9fb4e02aeb56911a60b8945710e82424ecc6c0ada0a1ed236335931e",
  "original_ppo_24576": "605436422dc2dd944c7a968abe6b9b62920ecf74d9be474f423c0a95f9ca7b91",
  "original_ppo_49152": "5b38a564dab054f77419ef249ed335a95c477402ceb61df26f0d805dec399973",
  "original_ppo_73728": "5503d2b3f7f3cbff973406d30c888490c62516797cdde9e29a9d67ff65ce0067",
  "original_ppo_98304": "fe6bcdc79336d8f83b2eff41a6d0f241701fddab19c9005557965aa072b5be0e",
  "p2_v1_full_24576": "7a4af0872320f626859fc24f0e9393110eae4206013c8f52b30999536c925e7b",
  "p2_v1_full_49152": "4dfed7e751a3b8e444402f0ce75e212e100e29e5950259f953652bd050e8bef9",
  "p2_v1_full_73728": "506fe1ded3c9932a0042d12c24a6891c8acf4613b527f74168e082b0036e0609",
  "p2_v1_full_98304": "a23293ad2ea4e6dbb6e75c4353e2f45246c81fbdd5cbc9f8d04e8bb1f4b54de2",
}

def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return {int(r["episode_idx"]): r for r in rows}
def truthy(v): return bool(v) if not isinstance(v, str) else v.lower() in ("true", "1", "yes")
def wilson(k, n, z=1.959963985):
    if n == 0: return (0.0, 0.0)
    p = k/n; den = 1 + z*z/n
    c = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return ((p + z*z/(2*n) - c)/den, (p + z*z/(2*n) + c)/den)

data = {name: load(p) for name, p in FILES.items()}
assert all(len(d) == 64 for d in data.values()), "each model must have 64 episodes"
seedsets = {name: set(d.keys()) for name, d in data.items()}
ref = seedsets["step0_common_init"]
seeds_ok = all(s == ref for s in seedsets.values())
seeds = sorted(ref)

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
    aonly = sum(1 for s in seeds if succ(a, s) and not succ(b, s))
    bonly = sum(1 for s in seeds if not succ(a, s) and succ(b, s))
    both = sum(1 for s in seeds if succ(a, s) and succ(b, s))
    neither = sum(1 for s in seeds if not succ(a, s) and not succ(b, s))
    chi2 = ((aonly-bonly)**2/(aonly+bonly)) if (aonly+bonly) > 0 else 0.0
    z = ((aonly-bonly)/math.sqrt(aonly+bonly)) if (aonly+bonly) > 0 else 0.0
    p = math.erfc(abs(z)/math.sqrt(2))
    return {"a_only_gained": aonly, "b_only_gained": bonly, "both_success": both,
            "neither": neither, "mcnemar_chi2": round(chi2, 4), "z": round(z, 4),
            "p_two_sided": round(p, 4)}

# OP vs P2 at each matched step (a=OP, b=P2)
op_vs_p2 = {}
for s in (24576, 49152, 73728, 98304):
    op_vs_p2[f"step_{s}"] = mcnemar(data[f"original_ppo_{s}"], data[f"p2_v1_full_{s}"], seeds)
# each model vs common step0 (a=model, b=step0): positive a_only => gained over init
vs_step0 = {}
for name in NAMES:
    if name == "step0_common_init": continue
    vs_step0[name] = mcnemar(data[name], data["step0_common_init"], seeds)

# learning curve table
curve = []
for s in STEPS:
    if s == 0:
        op = p2 = per_model["step0_common_init"]
        row = {"step": 0, "source": "step0_common_init (shared)",
               "OP_SR_pct": op["SR_pct"], "OP_DK": op["DEFEAT_KOBOLD"], "OP_wilson95_pct": op["wilson95_pct"],
               "P2_SR_pct": p2["SR_pct"], "P2_DK": p2["DEFEAT_KOBOLD"], "P2_wilson95_pct": p2["wilson95_pct"]}
    else:
        op = per_model[f"original_ppo_{s}"]; p2 = per_model[f"p2_v1_full_{s}"]
        row = {"step": s,
               "OP_SR_pct": op["SR_pct"], "OP_DK": op["DEFEAT_KOBOLD"], "OP_wilson95_pct": op["wilson95_pct"],
               "P2_SR_pct": p2["SR_pct"], "P2_DK": p2["DEFEAT_KOBOLD"], "P2_wilson95_pct": p2["wilson95_pct"],
               "OP_vs_P2_mcnemar_p": op_vs_p2[f"step_{s}"]["p_two_sided"]}
    curve.append(row)

report = {"protocol": "Stage4-native S4_dark, 64 world seeds (seed 42), stochastic, 4096 max steps, "
                       "success=DEFEAT_KOBOLD ever-set, evaluator parent SHA 06221187",
          "design": "from-scratch fair comparison, seed0; common bit-exact random init (params SHA e78426c8); "
                    "identical PPO hyperparams/gamma0.999/gae0.8/num_envs16/rollout128; 98304 steps=48 updates; "
                    "only difference = replay+hindsight ON (P2) vs OFF (Original PPO)",
          "evaluator_sha": EVAL_SHA, "paired_seeds_identical": seeds_ok, "n_seeds": len(seeds),
          "learning_curve": curve, "per_model": per_model,
          "paired_mcnemar_OP_vs_P2": op_vs_p2, "paired_mcnemar_vs_step0": vs_step0,
          "interpretation_boundary": "seed0 is engineering screening only; NO single-seed significance claim.",
          "files": FILES}
ev = "/home/oseasy/experiments/single_director_20260722/evidence"; os.makedirs(ev, exist_ok=True)
with open(os.path.join(ev, "p2_v1_from_scratch_unified_eval.json"), "w") as f:
    json.dump(report, f, indent=2, sort_keys=True, default=str); f.write("\n")

# learning-curve CSV
lc_csv = os.path.join(ev, "p2_v1_from_scratch_learning_curve.csv")
with open(lc_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["step", "OP_DK", "OP_SR_pct", "OP_wilson95_lo", "OP_wilson95_hi",
                "P2_DK", "P2_SR_pct", "P2_wilson95_lo", "P2_wilson95_hi", "OP_vs_P2_mcnemar_p"])
    for r in curve:
        w.writerow([r["step"], r["OP_DK"], r["OP_SR_pct"], r["OP_wilson95_pct"][0], r["OP_wilson95_pct"][1],
                    r["P2_DK"], r["P2_SR_pct"], r["P2_wilson95_pct"][0], r["P2_wilson95_pct"][1],
                    r.get("OP_vs_P2_mcnemar_p", "")])
# per-seed CSV (DK / floor3 / len for all 9 models)
ps_csv = os.path.join(ev, "p2_v1_from_scratch_per_seed.csv")
with open(ps_csv, "w", newline="") as f:
    w = csv.writer(f)
    hdr = ["episode_idx"]
    for name in NAMES: hdr += [f"{name}_DK", f"{name}_floor3", f"{name}_len"]
    w.writerow(hdr)
    for s in seeds:
        row = [s]
        for name in NAMES:
            r = data[name][s]
            row += [int(truthy(r["DEFEAT_KOBOLD"])), int(truthy(r["floor3_reach"])), r["episode_length"]]
        w.writerow(row)

# terse print
print("paired_seeds_identical =", seeds_ok, "| n_seeds =", len(seeds))
print("\n===== LEARNING CURVE (Stage4-native S4_dark, DK SR) =====")
print(f"{'step':>7s} | {'OP DK/64':>9s} {'OP SR%':>7s} {'OP Wilson95%':>16s} | {'P2 DK/64':>9s} {'P2 SR%':>7s} {'P2 Wilson95%':>16s} | {'McNemar p':>9s}")
for r in curve:
    p = r.get("OP_vs_P2_mcnemar_p", "")
    print(f"{r['step']:>7d} | {r['OP_DK']:>5d}/64 {r['OP_SR_pct']:>6.2f} {str(r['OP_wilson95_pct']):>16s} | "
          f"{r['P2_DK']:>5d}/64 {r['P2_SR_pct']:>6.2f} {str(r['P2_wilson95_pct']):>16s} | {str(p):>9s}")
print("\n===== per-model detail =====")
print(f"{'model':24s} {'DK/64':>7s} {'SR%':>7s} {'Wilson95%':>16s} {'floor3':>7s} {'DK|f3':>6s} {'sewers':>7s} {'death':>6s} {'TO':>4s} {'lenMean':>8s}")
for name in NAMES:
    m = per_model[name]
    print(f"{name:24s} {m['DEFEAT_KOBOLD']:>3d}/64 {m['SR_pct']:>6.2f} {str(m['wilson95_pct']):>16s} "
          f"{m['floor3_reach']:>7d} {str(m['conditional_kill_DK_given_floor3']):>6s} {m['ENTER_SEWERS']:>7d} "
          f"{m['death']:>6d} {m['timeout']:>4d} {m['ep_len_mean']:>8.1f}")
print("\n===== McNemar OP vs P2 (a=OP gained / b=P2 gained / both / chi2 / z / p) =====")
for k, v in op_vs_p2.items():
    print(f"  {k:10s} OP_only={v['a_only_gained']:2d} P2_only={v['b_only_gained']:2d} both={v['both_success']:2d} "
          f"chi2={v['mcnemar_chi2']:.3f} z={v['z']:.3f} p={v['p_two_sided']:.4f}")
print("\n===== McNemar vs common step0 (a=model gained over init / b=step0 only / both / p) =====")
for k, v in vs_step0.items():
    print(f"  {k:24s} gained={v['a_only_gained']:2d} init_only={v['b_only_gained']:2d} both={v['both_success']:2d} p={v['p_two_sided']:.4f}")
print("\nEVIDENCE_JSON =", os.path.join(ev, "p2_v1_from_scratch_unified_eval.json"))
print("LEARNING_CURVE_CSV =", lc_csv)
print("PER_SEED_CSV =", ps_csv)
