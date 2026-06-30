# -*- coding: utf-8 -*-
"""
把 measure_sigma_on_ckpt.py 在 Oscar 上提取的 per-seed S 矩阵（meplr-s{0,1,..}.npy）
喂给 sigma_measure.py 的判据层（纯 numpy），输出：
  1. 每个 seed 各自的 Σ / ρ / T3 / T4 判定
  2. 跨 seed 的 ρ 稳定性（复用 h_drift_check）——这是"单 seed 不够算 Σ"的核心校验：
     若某对 estimator 的 ρ 在 seed 间极差 < tol，则该对的冗余结论稳健（不随 seed 翻转）。

用法：python run_sigma_on_seeds.py  [sigma_S 目录，默认 ./sigma_S]
"""
import sys, os, json
import numpy as np
import sigma_measure as sm

d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "sigma_S")

# 收集所有 meplr-s*.npy
import glob
npys = sorted(glob.glob(os.path.join(d, "meplr-s*.npy")))
if not npys:
    print(f"在 {d} 下没找到 meplr-s*.npy"); sys.exit(1)

seeds = []
for p in npys:
    seed = os.path.basename(p).replace("meplr-s", "").replace(".npy", "")
    S = np.load(p)
    names_p = p.replace(".npy", ".names.json")
    names = json.load(open(names_p)) if os.path.exists(names_p) else ["PVL", "SFL", "CENIE"]
    seeds.append((seed, S, names))

print("=" * 78)
print(f"找到 {len(seeds)} 个 seed 的 S 矩阵：", [s[0] for s in seeds])
print("=" * 78)

rho_list = []
names0 = seeds[0][2]
for seed, S, names in seeds:
    print(f"\n{'#'*78}\n# SEED {seed}    S.shape = {S.shape}   行序 = {names}\n{'#'*78}")
    Sigma, report = sm.measure_from_S(S, names, alpha=1.0)
    sm.print_report(Sigma, report, names)
    rho_list.append(report['rho'])

# ---- 跨 seed 稳定性：ρ 结构是否随 seed 翻转 ----
if len(rho_list) >= 2:
    print(f"\n{'='*78}\n[跨 seed 稳定性] 每对 estimator 的 ρ 在 {len(rho_list)} 个 seed 间是否稳定")
    print("（tol=0.15：极差 < tol → 该对冗余结论不随 seed 翻转，单 seed 也可信）")
    print("=" * 78)
    drift = sm.h_drift_check(rho_list, names0, tol=0.15)
    for (j, k), dd in drift.items():
        flag = "✓稳定" if dd['stable'] else "✗翻转风险(需更多 seed/n)"
        print(f"  ρ({j:5s},{k:5s}): 各 seed={dd['values']}  极差={dd['range']:.3f}  {flag}")
