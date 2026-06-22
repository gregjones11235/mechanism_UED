"""离线重判：用修正后的饱和判据(p_std<0.05 或 frac_hi>0.95)重算 orthogonality_trace.csv 的
非饱和 epoch + 总裁决。**不需重训**——csv 已存每 epoch 的 raw 相关 + p_std + frac_hi/lo。

背景：旧判据 (frac_hi+frac_lo>0.8) 把 jaxnav 的双峰分布(p_std≈0.45 很分散)误报为饱和，
导致 n_valid_nonsaturated 只剩 2-3。新判据只认 p_std 低/单极高才算饱和，双峰算可信。

用法： python rejudge_trace.py <trace.csv> [trace2.csv ...]
"""
import sys
import csv
import numpy as np


def rejudge_saturated(p_std, frac_hi):
    """新判据：p_std<0.05(几乎恒定) 或 frac_hi>0.95(单极全解开) → 饱和。双峰不算。"""
    return (p_std < 0.05) or (frac_hi > 0.95)


def rejudge_file(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        return {"path": path, "error": "empty"}
    pair_cols = [c for c in rows[0] if c.startswith("rho[")]
    valid = []
    for r in rows:
        p_std = float(r["p_std"]); frac_hi = float(r["frac_sat_hi"])
        sat_new = rejudge_saturated(p_std, frac_hi)
        if not sat_new:
            valid.append(r)
    n_ready = sum(1 for r in valid if r["verdict"] == "ready_pool_sufficient")
    n_must = len(valid) - n_ready
    if not valid:
        verdict = "inconclusive_all_saturated"
    else:
        verdict = "ready_pool_sufficient" if n_ready >= n_must else "must_implement_expensive_signal"
    avg_rho = {}
    if valid:
        avg_rho = {c: round(float(np.mean([float(r[c]) for r in valid])), 4) for c in pair_cols}
    # 也报非饱和 epoch 的 update_count 范围（看覆盖训练哪一段）
    valid_steps = [int(r["update_count"]) for r in valid]
    return {
        "path": path,
        "n_epochs": len(rows),
        "n_valid_nonsaturated_NEW": len(valid),
        "n_ready": n_ready, "n_must_implement": n_must,
        "verdict_NEW": verdict,
        "nonsaturated_step_range": (min(valid_steps), max(valid_steps)) if valid_steps else None,
        "avg_rho_nonsaturated_NEW": avg_rho,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python rejudge_trace.py <trace.csv> [...]"); sys.exit(1)
    for path in sys.argv[1:]:
        rep = rejudge_file(path)
        print("=" * 70)
        for k, v in rep.items():
            print(f"{k}: {v}")
