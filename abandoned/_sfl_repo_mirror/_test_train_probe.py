"""train_probe.py 测试（CPU）—— 饱和判定 + 相关轨迹 csv + 跨 epoch 总裁决。

模拟训练轨迹：早期 success_rate 散开（非饱和、信号有效）→ 末期贴 1.0（饱和、相关不可信）。
验证探针能①在饱和 epoch 打警告、②总裁决只采非饱和 epoch。

跑法（minimax env）： python _test_train_probe.py
"""
import os, tempfile
import numpy as np

from train_probe import log_orthogonality_step, summarize_trace, _saturation_flags

PASS, FAIL = "PASS", "FAIL"
results = []
def check(name, cond):
    results.append((name, bool(cond))); print(f"[{PASS if cond else FAIL}] {name}")


def make_probe_data(rng, M, T, d, p_dist):
    """合成一个 epoch 的 probe_data。p_dist: 'spread'(0.2~0.8) 或 'saturated'(贴1.0)。"""
    if p_dist == "spread":
        succ = rng.uniform(0.2, 0.8, M).astype(np.float32)
    else:  # saturated
        succ = np.clip(rng.normal(0.98, 0.02, M), 0, 1).astype(np.float32)
    ne = rng.integers(1, 6, M).astype(np.int32)
    dones = (rng.uniform(0, 1, (T, M)) < 0.1)
    dones[-1, :] = True
    dones = dones.astype(bool)
    adv = rng.normal(0, 1, (T, M)).astype(np.float32)
    hidden = rng.normal(0, 1, (T * M, d)).astype(np.float32)
    pli = np.tile(np.arange(M), T).astype(np.int64)
    return {"success_rate": succ, "num_episodes": ne, "dones": dones,
            "advantages": adv, "hidden_feats": hidden, "per_level_index": pli, "n_levels": M}


# ── 饱和判定单测 ──
rng = np.random.default_rng(0)
spread_p = rng.uniform(0.2, 0.8, 100).astype(np.float32)
sat_p = np.clip(rng.normal(0.99, 0.01, 100), 0, 1).astype(np.float32)
ne = np.ones(100, dtype=np.int32)
f_spread = _saturation_flags(spread_p, ne)
f_sat = _saturation_flags(sat_p, ne)
check("饱和判定: 散开分布 → not saturated", not f_spread["saturated"])
check("饱和判定: 贴1.0分布 → saturated", f_sat["saturated"])
check("饱和判定: 散开 p_std 明显 > 饱和 p_std", f_spread["p_std"] > f_sat["p_std"])

# ── 回归：双峰分布(jaxnav 实测)必须 NOT saturated（旧 frac_hi+frac_lo 判据会误报）──
# 一半 level success≈1、一半 success≈0：frac_hi+frac_lo≈1.0 但 p_std≈0.5 很分散，相关可信。
bimodal_p = np.concatenate([np.full(50, 0.98), np.full(50, 0.02)]).astype(np.float32)
ne100 = np.ones(100, dtype=np.int32)
f_bi = _saturation_flags(bimodal_p, ne100)
check("饱和判定: 双峰(一半易一半难) → NOT saturated（修复核心）", not f_bi["saturated"])
check("饱和判定: 双峰 frac_hi+frac_lo 高(>0.9) 但 p_std 大(>0.4)",
      f_bi["frac_saturated_high"] + f_bi["frac_saturated_low"] > 0.9 and f_bi["p_std"] > 0.4)
# 单极饱和(几乎全解开 success>0.95) → saturated
unimodal_hi = np.clip(np.random.default_rng(9).normal(0.99, 0.01, 100), 0, 1).astype(np.float32)
f_uni = _saturation_flags(unimodal_hi, ne100)
check("饱和判定: 单极(几乎全 success>0.95) → saturated", f_uni["saturated"])


# ── 模拟训练轨迹：3 个非饱和 epoch + 2 个饱和 epoch ──
out = tempfile.mkdtemp(prefix="trainprobe_test_")
M, T, d = 80, 24, 16
steps = [100, 200, 300, 400, 500]
dists = ["spread", "spread", "spread", "saturated", "saturated"]
sat_logged = []
for step, dist in zip(steps, dists):
    pd = make_probe_data(rng, M, T, d, dist)
    res = log_orthogonality_step(pd, step, out, threshold=0.5)
    sat_logged.append(res["saturation"]["saturated"])
    check(f"epoch step={step} ({dist}): wandb_metrics 含 max_abs_offdiag",
          "probe/max_abs_offdiag" in res["wandb_metrics"])

check("轨迹: 散开 epoch 标记非饱和", not any(sat_logged[:3]))
check("轨迹: 末期 epoch 标记饱和", all(sat_logged[3:]))

# csv 落了 5 行
trace = os.path.join(out, "orthogonality_trace.csv")
rows = list(open(trace, encoding="utf-8").read().strip().split("\n"))
check("轨迹 csv: 1 表头 + 5 数据行", len(rows) == 6)
check("轨迹 csv: 含 rho[ 列", "rho[" in rows[0])


# ── 总裁决：只采非饱和 epoch ──
summ = summarize_trace(out)
check("总裁决: n_valid_nonsaturated = 3", summ["n_valid_nonsaturated"] == 3)
check("总裁决: 总 epoch = 5", summ["n_epochs"] == 5)
check("总裁决: verdict 合法且非 inconclusive",
      summ["verdict"] in ("ready_pool_sufficient", "must_implement_expensive_signal"))
check("总裁决: avg_rho_nonsaturated 含 4 对", len(summ["avg_rho_nonsaturated"]) == 6)


# ── 边界：全饱和 → inconclusive ──
out2 = tempfile.mkdtemp(prefix="trainprobe_sat_")
for step in [100, 200]:
    log_orthogonality_step(make_probe_data(rng, M, T, d, "saturated"), step, out2)
summ2 = summarize_trace(out2)
check("全饱和 → verdict=inconclusive_all_saturated",
      summ2["verdict"] == "inconclusive_all_saturated")


print(f"\n[trace dir] {out}")
n_pass = sum(1 for _, ok in results if ok)
print(f"{'='*40}\n{n_pass}/{len(results)} 通过")
if n_pass != len(results):
    print("失败项:", [n for n, ok in results if not ok]); raise SystemExit(1)
print("全部通过 ✓")
