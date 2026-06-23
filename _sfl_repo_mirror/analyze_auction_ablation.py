#!/usr/bin/env python
"""整理 auction(α) 必要性消融结果 —— 方案B §6.1 / STAGE3 w 轴。

三档 AUCTION_LAMBDA:
  inf  -> argmax (single-winner)   group=auction-abl-linf   (9 seed: 0-5,7,8,9)
  1.0  -> fractional 三方混合       group=auction-abl-l1_0   (8 seed: 0-7)
  3.0  -> 近 uniform (退化对照)      group=auction-abl-l3_0   (8 seed: 0-7)

判据(§6.1):三档下游 win rate 的 CI 分得开(混合>argmax) -> auction 必要性坐实;
          CI 重叠 -> 诚实砍掉用简单 argmax。

主口径 = sampled-test-metrics/eval-sampled/overall_win_rate (100图采样测试集,
对齐 SFL 论文 baseline)。副口径 = singleton-test-metrics/eval/:overall_win_rate
(5张固定测试图)。
"""
import math
import csv
import wandb

PROJECT = "multi_robot_ued"
GROUPS = {
    "inf (argmax)":   "auction-abl-linf",
    "1.0 (mix)":      "auction-abl-l1_0",
    "3.0 (uniform)":  "auction-abl-l3_0",
}


def mean_ci95(xs):
    """返回 (mean, half_width_95ci, n)。用 t 近似(小样本)。"""
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = sum(xs) / n
    if n == 1:
        return m, float("nan"), 1
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    # t_{0.975, df} 近似表 (df=1..15 然后用 ~1.96)
    tt = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
          7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
          13: 2.160, 14: 2.145, 15: 2.131}
    t = tt.get(n - 1, 1.96)
    return m, t * se, n


def pull(group):
    api = wandb.Api()
    runs = list(api.runs(PROJECT, filters={"group": group}))
    rows = []
    for r in runs:
        if r.state != "finished":
            continue
        if r.summary.get("update_count") != 2250:
            continue
        seed = r.config.get("SEED")
        sm = r.summary.get("sampled-test-metrics", {}) or {}
        sg = r.summary.get("singleton-test-metrics", {}) or {}
        rows.append({
            "seed": seed,
            "sampled_wr": sm.get("eval-sampled/overall_win_rate"),
            "singleton_wr": sg.get("eval/:overall_win_rate"),
            "sampled_return": sm.get("eval-sampled/overall_mean_return"),
            "learnability_mean": r.summary.get("learnability_set_mean_score"),
            "run": r.name,
        })
    # 同 seed 去重(保留最新)，按 seed 排序
    by_seed = {}
    for row in rows:
        by_seed[row["seed"]] = row
    return sorted(by_seed.values(), key=lambda z: (z["seed"] if z["seed"] is not None else -1))


def main():
    all_rows = []
    summary = []
    for label, grp in GROUPS.items():
        rows = pull(grp)
        for row in rows:
            row["lambda"] = label
            all_rows.append(row)
        samp = [r["sampled_wr"] for r in rows if isinstance(r["sampled_wr"], (int, float))]
        sing = [r["singleton_wr"] for r in rows if isinstance(r["singleton_wr"], (int, float))]
        ms, cs, ns = mean_ci95(samp)
        mg, cg, ng = mean_ci95(sing)
        summary.append((label, ns, ms, cs, mg, cg, sorted(r["seed"] for r in rows)))

    # 逐 run CSV
    with open("auction_ablation_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["lambda", "seed", "sampled_wr",
                            "singleton_wr", "sampled_return", "learnability_mean", "run"])
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    # 摘要表
    print("\n" + "=" * 78)
    print("AUCTION (alpha) ABLATION — 下游 win rate 三档对比  [mean ± 95% CI]")
    print("=" * 78)
    print(f"{'lambda':<16}{'n':>3}  {'sampled_wr (100图主口径)':<28}{'singleton_wr (5图)':<24}")
    print("-" * 78)
    for label, n, ms, cs, mg, cg, seeds in summary:
        print(f"{label:<16}{n:>3}  {ms:.4f} ± {cs:.4f}{'':>10}{mg:.4f} ± {cg:.4f}")
    print("-" * 78)
    for label, n, ms, cs, mg, cg, seeds in summary:
        print(f"  {label:<14} seeds={seeds}")

    # 判据:三档 sampled_wr CI 是否分得开
    print("\n" + "=" * 78)
    print("判据(§6.1 auction 必要性)：sampled_wr 95%CI 区间")
    print("=" * 78)
    intervals = {}
    for label, n, ms, cs, mg, cg, seeds in summary:
        lo, hi = ms - cs, ms + cs
        intervals[label] = (lo, hi)
        print(f"  {label:<16} [{lo:.4f}, {hi:.4f}]")
    labs = list(intervals.keys())
    print("\n  两两 CI 重叠检查:")
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            a, b = labs[i], labs[j]
            la, ha = intervals[a]
            lb, hb = intervals[b]
            overlap = not (ha < lb or hb < la)
            print(f"    {a:<16} vs {b:<16}: {'重叠(差异不显著)' if overlap else '分得开(差异显著)'}")
    with open("auction_ablation_summary.txt", "w") as f:
        for label, n, ms, cs, mg, cg, seeds in summary:
            f.write(f"{label}\tn={n}\tsampled_wr={ms:.4f}±{cs:.4f}\tsingleton_wr={mg:.4f}±{cg:.4f}\tseeds={seeds}\n")
    print("\n写出: auction_ablation_runs.csv (逐run), auction_ablation_summary.txt (摘要)")


if __name__ == "__main__":
    main()
