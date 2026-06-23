#!/usr/bin/env python
"""auction(α) 消融 —— 看学习曲线前中期是否分得开(而非饱和末值)。

末值挤在 0.98-0.99 是饱和天花板。真正区分 curriculum 机制的是:
  (1) 前中期(爬升段)同一 update 下的 win rate
  (2) 整条曲线 AUC (面积 = 样本效率)
  (3) 达到某阈值(如 wr>=0.9)所需 update 数 (收敛速度)
"""
import math, csv, json
import wandb

PROJECT = "multi_robot_ued"
GROUPS = {
    "inf": "auction-abl-linf",
    "1.0": "auction-abl-l1_0",
    "3.0": "auction-abl-l3_0",
}
WR = "sampled-test-metrics.eval-sampled/overall_win_rate"


def mean_ci95(xs):
    n = len(xs)
    if n == 0: return float("nan"), float("nan"), 0
    m = sum(xs)/n
    if n == 1: return m, float("nan"), 1
    sd = math.sqrt(sum((x-m)**2 for x in xs)/(n-1))
    se = sd/math.sqrt(n)
    tt = {1:12.71,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262}
    t = tt.get(n-1,1.96)
    return m, t*se, n


def pull_curve(group):
    """返回 {seed: [(update, wr), ...]}（只保留 finished & 跑满的）。"""
    api = wandb.Api()
    runs = list(api.runs(PROJECT, filters={"group": group}))
    out = {}
    for r in runs:
        if r.state != "finished" or r.summary.get("update_count") != 2250:
            continue
        seed = r.config.get("SEED")
        pts = []
        for row in r.scan_history(keys=["update_count", WR]):
            u = row.get("update_count"); wr = row.get(WR)
            if u is not None and wr is not None:
                pts.append((u, wr))
        pts.sort()
        if pts:
            out[seed] = pts
    return out


def auc(pts):
    """梯形面积 / 总宽度 = 时间平均 win rate。"""
    if len(pts) < 2: return float("nan")
    area = 0.0
    for (u0, w0), (u1, w1) in zip(pts, pts[1:]):
        area += (u1-u0)*(w0+w1)/2
    return area/(pts[-1][0]-pts[0][0])


def reach(pts, thr):
    """首次达到 wr>=thr 的 update（线性插值）；没达到返回 None。"""
    for (u0, w0), (u1, w1) in zip(pts, pts[1:]):
        if w0 < thr <= w1:
            return u0 + (thr-w0)/(w1-w0)*(u1-u0)
    if pts and pts[0][1] >= thr:
        return pts[0][0]
    return None


def main():
    curves = {lab: pull_curve(grp) for lab, grp in GROUPS.items()}

    # --- 统一 update 网格(各档取并集的公共评估点) ---
    all_updates = sorted({u for c in curves.values() for s in c.values() for u, _ in s})
    # 对每档每 seed 用其自有评估点的最近值插到网格(评估点本就同频 EVAL_FREQ=50, 直接按 update 对齐)
    def at(pts, u):
        d = dict(pts)
        if u in d: return d[u]
        # 最近邻
        best = min(pts, key=lambda p: abs(p[0]-u))
        return best[1] if abs(best[0]-u) <= 50 else None

    print("="*92)
    print("auction 消融 学习曲线 —— 同 update 下三档 sampled_wr [mean±95CI]  (爬升段是否分得开)")
    print("="*92)
    print(f"{'update':>8} | " + " | ".join(f"{lab:^22}" for lab in GROUPS))
    print("-"*92)
    rows_csv = []
    for u in all_updates:
        cells = []
        rec = {"update": u}
        for lab in GROUPS:
            vals = [at(pts, u) for pts in curves[lab].values()]
            vals = [v for v in vals if v is not None]
            m, ci, n = mean_ci95(vals)
            rec[f"{lab}_mean"] = round(m, 4) if not math.isnan(m) else ""
            rec[f"{lab}_ci"] = round(ci, 4) if not math.isnan(ci) else ""
            rec[f"{lab}_n"] = n
            cells.append(f"{m:.3f}±{ci:.3f}(n{n})" if n else " "*16)
        rows_csv.append(rec)
        # 只打印部分行(前中期密、后期疏)避免刷屏
        if u <= 600 or u % 250 == 0:
            print(f"{u:>8} | " + " | ".join(f"{c:^22}" for c in cells))

    # --- AUC & 收敛速度 ---
    print("\n" + "="*92)
    print("整条曲线 AUC (时间平均 win rate, 越高=样本效率越好) + 达 wr>=0.9 所需 update")
    print("="*92)
    for lab in GROUPS:
        aucs = [auc(p) for p in curves[lab].values()]
        aucs = [a for a in aucs if not math.isnan(a)]
        reaches = [reach(p, 0.9) for p in curves[lab].values()]
        reaches = [r for r in reaches if r is not None]
        ma, ca, na = mean_ci95(aucs)
        mr, cr, nr = mean_ci95(reaches)
        print(f"  {lab:<5} AUC={ma:.4f}±{ca:.4f} (n{na})   reach0.9: {mr:.0f}±{cr:.0f} upd (n{nr}/{len(curves[lab])})")

    with open("auction_curve_grid.csv", "w", newline="") as f:
        if rows_csv:
            w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
            w.writeheader(); w.writerows(rows_csv)
    print("\n写出 auction_curve_grid.csv (逐 update 三档 mean/ci/n)")


if __name__ == "__main__":
    main()
