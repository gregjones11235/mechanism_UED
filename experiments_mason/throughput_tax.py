"""吞吐税假说验证 —— 从 run 日志切片统计"净入库任务数/1e8 步",对照 return 增速。

假说(gap 报告 §6 讨论项):14B 一次写对率低 → 单位算力的有效新任务吞吐低 → iron 族
地板铺不满 → 3.27 分终点 gap 的机制来源。若"净入库速率 ↔ return 增速"跨 run 强相关,
假说获得初步证据(进论文 discussion);不相关,teacher 侧连讨论价值都关账。

输入:pod 上 grep 出的日志切片(见 runbook),每行含 Starting Session / [Preflight] kept
X/Y。输出:每 session 净入库数 → 按 1e8 步窗口聚合 → 与同窗口官方 eval 增速并排。

用法:
    python throughput_tax.py <日志切片.txt> <eval JSON> <steps_per_session,默认13.1e6>
"""
import json
import re
import sys


def parse_slice(path):
    kept_per_session, cur = {}, None
    for line in open(path, errors="ignore"):
        m = re.search(r"Starting Session (\d+)", line)
        if m:
            cur = int(m.group(1)); kept_per_session.setdefault(cur, 0)
        m = re.search(r"\[Preflight\] kept (\d+)/(\d+)", line)
        if m and cur is not None:
            kept_per_session[cur] += int(m.group(1))
    return kept_per_session


def main():
    slice_path, eval_path = sys.argv[1], sys.argv[2]
    sps = float(sys.argv[3]) if len(sys.argv) > 3 else 13.1e6
    kept = parse_slice(slice_path)
    ev = json.load(open(eval_path))
    pts = sorted((int(k), v["mean_return"]) for k, v in ev.items() if v["mean_return"] > 5)

    print(f"{'窗口(e8步)':>10} | {'净入库任务':>8} | {'return 增速(分/1e8)':>16}")
    win = 1e8
    max_step = max(kept) * sps
    w = 0
    rows = []
    while w * win < max_step:
        lo, hi = w * win, (w + 1) * win
        n_kept = sum(v for s, v in kept.items() if lo <= s * sps < hi)
        # return slope across the window from eval points (linear through nearest points)
        inwin = [(u * 127208, r) for u, r in pts if lo <= u * 127208 < hi + win]
        slope = ((inwin[-1][1] - inwin[0][1]) / max((inwin[-1][0] - inwin[0][0]) / win, 1e-9)
                 if len(inwin) >= 2 else float("nan"))
        rows.append((n_kept, slope))
        print(f"{w:>10} | {n_kept:>8} | {slope:>16.2f}")
        w += 1
    xs = [a for a, b in rows if b == b]
    ys = [b for a, b in rows if b == b]
    if len(xs) >= 3:
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        vx = sum((a - mx) ** 2 for a in xs) ** 0.5
        vy = sum((b - my) ** 2 for b in ys) ** 0.5
        r = cov / (vx * vy + 1e-9)
        print(f"\nPearson r(净入库, 增速) = {r:.3f}  (n={len(xs)} 窗口)")
        print("解读: |r|>0.6 → 吞吐税假说获初步支持;跨 run(你 vs Alec 日志)同号更硬。")


if __name__ == "__main__":
    main()
