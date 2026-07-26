#!/usr/bin/env python3
"""§14 revised-gate PAIRED 256-world analyzer (pure CPU, no GPU, no new deps).

Reads the two episode jsonls produced by eval_paired_256.py (baseline=ckpt17500,
control=lr_2e-5/24576) run on the SAME 256 fresh worlds (identical seed_base and
num_worlds -> world i is identical in both; common-random-number action streams).
Pairs by episode_idx and reports:
  - per-policy SR / floor3 / conditional kill P(DK|floor3) / death / timeout (+Wilson 95% CI)
  - SR drop (pp) and floor3 retention vs baseline
  - PAIRED 2x2 table (both / base-only / ctrl-only / neither) for DK and for floor3
  - McNemar exact two-sided p-value (binomial on discordant pairs)
  - paired SR-difference 95% CI (large-sample) and Wilson CIs
Acceptance flags (user-frozen revised gate, STAGE4 behavior part):
  c1: SR_drop <= 8pp     (SR_base - SR_ctrl <= 0.08)
  c2: floor3 retention >= 80% of baseline
Writes <out_dir>/paired_256_report.json. No checkpoint, no training, no GPU.
"""
import argparse, json, math, os

SR_DROP_MAX = 0.08
FLOOR3_MIN_FRAC = 0.80


def _load_episodes(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: int(r["episode_idx"]))
    return rows


def _wilson95(k, n):
    if n == 0:
        return (float("nan"), float("nan"))
    z = 1.959963984540054
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4.0 * n * n)) / denom
    return (center - half, center + half)


def _binom_cdf_le(k, n):
    """P(X <= k) for X ~ Binom(n, 0.5), exact (n small)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    p = 0.5 ** n
    s = 0.0
    for i in range(0, k + 1):
        s += math.comb(n, i)
    return s * p


def _mcnemar_exact_p(n10, n01):
    ndisc = n10 + n01
    if ndisc == 0:
        return 1.0
    k = min(n10, n01)
    return min(1.0, 2.0 * _binom_cdf_le(k, ndisc))


def _paired_diff_ci(n10, n01, N):
    """SR_ctrl - SR_base = (n01 - n10)/N, large-sample paired 95% CI."""
    diff = (n01 - n10) / N
    se2 = (n10 + n01) / (N * N) - ((n01 - n10) ** 2) / (N ** 3)
    se = math.sqrt(max(se2, 0.0))
    return diff, diff - 1.959963984540054 * se, diff + 1.959963984540054 * se


def _paired_table(base_flags, ctrl_flags):
    n11 = n10 = n01 = n00 = 0
    for b, c in zip(base_flags, ctrl_flags):
        if b and c:
            n11 += 1
        elif b and not c:
            n10 += 1
        elif (not b) and c:
            n01 += 1
        else:
            n00 += 1
    return dict(both=n11, base_only=n10, ctrl_only=n01, neither=n00)


def _policy_stats(rows):
    n = len(rows)
    dk = [bool(r["DEFEAT_KOBOLD"]) for r in rows]
    f3 = [bool(r["floor3_reach"]) for r in rows]
    death = [bool(r["death"]) for r in rows]
    timeout = [bool(r["timeout"]) for r in rows]
    sewers = [bool(r["ENTER_SEWERS"]) for r in rows]
    n_dk = sum(dk); n_f3 = sum(f3)
    n_f3_and_dk = sum(1 for a, b in zip(f3, dk) if a and b)
    wil_sr = _wilson95(n_dk, n)
    wil_f3 = _wilson95(n_f3, n)
    return dict(
        n=n, n_success=n_dk, SR=n_dk / n, SR_wilson95=list(wil_sr),
        n_floor3=n_f3, floor3_rate=n_f3 / n, floor3_wilson95=list(wil_f3),
        n_floor3_and_dk=n_f3_and_dk,
        conditional_kill_given_floor3=(n_f3_and_dk / n_f3) if n_f3 else float("nan"),
        n_death=sum(death), death_rate=sum(death) / n,
        n_timeout=sum(timeout), timeout_rate=sum(timeout) / n,
        n_sewers=sum(sewers), enter_sewers_rate=sum(sewers) / n,
        mean_episode_length=sum(float(r["episode_length"]) for r in rows) / n,
        _dk_flags=dk, _f3_flags=f3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_jsonl", required=True)
    ap.add_argument("--control_jsonl", required=True)
    ap.add_argument("--baseline_label", required=True)
    ap.add_argument("--control_label", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    base = _load_episodes(args.baseline_jsonl)
    ctrl = _load_episodes(args.control_jsonl)
    Nb, Nc = len(base), len(ctrl)
    assert Nb == Nc, f"paired eval needs equal worlds: baseline={Nb} control={Nc}"
    N = Nb
    sb = base[0]["seed_base"]; sc = ctrl[0]["seed_base"]
    assert sb == sc, f"seed_base mismatch (not paired): baseline={sb} control={sc}"
    # verify index alignment
    assert [int(r["episode_idx"]) for r in base] == [int(r["episode_idx"]) for r in ctrl]

    bs = _policy_stats(base)
    cs = _policy_stats(ctrl)

    # paired DK table
    dk_tbl = _paired_table(bs["_dk_flags"], cs["_dk_flags"])
    dk_diff, dk_lo, dk_hi = _paired_diff_ci(dk_tbl["base_only"], dk_tbl["ctrl_only"], N)
    dk_p = _mcnemar_exact_p(dk_tbl["base_only"], dk_tbl["ctrl_only"])
    # paired floor3 table
    f3_tbl = _paired_table(bs["_f3_flags"], cs["_f3_flags"])
    f3_diff, f3_lo, f3_hi = _paired_diff_ci(f3_tbl["base_only"], f3_tbl["ctrl_only"], N)
    f3_p = _mcnemar_exact_p(f3_tbl["base_only"], f3_tbl["ctrl_only"])

    sr_drop = bs["SR"] - cs["SR"]                      # positive => control worse
    floor3_retention = (cs["floor3_rate"] / bs["floor3_rate"]) if bs["floor3_rate"] > 0 else float("nan")

    c1 = sr_drop <= SR_DROP_MAX + 1e-12
    c2 = (cs["floor3_rate"] >= FLOOR3_MIN_FRAC * bs["floor3_rate"]) if bs["floor3_rate"] > 0 else True

    for d in (bs, cs):
        d.pop("_dk_flags", None); d.pop("_f3_flags", None)

    report = dict(
        protocol="§14 revised-gate paired 256-world Stage4_native comparison",
        paired=True, common_random_numbers=True, seed_base=sb, num_worlds=N,
        baseline_label=args.baseline_label, control_label=args.control_label,
        baseline_jsonl=args.baseline_jsonl, control_jsonl=args.control_jsonl,
        baseline=bs, control=cs,
        sr_drop_pp=sr_drop * 100.0, sr_drop=sr_drop,
        floor3_retention=floor3_retention, floor3_retention_pct=floor3_retention * 100.0,
        sr_drop_max_pp=SR_DROP_MAX * 100.0, floor3_min_frac=FLOOR3_MIN_FRAC,
        paired_DK=dk_tbl,
        paired_SR_diff_ctrl_minus_base=dk_diff,
        paired_SR_diff_95ci=[dk_lo, dk_hi],
        mcnemar_DK_exact_p=dk_p,
        paired_floor3=f3_tbl,
        paired_floor3_diff_ctrl_minus_base=f3_diff,
        paired_floor3_diff_95ci=[f3_lo, f3_hi],
        mcnemar_floor3_exact_p=f3_p,
        gate_c1_sr_drop_le_8pp=bool(c1),
        gate_c2_floor3_ge_80pct=bool(c2),
        behavioral_gates_pass=bool(c1 and c2))
    out = os.path.join(args.out_dir, "paired_256_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=str)
        f.write("\n")

    print("=" * 72)
    print(f"PAIRED {N}-world Stage4_native  (seed_base={sb})")
    print(f"  Baseline [{args.baseline_label}]: SR={bs['SR']*100:.2f}% "
          f"(Wilson95 {bs['SR_wilson95'][0]*100:.2f}-{bs['SR_wilson95'][1]*100:.2f})  "
          f"floor3={bs['floor3_rate']*100:.2f}%  cond_kill|f3={bs['conditional_kill_given_floor3']:.3f}  "
          f"death={bs['death_rate']*100:.2f}%  timeout={bs['timeout_rate']*100:.2f}%")
    print(f"  Control  [{args.control_label}]: SR={cs['SR']*100:.2f}% "
          f"(Wilson95 {cs['SR_wilson95'][0]*100:.2f}-{cs['SR_wilson95'][1]*100:.2f})  "
          f"floor3={cs['floor3_rate']*100:.2f}%  cond_kill|f3={cs['conditional_kill_given_floor3']:.3f}  "
          f"death={cs['death_rate']*100:.2f}%  timeout={cs['timeout_rate']*100:.2f}%")
    print(f"  SR_drop = {sr_drop*100:+.2f}pp  (<= {SR_DROP_MAX*100:.0f}pp ? {'OK' if c1 else 'FAIL'})")
    print(f"  floor3 retention = {floor3_retention*100:.1f}% of baseline  "
          f"(>= {FLOOR3_MIN_FRAC*100:.0f}% ? {'OK' if c2 else 'FAIL'})")
    print(f"  paired DK table: both={dk_tbl['both']} base_only={dk_tbl['base_only']} "
          f"ctrl_only={dk_tbl['ctrl_only']} neither={dk_tbl['neither']}")
    print(f"  McNemar DK exact p={dk_p:.4g}  paired SR diff(ctrl-base)={dk_diff*100:+.2f}pp "
          f"95CI [{dk_lo*100:+.2f},{dk_hi*100:+.2f}]")
    print(f"  paired floor3 table: both={f3_tbl['both']} base_only={f3_tbl['base_only']} "
          f"ctrl_only={f3_tbl['ctrl_only']} neither={f3_tbl['neither']}  McNemar p={f3_p:.4g}")
    print(f"  BEHAVIORAL GATES (c1 SR-drop, c2 floor3): {'PASS' if (c1 and c2) else 'FAIL'}")
    print(f"  report -> {out}")
    print("=" * 72)


if __name__ == "__main__":
    main()
