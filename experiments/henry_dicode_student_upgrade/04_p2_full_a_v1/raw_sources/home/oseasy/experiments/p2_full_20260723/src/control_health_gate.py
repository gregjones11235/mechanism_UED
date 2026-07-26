#!/usr/bin/env python3
"""§14 Control health-gate analyzer + LR selection rule (frozen).

Reads, per grid LR (2e-4 > 6e-5 > 2e-5):
  - training manifest  <grid_root>/lr_<lr>/control_manifest.json
      -> params_finite, restore_roundtrip_ok, params_advanced, trained_params_sha256
  - eval combined json <grid_root>/eval_<lr>/results/control_eval_combined.json
      -> STAGE4_NATIVE: SR, floor3_reach_rate, cumulative_policy_kl_mean
and the Baseline (session175 / ckpt17500, untrained) 64-world Stage4-native jsonl
to obtain the reference SR and floor3 reach.

§14 health gate (ALL must hold for an LR to PASS):
  1. SR drop vs Baseline        <= 8pp      (baseline_SR - control_SR <= 0.08)
  2. floor3 reach               >= 80% x Baseline floor3
  3. no NaN/Inf                            (params_finite & roundtrip & finite SR/KL)
  4. cumulative policy KL       < KL_MAX_RUN = 0.1
Selection rule: among passing LRs pick the HIGHEST LR. All fail ->
CONTROL_PROTOCOL_UNHEALTHY (stop; do not proceed to formal comparison).

Writes <grid_root>/control_health_gate_report.json and prints the decision.
Pure post-processing; touches no GPU and no checkpoint.
"""
import argparse, json, os

GRID_LRS = ["2e-4", "6e-5", "2e-5"]            # highest -> lowest
LR_FLOAT = {"2e-4": 2e-4, "6e-5": 6e-5, "2e-5": 2e-5}
KL_MAX_RUN = 0.1                                 # §14 frozen cumulative ceiling
SR_DROP_MAX = 0.08                               # 8 percentage points
FLOOR3_MIN_FRAC = 0.80                           # >= 80% of Baseline


def _load(p):
    with open(p) as f:
        return json.load(f)


def baseline_stats(jsonl_path):
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    n = len(rows)
    assert n == 64, f"baseline must have 64 episodes, got {n}"
    sr = sum(1 for r in rows if bool(r.get("DEFEAT_KOBOLD"))) / n
    f3 = sum(1 for r in rows if bool(r.get("floor3_reach"))) / n
    return dict(n=n, SR=sr, floor3_reach_rate=f3,
                n_success=int(sum(1 for r in rows if bool(r.get("DEFEAT_KOBOLD")))),
                n_floor3=int(sum(1 for r in rows if bool(r.get("floor3_reach")))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid_root", required=True)
    ap.add_argument("--baseline_jsonl", default=(
        "/home/oseasy/experiments/session175_dual_caliber_pilot_20260722/"
        "results/STAGE4_NATIVE_episodes.jsonl"))
    args = ap.parse_args()

    base = baseline_stats(args.baseline_jsonl)
    print(f"[gate] Baseline (session175/ckpt17500): SR={base['SR']*100:.2f}% "
          f"({base['n_success']}/64)  floor3={base['floor3_reach_rate']*100:.2f}% "
          f"({base['n_floor3']}/64)")

    per_lr = {}
    for lr in GRID_LRS:
        mf_path = os.path.join(args.grid_root, f"lr_{lr}", "control_manifest.json")
        ev_path = os.path.join(args.grid_root, f"eval_{lr}", "results",
                               "control_eval_combined.json")
        entry = dict(lr=lr, manifest=os.path.exists(mf_path),
                     eval=os.path.exists(ev_path))
        if not (entry["manifest"] and entry["eval"]):
            entry["error"] = "missing manifest and/or eval output"
            per_lr[lr] = entry
            print(f"[gate] lr={lr}: MISSING artifacts -> treated as FAIL")
            continue
        mf = _load(mf_path)
        ev = _load(ev_path)
        s4 = ev["STAGE4_NATIVE"]
        sr = float(s4["SR"]); f3 = float(s4["floor3_reach_rate"])
        kl = float(s4["cumulative_policy_kl_mean"])
        finite = bool(mf.get("params_finite")) and bool(mf.get("restore_roundtrip_ok"))
        advanced = bool(mf.get("params_advanced"))
        sr_drop = base["SR"] - sr
        f3_ok = (f3 >= FLOOR3_MIN_FRAC * base["floor3_reach_rate"]) if base["floor3_reach_rate"] > 0 else True
        c1 = sr_drop <= SR_DROP_MAX + 1e-12
        c2 = f3_ok
        c3 = finite and advanced and (sr == sr) and (kl == kl)   # finite & not NaN
        c4 = kl < KL_MAX_RUN
        passed = c1 and c2 and c3 and c4
        entry.update(dict(
            control_SR=sr, control_floor3=f3, cumulative_policy_kl=kl,
            params_finite=bool(mf.get("params_finite")),
            restore_roundtrip_ok=bool(mf.get("restore_roundtrip_ok")),
            params_advanced=advanced, trained_params_sha256=mf.get("trained_params_sha256"),
            source_checkpoint_sha256=mf.get("source_checkpoint_sha256"),
            sr_drop_vs_baseline=sr_drop, floor3_threshold=FLOOR3_MIN_FRAC * base["floor3_reach_rate"],
            gate_1_sr_drop_le_8pp=bool(c1), gate_2_floor3_ge_80pct=bool(c2),
            gate_3_no_nan_inf=bool(c3), gate_4_kl_lt_0p1=bool(c4),
            passed=bool(passed)))
        per_lr[lr] = entry
        print(f"[gate] lr={lr}: SR={sr*100:.2f}% (drop={sr_drop*100:+.2f}pp, "
              f"{'OK' if c1 else 'FAIL'})  floor3={f3*100:.2f}% "
              f"({'OK' if c2 else 'FAIL'})  finite={'OK' if c3 else 'FAIL'}  "
              f"KL={kl:.5f} ({'OK' if c4 else 'FAIL'})  => {'PASS' if passed else 'FAIL'}")

    passing = [lr for lr in GRID_LRS if per_lr[lr].get("passed")]
    # GRID_LRS is highest->lowest, so the first passing is the highest passing LR
    selected = passing[0] if passing else None
    healthy = selected is not None
    decision = {
        "protocol": "frozen §14 Control health gate + LR selection",
        "baseline": base, "baseline_jsonl": args.baseline_jsonl,
        "kl_max_run": KL_MAX_RUN, "sr_drop_max": SR_DROP_MAX,
        "floor3_min_frac": FLOOR3_MIN_FRAC, "grid_lrs": GRID_LRS,
        "per_lr": per_lr, "passing_lrs": passing,
        "selected_lr": (LR_FLOAT[selected] if selected else None),
        "selected_lr_str": selected,
        "control_healthy": healthy,
        "decision": ("CONTROL_HEALTHY" if healthy else "CONTROL_PROTOCOL_UNHEALTHY"),
        "next_action": (f"re-run 4096 Full P2 smoke at LR={LR_FLOAT[selected]} (item 8)"
                        if healthy else
                        "STOP: do not proceed to 98304 / formal 512 eval; report BLOCKED"),
    }
    out_path = os.path.join(args.grid_root, "control_health_gate_report.json")
    with open(out_path, "w") as f:
        json.dump(decision, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    print("\n" + "=" * 72)
    print(f"DECISION: {decision['decision']}  selected_lr={selected}  "
          f"passing={passing}")
    print(f"report -> {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
