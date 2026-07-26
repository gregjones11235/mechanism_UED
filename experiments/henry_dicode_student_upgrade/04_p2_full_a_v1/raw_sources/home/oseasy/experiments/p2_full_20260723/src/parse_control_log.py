#!/usr/bin/env python3
"""Parse §14 Control training log -> per-update trajectories (read-only).

Reads <grid_root>/lr_<lr>/control_log.jsonl (written by run_control_grid.py, which
captured every numeric 1-D array returned in run_training_session()['metrics']).
Reports, per metric key: n_updates, mean/max/min, full per-update values. If a
per-update KL key exists (approx_kl / kl / policy_kl / kl_div), also reports
mean_per_update_kl, max_per_update_kl, and cumulative_kl_from_baseline_proxy =
SUM of per-update KLs (a drift proxy; NOT the rollout KL). Writes parsed json.
Pure post-processing; no GPU, no checkpoint, no retraining.
"""
import argparse, json, os, statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid_root", required=True)
    ap.add_argument("--lr", required=True)
    args = ap.parse_args()
    log_path = os.path.join(args.grid_root, f"lr_{args.lr}", "control_log.jsonl")

    events = {}
    metrics = {}            # key -> per-update float list
    init_rec = None
    train_done = None
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ev = r.get("event")
            events[ev] = events.get(ev, 0) + 1
            if ev == "train_metric":
                metrics[r.get("key")] = [float(x) for x in r.get("values", [])]
            elif ev == "init":
                init_rec = r
            elif ev == "train_done":
                train_done = r

    print(f"[parse] lr={args.lr} log={log_path}")
    print(f"[parse] event_counts={events}")
    print(f"[parse] metric_keys={sorted(metrics.keys())}")

    parsed = {"lr": args.lr, "log": log_path, "event_counts": events,
              "init": init_rec, "train_done": train_done, "metrics": {}}
    for k, vals in sorted(metrics.items()):
        n = len(vals)
        summ = {"n": n, "values": vals}
        if n:
            summ.update(mean=statistics.mean(vals), max=max(vals),
                        min=min(vals), first=vals[0], last=vals[-1])
        parsed["metrics"][k] = summ
        print(f"[parse]   {k}: n={n} mean={summ.get('mean')} max={summ.get('max')} "
              f"min={summ.get('min')}")
        print(f"[parse]     values={[round(v, 6) for v in vals]}")

    kl_key = None
    for cand in ("approx_kl", "kl", "policy_kl", "kl_div", "kl_loss"):
        if cand in metrics:
            kl_key = cand
            break
    if kl_key:
        v = metrics[kl_key]
        parsed["per_update_kl_key"] = kl_key
        parsed["mean_per_update_kl"] = statistics.mean(v)
        parsed["max_per_update_kl"] = max(v)
        parsed["min_per_update_kl"] = min(v)
        parsed["cumulative_kl_from_baseline_proxy"] = sum(v)
        print(f"[parse] per-update KL key={kl_key}: mean={parsed['mean_per_update_kl']:.6f} "
              f"max={parsed['max_per_update_kl']:.6f} min={parsed['min_per_update_kl']:.6f} "
              f"cumulative_proxy(sum)={parsed['cumulative_kl_from_baseline_proxy']:.6f}")
    else:
        parsed["per_update_kl_key"] = None
        print("[parse] WARNING: no per-update KL key in training metrics; "
              "cannot derive per-update KL from this log")

    out = os.path.join(args.grid_root, f"lr_{args.lr}", "control_log_parsed.json")
    with open(out, "w") as f:
        json.dump(parsed, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    print(f"[parse] wrote {out}")


if __name__ == "__main__":
    main()
