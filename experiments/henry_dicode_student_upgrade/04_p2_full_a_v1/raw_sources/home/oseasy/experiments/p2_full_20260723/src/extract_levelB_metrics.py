"""Compact Gate-B metric extractor for the Level B 24576 run. Reads training_log.jsonl
and prints per-update replay/KL/ESS/ratio/entropy/grad/lag stats + conservation +
hindsight counters, plus min/max/mean aggregates. Read-only. No logs dumped verbatim.
"""
import sys, json, os


def main():
    log_path = sys.argv[1]
    recs = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))

    upd_keys = ["policy_kl", "ess", "ratio_max", "entropy", "grad_norm_pre_clip",
                "chosen_actor_step_scale", "loss", "policy_committed",
                "kl_rejected_update", "policy_lag", "max_policy_lag",
                "vtrace_actor", "vtrace_value", "awr_actor", "awr_value"]
    rows = []
    conservation = []
    hindsight = []
    eps_stat = []
    for r in recs:
        if r.get("event") != "rollout":
            continue
        u = r.get("update")
        conservation.append(bool(r.get("conservation_ok", True)))
        if isinstance(r.get("hindsight"), dict):
            hindsight.append(r["hindsight"])
        eps_stat.append((r.get("global_step"), r.get("completed_episodes"),
                         r.get("mean_ep_length"), r.get("pending_transitions"),
                         r.get("pending_anchors"), r.get("replay_size")))
        if not u:
            continue
        row = {"rollout": r.get("rollout"), "global_step": r.get("global_step")}
        for k in upd_keys:
            if k in u:
                row[k] = u[k]
        rows.append(row)

    print("=== per-update Gate-B metrics ===")
    for row in rows:
        compact = {k: (round(v, 5) if isinstance(v, float) else v)
                   for k, v in row.items()}
        print(json.dumps(compact, sort_keys=True))

    def agg(key, fn):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return fn(vals) if vals else None

    print("=== aggregates over %d fired updates ===" % len(rows))
    import math
    summary = {
        "n_fired_updates": len(rows),
        "policy_kl_max": agg("policy_kl", max),
        "policy_kl_mean": agg("policy_kl", lambda v: sum(v) / len(v)),
        "ess_min": agg("ess", min),
        "ratio_max_max": agg("ratio_max", max),
        "entropy_min": agg("entropy", min),
        "entropy_mean": agg("entropy", lambda v: sum(v) / len(v)),
        "grad_norm_pre_clip_max": agg("grad_norm_pre_clip", max),
        "all_policy_committed": all(r.get("policy_committed") for r in rows),
        "any_kl_rejected": any(r.get("kl_rejected_update") for r in rows),
        "all_finite": all(r.get("loss") is not None and math.isfinite(r["loss"])
                          for r in rows),
        "policy_lag_present": [r.get("policy_lag", r.get("max_policy_lag"))
                               for r in rows],
    }
    print(json.dumps(summary, sort_keys=True, default=str))

    print("=== conservation (every rollout) ===")
    print("all_conservation_ok=%s n_rollouts=%d" % (all(conservation), len(conservation)))

    if hindsight:
        h_attempts = sum(h.get("hindsight_attempts", h.get("attempts", 0)) for h in hindsight)
        h_eligible = sum(h.get("hindsight_eligible", h.get("eligible", 0)) for h in hindsight)
        h_accepted = sum(h.get("hindsight_accepted", h.get("accepted", 0)) for h in hindsight)
        print("=== hindsight aggregates ===")
        print(json.dumps({"sum_attempts": h_attempts, "sum_eligible": h_eligible,
                          "sum_accepted": h_accepted, "per_rollout": hindsight},
                         sort_keys=True, default=str))

    print("=== episode/pending per rollout (step, completed, mean_len, pending_t, pending_a, replay) ===")
    for e in eps_stat:
        print(e)


if __name__ == "__main__":
    main()
