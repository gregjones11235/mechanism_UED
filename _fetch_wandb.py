import os, re, json, sys
import wandb

# read key from .env (wandb_API=...)
key = None
with open(".env", encoding="utf-8") as f:
    for line in f:
        m = re.match(r"\s*wandb_API\s*=\s*(\S+)", line)
        if m:
            key = m.group(1); break
assert key, "no wandb_API in .env"
os.environ["WANDB_API_KEY"] = key

api = wandb.Api(timeout=60)
ENTITY = "gregjones11235-brown-university"
# (project, run_id, label)
TARGETS = [
    ("DiCode-repro",   "dicode-repro-s0-v1",   "baseline"),
    ("DiCode-auction", "dicode-auctionC-s0-v1","carm"),
]

def try_get_run(project, rid):
    # first try direct path
    try:
        return api.run(f"{ENTITY}/{project}/{rid}")
    except Exception as e:
        print(f"  direct path failed ({e}); searching by name/displayName...")
    for r in api.runs(f"{ENTITY}/{project}"):
        if r.id == rid or r.name == rid or getattr(r, "display_name", "") == rid:
            return r
    return None

summary = {}
for project, rid, label in TARGETS:
    print(f"\n=== {label}: {ENTITY}/{project}/{rid} ===")
    run = try_get_run(project, rid)
    if run is None:
        print("  RUN NOT FOUND");
        # list available runs to help
        try:
            print("  available runs in project:")
            for r in api.runs(f"{ENTITY}/{project}"):
                print("   -", r.id, "|", r.name, "|", r.state)
        except Exception as e:
            print("   (could not list:", e, ")")
        continue
    print(f"  found run id={run.id} name={run.name} state={run.state}")
    # pull history for eval metric keys; keys may be 'evaluation/mean_return'
    keys = ["evaluation/mean_return", "evaluation/mean_performance",
            "global_env_steps", "global_update_step", "session"]
    rows = []
    try:
        hist = run.history(keys=keys, pandas=False, samples=100000)
        for h in hist:
            rows.append(h)
    except Exception as e:
        print("  history(keys) failed:", e)
    # fallback: scan_history
    if not rows:
        try:
            for h in run.scan_history(keys=keys):
                rows.append(h)
        except Exception as e:
            print("  scan_history failed:", e)
    # extract clean series
    series = []
    for h in rows:
        mr = h.get("evaluation/mean_return")
        if mr is None:
            continue
        series.append({
            "env_steps": h.get("global_env_steps"),
            "update_step": h.get("global_update_step"),
            "session": h.get("session"),
            "mean_return": mr,
            "mean_performance": h.get("evaluation/mean_performance"),
        })
    series.sort(key=lambda x: (x["env_steps"] if x["env_steps"] is not None else 0))
    print(f"  eval points: {len(series)}")
    for s in series:
        print(f"    step={s['update_step']} env={s['env_steps']} sess={s['session']} "
              f"mean_return={s['mean_return']}")
    summary[label] = series

with open("_wandb_eval_curves.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nsaved -> _wandb_eval_curves.json")
