import os, re
import wandb
key = os.environ.get("WANDB_API_KEY")
if not key:
    with open(".env", encoding="utf-8") as f:
        for line in f:
            m = re.match(r"\s*wandb_API\s*=\s*(\S+)", line)
            if m:
                key = m.group(1); break
assert key, "no wandb key"
os.environ["WANDB_API_KEY"] = key
os.environ["WANDB_DISABLE_CODE"] = "true"
api = wandb.Api(timeout=90)
ENTITY = "gregjones11235-brown-university"
TARGETS = [("DiCode-auction", "dicode-auctionC-s0-v2", "C-v2", 1500),
           ("DiCode-repro", "dicode-repro-s0-v1", "baseline", 12700)]
def find(project, rid):
    try:
        return api.run(f"{ENTITY}/{project}/{rid}")
    except Exception:
        for r in api.runs(f"{ENTITY}/{project}"):
            if rid in (r.id, r.name, getattr(r, "display_name", "")):
                return r
    return None
for project, rid, label, restore in TARGETS:
    print(f"\n===== {label}  ({project}/{rid})  restore@{restore} =====")
    run = find(project, rid)
    if run is None:
        print("  RUN NOT FOUND; available:")
        for r in api.runs(f"{ENTITY}/{project}"):
            print("   -", r.id, "|", r.name, "|", r.state)
        continue
    print(f"  id={run.id} state={run.state}")
    pts = []
    for row in run.scan_history(keys=["_step", "evaluation/mean_return"]):
        mr = row.get("evaluation/mean_return"); st = row.get("_step")
        if mr is not None and st is not None:
            pts.append((st, mr))
    pts.sort()
    if not pts:
        print("  no eval/mean_return points yet"); continue
    print(f"  total eval points={len(pts)}  step range {pts[0][0]}..{pts[-1][0]}")
    lo, hi = restore - 800, restore + 800
    win = [(s, v) for s, v in pts if lo <= s <= hi]
    print(f"  --- window [{lo},{hi}] around restore ({len(win)} pts) ---")
    for s, v in win:
        print(f"    step {s:>6}  mean_return {v:.3f}")
    steps = [s for s, _ in pts]
    dup = len(steps) != len(set(steps))
    backw = any(steps[i] < steps[i-1] for i in range(1, len(steps)))
    print(f"  step_has_duplicates={dup}  step_goes_backward={backw}")
    if len(win) >= 2:
        jumps = [(win[i][1]-win[i-1][1], win[i-1][0], win[i][0]) for i in range(1, len(win))]
        big = max(jumps, key=lambda x: abs(x[0]))
        print(f"  biggest mean_return delta in window = {big[0]:+.3f} (step {big[1]}->{big[2]})")
