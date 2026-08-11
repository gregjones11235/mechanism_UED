#!/usr/bin/env python3
"""D052 read-only cell inventory (阶段2 scaffolding).

Reads the PRIOR unified_eval orchestration_state.json (treated as UNVERIFIED
leads), lists all 25 cells with seed + checkpoint path, and verifies on disk:
  - checkpoint_98304 dir exists
  - manifest.json / stage4_manifest.json present
Does NOT modify anything. Prints a compact table + aggregation/mechanism sets
+ the non-cell top-level keys (to find evaluator config / probe summaries).
"""
import json
import os

STATE = "/home/oseasy/experiments/d052_unified_eval_20260722/orchestration_state.json"

d = json.load(open(STATE))
cells = d.get("d052_cells", {})
print("total cells in state:", len(cells))

aggs = set()
mechs = set()
for name in sorted(cells):
    info = cells[name]
    a, m = name.split("_x_", 1)
    aggs.add(a)
    mechs.add(m)
    ck = info.get("checkpoint_98304", "")
    exists = os.path.isdir(ck)
    man = os.path.exists(os.path.join(ck, "manifest.json"))
    sm = os.path.exists(os.path.join(ck, "stage4_manifest.json"))
    seed = info.get("seed", "?")
    gate = info.get("gate", "?")
    tr = info.get("training", "?")
    print(f"{name:38s} {seed:18s} dir={int(exists)} man={int(man)} "
          f"s4man={int(sm)} gate={gate} train={tr}")

print()
print("AGGREGATIONS (%d): %s" % (len(aggs), sorted(aggs)))
print("MECHANISMS (%d): %s" % (len(mechs), sorted(mechs)))
print()
print("non-cell top-level keys:",
      [k for k in d.keys() if k not in ("gpu", "d052_cells")])
