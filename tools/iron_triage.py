import json, sys, numpy as np
from craftax.craftax.constants import Achievement
d = json.load(open(sys.argv[1])); det = d[list(d)[0]]
ach = np.asarray(det["ach_at_done"]); ret = np.asarray(det["return"])
died = np.asarray(det["died"]).astype(bool) & np.asarray(det["finished"]).astype(bool)
print(f"n(died,有成就位)={died.sum()}")
for name in ["COLLECT_IRON", "MAKE_IRON_PICKAXE", "MAKE_IRON_SWORD", "MAKE_IRON_ARMOUR", "COLLECT_DIAMOND"]:
    idx = Achievement[name].value
    got = ach[:, idx] > 0.5
    a, b = got & died, (~got) & died
    if a.sum() == 0: print(f"{name}: 0 达成"); continue
    print(f"{name}: 达成率={100*a.sum()/died.sum():.1f}%  达成者ret={ret[a].mean():.2f}  "
          f"未达成ret={ret[b].mean():.2f}  溢价={ret[a].mean()-ret[b].mean():+.2f}")
