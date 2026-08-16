"""死因判决表:eval_*_details.json -> 逐杀稠密生死裁决的原料"""
import json, sys
import numpy as np
d = json.load(open(sys.argv[1]))
det = d[list(d)[0]]
A = {k: np.asarray(v) for k, v in det.items()}
fin = A["finished"].astype(bool); died = A["died"].astype(bool) & fin
n = fin.sum(); F = A["steps_on_floor"].shape[1]
print(f"finished={n}/{len(fin)}  died={died.sum()} ({100*died.sum()/max(n,1):.1f}%)  "
      f"mean_return={A['return'][fin].mean():.2f}")
print("\n== 死亡楼层分布(died only)==")
fl = A["floor_at_done"][died]
for f in range(F):
    c = (fl == f).sum()
    if c: print(f"  floor {f}: {c} ({100*c/died.sum():.1f}%)")
print("\n== 分层账本(全员,均值/env)==")
print(f"{'floor':>5} {'steps':>8} {'dmg_in':>8} {'dmg_out':>8} {'交战比':>8} {'kills':>6}")
for f in range(F):
    s = A["steps_on_floor"][:, f].mean()
    if s < 0.5: continue
    di = A["dmg_taken_floor"][:, f].mean(); do = A["dmg_dealt_floor"][:, f].mean()
    kl = (A["kills_melee"][:, f].sum() + A["kills_ranged"][:, f].sum())
    print(f"{f:>5} {s:>8.1f} {di:>8.2f} {do:>8.2f} {do/max(di,1e-6):>8.2f} {kl:>6.0f}")
print("\n== 死亡上下文(died only)==")
md = A["min_melee_dist_death"][died]; rd = A["min_ranged_dist_death"][died]
fo = A["food_at_death"][died]; dr = A["drink_at_death"][died]
print(f"  近敌死(melee dist<=3): {100*(md<=3).mean():.1f}%   ranged<=5: {100*(rd<=5).mean():.1f}%")
print(f"  饿死疑(food==0): {100*(fo<=0).mean():.1f}%   渴死疑(drink==0): {100*(dr<=0).mean():.1f}%")
print(f"  无敌孤死(melee>10 & ranged>10): {100*((md>10)&(rd>10)).mean():.1f}%")
print("\n== 击杀矩阵 floor x type_id(全员合计)==")
for tag in ["kills_melee", "kills_ranged"]:
    K = A[tag].sum(0)
    for f in range(F):
        if K[f].sum() > 0.5:
            print(f"  {tag} floor {f}: " + " ".join(f"t{t}:{K[f,t]:.0f}" for t in range(8) if K[f,t] > 0.5))
# 裁决口径:1层死亡里"近敌死"占比高 + dmg_taken 集中于少数 type -> 逐杀稠密有靶子;
# 死因弥散(孤死/饿死占比高、伤害来源分散)-> 提案毙。
