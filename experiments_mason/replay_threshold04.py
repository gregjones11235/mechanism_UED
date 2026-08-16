"""mastery_threshold=0.4 离线回放预判 —— 零 GPU,十分钟出纸面裁决。

问题:0.4 相对 0.2 的赌注是"用中期前沿推进变慢换巩固窗口拉长"。本脚本拿 2e9 主跑的
全部官方离线快照(两段拼接,总 update 300→15300)逐点回放 pick_target,对比两个
threshold 下的目标池,量化三件事:
  1. 中期霸池:0.4 池中 SR∈[0.2,0.4) 的"中层滞留技能"占了几席;
  2. 前沿推迟:gnomish / 宝石 / armour 等关键前沿技能在 0.4 下首次入池比 0.2 晚多少;
  3. 巩固窗口:iron_sword / iron_pickaxe 在 0.4 下多留池多少个快照点。

纸面否决线(预注册):中期(总 2000-8000)有 ≥4 个快照点上 0.4 池被中层占 ≥4/6 席,
或 gnomish 入池推迟 ≥6 个快照点(≈ 0.23e9 步)——两者任一成立,0.4 不上真跑。

用法(pod 或本地均可,纯 CPU):
    cd dicode_src && PYTHONPATH=src:. python ../experiments_mason/replay_threshold04.py
"""
import json
import os
import sys

sys.path.insert(0, "src"); sys.path.insert(0, ".")
from dicode.skill_preflight.skill_scheduler import pick_target  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = json.load(open(f"{HERE}/eval/eval_C2LITE2E9_seed0.json"))
SUP = json.load(open(f"{HERE}/eval/eval_C2LITE2E9SUP_seed0.json"))
ORIG = {"300", "900", "1500", "2100", "2700", "3300", "3900", "4500", "5100",
        "5700", "6300", "6600"}

FOCUS = ["enter_gnomish_mines", "collect_sapphire", "collect_ruby",
         "make_iron_armour", "make_iron_sword", "make_iron_pickaxe"]

snaps = []
for src in (MAIN, SUP):
    for k, v in src.items():
        tot = int(k) if k in ORIG else int(k) + 6800
        snaps.append((tot, v["skills"]))
snaps.sort()
snaps = [(t, s) for t, s in snaps if s.get("skill_collect_wood", 0) > 1]  # 剔崩溃点

first_in = {0.2: {}, 0.4: {}}
window = {0.2: {k: 0 for k in FOCUS}, 0.4: {k: 0 for k in FOCUS}}
midhog = []
print(f"{'总upd':>6} | {'0.2 池':<58} | {'0.4 池(★=中层滞留)'}")
for tot, sk in snaps:
    pools = {}
    for th in (0.2, 0.4):
        t = pick_target(sk, threshold=th, frontier_mode="prereq", prereq_threshold=0.3)
        pools[th] = list(t.target_achievements)
        for a in pools[th]:
            first_in[th].setdefault(a, tot)
            if a in window[th]:
                window[th][a] += 1
    mid = [a for a in pools[0.4] if 20 <= sk.get(f"skill_{a}", 0) < 40]
    if 2000 <= tot <= 8000 and len(mid) >= 4:
        midhog.append(tot)
    tag = lambda a: f"{a}★" if a in mid else a
    print(f"{tot:>6} | {','.join(pools[0.2])[:58]:<58} | {','.join(tag(a) for a in pools[0.4])[:70]}")

print("\n== 判据读数 ==")
delay = {}
for a in FOCUS:
    d2, d4 = first_in[0.2].get(a), first_in[0.4].get(a)
    idx = lambda t: next((i for i, (tt, _) in enumerate(snaps) if tt == t), None)
    delay[a] = (idx(d4) - idx(d2)) if (d2 and d4) else None
    print(f"  {a:<24} 首入池: 0.2@{d2} vs 0.4@{d4} (推迟 {delay[a]} 个快照点)"
          f" | 留池窗口: {window[0.2][a]} vs {window[0.4][a]} 点")
n_hog = len(midhog)
gnom_delay = delay.get("enter_gnomish_mines") or 0
print(f"\n中期霸池点数(总2000-8000,中层≥4/6席): {n_hog}(否决线 ≥4)")
print(f"gnomish 入池推迟: {gnom_delay} 个快照点(否决线 ≥6)")
verdict = "❌ 纸面否决,0.4 不上真跑" if (n_hog >= 4 or gnom_delay >= 6) else \
          "✅ 纸面放行 —— 可作为 teacher 侧候选带上周五(真跑仍需拍板)"
print(f"\n裁决: {verdict}")
