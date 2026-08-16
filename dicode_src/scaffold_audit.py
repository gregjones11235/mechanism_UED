#!/usr/bin/env python3
"""
脚手架泄漏全量审计（静态 AST 检测，零 LLM 成本）

对各臂 task_graph.graphml 中全部生成任务代码,检测四种脚手架签名:
  S1 inventory : builder.set_player_inventory({...})        —— 直接送资源/工具
  S2 premark   : self.completed_achievements = [N 个成就]    —— 预标前置链(代码级跳过依赖)
  S3 mob_near  : builder.add_mobs_randomly_near(min_dist<=8) —— 目标怪刷在玩家旁
  S4 floor     : builder.set_starting_floor(n>0)             —— 跳过下潜过程

输出:
  - 每臂: 总任务数 / 任一脚手架泄漏率 / 各签名分布 / 深层定向子集泄漏率
  - 深度分桶: 任务定向深度(按 relevant_achievements 关键词) vs 泄漏率
  - scaffold_report.json(全量明细,归档用)

用法(pod, venv):
  python scaffold_audit.py \
    --arm probe=/path/to/probe/task_graph.graphml \
    --arm ARMAB=/path/to/armAB/task_graph.graphml \
    --arm ARMA_ext=/path/... --arm BASE_ext=/path/...

注:S2 阈值——completed_achievements 非空即计(预标 1 个也是跳依赖);
   S3 阈值 min_dist<=8(task_19 实录为 4-8);
   种子任务(task_1..4)通常无 code 或为手写,单独列出不计入生成任务分母。
"""
import argparse, ast, json, sys

# ---- 脚手架签名检测 ----------------------------------------------------------
SEED_IDS = {"task_1", "task_2", "task_3", "task_4"}

DEEP_KW = {  # 任务定向深度分桶(按 relevant_achievements 成员名关键词)
    3: ("DIAMOND", "LIZARD", "FIREBALL", "ICEBALL", "VAULT", "GNOME", "ORC",
        "TROLL", "KOBOLD", "KNIGHT", "ARCHER", "NECROMANCER", "ENCHANT",
        "SEWERS", "GRAVEYARD", "FIRE_REALM", "ICE_REALM", "PIGMAN", "DEEP_THING",
        "ELEMENTAL", "DUNGEON_2"),
    2: ("IRON", "DUNGEON", "BOW", "CHEST", "SNAIL", "BAT", "POTION", "MINES"),
    1: ("STONE", "SKELETON", "PLANT", "COAL", "TORCH", "ARROW", "FURNACE"),
    0: ("WOOD", "SAPLING", "TABLE", "COW", "DRINK", "ZOMBIE", "WAKE"),
}

def task_depth(relevant):
    """按 relevant_achievements 里最深的成员定深度。"""
    best = -1
    for name in relevant:
        u = name.upper()
        for d, kws in DEEP_KW.items():
            if any(k in u for k in kws):
                best = max(best, d)
    return best

def audit_code(code):
    """返回 dict: relevant[], scaffolds{S1..S4: detail}, parse_ok"""
    out = {"relevant": [], "scaffolds": {}, "parse_ok": True}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        out["parse_ok"] = False
        return out

    for node in ast.walk(tree):
        # relevant_achievements / completed_achievements 赋值
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                    and t.value.id == "self"):
                names = [n.attr for n in ast.walk(node.value)
                         if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                         and n.value.id == "Achievement"]
                if t.attr == "relevant_achievements":
                    out["relevant"] = names
                elif t.attr == "completed_achievements" and names:
                    out["scaffolds"]["S2_premark"] = {"n_premarked": len(names),
                                                      "members": names[:6]}
        # builder.* 调用
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            fn = node.func.attr
            if fn == "set_player_inventory":
                items = {}
                for a in node.args:
                    if isinstance(a, ast.Dict):
                        for k, v in zip(a.keys, a.values):
                            if isinstance(k, ast.Constant):
                                items[str(k.value)] = (v.value if isinstance(v, ast.Constant) else "?")
                out["scaffolds"]["S1_inventory"] = {"items": items}
            elif fn == "add_mobs_randomly_near":
                kw = {k.arg: (k.value.value if isinstance(k.value, ast.Constant) else "?")
                      for k in node.keywords if k.arg}
                md = kw.get("min_dist")
                if isinstance(md, (int, float)) and md <= 8:
                    out["scaffolds"]["S3_mob_near"] = {"min_dist": md,
                                                       "max_dist": kw.get("max_dist"),
                                                       "mob": kw.get("mob_name") or kw.get("type_id")}
            elif fn == "set_starting_floor":
                if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value:
                    out["scaffolds"]["S4_floor"] = {"floor": node.args[0].value}
    return out

# ---- 主流程 ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True,
                    help="label=/path/to/task_graph.graphml (可多次)")
    ap.add_argument("--out", default="scaffold_report.json")
    args = ap.parse_args()

    import networkx as nx
    report = {}
    for spec in args.arm:
        label, path = spec.split("=", 1)
        g = nx.read_graphml(path)
        rows, seeds = [], []
        for n, d in g.nodes(data=True):
            code = d.get("code")
            if not code or "class Env" not in code:
                continue
            r = audit_code(code)
            r["task"] = n
            r["depth"] = task_depth(r["relevant"])
            (seeds if n in SEED_IDS else rows).append(r)

        gen = rows
        leaked = [r for r in gen if r["scaffolds"]]
        by_sig = {s: sum(1 for r in gen if s in r["scaffolds"])
                  for s in ("S1_inventory", "S2_premark", "S3_mob_near", "S4_floor")}
        by_depth = {}
        for dep in (0, 1, 2, 3):
            sub = [r for r in gen if r["depth"] == dep]
            if sub:
                by_depth[dep] = {"n": len(sub),
                                 "leaked": sum(1 for r in sub if r["scaffolds"]),
                                 "rate": round(sum(1 for r in sub if r["scaffolds"]) / len(sub), 3)}
        premark_counts = [r["scaffolds"]["S2_premark"]["n_premarked"]
                          for r in gen if "S2_premark" in r["scaffolds"]]

        report[label] = {
            "n_generated": len(gen),
            "n_leaked": len(leaked),
            "leak_rate": round(len(leaked) / len(gen), 3) if gen else None,
            "by_signature": by_sig,
            "by_target_depth": by_depth,
            "premark_mean_n": round(sum(premark_counts) / len(premark_counts), 1) if premark_counts else 0,
            "seed_tasks_excluded": len(seeds),
            "examples_leaked": [r["task"] for r in leaked[:8]],
            "detail": gen,
        }
        print(f"\n===== {label} =====")
        print(f"  生成任务 {len(gen)} | 泄漏 {len(leaked)} | 泄漏率 {report[label]['leak_rate']}")
        print(f"  签名分布: {by_sig} | 预标均数 {report[label]['premark_mean_n']}")
        print(f"  按定向深度: " + " | ".join(
            f"tier{d}: {v['leaked']}/{v['n']}={v['rate']}" for d, v in sorted(by_depth.items())))

    json.dump(report, open(args.out, "w"), indent=2, ensure_ascii=False, default=str)
    print(f"\n[saved] {args.out}")

if __name__ == "__main__":
    main()
