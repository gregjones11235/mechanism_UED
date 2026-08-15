#!/usr/bin/env python3
"""Mid-course read for the surgical arm — run ON THE POD with system python3:

    python3 midread_surgical.py [graphml_path]

Default path: /root/outputs/surgical/task_graph.graphml
Stdlib only (no venv needed). Mirrors the verdict-day funnel exactly:
descriptions are PROPOSALS (Player-anchored regex), code is REALITY (AST on
set_starting_floor). References baked in (trained mob levels, code floor >=2):
longStack 0.0% (0/27) | gateOff 95.0% (191/201) | shufGraph 94.2% | base2e9 76.9%.

Surgical-specific expectations:
  * code f>=2 should head for ~90% (floor clause off -> floor-2 mob levels survive)
  * FLOOR rewrites (desc_floor > code_floor) ~ 0  — the gate may still repair
    premark/inventory violations, but those do not move floors
  * gate alive: check the log side separately (ScaffoldGate == R3-floor OFF counts)
"""
import sys, re, json, ast, collections
import xml.etree.ElementTree as ET

NS = "{http://graphml.graphdrawing.org/xmlns}"
START = re.compile(r"Player:\s*Starts?\s+on\s+floor\s+(\d)", re.I)
RELEV = re.compile(r"Relevant Achievements:\s*([A-Z_0-9,\s]+)")
MOBS = ("DEFEAT_GNOME_WARRIOR", "DEFEAT_GNOME_ARCHER", "EAT_BAT")
REF = "参照(被训练怪关卡 code f>=2): longStack 0.0% | gateOff 95.0% | shuf 94.2% | base 76.9%"


def parse(p):
    km, ns, cur = {}, [], None
    for ev, el in ET.iterparse(p, events=("start", "end")):
        t = el.tag.replace(NS, "")
        if ev == "end" and t == "key":
            km[el.attrib["id"]] = el.attrib.get("attr.name", el.attrib["id"]); el.clear()
        elif ev == "start" and t == "node":
            cur = {"id": el.attrib.get("id")}
        elif ev == "end" and t == "data" and cur is not None:
            cur[km.get(el.attrib["key"], el.attrib["key"])] = el.text or ""; el.clear()
        elif ev == "end" and t == "node":
            ns.append(cur); cur = None; el.clear()
    return ns


def code_floor(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    fl = 0
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "set_starting_floor"
                and n.args and isinstance(n.args[0], ast.Constant)):
            try:
                fl = int(n.args[0].value)
            except (TypeError, ValueError):
                pass
    return fl


def main(path):
    g = [n for n in parse(path) if n.get("type") != "seed"]
    mob = []
    for n in g:
        r = RELEV.search(n.get("description") or "")
        if r and any(s in r.group(1).upper() for s in MOBS):
            mob.append(n)
    coded = [n for n in mob if (n.get("code") or "").strip()]
    trained, rows, srs, sess = [], [], [], 0
    for n in coded:
        ph = (n.get("performance_history") or "").strip()
        if ph in ("", "[]", "null"):
            continue
        try:
            h = json.loads(ph)
        except ValueError:
            continue
        if not h:
            continue
        trained.append(n)
        sess += len(h)
        m = START.search(n.get("description") or "")
        df = int(m.group(1)) if m else None
        cf = code_floor(n["code"])
        if df is not None and cf is not None:
            rows.append((df, cf))
        for rec in h:
            for k, v in (rec.get("achievement_srs") or {}).items():
                if any(x in k.lower() for x in ("gnome_warrior", "gnome_archer", "eat_bat")):
                    srs.append(v)

    sc = [int(n["session_created"]) for n in g
          if (n.get("session_created") or "").lstrip("-").isdigit()]
    print("=== surgical 中读 · 档案跨度 s%d-%d · 生成节点 %d ===" %
          (min(sc) if sc else 0, max(sc) if sc else 0, len(g)))
    print("怪关卡: 生成 %d (%.1f%%) | 有代码 %d | 已训练 %d | 重放 %.1fx" %
          (len(mob), 100 * len(mob) / len(g) if g else 0, len(coded), len(trained),
           sess / len(trained) if trained else 0))
    if rows:
        ge2 = sum(1 for _, c in rows if c >= 2)
        down = sum(1 for d, c in rows if d > c)
        dc = collections.Counter(d for d, _ in rows)
        cc = collections.Counter(c for _, c in rows)
        print("code f>=2: %d/%d = %.1f%%   |   楼层改写(desc>code): %d  (预期 ~0)" %
              (ge2, len(rows), 100 * ge2 / len(rows), down))
        print("desc 楼层分布 %s  ->  code 楼层分布 %s" %
              (dict(sorted(dc.items())), dict(sorted(cc.items()))))
    grades = collections.Counter(n.get("status", "?") for n in trained)
    med = sorted(srs)[len(srs) // 2] if srs else float("nan")
    print("怪训练 SR 中位: %.1f   |   评级: %s" % (med, dict(grades.most_common(6))))
    print(REF)
    if len(trained) < 5:
        print("!! 已训练怪关卡不足 5 个 —— 样本太小,过几小时再跑一次")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/root/outputs/surgical/task_graph.graphml")
