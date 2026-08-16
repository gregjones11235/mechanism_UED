#!/usr/bin/env python3
"""
★C Stage 1 — 离线修复级联基准（不占训练 GPU；Ollama 轻量调用）

回答两个问题：
  Q-lint : C-0 静态 lint 能捕获多大比例的幻觉类失败？（预期 ~100% 的枚举幻觉）
  Q-repair: C-1 小模型修复的成功率与 token 成本是多少？（判据：修复通过率>70%，
            每修复 token 远低于一次完整重生成）

流程（全在 pod 上跑）：
  1. whitelist  : 运行时导入 craftax 真身枚举（与生成代码同一 import 路径）
  2. harvest    : 从训练日志收集有机幻觉清单（has no attribute 'X'），并与
                  prompt 白名单交叉（幻觉成员是"无中生有"还是"prompt 缺失"）
  3. bench      : 从 task_graph.graphml 取真实 14B 生成代码 → 注入取自有机清单
                  的幻觉故障 → C-0 lint 检测 → C-1 调 Ollama 修复 → 三级验证
                  （ast.parse / exec / Env 实例化，镜像 gen_manager.load_env）
  4. report     : 检测率 / 修复率 / token 成本，存 JSON

用法（pod，venv 内）：
  python stage1_repair_bench.py \
      --graphml outputs/2026-07-07_214222_740288/task_graph.graphml \
      --logs /workspace/baseline_run.log /workspace/run_AB.log /workspace/run_AB2.log \
      --n 30 --repair-attempts 2 \
      --ollama http://localhost:11434 --model qwen2.5-coder:14b

注意：
  - 训练同机跑时无碍（每次修复=一次普通 chat 调用），但建议避开设计 session 高峰。
  - 局限（报告里要写）：故障为"注入式"（取自真实幻觉分布），非端到端有机失败；
    深层故障（world-gen 期，如 BlockType.BAT）需 --deep 才触发 reset 级验证。
"""
import argparse, ast, difflib, json, os, random, re, sys, time, urllib.request

# ---------------------------------------------------------------- whitelist
def build_whitelist():
    from craftax.craftax.constants import Achievement, BlockType  # 运行时真身
    wl = {"Achievement": sorted(m.name for m in Achievement),
          "BlockType":   sorted(m.name for m in BlockType)}
    print(f"[whitelist] Achievement={len(wl['Achievement'])}  BlockType={len(wl['BlockType'])}")
    return wl

def prompt_whitelist(repo_root):
    """prompt 里展示给 LLM 的枚举清单（交叉参照，可选）"""
    p = os.path.join(repo_root, "src/dicode/dreaming/prompts/cl_/craftax_coder.py")
    if not os.path.isfile(p):
        return None
    src = open(p, encoding="utf-8", errors="replace").read()
    out = {}
    for cls in ("Achievement", "BlockType"):
        m = re.search(r"class " + cls + r"\(Enum\):(.*?)(?=\nclass |\Z)", src, re.S)
        out[cls] = sorted(set(re.findall(r"^\s+([A-Z_][A-Z0-9_]*)\s*=", m.group(1), re.M))) if m else []
    return out

# ---------------------------------------------------------------- harvest
HALLU_RE = re.compile(r"type object '(Achievement|BlockType)' has no attribute '([A-Z_][A-Z0-9_]*)'")

def harvest(logs):
    inv = {}
    for lp in logs:
        if not os.path.isfile(lp):
            print(f"[harvest] skip missing {lp}"); continue
        raw = open(lp, "rb").read().decode("utf-8", errors="replace")
        for cls, name in HALLU_RE.findall(raw):
            inv.setdefault(cls, {}).setdefault(name, 0)
            inv[cls][name] += 1
    total = sum(sum(v.values()) for v in inv.values())
    print(f"[harvest] organic hallucinations: {total} hits, "
          f"{sum(len(v) for v in inv.values())} unique members")
    return inv

# ---------------------------------------------------------------- C-0 lint
def lint(code, wl):
    """AST 扫 Achievement.X / BlockType.Y，返回 [(cls, bad_member, lineno, suggestion)]"""
    hits = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [("__syntax__", str(e), getattr(e, "lineno", -1), None)]
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id in wl and node.attr not in wl[node.value.id]):
            sug = difflib.get_close_matches(node.attr, wl[node.value.id], n=1, cutoff=0.4)
            hits.append((node.value.id, node.attr, node.lineno, sug[0] if sug else None))
    return hits

# ---------------------------------------------------------------- 三级验证（镜像 gen_manager.load_env）
def verify(code, deep=False):
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"ast: {e}"
    import types
    mod = types.ModuleType("cand")
    mod.__dict__["__file__"] = "<stage1>"
    try:
        exec(compile(code, "<stage1>", "exec"), mod.__dict__)
    except Exception as e:
        return False, f"exec: {type(e).__name__}: {e}"
    try:
        from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
        env_task = getattr(mod, "Env")(static_params=StaticEnvParams(), params=EnvParams())
    except Exception as e:
        return False, f"init: {type(e).__name__}: {e}"
    if deep:  # reset 级（触发 generate_world，捕获 world-gen 期故障；较慢）
        try:
            import jax
            from minicraftax.envs.base import MiniCraftaxTrain
            env = MiniCraftaxTrain(task=env_task)
            env.reset(jax.random.PRNGKey(0), env.default_params)
        except Exception as e:
            return False, f"reset: {type(e).__name__}: {e}"
    return True, "ok"

# ---------------------------------------------------------------- C-1 repair via Ollama
def ollama_chat(base, model, prompt, timeout=300):
    body = json.dumps({"model": model, "stream": False,
                       "messages": [{"role": "user", "content": prompt}],
                       "options": {"temperature": 0.2, "num_predict": 4096}}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    msg = resp["message"]["content"]
    tok_in  = resp.get("prompt_eval_count", -1)
    tok_out = resp.get("eval_count", -1)
    return msg, tok_in, tok_out, time.time() - t0

def strip_fence(text):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip() + "\n"

REPAIR_PROMPT = """You are a code repairer. The following Python file references enum members \
that DO NOT EXIST in the real API. Fix ONLY those references, changing nothing else.

Invalid references found (class, member, line, closest valid member):
{lint_report}

Valid members of {cls} include (partial): {wl_snippet}

Rules: output the COMPLETE corrected file in one ```python code block. Replace each invalid \
member with the semantically closest VALID member. Do not add, remove, or reorder anything else.

File:
```python
{code}
```"""

def repair(code, hits, wl, base, model, attempts=2, deep=False):
    tot_in = tot_out = 0
    for k in range(attempts):
        cls = hits[0][0] if hits and hits[0][0] != "__syntax__" else "Achievement"
        snippet = ", ".join(wl.get(cls, [])[:40])
        rep = "\n".join(f"  - {c}.{m} (line {ln}, closest: {s})" for c, m, ln, s in hits)
        msg, ti, to, _ = ollama_chat(base, model,
            REPAIR_PROMPT.format(lint_report=rep, cls=cls, wl_snippet=snippet, code=code))
        tot_in += max(ti, 0); tot_out += max(to, 0)
        code = strip_fence(msg)
        hits = [h for h in lint(code, wl) if h[0] != "__syntax__"]
        if not hits:
            ok, why = verify(code, deep=deep)
            return ok, why, k + 1, tot_in, tot_out
    return False, "lint still failing", attempts, tot_in, tot_out

# ---------------------------------------------------------------- bench
def load_codes(graphml, limit):
    import networkx as nx
    g = nx.read_graphml(graphml)
    codes = [(n, d["code"]) for n, d in g.nodes(data=True)
             if d.get("code") and "class Env" in d.get("code", "")]
    random.shuffle(codes)
    return codes[:limit]

def inject(code, inv, wl):
    """把一个合法枚举引用替换成有机清单里的幻觉成员；返回 (坏代码, (cls, fake))"""
    tree = ast.parse(code)
    legal = [(n.value.id, n.attr, n.lineno) for n in ast.walk(tree)
             if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
             and n.value.id in wl and n.attr in wl[n.value.id]]
    if not legal:
        return None, None
    pool = [(c, f) for c, d in inv.items() for f in d] or \
           [("Achievement", "DESCEND"), ("BlockType", "BAT")]
    cls, real, _ = random.choice([l for l in legal if l[0] in dict(pool)] or legal)
    fakes = [f for c, f in pool if c == cls] or ["DESCEND"]
    fake = random.choice(fakes)
    bad = re.sub(rf"\b{cls}\.{real}\b", f"{cls}.{fake}", code, count=1)
    return (bad, (cls, fake)) if bad != code else (None, None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphml", required=True)
    ap.add_argument("--logs", nargs="+", default=[])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--repair-attempts", type=int, default=2)
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument("--model", default="qwen2.5-coder:14b")
    ap.add_argument("--deep", action="store_true", help="reset 级验证（捕获 world-gen 故障，较慢）")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    wl = build_whitelist()
    pw = prompt_whitelist(args.repo_root)
    inv = harvest(args.logs)

    # 交叉参照：幻觉成员是否曾出现在 prompt 清单里(理论上不该)；真身成员 prompt 是否缺失
    cross = {}
    if pw:
        for cls in ("Achievement", "BlockType"):
            cross[cls] = {
                "hallucinated_but_in_prompt": sorted(set(inv.get(cls, {})) & set(pw[cls])),
                "real_missing_from_prompt":   sorted(set(wl[cls]) - set(pw[cls])),
                "prompt_extra_not_real":      sorted(set(pw[cls]) - set(wl[cls])),
            }
        print("[cross]", json.dumps(cross, ensure_ascii=False))

    codes = load_codes(args.graphml, args.n)
    print(f"[bench] {len(codes)} real generated task codes loaded")

    rows, det, rep_ok = [], 0, 0
    for tid, code in codes:
        bad, fault = inject(code, inv, wl)
        if bad is None:
            continue
        hits = [h for h in lint(bad, wl) if h[0] != "__syntax__"]
        detected = any(h[1] == fault[1] for h in hits)
        det += detected
        row = {"task": tid, "fault": fault, "detected": detected}
        if detected:
            ok, why, k, ti, to = repair(bad, hits, wl, args.ollama, args.model,
                                        args.repair_attempts, args.deep)
            rep_ok += ok
            row.update({"repaired": ok, "why": why, "attempts": k,
                        "tok_in": ti, "tok_out": to})
            print(f"  {tid}: fault={fault} lint={'✓' if detected else '✗'} "
                  f"repair={'✓' if ok else '✗ ' + why} tok={ti}+{to}")
        rows.append(row)

    n = len(rows)
    summary = {
        "n": n,
        "lint_detection_rate": round(det / n, 3) if n else None,
        "repair_success_rate_given_detected": round(rep_ok / det, 3) if det else None,
        "mean_repair_tokens": (round(sum(r.get("tok_in", 0) + r.get("tok_out", 0)
                               for r in rows if r.get("detected")) / det, 1) if det else None),
        "organic_inventory": {c: d for c, d in inv.items()},
        "prompt_cross_reference": cross,
    }
    print("\n===== STAGE 1 SUMMARY ====="); print(json.dumps(summary, indent=2, ensure_ascii=False))
    out = "stage1_report.json"
    json.dump({"summary": summary, "rows": rows}, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"[saved] {out}")

if __name__ == "__main__":
    main()
