#!/usr/bin/env python
"""Phase 0 go/no-go: can the local 14B write COMPILABLE Craftax env code?

Standalone: uses DiCode's real generation prompt + real seed examples + a
replicated check_compilation. Measures FIRST-TRY compile rate (no reflection).
DiCode adds reflection on top, so the 'usable' rate in the full pipeline is >= this.
"""
import os, re, json, time, ast, tempfile, importlib
from pathlib import Path
from collections import defaultdict

# --- point the DiCode LLM client at local Ollama ---
os.environ.setdefault("GENERATION_SERVER_URL", "http://localhost:11434/v1")
os.environ.setdefault("OPENAI_API_KEY", "ollama")

import jax
import jax.numpy as jnp
from dicode.dreaming.llm import LLM
from dicode.dreaming.gen_manager import Task

REPO = Path("/workspace/mechanism_UED/dicode_src")
OUT = Path("/workspace/phase0_out")
OUT.mkdir(parents=True, exist_ok=True)

MODEL = "qwen2.5-coder:14b"
N_PER_TARGET = 3            # generations per target
GO_THRESHOLD = 0.5         # first-try compile rate to call it GO

# --- real generation prompt + knowledge-base contexts (paths from conf/gen_manager/default.yaml) ---
gen_env = importlib.import_module("dicode.dreaming.prompts.cl_.gen_env")
CRAFTAX_CODE = importlib.import_module("dicode.dreaming.prompts.cl_.craftax_coder").context
MINICRAFTAX_CODE = importlib.import_module("dicode.dreaming.prompts.cl_.minicraftax_coder").context
MOBS_CODE = importlib.import_module("dicode.dreaming.prompts.dicode.mobs_code").context

# --- the 4 seed tasks DiCode ships (config example_paths) ---
SEED_PATHS = [
    "src/minicraftax/tasks/seed_tasks/collecting.py",
    "src/minicraftax/tasks/seed_tasks/combat.py",
    "src/minicraftax/tasks/seed_tasks/crafting.py",
    "src/minicraftax/tasks/seed_tasks/survive.py",
]


def class_docstring(path: Path) -> str:
    """Extract the Env class docstring = the natural-language task description."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            doc = ast.get_docstring(node)
            if doc:
                return doc
    return path.read_text()  # fallback


def examples_excluding(exclude: str) -> str:
    """Leave-one-out: show the OTHER seed tasks' code as examples."""
    parts = []
    for p in SEED_PATHS:
        if p == exclude:
            continue
        code = (REPO / p).read_text()
        parts.append(f"<example>\n{code}\n</example>\n")
    return "\n".join(parts)


def _strip_md_fence(code: str) -> str:
    """Remove a leading ```python / ``` fence and trailing ``` if the model wrapped
    its code in a markdown block inside the <code> tags (this caused SyntaxError line 1)."""
    if not code:
        return code
    code = code.strip()
    code = re.sub(r"^```[a-zA-Z0-9_+-]*\s*\n", "", code)   # leading ``` or ```python
    code = re.sub(r"\n```\s*$", "", code)                   # trailing ```
    return code.strip()


def extract_code(content: str):
    """Pull code out of <code>...</code>, then strip any inner markdown fence."""
    if not content:
        return None
    m = re.search(r"<code>\s*(.*?)\s*</code>", content, re.DOTALL)
    raw = m.group(1).strip() if m else content
    return _strip_md_fence(raw)


def check_compilation(code: str):
    """Replicates EnvGenerator.check_compilation: load env, run reset+step on CPU."""
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_file = f.name
        try:
            cpu = jax.devices("cpu")[0]
        except IndexError:
            cpu = jax.local_devices(backend="cpu")[0]
        with jax.default_device(cpu):
            env = Task(temp_file).env
            params = env.default_params
            rng = jax.random.PRNGKey(0)
            rng, rk = jax.random.split(rng)
            obs, state = env.reset(rk, params)
            action = env.action_space(params).sample(rng)
            obs, state, reward, done, info = env.step(rng, state, action, params)
            for fn, v in state.inventory.__dict__.items():
                if hasattr(v, "dtype") and v.dtype != jnp.int32:
                    raise ValueError(f"Inventory '{fn}' is {v.dtype}, expected int32")
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    finally:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)


def main():
    llm = LLM(provider="local",
              base_url=os.environ["GENERATION_SERVER_URL"],
              model=MODEL, llm_type="generation",
              max_tokens=8192, temperature=0.6, top_p=0.95, think=False)

    system_prompt = gen_env.system_prompt.format(
        CRAFTAX_CODE=CRAFTAX_CODE, MINICRAFTAX_CODE=MINICRAFTAX_CODE, MOBS=MOBS_CODE)
    print(f"[info] system prompt: {len(system_prompt):,} chars "
          f"(~{len(system_prompt)//4:,} tokens) — make sure Ollama context covers this")

    results = []
    for tgt in SEED_PATHS:
        name = Path(tgt).stem
        desc = class_docstring(REPO / tgt)
        user_prompt = gen_env.user_prompt.format(
            CODE_EXAMPLES=examples_excluding(tgt), TASK_DESCRIPTION=desc)
        print(f"\n=== target: {name}  (user prompt {len(user_prompt):,} chars) ===")
        for i in range(N_PER_TARGET):
            t0 = time.time()
            code, ok, msg = None, False, ""
            try:
                resp = llm.query(system_prompt, [user_prompt])
                item = resp[0] if resp else {}
                content, err0 = item.get("content"), item.get("error")
                code = extract_code(content)
                if not code:
                    ok, msg = False, f"no code extracted (llm error={err0})"
                else:
                    ok, msg = check_compilation(code)
            except Exception as e:
                ok, msg = False, f"EXC {type(e).__name__}: {e}"
            dt = time.time() - t0
            if code:
                (OUT / f"{name}_{i}.py").write_text(code)
            results.append({"target": name, "i": i, "compiled": ok,
                            "error": msg[:400], "secs": round(dt, 1),
                            "code_len": len(code) if code else 0})
            print(f"  {name}#{i}: {'OK ' if ok else 'FAIL'} "
                  f"({dt:4.0f}s, {len(code) if code else 0} chars)"
                  f"{'' if ok else '  <- ' + msg[:140]}")

    # ---- report ----
    n = len(results)
    n_ok = sum(r["compiled"] for r in results)
    rate = n_ok / n if n else 0.0
    print("\n" + "=" * 46)
    print("PHASE 0 RESULT")
    print("=" * 46)
    print(f"Model: {MODEL}")
    print(f"First-try compile rate: {n_ok}/{n} = {100 * rate:.0f}%")
    bt = defaultdict(lambda: [0, 0])
    for r in results:
        bt[r["target"]][0] += r["compiled"]
        bt[r["target"]][1] += 1
    for t, (o, c) in bt.items():
        print(f"  {t:12s}: {o}/{c}")
    json.dump(results, open(OUT / "phase0_results.json", "w"), indent=2)
    print(f"\nGenerated code + results saved to {OUT}")

    print("\nVERDICT:", end=" ")
    if rate >= GO_THRESHOLD:
        print("GO — 14B produces compilable Craftax env code at a workable first-try rate.")
        print("(DiCode adds reflection retries on top, so usable rate will be higher.)")
    elif rate > 0:
        print("MARGINAL — some compile. Inspect errors in phase0_out; reflection or a")
        print("bigger/less-quantized model may push it over. Not dead, but check quality.")
    else:
        print("NO-GO — 14B cannot produce compilable env code. Method needs rethink")
        print("(bigger model / fp16 / different approach). Inspect errors before deciding.")


if __name__ == "__main__":
    main()
