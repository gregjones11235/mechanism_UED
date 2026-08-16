"""Server CPU isolation smoke: imports the research tools and exercises
static_lint + cpu_jax_validation on CPU only. Touches NO GPU.

Run inside the /tmp sandbox on the server with the skill_preflight_e0e1 venv.
"""
import os
import sys

TOOLS = os.environ.get("LLM_TOOLS", "/tmp/llm_research_d_BrRxKL/tools")
SRC = "/home/oseasy/e2_data_disk2/skill_preflight_runs/mason_full_budget_gpu3_91a75e5_20260810T1805/source/dicode_src/src"
sys.path.insert(0, TOOLS)
sys.path.insert(0, SRC)

import llm_replay_manifest as m
import llm_replay_harness as h
import llm_replay_benchmark as b
import llm_replay_report as r  # noqa: F401

print("=== imports OK ===")

print("=== static_lint ===")
print("syntax_error ->", h.static_lint("def foo(:"))
print("valid_enum ->", h.static_lint(
    "from craftax.craftax.constants import BlockType\nx = BlockType.COAL"))
print("invalid_enum ->", h.static_lint(
    "from craftax.craftax.constants import BlockType\nx = BlockType.NONEXISTENT"))

seed_path = os.path.join(SRC, "minicraftax/tasks/seed_tasks/collecting.py")
seed_code = open(seed_path).read()
print("=== cpu_jax_validation on real seed task (CPU only) ===")
ok, msg = h.cpu_jax_validation(seed_code)
print("cpu_jax_validation ->", ok, (msg[:160] if msg else ""))

# verify a real invalid code fails the CPU validation (not silently passed)
print("=== cpu_jax_validation on invalid code ===")
ok2, msg2 = h.cpu_jax_validation("class Env: pass")
print("invalid_cpu_jax ->", ok2, (msg2[:160] if msg2 else ""))

print("=== manifest roundtrip smoke ===")
spec = {
    "classification": "LLM_REPLAY_MANIFEST",
    "source_commit": "91a75e5",
    "provider": "local", "model": "qwen2.5-coder:14b",
    "base_url": "http://127.0.0.1:11434/v1",
    "temperature": 0.6, "top_p": 0.95, "max_tokens": 8192,
    "system_prompt": "sys", "user_prompts": ["p1", "p2"],
    "candidate_slots": ["s0", "s1"],
}
man = m.build_replay_manifest(spec)
out = os.path.join(TOOLS, "_cpu_smoke_manifest.json")
written = m.write_manifest(man, out)
reloaded = m.load_manifest(out)
print("manifest_roundtrip ->", reloaded["manifest_sha256"] == written["manifest_sha256"])
print("=== CPU ISOLATION SMOKE DONE ===")
