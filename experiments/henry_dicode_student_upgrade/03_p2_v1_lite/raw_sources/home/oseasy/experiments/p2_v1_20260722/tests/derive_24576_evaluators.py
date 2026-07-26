#!/usr/bin/env python3
"""Derive TWO 64-episode Stage4 evaluators from the frozen session175 pilot
evaluator (parent SHA 06221187) — byte-identical protocol EXCEPT three literals
(CKPT / CKPT_LABEL / OUT) + a derivation note, per model:

  (A) P2-v1 Full @24576      -> checkpoints/24576
  (B) Original PPO @24576    -> checkpoints_original_ppo/24576

Protocol unchanged: seed 42, 64 episodes (64 parallel envs), S4_dark scaffold,
DEFEAT_KOBOLD ever-set dual-channel success, stochastic pi.sample, 4096 max
steps, BIG network config. load_weights_only reused unchanged (80/80 bit-exact
vs restore_p2_v1_checkpoint, proven by probe_level2_weight_load.py).
Prints unified diffs + derived SHA256 for each. Writes both files. Does NOT run.
"""
import difflib, hashlib, os

SRC = "/home/oseasy/experiments/session175_dual_caliber_pilot_20260722/s175_dual_caliber_pilot.py"
PARENT_SHA = "06221187ac06d7da59dac64e6273abfc865b3baafdc75615e0808fc5065d26e2"
ORIG_CKPT = 'CKPT = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/base_ckpt_v7fix55_armA_s0/rl_checkpoints/17500"'
ORIG_LABEL = 'CKPT_LABEL = "session175_base_ckpt_v7fix55_armA_s0_rl_17500"'
ORIG_OUT = 'OUT = "/home/oseasy/experiments/session175_dual_caliber_pilot_20260722"'
MARKER = '"""session175 dual-caliber pilot evaluator (directive section 七).'

with open(SRC, "rb") as f: src_bytes = f.read()
parent_sha = hashlib.sha256(src_bytes).hexdigest()
assert parent_sha == PARENT_SHA, f"parent SHA mismatch: {parent_sha}"
src_text = src_bytes.decode("utf-8")

SPECS = [
    {"name": "p2_v1_full_24576",
     "ckpt": 'CKPT = "/home/oseasy/experiments/p2_v1_20260722/checkpoints/24576"',
     "label": 'CKPT_LABEL = "p2_v1_full_24576_isolated_plan2"',
     "out": 'OUT = "/home/oseasy/experiments/p2_v1_full_24576_64ep_stage4_eval_20260723"',
     "dst": "/home/oseasy/experiments/p2_v1_full_24576_64ep_stage4_eval_20260723/eval_p2_v1_full_24576_64ep.py",
     "desc": "P2-v1 Full @24576 (方案2 isolated critic-only replay-aux)"},
    {"name": "original_ppo_24576",
     "ckpt": 'CKPT = "/home/oseasy/experiments/p2_v1_20260722/checkpoints_original_ppo/24576"',
     "label": 'CKPT_LABEL = "original_ppo_24576_control"',
     "out": 'OUT = "/home/oseasy/experiments/original_ppo_24576_64ep_stage4_eval_20260723"',
     "dst": "/home/oseasy/experiments/original_ppo_24576_64ep_stage4_eval_20260723/eval_original_ppo_24576_64ep.py",
     "desc": "Original PPO @24576 (pure Henry native PPO control, replay/hindsight OFF)"},
]

results = []
for s in SPECS:
    text = src_text
    for old, new in [(ORIG_CKPT, s["ckpt"]), (ORIG_LABEL, s["label"]), (ORIG_OUT, s["out"])]:
        assert text.count(old) == 1, f"need exactly 1 occurrence of: {old[:50]}"
        text = text.replace(old, new)
    note = (MARKER +
        f'\n\nDERIVED EVALUATOR (P2-v1 三模型统一评估, 2026-07-23): {s["desc"]}.\n'
        'Byte-identical to parent SHA 06221187ac06d7da59dac64e6273abfc865b3baafdc75615e0808fc5065d26e2\n'
        'EXCEPT three literals (CKPT / CKPT_LABEL / OUT) + this note. Protocol UNCHANGED:\n'
        'seed 42, 64 episodes, S4_dark, DEFEAT_KOBOLD ever-set dual-channel, stochastic,\n'
        '4096 max steps, BIG net config. load_weights_only reused unchanged (80/80 bit-exact).\n')
    assert MARKER in text
    text = text.replace(MARKER, note, 1)
    assert text.count("DERIVED EVALUATOR") == 1
    os.makedirs(os.path.dirname(s["dst"]), exist_ok=True)
    with open(s["dst"], "w") as f: f.write(text)
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    diff = "".join(difflib.unified_diff(src_text.splitlines(True), text.splitlines(True),
                fromfile="parent_06221187.py", tofile=os.path.basename(s["dst"]), n=0))
    # sanity: diff lines that change code (exclude the note block) are only the 3 literals
    code_changes = [l for l in diff.splitlines()
                    if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))
                    and "DERIVED" not in l and "Byte-identical" not in l and "seed 42" not in l
                    and "4096 max" not in l and "load_weights_only reused" not in l
                    and "EXCEPT three" not in l and l.strip() not in ("+", "-")]
    print("=" * 72)
    print(f"MODEL: {s['name']}  ({s['desc']})")
    print(f"  dst = {s['dst']}")
    print(f"  sha256 = {sha}")
    print(f"  code-literal diff lines ({len(code_changes)}):")
    for l in code_changes: print("    " + l)
    results.append({"name": s["name"], "dst": s["dst"], "sha256": sha})

print("=" * 72)
print("parent_sha256 =", parent_sha)
import json
print("DERIVED_EVALUATORS_JSON =", json.dumps(results))
