#!/usr/bin/env python3
"""Derive NINE 64-episode Stage4 evaluators from the frozen session175 pilot
evaluator (parent SHA 06221187) — byte-identical protocol EXCEPT three literals
(CKPT / CKPT_LABEL / OUT) + a derivation note, for the from-scratch fair
comparison learning curve:

  step0 (common random init)  -> scratch_init_seed0/0   (evaluated ONCE, shared)
  Original PPO @ 24576/49152/73728/98304 -> checkpoints_from_scratch_op/<step>
  P2-v1 Full   @ 24576/49152/73728/98304 -> checkpoints_from_scratch_p2/<step>

Protocol unchanged: seed 42, 64 episodes (64 parallel envs), S4_dark scaffold,
DEFEAT_KOBOLD ever-set dual-channel success, stochastic pi.sample, 4096 max
steps, BIG net config. load_weights_only reused unchanged. The from-scratch
checkpoints are params-only (replay_meta rolled-stripped) which the parent's
load path handles (it was built for the replay-less 17500 checkpoint).
Prints unified diffs + derived SHA256 for each. Writes all files. Does NOT run.
"""
import difflib, hashlib, os, json

SRC = "/home/oseasy/experiments/session175_dual_caliber_pilot_20260722/s175_dual_caliber_pilot.py"
PARENT_SHA = "06221187ac06d7da59dac64e6273abfc865b3baafdc75615e0808fc5065d26e2"
ORIG_CKPT = 'CKPT = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/base_ckpt_v7fix55_armA_s0/rl_checkpoints/17500"'
ORIG_LABEL = 'CKPT_LABEL = "session175_base_ckpt_v7fix55_armA_s0_rl_17500"'
ORIG_OUT = 'OUT = "/home/oseasy/experiments/session175_dual_caliber_pilot_20260722"'
MARKER = '"""session175 dual-caliber pilot evaluator (directive section 七).'
E = "/home/oseasy/experiments"
DATE = "20260723"

with open(SRC, "rb") as f: src_bytes = f.read()
parent_sha = hashlib.sha256(src_bytes).hexdigest()
assert parent_sha == PARENT_SHA, f"parent SHA mismatch: {parent_sha}"
src_text = src_bytes.decode("utf-8")

def spec(name, ckpt_path, label, desc):
    out = f"{E}/fs_{name}_64ep_stage4_eval_{DATE}"
    return {"name": name,
            "ckpt": f'CKPT = "{ckpt_path}"',
            "label": f'CKPT_LABEL = "{label}"',
            "out": f'OUT = "{out}"',
            "dst": f"{out}/eval_fs_{name}_64ep.py",
            "desc": desc}

OP = f"{E}/p2_v1_20260722/checkpoints_from_scratch_op"
P2 = f"{E}/p2_v1_20260722/checkpoints_from_scratch_p2"
SPECS = [
    spec("step0_common_init", f"{E}/p2_v1_20260722/scratch_init_seed0/0",
         "from_scratch_step0_common_random_init_seed0",
         "Common random init step0 (shared bit-exact start, params SHA e78426c8)"),
]
for st in (24576, 49152, 73728, 98304):
    SPECS.append(spec(f"original_ppo_{st}", f"{OP}/{st}",
                      f"from_scratch_original_ppo_seed0_step{st}",
                      f"Original PPO from scratch seed0 @ {st} (pure Henry native PPO, replay/hindsight OFF)"))
for st in (24576, 49152, 73728, 98304):
    SPECS.append(spec(f"p2_v1_full_{st}", f"{P2}/{st}",
                      f"from_scratch_p2_v1_full_seed0_step{st}",
                      f"P2-v1 Full from scratch seed0 @ {st} (Plan B + hindsight + isolated critic-only replay-aux)"))

results = []
for s in SPECS:
    text = src_text
    for old, new in [(ORIG_CKPT, s["ckpt"]), (ORIG_LABEL, s["label"]), (ORIG_OUT, s["out"])]:
        assert text.count(old) == 1, f"need exactly 1 occurrence of: {old[:50]}"
        text = text.replace(old, new)
    note = (MARKER +
        f'\n\nDERIVED EVALUATOR (P2-v1 从零公平对比评估, {DATE}): {s["desc"]}.\n'
        'Byte-identical to parent SHA 06221187ac06d7da59dac64e6273abfc865b3baafdc75615e0808fc5065d26e2\n'
        'EXCEPT three literals (CKPT / CKPT_LABEL / OUT) + this note. Protocol UNCHANGED:\n'
        'seed 42, 64 episodes, S4_dark, DEFEAT_KOBOLD ever-set dual-channel, stochastic,\n'
        '4096 max steps, BIG net config. load_weights_only reused unchanged (params-only ckpt OK).\n')
    assert MARKER in text
    text = text.replace(MARKER, note, 1)
    assert text.count("DERIVED EVALUATOR") == 1
    os.makedirs(os.path.dirname(s["dst"]), exist_ok=True)
    with open(s["dst"], "w") as f: f.write(text)
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    diff = "".join(difflib.unified_diff(src_text.splitlines(True), text.splitlines(True),
                fromfile="parent_06221187.py", tofile=os.path.basename(s["dst"]), n=0))
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
    # each derived eval must differ from parent by EXACTLY the 3 literal code lines (+/-)
    assert len(code_changes) == 6, f"expected 6 code-literal diff lines (3 removed + 3 added), got {len(code_changes)}"
    results.append({"name": s["name"], "dst": s["dst"], "ckpt": s["ckpt"].split('"')[1],
                    "label": s["label"].split('"')[1], "out": s["out"].split('"')[1], "sha256": sha})

print("=" * 72)
print("parent_sha256 =", parent_sha)
print("DERIVED_EVALUATORS_JSON =", json.dumps(results))
ev = "/home/oseasy/experiments/single_director_20260722/evidence"; os.makedirs(ev, exist_ok=True)
with open(os.path.join(ev, "p2_v1_from_scratch_derived_evaluators.json"), "w") as f:
    json.dump({"parent_sha256": parent_sha, "evaluators": results}, f, indent=2, sort_keys=True); f.write("\n")
print(f"WROTE {len(results)} derived evaluators + evidence JSON", flush=True)
