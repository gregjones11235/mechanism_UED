#!/usr/bin/env python3
"""P2-v1 Original-PPO FAIR CONTROL (24576 env steps, GPU0).

Runs the FROZEN P2-v1 launcher's exact PPO main-update path with the replay
auxiliary update and hindsight relabel DISABLED, so the result is pure Henry
native on-policy PPO. Nothing in the frozen source is modified — this driver
only (a) points the launcher at independent output/checkpoint dirs (no reuse of
the P2-v1 Full dirs) and (b) runtime-patches p2_v1_update to force
replay_aux=False + relabel_callback=None. The launcher/learner/p2_v1_core file
SHAs therefore stay byte-identical to the frozen set.

Fairness vs P2-v1 Full (Level3, 24576):
  - SAME session175 weights-only init (load_opt_state=False, frozen base 17500)
  - SAME Stage4 config (Cfg: 16 envs x 128 rollout x 12 updates = 24576)
  - SAME training seed (P2_V1_MASTER_SEED=42 -> identical JAX + action RNG)
  - SAME env steps (24576), SAME fresh optimizer / empty replay / global_step=0
  - ONLY difference: replay-aux critic update + hindsight OFF here.
The PPO main update (clipped surrogate + value + entropy + GAE) is the identical
code path; with replay_aux=False the aux block (and relabel) is skipped entirely
(p2_v1_core.p2_v1_update: `if replay_aux and replay.can_sample():`).
"""
import os, sys, json, hashlib

GPU_UUID = "GPU-e8c08612-c22a-c8a4-6df5-affb2dd1f9a6"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID

P2 = "/home/oseasy/experiments/p2_v1_20260722/src"
V7 = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src"
for p in (V7, P2):
    if p not in sys.path:
        sys.path.insert(0, p)

import stage4_continue_launcher as L

# ── independent dirs (NO reuse of P2-v1 Full outputs/checkpoints) ─────
L.OUTPUT_ROOT = "/home/oseasy/experiments/p2_v1_20260722/outputs_original_ppo"
L.CKPT_ROOT = "/home/oseasy/experiments/p2_v1_20260722/checkpoints_original_ppo"
os.makedirs(L.OUTPUT_ROOT, exist_ok=True)
os.makedirs(L.CKPT_ROOT, exist_ok=True)

# ── force pure PPO: disable replay aux + hindsight relabel ────────────
_orig_update = L.p2_v1_update
def _ppo_only_update(*args, **kwargs):
    kwargs["replay_aux"] = False
    kwargs["relabel_callback"] = None
    return _orig_update(*args, **kwargs)
L.p2_v1_update = _ppo_only_update

def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

note = {
    "run": "P2-v1 Original-PPO fair control (24576 env steps, GPU0)",
    "directive": "第九节 P2-v1公平对照Original PPO",
    "gpu_uuid": GPU_UUID,
    "method": "frozen launcher main(1,12) with p2_v1_update runtime-patched to "
              "replay_aux=False, relabel_callback=None; OUTPUT_ROOT/CKPT_ROOT "
              "redirected to *_original_ppo dirs",
    "frozen_source_sha256": {
        "stage4_continue_launcher.py": _sha(os.path.join(P2, "stage4_continue_launcher.py")),
        "long_context_learner.py": _sha(os.path.join(P2, "long_context_learner.py")),
        "p2_v1_core.py": _sha(os.path.join(P2, "p2_v1_core.py")),
        "checkpointing.py": _sha(os.path.join(P2, "checkpointing.py")),
    },
    "expected_frozen_sha": {
        "stage4_continue_launcher.py": "36ec9cd9eef7f3408b6b8680be7d2d21552be4577e78946cf0948b0b9ca9079f",
        "long_context_learner.py": "6689426b77bb030c8ce3a3a3c97ddab7bd0248d2eaa1d477146dff44ccf1c386",
        "p2_v1_core.py": "6e20d2e60b638e45bba7ba32cdb44b3b871d8ca69b61f47239dff23e1e798974",
        "checkpointing.py": "9b8cf1a276aeda4173494ae3d9575dec74df7e93d435ff2c056a810bc3c5a56a",
    },
    "fairness": {
        "init": "weights-only from session175 base 17500 (load_opt_state=False)",
        "seed": 42, "num_envs": 16, "rollout_steps": 128, "num_updates": 12,
        "env_steps": 24576, "replay_aux": False, "hindsight": False,
        "optimizer": "fresh", "global_step_start": 0, "update_count_start": 0,
    },
    "output_root": L.OUTPUT_ROOT, "ckpt_root": L.CKPT_ROOT,
}
# fail-closed if a frozen SHA drifted
for k, v in note["expected_frozen_sha"].items():
    assert note["frozen_source_sha256"][k] == v, f"FROZEN SHA DRIFT: {k}"
ev = "/home/oseasy/experiments/single_director_20260722/evidence"
os.makedirs(ev, exist_ok=True)
with open(os.path.join(ev, "p2_v1_original_ppo_control_note.json"), "w") as f:
    json.dump(note, f, indent=2, sort_keys=True); f.write("\n")
print("[original-ppo-control] frozen SHAs verified; replay_aux+hindsight DISABLED", flush=True)
print(f"[original-ppo-control] OUTPUT_ROOT={L.OUTPUT_ROOT}", flush=True)
print(f"[original-ppo-control] CKPT_ROOT={L.CKPT_ROOT}", flush=True)

# ── run the frozen launcher pipeline (pure PPO) ───────────────────────
L.main(max_sessions=1, num_updates=12, resume_from=None)
