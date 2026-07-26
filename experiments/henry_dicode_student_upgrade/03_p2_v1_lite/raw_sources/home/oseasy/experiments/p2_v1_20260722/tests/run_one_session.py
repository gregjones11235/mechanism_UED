#!/usr/bin/env python3
"""From-scratch single-session driver (GPU0). Usage: run_one_session.py MODE RESUME
  MODE   in {original_ppo, p2_full}
  RESUME = "none" (session 1, fresh init from common scratch) or a step int
           (24576/49152/73728 -> 方案B exact-resume continuation)

Each invocation is a FRESH process running ONE 12-update (24576-step) session via
the frozen launcher main(). The sh launcher chains 4 invocations (none/24576/
49152/73728) to reach 98304 steps with checkpoints at each boundary. Frozen code
unmodified; runtime patches only: SESSION175_CKPT -> scratch_init_seed0/0,
CKPT_ROOT/OUTPUT_ROOT -> per-group dirs, and (original_ppo) p2_v1_update forced
replay_aux=False + relabel_callback=None (pure Henry native PPO).
"""
import os, sys, json, shutil, hashlib
GPU_UUID = "GPU-e8c08612-c22a-c8a4-6df5-affb2dd1f9a6"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID
MODE = sys.argv[1]; RESUME = sys.argv[2]
assert MODE in ("original_ppo", "p2_full")
resume_from = None if RESUME == "none" else int(RESUME)

P2 = "/home/oseasy/experiments/p2_v1_20260722/src"
V7 = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src"
for p in (V7, P2):
    if p not in sys.path: sys.path.insert(0, p)
import stage4_continue_launcher as L

EXP = "/home/oseasy/experiments/p2_v1_20260722"
SCRATCH_STEP0 = os.path.join(EXP, "scratch_init_seed0", "0")
SUFFIX = {"original_ppo": "op", "p2_full": "p2"}[MODE]
L.CKPT_ROOT = os.path.join(EXP, f"checkpoints_from_scratch_{SUFFIX}")
L.OUTPUT_ROOT = os.path.join(EXP, f"outputs_from_scratch_{SUFFIX}")
os.makedirs(L.CKPT_ROOT, exist_ok=True); os.makedirs(L.OUTPUT_ROOT, exist_ok=True)
L.SESSION175_CKPT = SCRATCH_STEP0   # common random start (verify_not_v0_source passes)

if MODE == "original_ppo":
    _orig = L.p2_v1_update
    def _ppo_only(*a, **kw):
        kw["replay_aux"] = False; kw["relabel_callback"] = None
        return _orig(*a, **kw)
    L.p2_v1_update = _ppo_only

# session 1: place the bit-identical step-0 checkpoint (copy of common scratch)
if resume_from is None:
    g0 = os.path.join(L.CKPT_ROOT, "0")
    if not os.path.isdir(g0):
        shutil.copytree(SCRATCH_STEP0, g0)
    def sha(p):
        with open(p, "rb") as f: return hashlib.sha256(f.read()).hexdigest()
    note = {"run": f"from-scratch {MODE} seed0", "mode": MODE, "gpu_uuid": GPU_UUID,
            "scratch_init": SCRATCH_STEP0,
            "scratch_params_sha256": "e78426c8fe9097d26039c982e64185fbc8db4695a175ed2e37a839fb7e37d48e",
            "ckpt_root": L.CKPT_ROOT, "output_root": L.OUTPUT_ROOT,
            "replay_aux": MODE == "p2_full", "hindsight": MODE == "p2_full",
            "schedule": "4x12-update exact-resume chain -> 24576/49152/73728/98304",
            "training_seed": L.P2_V1_MASTER_SEED, "init_seed": 0,
            "frozen_code_sha256": {fn: sha(os.path.join(P2, fn)) for fn in
                ["long_context_learner.py","stage4_continue_launcher.py","p2_v1_core.py","checkpointing.py"]}}
    ev = "/home/oseasy/experiments/single_director_20260722/evidence"; os.makedirs(ev, exist_ok=True)
    with open(os.path.join(ev, f"p2_v1_from_scratch_{SUFFIX}_note.json"), "w") as f:
        json.dump(note, f, indent=2, sort_keys=True); f.write("\n")

print(f"[session {MODE}] resume_from={resume_from}  CKPT_ROOT={L.CKPT_ROOT}  "
      f"replay_aux={MODE=='p2_full'}", flush=True)
L.main(max_sessions=1, num_updates=12, resume_from=resume_from)
print(f"[session {MODE}] SESSION DONE resume_from={resume_from}", flush=True)
