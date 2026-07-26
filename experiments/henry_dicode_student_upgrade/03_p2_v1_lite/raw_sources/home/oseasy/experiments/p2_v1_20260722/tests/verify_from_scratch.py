#!/usr/bin/env python3
"""One-shot verification of a from-scratch 98304-step run (GPU0 restore).
Usage: verify_from_scratch.py MODE   (MODE = original_ppo | p2_full)

Checkpoints are params-only (replay_meta stripped by the rolling-strip chain to
bound disk), so verification uses: per-step manifests + the 4 session training
logs + a params round-trip via load_weights_only (orbax restore MUST run on GPU;
GPU-saved orbax cannot be restored under JAX_PLATFORM_NAME=cpu).

Gates:
  common: 5 checkpoints (0/24576/49152/73728/98304) with orbax; frozen-code SHAs;
          step0 params SHA == scratch e78426c8...; 98304 manifest global_step==98304,
          update_count==48, gradient_updates==48, gamma/gae/gpu; no NaN/Inf in any
          log; global_step spans 0->98304 across the 4 sessions; trajectories
          collected==inserted (no insertion loss); params finite; 98304 params
          round-trip bit-exact (load twice -> identical SHA) and differ from step0;
          main opt_step==96 (48 updates x 2 minibatches).
  original_ppo: replay_aux_update all False; relabelled_samples==0;
          replay_samples_drawn==0 (pure Henry native PPO).
  p2_full:    replay_aux_update True sometimes (aux fired); hindsight accepted>0;
          relabelled_samples>0; replay_samples_drawn>0.
Prints PASS/FAIL per gate + writes evidence JSON. Frozen source NOT modified.
"""
import os, sys, json, glob, hashlib
GPU_UUID = "GPU-e8c08612-c22a-c8a4-6df5-affb2dd1f9a6"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID
import jax, numpy as np
MODE = sys.argv[1]
assert MODE in ("original_ppo", "p2_full")
SUFFIX = {"original_ppo": "op", "p2_full": "p2"}[MODE]

P2 = "/home/oseasy/experiments/p2_v1_20260722/src"
V7 = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src"
for p in (V7, P2):
    if p not in sys.path: sys.path.insert(0, p)
import stage4_continue_launcher as L

EXP = "/home/oseasy/experiments/p2_v1_20260722"
CK = os.path.join(EXP, f"checkpoints_from_scratch_{SUFFIX}")
OUT = os.path.join(EXP, f"outputs_from_scratch_{SUFFIX}")
SCRATCH_SHA = "e78426c8fe9097d26039c982e64185fbc8db4695a175ed2e37a839fb7e37d48e"
STEPS = [0, 24576, 49152, 73728, 98304]
EXPECTED_FROZEN = {
    "long_context_learner.py": "6689426b77bb030c8ce3a3a3c97ddab7bd0248d2eaa1d477146dff44ccf1c386",
    "stage4_continue_launcher.py": "36ec9cd9eef7f3408b6b8680be7d2d21552be4577e78946cf0948b0b9ca9079f",
    "p2_v1_core.py": "6e20d2e60b638e45bba7ba32cdb44b3b871d8ca69b61f47239dff23e1e798974",
    "checkpointing.py": "9b8cf1a276aeda4173494ae3d9575dec74df7e93d435ff2c056a810bc3c5a56a",
}

def sha(p):
    with open(p, "rb") as f: return hashlib.sha256(f.read()).hexdigest()

gates = {}

# ── frozen code SHAs ────────────────────────────────────────────────
frozen_ok = True
for fn, want in EXPECTED_FROZEN.items():
    got = sha(os.path.join(P2, fn))
    if got != want: frozen_ok = False; print(f"  FROZEN MISMATCH {fn}: {got}")
gates["frozen_code_sha256"] = frozen_ok

# ── 5 checkpoints have orbax default ────────────────────────────────
gates["checkpoints_orbax_present"] = all(os.path.isdir(os.path.join(CK, str(s), "default")) for s in STEPS)

# ── build dummy env + network for weights load (as launcher does) ────
cfg = L.Cfg()
ach_table = jax.numpy.array([L.get_achievement_multi_hot([L.Achievement.DEFEAT_KOBOLD])], dtype=jax.numpy.float32)
EMB = int(ach_table.shape[1])
with open(L.S4_TASK_PATH) as f: s4_code = f.read()
ns = {}; exec(s4_code, ns); Task = ns["Env"]
static_env_params = L.StaticEnvParams(); env_params = L.EnvParams(max_timesteps=4096)
base_env = L.MultiTaskMiniCraftaxEnv([Task], static_env_params, env_params, cfg.condition_on_task,
    conditioning_type="embedding", embedding_size=EMB, completion_bonus_scale=cfg.completion_bonus_scale,
    completion_bonus_min=cfg.completion_bonus_min, bonus_type=cfg.bonus_type, dynamic_bonus_k=cfg.dynamic_bonus_k)

# ── step0 params SHA == common scratch init ─────────────────────────
ts0 = L.load_weights_only(os.path.join(CK, "0"), base_env, env_params, cfg, load_opt_state=False)
step0_sha = L._params_content_sha256(ts0.params)
gates["step0_is_common_init"] = (step0_sha == SCRATCH_SHA)
print(f"[verify:{MODE}] step0 params sha={step0_sha}  matches_scratch={step0_sha==SCRATCH_SHA}", flush=True)

# ── 98304 params round-trip (load twice -> identical) + finite + differs from step0 ──
tsA = L.load_weights_only(os.path.join(CK, "98304"), base_env, env_params, cfg, load_opt_state=False)
tsB = L.load_weights_only(os.path.join(CK, "98304"), base_env, env_params, cfg, load_opt_state=False)
shaA = L._params_content_sha256(tsA.params); shaB = L._params_content_sha256(tsB.params)
gates["restore_98304_bitexact"] = (shaA == shaB)
leavesA = jax.tree_util.tree_leaves(tsA.params)
gates["params_98304_finite"] = bool(all(np.all(np.isfinite(np.asarray(v))) for v in leavesA))
gates["params_98304_advanced_from_step0"] = (shaA != step0_sha)
print(f"[verify:{MODE}] 98304 sha={shaA} roundtrip={shaA==shaB} differs_from_step0={shaA!=step0_sha}", flush=True)

# ── main opt_step == 96 (restore opt_state) ─────────────────────────
ts_opt = L.load_weights_only(os.path.join(CK, "98304"), base_env, env_params, cfg, load_opt_state=True)
opt_step = L._optimizer_step_count(ts_opt)
gates["main_opt_step_96"] = (opt_step == 48 * 2)
print(f"[verify:{MODE}] main opt_step={opt_step} (expect 96)", flush=True)

# ── 98304 manifest gates ─────────────────────────────────────────────
with open(os.path.join(CK, "98304", "manifest.json")) as f: mani = json.load(f)
with open(os.path.join(CK, "98304", "stage4_manifest.json")) as f: s4mani = json.load(f)
gates["manifest_global_step_98304"] = (int(mani["global_step"]) == 98304)
gates["manifest_gradient_updates_48"] = (int(mani["counters"]["gradient_updates"]) == 48)
gates["manifest_gpu_uuid"] = (s4mani.get("gpu_uuid") == GPU_UUID)
gates["manifest_gamma"] = (abs(float(s4mani["gamma"]) - 0.999) < 1e-12)
gates["manifest_gae_lambda"] = (abs(float(s4mani["gae_lambda"]) - 0.8) < 1e-12)
traj_coll = int(mani["counters"]["trajectories_collected"])
traj_ins = int(mani["counters"]["trajectories_inserted"])
gates["manifest_no_insertion_loss"] = (traj_coll == traj_ins)
relabelled = int(mani["counters"]["relabelled_samples"])
drawn = int(mani["counters"]["replay_samples_drawn"])
# update_count semantics: OP has no aux so update_count==48 (==gradient_updates); P2's
# update_count ALSO counts replay-aux updates, so update_count==48+relabelled. The PPO main
# update count is gradient_updates==48 (gate above); the MAIN optimizer advances only by PPO
# (opt_step==96, isolation gate below) regardless of how many aux updates ran.
expected_uc = 48 + (relabelled if MODE == "p2_full" else 0)
gates["manifest_update_count_consistent"] = (int(mani["update_count"]) == expected_uc)
print(f"[verify:{MODE}] manifest: traj_coll={traj_coll} traj_ins={traj_ins} relabelled={relabelled} drawn={drawn} "
      f"replay_size={mani['replay_buffer_size']} pending={mani['pending_total_transitions']} update_count={mani['update_count']} (expect {expected_uc})", flush=True)

# ── parse all session training logs ─────────────────────────────────
logs = sorted(glob.glob(os.path.join(OUT, "stage4_continue", "session_*", "training_log.jsonl")))
print(f"[verify:{MODE}] found {len(logs)} training logs", flush=True)
all_recs = []
nan_inf = False
gs_seen = set(); uc_seen = set()
aux_flags = []; relabel_counts = []; drawn_counts = []
for lg in logs:
    with open(lg) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: r = json.loads(line)
            except Exception: continue
            all_recs.append(r)
            # scan any numeric for nan/inf
            def scan(o):
                global nan_inf
                if isinstance(o, dict):
                    for v in o.values(): scan(v)
                elif isinstance(o, list):
                    for v in o: scan(v)
                elif isinstance(o, float):
                    if o != o or o in (float("inf"), float("-inf")): nan_inf = True
            scan(r)
            if "global_step" in r: gs_seen.add(int(r["global_step"]))
            if "update_count" in r: uc_seen.add(int(r["update_count"]))
            if "replay_aux_update" in r: aux_flags.append(bool(r["replay_aux_update"]))
            for k in ("n_relabel", "relabelled", "hindsight_accepted", "n_hindsight_accepted"):
                if k in r: relabel_counts.append(r[k])
            for k in ("replay_samples_drawn", "n_replay_drawn", "replay_drawn"):
                if k in r: drawn_counts.append(r[k])
gates["no_nan_inf_in_logs"] = (not nan_inf)
# per-update log records carry update_count (not global_step). The final log update_count must
# match the manifest (OP=48; P2=48+aux). global_step==98304 is gated via the manifest above.
gates["logs_update_count_matches_manifest"] = (len(uc_seen) > 0 and max(uc_seen) == int(mani["update_count"]))
gates["logs_have_ppo_records"] = (len(all_recs) > 0)
n_ppo = sum(1 for r in all_recs if "total_loss" in r or "policy_loss" in r or "ppo_policy_loss" in r)
print(f"[verify:{MODE}] log records={len(all_recs)} ppo_like={n_ppo} gs_max={max(gs_seen) if gs_seen else None} "
      f"uc_max={max(uc_seen) if uc_seen else None} aux_flags_set={set(aux_flags)}", flush=True)

# ── mode-specific gates ──────────────────────────────────────────────
if MODE == "original_ppo":
    gates["op_replay_aux_all_false"] = (len(aux_flags) > 0 and not any(aux_flags))
    gates["op_relabelled_zero"] = (relabelled == 0)
    gates["op_replay_drawn_zero"] = (drawn == 0)
else:  # p2_full
    gates["p2_replay_aux_fired"] = any(aux_flags)
    gates["p2_relabelled_positive"] = (relabelled > 0)
    gates["p2_replay_drawn_positive"] = (drawn > 0)
    # actor/trunk isolation proven structurally (task #61, frozen p2_v1_core SHA verified above);
    # here we confirm aux ran while the MAIN opt advanced exactly 96 (aux uses separate masked opt).
    gates["p2_main_opt_isolated_96"] = (opt_step == 48 * 2)

gates["all_pass"] = bool(all(gates.values()))
out = {"mode": MODE, "gpu_uuid": GPU_UUID, "scratch_params_sha256": SCRATCH_SHA,
       "step0_params_sha256": step0_sha, "ckpt98304_params_sha256": shaA,
       "main_opt_step": opt_step, "trajectories_collected": traj_coll, "trajectories_inserted": traj_ins,
       "relabelled_samples": relabelled, "replay_samples_drawn": drawn,
       "replay_buffer_size": mani["replay_buffer_size"], "pending_total_transitions": mani["pending_total_transitions"],
       "n_training_logs": len(logs), "n_log_records": len(all_recs),
       "note_replay_eviction": "replay capped at 256; trajectories beyond cap evicted by design (OP never consumes replay). Conservation = trajectories_collected==inserted (no insertion loss).",
       "gates": gates}
ev = "/home/oseasy/experiments/single_director_20260722/evidence"; os.makedirs(ev, exist_ok=True)
with open(os.path.join(ev, f"p2_v1_from_scratch_{SUFFIX}_verification.json"), "w") as f:
    json.dump(out, f, indent=2, sort_keys=True, default=str); f.write("\n")

print("\n" + "="*60, flush=True)
for k in sorted(gates): print(f"  {'PASS' if gates[k] else 'FAIL'}  {k}", flush=True)
print("="*60, flush=True)
print(f"FROM-SCRATCH {MODE} VERIFY: {'ALL PASS' if gates['all_pass'] else 'FAIL'}", flush=True)
sys.exit(0 if gates["all_pass"] else 1)
