"""P2-Full-A EXACT-RESUME continuation trainer (24576 -> 98304 env steps), GPU0 only.

EXPLORATORY_DELAYED_ONSET_EXTENSION / plan-1 section 五. This is NOT a fresh train and
NOT a weights-only resume: it restores the COMPLETE P2-Full-A state at step 24576
(params, EMA target, optimizer, replay buffer incl anchors, pending episode buffers,
JAX rng_key, numpy action_rng bit-generator state, and the full collector state
env_state/obsv/memories/mem_mask/mem_idx) from the Level-B checkpoint, then continues
the SAME frozen protocol for 36 more rollouts (24576 -> 98304).

Because every source of nondeterminism is restored bit-exactly (and the env is stepped
forward from the restored env_state with NO env.reset), the action/rollout/update stream
is the exact continuation of the Level-B run that produced P2@24576 — as if it had never
stopped. Action sampling resumes from the saved PCG64 bit-generator state; the JAX rng
key resumes from the saved rng_key.

The Level-B P2@24576 checkpoint dir is READ-ONLY and is never written to or modified;
this resume writes its own checkpoints to a NEW dir with keep=0 (no stripping), saving
full state at {49152, 73728, 98304}, each followed by a bit-exact restore round-trip.

Fail-closed HARD STOPS (non-zero exit, no corrupted checkpoint written):
  * GPU bind error (visible device set != GPU0 UUID, or != 1 device)
  * output-dir reuse (run_dir/ckpt_dir not empty / log already present)
  * source P2@24576 checkpoint missing, OR restored params sha != frozen bd084220...,
    OR restored global_step != 24576 / update_count != 11
  * low disk (< 10 GB free before any save)
  * NaN/Inf in any update
  * checkpoint restore round-trip mismatch
  * trajectory-conservation break (collected != inserted)  [counters restored, not reset]
  * an ACCEPTED (policy_committed) update with policy_kl > kl_replay_max
  * a KL-rejected update that nonetheless reports policy_committed
  * entropy collapse (a fired update with entropy < ent_floor)
  * replay actor path unreachable while eligible data exists for >= 6 rollouts

Algorithm modules (compat_init / full_p2_core / full_p2_learner / replay_buffer /
pending_episodes / hindsight / memory_anchor / rng_utils / checkpointing) are used
UNCHANGED — this file is only the training HARNESS (exact-resume + longer horizon +
periodic checkpointing + hard-stop gates + conservation bookkeeping).
"""
import os
GPU_UUID = "GPU-e8c08612-c22a-c8a4-6df5-affb2dd1f9a6"   # GPU0
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID            # MUST precede jax import
import sys, os.path, json, argparse, shutil

BASE_SRC = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
HENRY_SRC = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src"
if BASE_SRC in sys.path:
    sys.path.remove(BASE_SRC)
sys.path.insert(0, BASE_SRC)
if HENRY_SRC not in sys.path:
    sys.path.insert(1, HENRY_SRC)

import numpy as np
import jax
import jax.numpy as jnp

# ---- real Stage4-native env (Henry / craftax / minicraftax only) ----
from craftax.craftax.constants import Achievement
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from dicode.task_utils import get_achievement_multi_hot
from dicode.wrappers_cl import DistributedMultiTaskOptimisticLogWrapper
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv

# ---- P2-Full-A modules (BASE_SRC wins; unchanged) ----
import compat_init as CI
import rng_utils as RU
import memory_anchor as MA
import hindsight as H
import full_p2_core as CORE
import full_p2_learner as FL
from full_p2_learner import FullP2Config, build_optimizer, full_p2_update
from replay_buffer import ReplayBuffer
from pending_episodes import PendingEpisodeBuffers
import checkpointing as CK

S4_TASK_PATH = "/home/oseasy/experiments/p2_v1_20260722/evidence/s4_task_code.py"

# ---- frozen constants (identical protocol to Level B) ----
NUM_ENVS = 16
ROLLOUT_STEPS = 128                       # 1 rollout == NUM_ENVS*ROLLOUT_STEPS == 2048 env steps
STEPS_PER_ROLLOUT = NUM_ENVS * ROLLOUT_STEPS
K_BATCH = 4                               # sampled windows per update
L_SEQ = 129                               # loss-window length
OPTIMISTIC_RESET_RATIO = 16
MASTER_SEED = 42
RESUME_LR = 2e-5
DISK_SAFE_BYTES = 10 * 1024 ** 3          # require >= 10 GB free before each save
UNREACHABLE_GUARD_ROLLOUTS = 6           # eligible-but-no-update streak that trips the guard

# ---- frozen RESUME source: the Level-B P2-Full-A checkpoint at step 24576 (READ-ONLY) ----
P2_24576_CKPT_DIR = "/home/oseasy/experiments/p2_full_20260723/checkpoints/p2_full_levelB_24576_20260724"
P2_24576_STEP = 24576
P2_24576_PARAMS_SHA = "bd08422042788f6322b76d0963042598d6868c1cedf5756178492c2215a10d28"
P2_24576_UPDATE_COUNT = 11
START_STEP = 24576
END_STEP = 98304
SAVE_STEPS_RESUME = (49152, 73728, 98304)


def _metric_val(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (list, dict, tuple)):
        return v
    if isinstance(v, str):
        return v
    if np.ndim(v) == 0:
        return float(v)
    return str(v)


def build_env(seed):
    ach_table = jnp.array(
        [get_achievement_multi_hot([Achievement.DEFEAT_KOBOLD])], dtype=jnp.float32)
    emb = int(ach_table.shape[1])
    with open(S4_TASK_PATH) as f:
        s4_code = f.read()
    ns = {}
    exec(s4_code, ns)
    Task = ns["Env"]
    static_env_params = StaticEnvParams()
    env_params = EnvParams(max_timesteps=4096)
    base_env = MultiTaskMiniCraftaxEnv(
        [Task], static_env_params, env_params, True,
        conditioning_type="embedding", embedding_size=emb,
        completion_bonus_scale=0.0, completion_bonus_min=0.0,
        bonus_type="none", dynamic_bonus_k=0.0)
    env = DistributedMultiTaskOptimisticLogWrapper(
        base_env, jax.random.PRNGKey(seed), NUM_ENVS, 1,
        OPTIMISTIC_RESET_RATIO, jnp.array([1.0]), ach_table)

    class _EnvAdapter:
        def step(self, step_rng, env_state, actions):
            return env.step(step_rng, env_state, actions, env_params)
        def reset(self, reset_rng, ep=None):
            return env.reset(reset_rng, env_params)
    return _EnvAdapter(), env_params, np.asarray(ach_table[0]).astype(np.float32), emb


# ----------------------------- hard-stop guards -----------------------------

def guard_gpu_bind():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == GPU_UUID, \
        "HARD STOP gpu-bind: CUDA_VISIBLE_DEVICES != GPU0 UUID"
    devs = jax.devices()
    assert len(devs) == 1, f"HARD STOP gpu-bind: expected 1 visible device, got {devs}"


def guard_fresh_dirs(run_dir, ckpt_dir):
    assert not os.path.exists(os.path.join(run_dir, "training_log.jsonl")), \
        f"HARD STOP dir-reuse: {run_dir}/training_log.jsonl already exists"
    for d in (run_dir, ckpt_dir):
        if os.path.isdir(d):
            assert not os.listdir(d), f"HARD STOP dir-reuse: {d} is not empty"


def guard_disk(path):
    free = shutil.disk_usage(path).free
    assert free > DISK_SAFE_BYTES, \
        f"HARD STOP disk-low: {free / 1024 ** 3:.1f} GB free < {DISK_SAFE_BYTES / 1024 ** 3:.0f} GB required at {path}"


def guard_source_checkpoint():
    pkl = os.path.join(P2_24576_CKPT_DIR, str(P2_24576_STEP), "full_state.pkl")
    assert os.path.exists(pkl), \
        f"HARD STOP source-missing: P2@24576 full_state.pkl not found at {pkl}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--seed", type=int, default=MASTER_SEED)
    ap.add_argument("--lr", type=float, default=RESUME_LR)
    ap.add_argument("--end_step", type=int, default=END_STEP)
    args = ap.parse_args()

    assert args.end_step == END_STEP, \
        f"HARD STOP budget: end_step={args.end_step} != {END_STEP} (not authorized this round)"

    os.makedirs(args.run_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    guard_gpu_bind()
    guard_fresh_dirs(args.run_dir, args.ckpt_dir)
    guard_disk(args.ckpt_dir)
    guard_source_checkpoint()

    log_path = os.path.join(args.run_dir, "training_log.jsonl")
    logf = open(log_path, "a")

    def log(rec):
        logf.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
        logf.flush()

    cfg = FullP2Config()
    assert cfg.adam_eps == 1e-5, f"HARD STOP fairness: cfg.adam_eps={cfg.adam_eps} != 1e-5"
    assert cfg.gamma == 0.999, f"HARD STOP fairness: gamma={cfg.gamma}"
    assert abs(args.lr - 2e-5) < 1e-12, f"HARD STOP fairness: lr={args.lr} != 2e-5"

    num_rollouts_resume = (args.end_step - START_STEP) // STEPS_PER_ROLLOUT
    save_steps = tuple(s for s in SAVE_STEPS_RESUME if s <= args.end_step)

    budget = dict(num_envs=NUM_ENVS, rollout_steps=ROLLOUT_STEPS,
                  num_rollouts_resume=num_rollouts_resume,
                  steps_per_rollout=STEPS_PER_ROLLOUT,
                  start_step=START_STEP, end_step=args.end_step,
                  resume_env_steps=args.end_step - START_STEP,
                  save_steps=list(save_steps))
    if args.end_step == END_STEP:                 # the frozen resume budget, asserted exactly
        assert budget["num_envs"] == 16, budget
        assert budget["rollout_steps"] == 128, budget
        assert budget["num_rollouts_resume"] == 36, budget
        assert budget["resume_env_steps"] == 73728, budget
        assert save_steps == (49152, 73728, 98304), save_steps
    log({"event": "budget", **budget})

    # 1. compatible init — network architecture + apply fns ONLY (params NOT taken from here)
    ci = CI.compatible_init(strict=True)
    network = ci["network"]
    a_rec, a_raw, scan_fn = ci["apply_eval_recon"], ci["apply_eval_raw"], ci["scan_fn"]
    jit_fwd = CORE.make_jit_forward(network)
    log({"event": "compatible_init", "fingerprint": ci["fingerprint"],
         "source_checkpoint": ci["source_checkpoint"], "kernels": ci["kernels"]})
    print("[resume] compatible_init OK  network_sha=%s params=%d" % (
        ci["fingerprint"]["params_sha256"], ci["fingerprint"]["param_count"]), flush=True)

    # 2. env (Stage4-native, DEFEAT_KOBOLD, seed=42) — env object only; state is RESTORED
    env, env_params, target_achievement, emb = build_env(args.seed)
    assert emb == cfg.embed or emb == 67, emb

    # 3. EXACT-RESUME: restore the COMPLETE P2@24576 state (no env.reset, no fresh init)
    ck = CK.restore_full_checkpoint(P2_24576_CKPT_DIR, step=P2_24576_STEP)

    # ---- source gate: refuse to resume from anything but the frozen P2@24576 params ----
    restored_params_sha = CK.params_content_sha256(ck["params"])
    assert restored_params_sha == P2_24576_PARAMS_SHA, (
        "HARD STOP source-mismatch: restored P2@24576 params_sha=%s != frozen %s" % (
            restored_params_sha, P2_24576_PARAMS_SHA))
    assert ck["global_step"] == START_STEP, \
        f"HARD STOP source-step: restored global_step={ck['global_step']} != {START_STEP}"
    assert ck["update_count"] == P2_24576_UPDATE_COUNT, \
        f"HARD STOP source-update_count: {ck['update_count']} != {P2_24576_UPDATE_COUNT}"
    assert ck["collector_state"] is not None, "HARD STOP source: collector_state missing"
    assert ck["pending"] is not None, "HARD STOP source: pending buffers missing"
    assert ck["action_rng_state"] is not None, "HARD STOP source: action_rng_state missing"

    params = ck["params"]
    target_params = ck["target_params"]
    opt_state = ck["opt_state"]
    replay = ck["replay_buffer"]               # ReplayBuffer (counters preserved, NOT reset)
    pending = ck["pending"]                    # PendingEpisodeBuffers (anchors preserved)
    rng = ck["rng_key"]                        # JAX rng key, exact
    action_rng = RU.restore_action_rng(ck["action_rng_state"], seed=args.seed)  # PCG64 state, exact
    cs = ck["collector_state"]
    env_state = cs["env_state"]                # NO env.reset — resume mid-stream
    obsv = cs["obsv"]
    memories = cs["memories"]
    mem_mask = cs["mem_mask"]
    mem_idx = cs["mem_idx"]
    global_step = int(ck["global_step"])       # 24576
    update_count = int(ck["update_count"])     # 11

    # optimizer object for the update RULE (opt_state is restored, not re-init'd)
    opt = build_optimizer(args.lr, cfg)        # adam eps = cfg.adam_eps = 1e-5

    src_manifest = ck["manifest"]
    log({"event": "restore",
         "source_dir": P2_24576_CKPT_DIR, "source_step": P2_24576_STEP,
         "params_sha256": restored_params_sha,
         "global_step": global_step, "update_count": update_count,
         "replay_size": len(replay),
         "trajectories_collected": replay.counters.trajectories_collected,
         "trajectories_inserted": replay.counters.trajectories_inserted,
         "total_anchors_stored": replay.counters.total_anchors_stored,
         "pending_transitions": pending.total_pending_transitions(),
         "pending_anchors": pending.total_pending_anchors(),
         "action_rng_restored": True, "collector_state_restored": True,
         "source_manifest_sha256": src_manifest.get("params_sha256")})
    print("[resume] EXACT-RESUME from P2@24576  params_sha=%s  step=%d  update_count=%d  "
          "replay=%d  pending=%d/%d  obsv=%s" % (
        restored_params_sha[:16], global_step, update_count, len(replay),
        pending.total_pending_transitions(), pending.total_pending_anchors(),
        np.asarray(obsv).shape), flush=True)

    # ---- counters: replay.counters are restored (conservation continues); harness tallies
    #      carry the cumulative Level-B values so the final summary spans 24576->98304 ----
    total_episodes = 0                          # segment-local
    total_updates = 0                           # segment-local
    accepted_policy_updates = int(src_manifest.get("accepted_policy_updates", update_count))
    kl_rejected_updates = int(src_manifest.get("kl_rejected_updates", 0))
    hindsight_attempts = int(src_manifest.get("hindsight_attempts", 0))
    hindsight_eligible = int(src_manifest.get("hindsight_eligible", 0))
    eligible_no_update_streak = 0
    any_nan = False
    saved_shas = {}

    def _collector_state():
        return {"env_state": env_state, "obsv": obsv, "memories": memories,
                "mem_mask": mem_mask, "mem_idx": mem_idx}

    def _save(step):
        nonlocal saved_shas
        em = {"label": "P2-Full-A-Resume-98304",
              "experiment": "EXPLORATORY_DELAYED_ONSET_EXTENSION",
              "resume_from_step": START_STEP, "resume_to_step": args.end_step,
              "source_p2_24576_params_sha256": P2_24576_PARAMS_SHA,
              "source_p2_24576_dir": P2_24576_CKPT_DIR,
              "gpu_uuid": GPU_UUID, "seed": args.seed, "lr": args.lr,
              "adam_eps": cfg.adam_eps, "budget": budget,
              "accepted_policy_updates": accepted_policy_updates,
              "kl_rejected_updates": kl_rejected_updates,
              "hindsight_attempts": hindsight_attempts, "hindsight_eligible": hindsight_eligible}
        guard_disk(args.ckpt_dir)
        CK.save_full_checkpoint(
            params, target_params, opt_state, replay, rng, global_step=step,
            path=args.ckpt_dir, step=step,
            action_rng_state=RU.action_rng_state(action_rng), update_count=update_count,
            pending=pending, collector_state=_collector_state(), config=cfg,
            keep=0, extra_manifest=em)            # keep=0 -> strip NOTHING (all saves retained)
        saved_sha = CK.params_content_sha256(params)
        restored = CK.restore_full_checkpoint(args.ckpt_dir, step=step)
        restored_sha = CK.params_content_sha256(restored["params"])
        roundtrip_ok = (restored_sha == saved_sha
                        and restored["global_step"] == step
                        and restored["update_count"] == update_count
                        and restored["pending"].total_pending_transitions()
                            == pending.total_pending_transitions())
        assert roundtrip_ok, f"HARD STOP roundtrip mismatch at step {step}"
        saved_shas[step] = saved_sha
        log({"event": "checkpoint", "step": step, "params_sha256": saved_sha,
             "restored_sha256": restored_sha, "roundtrip_ok": True,
             "update_count": update_count,
             "replay_size": len(replay),
             "pending_transitions": pending.total_pending_transitions(),
             "pending_anchors": pending.total_pending_anchors()})
        print("[resume] checkpoint step=%d  sha=%s  roundtrip=OK" % (step, saved_sha[:16]),
              flush=True)

    for r in range(num_rollouts_resume):
        trajs, carry, stats = CORE.collect_rollout(
            env, env_state, network, params, obsv,
            memories, mem_mask, mem_idx, rng, action_rng,
            pending, target_achievement, rollout_steps=ROLLOUT_STEPS,
            window_mem=cfg.window_mem, num_heads=cfg.num_heads,
            collected_update_count=update_count, jit_forward=jit_fwd)
        env_state = carry["env_state"]; obsv = carry["obsv"]
        memories = carry["memories"]; mem_mask = carry["mem_mask"]
        mem_idx = carry["mem_idx"]; rng = carry["rng"]
        global_step += STEPS_PER_ROLLOUT
        total_episodes += len(trajs)

        # ---- terminal-episode boundary integrity (each collected traj must be a single
        #      done-terminated episode; insert() also fail-closed-rejects non-terminal) ----
        for t in trajs:
            assert bool(np.asarray(t.dones)[-1]), "HARD STOP episode-boundary: collected non-terminal trajectory"
            replay.insert(t)                        # raises ValueError on bad anchors / non-terminal
            replay.counters.trajectories_collected += 1

        # ---- trajectory conservation: every collected trajectory is inserted ----
        assert replay.counters.trajectories_collected == replay.counters.trajectories_inserted, \
            ("HARD STOP conservation: collected=%d != inserted=%d" % (
                replay.counters.trajectories_collected, replay.counters.trajectories_inserted))

        # ---- try a combined update on relabelable windows ----
        update_metrics = None
        updates_before = total_updates
        if replay.can_sample():
            so, sr = [], []
            for _ in range(K_BATCH):
                hindsight_attempts += 1
                s = replay.sample(sequence_length=L_SEQ)
                try:
                    rel = H.relabel_sample(s)       # goal_index=None -> min achieved (Gate 5/6)
                except ValueError:
                    continue                        # not relabelable
                hindsight_eligible += 1
                so.append(s); sr.append(rel)
            if len(so) >= 2:                        # lax.scan needs B>=2
                params, target_params, opt_state, m = full_p2_update(
                    params, target_params, opt_state, opt, a_rec, a_raw, scan_fn,
                    so, sr, cfg, update_count)
                update_count += 1
                total_updates += 1
                finite = bool(m["finite"])
                any_nan = any_nan or (not finite)
                assert finite, "HARD STOP NaN/Inf in update loss"
                # accepted update must obey the transactional KL gate
                if bool(m.get("policy_committed")):
                    assert float(m["policy_kl"]) <= cfg.kl_replay_max + 1e-12, \
                        "HARD STOP accepted update policy_kl=%.5f > kl_replay_max=%.3f" % (
                            float(m["policy_kl"]), cfg.kl_replay_max)
                    accepted_policy_updates += 1
                # a KL-rejected update must NOT report policy_committed
                if bool(m.get("kl_rejected_update")):
                    assert not bool(m.get("policy_committed")), \
                        "HARD STOP KL-rejected update committed policy params"
                    kl_rejected_updates += 1
                # entropy collapse guard (frozen floor)
                assert float(m["entropy"]) >= cfg.ent_floor, \
                    "HARD STOP entropy collapse: %.4f < floor %.3f" % (
                        float(m["entropy"]), cfg.ent_floor)
                update_metrics = {k: _metric_val(v) for k, v in m.items()}
                update_metrics["finite"] = finite
                update_metrics["batch"] = len(so)

        # ---- replay-actor-unreachable guard ----
        if replay.can_sample() and total_updates == updates_before:
            eligible_no_update_streak += 1
        else:
            eligible_no_update_streak = 0
        assert eligible_no_update_streak < UNREACHABLE_GUARD_ROLLOUTS, \
            ("HARD STOP replay-actor unreachable: %d consecutive rollouts with eligible "
             "data but no update fired" % eligible_no_update_streak)

        log({"event": "rollout", "rollout": r, "global_step": global_step,
             "completed_episodes": stats["completed_episodes"],
             "mean_ep_return": stats["mean_ep_return"],
             "mean_ep_length": stats["mean_ep_length"],
             "pending_transitions": stats["pending_transitions"],
             "pending_anchors": stats["pending_anchors"],
             "replay_size": len(replay),
             "replay_can_sample": replay.can_sample(),
             "trajectories_collected": replay.counters.trajectories_collected,
             "trajectories_inserted": replay.counters.trajectories_inserted,
             "total_anchors_stored": replay.counters.total_anchors_stored,
             "conservation_ok": replay.counters.trajectories_collected == replay.counters.trajectories_inserted,
             "update_count": update_count,
             "accepted_policy_updates": accepted_policy_updates,
             "kl_rejected_updates": kl_rejected_updates,
             "hindsight_attempts": hindsight_attempts,
             "hindsight_eligible": hindsight_eligible,
             "eligible_no_update_streak": eligible_no_update_streak,
             "update": update_metrics})
        print("[resume] rollout %d  step=%d  episodes=%d  replay=%d  updates=%d  accepted=%d%s" % (
            r, global_step, stats["completed_episodes"], len(replay), total_updates,
            accepted_policy_updates,
            ("  loss=%.4f kl=%.4f" % (update_metrics["loss"], update_metrics["policy_kl"]))
            if update_metrics else "  (no update)"), flush=True)

        # ---- periodic checkpoint at the frozen resume save steps ----
        if global_step in save_steps:
            _save(global_step)

    # ---- final fail-closed gates ----
    assert not any_nan, "HARD STOP NaN/Inf encountered during resume"
    assert global_step == args.end_step, f"HARD STOP ran to {global_step} != {args.end_step}"
    assert replay.counters.trajectories_collected == replay.counters.trajectories_inserted, \
        "HARD STOP final conservation mismatch"
    assert 98304 in saved_shas, "HARD STOP final checkpoint at 98304 was not saved"

    summary = {
        "label": "P2-Full-A-Resume-98304",
        "experiment": "EXPLORATORY_DELAYED_ONSET_EXTENSION",
        "resume_from_step": START_STEP, "resume_to_step": args.end_step,
        "source_p2_24576_params_sha256": P2_24576_PARAMS_SHA,
        "source_p2_24576_dir": P2_24576_CKPT_DIR,
        "gpu_uuid": GPU_UUID, "seed": args.seed, "lr": args.lr, "adam_eps": cfg.adam_eps,
        "global_step": global_step, "update_count": update_count,
        "segment_episodes": total_episodes, "segment_updates": total_updates,
        "accepted_policy_updates": accepted_policy_updates,
        "kl_rejected_updates": kl_rejected_updates,
        "hindsight_attempts": hindsight_attempts, "hindsight_eligible": hindsight_eligible,
        "kl_replay_max": cfg.kl_replay_max, "kl_run_max": cfg.kl_run_max,
        "any_nan_or_inf": any_nan,
        "trajectories_collected": replay.counters.trajectories_collected,
        "trajectories_inserted": replay.counters.trajectories_inserted,
        "total_anchors_stored": replay.counters.total_anchors_stored,
        "conservation_ok": replay.counters.trajectories_collected == replay.counters.trajectories_inserted,
        "final_params_sha256": saved_shas.get(98304),
        "saved_checkpoints": {str(k): v for k, v in sorted(saved_shas.items())},
        "replay_size": len(replay), "replay_can_sample": replay.can_sample(),
        "pending_transitions": pending.total_pending_transitions(),
        "pending_anchors": pending.total_pending_anchors(),
        "budget": budget, "num_envs": NUM_ENVS, "rollout_steps": ROLLOUT_STEPS,
        "network_fingerprint": ci["fingerprint"],
    }
    with open(os.path.join(args.run_dir, "resume_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    log({"event": "summary", **summary})
    logf.close()

    print("RESUME_OK steps=%d->%d segment_episodes=%d segment_updates=%d accepted=%d "
          "rejected=%d conservation=%s nan=%s checkpoints=%s final_sha=%s" % (
        START_STEP, global_step, total_episodes, total_updates, accepted_policy_updates,
        kl_rejected_updates, summary["conservation_ok"], any_nan,
        sorted(saved_shas.keys()), str(saved_shas.get(98304))[:16]), flush=True)


if __name__ == "__main__":
    main()
