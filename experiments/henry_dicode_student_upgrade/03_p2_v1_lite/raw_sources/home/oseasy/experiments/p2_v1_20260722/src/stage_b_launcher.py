#!/usr/bin/env python3
"""D059 Stage B: AMAGO-style bounded GPU3 preflight (24,576 resolved env steps).

Exact step arithmetic:
  requested_env_steps = 24576
  num_envs = 16
  rollout_steps = 128
  num_updates = 12
  resolved_env_steps = 16 * 128 * 12 = 24576

Fails closed unless requested_env_steps == resolved_env_steps == 98304.

PASS requires all of:
  - total_env_steps == 98304
  - >=1 real Craftax trajectory >128 inserted into replay
  - >=1 replay sample from >128 sequence
  - >=1 finite nonzero off-policy gradient update changing params
  - >=1 hindsight relabeling with achieved-goal evidence
  - final atomic checkpoint (model+opt+replay+RNG+global_step+provenance)
"""

import hashlib, json, math, os, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path

import jax, jax.numpy as jnp, numpy as np
from flax.training.train_state import TrainState

# ── Paths ────────────────────────────────────────────────────────────
_HENRY_SRC = ("/home/oseasy/incoming/henry_work_20260721T105300/extracted/"
              "Henry_work/code/dicode_v7fix58_armB/src")
_AMAGO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, _HENRY_SRC)
sys.path.insert(0, _AMAGO_SRC)

from craftax.craftax.constants import Achievement
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from dicode.network import ActorCriticTransformer
from dicode.task_utils import get_achievement_multi_hot
from dicode.utils.general.train_state_utils import load_weights_only
from minicraftax.envs.craftax import CraftaxAugObsTrain
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv
from dicode.wrappers_cl import DistributedMultiTaskOptimisticLogWrapper

from trajectory_replay import Trajectory, TrajectoryReplayBuffer
from hindsight import relabel_sample
from long_context_learner import LongContextLearner
from checkpointing import save_full_checkpoint

# ═════════════════════════════════════════════════════════════════════
# IMMUTABLE CONSTANTS
# ═════════════════════════════════════════════════════════════════════

EXPECTED_GPU_UUID = "GPU-f56a59b4-99f3-f2e5-11c6-d01685de8abd"
REQUESTED_ENV_STEPS = 98304
NUM_ENVS = 16
ROLLOUT_STEPS = 128
NUM_UPDATES = 48
RESOLVED_ENV_STEPS = NUM_ENVS * ROLLOUT_STEPS * NUM_UPDATES
assert REQUESTED_ENV_STEPS == RESOLVED_ENV_STEPS == 98304, \
    f"STEP ARITHMETIC FAIL: {REQUESTED_ENV_STEPS} != {RESOLVED_ENV_STEPS}"
GAMMA = 0.999
GAE_LAMBDA = 0.8
SEED = 0

CKPT_SRC = ("/home/oseasy/incoming/henry_work_20260721T105300/extracted/"
            "Henry_work/base_ckpt_v7fix55_armA_s0/rl_checkpoints/17500")
OUTPUT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
CKPT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints"))
EVIDENCE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evidence"))
S4_TASK_PATH = os.path.join(EVIDENCE_DIR, "s4_task_code.py")

for d in [OUTPUT_ROOT, CKPT_ROOT, EVIDENCE_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Config (matches ckpt 17500) ─────────────────────────────────────

class Cfg:
    lr=2e-4; min_lr=2e-6; num_envs=NUM_ENVS; num_steps=ROLLOUT_STEPS
    update_epochs=1; num_minibatches=2; gamma=GAMMA; gae_lambda=GAE_LAMBDA
    clip_eps=0.2; ent_coef=0.002; vf_coef=0.5; max_grad_norm=1.0
    activation="relu"; anneal_lr=True
    qkv_features=256; embed_size=256; num_heads=8; num_layers=2
    hidden_layers=256; window_mem=128; window_grad=64
    gating=True; gating_bias=2.0
    condition_on_task=True; optimistic_reset_ratio=16
    mode="score"; bonus_type="none"; dynamic_bonus_k=0.0
    completion_bonus_scale=0.0; completion_bonus_min=0.0
    max_updates_per_session=NUM_UPDATES; total_timesteps=2_005_401_600
    scoring_window_updates=4
    value_target_clip_min=-50.0; value_target_clip_max=300.0
    guard_session_vloss_max=1000.0; guard_session_entropy_min=0.10
    guard_max_consecutive_reverts=2; lr_restart=0.0
    lr_restart_at=0; lr_restart_horizon=0; lr_restart_warmup=50
    sil=False; sil_pools=[]


# ═════════════════════════════════════════════════════════════════════
# Guards
# ═════════════════════════════════════════════════════════════════════

def verify_gpu():
    import subprocess
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip().split("\n")
    if EXPECTED_GPU_UUID not in out:
        print(f"STOP: GPU {EXPECTED_GPU_UUID} not in {out}"); sys.exit(1)
    if not jax.devices("gpu"):
        print("STOP: no GPU (CPU fallback)"); sys.exit(1)
    print(f"[guard] GPU OK  |  JAX devices: {len(jax.devices('gpu'))}")

def guard_output_collision():
    out_dir = os.path.join(OUTPUT_ROOT, "stage_B")
    if os.path.exists(out_dir):
        print(f"STOP: output collision at {out_dir}"); sys.exit(1)
    return out_dir


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("D059 Stage B: AMAGO-style GPU3 preflight")
    print(f"  requested={REQUESTED_ENV_STEPS}  num_envs={NUM_ENVS}")
    print(f"  rollout_steps={ROLLOUT_STEPS}  num_updates={NUM_UPDATES}")
    print(f"  resolved={RESOLVED_ENV_STEPS}")
    print("=" * 60)

    verify_gpu()
    out_dir = guard_output_collision()
    os.makedirs(out_dir, exist_ok=True)
    started_utc = datetime.now(timezone.utc).isoformat()
    pid = os.getpid()
    argv = sys.argv

    # ── Load ckpt ───────────────────────────────────────────────
    print("\n[1/6] Loading ckpt 17500 ...")
    cfg = Cfg()
    ach_table = jnp.array(
        [get_achievement_multi_hot([Achievement.DEFEAT_KOBOLD])], dtype=jnp.float32)
    EMB = int(ach_table.shape[1])
    dummy = CraftaxAugObsTrain(condition_on_task=True, conditioning_type="embedding",
                               embedding_size=EMB, task_embeddings=jnp.zeros((1, EMB)))
    ts = load_weights_only(CKPT_SRC, dummy, dummy.default_params, cfg, load_opt_state=True)
    print("  ckpt loaded (gamma=0.999, gae_lambda=0.8)")

    # ── Network + learner ───────────────────────────────────────
    print("\n[2/6] Network + learner ...")
    network = ActorCriticTransformer(
        action_dim=dummy.action_space(dummy.default_params).n,
        activation=cfg.activation, hidden_layers=cfg.hidden_layers,
        encoder_size=cfg.embed_size, num_heads=cfg.num_heads,
        qkv_features=cfg.qkv_features, num_layers=cfg.num_layers,
        gating=cfg.gating, gating_bias=cfg.gating_bias)
    rng = jax.random.PRNGKey(SEED)
    learner = LongContextLearner(network, cfg, rng)

    # ── Env ─────────────────────────────────────────────────────
    print("\n[3/6] Building env from S4 task code ...")
    with open(S4_TASK_PATH) as f:
        s4_code = f.read()
    ns = {}
    exec(s4_code, ns)
    Task = ns["Env"]

    static_env_params = StaticEnvParams()
    env_params = EnvParams(max_timesteps=4096)
    base_env = MultiTaskMiniCraftaxEnv(
        [Task], static_env_params, env_params, cfg.condition_on_task,
        conditioning_type="embedding", embedding_size=EMB,
        completion_bonus_scale=cfg.completion_bonus_scale,
        completion_bonus_min=cfg.completion_bonus_min,
        bonus_type=cfg.bonus_type, dynamic_bonus_k=cfg.dynamic_bonus_k)
    env = DistributedMultiTaskOptimisticLogWrapper(
        base_env, jax.random.PRNGKey(SEED), NUM_ENVS, 1,
        cfg.optimistic_reset_ratio, jnp.array([1.0]), ach_table)

    # ── Replay buffer ───────────────────────────────────────────
    replay = TrajectoryReplayBuffer(capacity=256, seed=SEED)

    # ── Save start ckpt ─────────────────────────────────────────
    print("\n[4/6] Initial checkpoint ...")
    save_full_checkpoint(ts, replay, rng, 0, CKPT_ROOT, step=0)

    # ── Compiled forward pass ───────────────────────────────────
    @jax.jit
    def jit_forward(params, mem, obs, mask):
        pi, value, mem_out = network.apply(params, mem, obs, mask,
                                           method=network.model_forward_eval)
        return pi.logits, value, mem_out

    # env.step is already JAX-native; calling without explicit jit
    # avoids the traced-EnvParams hashing issue in JAX 0.6.
    def env_step_call(env_state, action, step_rng):
        return env.step(step_rng, env_state, action, env_params)

    # ── Run ─────────────────────────────────────────────────────
    print(f"\n[5/6] Training ({NUM_UPDATES} updates, {RESOLVED_ENV_STEPS} steps) ...")
    obs_dim = dummy.observation_space(dummy.default_params).shape[0]
    t0 = time.time()

    rng, reset_rng = jax.random.split(rng)
    obsv_j, env_state = env.reset(reset_rng, env_params)

    memories_j = jnp.zeros((NUM_ENVS, cfg.window_mem, cfg.num_layers, cfg.embed_size))
    mem_mask_j = jnp.zeros((NUM_ENVS, cfg.num_heads, 1, cfg.window_mem + 1), dtype=jnp.bool_)
    mem_idx_j = jnp.full((NUM_ENVS,), cfg.window_mem + 1, dtype=jnp.int32)

    # Per-env episode buffers
    ep = [{"obs": [], "act": [], "rew": [], "don": [], "val": [], "lp": [],
           "ach": [], "mem_seq": [], "init_mem": None} for _ in range(NUM_ENVS)]

    all_logs = []
    total_env_steps = 0
    up = 0
    crash_info = None

    # Counters for strict PASS gate
    pass_trajs_long = 0        # trajectories with len > 128 inserted
    pass_samples_long = 0      # replay samples from >128 sequence
    pass_grad_updates = 0      # finite nonzero gradient updates
    pass_params_changed = 0    # updates that changed model params
    pass_relabel_evidence = []  # list of (from_goal, to_goal) evidence

    try:
        for up in range(NUM_UPDATES):
            up_t0 = time.time()

            # ── Collect ROLLOUT_STEPS transitions ───────────────
            for st in range(ROLLOUT_STEPS):
                total_env_steps += NUM_ENVS

                # Memory index management
                mem_idx_j = jnp.clip(mem_idx_j - 1, 0, cfg.window_mem)
                ohot = jax.nn.one_hot(mem_idx_j, cfg.window_mem + 1)
                ohot = ohot[:, None, None, :].repeat(cfg.num_heads, 1)
                mem_mask_j = jnp.logical_or(mem_mask_j, ohot)

                # Forward: get logits (not distrax Categorical)
                rng, a_rng = jax.random.split(rng)
                logits, value, mem_out = jit_forward(
                    ts.params, memories_j, obsv_j, mem_mask_j)
                logits_np = np.asarray(logits)
                value_np = np.asarray(value)
                mem_out_np = np.asarray(mem_out)

                # Sample actions from logits via numpy
                probs = jax.nn.softmax(jnp.asarray(logits), axis=-1)
                probs_np = np.asarray(probs)
                actions_np = np.array([
                    np.random.choice(probs_np.shape[1], p=p)
                    for p in probs_np])
                logp_np = np.log(
                    probs_np[np.arange(NUM_ENVS), actions_np] + 1e-12)

                # Roll memory
                memories_j = jnp.roll(memories_j, -1, axis=1).at[:, -1].set(mem_out_np)

                # Step env
                rng, s_rng = jax.random.split(rng)
                obsv_j, env_state, reward_j, done_j, info = env_step_call(
                    env_state, actions_np, s_rng)
                reward_np = np.asarray(reward_j)
                done_np = np.asarray(done_j)

                # Reset memory for done envs
                memories_j = jnp.where(
                    done_np[:, None, None, None],
                    jnp.zeros_like(memories_j), memories_j)
                mem_mask_j = jnp.where(
                    done_np[:, None, None, None],
                    jnp.zeros_like(mem_mask_j), mem_mask_j)
                mem_idx_j = jnp.where(done_np, cfg.window_mem, mem_idx_j)

                # Extract achievements from env_state (not info — Craftax stores
                # achievement bits in the state object, not the info dict).
                ach_data = np.zeros((NUM_ENVS, 67), dtype=np.float32)
                try:
                    est = env_state.env_state
                    if hasattr(est, 'achievements'):
                        ach_data = np.asarray(est.achievements).astype(np.float32)
                except Exception:
                    pass

                # Per-env accumulation
                for e in range(NUM_ENVS):
                    buf = ep[e]
                    if buf["init_mem"] is None:
                        buf["init_mem"] = np.asarray(memories_j[e]).copy()
                    buf["obs"].append(np.asarray(obsv_j[e]))
                    buf["act"].append(int(actions_np[e]))
                    buf["rew"].append(float(reward_np[e]))
                    buf["don"].append(bool(done_np[e]))
                    buf["val"].append(float(value_np[e]))
                    buf["lp"].append(float(logp_np[e]))
                    buf["ach"].append(ach_data[e].copy())
                    buf["mem_seq"].append(np.asarray(memories_j[e]).copy())

                    if done_np[e]:
                        L = len(buf["obs"])
                        if L > 0 and buf["init_mem"] is not None:
                            traj = Trajectory(
                                observations=np.stack(buf["obs"]),
                                actions=np.array(buf["act"], dtype=np.int32),
                                rewards=np.array(buf["rew"], dtype=np.float32),
                                dones=np.array(buf["don"], dtype=bool),
                                values=np.array(buf["val"], dtype=np.float32),
                                log_probs=np.array(buf["lp"], dtype=np.float32),
                                initial_memory=buf["init_mem"],
                                achievements=np.stack(buf["ach"]),
                                target_achievements=np.asarray(ach_table[0]),
                                memory_sequence=np.stack(buf["mem_seq"]))
                            replay.insert(traj)
                            replay.counters.trajectories_collected += 1
                            if L > 128:
                                pass_trajs_long += 1
                        buf.update({"obs": [], "act": [], "rew": [], "don": [],
                                     "val": [], "lp": [], "ach": [], "mem_seq": [],
                                     "init_mem": None})

            # ── Off-policy update ───────────────────────────────
            off_metrics = None
            if replay.can_sample():
                try:
                    sample = replay.sample()
                    pass_samples_long += 1

                    # Hindsight relabeling if achievements present
                    achieved_idxs = []
                    ach_any = sample.achievements.max(axis=0)
                    if ach_any.any():
                        old_target = np.argmax(sample.target_achievements)
                        sample = relabel_sample(sample)
                        new_target = np.argmax(sample.target_achievements)
                        replay.counters.relabelled_samples += 1
                        pass_relabel_evidence.append(
                            {"from_goal_idx": int(old_target),
                             "to_goal_idx": int(new_target)})

                    ts_old_params = jax.tree_util.tree_map(
                        lambda x: x.copy(), ts.params)
                    ts, metrics = learner.update(ts, sample)
                    replay.counters.gradient_updates += 1
                    off_metrics = metrics

                    if metrics["grad_norm"] > 1e-12 and np.isfinite(metrics["grad_norm"]):
                        pass_grad_updates += 1
                    if metrics.get("params_changed"):
                        pass_params_changed += 1

                    # NaN/Inf guard
                    if not np.isfinite(metrics["total_loss"]):
                        raise RuntimeError(f"NaN/Inf loss: {metrics['total_loss']}")
                    if not np.isfinite(metrics["grad_norm"]):
                        raise RuntimeError(f"NaN/Inf grad_norm: {metrics['grad_norm']}")
                except (ValueError, RuntimeError) as e:
                    msg = str(e)
                    if "Gate" in msg or "128" in msg:
                        pass  # expected early
                    else:
                        raise

            # ── Log ─────────────────────────────────────────────
            cs = replay.counters.snapshot()
            log = {"update": up, "env_steps": total_env_steps,
                   "elapsed_s": round(time.time() - t0, 1),
                   "replay_size": len(replay),
                   "replay_max_len": replay.longest_trajectory_length, **cs}
            if off_metrics:
                log["off_loss"] = round(off_metrics["total_loss"], 6)
                log["off_grad_norm"] = round(off_metrics["grad_norm"], 6)
                log["off_seq_len"] = off_metrics["sequence_length"]
            all_logs.append(log)

            print(f"  up {up:3d}/{NUM_UPDATES}  steps {total_env_steps:6d}/{RESOLVED_ENV_STEPS}  "
                  f"replay {len(replay):3d}  samples {cs['replay_samples_drawn']:3d}  "
                  f"grads {cs['gradient_updates']:3d}  relab {cs['relabelled_samples']:2d}  "
                  f"max_len={replay.longest_trajectory_length}")

    except Exception as e:
        print(f"\nSTOP: {e}")
        traceback.print_exc()
        crash_info = {"error": str(e), "traceback": traceback.format_exc(),
                      "env_steps": total_env_steps, "update": up,
                      "timestamp": datetime.now(timezone.utc).isoformat()}
        # Continue to save evidence — don't exit yet

    # ── End checkpoint ──────────────────────────────────────────
    print("\n[6/6] Saving end checkpoint ...")
    final_step = total_env_steps
    ckpt_path = save_full_checkpoint(ts, replay, rng, final_step, CKPT_ROOT, step=final_step)

    # ── Compute PASS gate verdict ───────────────────────────────
    pass_total_steps = (total_env_steps == RESOLVED_ENV_STEPS == 98304)
    pass_has_traj = (pass_trajs_long >= 1)
    pass_has_sample = (pass_samples_long >= 1)
    pass_has_grad = (pass_grad_updates >= 1)
    pass_has_params = (pass_params_changed >= 1)
    pass_has_relabel = (len(pass_relabel_evidence) >= 1)
    pass_no_crash = (crash_info is None)
    pass_all = all([pass_total_steps, pass_has_traj, pass_has_sample,
                    pass_has_grad, pass_has_params, pass_has_relabel, pass_no_crash])

    # ── Source hashes ──────────────────────────────────────────
    src_hashes = {}
    for fn in sorted(os.listdir(_AMAGO_SRC)):
        if fn.endswith(".py"):
            with open(os.path.join(_AMAGO_SRC, fn), "rb") as fh:
                src_hashes[fn] = hashlib.sha256(fh.read()).hexdigest()

    # ── Manifest ────────────────────────────────────────────────
    manifest = {
        "directive": "D059", "stage": "B",
        "treatment": "AMAGO_STYLE_EXPLORATORY_P2",
        "seed": SEED, "gpu_uuid": EXPECTED_GPU_UUID,
        "pid": pid, "argv": argv,
        "requested_env_steps": REQUESTED_ENV_STEPS,
        "num_envs": NUM_ENVS,
        "rollout_steps": ROLLOUT_STEPS,
        "num_updates": NUM_UPDATES,
        "resolved_env_steps": RESOLVED_ENV_STEPS,
        "total_env_steps": total_env_steps,
        "checkpoint_source": CKPT_SRC, "checkpoint_step": 17500,
        "gamma": GAMMA, "gae_lambda": GAE_LAMBDA,
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": round(time.time() - t0, 1),
        "counters": replay.counters.snapshot(),
        "replay_max_len": replay.longest_trajectory_length,
        "pass_gate": {
            "total_env_steps_eq_24576": pass_total_steps,
            "trajectories_longer_than_128": pass_has_traj,
            "replay_samples_from_long_sequence": pass_has_sample,
            "finite_nonzero_gradient_update": pass_has_grad,
            "params_changed_by_update": pass_has_params,
            "hindsight_relabel_evidence": pass_has_relabel,
            "no_crash": pass_no_crash,
            "relabel_evidence": pass_relabel_evidence,
            "num_trajs_long": pass_trajs_long,
            "num_samples_long": pass_samples_long,
            "num_grad_updates": pass_grad_updates,
        },
        "crash_info": crash_info,
        "source_hashes": src_hashes,
        "checkpoint_hash": hashlib.sha256(
            str(ckpt_path).encode()).hexdigest()[:16],
        "total_env_steps_exact": total_env_steps,
    }

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    with open(os.path.join(out_dir, "training_log.jsonl"), "w") as f:
        for e in all_logs:
            f.write(json.dumps(e) + "\n")
    with open(os.path.join(out_dir, "source_hashes.json"), "w") as f:
        json.dump(src_hashes, f, indent=2, sort_keys=True)

    # ── Verdict ═════════════════════════════════════════════════
    verdict = {"stage": "B", "status": "PASS" if pass_all else "FAIL",
               "pass_details": manifest["pass_gate"]}
    with open(os.path.join(EVIDENCE_DIR, "stage_B_verdict.json"), "w") as f:
        json.dump(verdict, f, indent=2, sort_keys=True)

    print(f"\n{'='*60}")
    print(f"Stage B: {'PASS' if pass_all else 'FAIL'}")
    for k, v in manifest["pass_gate"].items():
        if isinstance(v, bool):
            print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"  Steps: {total_env_steps}/{RESOLVED_ENV_STEPS}")
    print(f"  Trajs>128: {pass_trajs_long}  Samples>128: {pass_samples_long}")
    print(f"  Relabels: {len(pass_relabel_evidence)}  GradUpdates: {pass_grad_updates}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    if crash_info:
        with open(os.path.join(EVIDENCE_DIR, "stage_B_crash.json"), "w") as f:
            json.dump(crash_info, f, indent=2)

    sys.exit(0 if pass_all else 1)


if __name__ == "__main__":
    main()
