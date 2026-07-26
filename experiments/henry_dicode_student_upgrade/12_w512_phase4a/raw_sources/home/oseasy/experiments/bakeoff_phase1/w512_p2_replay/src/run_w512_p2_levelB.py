"""W512 × P2 Replay Level B trainer (24576 env steps), GPU-parameterized.

Trains W512 with P2-Full-A frozen replay (V-trace + AWR) from ckpt17500.
Two carry modes:
  --carry_mode persistent  : long_buf/long_mask persist across rollout boundary
  --carry_mode reset128    : long_buf/long_mask cleared at rollout boundary

P2-Full-A frozen hyperparameters: L_SEQ=129, K_BATCH=4, capacity=64,
kl_replay_max=0.05, kl_run_max=0.10, ema_tau=0.995, max_policy_lag=16,
w_vtrace=0.5, w_awr=0.5, rho_bar=1.0, c_bar=1.0, beta=1.0, w_max=20.0,
lambda_kl=0.01, actor_step_scales=(1.0,0.5,0.25,0.125), ent_floor=0.05.

Common init: ckpt17500, seed=42, deterministic ops, Stage4-native,
goal=DEFEAT_KOBOLD, num_envs=16, rollout=128, LR=2e-5, Adam eps=1e-5,
gamma=0.999, GAE lambda=0.8, total_steps=24576.

Saves checkpoints at {0, 4096, 8192, 12288, 16384, 20480, 24576}.
Fail-closed HARD STOPS identical to P2-Full-A Level B.
"""
import os
import sys
import json
import argparse
import shutil
import time

# ---- GPU must be set BEFORE jax import ----
def _setup_gpu(gpu_uuid):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid

def _setup_paths():
    THIS_SRC = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
    BAKE_SRC = "/home/oseasy/experiments/bakeoff_phase1/gpu0_w512/src"
    P2_SRC = "/home/oseasy/experiments/p2_full_20260723/src"
    HENRY_SRC = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src"
    for p in [THIS_SRC, BAKE_SRC, P2_SRC]:
        if p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, THIS_SRC)
    sys.path.insert(1, BAKE_SRC)
    sys.path.insert(2, P2_SRC)
    if HENRY_SRC not in sys.path:
        sys.path.insert(3, HENRY_SRC)


# ---- frozen constants ----
NUM_ENVS = 16
ROLLOUT_STEPS = 128
STEPS_PER_ROLLOUT = NUM_ENVS * ROLLOUT_STEPS
K_BATCH = 4
L_SEQ = 129
OPTIMISTIC_RESET_RATIO = 16
MASTER_SEED = 42
LEVELB_LR = 2e-5
SAVE_STEPS = (0, 4096, 8192, 12288, 16384, 20480, 24576)
DISK_SAFE_BYTES = 10 * 1024 ** 3
UNREACHABLE_GUARD_ROLLOUTS = 6

S4_TASK_PATH = "/home/oseasy/experiments/p2_v1_20260722/evidence/s4_task_code.py"


def _metric_val(v):
    if isinstance(v, (bool, str, list, dict, tuple)):
        return v
    import numpy as np
    if np.ndim(v) == 0:
        return float(v)
    return str(v)


def build_env(seed):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from craftax.craftax.constants import Achievement
    from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
    from dicode.task_utils import get_achievement_multi_hot
    from dicode.wrappers_cl import DistributedMultiTaskOptimisticLogWrapper
    from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=24576)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--seed", type=int, default=MASTER_SEED)
    ap.add_argument("--lr", type=float, default=LEVELB_LR)
    ap.add_argument("--carry_mode", required=True,
                    choices=["persistent", "reset128"])
    ap.add_argument("--gpu_uuid", required=True)
    args = ap.parse_args()

    _setup_gpu(args.gpu_uuid)
    _setup_paths()

    import numpy as np
    import jax
    import jax.numpy as jnp

    import rng_utils as RU
    import w512_memory as w5m
    from w512_compat_init import compatible_init_w512
    from w512_p2_core import collect_rollout_w512
    from w512_p2_learner import full_p2_update_w512
    from w512_replay_buffer import W512ReplayBuffer, relabel_sample_w512
    from w512_pending_episodes import W512PendingEpisodeBuffers
    from full_p2_learner import FullP2Config, build_optimizer
    import hindsight as H
    import checkpointing as CK

    os.makedirs(args.run_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ---- hard-stop guards ----
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == args.gpu_uuid
    devs = jax.devices()
    assert len(devs) == 1, f"HARD STOP gpu-bind: {devs}"
    assert not os.path.exists(os.path.join(args.run_dir, "training_log.jsonl")), \
        f"HARD STOP dir-reuse: {args.run_dir}"
    for d in (args.run_dir, args.ckpt_dir):
        if os.path.isdir(d):
            assert not os.listdir(d), f"HARD STOP dir-reuse: {d} not empty"
    free = shutil.disk_usage(args.ckpt_dir).free
    assert free > DISK_SAFE_BYTES, f"HARD STOP disk-low: {free/1024**3:.1f}GB"

    log_path = os.path.join(args.run_dir, "training_log.jsonl")
    logf = open(log_path, "a")
    def log(rec):
        logf.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
        logf.flush()

    cfg = FullP2Config()
    assert cfg.adam_eps == 1e-5
    assert cfg.gamma == 0.999
    assert abs(args.lr - 2e-5) < 1e-12
    num_rollouts = args.steps // STEPS_PER_ROLLOUT
    save_steps = tuple(s for s in SAVE_STEPS if s <= args.steps)

    label = f"W512-{args.carry_mode.upper()}-P2REPLAY"
    budget = dict(num_envs=NUM_ENVS, rollout_steps=ROLLOUT_STEPS,
                  num_updates=num_rollouts, total_env_steps=args.steps,
                  save_steps=list(save_steps), carry_mode=args.carry_mode)
    log({"event": "budget", "label": label, **budget})
    print(f"[{label}] budget={budget}", flush=True)

    # 1. W512 compatible init
    ci = compatible_init_w512(strict=True)
    params, target_params = ci["params"], ci["target_params"]
    network = ci["network"]
    w5_cfg = ci["w5_cfg"]
    a_rec = ci["apply_eval_recon"]
    a_raw = ci["apply_eval_raw"]
    scan_fn = ci["scan_fn"]
    log({"event": "compatible_init", "fingerprint": ci["fingerprint"],
         "source_checkpoint": ci["source_checkpoint"]})
    print(f"[{label}] init OK sha={ci['fingerprint']['params_sha256'][:16]} "
          f"params={ci['fingerprint']['param_count']}", flush=True)

    # 2. env
    env, env_params, target_achievement, emb = build_env(args.seed)

    # 3. fresh state
    memories, mem_mask, mem_idx = jnp.zeros((NUM_ENVS, 128, 2, 256)), \
        jnp.zeros((NUM_ENVS, 8, 1, 129), dtype=jnp.bool_), \
        jnp.full((NUM_ENVS,), 128, dtype=jnp.int32)
    w512_state = w5m.w512_init(NUM_ENVS, w5_cfg)
    rng = jax.random.PRNGKey(args.seed)
    action_rng = RU.make_action_rng(args.seed)
    pending = W512PendingEpisodeBuffers(NUM_ENVS, first_policy_version=0)
    replay = W512ReplayBuffer(capacity=64, seed=args.seed)
    opt = build_optimizer(args.lr, cfg)
    opt_state = opt.init(params)
    rng, reset_rng = jax.random.split(rng)
    obsv, env_state = env.reset(reset_rng, env_params)
    print(f"[{label}] env reset OK obsv={np.asarray(obsv).shape}", flush=True)

    global_step = 0
    update_count = 0
    total_episodes = 0
    total_updates = 0
    accepted_policy_updates = 0
    kl_rejected_updates = 0
    hindsight_attempts = 0
    hindsight_eligible = 0
    eligible_no_update_streak = 0
    any_nan = False
    saved_shas = {}
    t_start = time.time()

    def _save(step):
        nonlocal saved_shas
        free = shutil.disk_usage(args.ckpt_dir).free
        assert free > DISK_SAFE_BYTES
        import pickle
        ckpt_path = os.path.join(args.ckpt_dir, str(step))
        os.makedirs(ckpt_path, exist_ok=True)
        with open(os.path.join(ckpt_path, "params.pkl"), "wb") as f:
            pickle.dump(params, f)
        import hashlib
        sha = hashlib.sha256()
        for leaf in jax.tree_util.tree_leaves(params):
            sha.update(np.asarray(leaf).tobytes())
        saved_sha = sha.hexdigest()
        manifest = {
            "step": step, "arm": "w512",
            "carry_mode": args.carry_mode.upper(),
            "replay": "P2REPLAY",
            "params_sha256": saved_sha,
            "gpu": args.gpu_uuid,
            "seed": args.seed, "lr": args.lr,
            "update_count": update_count,
            "replay_size": len(replay),
            "accepted_policy_updates": accepted_policy_updates,
            "kl_rejected_updates": kl_rejected_updates,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(os.path.join(ckpt_path, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        saved_shas[step] = saved_sha
        log({"event": "checkpoint", "step": step, "params_sha256": saved_sha,
             "update_count": update_count, "replay_size": len(replay)})
        print(f"[{label}] ckpt step={step} sha={saved_sha[:16]}", flush=True)

    if 0 in save_steps:
        _save(0)

    # ---- training loop ----
    for r in range(num_rollouts):
        t0 = time.time()
        trajs, carry, stats = collect_rollout_w512(
            env, env_state, network, params, obsv,
            memories, mem_mask, mem_idx, w512_state,
            rng, action_rng, pending, target_achievement,
            rollout_steps=ROLLOUT_STEPS, window_mem=128, num_heads=8,
            w5_cfg=w5_cfg, carry_mode=args.carry_mode,
            collected_update_count=update_count)
        env_state = carry["env_state"]; obsv = carry["obsv"]
        memories = carry["memories"]; mem_mask = carry["mem_mask"]
        mem_idx = carry["mem_idx"]; rng = carry["rng"]
        w512_state = carry["w512_state"]
        global_step += STEPS_PER_ROLLOUT
        total_episodes += len(trajs)

        for t in trajs:
            assert bool(np.asarray(t.dones)[-1]), "HARD STOP non-terminal traj"
            replay.insert(t)
            replay.trajectories_collected += 1

        assert replay.trajectories_collected == replay.trajectories_inserted, \
            "HARD STOP conservation"

        # ---- replay update ----
        update_metrics = None
        updates_before = total_updates
        if replay.can_sample():
            so, sr = [], []
            for _ in range(K_BATCH):
                hindsight_attempts += 1
                s = replay.sample(sequence_length=L_SEQ)
                try:
                    rel = relabel_sample_w512(s)
                except ValueError:
                    continue
                hindsight_eligible += 1
                so.append(s); sr.append(rel)
            if len(so) >= 2:
                params, target_params, opt_state, m = full_p2_update_w512(
                    params, target_params, opt_state, opt,
                    a_rec, a_raw, scan_fn,
                    so, sr, cfg, update_count, w5_cfg)
                update_count += 1
                total_updates += 1
                finite = bool(m["finite"])
                any_nan = any_nan or (not finite)
                assert finite, "HARD STOP NaN/Inf"
                if bool(m.get("policy_committed")):
                    assert float(m["policy_kl"]) <= cfg.kl_replay_max + 1e-12
                    accepted_policy_updates += 1
                if bool(m.get("kl_rejected_update")):
                    assert not bool(m.get("policy_committed"))
                    kl_rejected_updates += 1
                assert float(m["entropy"]) >= cfg.ent_floor, \
                    f"HARD STOP entropy collapse: {m['entropy']}"
                update_metrics = {k: _metric_val(v) for k, v in m.items()}

        # unreachable guard
        if replay.can_sample() and total_updates == updates_before:
            eligible_no_update_streak += 1
        else:
            eligible_no_update_streak = 0
        assert eligible_no_update_streak < UNREACHABLE_GUARD_ROLLOUTS

        dt = time.time() - t0
        log({"event": "rollout", "rollout": r, "global_step": global_step,
             "completed_episodes": stats["completed_episodes"],
             "replay_size": len(replay), "update_count": update_count,
             "accepted": accepted_policy_updates,
             "kl_rejected": kl_rejected_updates,
             "hindsight_attempts": hindsight_attempts,
             "hindsight_eligible": hindsight_eligible,
             "time_s": round(dt, 1),
             "update": update_metrics})
        print(f"[{label}] rollout {r} step={global_step} ep={stats['completed_episodes']} "
              f"replay={len(replay)} updates={total_updates} acc={accepted_policy_updates}"
              f"{' loss=%.4f kl=%.4f' % (update_metrics['loss'], update_metrics['policy_kl']) if update_metrics else ' (no update)'}"
              f" ({dt:.1f}s)", flush=True)

        if global_step in save_steps:
            _save(global_step)

    # ---- final ----
    assert not any_nan
    assert global_step == args.steps
    total_time = time.time() - t_start

    summary = {
        "label": label, "steps": args.steps, "gpu_uuid": args.gpu_uuid,
        "carry_mode": args.carry_mode, "seed": args.seed, "lr": args.lr,
        "global_step": global_step, "update_count": update_count,
        "total_episodes": total_episodes, "total_updates": total_updates,
        "accepted_policy_updates": accepted_policy_updates,
        "kl_rejected_updates": kl_rejected_updates,
        "hindsight_attempts": hindsight_attempts,
        "hindsight_eligible": hindsight_eligible,
        "any_nan": any_nan, "total_time_s": round(total_time, 1),
        "saved_checkpoints": {str(k): v for k, v in sorted(saved_shas.items())},
        "replay_size": len(replay),
        "source_fingerprint": ci["fingerprint"],
        "budget": budget,
    }
    with open(os.path.join(args.run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log({"event": "summary", **summary})
    logf.close()

    print(f"\n{'='*72}\n{label} training complete. final step={global_step}\n"
          f"  checkpoints: {args.ckpt_dir}\n"
          f"  log: {log_path}\n{'='*72}", flush=True)


if __name__ == "__main__":
    main()
