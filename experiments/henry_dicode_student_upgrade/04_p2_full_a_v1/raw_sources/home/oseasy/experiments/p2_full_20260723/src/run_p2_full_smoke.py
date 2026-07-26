"""P2-Full-A GPU0 smoke launcher (2048 / 4096 env steps). NO long training.

Pipeline validated end-to-end on GPU0 from the Henry base ckpt17500:
  1. compatible_init  — orbax load of ckpt17500 + bit-exact fingerprint (fail-closed)
  2. real Stage4 env  — MultiTaskMiniCraftaxEnv + optimistic-log wrapper (Henry/craftax
                        imports only; the P2-v1 launcher is NOT imported, so P2-Full-A's
                        same-named modules are never shadowed)
  3. collect_rollout  — real vectorized rollouts, sparse entering-state anchors
  4. replay + update  — insert done-terminated >=129-step episodes; fire the combined
                        V-trace+AWR update on K relabelable windows (B>=2). Honest: if no
                        episode completes / none relabelable in the budget, updates_fired=0
                        (the update path is independently CPU-proven by Gate 2-4).
  5. checkpoint       — pure-pickle full state (params/target/opt/replay/pending/rng) +
                        restore round-trip provenance check (params SHA must match).

Memory convention: fresh init + done-reset use mem_idx=window_mem (128), self-consistent
with derive_anchor_entering_state (Gate G1.4 bit-exact). This is P2-Full-A's frozen
convention and deliberately NOT the P2-v1 launcher's window_mem+1 fresh-init.

Output: <run_dir>/training_log.jsonl, <run_dir>/smoke_summary.json,
        <ckpt_dir>/<step>/{full_state.pkl,manifest.json}.
"""
import os
GPU_UUID = "GPU-e8c08612-c22a-c8a4-6df5-affb2dd1f9a6"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID   # MUST precede jax import
import sys, os.path, json, argparse

BASE_SRC = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__))))
HENRY_SRC = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src"
if BASE_SRC in sys.path:
    sys.path.remove(BASE_SRC)
sys.path.insert(0, BASE_SRC)
if HENRY_SRC not in sys.path:
    sys.path.insert(1, HENRY_SRC)

import numpy as np
import jax
import jax.numpy as jnp

# ---- real Stage4 env (Henry / craftax / minicraftax only; NO P2-v1 launcher) ----
from craftax.craftax.constants import Achievement
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from dicode.task_utils import get_achievement_multi_hot
from dicode.wrappers_cl import DistributedMultiTaskOptimisticLogWrapper
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv

# ---- P2-Full-A modules (BASE_SRC wins; never shadowed by P2-v1) ----
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

# ---- frozen constants ----
NUM_ENVS = 16
ROLLOUT_STEPS = 128                 # 1 rollout == NUM_ENVS*ROLLOUT_STEPS == 2048 env steps
STEPS_PER_ROLLOUT = NUM_ENVS * ROLLOUT_STEPS
K_BATCH = 4                         # sampled windows per update
L_SEQ = 129                         # smoke loss-window length (min replayable; formal run uses 512)
OPTIMISTIC_RESET_RATIO = 16
MASTER_SEED = 42
SMOKE_LR = 6e-5                     # default; overridden by --lr (selected Control-grid LR)


def _metric_val(v):
    """JSON-friendly metric conversion: keep bool/list/dict/str native, scalar -> float."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, required=True, choices=[2048, 4096])
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--seed", type=int, default=MASTER_SEED)
    ap.add_argument("--lr", type=float, default=SMOKE_LR,
                    help="training LR; use the LR selected by the frozen Control grid")
    args = ap.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    log_path = os.path.join(args.run_dir, "training_log.jsonl")
    logf = open(log_path, "a")

    def log(rec):
        logf.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
        logf.flush()

    cfg = FullP2Config()
    num_rollouts = args.steps // STEPS_PER_ROLLOUT
    assert num_rollouts >= 1

    # ---- frozen smoke budget manifest (directive item 6) ----
    budget = dict(
        num_envs=NUM_ENVS,
        rollout_steps=ROLLOUT_STEPS,
        num_updates=num_rollouts,                 # update opportunities (1/2048 -> 2/4096)
        steps_per_env=args.steps // NUM_ENVS,
        steps_per_rollout=STEPS_PER_ROLLOUT,
        total_env_steps=args.steps,
    )
    if args.steps == 4096:                        # the frozen 4096 budget, asserted exactly
        assert budget["num_envs"] == 16, budget
        assert budget["rollout_steps"] == 128, budget
        assert budget["num_updates"] == 2, budget
        assert budget["steps_per_env"] == 256, budget
        assert budget["total_env_steps"] == 4096, budget
    log({"event": "budget", **budget})

    # 1. compatible init (load + fingerprint, fail-closed)
    ci = CI.compatible_init(strict=True)
    params, target_params = ci["params"], ci["target_params"]
    network = ci["network"]
    a_rec, a_raw, scan_fn = ci["apply_eval_recon"], ci["apply_eval_raw"], ci["scan_fn"]
    jit_fwd = CORE.make_jit_forward(network)
    log({"event": "compatible_init", "fingerprint": ci["fingerprint"],
         "source_checkpoint": ci["source_checkpoint"], "kernels": ci["kernels"]})
    print("[smoke] compatible_init OK  sha=%s params=%d value=%.4f top=%d" % (
        ci["fingerprint"]["params_sha256"], ci["fingerprint"]["param_count"],
        ci["fingerprint"]["value"], ci["fingerprint"]["top_action"]))

    # 2. env
    env, env_params, target_achievement, emb = build_env(args.seed)
    assert emb == cfg.embed or emb == 67, emb

    # 3. init state (P2-Full-A convention: mem_idx = window_mem at fresh start)
    memories, mem_mask, mem_idx = MA.fresh_rollout_state(
        cfg.window_mem, cfg.num_heads, cfg.num_layers, cfg.embed, NUM_ENVS)
    rng = jax.random.PRNGKey(args.seed)
    action_rng = RU.make_action_rng(args.seed)
    pending = PendingEpisodeBuffers(NUM_ENVS, first_policy_version=0)
    replay = ReplayBuffer(capacity=64, seed=args.seed)
    opt = build_optimizer(args.lr, cfg)
    opt_state = opt.init(params)
    rng, reset_rng = jax.random.split(rng)
    obsv, env_state = env.reset(reset_rng, env_params)
    print("[smoke] env reset OK  obsv=%s" % (np.asarray(obsv).shape,))

    global_step = 0
    update_count = 0
    total_episodes = 0
    total_updates = 0
    accepted_policy_updates = 0          # updates committed with KL<=kl_replay_max (item 8)
    kl_rejected_updates = 0             # updates where every actor scale breached the gate
    any_nan = False

    for r in range(num_rollouts):
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
        for t in trajs:
            replay.insert(t)
            replay.counters.trajectories_collected += 1

        # ---- try a combined update on relabelable windows ----
        update_metrics = None
        if replay.can_sample():
            so, sr = [], []
            for _ in range(K_BATCH):
                s = replay.sample(sequence_length=L_SEQ)
                try:
                    rel = H.relabel_sample(s)   # goal_index=None -> min achieved (Gate 5/6)
                except ValueError:
                    continue                    # no achievement in window -> not relabelable
                so.append(s); sr.append(rel)
            if len(so) >= 2:                    # lax.scan needs B>=2
                params, target_params, opt_state, m = full_p2_update(
                    params, target_params, opt_state, opt, a_rec, a_raw, scan_fn,
                    so, sr, cfg, update_count)
                update_count += 1
                total_updates += 1
                finite = bool(m["finite"])
                any_nan = any_nan or (not finite)
                update_metrics = {k: _metric_val(v) for k, v in m.items()}
                update_metrics["finite"] = finite
                update_metrics["batch"] = len(so)
                # Level-A acceptance bookkeeping (directive item 8): a policy replay
                # update counts as ACCEPTED only if it committed with KL<=kl_replay_max.
                if bool(m.get("policy_committed")) and \
                        float(m.get("policy_kl", 1.0)) <= cfg.kl_replay_max:
                    accepted_policy_updates += 1
                if bool(m.get("kl_rejected_update")):
                    kl_rejected_updates += 1

        rec = {"event": "rollout", "rollout": r, "global_step": global_step,
               "completed_episodes": stats["completed_episodes"],
               "mean_ep_return": stats["mean_ep_return"],
               "mean_ep_length": stats["mean_ep_length"],
               "pending_transitions": stats["pending_transitions"],
               "pending_anchors": stats["pending_anchors"],
               "replay_size": len(replay),
               "replay_can_sample": replay.can_sample(),
               "update_count": update_count,
               "update": update_metrics}
        log(rec)
        print("[smoke] rollout %d  step=%d  episodes=%d  replay=%d  updates=%d%s" % (
            r, global_step, stats["completed_episodes"], len(replay), total_updates,
            ("  loss=%.4f" % update_metrics["loss"]) if update_metrics else "  (no update)"))

    # 4. checkpoint (pure pickle, full state) + restore round-trip provenance check
    collector_state = {"env_state": env_state, "obsv": obsv, "memories": memories,
                       "mem_mask": mem_mask, "mem_idx": mem_idx}
    ck_dir = CK.save_full_checkpoint(
        params, target_params, opt_state, replay, rng,
        global_step=global_step, path=args.ckpt_dir, step=global_step,
        action_rng_state=RU.action_rng_state(action_rng), update_count=update_count,
        pending=pending, collector_state=collector_state, config=cfg, keep=3,
        extra_manifest={"label": "P2-Full-A", "smoke_steps": args.steps,
                        "gpu_uuid": GPU_UUID, "seed": args.seed, "lr": args.lr,
                        "budget": budget,
                        "accepted_policy_updates": accepted_policy_updates,
                        "kl_rejected_updates": kl_rejected_updates,
                        "source_checkpoint_sha256": ci["fingerprint"]["params_sha256"]})
    saved_sha = CK.params_content_sha256(params)
    restored = CK.restore_full_checkpoint(args.ckpt_dir, step=global_step)
    restored_sha = CK.params_content_sha256(restored["params"])
    roundtrip_ok = (restored_sha == saved_sha
                    and restored["global_step"] == global_step
                    and restored["update_count"] == update_count
                    and restored["pending"].total_pending_transitions()
                        == pending.total_pending_transitions())
    print("[smoke] checkpoint step=%d  saved_sha=%s  restored_sha=%s  roundtrip_ok=%s" % (
        global_step, saved_sha[:16], restored_sha[:16], roundtrip_ok))

    summary = {
        "label": "P2-Full-A", "smoke_steps": args.steps, "gpu_uuid": GPU_UUID,
        "seed": args.seed, "global_step": global_step, "update_count": update_count,
        "total_episodes": total_episodes, "total_updates": total_updates,
        "accepted_policy_updates": accepted_policy_updates,
        "kl_rejected_updates": kl_rejected_updates,
        # Level A is officially PASS only with >=1 KL<=0.05 accepted policy replay update
        "level_a_accepted_update_present": accepted_policy_updates >= 1,
        "kl_replay_max": cfg.kl_replay_max,
        "any_nan_or_inf": any_nan,
        "source_checkpoint": ci["source_checkpoint"],
        "source_fingerprint": ci["fingerprint"],
        "checkpoint_dir": ck_dir, "checkpoint_step": global_step,
        "params_sha256": saved_sha, "restore_roundtrip_ok": roundtrip_ok,
        "replay_size": len(replay), "replay_can_sample": replay.can_sample(),
        "lr": args.lr, "smoke_lr": args.lr, "k_batch": K_BATCH, "l_seq": L_SEQ,
        "budget": budget,
        "num_envs": NUM_ENVS, "rollout_steps": ROLLOUT_STEPS,
    }
    with open(os.path.join(args.run_dir, "smoke_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    log({"event": "summary", **summary})
    logf.close()

    # fail-closed gates for the smoke (Level-A acceptance is a REPORTED criterion, not a
    # crash gate: a 2048 smoke legitimately fires 0 updates and a KL_REJECTED update is a
    # valid gate outcome, not a pipeline failure).
    assert not any_nan, "FAIL: NaN/Inf encountered during smoke"
    assert roundtrip_ok, "FAIL: checkpoint restore round-trip mismatch"
    assert global_step == args.steps, f"FAIL: ran {global_step} != {args.steps}"
    print("SMOKE_OK steps=%d episodes=%d updates=%d accepted_kl=%.2gx%d rejected=%d "
          "nan=%s roundtrip=%s level_a_accepted=%s" % (
        global_step, total_episodes, total_updates, cfg.kl_replay_max,
        accepted_policy_updates, kl_rejected_updates, any_nan, roundtrip_ok,
        accepted_policy_updates >= 1))


if __name__ == "__main__":
    main()
