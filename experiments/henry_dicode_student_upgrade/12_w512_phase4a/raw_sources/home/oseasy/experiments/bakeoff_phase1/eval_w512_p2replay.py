#!/usr/bin/env python3
"""W512 × P2 Replay — 256-world Stage4 evaluator for P2Replay arms.

Frozen protocol identical to eval_a_side_unified.py (SHA dcf7fe20):
  stochastic policy, seed=42, 256 worlds, max 4096 steps, S4_dark native start,
  optimistic-reset wrapper, all arms PAIRED by world index.
  DEFEAT_KOBOLD = pre-step ever-set from env_state.achievements.
  floor3 = max player_level >= 3 (pre-step).

Arms:
  W512_Persistent_P2Replay  (w512_p2_replay/.../24576/params.pkl)
  W512_Reset128_P2Replay    (w512_p2_replay/.../24576/params.pkl)
"""
import hashlib, json, os, sys, time, pickle
import numpy as np

# ---- GPU ----
GPU_UUID = os.environ.get("P2REPLAY_EVAL_GPU", "GPU-e8c08612-c22a-c8a4-6df5-affb2dd1f9a6")
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

import jax, jax.numpy as jnp

# ---- paths ----
BAKE = "/home/oseasy/experiments/bakeoff_phase1"
W512_SRC = f"{BAKE}/gpu0_w512/src"
V7_SRC = ("/home/oseasy/incoming/henry_work_20260721T105300/"
          "extracted/Henry_work/code/dicode_v7fix58_armB/src")
V7 = ("/home/oseasy/incoming/henry_work_20260721T105300/"
      "extracted/Henry_work/code/dicode_v7fix58_armB")
for p in (W512_SRC, V7_SRC, V7):
    if p not in sys.path: sys.path.insert(0, p)

from craftax.craftax.constants import Achievement
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from dicode.task_utils import get_achievement_multi_hot
from dicode.wrappers_cl import DistributedMultiTaskOptimisticLogWrapper
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv

from network_w512 import ActorCriticTransformerW512
import w512_memory as w5m

# ---- frozen config ----
_cfg = dict(activation="relu", embed_size=256, hidden_layers=256, num_heads=8,
            qkv_features=256, num_layers=2, gating=True, gating_bias=2.0,
            window_mem=128, window_grad=64)
cfg = type("C", (), _cfg)()

W512_LONG_SIZE = 384
W512_DELAY_SIZE = 128

S4_TASK_PATH = "/home/oseasy/experiments/p2_v1_20260722/evidence/s4_task_code.py"
EVAL_STEP = 24576

CKPT = {
    "W512_Persistent_P2Replay": f"{BAKE}/w512_p2_replay/w512_persistent_p2replay_24576/checkpoints/{EVAL_STEP}/params.pkl",
    "W512_Reset128_P2Replay":   f"{BAKE}/w512_p2_replay/w512_reset128_p2replay_24576/checkpoints/{EVAL_STEP}/params.pkl",
}

DK = int(Achievement.DEFEAT_KOBOLD.value)
SEWERS = int(Achievement.ENTER_SEWERS.value)
NUM_ENVS = 256
NUM_STEPS = 4096
EVAL_SEED = 42

with open(__file__, "rb") as f:
    EVAL_SHA256 = hashlib.sha256(f.read()).hexdigest()

# ---- env ----
ctor = EnvParams(max_timesteps=NUM_STEPS)
table = jnp.array([get_achievement_multi_hot([Achievement.DEFEAT_KOBOLD])], dtype=jnp.float32)
EMB = int(table.shape[1])
ns4 = {}; exec(open(S4_TASK_PATH).read(), ns4); S4Cls = ns4["Env"]
s4_base = MultiTaskMiniCraftaxEnv([S4Cls], StaticEnvParams(), ctor, True,
                                  conditioning_type="embedding", embedding_size=EMB)
ACTION_DIM = int(s4_base.action_space(ctor).n)
OBS_DIM = int(s4_base.observation_space(ctor).shape[0])

# ---- network ----
_net_kw = dict(action_dim=ACTION_DIM, activation=cfg.activation, encoder_size=cfg.embed_size,
    hidden_layers=cfg.hidden_layers, num_heads=cfg.num_heads, qkv_features=cfg.qkv_features,
    num_layers=cfg.num_layers, gating=cfg.gating, gating_bias=cfg.gating_bias)

w512_net = ActorCriticTransformerW512(**_net_kw, long_size=W512_LONG_SIZE)
w5_cfg = w5m.W512Config(long_size=W512_LONG_SIZE, delay_size=W512_DELAY_SIZE,
                         encoder_size=cfg.embed_size)


def load_params_pkl(path):
    with open(path, "rb") as f:
        raw = jax.tree_util.tree_map(jnp.asarray, pickle.load(f))
    # P2Replay checkpoints store raw param dict (unwrapped);
    # wrap in {"params": ...} for flax apply
    if "params" not in raw:
        return {"params": raw}
    return raw


# ---- core eval loop (frozen protocol, identical to eval_a_side_unified.py) ----
def run_w512_eval(name, params):
    """Evaluate W512 arm with mode='on' (full long memory)."""
    env = DistributedMultiTaskOptimisticLogWrapper(s4_base, jax.random.PRNGKey(0),
            NUM_ENVS, 1, 16, jnp.array([1.0]), table)
    memories = jnp.zeros((NUM_ENVS, cfg.window_mem, cfg.num_layers, cfg.embed_size))
    mem_mask = jnp.zeros((NUM_ENVS, cfg.num_heads, 1, cfg.window_mem + 1), dtype=jnp.bool_)
    mem_idx = jnp.zeros((NUM_ENVS,), dtype=jnp.int32) + (cfg.window_mem + 1)
    ls = w5m.w512_init(NUM_ENVS, w5_cfg)

    def _step(carry, _):
        (log_state, memories, mem_mask, mem_idx, last_obs, done, true_done, ls,
         finished, ep_len, max_floor, seen, info_acc, ep_return, sewers, flip_floor, rng,
         step_count) = carry

        mem_idx = jnp.where(done, cfg.window_mem, jnp.clip(mem_idx - 1, 0, cfg.window_mem))
        mem_mask = jnp.where(done[:, None, None, None], jnp.zeros_like(mem_mask), mem_mask)
        ohot = jax.nn.one_hot(mem_idx, cfg.window_mem + 1)[:, None, None, :].repeat(cfg.num_heads, 1)
        mem_mask = jnp.logical_or(mem_mask, ohot)

        rng, a_rng, s_rng = jax.random.split(rng, 3)

        # W512 forward with full long memory (mode="on")
        pi, _, mem_out, h_t = w512_net.apply(params, memories, last_obs, mem_mask,
            long_buf=ls["long_buf"], long_mask=ls["long_mask"],
            method=w512_net.model_forward_eval)
        ls_new = w5m.w512_step(ls, h_t, true_done, w5_cfg)

        action = pi.sample(seed=a_rng)
        memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(mem_out)

        # Pre-step achievements (from log_state BEFORE env.step)
        pre = log_state.env_state
        pre_pl = pre.player_level
        pre_dk = pre.achievements[:, DK].astype(bool)
        pre_sw = pre.achievements[:, SEWERS].astype(bool)

        next_obs, next_log_state, reward, next_done, info = env.step(s_rng, log_state, action, ctor)
        true_done_next = info["returned_episode"]

        active = ~finished
        ep_len = ep_len + active.astype(jnp.int32)
        ep_return = ep_return + jnp.asarray(reward, jnp.float32).reshape(-1) * active.astype(jnp.float32)
        max_floor = jnp.where(active, jnp.maximum(max_floor, pre_pl), max_floor)
        newly = pre_dk & active & ~seen
        flip_floor = jnp.where(newly, pre_pl, flip_floor)
        seen = seen | (pre_dk & active)
        sewers = sewers | (pre_sw & active)
        keys = [k for k in info if "Achievements" in k and "kobold" in k.lower()]
        if keys:
            info_acc = info_acc + jnp.asarray(info[keys[0]], jnp.float32).reshape(-1) * active.astype(jnp.float32)
        finished = finished | next_done

        return (next_log_state, memories, mem_mask, mem_idx, next_obs, next_done, true_done_next,
                ls_new, finished, ep_len, max_floor, seen, info_acc, ep_return, sewers, flip_floor, rng,
                step_count + 1), None

    rng = jax.random.PRNGKey(EVAL_SEED)
    rng, reset_rng = jax.random.split(rng)
    obsv, log_state = env.reset(reset_rng, ctor)
    init = (log_state, memories, mem_mask, mem_idx, obsv,
            jnp.zeros((NUM_ENVS,), dtype=jnp.bool_), jnp.zeros((NUM_ENVS,), dtype=jnp.bool_),
            ls, jnp.zeros((NUM_ENVS,), dtype=jnp.bool_),
            jnp.zeros((NUM_ENVS,), dtype=jnp.int32),
            jnp.full((NUM_ENVS,), 2, dtype=jnp.int32),
            jnp.zeros((NUM_ENVS,), dtype=jnp.bool_), jnp.zeros((NUM_ENVS,), dtype=jnp.float32),
            jnp.zeros((NUM_ENVS,), dtype=jnp.float32), jnp.zeros((NUM_ENVS,), dtype=jnp.bool_),
            jnp.full((NUM_ENVS,), -1, dtype=jnp.int32), rng,
            jnp.int32(0))
    t0 = time.time()
    final, _ = jax.lax.scan(_step, init, None, NUM_STEPS)
    (_, _, _, _, _, _, _, _, finished, ep_len, max_floor, seen, info_acc,
     ep_return, sewers, flip_floor, _, _) = final
    jax.block_until_ready(final)
    roll_time = time.time() - t0

    finished_np = np.asarray(finished); ep_len_np = np.asarray(ep_len)
    max_floor_np = np.asarray(max_floor); seen_np = np.asarray(seen)
    info_acc_np = np.asarray(info_acc); sewers_np = np.asarray(sewers)
    success_np = seen_np | (info_acc_np > 0)
    timeout_np = finished_np & (ep_len_np >= NUM_STEPS) & ~success_np
    died_np = finished_np & ~success_np & ~timeout_np
    n_not_finished = int(np.sum(~finished_np))
    if n_not_finished > 0: timeout_np = timeout_np | ~finished_np
    n_success = int(success_np.sum()); n_died = int(died_np.sum()); n_timeout = int(timeout_np.sum())
    n_sewers = int(sewers_np.sum()); n_floor3 = int((max_floor_np >= 3).sum())
    sr = n_success / NUM_ENVS; floor3 = n_floor3 / NUM_ENVS
    cond_kill = (n_success / n_floor3) if n_floor3 > 0 else 0.0
    summary = dict(variant=name, arm_type="w512", mode="on",
        num_episodes=NUM_ENVS, evaluation_seed=EVAL_SEED,
        policy_mode="stochastic", spawn_floor=2,
        SR=sr, n_success=n_success, floor3_reach_rate=floor3, n_floor3=n_floor3,
        conditional_kill_rate=cond_kill, n_died=n_died, n_timeout=n_timeout,
        n_not_finished=n_not_finished, death_rate=n_died / NUM_ENVS, timeout_rate=n_timeout / NUM_ENVS,
        ENTER_SEWERS_rate=n_sewers / NUM_ENVS, mean_episode_length=float(ep_len_np.mean()),
        median_episode_length=float(np.median(ep_len_np)),
        max_floor_max=int(max_floor_np.max()),
        success_per_world=[bool(x) for x in success_np],
        floor3_per_world=[bool(x) for x in (max_floor_np >= 3)],
        died_per_world=[bool(x) for x in died_np],
        sewers_per_world=[bool(x) for x in sewers_np],
        ep_len_per_world=[int(x) for x in ep_len_np],
        rollout_time_s=round(roll_time, 1))
    print(f"[{name}] SR={sr*100:.2f}% ({n_success}/{NUM_ENVS})  floor3={floor3*100:.2f}% ({n_floor3})  "
          f"cond_kill={cond_kill*100:.1f}%  died={n_died} timeout={n_timeout} sewers={n_sewers} "
          f"eplen={ep_len_np.mean():.0f}  ({roll_time:.1f}s)", flush=True)
    return summary


# ---- statistics ----
def mcnemar_paired(a_succ, b_succ):
    a = np.asarray(a_succ, bool); b = np.asarray(b_succ, bool)
    both = int(np.sum(a & b)); a_only = int(np.sum(a & ~b)); b_only = int(np.sum(~a & b))
    neither = int(np.sum(~a & ~b)); n_disc = a_only + b_only
    if n_disc == 0:
        chi2 = 0.0; pval = 1.0
    else:
        chi2 = (abs(a_only - b_only) - 1) ** 2 / n_disc
        from math import erfc, sqrt
        pval = erfc(sqrt(chi2 / 2.0))
    return dict(both=both, a_only=a_only, b_only=b_only, neither=neither, n_discordant=n_disc,
                mcnemar_chi2=round(float(chi2), 4), mcnemar_p=round(float(pval), 6))

def paired_bootstrap_ci(a_succ, b_succ, n=20000, seed=42):
    a = np.asarray(a_succ, np.float64); b = np.asarray(b_succ, np.float64); d = a - b
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return dict(diff_pp=round(float(d.mean()) * 100, 3),
                ci95_low_pp=round(float(lo) * 100, 3), ci95_high_pp=round(float(hi) * 100, 3))

def paired_compare(a_summary, b_summary, label):
    def d(k): return (a_summary[k] - b_summary[k]) * 100.0
    return dict(label=label,
        a=a_summary["variant"], b=b_summary["variant"],
        a_SR_pp=a_summary["SR"] * 100, b_SR_pp=b_summary["SR"] * 100,
        SR_delta_pp=d("SR"), SR_delta_worlds=a_summary["n_success"] - b_summary["n_success"],
        floor3_delta_pp=d("floor3_reach_rate"), cond_kill_delta_pp=d("conditional_kill_rate"),
        death_delta_pp=d("death_rate"), timeout_delta_pp=d("timeout_rate"),
        sewers_delta_pp=d("ENTER_SEWERS_rate"),
        eplen_delta=a_summary["mean_episode_length"] - b_summary["mean_episode_length"],
        mcnemar=mcnemar_paired(a_summary["success_per_world"], b_summary["success_per_world"]),
        bootstrap_SR=paired_bootstrap_ci(a_summary["success_per_world"], b_summary["success_per_world"]))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default=GPU_UUID)
    ap.add_argument("--out", default=f"{BAKE}/reports/w512_p2replay_eval.json")
    args = ap.parse_args()

    # Load params and compute SHA
    P = {}
    shas = {}
    for arm_name, ckpt_path in CKPT.items():
        print(f"[eval] Loading {arm_name} from {ckpt_path}", flush=True)
        params = load_params_pkl(ckpt_path)
        P[arm_name] = params
        sha = hashlib.sha256()
        for leaf in jax.tree_util.tree_leaves(params):
            sha.update(np.asarray(leaf).tobytes())
        shas[arm_name] = sha.hexdigest()
        print(f"[eval] {arm_name} sha={sha.hexdigest()[:16]}", flush=True)

    # Run evaluations
    all_results = {}
    for arm_name in CKPT:
        res = run_w512_eval(arm_name, P[arm_name])
        res["params_sha256"] = shas[arm_name]
        res["ckpt_path"] = CKPT[arm_name]
        all_results[arm_name] = res

    # Paired comparisons between the two P2Replay arms
    cmp_p2replay = paired_compare(
        all_results["W512_Persistent_P2Replay"],
        all_results["W512_Reset128_P2Replay"],
        "CARRY_WITH_REPLAY = W512_Persistent_P2Replay - W512_Reset128_P2Replay")

    # Load existing A-side results for cross-comparison
    a_side_path = f"{BAKE}/reports/cc1_corrected_eval/a_side_unified_eval.json"
    existing = {}
    if os.path.exists(a_side_path):
        with open(a_side_path) as f:
            a_data = json.load(f)
        for arm_key, arm_data in a_data.get("arms", {}).items():
            existing[arm_key] = arm_data

    # Cross-comparisons with existing arms
    cross_comparisons = {}
    if existing:
        # REPLAY_EFFECT_PERSISTENT = P2Replay_Persistent - PPO_Persistent
        for existing_key in ["W512_Persistent_on", "W512_Persistent"]:
            if existing_key in existing:
                cross_comparisons["REPLAY_EFFECT_PERSISTENT"] = paired_compare(
                    all_results["W512_Persistent_P2Replay"],
                    existing[existing_key],
                    "REPLAY_EFFECT_PERSISTENT")
                break
        # REPLAY_EFFECT_RESET = P2Replay_Reset128 - PPO_Reset128
        for existing_key in ["W512_Reset128_on", "W512_Reset128"]:
            if existing_key in existing:
                cross_comparisons["REPLAY_EFFECT_RESET"] = paired_compare(
                    all_results["W512_Reset128_P2Replay"],
                    existing[existing_key],
                    "REPLAY_EFFECT_RESET")
                break

    output = {
        "evaluator": "eval_w512_p2replay.py",
        "evaluator_sha256": EVAL_SHA256,
        "protocol": {
            "num_worlds": NUM_ENVS, "seed": EVAL_SEED,
            "max_steps": NUM_STEPS, "policy": "stochastic",
            "start": "S4_dark_native", "gpu": args.gpu,
        },
        "arms": all_results,
        "paired_comparisons": {
            "CARRY_WITH_REPLAY": cmp_p2replay,
            **cross_comparisons,
        },
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n[eval] Results saved to {args.out}", flush=True)

    # Print summary table
    print(f"\n{'='*72}")
    print(f"{'Arm':<40} {'DK SR':>8} {'n':>5} {'floor3':>8} {'death':>6} {'eplen':>6}")
    print(f"{'-'*72}")
    for arm_name, res in all_results.items():
        print(f"{arm_name:<40} {res['SR']*100:>7.2f}% {res['n_success']:>5} "
              f"{res['floor3_reach_rate']*100:>7.2f}% {res['n_died']:>6} "
              f"{res['mean_episode_length']:>6.0f}")
    print(f"{'='*72}")
    print(f"\nCARRY_WITH_REPLAY: {cmp_p2replay['SR_delta_pp']:+.2f}pp "
          f"(p={cmp_p2replay['mcnemar']['mcnemar_p']:.4f})")
    for k, v in cross_comparisons.items():
        print(f"{k}: {v['SR_delta_pp']:+.2f}pp (p={v['mcnemar']['mcnemar_p']:.4f})")


if __name__ == "__main__":
    main()
