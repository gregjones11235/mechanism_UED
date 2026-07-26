#!/usr/bin/env python3
"""§14 Control REVISED-gate 256-world PAIRED Stage4-native evaluator.

Derived from eval_control_64ep.py (which itself derives from parent dual-caliber
06221187). Purpose: evaluate ONE checkpoint on NUM_WORLDS fresh Stage4_native
worlds. Run this script TWICE with the SAME --num_worlds and SAME --seed_base
(baseline=ckpt17500, then control=lr_2e-5/24576) -> world i is the IDENTICAL
environment in both runs (the wrapper splits the reset key into num_worlds
deterministic subkeys) AND the per-step action-sampling RNG stream is identical
(common random numbers). The ONLY thing that differs between the two runs is the
policy params -> a clean PAIRED comparison (McNemar on per-world outcomes).

Changes vs eval_control_64ep.py (documented, NOT a protocol change):
  1. NUM_ENVS = --num_worlds (default 256, was hardcoded 64).
  2. EVAL_SEED = --seed_base (default 100000, a FRESH master seed distinct from the
     original 64-world seed 42) so all worlds are fresh.
  3. STAGE4_NATIVE only (OFFICIAL_FULL dropped — not part of the revised health
     gate; saves one rollout). spawn_floor=2, max_steps=4096 UNCHANGED.
  4. --compute_kl flag: cumulative-vs-ckpt17500 KL is now DIAGNOSTIC only, so the
     baseline run sets it False (KL trivially 0); the control run sets it True.
  5. episodes.jsonl records seed_base + episode_idx so the paired analyzer can
     match world i across the two runs unambiguously.

Stage4 scaffold, DEFEAT_KOBOLD ever-set dual-channel, stochastic pi.sample,
BIG-net config, load_weights_only — all UNCHANGED from the parent.
GPU0 only (UUID bound). Read-only w.r.t. checkpoints; writes only under --out.
"""
import argparse, hashlib, json, os, sys, time

GPU_UUID = "GPU-e8c08612-c22a-c8a4-6df5-affb2dd1f9a6"   # GPU0
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--ckpt_label", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--base_ckpt", default=(
    "/home/oseasy/incoming/henry_work_20260721T105300/extracted/"
    "Henry_work/base_ckpt_v7fix55_armA_s0/rl_checkpoints/17500"))
ap.add_argument("--num_worlds", type=int, default=256)
ap.add_argument("--seed_base", type=int, default=100000)
ap.add_argument("--compute_kl", type=int, default=1)
args = ap.parse_args()

import jax, jax.numpy as jnp
import numpy as np

V7_SRC = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src"
if V7_SRC not in sys.path:
    sys.path.insert(0, V7_SRC)

from craftax.craftax.constants import Achievement
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from dicode.network import ActorCriticTransformer
from dicode.task_utils import get_achievement_multi_hot
from dicode.utils.general.train_state_utils import load_weights_only
from dicode.wrappers_cl import DistributedMultiTaskOptimisticLogWrapper
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv

_cfg = dict(activation="relu", embed_size=256, hidden_layers=256, num_heads=8,
            qkv_features=256, num_layers=2, gating=True, gating_bias=2.0,
            window_mem=128, window_grad=64, anneal_lr=False, lr=2e-4, min_lr=2e-6,
            max_grad_norm=1.0, total_timesteps=2005401600, num_envs=1024, num_steps=128,
            update_epochs=4, num_minibatches=8, max_updates_per_session=500)
cfg = type("C", (), _cfg)()

CKPT = args.ckpt
CKPT_LABEL = args.ckpt_label
OUT = args.out
BASE_CKPT = args.base_ckpt
NUM_ENVS = int(args.num_worlds)
EVAL_SEED = int(args.seed_base)
COMPUTE_KL = bool(args.compute_kl)
RES = os.path.join(OUT, "results")
os.makedirs(RES, exist_ok=True)

DK = int(Achievement.DEFEAT_KOBOLD.value)
SEWERS = int(Achievement.ENTER_SEWERS.value)
KOBOLD_FLOOR_DIAG = 3
NUM_STEPS = 4096

with open(__file__, "rb") as f:
    EVAL_SHA256 = hashlib.sha256(f.read()).hexdigest()

S4_TASK_CODE = '''
import jax
from craftax.craftax.constants import Achievement, ItemType
from minicraftax.craftax_state import TaskParams
from minicraftax.tasks.base_task import BaseTask
from minicraftax.world_builder import WorldBuilder
class Env(BaseTask):
    def __init__(self, sp, p):
        super().__init__(sp, p)
        self.relevant_achievements=[Achievement.DEFEAT_KOBOLD]; self.completed_achievements=[]; self.label="DEFEAT_KOBOLD"
    def get_task_params(self): return TaskParams(needs_depletion_multiplier=0.3)
    def generate_world(self, rng):
        rng,_r=jax.random.split(rng); b=WorldBuilder(_r,self.static_params,self.params)
        b.set_starting_floor(2); b.set_monsters_killed(2,8)
        b.set_player_inventory({"wood":7,"stone":27,"coal":3,"iron":3,"sapling":1,"pickaxe":3,"sword":3,"bow":1,"arrows":7,"torches":10})
        s=b.build(rng); up=b.ladders_up[2]
        return s.replace(item_map=s.item_map.at[2,up[0],up[1]].set(ItemType.NONE.value))
'''
ctor = EnvParams(max_timesteps=NUM_STEPS)
table = jnp.array([get_achievement_multi_hot([Achievement.DEFEAT_KOBOLD])], dtype=jnp.float32)
EMB = int(table.shape[1])

ns4 = {}; exec(S4_TASK_CODE, ns4); S4Cls = ns4["Env"]
s4_base = MultiTaskMiniCraftaxEnv([S4Cls], StaticEnvParams(), ctor, True,
                                  conditioning_type="embedding", embedding_size=EMB)
ACTION_DIM = int(s4_base.action_space(ctor).n)
OBS_DIM = int(s4_base.observation_space(ctor).shape[0])

print("=" * 72, flush=True)
print("PAIRED 256-world Stage4-native evaluator (derived from eval_control_64ep)", flush=True)
print(f"  checkpoint: {CKPT}  label: {CKPT_LABEL}", flush=True)
print(f"  base (KL ref): {BASE_CKPT}  compute_kl={COMPUTE_KL}", flush=True)
print(f"  GPU_UUID: {GPU_UUID}  devices: {[str(d) for d in jax.devices()]}", flush=True)
print(f"  obs_dim={OBS_DIM} action_dim={ACTION_DIM} embedding_size={EMB}", flush=True)
print(f"  NUM_WORLDS={NUM_ENVS} NUM_STEPS={NUM_STEPS} seed_base={EVAL_SEED}", flush=True)
print(f"  evaluator_sha256={EVAL_SHA256}", flush=True)
print("=" * 72, flush=True)

t0 = time.time()
ts = load_weights_only(CKPT, s4_base, ctor, cfg, load_opt_state=False)
PARAM_LEAVES = len(jax.tree_util.tree_leaves(ts.params))
if COMPUTE_KL:
    ts_base = load_weights_only(BASE_CKPT, s4_base, ctor, cfg, load_opt_state=False)
else:
    ts_base = None
network = ActorCriticTransformer(action_dim=ACTION_DIM, activation=cfg.activation,
    encoder_size=cfg.embed_size, hidden_layers=cfg.hidden_layers, num_heads=cfg.num_heads,
    qkv_features=cfg.qkv_features, num_layers=cfg.num_layers, gating=cfg.gating,
    gating_bias=cfg.gating_bias)
print(f"[load] leaves={PARAM_LEAVES}  ({time.time()-t0:.1f}s)", flush=True)


def _kl_from_logits(lt, lb):
    logpt = lt - jax.scipy.special.logsumexp(lt, axis=-1, keepdims=True)
    logpb = lb - jax.scipy.special.logsumexp(lb, axis=-1, keepdims=True)
    pt = jax.nn.softmax(lt, axis=-1)
    return (pt * (logpt - logpb)).sum(axis=-1)


def run_stage4():
    env = DistributedMultiTaskOptimisticLogWrapper(s4_base, jax.random.PRNGKey(0),
            NUM_ENVS, 1, 16, jnp.array([1.0]), table)
    memories = jnp.zeros((NUM_ENVS, cfg.window_mem, cfg.num_layers, cfg.embed_size))
    mem_mask = jnp.zeros((NUM_ENVS, cfg.num_heads, 1, cfg.window_mem + 1), dtype=jnp.bool_)
    mem_idx = jnp.zeros((NUM_ENVS,), dtype=jnp.int32) + (cfg.window_mem + 1)
    base_params = ts_base.params if COMPUTE_KL else ts.params

    def _step(carry, _):
        (log_state, memories, mem_mask, mem_idx, last_obs, done, finished, ep_len,
         max_floor, seen, info_acc, ep_return, sewers, flip_floor, kl_sum, kl_cnt,
         ent_tr_sum, ent_ba_sum, rng) = carry
        mem_idx = jnp.where(done, cfg.window_mem, jnp.clip(mem_idx - 1, 0, cfg.window_mem))
        mem_mask = jnp.where(done[:, None, None, None], jnp.zeros_like(mem_mask), mem_mask)
        ohot = jax.nn.one_hot(mem_idx, cfg.window_mem + 1)[:, None, None, :].repeat(cfg.num_heads, 1)
        mem_mask = jnp.logical_or(mem_mask, ohot)
        rng, a_rng, s_rng = jax.random.split(rng, 3)
        pi, _, mem_out = network.apply(ts.params, memories, last_obs, mem_mask,
                                       method=network.model_forward_eval)
        action = pi.sample(seed=a_rng)   # STOCHASTIC; a_rng identical across paired runs
        if COMPUTE_KL:
            pi_b, _, _ = network.apply(base_params, memories, last_obs, mem_mask,
                                       method=network.model_forward_eval)
            kl_t = _kl_from_logits(pi.logits, pi_b.logits)
            ent_t = pi.entropy()        # trained-policy entropy (nats) per env
            ent_b = pi_b.entropy()      # base-policy entropy on the SAME state
        else:
            kl_t = jnp.zeros((NUM_ENVS,), dtype=jnp.float32)
            ent_t = jnp.zeros((NUM_ENVS,), dtype=jnp.float32)
            ent_b = jnp.zeros((NUM_ENVS,), dtype=jnp.float32)
        memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(mem_out)
        pre = log_state.env_state
        pre_pl = pre.player_level
        pre_dk = pre.achievements[:, DK].astype(bool)
        pre_sw = pre.achievements[:, SEWERS].astype(bool)
        next_obs, next_log_state, reward, next_done, info = env.step(s_rng, log_state, action, ctor)
        active = (~finished).astype(jnp.float32)
        ep_len = ep_len + active.astype(jnp.int32)
        ep_return = ep_return + jnp.asarray(reward, jnp.float32).reshape(-1) * active
        max_floor = jnp.where(active > 0, jnp.maximum(max_floor, pre_pl), max_floor)
        newly = pre_dk & (active > 0) & ~seen
        flip_floor = jnp.where(newly, pre_pl, flip_floor)
        seen = seen | (pre_dk & (active > 0))
        sewers = sewers | (pre_sw & (active > 0))
        kl_sum = kl_sum + kl_t * active
        kl_cnt = kl_cnt + active
        ent_tr_sum = ent_tr_sum + ent_t * active
        ent_ba_sum = ent_ba_sum + ent_b * active
        keys = [k for k in info if "Achievements" in k and "kobold" in k.lower()]
        if keys:
            info_acc = info_acc + jnp.asarray(info[keys[0]], jnp.float32).reshape(-1) * active
        finished = finished | next_done
        return (next_log_state, memories, mem_mask, mem_idx, next_obs, next_done, finished,
                ep_len, max_floor, seen, info_acc, ep_return, sewers, flip_floor,
                kl_sum, kl_cnt, ent_tr_sum, ent_ba_sum, rng), None

    rng = jax.random.PRNGKey(EVAL_SEED)
    rng, reset_rng = jax.random.split(rng)
    obsv, log_state = env.reset(reset_rng, ctor)
    init = (log_state, memories, mem_mask, mem_idx, obsv,
            jnp.zeros((NUM_ENVS,), dtype=jnp.bool_),
            jnp.zeros((NUM_ENVS,), dtype=jnp.bool_),
            jnp.zeros((NUM_ENVS,), dtype=jnp.int32),
            jnp.full((NUM_ENVS,), 2, dtype=jnp.int32),
            jnp.zeros((NUM_ENVS,), dtype=jnp.bool_),
            jnp.zeros((NUM_ENVS,), dtype=jnp.float32),
            jnp.zeros((NUM_ENVS,), dtype=jnp.float32),
            jnp.zeros((NUM_ENVS,), dtype=jnp.bool_),
            jnp.full((NUM_ENVS,), -1, dtype=jnp.int32),
            jnp.zeros((NUM_ENVS,), dtype=jnp.float32),
            jnp.zeros((NUM_ENVS,), dtype=jnp.float32),
            jnp.zeros((NUM_ENVS,), dtype=jnp.float32),
            jnp.zeros((NUM_ENVS,), dtype=jnp.float32),
            rng)
    t0 = time.time()
    final, _ = jax.lax.scan(_step, init, None, NUM_STEPS)
    (_, _, _, _, _, _, finished, ep_len, max_floor, seen, info_acc,
     ep_return, sewers, flip_floor, kl_sum, kl_cnt, ent_tr_sum, ent_ba_sum, _) = final
    jax.block_until_ready(final)
    roll_time = time.time() - t0

    finished_np = np.asarray(finished); ep_len_np = np.asarray(ep_len)
    max_floor_np = np.asarray(max_floor); seen_np = np.asarray(seen)
    info_acc_np = np.asarray(info_acc); ep_return_np = np.asarray(ep_return)
    sewers_np = np.asarray(sewers); flip_floor_np = np.asarray(flip_floor)
    kl_sum_np = np.asarray(kl_sum); kl_cnt_np = np.asarray(kl_cnt)
    ent_tr_np = np.asarray(ent_tr_sum); ent_ba_np = np.asarray(ent_ba_sum)

    success_np = seen_np | (info_acc_np > 0)
    timeout_np = finished_np & (ep_len_np >= NUM_STEPS) & ~success_np
    died_np = finished_np & ~success_np & ~timeout_np
    n_not_finished = int(np.sum(~finished_np))
    if n_not_finished > 0:
        timeout_np = timeout_np | ~finished_np

    n_success = int(success_np.sum()); n_died = int(died_np.sum()); n_timeout = int(timeout_np.sum())
    n_sewers = int(sewers_np.sum()); n_floor3 = int((max_floor_np >= 3).sum())
    n_floor3_and_dk = int(((max_floor_np >= 3) & success_np).sum())
    sr = n_success / NUM_ENVS
    kl_total = float(kl_sum_np.sum()); kl_n = float(kl_cnt_np.sum())
    cumulative_kl_mean = (kl_total / kl_n) if kl_n > 0 else float("nan")
    # entropy-collapse diagnostic (COMPUTE_KL only): mean entropy of the trained
    # policy and of the base policy, both over the SAME active steps the trained
    # policy visits. Max entropy over 43 actions = ln(43) ~= 3.76 nats. Collapse
    # flag uses the frozen training-time guard guard_session_entropy_min = 0.10.
    mean_ent_trained = (float(ent_tr_np.sum()) / kl_n) if (kl_n > 0 and COMPUTE_KL) else float("nan")
    mean_ent_base = (float(ent_ba_np.sum()) / kl_n) if (kl_n > 0 and COMPUTE_KL) else float("nan")
    ent_ratio_trained_over_base = (mean_ent_trained / mean_ent_base) if (COMPUTE_KL and mean_ent_base == mean_ent_base and mean_ent_base > 0) else float("nan")
    entropy_collapse = bool(COMPUTE_KL and (mean_ent_trained == mean_ent_trained) and mean_ent_trained < 0.10)

    jl_path = os.path.join(RES, f"{CKPT_LABEL}_episodes.jsonl")
    with open(jl_path, "w") as f:
        for i in range(NUM_ENVS):
            rec = dict(checkpoint_label=CKPT_LABEL, seed_base=EVAL_SEED,
                episode_idx=i, policy_mode="stochastic", spawn_floor=2,
                episode_length=int(ep_len_np[i]), **{"return": float(ep_return_np[i])},
                DEFEAT_KOBOLD=bool(success_np[i]), ENTER_SEWERS=bool(sewers_np[i]),
                floor3_reach=bool(max_floor_np[i] >= 3),
                death=bool(died_np[i]), timeout=bool(timeout_np[i]),
                max_floor=int(max_floor_np[i]), flip_floor=int(flip_floor_np[i]))
            f.write(json.dumps(rec) + "\n")

    summary = dict(checkpoint=CKPT, checkpoint_label=CKPT_LABEL, base_checkpoint=BASE_CKPT,
        num_worlds=NUM_ENVS, seed_base=EVAL_SEED, policy_mode="stochastic", spawn_floor=2,
        kobold_floor_diag=KOBOLD_FLOOR_DIAG, compute_kl=COMPUTE_KL,
        SR=sr, n_success=n_success, n_died=n_died, n_timeout=n_timeout,
        n_not_finished=n_not_finished,
        floor3_reach_rate=n_floor3 / NUM_ENVS, n_floor3=n_floor3,
        n_floor3_and_dk=n_floor3_and_dk,
        conditional_kill_given_floor3=(n_floor3_and_dk / n_floor3) if n_floor3 else float("nan"),
        ENTER_SEWERS_rate=n_sewers / NUM_ENVS,
        death_rate=n_died / NUM_ENVS, timeout_rate=n_timeout / NUM_ENVS,
        mean_episode_length=float(ep_len_np.mean()),
        cumulative_policy_kl_mean=cumulative_kl_mean,
        cumulative_policy_kl_active_steps=kl_n,
        mean_entropy_trained_policy_nats=mean_ent_trained,
        mean_entropy_base_policy_same_states_nats=mean_ent_base,
        entropy_ratio_trained_over_base=ent_ratio_trained_over_base,
        entropy_collapse_flag=entropy_collapse,
        max_possible_entropy_nats=float(np.log(ACTION_DIM)),
        rollout_time_s=round(roll_time, 1),
        evaluator_sha256=EVAL_SHA256,
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with open(os.path.join(RES, f"{CKPT_LABEL}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[{CKPT_LABEL}] SR={sr*100:.2f}% ({n_success}/{NUM_ENVS})  "
          f"floor3={n_floor3}/{NUM_ENVS}  cond_kill|floor3="
          f"{(n_floor3_and_dk / n_floor3) if n_floor3 else float('nan'):.3f}  "
          f"died={n_died} timeout={n_timeout}  cum_KL={cumulative_kl_mean:.5f}  "
          f"({roll_time:.1f}s)", flush=True)
    if COMPUTE_KL:
        print(f"[{CKPT_LABEL}] entropy: trained={mean_ent_trained:.4f} nats  "
              f"base(same states)={mean_ent_base:.4f} nats  "
              f"ratio={ent_ratio_trained_over_base:.4f}  "
              f"max={np.log(ACTION_DIM):.4f}  collapse_flag={entropy_collapse}", flush=True)
    return summary


s = run_stage4()
print("\n" + "=" * 72, flush=True)
print(f"SR_STAGE4_NATIVE = {s['SR']*100:.2f}%  ({s['n_success']}/{NUM_ENVS})  "
      f"floor3={s['floor3_reach_rate']*100:.2f}%", flush=True)
print(f"evaluator_sha256 = {EVAL_SHA256}", flush=True)
print("DONE", flush=True)
