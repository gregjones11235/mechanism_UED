#!/usr/bin/env python3
"""P7-EGOMAP 256-world PAIRED Stage4-native evaluator.

Follows the EXACT same protocol as eval_paired_256.py (SHA 51c37c27...) but
uses ActorCriticTransformerEgoMap with egomap_enabled=True. Loads params from
pkl (P7 checkpoint format) instead of orbax.

Protocol (IDENTICAL to eval_paired_256.py):
  - 256 worlds, seed_base=100000
  - Stage4-native, spawn_floor=2
  - max_steps=4096/world
  - stochastic (pi.sample)
  - DEFEAT_KOBOLD ever-set
  - terminal auto-reset前累计achievement
  - 主结果正好256条

Usage:
  python eval_p7_egomap_paired_256.py \
    --pkl /path/to/params_98304.pkl \
    --ckpt_label p7_egomap_98304 \
    --out /path/to/output \
    [--arm egomap|control]  # egomap=enabled, control=disabled
"""
import argparse, hashlib, json, os, sys, time, pickle

GPU_UUID = "GPU-3c7a2864-755b-7045-b293-6f80e748283f"   # GPU1
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID

ap = argparse.ArgumentParser()
ap.add_argument("--pkl", required=True, help="P7 pkl params file")
ap.add_argument("--ckpt_label", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--arm", choices=["egomap", "control"], default="egomap",
                help="egomap=egomap_enabled, control=egomap_disabled")
ap.add_argument("--num_worlds", type=int, default=256)
ap.add_argument("--seed_base", type=int, default=100000)
args = ap.parse_args()

import jax, jax.numpy as jnp
import numpy as np

V7_SRC = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src"
P7_SRC = "/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu1_p7_egomap/src"
for p in [V7_SRC, P7_SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

from craftax.craftax.constants import Achievement
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from dicode.task_utils import get_achievement_multi_hot
from dicode.wrappers_cl import DistributedMultiTaskOptimisticLogWrapper
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv
from network_egomap import ActorCriticTransformerEgoMap
import egomap as egomap_lib

_cfg = dict(activation="relu", embed_size=256, hidden_layers=256, num_heads=8,
            qkv_features=256, num_layers=2, gating=True, gating_bias=2.0,
            window_mem=128, window_grad=64)
cfg = type("C", (), _cfg)()

PKL = args.pkl
CKPT_LABEL = args.ckpt_label
OUT = args.out
EGOMAP_ENABLED = (args.arm == "egomap")
NUM_ENVS = int(args.num_worlds)
EVAL_SEED = int(args.seed_base)
RES = os.path.join(OUT, "results")
os.makedirs(RES, exist_ok=True)

DK = int(Achievement.DEFEAT_KOBOLD.value)
SEWERS = int(Achievement.ENTER_SEWERS.value)
NUM_STEPS = 4096

# EgoMap config (same as launcher_p7.py defaults)
EGOMAP_MAP_SIZE = 32
EGOMAP_NUM_FLOORS = 9
egomap_cfg = egomap_lib.EgoMapConfig(
    map_size=EGOMAP_MAP_SIZE, num_floors=EGOMAP_NUM_FLOORS, enabled=EGOMAP_ENABLED)

# Stage4 task code (IDENTICAL to eval_paired_256.py)
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
print("P7-EGOMAP 256-world Stage4-native paired evaluator", flush=True)
print(f"  pkl: {PKL}  label: {CKPT_LABEL}  arm: {args.arm}", flush=True)
print(f"  egomap_enabled: {EGOMAP_ENABLED}", flush=True)
print(f"  GPU_UUID: {GPU_UUID}  devices: {[str(d) for d in jax.devices()]}", flush=True)
print(f"  obs_dim={OBS_DIM} action_dim={ACTION_DIM} embedding_size={EMB}", flush=True)
print(f"  NUM_WORLDS={NUM_ENVS} NUM_STEPS={NUM_STEPS} seed_base={EVAL_SEED}", flush=True)
print("=" * 72, flush=True)

# Load params from pkl
t0 = time.time()
with open(PKL, "rb") as f:
    flat_params = pickle.load(f)
params = {"params": jax.tree_util.tree_map(jnp.asarray, flat_params)}
PARAM_LEAVES = len(jax.tree_util.tree_leaves(params))
print(f"[load] leaves={PARAM_LEAVES}  ({time.time()-t0:.1f}s)", flush=True)

# Build network
network = ActorCriticTransformerEgoMap(
    action_dim=ACTION_DIM, activation=cfg.activation,
    encoder_size=cfg.embed_size, hidden_layers=cfg.hidden_layers,
    num_heads=cfg.num_heads, qkv_features=cfg.qkv_features,
    num_layers=cfg.num_layers, gating=cfg.gating, gating_bias=cfg.gating_bias,
    egomap_channels=egomap_lib.N_MAP_CH, egomap_cnn_features=(16, 32))


def run_stage4():
    env = DistributedMultiTaskOptimisticLogWrapper(s4_base, jax.random.PRNGKey(0),
            NUM_ENVS, 1, 16, jnp.array([1.0]), table)
    memories = jnp.zeros((NUM_ENVS, cfg.window_mem, cfg.num_layers, cfg.embed_size))
    mem_mask = jnp.zeros((NUM_ENVS, cfg.num_heads, 1, cfg.window_mem + 1), dtype=jnp.bool_)
    mem_idx = jnp.zeros((NUM_ENVS,), dtype=jnp.int32) + (cfg.window_mem + 1)

    # Init egomap state
    egomap_state = egomap_lib.egomap_init_state(NUM_ENVS, egomap_cfg)

    def _step(carry, _):
        (log_state, memories, mem_mask, mem_idx, last_obs, done, finished, ep_len,
         max_floor, seen, info_acc, ep_return, sewers, flip_floor,
         egomap_state, rng) = carry
        mem_idx = jnp.where(done, cfg.window_mem, jnp.clip(mem_idx - 1, 0, cfg.window_mem))
        mem_mask = jnp.where(done[:, None, None, None], jnp.zeros_like(mem_mask), mem_mask)
        ohot = jax.nn.one_hot(mem_idx, cfg.window_mem + 1)[:, None, None, :].repeat(cfg.num_heads, 1)
        mem_mask = jnp.logical_or(mem_mask, ohot)
        rng, a_rng, s_rng = jax.random.split(rng, 3)

        # P7: read egomap features (map up to t-1, around pos_t)
        ego_features = egomap_lib.egomap_read(egomap_state, last_obs, egomap_cfg)

        pi, _, mem_out = network.apply(params, memories, last_obs, mem_mask,
                                       ego_features=ego_features,
                                       egomap_enabled=EGOMAP_ENABLED,
                                       method=network.model_forward_eval)
        action = pi.sample(seed=a_rng)   # STOCHASTIC

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
        keys = [k for k in info if "Achievements" in k and "kobold" in k.lower()]
        if keys:
            info_acc = info_acc + jnp.asarray(info[keys[0]], jnp.float32).reshape(-1) * active
        finished = finished | next_done

        # P7: update egomap carry (same ordering as training)
        egomap_state = egomap_lib.egomap_update(egomap_state, last_obs, action, next_done, egomap_cfg)

        return (next_log_state, memories, mem_mask, mem_idx, next_obs, next_done, finished,
                ep_len, max_floor, seen, info_acc, ep_return, sewers, flip_floor,
                egomap_state, rng), None

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
            egomap_state,
            rng)
    t0 = time.time()
    final, _ = jax.lax.scan(_step, init, None, NUM_STEPS)
    (_, _, _, _, _, _, finished, ep_len, max_floor, seen, info_acc,
     ep_return, sewers, flip_floor, _, _) = final
    jax.block_until_ready(final)
    roll_time = time.time() - t0

    finished_np = np.asarray(finished); ep_len_np = np.asarray(ep_len)
    max_floor_np = np.asarray(max_floor); seen_np = np.asarray(seen)
    info_acc_np = np.asarray(info_acc); ep_return_np = np.asarray(ep_return)
    sewers_np = np.asarray(sewers); flip_floor_np = np.asarray(flip_floor)

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

    jl_path = os.path.join(RES, f"{CKPT_LABEL}_episodes.jsonl")
    with open(jl_path, "w") as f:
        for i in range(NUM_ENVS):
            rec = dict(checkpoint_label=CKPT_LABEL, seed_base=EVAL_SEED,
                episode_idx=i, policy_mode="stochastic", spawn_floor=2,
                episode_length=int(ep_len_np[i]), **{"return": float(ep_return_np[i])},
                DEFEAT_KOBOLD=bool(success_np[i]), ENTER_SEWERS=bool(sewers_np[i]),
                floor3_reach=bool(max_floor_np[i] >= 3),
                death=bool(died_np[i]), timeout=bool(timeout_np[i]),
                max_floor=int(max_floor_np[i]), flip_floor=int(flip_floor_np[i]),
                egomap_enabled=EGOMAP_ENABLED)
            f.write(json.dumps(rec) + "\n")

    summary = dict(checkpoint=PKL, checkpoint_label=CKPT_LABEL,
        num_worlds=NUM_ENVS, seed_base=EVAL_SEED, policy_mode="stochastic", spawn_floor=2,
        egomap_enabled=EGOMAP_ENABLED, arm=args.arm,
        SR=sr, n_success=n_success, n_died=n_died, n_timeout=n_timeout,
        n_not_finished=n_not_finished,
        floor3_reach_rate=n_floor3 / NUM_ENVS, n_floor3=n_floor3,
        n_floor3_and_dk=n_floor3_and_dk,
        conditional_kill_given_floor3=(n_floor3_and_dk / n_floor3) if n_floor3 else float("nan"),
        ENTER_SEWERS_rate=n_sewers / NUM_ENVS,
        death_rate=n_died / NUM_ENVS, timeout_rate=n_timeout / NUM_ENVS,
        mean_episode_length=float(ep_len_np.mean()),
        rollout_time_s=round(roll_time, 1),
        param_leaves=PARAM_LEAVES,
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with open(os.path.join(RES, f"{CKPT_LABEL}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[{CKPT_LABEL}] SR={sr*100:.2f}% ({n_success}/{NUM_ENVS})  "
          f"floor3={n_floor3}/{NUM_ENVS}  cond_kill|floor3="
          f"{(n_floor3_and_dk / n_floor3) if n_floor3 else float('nan'):.3f}  "
          f"died={n_died} timeout={n_timeout}  "
          f"({roll_time:.1f}s)", flush=True)
    return summary


s = run_stage4()
print("\n" + "=" * 72, flush=True)
print(f"SR_STAGE4_NATIVE = {s['SR']*100:.2f}%  ({s['n_success']}/{NUM_ENVS})  "
      f"floor3={s['floor3_reach_rate']*100:.2f}%", flush=True)
print("DONE", flush=True)
