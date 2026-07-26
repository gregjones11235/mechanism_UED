#!/usr/bin/env python3
"""§14 Control 64-world Stage4-native evaluator (derived, frozen protocol).

Derived from the parent dual-caliber evaluator (parent SHA
06221187ac06d7da59dac64e6273abfc865b3baafdc75615e0808fc5065d26e2 — the validated
session175 three-model evaluator). Protocol UNCHANGED: seed 42, 64 episodes,
S4_dark scaffold (floor-2 spawn, gate open, winner kit, up-ladder removed,
needs_depletion 0.3) + OFFICIAL_FULL, DEFEAT_KOBOLD ever-set dual-channel,
stochastic policy (pi.sample), 4096 max steps, BIG net config, load_weights_only
reused unchanged.

Changes vs parent (documented, NOT a protocol change):
  1. CKPT / CKPT_LABEL / OUT are CLI args (--ckpt/--ckpt_label/--out) so one
     script serves every Control grid LR (parent hardcoded 3 literals).
  2. Added a CUMULATIVE POLICY KL diagnostic for §14 health-gate item 4
     (KL_MAX_RUN=0.1, cumulative over the 24576 run). Definition (stated exactly,
     not fabricated): along the TRAINED policy's 64-world STAGE4_NATIVE rollout
     trajectory, at each active (not-yet-finished) step compute
       kl_t = sum_a p_trained(a|s_t) * [log p_trained(a|s_t) - log p_base(a|s_t)]
     where p_base is the ckpt17500 (session175 start) policy evaluated on the
     SAME (memory, obs, mask) the trained policy visits. cumulative_policy_kl_mean
     is kl_t averaged over all active steps. This is the eval-time operational
     measure of "how far the trained policy drifted from its ckpt17500 start".

GPU0 only (UUID bound). Read-only w.r.t. both checkpoints; writes only under --out.
"""
import argparse, hashlib, json, os, sys, time

GPU_UUID = "GPU-e8c08612-c22a-c8a4-6df5-affb2dd1f9a6"   # GPU0
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True, help="trained Control checkpoint dir (step subdir)")
ap.add_argument("--ckpt_label", required=True)
ap.add_argument("--out", required=True, help="output dir (results/ written inside)")
ap.add_argument("--base_ckpt", default=(
    "/home/oseasy/incoming/henry_work_20260721T105300/extracted/"
    "Henry_work/base_ckpt_v7fix55_armA_s0/rl_checkpoints/17500"),
    help="reference start checkpoint for cumulative policy KL (default ckpt17500)")
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

# ---- BIG network config (matches session175 / frozen v7 evaluator) ----
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
RES = os.path.join(OUT, "results")
os.makedirs(RES, exist_ok=True)

DK = int(Achievement.DEFEAT_KOBOLD.value)            # 41
SEWERS = int(Achievement.ENTER_SEWERS.value)         # 30
KOBOLD_FLOOR_DIAG = 3
NUM_ENVS = 64
NUM_STEPS = 4096
EVAL_SEED = 42

with open(__file__, "rb") as f:
    EVAL_SHA256 = hashlib.sha256(f.read()).hexdigest()

# ---- task classes (UNCHANGED from parent) ----
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
FULL_TASK_CODE = '''
import jax
from craftax.craftax.constants import Achievement
from minicraftax.craftax_state import TaskParams
from minicraftax.tasks.base_task import BaseTask
from minicraftax.world_builder import WorldBuilder
class Env(BaseTask):
    def __init__(self, sp, p):
        super().__init__(sp, p)
        self.relevant_achievements=[Achievement.DEFEAT_KOBOLD]; self.completed_achievements=[]; self.label="FULL_DEFEAT_KOBOLD"
    def get_task_params(self): return TaskParams()
    def generate_world(self, rng):
        rng,_r=jax.random.split(rng); b=WorldBuilder(_r,self.static_params,self.params)
        return b.build(rng)
'''

ctor = EnvParams(max_timesteps=NUM_STEPS)
table = jnp.array([get_achievement_multi_hot([Achievement.DEFEAT_KOBOLD])], dtype=jnp.float32)
EMB = int(table.shape[1])

ns4 = {}; exec(S4_TASK_CODE, ns4); S4Cls = ns4["Env"]
nfl = {}; exec(FULL_TASK_CODE, nfl); FullCls = nfl["Env"]

s4_base = MultiTaskMiniCraftaxEnv([S4Cls], StaticEnvParams(), ctor, True,
                                  conditioning_type="embedding", embedding_size=EMB)
full_base = MultiTaskMiniCraftaxEnv([FullCls], StaticEnvParams(), ctor, True,
                                    conditioning_type="embedding", embedding_size=EMB)
ACTION_DIM = int(s4_base.action_space(ctor).n)
OBS_DIM = int(s4_base.observation_space(ctor).shape[0])

print("=" * 72, flush=True)
print("Control grid evaluator (derived from parent 06221187)", flush=True)
print(f"  checkpoint: {CKPT}  label: {CKPT_LABEL}", flush=True)
print(f"  base (KL ref): {BASE_CKPT}", flush=True)
print(f"  GPU_UUID: {GPU_UUID}  devices: {[str(d) for d in jax.devices()]}", flush=True)
print(f"  obs_dim={OBS_DIM} action_dim={ACTION_DIM} embedding_size={EMB}", flush=True)
print(f"  NUM_ENVS={NUM_ENVS} NUM_STEPS={NUM_STEPS} seed={EVAL_SEED}", flush=True)
print(f"  evaluator_sha256={EVAL_SHA256}", flush=True)
print("=" * 72, flush=True)

# ---- load trained checkpoint + base (ckpt17500) reference, ONCE each ----
t0 = time.time()
ts = load_weights_only(CKPT, s4_base, ctor, cfg, load_opt_state=False)
PARAM_LEAVES = len(jax.tree_util.tree_leaves(ts.params))
ts_base = load_weights_only(BASE_CKPT, s4_base, ctor, cfg, load_opt_state=False)
network = ActorCriticTransformer(action_dim=ACTION_DIM, activation=cfg.activation,
    encoder_size=cfg.embed_size, hidden_layers=cfg.hidden_layers, num_heads=cfg.num_heads,
    qkv_features=cfg.qkv_features, num_layers=cfg.num_layers, gating=cfg.gating,
    gating_bias=cfg.gating_bias)
print(f"[load] trained leaves={PARAM_LEAVES}  base loaded  ({time.time()-t0:.1f}s)", flush=True)


def _kl_from_logits(lt, lb):
    """KL(Categorical(lt) || Categorical(lb)) per row, from unnormalized logits."""
    logpt = lt - jax.scipy.special.logsumexp(lt, axis=-1, keepdims=True)
    logpb = lb - jax.scipy.special.logsumexp(lb, axis=-1, keepdims=True)
    pt = jax.nn.softmax(lt, axis=-1)
    return (pt * (logpt - logpb)).sum(axis=-1)


def run_caliber(name, base_env, spawn_floor, compute_kl):
    env = DistributedMultiTaskOptimisticLogWrapper(base_env, jax.random.PRNGKey(0),
            NUM_ENVS, 1, 16, jnp.array([1.0]), table)
    memories = jnp.zeros((NUM_ENVS, cfg.window_mem, cfg.num_layers, cfg.embed_size))
    mem_mask = jnp.zeros((NUM_ENVS, cfg.num_heads, 1, cfg.window_mem + 1), dtype=jnp.bool_)
    mem_idx = jnp.zeros((NUM_ENVS,), dtype=jnp.int32) + (cfg.window_mem + 1)

    def _step(carry, _):
        (log_state, memories, mem_mask, mem_idx, last_obs, done, finished, ep_len,
         max_floor, seen, info_acc, ep_return, sewers, flip_floor, kl_sum, kl_cnt, rng) = carry
        mem_idx = jnp.where(done, cfg.window_mem, jnp.clip(mem_idx - 1, 0, cfg.window_mem))
        mem_mask = jnp.where(done[:, None, None, None], jnp.zeros_like(mem_mask), mem_mask)
        ohot = jax.nn.one_hot(mem_idx, cfg.window_mem + 1)[:, None, None, :].repeat(cfg.num_heads, 1)
        mem_mask = jnp.logical_or(mem_mask, ohot)
        rng, a_rng, s_rng = jax.random.split(rng, 3)
        pi, _, mem_out = network.apply(ts.params, memories, last_obs, mem_mask,
                                       method=network.model_forward_eval)
        action = pi.sample(seed=a_rng)   # STOCHASTIC
        # cumulative policy KL vs base (ckpt17500) on the SAME visited state
        if compute_kl:
            pi_b, _, _ = network.apply(ts_base.params, memories, last_obs, mem_mask,
                                       method=network.model_forward_eval)
            lt = getattr(pi, "logits", None)
            lb = getattr(pi_b, "logits", None)
            if lt is None or lb is None:
                raise RuntimeError("policy distribution exposes no .logits; cannot compute KL")
            kl_t = _kl_from_logits(lt, lb)
        else:
            kl_t = jnp.zeros((NUM_ENVS,), dtype=jnp.float32)
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
        keys = [k for k in info if "Achievements" in k and "kobold" in k.lower()]
        if keys:
            info_acc = info_acc + jnp.asarray(info[keys[0]], jnp.float32).reshape(-1) * active
        finished = finished | next_done
        return (next_log_state, memories, mem_mask, mem_idx, next_obs, next_done, finished,
                ep_len, max_floor, seen, info_acc, ep_return, sewers, flip_floor,
                kl_sum, kl_cnt, rng), None

    rng = jax.random.PRNGKey(EVAL_SEED)
    rng, reset_rng = jax.random.split(rng)
    obsv, log_state = env.reset(reset_rng, ctor)
    init = (log_state, memories, mem_mask, mem_idx, obsv,
            jnp.zeros((NUM_ENVS,), dtype=jnp.bool_),
            jnp.zeros((NUM_ENVS,), dtype=jnp.bool_),
            jnp.zeros((NUM_ENVS,), dtype=jnp.int32),
            jnp.full((NUM_ENVS,), spawn_floor, dtype=jnp.int32),
            jnp.zeros((NUM_ENVS,), dtype=jnp.bool_),
            jnp.zeros((NUM_ENVS,), dtype=jnp.float32),
            jnp.zeros((NUM_ENVS,), dtype=jnp.float32),
            jnp.zeros((NUM_ENVS,), dtype=jnp.bool_),
            jnp.full((NUM_ENVS,), -1, dtype=jnp.int32),
            jnp.zeros((NUM_ENVS,), dtype=jnp.float32),
            jnp.zeros((NUM_ENVS,), dtype=jnp.float32),
            rng)
    t0 = time.time()
    final, _ = jax.lax.scan(_step, init, None, NUM_STEPS)
    (_, _, _, _, _, _, finished, ep_len, max_floor, seen, info_acc,
     ep_return, sewers, flip_floor, kl_sum, kl_cnt, _) = final
    jax.block_until_ready(final)
    roll_time = time.time() - t0

    finished_np = np.asarray(finished); ep_len_np = np.asarray(ep_len)
    max_floor_np = np.asarray(max_floor); seen_np = np.asarray(seen)
    info_acc_np = np.asarray(info_acc); ep_return_np = np.asarray(ep_return)
    sewers_np = np.asarray(sewers); flip_floor_np = np.asarray(flip_floor)
    kl_sum_np = np.asarray(kl_sum); kl_cnt_np = np.asarray(kl_cnt)

    success_np = seen_np | (info_acc_np > 0)
    timeout_np = finished_np & (ep_len_np >= NUM_STEPS) & ~success_np
    died_np = finished_np & ~success_np & ~timeout_np
    n_not_finished = int(np.sum(~finished_np))
    if n_not_finished > 0:
        timeout_np = timeout_np | ~finished_np

    n_success = int(success_np.sum()); n_died = int(died_np.sum()); n_timeout = int(timeout_np.sum())
    n_sewers = int(sewers_np.sum()); n_floor3 = int((max_floor_np >= 3).sum())
    n_kobold_enc = int((max_floor_np >= KOBOLD_FLOOR_DIAG).sum())
    sr = n_success / NUM_ENVS
    ff = flip_floor_np[success_np]
    flip_counts = {int(k): int(v) for k, v in zip(*np.unique(ff, return_counts=True))} if n_success else {}
    # cumulative policy KL vs base, averaged over active steps (sum kl / sum active)
    kl_total = float(kl_sum_np.sum()); kl_n = float(kl_cnt_np.sum())
    cumulative_kl_mean = (kl_total / kl_n) if kl_n > 0 else float("nan")

    jl_path = os.path.join(RES, f"{name}_episodes.jsonl")
    with open(jl_path, "w") as f:
        for i in range(NUM_ENVS):
            rec = dict(caliber=name, checkpoint=CKPT_LABEL, evaluation_seed=EVAL_SEED,
                episode_idx=i, policy_mode="stochastic",
                episode_length=int(ep_len_np[i]), **{"return": float(ep_return_np[i])},
                DEFEAT_KOBOLD=bool(success_np[i]), ENTER_SEWERS=bool(sewers_np[i]),
                floor3_reach=bool(max_floor_np[i] >= 3),
                kobold_encounter=bool(max_floor_np[i] >= KOBOLD_FLOOR_DIAG),
                death=bool(died_np[i]), timeout=bool(timeout_np[i]),
                max_floor=int(max_floor_np[i]), flip_floor=int(flip_floor_np[i]))
            f.write(json.dumps(rec) + "\n")

    summary = dict(caliber=name, checkpoint=CKPT, checkpoint_label=CKPT_LABEL,
        base_checkpoint=BASE_CKPT, num_episodes=NUM_ENVS, evaluation_seed=EVAL_SEED,
        policy_mode="stochastic", kobold_floor_diag=KOBOLD_FLOOR_DIAG, spawn_floor=spawn_floor,
        SR=n_success / NUM_ENVS, n_success=n_success, n_died=n_died, n_timeout=n_timeout,
        n_not_finished=n_not_finished, ENTER_SEWERS_rate=n_sewers / NUM_ENVS,
        floor3_reach_rate=n_floor3 / NUM_ENVS, kobold_encounter_rate=n_kobold_enc / NUM_ENVS,
        death_rate=n_died / NUM_ENVS, timeout_rate=n_timeout / NUM_ENVS,
        mean_episode_length=float(ep_len_np.mean()), median_episode_length=float(np.median(ep_len_np)),
        max_floor_max=int(max_floor_np.max()), max_floor_median=float(np.median(max_floor_np)),
        flip_floor_counts=flip_counts,
        cumulative_policy_kl_mean=cumulative_kl_mean,
        cumulative_policy_kl_sum=kl_total, cumulative_policy_kl_active_steps=kl_n,
        rollout_time_s=round(roll_time, 1),
        evaluator_sha256=EVAL_SHA256, timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with open(os.path.join(RES, f"{name}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[{name}] SR={sr*100:.2f}% ({n_success}/{NUM_ENVS})  "
          f"ENTER_SEWERS={n_sewers}/{NUM_ENVS}  floor3+={n_floor3}/{NUM_ENVS}  "
          f"died={n_died} timeout={n_timeout} not_finished={n_not_finished}", flush=True)
    print(f"[{name}] cum_policy_KL_vs_base={cumulative_kl_mean:.5f}  "
          f"flip_floor_counts={flip_counts}  mean_ep_len={ep_len_np.mean():.1f}  ({roll_time:.1f}s)",
          flush=True)
    return summary


s_stage4 = run_caliber("STAGE4_NATIVE", s4_base, spawn_floor=2, compute_kl=True)
s_full = run_caliber("OFFICIAL_FULL", full_base, spawn_floor=0, compute_kl=False)

combined = dict(
    pilot="control_grid_eval", checkpoint=CKPT, checkpoint_label=CKPT_LABEL,
    base_checkpoint=BASE_CKPT, gpu_uuid=GPU_UUID, devices=[str(d) for d in jax.devices()],
    evaluator_sha256=EVAL_SHA256, parent_evaluator_sha="06221187ac06d7da59dac64e6273abfc865b3baafdc75615e0808fc5065d26e2",
    v7_codebase=V7_SRC, obs_dim=OBS_DIM, action_dim=ACTION_DIM, embedding_size=EMB,
    DK_idx=DK, ENTER_SEWERS_idx=SEWERS, kobold_floor_diag=KOBOLD_FLOOR_DIAG,
    num_envs=NUM_ENVS, num_steps=NUM_STEPS, evaluation_seed=EVAL_SEED,
    param_leaves=PARAM_LEAVES,
    SR_STAGE4_NATIVE_DEFEAT_KOBOLD=s_stage4["SR"],
    SR_OFFICIAL_FULL_DEFEAT_KOBOLD=s_full["SR"],
    STAGE4_NATIVE_cumulative_policy_kl_mean=s_stage4["cumulative_policy_kl_mean"],
    STAGE4_NATIVE=s_stage4, OFFICIAL_FULL=s_full,
    timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
with open(os.path.join(RES, "control_eval_combined.json"), "w") as f:
    json.dump(combined, f, indent=2, default=str)

print("\n" + "=" * 72, flush=True)
print(f"SR_STAGE4_NATIVE = {s_stage4['SR']*100:.2f}%  ({s_stage4['n_success']}/{NUM_ENVS})  "
      f"floor3={s_stage4['floor3_reach_rate']*100:.2f}%  "
      f"cum_KL={s_stage4['cumulative_policy_kl_mean']:.5f}", flush=True)
print(f"SR_OFFICIAL_FULL = {s_full['SR']*100:.2f}%  ({s_full['n_success']}/{NUM_ENVS})", flush=True)
print(f"evaluator_sha256 = {EVAL_SHA256}", flush=True)
print("DONE", flush=True)
