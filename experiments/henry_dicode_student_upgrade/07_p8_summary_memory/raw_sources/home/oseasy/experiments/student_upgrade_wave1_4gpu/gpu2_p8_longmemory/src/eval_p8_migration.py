#!/usr/bin/env python3
"""P8-LONGMEM-SUMMARY — frozen 64-world Stage4 migration-gate evaluator (GPU2).

Adapted byte-for-byte in PROTOCOL from the canonical frozen 64-world Stage4 evaluator
(fs_p2_v1_full_49152_64ep_stage4_eval_20260723/eval_fs_p2_v1_full_49152_64ep.py):
  seed 42, 64 episodes, S4_dark (floor-2 spawn, gate open, winner kit, up-ladder removed,
  needs_depletion 0.3), DEFEAT_KOBOLD ever-set dual-channel, STOCHASTIC policy (pi.sample),
  4096 max steps, DistributedMultiTaskOptimisticLogWrapper (optimistic_reset_ratio=16),
  condition on DEFEAT_KOBOLD embedding (size 67). Metrics: DK SR, floor3 reach (max_floor>=3),
  death / timeout / ENTER_SEWERS. The GTrXL short-term memory-mask machinery in _step is
  IDENTICAL to that evaluator (and to training's _env_step).

The ONLY addition for P8: when use_p8=True the forward is ActorCriticLongMem.forward_eval,
which carries a long-term summary state (longstate) through the scan and resets it on
true_done (info["returned_episode"]); the actor reads the long-term context. When
use_p8=False (Baseline) the forward is the teacher ActorCriticTransformer.model_forward_eval
(no long state) — this is the canonical Baseline = ckpt17500.

Variants evaluated (SAME protocol, SAME eval seed -> paired by world index):
  BASELINE   : teacher ckpt17500 (ActorCriticTransformer)           -> the reference Baseline
  P8_DISTILL : distilled init, long-mem ON (the 98304 entry point)
  P8_OFF     : distilled init, long-mem OFF (== teacher exactly)    -> feature-off path check
  P8_4096    : 4096 smoke ckpt, long-mem ON                          -> early-training health

Gate (design §4): P8 DK SR drop <= 5pp AND floor3 drop <= 5pp vs Baseline; feature-off drop ~= 0.
Read-only w.r.t. checkpoints; writes only under the P8 eval output dir. GPU2 only.
"""
import hashlib, json, os, sys, time, pickle, argparse

GPU_UUID = "GPU-8df11537-ab79-722d-606f-411966196c4c"   # GPU2 ONLY
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

import jax, jax.numpy as jnp
import numpy as np

P8_SRC = "/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu2_p8_longmemory/src"
V7_SRC = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src"
V7 = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB"
for p in (P8_SRC, V7_SRC, V7):
    if p not in sys.path:
        sys.path.insert(0, p)

from craftax.craftax.constants import Achievement
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from dicode.network import ActorCriticTransformer
from dicode.task_utils import get_achievement_multi_hot
from dicode.utils.general.train_state_utils import load_weights_only
from dicode.wrappers_cl import DistributedMultiTaskOptimisticLogWrapper
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv
from p8_network import ActorCriticLongMem, init_longstate

_cfg = dict(activation="relu", embed_size=256, hidden_layers=256, num_heads=8,
            qkv_features=256, num_layers=2, gating=True, gating_bias=2.0,
            window_mem=128, window_grad=64, anneal_lr=False, lr=2e-4, min_lr=2e-6,
            max_grad_norm=1.0, total_timesteps=2005401600, num_envs=1024, num_steps=128,
            update_epochs=4, num_minibatches=8, max_updates_per_session=500)
cfg = type("C", (), _cfg)()

TEACHER_CKPT = ("/home/oseasy/incoming/henry_work_20260721T105300/extracted/"
                "Henry_work/base_ckpt_v7fix55_armA_s0/rl_checkpoints/17500")
DISTILL_PARAMS = "/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu2_p8_longmemory/distill/P8_DISTILLED_INIT/params.pkl"
P8_4096_PARAMS = "/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu2_p8_longmemory/smoke/smoke_A_ckpt/4096/full_state.pkl"

DK = int(Achievement.DEFEAT_KOBOLD.value)            # 41
SEWERS = int(Achievement.ENTER_SEWERS.value)         # 30
KOBOLD_FLOOR_DIAG = 3
NUM_ENVS = 64
NUM_STEPS = 4096
EVAL_SEED = 42

with open(__file__, "rb") as f:
    EVAL_SHA256 = hashlib.sha256(f.read()).hexdigest()

S4_TASK_PATH = "/home/oseasy/experiments/p2_v1_20260722/evidence/s4_task_code.py"
with open(S4_TASK_PATH) as f:
    S4_TASK_CODE = f.read()

ctor = EnvParams(max_timesteps=NUM_STEPS)
table = jnp.array([get_achievement_multi_hot([Achievement.DEFEAT_KOBOLD])], dtype=jnp.float32)
EMB = int(table.shape[1])
ns4 = {}; exec(S4_TASK_CODE, ns4); S4Cls = ns4["Env"]
s4_base = MultiTaskMiniCraftaxEnv([S4Cls], StaticEnvParams(), ctor, True,
                                  conditioning_type="embedding", embedding_size=EMB)
ACTION_DIM = int(s4_base.action_space(ctor).n)
OBS_DIM = int(s4_base.observation_space(ctor).shape[0])

print("=" * 72, flush=True)
print("P8-LONGMEM migration-gate evaluator (frozen 64-world Stage4)", flush=True)
print(f"  GPU_UUID={GPU_UUID} devices={[str(d) for d in jax.devices()]}", flush=True)
print(f"  obs_dim={OBS_DIM} action_dim={ACTION_DIM} emb={EMB} DK_idx={DK}", flush=True)
print(f"  NUM_ENVS={NUM_ENVS} NUM_STEPS={NUM_STEPS} seed={EVAL_SEED} policy=stochastic", flush=True)
print(f"  evaluator_sha256={EVAL_SHA256}", flush=True)
print("=" * 72, flush=True)

# ---- networks ----
teacher_net = ActorCriticTransformer(action_dim=ACTION_DIM, activation=cfg.activation,
    encoder_size=cfg.embed_size, hidden_layers=cfg.hidden_layers, num_heads=cfg.num_heads,
    qkv_features=cfg.qkv_features, num_layers=cfg.num_layers, gating=cfg.gating, gating_bias=cfg.gating_bias)
p8_net = ActorCriticLongMem(action_dim=ACTION_DIM, activation=cfg.activation,
    encoder_size=cfg.embed_size, hidden_layers=cfg.hidden_layers, num_heads=cfg.num_heads,
    qkv_features=cfg.qkv_features, num_layers=cfg.num_layers, gating=cfg.gating, gating_bias=cfg.gating_bias,
    use_longmem=True)
p8_net_off = ActorCriticLongMem(action_dim=ACTION_DIM, activation=cfg.activation,
    encoder_size=cfg.embed_size, hidden_layers=cfg.hidden_layers, num_heads=cfg.num_heads,
    qkv_features=cfg.qkv_features, num_layers=cfg.num_layers, gating=cfg.gating, gating_bias=cfg.gating_bias,
    use_longmem=False)

# ---- params ----
teacher_params = load_weights_only(TEACHER_CKPT, s4_base, ctor, cfg, load_opt_state=False).params
distill_params = jax.tree_util.tree_map(jnp.asarray, pickle.load(open(DISTILL_PARAMS, "rb")))
_p8_4096 = pickle.load(open(P8_4096_PARAMS, "rb"))
_leaves, _td = _p8_4096["params"]
p8_4096_params = jax.tree_util.tree_unflatten(_td, [jnp.asarray(l) for l in _leaves])


def run_variant(name, network, params, use_p8):
    env = DistributedMultiTaskOptimisticLogWrapper(s4_base, jax.random.PRNGKey(0),
            NUM_ENVS, 1, 16, jnp.array([1.0]), table)
    memories = jnp.zeros((NUM_ENVS, cfg.window_mem, cfg.num_layers, cfg.embed_size))
    mem_mask = jnp.zeros((NUM_ENVS, cfg.num_heads, 1, cfg.window_mem + 1), dtype=jnp.bool_)
    mem_idx = jnp.zeros((NUM_ENVS,), dtype=jnp.int32) + (cfg.window_mem + 1)

    def _step(carry, _):
        (log_state, memories, mem_mask, mem_idx, last_obs, done, true_done, longstate,
         finished, ep_len, max_floor, seen, info_acc, ep_return, sewers, flip_floor, rng) = carry
        mem_idx = jnp.where(done, cfg.window_mem, jnp.clip(mem_idx - 1, 0, cfg.window_mem))
        mem_mask = jnp.where(done[:, None, None, None], jnp.zeros_like(mem_mask), mem_mask)
        ohot = jax.nn.one_hot(mem_idx, cfg.window_mem + 1)[:, None, None, :].repeat(cfg.num_heads, 1)
        mem_mask = jnp.logical_or(mem_mask, ohot)
        rng, a_rng, s_rng = jax.random.split(rng, 3)
        if use_p8:
            pi, _, mem_out, ls_new = network.apply(params, memories, last_obs, mem_mask,
                                                   longstate, true_done, method=network.forward_eval)
        else:
            pi, _, mem_out = network.apply(params, memories, last_obs, mem_mask,
                                           method=network.model_forward_eval)
            ls_new = longstate
        action = pi.sample(seed=a_rng)   # STOCHASTIC (frozen protocol)
        memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(mem_out)
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
                ls_new, finished, ep_len, max_floor, seen, info_acc, ep_return, sewers, flip_floor, rng), None

    rng = jax.random.PRNGKey(EVAL_SEED)
    rng, reset_rng = jax.random.split(rng)
    obsv, log_state = env.reset(reset_rng, ctor)
    init = (log_state, memories, mem_mask, mem_idx, obsv,
            jnp.zeros((NUM_ENVS,), dtype=jnp.bool_),     # done
            jnp.zeros((NUM_ENVS,), dtype=jnp.bool_),     # true_done
            init_longstate(NUM_ENVS),                    # longstate
            jnp.zeros((NUM_ENVS,), dtype=jnp.bool_),     # finished
            jnp.zeros((NUM_ENVS,), dtype=jnp.int32),     # ep_len
            jnp.full((NUM_ENVS,), 2, dtype=jnp.int32),   # max_floor (spawn floor 2)
            jnp.zeros((NUM_ENVS,), dtype=jnp.bool_),     # seen DK
            jnp.zeros((NUM_ENVS,), dtype=jnp.float32),   # info_acc
            jnp.zeros((NUM_ENVS,), dtype=jnp.float32),   # ep_return
            jnp.zeros((NUM_ENVS,), dtype=jnp.bool_),     # sewers
            jnp.full((NUM_ENVS,), -1, dtype=jnp.int32),  # flip_floor
            rng)
    t0 = time.time()
    final, _ = jax.lax.scan(_step, init, None, NUM_STEPS)
    (_, _, _, _, _, _, _, _, finished, ep_len, max_floor, seen, info_acc,
     ep_return, sewers, flip_floor, _) = final
    jax.block_until_ready(final)
    roll_time = time.time() - t0

    finished_np = np.asarray(finished); ep_len_np = np.asarray(ep_len)
    max_floor_np = np.asarray(max_floor); seen_np = np.asarray(seen)
    info_acc_np = np.asarray(info_acc); sewers_np = np.asarray(sewers); flip_floor_np = np.asarray(flip_floor)

    success_np = seen_np | (info_acc_np > 0)
    timeout_np = finished_np & (ep_len_np >= NUM_STEPS) & ~success_np
    died_np = finished_np & ~success_np & ~timeout_np
    n_not_finished = int(np.sum(~finished_np))
    if n_not_finished > 0:
        timeout_np = timeout_np | ~finished_np
    n_success = int(success_np.sum()); n_died = int(died_np.sum()); n_timeout = int(timeout_np.sum())
    n_sewers = int(sewers_np.sum()); n_floor3 = int((max_floor_np >= 3).sum())
    sr = n_success / NUM_ENVS; floor3 = n_floor3 / NUM_ENVS

    summary = dict(variant=name, use_p8=use_p8, num_episodes=NUM_ENVS, evaluation_seed=EVAL_SEED,
        policy_mode="stochastic", spawn_floor=2, kobold_floor_diag=KOBOLD_FLOOR_DIAG,
        SR=sr, n_success=n_success, floor3_reach_rate=floor3, n_floor3=n_floor3,
        n_died=n_died, n_timeout=n_timeout, n_not_finished=n_not_finished,
        death_rate=n_died / NUM_ENVS, timeout_rate=n_timeout / NUM_ENVS,
        ENTER_SEWERS_rate=n_sewers / NUM_ENVS,
        mean_episode_length=float(ep_len_np.mean()), max_floor_max=int(max_floor_np.max()),
        success_per_world=[bool(x) for x in success_np],
        floor3_per_world=[bool(x) for x in (max_floor_np >= 3)],
        died_per_world=[bool(x) for x in died_np],
        rollout_time_s=round(roll_time, 1), evaluator_sha256=EVAL_SHA256)
    print(f"[{name}] SR={sr*100:.2f}% ({n_success}/{NUM_ENVS})  floor3={floor3*100:.2f}% ({n_floor3}/{NUM_ENVS})  "
          f"died={n_died} timeout={n_timeout} sewers={n_sewers}  ({roll_time:.1f}s)", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu2_p8_longmemory/eval")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    results = {}
    results["BASELINE"] = run_variant("BASELINE", teacher_net, teacher_params, use_p8=False)
    results["P8_OFF"] = run_variant("P8_OFF", p8_net_off, distill_params, use_p8=True)
    results["P8_DISTILL"] = run_variant("P8_DISTILL", p8_net, distill_params, use_p8=True)
    results["P8_4096"] = run_variant("P8_4096", p8_net, p8_4096_params, use_p8=True)

    base = results["BASELINE"]
    def drop(v, key):
        return (base[key] - results[v][key]) * 100.0   # pp drop vs Baseline (positive = P8 worse)
    gate = {}
    for v in ("P8_DISTILL", "P8_4096"):
        gate[v] = dict(SR_drop_pp=drop(v, "SR"), floor3_drop_pp=drop(v, "floor3_reach_rate"),
                       SR_pass=drop(v, "SR") <= 5.0, floor3_pass=drop(v, "floor3_reach_rate") <= 5.0)
    gate["P8_OFF"] = dict(SR_drop_pp=drop("P8_OFF", "SR"), floor3_drop_pp=drop("P8_OFF", "floor3_reach_rate"),
                          feature_off_ok=(abs(drop("P8_OFF", "SR")) <= 1.0 and abs(drop("P8_OFF", "floor3_reach_rate")) <= 1.0))
    all_pass = all(g["SR_pass"] and g["floor3_pass"] for g in (gate["P8_DISTILL"], gate["P8_4096"])) and gate["P8_OFF"]["feature_off_ok"]

    combined = dict(label="P8_LONGMEM_MIGRATION_GATE", protocol="frozen 64-world Stage4 stochastic seed42",
        gpu_uuid=GPU_UUID, evaluator_sha256=EVAL_SHA256,
        baseline_ckpt=TEACHER_CKPT, p8_distill_params=DISTILL_PARAMS, p8_4096_params=P8_4096_PARAMS,
        results=results, gate=gate, MIGRATION_GATE_PASS=bool(all_pass),
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with open(os.path.join(args.out, "p8_migration_gate.json"), "w") as f:
        json.dump(combined, f, indent=2, default=str)

    print("\n" + "=" * 72, flush=True)
    print(f"BASELINE   SR={base['SR']*100:.2f}%  floor3={base['floor3_reach_rate']*100:.2f}%", flush=True)
    for v in ("P8_OFF", "P8_DISTILL", "P8_4096"):
        print(f"{v:11s} SR_drop={gate[v]['SR_drop_pp']:+.2f}pp  floor3_drop={gate[v]['floor3_drop_pp']:+.2f}pp", flush=True)
    print(f"MIGRATION_GATE_PASS = {all_pass}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
