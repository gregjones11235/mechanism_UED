"""P9-AUTHENTIC-RESET — frozen 256-world Stage4 FINAL evaluator + 98304 positive gate (GPU3).

FINAL EVALUATION USES 100% NATURAL Stage4 STARTS — NO reset injection (design §4). The network
is the ORIGINAL ActorCriticTransformer (P9 changed only the training reset source, not the
network), so this is the canonical natural-start Stage4 evaluator with the same memory machinery
as the Baseline/Control/P8 evaluators. For each saved step in {24576,49152,73728,98304} it
evaluates BOTH the P9 checkpoint and the SAME-STEP Control checkpoint under an IDENTICAL protocol
and seed -> paired by world index.

Positive gate (design §5, applied at 98304; reported at every step):
  natural-eval DK SR  >= Control + 8 pp
  floor3 reach rate   >= Control
  >= 1 death/search metric improved (death_rate down OR timeout_rate down OR ENTER_SEWERS up)
  no numeric / entropy collapse (all metrics finite)
Allowed labels only: EXPLORATORY_POSITIVE_SIGNAL / NO_POSITIVE_SIGNAL / ENGINEERING_FAIL.
Read-only w.r.t. checkpoints; writes only under the P9 eval dir. GPU3 only.
"""
import hashlib, json, os, sys, time, pickle, argparse

GPU_UUID = "GPU-f56a59b4-99f3-f2e5-11c6-d01685de8abd"   # GPU3 ONLY
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

import jax, jax.numpy as jnp
import numpy as np

V7_SRC = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src"
V7 = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB"
for p in (V7_SRC, V7):
    if p not in sys.path:
        sys.path.insert(0, p)

from craftax.craftax.constants import Achievement
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from dicode.network import ActorCriticTransformer
from dicode.task_utils import get_achievement_multi_hot
from dicode.wrappers_cl import DistributedMultiTaskOptimisticLogWrapper
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv

_cfg = dict(activation="relu", embed_size=256, hidden_layers=256, num_heads=8,
            qkv_features=256, num_layers=2, gating=True, gating_bias=2.0,
            window_mem=128, window_grad=64)
cfg = type("C", (), _cfg)()

P9_CKPT_ROOT = "/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu3_p9_authentic_reset/train_98304/ckpt"
CTL_CKPT_ROOT = "/home/oseasy/experiments/exploratory_delayed_onset_20260724/control_RUN2/ckpt"
STEPS = [24576, 49152, 73728, 98304]

DK = int(Achievement.DEFEAT_KOBOLD.value)
SEWERS = int(Achievement.ENTER_SEWERS.value)
NUM_ENVS = 256
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
print("P9-AUTHENTIC-RESET FINAL evaluator (frozen 256-world, 100% natural Stage4 starts)", flush=True)
print(f"  GPU_UUID={GPU_UUID} devices={[str(d) for d in jax.devices()]}", flush=True)
print(f"  obs_dim={OBS_DIM} action_dim={ACTION_DIM} emb={EMB} DK_idx={DK}", flush=True)
print(f"  NUM_ENVS={NUM_ENVS} NUM_STEPS={NUM_STEPS} seed={EVAL_SEED} policy=stochastic", flush=True)
print(f"  evaluator_sha256={EVAL_SHA256}", flush=True)
print("=" * 72, flush=True)

net = ActorCriticTransformer(action_dim=ACTION_DIM, activation=cfg.activation,
    encoder_size=cfg.embed_size, hidden_layers=cfg.hidden_layers, num_heads=cfg.num_heads,
    qkv_features=cfg.qkv_features, num_layers=cfg.num_layers, gating=cfg.gating, gating_bias=cfg.gating_bias)


def load_params(path):
    d = pickle.load(open(path, "rb"))
    leaves, td = d["params"]
    return jax.tree_util.tree_unflatten(td, [jnp.asarray(l) for l in leaves]), d


def run_variant(name, params):
    """100% natural Stage4 starts, original network, stochastic policy. NO reset injection."""
    env = DistributedMultiTaskOptimisticLogWrapper(s4_base, jax.random.PRNGKey(0),
            NUM_ENVS, 1, 16, jnp.array([1.0]), table)
    memories = jnp.zeros((NUM_ENVS, cfg.window_mem, cfg.num_layers, cfg.embed_size))
    mem_mask = jnp.zeros((NUM_ENVS, cfg.num_heads, 1, cfg.window_mem + 1), dtype=jnp.bool_)
    mem_idx = jnp.zeros((NUM_ENVS,), dtype=jnp.int32) + (cfg.window_mem + 1)

    def _step(carry, _):
        (log_state, memories, mem_mask, mem_idx, last_obs, done,
         finished, ep_len, max_floor, seen, info_acc, ep_return, sewers, rng) = carry
        mem_idx = jnp.where(done, cfg.window_mem, jnp.clip(mem_idx - 1, 0, cfg.window_mem))
        mem_mask = jnp.where(done[:, None, None, None], jnp.zeros_like(mem_mask), mem_mask)
        ohot = jax.nn.one_hot(mem_idx, cfg.window_mem + 1)[:, None, None, :].repeat(cfg.num_heads, 1)
        mem_mask = jnp.logical_or(mem_mask, ohot)
        rng, a_rng, s_rng = jax.random.split(rng, 3)
        pi, _, mem_out = net.apply(params, memories, last_obs, mem_mask, method=net.model_forward_eval)
        action = pi.sample(seed=a_rng)
        memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(mem_out)
        pre = log_state.env_state
        pre_pl = pre.player_level
        pre_dk = pre.achievements[:, DK].astype(bool)
        pre_sw = pre.achievements[:, SEWERS].astype(bool)
        next_obs, next_log_state, reward, next_done, info = env.step(s_rng, log_state, action, ctor)
        active = ~finished
        ep_len = ep_len + active.astype(jnp.int32)
        ep_return = ep_return + jnp.asarray(reward, jnp.float32).reshape(-1) * active.astype(jnp.float32)
        max_floor = jnp.where(active, jnp.maximum(max_floor, pre_pl), max_floor)
        seen = seen | (pre_dk & active)
        sewers = sewers | (pre_sw & active)
        keys = [k for k in info if "Achievements" in k and "kobold" in k.lower()]
        if keys:
            info_acc = info_acc + jnp.asarray(info[keys[0]], jnp.float32).reshape(-1) * active.astype(jnp.float32)
        finished = finished | next_done
        return (next_log_state, memories, mem_mask, mem_idx, next_obs, next_done,
                finished, ep_len, max_floor, seen, info_acc, ep_return, sewers, rng), None

    rng = jax.random.PRNGKey(EVAL_SEED)
    rng, reset_rng = jax.random.split(rng)
    obsv, log_state = env.reset(reset_rng, ctor)
    init = (log_state, memories, mem_mask, mem_idx, obsv,
            jnp.zeros((NUM_ENVS,), dtype=jnp.bool_), jnp.zeros((NUM_ENVS,), dtype=jnp.bool_),
            jnp.zeros((NUM_ENVS,), dtype=jnp.int32), jnp.full((NUM_ENVS,), 2, dtype=jnp.int32),
            jnp.zeros((NUM_ENVS,), dtype=jnp.bool_), jnp.zeros((NUM_ENVS,), dtype=jnp.float32),
            jnp.zeros((NUM_ENVS,), dtype=jnp.float32), jnp.zeros((NUM_ENVS,), dtype=jnp.bool_), rng)
    t0 = time.time()
    final, _ = jax.lax.scan(_step, init, None, NUM_STEPS)
    (_, _, _, _, _, _, finished, ep_len, max_floor, seen, info_acc, ep_return, sewers, _) = final
    jax.block_until_ready(final)
    roll_time = time.time() - t0

    finished_np = np.asarray(finished); ep_len_np = np.asarray(ep_len)
    max_floor_np = np.asarray(max_floor); seen_np = np.asarray(seen)
    info_acc_np = np.asarray(info_acc); sewers_np = np.asarray(sewers)

    success_np = seen_np | (info_acc_np > 0)
    timeout_np = finished_np & (ep_len_np >= NUM_STEPS) & ~success_np
    died_np = finished_np & ~success_np & ~timeout_np
    n_not_finished = int(np.sum(~finished_np))
    if n_not_finished > 0:
        timeout_np = timeout_np | ~finished_np
    n_success = int(success_np.sum()); n_died = int(died_np.sum()); n_timeout = int(timeout_np.sum())
    n_sewers = int(sewers_np.sum()); n_floor3 = int((max_floor_np >= 3).sum())
    sr = n_success / NUM_ENVS; floor3 = n_floor3 / NUM_ENVS

    summary = dict(variant=name, num_episodes=NUM_ENVS, evaluation_seed=EVAL_SEED,
        policy_mode="stochastic", spawn_floor=2, reset_injection="OFF (100% natural Stage4 starts)",
        SR=sr, n_success=n_success, floor3_reach_rate=floor3, n_floor3=n_floor3,
        n_died=n_died, n_timeout=n_timeout, n_not_finished=n_not_finished,
        death_rate=n_died / NUM_ENVS, timeout_rate=n_timeout / NUM_ENVS,
        ENTER_SEWERS_rate=n_sewers / NUM_ENVS, mean_episode_length=float(ep_len_np.mean()),
        max_floor_max=int(max_floor_np.max()),
        success_per_world=[bool(x) for x in success_np],
        floor3_per_world=[bool(x) for x in (max_floor_np >= 3)],
        died_per_world=[bool(x) for x in died_np],
        rollout_time_s=round(roll_time, 1), evaluator_sha256=EVAL_SHA256)
    print(f"[{name}] SR={sr*100:.2f}% ({n_success}/{NUM_ENVS})  floor3={floor3*100:.2f}% ({n_floor3}/{NUM_ENVS})  "
          f"died={n_died} timeout={n_timeout} sewers={n_sewers}  ({roll_time:.1f}s)", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu3_p9_authentic_reset/eval")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    results = {}
    for st in STEPS:
        p9_params, _ = load_params(os.path.join(P9_CKPT_ROOT, str(st), "full_state.pkl"))
        ctl_params, _ = load_params(os.path.join(CTL_CKPT_ROOT, str(st), "full_state.pkl"))
        results[f"P9_{st}"] = run_variant(f"P9_{st}", p9_params)
        results[f"CTL_{st}"] = run_variant(f"CTL_{st}", ctl_params)

    def delta(st, key):
        return (results[f"P9_{st}"][key] - results[f"CTL_{st}"][key]) * 100.0
    per_step = {}
    for st in STEPS:
        per_step[st] = dict(SR_pp=results[f"P9_{st}"]["SR"]*100, CTL_SR_pp=results[f"CTL_{st}"]["SR"]*100,
            SR_delta_pp=delta(st, "SR"), floor3_delta_pp=delta(st, "floor3_reach_rate"),
            death_delta_pp=delta(st, "death_rate"), timeout_delta_pp=delta(st, "timeout_rate"),
            sewers_delta_pp=delta(st, "ENTER_SEWERS_rate"),
            eplen_delta=results[f"P9_{st}"]["mean_episode_length"]-results[f"CTL_{st}"]["mean_episode_length"])

    g = per_step[98304]
    sr_pass = g["SR_delta_pp"] >= 8.0
    floor3_pass = g["floor3_delta_pp"] >= -1e-9
    death_search_pass = (g["death_delta_pp"] < -1e-9) or (g["timeout_delta_pp"] < -1e-9) or (g["sewers_delta_pp"] > 1e-9)
    finite_pass = all(np.isfinite(results[f"P9_98304"][k]) for k in ("SR","floor3_reach_rate","death_rate","timeout_rate","mean_episode_length"))
    gate = dict(at_step=98304, SR_delta_pp=g["SR_delta_pp"], SR_pass=sr_pass,
                floor3_delta_pp=g["floor3_delta_pp"], floor3_pass=floor3_pass,
                death_delta_pp=g["death_delta_pp"], timeout_delta_pp=g["timeout_delta_pp"],
                sewers_delta_pp=g["sewers_delta_pp"], death_search_pass=death_search_pass,
                finite_pass=bool(finite_pass))
    positive = sr_pass and floor3_pass and death_search_pass and finite_pass
    label = "EXPLORATORY_POSITIVE_SIGNAL" if positive else "NO_POSITIVE_SIGNAL"

    combined = dict(label="P9_AUTHENTIC_RESET_FINAL_GATE", gate_label=label,
        protocol="frozen 256-world Stage4 stochastic seed42, 100% natural starts, P9 vs same-step Control paired",
        gpu_uuid=GPU_UUID, evaluator_sha256=EVAL_SHA256, steps=STEPS,
        p9_ckpt_root=P9_CKPT_ROOT, ctl_ckpt_root=CTL_CKPT_ROOT,
        results=results, per_step=per_step, gate=gate,
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with open(os.path.join(args.out, "p9_final_gate.json"), "w") as f:
        json.dump(combined, f, indent=2, default=str)

    print("\n" + "=" * 72, flush=True)
    for st in STEPS:
        s = per_step[st]
        print(f"step {st:6d}: P9_SR={s['SR_pp']:.2f}% CTL_SR={s['CTL_SR_pp']:.2f}% "
              f"dSR={s['SR_delta_pp']:+.2f}pp dFloor3={s['floor3_delta_pp']:+.2f}pp "
              f"dDeath={s['death_delta_pp']:+.2f}pp dSewers={s['sewers_delta_pp']:+.2f}pp", flush=True)
    print(f"GATE@98304: SR_pass={sr_pass} floor3_pass={floor3_pass} "
          f"death_search_pass={death_search_pass} finite_pass={finite_pass}", flush=True)
    print(f"P9_FINAL_LABEL = {label}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
