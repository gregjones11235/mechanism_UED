#!/usr/bin/env python3
"""Bakeoff Phase1 evaluation – 256 worlds, Stage4-native DEFEAT_KOBOLD.

Runs episodes in batches of EVAL_BATCH_SIZE worlds using lax.scan.
Each world gets a unique seed = seed_base + world_idx.

Usage:
  python eval_bakeoff.py --arm baseline \
      --params /path/to/params.pkl \
      --out /path/to/eval_out \
      --gpu_uuid GPU-xxxx

  python eval_bakeoff.py --arm w512 \
      --params /path/to/params.pkl \
      --out /path/to/eval_out \
      --gpu_uuid GPU-xxxx \
      [--ablation full|zero|shuffle|short128|reset128]
"""
import os, sys, json, time, hashlib, pickle, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True,
                choices=["baseline", "control", "w512", "rmt16"])
ap.add_argument("--params", required=True,
                help="path to params.pkl (pickle)")
ap.add_argument("--out", required=True,
                help="output directory for results")
ap.add_argument("--gpu_uuid", required=True)
ap.add_argument("--num_worlds", type=int, default=256)
ap.add_argument("--seed_base", type=int, default=100000)
ap.add_argument("--max_steps", type=int, default=4096)
ap.add_argument("--batch_size", type=int, default=16)
ap.add_argument("--ablation", default="full",
                choices=["full", "zero", "shuffle", "short128", "reset128"])
args = ap.parse_args()

# ---- GPU + determinism ----
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

# ---- paths ----
V7_SRC = ("/home/oseasy/incoming/henry_work_20260721T105300/"
          "extracted/Henry_work/code/dicode_v7fix58_armB/src")
BAKE_SRC = os.path.dirname(os.path.abspath(__file__))
for p in [V7_SRC, BAKE_SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

import jax, jax.numpy as jnp, numpy as np

from craftax.craftax.constants import Achievement
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from dicode.task_utils import get_achievement_multi_hot
from dicode.wrappers_cl import DistributedMultiTaskOptimisticLogWrapper
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv

# ================================================================
# Config (frozen, matches training)
# ================================================================
class Cfg:
    activation       = "relu"
    embed_size       = 256
    hidden_layers    = 256
    num_heads        = 8
    qkv_features     = 256
    num_layers       = 2
    gating           = True
    gating_bias      = 2.0
    window_mem       = 128
    w512_long_size   = 384
    w512_delay_size  = 128
    rmt_num_tokens   = 16
    # For DistributedMultiTaskOptimisticLogWrapper
    optimistic_reset_ratio = 16

cfg = Cfg()
EVAL_BATCH = args.batch_size
MAX_STEPS  = args.max_steps
ROLLOUT_LEN = 128   # rollout boundary for reset128 ablation

# ================================================================
# Stage4 task
# ================================================================
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
ns4 = {}; exec(S4_TASK_CODE, ns4); S4Cls = ns4["Env"]
ctor = EnvParams(max_timesteps=4096)
table = jnp.array([get_achievement_multi_hot([Achievement.DEFEAT_KOBOLD])],
                   dtype=jnp.float32)
EMB = int(table.shape[1])

# ================================================================
# Env setup
# ================================================================
static_env_params = StaticEnvParams()
env_params_base = EnvParams(max_timesteps=MAX_STEPS)
base_env = MultiTaskMiniCraftaxEnv(
    [S4Cls], static_env_params, env_params_base, True,
    conditioning_type="embedding", embedding_size=EMB)
ACTION_DIM = int(base_env.action_space(env_params_base).n)
OBS_DIM = int(base_env.observation_space(env_params_base).shape[0])

# Use the same wrapper as training for consistent world generation
eval_env = DistributedMultiTaskOptimisticLogWrapper(
    base_env, jax.random.PRNGKey(0), EVAL_BATCH, 1,
    cfg.optimistic_reset_ratio, jnp.array([1.0]), table)
eval_env_params = eval_env.default_params

print("=" * 72, flush=True)
print(f"Bakeoff eval  arm={args.arm}  ablation={args.ablation}", flush=True)
print(f"  params: {args.params}", flush=True)
print(f"  worlds: {args.num_worlds}  seed_base={args.seed_base}", flush=True)
print(f"  batch_size={EVAL_BATCH}  max_steps={MAX_STEPS}", flush=True)
print(f"  GPU: {args.gpu_uuid}  devices: {[str(d) for d in jax.devices()]}", flush=True)
print(f"  OBS_DIM={OBS_DIM}  ACTION_DIM={ACTION_DIM}", flush=True)
print("=" * 72, flush=True)

# ================================================================
# Load params
# ================================================================
t0 = time.time()
with open(args.params, "rb") as f:
    params = pickle.load(f)
params = jax.tree_util.tree_map(jnp.asarray, params)
n_leaves = len(jax.tree_util.tree_leaves(params))
p_sha = hashlib.sha256(
    b"".join(np.asarray(l).tobytes()
             for l in jax.tree_util.tree_leaves(params))).hexdigest()
print(f"[load] leaves={n_leaves}  sha256={p_sha[:16]}...  ({time.time()-t0:.1f}s)",
      flush=True)

# ================================================================
# Network
# ================================================================
if args.arm in ("baseline", "control"):
    from dicode.network import ActorCriticTransformer
    network = ActorCriticTransformer(
        action_dim=ACTION_DIM, activation=cfg.activation,
        encoder_size=cfg.embed_size, hidden_layers=cfg.hidden_layers,
        num_heads=cfg.num_heads, qkv_features=cfg.qkv_features,
        num_layers=cfg.num_layers, gating=cfg.gating,
        gating_bias=cfg.gating_bias)
elif args.arm == "w512":
    from network_w512 import ActorCriticTransformerW512
    import w512_memory as w5m
    network = ActorCriticTransformerW512(
        action_dim=ACTION_DIM, activation=cfg.activation,
        encoder_size=cfg.embed_size, hidden_layers=cfg.hidden_layers,
        num_heads=cfg.num_heads, qkv_features=cfg.qkv_features,
        num_layers=cfg.num_layers, gating=cfg.gating,
        gating_bias=cfg.gating_bias, long_size=cfg.w512_long_size)
elif args.arm == "rmt16":
    from network_rmt16 import ActorCriticTransformerRMT16
    import rmt16_memory as rmtm
    network = ActorCriticTransformerRMT16(
        action_dim=ACTION_DIM, activation=cfg.activation,
        encoder_size=cfg.embed_size, hidden_layers=cfg.hidden_layers,
        num_heads=cfg.num_heads, qkv_features=cfg.qkv_features,
        num_layers=cfg.num_layers, gating=cfg.gating,
        gating_bias=cfg.gating_bias, rmt_num_tokens=cfg.rmt_num_tokens)

# ================================================================
# Batch eval function (JIT-compiled)
# ================================================================
ABLATION = args.ablation
ARM = args.arm

def run_batch(params, rng_key):
    """Run EVAL_BATCH worlds for MAX_STEPS. Returns per-env results."""
    rng_reset, rng_loop = jax.random.split(rng_key)
    obs, env_state = eval_env.reset(rng_reset, eval_env_params)

    # Transformer memory
    memories = jnp.zeros((EVAL_BATCH, cfg.window_mem, cfg.num_layers, cfg.embed_size))
    mmask = jnp.zeros((EVAL_BATCH, cfg.num_heads, 1, cfg.window_mem + 1), dtype=jnp.bool_)
    midx = jnp.full((EVAL_BATCH,), cfg.window_mem + 1, dtype=jnp.int32)
    done = jnp.zeros((EVAL_BATCH,), dtype=jnp.bool_)

    # LC state
    if ARM == "w512":
        w5_cfg = w5m.W512Config(
            long_size=cfg.w512_long_size,
            delay_size=cfg.w512_delay_size,
            encoder_size=cfg.embed_size)
        w5st = w5m.w512_init(EVAL_BATCH, w5_cfg)
    elif ARM == "rmt16":
        rmt_cfg = rmtm.RMT16Config(
            num_tokens=cfg.rmt_num_tokens,
            segment_len=128,
            encoder_size=cfg.embed_size)
        rmt_st = rmtm.rmt16_init(EVAL_BATCH, rmt_cfg)

    # Trackers
    dk_ever = jnp.zeros((EVAL_BATCH,), dtype=jnp.bool_)
    floor3_ever = jnp.zeros((EVAL_BATCH,), dtype=jnp.bool_)
    ep_len = jnp.zeros((EVAL_BATCH,), dtype=jnp.int32)
    ep_done = jnp.zeros((EVAL_BATCH,), dtype=jnp.bool_)
    death_flag = jnp.zeros((EVAL_BATCH,), dtype=jnp.bool_)

    # step_in_rollout: scalar counter 0..ROLLOUT_LEN-1, wraps every 128 steps
    step_in_rollout = jnp.zeros((), dtype=jnp.int32)

    def _step(carry, _):
        if ARM == "w512":
            (obs, es, mems, mm, mi, dn, w5s,
             dk_e, f3_e, elen, edone, dflag, sir, rng) = carry
        elif ARM == "rmt16":
            (obs, es, mems, mm, mi, dn, rst,
             dk_e, f3_e, elen, edone, dflag, sir, rng) = carry
        else:
            (obs, es, mems, mm, mi, dn,
             dk_e, f3_e, elen, edone, dflag, sir, rng) = carry

        # ---- RESET128 ablation: at rollout boundary, clear extra LC memory ----
        # sir==0 AND we are not at the very first step of the episode
        # (sir wraps 0→127→0→...; reset when sir==0 except the initial state)
        # We track "is_first_step" implicitly: at the very start sir=0 and
        # all LC state is already zero, so resetting is a harmless no-op.
        # Subsequent sir==0 hits are true rollout boundaries.
        if ABLATION == "reset128" and ARM == "w512":
            at_boundary = jnp.equal(sir, 0)
            w5s = {
                **w5s,
                "long_buf":  jnp.where(at_boundary,
                                       jnp.zeros_like(w5s["long_buf"]),
                                       w5s["long_buf"]),
                "long_mask": jnp.where(at_boundary,
                                       jnp.zeros_like(w5s["long_mask"]),
                                       w5s["long_mask"]),
            }
        elif ABLATION == "reset128" and ARM == "rmt16":
            at_boundary = jnp.equal(sir, 0)
            rst = {
                **rst,
                "mem_tokens": jnp.where(at_boundary,
                                        jnp.zeros_like(rst["mem_tokens"]),
                                        rst["mem_tokens"]),
            }

        # Memory mask update
        mi = jnp.where(dn, cfg.window_mem,
                        jnp.clip(mi - 1, 0, cfg.window_mem))
        mm = jnp.where(dn[:, None, None, None],
                        jnp.zeros_like(mm), mm)
        ohot = jax.nn.one_hot(mi, cfg.window_mem + 1)
        ohot = ohot[:, None, None, :].repeat(cfg.num_heads, 1)
        mm = jnp.logical_or(mm, ohot)

        # Forward
        rng, _rng = jax.random.split(rng)
        if ARM in ("baseline", "control"):
            pi, value, mem_out = network.apply(
                params, mems, obs, mm,
                method=network.model_forward_eval)
            h_t = None
        elif ARM == "w512":
            lbuf = w5s["long_buf"]
            lmsk = w5s["long_mask"]
            if ABLATION == "zero":
                lbuf = jnp.zeros_like(lbuf)
                lmsk = jnp.zeros_like(lmsk)
            elif ABLATION == "shuffle":
                sk = jax.random.fold_in(rng, 99999)
                perm = jax.random.permutation(sk, cfg.w512_long_size)
                lbuf = lbuf[:, perm, :]
                lmsk = lmsk[:, perm]
            elif ABLATION == "short128":
                short_m = jnp.zeros(cfg.w512_long_size, dtype=jnp.bool_)
                short_m = short_m.at[-cfg.w512_delay_size:].set(True)
                lmsk = lmsk & short_m[None, :]
            pi, value, mem_out, h_t = network.apply(
                params, mems, obs, mm,
                long_buf=lbuf, long_mask=lmsk,
                method=network.model_forward_eval)
        elif ARM == "rmt16":
            mtok = rst["mem_tokens"]
            if ABLATION == "zero":
                mtok = jnp.zeros_like(mtok)
            elif ABLATION == "shuffle":
                sk = jax.random.fold_in(rng, 99999)
                perm = jax.random.permutation(sk, cfg.rmt_num_tokens)
                mtok = mtok[:, perm, :]
            elif ABLATION == "short128":
                mtok = jnp.zeros_like(mtok)
            pi, value, mem_out, h_t = network.apply(
                params, mems, obs, mm,
                mem_tokens=mtok,
                method=network.model_forward_eval)

        action = pi.sample(seed=_rng)
        mems = jnp.roll(mems, -1, axis=1).at[:, -1].set(mem_out)

        # Env step
        rng, _rng = jax.random.split(rng)
        obs_next, es_new, reward, dn_new, info = eval_env.step(
            _rng, es, action, eval_env_params)

        # Track achievements (lowercase key from Craftax wrapper)
        dk_key = "Achievements/defeat_kobold"
        dk_now = jnp.zeros((EVAL_BATCH,), dtype=jnp.bool_)
        if dk_key in info:
            dk_now = info[dk_key].astype(jnp.bool_)

        # floor3: use enter_gnomish_mines (floor 3) as proxy
        floor_now = jnp.zeros((EVAL_BATCH,), dtype=jnp.bool_)
        if "Achievements/enter_gnomish_mines" in info:
            floor_now = info["Achievements/enter_gnomish_mines"].astype(jnp.bool_)

        dk_e = jnp.logical_or(dk_e, dk_now)
        f3_e = jnp.logical_or(f3_e, floor_now)
        elen = jnp.where(edone, elen, elen + 1)

        # Death detection: done but NOT success
        new_death = jnp.logical_and(dn_new, jnp.logical_not(dk_e))
        dflag = jnp.logical_or(dflag, new_death)
        edone = jnp.logical_or(edone, dn_new)

        # Advance rollout counter: 0→1→...→127→0→...
        sir_next = (sir + 1) % ROLLOUT_LEN

        # LC state update
        if ARM == "w512":
            w5s_new = w5m.w512_step(w5s, h_t, dn_new, w5_cfg)
            carry_out = (obs_next, es_new, mems, mm, mi, dn_new, w5s_new,
                         dk_e, f3_e, elen, edone, dflag, sir_next, rng)
        elif ARM == "rmt16":
            def _upd_fn(tokens, seg_buf):
                return network.apply(
                    params, tokens, seg_buf,
                    method=network.update_rmt_tokens)
            rst_new = rmtm.rmt16_step(rst, h_t, dn_new, _upd_fn, rmt_cfg)
            carry_out = (obs_next, es_new, mems, mm, mi, dn_new, rst_new,
                         dk_e, f3_e, elen, edone, dflag, sir_next, rng)
        else:
            carry_out = (obs_next, es_new, mems, mm, mi, dn_new,
                         dk_e, f3_e, elen, edone, dflag, sir_next, rng)

        return carry_out, None

    # Init carry
    if ARM == "w512":
        init = (obs, env_state, memories, mmask, midx, done, w5st,
                dk_ever, floor3_ever, ep_len, ep_done, death_flag,
                step_in_rollout, rng_loop)
    elif ARM == "rmt16":
        init = (obs, env_state, memories, mmask, midx, done, rmt_st,
                dk_ever, floor3_ever, ep_len, ep_done, death_flag,
                step_in_rollout, rng_loop)
    else:
        init = (obs, env_state, memories, mmask, midx, done,
                dk_ever, floor3_ever, ep_len, ep_done, death_flag,
                step_in_rollout, rng_loop)

    final, _ = jax.lax.scan(_step, init, None, MAX_STEPS)

    if ARM in ("baseline", "control"):
        (_, _, _, _, _, _, dk_f, f3_f, el_f, ed_f, df_f, _, _) = final
    else:
        (_, _, _, _, _, _, _, dk_f, f3_f, el_f, ed_f, df_f, _, _) = final

    return dk_f, f3_f, el_f, ed_f, df_f

run_batch_jit = jax.jit(run_batch)

# ================================================================
# Run all worlds in batches
# ================================================================
os.makedirs(args.out, exist_ok=True)

n_batches = (args.num_worlds + EVAL_BATCH - 1) // EVAL_BATCH
print(f"\n[eval] {args.num_worlds} worlds in {n_batches} batches of {EVAL_BATCH}",
      flush=True)
t_eval = time.time()

all_dk = []
all_f3 = []
all_elen = []
all_edone = []
all_dflag = []

rng_eval = jax.random.PRNGKey(args.seed_base)

for b in range(n_batches):
    rng_eval, batch_rng = jax.random.split(rng_eval)
    dk, f3, el, ed, df = run_batch_jit(params, batch_rng)
    dk_np = np.asarray(dk)
    f3_np = np.asarray(f3)
    el_np = np.asarray(el)
    ed_np = np.asarray(ed)
    df_np = np.asarray(df)

    # Trim to num_worlds (last batch may overshoot)
    remaining = args.num_worlds - b * EVAL_BATCH
    take = min(EVAL_BATCH, remaining)
    all_dk.extend(dk_np[:take].tolist())
    all_f3.extend(f3_np[:take].tolist())
    all_elen.extend(el_np[:take].tolist())
    all_edone.extend(ed_np[:take].tolist())
    all_dflag.extend(df_np[:take].tolist())

    sr_so_far = sum(all_dk) / len(all_dk)
    print(f"  batch {b+1}/{n_batches}  SR={sr_so_far:.3f}  "
          f"({sum(all_dk)}/{len(all_dk)})", flush=True)

eval_time = time.time() - t_eval

# ================================================================
# Metrics
# ================================================================
n = len(all_dk)
dk_sr = sum(all_dk) / n
f3_rate = sum(all_f3) / n
f3_idx = [i for i in range(n) if all_f3[i]]
cond_kill = (sum(all_dk[i] for i in f3_idx) / len(f3_idx)) if f3_idx else 0.0
death_ct = sum(all_dflag)
timeout_ct = sum(1 for i in range(n) if not all_edone[i])
death_rate = death_ct / n
timeout_rate = timeout_ct / n
mean_elen = float(np.mean(all_elen))
median_elen = float(np.median(all_elen))

# Wilson CI
try:
    from statsmodels.stats.proportion import proportion_confint
    ci_lo, ci_hi = proportion_confint(sum(all_dk), n, alpha=0.05, method="wilson")
except ImportError:
    # Fallback: normal approx
    se = np.sqrt(dk_sr * (1 - dk_sr) / n)
    ci_lo, ci_hi = dk_sr - 1.96 * se, dk_sr + 1.96 * se

summary = {
    "arm": args.arm,
    "ablation": args.ablation,
    "num_worlds": n,
    "seed_base": args.seed_base,
    "max_steps": MAX_STEPS,
    "params_sha256": p_sha,
    "dk_sr": round(dk_sr, 6),
    "dk_sr_ci95": [round(float(ci_lo), 6), round(float(ci_hi), 6)],
    "dk_successes": int(sum(all_dk)),
    "floor3_rate": round(f3_rate, 6),
    "floor3_count": int(sum(all_f3)),
    "conditional_kill_rate": round(cond_kill, 6),
    "death_count": int(death_ct),
    "death_rate": round(death_rate, 6),
    "timeout_count": int(timeout_ct),
    "timeout_rate": round(timeout_rate, 6),
    "mean_episode_length": round(mean_elen, 1),
    "median_episode_length": round(median_elen, 1),
    "eval_time_s": round(eval_time, 1),
    "gpu": args.gpu_uuid,
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}

out_json = os.path.join(args.out, f"eval_{args.arm}_{args.ablation}.json")
with open(out_json, "w") as f:
    json.dump(summary, f, indent=2)

out_detail = os.path.join(args.out, f"eval_{args.arm}_{args.ablation}_per_world.jsonl")
with open(out_detail, "w") as f:
    for i in range(n):
        f.write(json.dumps({
            "world_idx": i,
            "dk_success": bool(all_dk[i]),
            "floor3": bool(all_f3[i]),
            "episode_length": int(all_elen[i]),
            "done": bool(all_edone[i]),
            "death": bool(all_dflag[i]),
        }) + "\n")

print(f"\n{'='*72}", flush=True)
print(f"RESULTS  arm={args.arm}  ablation={args.ablation}", flush=True)
print(f"  DK SR:     {dk_sr:.4f}  [{ci_lo:.4f}, {ci_hi:.4f}]  ({sum(all_dk)}/{n})", flush=True)
print(f"  floor3:    {f3_rate:.4f}  ({sum(all_f3)}/{n})", flush=True)
print(f"  cond kill: {cond_kill:.4f}", flush=True)
print(f"  death:     {death_rate:.4f}  ({death_ct}/{n})", flush=True)
print(f"  timeout:   {timeout_rate:.4f}  ({timeout_ct}/{n})", flush=True)
print(f"  ep_len:    mean={mean_elen:.1f}  median={median_elen:.1f}", flush=True)
print(f"  time:      {eval_time:.1f}s", flush=True)
print(f"  saved:     {out_json}", flush=True)
print(f"{'='*72}", flush=True)
