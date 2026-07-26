#!/usr/bin/env python3
"""Bakeoff Phase1 launcher – LC-W512 (GPU0) / LC-RMT16 (GPU1).

Canonical Control hyperparams (frozen):
  update_epochs=1, num_minibatches=2, lr=2e-5 constant, Adam eps=1e-5,
  gamma=0.999, gae_lambda=0.8, clip=0.2, ent_coef=0.002, vf_coef=0.5,
  max_grad_norm=1.0, num_envs=16, num_steps=128,
  optimistic_reset_ratio=16, mode=score, MASTER_SEED=42,
  XLA_FLAGS=--xla_gpu_deterministic_ops=true.

Checkpoint schedule: 0 / 4096 / 24576
Total: 24576 updates.
"""
import os, sys, json, time, hashlib, pickle, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=["w512", "rmt16"])
ap.add_argument("--ckpt17500", required=True,
                help="orbax dir of base ckpt17500")
ap.add_argument("--out", required=True,
                help="output root")
ap.add_argument("--gpu_uuid", required=True)
ap.add_argument("--segment", type=int, default=4096,
                help="updates per compiled segment (default 4096)")
ap.add_argument("--resume", default=None,
                help="path to runner_state.pkl for exact resume")
args = ap.parse_args()

# ---- GPU + determinism (BEFORE any JAX import) ----
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

# ---- paths ----
V7_SRC = ("/home/oseasy/incoming/henry_work_20260721T105300/"
          "extracted/Henry_work/code/dicode_v7fix58_armB/src")
BAKE_SRC = os.path.dirname(os.path.abspath(__file__))
for p in [V7_SRC, BAKE_SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

import jax, jax.numpy as jnp, numpy as np, optax
import orbax.checkpoint as ocp
from flax.training.train_state import TrainState

from craftax.craftax.constants import Achievement
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from dicode.task_utils import get_achievement_multi_hot
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv

# ================================================================
# Config – canonical Control freeze record
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
    window_grad      = 64
    lr               = 2e-5
    max_grad_norm    = 1.0
    gamma            = 0.999
    gae_lambda       = 0.8
    clip_eps         = 0.2
    vf_coef          = 0.5
    ent_coef         = 0.002
    update_epochs    = 1
    num_minibatches  = 2
    num_envs         = 16
    num_steps        = 128
    optimistic_reset_ratio = 16
    condition_on_task = True
    value_target_clip_min = -50.0
    value_target_clip_max = 300.0
    # W512
    w512_long_size   = 384
    w512_delay_size  = 128
    # RMT16
    rmt_num_tokens   = 16
    # training budget
    total_timesteps  = 24576 * 16 * 128
    max_updates_per_session = 24576

cfg = Cfg()
MASTER_SEED = 42
TOTAL_UPDATES = 24576
CKPT_STEPS = [0, 4096, 24576]
SEGMENT = args.segment

# ================================================================
# Arm-specific imports
# ================================================================
if args.arm == "w512":
    from network_w512 import ActorCriticTransformerW512 as NetCls
    import w512_memory as mem_mod
    from ppo_tr_w512 import make_train
    ARM_LABEL = "LC-W512-PPO"
else:
    from network_rmt16 import ActorCriticTransformerRMT16 as NetCls
    import rmt16_memory as mem_mod
    from ppo_tr_rmt16 import make_train
    ARM_LABEL = "LC-RMT16-PPO"

# ================================================================
# Stage4 task (IDENTICAL to eval_paired_256.py / canonical Control)
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
# Load ckpt17500 base params (orbax)
# ================================================================
print("=" * 72, flush=True)
print(f"{ARM_LABEL} launcher  Bakeoff Phase1", flush=True)
print(f"  ckpt17500 : {args.ckpt17500}", flush=True)
print(f"  out       : {args.out}", flush=True)
print(f"  segment   : {SEGMENT} updates", flush=True)
print(f"  GPU       : {args.gpu_uuid}  devices: {[str(d) for d in jax.devices()]}", flush=True)
print("=" * 72, flush=True)

t0 = time.time()
ckpt_mgr = ocp.CheckpointManager(os.path.dirname(args.ckpt17500))
raw = ckpt_mgr.restore(int(os.path.basename(args.ckpt17500)))
base_params = raw["params"]  # {"params": {...}}
base_inner = base_params["params"]
base_leaves = jax.tree_util.tree_leaves(base_inner)
base_sha = hashlib.sha256(
    b"".join(np.asarray(l).tobytes() for l in base_leaves)).hexdigest()
print(f"[load] ckpt17500  leaves={len(base_leaves)}  "
      f"sha256={base_sha[:16]}...  ({time.time()-t0:.1f}s)", flush=True)

# ================================================================
# Build network, merge base + new params
# ================================================================
s4_base = MultiTaskMiniCraftaxEnv([S4Cls], StaticEnvParams(), ctor, True,
                                  conditioning_type="embedding",
                                  embedding_size=EMB)
ACTION_DIM = int(s4_base.action_space(ctor).n)
OBS_DIM = int(s4_base.observation_space(ctor).shape[0])

if args.arm == "w512":
    network = NetCls(
        action_dim=ACTION_DIM, activation=cfg.activation,
        encoder_size=cfg.embed_size, hidden_layers=cfg.hidden_layers,
        num_heads=cfg.num_heads, qkv_features=cfg.qkv_features,
        num_layers=cfg.num_layers, gating=cfg.gating,
        gating_bias=cfg.gating_bias, long_size=cfg.w512_long_size)
else:
    network = NetCls(
        action_dim=ACTION_DIM, activation=cfg.activation,
        encoder_size=cfg.embed_size, hidden_layers=cfg.hidden_layers,
        num_heads=cfg.num_heads, qkv_features=cfg.qkv_features,
        num_layers=cfg.num_layers, gating=cfg.gating,
        gating_bias=cfg.gating_bias, rmt_num_tokens=cfg.rmt_num_tokens)

rng_init = jax.random.PRNGKey(MASTER_SEED)
rng_init, _rng = jax.random.split(rng_init)
dummy_obs  = jnp.zeros((2, OBS_DIM))
dummy_mem  = jnp.zeros((2, cfg.window_mem, cfg.num_layers, cfg.embed_size))
dummy_mask = jnp.zeros((2, cfg.num_heads, 1, cfg.window_mem + 1), dtype=jnp.bool_)
if args.arm == "w512":
    full_params = network.init(_rng, dummy_mem, dummy_obs, dummy_mask,
                               long_buf=jnp.zeros((2, cfg.w512_long_size, cfg.embed_size)),
                               long_mask=jnp.zeros((2, cfg.w512_long_size), dtype=jnp.bool_))
else:
    full_params = network.init(_rng, dummy_mem, dummy_obs, dummy_mask,
                               mem_tokens=jnp.zeros((2, cfg.rmt_num_tokens, cfg.embed_size)),
                               seg_buf=jnp.zeros((2, cfg.num_steps, cfg.embed_size)),
                               method=network.init_all)

def _merge(base, full):
    if isinstance(base, dict) and isinstance(full, dict):
        out = dict(full)
        for k in base:
            if k in full:
                out[k] = _merge(base[k], full[k])
        return out
    return base

merged_params = _merge(base_params, full_params)
print(f"[merge] full_leaves={len(jax.tree_util.tree_leaves(merged_params))}  "
      f"base_sha={base_sha[:16]}...", flush=True)

# ================================================================
# Optimizer + TrainState
# ================================================================
tx = optax.chain(
    optax.clip_by_global_norm(cfg.max_grad_norm),
    optax.adam(cfg.lr, eps=1e-5))
train_state = TrainState.create(
    apply_fn=network.apply, params=merged_params, tx=tx)

# ================================================================
# Checkpoint helpers
# ================================================================
CKPT_DIR = os.path.join(args.out, "checkpoints")
LOG_DIR  = os.path.join(args.out, "logs")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def _to_np(pytree):
    return jax.tree_util.tree_map(lambda x: np.asarray(x), pytree)

def save_ckpt(runner_state, step):
    d = os.path.join(CKPT_DIR, str(step))
    os.makedirs(d, exist_ok=True)
    ts = runner_state[0]
    params_np = _to_np(ts.params)
    with open(os.path.join(d, "params.pkl"), "wb") as f:
        pickle.dump(params_np, f, protocol=4)
    # Strip non-picklable TrainState fields (apply_fn, tx)
    rs_save = {
        "ts_step": np.asarray(ts.step),
        "ts_params": params_np,
        "ts_opt_state": _to_np(ts.opt_state),
        "rest": _to_np(runner_state[1:]),
    }
    with open(os.path.join(d, "runner_state.pkl"), "wb") as f:
        pickle.dump(rs_save, f, protocol=4)
    p_sha = hashlib.sha256(
        b"".join(np.asarray(l).tobytes()
                 for l in jax.tree_util.tree_leaves(ts.params))).hexdigest()
    manifest = dict(step=step, arm=args.arm, params_sha256=p_sha,
                    gpu=args.gpu_uuid, seed=MASTER_SEED,
                    update_epochs=cfg.update_epochs,
                    num_minibatches=cfg.num_minibatches,
                    lr=cfg.lr,
                    timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                time.gmtime()))
    if args.arm == "w512":
        manifest["w512_long_size"] = cfg.w512_long_size
    else:
        manifest["rmt_num_tokens"] = cfg.rmt_num_tokens
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[ckpt] step={step}  params_sha={p_sha[:16]}...  dir={d}", flush=True)

# ================================================================
# Resume or fresh start
# ================================================================
if args.resume:
    print(f"[resume] loading {args.resume}", flush=True)
    with open(args.resume, "rb") as f:
        rs_raw = pickle.load(f)
    # Reconstruct TrainState from saved components (apply_fn/tx not pickled)
    ts_resumed = TrainState.create(
        apply_fn=network.apply,
        params=jax.tree_util.tree_map(jnp.asarray, rs_raw["ts_params"]),
        tx=tx)
    ts_resumed = ts_resumed.replace(
        opt_state=jax.tree_util.tree_map(jnp.asarray, rs_raw["ts_opt_state"]),
        step=jnp.asarray(rs_raw["ts_step"]))
    rest = jax.tree_util.tree_map(jnp.asarray, rs_raw["rest"])
    resume_runner_state = (ts_resumed,) + rest
    current_step = int(resume_runner_state[9])
    print(f"[resume] resuming from step={current_step}", flush=True)
else:
    resume_runner_state = None
    current_step = 0
    train_fn_0 = make_train(cfg, [S4Cls], 0,
                            task_embeddings=table,
                            task_distribution_proportions=jnp.array([1.0]),
                            initial_global_update_step=17500)
    rs0, _ = train_fn_0(jax.random.PRNGKey(MASTER_SEED),
                        train_state=train_state)
    save_ckpt(rs0, 0)
    resume_runner_state = rs0
    current_step = 0

# ================================================================
# Segmented training loop
# ================================================================
rng_train = jax.random.PRNGKey(MASTER_SEED + 1)
log_path = os.path.join(LOG_DIR, "training_log.jsonl")

while current_step < TOTAL_UPDATES:
    seg_end = min(current_step + SEGMENT, TOTAL_UPDATES)
    seg_updates = seg_end - current_step
    print(f"\n[train] segment {current_step} → {seg_end}  "
          f"({seg_updates} updates)", flush=True)

    train_fn = make_train(cfg, [S4Cls], seg_updates,
                          task_embeddings=table,
                          task_distribution_proportions=jnp.array([1.0]),
                          initial_global_update_step=17500 + current_step)

    t_seg = time.time()
    resume_runner_state, scan_info = train_fn(
        rng_train,
        train_state=resume_runner_state[0],
        resume_runner_state=resume_runner_state)
    jax.block_until_ready(resume_runner_state)
    seg_time = time.time() - t_seg

    current_step = seg_end
    metrics_np = jax.tree_util.tree_map(np.asarray, scan_info)
    entry = dict(step=current_step,
                 arm=args.arm,
                 seg_updates=seg_updates,
                 seg_time_s=round(seg_time, 1),
                 total_loss_mean=float(np.mean(metrics_np[0])),
                 value_loss_mean=float(np.mean(metrics_np[1])),
                 actor_loss_mean=float(np.mean(metrics_np[2])),
                 entropy_mean=float(np.mean(metrics_np[3])),
                 grad_norm_mean=float(np.mean(metrics_np[4])),
                 grad_norm_max=float(np.max(metrics_np[5])),
                 timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                             time.gmtime()))
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[train] step={current_step}  "
          f"actor_loss={entry['actor_loss_mean']:.4f}  "
          f"entropy={entry['entropy_mean']:.4f}  "
          f"gn_max={entry['grad_norm_max']:.4f}  "
          f"({seg_time:.1f}s)", flush=True)

    if not np.isfinite(entry['actor_loss_mean']) or \
       not np.isfinite(entry['entropy_mean']) or \
       not np.isfinite(entry['grad_norm_max']):
        print(f"[HARD_STOP] NaN/Inf detected at step={current_step}", flush=True)
        sys.exit(1)

    if current_step in CKPT_STEPS:
        save_ckpt(resume_runner_state, current_step)

    rng_train = jax.random.fold_in(rng_train, current_step)

print("\n" + "=" * 72, flush=True)
print(f"{ARM_LABEL} training complete.  final step={current_step}", flush=True)
print(f"  checkpoints: {CKPT_DIR}", flush=True)
print(f"  log: {log_path}", flush=True)
print("=" * 72, flush=True)
