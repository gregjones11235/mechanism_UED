#!/usr/bin/env python3
"""§14 Control health-calibration trainer (frozen Control protocol).

Control = ckpt17500 + ORIGINAL Henry GTrXL-PPO (ppo_tr.run_training_session /
make_train, i.e. compute_on_policy_gae + native PPO update; replay/hindsight are
ABSENT by construction — they are P2-v1 additions that live OUTSIDE Henry's
run_training_session and are simply never called here). Static S4_dark training
task, 24576 env steps (12 updates) for the grid, frozen §14 config
(gamma=0.999 / gae_lambda=0.8 / clip=0.2 / vf=0.5 / ent=0.002 / gradnorm=1.0 /
num_envs=16 / num_steps=128 / anneal_lr=False), LR = grid value (--lr).

Starting point is weights-only from the healthy session175 base ckpt17500 via
load_weights_only(load_opt_state=False) -> restored params + FRESH Adam
(clip_by_global_norm(1.0)+adam(lr, eps=1e-5)), step=0. Encoder kernel shape is
fail-closed verified (obs_dim, embed_size). Source-weight SHA256 recorded.

Output (per LR, independent dirs — never reused):
  <ckpt_root>/<steps>/default/   orbax FLAT TrainState (loadable by load_weights_only)
  <out_dir>/control_manifest.json  provenance + config + NaN/finite check
  <out_dir>/control_log.jsonl       per-update loss/entropy/grad_norm (from make_train)

GPU0 only (UUID bound). wandb is neutralized (mode=disabled) because make_train's
_log_callback calls wandb.log unconditionally every update.
"""
import os
GPU_UUID = "GPU-e8c08612-c22a-c8a4-6df5-affb2dd1f9a6"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID           # MUST precede jax import
os.environ["WANDB_MODE"] = "disabled"                    # neutralize make_train wandb.log
os.environ["WANDB_SILENT"] = "true"
import sys, json, argparse, hashlib, time

V7_SRC = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src"
if V7_SRC not in sys.path:
    sys.path.insert(0, V7_SRC)

import wandb
try:
    wandb.init(mode="disabled")                          # no-op run; wandb.log is a no-op
except Exception as e:
    print(f"[control] wandb.init(disabled) note: {e}")

import numpy as np
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from craftax.craftax.constants import Achievement
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from dicode.network import ActorCriticTransformer
from dicode.task_utils import get_achievement_multi_hot
from dicode.utils.general.train_state_utils import load_weights_only
from dicode.ppo_tr import run_training_session
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv

SESSION175_CKPT = ("/home/oseasy/incoming/henry_work_20260721T105300/extracted/"
                   "Henry_work/base_ckpt_v7fix55_armA_s0/rl_checkpoints/17500")
S4_TASK_PATH = "/home/oseasy/experiments/p2_v1_20260722/evidence/s4_task_code.py"

NUM_ENVS = 16
ROLLOUT_STEPS = 128
STEPS_PER_UPDATE = NUM_ENVS * ROLLOUT_STEPS              # 2048
GAMMA = 0.999
GAE_LAMBDA = 0.8
MASTER_SEED = 42                                          # shared training+env seed (== §14 same seed)
GRID_LRS = (2e-4, 6e-5, 2e-5)


class Cfg:
    """Frozen §14 Control config (field set == proven p2_v1 stage4 Cfg). LR per run."""
    lr = 2e-4
    min_lr = 2e-6
    num_envs = NUM_ENVS
    num_steps = ROLLOUT_STEPS
    update_epochs = 1
    num_minibatches = 2
    gamma = GAMMA
    gae_lambda = GAE_LAMBDA
    clip_eps = 0.2
    ent_coef = 0.002
    vf_coef = 0.5
    max_grad_norm = 1.0
    activation = "relu"
    anneal_lr = False
    qkv_features = 256
    embed_size = 256
    num_heads = 8
    num_layers = 2
    hidden_layers = 256
    window_mem = 128
    window_grad = 64
    gating = True
    gating_bias = 2.0
    condition_on_task = True
    optimistic_reset_ratio = 16
    mode = "score"
    bonus_type = "none"
    dynamic_bonus_k = 0.0
    completion_bonus_scale = 0.0
    completion_bonus_min = 0.0
    max_updates_per_session = 12
    total_timesteps = 2_005_401_600
    scoring_window_updates = 4
    value_target_clip_min = -50.0
    value_target_clip_max = 300.0
    guard_session_vloss_max = 1000.0
    guard_session_entropy_min = 0.10
    guard_max_consecutive_reverts = 2
    lr_restart = 0.0
    lr_restart_at = 0
    lr_restart_horizon = 0
    lr_restart_warmup = 50
    sil = False
    sil_pools = []
    # extras read at top level by run_training_session / make_train logging path
    use_wandb = False
    debug = False
    validation = None
    dicode_manager = None

    def get(self, key, default=None):
        return getattr(self, key, default)


def _cfg_resolved_dict(cfg):
    out = {}
    for k in dir(cfg):
        if k.startswith("_") or k == "training":
            continue
        v = getattr(cfg, k)
        if callable(v):
            continue
        out[k] = v
    return out


def _params_sha(params):
    leaves = jax.tree_util.tree_leaves(params)
    h = hashlib.sha256()
    for v in leaves:
        a = np.ascontiguousarray(np.asarray(v))
        h.update(a.tobytes())
    return h.hexdigest()


def _find_encoder_kernel_shape(params):
    """Locate the encoder's first Dense kernel shape (obs_dim, embed)."""
    flat = jax.tree_util.tree_flatten_with_path(params)[0]
    for kp, v in flat:
        ks = "/".join(getattr(k, "key", str(getattr(k, "idx", "?"))) for k in kp)
        if "encoder" in ks and ks.endswith("kernel") and np.asarray(v).ndim == 2:
            return tuple(int(x) for x in np.asarray(v).shape)
    raise RuntimeError("encoder kernel not found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, required=True,
                    help="total env steps; must be a multiple of 2048 (24576 for the grid)")
    ap.add_argument("--lr", type=float, required=True, help="grid LR (2e-4 / 6e-5 / 2e-5)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckpt_root", required=True)
    ap.add_argument("--seed", type=int, default=MASTER_SEED)
    args = ap.parse_args()

    assert args.steps % STEPS_PER_UPDATE == 0, "steps must be a multiple of 2048"
    num_updates = args.steps // STEPS_PER_UPDATE
    assert args.lr in GRID_LRS, f"LR {args.lr} not in frozen grid {GRID_LRS}"

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ckpt_root, exist_ok=True)
    log_path = os.path.join(args.out_dir, "control_log.jsonl")
    logf = open(log_path, "a")

    def log(rec):
        logf.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
        logf.flush()

    cfg = Cfg()
    cfg.lr = float(args.lr)
    cfg.max_updates_per_session = num_updates
    cfg.training = cfg                                    # run_training_session: config_t = config.training

    # GPU UUID binding check (fail-closed)
    devs = jax.local_devices()
    print(f"[control] jax devices: {devs}")

    # [1] static S4_dark task + base env (for weights load + make_train env config)
    ach_table = jnp.array(
        [get_achievement_multi_hot([Achievement.DEFEAT_KOBOLD])], dtype=jnp.float32)
    EMB = int(ach_table.shape[1])
    with open(S4_TASK_PATH) as f:
        s4_code = f.read()
    ns = {}
    exec(s4_code, ns)
    Task = ns["Env"]
    static_env_params = StaticEnvParams()
    env_params = EnvParams(max_timesteps=4096)
    base_env = MultiTaskMiniCraftaxEnv(
        [Task], static_env_params, env_params, cfg.condition_on_task,
        conditioning_type="embedding", embedding_size=EMB,
        completion_bonus_scale=cfg.completion_bonus_scale,
        completion_bonus_min=cfg.completion_bonus_min,
        bonus_type=cfg.bonus_type, dynamic_bonus_k=cfg.dynamic_bonus_k)
    obs_dim = int(base_env.observation_space(env_params).shape[0])
    action_dim = int(base_env.action_space(env_params).n)
    print(f"[control] S4_dark env: obs_dim={obs_dim} action_dim={action_dim} emb={EMB}")

    # [2] weights-only init from ckpt17500 + fresh Adam (load_opt_state=False)
    ts = load_weights_only(SESSION175_CKPT, base_env, env_params, cfg,
                           load_opt_state=False)
    source_sha = _params_sha(ts.params)
    kshape = _find_encoder_kernel_shape(ts.params)
    assert kshape == (obs_dim, cfg.embed_size), \
        f"REFUSED: encoder kernel {kshape} != ({obs_dim},{cfg.embed_size})"
    opt_step0 = int(ts.step)
    assert opt_step0 == 0, f"REFUSED: fresh opt step != 0 (got {opt_step0})"
    n_leaves = len(jax.tree_util.tree_leaves(ts.params))
    print(f"[control] loaded ckpt17500: sha={source_sha} kernel={kshape} "
          f"leaves={n_leaves} opt_step={opt_step0} lr={cfg.lr}")
    log({"event": "init", "source_checkpoint": SESSION175_CKPT,
         "source_sha256": source_sha, "encoder_kernel": list(kshape),
         "obs_dim": obs_dim, "action_dim": action_dim, "emb": EMB,
         "param_leaves": n_leaves, "opt_step0": opt_step0, "lr": cfg.lr})

    # [3] run ORIGINAL Henry PPO (run_training_session; no replay/hindsight)
    rng = jax.random.PRNGKey(args.seed)
    task_embeddings = ach_table                            # [1, 67]
    t0 = time.time()
    results = run_training_session(
        cfg, rng, [Task], num_updates,
        task_embeddings=task_embeddings, train_state=ts,
        global_update_step=0, current_original_return=0.0)
    elapsed = time.time() - t0
    trained_ts = results["train_state"]
    res_keys = sorted(str(k) for k in results.keys())
    print(f"[control] run_training_session done in {elapsed:.1f}s  "
          f"result_keys={res_keys}")

    # [4] NaN/Inf fail-closed check on trained params
    leaves = jax.tree_util.tree_leaves(trained_ts.params)
    finite = bool(all(np.all(np.isfinite(np.asarray(v))) for v in leaves))
    trained_sha = _params_sha(trained_ts.params)
    advanced = (trained_sha != source_sha)
    log({"event": "train_done", "elapsed_s": round(elapsed, 1),
         "result_keys": res_keys, "params_finite": finite,
         "trained_sha256": trained_sha, "params_advanced": advanced,
         "final_opt_step": int(trained_ts.step)})
    if not finite:
        raise RuntimeError("FAIL: NaN/Inf in trained Control params")
    if not advanced:
        raise RuntimeError("FAIL: Control params did not advance from ckpt17500")

    # per-update loss/entropy/grad_norm if make_train returned them
    for k in res_keys:
        if k == "train_state":
            continue
        v = results[k]
        try:
            arr = np.asarray(v)
            if arr.ndim <= 1 and np.issubdtype(arr.dtype, np.number):
                log({"event": "train_metric", "key": k,
                     "values": [float(x) for x in arr.reshape(-1)]})
        except Exception:
            pass

    # [5] save orbax FLAT TrainState (loadable by load_weights_only Attempt-B)
    mgr = ocp.CheckpointManager(
        args.ckpt_root, ocp.PyTreeCheckpointer(),
        options=ocp.CheckpointManagerOptions(create=True))
    mgr.save(args.steps, items=trained_ts)
    mgr.wait_until_finished()
    ckpt_dir = os.path.join(args.ckpt_root, str(args.steps))
    print(f"[control] saved checkpoint -> {ckpt_dir}")

    # [6] restore round-trip provenance (bit-exact via load_weights_only)
    ts_rt = load_weights_only(ckpt_dir, base_env, env_params, cfg,
                              load_opt_state=False)
    rt_sha = _params_sha(ts_rt.params)
    roundtrip_ok = (rt_sha == trained_sha)
    print(f"[control] restore round-trip: saved={trained_sha[:16]} "
          f"restored={rt_sha[:16]} ok={roundtrip_ok}")
    if not roundtrip_ok:
        raise RuntimeError("FAIL: checkpoint restore round-trip SHA mismatch")

    manifest = {
        "label": "Control_original_Henry_PPO",
        "protocol": "frozen §14 Control: ckpt17500 + run_training_session (native "
                    "GTrXL-PPO; replay/hindsight OFF by construction), static S4_dark",
        "source_checkpoint": SESSION175_CKPT,
        "source_checkpoint_sha256": source_sha,
        "lr": cfg.lr, "lr_grid": list(GRID_LRS),
        "total_env_steps": args.steps, "num_updates": num_updates,
        "num_envs": NUM_ENVS, "num_steps": ROLLOUT_STEPS,
        "steps_per_update": STEPS_PER_UPDATE,
        "gamma": GAMMA, "gae_lambda": GAE_LAMBDA, "clip_eps": cfg.clip_eps,
        "vf_coef": cfg.vf_coef, "ent_coef": cfg.ent_coef,
        "max_grad_norm": cfg.max_grad_norm, "anneal_lr": cfg.anneal_lr,
        "update_epochs": cfg.update_epochs, "num_minibatches": cfg.num_minibatches,
        "mode": cfg.mode, "condition_on_task": cfg.condition_on_task,
        "optimistic_reset_ratio": cfg.optimistic_reset_ratio,
        "replay_hindsight": "OFF (pure Henry run_training_session)",
        "master_seed": args.seed, "gpu_uuid": GPU_UUID,
        "encoder_kernel": list(kshape), "obs_dim": obs_dim,
        "action_dim": action_dim, "param_leaves": n_leaves,
        "trained_params_sha256": trained_sha, "params_finite": finite,
        "params_advanced": advanced, "final_opt_step": int(trained_ts.step),
        "restore_roundtrip_ok": roundtrip_ok, "elapsed_s": round(elapsed, 1),
        "result_keys": res_keys,
        "config_resolved": _cfg_resolved_dict(cfg),
    }
    with open(os.path.join(args.out_dir, "control_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    log({"event": "manifest", **manifest})
    logf.close()
    print(f"CONTROL_TRAIN_OK lr={cfg.lr} steps={args.steps} finite={finite} "
          f"advanced={advanced} roundtrip={roundtrip_ok} ckpt={ckpt_dir}")


if __name__ == "__main__":
    main()
