#!/usr/bin/env python3
"""Convert P7 pkl checkpoints to orbax format for eval_paired_256.py.

Usage:
  python convert_p7_to_orbax.py --arm control --step 98304 --out /path/to/orbax
"""
import argparse, os, sys, pickle
import jax, jax.numpy as jnp

V7_SRC = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src"
if V7_SRC not in sys.path:
    sys.path.insert(0, V7_SRC)

from craftax.craftax.constants import Achievement
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from dicode.network import ActorCriticTransformer
from dicode.task_utils import get_achievement_multi_hot
from dicode.utils.general.train_state_utils import load_weights_only
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv
from minicraftax.tasks.seed_tasks.original import Env as OriginalTask
import orbax.checkpoint as ocp

ap = argparse.ArgumentParser()
ap.add_argument("--pkl", required=True, help="Path to P7 pkl params file")
ap.add_argument("--out", required=True, help="Output orbax checkpoint dir")
ap.add_argument("--step", type=int, required=True)
ap.add_argument("--base_only", action="store_true", help="Strip ego_encoder/ego_gate leaves")
args = ap.parse_args()

_cfg = dict(activation="relu", embed_size=256, hidden_layers=256, num_heads=8,
            qkv_features=256, num_layers=2, gating=True, gating_bias=2.0,
            window_mem=128, window_grad=64, anneal_lr=False, lr=2e-4, min_lr=2e-6,
            max_grad_norm=1.0, total_timesteps=2005401600, num_envs=1024, num_steps=128,
            update_epochs=4, num_minibatches=8, max_updates_per_session=500)
cfg = type("C", (), _cfg)()

table = jnp.array([get_achievement_multi_hot([Achievement.DEFEAT_KOBOLD])], dtype=jnp.float32)
EMB = int(table.shape[1])
ctor = EnvParams(max_timesteps=4096)
base_env = MultiTaskMiniCraftaxEnv([OriginalTask], StaticEnvParams(), ctor, True,
    conditioning_type="embedding", embedding_size=EMB)

# Load baseline to get TrainState structure
BASE_CKPT = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/base_ckpt_v7fix55_armA_s0/rl_checkpoints/17500"
ts_template = load_weights_only(BASE_CKPT, base_env, ctor, cfg, load_opt_state=False)
print(f"Template params leaves: {len(jax.tree_util.tree_leaves(ts_template.params))}")

# Load P7 pkl
with open(args.pkl, "rb") as f:
    p7_flat = pickle.load(f)

if args.base_only:
    # Strip ego keys
    p7_base = {k: v for k, v in p7_flat.items()
               if not k.startswith("ego_encoder") and not k.startswith("ego_gate")}
    print(f"Base-only leaves: {len(jax.tree_util.tree_leaves(p7_base))}")
    new_params = {"params": p7_base}
else:
    new_params = {"params": p7_flat}

# Replace params in template
new_ts = ts_template.replace(params=new_params)
print(f"New params leaves: {len(jax.tree_util.tree_leaves(new_ts.params))}")

# Save as orbax
os.makedirs(args.out, exist_ok=True)
options = ocp.CheckpointManagerOptions(max_to_keep=1)
mngr = ocp.CheckpointManager(args.out, options=options)
mngr.save(args.step, args=ocp.args.StandardSave(new_ts))
mngr.wait_until_finished()
print(f"Saved orbax checkpoint to {args.out}/{args.step}")

# Verify
ts_check = load_weights_only(os.path.join(args.out, str(args.step)), base_env, ctor, cfg, load_opt_state=False)
check_leaves = jax.tree_util.tree_leaves(ts_check.params)
print(f"Verified: {len(check_leaves)} leaves loaded back")

import hashlib
h = hashlib.sha256()
for leaf in check_leaves:
    h.update(jnp.asarray(leaf).tobytes())
print(f"Verified SHA: {h.hexdigest()}")
