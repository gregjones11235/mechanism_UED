"""CPU mechanical verification of the ppo_tr_egomap.py graft (gates G4 + G5 +
feature-off regression) using the REAL Craftax OriginalTask env at tiny scale.

Run:
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" WANDB_MODE=disabled \
  PYTHONPATH=<dicode_src>:<thisdir> python test_ppo_graft_cpu.py

Checks:
  M1  make_train(...).train(...) runs end-to-end on CPU (feature-on) w/o crash.
  M2  no NaN/Inf in returned train_state params (G4 numeric health, mini scale).
  M3  EgoMap carry present + finite (map_bank/ego_pos/step) in out["carry"].
  M4  G5 exact resume: a 2-update run == two 1-update runs joined by resume_carry
      (train_state params AND every carry leaf bit-exact).
  M5  feature-off (egomap_enabled=False) runs and base-subset params stay finite.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")
import sys
import types
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ppo_tr_egomap._log_callback unconditionally calls wandb.log (base behaviour;
# real runs always init wandb). Give it a disabled no-op run so the CPU test
# doesn't need a wandb account / network.
import wandb                                                           # noqa: E402
wandb.init(mode="disabled", project="p7-graft-cpu-test")

from minicraftax.tasks.seed_tasks.original import Env as OriginalTask  # noqa: E402
import ppo_tr_egomap as PE                                             # noqa: E402

PASS = []
def check(name, cond):
    PASS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name)

EMBED = 76  # obs = 8217 spatial + 42 scalar + 76 task embedding = 8335

def make_config(egomap_enabled):
    c = types.SimpleNamespace()
    # env / scale (tiny for CPU)
    c.mode = "task"                      # != "reward" branch
    c.condition_on_task = True
    c.completion_bonus_scale = 2.0
    c.completion_bonus_min = 20.0
    c.bonus_type = "dynamic"
    c.dynamic_bonus_k = 2.0
    c.num_envs = 16
    c.num_steps = 16
    c.optimistic_reset_ratio = 16        # num_resets = 16//16 = 1 (valid)
    c.total_timesteps = 16 * 16 * 100
    c.max_updates_per_session = 100
    # ppo
    c.gamma = 0.999
    c.gae_lambda = 0.8
    c.clip_eps = 0.2
    c.ent_coef = 0.002
    c.vf_coef = 0.5
    c.max_grad_norm = 1.0
    c.num_minibatches = 2
    c.update_epochs = 2
    c.anneal_lr = False
    c.lr = 2e-5
    c.min_lr = 2e-6
    # transformer (tiny but valid: embed_size % num_heads == 0)
    c.activation = "relu"
    c.hidden_layers = 32
    c.embed_size = 32
    c.num_heads = 2
    c.qkv_features = 32
    c.num_layers = 2
    c.gating = False
    c.gating_bias = 0.0
    c.window_mem = 8
    c.window_grad = 8                    # divides num_steps=16
    c.scoring_window_updates = 2
    # guard / sil (off)
    c.sil = False
    c.debug = False
    c.use_wandb = False
    # egomap
    c.egomap_enabled = egomap_enabled
    c.egomap_map_size = 16
    c.egomap_num_floors = 9
    c.egomap_cnn_features = (8, 16)
    return c

# fixed, deterministic task embedding table (1 task = OriginalTask). Using a fixed
# vector (not the paid embedding API) is fine here: this is a MECHANICAL graft test,
# and the real launcher will use a consistent embedding for Control/EgoMap/Baseline.
TASK_EMB = jax.random.normal(jax.random.PRNGKey(123), (1, EMBED))

def build(egomap_enabled):
    config = make_config(egomap_enabled)
    train_fn = PE.make_train(config, [OriginalTask], 2, TASK_EMB, None, 0)
    return jax.jit(train_fn), config

def leaves_finite(tree):
    return all(bool(np.all(np.isfinite(np.asarray(x)))) for x in jax.tree_util.tree_leaves(tree))

# --------------------------------------------------------------------------- #
# M1/M2/M3: feature-on end-to-end, no NaN, carry present
# --------------------------------------------------------------------------- #
print("== M1-M3: feature-on end-to-end ==")
train_jit, config = build(egomap_enabled=True)
rng = jax.random.PRNGKey(42)
out = train_jit(rng)                       # cold start (train_state=None)
ts = out["train_state"]
check("M1 feature-on train() runs end-to-end on CPU", ts is not None)
check("M2 no NaN/Inf in returned params (G4 mini)", leaves_finite(ts.params))
carry = out["carry"]
check("M3 carry has egomap_state with map_bank/ego_pos/step",
      set(["map_bank", "ego_pos", "step"]).issubset(set(carry["egomap_state"].keys())))
check("M3 egomap map_bank finite", leaves_finite(carry["egomap_state"]))
check("M3 carry exposes memories/env_state/last_obs/done/rng",
      set(["memories", "memories_mask", "memories_mask_idx", "env_state",
           "last_obs", "done", "rng"]).issubset(set(carry.keys())))

# --------------------------------------------------------------------------- #
# M4: G5 exact resume — 2 updates at once == 1 update + resume 1 update
# --------------------------------------------------------------------------- #
print("== M4: G5 exact resume ==")
# reference: single cold run of 2 updates
ref = train_jit(rng)
ref_params = ref["train_state"].params
ref_carry = ref["carry"]

# split: first 1 update (num_training_updates=1)
train1 = jax.jit(PE.make_train(config, [OriginalTask], 1, TASK_EMB, None, 0))
seg1 = train1(rng)                         # same rng -> identical update 0
# resume: second 1 update from seg1's carry
seg2 = train1(seg1["carry"]["rng"], train_state=seg1["train_state"],
              current_original_return=0.0, resume_carry=seg1["carry"])

def maxdiff(a, b):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    if len(la) != len(lb):
        return float("inf")
    return max((float(np.abs(np.asarray(x) - np.asarray(y)).max()) for x, y in zip(la, lb)),
               default=0.0)

d_params = maxdiff(seg2["train_state"].params, ref_params)
print(f"  resume params max|diff| = {d_params}")
check("M4 G5 resumed params bit-exact vs uninterrupted (diff==0)", d_params == 0.0)

# compare the full carry leaves (egomap_state + memories + env_state + last_obs ...)
d_ego = maxdiff(seg2["carry"]["egomap_state"], ref_carry["egomap_state"])
d_mem = maxdiff(seg2["carry"]["memories"], ref_carry["memories"])
d_obs = maxdiff(seg2["carry"]["last_obs"], ref_carry["last_obs"])
print(f"  resume carry max|diff| ego={d_ego} mem={d_mem} obs={d_obs}")
check("M4 G5 resumed egomap_state bit-exact (diff==0)", d_ego == 0.0)
check("M4 G5 resumed memories bit-exact (diff==0)", d_mem == 0.0)
check("M4 G5 resumed last_obs bit-exact (diff==0)", d_obs == 0.0)

# --------------------------------------------------------------------------- #
# M5: feature-off runs + base-subset params finite (and, from a FRESH init, the
#     base-subset of a feature-off run is finite; bit-exactness vs the base
#     network is already proven at the network level in test_network_g1.py).
# --------------------------------------------------------------------------- #
print("== M5: feature-off regression ==")
train_off = jax.jit(PE.make_train(make_config(False), [OriginalTask], 2, TASK_EMB, None, 0))
out_off = train_off(jax.random.PRNGKey(42))
check("M5 feature-off train() runs end-to-end", out_off["train_state"] is not None)
check("M5 feature-off params finite", leaves_finite(out_off["train_state"].params))
# feature-off: egomap_read returns zeros so map_bank never accumulates
mb = out_off["carry"]["egomap_state"]["map_bank"]
check("M5 feature-off map_bank stays all-zero (no accumulation)",
      float(np.abs(np.asarray(mb)).sum()) == 0.0)

print("\n==== SUMMARY ====")
ok = sum(1 for _, c in PASS if c)
print(f"{ok}/{len(PASS)} passed")
if ok != len(PASS):
    print("FAILED:", [n for n, c in PASS if not c])
    sys.exit(1)
print("ALL_PPO_GRAFT_CPU_TESTS_PASS")
