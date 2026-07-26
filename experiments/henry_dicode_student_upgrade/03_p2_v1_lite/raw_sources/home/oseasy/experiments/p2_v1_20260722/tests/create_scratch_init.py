#!/usr/bin/env python3
"""Create + verify the COMMON random initialization scratch_init_seed0 (GPU0).

One random network init from a fixed seed (0), saved as an orbax checkpoint that
the frozen launcher's load_weights_only can read. Both Original PPO and P2-v1
will point SESSION175_CKPT at this artifact so init_fresh_from_session175 loads
the IDENTICAL params bit-exact (per-leaf), then builds fresh optimizer/replay/
RNG/global_step=0/update_count=0. No session175 / P2-v0 / P2-v1 weights used.

Verifies:
  - random params generated (seed 0), encoder kernel == (8335, 256), opt_step==0
  - saved to scratch_init_seed0/0 (orbax + replay_meta + manifest)
  - load_weights_only reload is bit-exact (params content SHA matches)
  - the REAL launcher init path (init_fresh_from_session175 with SESSION175_CKPT
    -> scratch) returns bit-exact params + FRESH optimizer/replay + step0/update0
Records INIT_METADATA.json: params SHA256, init seed, network config, obs schema,
optimizer config, code SHA256. Prints PASS/FAIL. Frozen source is NOT modified.
"""
import os, sys, json, hashlib
GPU_UUID = "GPU-e8c08612-c22a-c8a4-6df5-affb2dd1f9a6"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID
import jax, jax.numpy as jnp
import numpy as np

P2 = "/home/oseasy/experiments/p2_v1_20260722/src"
V7 = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src"
for p in (V7, P2):
    if p not in sys.path: sys.path.insert(0, p)
import stage4_continue_launcher as L
from checkpointing import save_full_checkpoint

INIT_SEED = 0
SCRATCH_PARENT = "/home/oseasy/experiments/p2_v1_20260722/scratch_init_seed0"
SCRATCH_STEP0 = os.path.join(SCRATCH_PARENT, "0")   # load_weights_only reads this step dir
OBS_DIM_EXPECTED = 8335

def sha(path):
    with open(path, "rb") as f: return hashlib.sha256(f.read()).hexdigest()

gates = {}
cfg = L.Cfg()

# ── build dummy env (obs_dim/action_dim) exactly as main() [1/6] ──────
ach_table = jnp.array([L.get_achievement_multi_hot([L.Achievement.DEFEAT_KOBOLD])], dtype=jnp.float32)
EMB = int(ach_table.shape[1])
dummy = L.CraftaxAugObsTrain(condition_on_task=True, conditioning_type="embedding",
    embedding_size=EMB, task_embeddings=jnp.zeros((1, EMB)))
obs_dim = dummy.observation_space(dummy.default_params).shape[0]
action_dim = dummy.action_space(dummy.default_params).n
gates["obs_dim_8335"] = (obs_dim == OBS_DIM_EXPECTED)
print(f"[scratch] obs_dim={obs_dim} action_dim={action_dim} EMB={EMB}", flush=True)

# ── build network + random init (seed 0) ─────────────────────────────
network = L.ActorCriticTransformer(action_dim=action_dim, activation=cfg.activation,
    hidden_layers=cfg.hidden_layers, encoder_size=cfg.embed_size, num_heads=cfg.num_heads,
    qkv_features=cfg.qkv_features, num_layers=cfg.num_layers, gating=cfg.gating, gating_bias=cfg.gating_bias)
rng_init = jax.random.PRNGKey(INIT_SEED)
params = L.init_network_params(network, obs_dim, cfg, rng_init)
kernel = L._find_encoder_kernel_shape(params)
gates["encoder_kernel_8335_256"] = (kernel == (OBS_DIM_EXPECTED, int(cfg.embed_size)))
n_leaves = len(jax.tree_util.tree_leaves(params))
gates["params_finite"] = bool(all(np.all(np.isfinite(np.asarray(v))) for v in jax.tree_util.tree_leaves(params)))
params_sha = L._params_content_sha256(params)
print(f"[scratch] random init seed={INIT_SEED}  leaves={n_leaves}  kernel={kernel}  sha={params_sha}", flush=True)

# fresh optimizer + verify step 0
ts = L.build_stage4_train_state(network, params, cfg)
gates["fresh_opt_step_0"] = (L._optimizer_step_count(ts) == 0)

# ── save artifact scratch_init_seed0/0 ───────────────────────────────
replay = L.TrajectoryReplayBuffer(capacity=L.REPLAY_CAPACITY, seed=L.P2_V1_MASTER_SEED)
rng = jax.random.PRNGKey(L.P2_V1_MASTER_SEED)
action_rng = L.make_action_rng(L.P2_V1_MASTER_SEED)
os.makedirs(SCRATCH_PARENT, exist_ok=True)
saved_dir = save_full_checkpoint(ts, replay, rng, 0, SCRATCH_PARENT, step=0,
    action_rng_state=action_rng.bit_generator.state, update_count=0,
    pending_state=None, collector_state=None, aux_opt_state=None)
gates["artifact_saved"] = os.path.isdir(SCRATCH_STEP0)
print(f"[scratch] saved -> {saved_dir}", flush=True)

# ── reload via load_weights_only (bit-exact) ─────────────────────────
# build the real Stage4 base env exactly as main() [3/6] for the weights load
with open(L.S4_TASK_PATH) as f: s4_code = f.read()
ns = {}; exec(s4_code, ns); Task = ns["Env"]
static_env_params = L.StaticEnvParams(); env_params = L.EnvParams(max_timesteps=4096)
base_env = L.MultiTaskMiniCraftaxEnv([Task], static_env_params, env_params, cfg.condition_on_task,
    conditioning_type="embedding", embedding_size=EMB, completion_bonus_scale=cfg.completion_bonus_scale,
    completion_bonus_min=cfg.completion_bonus_min, bonus_type=cfg.bonus_type, dynamic_bonus_k=cfg.dynamic_bonus_k)
ts_reload = L.load_weights_only(SCRATCH_STEP0, base_env, env_params, cfg, load_opt_state=False)
reload_sha = L._params_content_sha256(ts_reload.params)
gates["reload_bitexact"] = (reload_sha == params_sha)
print(f"[scratch] load_weights_only reload sha={reload_sha}  bitexact={reload_sha==params_sha}", flush=True)

# ── verify the REAL launcher init path loads it bit-exact + fresh ────
_orig_ckpt = L.SESSION175_CKPT
L.SESSION175_CKPT = SCRATCH_STEP0
fresh = L.init_fresh_from_session175(network, base_env, env_params, cfg, obs_dim)
L.SESSION175_CKPT = _orig_ckpt
fresh_sha = fresh["source_checkpoint_sha256"]
gates["launcher_init_bitexact"] = (fresh_sha == params_sha)
gates["launcher_init_fresh_opt0"] = (fresh["optimizer_step"] == 0)
gates["launcher_init_replay_empty"] = (fresh["replay_size"] == 0)
gates["launcher_init_step0"] = (int(fresh["global_step"]) == 0)
gates["launcher_init_update0"] = (int(fresh["update_count"]) == 0)
# fresh optimizer must be distinct object with zero count
gates["launcher_init_opt_count0"] = (L._optimizer_step_count(fresh["train_state"]) == 0)
print(f"[scratch] launcher init_fresh_from_session175 -> sha={fresh_sha} bitexact={fresh_sha==params_sha} "
      f"opt_step={fresh['optimizer_step']} replay={fresh['replay_size']} step={fresh['global_step']} uc={fresh['update_count']}", flush=True)

# ── metadata + code SHAs ─────────────────────────────────────────────
meta = {
    "artifact": "scratch_init_seed0 (common random init for from-scratch fair comparison)",
    "init_seed": INIT_SEED, "training_seed": L.P2_V1_MASTER_SEED,
    "scratch_step0_dir": SCRATCH_STEP0,
    "params_sha256": params_sha, "param_leaves": n_leaves,
    "encoder_kernel": list(kernel), "obs_dim": int(obs_dim), "action_dim": int(action_dim),
    "embedding_size": EMB,
    "network_config": {k: getattr(cfg, k) for k in
        ["activation","hidden_layers","embed_size","num_heads","qkv_features","num_layers",
         "gating","gating_bias","window_mem","window_grad"]},
    "optimizer_config": {"type": "optax.chain(clip_by_global_norm(max_grad_norm), adam(lr, eps=1e-5))",
        "max_grad_norm": cfg.max_grad_norm, "lr": cfg.lr, "eps": 1e-5, "anneal_lr": cfg.anneal_lr,
        "fresh_opt_step_at_init": 0},
    "ppo_config": {"gamma": cfg.gamma, "gae_lambda": cfg.gae_lambda, "clip_eps": cfg.clip_eps,
        "ent_coef": cfg.ent_coef, "vf_coef": cfg.vf_coef, "num_minibatches": cfg.num_minibatches,
        "update_epochs": cfg.update_epochs, "num_envs": L.NUM_ENVS, "rollout_steps": L.ROLLOUT_STEPS},
    "frozen_code_sha256": {
        "long_context_learner.py": sha(os.path.join(P2, "long_context_learner.py")),
        "stage4_continue_launcher.py": sha(os.path.join(P2, "stage4_continue_launcher.py")),
        "p2_v1_core.py": sha(os.path.join(P2, "p2_v1_core.py")),
        "checkpointing.py": sha(os.path.join(P2, "checkpointing.py")),
    },
    "gates": gates,
}
meta["all_pass"] = bool(all(gates.values()))
with open(os.path.join(SCRATCH_PARENT, "INIT_METADATA.json"), "w") as f:
    json.dump(meta, f, indent=2, sort_keys=True, default=str); f.write("\n")
ev = "/home/oseasy/experiments/single_director_20260722/evidence"
os.makedirs(ev, exist_ok=True)
with open(os.path.join(ev, "p2_v1_scratch_init_seed0.json"), "w") as f:
    json.dump(meta, f, indent=2, sort_keys=True, default=str); f.write("\n")

print("\n" + "="*60, flush=True)
for k in sorted(gates): print(f"  {'PASS' if gates[k] else 'FAIL'}  {k}", flush=True)
print("="*60, flush=True)
print(f"params_sha256 = {params_sha}", flush=True)
print(f"SCRATCH INIT VERIFY: {'ALL PASS' if meta['all_pass'] else 'FAIL'}", flush=True)
sys.exit(0 if meta["all_pass"] else 1)
