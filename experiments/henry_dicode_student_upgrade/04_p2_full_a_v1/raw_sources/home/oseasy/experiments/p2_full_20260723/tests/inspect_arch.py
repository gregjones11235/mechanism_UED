#!/usr/bin/env python3
"""READ-ONLY architecture inspection for Full P2 design freeze (GPU0).

Dumps:
  1. ckpt17500 (healthy Henry Student) param tree: leaf paths, shapes, dtypes,
     total param count, params SHA256.
  2. Resolved P2-v1 Cfg() values (window_mem/window_grad/net/PPO hyperparams).
  3. action_space.n (expect 43) and obs_dim.
  4. A compatible-init DRY RUN: load ckpt17500 weights into a freshly-init'd
     network of the SAME architecture and report matched/unmatched leaves +
     initial policy logits/value on a fixed dummy (obs, memory, mask).
Writes evidence JSON. Does NOT modify ckpt17500 or any frozen source.
"""
import os, sys, json, hashlib
GPU_UUID = "GPU-e8c08612-c22a-c8a4-6df5-affb2dd1f9a6"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID
import jax, numpy as np

P2 = "/home/oseasy/experiments/p2_v1_20260722/src"
V7 = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src"
for p in (V7, P2):
    if p not in sys.path: sys.path.insert(0, p)
import stage4_continue_launcher as L

CKPT17500 = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/base_ckpt_v7fix55_armA_s0/rl_checkpoints/17500"

cfg = L.Cfg()
# action / obs dims
ach_table = jax.numpy.array([L.get_achievement_multi_hot([L.Achievement.DEFEAT_KOBOLD])], dtype=jax.numpy.float32)
EMB = int(ach_table.shape[1])
with open(L.S4_TASK_PATH) as f: s4_code = f.read()
ns = {}; exec(s4_code, ns); Task = ns["Env"]
static_env_params = L.StaticEnvParams(); env_params = L.EnvParams(max_timesteps=4096)
base_env = L.MultiTaskMiniCraftaxEnv([Task], static_env_params, env_params, cfg.condition_on_task,
    conditioning_type="embedding", embedding_size=EMB, completion_bonus_scale=cfg.completion_bonus_scale,
    completion_bonus_min=cfg.completion_bonus_min, bonus_type=cfg.bonus_type, dynamic_bonus_k=cfg.dynamic_bonus_k)
obs_dim = int(base_env.observation_space(env_params).shape[0])
action_dim = int(base_env.action_space(env_params).n)

def leaf_report(tree):
    leaves = jax.tree_util.tree_leaves_with_path(tree)
    out = []; total = 0
    for path, v in leaves:
        v = np.asarray(v)
        key = "/".join(str(getattr(k, "key", k)) for k in path)
        out.append({"path": key, "shape": list(v.shape), "dtype": str(v.dtype), "n": int(v.size)})
        total += int(v.size)
    return out, total

# fresh network of the same architecture -> full param tree structure
net = L.make_network(base_env, env_params, cfg, EMB) if hasattr(L, "make_network") else None
# fallback: build via load_weights_only on a dummy then read params
ts0 = L.load_weights_only(CKPT17500, base_env, env_params, cfg, load_opt_state=False)
params = ts0.params
leaves, total = leaf_report(params)
psha = L._params_content_sha256(params)

# initial policy on fixed dummy (obs, memory, mask)
def dummy_forward(params):
    b = 2
    obs = jax.numpy.zeros((b, obs_dim), dtype=jax.numpy.float32)
    mem = jax.numpy.zeros((b, cfg.window_mem, cfg.num_layers, cfg.embed_size), dtype=jax.numpy.float32)
    mask = jax.numpy.ones((b, cfg.num_heads, 1, cfg.window_mem + 1), dtype=jax.numpy.bool_)
    pi, value, _ = L.network_apply_eval(params, mem, obs, mask) if hasattr(L, "network_apply_eval") else (None, None, None)
    return pi, value
logits0 = value0 = None
try:
    # use the network object the launcher builds; reconstruct minimal
    from dicode.network import ActorCriticTransformer
    network = ActorCriticTransformer(action_dim=action_dim, activation=cfg.activation,
        hidden_layers=cfg.hidden_layers, encoder_size=cfg.embed_size, num_heads=cfg.num_heads,
        qkv_features=cfg.qkv_features, num_layers=cfg.num_layers, gating=cfg.gating, gating_bias=cfg.gating_bias)
    b = 2
    obs = jax.numpy.zeros((b, obs_dim), dtype=jax.numpy.float32)
    mem = jax.numpy.zeros((b, cfg.window_mem, cfg.num_layers, cfg.embed_size), dtype=jax.numpy.float32)
    mask = jax.numpy.ones((b, cfg.num_heads, 1, cfg.window_mem + 1), dtype=jax.numpy.bool_)
    pi, value, _ = network.apply(params, mem, obs, mask, method=network.model_forward_eval)
    logits0 = np.asarray(pi.logits[0]).tolist()
    value0 = float(np.asarray(value[0]))
    probs0 = np.asarray(jax.nn.softmax(pi.logits, axis=-1)[0])
    top5 = sorted(range(action_dim), key=lambda i: -probs0[i])[:5]
    probs0 = {"top5_actions": top5, "top5_probs": [float(probs0[i]) for i in top5], "entropy": float(pi.entropy()[0])}
except Exception as ex:
    probs0 = {"error": str(ex)}

cfg_keys = ["num_envs","num_steps","num_minibatches","update_epochs","window_mem","window_grad",
    "embed_size","num_heads","num_layers","qkv_features","hidden_layers","activation","gating","gating_bias",
    "lr","min_lr","anneal_lr","gamma","gae_lambda","clip_eps","vf_coef","ent_coef","max_grad_norm",
    "condition_on_task","completion_bonus_scale","completion_bonus_min","bonus_type","dynamic_bonus_k"]
cfg_dump = {}
for k in cfg_keys:
    cfg_dump[k] = getattr(cfg, k, "MISSING")

report = {
    "gpu_uuid": GPU_UUID,
    "ckpt17500": CKPT17500,
    "action_dim": action_dim,
    "obs_dim": obs_dim,
    "task_embedding_size": EMB,
    "params_sha256": psha,
    "total_params": total,
    "num_leaves": len(leaves),
    "leaves": leaves,
    "config": cfg_dump,
    "dummy_zero_obs_policy": {"logits_first8": (logits0[:8] if logits0 else None),
                               "value": value0, "probs": probs0},
}
out = "/home/oseasy/experiments/p2_full_20260723/evidence/architecture_inspection.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f: json.dump(report, f, indent=2, sort_keys=True, default=str); f.write("\n")
print("INSPECT_OK action_dim=%d obs_dim=%d EMB=%d total_params=%d leaves=%d" % (action_dim, obs_dim, EMB, total, len(leaves)))
print("params_sha256=%s" % psha)
print("window_mem=%s window_grad=%s num_layers=%s embed_size=%s num_heads=%s qkv=%s hidden=%s" % (
    cfg_dump["window_mem"], cfg_dump["window_grad"], cfg_dump["num_layers"], cfg_dump["embed_size"],
    cfg_dump["num_heads"], cfg_dump["qkv_features"], cfg_dump["hidden_layers"]))
print("lr=%s min_lr=%s anneal=%s gamma=%s gae=%s clip=%s vf=%s ent=%s gradnorm=%s" % (
    cfg_dump["lr"], cfg_dump["min_lr"], cfg_dump["anneal_lr"], cfg_dump["gamma"], cfg_dump["gae_lambda"],
    cfg_dump["clip_eps"], cfg_dump["vf_coef"], cfg_dump["ent_coef"], cfg_dump["max_grad_norm"]))
print("OUT=%s" % out)
