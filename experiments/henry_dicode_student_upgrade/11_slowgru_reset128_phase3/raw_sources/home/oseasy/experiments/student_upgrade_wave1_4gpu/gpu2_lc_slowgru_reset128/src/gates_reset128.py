#!/usr/bin/env python3
"""LC-SLOWGRU-RESET128 (Phase2 LONG_MEMORY_CAUSAL_CARRY_ABLATION) — network/init engineering gates.
Verifies the Reset128 arm is bit-identical to the Persistent arm EXCEPT carry/reset semantics.

Non-training gates verified here (mapped to the 13-gate spec):
  REQ1  schema identical      : Reset128 init param tree_structure == Persistent ckpt@0 params structure
  REQ2  step0 bit-identical   : Reset128 init _params_sha == Persistent ckpt@0 _params_sha == known init_sha
  REQ3  param count identical : leaf count + total element count == Persistent ckpt@0
  REQ4  within-rollout R/W    : with reset=False the long state accumulates across steps (single rollout)
  REQ7  env no crosstalk      : perturbing env0 long state changes ONLY env0 output
  REQ8  true-done reset       : reset=True clears long state to init
  REQ13 long-module gradient  : long-mem params (incl. residual gate) get finite NON-ZERO grads
Plus sanity: feature-off + init zero-gate output == teacher bit-exact (network truly unchanged).

Trainer-level gates (verified by the training run, not here):
  REQ5  rollout-boundary clear : reset128_gates.boundary_clear_pass (per-rollout longstate hash == init)
  REQ6  GTrXL unchanged        : network-file sha identical + trainer diff contains ONLY the clear block
  REQ9  checkpoint full save   : trainer roundtrip_ok asserts at 0/4096/24576
  REQ10 exact resume bit-ident : smoke+resume (continuous vs resumed params_sha equal)
  REQ11 no NaN/Inf             : trainer _finite asserts per chunk
  REQ12 entropy no collapse    : per_update.jsonl entropy range

GPU2 only; deterministic ops; read-only w.r.t. training.
"""
import os
GPU_UUID = "GPU-8df11537-ab79-722d-606f-411966196c4c"   # GPU2
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
import sys, json, hashlib, time, pickle
import numpy as np, jax, jax.numpy as jnp

ARM = "LC_SLOWGRU_RESET128"
SRC = "/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu2_lc_slowgru_reset128/src"
W = "/home/oseasy/experiments/student_upgrade_wave1_4gpu"
V7_SRC = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src"
V7 = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB"
for p in (SRC, V7_SRC, V7):
    if p not in sys.path: sys.path.insert(0, p)

from craftax.craftax.constants import Achievement
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from dicode.network import ActorCriticTransformer
from dicode.task_utils import get_achievement_multi_hot
from dicode.utils.general.train_state_utils import load_weights_only
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv
from slowgru_network import ActorCriticSlowGRU, init_longstate

TEACHER_CKPT = ("/home/oseasy/incoming/henry_work_20260721T105300/extracted/"
                "Henry_work/base_ckpt_v7fix55_armA_s0/rl_checkpoints/17500")
S4_TASK_PATH = "/home/oseasy/experiments/p2_v1_20260722/evidence/s4_task_code.py"
PERSISTENT_CKPT0 = f"{W}/gpu2_lc_slowgru/train_24576/ckpt/0/full_state.pkl"
KNOWN_PERSISTENT_INIT_SHA16 = "5ae94ed0257f50fa"
NETWORK_MODULE = "slowgru_network.py"

E, WM, NL, NH, DIM = 4, 128, 2, 8, 256
_cfg = dict(activation="relu", embed_size=256, hidden_layers=256, num_heads=8, qkv_features=256,
            num_layers=2, gating=True, gating_bias=2.0, window_mem=128, window_grad=64,
            condition_on_task=True, optimistic_reset_ratio=16, mode="score", bonus_type="none",
            dynamic_bonus_k=0.0, completion_bonus_scale=0.0, completion_bonus_min=0.0,
            value_target_clip_min=-50.0, value_target_clip_max=300.0, lr=2e-5, num_envs=16,
            num_steps=128, update_epochs=1, num_minibatches=2, gamma=0.999, gae_lambda=0.8,
            clip_eps=0.2, ent_coef=0.002, vf_coef=0.5, max_grad_norm=1.0, anneal_lr=False)
cfg = type("C", (), _cfg)(); cfg.get = lambda k, d=None: getattr(cfg, k, d); cfg.training = cfg


def _params_sha(params):
    h = hashlib.sha256()
    for v in jax.tree_util.tree_leaves(params):
        h.update(np.ascontiguousarray(np.asarray(v)).tobytes())
    return h.hexdigest()


def _n_elements(params):
    return int(sum(int(np.asarray(v).size) for v in jax.tree_util.tree_leaves(params)))


ach = jnp.array([get_achievement_multi_hot([Achievement.DEFEAT_KOBOLD])], dtype=jnp.float32)
EMB = int(ach.shape[1])
ns = {}; exec(open(S4_TASK_PATH).read(), ns); Task = ns["Env"]
epc = EnvParams(max_timesteps=4096)
base = MultiTaskMiniCraftaxEnv([Task], StaticEnvParams(), epc, True,
    conditioning_type="embedding", embedding_size=EMB)
OBS_DIM = int(base.observation_space(epc).shape[0]); ACTION_DIM = int(base.action_space(epc).n)
print(f"[{ARM}] gates: obs_dim={OBS_DIM} action_dim={ACTION_DIM} devices={[str(d) for d in jax.devices()]}")


class Cfg: pass
for k, v in _cfg.items(): setattr(Cfg, k, v)
Cfg.get = lambda k, d=None: getattr(Cfg, k, d); Cfg.training = Cfg
teacher_vars = load_weights_only(TEACHER_CKPT, base, epc, Cfg, load_opt_state=False).params
teacher_inner = teacher_vars["params"]

net_on = ActorCriticSlowGRU(action_dim=ACTION_DIM, activation="relu", hidden_layers=256,
    encoder_size=256, num_heads=NH, qkv_features=256, num_layers=NL, gating=True, gating_bias=2.0,
    use_longmem=True)
net_off = ActorCriticSlowGRU(action_dim=ACTION_DIM, activation="relu", hidden_layers=256,
    encoder_size=256, num_heads=NH, qkv_features=256, num_layers=NL, gating=True, gating_bias=2.0,
    use_longmem=False)
teacher_net = ActorCriticTransformer(action_dim=ACTION_DIM, activation="relu", hidden_layers=256,
    encoder_size=256, num_heads=NH, qkv_features=256, num_layers=NL, gating=True, gating_bias=2.0)

# Reset128 init = teacher (inherited, inner) + fresh long-mem (zero residual gate); re-wrap for apply.
# IDENTICAL init-merge to the frozen Persistent trainer (same network file, same seed=0).
_dummy_mem = jnp.zeros((2, WM, NL, DIM)); _dummy_obs = jnp.zeros((2, OBS_DIM))
_dummy_mask = jnp.zeros((2, NH, 1, WM + 1), jnp.bool_); _dummy_ls = init_longstate(2)
_dummy_reset = jnp.zeros((2,), jnp.bool_)
full_inner = net_on.init(jax.random.PRNGKey(0), _dummy_mem, _dummy_obs, _dummy_mask, _dummy_ls,
                         _dummy_reset, method=net_on.forward_eval)["params"]
missing = [k for k in teacher_inner if k not in full_inner]
assert not missing, f"teacher keys missing: {missing}"
_init_inner = dict(full_inner)
for k in teacher_inner: _init_inner[k] = teacher_inner[k]
init_params = {"params": _init_inner}

# ---- load the FROZEN Persistent arm's step-0 checkpoint params (the reference) ----
with open(PERSISTENT_CKPT0, "rb") as f: rd = pickle.load(f)
p_leaves, p_treedef = rd["params"]
persistent_params = jax.tree_util.tree_unflatten(p_treedef, [jnp.asarray(l) for l in p_leaves])

key = jax.random.PRNGKey(123)
k1, k2, k3 = jax.random.split(key, 3)
obs = jax.random.normal(k1, (E, OBS_DIM))
memories = jax.random.normal(k2, (E, WM, NL, DIM)) * 0.1
mask = jnp.zeros((E, NH, 1, WM + 1), jnp.bool_).at[:, :, :, -1].set(True)
reset0 = jnp.zeros((E,), jnp.bool_)


def eq(a, b): return np.array_equal(np.asarray(a), np.asarray(b))


results = {}

# ---------- REQ1 schema identical (Reset128 init vs Persistent ckpt@0) ----------
struct_r = jax.tree_util.tree_structure(init_params)
struct_p = jax.tree_util.tree_structure(persistent_params)
shapes_r = [np.asarray(v).shape for v in jax.tree_util.tree_leaves(init_params)]
shapes_p = [np.asarray(v).shape for v in jax.tree_util.tree_leaves(persistent_params)]
req1 = bool(struct_r == struct_p and shapes_r == shapes_p)
results["req1_schema_identical"] = dict(pass_=req1, n_leaves_reset128=len(shapes_r),
    n_leaves_persistent=len(shapes_p), tree_struct_equal=bool(struct_r == struct_p),
    shapes_equal=bool(shapes_r == shapes_p))
print(f"[req1] schema_identical={req1} (leaves {len(shapes_r)} vs {len(shapes_p)})")

# ---------- REQ2 step0 params bit-identical ----------
sha_r = _params_sha(init_params); sha_p = _params_sha(persistent_params)
req2 = bool(sha_r == sha_p and sha_r[:16] == KNOWN_PERSISTENT_INIT_SHA16)
results["req2_step0_bit_identical"] = dict(pass_=req2, reset128_init_sha256=sha_r,
    persistent_ckpt0_sha256=sha_p, known_persistent_init_sha16=KNOWN_PERSISTENT_INIT_SHA16,
    match_persistent=bool(sha_r == sha_p), match_known=bool(sha_r[:16] == KNOWN_PERSISTENT_INIT_SHA16))
print(f"[req2] step0_bit_identical={req2} reset128_sha={sha_r[:16]} persistent_sha={sha_p[:16]} "
      f"known={KNOWN_PERSISTENT_INIT_SHA16}")

# ---------- REQ3 param count identical ----------
ne_r = _n_elements(init_params); ne_p = _n_elements(persistent_params)
req3 = bool(len(shapes_r) == len(shapes_p) and ne_r == ne_p)
results["req3_param_count_identical"] = dict(pass_=req3, n_elements_reset128=ne_r,
    n_elements_persistent=ne_p, n_leaves_reset128=len(shapes_r), n_leaves_persistent=len(shapes_p))
print(f"[req3] param_count_identical={req3} (elements {ne_r} vs {ne_p})")

# ---------- sanity: feature-off + init zero-gate == teacher bit-exact (network unchanged) ----------
pi_t, v_t, mem_t = teacher_net.apply(teacher_vars, memories, obs, mask, method=teacher_net.model_forward_eval)
pi_off, v_off, mem_off, _ = net_off.apply(init_params, memories, obs, mask, init_longstate(E), reset0,
                                          method=net_off.forward_eval)
pi_on, v_on, mem_on, _ = net_on.apply(init_params, memories, obs, mask, init_longstate(E), reset0,
                                      method=net_on.forward_eval)
g_off = bool(eq(pi_off.logits, pi_t.logits) and eq(v_off, v_t) and eq(mem_off, mem_t))
g_init = bool(eq(pi_on.logits, pi_t.logits) and eq(v_on, v_t) and eq(mem_on, mem_t))
results["sanity_feature_off_eq_teacher"] = dict(pass_=g_off,
    max_logit_diff=float(np.abs(np.asarray(pi_off.logits) - np.asarray(pi_t.logits)).max()))
results["sanity_init_zero_gate_eq_teacher"] = dict(pass_=g_init,
    max_logit_diff=float(np.abs(np.asarray(pi_on.logits) - np.asarray(pi_t.logits)).max()))
print(f"[sanity] feature_off_eq_teacher={g_off} init_zero_gate_eq_teacher={g_init}")

# ---------- REQ7 env no crosstalk ----------
ls_a = init_longstate(E)
ls_b = init_longstate(E)
ls_b_filled = jax.tree_util.tree_map(
    lambda x: x.at[0].set(jax.random.normal(k3, x.shape[1:]) * 0.5) if x.ndim >= 1 else x, ls_b)
if "valid" in ls_b_filled:
    ls_b_filled["valid"] = ls_b_filled["valid"].at[0].set(True)
pi_a, v_a, _, _ = net_on.apply(init_params, memories, obs, mask, ls_a, reset0, method=net_on.forward_eval)
pi_b, v_b, _, _ = net_on.apply(init_params, memories, obs, mask, ls_b_filled, reset0, method=net_on.forward_eval)
req7 = bool(eq(pi_a.logits[1:], pi_b.logits[1:]) and eq(v_a[1:], v_b[1:]))
results["req7_env_no_crosstalk"] = dict(pass_=req7,
    env0_changed=bool(not eq(pi_a.logits[0], pi_b.logits[0])))
print(f"[req7] env_no_crosstalk={req7}")

# ---------- REQ4 within-rollout read/write (no reset -> long state accumulates) ----------
ls = init_longstate(E)
for _ in range(5):
    _, _, _, ls = net_on.apply(init_params, memories, obs, mask, ls, reset0, method=net_on.forward_eval)
req4 = bool(not all(eq(a, b) for a, b in
              zip(jax.tree_util.tree_leaves(ls), jax.tree_util.tree_leaves(init_longstate(E)))))
results["req4_within_rollout_readwrite"] = dict(pass_=req4)
print(f"[req4] within_rollout_readwrite={req4}")

# ---------- REQ8 true-done reset clears long state ----------
reset_all = jnp.ones((E,), jnp.bool_)
_, _, _, ls_after_reset = net_on.apply(init_params, memories, obs, mask, ls, reset_all, method=net_on.forward_eval)
_, _, _, ls_from_empty = net_on.apply(init_params, memories, obs, mask, init_longstate(E), reset_all,
                                      method=net_on.forward_eval)
req8 = bool(all(eq(a, b) for a, b in
              zip(jax.tree_util.tree_leaves(ls_after_reset), jax.tree_util.tree_leaves(ls_from_empty))))
results["req8_true_done_reset"] = dict(pass_=req8)
print(f"[req8] true_done_reset={req8}")

# ---------- REQ13 long-module finite non-zero gradient (populated memory) ----------
ls = init_longstate(E); mem_roll = memories
for _ in range(40):
    _, _, mout, ls = net_on.apply(init_params, mem_roll, obs, mask, ls, reset0, method=net_on.forward_eval)
    mem_roll = jnp.roll(mem_roll, -1, axis=1).at[:, -1].set(mout)

def long_loss(params):
    pi, _, _, _ = net_on.apply(params, mem_roll, obs, mask, ls, reset0, method=net_on.forward_eval)
    return jnp.mean(pi.logits)

grads = jax.grad(long_loss)(init_params)["params"]
long_keys = [k for k in _init_inner if k not in teacher_inner]
gnorm = {}
for k in long_keys:
    gleaves = jax.tree_util.tree_leaves(grads[k])
    n = float(np.sqrt(sum(float(np.sum(np.asarray(g) ** 2)) for g in gleaves)))
    finite = bool(all(np.all(np.isfinite(np.asarray(g))) for g in gleaves))
    gnorm[k] = dict(norm=n, finite=finite)
all_finite = all(v["finite"] for v in gnorm.values())
any_nonzero = any(v["norm"] > 0 for v in gnorm.values())
res_key = [k for k in long_keys if "to_actor" in k][0]
res_ok = gnorm[res_key]["norm"] > 0 and gnorm[res_key]["finite"]
req13 = bool(all_finite and any_nonzero and res_ok)
results["req13_long_module_grad"] = dict(pass_=req13, all_finite=all_finite, any_nonzero=any_nonzero,
    residual_gate_key=res_key, residual_gate_norm=gnorm[res_key]["norm"],
    residual_gate_finite=gnorm[res_key]["finite"],
    per_module_norm={k: round(v["norm"], 8) for k, v in gnorm.items()})
print(f"[req13] long_module_grad finite={all_finite} nonzero={any_nonzero} "
      f"residual_gate({res_key})_norm={gnorm[res_key]['norm']:.3e}")

ALL = [results["req1_schema_identical"]["pass_"], results["req2_step0_bit_identical"]["pass_"],
       results["req3_param_count_identical"]["pass_"], results["sanity_feature_off_eq_teacher"]["pass_"],
       results["sanity_init_zero_gate_eq_teacher"]["pass_"], results["req7_env_no_crosstalk"]["pass_"],
       results["req4_within_rollout_readwrite"]["pass_"], results["req8_true_done_reset"]["pass_"],
       results["req13_long_module_grad"]["pass_"]]
ok = bool(all(ALL))
out = dict(label=f"{ARM}_RESET128_GATES", network="LC-SLOWGRU-RESET128", carry_mode="RESET128",
    gates=results, NON_TRAINING_GATES_PASS=ok,
    trainer_level_gates="req5(boundary_clear)/req6(GTrXL_unchanged)/req9(ckpt)/req10(exact_resume)/req11(no_nan)/req12(entropy) verified by training run",
    persistent_ckpt0=PERSISTENT_CKPT0,
    timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
json.dump(out, open(os.path.join(SRC, f"{ARM}_gates.json"), "w"), indent=2, default=str)
print(f"\n[{ARM}] RESET128_GATES_{'PASS' if ok else 'FAIL'} (req1/2/3/4/7/8/13 + sanity)")
