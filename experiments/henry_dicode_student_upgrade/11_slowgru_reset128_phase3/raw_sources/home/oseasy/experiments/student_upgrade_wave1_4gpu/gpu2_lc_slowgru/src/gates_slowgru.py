#!/usr/bin/env python3
"""LC-EVENTMEM32-PPO — network-level engineering gates (GPU2, no training). Covers:
  (1) FEATURE-OFF bit-identical to teacher: use_longmem=False output == teacher ActorCriticTransformer
      (inherited params) AND use_longmem=True AT INIT (zero residual gate) == teacher (logits/value/mem
      bit-exact).
  (2) ENV ISOLATION: perturbing one env's long-state changes ONLY that env's output.
  (3) ROLLOUT CONTINUITY: with reset=False the long-state accumulates across steps (not reset).
  (4) TRUE-DONE RESET: reset=True clears the long-state to init.
  (8) LONG-PATH FINITE NON-ZERO GRADIENT: after populating memory, grads of the long-mem params
      (incl. the residual gate) are finite and non-zero.
Gates 5 (roundtrip), 6 (exact resume), 7 (4096 smoke), 9 (no NaN/entropy collapse) are verified by the
trainer smoke+resume run. Gate 10 (clear-state action KL>0 post-training) is in the eval/ablation.
Read-only w.r.t. training; GPU2 only; deterministic ops.

NOTE on param convention: this repo stores/applies params in the WRAPPED flax form {'params': {...}}
(network.init returns that; TrainState.create(params=...); network.apply(train_state.params)). So the
init-merge is done at the INNER level and re-wrapped before any .apply call.
"""
import os
GPU_UUID = "GPU-8df11537-ab79-722d-606f-411966196c4c"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
import sys, json, hashlib, time
import numpy as np, jax, jax.numpy as jnp

ARM = "LC_SLOWGRU"
SRC = "/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu2_lc_slowgru/src"
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

E, WM, NL, NH, DIM = 4, 128, 2, 8, 256
_cfg = dict(activation="relu", embed_size=256, hidden_layers=256, num_heads=8, qkv_features=256,
            num_layers=2, gating=True, gating_bias=2.0, window_mem=128, window_grad=64,
            condition_on_task=True, optimistic_reset_ratio=16, mode="score", bonus_type="none",
            dynamic_bonus_k=0.0, completion_bonus_scale=0.0, completion_bonus_min=0.0,
            value_target_clip_min=-50.0, value_target_clip_max=300.0, lr=2e-5, num_envs=16,
            num_steps=128, update_epochs=1, num_minibatches=2, gamma=0.999, gae_lambda=0.8,
            clip_eps=0.2, ent_coef=0.002, vf_coef=0.5, max_grad_norm=1.0, anneal_lr=False)
cfg = type("C", (), _cfg)(); cfg.get = lambda k, d=None: getattr(cfg, k, d); cfg.training = cfg

ach = jnp.array([get_achievement_multi_hot([Achievement.DEFEAT_KOBOLD])], dtype=jnp.float32)
EMB = int(ach.shape[1])
ns = {}; exec(open(S4_TASK_PATH).read(), ns); Task = ns["Env"]
epc = EnvParams(max_timesteps=4096)
base = MultiTaskMiniCraftaxEnv([Task], StaticEnvParams(), epc, True,
    conditioning_type="embedding", embedding_size=EMB)
OBS_DIM = int(base.observation_space(epc).shape[0]); ACTION_DIM = int(base.action_space(epc).n)
print(f"[{ARM}] gates: obs_dim={OBS_DIM} action_dim={ACTION_DIM} devices={[str(d) for d in jax.devices()]}")


class Cfg:
    pass
for k, v in _cfg.items(): setattr(Cfg, k, v)
Cfg.get = lambda k, d=None: getattr(Cfg, k, d)
Cfg.training = Cfg
teacher_vars = load_weights_only(TEACHER_CKPT, base, epc, Cfg, load_opt_state=False).params
teacher_inner = teacher_vars["params"]    # repo convention: wrapped {'params': {...}} -> inner flat dict

net_on = ActorCriticSlowGRU(action_dim=ACTION_DIM, activation="relu", hidden_layers=256,
    encoder_size=256, num_heads=NH, qkv_features=256, num_layers=NL, gating=True, gating_bias=2.0,
    use_longmem=True)
net_off = ActorCriticSlowGRU(action_dim=ACTION_DIM, activation="relu", hidden_layers=256,
    encoder_size=256, num_heads=NH, qkv_features=256, num_layers=NL, gating=True, gating_bias=2.0,
    use_longmem=False)
teacher_net = ActorCriticTransformer(action_dim=ACTION_DIM, activation="relu", hidden_layers=256,
    encoder_size=256, num_heads=NH, qkv_features=256, num_layers=NL, gating=True, gating_bias=2.0)

# init params = teacher (inherited, inner) + fresh long-mem (zero residual gate); re-wrap for apply
_dummy_mem = jnp.zeros((2, WM, NL, DIM)); _dummy_obs = jnp.zeros((2, OBS_DIM))
_dummy_mask = jnp.zeros((2, NH, 1, WM + 1), jnp.bool_); _dummy_ls = init_longstate(2)
_dummy_reset = jnp.zeros((2,), jnp.bool_)
full_inner = net_on.init(jax.random.PRNGKey(0), _dummy_mem, _dummy_obs, _dummy_mask, _dummy_ls,
                         _dummy_reset, method=net_on.forward_eval)["params"]
missing = [k for k in teacher_inner if k not in full_inner]
assert not missing, f"teacher keys missing in slow-gru net: {missing}"
_init_inner = dict(full_inner)
for k in teacher_inner: _init_inner[k] = teacher_inner[k]
init_params = {"params": _init_inner}      # wrapped -> matches network.apply(train_state.params)

key = jax.random.PRNGKey(123)
k1, k2, k3 = jax.random.split(key, 3)
obs = jax.random.normal(k1, (E, OBS_DIM))
memories = jax.random.normal(k2, (E, WM, NL, DIM)) * 0.1
mask = jnp.zeros((E, NH, 1, WM + 1), jnp.bool_).at[:, :, :, -1].set(True)
reset0 = jnp.zeros((E,), jnp.bool_)


def eq(a, b):
    return np.array_equal(np.asarray(a), np.asarray(b))


results = {}

# ---------- Gate 1: feature-off + init bit-identical to teacher ----------
pi_t, v_t, mem_t = teacher_net.apply(teacher_vars, memories, obs, mask, method=teacher_net.model_forward_eval)
pi_off, v_off, mem_off, _ = net_off.apply(init_params, memories, obs, mask, init_longstate(E), reset0,
                                          method=net_off.forward_eval)
pi_on, v_on, mem_on, _ = net_on.apply(init_params, memories, obs, mask, init_longstate(E), reset0,
                                      method=net_on.forward_eval)
g1_off = bool(eq(pi_off.logits, pi_t.logits) and eq(v_off, v_t) and eq(mem_off, mem_t))
g1_init = bool(eq(pi_on.logits, pi_t.logits) and eq(v_on, v_t) and eq(mem_on, mem_t))
results["gate1_feature_off"] = dict(pass_=g1_off,
    max_logit_diff_off=float(np.abs(np.asarray(pi_off.logits) - np.asarray(pi_t.logits)).max()))
results["gate1_init_zero_gate"] = dict(pass_=g1_init,
    max_logit_diff_init=float(np.abs(np.asarray(pi_on.logits) - np.asarray(pi_t.logits)).max()))
print(f"[gate1] feature_off_bit_identical={g1_off} init_zero_gate_bit_identical={g1_init}")

# ---------- Gate 2: env isolation ----------
ls_a = init_longstate(E)
ls_b = init_longstate(E)
# populate ls_b env 0 only with distinct content
ls_b_filled = jax.tree_util.tree_map(
    lambda x: x.at[0].set(jax.random.normal(k3, x.shape[1:]) * 0.5) if x.ndim >= 1 else x, ls_b)
# make env0 valid (slowgru has no valid leaf; guard skips)
if "valid" in ls_b_filled:
    ls_b_filled["valid"] = ls_b_filled["valid"].at[0].set(True)
pi_a, v_a, _, _ = net_on.apply(init_params, memories, obs, mask, ls_a, reset0, method=net_on.forward_eval)
pi_b, v_b, _, _ = net_on.apply(init_params, memories, obs, mask, ls_b_filled, reset0, method=net_on.forward_eval)
# envs 1..E-1 must be bit-identical between the two runs (only env 0 may differ)
iso_logits = eq(pi_a.logits[1:], pi_b.logits[1:])
iso_value = eq(v_a[1:], v_b[1:])
g2 = bool(iso_logits and iso_value)
results["gate2_env_isolation"] = dict(pass_=g2, env0_changed=bool(not eq(pi_a.logits[0], pi_b.logits[0])))
print(f"[gate2] env_isolation={g2} (env0_changed={not eq(pi_a.logits[0], pi_b.logits[0])})")

# ---------- Gate 3: rollout continuity (no reset -> accumulates) ----------
ls = init_longstate(E)
for _ in range(5):
    _, _, _, ls = net_on.apply(init_params, memories, obs, mask, ls, reset0, method=net_on.forward_eval)
changed = not all(eq(a, b) for a, b in
                  zip(jax.tree_util.tree_leaves(ls), jax.tree_util.tree_leaves(init_longstate(E))))
g3 = bool(changed)
results["gate3_rollout_continuity"] = dict(pass_=g3)
print(f"[gate3] rollout_continuity_accumulates={g3}")

# ---------- Gate 4: true-done reset clears long-state ----------
ls_pop = ls  # populated from gate 3
reset_all = jnp.ones((E,), jnp.bool_)
_, _, _, ls_after_reset = net_on.apply(init_params, memories, obs, mask, ls_pop, reset_all,
                                       method=net_on.forward_eval)
# Strong check: running with reset on an empty vs populated prior gives the SAME result (prior wiped).
_, _, _, ls_from_empty = net_on.apply(init_params, memories, obs, mask, init_longstate(E), reset_all,
                                      method=net_on.forward_eval)
g4 = bool(all(eq(a, b) for a, b in
              zip(jax.tree_util.tree_leaves(ls_after_reset), jax.tree_util.tree_leaves(ls_from_empty))))
results["gate4_true_done_reset"] = dict(pass_=g4)
print(f"[gate4] true_done_reset_clears_prior={g4}")

# ---------- Gate 8: long-path finite non-zero gradient (populated memory) ----------
# populate memory by rolling 40 steps (no reset) so ctx != 0
ls = init_longstate(E)
mem_roll = memories
for _ in range(40):
    _, _, mout, ls = net_on.apply(init_params, mem_roll, obs, mask, ls, reset0, method=net_on.forward_eval)
    mem_roll = jnp.roll(mem_roll, -1, axis=1).at[:, -1].set(mout)

def long_loss(params):
    pi, _, _, _ = net_on.apply(params, mem_roll, obs, mask, ls, reset0, method=net_on.forward_eval)
    return jnp.mean(pi.logits)

grads = jax.grad(long_loss)(init_params)              # grads wrapped {'params': {...}}
grads_inner = grads["params"]
long_keys = [k for k in _init_inner if k not in teacher_inner]
gnorm = {}
for k in long_keys:
    leaves = jax.tree_util.tree_leaves(_init_inner[k])
    gleaves = jax.tree_util.tree_leaves(grads_inner[k])
    n = float(np.sqrt(sum(float(np.sum(np.asarray(g) ** 2)) for g in gleaves)))
    finite = bool(all(np.all(np.isfinite(np.asarray(g))) for g in gleaves))
    gnorm[k] = dict(norm=n, finite=finite)
all_finite = all(v["finite"] for v in gnorm.values())
any_nonzero = any(v["norm"] > 0 for v in gnorm.values())
# the residual gate specifically must have non-zero finite grad
res_key = [k for k in long_keys if "to_actor" in k][0]
res_ok = gnorm[res_key]["norm"] > 0 and gnorm[res_key]["finite"]
g8 = bool(all_finite and any_nonzero and res_ok)
results["gate8_long_grad"] = dict(pass_=g8, all_finite=all_finite, any_nonzero=any_nonzero,
    residual_gate_norm=gnorm[res_key]["norm"], residual_gate_finite=gnorm[res_key]["finite"],
    per_module_norm={k: round(v["norm"], 8) for k, v in gnorm.items()})
print(f"[gate8] long_grad finite={all_finite} nonzero={any_nonzero} residual_gate_norm={gnorm[res_key]['norm']:.3e}")

ALL = [results["gate1_feature_off"]["pass_"], results["gate1_init_zero_gate"]["pass_"],
       results["gate2_env_isolation"]["pass_"], results["gate3_rollout_continuity"]["pass_"],
       results["gate4_true_done_reset"]["pass_"], results["gate8_long_grad"]["pass_"]]
ok = bool(all(ALL))
out = dict(label=f"{ARM}_GATES", network=ARM, gates=results, ALL_PASS=ok,
           timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
json.dump(out, open(os.path.join(SRC, f"{ARM}_gates.json"), "w"), indent=2, default=str)
print(f"\n[{ARM}] GATES_{'PASS' if ok else 'FAIL'} (1,2,3,4,8; 5/6/7/9 via smoke+resume; 10 via post-train eval)")
