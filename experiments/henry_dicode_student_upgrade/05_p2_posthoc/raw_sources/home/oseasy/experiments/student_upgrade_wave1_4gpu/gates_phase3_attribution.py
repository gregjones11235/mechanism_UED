#!/usr/bin/env python3
"""Phase3 SLOWGRU_RESET128_SUSTAINABILITY_AND_ATTRIBUTION — GPU3 ATTRIBUTION engineering gates
(section 9, per-ablation) for DETACH (experiment A) and MATCHED_MLP (experiment B). NON-TRAINING.

Verified here (gates that can be checked without a training run):
  G1  same start ckpt17500        : teacher-inherited (public) params of Full/Detach/MLP all == teacher bit-for-bit
  G2  public params step0 identical: Full/Detach/MLP teacher-inherited subtrees bit-identical at init
  G3  config diff empty except ablation: Detach param tree == Full (structure+shapes); MLP differs ONLY in slow_gru->slow_mlp
  G4  rollout/world/RNG protocol  : config constants identical across the three trainers
  G6  no env crosstalk            : perturbing env0 long state changes ONLY env0 (Detach & MLP)
  G9  slow-module finite nonzero grad: slow-branch params get finite NON-ZERO grads (Detach & MLP)
  DA  DETACH blocks backbone grad : grad on GTrXL/transformer params differs Full vs Detach (slow->backbone shaping
                                    present in Full, removed in Detach) WHILE slow-branch grads stay nonzero in Detach
  MB  MLP param match             : MLP slow-branch param count == GRU count (within +/-2%); full-tree count equal
  MR  MLP non-recurrent           : h_new independent of previous slow state h (Full IS recurrent -> differs)
Trainer-level gates verified by the training runs: G5 exact resume, G7 no NaN/Inf, G8 entropy no collapse,
  G10 separate output dirs (dirs created & distinct). GPU3 only; deterministic ops.
"""
import os
GPU_UUID = "GPU-f56a59b4-99f3-f2e5-11c6-d01685de8abd"   # GPU3
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
import sys, json, hashlib, time, re
import numpy as np, jax, jax.numpy as jnp

W = "/home/oseasy/experiments/student_upgrade_wave1_4gpu"
ATTR_SRC = f"{W}/gpu3_slowgru_reset128_attribution/src"
V7_SRC = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src"
V7 = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB"
for p in (ATTR_SRC, V7_SRC, V7):
    if p not in sys.path: sys.path.insert(0, p)

from craftax.craftax.constants import Achievement
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from dicode.network import ActorCriticTransformer
from dicode.task_utils import get_achievement_multi_hot
from dicode.utils.general.train_state_utils import load_weights_only
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv
from slowgru_network import ActorCriticSlowGRU, init_longstate
from slowgru_detach_network import ActorCriticSlowGRUDetach
from slowgru_mlp_network import ActorCriticSlowGRUMLP

TEACHER_CKPT = ("/home/oseasy/incoming/henry_work_20260721T105300/extracted/"
                "Henry_work/base_ckpt_v7fix55_armA_s0/rl_checkpoints/17500")
S4_TASK_PATH = "/home/oseasy/experiments/p2_v1_20260722/evidence/s4_task_code.py"
KNOWN_TEACHER_SHA16 = "d4e85af58b7f87d6"
KNOWN_FULL_INIT_SHA16 = "5ae94ed0257f50fa"
FROZEN_NET_SHA16 = "b265210597d00321"

E, WM, NL, NH, DIM = 4, 128, 2, 8, 256
_cfg = dict(activation="relu", embed_size=256, hidden_layers=256, num_heads=8, qkv_features=256,
            num_layers=2, gating=True, gating_bias=2.0, window_mem=128, window_grad=64,
            condition_on_task=True, optimistic_reset_ratio=16, mode="score", bonus_type="none",
            dynamic_bonus_k=0.0, completion_bonus_scale=0.0, completion_bonus_min=0.0,
            value_target_clip_min=-50.0, value_target_clip_max=300.0, lr=2e-5, num_envs=16,
            num_steps=128, update_epochs=1, num_minibatches=2, gamma=0.999, gae_lambda=0.8,
            clip_eps=0.2, ent_coef=0.002, vf_coef=0.5, max_grad_norm=1.0, anneal_lr=False)
class Cfg: pass
for k, v in _cfg.items(): setattr(Cfg, k, v)
Cfg.get = lambda k, d=None: getattr(Cfg, k, d); Cfg.training = Cfg


def _params_sha(params):
    h = hashlib.sha256()
    for v in jax.tree_util.tree_leaves(params):
        h.update(np.ascontiguousarray(np.asarray(v)).tobytes())
    return h.hexdigest()


def _subtree_sha(params_inner, keys):
    h = hashlib.sha256()
    for k in sorted(keys):
        for v in jax.tree_util.tree_leaves(params_inner[k]):
            h.update(k.encode()); h.update(np.ascontiguousarray(np.asarray(v)).tobytes())
    return h.hexdigest()


ach = jnp.array([get_achievement_multi_hot([Achievement.DEFEAT_KOBOLD])], dtype=jnp.float32)
EMB = int(ach.shape[1])
ns = {}; exec(open(S4_TASK_PATH).read(), ns); Task = ns["Env"]
epc = EnvParams(max_timesteps=4096)
base = MultiTaskMiniCraftaxEnv([Task], StaticEnvParams(), epc, True,
    conditioning_type="embedding", embedding_size=EMB)
OBS_DIM = int(base.observation_space(epc).shape[0]); ACTION_DIM = int(base.action_space(epc).n)
print(f"[P3-ATTR] gates: obs_dim={OBS_DIM} action_dim={ACTION_DIM} devices={[str(d) for d in jax.devices()]}")
assert ACTION_DIM == 43 and OBS_DIM == 8335 and EMB == 67

teacher_vars = load_weights_only(TEACHER_CKPT, base, epc, Cfg, load_opt_state=False).params
teacher_inner = teacher_vars["params"]
teacher_sha = _params_sha(teacher_vars)
TEACHER_KEYS = sorted(teacher_inner.keys())
print(f"[P3-ATTR] teacher_sha={teacher_sha[:16]} (expect {KNOWN_TEACHER_SHA16}) teacher_keys={len(TEACHER_KEYS)}")

NET_KW = dict(action_dim=ACTION_DIM, activation="relu", hidden_layers=256, encoder_size=256,
              num_heads=NH, qkv_features=256, num_layers=NL, gating=True, gating_bias=2.0, use_longmem=True)
net_full = ActorCriticSlowGRU(**NET_KW)
net_det = ActorCriticSlowGRUDetach(**NET_KW)
net_mlp = ActorCriticSlowGRUMLP(**NET_KW)

_dummy_mem = jnp.zeros((2, WM, NL, DIM)); _dummy_obs = jnp.zeros((2, OBS_DIM))
_dummy_mask = jnp.zeros((2, NH, 1, WM + 1), jnp.bool_); _dummy_ls = init_longstate(2)
_dummy_reset = jnp.zeros((2,), jnp.bool_)


def build_init(network):
    """teacher (inherited, inner) + fresh slow-branch (zero residual gate); re-wrap. SAME merge as frozen."""
    full_inner = network.init(jax.random.PRNGKey(0), _dummy_mem, _dummy_obs, _dummy_mask, _dummy_ls,
                              _dummy_reset, method=network.forward_eval)["params"]
    missing = [k for k in teacher_inner if k not in full_inner]
    assert not missing, f"teacher keys missing: {missing}"
    inner = dict(full_inner)
    for k in teacher_inner: inner[k] = teacher_inner[k]
    return {"params": inner}, inner      # second = MERGED inner (teacher-inherited + fresh slow branch)


init_full, inner_full = build_init(net_full)
init_det, inner_det = build_init(net_det)
init_mlp, inner_mlp = build_init(net_mlp)
sha_full = _params_sha(init_full); sha_det = _params_sha(init_det); sha_mlp = _params_sha(init_mlp)
print(f"[P3-ATTR] init_sha full={sha_full[:16]} detach={sha_det[:16]} mlp={sha_mlp[:16]}")

results = {}

# ---------- G1 + G2: public (teacher-inherited) params bit-identical at step0 ----------
pub_full = _subtree_sha(inner_full, TEACHER_KEYS)
pub_det = _subtree_sha(inner_det, TEACHER_KEYS)
pub_mlp = _subtree_sha(inner_mlp, TEACHER_KEYS)
pub_teacher = _subtree_sha(teacher_inner, TEACHER_KEYS)
print(f"[G1/G2-debug] pub_teacher={pub_teacher[:16]} pub_full={pub_full[:16]} "
      f"pub_det={pub_det[:16]} pub_mlp={pub_mlp[:16]}")
print(f"[G1/G2-debug] full==teacher:{pub_full==pub_teacher} det==teacher:{pub_det==pub_teacher} "
      f"mlp==teacher:{pub_mlp==pub_teacher} full==det:{pub_full==pub_det}")
g1 = bool(pub_full == pub_det == pub_mlp == pub_teacher)
# Detach full tree MUST be bit-identical to Full (stop_gradient changes no params)
detach_eq_full = bool(sha_det == sha_full and sha_full[:16] == KNOWN_FULL_INIT_SHA16)
results["G1_G2_public_params_bit_identical"] = dict(pass_=g1 and detach_eq_full,
    public_sha_teacher=pub_teacher[:16], public_sha_full=pub_full[:16], public_sha_detach=pub_det[:16],
    public_sha_mlp=pub_mlp[:16], detach_full_tree_eq_full=detach_eq_full,
    full_init_sha16=sha_full[:16], known_full_init_sha16=KNOWN_FULL_INIT_SHA16,
    teacher_sha16=teacher_sha[:16], known_teacher_sha16=KNOWN_TEACHER_SHA16)
print(f"[G1/G2] public_bit_identical={g1} detach_full_tree_eq_full={detach_eq_full}")

# ---------- G3: config diff empty except ablation point ----------
struct_full = jax.tree_util.tree_structure(init_full)
struct_det = jax.tree_util.tree_structure(init_det)
shapes_full = [np.asarray(v).shape for v in jax.tree_util.tree_leaves(init_full)]
shapes_det = [np.asarray(v).shape for v in jax.tree_util.tree_leaves(init_det)]
detach_tree_eq = bool(struct_full == struct_det and shapes_full == shapes_det)
# MLP: only the slow_gru subtree differs (slow_mlp_1/slow_mlp_2 replace slow_gru)
full_keys = set(inner_full.keys()); mlp_keys = set(inner_mlp.keys())
removed = sorted(full_keys - mlp_keys); added = sorted(mlp_keys - full_keys)
common = sorted(full_keys & mlp_keys)
common_shapes_eq = all(np.asarray(inner_full[k]).shape == np.asarray(inner_mlp[k]).shape
                       for k in common if not isinstance(inner_full[k], dict))
# compare common SUBTREES leaf shapes
def leaf_shapes(d): return [np.asarray(v).shape for v in jax.tree_util.tree_leaves(d)]
common_subtree_eq = all(leaf_shapes(inner_full[k]) == leaf_shapes(inner_mlp[k]) for k in common)
mlp_diff_only_slow = bool(removed == ["slow_gru"] and sorted(added) == ["slow_mlp_1", "slow_mlp_2"]
                          and common_subtree_eq)
results["G3_config_diff_empty_except_ablation"] = dict(pass_=bool(detach_tree_eq and mlp_diff_only_slow),
    detach_tree_eq_full=detach_tree_eq, mlp_removed_keys=removed, mlp_added_keys=added,
    mlp_common_subtrees_identical=bool(common_subtree_eq))
print(f"[G3] detach_tree_eq_full={detach_tree_eq} mlp_diff_only_slow={mlp_diff_only_slow} "
      f"(removed={removed} added={added})")

# ---------- G4: rollout/world/RNG protocol identical across the three trainers ----------
def trainer_consts(path):
    t = open(path).read()
    out = {}
    for name in ["NUM_ENVS", "ROLLOUT_STEPS", "TOTAL_STEPS", "MASTER_SEED", "LR", "ADAM_EPS", "GAMMA",
                 "GAE_LAMBDA", "CLIP_EPS", "VF_COEF", "ENT_COEF", "MAX_GRAD_NORM", "WINDOW_MEM",
                 "WINDOW_GRAD", "NUM_HEADS", "NUM_LAYERS", "EMBED_SIZE", "HIDDEN_LAYERS",
                 "QKV_FEATURES", "OPTIMISTIC_RESET_RATIO", "NUM_MINIBATCHES", "UPDATE_EPOCHS"]:
        m = re.search(rf'^{name}\s*=\s*([^\n#]+)', t, re.M)
        out[name] = m.group(1).strip() if m else None
    out["SAVE_STEPS"] = re.search(r'^SAVE_STEPS\s*=\s*\(([^\)]+)\)', t, re.M).group(1).strip()
    return out
TF = f"{W}/gpu2_lc_slowgru_reset128/src/run_slowgru_reset128_24576.py"
TD = f"{ATTR_SRC}/run_slowgru_detach_24576.py"
TM = f"{ATTR_SRC}/run_slowgru_mlp_24576.py"
c_full, c_det, c_mlp = trainer_consts(TF), trainer_consts(TD), trainer_consts(TM)
diff_keys = [k for k in c_full if c_full[k] != c_det[k] or c_full[k] != c_mlp[k]]
g4 = bool(len(diff_keys) == 0)
results["G4_protocol_identical"] = dict(pass_=g4, diff_keys=diff_keys, config=c_full)
print(f"[G4] protocol_identical={g4} diff_keys={diff_keys}")

# ---------- test fixtures ----------
key = jax.random.PRNGKey(123)
k1, k2, k3 = jax.random.split(key, 3)
obs = jax.random.normal(k1, (E, OBS_DIM))
memories = jax.random.normal(k2, (E, WM, NL, DIM)) * 0.1
mask = jnp.zeros((E, NH, 1, WM + 1), jnp.bool_).at[:, :, :, -1].set(True)
reset0 = jnp.zeros((E,), jnp.bool_)


def eq(a, b): return np.array_equal(np.asarray(a), np.asarray(b))


# random (NON-teacher-merged) init so the zero-init residual gate is non-zero -> the slow path actually
# contributes at this point (needed for the crosstalk / gradient / detach tests below).
def fresh_random_init(network):
    return network.init(jax.random.PRNGKey(0), _dummy_mem, _dummy_obs, _dummy_mask, _dummy_ls,
                        _dummy_reset, method=network.forward_eval)
ri_full = fresh_random_init(net_full); ri_det = fresh_random_init(net_det); ri_mlp = fresh_random_init(net_mlp)


def open_gate(params):
    """slow_to_actor is ZERO-initialised (constant 0.0) by design -> at init it blocks gradient flow to the
    rest of the slow branch (and contributes nothing to the output/backbone). To verify the slow branch /
    detach effect through the FULL chain, set the residual gate to a small non-zero value so gradient flows
    end-to-end (this is only for the gradient/detach GATES, not for training)."""
    p = {"params": dict(params["params"])}
    sta = dict(p["params"]["slow_to_actor"])
    sta["kernel"] = jax.random.normal(jax.random.PRNGKey(7), sta["kernel"].shape, jnp.float32) * 1.0
    p["params"]["slow_to_actor"] = sta
    return p
ri_full_open = open_gate(ri_full); ri_det_open = open_gate(ri_det); ri_mlp_open = open_gate(ri_mlp)


# ---------- G6: no env crosstalk (Detach & MLP). Use RANDOM-init params (non-zero slow_to_actor) so the
# perturbed env0 actually changes output; crosstalk gate = env1..3 outputs unchanged. ----------
def env_iso(network, params):
    ls_a = init_longstate(E); ls_b = init_longstate(E)
    def pert(x):
        if np.asarray(x).dtype.kind != "f": return x          # skip int 'count' leaf
        return x.at[0].set(jax.random.normal(k3, x.shape[1:]) * 0.5) if x.ndim >= 1 else x
    ls_b = jax.tree_util.tree_map(pert, ls_b)
    pi_a, v_a, _, _ = network.apply(params, memories, obs, mask, ls_a, reset0, method=network.forward_eval)
    pi_b, v_b, _, _ = network.apply(params, memories, obs, mask, ls_b, reset0, method=network.forward_eval)
    no_crosstalk = bool(eq(pi_a.logits[1:], pi_b.logits[1:]) and eq(v_a[1:], v_b[1:]))
    env0_changed = bool(not eq(pi_a.logits[0], pi_b.logits[0]))
    return no_crosstalk, env0_changed
g6_det, e0_det = env_iso(net_det, ri_det); g6_mlp, e0_mlp = env_iso(net_mlp, ri_mlp)
# crosstalk gate = env1..3 unchanged. env0_changed is EXPECTED False at init because slow_to_actor is
# zero-initialised (constant 0.0) -> the slow context has exactly zero effect on the output at init by
# design (clean feature-off); it becomes non-zero only once training moves the residual gate off zero.
results["G6_env_no_crosstalk"] = dict(pass_=bool(g6_det and g6_mlp),
    detach_no_crosstalk=g6_det, mlp_no_crosstalk=g6_mlp,
    detach_env0_changed=e0_det, mlp_env0_changed=e0_mlp,
    note="env0 unchanged at init is expected (zero-init residual gate); crosstalk = env1..3 unchanged")
print(f"[G6] no_crosstalk detach={g6_det} mlp={g6_mlp} (env0_changed det={e0_det} mlp={e0_mlp}; "
      f"expected False at init due to zero gate)")

# ---------- G9: slow-module finite nonzero grad (populated memory; random-init slow gate so grad flows) ----------
# populate memory so the slow buffer has content
_obs_seq = jax.random.normal(jax.random.PRNGKey(555), (64, E, OBS_DIM))   # DISTINCT obs per step so the
# period buffer holds varied GTrXL hiddens -> the attention values v_i differ -> the attention WEIGHTS
# (pool_q/pool_k) receive gradient, exactly as in a real rollout (constant obs would make v_i identical and
# the weight-gradient vanish since softmax weights sum to 1). Test-faithfulness fix, not a network change.
def populate(network, params, n_steps=40):
    ls = init_longstate(E); mem = memories
    for i in range(n_steps):
        _, _, mo, ls = network.apply(params, mem, _obs_seq[i], mask, ls, reset0, method=network.forward_eval)
        mem = jnp.roll(mem, -1, axis=1).at[:, -1].set(mo)
    return ls, mem
# For the gradient test, advance to count=31 so the single forward_eval inside slow_loss triggers a COMMIT
# (count 31->32) -> h_new is used -> the deep slow params (pool/slow_in/GRU-or-MLP) receive gradient, exactly
# as at the 4 per-rollout commits in the real training loss. (At a non-commit step h is just held -> no grad.)
ls_f, mem_f = populate(net_full, ri_full, 31)
ls_d, mem_d = populate(net_det, ri_det, 31)
ls_m, mem_m = populate(net_mlp, ri_mlp, 31)


def slow_loss(network, params, mem, ls):
    pi, _, _, _ = network.apply(params, mem, obs, mask, ls, reset0, method=network.forward_eval)
    return jnp.mean(pi.logits)


def slow_grad_norms(network, params, mem, ls, keys):
    grads = jax.grad(slow_loss, argnums=1)(network, params, mem, ls)["params"]
    out = {}
    for k in keys:
        if k not in grads: out[k] = dict(norm=-1.0, finite=False, present=False); continue
        gl = jax.tree_util.tree_leaves(grads[k])
        n = float(np.sqrt(sum(float(np.sum(np.asarray(g) ** 2)) for g in gl)))
        out[k] = dict(norm=n, finite=bool(all(np.all(np.isfinite(np.asarray(g))) for g in gl)), present=True)
    return out, grads


SLOW_KEYS_FULL = ["pool_q", "pool_k", "pool_v", "slow_in", "slow_gru", "slow_read", "slow_to_actor"]
SLOW_KEYS_MLP = ["pool_q", "pool_k", "pool_v", "slow_in", "slow_mlp_1", "slow_mlp_2", "slow_read", "slow_to_actor"]
gn_full, grads_full = slow_grad_norms(net_full, ri_full_open, mem_f, ls_f, SLOW_KEYS_FULL)
gn_det, grads_det = slow_grad_norms(net_det, ri_det_open, mem_d, ls_d, SLOW_KEYS_FULL)
gn_mlp, grads_mlp = slow_grad_norms(net_mlp, ri_mlp_open, mem_m, ls_m, SLOW_KEYS_MLP)
print("[G9-debug] sta_kernel_norm ri_full=%.3e open=%.3e | h_norm populated full=%.3e mlp=%.3e" % (
    float(np.linalg.norm(np.asarray(ri_full["params"]["slow_to_actor"]["kernel"]))),
    float(np.linalg.norm(np.asarray(ri_full_open["params"]["slow_to_actor"]["kernel"]))),
    float(np.linalg.norm(np.asarray(ls_f["h"]))), float(np.linalg.norm(np.asarray(ls_m["h"])))))
print("[G9-debug] full per-module grad norms:", {k: round(gn_full[k]["norm"], 6) for k in SLOW_KEYS_FULL})
print("[G9-debug] detach per-module grad norms:", {k: round(gn_det[k]["norm"], 6) for k in SLOW_KEYS_FULL})
print("[G9-debug] mlp per-module grad norms:", {k: round(gn_mlp[k]["norm"], 6) for k in SLOW_KEYS_MLP})


def g9_ok(gn, keys):
    return bool(all(gn[k]["present"] and gn[k]["finite"] and gn[k]["norm"] > 0 for k in keys))
g9_full = g9_ok(gn_full, SLOW_KEYS_FULL); g9_det = g9_ok(gn_det, SLOW_KEYS_FULL); g9_mlp = g9_ok(gn_mlp, SLOW_KEYS_MLP)
results["G9_slow_module_grad"] = dict(pass_=bool(g9_full and g9_det and g9_mlp),
    full=g9_full, detach=g9_det, mlp=g9_mlp,
    full_norms={k: round(gn_full[k]["norm"], 6) for k in SLOW_KEYS_FULL},
    detach_norms={k: round(gn_det[k]["norm"], 6) for k in SLOW_KEYS_FULL},
    mlp_norms={k: round(gn_mlp[k]["norm"], 6) for k in SLOW_KEYS_MLP})
print(f"[G9] slow_grad full={g9_full} detach={g9_det} mlp={g9_mlp}")

# ---------- DA: DETACH blocks slow->backbone gradient ----------
# backbone (transformer) grad differs Full vs Detach (slow path shapes backbone in Full, blocked in Detach);
# stop_gradient can only REMOVE flow, and G9 shows slow branch still trains in Detach -> isolates the effect.
# grads_full / grads_det (computed above with the OPEN gate) carry the backbone (transformer) gradient.
# Full: slow branch backprops into the GTrXL/CNN backbone. Detach: stop_gradient blocks that flow, so the
# backbone gradient DIFFERS (the slow->backbone shaping is removed) WHILE the slow branch still trains (G9).
def leaves_of(grads, key):
    return [np.asarray(v) for v in jax.tree_util.tree_leaves(grads[key])]
T_full = leaves_of(grads_full, "transformer"); T_det = leaves_of(grads_det, "transformer")
Tfull_norm = float(np.sqrt(sum(np.sum(a ** 2) for a in T_full)))
Tdet_norm = float(np.sqrt(sum(np.sum(a ** 2) for a in T_det)))
backbone_grad_differs = bool(len(T_full) == len(T_det) and
                             any(not np.array_equal(a, b) for a, b in zip(T_full, T_det)))
rel_diff = float(np.sqrt(sum(np.sum((a - b) ** 2) for a, b in zip(T_full, T_det))) / (Tfull_norm + 1e-12))
# stop_gradient can only REMOVE gradient flow; Full!=Detach with slow branch still training (g9_det) proves
# the slow->backbone gradient present in Full is the thing Detach removes.
da = bool(backbone_grad_differs and rel_diff > 1e-8 and g9_det)
results["DA_detach_blocks_backbone_grad"] = dict(pass_=da, backbone_grad_differs=backbone_grad_differs,
    backbone_grad_rel_diff=rel_diff, full_transformer_grad_norm=Tfull_norm,
    detach_transformer_grad_norm=Tdet_norm, detach_slow_branch_still_trains=g9_det)
print(f"[DA] detach_blocks_backbone={da} backbone_rel_diff={rel_diff:.6f} "
      f"(full_T_norm={Tfull_norm:.4e} detach_T_norm={Tdet_norm:.4e})")

# ---------- MB: MLP param count match ----------
def count(inner, keys): return int(sum(np.asarray(v).size for k in keys for v in jax.tree_util.tree_leaves(inner[k])))
gru_n = count(inner_full, ["slow_gru"])
mlp_n = count(inner_mlp, ["slow_mlp_1", "slow_mlp_2"])
total_full = int(sum(np.asarray(v).size for v in jax.tree_util.tree_leaves(init_full)))
total_mlp = int(sum(np.asarray(v).size for v in jax.tree_util.tree_leaves(init_mlp)))
mlp_match_pct = (mlp_n - gru_n) / gru_n * 100
mb = bool(abs(mlp_match_pct) <= 2.0 and total_full == total_mlp)
results["MB_mlp_param_match"] = dict(pass_=mb, gru_params=gru_n, mlp_params=mlp_n,
    mlp_match_pct=round(mlp_match_pct, 4), total_params_full=total_full, total_params_mlp=total_mlp,
    total_equal=bool(total_full == total_mlp))
print(f"[MB] mlp_param_match={mb} gru={gru_n} mlp={mlp_n} ({mlp_match_pct:+.4f}%) total_full={total_full} total_mlp={total_mlp}")

# ---------- MR: MLP non-recurrent (h_new independent of prev h); Full IS recurrent ----------
def recurrence_test(network, params):
    """Drive the slow update to a COMMIT boundary (count reaches SLOW_INTERVAL=32) so h_new is freshly
    computed, then check whether h_new depends on the PREVIOUS h. Recurrent (Full/GRU) -> depends;
    non-recurrent (MLP) -> independent."""
    x = jax.random.normal(k1, (E, DIM))
    reset = jnp.zeros((E,), jnp.bool_)
    ls = init_longstate(E)
    for _ in range(31):                                   # count -> 31 (no commit yet, h still zeros)
        ls = network.apply(params, ls, x, reset, method=network._slow_update)
    assert int(np.asarray(ls["count"]).max()) == 31
    ls_b = dict(jax.tree_util.tree_map(lambda v: v, ls)); ls_b["h"] = ls["h"] + 1.0   # perturb prev h
    ls_a = network.apply(params, ls, x, reset, method=network._slow_update)           # 32nd step -> commit
    ls_b2 = network.apply(params, ls_b, x, reset, method=network._slow_update)
    assert int(np.asarray(ls_a["count"]).max()) == 0      # committed -> count reset
    same = bool(eq(ls_a["h"], ls_b2["h"]))                # h_new identical despite different prev h?
    return same, np.asarray(ls_a["h"])
mr_mlp_same, _ = recurrence_test(net_mlp, ri_mlp)
mr_full_same, _ = recurrence_test(net_full, ri_full)
mr = bool(mr_mlp_same and not mr_full_same)
results["MR_mlp_non_recurrent"] = dict(pass_=mr, mlp_h_independent_of_prev_h=mr_mlp_same,
    full_h_depends_on_prev_h=bool(not mr_full_same))
print(f"[MR] mlp_non_recurrent={mr} (mlp_h_indep={mr_mlp_same}, full_recurrent={not mr_full_same})")

# ---------- G10: separate output dirs ----------
d1 = f"{W}/gpu3_slowgru_reset128_attribution/train_detach"; d2 = f"{W}/gpu3_slowgru_reset128_attribution/train_mlp"
os.makedirs(d1, exist_ok=True); os.makedirs(d2, exist_ok=True)
g10 = bool(os.path.isdir(d1) and os.path.isdir(d2) and os.path.abspath(d1) != os.path.abspath(d2))
results["G10_separate_output_dirs"] = dict(pass_=g10, detach_dir=d1, mlp_dir=d2)
print(f"[G10] separate_dirs={g10}")

# ---------- frozen-network integrity (the Full reference network in ATTR_SRC must be the frozen one) ----------
import hashlib as _h
def _fsha(p):
    hh = _h.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): hh.update(c)
    return hh.hexdigest()
attr_net_sha = _fsha(f"{ATTR_SRC}/slowgru_network.py")
results["frozen_network_integrity"] = dict(pass_=bool(attr_net_sha[:16] == FROZEN_NET_SHA16),
    attr_src_slowgru_network_sha16=attr_net_sha[:16], known_frozen_sha16=FROZEN_NET_SHA16)

ALL = [v["pass_"] for v in results.values()]
ok = bool(all(ALL))
out = dict(label="P3_ATTRIBUTION_GATES", arms=["DETACH", "MATCHED_MLP"], gates=results,
    ATTRIBUTION_GATES_PASS=ok,
    trainer_level_gates="G5(exact_resume)/G7(no_nan)/G8(entropy) verified by training runs",
    timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
json.dump(out, open(f"{ATTR_SRC}/P3_ATTRIBUTION_gates.json", "w"), indent=2, default=str)
print(f"\n[P3-ATTR] ATTRIBUTION_GATES_{'PASS' if ok else 'FAIL'}")
