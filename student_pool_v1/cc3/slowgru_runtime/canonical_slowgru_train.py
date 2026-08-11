#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CC3 canonical SlowGRU trainer (task CC3_CORRECTED_MATCHED_SLOWGRU_STUDENTS).

Built VERBATIM on the validated wave1 driver machinery
(run_slowgru_reset128_longrun.py / run_slowgru_24576.py, network SHA b2652105...):
same env wrapper / optax / Transition / _env_step mask mechanics / _loss_fn /
chunked-scan outer loop / full-state save + bit-exact roundtrip, under
--xla_gpu_deterministic_ops=true on a single allowed GPU (GPU2/GPU3 only).

ONLY intentional differences vs the historical drivers:
  * carry_mode is a parameter: RESET128 keeps the boundary-clear block at the head of
    every _update_step; PERSISTENT omits it (the documented ONLY difference between
    the two historical trainers). The two are NOT unified into the same behavior.
  * CC3 EXACT-RESUME PROOF phase (resume mode): after loading the historical
    24576 full_state.pkl, run two INDEPENDENT 4096-step continuations from the same
    restored state; require BIT-EXACT params SHA equality (determinism), plus
    finiteness/entropy engineering gates, BEFORE any long run is allowed.
  * fresh mode: canonical rebuild from teacher ckpt17500 (bit-for-bit inherited
    paths + zero-init slow-GRU); requires reproducing the canonical init params SHA
    5ae94ed0... as a gate.

FROZEN protocol (unchanged): num_envs=16, num_steps=128, seed=42, LR=2e-5,
Adam eps=1e-5, gamma=0.999, gae_lambda=0.8, clip=0.2, vf=0.5, ent=0.002,
gradnorm=1.0, anneal_lr=False, window_mem=128, window_grad=64, heads=8, layers=2,
embed/qkv=256, hidden=256, gating=True, gating_bias=2.0, optimistic_reset_ratio=16,
minibatches=2, epochs=1. Original PPO ONLY (no replay/vtrace/hindsight/awr/novelty/
nav_aux/egomap). NO performance early-stop — engineering gates only.

Any gate failure -> hard stop with a nonzero exit and STATUS line (no auto-retry).
"""
import os

GPU_ALLOWED = {"GPU-8df11537-ab79-722d-606f-411966196c4c",
               "GPU-f56a59b4-99f3-f2e5-11c6-d01685de8abd"}


def _early_arg(argv, flag):
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
    return None


_GPU_UUID = _early_arg(__import__("sys").argv, "--gpu-uuid")
assert _GPU_UUID in GPU_ALLOWED, "gpu %s not in allowed set (GPU2/GPU3 only)" % _GPU_UUID
os.environ["CUDA_VISIBLE_DEVICES"] = _GPU_UUID
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"

import sys, json, argparse, hashlib, time, pickle, shutil   # noqa: E402

ARM_BASE = "CC3_SLOWGRU_CANONICAL"
V7_SRC = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src"
V7 = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB"
TEACHER_CKPT = ("/home/oseasy/incoming/henry_work_20260721T105300/extracted/"
                "Henry_work/base_ckpt_v7fix55_armA_s0/rl_checkpoints/17500")
S4_TASK_PATH = "/home/oseasy/experiments/p2_v1_20260722/evidence/s4_task_code.py"

import numpy as np            # noqa: E402
import jax                    # noqa: E402
import jax.numpy as jnp       # noqa: E402
import optax                  # noqa: E402
from flax.training.train_state import TrainState   # noqa: E402

from craftax.craftax.constants import Achievement                      # noqa: E402
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams   # noqa: E402
from dicode.network import Transition                                  # noqa: E402
from dicode.task_utils import get_achievement_multi_hot                # noqa: E402
from dicode.utils.general.train_state_utils import load_weights_only   # noqa: E402
from dicode.wrappers_cl import DistributedMultiTaskOptimisticLogWrapper  # noqa: E402
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv         # noqa: E402

# --------------------------------------------------------------------- frozen protocol
NUM_ENVS = 16
ROLLOUT_STEPS = 128
STEPS_PER_UPDATE = NUM_ENVS * ROLLOUT_STEPS      # 2048
TOTAL_STEPS = 98304
TOTAL_UPDATES = TOTAL_STEPS // STEPS_PER_UPDATE   # 48
UPDATES_PER_CHUNK = 2                             # 4096-step chunks
MASTER_SEED = 42
LR = 2e-5
ADAM_EPS = 1e-5
GAMMA = 0.999
GAE_LAMBDA = 0.8
CLIP_EPS = 0.2
VF_COEF = 0.5
ENT_COEF = 0.002
MAX_GRAD_NORM = 1.0
WINDOW_MEM = 128
WINDOW_GRAD = 64
NUM_HEADS = 8
NUM_LAYERS = 2
EMBED_SIZE = 256
HIDDEN_LAYERS = 256
QKV_FEATURES = 256
OPTIMISTIC_RESET_RATIO = 16
NUM_MINIBATCHES = 2
UPDATE_EPOCHS = 1
VT_CLIP_MIN = -50.0
VT_CLIP_MAX = 300.0

KNOWN_TEACHER_SHA = "d4e85af58b7f87d689fadea12eec70c852fa098a09f5ea8907448684b3bf60f5"
KNOWN_INIT_SHA = "5ae94ed0257f50fabfba6a928169acf0eb8e05aa34705c3763bc342eed3218ab"
KNOWN_NETWORK_SHA = "b265210597d003218e303ef458ff697b5c9c6a14bfccca098ccce8014bf3eb0b"
KNOWN_S4_TASK_SHA = "45fdd17c5b34b9f32a7f85b8030437f74d63d16bed2d6f2c683d80454e4d824d"
# historical persistent @24576 artifact (identity-verified by CC3 recovery_probe 36/36)
KNOWN_PERSISTENT_24576_PARAMS_SHA = "1bd4fbfe91ab4da44c274ef20f372e04bf6a7e8367869e39e0b65f044c85e9f2"
KNOWN_PERSISTENT_24576_FILE_SHA = "d4e7008da7d2a78a7765b32379704719165288015d031856850ad7c8a0e7495e"


class Cfg:
    lr = LR; num_envs = NUM_ENVS; num_steps = ROLLOUT_STEPS; update_epochs = UPDATE_EPOCHS
    num_minibatches = NUM_MINIBATCHES; gamma = GAMMA; gae_lambda = GAE_LAMBDA; clip_eps = CLIP_EPS
    ent_coef = ENT_COEF; vf_coef = VF_COEF; max_grad_norm = MAX_GRAD_NORM; activation = "relu"
    anneal_lr = False; qkv_features = QKV_FEATURES; embed_size = EMBED_SIZE; num_heads = NUM_HEADS
    num_layers = NUM_LAYERS; hidden_layers = HIDDEN_LAYERS; window_mem = WINDOW_MEM
    window_grad = WINDOW_GRAD; gating = True; gating_bias = 2.0; condition_on_task = True
    optimistic_reset_ratio = OPTIMISTIC_RESET_RATIO; mode = "score"; bonus_type = "none"
    dynamic_bonus_k = 0.0; completion_bonus_scale = 0.0; completion_bonus_min = 0.0
    value_target_clip_min = VT_CLIP_MIN; value_target_clip_max = VT_CLIP_MAX
    def get(self, key, default=None): return getattr(self, key, default)


# ------------------------------------------------------------- hashing helpers (driver-exact)
def _params_sha(params):
    h = hashlib.sha256()
    for v in jax.tree_util.tree_leaves(params):
        h.update(np.ascontiguousarray(np.asarray(v)).tobytes())
    return h.hexdigest()


def _leaf_hash(pytree):
    h = hashlib.sha256()
    for l in jax.tree_util.tree_leaves(pytree):
        a = np.asarray(l); h.update(str(a.shape).encode()); h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _leaves_equal(a, b):
    la = jax.tree_util.tree_leaves(a); lb = jax.tree_util.tree_leaves(b)
    if len(la) != len(lb): return False
    return all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb))


def _finite(pytree):
    return bool(all(np.all(np.isfinite(np.asarray(v))) for v in jax.tree_util.tree_leaves(pytree)
                    if np.asarray(v).dtype.kind in "fi"))


def _pack(pytree):
    leaves, treedef = jax.tree_util.tree_flatten(pytree)
    return [np.asarray(l) for l in leaves], treedef


def _unpack(packed):
    leaves, treedef = packed
    return jax.tree_util.tree_unflatten(treedef, [jnp.asarray(l) for l in leaves])


def _sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carry-mode", required=True, choices=["persistent", "reset128"])
    ap.add_argument("--mode", required=True, choices=["resume", "fresh"])
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--arm-src", required=True, help="historical arm src dir (slowgru_network.py)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckpt_root", required=True)
    ap.add_argument("--gpu-uuid", required=True)
    ap.add_argument("--save_steps", default="28672,98304")
    args = ap.parse_args()
    args.out_dir = os.path.abspath(args.out_dir); args.ckpt_root = os.path.abspath(args.ckpt_root)
    save_steps = tuple(int(x) for x in args.save_steps.split(",") if x.strip())
    CARRY_MODE = args.carry_mode.upper()
    ARM = "%s_%s" % (ARM_BASE, CARRY_MODE)
    METRICS_LOG = []
    LS_START_HASH = []          # longstate hash at each rollout start (post-clear for RESET128)
    LS_CARRIED_HASH = []        # longstate hash carried into the rollout (pre-clear)

    SRC = args.arm_src
    for p in (SRC, V7_SRC, V7):
        if p not in sys.path:
            sys.path.insert(0, p)
    assert _sha_file(os.path.join(SRC, "slowgru_network.py")) == KNOWN_NETWORK_SHA, \
        "NETWORK_SRC_SHA_MISMATCH (server file is SHA-authoritative)"
    assert _sha_file(S4_TASK_PATH) == KNOWN_S4_TASK_SHA, "S4_TASK_SHA_MISMATCH"
    from slowgru_network import ActorCriticSlowGRU, init_longstate, SLOW_INTERVAL, SLOW_DIM
    assert SLOW_INTERVAL == 32 and SLOW_DIM == 256

    import wandb  # driver-fidelity: disabled wandb init, no network, no tracking
    wandb.init(mode="disabled")

    devs = jax.local_devices()
    print("[%s] devices=%s gpu=%s carry_mode=%s mode=%s" % (ARM, devs, args.gpu_uuid,
                                                            CARRY_MODE, args.mode), flush=True)
    assert len(devs) == 1, f"expected exactly 1 visible device, got {devs}"
    for d in (args.out_dir, args.ckpt_root): os.makedirs(d, exist_ok=True)
    free_gb = shutil.disk_usage(args.ckpt_root).free / 1024 ** 3
    print("[%s] disk free = %.1f GB" % (ARM, free_gb), flush=True)
    assert free_gb > 5.0, f"HARD STOP low disk {free_gb:.1f} GB"

    cfg = Cfg(); cfg.training = cfg
    code_sha = {"launcher": _sha_file(os.path.abspath(__file__)),
                "network": _sha_file(os.path.join(SRC, "slowgru_network.py")),
                "s4_task": _sha_file(S4_TASK_PATH)}

    # ---- S4_dark env (EXACT driver construction) ----
    ach_table = jnp.array([get_achievement_multi_hot([Achievement.DEFEAT_KOBOLD])], dtype=jnp.float32)
    EMB = int(ach_table.shape[1])
    with open(S4_TASK_PATH) as f: s4_code = f.read()
    ns = {}; exec(s4_code, ns); Task = ns["Env"]
    static_env_params = StaticEnvParams()
    env_params_ctor = EnvParams(max_timesteps=4096)
    base_env = MultiTaskMiniCraftaxEnv([Task], static_env_params, env_params_ctor, cfg.condition_on_task,
        conditioning_type="embedding", embedding_size=EMB, completion_bonus_scale=cfg.completion_bonus_scale,
        completion_bonus_min=cfg.completion_bonus_min, bonus_type=cfg.bonus_type, dynamic_bonus_k=cfg.dynamic_bonus_k)
    env = DistributedMultiTaskOptimisticLogWrapper(base_env, jax.random.PRNGKey(0), cfg.num_envs, 1,
        cfg.optimistic_reset_ratio, jnp.ones(1), ach_table)
    env_params = env.default_params
    obs_dim = int(env.observation_space(env_params).shape[0])
    action_dim = int(env.action_space(env_params).n)
    print("[%s] S4_dark env: obs_dim=%d action_dim=%d emb=%d" % (ARM, obs_dim, action_dim, EMB), flush=True)
    assert action_dim == 43 and obs_dim == 8335 and EMB == 67

    network = ActorCriticSlowGRU(action_dim=action_dim, activation=cfg.activation,
        hidden_layers=cfg.hidden_layers, encoder_size=cfg.embed_size, num_heads=cfg.num_heads,
        qkv_features=cfg.qkv_features, num_layers=cfg.num_layers, gating=cfg.gating,
        gating_bias=cfg.gating_bias, use_longmem=True)
    tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.lr, eps=ADAM_EPS))

    current_original_return = 0.0
    vt_clip_min = VT_CLIP_MIN; vt_clip_max = VT_CLIP_MAX
    initial_global_update_step = 0
    config = cfg

    batch_indices_select = jax.vmap(lambda x, y: x[y])
    roll_vmap = jax.vmap(jnp.roll, in_axes=(-2, 0, None), out_axes=-2)
    batchify = lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1],) + x.shape[2:])

    def _log_callback(metrics, step):
        t_loss, v_loss, a_loss, ent, g_norm_mean, g_norm_max = metrics
        METRICS_LOG.append(dict(total_loss=float(np.asarray(t_loss)), value_loss=float(np.asarray(v_loss)),
            actor_loss=float(np.asarray(a_loss)), entropy=float(np.asarray(ent)),
            grad_norm_mean=float(np.asarray(g_norm_mean)), grad_norm_max=float(np.asarray(g_norm_max)),
            update_step=int(np.asarray(step))))

    def _ls_hash_callback(ls):
        LS_START_HASH.append(_leaf_hash(ls))

    def _ls_carried_callback(ls):
        LS_CARRIED_HASH.append(_leaf_hash(ls))

    # ============================================================== _update_step ======
    def _update_step(runner_state, unused_scan_input):
        if CARRY_MODE == "RESET128":
            # trainer boundary-clear block VERBATIM: clear ONLY the slow longstate at every
            # 128-step rollout boundary; fast memories/mask/idx carry.
            jax.debug.callback(_ls_carried_callback, runner_state[8])
            _ls_cleared = init_longstate(config.num_envs)
            runner_state = (runner_state[0], runner_state[1], runner_state[2], runner_state[3],
                            runner_state[4], runner_state[5], runner_state[6], runner_state[7],
                            _ls_cleared, runner_state[9], runner_state[10], runner_state[11])
        jax.debug.callback(_ls_hash_callback, runner_state[8])

        def _env_step(runner_state, _):
            (train_state, env_state, memories, memories_mask, memories_mask_idx,
             last_obs, done, true_done, longstate, step_env_currentloop, update_step, rng) = runner_state

            memories_mask_idx = jnp.where(done, config.window_mem,
                                          jnp.clip(memories_mask_idx - 1, 0, config.window_mem))
            memories_mask = jnp.where(done[:, None, None, None],
                jnp.zeros((config.num_envs, config.num_heads, 1, config.window_mem + 1), dtype=jnp.bool_),
                memories_mask)
            ohot = jax.nn.one_hot(memories_mask_idx, config.window_mem + 1)[:, None, None, :].repeat(config.num_heads, 1)
            memories_mask = jnp.logical_or(memories_mask, ohot)

            rng, _rng = jax.random.split(rng)
            pi, value, memories_out, ls_new = network.apply(train_state.params, memories, last_obs,
                memories_mask, longstate, true_done, method=network.forward_eval)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)
            memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(memories_out)

            rng, _rng = jax.random.split(rng)
            obsv, env_state, reward, done, info = env.step(_rng, env_state, action, env_params)
            env_state = env_state.replace(running_original_return=jnp.full(
                (config.num_envs,), current_original_return, dtype=jnp.float32))
            true_done_next = info["returned_episode"]

            memory_indices = jnp.arange(0, config.window_mem)[None, :] + \
                step_env_currentloop * jnp.ones((config.num_envs, 1), dtype=jnp.int32)
            transition = Transition(done, action, value, reward, log_prob,
                                    memories_mask.squeeze(), memory_indices, last_obs, info)
            carry = (train_state, env_state, memories, memories_mask, memories_mask_idx,
                     obsv, done, true_done_next, ls_new, step_env_currentloop + 1, update_step, rng)
            return carry, (transition, memories_out)

        memories_previous = runner_state[2]
        longstate_previous = runner_state[8]
        true_done_start = runner_state[7]
        (final_state_carry), (traj_batch, memories_batch) = jax.lax.scan(
            _env_step, runner_state, None, config.num_steps)
        (train_state, final_env_state, final_memories, final_mask, final_mask_idx,
         final_obs, done, final_true_done, final_longstate, final_step_loop, update_step, rng) = final_state_carry

        _, last_val, _, _ = network.apply(train_state.params, final_memories, final_obs, final_mask,
                                          final_longstate, final_true_done, method=network.forward_eval)

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(carry, transition):
                gae, next_value = carry
                done, value, reward = transition.done, transition.value, transition.reward
                delta = reward + config.gamma * next_value * (1 - done) - value
                gae = delta + config.gamma * config.gae_lambda * (1 - done) * gae
                return (gae, value), gae
            _, advantages = jax.lax.scan(_get_advantages, (jnp.zeros_like(last_val), last_val),
                                         traj_batch, reverse=True, unroll=16)
            return advantages, jnp.clip(advantages + traj_batch.value, vt_clip_min, vt_clip_max)
        advantages, targets = _calculate_gae(traj_batch, last_val)

        memories_batch = jnp.concatenate([jnp.swapaxes(memories_previous, 0, 1), memories_batch], axis=0)

        def _update_epoch(update_state, unused):
            def _update_minbatch(train_state, batch_info):
                traj_batch, memories_batch, advantages, targets, ls_prev_mb, td_start_mb = batch_info
                def _loss_fn(params, traj_batch, memories_batch, gae, targets, ls_prev, td_start):
                    memories_batch = batch_indices_select(memories_batch,
                        traj_batch.memories_indices[:, :: config.window_grad])
                    memories_batch = batchify(memories_batch)
                    memories_mask = traj_batch.memories_mask.reshape(
                        (-1, config.window_grad) + traj_batch.memories_mask.shape[2:])
                    memories_mask = jnp.swapaxes(memories_mask, 1, 2)
                    memories_mask = jnp.concatenate((memories_mask,
                        jnp.zeros(memories_mask.shape[:-1] + (config.window_grad - 1,), dtype=jnp.bool_)), axis=-1)
                    memories_mask = roll_vmap(memories_mask, jnp.arange(0, config.window_grad), -1)
                    obs = traj_batch.obs.reshape((-1, config.window_grad) + traj_batch.obs.shape[2:])
                    traj_batch_r, targets_r, gae_r = jax.tree_util.tree_map(
                        lambda x: jnp.reshape(x, (-1, config.window_grad) + x.shape[2:]),
                        (traj_batch, targets, gae))
                    info_re = traj_batch.info["returned_episode"]
                    T_roll = info_re.shape[1]
                    true_done_ET = jnp.concatenate([td_start[:, None], info_re[:, : T_roll - 1]], axis=1)
                    pi, value = network.apply(params, memories_batch, obs, memories_mask, true_done_ET,
                                              ls_prev, method=network.model_forward_train_longmem)
                    log_prob = pi.log_prob(traj_batch_r.action)
                    value_pred_clipped = traj_batch_r.value + (value - traj_batch_r.value).clip(
                        -config.clip_eps, config.clip_eps)
                    value_losses = jnp.square(value - targets_r)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets_r)
                    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    ratio = jnp.exp(log_prob - traj_batch_r.log_prob)
                    gae_r = (gae_r - gae_r.mean()) / (gae_r.std() + 1e-8)
                    loss_actor1 = ratio * gae_r
                    loss_actor2 = jnp.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * gae_r
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                    entropy = pi.entropy().mean()
                    total_loss = loss_actor + config.vf_coef * value_loss - config.ent_coef * entropy
                    return total_loss, (value_loss, loss_actor, entropy)
                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                (total_loss, (value_loss, loss_actor, entropy)), grads = grad_fn(
                    train_state.params, traj_batch, memories_batch, advantages, targets,
                    ls_prev_mb, td_start_mb)
                grad_norm = optax.global_norm(grads)
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, (total_loss, value_loss, loss_actor, entropy, grad_norm)

            (train_state, traj_batch, memories_batch, advantages, targets,
             ls_prev, td_start, update_step, rng) = update_state
            rng, _rng = jax.random.split(rng)
            permutation = jax.random.permutation(_rng, config.num_envs)
            batch = (traj_batch, memories_batch, advantages, targets)
            batch = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), batch)
            shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, [config.num_minibatches, -1] + list(x.shape[1:])), shuffled_batch)
            ls_prev_shuffled = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), ls_prev)
            ls_prev_mb = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, [config.num_minibatches, -1] + list(x.shape[1:])), ls_prev_shuffled)
            td_start_shuffled = jnp.take(td_start, permutation, axis=0)
            td_start_mb = jnp.reshape(td_start_shuffled, [config.num_minibatches, -1])
            mb_full = minibatches + (ls_prev_mb, td_start_mb)
            train_state, (total_loss, value_loss, loss_actor, entropy, grad_norm) = \
                jax.lax.scan(_update_minbatch, train_state, mb_full)
            return (train_state, traj_batch, memories_batch, advantages, targets,
                    ls_prev, td_start, update_step, rng), (total_loss, value_loss, loss_actor, entropy, grad_norm)

        update_state = (train_state, traj_batch, memories_batch, advantages, targets,
                        longstate_previous, true_done_start, update_step, rng)
        update_state, rl_info = jax.lax.scan(_update_epoch, update_state, None, config.update_epochs)

        losses_and_ent = rl_info[:4]; grad_norms = rl_info[4]
        losses_mean = jax.tree_util.tree_map(lambda x: jnp.mean(x), losses_and_ent)
        gn_mean = jnp.mean(grad_norms); gn_max = jnp.max(grad_norms)
        metrics_to_log = (*losses_mean, gn_mean, gn_max)
        current_step = initial_global_update_step + update_step
        jax.debug.callback(_log_callback, metrics_to_log, current_step)

        train_state = update_state[0]; rng = update_state[-1]
        next_runner_state = (train_state, final_env_state, final_memories, final_mask,
                             final_mask_idx, final_obs, done, final_true_done, final_longstate,
                             0, update_step + 1, rng)
        return next_runner_state, None

    # ============================================================== resume loader ======
    def _load_resume(pkl_path):
        with open(pkl_path, "rb") as f:
            rd = pickle.load(f)
        ts = TrainState.create(apply_fn=network.apply, params=_unpack(rd["params"]), tx=tx).replace(
            opt_state=_unpack(rd["opt_state"]), step=jnp.asarray(rd["opt_step"], jnp.int32))
        rs = (ts, _unpack(rd["env_state"]), jnp.asarray(rd["memories"]),
              jnp.asarray(rd["memories_mask"]), jnp.asarray(rd["memories_mask_idx"]),
              jnp.asarray(rd["obs"]), jnp.asarray(rd["done"]), jnp.asarray(rd["true_done"]),
              _unpack(rd["longstate"]), int(rd["step_env_currentloop"]),
              int(rd["update_step"]), jnp.asarray(rd["rng"]))
        return rd, ts, rs

    # ============================================================== init / resume ======
    teacher_params = load_weights_only(TEACHER_CKPT, base_env, env_params_ctor, cfg,
                                       load_opt_state=False).params
    teacher_inner = teacher_params["params"]
    teacher_sha = _params_sha(teacher_params)
    assert teacher_sha == KNOWN_TEACHER_SHA, \
        f"TEACHER_SHA_MISMATCH {teacher_sha} != {KNOWN_TEACHER_SHA}"
    dummy_mem = jnp.zeros((2, cfg.window_mem, cfg.num_layers, cfg.embed_size))
    dummy_obs = jnp.zeros((2, obs_dim))
    dummy_mask = jnp.zeros((2, cfg.num_heads, 1, cfg.window_mem + 1), jnp.bool_)
    dummy_ls = init_longstate(2)
    dummy_reset = jnp.zeros((2,), jnp.bool_)
    full_vars = network.init(jax.random.PRNGKey(0), dummy_mem, dummy_obs, dummy_mask, dummy_ls,
                             dummy_reset, method=network.forward_eval)
    full_inner = full_vars["params"]
    missing = [k for k in teacher_inner if k not in full_inner]
    assert not missing, f"teacher keys not in slow-gru net: {missing}"
    init_inner = dict(full_inner)
    for k in teacher_inner:
        init_inner[k] = teacher_inner[k]
    init_params = {"params": init_inner}
    init_params_sha = _params_sha(init_params)
    print("[%s] init_sha=%s teacher_sha=%s" % (ARM, init_params_sha[:16], teacher_sha[:16]), flush=True)

    if args.mode == "resume":
        assert args.resume_from, "--resume-from required in resume mode"
        assert _sha_file(args.resume_from) == KNOWN_PERSISTENT_24576_FILE_SHA, \
            "RESUME_FILE_SHA_MISMATCH (identity must be verified before resume)"
        rd0, ts0, runner_state = _load_resume(args.resume_from)
        start_update = int(rd0["update_step"])
        resumed_params_sha = _params_sha(runner_state[0].params)
        print("[%s] resumed at update_step=%d params_sha=%s" % (ARM, start_update,
                                                                resumed_params_sha[:16]), flush=True)
        # identity gates before ANY training continuation
        assert resumed_params_sha == KNOWN_PERSISTENT_24576_PARAMS_SHA, "RESUME_PARAMS_SHA_MISMATCH"
        assert int(rd0["opt_step"]) == 24 and start_update == 12, "RESUME_STEP_COUNTERS_UNEXPECTED"
        assert rd0["manifest"].get("params_sha256") == KNOWN_PERSISTENT_24576_PARAMS_SHA
        assert _leaf_hash(runner_state[0].opt_state) == rd0["manifest"]["opt_state_leaf_hash"], \
            "RESUME_OPT_STATE_LEAF_HASH_MISMATCH"
        assert _leaf_hash(runner_state[8]) == rd0["manifest"]["longstate_leaf_hash"], \
            "RESUME_LONGSTATE_LEAF_HASH_MISMATCH"
        assert _finite(runner_state[0].params) and _finite(runner_state[0].opt_state), \
            "RESUME_NON_FINITE"
    else:
        assert init_params_sha == KNOWN_INIT_SHA, \
            f"FRESH_INIT_SHA_MISMATCH {init_params_sha} != canonical {KNOWN_INIT_SHA}"
        ts = TrainState.create(apply_fn=network.apply, params=init_params, tx=tx)
        assert int(ts.step) == 0
        rng0 = jax.random.PRNGKey(MASTER_SEED); rng0, _rng = jax.random.split(rng0)
        obsv, env_state = env.reset(_rng, env_params)
        env_state = env_state.replace(running_original_return=jnp.full(
            (cfg.num_envs,), current_original_return, dtype=jnp.float32))
        memories = jnp.zeros((cfg.num_envs, cfg.window_mem, cfg.num_layers, cfg.embed_size))
        memories_mask = jnp.zeros((cfg.num_envs, cfg.num_heads, 1, cfg.window_mem + 1), dtype=jnp.bool_)
        memories_mask_idx = jnp.zeros((cfg.num_envs,), dtype=jnp.int32) + (cfg.window_mem + 1)
        done = jnp.zeros((cfg.num_envs,), dtype=jnp.bool_)
        true_done = jnp.zeros((cfg.num_envs,), dtype=jnp.bool_)
        longstate = init_longstate(cfg.num_envs)
        rng0, _rng = jax.random.split(rng0)
        runner_state = (ts, env_state, memories, memories_mask, memories_mask_idx, obsv, done,
                        true_done, longstate, 0, 0, _rng)
        start_update = 0

    # ============================================================== save helper (driver verbatim) ======
    cfg_dict = {k: v for k, v in vars(Cfg).items() if not k.startswith("__") and not callable(v)}

    def _save_full(runner_state, global_step, params_sha):
        (train_state_, env_state_, memories_, mask_, mask_idx_, obs_, done_,
         true_done_, longstate_, step_loop_, update_step_, rng_) = runner_state
        ckpt_dir = os.path.join(args.ckpt_root, str(global_step)); os.makedirs(ckpt_dir, exist_ok=True)
        full = dict(params=_pack(train_state_.params), opt_state=_pack(train_state_.opt_state),
            opt_step=int(train_state_.step), env_state=_pack(env_state_),
            memories=np.asarray(memories_), memories_mask=np.asarray(mask_),
            memories_mask_idx=np.asarray(mask_idx_), obs=np.asarray(obs_), done=np.asarray(done_),
            true_done=np.asarray(true_done_), longstate=_pack(longstate_),
            step_env_currentloop=int(step_loop_), update_step=int(update_step_), rng=np.asarray(rng_),
            global_step=global_step, update_count=int(update_step_), config=cfg_dict, code_sha256=code_sha,
            manifest=dict(label=ARM, arm=ARM, carry_mode=CARRY_MODE, rng_seed=MASTER_SEED,
                          xla_flags=os.environ["XLA_FLAGS"], gpu_uuid=args.gpu_uuid,
                          params_sha256=params_sha,
                          opt_state_leaf_hash=_leaf_hash(train_state_.opt_state),
                          longstate_leaf_hash=_leaf_hash(longstate_),
                          teacher_init_sha256=teacher_sha, slow_interval=SLOW_INTERVAL, slow_dim=SLOW_DIM,
                          cc3_mode=args.mode,
                          cc3_resume_source=(args.resume_from if args.mode == "resume" else None),
                          cc3_resume_params_sha256=(KNOWN_PERSISTENT_24576_PARAMS_SHA
                                                    if args.mode == "resume" else None)))
        pkl_path = os.path.join(ckpt_dir, "full_state.pkl"); tmp_path = pkl_path + ".tmp"
        with open(tmp_path, "wb") as f: pickle.dump(full, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, pkl_path)
        with open(pkl_path, "rb") as f: rd = pickle.load(f)
        rt_ok = (_params_sha(_unpack(rd["params"])) == params_sha
                 and _leaves_equal(_unpack(rd["opt_state"]), train_state_.opt_state)
                 and np.array_equal(rd["rng"], np.asarray(rng_))
                 and _leaves_equal(_unpack(rd["env_state"]), env_state_)
                 and np.array_equal(rd["memories"], np.asarray(memories_))
                 and np.array_equal(rd["memories_mask"], np.asarray(mask_))
                 and np.array_equal(rd["memories_mask_idx"], np.asarray(mask_idx_))
                 and np.array_equal(rd["obs"], np.asarray(obs_))
                 and np.array_equal(rd["done"], np.asarray(done_))
                 and np.array_equal(rd["true_done"], np.asarray(true_done_))
                 and _leaves_equal(_unpack(rd["longstate"]), longstate_)
                 and int(rd["opt_step"]) == int(train_state_.step)
                 and int(rd["global_step"]) == global_step
                 and int(rd["update_count"]) == int(update_step_))
        print("[%s] saved FULL state -> %s (sha=%s) roundtrip_ok=%s" % (
            ARM, pkl_path, _sha_file(pkl_path)[:16], rt_ok), flush=True)
        assert rt_ok, f"FAIL roundtrip bit-exact at {global_step}"
        return pkl_path, _sha_file(pkl_path)

    # ============================================================== CC3 EXACT-RESUME PROOF ======
    proof = None
    if args.mode == "resume":
        run_chunk = jax.jit(lambda rs: jax.lax.scan(_update_step, rs, None, UPDATES_PER_CHUNK))
        def _wait_metrics(n, timeout_s=60.0):
            deadline = time.time() + timeout_s
            while len(METRICS_LOG) < n and time.time() < deadline:
                time.sleep(0.2)

        print("[%s] EXACT-RESUME PROOF run A (4096 steps)..." % ARM, flush=True)
        rs_a, _ = run_chunk(runner_state); jax.block_until_ready(rs_a)
        _wait_metrics(UPDATES_PER_CHUNK)
        sha_a = _params_sha(rs_a[0].params)
        ent_a = METRICS_LOG[-1]["entropy"] if METRICS_LOG else float("nan")
        gn_a = METRICS_LOG[-1]["grad_norm_max"] if METRICS_LOG else float("nan")
        finite_a = _finite(rs_a[0].params) and _finite(rs_a[0].opt_state)
        print("[%s] run A done sha28672=%s entropy=%.4f grad_norm_max=%.4f finite=%s"
              % (ARM, sha_a[:16], ent_a, gn_a, finite_a), flush=True)
        # independent reload + re-continuation
        rd_b, ts_b, rs_b0 = _load_resume(args.resume_from)
        assert _params_sha(rs_b0[0].params) == KNOWN_PERSISTENT_24576_PARAMS_SHA
        n_metrics_before = len(METRICS_LOG)
        print("[%s] EXACT-RESUME PROOF run B (independent reload, 4096 steps)..." % ARM, flush=True)
        rs_b, _ = run_chunk(rs_b0); jax.block_until_ready(rs_b)
        _wait_metrics(n_metrics_before + UPDATES_PER_CHUNK)
        sha_b = _params_sha(rs_b[0].params)
        ent_b = METRICS_LOG[-1]["entropy"] if METRICS_LOG else float("nan")
        deterministic = (sha_a == sha_b)
        print("[%s] run B done sha28672=%s deterministic=%s" % (ARM, sha_b[:16], deterministic), flush=True)
        # engineering gates ONLY (no performance judgment): determinism, finiteness,
        # entropy within (0.01, ln(43)), grad norm finite
        entropy_ok = bool(np.isfinite(ent_a) and 0.01 < ent_a < 3.762)
        gradnorm_ok = bool(np.isfinite(gn_a))
        proof = dict(
            resume_source=args.resume_from,
            resume_params_sha256=KNOWN_PERSISTENT_24576_PARAMS_SHA,
            resume_file_sha256=KNOWN_PERSISTENT_24576_FILE_SHA,
            params_sha256_28672_runA=sha_a,
            params_sha256_28672_runB=sha_b,
            bit_exact_determinism=bool(deterministic),
            params_finite=bool(finite_a),
            entropy_28672_runA=float(ent_a),
            entropy_28672_runB=float(ent_b),
            entropy_in_engineering_range=bool(entropy_ok),
            grad_norm_max_28672_runA=float(gn_a),
            grad_norm_finite=bool(gradnorm_ok),
        )
        proof["EXACT_RESUME_PROOF_PASS"] = bool(deterministic and finite_a and entropy_ok and gradnorm_ok)
        with open(os.path.join(args.out_dir, "exact_resume_proof.json"), "w") as f:
            json.dump(proof, f, indent=2, sort_keys=True, default=str); f.write("\n")
        if not proof["EXACT_RESUME_PROOF_PASS"]:
            print("STATUS: %s_EXACT_RESUME_PROOF_FAIL — stopping; canonical fresh run required" % ARM,
                  flush=True)
            sys.exit(2)
        # continue from run B state (== run A state, bit-exact); per_update_metrics records
        # only the production continuation from here on (proof metrics live in exact_resume_proof.json)
        runner_state = rs_b
        assert len(METRICS_LOG) == 2 * UPDATES_PER_CHUNK, \
            "expected exactly %d proof metrics, got %d" % (2 * UPDATES_PER_CHUNK, len(METRICS_LOG))
        assert n_metrics_before == UPDATES_PER_CHUNK
        METRICS_LOG.clear()
        LS_START_HASH.clear(); LS_CARRIED_HASH.clear()
        print("[%s] EXACT_RESUME_PROOF_PASS — proceeding to 98304" % ARM, flush=True)

    # ============================================================== chunked training ======
    _upc = UPDATES_PER_CHUNK
    run_chunk = jax.jit(lambda rs: jax.lax.scan(_update_step, rs, None, _upc))
    chunk_records = []; t_start = time.time()

    # chunk count is computed from the CURRENT runner_state update counter — in resume mode
    # the exact-resume proof above has already advanced UPDATES_PER_CHUNK updates (24576->28672),
    # so using the pre-proof start_update here would overshoot the 98304 budget by one chunk.
    current_update_for_loop = int(runner_state[10])
    start_global = current_update_for_loop * STEPS_PER_UPDATE
    for ci in range((TOTAL_UPDATES - current_update_for_loop) // _upc):
        t0 = time.time()
        runner_state, _ = run_chunk(runner_state)
        jax.block_until_ready(runner_state)
        train_state = runner_state[0]
        update_step = int(runner_state[10])
        global_step = update_step * STEPS_PER_UPDATE
        elapsed = time.time() - t0
        params_sha = _params_sha(train_state.params)
        ent_now = METRICS_LOG[-1]["entropy"] if METRICS_LOG else float("nan")
        print("[%s] chunk %d done  global_step=%d update_count=%d params_sha=%s entropy=%.4f (%.1fs)"
              % (ARM, ci, global_step, update_step, params_sha[:16], ent_now, elapsed), flush=True)
        assert _finite(train_state.params), f"NUMERIC_FAIL params non-finite at {global_step}"
        assert _finite(train_state.opt_state), f"NUMERIC_FAIL opt_state non-finite at {global_step}"
        rec = dict(chunk=ci, global_step=global_step, update_count=update_step, params_sha256=params_sha,
                   opt_step=int(train_state.step), params_finite=True, opt_finite=True,
                   entropy=ent_now, elapsed_s=round(elapsed, 1))
        if args.mode == "fresh" and ci == 0:
            # fresh fallback path: 4096 engineering smoke gate before any long run
            gate = dict(gate_step=global_step, entropy=ent_now, params_finite=True, opt_finite=True,
                        entropy_ok=bool(np.isfinite(ent_now) and 0.01 < ent_now < 3.762))
            gate["ENGINEERING_GATE_4096_PASS"] = bool(gate["entropy_ok"])
            with open(os.path.join(args.out_dir, "engineering_gate_4096.json"), "w") as f:
                json.dump(gate, f, indent=2, sort_keys=True, default=str); f.write("\n")
            if not gate["ENGINEERING_GATE_4096_PASS"]:
                print("STATUS: %s_ENGINEERING_GATE_4096_FAIL — hard stop (engineering gate only, "
                      "NOT a performance early-stop)" % ARM, flush=True)
                sys.exit(2)
        if global_step in save_steps:
            pkl_path, ckpt_sha = _save_full(runner_state, global_step, params_sha)
            rec["checkpoint"] = pkl_path; rec["checkpoint_sha256"] = ckpt_sha; rec["roundtrip_ok"] = True
        chunk_records.append(rec)

    # ============================================================== done ======
    final_train_state = runner_state[0]
    final_global_step = int(runner_state[10]) * STEPS_PER_UPDATE
    status = f"{ARM}_TRAIN_OK" if final_global_step == TOTAL_STEPS else f"{ARM}_TRAIN_INCOMPLETE"
    _init_ls_hash = _leaf_hash(init_longstate(NUM_ENVS))
    carry_report = dict(carry_mode=CARRY_MODE, init_ls_hash=_init_ls_hash,
        n_rollouts=len(LS_START_HASH),
        rollout_start_hashes=list(LS_START_HASH),
        rollout_carried_hashes=list(LS_CARRIED_HASH))
    if CARRY_MODE == "RESET128":
        carry_report["boundary_clear_pass"] = bool(
            len(LS_START_HASH) >= 1 and all(h == _init_ls_hash for h in LS_START_HASH))
        carry_report["clear_nontrivial_pass"] = bool(
            any(h != _init_ls_hash for h in LS_CARRIED_HASH[1:]))
    else:
        carry_report["persistent_carry_nontrivial_pass"] = bool(
            any(h != _init_ls_hash for h in LS_START_HASH[1:]))
        carry_report["no_forced_boundary_clear"] = True
    summary = dict(label=ARM, arm=ARM, status=status,
        protocol=dict(start=("exact resume from identity-verified persistent@24576" if args.mode == "resume"
                             else "teacher ckpt17500 (inherited bit-for-bit) + fresh slow-GRU (zero-init gate)"),
                      algorithm="Original GTrXL-PPO + SlowGRU long-term channel",
                      xla_flags=os.environ["XLA_FLAGS"], seed=MASTER_SEED, lr=LR, adam_eps=ADAM_EPS,
                      gamma=GAMMA, gae_lambda=GAE_LAMBDA, num_envs=NUM_ENVS, rollout_steps=ROLLOUT_STEPS,
                      update_epochs=UPDATE_EPOCHS, num_minibatches=NUM_MINIBATCHES, clip_eps=CLIP_EPS,
                      vf_coef=VF_COEF, ent_coef=ENT_COEF, max_grad_norm=MAX_GRAD_NORM, anneal_lr=False,
                      optimistic_reset_ratio=OPTIMISTIC_RESET_RATIO, mode="score", replay="OFF",
                      vtrace="OFF", hindsight="OFF", novelty="OFF", nav_aux="OFF", egomap="OFF",
                      goal="DEFEAT_KOBOLD", stage="S4_dark native", total_env_steps=TOTAL_STEPS,
                      gpu_uuid=args.gpu_uuid, slow_interval=SLOW_INTERVAL, slow_dim=SLOW_DIM,
                      carry_mode=CARRY_MODE, cc3_mode=args.mode,
                      cc3_resume_source=(args.resume_from if args.mode == "resume" else None)),
        teacher_init_sha256=teacher_sha, init_params_sha256=init_params_sha, code_sha256=code_sha,
        config=cfg_dict, exact_resume_proof=proof, chunks=chunk_records,
        per_update_metrics=list(METRICS_LOG), carry_report=carry_report,
        elapsed_total_s=round(time.time() - t_start, 1),
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with open(os.path.join(args.out_dir, f"{ARM}_train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True, default=str); f.write("\n")
    with open(os.path.join(args.out_dir, f"{ARM}_per_update.jsonl"), "w") as f:
        for m in METRICS_LOG: f.write(json.dumps(m, default=str) + "\n")
    print("\nSTATUS: %s  final_global_step=%d params_sha=%s" % (
        status, final_global_step, _params_sha(final_train_state.params)), flush=True)
    sys.exit(0 if status.endswith("_TRAIN_OK") else 3)


if __name__ == "__main__":
    main()
