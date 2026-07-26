#!/usr/bin/env python3
"""P8-LONGMEM-SUMMARY — deterministic GTrXL-PPO trainer with a long-term summary channel.

Built VERBATIM on the validated Control continuous trainer (run_control_continuous_98304.py):
same _update_step/_env_step/_loss_fn memory machinery, same env wrapper / optax / Transition /
chunked-scan outer loop, same full-state save + bit-exact roundtrip. The ONLY changes:

  1. Network = ActorCriticLongMem (original GTrXL inherited bit-for-bit + a long-term summary
     channel; zero-init additive path so feature-off == teacher exactly).
  2. runner_state gains TWO fields carried across rollouts:
        - longstate : the long-term summary state (summaries/valid/accum_sum/accum_count),
                      evolved by forward_eval, reset on true_done only.
        - true_done : info["returned_episode"] from the previous step (the long-term reset
                      signal ENTERING the current step), carried exactly like `done`.
  3. _env_step calls forward_eval(..., longstate, true_done) -> (pi, value, memories_out, ls_new).
  4. GAE bootstrap value uses forward_eval with the final longstate/true_done.
  5. _loss_fn calls model_forward_train_longmem(..., true_done_ET, ls_prev). true_done_ET is the
     true_done ENTERING each rollout step = concat([td_start, info["returned_episode"][:, :T-1]]).
     ls_prev (the long-state at rollout start) and td_start are permuted/reshaped over the ENV
     axis with the SAME permutation as traj_batch, so each minibatch gets its own envs' long-state.
  6. Initial TrainState is built from the behaviorally-distilled init (P8_DISTILLED_INIT/params.pkl)
     with a FRESH Adam (step=0), exactly as the control builds from ckpt17500.

BIT-EXACT CHUNKING (same argument as control): scan(body, s0, 2k) == scan(body, scan(body,s0,k), k)
under XLA det-ops, so updates_per_chunk=2 (4096-step chunks, enabling the 4096 save node) is
bit-identical to a single continuous run, and a checkpoint resumed mid-run reproduces the
continuous trajectory exactly (verified by the exact-resume test).

FROZEN protocol (must NOT change): num_envs=16, num_steps=128, seed=42, LR=2e-5, Adam eps=1e-5,
gamma=0.999, gae_lambda=0.8, clip=0.2, vf=0.5, ent=0.002, gradnorm=1.0, anneal_lr=False,
window_mem=128, window_grad=64, heads=8, layers=2, embed/qkv=256, hidden=256, gating=True,
gating_bias=2.0, optimistic_reset_ratio=16, total=98304 (48 updates). GPU2 only.
"""
import os
GPU_UUID = "GPU-8df11537-ab79-722d-606f-411966196c4c"   # GPU2 ONLY
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"
import sys, json, argparse, hashlib, time, pickle, shutil

P8_SRC = "/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu2_p8_longmemory/src"
V7_SRC = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src"
V7 = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB"
for p in (P8_SRC, V7_SRC, V7):
    if p not in sys.path:
        sys.path.insert(0, p)

import wandb
wandb.init(mode="disabled")

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from craftax.craftax.constants import Achievement
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from dicode.network import Transition
from dicode.task_utils import get_achievement_multi_hot
from dicode.wrappers_cl import DistributedMultiTaskOptimisticLogWrapper
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv
from p8_network import ActorCriticLongMem, init_longstate

METRICS_LOG = []

# --------------------------------------------------------------------- frozen anchors
DISTILL_PARAMS = "/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu2_p8_longmemory/distill/P8_DISTILLED_INIT/params.pkl"
DISTILL_SUMMARY = "/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu2_p8_longmemory/distill/P8_DISTILLED_INIT/summary.json"
S4_TASK_PATH = "/home/oseasy/experiments/p2_v1_20260722/evidence/s4_task_code.py"
TEACHER_CKPT17500_SHA = "d4e85af58b7f87d689fadea12eec70c852fa098a09f5ea8907448684b3bf60f5"

# --------------------------------------------------------------------- frozen protocol
NUM_ENVS = 16
ROLLOUT_STEPS = 128
STEPS_PER_UPDATE = NUM_ENVS * ROLLOUT_STEPS      # 2048
TOTAL_STEPS = 98304
TOTAL_UPDATES = TOTAL_STEPS // STEPS_PER_UPDATE   # 48
UPDATES_PER_CHUNK = 2                             # 4096 env steps per chunk -> 4096 save node
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
SAVE_STEPS = (4096, 24576, 49152, 73728, 98304)
LONG_K = 64
LONG_N = 16


class Cfg:
    lr = LR
    num_envs = NUM_ENVS
    num_steps = ROLLOUT_STEPS
    update_epochs = UPDATE_EPOCHS
    num_minibatches = NUM_MINIBATCHES
    gamma = GAMMA
    gae_lambda = GAE_LAMBDA
    clip_eps = CLIP_EPS
    ent_coef = ENT_COEF
    vf_coef = VF_COEF
    max_grad_norm = MAX_GRAD_NORM
    activation = "relu"
    anneal_lr = False
    qkv_features = QKV_FEATURES
    embed_size = EMBED_SIZE
    num_heads = NUM_HEADS
    num_layers = NUM_LAYERS
    hidden_layers = HIDDEN_LAYERS
    window_mem = WINDOW_MEM
    window_grad = WINDOW_GRAD
    gating = True
    gating_bias = 2.0
    condition_on_task = True
    optimistic_reset_ratio = OPTIMISTIC_RESET_RATIO
    mode = "score"
    bonus_type = "none"
    dynamic_bonus_k = 0.0
    completion_bonus_scale = 0.0
    completion_bonus_min = 0.0
    value_target_clip_min = VT_CLIP_MIN
    value_target_clip_max = VT_CLIP_MAX

    def get(self, key, default=None):
        return getattr(self, key, default)


# ------------------------------------------------------------- hashing helpers
def _params_sha(params):
    h = hashlib.sha256()
    for v in jax.tree_util.tree_leaves(params):
        h.update(np.ascontiguousarray(np.asarray(v)).tobytes())
    return h.hexdigest()


def _leaf_hash(pytree):
    h = hashlib.sha256()
    for l in jax.tree_util.tree_leaves(pytree):
        a = np.asarray(l)
        h.update(str(a.shape).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _leaves_equal(a, b):
    la = jax.tree_util.tree_leaves(a)
    lb = jax.tree_util.tree_leaves(b)
    if len(la) != len(lb):
        return False
    return all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb))


def _finite(pytree):
    return bool(all(np.all(np.isfinite(np.asarray(v)))
                    for v in jax.tree_util.tree_leaves(pytree)
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
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckpt_root", required=True)
    ap.add_argument("--num_chunks", type=int, default=TOTAL_UPDATES // UPDATES_PER_CHUNK)
    ap.add_argument("--updates_per_chunk", type=int, default=UPDATES_PER_CHUNK)
    ap.add_argument("--resume_from", default=None,
                    help="path to a full_state.pkl to resume from (exact resume).")
    ap.add_argument("--save_steps", default=",".join(map(str, SAVE_STEPS)))
    args = ap.parse_args()
    args.out_dir = os.path.abspath(args.out_dir)
    args.ckpt_root = os.path.abspath(args.ckpt_root)
    save_steps = tuple(int(x) for x in args.save_steps.split(",") if x.strip())

    devs = jax.local_devices()
    print(f"[p8] devices={devs} (expect GPU2 only)", flush=True)
    assert len(devs) == 1, f"expected exactly 1 visible device (GPU2), got {devs}"

    for d in (args.out_dir, args.ckpt_root):
        os.makedirs(d, exist_ok=True)
    free_gb = shutil.disk_usage(args.ckpt_root).free / 1024 ** 3
    print(f"[p8] disk free = {free_gb:.1f} GB", flush=True)
    assert free_gb > 10.0, f"HARD STOP low disk {free_gb:.1f} GB"

    cfg = Cfg()
    cfg.training = cfg

    code_sha = {
        "launcher": _sha_file(os.path.abspath(__file__)),
        "p8_network": _sha_file(os.path.join(P8_SRC, "p8_network.py")),
        "s4_task": _sha_file(S4_TASK_PATH),
    }

    # ---- S4_dark env (EXACT control construction) ----
    ach_table = jnp.array([get_achievement_multi_hot([Achievement.DEFEAT_KOBOLD])], dtype=jnp.float32)
    EMB = int(ach_table.shape[1])
    with open(S4_TASK_PATH) as f:
        s4_code = f.read()
    ns = {}; exec(s4_code, ns); Task = ns["Env"]
    num_tasks = 1
    static_env_params = StaticEnvParams()
    env_params_ctor = EnvParams(max_timesteps=4096)
    base_env = MultiTaskMiniCraftaxEnv(
        [Task], static_env_params, env_params_ctor, cfg.condition_on_task,
        conditioning_type="embedding", embedding_size=EMB,
        completion_bonus_scale=cfg.completion_bonus_scale,
        completion_bonus_min=cfg.completion_bonus_min,
        bonus_type=cfg.bonus_type, dynamic_bonus_k=cfg.dynamic_bonus_k)
    task_distribution_proportions = jnp.ones(num_tasks) / num_tasks
    env = DistributedMultiTaskOptimisticLogWrapper(
        base_env, jax.random.PRNGKey(0), cfg.num_envs, num_tasks,
        cfg.optimistic_reset_ratio, task_distribution_proportions, ach_table)
    env_params = env.default_params
    obs_dim = int(env.observation_space(env_params).shape[0])
    action_dim = int(env.action_space(env_params).n)
    print(f"[p8] S4_dark env: obs_dim={obs_dim} action_dim={action_dim} emb={EMB}", flush=True)
    assert action_dim == 43 and obs_dim == 8335 and EMB == 67

    # ---- network + optimizer (anneal_lr=False) ----
    network = ActorCriticLongMem(
        action_dim=action_dim, activation=cfg.activation, hidden_layers=cfg.hidden_layers,
        encoder_size=cfg.embed_size, num_heads=cfg.num_heads, qkv_features=cfg.qkv_features,
        num_layers=cfg.num_layers, gating=cfg.gating, gating_bias=cfg.gating_bias,
        use_longmem=True)
    tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm),
                     optax.adam(cfg.lr, eps=ADAM_EPS))

    current_original_return = 0.0
    vt_clip_min = VT_CLIP_MIN
    vt_clip_max = VT_CLIP_MAX
    initial_global_update_step = 0
    config = cfg

    indices_select = lambda x, y: x[y]
    batch_indices_select = jax.vmap(indices_select)
    roll_vmap = jax.vmap(jnp.roll, in_axes=(-2, 0, None), out_axes=-2)
    batchify = lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1],) + x.shape[2:])

    def _log_callback(metrics, step):
        t_loss, v_loss, a_loss, ent, g_norm_mean, g_norm_max = metrics
        METRICS_LOG.append(dict(
            total_loss=float(np.asarray(t_loss)), value_loss=float(np.asarray(v_loss)),
            actor_loss=float(np.asarray(a_loss)), entropy=float(np.asarray(ent)),
            grad_norm_mean=float(np.asarray(g_norm_mean)),
            grad_norm_max=float(np.asarray(g_norm_max)),
            update_step=int(np.asarray(step))))

    # ============================================================== _update_step ======
    def _update_step(runner_state, unused_scan_input):
        # === A. COLLECT TRAJECTORIES ===
        def _env_step(runner_state, _):
            (train_state, env_state, memories, memories_mask, memories_mask_idx,
             last_obs, done, true_done, longstate, step_env_currentloop, update_step, rng) = runner_state

            # short-term mask reset on RAW done (UNCHANGED from control)
            memories_mask_idx = jnp.where(
                done, config.window_mem, jnp.clip(memories_mask_idx - 1, 0, config.window_mem))
            memories_mask = jnp.where(
                done[:, None, None, None],
                jnp.zeros((config.num_envs, config.num_heads, 1, config.window_mem + 1),
                          dtype=jnp.bool_),
                memories_mask)
            memories_mask_idx_ohot = jax.nn.one_hot(memories_mask_idx, config.window_mem + 1)
            memories_mask_idx_ohot = memories_mask_idx_ohot[:, None, None, :].repeat(config.num_heads, 1)
            memories_mask = jnp.logical_or(memories_mask, memories_mask_idx_ohot)

            rng, _rng = jax.random.split(rng)
            # long-term reset on TRUE_DONE inside forward_eval
            pi, value, memories_out, ls_new = network.apply(
                train_state.params, memories, last_obs, memories_mask, longstate, true_done,
                method=network.forward_eval)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)

            memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(memories_out)

            rng, _rng = jax.random.split(rng)
            obsv, env_state, reward, done, info = env.step(_rng, env_state, action, env_params)
            env_state = env_state.replace(
                running_original_return=jnp.full(
                    (config.num_envs,), current_original_return, dtype=jnp.float32))
            true_done_next = info["returned_episode"]   # true_done ENTERING the next step

            memory_indices = jnp.arange(0, config.window_mem)[None, :] + \
                step_env_currentloop * jnp.ones((config.num_envs, 1), dtype=jnp.int32)

            transition = Transition(
                done, action, value, reward, log_prob,
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

        # === B. CALCULATE ADVANTAGES (GAE) ===
        _, last_val, _, _ = network.apply(
            train_state.params, final_memories, final_obs, final_mask, final_longstate,
            final_true_done, method=network.forward_eval)

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(carry, transition):
                gae, next_value = carry
                done, value, reward = transition.done, transition.value, transition.reward
                delta = reward + config.gamma * next_value * (1 - done) - value
                gae = delta + config.gamma * config.gae_lambda * (1 - done) * gae
                return (gae, value), gae
            _, advantages = jax.lax.scan(
                _get_advantages, (jnp.zeros_like(last_val), last_val),
                traj_batch, reverse=True, unroll=16)
            return advantages, jnp.clip(advantages + traj_batch.value, vt_clip_min, vt_clip_max)

        advantages, targets = _calculate_gae(traj_batch, last_val)

        # === C. UPDATE NETWORK (TRANSFORMER LOSS + LONG-MEM CONTEXT) ===
        memories_batch = jnp.concatenate(
            [jnp.swapaxes(memories_previous, 0, 1), memories_batch], axis=0)

        def _update_epoch(update_state, unused):
            def _update_minbatch(train_state, batch_info):
                traj_batch, memories_batch, advantages, targets, ls_prev_mb, td_start_mb = batch_info

                def _loss_fn(params, traj_batch, memories_batch, gae, targets, ls_prev, td_start):
                    memories_batch = batch_indices_select(
                        memories_batch, traj_batch.memories_indices[:, :: config.window_grad])
                    memories_batch = batchify(memories_batch)
                    memories_mask = traj_batch.memories_mask.reshape(
                        (-1, config.window_grad) + traj_batch.memories_mask.shape[2:])
                    memories_mask = jnp.swapaxes(memories_mask, 1, 2)
                    memories_mask = jnp.concatenate(
                        (memories_mask,
                         jnp.zeros(memories_mask.shape[:-1] + (config.window_grad - 1,),
                                   dtype=jnp.bool_)),
                        axis=-1)
                    memories_mask = roll_vmap(memories_mask, jnp.arange(0, config.window_grad), -1)
                    obs = traj_batch.obs.reshape((-1, config.window_grad) + traj_batch.obs.shape[2:])
                    traj_batch_r, targets_r, gae_r = jax.tree_util.tree_map(
                        lambda x: jnp.reshape(x, (-1, config.window_grad) + x.shape[2:]),
                        (traj_batch, targets, gae))
                    # true_done ENTERING each rollout step (E_mb, T)
                    info_re = traj_batch.info["returned_episode"]            # (E_mb, T)
                    T_roll = info_re.shape[1]
                    true_done_ET = jnp.concatenate(
                        [td_start[:, None], info_re[:, : T_roll - 1]], axis=1)
                    pi, value = network.apply(
                        params, memories_batch, obs, memories_mask, true_done_ET, ls_prev,
                        method=network.model_forward_train_longmem)
                    log_prob = pi.log_prob(traj_batch_r.action)
                    value_pred_clipped = traj_batch_r.value + (value - traj_batch_r.value).clip(
                        -config.clip_eps, config.clip_eps)
                    value_losses = jnp.square(value - targets_r)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets_r)
                    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    ratio = jnp.exp(log_prob - traj_batch_r.log_prob)
                    gae_r = (gae_r - gae_r.mean()) / (gae_r.std() + 1e-8)
                    loss_actor1 = ratio * gae_r
                    loss_actor2 = (jnp.clip(ratio, 1.0 - config.clip_eps,
                                            1.0 + config.clip_eps) * gae_r)
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                    entropy = pi.entropy().mean()
                    total_loss = (loss_actor + config.vf_coef * value_loss
                                  - config.ent_coef * entropy)
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
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch)
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, [config.num_minibatches, -1] + list(x.shape[1:])),
                shuffled_batch)
            # permute + minibatch-reshape longstate_prev & td_start over the ENV axis (axis 0),
            # with the SAME permutation as traj_batch (env axis is already axis 0 here).
            ls_prev_shuffled = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), ls_prev)
            ls_prev_mb = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, [config.num_minibatches, -1] + list(x.shape[1:])),
                ls_prev_shuffled)
            td_start_shuffled = jnp.take(td_start, permutation, axis=0)
            td_start_mb = jnp.reshape(td_start_shuffled, [config.num_minibatches, -1])
            mb_full = minibatches + (ls_prev_mb, td_start_mb)
            train_state, (total_loss, value_loss, loss_actor, entropy, grad_norm) = \
                jax.lax.scan(_update_minbatch, train_state, mb_full)
            return (train_state, traj_batch, memories_batch, advantages, targets,
                    ls_prev, td_start, update_step, rng), \
                   (total_loss, value_loss, loss_actor, entropy, grad_norm)

        update_state = (train_state, traj_batch, memories_batch, advantages, targets,
                        longstate_previous, true_done_start, update_step, rng)
        update_state, rl_info = jax.lax.scan(_update_epoch, update_state, None, config.update_epochs)

        losses_and_ent = rl_info[:4]
        grad_norms = rl_info[4]
        losses_mean = jax.tree_util.tree_map(lambda x: jnp.mean(x), losses_and_ent)
        gn_mean = jnp.mean(grad_norms)
        gn_max = jnp.max(grad_norms)
        metrics_to_log = (*losses_mean, gn_mean, gn_max)
        current_step = initial_global_update_step + update_step
        jax.debug.callback(_log_callback, metrics_to_log, current_step)

        train_state = update_state[0]
        rng = update_state[-1]
        next_runner_state = (train_state, final_env_state, final_memories, final_mask,
                             final_mask_idx, final_obs, done, final_true_done, final_longstate,
                             0, update_step + 1, rng)
        return next_runner_state, None

    # ============================================================== initial / resumed state
    distill_summary = json.load(open(DISTILL_SUMMARY))
    distilled_params = jax.tree_util.tree_map(jnp.asarray, pickle.load(open(DISTILL_PARAMS, "rb")))
    init_params_sha = _params_sha(distilled_params)
    print(f"[p8] distilled init params: leaves={len(jax.tree_util.tree_leaves(distilled_params))} "
          f"sha={init_params_sha[:16]} teacher_sha={distill_summary.get('teacher_sha','?')[:16]} "
          f"held_out_kl={distill_summary.get('held_out_kl','?')}", flush=True)

    if args.resume_from:
        print(f"[p8] RESUME from {args.resume_from}", flush=True)
        with open(args.resume_from, "rb") as f:
            rd = pickle.load(f)
        ts = TrainState.create(apply_fn=network.apply, params=_unpack(rd["params"]),
                               tx=tx).replace(opt_state=_unpack(rd["opt_state"]),
                                              step=jnp.asarray(rd["opt_step"], jnp.int32))
        env_state_r = _unpack(rd["env_state"])
        runner_state = (ts, env_state_r, jnp.asarray(rd["memories"]), jnp.asarray(rd["memories_mask"]),
                        jnp.asarray(rd["memories_mask_idx"]), jnp.asarray(rd["obs"]),
                        jnp.asarray(rd["done"]), jnp.asarray(rd["true_done"]),
                        _unpack(rd["longstate"]), int(rd["step_env_currentloop"]),
                        int(rd["update_step"]), jnp.asarray(rd["rng"]))
        start_update = int(rd["update_step"])
        print(f"[p8] resumed at update_step={start_update} "
              f"params_sha={_params_sha(ts.params)[:16]}", flush=True)
    else:
        ts = TrainState.create(apply_fn=network.apply, params=distilled_params, tx=tx)
        assert int(ts.step) == 0
        rng0 = jax.random.PRNGKey(MASTER_SEED)
        rng0, _rng = jax.random.split(rng0)
        obsv, env_state = env.reset(_rng, env_params)
        env_state = env_state.replace(
            running_original_return=jnp.full((cfg.num_envs,), current_original_return, dtype=jnp.float32))
        memories = jnp.zeros((cfg.num_envs, cfg.window_mem, cfg.num_layers, cfg.embed_size))
        memories_mask = jnp.zeros((cfg.num_envs, cfg.num_heads, 1, cfg.window_mem + 1), dtype=jnp.bool_)
        memories_mask_idx = jnp.zeros((cfg.num_envs,), dtype=jnp.int32) + (cfg.window_mem + 1)
        done = jnp.zeros((cfg.num_envs,), dtype=jnp.bool_)
        true_done = jnp.zeros((cfg.num_envs,), dtype=jnp.bool_)
        longstate = init_longstate(cfg.num_envs)
        rng0, _rng = jax.random.split(rng0)
        runner_state = (ts, env_state, memories, memories_mask, memories_mask_idx,
                        obsv, done, true_done, longstate, 0, 0, _rng)
        start_update = 0

    # ============================================================== chunked training ======
    _upc = args.updates_per_chunk
    run_chunk = jax.jit(lambda rs: jax.lax.scan(_update_step, rs, None, _upc))

    cfg_dict = {k: v for k, v in vars(Cfg).items() if not k.startswith("__") and not callable(v)}
    chunk_records = []
    t_start = time.time()

    for ci in range(args.num_chunks):
        t0 = time.time()
        runner_state, _ = run_chunk(runner_state)
        jax.block_until_ready(runner_state)
        train_state = runner_state[0]
        update_step = int(runner_state[10])
        global_step = update_step * STEPS_PER_UPDATE
        elapsed = time.time() - t0
        params_sha = _params_sha(train_state.params)
        ent_now = METRICS_LOG[-1]["entropy"] if METRICS_LOG else float("nan")
        print(f"[p8] chunk {ci} done  global_step={global_step} update_count={update_step} "
              f"params_sha={params_sha[:16]} entropy={ent_now:.4f}  ({elapsed:.1f}s)", flush=True)

        # numeric / entropy collapse guard
        assert _finite(train_state.params), f"NUMERIC_FAIL params non-finite at {global_step}"
        assert _finite(train_state.opt_state), f"NUMERIC_FAIL opt_state non-finite at {global_step}"

        rec = dict(chunk=ci, global_step=global_step, update_count=update_step,
                   params_sha256=params_sha, opt_step=int(train_state.step),
                   params_finite=_finite(train_state.params),
                   opt_finite=_finite(train_state.opt_state),
                   entropy=ent_now, elapsed_s=round(elapsed, 1))

        if global_step in save_steps:
            ckpt_dir = os.path.join(args.ckpt_root, str(global_step))
            os.makedirs(ckpt_dir, exist_ok=True)
            (train_state_, env_state_, memories_, mask_, mask_idx_, obs_, done_,
             true_done_, longstate_, step_loop_, update_step_, rng_) = runner_state
            full = dict(
                params=_pack(train_state_.params),
                opt_state=_pack(train_state_.opt_state),
                opt_step=int(train_state_.step),
                env_state=_pack(env_state_),
                memories=np.asarray(memories_),
                memories_mask=np.asarray(mask_),
                memories_mask_idx=np.asarray(mask_idx_),
                obs=np.asarray(obs_),
                done=np.asarray(done_),
                true_done=np.asarray(true_done_),
                longstate=_pack(longstate_),
                step_env_currentloop=int(step_loop_),
                update_step=int(update_step_),
                rng=np.asarray(rng_),
                global_step=global_step,
                update_count=int(update_step_),
                config=cfg_dict,
                code_sha256=code_sha,
                manifest=dict(label="P8_LONGMEM_SUMMARY", arm="P8_LONGMEM",
                              rng_seed=MASTER_SEED, xla_flags=os.environ["XLA_FLAGS"],
                              gpu_uuid=GPU_UUID, params_sha256=params_sha,
                              opt_state_leaf_hash=_leaf_hash(train_state_.opt_state),
                              longstate_leaf_hash=_leaf_hash(longstate_),
                              distilled_init_sha256=init_params_sha))
            pkl_path = os.path.join(ckpt_dir, "full_state.pkl")
            tmp_path = pkl_path + ".tmp"
            with open(tmp_path, "wb") as f:
                pickle.dump(full, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, pkl_path)
            rec["checkpoint"] = pkl_path
            rec["checkpoint_sha256"] = _sha_file(pkl_path)
            print(f"[p8] saved FULL state -> {pkl_path} (sha={rec['checkpoint_sha256'][:16]})", flush=True)

            # non-disturbing save->restore roundtrip (separate object)
            with open(pkl_path, "rb") as f:
                rd = pickle.load(f)
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
            rec["roundtrip_ok"] = bool(rt_ok)
            print(f"[p8] roundtrip@{global_step} ok={rt_ok} (live runner_state untouched)", flush=True)
            assert rt_ok, f"FAIL roundtrip bit-exact at {global_step}"

        chunk_records.append(rec)

    # ============================================================== done ======
    final_train_state = runner_state[0]
    final_update_step = int(runner_state[10])
    final_global_step = final_update_step * STEPS_PER_UPDATE
    status = "P8_TRAIN_OK" if final_global_step == (start_update + args.num_chunks * _upc) * STEPS_PER_UPDATE \
        else "P8_TRAIN_INCOMPLETE"
    summary = dict(
        label="P8_LONGMEM_SUMMARY", arm="P8_LONGMEM", status=status,
        protocol=dict(start="P8_DISTILLED_INIT(teacher ckpt17500)", algorithm="GTrXL-PPO + long-mem summary",
                      xla_flags=os.environ["XLA_FLAGS"], seed=MASTER_SEED, lr=LR, adam_eps=ADAM_EPS,
                      gamma=GAMMA, gae_lambda=GAE_LAMBDA, num_envs=NUM_ENVS, rollout_steps=ROLLOUT_STEPS,
                      update_epochs=UPDATE_EPOCHS, num_minibatches=NUM_MINIBATCHES, clip_eps=CLIP_EPS,
                      vf_coef=VF_COEF, ent_coef=ENT_COEF, max_grad_norm=MAX_GRAD_NORM, anneal_lr=False,
                      optimistic_reset_ratio=OPTIMISTIC_RESET_RATIO, mode="score", replay="OFF",
                      hindsight="OFF", novelty="OFF", nav_aux="OFF", goal="DEFEAT_KOBOLD",
                      stage="S4_dark native", total_env_steps=TOTAL_STEPS, gpu_uuid=GPU_UUID,
                      long_K=LONG_K, long_N=LONG_N, long_coverage=LONG_K * LONG_N),
        distilled_init=dict(params_sha256=init_params_sha, summary=distill_summary),
        code_sha256=code_sha, config=cfg_dict, chunks=chunk_records,
        per_update_metrics=list(METRICS_LOG),
        elapsed_total_s=round(time.time() - t_start, 1),
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with open(os.path.join(args.out_dir, "p8_train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    with open(os.path.join(args.out_dir, "p8_per_update.jsonl"), "w") as f:
        for m in METRICS_LOG:
            f.write(json.dumps(m, default=str) + "\n")
    print(f"\nSTATUS: {status}  final_global_step={final_global_step} "
          f"params_sha={_params_sha(final_train_state.params)}", flush=True)


if __name__ == "__main__":
    main()
