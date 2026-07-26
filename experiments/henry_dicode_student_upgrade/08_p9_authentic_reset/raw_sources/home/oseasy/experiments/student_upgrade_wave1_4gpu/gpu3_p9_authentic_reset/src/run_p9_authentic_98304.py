#!/usr/bin/env python3
"""P9-AUTHENTIC-RESET — deterministic GTrXL-PPO trainer with authentic reached-state reset
injection (GPU3). NO network change: original ActorCriticTransformer from healthy ckpt17500.

Built VERBATIM on the validated control/P8 continuous trainer machinery: same _update_step /
_env_step / _loss_fn short-term-memory machinery, same env wrapper / optax / Transition /
chunked-scan outer loop, same full-state save + bit-exact roundtrip + exact resume. The ONLY
change is the episode-reset SOURCE inside _env_step (design §3):

  At each episode boundary (raw `done`), for each env independently, draw Bernoulli(0.5):
    natural   (50%): keep the wrapper's fresh Stage4 reset (unchanged path).
    authentic (50%): overwrite the carried (env_state [full LogEnvState], obs, memories,
                     memories_mask, memories_mask_idx) with a uniformly-sampled snapshot from
                     the READ-ONLY authentic library, and set the carried done=False so the next
                     step's short-term mask-reset preserves the injected reached-state context.
  The 50/50 mix is FIXED for the whole run (never tuned by results). The PPO objective, the
  network, and the natural-start branch are untouched. The authentic branch only changes WHERE
  an episode starts.

The injection draws (Bernoulli + snapshot index) are derived from the carried main `rng`
(jax.random.split inside _env_step), so they are fully determined by the saved rng -> exact
resume holds (P9 continuous == P9 chunked == P9 resumed, verified by the exact-resume smoke).

The Transition for the boundary step is built with the ORIGINAL `done` (the episode genuinely
ended -> GAE bootstrap cut is correct); only the CARRY's done is masked for injected envs.

BIT-EXACT CHUNKING (same argument as control/P8): scan(body,s0,2k)==scan(body,scan(body,s0,k),k)
under XLA det-ops, so updates_per_chunk=2 (4096-step chunks) is bit-identical to continuous.

FROZEN protocol (must NOT change): num_envs=16, num_steps=128, seed=42, LR=2e-5, Adam eps=1e-5,
gamma=0.999, gae_lambda=0.8, clip=0.2, vf=0.5, ent=0.002, gradnorm=1.0, anneal_lr=False,
window_mem=128, window_grad=64, heads=8, layers=2, embed/qkv=256, hidden=256, gating=True,
gating_bias=2.0, optimistic_reset_ratio=16, total=98304 (48 updates). GPU3 only.
"""
import os
GPU_UUID = "GPU-f56a59b4-99f3-f2e5-11c6-d01685de8abd"   # GPU3 ONLY
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"
import sys, json, argparse, hashlib, time, pickle, shutil

P9_SRC = "/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu3_p9_authentic_reset/src"
V7_SRC = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src"
V7 = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB"
for p in (P9_SRC, V7_SRC, V7):
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
from dicode.network import ActorCriticTransformer, Transition
from dicode.task_utils import get_achievement_multi_hot
from dicode.utils.general.train_state_utils import load_weights_only
from dicode.wrappers_cl import DistributedMultiTaskOptimisticLogWrapper
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv

METRICS_LOG = []

# --------------------------------------------------------------------- frozen anchors
TEACHER_CKPT = ("/home/oseasy/incoming/henry_work_20260721T105300/extracted/"
                "Henry_work/base_ckpt_v7fix55_armA_s0/rl_checkpoints/17500")
LIB_PATH = "/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu3_p9_authentic_reset/lib/p9_library.pkl"
S4_TASK_PATH = "/home/oseasy/experiments/p2_v1_20260722/evidence/s4_task_code.py"

# --------------------------------------------------------------------- frozen protocol
NUM_ENVS = 16
ROLLOUT_STEPS = 128
STEPS_PER_UPDATE = NUM_ENVS * ROLLOUT_STEPS      # 2048
TOTAL_STEPS = 98304
TOTAL_UPDATES = TOTAL_STEPS // STEPS_PER_UPDATE   # 48
UPDATES_PER_CHUNK = 2
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
AUTHENTIC_PROB = 0.5   # FROZEN reset mix; never tuned by results


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


# ------------------------------------------------------------- hashing helpers
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
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckpt_root", required=True)
    ap.add_argument("--num_chunks", type=int, default=TOTAL_UPDATES // UPDATES_PER_CHUNK)
    ap.add_argument("--updates_per_chunk", type=int, default=UPDATES_PER_CHUNK)
    ap.add_argument("--resume_from", default=None)
    ap.add_argument("--save_steps", default=",".join(map(str, SAVE_STEPS)))
    ap.add_argument("--lib_path", default=LIB_PATH)
    args = ap.parse_args()
    args.out_dir = os.path.abspath(args.out_dir); args.ckpt_root = os.path.abspath(args.ckpt_root)
    save_steps = tuple(int(x) for x in args.save_steps.split(",") if x.strip())

    devs = jax.local_devices()
    print(f"[p9] devices={devs} (expect GPU3 only)", flush=True)
    assert len(devs) == 1, f"expected exactly 1 visible device (GPU3), got {devs}"
    for d in (args.out_dir, args.ckpt_root): os.makedirs(d, exist_ok=True)
    free_gb = shutil.disk_usage(args.ckpt_root).free / 1024 ** 3
    print(f"[p9] disk free = {free_gb:.1f} GB", flush=True)
    assert free_gb > 10.0, f"HARD STOP low disk {free_gb:.1f} GB"

    cfg = Cfg(); cfg.training = cfg
    code_sha = {"launcher": _sha_file(os.path.abspath(__file__)), "s4_task": _sha_file(S4_TASK_PATH),
                "library": _sha_file(args.lib_path)}

    # ---- S4_dark env (EXACT control construction) ----
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
        cfg.optimistic_reset_ratio, jnp.ones(1) / 1, ach_table)
    env_params = env.default_params
    obs_dim = int(env.observation_space(env_params).shape[0])
    action_dim = int(env.action_space(env_params).n)
    print(f"[p9] S4_dark env: obs_dim={obs_dim} action_dim={action_dim} emb={EMB}", flush=True)
    assert action_dim == 43 and obs_dim == 8335 and EMB == 67

    # ---- network (ORIGINAL, unchanged) + optimizer ----
    network = ActorCriticTransformer(action_dim=action_dim, activation=cfg.activation,
        hidden_layers=cfg.hidden_layers, encoder_size=cfg.embed_size, num_heads=cfg.num_heads,
        qkv_features=cfg.qkv_features, num_layers=cfg.num_layers, gating=cfg.gating, gating_bias=cfg.gating_bias)
    tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.lr, eps=ADAM_EPS))

    # ---- authentic library -> closed-over constant pytrees (leading axis LIB_N) ----
    lib = pickle.load(open(args.lib_path, "rb"))
    snaps = lib["snaps"]; LIB_N = len(snaps)
    assert LIB_N > 0, "EMPTY authentic library — run p9_capture.py first"
    lib_logstate = jax.tree_util.tree_map(lambda *xs: jnp.asarray(jnp.stack(xs, 0)),
                                          *[s["logstate"] for s in snaps])
    lib_obs   = jnp.stack([jnp.asarray(s["obs"]) for s in snaps], 0)
    lib_mem   = jnp.stack([jnp.asarray(s["memories"]) for s in snaps], 0)
    lib_mask  = jnp.stack([jnp.asarray(s["mask"]) for s in snaps], 0)
    lib_midx  = jnp.stack([jnp.asarray(s["midx"], jnp.int32) for s in snaps], 0)
    lib_done  = jnp.stack([jnp.asarray(s["done"], jnp.bool_) for s in snaps], 0)
    lib_bytes = sum(int(np.prod(l.shape)) * l.dtype.itemsize for l in jax.tree_util.tree_leaves(lib_logstate)) \
                + int(np.prod(lib_obs.shape)) * 4 + int(np.prod(lib_mem.shape)) * 4
    print(f"[p9] authentic library: LIB_N={LIB_N} counts={lib['counts']} "
          f"~{lib_bytes/1e6:.1f}MB on-device constants", flush=True)

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

    def _where_env(cond, new, old):
        """jnp.where with a per-env (num_envs,) condition broadcast to new's ndim."""
        cb = cond.reshape((config.num_envs,) + (1,) * (new.ndim - 1))
        return jnp.where(cb, new, old)

    # ============================================================== _update_step ======
    def _update_step(runner_state, unused_scan_input):
        def _env_step(runner_state, _):
            (train_state, env_state, memories, memories_mask, memories_mask_idx,
             last_obs, done, step_env_currentloop, update_step, rng) = runner_state

            memories_mask_idx = jnp.where(done, config.window_mem,
                                          jnp.clip(memories_mask_idx - 1, 0, config.window_mem))
            memories_mask = jnp.where(done[:, None, None, None],
                jnp.zeros((config.num_envs, config.num_heads, 1, config.window_mem + 1), dtype=jnp.bool_),
                memories_mask)
            ohot = jax.nn.one_hot(memories_mask_idx, config.window_mem + 1)[:, None, None, :].repeat(config.num_heads, 1)
            memories_mask = jnp.logical_or(memories_mask, ohot)

            rng, _rng = jax.random.split(rng)
            pi, value, memories_out = network.apply(train_state.params, memories, last_obs, memories_mask,
                                                    method=network.model_forward_eval)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)
            memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(memories_out)

            rng, _rng = jax.random.split(rng)
            obsv, env_state, reward, done, info = env.step(_rng, env_state, action, env_params)
            env_state = env_state.replace(running_original_return=jnp.full(
                (config.num_envs,), current_original_return, dtype=jnp.float32))

            memory_indices = jnp.arange(0, config.window_mem)[None, :] + \
                step_env_currentloop * jnp.ones((config.num_envs, 1), dtype=jnp.int32)
            # Transition uses the ORIGINAL done (boundary recorded for GAE) ----
            transition = Transition(done, action, value, reward, log_prob,
                                    memories_mask.squeeze(), memory_indices, last_obs, info)

            # ==== P9 authentic-reset injection (FROZEN Bernoulli(0.5) at episode boundaries) ====
            rng, inj_rng, sel_rng = jax.random.split(rng, 3)
            use_auth = jax.random.bernoulli(inj_rng, AUTHENTIC_PROB, (config.num_envs,))
            inject = done & use_auth
            sel = jax.random.randint(sel_rng, (config.num_envs,), 0, LIB_N)
            g_log  = jax.tree_util.tree_map(lambda L: L[sel], lib_logstate)
            g_obs  = lib_obs[sel]; g_mem = lib_mem[sel]; g_mask = lib_mask[sel]; g_midx = lib_midx[sel]
            env_state = jax.tree_util.tree_map(lambda old, new: _where_env(inject, new, old), env_state, g_log)
            obsv = _where_env(inject, g_obs, obsv)
            memories = _where_env(inject, g_mem, memories)
            memories_mask = _where_env(inject, g_mask, memories_mask)
            memories_mask_idx = _where_env(inject, g_midx, memories_mask_idx)
            done_carry = done & ~inject   # injected envs start a (continued) episode: keep injected memory

            carry = (train_state, env_state, memories, memories_mask, memories_mask_idx,
                     obsv, done_carry, step_env_currentloop + 1, update_step, rng)
            return carry, (transition, memories_out)

        memories_previous = runner_state[2]
        (final_state_carry), (traj_batch, memories_batch) = jax.lax.scan(
            _env_step, runner_state, None, config.num_steps)
        (train_state, final_env_state, final_memories, final_mask, final_mask_idx,
         final_obs, done, final_step_loop, update_step, rng) = final_state_carry

        _, last_val, _ = network.apply(train_state.params, final_memories, final_obs, final_mask,
                                       method=network.model_forward_eval)

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
                traj_batch, memories_batch, advantages, targets = batch_info
                def _loss_fn(params, traj_batch, memories_batch, gae, targets):
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
                    pi, value = network.apply(params, memories_batch, obs, memories_mask,
                                              method=network.model_forward_train)
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
                    train_state.params, traj_batch, memories_batch, advantages, targets)
                grad_norm = optax.global_norm(grads)
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, (total_loss, value_loss, loss_actor, entropy, grad_norm)

            (train_state, traj_batch, memories_batch, advantages, targets, update_step, rng) = update_state
            rng, _rng = jax.random.split(rng)
            permutation = jax.random.permutation(_rng, config.num_envs)
            batch = (traj_batch, memories_batch, advantages, targets)
            batch = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), batch)
            shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, [config.num_minibatches, -1] + list(x.shape[1:])), shuffled_batch)
            train_state, (total_loss, value_loss, loss_actor, entropy, grad_norm) = \
                jax.lax.scan(_update_minbatch, train_state, minibatches)
            return (train_state, traj_batch, memories_batch, advantages, targets,
                    update_step, rng), (total_loss, value_loss, loss_actor, entropy, grad_norm)

        update_state = (train_state, traj_batch, memories_batch, advantages, targets, update_step, rng)
        update_state, rl_info = jax.lax.scan(_update_epoch, update_state, None, config.update_epochs)

        losses_and_ent = rl_info[:4]; grad_norms = rl_info[4]
        losses_mean = jax.tree_util.tree_map(lambda x: jnp.mean(x), losses_and_ent)
        gn_mean = jnp.mean(grad_norms); gn_max = jnp.max(grad_norms)
        metrics_to_log = (*losses_mean, gn_mean, gn_max)
        current_step = initial_global_update_step + update_step
        jax.debug.callback(_log_callback, metrics_to_log, current_step)

        train_state = update_state[0]; rng = update_state[-1]
        next_runner_state = (train_state, final_env_state, final_memories, final_mask,
                             final_mask_idx, final_obs, done, 0, update_step + 1, rng)
        return next_runner_state, None

    # ============================================================== init / resume ======
    teacher_params = load_weights_only(TEACHER_CKPT, base_env, env_params_ctor, cfg,
                                       load_opt_state=False).params
    init_params_sha = _params_sha(teacher_params)
    print(f"[p9] teacher ckpt17500 init params: leaves={len(jax.tree_util.tree_leaves(teacher_params))} "
          f"sha={init_params_sha[:16]}", flush=True)

    if args.resume_from:
        print(f"[p9] RESUME from {args.resume_from}", flush=True)
        with open(args.resume_from, "rb") as f: rd = pickle.load(f)
        ts = TrainState.create(apply_fn=network.apply, params=_unpack(rd["params"]), tx=tx).replace(
            opt_state=_unpack(rd["opt_state"]), step=jnp.asarray(rd["opt_step"], jnp.int32))
        runner_state = (ts, _unpack(rd["env_state"]),
                        jnp.asarray(rd["memories"]), jnp.asarray(rd["memories_mask"]),
                        jnp.asarray(rd["memories_mask_idx"]), jnp.asarray(rd["obs"]),
                        jnp.asarray(rd["done"]), int(rd["step_env_currentloop"]),
                        int(rd["update_step"]), jnp.asarray(rd["rng"]))
        start_update = int(rd["update_step"])
        print(f"[p9] resumed at update_step={start_update} params_sha={_params_sha(ts.params)[:16]}", flush=True)
    else:
        ts = TrainState.create(apply_fn=network.apply, params=teacher_params, tx=tx)
        assert int(ts.step) == 0
        rng0 = jax.random.PRNGKey(MASTER_SEED); rng0, _rng = jax.random.split(rng0)
        obsv, env_state = env.reset(_rng, env_params)
        env_state = env_state.replace(running_original_return=jnp.full(
            (cfg.num_envs,), current_original_return, dtype=jnp.float32))
        memories = jnp.zeros((cfg.num_envs, cfg.window_mem, cfg.num_layers, cfg.embed_size))
        memories_mask = jnp.zeros((cfg.num_envs, cfg.num_heads, 1, cfg.window_mem + 1), dtype=jnp.bool_)
        memories_mask_idx = jnp.zeros((cfg.num_envs,), dtype=jnp.int32) + (cfg.window_mem + 1)
        done = jnp.zeros((cfg.num_envs,), dtype=jnp.bool_)
        rng0, _rng = jax.random.split(rng0)
        runner_state = (ts, env_state, memories, memories_mask, memories_mask_idx, obsv, done, 0, 0, _rng)
        start_update = 0

    # ============================================================== save helper ======
    def _save_full(runner_state, global_step, params_sha):
        (train_state_, env_state_, memories_, mask_, mask_idx_, obs_, done_,
         step_loop_, update_step_, rng_) = runner_state
        ckpt_dir = os.path.join(args.ckpt_root, str(global_step))
        os.makedirs(ckpt_dir, exist_ok=True)
        full = dict(params=_pack(train_state_.params), opt_state=_pack(train_state_.opt_state),
            opt_step=int(train_state_.step), env_state=_pack(env_state_),
            memories=np.asarray(memories_), memories_mask=np.asarray(mask_),
            memories_mask_idx=np.asarray(mask_idx_), obs=np.asarray(obs_), done=np.asarray(done_),
            step_env_currentloop=int(step_loop_), update_step=int(update_step_), rng=np.asarray(rng_),
            global_step=global_step, update_count=int(update_step_), config=cfg_dict, code_sha256=code_sha,
            manifest=dict(label="P9_AUTHENTIC_RESET", arm="P9_AUTHENTIC_RESET", rng_seed=MASTER_SEED,
                          xla_flags=os.environ["XLA_FLAGS"], gpu_uuid=GPU_UUID, params_sha256=params_sha,
                          opt_state_leaf_hash=_leaf_hash(train_state_.opt_state),
                          authentic_prob=AUTHENTIC_PROB, lib_n=LIB_N, lib_counts=lib["counts"],
                          teacher_init_sha256=init_params_sha))
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
                 and int(rd["opt_step"]) == int(train_state_.step)
                 and int(rd["global_step"]) == global_step
                 and int(rd["update_count"]) == int(update_step_))
        print(f"[p9] saved FULL state -> {pkl_path} (sha={_sha_file(pkl_path)[:16]}) roundtrip_ok={rt_ok}", flush=True)
        assert rt_ok, f"FAIL roundtrip bit-exact at {global_step}"
        return pkl_path, _sha_file(pkl_path)

    # ============================================================== chunked training ======
    _upc = args.updates_per_chunk
    run_chunk = jax.jit(lambda rs: jax.lax.scan(_update_step, rs, None, _upc))
    cfg_dict = {k: v for k, v in vars(Cfg).items() if not k.startswith("__") and not callable(v)}
    chunk_records = []; t_start = time.time()

    # step-0 save (initial authentic-reset trainer state)
    if 0 in save_steps:
        pkl0, sha0 = _save_full(runner_state, 0, init_params_sha)
        chunk_records.append(dict(chunk=-1, global_step=0, update_count=0,
            params_sha256=init_params_sha, checkpoint=pkl0, checkpoint_sha256=sha0, roundtrip_ok=True))

    for ci in range(args.num_chunks):
        t0 = time.time()
        runner_state, _ = run_chunk(runner_state)
        jax.block_until_ready(runner_state)
        train_state = runner_state[0]
        update_step = int(runner_state[8])
        global_step = update_step * STEPS_PER_UPDATE
        elapsed = time.time() - t0
        params_sha = _params_sha(train_state.params)
        ent_now = METRICS_LOG[-1]["entropy"] if METRICS_LOG else float("nan")
        print(f"[p9] chunk {ci} done  global_step={global_step} update_count={update_step} "
              f"params_sha={params_sha[:16]} entropy={ent_now:.4f}  ({elapsed:.1f}s)", flush=True)
        assert _finite(train_state.params), f"NUMERIC_FAIL params non-finite at {global_step}"
        assert _finite(train_state.opt_state), f"NUMERIC_FAIL opt_state non-finite at {global_step}"
        rec = dict(chunk=ci, global_step=global_step, update_count=update_step, params_sha256=params_sha,
                   opt_step=int(train_state.step), params_finite=True, opt_finite=True,
                   entropy=ent_now, elapsed_s=round(elapsed, 1))
        if global_step in save_steps:
            pkl_path, ckpt_sha = _save_full(runner_state, global_step, params_sha)
            rec["checkpoint"] = pkl_path; rec["checkpoint_sha256"] = ckpt_sha; rec["roundtrip_ok"] = True
        chunk_records.append(rec)

    # ============================================================== done ======
    final_train_state = runner_state[0]
    final_global_step = int(runner_state[8]) * STEPS_PER_UPDATE
    status = "P9_TRAIN_OK" if final_global_step == (start_update + args.num_chunks * _upc) * STEPS_PER_UPDATE \
        else "P9_TRAIN_INCOMPLETE"
    summary = dict(label="P9_AUTHENTIC_RESET", arm="P9_AUTHENTIC_RESET", status=status,
        protocol=dict(start="teacher ckpt17500 (original ActorCriticTransformer, NO network change)",
                      algorithm="Henry Original GTrXL-PPO + authentic reached-state reset injection",
                      reset_mix=f"{int(AUTHENTIC_PROB*100)}/{int((1-AUTHENTIC_PROB)*100)} natural/authentic (FROZEN)",
                      xla_flags=os.environ["XLA_FLAGS"], seed=MASTER_SEED, lr=LR, adam_eps=ADAM_EPS,
                      gamma=GAMMA, gae_lambda=GAE_LAMBDA, num_envs=NUM_ENVS, rollout_steps=ROLLOUT_STEPS,
                      update_epochs=UPDATE_EPOCHS, num_minibatches=NUM_MINIBATCHES, clip_eps=CLIP_EPS,
                      vf_coef=VF_COEF, ent_coef=ENT_COEF, max_grad_norm=MAX_GRAD_NORM, anneal_lr=False,
                      optimistic_reset_ratio=OPTIMISTIC_RESET_RATIO, mode="score", replay="OFF",
                      hindsight="OFF", novelty="OFF", nav_aux="OFF", goal="DEFEAT_KOBOLD",
                      stage="S4_dark native", total_env_steps=TOTAL_STEPS, gpu_uuid=GPU_UUID),
        library=dict(path=args.lib_path, lib_n=LIB_N, counts=lib["counts"], cat_cap=lib.get("cat_cap"),
                     view_radius=lib.get("view_radius"), recent_window=lib.get("recent_window"),
                     capture_code_sha=lib.get("code_sha")),
        teacher_init_sha256=init_params_sha, code_sha256=code_sha, config=cfg_dict, chunks=chunk_records,
        per_update_metrics=list(METRICS_LOG), elapsed_total_s=round(time.time() - t_start, 1),
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with open(os.path.join(args.out_dir, "p9_train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True, default=str); f.write("\n")
    with open(os.path.join(args.out_dir, "p9_per_update.jsonl"), "w") as f:
        for m in METRICS_LOG: f.write(json.dumps(m, default=str) + "\n")
    print(f"\nSTATUS: {status}  final_global_step={final_global_step} "
          f"params_sha={_params_sha(final_train_state.params)}", flush=True)


if __name__ == "__main__":
    main()
