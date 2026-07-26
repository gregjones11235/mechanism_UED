#!/usr/bin/env python3
"""EXPLORATORY_DELAYED_ONSET_EXTENSION — Control continuous deterministic re-train.

Reconstructs the canonical long-Control trajectory (Henry Original GTrXL-PPO) from the
healthy ckpt17500 continuously to 98304 env steps, in a SINGLE active process, saving
FULL resumable state at 24576/49152/73728/98304 so it can be compared same-step against
P2-Full-A.

WHY THIS EXISTS (architectural fact, verified by reading dicode/ppo_tr.py):
Henry's make_train.train() ALWAYS env.reset()s at session start (no resume injection)
and runs the whole update loop as ONE opaque jax.lax.scan, returning ONLY train_state
(no env/collector/RNG/memory). So it is impossible to (a) capture the live full state at
intermediate steps or (b) verify the 24576 anchor before continuing using the stock
trainer unmodified. This launcher therefore reproduces Henry's _update_step/_env_step/
_loss_fn VERBATIM (reusing the REAL env wrapper / network / optax / Transition — only the
OUTER loop is re-expressed as 4 chunks of 12 updates) and carries the full runner_state
across chunks.

BIT-EXACT EQUIVALENCE: scan(body, s0, length=24) == scan(body, scan(body, s0, 12), 12)
because scan is iterative application of the same pure body; under
XLA_FLAGS=--xla_gpu_deterministic_ops=true each body application yields identical bits.
So chunked execution is bit-identical to a single continuous 48-update run, while letting
us capture state and gate at the 24576 boundary.

24576 DETERMINISTIC ANCHOR HARD GATE (directive section 三): before continuing past
24576, verify the re-trained state reproduces the FROZEN deterministic Control
realization ece6fa99... :
  - params _params_sha (byte-concat) == ECE6FA99_ANCHOR_PARAMS_SHA
  - optimizer state leaf-by-leaf bit-exact vs the frozen ece6fa99 TrainState
  - optimizer step == 24 (=12 updates x num_minibatches=2)
  - global_step/update_count == 24576/12
  - params & opt finite (no NaN/Inf)
The frozen ece6fa99 checkpoint has NO collector/env/RNG, so those fields are saved from
this new run only and are NOT compared to the old checkpoint (per directive). On ANY
params/opt/step mismatch -> CONTROL_RECONSTRUCTION_MISMATCH, exit(1), do NOT continue to
98304, never "close enough". On match -> CONTROL_24576_ANCHOR_REPRODUCED=true and the SAME
active runner_state continues (no reload, no env/RNG reset).

FULL-STATE ROUNDTRIP (section 四): at each save step, after writing full_state.pkl, restore
it into a SEPARATE object (temp dir) and verify params/opt/RNG/collector/env/memory are
bit-exact; the live runner_state is untouched and continues forward.

Frozen Control protocol (section 一): ckpt17500 start; Henry Original GTrXL-PPO;
XLA det-ops; seed=42; LR=2e-5; Adam eps=1e-5; gamma=0.999; GAE lambda=0.8; num_envs=16;
rollout_steps=128; update_epochs=1; num_minibatches=2; clip=0.2; vf=0.5; ent=0.002;
gradnorm=1.0; anneal_lr=False; optimistic_reset_ratio=16; mode=score; replay/hindsight OFF;
Stage4-native S4_dark; goal=DEFEAT_KOBOLD; total=98304 env steps (48 updates). GPU0 only.
"""
import os
GPU_UUID = "GPU-e8c08612-c22a-c8a4-6df5-affb2dd1f9a6"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"
import sys, json, argparse, hashlib, time, pickle, shutil

V7_SRC = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src"
if V7_SRC not in sys.path:
    sys.path.insert(0, V7_SRC)

import wandb
wandb.init(mode="disabled")

import numpy as np
import jax
import jax.numpy as jnp
import optax

from craftax.craftax.constants import Achievement
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from dicode.network import ActorCriticTransformer, Transition
from dicode.task_utils import get_achievement_multi_hot
from dicode.utils.general.train_state_utils import load_weights_only
from dicode.wrappers_cl import DistributedMultiTaskOptimisticLogWrapper
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv

# per-update metrics captured side-effect-free via jax.debug.callback (proven bit-exact
# preserving under det-ops by the Control freeze investigation, evidence_3).
METRICS_LOG = []

# --------------------------------------------------------------------- frozen anchors
SESSION175_CKPT = ("/home/oseasy/incoming/henry_work_20260721T105300/extracted/"
                   "Henry_work/base_ckpt_v7fix55_armA_s0/rl_checkpoints/17500")
FROZEN_CONTROL_24576_CKPT = "/home/oseasy/experiments/p2_full_20260723/checkpoints/det_telemetry_full_20260724/24576"
S4_TASK_PATH = "/home/oseasy/experiments/p2_v1_20260722/evidence/s4_task_code.py"

# byte-concat _params_sha scheme (EXACT telemetry/freeze scheme)
ECE6FA99_ANCHOR_PARAMS_SHA = "ece6fa9962e815123ce947577a93040057bc9df0b1e686dd28424cb2bbdabf55"
SOURCE_CKPT17500_SHA = "d4e85af58b7f87d689fadea12eec70c852fa098a09f5ea8907448684b3bf60f5"

# --------------------------------------------------------------------- frozen protocol
NUM_ENVS = 16
ROLLOUT_STEPS = 128
STEPS_PER_UPDATE = NUM_ENVS * ROLLOUT_STEPS      # 2048
TOTAL_STEPS = 98304
TOTAL_UPDATES = TOTAL_STEPS // STEPS_PER_UPDATE   # 48
UPDATES_PER_CHUNK = 12                            # 24576 env steps per chunk
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
SAVE_STEPS = (24576, 49152, 73728, 98304)


class Cfg:
    """Frozen §14 Control config — IDENTICAL to run_control_kl_telemetry.Cfg."""
    lr = LR
    min_lr = 2e-6
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
    max_updates_per_session = UPDATES_PER_CHUNK
    total_timesteps = 2_005_401_600
    scoring_window_updates = 4
    value_target_clip_min = VT_CLIP_MIN
    value_target_clip_max = VT_CLIP_MAX
    guard_session_vloss_max = 1000.0
    guard_session_entropy_min = 0.10
    guard_max_consecutive_reverts = 2
    lr_restart = 0.0
    lr_restart_at = 0
    lr_restart_horizon = 0
    lr_restart_warmup = 50
    sil = False
    sil_pools = []
    use_wandb = False
    debug = False
    validation = None
    dicode_manager = None

    def get(self, key, default=None):
        return getattr(self, key, default)


# ------------------------------------------------------------- hashing helpers
def _params_sha(params):
    """EXACT telemetry/freeze byte-concat params SHA256 (over tree_leaves in order)."""
    h = hashlib.sha256()
    for v in jax.tree_util.tree_leaves(params):
        h.update(np.ascontiguousarray(np.asarray(v)).tobytes())
    return h.hexdigest()


def _leaf_hash(pytree):
    """Stable SHA256 over all leaves (shape+bytes) of a pytree, in flatten order."""
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


# ------------------------------------------------------------- pytree (de)serialization
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
    ap.add_argument("--num_chunks", type=int, default=TOTAL_UPDATES // UPDATES_PER_CHUNK,
                    help="4 for the full 0->98304 run; 1 for a cheap anchor validation")
    ap.add_argument("--updates_per_chunk", type=int, default=UPDATES_PER_CHUNK,
                    help="DIAGNOSTIC ONLY: set to 1 to emit a per-update params SHA so the "
                         "reimplementation can be diffed against make_train update-by-update. "
                         "The frozen run uses the default 12 (== make_train NUM_UPDATES).")
    args = ap.parse_args()
    args.out_dir = os.path.abspath(args.out_dir)
    args.ckpt_root = os.path.abspath(args.ckpt_root)

    # ---- guards: GPU bind + fresh dirs + disk ----
    devs = jax.local_devices()
    dev_id = getattr(devs[0], "id", None)
    print(f"[control] devices={devs} (expect GPU0 only)", flush=True)
    assert len(devs) == 1, f"expected exactly 1 visible device (GPU0), got {devs}"

    for d in (args.out_dir, args.ckpt_root):
        if os.path.exists(d):
            assert not os.listdir(d), f"HARD STOP dir-reuse: {d} not empty"
        else:
            os.makedirs(d, exist_ok=True)
    free_gb = shutil.disk_usage(args.ckpt_root).free / 1024 ** 3
    print(f"[control] disk free = {free_gb:.1f} GB", flush=True)
    assert free_gb > 10.0, f"HARD STOP low disk {free_gb:.1f} GB"

    cfg = Cfg()
    cfg.training = cfg

    code_sha = {
        "launcher": _sha_file(os.path.abspath(__file__)),
        "ppo_tr": _sha_file(os.path.join(V7_SRC, "dicode", "ppo_tr.py")),
        "s4_task": _sha_file(S4_TASK_PATH),
    }

    # ---- S4_dark env (EXACT make_train construction) ----
    ach_table = jnp.array([get_achievement_multi_hot([Achievement.DEFEAT_KOBOLD])],
                          dtype=jnp.float32)
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
    print(f"[control] S4_dark env: obs_dim={obs_dim} action_dim={action_dim} emb={EMB}",
          flush=True)
    assert action_dim == 43 and obs_dim == 8335 and EMB == 67

    # ---- network + optimizer (EXACT make_train, anneal_lr=False) ----
    network = ActorCriticTransformer(
        action_dim=action_dim, activation=cfg.activation, hidden_layers=cfg.hidden_layers,
        encoder_size=cfg.embed_size, num_heads=cfg.num_heads, qkv_features=cfg.qkv_features,
        num_layers=cfg.num_layers, gating=cfg.gating, gating_bias=cfg.gating_bias)
    tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm),
                     optax.adam(cfg.lr, eps=ADAM_EPS))

    # ---- load ckpt17500 (weights-only + FRESH Adam, step=0) ----
    ts = load_weights_only(SESSION175_CKPT, base_env, env_params_ctor, cfg,
                           load_opt_state=False)
    source_sha = _params_sha(ts.params)
    assert source_sha == SOURCE_CKPT17500_SHA, \
        f"REFUSED source sha {source_sha} != {SOURCE_CKPT17500_SHA}"
    assert int(ts.step) == 0
    print(f"[control] ckpt17500 loaded: source_sha={source_sha} opt_step=0", flush=True)

    # ---- load FROZEN ece6fa99 TrainState (params+opt+step) for the anchor gate ----
    ts_frozen = load_weights_only(FROZEN_CONTROL_24576_CKPT, base_env, env_params_ctor, cfg,
                                  load_opt_state=True)
    frozen_params_sha = _params_sha(ts_frozen.params)
    frozen_opt_step = int(ts_frozen.step)
    print(f"[control] frozen anchor ckpt loaded: params_sha={frozen_params_sha} "
          f"opt_step={frozen_opt_step}", flush=True)
    assert frozen_params_sha == ECE6FA99_ANCHOR_PARAMS_SHA, \
        f"REFUSED frozen anchor sha {frozen_params_sha} != {ECE6FA99_ANCHOR_PARAMS_SHA}"
    assert frozen_opt_step == 24, f"frozen anchor opt_step {frozen_opt_step} != 24"

    # ============================================================== _update_step (VERBATIM
    # from dicode/ppo_tr.py make_train.train._update_step, sil_on=False branches removed) ==
    current_original_return = 0.0
    vt_clip_min = VT_CLIP_MIN
    vt_clip_max = VT_CLIP_MAX
    initial_global_update_step = 0
    config = cfg

    # helper funcs (EXACT ppo_tr.py)
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

    def _update_step(runner_state, unused_scan_input):
        # === A. COLLECT TRAJECTORIES ===
        def _env_step(runner_state, _):
            (train_state, env_state, memories, memories_mask, memories_mask_idx,
             last_obs, done, step_env_currentloop, update_step, rng) = runner_state

            memories_mask_idx = jnp.where(
                done, config.window_mem, jnp.clip(memories_mask_idx - 1, 0, config.window_mem))
            memories_mask = jnp.where(
                done[:, None, None, None],
                jnp.zeros((config.num_envs, config.num_heads, 1, config.window_mem + 1),
                          dtype=jnp.bool_),
                memories_mask)

            memories_mask_idx_ohot = jax.nn.one_hot(memories_mask_idx, config.window_mem + 1)
            memories_mask_idx_ohot = memories_mask_idx_ohot[:, None, None, :].repeat(
                config.num_heads, 1)
            memories_mask = jnp.logical_or(memories_mask, memories_mask_idx_ohot)

            rng, _rng = jax.random.split(rng)
            pi, value, memories_out = network.apply(
                train_state.params, memories, last_obs, memories_mask,
                method=network.model_forward_eval)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)

            memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(memories_out)

            rng, _rng = jax.random.split(rng)
            obsv, env_state, reward, done, info = env.step(_rng, env_state, action, env_params)
            env_state = env_state.replace(
                running_original_return=jnp.full(
                    (config.num_envs,), current_original_return, dtype=jnp.float32))

            memory_indices = jnp.arange(0, config.window_mem)[None, :] + \
                step_env_currentloop * jnp.ones((config.num_envs, 1), dtype=jnp.int32)

            transition = Transition(
                done, action, value, reward, log_prob,
                memories_mask.squeeze(), memory_indices, last_obs, info)

            carry = (train_state, env_state, memories, memories_mask, memories_mask_idx,
                     obsv, done, step_env_currentloop + 1, update_step, rng)
            return carry, (transition, memories_out)

        memories_previous = runner_state[2]
        (final_state_carry), (traj_batch, memories_batch) = jax.lax.scan(
            _env_step, runner_state, None, config.num_steps)
        (train_state, final_env_state, final_memories, final_mask, final_mask_idx,
         final_obs, done, final_step_loop, update_step, rng) = final_state_carry

        # === B. CALCULATE ADVANTAGES (GAE) ===
        _, last_val, _ = network.apply(
            train_state.params, final_memories, final_obs, final_mask,
            method=network.model_forward_eval)

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
            return advantages, jnp.clip(
                advantages + traj_batch.value, vt_clip_min, vt_clip_max)

        advantages, targets = _calculate_gae(traj_batch, last_val)

        # === C. UPDATE NETWORK (TRANSFORMER LOSS) ===
        memories_batch = jnp.concatenate(
            [jnp.swapaxes(memories_previous, 0, 1), memories_batch], axis=0)

        def _update_epoch(update_state, unused):
            def _update_minbatch(train_state, batch_info):
                traj_batch, memories_batch, advantages, targets = batch_info

                def _loss_fn(params, traj_batch, memories_batch, gae, targets):
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
                    obs = traj_batch.obs.reshape(
                        (-1, config.window_grad) + traj_batch.obs.shape[2:])
                    traj_batch_r, targets_r, gae_r = jax.tree_util.tree_map(
                        lambda x: jnp.reshape(x, (-1, config.window_grad) + x.shape[2:]),
                        (traj_batch, targets, gae))
                    pi, value = network.apply(
                        params, memories_batch, obs, memories_mask,
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
                    loss_actor2 = (jnp.clip(ratio, 1.0 - config.clip_eps,
                                            1.0 + config.clip_eps) * gae_r)
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                    entropy = pi.entropy().mean()
                    total_loss = (loss_actor + config.vf_coef * value_loss
                                  - config.ent_coef * entropy)
                    return total_loss, (value_loss, loss_actor, entropy)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                (total_loss, (value_loss, loss_actor, entropy)), grads = grad_fn(
                    train_state.params, traj_batch, memories_batch, advantages, targets)
                grad_norm = optax.global_norm(grads)
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, (total_loss, value_loss, loss_actor, entropy, grad_norm)

            (train_state, traj_batch, memories_batch, advantages, targets,
             update_step, rng) = update_state
            rng, _rng = jax.random.split(rng)
            permutation = jax.random.permutation(_rng, config.num_envs)
            batch = (traj_batch, memories_batch, advantages, targets)
            batch = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), batch)
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch)
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, [config.num_minibatches, -1] + list(x.shape[1:])),
                shuffled_batch)
            train_state, (total_loss, value_loss, loss_actor, entropy, grad_norm) = \
                jax.lax.scan(_update_minbatch, train_state, minibatches)
            return (train_state, traj_batch, memories_batch, advantages, targets,
                    update_step, rng), (total_loss, value_loss, loss_actor, entropy, grad_norm)

        update_state = (train_state, traj_batch, memories_batch, advantages, targets,
                        update_step, rng)
        update_state, rl_info = jax.lax.scan(
            _update_epoch, update_state, None, config.update_epochs)

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
                             final_mask_idx, final_obs, done, 0, update_step + 1, rng)
        return next_runner_state, None

    # ============================================================== initial runner_state
    # (VERBATIM make_train.train init, train_state provided so no param init) ==============
    rng0 = jax.random.PRNGKey(MASTER_SEED)
    rng0, _rng = jax.random.split(rng0)
    obsv, env_state = env.reset(_rng, env_params)
    env_state = env_state.replace(
        running_original_return=jnp.full((cfg.num_envs,), current_original_return,
                                         dtype=jnp.float32))
    memories = jnp.zeros((cfg.num_envs, cfg.window_mem, cfg.num_layers, cfg.embed_size))
    memories_mask = jnp.zeros((cfg.num_envs, cfg.num_heads, 1, cfg.window_mem + 1),
                              dtype=jnp.bool_)
    memories_mask_idx = jnp.zeros((cfg.num_envs,), dtype=jnp.int32) + (cfg.window_mem + 1)
    done = jnp.zeros((cfg.num_envs,), dtype=jnp.bool_)
    rng0, _rng = jax.random.split(rng0)
    runner_state = (ts, env_state, memories, memories_mask, memories_mask_idx,
                    obsv, done, 0, 0, _rng)

    # ============================================================== chunked training ======
    _upc = args.updates_per_chunk
    run_chunk = jax.jit(lambda rs: jax.lax.scan(_update_step, rs, None, _upc))

    cfg_dict = {k: v for k, v in vars(Cfg).items()
                if not k.startswith("__") and not callable(v)}
    chunk_records = []
    anchor_result = None
    t_start = time.time()

    for ci in range(args.num_chunks):
        t0 = time.time()
        runner_state, _ = run_chunk(runner_state)
        jax.block_until_ready(runner_state)
        train_state = runner_state[0]
        update_step = int(runner_state[8])
        global_step = update_step * STEPS_PER_UPDATE
        elapsed = time.time() - t0
        params_sha = _params_sha(train_state.params)
        print(f"[control] chunk {ci} done  global_step={global_step} update_count={update_step} "
              f"params_sha={params_sha[:16]}  ({elapsed:.1f}s)", flush=True)

        rec = dict(chunk=ci, global_step=global_step, update_count=update_step,
                   params_sha256=params_sha, opt_step=int(train_state.step),
                   params_finite=_finite(train_state.params),
                   opt_finite=_finite(train_state.opt_state),
                   elapsed_s=round(elapsed, 1))

        if global_step in SAVE_STEPS:
            # ---- 24576 ANCHOR HARD GATE (before continuing) ----
            if global_step == 24576:
                a_params = (params_sha == ECE6FA99_ANCHOR_PARAMS_SHA)
                a_opt = _leaves_equal(train_state.opt_state, ts_frozen.opt_state)
                a_step = (int(train_state.step) == 24)
                a_gs = (global_step == 24576) and (update_step == 12)
                a_fin = rec["params_finite"] and rec["opt_finite"]
                anchor_result = dict(
                    params_sha_match=a_params, opt_state_leaf_exact=a_opt,
                    opt_step_match=a_step, global_step_update_count_match=a_gs,
                    finite=a_fin, reproduced=bool(a_params and a_opt and a_step and a_gs and a_fin))
                rec["anchor"] = anchor_result
                print(f"[control] ANCHOR@24576: params={a_params} opt_leaf_exact={a_opt} "
                      f"step={a_step} gs/uc={a_gs} finite={a_fin} "
                      f"=> reproduced={anchor_result['reproduced']}", flush=True)
                if not anchor_result["reproduced"]:
                    rec["status"] = "CONTROL_RECONSTRUCTION_MISMATCH"
                    chunk_records.append(rec)
                    _write_summary(args.out_dir, cfg_dict, code_sha, chunk_records,
                                   anchor_result, "CONTROL_RECONSTRUCTION_MISMATCH",
                                   args, t_start)
                    print("CONTROL_RECONSTRUCTION_MISMATCH — stopping, NOT continuing to 98304",
                          flush=True)
                    sys.exit(1)
                print("CONTROL_24576_ANCHOR_REPRODUCED=true — continuing same active process",
                      flush=True)

            # ---- save FULL state ----
            ckpt_dir = os.path.join(args.ckpt_root, str(global_step))
            assert not os.path.exists(ckpt_dir), f"HARD STOP out-reuse {ckpt_dir}"
            os.makedirs(ckpt_dir, exist_ok=True)
            (train_state_, env_state_, memories_, mask_, mask_idx_,
             obs_, done_, step_loop_, update_step_, rng_) = runner_state
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
                step_env_currentloop=int(step_loop_),
                update_step=int(update_step_),
                rng=np.asarray(rng_),
                global_step=global_step,
                update_count=int(update_step_),
                config=cfg_dict,
                code_sha256=code_sha,
                manifest=dict(label="EXPLORATORY_DELAYED_ONSET_EXTENSION",
                              arm="CONTROL_ORIGINAL_PPO", rng_seed=MASTER_SEED,
                              xla_flags=os.environ["XLA_FLAGS"], gpu_uuid=GPU_UUID,
                              params_sha256=params_sha,
                              opt_state_leaf_hash=_leaf_hash(train_state_.opt_state),
                              anchor=(anchor_result if global_step == 24576 else None)))
            pkl_path = os.path.join(ckpt_dir, "full_state.pkl")
            tmp_path = pkl_path + ".tmp"
            with open(tmp_path, "wb") as f:
                pickle.dump(full, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, pkl_path)
            rec["checkpoint"] = pkl_path
            rec["checkpoint_sha256"] = _sha_file(pkl_path)
            print(f"[control] saved FULL state -> {pkl_path} "
                  f"(sha={rec['checkpoint_sha256'][:16]})", flush=True)

            # ---- non-disturbing save->restore roundtrip (separate object) ----
            with open(pkl_path, "rb") as f:
                rd = pickle.load(f)
            rt_params = _unpack(rd["params"])
            rt_opt = _unpack(rd["opt_state"])
            rt_env = _unpack(rd["env_state"])
            rt_ok = (_params_sha(rt_params) == params_sha
                     and _leaves_equal(rt_opt, train_state_.opt_state)
                     and np.array_equal(rd["rng"], np.asarray(rng_))
                     and _leaves_equal(rt_env, env_state_)
                     and np.array_equal(rd["memories"], np.asarray(memories_))
                     and np.array_equal(rd["memories_mask"], np.asarray(mask_))
                     and np.array_equal(rd["memories_mask_idx"], np.asarray(mask_idx_))
                     and np.array_equal(rd["obs"], np.asarray(obs_))
                     and np.array_equal(rd["done"], np.asarray(done_))
                     and int(rd["opt_step"]) == int(train_state_.step)
                     and int(rd["global_step"]) == global_step
                     and int(rd["update_count"]) == int(update_step_))
            rec["roundtrip_ok"] = bool(rt_ok)
            print(f"[control] roundtrip@{global_step} ok={rt_ok} "
                  f"(live runner_state untouched, continues)", flush=True)
            assert rt_ok, f"FAIL roundtrip bit-exact at {global_step}"

        chunk_records.append(rec)

    # ============================================================== done ==================
    final_train_state = runner_state[0]
    final_update_step = int(runner_state[8])
    final_global_step = final_update_step * STEPS_PER_UPDATE
    status = ("CONTROL_CONTINUOUS_RECONSTRUCTION_OK"
              if (final_global_step == args.num_chunks * args.updates_per_chunk * STEPS_PER_UPDATE
                  and (anchor_result is None or anchor_result["reproduced"]))
              else "CONTROL_RECONSTRUCTION_INCOMPLETE")
    _write_summary(args.out_dir, cfg_dict, code_sha, chunk_records, anchor_result,
                   status, args, t_start)
    print(f"\nSTATUS: {status}  final_global_step={final_global_step} "
          f"params_sha={_params_sha(final_train_state.params)}", flush=True)
    print(f"CONTROL_ANCHOR_REPRODUCED={anchor_result['reproduced'] if anchor_result else None}",
          flush=True)


def _write_summary(out_dir, cfg_dict, code_sha, chunk_records, anchor_result,
                   status, args, t_start):
    summary = dict(
        label="EXPLORATORY_DELAYED_ONSET_EXTENSION",
        arm="CONTROL_ORIGINAL_PPO_CONTINUOUS_RETRAIN",
        status=status,
        protocol=dict(start="ckpt17500", algorithm="Henry Original GTrXL-PPO",
                      xla_flags=os.environ["XLA_FLAGS"], seed=MASTER_SEED, lr=LR,
                      adam_eps=ADAM_EPS, gamma=GAMMA, gae_lambda=GAE_LAMBDA,
                      num_envs=NUM_ENVS, rollout_steps=ROLLOUT_STEPS,
                      update_epochs=UPDATE_EPOCHS, num_minibatches=NUM_MINIBATCHES,
                      clip_eps=CLIP_EPS, vf_coef=VF_COEF, ent_coef=ENT_COEF,
                      max_grad_norm=MAX_GRAD_NORM, anneal_lr=False,
                      optimistic_reset_ratio=OPTIMISTIC_RESET_RATIO, mode="score",
                      replay="OFF", hindsight="OFF", goal="DEFEAT_KOBOLD",
                      stage="S4_dark native", total_env_steps=TOTAL_STEPS,
                      total_updates=TOTAL_UPDATES, gpu_uuid=GPU_UUID),
        anchors=dict(source_ckpt17500_sha256=SOURCE_CKPT17500_SHA,
                     frozen_control_24576_sha256=ECE6FA99_ANCHOR_PARAMS_SHA,
                     frozen_control_24576_ckpt=FROZEN_CONTROL_24576_CKPT),
        anchor_24576=anchor_result,
        code_sha256=code_sha,
        config=cfg_dict,
        chunks=chunk_records,
        per_update_metrics=list(METRICS_LOG),
        elapsed_total_s=round(time.time() - t_start, 1),
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with open(os.path.join(out_dir, "control_continuous_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    jl = os.path.join(out_dir, "control_per_update.jsonl")
    with open(jl, "w") as f:
        for m in METRICS_LOG:
            f.write(json.dumps(m, default=str) + "\n")


if __name__ == "__main__":
    main()
