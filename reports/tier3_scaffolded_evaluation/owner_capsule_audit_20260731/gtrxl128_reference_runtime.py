#!/usr/bin/env python3
"""GTRXL128_REFERENCE_RUNTIME — CC1 shared reference runtime for GTrXL128 (window_mem=128,
num_layers=2, embed=256, heads=8) Student candidates. ONE network implementation (dicode's
ActorCriticTransformer, imported — NOT copied). Two thin loaders:

  - THIN_GTRXL128_PICKLE_CONTROL : loads a pickled params pytree (eval_bakeoff convention).
  - THIN_GTRXL128_ORBAX_BASE     : loads an orbax training checkpoint via dicode's own
                                    load_weights_only (TrainState -> .params).

CC1 does NOT implement scientific metrics (graph_distance / transition / defeat). This module
only provides: network construction, checkpoint loading, params/file SHA, memory init/step/reset,
greedy policy_step, and metadata — the runtime ABI consumed by candidate_runtime.py and the
interface smoke. Metric semantics belong to CC4's common evaluator (formal_eval_binding=WAITING).
"""
import os, sys, json, hashlib, pickle

# ── Henry dicode source (authoritative network/env) — injected, not copied ──
DICODE_SRC = ("/home/oseasy/incoming/henry_work_20260721T105300/extracted/"
              "Henry_work/code/dicode_v7fix58_armB/src")

# Frozen network/training construction config (matches bakeoff frozen design + launch_baseline Cfg).
DEFAULT_CFG = {
    "activation": "relu",
    "encoder_size": 256,     # ActorCriticTransformer kwarg
    "embed_size": 256,       # dicode.load_weights_only reads config.embed_size (same value)
    "hidden_layers": 256,
    "num_heads": 8,
    "qkv_features": 256,
    "num_layers": 2,
    "gating": True,
    "gating_bias": 2.0,
    "window_mem": 128,
    "optimistic_reset_ratio": 16,
    # fields needed by dicode.load_weights_only's optimizer reconstruction (values from launch_baseline)
    "anneal_lr": True, "lr": 2e-4, "min_lr": 2e-6, "num_envs": 16, "num_steps": 128,
    "update_epochs": 1, "num_minibatches": 2, "max_grad_norm": 1.0,
    "max_updates_per_session": 48, "total_timesteps": 2_005_401_600,
    "lr_restart": 0.0, "lr_restart_at": 0, "lr_restart_horizon": 0, "lr_restart_warmup": 50,
}

LOADER_PICKLE_CONTROL = "THIN_GTRXL128_PICKLE_CONTROL"
LOADER_ORBAX_BASE = "THIN_GTRXL128_ORBAX_BASE"


def ensure_dicode_path():
    if DICODE_SRC not in sys.path:
        sys.path.insert(0, DICODE_SRC)


def _cfg_obj(cfg):
    """Dict -> attribute object for dicode loaders that expect a config namespace."""
    class _C: pass
    c = _C()
    for k, v in cfg.items():
        setattr(c, k, v)
    return c


# ───────────────────────── network ─────────────────────────
def build_network(action_dim, cfg=None):
    ensure_dicode_path()
    from dicode.network import ActorCriticTransformer
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    return ActorCriticTransformer(
        action_dim=int(action_dim), activation=cfg["activation"],
        encoder_size=cfg["encoder_size"], hidden_layers=cfg["hidden_layers"],
        num_heads=cfg["num_heads"], qkv_features=cfg["qkv_features"],
        num_layers=cfg["num_layers"], gating=cfg["gating"],
        gating_bias=cfg["gating_bias"])


# ───────────────────────── environment (Stage4 DEFEAT_KOBOLD) ─────────────────────────
_S4_TASK_CODE = '''
import jax
from craftax.craftax.constants import Achievement, ItemType
from minicraftax.craftax_state import TaskParams
from minicraftax.tasks.base_task import BaseTask
from minicraftax.world_builder import WorldBuilder
class Env(BaseTask):
    def __init__(self, sp, p):
        super().__init__(sp, p)
        self.relevant_achievements=[Achievement.DEFEAT_KOBOLD]; self.completed_achievements=[]; self.label="DEFEAT_KOBOLD"
    def get_task_params(self): return TaskParams(needs_depletion_multiplier=0.3)
    def generate_world(self, rng):
        rng,_r=jax.random.split(rng); b=WorldBuilder(_r,self.static_params,self.params)
        b.set_starting_floor(2); b.set_monsters_killed(2,8)
        b.set_player_inventory({"wood":7,"stone":27,"coal":3,"iron":3,"sapling":1,"pickaxe":3,"sword":3,"bow":1,"arrows":7,"torches":10})
        s=b.build(rng); up=b.ladders_up[2]
        return s.replace(item_map=s.item_map.at[2,up[0],up[1]].set(ItemType.NONE.value))
'''


def build_stage4_env(batch_size, max_steps=4096, cfg=None):
    """Returns dict with base_env, eval_env, env_params, static_params, OBS_DIM, ACTION_DIM, EMB, table."""
    ensure_dicode_path()
    import jax, jax.numpy as jnp
    from craftax.craftax.constants import Achievement
    from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
    from dicode.task_utils import get_achievement_multi_hot
    from dicode.wrappers_cl import DistributedMultiTaskOptimisticLogWrapper
    from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv
    cfg = {**DEFAULT_CFG, **(cfg or {})}

    ns = {}; exec(_S4_TASK_CODE, ns); S4Cls = ns["Env"]
    static_params = StaticEnvParams()
    env_params = EnvParams(max_timesteps=max_steps)
    table = jnp.array([get_achievement_multi_hot([Achievement.DEFEAT_KOBOLD])], dtype=jnp.float32)
    EMB = int(table.shape[1])
    base_env = MultiTaskMiniCraftaxEnv([S4Cls], static_params, env_params, True,
                                       conditioning_type="embedding", embedding_size=EMB)
    ACTION_DIM = int(base_env.action_space(env_params).n)
    OBS_DIM = int(base_env.observation_space(env_params).shape[0])
    eval_env = DistributedMultiTaskOptimisticLogWrapper(
        base_env, jax.random.PRNGKey(0), batch_size, 1,
        cfg["optimistic_reset_ratio"], jnp.array([1.0]), table)
    return {"base_env": base_env, "eval_env": eval_env, "env_params": env_params,
            "static_params": static_params, "table": table, "EMB": EMB,
            "OBS_DIM": OBS_DIM, "ACTION_DIM": ACTION_DIM,
            "eval_env_params": eval_env.default_params}


# ───────────────────────── loaders ─────────────────────────
def load_params_pickle(path):
    """THIN_GTRXL128_PICKLE_CONTROL — pickle params pytree -> jnp arrays."""
    import jax, jax.numpy as jnp
    with open(path, "rb") as f:
        params = pickle.load(f)
    return jax.tree_util.tree_map(jnp.asarray, params)


def load_params_orbax(path, network, env_bundle, cfg=None):
    """THIN_GTRXL128_ORBAX_BASE — orbax training checkpoint -> params, via dicode load_weights_only."""
    ensure_dicode_path()
    from dicode.utils.general.train_state_utils import load_weights_only
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    base_env = env_bundle["base_env"]; env_params = env_bundle["env_params"]
    out = load_weights_only(path, base_env, env_params, _cfg_obj(cfg), load_opt_state=False)
    params = getattr(out, "params", out)
    import jax, jax.numpy as jnp
    return jax.tree_util.tree_map(jnp.asarray, params)


def load_params(contract, network=None, env_bundle=None, cfg=None):
    """Dispatch on contract['loader']."""
    loader = contract["loader"]
    path = contract["checkpoint_path"]
    if loader == LOADER_PICKLE_CONTROL:
        return load_params_pickle(path)
    if loader == LOADER_ORBAX_BASE:
        if network is None or env_bundle is None:
            raise ValueError("orbax loader requires network + env_bundle")
        return load_params_orbax(path, network, env_bundle, cfg)
    raise ValueError("unknown loader: %s" % loader)


# ───────────────────────── hashing ─────────────────────────
def params_sha256(params):
    """eval_bakeoff convention: sha256 of concatenated little-endian leaf bytes (flax order)."""
    import jax, numpy as np
    return hashlib.sha256(
        b"".join(np.asarray(l).tobytes() for l in jax.tree_util.tree_leaves(params))).hexdigest()


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def dir_sha256(root):
    """Deterministic digest of a directory (orbax checkpoint): sha256 over sorted
    (relpath, file_sha256) pairs. Documented in checkpoint_contract.json."""
    entries = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            full = os.path.join(dp, fn)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            entries.append((rel, file_sha256(full)))
    entries.sort()
    h = hashlib.sha256()
    for rel, sha in entries:
        h.update(rel.encode("utf-8")); h.update(b"\0"); h.update(sha.encode("ascii")); h.update(b"\n")
    return h.hexdigest()


def checkpoint_file_sha256(contract):
    path = contract["checkpoint_path"]
    return dir_sha256(path) if os.path.isdir(path) else file_sha256(path)


def params_finite(params):
    import jax, jax.numpy as jnp
    leaves = jax.tree_util.tree_leaves(params)
    return bool(all(jnp.all(jnp.isfinite(l)) for l in leaves))


# ───────────────────────── memory / policy ABI ─────────────────────────
def init_memory(batch_size, cfg=None):
    """Stable zero-init GTrXL128 memory state (also used by Base/Teacher = no extra memory)."""
    import jax.numpy as jnp
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    wm, nl, es, nh = cfg["window_mem"], cfg["num_layers"], cfg["encoder_size"], cfg["num_heads"]
    return {
        "memories": jnp.zeros((batch_size, wm, nl, es)),
        "mmask": jnp.zeros((batch_size, nh, 1, wm + 1), dtype=jnp.bool_),
        "midx": jnp.full((batch_size,), wm + 1, dtype=jnp.int32),
    }


def reset_memory(memory_state, reset_mask, cfg=None):
    """Zero memory where reset_mask (per-env episode reset). reset_mask: (B,) bool."""
    import jax.numpy as jnp
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    wm = cfg["window_mem"]
    rm = reset_mask[:, None, None, None]
    return {
        "memories": jnp.where(rm, jnp.zeros_like(memory_state["memories"]), memory_state["memories"]),
        "mmask": jnp.where(reset_mask[:, None, None, None], jnp.zeros_like(memory_state["mmask"]), memory_state["mmask"]),
        "midx": jnp.where(reset_mask, jnp.full_like(memory_state["midx"], wm + 1), memory_state["midx"]),
    }


def policy_step(network, params, observation, memory_state, done_mask, cfg=None, greedy=True, rng=None):
    """One greedy/stochastic forward + GTrXL128 memory advance (mirrors eval_bakeoff baseline/control).
    done_mask: (B,) bool — dones from the PREVIOUS env step (drives memory-mask reset).
    Returns dict(action, logits, value, memory_state)."""
    import jax, jax.numpy as jnp
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    wm, nh = cfg["window_mem"], cfg["num_heads"]

    mi = jnp.where(done_mask, wm, jnp.clip(memory_state["midx"] - 1, 0, wm))
    mm = jnp.where(done_mask[:, None, None, None], jnp.zeros_like(memory_state["mmask"]), memory_state["mmask"])
    ohot = jax.nn.one_hot(mi, wm + 1)[:, None, None, :].repeat(nh, 1)
    mm = jnp.logical_or(mm, ohot)

    pi, value, mem_out = network.apply(params, memory_state["memories"], observation, mm,
                                       method=network.model_forward_eval)
    if greedy:
        action = pi.mode()
    else:
        if rng is None:
            raise ValueError("rng required for stochastic policy_step")
        action = pi.sample(seed=rng)
    memories = jnp.roll(memory_state["memories"], -1, axis=1).at[:, -1].set(mem_out)
    new_state = {"memories": memories, "mmask": mm, "midx": mi}
    return {"action": action, "logits": pi.logits, "value": value, "memory_state": new_state}


def candidate_metadata(candidate_id, cfg=None, obs_dim=None, action_dim=None):
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    return {
        "candidate_id": candidate_id,
        "network_family": "GTrXL128",
        "network_class": "dicode.network.ActorCriticTransformer",
        "memory_mode": "gtrxl_window128",
        "window_mem": cfg["window_mem"], "num_layers": cfg["num_layers"],
        "encoder_size": cfg["encoder_size"], "num_heads": cfg["num_heads"],
        "qkv_features": cfg["qkv_features"], "gating": cfg["gating"], "gating_bias": cfg["gating_bias"],
        "observation_dim": obs_dim, "action_dim": action_dim,
    }
