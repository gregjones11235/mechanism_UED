"""Convert a P2-Full-A pure-pickle checkpoint (full_state.pkl) into the FLAT orbax
TrainState format that the FROZEN evaluator (eval_paired_256.py, load_weights_only)
loads. The evaluator SHA must NOT change, so we produce a checkpoint in the exact
orbax layout load_weights_only expects (same network config / init shapes / FLAT
TrainState as the frozen Control checkpoint).

Read-only w.r.t. the source Full P2 checkpoint. Writes ONLY to a fresh --out_root.
Runs on GPU0 (network.init only; no training, no env rollout).

Verification (fail-closed): after saving, reload via load_weights_only(load_opt_state=
False) and assert the restored params content SHA == source Full P2 params SHA, and the
restored leaf count/paths/shapes match the network.init template. Prints CONVERT_OK.
"""
import os
GPU_UUID = "GPU-e8c08612-c22a-c8a4-6df5-affb2dd1f9a6"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID          # MUST precede jax import
import sys, argparse, hashlib

BASE_SRC = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
V7_SRC = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src"
if BASE_SRC in sys.path:
    sys.path.remove(BASE_SRC)
sys.path.insert(0, BASE_SRC)
if V7_SRC not in sys.path:
    sys.path.insert(1, V7_SRC)

import numpy as np
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from orbax.checkpoint import CheckpointManager, PyTreeCheckpointer, CheckpointManagerOptions
from flax.training.train_state import TrainState

from craftax.craftax.constants import Achievement
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from dicode.network import ActorCriticTransformer
from dicode.task_utils import get_achievement_multi_hot
from dicode.utils.general.train_state_utils import load_weights_only
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv

import checkpointing as CK

# ---- EXACT eval_paired_256.py config (frozen evaluator) ----
_cfg = dict(activation="relu", embed_size=256, hidden_layers=256, num_heads=8,
            qkv_features=256, num_layers=2, gating=True, gating_bias=2.0,
            window_mem=128, window_grad=64, anneal_lr=False, lr=2e-4, min_lr=2e-6,
            max_grad_norm=1.0, total_timesteps=2005401600, num_envs=1024, num_steps=128,
            update_epochs=4, num_minibatches=8, max_updates_per_session=500)
cfg = type("C", (), _cfg)()
NUM_STEPS = 4096

# ---- EXACT eval_paired_256.py Stage4-native task (DEFEAT_KOBOLD) ----
S4_TASK_CODE = '''
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


def build_eval_env():
    ctor = EnvParams(max_timesteps=NUM_STEPS)
    table = jnp.array([get_achievement_multi_hot([Achievement.DEFEAT_KOBOLD])], dtype=jnp.float32)
    emb = int(table.shape[1])
    ns4 = {}
    exec(S4_TASK_CODE, ns4)
    S4Cls = ns4["Env"]
    s4_base = MultiTaskMiniCraftaxEnv([S4Cls], StaticEnvParams(), ctor, True,
                                      conditioning_type="embedding", embedding_size=emb)
    return s4_base, ctor, emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p2_ckpt_dir", required=True,
                    help="Full P2 checkpoint ROOT containing <step>/full_state.pkl")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--out_root", required=True,
                    help="FRESH orbax output root; checkpoint written to <out_root>/<step>/")
    args = ap.parse_args()
    args.p2_ckpt_dir = os.path.abspath(args.p2_ckpt_dir)
    args.out_root = os.path.abspath(args.out_root)        # orbax tensorstore needs absolute paths

    with open(os.path.abspath(__file__), "rb") as f:
        converter_sha = hashlib.sha256(f.read()).hexdigest()

    # 1. eval env (for action/obs dims; mirrors load_weights_only exactly)
    s4_base, ctor, emb = build_eval_env()
    action_dim = int(s4_base.action_space(ctor).n)
    obs_dim = int(s4_base.observation_space(ctor).shape[0])
    print(f"[convert] env action_dim={action_dim} obs_dim={obs_dim} emb={emb}", flush=True)
    assert action_dim == 43, action_dim
    assert obs_dim == 8335, obs_dim
    assert emb == 67, emb

    # 2. network template (EXACT eval config + EXACT load_weights_only init shapes)
    network = ActorCriticTransformer(
        action_dim=action_dim, activation=cfg.activation,
        encoder_size=cfg.embed_size, hidden_layers=cfg.hidden_layers,
        num_heads=cfg.num_heads, qkv_features=cfg.qkv_features,
        num_layers=cfg.num_layers, gating=cfg.gating, gating_bias=cfg.gating_bias)
    rng = jax.random.PRNGKey(0)
    rng, _rng = jax.random.split(rng)
    init_obs = jnp.zeros((2, obs_dim))
    init_memory = jnp.zeros((2, cfg.window_mem, cfg.num_layers, cfg.embed_size))
    init_mask = jnp.zeros((2, cfg.num_heads, 1, cfg.window_mem + 1), dtype=jnp.bool_)
    dummy_params = network.init(_rng, init_memory, init_obs, init_mask)

    # 3. load Full P2 params (read-only) -> jnp. compat_init stores the INNER params
    #    dict (unwrapped); load_weights_only / network.init use the flax VARIABLES dict
    #    {"params": {...}}. Re-wrap one level (exact inverse of compat_init's unwrap).
    restored = CK.restore_full_checkpoint(args.p2_ckpt_dir, step=args.step)
    p2_inner = jax.tree_util.tree_map(jnp.asarray, restored["params"])
    src_sha = CK.params_content_sha256(p2_inner)          # canonical Full P2 params SHA
    p2_params = {"params": p2_inner}                       # re-wrap for the flax TrainState
    print(f"[convert] source Full P2 params sha={src_sha} step={args.step} (re-wrapped 'params')", flush=True)

    # 4. fail-closed structure check: paths + shapes must match the eval template
    d_flat = jax.tree_util.tree_leaves_with_path(dummy_params)
    p_flat = jax.tree_util.tree_leaves_with_path(p2_params)
    assert len(d_flat) == len(p_flat), f"leaf count {len(d_flat)} != {len(p_flat)}"
    for (dp, da), (pp, pa) in zip(d_flat, p_flat):
        assert dp == pp, f"path mismatch {dp} != {pp}"
        assert da.shape == pa.shape, f"shape mismatch at {dp}: {da.shape} != {pa.shape}"
    n_params = sum(int(np.prod(l.shape)) for _, l in p_flat)
    print(f"[convert] structure OK  leaves={len(p_flat)}  params={n_params}", flush=True)
    assert n_params == 4906028, n_params

    # 5. build TrainState (tx irrelevant for load_opt_state=False; keep valid/eval-like)
    tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm),
                     optax.adam(cfg.lr, eps=1e-5))
    ts = TrainState.create(apply_fn=network.apply, params=p2_params, tx=tx)

    # 6. save FLAT orbax (same layout as frozen Control: mgr.save(step, items=ts))
    assert not os.path.exists(os.path.join(args.out_root, str(args.step))), \
        f"HARD STOP out-reuse: {args.out_root}/{args.step} already exists"
    mgr = CheckpointManager(args.out_root, PyTreeCheckpointer(),
                            options=CheckpointManagerOptions(create=True))
    mgr.save(args.step, items=ts)
    mgr.wait_until_finished()
    print(f"[convert] saved FLAT orbax -> {args.out_root}/{args.step}", flush=True)

    # 7. fail-closed reload via the FROZEN evaluator loader
    ts2 = load_weights_only(os.path.join(args.out_root, str(args.step)),
                            s4_base, ctor, cfg, load_opt_state=False)
    rt_sha = CK.params_content_sha256(ts2.params)
    wrap_sha = CK.params_content_sha256(p2_params)
    leaves2 = len(jax.tree_util.tree_leaves(ts2.params))
    # leaf-by-leaf bit-exact equality (float32 restored from pickle->orbax->restore)
    rt_leaves = jax.tree_util.tree_leaves(ts2.params)
    src_leaves = jax.tree_util.tree_leaves(p2_params)
    leaf_exact = (len(rt_leaves) == len(src_leaves)) and all(
        np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(rt_leaves, src_leaves))
    roundtrip_ok = (rt_sha == wrap_sha) and (leaves2 == len(p_flat)) and leaf_exact
    print(f"[convert] reload sha={rt_sha} leaves={leaves2} leaf_exact={leaf_exact} roundtrip_ok={roundtrip_ok}", flush=True)
    assert roundtrip_ok, f"FAIL: reload mismatch (rt_sha={rt_sha} wrap_sha={wrap_sha} leaf_exact={leaf_exact})"

    print("CONVERT_OK step=%d src_sha=%s out=%s/%d leaves=%d params=%d converter_sha=%s" % (
        args.step, src_sha, args.out_root, args.step, leaves2, n_params, converter_sha),
        flush=True)


if __name__ == "__main__":
    main()
