"""Convert a CONTROL continuous-retrain full_state.pkl (the custom Control-launcher
pickle that stores FLAX-WRAPPED params {"params": ...} via _pack=(leaves,treedef)) into
the FLAT orbax TrainState format that the FROZEN evaluator (eval_paired_256.py,
load_weights_only) loads. The evaluator SHA must NOT change, so we produce a checkpoint
in the exact orbax layout load_weights_only expects (same network config / init shapes /
FLAT TrainState as the frozen Control checkpoint).

This is the Control analogue of convert_fullp2_to_orbax.py. The ONLY difference is the
source reader: Control full_state.pkl is a custom pickle whose "params" entry is
_pack(train_state.params) = (numpy_leaves_list, treedef) of the FLAX-WRAPPED variables
dict {"params": {...}}. We _unpack it (tree_unflatten) and use it DIRECTLY as the
TrainState params (NO re-wrap, unlike the unwrapped Full P2 pickle).

Read-only w.r.t. the source Control checkpoint. Writes ONLY to a fresh --out_root.
Runs on GPU0 (network.init only; no training, no env rollout).

Verification (fail-closed):
  * source params _params_sha (byte-concat) == --expected_params_sha (the training-time
    SHA logged by the Control run for this step) — ties the orbax ckpt to the verified
    continuous trajectory.
  * after saving, reload via load_weights_only(load_opt_state=False) and assert the
    reloaded params _params_sha == source _params_sha AND leaf-by-leaf bit-exact AND the
    leaf paths/shapes match the network.init template. Prints CONVERT_OK.
"""
import os
GPU_UUID = "GPU-e8c08612-c22a-c8a4-6df5-affb2dd1f9a6"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID          # MUST precede jax import
import sys, argparse, hashlib, pickle

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


def _unpack(packed):
    """Inverse of the Control launcher's _pack: (numpy_leaves, treedef) -> pytree."""
    leaves, treedef = packed
    return jax.tree_util.tree_unflatten(treedef, [jnp.asarray(l) for l in leaves])


def _params_sha(params):
    """Byte-concat SHA over tree leaves (SAME scheme the Control run logged at train time)."""
    h = hashlib.sha256()
    for v in jax.tree_util.tree_leaves(params):
        h.update(np.ascontiguousarray(np.asarray(v)).tobytes())
    return h.hexdigest()


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
    ap.add_argument("--control_pkl", required=True,
                    help="Control full_state.pkl path (contains _pack'd wrapped params)")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--expected_params_sha", required=True,
                    help="training-time _params_sha (byte-concat) for this step")
    ap.add_argument("--out_root", required=True,
                    help="FRESH orbax output root; checkpoint written to <out_root>/<step>/")
    args = ap.parse_args()
    args.control_pkl = os.path.abspath(args.control_pkl)
    args.out_root = os.path.abspath(args.out_root)        # orbax tensorstore needs absolute paths

    with open(os.path.abspath(__file__), "rb") as f:
        converter_sha = hashlib.sha256(f.read()).hexdigest()

    # 1. eval env (for action/obs dims; mirrors load_weights_only exactly)
    s4_base, ctor, emb = build_eval_env()
    action_dim = int(s4_base.action_space(ctor).n)
    obs_dim = int(s4_base.observation_space(ctor).shape[0])
    print(f"[convert-ctl] env action_dim={action_dim} obs_dim={obs_dim} emb={emb}", flush=True)
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

    # 3. load Control params (read-only) from the custom pickle. _unpack yields the
    #    FLAX-WRAPPED variables dict {"params": {...}} (exactly what TrainState wants).
    with open(args.control_pkl, "rb") as f:
        rd = pickle.load(f)
    assert "params" in rd, f"Control pkl missing 'params' key; keys={sorted(rd.keys())}"
    ctl_params = _unpack(rd["params"])                     # {"params": {...}} wrapped
    # sanity: must be a mapping with a "params" member (flax variables dict)
    assert isinstance(ctl_params, dict) and "params" in ctl_params, \
        f"expected wrapped flax variables dict, got top-level keys={list(ctl_params.keys()) if isinstance(ctl_params, dict) else type(ctl_params)}"
    src_sha = _params_sha(ctl_params)
    print(f"[convert-ctl] source Control params sha={src_sha} step={args.step} (wrapped 'params')",
          flush=True)

    # 4. training-tie HARD GATE: source params must equal the training-time SHA
    assert src_sha == args.expected_params_sha, (
        "HARD STOP source-mismatch: Control pkl _params_sha=%s != expected training SHA %s"
        % (src_sha, args.expected_params_sha))

    # 5. fail-closed structure check: paths + shapes must match the eval template
    d_flat = jax.tree_util.tree_leaves_with_path(dummy_params)
    p_flat = jax.tree_util.tree_leaves_with_path(ctl_params)
    assert len(d_flat) == len(p_flat), f"leaf count {len(d_flat)} != {len(p_flat)}"
    for (dp, da), (pp, pa) in zip(d_flat, p_flat):
        assert dp == pp, f"path mismatch {dp} != {pp}"
        assert da.shape == pa.shape, f"shape mismatch at {dp}: {da.shape} != {pa.shape}"
    n_params = sum(int(np.prod(l.shape)) for _, l in p_flat)
    print(f"[convert-ctl] structure OK  leaves={len(p_flat)}  params={n_params}", flush=True)
    assert n_params == 4906028, n_params

    # 6. build TrainState (tx irrelevant for load_opt_state=False; keep valid/eval-like)
    tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm),
                     optax.adam(cfg.lr, eps=1e-5))
    ts = TrainState.create(apply_fn=network.apply, params=ctl_params, tx=tx)

    # 7. save FLAT orbax (same layout as frozen Control: mgr.save(step, items=ts))
    assert not os.path.exists(os.path.join(args.out_root, str(args.step))), \
        f"HARD STOP out-reuse: {args.out_root}/{args.step} already exists"
    mgr = CheckpointManager(args.out_root, PyTreeCheckpointer(),
                            options=CheckpointManagerOptions(create=True))
    mgr.save(args.step, items=ts)
    mgr.wait_until_finished()
    print(f"[convert-ctl] saved FLAT orbax -> {args.out_root}/{args.step}", flush=True)

    # 8. fail-closed reload via the FROZEN evaluator loader
    ts2 = load_weights_only(os.path.join(args.out_root, str(args.step)),
                            s4_base, ctor, cfg, load_opt_state=False)
    rt_sha = _params_sha(ts2.params)
    leaves2 = len(jax.tree_util.tree_leaves(ts2.params))
    rt_leaves = jax.tree_util.tree_leaves(ts2.params)
    src_leaves = jax.tree_util.tree_leaves(ctl_params)
    leaf_exact = (len(rt_leaves) == len(src_leaves)) and all(
        np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(rt_leaves, src_leaves))
    roundtrip_ok = (rt_sha == src_sha) and (leaves2 == len(p_flat)) and leaf_exact
    print(f"[convert-ctl] reload sha={rt_sha} leaves={leaves2} leaf_exact={leaf_exact} "
          f"roundtrip_ok={roundtrip_ok}", flush=True)
    assert roundtrip_ok, f"FAIL: reload mismatch (rt_sha={rt_sha} src_sha={src_sha} leaf_exact={leaf_exact})"

    print("CONVERT_OK step=%d src_sha=%s out=%s/%d leaves=%d params=%d converter_sha=%s" % (
        args.step, src_sha, args.out_root, args.step, leaves2, n_params, converter_sha),
        flush=True)


if __name__ == "__main__":
    main()
