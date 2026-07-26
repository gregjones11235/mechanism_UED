"""Checkpoint round-trip conservation (CPU, pure pickle).

Proves a P2-Full-A checkpoint restores BIT-EXACTLY on CPU:
  CKPT.rt     params / EMA target_params / opt_state / replay (incl anchors) /
              pending (incl anchors) / rng_key / action_rng / global_step /
              update_count all survive save->restore with zero diff
  CKPT.anchor replayed trajectory anchors + pending anchors are byte-conserved
  CKPT.strip  rolling strip keeps only the newest N; detect_latest_step/inventory OK
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import sys, os.path, tempfile, shutil
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
import jax
import jax.numpy as jnp

import fputil
import rng_utils as RU
import checkpointing as CK
from replay_buffer import ReplayBuffer
from pending_episodes import PendingEpisodeBuffers
from full_p2_learner import build_optimizer, full_p2_update

CFG = fputil.CFG


def _leaves_equal(a, b):
    la = jax.tree_util.tree_leaves(a)
    lb = jax.tree_util.tree_leaves(b)
    if len(la) != len(lb):
        return False
    for x, y in zip(la, lb):
        x = np.asarray(x); y = np.asarray(y)
        if x.shape != y.shape or x.dtype != y.dtype:
            return False
        if not np.array_equal(x, y):
            return False
    return True


def test_checkpoint_roundtrip_bitexact():
    _, params, a_rec, a_raw, scan_fn = fputil.build_net()
    so, sr, traj = fputil.make_samples(CFG, params, a_rec, K=2, L_seq=129)

    target = jax.tree_util.tree_map(lambda x: x, params)
    opt = build_optimizer(3e-4, CFG)
    opt_state = opt.init(params)
    # do a real update so params/target/opt_state carry non-trivial values
    params, target, opt_state, m = full_p2_update(
        params, target, opt_state, opt, a_rec, a_raw, scan_fn, so, sr, CFG, 0)
    assert m["finite"]

    # replay buffer WITH the anchored trajectory
    buf = ReplayBuffer(capacity=8, seed=7)
    buf.insert(traj)
    buf.sample(sequence_length=129, start_step=10)   # advance replay RNG state

    # pending buffers WITH an anchor
    pend = PendingEpisodeBuffers(num_envs=3, first_episode_id=5)
    fake_mem = np.random.RandomState(0).standard_normal(
        (CFG.window_mem, CFG.num_layers, CFG.embed)).astype(np.float32)
    fake_mask = np.ones((CFG.num_heads, 1, CFG.window_mem + 1), bool)
    pend.slots[1]["obs"].append(np.zeros(CFG.obs_dim, np.float32))
    pend.add_anchor(1, 0, fake_mem.copy(), fake_mask.copy(), CFG.window_mem)

    rng_key = jax.random.PRNGKey(123)
    action_rng = RU.make_action_rng(99)
    _ = RU.sample_actions(action_rng, np.full((2, CFG.action_dim), 1.0 / CFG.action_dim))
    ar_state = RU.action_rng_state(action_rng)

    tmp = tempfile.mkdtemp(prefix="p2full_ckpt_")
    try:
        CK.save_full_checkpoint(
            params, target, opt_state, buf, rng_key, global_step=2048,
            path=tmp, step=2048, action_rng_state=ar_state, update_count=4,
            pending=pend, collector_state={"x": jnp.ones(3)}, config=CFG, keep=5)

        r = CK.restore_full_checkpoint(tmp, step=2048)
        assert _leaves_equal(r["params"], params), "params not bit-exact"
        assert _leaves_equal(r["target_params"], target), "target not bit-exact"
        assert _leaves_equal(r["opt_state"], opt_state), "opt_state not bit-exact"
        assert np.array_equal(np.asarray(r["rng_key"]), np.asarray(rng_key))
        assert r["global_step"] == 2048 and r["update_count"] == 4

        # replay conservation (incl anchors) via hash digest + anchor bytes
        assert r["replay_buffer"].hash_digest() == buf.hash_digest()
        rt = r["replay_buffer"]._get_by_id(traj.trajectory_id)
        assert rt is not None
        assert np.array_equal(np.asarray(rt.memory_anchors), np.asarray(traj.memory_anchors))
        assert np.array_equal(np.asarray(rt.anchor_steps), np.asarray(traj.anchor_steps))
        rt.validate_anchors()

        # pending conservation (incl anchor bytes)
        rp = r["pending"]
        assert rp.total_pending_transitions() == pend.total_pending_transitions()
        assert rp.total_pending_anchors() == pend.total_pending_anchors()
        assert np.array_equal(np.asarray(rp.slots[1]["anchor_mem"][0]), fake_mem)
        assert rp.slots[1]["anchor_idx"][0] == CFG.window_mem

        # action RNG resumes the SAME stream
        rng2 = RU.restore_action_rng(r["action_rng_state"], seed=99)
        p = np.full((2, CFG.action_dim), 1.0 / CFG.action_dim)
        a_cont = RU.sample_actions(action_rng, p)     # continuing original stream
        # (action_rng already advanced above; re-create from saved state to compare)
        rng_a = RU.restore_action_rng(ar_state, seed=99)
        rng_b = RU.restore_action_rng(r["action_rng_state"], seed=99)
        assert np.array_equal(RU.sample_actions(rng_a, p), RU.sample_actions(rng_b, p))

        # collector state
        assert np.array_equal(np.asarray(r["collector_state"]["x"]), np.ones(3))
        print("PASS CKPT.rt + CKPT.anchor bit-exact params/target/opt/replay/pending/rng")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_checkpoint_strip_and_inventory():
    _, params, a_rec, _, _ = fputil.build_net()
    buf = ReplayBuffer(capacity=4, seed=1)
    target = jax.tree_util.tree_map(lambda x: x, params)
    opt = build_optimizer(3e-4, CFG); opt_state = opt.init(params)
    tmp = tempfile.mkdtemp(prefix="p2full_strip_")
    try:
        for s in [100, 200, 300, 400, 500]:
            CK.save_full_checkpoint(params, target, opt_state, buf,
                                    jax.random.PRNGKey(0), global_step=s,
                                    path=tmp, step=s, update_count=s // 100,
                                    config=CFG, keep=3)
        inv = CK.checkpoint_inventory(tmp)
        steps = [e["step"] for e in inv["steps"]]
        assert steps == [300, 400, 500], f"strip failed, kept {steps}"
        assert CK.detect_latest_step(tmp) == 500
        assert inv["latest_step"] == 500
        # manifest carries provenance
        man = inv["steps"][-1]["manifest"]
        assert man["params_sha256"] == CK.params_content_sha256(params)
        assert man["param_count"] == CK.param_count(params)
        assert man["format"] == "p2_full_a_pure_pickle_v1"
        print(f"PASS CKPT.strip kept {steps}; detect_latest=500; manifest provenance OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_checkpoint_roundtrip_bitexact()
    test_checkpoint_strip_and_inventory()
    print("ALL_CHECKPOINT_ROUNDTRIP_TESTS_PASS")
