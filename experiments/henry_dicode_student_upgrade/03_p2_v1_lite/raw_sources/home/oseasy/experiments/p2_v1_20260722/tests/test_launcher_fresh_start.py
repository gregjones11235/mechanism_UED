#!/usr/bin/env python3
"""P2-v1 launcher fresh-start CPU regression (第五节).

Complements the GPU `--smoke-test` (which does the real weights-only load via
load_weights_only, requiring GPU).  This file runs on CPU and verifies the
launcher's fresh-start CONTRACT and helpers without any GPU/weights load:

  1. v0 source guard rejects 98304/122880/henry_student_p2_amago and accepts
     the healthy session175 path (fail closed, no override).
  2. P2-v1 start constants: source ends with /17500, global_step/update_count
     start at 0, master seed is a documented int, GPU0 uuid.
  3. encoder-kernel finder recovers (obs_dim, embed_size) from a params tree.
  4. fresh optimizer step counter == 0 (brand-new Adam, no carried moments).
  5. fresh replay buffer is empty (size 0).
  6. action RNG determinism/independence: same seed -> identical stream,
     different seed -> different stream (no global np.random).
  7. P2-v1 checkpoint round-trip on CPU: save_full_checkpoint ->
     restore_full_checkpoint preserves params/optimizer/rng/global_step/
     update_count/action_rng bit-exact.
  8. py_compile of launcher + all src files.

This does NOT replace the GPU weights-only load check, and prior CPU-suite green
does NOT replace this launcher regression (per 第五节).
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import jax
import jax.numpy as jnp

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC)

# Importing the launcher inserts _HENRY_SRC (dicode) and _AMAGO_SRC (src) on path.
import stage4_continue_launcher as L
from rng_utils import (make_action_rng, sample_actions, action_rng_state,
                       restore_action_rng)
from trajectory_replay import TrajectoryReplayBuffer
from checkpointing import save_full_checkpoint, restore_full_checkpoint


GPU0_UUID = "GPU-e8c08612-c22a-c8a4-6df5-affb2dd1f9a6"


def _sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ---------------------------------------------------------------------------
# 1. v0 source guard
# ---------------------------------------------------------------------------
def test_v0_source_guard():
    forbidden = [
        "/home/oseasy/experiments/henry_student_p2_amago_20260721/checkpoints/98304",
        "/some/path/122880",
        "henry_student_p2_amago_20260721/whatever",
    ]
    for p in forbidden:
        try:
            L.verify_not_v0_source(p)
            assert False, f"guard FAILED to reject forbidden path: {p}"
        except RuntimeError as e:
            assert "REFUSED" in str(e)
    # healthy session175 must be accepted
    L.verify_not_v0_source(L.SESSION175_CKPT)
    print("  PASS v0 source guard: rejects 98304/122880/v0-dir, accepts session175")


# ---------------------------------------------------------------------------
# 2. start constants
# ---------------------------------------------------------------------------
def test_start_constants():
    assert L.SESSION175_CKPT.rstrip("/").endswith("/17500"), L.SESSION175_CKPT
    assert "base_ckpt_v7fix55_armA_s0" in L.SESSION175_CKPT
    assert L.P2_V1_GLOBAL_STEP_START == 0
    assert L.P2_V1_UPDATE_COUNT_START == 0
    assert isinstance(L.P2_V1_MASTER_SEED, int)
    assert L.EXPECTED_GPU_UUID == GPU0_UUID
    assert L.REPLAY_CAPACITY > 0
    # arithmetic invariant still holds
    assert L.NUM_ENVS * L.ROLLOUT_STEPS * L.NUM_UPDATES_PER_SESSION == L.SESSION_ENV_STEPS
    print("  PASS start constants: source=/17500 (v7fix55), global_step0, "
          "update_count0, seed int, GPU0, replay cap>0")


# ---------------------------------------------------------------------------
# 3. encoder-kernel finder
# ---------------------------------------------------------------------------
def _build_small_network_and_params(obs_dim=64):
    cfg = L.Cfg()
    network = L.ActorCriticTransformer(
        action_dim=43, activation=cfg.activation, hidden_layers=cfg.hidden_layers,
        encoder_size=cfg.embed_size, num_heads=cfg.num_heads,
        qkv_features=cfg.qkv_features, num_layers=cfg.num_layers,
        gating=cfg.gating, gating_bias=cfg.gating_bias)
    params = L.init_network_params(network, obs_dim, cfg, jax.random.PRNGKey(0))
    return cfg, network, params, obs_dim


def test_encoder_kernel_finder():
    cfg, network, params, obs_dim = _build_small_network_and_params(obs_dim=64)
    shape = L._find_encoder_kernel_shape(params)
    assert shape == (64, cfg.embed_size), f"kernel finder got {shape}"
    print(f"  PASS encoder-kernel finder: recovered {shape} == (64, {cfg.embed_size})")


# ---------------------------------------------------------------------------
# 4. fresh optimizer step == 0
# ---------------------------------------------------------------------------
def test_fresh_optimizer_step_zero():
    cfg, network, params, obs_dim = _build_small_network_and_params(obs_dim=64)
    ts = L.build_stage4_train_state(network, params, cfg)
    step = L._optimizer_step_count(ts)
    assert step == 0, f"fresh optimizer step {step} != 0"
    print(f"  PASS fresh optimizer step == {step} (brand-new Adam)")


# ---------------------------------------------------------------------------
# 5. fresh replay buffer empty
# ---------------------------------------------------------------------------
def test_fresh_replay_empty():
    buf = TrajectoryReplayBuffer(capacity=L.REPLAY_CAPACITY, seed=L.P2_V1_MASTER_SEED)
    assert len(buf) == 0, f"fresh replay not empty (len={len(buf)})"
    print(f"  PASS fresh replay buffer empty (len={len(buf)}, cap={L.REPLAY_CAPACITY})")


# ---------------------------------------------------------------------------
# 6. action RNG determinism / independence
# ---------------------------------------------------------------------------
def test_action_rng_determinism():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((4, 43))
    softmax = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = softmax / softmax.sum(axis=-1, keepdims=True)

    def stream(g, n=10):
        return [int(sample_actions(g, probs)[0]) for _ in range(n)]

    a = stream(make_action_rng(L.P2_V1_MASTER_SEED))
    b = stream(make_action_rng(L.P2_V1_MASTER_SEED))
    c = stream(make_action_rng(L.P2_V1_MASTER_SEED + 1))
    assert a == b, "same seed produced different streams"
    assert a != c, "different seeds produced identical streams"

    # state save/restore reproduces the continuing stream
    g_live = make_action_rng(L.P2_V1_MASTER_SEED)
    _ = stream(g_live, n=3)  # advance
    st = action_rng_state(g_live)
    g_restored = restore_action_rng(st, seed=L.P2_V1_MASTER_SEED)
    assert stream(g_live) == stream(g_restored), "restored action RNG diverges"
    print("  PASS action RNG: same-seed identical, diff-seed differs, "
          "state save/restore continues identically")


# ---------------------------------------------------------------------------
# 7. checkpoint round-trip on CPU (P2-v1 self-checkpoint format)
# ---------------------------------------------------------------------------
def test_checkpoint_roundtrip_cpu(tmp_dir):
    cfg, network, params, obs_dim = _build_small_network_and_params(obs_dim=64)
    ts = L.build_stage4_train_state(network, params, cfg)
    replay = TrajectoryReplayBuffer(capacity=L.REPLAY_CAPACITY, seed=L.P2_V1_MASTER_SEED)
    rng = jax.random.PRNGKey(L.P2_V1_MASTER_SEED)
    action_rng = make_action_rng(L.P2_V1_MASTER_SEED)
    global_step = 0
    update_count = 0

    save_full_checkpoint(
        ts, replay, rng, global_step, tmp_dir, step=global_step,
        action_rng_state=action_rng_state(action_rng), update_count=update_count)

    dummy_ts = L.build_stage4_train_state(
        network, L.init_network_params(network, obs_dim, cfg, jax.random.PRNGKey(1)), cfg)
    r = restore_full_checkpoint(tmp_dir, dummy_ts, step=global_step)

    assert r["global_step"] == global_step, "global_step mismatch"
    assert r["update_count"] == update_count, "update_count mismatch"
    assert len(r["replay_buffer"]) == len(replay), "replay size mismatch"
    assert bool(jnp.all(r["rng"] == rng)), "JAX rng mismatch"
    assert r["action_rng_state"] is not None, "action_rng_state missing"

    params_eq = all(bool(jnp.all(l1 == l2)) for l1, l2 in zip(
        jax.tree_util.tree_leaves(ts.params),
        jax.tree_util.tree_leaves(r["train_state"].params)))
    assert params_eq, "params mismatch after round-trip"

    opt_eq = all(bool(jnp.all(jnp.asarray(l1) == jnp.asarray(l2)))
                 for l1, l2 in zip(
                     jax.tree_util.tree_leaves(ts.opt_state),
                     jax.tree_util.tree_leaves(r["train_state"].opt_state)))
    assert opt_eq, "optimizer state mismatch after round-trip"

    # restored action RNG continues identically to the live one
    rng_probes = np.random.default_rng(7).standard_normal((4, 43))
    sm = np.exp(rng_probes - rng_probes.max(axis=-1, keepdims=True))
    pr = sm / sm.sum(axis=-1, keepdims=True)
    g_restored = restore_action_rng(r["action_rng_state"], seed=L.P2_V1_MASTER_SEED)
    seq_r = [int(sample_actions(g_restored, pr)[0]) for _ in range(8)]
    seq_l = [int(sample_actions(action_rng, pr)[0]) for _ in range(8)]
    assert seq_r == seq_l, "restored action RNG stream diverges from live"
    print("  PASS checkpoint round-trip (CPU): params/optimizer/rng/global_step/"
          "update_count/action_rng all bit-exact")


# ---------------------------------------------------------------------------
# 8. py_compile launcher + all src
# ---------------------------------------------------------------------------
def test_py_compile():
    import py_compile
    for fn in sorted(os.listdir(SRC)):
        if fn.endswith(".py"):
            py_compile.compile(os.path.join(SRC, fn), doraise=True)
    print("  PASS py_compile: all src/*.py compile")


def main():
    import shutil
    import tempfile
    results = {}
    tmp_dir = tempfile.mkdtemp(prefix="p2v1_launcher_rt_")
    tests = [
        ("v0 source guard", test_v0_source_guard, ()),
        ("start constants", test_start_constants, ()),
        ("encoder-kernel finder", test_encoder_kernel_finder, ()),
        ("fresh optimizer step==0", test_fresh_optimizer_step_zero, ()),
        ("fresh replay empty", test_fresh_replay_empty, ()),
        ("action RNG determinism", test_action_rng_determinism, ()),
        ("checkpoint round-trip CPU", test_checkpoint_roundtrip_cpu, (tmp_dir,)),
        ("py_compile", test_py_compile, ()),
    ]
    passed = failed = 0
    for name, fn, args in tests:
        try:
            fn(*args)
            passed += 1
            results[name] = "PASS"
        except Exception as e:
            import traceback
            failed += 1
            results[name] = f"FAIL: {e}"
            print(f"  FAIL {name}: {e}")
            traceback.print_exc()
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n{'='*60}")
    print(f"P2-v1 launcher fresh-start CPU regression: {passed}/{passed+failed} PASS")
    print(f"{'='*60}")

    report = {
        "directive": "P2-v1 第五节 launcher regression (CPU part)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "passed": passed, "failed": failed, "total": passed + failed,
        "results": results,
        "source_hashes": {
            "stage4_continue_launcher.py": _sha256_file(
                os.path.join(SRC, "stage4_continue_launcher.py")),
        },
        "note": "weights-only LOAD check runs separately on GPU0 via --smoke-test",
    }
    ev = os.path.join(os.path.dirname(__file__), "..", "evidence")
    os.makedirs(ev, exist_ok=True)
    with open(os.path.join(ev, "launcher_cpu_regression_report.json"), "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")

    if failed:
        sys.exit(1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
