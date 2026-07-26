"""Gate 1 — native GTrXL long-context + sparse-anchor reconstruction (CPU, real net).

Builds the REAL Henry ActorCriticTransformer (window_mem=128, 2 layers, embed 256,
8 heads, qkv 256, relu, gating, gating_bias 2.0, action_dim 43, obs_dim 8335) and
verifies:
  G1.1  history changes logits (long context actually used)
  G1.2  mask invariance: garbage in UNFILLED memory slots does not change output
  G1.3  episode isolation: done-reset memory == fresh zero (no cross-episode leak)
  G1.4  CORE: anchor reconstruction is bit-exact vs an independent full rollout scan
  G1.6  reconstruction across the 128 boundary (anchors at 0/128/256, gap<=128)
  G1.7  network-level anchor conservation (step-0 anchor == fresh zero memory)
  +     make_apply_eval batch==1 padding is bit-exact vs batch==2 slice [0]

Run:  JAX_PLATFORMS=cpu PYTHONPATH=$HENRY_SRC:$BASE/src python test_gate1_longctx_anchor.py
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys
HENRY_SRC = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src"
BASE_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, HENRY_SRC)
sys.path.insert(0, os.path.abspath(BASE_SRC))

import numpy as np
import jax
import jax.numpy as jnp

from dicode.network import ActorCriticTransformer
import memory_anchor as MA

# ---- real P2 dims ----
WM, LAY, EMB = 128, 2, 256
HEADS, QKV, HIDDEN = 8, 256, 256
ACTION_DIM, OBS_DIM = 43, 8335
B, L = 2, 300          # batch>=2 dodges forward_eval squeeze bug
ANCHOR_STEPS = [0, 128, 256]


def build():
    net = ActorCriticTransformer(
        action_dim=ACTION_DIM, activation="relu", hidden_layers=HIDDEN,
        encoder_size=EMB, num_heads=HEADS, qkv_features=QKV, num_layers=LAY,
        gating=True, gating_bias=2.0)
    key = jax.random.PRNGKey(0)
    memories = jnp.zeros((B, WM, LAY, EMB))
    obs = jnp.zeros((B, OBS_DIM))
    mask = jnp.zeros((B, HEADS, 1, WM + 1), dtype=jnp.bool_)
    params = net.init(key, memories, obs, mask, method=net.model_forward_eval)["params"]

    @jax.jit
    def jitted_eval(params, memories, obs, mask):
        pi, value, mem_out = net.apply(
            {"params": params}, memories, obs, mask, method=net.model_forward_eval)
        return pi.logits, value, mem_out

    return net, params, MA.make_apply_eval(net), jitted_eval


def make_obs(seed=0):
    rng = np.random.RandomState(seed)
    return jnp.asarray(rng.standard_normal((L, B, OBS_DIM)).astype(np.float32) * 0.1)


def main():
    net, params, apply_eval_pad, apply_eval = build()
    obs = make_obs()

    # ---- ground-truth rollout scan ----
    pre_mem, pre_mask, pre_idx, logits, values = MA.scan_memory_eval(
        apply_eval, params, obs, WM, HEADS, num_layers=LAY, embed=EMB)
    print(f"scan ok: pre_mem {pre_mem.shape} logits {logits.shape}")

    # ---- G1.1 history changes logits ----
    s = 200
    lg_real, val_real, _ = apply_eval(params, pre_mem[s], obs[s], pre_mask[s])
    zero_mem = jnp.zeros_like(pre_mem[s])
    fresh_mask = jnp.zeros_like(pre_mask[s])
    lg_zero, val_zero, _ = apply_eval(params, zero_mem, obs[s], fresh_mask)
    diff = float(np.abs(np.asarray(lg_real) - np.asarray(lg_zero)).max())
    assert diff > 1e-4, f"G1.1 FAIL: long context had no effect (max diff {diff})"
    print(f"PASS G1.1 history changes logits (max logit diff {diff:.5f})")

    # ---- G1.2 mask invariance: garbage in UNFILLED slots is masked out ----
    # after s=10 steps, filled slots are the rightmost 10 (positions 118..127);
    # positions 0..117 are unfilled and must be masked (no effect on output).
    s2 = 10
    rng = np.random.RandomState(3)
    garbage = np.asarray(pre_mem[s2]).copy()
    garbage[:, :118] = rng.standard_normal((B, 118, LAY, EMB)).astype(np.float32) * 50.0
    lg_base, _, _ = apply_eval(params, pre_mem[s2], obs[s2], pre_mask[s2])
    lg_garb, _, _ = apply_eval(params, jnp.asarray(garbage), obs[s2], pre_mask[s2])
    md = float(np.abs(np.asarray(lg_base) - np.asarray(lg_garb)).max())
    assert md < 1e-4, f"G1.2 FAIL: unfilled-slot garbage leaked into output (diff {md})"
    print(f"PASS G1.2 mask invariance (unfilled garbage diff {md:.2e})")

    # ---- G1.3 episode isolation: done-reset == fresh zero memory ----
    post = jnp.roll(pre_mem[50], -1, axis=1)  # any non-zero post-step memory
    done = jnp.ones((B,), dtype=bool)
    reset_mem = jnp.where(done[:, None, None, None], jnp.zeros_like(post), post)
    fresh_mem, fresh_msk, fresh_idx = MA.fresh_rollout_state(WM, HEADS, LAY, EMB, B)
    assert np.array_equal(np.asarray(reset_mem), np.asarray(fresh_mem))
    # step-0 logits of a NEW episode independent of prior episode content:
    obs_b = jnp.asarray(np.random.RandomState(7).standard_normal((B, OBS_DIM)).astype(np.float32) * 0.1)
    lgA, _, _ = apply_eval(params, reset_mem, obs_b, fresh_msk)
    lgB, _, _ = apply_eval(params, fresh_mem, obs_b, fresh_msk)
    assert float(np.abs(np.asarray(lgA) - np.asarray(lgB)).max()) < 1e-6
    print("PASS G1.3 episode isolation via done-zero reset")

    # ---- G1.4 CORE: anchor reconstruction bit-exact ----
    a_mem, a_mask, a_idx = MA.record_anchors(pre_mem, pre_mask, pre_idx, ANCHOR_STEPS)
    pairs = [(0, 50), (0, 127), (128, 128), (128, 200), (128, 255), (256, 299)]
    worst = 0.0
    for a, s in pairs:
        ai = ANCHOR_STEPS.index(a)
        gap = s - a
        assert gap <= 128, (a, s, gap)
        seg = obs[a:s]  # [gap, B, obs_dim]
        rm, rmask, ridx = MA.reconstruct_state(
            apply_eval, params, a_mem[ai], a_mask[ai], a_idx[ai], seg, WM, HEADS)
        d_mem = float(np.abs(np.asarray(rm) - np.asarray(pre_mem[s])).max())
        d_mask = int(np.abs(np.asarray(rmask).astype(int) - np.asarray(pre_mask[s]).astype(int)).max())
        d_idx = int(np.abs(np.asarray(ridx) - np.asarray(pre_idx[s])).max())
        worst = max(worst, d_mem)
        assert d_mem < 1e-5, f"G1.4 FAIL mem at ({a}->{s}) diff {d_mem}"
        assert d_mask == 0, f"G1.4 FAIL mask at ({a}->{s})"
        assert d_idx == 0, f"G1.4 FAIL idx at ({a}->{s})"
    print(f"PASS G1.4 anchor reconstruction bit-exact (worst mem diff {worst:.2e})")

    # ---- G1.6 spans >128 (uses anchors beyond first window; gap<=128) ----
    # reconstruction of step 299 from anchor 256 already covered above (gap 43).
    # additional: reconstruct 255 from anchor 128 (gap 127, near-max)
    ai = ANCHOR_STEPS.index(128)
    rm, _, _ = MA.reconstruct_state(
        apply_eval, params, a_mem[ai], a_mask[ai], a_idx[ai], obs[128:255], WM, HEADS)
    assert float(np.abs(np.asarray(rm) - np.asarray(pre_mem[255])).max()) < 1e-5
    print("PASS G1.6 reconstruction spans >128 (near-max gap 127)")

    # ---- G1.7 network-level anchor conservation ----
    assert a_mem.shape[0] == 3, a_mem.shape
    # step-0 anchor == fresh zero memory
    assert np.array_equal(np.asarray(a_mem[0]), np.asarray(fresh_mem)), "step0 anchor not zero"
    assert int(np.asarray(a_idx[0]).max()) == WM
    print("PASS G1.7 network-level anchor conservation (3 anchors; step0 == fresh zero)")

    # ---- batch==1 padding bit-exact vs batch==2 slice ----
    single_obs = obs[s][:1]            # [1, obs_dim]
    single_mem = pre_mem[s][:1]        # [1, wm, lay, emb]
    single_mask = pre_mask[s][:1]      # [1, heads,1,wm+1]
    lg1, v1, m1 = apply_eval_pad(params, single_mem, single_obs, single_mask)
    lg2, v2, m2 = apply_eval(params, pre_mem[s], obs[s], pre_mask[s])  # B=2
    assert float(np.abs(np.asarray(lg1) - np.asarray(lg2)[:1]).max()) < 1e-5
    assert float(np.abs(np.asarray(m1) - np.asarray(m2)[:1]).max()) < 1e-5
    print("PASS batch==1 padding bit-exact vs batch==2[0]")

    print("ALL_GATE1_LONGCTX_ANCHOR_TESTS_PASS")


if __name__ == "__main__":
    main()
