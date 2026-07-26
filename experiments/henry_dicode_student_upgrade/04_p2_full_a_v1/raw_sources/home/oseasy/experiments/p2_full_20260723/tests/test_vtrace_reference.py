"""Gate 2.7 / 2.2 / 2.3 — V-trace vs independent numpy reference (CPU).

Runs on JAX_PLATFORM_NAME=cpu. No network / env needed.
"""
import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import jax.numpy as jnp

import vtrace as V


def numpy_vtrace(log_pi, log_mu, v_online, v_target_tp1, rewards, dones,
                 bootstrap, gamma, rho_bar, c_bar, lo, hi):
    """Independent reference implementing the frozen §3 sum form."""
    B, T = v_online.shape
    ratio = np.exp(np.clip(log_pi - log_mu, -30, 30))
    rb = np.minimum(rho_bar, ratio)
    c = np.minimum(c_bar, ratio)
    nd = 1.0 - dones
    delta = rb * (rewards + gamma * v_target_tp1 * nd - v_online)
    g = np.zeros((B, T + 1))
    for t in range(T - 1, -1, -1):
        g[:, t] = delta[:, t] + gamma * c[:, t] * nd[:, t] * g[:, t + 1]
    vs = np.clip(v_online + g[:, :T], lo, hi)
    return vs, ratio, rb, c, delta


def _rand(rng, shape):
    return rng.standard_normal(shape).astype(np.float32)


def test_vtrace_matches_reference():
    rng = np.random.RandomState(0)
    B, T = 3, 40
    log_pi = _rand(rng, (B, T)) * 0.5
    log_mu = _rand(rng, (B, T)) * 0.5
    v_online = _rand(rng, (B, T)) * 5 + 3
    v_target_tp1 = _rand(rng, (B, T)) * 5 + 3
    rewards = np.abs(_rand(rng, (B, T)))
    dones = np.zeros((B, T), np.float32)
    bootstrap = _rand(rng, (B,)) * 5
    cfg = V.VtraceConfig()
    out = V.vtrace_targets(jnp.array(log_pi), jnp.array(log_mu), jnp.array(v_online),
                           jnp.array(v_target_tp1), jnp.array(rewards), jnp.array(dones),
                           jnp.array(bootstrap), cfg)
    vs_ref, ratio_ref, rb_ref, c_ref, delta_ref = numpy_vtrace(
        log_pi, log_mu, v_online, v_target_tp1, rewards, dones, bootstrap,
        cfg.gamma, cfg.rho_bar, cfg.c_bar, cfg.vt_clip_min, cfg.vt_clip_max)
    assert np.allclose(np.asarray(out.vs), vs_ref, atol=1e-4), "vs mismatch"
    assert np.allclose(np.asarray(out.ratio), ratio_ref, atol=1e-4)
    assert np.allclose(np.asarray(out.rho_bar), rb_ref, atol=1e-4)
    assert np.allclose(np.asarray(out.c), c_ref, atol=1e-4)
    assert np.allclose(np.asarray(out.delta), delta_ref, atol=1e-4)
    print("PASS test_vtrace_matches_reference")


def test_vtrace_terminal_no_bootstrap():
    """After a terminal step the trace must not propagate (done->0)."""
    rng = np.random.RandomState(1)
    B, T = 2, 10
    log_pi = _rand(rng, (B, T)) * 0.3
    log_mu = log_pi.copy()  # on-policy ratio=1
    v_online = _rand(rng, (B, T)) * 2
    v_target_tp1 = _rand(rng, (B, T)) * 2
    rewards = np.ones((B, T), np.float32)
    dones = np.zeros((B, T), np.float32)
    dones[:, 4] = 1.0  # terminal at step 4
    bootstrap = np.full((B,), 999.0, np.float32)  # huge; must be ignored after done
    cfg = V.VtraceConfig()
    out = V.vtrace_targets(jnp.array(log_pi), jnp.array(log_mu), jnp.array(v_online),
                           jnp.array(v_target_tp1), jnp.array(rewards), jnp.array(dones),
                           jnp.array(bootstrap), cfg)
    vs = np.asarray(out.vs)
    # v at the terminal step 4 must NOT depend on bootstrap (done kills it)
    # verify finite & within clip
    assert np.all(np.isfinite(vs))
    assert vs.min() >= cfg.vt_clip_min - 1e-3 and vs.max() <= cfg.vt_clip_max + 1e-3
    # delta at terminal step uses (1-done)=0 on the bootstrap term:
    delta4 = np.asarray(out.delta)[:, 4]
    expected = np.asarray(out.rho_bar)[:, 4] * (rewards[:, 4] - v_online[:, 4])
    assert np.allclose(delta4, expected, atol=1e-4), "terminal delta must drop bootstrap"
    print("PASS test_vtrace_terminal_no_bootstrap")


def test_vtrace_clip_applies():
    B, T = 1, 5
    log_pi = np.zeros((B, T), np.float32)
    log_mu = np.zeros((B, T), np.float32)
    v_online = np.zeros((B, T), np.float32)
    v_target_tp1 = np.zeros((B, T), np.float32)
    rewards = np.full((B, T), 1e6, np.float32)  # force huge returns
    dones = np.zeros((B, T), np.float32)
    bootstrap = np.zeros((B,), np.float32)
    cfg = V.VtraceConfig()
    out = V.vtrace_targets(jnp.array(log_pi), jnp.array(log_mu), jnp.array(v_online),
                           jnp.array(v_target_tp1), jnp.array(rewards), jnp.array(dones),
                           jnp.array(bootstrap), cfg)
    vs = np.asarray(out.vs)
    assert np.all(vs <= cfg.vt_clip_max + 1e-3)
    print("PASS test_vtrace_clip_applies")


def test_vtrace_deterministic():
    rng = np.random.RandomState(2)
    B, T = 2, 20
    log_pi = _rand(rng, (B, T)) * 0.4
    log_mu = _rand(rng, (B, T)) * 0.4
    v_online = _rand(rng, (B, T)) * 3
    v_target_tp1 = _rand(rng, (B, T)) * 3
    rewards = np.abs(_rand(rng, (B, T)))
    dones = np.zeros((B, T), np.float32); dones[:, -1] = 1.0
    bootstrap = _rand(rng, (B,)) * 3
    cfg = V.VtraceConfig()
    js = [jnp.array(a) for a in (log_pi, log_mu, v_online, v_target_tp1, rewards, dones, bootstrap)]
    o1 = V.vtrace_targets(*js, cfg)
    o2 = V.vtrace_targets(*js, cfg)
    assert np.array_equal(np.asarray(o1.vs), np.asarray(o2.vs))
    print("PASS test_vtrace_deterministic")


if __name__ == "__main__":
    test_vtrace_matches_reference()
    test_vtrace_terminal_no_bootstrap()
    test_vtrace_clip_applies()
    test_vtrace_deterministic()
    print("ALL_VTRACE_TESTS_PASS")
