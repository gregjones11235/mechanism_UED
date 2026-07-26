"""Gate 2 — V-trace value path on the REAL network (CPU).

  G2.ref    the network-fed V-trace target vs matches an independent numpy reference
  G2.onpol  at init current==behavior policy -> ratio=1, ess=1 (ratio uses stored mu)
  G2.value  combined update keeps value loss finite/bounded, params change, KL reported
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import sys, os.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
import jax

import fputil
from full_p2_learner import (FullP2Config, build_optimizer, full_p2_update,
                             diagnose_vtrace)

CFG = fputil.CFG


def numpy_vtrace(log_pi, log_mu, v_online, v_target_tp1, rewards, dones,
                 bootstrap, gamma, rho_bar, c_bar, lo, hi):
    B, T = v_online.shape
    ratio = np.exp(np.clip(log_pi - log_mu, -30, 30))
    rb = np.minimum(rho_bar, ratio); c = np.minimum(c_bar, ratio)
    nd = 1.0 - dones
    delta = rb * (rewards + gamma * v_target_tp1 * nd - v_online)
    g = np.zeros((B, T + 1))
    for t in range(T - 1, -1, -1):
        g[:, t] = delta[:, t] + gamma * c[:, t] * nd[:, t] * g[:, t + 1]
    vs = np.clip(v_online + g[:, :T], lo, hi)
    return vs, ratio


def test_gate2_target_matches_reference():
    _, params, a_rec, a_raw, scan_fn = fputil.build_net()
    so, sr, _ = fputil.make_samples(CFG, params, a_rec, K=2, L_seq=129)
    d = diagnose_vtrace(params, params, scan_fn, a_rec, so, CFG)  # target init = online
    vs_ref, ratio_ref = numpy_vtrace(
        d["log_pi"], d["log_mu"], d["v_online"], d["v_target_tp1"], d["rewards"],
        d["dones"], d["bootstrap"], CFG.gamma, CFG.rho_bar, CFG.c_bar,
        CFG.vt_clip_min, CFG.vt_clip_max)
    assert np.allclose(d["vs"], vs_ref, atol=1e-3), "network V-trace target != reference"
    assert np.allclose(d["ratio"], ratio_ref, atol=1e-3)
    assert np.allclose(d["log_pi"], d["log_mu"], atol=1e-4), "behavior mu mismatch at init"
    assert np.allclose(d["ratio"], 1.0, atol=1e-4)
    assert abs(d["ess"] - 1.0) < 1e-3, d["ess"]
    assert np.all(np.isfinite(d["vs"]))
    print(f"PASS G2.ref/G2.onpol vs matches ref; ratio=1 ess={d['ess']:.4f}")


def test_gate2_value_update_healthy():
    _, params, a_rec, a_raw, scan_fn = fputil.build_net()
    so, sr, _ = fputil.make_samples(CFG, params, a_rec, K=2, L_seq=129)
    target = jax.tree_util.tree_map(lambda x: x, params)
    # LR inside the frozen KL replay gate so each combined update is ACCEPTED.
    opt = build_optimizer(1e-5, CFG)
    opt_state = opt.init(params)
    params0 = params
    vv = []
    for uc in range(3):
        params, target, opt_state, m = full_p2_update(
            params, target, opt_state, opt, a_rec, a_raw, scan_fn, so, sr, CFG, uc)
        assert m["finite"], "non-finite loss"
        assert np.isfinite(float(m["vtrace_value"]))
        assert np.isfinite(float(m["vtrace_actor"]))
        vv.append(float(m["vtrace_value"]))
        assert "policy_kl" in m and "kl_gate_pass" in m
        assert m["policy_committed"] is True, "combined update rejected by KL gate"
        assert float(m["policy_kl"]) <= CFG.kl_replay_max, m["policy_kl"]
    # Shared-backbone actor-critic: actor updates perturb the backbone, so the value
    # loss need NOT be monotone over a few combined steps (vs is stop-gradient target).
    # Health gate = finite, no explosion, params move (correctness proven in G2.ref).
    assert all(np.isfinite(v) for v in vv), f"non-finite value loss: {vv}"
    assert max(vv) < 1e5, f"value loss exploded: {vv}"
    assert not np.allclose(np.asarray(params0["actor_out"]["kernel"]),
                           np.asarray(params["actor_out"]["kernel"])), "actor unchanged"
    assert not np.allclose(np.asarray(params0["critic_out"]["kernel"]),
                           np.asarray(params["critic_out"]["kernel"])), "critic unchanged"
    print(f"PASS G2.value finite/bounded over updates {vv}; actor+critic moved; KL reported")


if __name__ == "__main__":
    test_gate2_target_matches_reference()
    test_gate2_value_update_healthy()
    print("ALL_GATE2_VTRACE_VALUE_TESTS_PASS")
