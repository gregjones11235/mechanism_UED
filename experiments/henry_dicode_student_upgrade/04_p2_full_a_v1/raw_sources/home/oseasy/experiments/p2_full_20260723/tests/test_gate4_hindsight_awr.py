"""Gate 4 — hindsight AWR (relabeled) on the REAL network (CPU).

  G4.nocross  NETWORK-LEVEL no-cross-goal-ratio: perturbing behavior log_probs (mu)
              leaves the AWR actor/value terms IDENTICAL but changes the V-trace term
  G4.relabel  relabeled obs/reward really differ from original (goal conditioning active)
  G4.awr      AWR weights bounded by w_max, KL finite, value loss finite; params move
  G4.gate56   relabel to an UNACHIEVED goal is REJECTED (Gate 6); achieved accepted (Gate 5)
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import sys, os.path, copy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
import jax

import fputil
import hindsight as H
from full_p2_learner import (FullP2Config, build_optimizer, full_p2_update,
                             reconstruct_batch, pack_batch, compute_loss,
                             _ext_obs, _target_scan)

CFG = fputil.CFG


def run_loss(params, target, a_rec, a_raw, scan_fn, so, sr, cfg, uc=0):
    po = pack_batch(so); pr = pack_batch(sr)
    obs_o_ext = _ext_obs(po); obs_r_ext = _ext_obs(pr)
    recon_o = reconstruct_batch(a_rec, params, so, cfg)
    recon_r = reconstruct_batch(a_rec, params, sr, cfg)
    recon_o_t = reconstruct_batch(a_rec, target, so, cfg)
    recon_r_t = reconstruct_batch(a_rec, target, sr, cfg)
    tvo = _target_scan(scan_fn, target, recon_o_t, obs_o_ext)
    tvr = _target_scan(scan_fn, target, recon_r_t, obs_r_ext)
    _, m = compute_loss(params, a_raw, po, pr, obs_o_ext, obs_r_ext, tvo, tvr,
                        recon_o, recon_r, cfg, uc)
    return m


def test_gate4_no_cross_goal_ratio():
    _, params, a_rec, a_raw, scan_fn = fputil.build_net()
    so, sr, _ = fputil.make_samples(CFG, params, a_rec, K=2, L_seq=129)
    m_a = run_loss(params, params, a_rec, a_raw, scan_fn, so, sr, CFG)
    so_b = [copy.deepcopy(s) for s in so]
    rng = np.random.RandomState(0)
    for s in so_b:
        s.log_probs = s.log_probs + rng.standard_normal(s.log_probs.shape).astype(np.float32) * 3.0
    m_b = run_loss(params, params, a_rec, a_raw, scan_fn, so_b, sr, CFG)
    assert abs(float(m_a["awr_actor"]) - float(m_b["awr_actor"])) < 1e-5, "AWR actor depends on mu!"
    assert abs(float(m_a["awr_value"]) - float(m_b["awr_value"])) < 1e-5, "AWR value depends on mu!"
    assert abs(float(m_a["vtrace_actor"]) - float(m_b["vtrace_actor"])) > 1e-6, "V-trace insensitive to mu"
    assert float(m_a["ess"]) != float(m_b["ess"])
    print("PASS G4.nocross AWR invariant to behavior mu; V-trace sensitive to mu")


def test_gate4_relabel_active():
    _, params, a_rec, a_raw, scan_fn = fputil.build_net()
    so, sr, _ = fputil.make_samples(CFG, params, a_rec, K=2, L_seq=129)
    assert not np.allclose(np.asarray(sr[0].observations)[:, -67:],
                           np.asarray(so[0].observations)[:, -67:])
    assert np.allclose(np.asarray(sr[0].observations)[:, -67:], H.goal_embedding(2, 67))
    assert float(np.asarray(sr[0].rewards).sum()) != float(np.asarray(so[0].rewards).sum())
    print("PASS G4.relabel goal conditioning + reward active on relabeled samples")


def test_gate4_awr_healthy_and_moves_params():
    _, params, a_rec, a_raw, scan_fn = fputil.build_net()
    so, sr, _ = fputil.make_samples(CFG, params, a_rec, K=2, L_seq=129)
    m = run_loss(params, params, a_rec, a_raw, scan_fn, so, sr, CFG)
    assert np.isfinite(float(m["awr_actor"]))
    assert np.isfinite(float(m["awr_value"]))
    assert np.isfinite(float(m["awr_kl"]))
    assert float(m["awr_w_mean"]) <= CFG.w_max + 1e-4
    target = jax.tree_util.tree_map(lambda x: x, params)
    # LR inside the frozen KL replay gate so the combined update is ACCEPTED.
    opt = build_optimizer(1e-5, CFG); opt_state = opt.init(params)
    params0 = params
    params, target, opt_state, m2 = full_p2_update(
        params, target, opt_state, opt, a_rec, a_raw, scan_fn, so, sr, CFG, 0)
    assert m2["finite"]
    assert m2["policy_committed"] is True and m2["kl_rejected_update"] is False, m2
    assert float(m2["policy_kl"]) <= CFG.kl_replay_max, m2["policy_kl"]
    assert not np.allclose(np.asarray(params0["actor_out"]["kernel"]),
                           np.asarray(params["actor_out"]["kernel"]))
    print(f"PASS G4.awr weights<=w_max, KL finite, params moved through gate "
          f"(w_mean={float(m['awr_w_mean']):.3f} kl={float(m2['policy_kl']):.5f})")


def test_gate4_gate56_enforced():
    _, params, a_rec, a_raw, scan_fn = fputil.build_net()
    so, sr, _ = fputil.make_samples(CFG, params, a_rec, K=1, L_seq=129)
    r = H.relabel_sample(so[0], goal_index=fputil.HINDSIGHT_GOAL, embedding_size=67)
    assert r.target_achievements[fputil.HINDSIGHT_GOAL] == 1.0
    try:
        H.relabel_sample(so[0], goal_index=3, embedding_size=67)  # goal 3 never achieved
        raise AssertionError("Gate 6 should reject unachieved goal")
    except ValueError:
        pass
    print("PASS G4.gate56 achieved accepted, unachieved rejected")


if __name__ == "__main__":
    test_gate4_no_cross_goal_ratio()
    test_gate4_relabel_active()
    test_gate4_awr_healthy_and_moves_params()
    test_gate4_gate56_enforced()
    print("ALL_GATE4_HINDSIGHT_AWR_TESTS_PASS")
