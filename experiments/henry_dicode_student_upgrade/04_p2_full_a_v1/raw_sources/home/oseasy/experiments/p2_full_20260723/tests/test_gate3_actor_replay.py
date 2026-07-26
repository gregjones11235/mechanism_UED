"""Gate 3 — actor + sequence replay on the REAL network (CPU).

  G3.replay   loss region (>128 steps) is forwarded from RECONSTRUCTED anchor memory;
              the reconstructed forward is history-dependent (long context actually used)
  G3.actor    V-trace actor gradient changes the actor head; loss finite; ess sane
  G3.lag      policy-lag gate excludes samples staler than MAX_POLICY_LAG from AWR
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import sys, os.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
import jax
import jax.numpy as jnp

import fputil
from full_p2_learner import (FullP2Config, build_optimizer, full_p2_update,
                             reconstruct_batch, pack_batch, _ext_obs)

CFG = fputil.CFG


def test_gate3_replay_history_dependent():
    _, params, a_rec, a_raw, scan_fn = fputil.build_net()
    so, sr, _ = fputil.make_samples(CFG, params, a_rec, K=2, L_seq=129)
    assert so[0].length == 129 and so[0].length > 128
    recon = reconstruct_batch(a_rec, params, so, CFG)
    obs_ext = _ext_obs(pack_batch(so))
    lg_real, _, _, _, _ = scan_fn(params, *recon, obs_ext)
    zero_recon = (jnp.zeros_like(recon[0]), recon[1], recon[2])
    lg_zero, _, _, _, _ = scan_fn(params, *zero_recon, obs_ext)
    diff = float(np.abs(np.asarray(lg_real) - np.asarray(lg_zero)).max())
    assert diff > 1e-4, f"G3.replay FAIL: reconstructed memory had no effect ({diff})"
    print(f"PASS G3.replay loss region ({so[0].length}>128) uses reconstructed memory (diff {diff:.5f})")


def test_gate3_actor_changes():
    _, params, a_rec, a_raw, scan_fn = fputil.build_net()
    so, sr, _ = fputil.make_samples(CFG, params, a_rec, K=2, L_seq=129)
    target = jax.tree_util.tree_map(lambda x: x, params)
    # LR chosen inside the frozen KL replay gate so the combined update is ACCEPTED
    # (actor moves through the gate). The rejection/rollback path is covered separately
    # by test_kl_transactional_gate.
    opt = build_optimizer(1e-5, CFG)
    opt_state = opt.init(params)
    params0 = params
    params, target, opt_state, m = full_p2_update(
        params, target, opt_state, opt, a_rec, a_raw, scan_fn, so, sr, CFG, 0)
    assert m["finite"]
    assert np.isfinite(float(m["vtrace_actor"]))
    assert float(m["ess"]) > 0.5, m["ess"]
    assert m["policy_committed"] is True and m["kl_rejected_update"] is False, m
    assert float(m["policy_kl"]) <= CFG.kl_replay_max, m["policy_kl"]
    assert not np.allclose(np.asarray(params0["actor_out"]["kernel"]),
                           np.asarray(params["actor_out"]["kernel"])), "actor unchanged"
    print(f"PASS G3.actor V-trace actor moved actor head through KL gate; "
          f"ess={float(m['ess']):.4f} kl={float(m['policy_kl']):.5f}")


def test_gate3_policy_lag_gate():
    _, params, a_rec, a_raw, scan_fn = fputil.build_net()
    so, sr, _ = fputil.make_samples(CFG, params, a_rec, K=2, L_seq=129)
    target = jax.tree_util.tree_map(lambda x: x, params)
    opt = build_optimizer(3e-4, CFG)
    opt_state = opt.init(params)
    _, _, _, m_fresh = full_p2_update(
        params, target, opt_state, opt, a_rec, a_raw, scan_fn, so, sr, CFG, 0)
    assert float(m_fresh["awr_valid_frac"]) == 1.0, m_fresh["awr_valid_frac"]
    _, _, _, m_stale = full_p2_update(
        params, target, opt_state, opt, a_rec, a_raw, scan_fn, so, sr, CFG,
        CFG.max_policy_lag + 10)
    assert float(m_stale["awr_valid_frac"]) == 0.0, m_stale["awr_valid_frac"]
    print("PASS G3.lag policy-lag gate: fresh->AWR in (1.0), stale->AWR out (0.0)")


if __name__ == "__main__":
    test_gate3_replay_history_dependent()
    test_gate3_actor_changes()
    test_gate3_policy_lag_gate()
    print("ALL_GATE3_ACTOR_REPLAY_TESTS_PASS")
