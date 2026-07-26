"""KL transactional gate (directive items 1-4) on the REAL network (CPU).

  KL.partition   critic-only leaves are EXACTLY critic_ln1/ln2/out; encoder + GTrXL
                 trunk + actor head are policy-affecting; 6 critic leaves (3 modules x
                 kernel+bias); counts sum to total param count.
  KL.criticinv   perturbing a critic-only leaf (critic_out/kernel) leaves the policy
                 logits UNCHANGED but moves the value; perturbing an actor leaf
                 (actor_out/kernel) moves the logits -> the partition matches the actor
                 forward (a critic-only head truly cannot bypass the KL gate's scope).
  KL.accept      a small-LR update is accepted at actor step scale 1.0
                 (policy_committed, policy_kl<=0.05, actor head moves, finite).
  KL.scale       when scale 1.0 breaches the gate the retry engages a smaller scale and
                 still commits an accepted update (chosen_scale<1.0, KL<=threshold).
  KL.rollback    when EVERY scale breaches (threshold~0): KL_REJECTED_UPDATE ->
                 policy-affecting params BIT-EXACT unchanged; their optimizer moments
                 rolled back; their EMA target frozen; the pure critic head still
                 commits; policy logits bit-exact unchanged while the value moves.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import sys, os.path
from dataclasses import replace
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
import jax
import jax.numpy as jnp

import fputil
import full_p2_learner as FL
from full_p2_learner import (FullP2Config, build_optimizer, full_p2_update,
                             classify_params, critic_only_mask, is_critic_only_path,
                             _path_str, reconstruct_batch, pack_batch, _ext_obs)

CFG = fputil.CFG


def _leaf_dict(tree):
    """{path_str: np.array} for every leaf."""
    pl, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {_path_str(kp): np.asarray(v) for kp, v in pl}


def _perturb(params, pred, delta=1.0):
    """Add delta to every leaf whose path satisfies pred(path_str)."""
    return jax.tree_util.tree_map_with_path(
        lambda kp, v: (v + delta) if pred(_path_str(kp)) else v, params)


def _window_outputs(a_rec, scan_fn, params, so, cfg):
    recon = reconstruct_batch(a_rec, params, so, cfg)
    obs_ext = _ext_obs(pack_batch(so))
    lg, val, _, _, _ = scan_fn(params, *recon, obs_ext)
    return np.asarray(lg), np.asarray(val)


def _setup(K=2, L_seq=129):
    _, params, a_rec, a_raw, scan_fn = fputil.build_net()
    so, sr, _ = fputil.make_samples(CFG, params, a_rec, K=K, L_seq=L_seq)
    target = jax.tree_util.tree_map(lambda x: x, params)
    return params, a_rec, a_raw, scan_fn, so, sr, target


# ----------------------------- partition -----------------------------

def test_partition():
    _, params, *_ = fputil.build_net()
    c = classify_params(params)
    assert c["n_critic_leaves"] == 6, c["n_critic_leaves"]      # 3 critic modules x (kernel+bias)
    for p in c["critic_only_paths"]:
        assert ("critic_ln1" in p) or ("critic_ln2" in p) or ("critic_out" in p), p
    # encoder + actor head + trunk are policy-affecting
    pol = "\n".join(c["policy_affecting_paths"])
    assert "actor_out/kernel" in pol
    assert "actor_ln1/kernel" in pol
    assert any("encoder" in p for p in c["policy_affecting_paths"]), "encoder must be policy-affecting"
    assert any("transformer" in p for p in c["policy_affecting_paths"]), "trunk must be policy-affecting"
    # no critic leaf leaks into the policy set
    assert not any(is_critic_only_path(p) for p in c["policy_affecting_paths"])
    # counts sum to total
    total = sum(l["size"] for l in c["leaves"])
    assert c["critic_param_count"] + c["policy_param_count"] == total
    assert c["policy_param_count"] > c["critic_param_count"]
    print(f"PASS KL.partition critic_leaves=6 critic_params={c['critic_param_count']} "
          f"policy_params={c['policy_param_count']} total={total}")


def test_critic_invariance():
    params, a_rec, a_raw, scan_fn, so, sr, _ = _setup()
    lg0, val0 = _window_outputs(a_rec, scan_fn, params, so, CFG)
    # perturb the pure critic head -> logits UNCHANGED, value CHANGED
    p_c = _perturb(params, lambda p: "critic_out/kernel" in p, delta=1.0)
    lg_c, val_c = _window_outputs(a_rec, scan_fn, p_c, so, CFG)
    assert np.allclose(lg0, lg_c, atol=1e-6), "critic-only perturbation moved policy logits!"
    assert not np.allclose(val0, val_c, atol=1e-6), "critic perturbation did not move value"
    # perturb an actor-head leaf -> logits CHANGED. We perturb actor_ln1/bias (not the
    # final kernel) because the tiny synthetic obs can leave the last relu near-zero,
    # which masks a constant-per-action final-kernel shift; +1 on actor_ln1/bias keeps
    # every relu unit active and yields a NON-uniform logit change (policy moves).
    p_a = _perturb(params, lambda p: "actor_ln1/bias" in p, delta=1.0)
    lg_a, _ = _window_outputs(a_rec, scan_fn, p_a, so, CFG)
    assert not np.allclose(lg0, lg_a, atol=1e-4), "actor perturbation did not move logits"
    # the change is non-uniform across actions (the policy distribution moves, not just a
    # softmax-invariant constant shift)
    d = np.asarray(lg_a) - np.asarray(lg0)
    assert np.abs(d - d[..., :1]).max() > 1e-4, "actor perturbation only added a constant"
    print("PASS KL.criticinv critic-only leaf leaves logits fixed (value moves); "
          "actor leaf moves logits (non-uniform)")


# ----------------------------- acceptance -----------------------------

def test_accept_scale1():
    params, a_rec, a_raw, scan_fn, so, sr, target = _setup()
    opt = build_optimizer(1e-6, CFG)                 # tiny step -> scale 1.0 passes
    opt_state = opt.init(params)
    p0 = params
    new_p, new_t, new_opt, m = full_p2_update(
        params, target, opt_state, opt, a_rec, a_raw, scan_fn, so, sr, CFG, 0)
    assert m["finite"]
    assert m["policy_committed"] is True
    assert m["kl_rejected_update"] is False
    assert m["kl_gate_pass"] is True
    assert m["chosen_actor_step_scale"] == 1.0, m["chosen_actor_step_scale"]
    assert float(m["policy_kl"]) <= CFG.kl_replay_max
    assert not np.allclose(np.asarray(p0["actor_out"]["kernel"]),
                           np.asarray(new_p["actor_out"]["kernel"])), "actor did not move on accept"
    print(f"PASS KL.accept scale1 policy_kl={float(m['policy_kl']):.6f}<=0.05 "
          f"chosen={m['chosen_actor_step_scale']} committed={m['policy_committed']}")


def test_scale_search_engages():
    params, a_rec, a_raw, scan_fn, so, sr, target = _setup()
    lr = 3e-4
    # 1) measure the scale-1.0 KL by forcing a rejection (threshold ~0, scales=(1.0,))
    cfg_probe = replace(CFG, actor_step_scales=(1.0,), kl_replay_max=1e-12)
    opt = build_optimizer(lr, CFG); opt_state = opt.init(params)
    _, _, _, mp = full_p2_update(
        params, target, opt_state, opt, a_rec, a_raw, scan_fn, so, sr, cfg_probe, 0)
    assert mp["kl_rejected_update"] is True
    K1 = float(mp["policy_kl"])                       # == KL at actor step scale 1.0
    assert K1 > 1e-6, "scale-1.0 update had no policy effect; cannot test search"
    # 2) threshold below K1 (so scale 1.0 fails) but above the smallest scale's KL
    thr = 0.6 * K1
    cfg_search = replace(CFG, actor_step_scales=(1.0, 0.5, 0.25, 0.125), kl_replay_max=thr)
    opt2 = build_optimizer(lr, CFG); opt_state2 = opt2.init(params)
    new_p, _, _, m = full_p2_update(
        params, target, opt_state2, opt2, a_rec, a_raw, scan_fn, so, sr, cfg_search, 0)
    assert m["policy_committed"] is True, m
    assert m["chosen_actor_step_scale"] < 1.0, m["chosen_actor_step_scale"]   # searched past 1.0
    assert m["chosen_actor_step_scale"] in (0.5, 0.25, 0.125), m
    assert float(m["policy_kl"]) <= thr, (float(m["policy_kl"]), thr)
    assert not np.allclose(np.asarray(params["actor_out"]["kernel"]),
                           np.asarray(new_p["actor_out"]["kernel"]))
    print(f"PASS KL.scale scale1_KL={K1:.4f} -> accepted at scale="
          f"{m['chosen_actor_step_scale']} KL={float(m['policy_kl']):.4f}<={thr:.4f}")


# ----------------------------- rejection + rollback -----------------------------

def test_rejection_rollback():
    params, a_rec, a_raw, scan_fn, so, sr, target = _setup()
    lg0, val0 = _window_outputs(a_rec, scan_fn, params, so, CFG)
    cfg_rej = replace(CFG, actor_step_scales=(1.0, 0.5, 0.25, 0.125), kl_replay_max=1e-12)
    opt = build_optimizer(3e-4, CFG)
    opt_state = opt.init(params)
    opt_before = _leaf_dict(opt_state)
    new_p, new_t, new_opt, m = full_p2_update(
        params, target, opt_state, opt, a_rec, a_raw, scan_fn, so, sr, cfg_rej, 0)

    assert m["finite"]
    assert m["kl_rejected_update"] is True
    assert m["policy_committed"] is False
    assert m["kl_gate_pass"] is False
    assert m["chosen_actor_step_scale"] == -1.0
    assert m["n_actor_scale_trials"] == 4
    assert float(m["policy_kl"]) > 1e-12           # a real (rejected) update was measured

    c = classify_params(params)
    new_d = _leaf_dict(new_p)
    old_d = _leaf_dict(params)
    # (a) EVERY policy-affecting param leaf is bit-exact unchanged
    for p in c["policy_affecting_paths"]:
        assert np.array_equal(old_d[p], new_d[p]), f"policy param moved on rejection: {p}"
    # (b) the pure critic head DID commit (critic_out kernel changed, finite)
    assert not np.array_equal(old_d["critic_out/kernel"], new_d["critic_out/kernel"]), \
        "critic head did not commit independently"
    assert np.all(np.isfinite(new_d["critic_out/kernel"]))

    # (c) EMA target: policy leaves frozen (== old target == old params at init), critic moved
    tgt_d = _leaf_dict(new_t)
    tgt0_d = _leaf_dict(target)
    for p in c["policy_affecting_paths"]:
        assert np.array_equal(tgt0_d[p], tgt_d[p]), f"policy EMA target moved on rejection: {p}"
    assert not np.array_equal(tgt0_d["critic_out/kernel"], tgt_d["critic_out/kernel"]), \
        "critic EMA target did not move"

    # (d) optimizer state: policy moments rolled back (== old), critic moments advanced
    opt_new_d = _leaf_dict(new_opt)
    n_policy_equal = n_policy_total = 0
    n_critic_diff = 0
    for path, old_v in opt_before.items():
        nv = opt_new_d[path]
        if is_critic_only_path(path):
            if not np.array_equal(old_v, nv):
                n_critic_diff += 1
        else:
            n_policy_total += 1
            assert np.array_equal(old_v, nv), f"policy opt moment NOT rolled back: {path}"
            n_policy_equal += 1
    assert n_policy_equal == n_policy_total and n_policy_total > 0
    assert n_critic_diff > 0, "no critic optimizer moment advanced"

    # (e) policy logits bit-exact unchanged; value moved (critic committed)
    lg1, val1 = _window_outputs(a_rec, scan_fn, new_p, so, CFG)
    assert np.array_equal(lg0, lg1), "policy logits changed despite full policy rollback"
    assert not np.allclose(val0, val1, atol=1e-6), "value did not move (critic head failed)"
    print(f"PASS KL.rollback policy params+opt+EMA frozen ({n_policy_total} opt leaves rolled "
          f"back), critic committed ({n_critic_diff} critic opt leaves advanced), "
          f"policy logits bit-exact, value moved")


if __name__ == "__main__":
    test_partition()
    test_critic_invariance()
    test_accept_scale1()
    test_scale_search_engages()
    test_rejection_rollback()
    print("ALL_KL_TRANSACTIONAL_GATE_TESTS_PASS")
