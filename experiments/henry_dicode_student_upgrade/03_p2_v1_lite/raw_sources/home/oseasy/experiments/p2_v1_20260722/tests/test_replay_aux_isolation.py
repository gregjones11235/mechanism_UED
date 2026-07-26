#!/usr/bin/env python3
"""方案2 (2026-07-23) hard tests: replay value-auxiliary parameter isolation.

User ruling: Level-3 entry condition (3) "Actor NOT modified by replay aux path"
is NOT satisfied under the broad reading, because the replay value auxiliary
updated the SHARED actor/critic trunk (68/68 leaves) — part of the policy
forward path — even though it left the actor-specific heads (0/6) untouched.
The fix isolates the replay value auxiliary to the critic-specific value head:

  1. replay value auxiliary updates ONLY the critic value head (shared trunk is
     stop-gradient / param-masked so it does NOT update);
  2. replay updates only critic-specific value head + auxiliary-specific params
     (the current Henry network has no separate goal/success head — goal
     conditioning flows through the shared encoder — so the auxiliary-specific
     params are exactly the critic value head);
  3. on-policy PPO still updates the shared trunk + actor head + critic head;
  4. the original hard gates are NOT lowered or reinterpreted.

These tests assert (all HARD, on the REAL ActorCriticTransformer, CPU):

  A. replay aux-only param change:
       actor head           -> 0 leaves changed (bit-exact)
       shared trunk (actor) -> 0 leaves changed (bit-exact)
       critic-specific head -> non-zero change (all critic leaves move)
       auxiliary-specific head -> non-zero per config (none exist beyond the
                                  critic head in this architecture)
  B. policy-function invariance (fixed obs/memory/mask):
       policy logits       -> bit-exact before/after the aux update
       action probabilities-> bit-exact
       entropy             -> bit-exact
  C. critic effective update:
       value prediction    -> changes
       replay value loss   -> non-zero finite gradient on the critic head
       critic-specific params -> change
  E. (bonus) the main PPO optimizer state is untouched by the aux update.
  F. (bonus) the dedicated aux critic-head optimizer state persists and
       round-trips through pickle bit-exactly (exact resume across processes).

Test D (native-PPO equivalence with aux off) is re-run via the existing
tests/test_p2_v1.py::test_original_ppo_equivalence (PPO code is untouched by
this fix); it is part of the full CPU suite run after this test.

Exit 0 only if every hard check passes; writes evidence JSON.
"""
import os
import sys
import json
import pickle
import io
from datetime import datetime, timezone

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import stage4_continue_launcher as L
from long_context_learner import LongContextLearner
from trajectory_replay import ReplaySample

OBS_DIM = 128          # isolation is a param-routing property, not obs-dim-related
EMB = 67               # DEFEAT_KOBOLD achievement embedding size (real value)
N_ACH = 67
ACTION_DIM = 43


def classify(path_str):
    p = path_str.lower()
    if "actor" in p:
        return "actor_head"
    if "critic" in p:
        return "critic_head"
    return "shared_trunk"   # encoder + transformer layers


def build():
    cfg = L.Cfg()
    network = L.ActorCriticTransformer(
        action_dim=ACTION_DIM, activation=cfg.activation,
        hidden_layers=cfg.hidden_layers, encoder_size=cfg.embed_size,
        num_heads=cfg.num_heads, qkv_features=cfg.qkv_features,
        num_layers=cfg.num_layers, gating=cfg.gating,
        gating_bias=cfg.gating_bias)
    # Init via model_forward_train (batch-safe; model_forward_eval at batch=1
    # trips the Henry squeeze — params are shared across forward methods).
    WM, LAY, E, W = cfg.window_mem, cfg.num_layers, cfg.embed_size, cfg.window_grad
    rng = jax.random.PRNGKey(0)
    params = network.init(
        rng, jnp.zeros((1, WM, LAY, E)), jnp.zeros((1, W, OBS_DIM)),
        jnp.zeros((1, cfg.num_heads, W, WM + W), dtype=jnp.bool_),
        method=network.model_forward_train)
    # Production optimizer (anneal_lr=False): clip + constant-lr Adam.
    tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm),
                     optax.adam(cfg.lr, eps=1e-5))
    ts = TrainState.create(apply_fn=network.apply, params=params, tx=tx)
    learner = LongContextLearner(network, cfg, jax.random.PRNGKey(0))
    return cfg, network, ts, learner


def make_nonterminal_sample(cfg, length=200):
    WM, LAY, E = cfg.window_mem, cfg.num_layers, cfg.embed_size
    r = np.random.RandomState(0)
    obs = (r.randn(length, OBS_DIM) * 0.1).astype(np.float32)
    tgt = np.zeros(N_ACH, np.float32); tgt[0] = 1.0
    obs[:, OBS_DIM - EMB:] = tgt[:EMB]
    ach = np.zeros((length, N_ACH), np.float32)
    return ReplaySample(
        observations=obs,
        actions=r.randint(0, ACTION_DIM, length).astype(np.int32),
        rewards=(r.randn(length) * 0.1).astype(np.float32),
        dones=np.zeros(length, bool),
        values=(r.randn(length) * 0.1).astype(np.float32),
        log_probs=r.randn(length).astype(np.float32),
        initial_memory=(r.randn(WM, LAY, E) * 0.01).astype(np.float32),
        achievements=ach, target_achievements=tgt, source_trajectory_id=0,
        start_step=0, length=length,
        memory_sequence=(r.randn(length, WM, LAY, E) * 0.01).astype(np.float32),
        next_observations=np.concatenate([obs[1:], obs[-1:]], 0).copy(),
        next_value=0.0, episode_done=False)   # nonterminal -> value bootstrap


def leaf_groups(params_a, params_b):
    """Return {group: [changed, total]} comparing two param trees leaf-wise."""
    flat_a = jax.tree_util.tree_leaves_with_path(params_a)
    flat_b = dict((p, v) for p, v in
                  jax.tree_util.tree_leaves_with_path(params_b))
    groups = {"actor_head": [0, 0], "critic_head": [0, 0],
              "shared_trunk": [0, 0]}
    for path, av in flat_a:
        g = classify(jax.tree_util.keystr(path))
        groups[g][1] += 1
        if bool(jnp.any(jnp.asarray(av) != jnp.asarray(flat_b[path]))):
            groups[g][0] += 1
    return groups


def main():
    print("=" * 68)
    print("方案2 replay value-auxiliary PARAMETER ISOLATION tests (real net, CPU)")
    print(f"  JAX devices: {jax.devices()}")
    print("=" * 68)

    cfg, network, ts, learner = build()
    sample = make_nonterminal_sample(cfg, length=200)
    WM, LAY, E = cfg.window_mem, cfg.num_layers, cfg.embed_size

    results = {}

    # ── A. replay aux-only param change ─────────────────────────────
    ts2, metrics = learner.update(ts, sample, current_update_count=2,
                                  replay_actor_update=False)
    groups = leaf_groups(ts.params, ts2.params)
    print(f"  [A] actor_enabled={metrics['actor_enabled']} "
          f"actor_loss={metrics['actor_loss']} "
          f"value_loss={metrics['value_loss']:.6f} "
          f"grad_norm={metrics['grad_norm']:.6f}")
    for g in ("actor_head", "critic_head", "shared_trunk"):
        c, t = groups[g]
        print(f"      {g:14s}: {c}/{t} changed")
    results["A_actor_head_0_changed"] = (groups["actor_head"][0] == 0
                                         and groups["actor_head"][1] == 6)
    results["A_shared_trunk_0_changed"] = (groups["shared_trunk"][0] == 0
                                           and groups["shared_trunk"][1] == 68)
    results["A_critic_head_nonzero"] = (groups["critic_head"][0] == groups["critic_head"][1]
                                        and groups["critic_head"][1] == 6
                                        and groups["critic_head"][0] > 0)
    # No auxiliary-specific head exists beyond the critic head in this
    # architecture; "per config" the aux-specific param set == critic head, which
    # is non-zero above.  Record the updatable key set for the evidence.
    results["A_aux_specific_substrings"] = list(learner._replay_aux_updatable_substrings)

    # ── B. policy-function invariance (fixed obs/memory/mask) ───────
    # Use model_forward_eval (the policy's single-step path) at batch=2 with two
    # IDENTICAL rows and read [0] (proven batch-independent by
    # test_replay_aux_bootstrap.py; batch=1 trips the Henry squeeze).  The actor
    # path = shared encoder + transformer trunk + actor head, all of which must
    # be bit-exact unchanged by the aux update.
    rb = np.random.RandomState(1)
    mem_b = jnp.asarray(rb.randn(1, WM, LAY, E).astype(np.float32) * 0.01)
    obs_b = jnp.asarray(rb.randn(1, OBS_DIM).astype(np.float32) * 0.1)
    mask_b = jnp.ones((1, cfg.num_heads, 1, WM + 1), dtype=jnp.bool_)
    mem2 = jnp.tile(mem_b, (2, 1, 1, 1))
    obs2 = jnp.tile(obs_b, (2, 1))

    def eval_policy(params):
        pi, _v, _m = network.apply(params, mem2, obs2, mask_b,
                                   method=network.model_forward_eval)
        logits = pi.logits
        probs = jax.nn.softmax(logits, axis=-1)
        ent = pi.entropy()
        return logits[0], probs[0], ent[0]

    lg_b, pr_b, en_b = eval_policy(ts.params)    # BEFORE aux update
    lg_a, pr_a, en_a = eval_policy(ts2.params)   # AFTER aux update
    results["B_logits_bit_exact"] = bool(jnp.array_equal(lg_b, lg_a))
    results["B_probs_bit_exact"] = bool(jnp.array_equal(pr_b, pr_a))
    results["B_entropy_bit_exact"] = bool(jnp.array_equal(en_b, en_a))
    print(f"  [B] logits_equal={results['B_logits_bit_exact']} "
          f"probs_equal={results['B_probs_bit_exact']} "
          f"entropy_equal={results['B_entropy_bit_exact']} "
          f"(entropy_before={float(en_b):.6f})")

    # ── C. critic effective update ──────────────────────────────────
    # C1: value prediction changes on a fixed windowed input.
    W = cfg.window_grad
    rc = np.random.RandomState(2)
    c_mem = jnp.asarray(rc.randn(1, WM, LAY, E).astype(np.float32) * 0.01)
    c_obs = jnp.asarray(rc.randn(1, W, OBS_DIM).astype(np.float32) * 0.1)
    c_mask = jnp.ones((1, cfg.num_heads, W, WM + W), dtype=jnp.bool_)
    _, val_before = network.apply(ts.params, c_mem, c_obs, c_mask,
                                  method=network.model_forward_train)
    _, val_after = network.apply(ts2.params, c_mem, c_obs, c_mask,
                                 method=network.model_forward_train)
    val_changed = bool(jnp.any(val_before != val_after))
    val_finite = bool(jnp.all(jnp.isfinite(val_after)))
    results["C_value_prediction_changes"] = val_changed
    results["C_value_finite"] = val_finite
    # C2: replay value loss produces a NON-ZERO FINITE gradient on the critic head.
    grad_fn = jax.grad(lambda p: learner._replay_aux_loss(p, sample, 2, False)[0])
    grads = grad_fn(ts.params)
    critic_grads = jax.tree_util.tree_map(
        lambda m, g: g if m else jnp.zeros_like(g), learner._aux_mask, grads)
    cg_norm = float(optax.global_norm(critic_grads))
    results["C_critic_grad_finite"] = bool(np.isfinite(cg_norm))
    results["C_critic_grad_nonzero"] = bool(cg_norm > 1e-12)
    results["C_critic_grad_norm"] = cg_norm
    print(f"  [C] value_changed={val_changed} value_finite={val_finite} "
          f"critic_grad_norm={cg_norm:.6f} (finite&>1e-12="
          f"{results['C_critic_grad_finite'] and results['C_critic_grad_nonzero']})")

    # ── E. main PPO optimizer state untouched by the aux update ─────
    opt_leaves_before = jax.tree_util.tree_leaves(ts.opt_state)
    opt_leaves_after = jax.tree_util.tree_leaves(ts2.opt_state)
    opt_untouched = (len(opt_leaves_before) == len(opt_leaves_after)
                     and all(bool(jnp.array_equal(a, b))
                             for a, b in zip(opt_leaves_before, opt_leaves_after)))
    results["E_main_opt_state_untouched"] = bool(opt_untouched)
    print(f"  [E] main_ppo_opt_state_untouched={opt_untouched} "
          f"({len(opt_leaves_before)} leaves)")

    # ── F. aux critic-head optimizer state persists + pickles ───────
    aux_state = learner.get_aux_optimizer_state()
    f_not_none = aux_state is not None
    f_roundtrip = False
    if f_not_none:
        buf = io.BytesIO()
        pickle.dump(aux_state, buf)
        buf.seek(0)
        rt = pickle.load(buf)
        la = jax.tree_util.tree_leaves(aux_state)
        lb = jax.tree_util.tree_leaves(rt)
        f_roundtrip = (len(la) == len(lb)
                       and all(bool(jnp.array_equal(x, y)) for x, y in zip(la, lb)))
    results["F_aux_state_present"] = bool(f_not_none)
    results["F_aux_state_pickle_roundtrip"] = bool(f_roundtrip)
    print(f"  [F] aux_state_present={f_not_none} pickle_roundtrip={f_roundtrip}")

    hard = ["A_actor_head_0_changed", "A_shared_trunk_0_changed",
            "A_critic_head_nonzero", "B_logits_bit_exact", "B_probs_bit_exact",
            "B_entropy_bit_exact", "C_value_prediction_changes", "C_value_finite",
            "C_critic_grad_finite", "C_critic_grad_nonzero",
            "E_main_opt_state_untouched", "F_aux_state_present",
            "F_aux_state_pickle_roundtrip"]
    all_pass = all(results[k] for k in hard)

    print("\n" + "=" * 68)
    for k in sorted(results):
        v = results[k]
        if isinstance(v, bool):
            print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("=" * 68)
    print(f"方案2 ISOLATION: {'ALL PASS' if all_pass else 'FAIL'}")

    ev = os.path.join(os.path.dirname(__file__), "..", "evidence")
    os.makedirs(ev, exist_ok=True)
    with open(os.path.join(ev, "replay_aux_isolation_test_report.json"), "w") as f:
        json.dump({
            "directive": "P2-v1 方案2 replay-aux parameter isolation (#57/#61)",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "jax_devices": str(jax.devices()),
            "obs_dim": OBS_DIM, "emb": EMB,
            "cfg": {"window_grad": cfg.window_grad, "window_mem": cfg.window_mem,
                    "num_heads": cfg.num_heads, "num_layers": cfg.num_layers,
                    "embed_size": cfg.embed_size, "lr": cfg.lr,
                    "max_grad_norm": cfg.max_grad_norm},
            "param_groups": {g: {"changed": groups[g][0], "total": groups[g][1]}
                             for g in groups},
            "replay_aux_updatable_substrings": list(learner._replay_aux_updatable_substrings),
            "results": results, "all_pass": bool(all_pass),
        }, f, indent=2, sort_keys=True)
        f.write("\n")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
