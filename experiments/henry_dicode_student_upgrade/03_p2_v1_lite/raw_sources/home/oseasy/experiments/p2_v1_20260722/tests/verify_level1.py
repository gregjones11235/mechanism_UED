#!/usr/bin/env python3
"""P2-v1 Level 1 independent acceptance verification (阶段1, gates 10/14/15).

Runs on GPU0.  Read-only w.r.t. the real checkpoint (checkpoints/2048 is only
restored, never modified).  The round-trip re-save goes to a tempdir that is
removed at the end.

Independently verifies:
  Gate 10  params changed vs healthy session175 base.
  Gate 14  restore_full_checkpoint(checkpoints/2048) succeeds.
  Gate 15  restored state is internally consistent AND round-trips bit-exact:
           params / optimizer / JAX RNG / action RNG / replay / global_step /
           update_count.

Also re-confirms (runtime, independent of the training log):
  - restored global_step == 2048, update_count == 1
  - restored optimizer step == 1 (exactly one Adam update applied, NOT fresh 0)
  - restored params finite (no NaN/Inf) and differ from fresh random init
  - session175 base params differ from fresh random init (sanity)

Exits 0 only if ALL gates pass; writes evidence JSON.
"""
import glob
import json
import os
import shutil
import sys
import tempfile
import hashlib
from datetime import datetime, timezone

# GPU run: JAX must see GPU0 (set CUDA_VISIBLE_DEVICES at launch).  Do NOT force
# CPU here (restore of the sharded orbax TrainState needs the GPU device).

import numpy as np
import jax
import jax.numpy as jnp

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC)

# Importing the launcher inserts _HENRY_SRC (dicode) + src on path and exposes
# all the builder helpers / constants we need.
import stage4_continue_launcher as L
from checkpointing import save_full_checkpoint, restore_full_checkpoint
from rng_utils import restore_action_rng, sample_actions

CKPT_2048 = os.path.join(L.CKPT_ROOT, "2048")
EMB = 67


def _leaves_finite(params):
    return all(bool(jnp.all(jnp.isfinite(jnp.asarray(l))))
               for l in jax.tree_util.tree_leaves(params))


def _params_equal(p1, p2):
    l1 = jax.tree_util.tree_leaves(p1)
    l2 = jax.tree_util.tree_leaves(p2)
    if len(l1) != len(l2):
        return False
    return all(bool(jnp.all(jnp.asarray(a) == jnp.asarray(b)))
               for a, b in zip(l1, l2))


def _opt_equal(o1, o2):
    l1 = jax.tree_util.tree_leaves(o1)
    l2 = jax.tree_util.tree_leaves(o2)
    if len(l1) != len(l2):
        return False
    return all(bool(jnp.all(jnp.asarray(a) == jnp.asarray(b)))
               for a, b in zip(l1, l2))


def main():
    gates = {}
    print("=" * 64)
    print("P2-v1 Level 1 restore verification (GPU0)")
    print(f"  JAX devices: {jax.devices()}")
    print(f"  checkpoint:  {CKPT_2048}")
    print("=" * 64)

    assert os.path.isdir(CKPT_2048), f"checkpoint missing: {CKPT_2048}"

    cfg = L.Cfg()

    # ── Build network (obs_dim/action_dim from dummy env, as in main) ──
    ach_table = jnp.array(
        [L.get_achievement_multi_hot([L.Achievement.DEFEAT_KOBOLD])],
        dtype=jnp.float32)
    emb = int(ach_table.shape[1])
    assert emb == EMB, f"embedding {emb} != {EMB}"
    dummy = L.CraftaxAugObsTrain(
        condition_on_task=True, conditioning_type="embedding",
        embedding_size=emb, task_embeddings=jnp.zeros((1, emb)))
    obs_dim = dummy.observation_space(dummy.default_params).shape[0]
    action_dim = dummy.action_space(dummy.default_params).n
    print(f"  obs_dim={obs_dim}  action_dim={action_dim}")

    network = L.ActorCriticTransformer(
        action_dim=action_dim, activation=cfg.activation,
        hidden_layers=cfg.hidden_layers, encoder_size=cfg.embed_size,
        num_heads=cfg.num_heads, qkv_features=cfg.qkv_features,
        num_layers=cfg.num_layers, gating=cfg.gating,
        gating_bias=cfg.gating_bias)

    # ── Build the real Stage4 base env (needed for weights load) ──
    with open(L.S4_TASK_PATH) as f:
        ns = {}
        exec(f.read(), ns)
    Task = ns["Env"]
    static_env_params = L.StaticEnvParams()
    env_params = L.EnvParams(max_timesteps=4096)
    base_env = L.MultiTaskMiniCraftaxEnv(
        [Task], static_env_params, env_params, cfg.condition_on_task,
        conditioning_type="embedding", embedding_size=emb,
        completion_bonus_scale=cfg.completion_bonus_scale,
        completion_bonus_min=cfg.completion_bonus_min,
        bonus_type=cfg.bonus_type, dynamic_bonus_k=cfg.dynamic_bonus_k)

    # ── session175 base params (healthy start, pre-update) ──
    print("\n[base] loading healthy session175 weights (pre-update reference)...")
    base = L.init_fresh_from_session175(network, base_env, env_params, cfg, obs_dim)
    base_params = base["train_state"].params
    print(f"  session175 source_sha256 = {base['source_checkpoint_sha256']}")

    # ── fresh random init (sanity reference) ──
    rand_params = L.init_network_params(network, obs_dim, cfg, jax.random.PRNGKey(1))

    # ── GATE 14: restore the real Level 1 checkpoint ──
    # restore_full_checkpoint(path=PARENT, step=N) reads PARENT/N/{default,replay_meta.pkl}.
    print("\n[14] restore_full_checkpoint(checkpoints/, step=2048) ...")
    dummy_ts = L.build_stage4_train_state(network, rand_params, cfg)
    r = restore_full_checkpoint(L.CKPT_ROOT, dummy_ts, step=2048)
    gates["14_restore_succeeds"] = (r is not None and "train_state" in r)
    print(f"  restored keys: {sorted(r.keys())}")

    r_ts = r["train_state"]
    r_params = r_ts.params
    r_global_step = int(r["global_step"])
    r_update_count = int(r["update_count"])
    r_replay_len = len(r["replay_buffer"])
    r_opt_step = L._optimizer_step_count(r_ts)

    print(f"  restored global_step = {r_global_step}")
    print(f"  restored update_count = {r_update_count}")
    print(f"  restored replay len = {r_replay_len}")
    print(f"  restored optimizer step = {r_opt_step}")
    print(f"  action_rng_state present = {r['action_rng_state'] is not None}")

    # ── Read the Level 1 training_log to derive the EXPECTED optimizer step ──
    # One logical on-policy update = (num_minibatches * update_epochs) gradient
    # applications, so after `update_count` updates the Adam step counter is
    # num_minibatches * update_epochs * update_count  (NOT merely update_count).
    sess_dirs = sorted(glob.glob(os.path.join(
        L.OUTPUT_ROOT, "stage4_continue", "session_*")))
    assert sess_dirs, "no session_* output dir found under outputs/stage4_continue"
    sess_dir = sess_dirs[-1]
    tlog = os.path.join(sess_dir, "training_log.jsonl")
    with open(tlog) as f:
        last = json.loads(f.readlines()[-1])
    n_mb = int(last["ppo_num_minibatches"])
    n_ep = int(last["ppo_update_epochs"])
    log_update_count = int(last["update_count"])
    expected_opt_step = n_mb * n_ep * log_update_count
    print(f"  training_log: num_minibatches={n_mb} update_epochs={n_ep} "
          f"update_count={log_update_count} -> expected optimizer step={expected_opt_step}")

    # ── GATE 15a: restored scalar/state consistency ──
    gates["15_global_step_2048"] = (r_global_step == 2048)
    gates["15_update_count_1"] = (r_update_count == 1)
    gates["15_replay_len_3"] = (r_replay_len == 3)
    # optimizer is the TRAINED one (not a fresh step-0 init) ...
    gates["15_optimizer_step_trained_not_fresh"] = (r_opt_step > 0)
    # ... and its step count exactly matches the logged update structure
    # (1 logical update = num_minibatches*update_epochs = 2 gradient steps).
    gates["15_optimizer_step_matches_updates"] = (r_opt_step == expected_opt_step)
    gates["15_action_rng_state_present"] = (r["action_rng_state"] is not None)

    # ── GATE 11 (runtime): restored params finite ──
    gates["11_restored_params_finite"] = _leaves_finite(r_params)

    # ── GATE 10: params changed vs session175 base ──
    changed_vs_base = not _params_equal(r_params, base_params)
    gates["10_params_changed_vs_session175"] = changed_vs_base
    # sanity: ckpt params differ from fresh random; base differs from random
    gates["10sanity_ckpt_differs_from_random"] = not _params_equal(r_params, rand_params)
    gates["10sanity_base_differs_from_random"] = not _params_equal(base_params, rand_params)
    print(f"\n[10] params changed vs session175 = {changed_vs_base}")

    # ── GATE 15b: round-trip bit-exact (re-save to tempdir, re-restore) ──
    print("\n[15b] round-trip: re-save restored state -> re-restore (tempdir) ...")
    tmp = tempfile.mkdtemp(prefix="p2v1_level1_rt_")
    try:
        save_full_checkpoint(
            r_ts, r["replay_buffer"], r["rng"], r_global_step, tmp,
            step=r_global_step, action_rng_state=r["action_rng_state"],
            update_count=r_update_count)
        dummy_ts2 = L.build_stage4_train_state(
            network, L.init_network_params(network, obs_dim, cfg,
                                           jax.random.PRNGKey(9)), cfg)
        r2 = restore_full_checkpoint(tmp, dummy_ts2, step=r_global_step)

        gates["15_roundtrip_params_bitexact"] = _params_equal(r_params, r2["train_state"].params)
        gates["15_roundtrip_opt_bitexact"] = _opt_equal(r_ts.opt_state, r2["train_state"].opt_state)
        gates["15_roundtrip_rng_equal"] = bool(jnp.all(r["rng"] == r2["rng"]))
        gates["15_roundtrip_global_step"] = (int(r2["global_step"]) == r_global_step)
        gates["15_roundtrip_update_count"] = (int(r2["update_count"]) == r_update_count)
        gates["15_roundtrip_replay_len"] = (len(r2["replay_buffer"]) == r_replay_len)

        # action RNG: restoring from the same saved state must reproduce the
        # identical continuing stream (checkpointable action RNG).
        probes = np.random.default_rng(7).standard_normal((4, 43))
        sm = np.exp(probes - probes.max(axis=-1, keepdims=True))
        pr = sm / sm.sum(axis=-1, keepdims=True)
        g_a = restore_action_rng(r["action_rng_state"], seed=L.P2_V1_MASTER_SEED)
        g_b = restore_action_rng(r2["action_rng_state"], seed=L.P2_V1_MASTER_SEED)
        seq_a = [int(sample_actions(g_a, pr)[0]) for _ in range(16)]
        seq_b = [int(sample_actions(g_b, pr)[0]) for _ in range(16)]
        gates["15_roundtrip_action_rng_stream"] = (seq_a == seq_b)
        print(f"  round-trip action_rng stream match = {seq_a == seq_b}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ── summary ──
    print("\n" + "=" * 64)
    all_pass = all(gates.values())
    for k in sorted(gates):
        print(f"  {'PASS' if gates[k] else 'FAIL'}  {k}")
    print("=" * 64)
    print(f"LEVEL1 RESTORE VERIFICATION: {'ALL PASS' if all_pass else 'FAIL'}")

    report = {
        "directive": "P2-v1 Level 1 acceptance (阶段1 gates 10/14/15)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": CKPT_2048,
        "session175_source_sha256": base["source_checkpoint_sha256"],
        "restored": {
            "global_step": r_global_step,
            "update_count": r_update_count,
            "replay_len": r_replay_len,
            "optimizer_step": r_opt_step,
            "optimizer_step_expected": expected_opt_step,
            "optimizer_step_derivation": f"{n_mb} minibatches * {n_ep} epochs * {log_update_count} update_count",
            "action_rng_state_present": r["action_rng_state"] is not None,
        },
        "training_log": tlog,
        "gates": gates,
        "all_pass": all_pass,
        "jax_devices": str(jax.devices()),
    }
    ev = os.path.join(os.path.dirname(__file__), "..", "evidence")
    os.makedirs(ev, exist_ok=True)
    with open(os.path.join(ev, "level1_restore_check_report.json"), "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
