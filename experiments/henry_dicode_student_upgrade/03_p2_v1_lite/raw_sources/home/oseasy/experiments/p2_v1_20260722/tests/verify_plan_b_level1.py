#!/usr/bin/env python3
"""P2-v1 方案B Level 1 full-checkpoint restore verification (GPU0).

Extends verify_level1.py (gates 10/14/15) to the 方案B contract.  Read-only
w.r.t. the real checkpoint (checkpoints/2048 only restored, never modified); the
round-trip re-save goes to a tempdir removed at the end.

Uses the PRODUCTION restore primitive ``restore_p2_v1_checkpoint`` (which
fail-closes if pending_state / collector_state are absent) and verifies the full
方案B resume state restores bit-exact:

  params / optimizer / JAX RNG / action RNG / env RNG / replay RNG /
  replay contents / pending buffers / collector state (obsv/env_state/
  memories/mem_mask/mem_idx) / episode ids / initial memories /
  global_step / update_count / policy_version.

Plus the runtime invariants the user's hard gates require:
  gate 11  completed-replay + pending transitions CONSERVED (== 2048 collected)
  gate 12  no cross-env trajectory concatenation (each completed traj has exactly
           one terminal done at its last step; pending slots hold NO done)
  gate 13  rollout boundary NOT faked as done (pending slots that crossed the
           128-step boundary carry transitions with dones all False)

Exits 0 only if ALL gates pass; writes evidence JSON.
"""
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np
import jax
import jax.numpy as jnp

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC)

import stage4_continue_launcher as L
from checkpointing import save_full_checkpoint
from pending_episodes import PendingEpisodeBuffers
from rng_utils import restore_action_rng, sample_actions

CKPT_2048 = os.path.join(L.CKPT_ROOT, "2048")
EMB = 67
NUM_ENVS = L.NUM_ENVS            # 16
ROLLOUT_STEPS = L.ROLLOUT_STEPS  # 128
COLLECTED = NUM_ENVS * ROLLOUT_STEPS * 1  # 1 update this Level 1 run = 2048


def _leaves_finite(pytree):
    return all(bool(jnp.all(jnp.isfinite(jnp.asarray(l))))
               for l in jax.tree_util.tree_leaves(pytree))


def _params_equal(p1, p2):
    l1 = jax.tree_util.tree_leaves(p1)
    l2 = jax.tree_util.tree_leaves(p2)
    return (len(l1) == len(l2) and
            all(bool(jnp.all(jnp.asarray(a) == jnp.asarray(b)))
                for a, b in zip(l1, l2)))


def _bytes_equal(a, b):
    return np.asarray(a).tobytes() == np.asarray(b).tobytes()


def main():
    gates = {}
    print("=" * 64)
    print("P2-v1 方案B Level 1 full-checkpoint restore verification (GPU0)")
    print(f"  JAX devices: {jax.devices()}")
    print(f"  checkpoint:  {CKPT_2048}")
    print(f"  collected transitions this run = {COLLECTED}")
    print("=" * 64)

    assert os.path.isdir(CKPT_2048), f"checkpoint missing: {CKPT_2048}"
    cfg = L.Cfg()

    # ── Build network + env (as in main) ─────────────────────────────
    ach_table = jnp.array(
        [L.get_achievement_multi_hot([L.Achievement.DEFEAT_KOBOLD])],
        dtype=jnp.float32)
    emb = int(ach_table.shape[1])
    assert emb == EMB
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

    with open(L.S4_TASK_PATH) as f:
        ns = {}
        exec(f.read(), ns)
    Task = ns["Env"]
    base_env = L.MultiTaskMiniCraftaxEnv(
        [Task], L.StaticEnvParams(), L.EnvParams(max_timesteps=4096),
        cfg.condition_on_task, conditioning_type="embedding",
        embedding_size=emb, completion_bonus_scale=cfg.completion_bonus_scale,
        completion_bonus_min=cfg.completion_bonus_min,
        bonus_type=cfg.bonus_type, dynamic_bonus_k=cfg.dynamic_bonus_k)

    # ── session175 base params (pre-update reference, gate 10) ──────
    print("\n[base] loading healthy session175 weights (pre-update reference)...")
    base = L.init_fresh_from_session175(network, base_env,
                                        L.EnvParams(max_timesteps=4096), cfg,
                                        obs_dim)
    base_params = base["train_state"].params
    source_sha = base["source_checkpoint_sha256"]
    print(f"  session175 source_sha256 = {source_sha}")

    # ── GATE 14: restore via the PRODUCTION 方案B primitive ─────────
    print("\n[14] restore_p2_v1_checkpoint(checkpoints/, step=2048) ...")
    r = L.restore_p2_v1_checkpoint(L.CKPT_ROOT, 2048, network, cfg, obs_dim)
    gates["14_restore_succeeds"] = isinstance(r, dict) and "train_state" in r

    r_ts = r["train_state"]
    r_params = r_ts.params
    gs = int(r["global_step"])
    uc = int(r["update_count"])
    replay = r["replay_buffer"]
    r_replay_len = len(replay)
    opt_step = L._optimizer_step_count(r_ts)
    pending_state = r["pending_state"]
    collector_state = r["collector_state"]

    print(f"  global_step={gs} update_count={uc} replay_len={r_replay_len} "
          f"opt_step={opt_step}")
    print(f"  pending_state present = {pending_state is not None}")
    print(f"  collector_state present = {collector_state is not None}")

    # ── standard gates ───────────────────────────────────────────────
    gates["g2_global_step_2048"] = (gs == 2048)
    gates["g3_update_count_1"] = (uc == 1)
    gates["g11_restored_params_finite"] = _leaves_finite(r_params)
    gates["g9_params_changed_vs_session175"] = not _params_equal(r_params,
                                                                 base_params)
    gates["opt_step_matches_2_minibatches"] = (opt_step == 2)
    gates["action_rng_state_present"] = (r["action_rng_state"] is not None)

    # PPO metrics finite (gate 4-8,10): read from the training_log
    import glob
    sess_dirs = sorted(glob.glob(os.path.join(
        L.OUTPUT_ROOT, "stage4_continue", "session_*")))
    sess_dir = sess_dirs[-1]
    with open(os.path.join(sess_dir, "training_log.jsonl")) as f:
        last = json.loads(f.readlines()[-1])
    ppo_vals = [last["ppo_total_loss"], last["ppo_policy_loss"],
                last["ppo_value_loss"], last["ppo_entropy"],
                last["ppo_grad_norm"]]
    gates["g4_8_ppo_metrics_finite"] = all(
        np.isfinite(v) for v in ppo_vals)
    gates["g8_grad_norm_nonzero"] = (last["ppo_grad_norm"] > 0.0)
    gates["g10_no_nan_inf"] = all(np.isfinite(v) for v in ppo_vals)

    # ── 方案B presence gates ─────────────────────────────────────────
    gates["B_pending_state_present"] = (pending_state is not None)
    gates["B_collector_state_present"] = (collector_state is not None)
    if pending_state is None or collector_state is None:
        print("FATAL: 方案B resume state missing — cannot continue.")
        _finish(gates, False, r, source_sha, sess_dir, None, None)
        return

    pending = PendingEpisodeBuffers.from_state_dict(pending_state)
    completed = sum(int(t.length) for t in replay._buffer)
    pending_tot = pending.total_pending_transitions()
    print(f"  completed(replay)={completed}  pending={pending_tot}  "
          f"sum={completed + pending_tot}  (collected={COLLECTED})")

    # ── GATE 11: conservation ────────────────────────────────────────
    gates["g11_conservation_completed_plus_pending"] = (
        completed + pending_tot == COLLECTED)

    # ── GATE 12: no cross-env concatenation ──────────────────────────
    # each completed trajectory: exactly one done, at the final step only.
    done_ok = True
    for t in replay._buffer:
        d = np.asarray(t.dones).astype(bool)
        if not (d.sum() == 1 and bool(d[-1])):
            done_ok = False
    # each pending slot: NO done (in-progress episode never terminated).
    pending_no_done = all(
        not any(bool(x) for x in s["don"]) for s in pending.slots)
    gates["g12_no_cross_env_concat"] = done_ok and pending_no_done

    # ── GATE 13: rollout boundary NOT faked as done ──────────────────
    # A slot that crossed the 128-step boundary still holds its transitions
    # with dones all False (the boundary was not written as a terminal).
    max_pending_len = max(pending.slot_lengths())
    boundary_not_faked = pending_no_done and (max_pending_len > 0)
    gates["g13_boundary_not_faked_done"] = boundary_not_faked
    print(f"  max pending slot length = {max_pending_len} (no done -> boundary "
          f"not faked)")

    # ── episode ids / policy_version / initial memories ──────────────
    ep_ids = list(pending.episode_id)
    gates["B_episode_ids_distinct"] = (len(set(ep_ids)) == NUM_ENVS)
    # next_episode_id == NUM_ENVS (initial 0..15) + #completions (reset_slot).
    gates["B_next_episode_id_consistent"] = (
        pending.next_episode_id == NUM_ENVS + r_replay_len)
    pv = list(pending.policy_version)
    gates["B_policy_version_present"] = (len(pv) == NUM_ENVS)
    gates["B_policy_version_zero_at_update0"] = all(int(v) == 0 for v in pv)
    # initial memory set for every non-empty pending slot.
    init_mem_ok = all(
        (s["init_mem"] is not None) for s in pending.slots
        if len(s["obs"]) > 0)
    gates["B_initial_memory_set"] = init_mem_ok

    # ── collector state integrity ────────────────────────────────────
    ckeys = set(collector_state.keys())
    need = {"env_state", "obsv", "memories", "mem_mask", "mem_idx"}
    gates["B_collector_keys_complete"] = need.issubset(ckeys)
    gates["B_collector_obsv_finite"] = _leaves_finite(collector_state["obsv"])
    gates["B_collector_memories_finite"] = _leaves_finite(
        collector_state["memories"])

    # ── GATE 15: round-trip bit-exact (re-save w/ pending+collector) ─
    print("\n[15] round-trip: re-save (pending+collector) -> re-restore ...")
    tmp = tempfile.mkdtemp(prefix="p2v1_planb_level1_rt_")
    try:
        save_full_checkpoint(
            r_ts, replay, r["rng"], gs, tmp, step=gs,
            action_rng_state=r["action_rng_state"], update_count=uc,
            pending_state=pending_state, collector_state=collector_state)
        r2 = L.restore_p2_v1_checkpoint(tmp, gs, network, cfg, obs_dim)

        gates["15_roundtrip_params_bitexact"] = _params_equal(
            r_params, r2["train_state"].params)
        gates["15_roundtrip_opt_bitexact"] = _params_equal(
            r_ts.opt_state, r2["train_state"].opt_state)
        gates["15_roundtrip_jax_rng_equal"] = bool(
            jnp.all(r["rng"] == r2["rng"]))
        gates["15_roundtrip_global_step"] = (int(r2["global_step"]) == gs)
        gates["15_roundtrip_update_count"] = (int(r2["update_count"]) == uc)

        # replay contents bit-exact
        rb2 = r2["replay_buffer"]
        rt_ok = (len(rb2) == r_replay_len)
        if rt_ok:
            for ta, tb in zip(replay._buffer, rb2._buffer):
                if not (_bytes_equal(ta.observations, tb.observations)
                        and _bytes_equal(ta.actions, tb.actions)
                        and _bytes_equal(ta.rewards, tb.rewards)
                        and _bytes_equal(ta.dones, tb.dones)):
                    rt_ok = False
        gates["15_roundtrip_replay_contents_bitexact"] = rt_ok

        # pending buffers bit-exact
        p2 = PendingEpisodeBuffers.from_state_dict(r2["pending_state"])
        pend_ok = (p2.slot_lengths() == pending.slot_lengths()
                   and p2.next_episode_id == pending.next_episode_id
                   and list(p2.episode_id) == ep_ids
                   and list(p2.policy_version) == pv)
        if pend_ok:
            for sa, sb in zip(pending.slots, p2.slots):
                for k in ("obs", "act", "rew", "don", "val", "lp",
                          "next_obs", "mem_pre", "mask_pre", "ach"):
                    for xa, xb in zip(sa[k], sb[k]):
                        if not _bytes_equal(xa, xb):
                            pend_ok = False
                if not _bytes_equal(sa["init_mem"], sb["init_mem"]):
                    pend_ok = False
        gates["15_roundtrip_pending_bitexact"] = pend_ok

        # collector state bit-exact
        c2 = r2["collector_state"]
        coll_ok = (_bytes_equal(collector_state["obsv"], c2["obsv"])
                   and _bytes_equal(collector_state["memories"], c2["memories"])
                   and _bytes_equal(collector_state["mem_mask"], c2["mem_mask"])
                   and _bytes_equal(collector_state["mem_idx"], c2["mem_idx"])
                   and _params_equal(collector_state["env_state"],
                                     c2["env_state"]))
        gates["15_roundtrip_collector_bitexact"] = coll_ok

        # action RNG continuing stream identical
        probes = np.random.default_rng(7).standard_normal((4, 43))
        sm = np.exp(probes - probes.max(axis=-1, keepdims=True))
        pr = sm / sm.sum(axis=-1, keepdims=True)
        g_a = restore_action_rng(r["action_rng_state"], seed=L.P2_V1_MASTER_SEED)
        g_b = restore_action_rng(r2["action_rng_state"],
                                 seed=L.P2_V1_MASTER_SEED)
        seq_a = [int(sample_actions(g_a, pr)[0]) for _ in range(16)]
        seq_b = [int(sample_actions(g_b, pr)[0]) for _ in range(16)]
        gates["15_roundtrip_action_rng_stream"] = (seq_a == seq_b)
        print(f"  round-trip: params={gates['15_roundtrip_params_bitexact']} "
              f"opt={gates['15_roundtrip_opt_bitexact']} "
              f"pending={pend_ok} collector={coll_ok} "
              f"action_rng={seq_a == seq_b}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    _finish(gates, all(gates.values()), r, source_sha, sess_dir,
            {"completed": completed, "pending": pending_tot,
             "max_pending_len": max_pending_len,
             "next_episode_id": pending.next_episode_id,
             "slot_lengths": pending.slot_lengths()},
            ppo_vals)


def _finish(gates, all_pass, r, source_sha, sess_dir, conservation, ppo_vals):
    print("\n" + "=" * 64)
    for k in sorted(gates):
        print(f"  {'PASS' if gates[k] else 'FAIL'}  {k}")
    print("=" * 64)
    print(f"方案B LEVEL1 RESTORE VERIFICATION: "
          f"{'ALL PASS' if all_pass else 'FAIL'}")
    report = {
        "directive": "P2-v1 方案B Level 1 full-checkpoint restore (#57 二)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": CKPT_2048,
        "session175_source_sha256": source_sha,
        "jax_devices": str(jax.devices()),
        "training_log": os.path.join(sess_dir, "training_log.jsonl"),
        "ppo_metrics": (None if ppo_vals is None else {
            "total": ppo_vals[0], "policy": ppo_vals[1], "value": ppo_vals[2],
            "entropy": ppo_vals[3], "grad_norm": ppo_vals[4]}),
        "conservation": conservation,
        "gates": gates,
        "all_pass": bool(all_pass),
    }
    ev = os.path.join(os.path.dirname(__file__), "..", "evidence")
    os.makedirs(ev, exist_ok=True)
    with open(os.path.join(ev, "plan_b_level1_restore_report.json"), "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
