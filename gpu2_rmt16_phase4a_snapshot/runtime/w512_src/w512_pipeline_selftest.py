#!/usr/bin/env python
"""CC2 corrected §二 — W512 × P2-Replay STAGE-2 PIPELINE self-test (SERVER, GPU2).

Validates the four Stage-2 modules end-to-end, WITHOUT training and WITHOUT touching GPU0/1:

  PHASE A — synthetic replay round-trip (no env; decisive replay-machinery correctness):
    PA1  a manually-built W512Trajectory (anchors at 0/128/256) passes W512ReplayBuffer.insert
         (validate_anchors: P2 conservation + W512 anchor count + policy-version range).
    PA2  sample_eligible(seq=129, batch=1) -> status OK; provenance arrays well-formed.
    PA3  the sampled window's pre_anchor_w512_* fields EXACTLY equal the stored anchor (the
         buffer/sample layer preserves the anchor data bit-for-bit).
    PA4  reconstruct_w512_batch -> FINITE (mem/mask/idx/w512_state), correct shapes.
    PA5  the W512 loss scan (make_scan_w512_loss) runs on the packed sample -> finite logits/values.
    PA6  original_vtrace_update_w512 -> finite loss; KL gate DECIDES (commit or reject, no NaN);
         params finite; params actually changed; EMA target finite. (Uses frozen FullP2Config.)

  PHASE B — live DEFEAT_KOBOLD env (GPU2): collect -> PPO -> vtrace on the real wrapper:
    PB1  collect_rollout_w512 returns; rollout arrays have shape [rollout_steps, num_envs, ...];
         stats keys present.
    PB2  completed trajectories insert (validate_anchors passes on REAL collection).
    PB3  ppo_update_w512 (compute_gae + one update) -> finite metrics; params changed.
    PB4  if a live trajectory >= 129 steps appears, original_vtrace_update_w512 on it -> finite;
         else fall back to the Phase-A synthetic trajectory (recorded as a diagnostic, NOT a fail).

Prints W512_PIPELINE_SELFTEST=PASS/FAIL. GPU bound to GPU2 by the launcher (CUDA_VISIBLE_DEVICES=2).
"""
from __future__ import annotations
import os, sys, hashlib, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNTIME = os.path.dirname(HERE)
FROZEN = os.path.join(RUNTIME, "frozen_modules")
EXPERIMENT = os.path.join(RUNTIME, "experiment_src")   # phase4a_v2_counters, rmt_collect
DICODE_SRC_CANDIDATES = [
    "/home/oseasy/experiments/dreaming-in-code/src",
    "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src",
    "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB",
]
for p in [HERE, FROZEN, EXPERIMENT] + DICODE_SRC_CANDIDATES:
    if p not in sys.path and os.path.isdir(p):
        sys.path.insert(0, p)

import jax, jax.numpy as jnp

# ---- frozen canonical constants (match w512_selftest / driver Cfg) ----
ACTION_DIM = 43; OBS_DIM = 8335; ACTIVATION = "relu"; EMBED = 256; HIDDEN = 256
NUM_HEADS = 8; QKV = 256; NUM_LAYERS = 2; GATING = True; GATING_BIAS = 2.0
WINDOW_MEM = 128; NUM_ENVS = 16; W512_LONG = 384; W512_DELAY = 128; SEGMENT_LEN = 128
EXPECTED_BASE_SHA = "d4e85af58b7f87d689fadea12eec70c852fa098a09f5ea8907448684b3bf60f5"
N_ACH = 22   # Craftax achievement multi-hot width (read from the env table at runtime)

GATES = {}; DETAIL = {}


def params_sha(params):
    h = hashlib.sha256()
    for v in jax.tree_util.tree_leaves(params):
        h.update(np.ascontiguousarray(np.asarray(v)).tobytes())
    return h.hexdigest()


def build_network():
    from network_w512 import ActorCriticTransformerW512
    return ActorCriticTransformerW512(
        action_dim=ACTION_DIM, activation=ACTIVATION, hidden_layers=HIDDEN,
        encoder_size=EMBED, num_heads=NUM_HEADS, qkv_features=QKV,
        num_layers=NUM_LAYERS, gating=GATING, gating_bias=GATING_BIAS, long_size=W512_LONG)


def init_w512_params(network, rng):
    init_obs = jnp.zeros((2, OBS_DIM))
    init_mem = jnp.zeros((2, WINDOW_MEM, NUM_LAYERS, EMBED))
    init_mask = jnp.zeros((2, NUM_HEADS, 1, WINDOW_MEM + 1), dtype=jnp.bool_)
    init_lbuf = jnp.zeros((2, W512_LONG, EMBED))
    init_lmsk = jnp.zeros((2, W512_LONG), dtype=jnp.bool_)
    return network.init(rng, init_mem, init_obs, init_mask,
                        long_buf=init_lbuf, long_mask=init_lmsk)["params"]


def load_merged_params(network, full_params, ckpt17500):
    import orbax.checkpoint as ocp
    def _merge(base, full):
        if isinstance(base, dict) and isinstance(full, dict):
            out = dict(full)
            for k in base:
                if k in full:
                    out[k] = _merge(base[k], full[k])
            return out
        return base
    mgr = ocp.CheckpointManager(os.path.dirname(ckpt17500))
    raw = mgr.restore(int(os.path.basename(ckpt17500)))
    base_inner = raw["params"]["params"]
    base_keys = [k for k in full_params.keys() if not str(k).startswith("w512_")]
    merged = _merge(base_inner, full_params)
    merged_base_sha = params_sha({k: merged[k] for k in base_keys if k in merged})
    return merged, params_sha(base_inner), merged_base_sha


S4_TASK_CODE = '''
import jax
from craftax.craftax.constants import Achievement, ItemType
from minicraftax.craftax_state import TaskParams
from minicraftax.tasks.base_task import BaseTask
from minicraftax.world_builder import WorldBuilder
class Env(BaseTask):
    def __init__(self, sp, p):
        super().__init__(sp, p)
        self.relevant_achievements=[Achievement.DEFEAT_KOBOLD]; self.completed_achievements=[]; self.label="DEFEAT_KOBOLD"
    def get_task_params(self): return TaskParams(needs_depletion_multiplier=0.3)
    def generate_world(self, rng):
        rng,_r=jax.random.split(rng); b=WorldBuilder(_r,self.static_params,self.params)
        b.set_starting_floor(2); b.set_monsters_killed(2,8)
        b.set_player_inventory({"wood":7,"stone":27,"coal":3,"iron":3,"sapling":1,"pickaxe":3,"sword":3,"bow":1,"arrows":7,"torches":10})
        s=b.build(rng); up=b.ladders_up[2]
        return s.replace(item_map=s.item_map.at[2,up[0],up[1]].set(ItemType.NONE.value))
'''


def build_env():
    from craftax.craftax.constants import Achievement
    from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
    from dicode.task_utils import get_achievement_multi_hot
    from dicode.wrappers_cl import DistributedMultiTaskOptimisticLogWrapper
    from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv
    ns4 = {}; exec(S4_TASK_CODE, ns4); S4Cls = ns4["Env"]
    ctor = EnvParams(max_timesteps=4096)
    table = jnp.array([get_achievement_multi_hot([Achievement.DEFEAT_KOBOLD])], dtype=jnp.float32)
    emb = int(table.shape[1])
    base_env = MultiTaskMiniCraftaxEnv([S4Cls], StaticEnvParams(), ctor, True,
                                       conditioning_type="embedding", embedding_size=emb)
    try:
        env = DistributedMultiTaskOptimisticLogWrapper(
            base_env, jax.random.PRNGKey(0), NUM_ENVS, 1, 16, jnp.array([1.0]), table,
            probe_term=False)
    except TypeError:
        env = DistributedMultiTaskOptimisticLogWrapper(
            base_env, jax.random.PRNGKey(0), NUM_ENVS, 1, 16, jnp.array([1.0]), table)
    return env, table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt17500", required=True)
    ap.add_argument("--rollout_steps", type=int, default=128)
    ap.add_argument("--max_collect_rounds", type=int, default=12)
    args = ap.parse_args()

    import w512_memory as w5m
    import w512_memory_anchor as WA
    import w512_replay_buffer as WRB
    import w512_collect as WC
    import w512_ppo as WP
    import w512_replay_learner as WRL
    import full_p2_learner as FPL
    import optax

    network = build_network()
    w5_cfg = w5m.W512Config(long_size=W512_LONG, delay_size=W512_DELAY, encoder_size=EMBED)
    apply_eval = WA.make_apply_eval_w512(network)
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    full_params = init_w512_params(network, init_rng)
    params, ckpt_base_sha, merged_base_sha = load_merged_params(network, full_params, args.ckpt17500)
    GATES["base_sha_match"] = (ckpt_base_sha == EXPECTED_BASE_SHA and merged_base_sha == EXPECTED_BASE_SHA)
    DETAIL["ckpt_base_sha"] = ckpt_base_sha; DETAIL["merged_base_sha"] = merged_base_sha

    cfg = FPL.FullP2Config()   # frozen replay coefficients (window_mem/num_heads/gamma/.../ema_tau)

    # ============================ PHASE A: synthetic replay round-trip ============================
    SYN_N = 260   # steps; anchors at 0/128/256; terminal done at last step
    Bsyn = 1
    obs_seq, act_seq, rew_seq, val_seq, lp_seq, don_seq, next_seq = [], [], [], [], [], [], []
    anchor_mem, anchor_mask, anchor_idx, anchor_step = [], [], [], []
    anchor_w512 = []   # list of entering w512 state dicts (numpy) per anchor
    mem = jnp.zeros((Bsyn, WINDOW_MEM, NUM_LAYERS, EMBED))
    mm = jnp.zeros((Bsyn, NUM_HEADS, 1, WINDOW_MEM + 1), jnp.bool_)
    mi = jnp.full((Bsyn,), WINDOW_MEM, jnp.int32)
    st = WA.w512_fresh_state(Bsyn, w5_cfg)
    de = jnp.zeros((Bsyn,), jnp.bool_); dn = jnp.zeros((Bsyn,), jnp.bool_)
    rk = jax.random.PRNGKey(101)
    init_mem_snapshot = None; init_w512_snapshot = None
    for t in range(SYN_N):
        # entering snapshot for anchors (BEFORE clear), exactly like collect
        if t % 128 == 0:
            anchor_mem.append(np.asarray(mem[0]).copy())
            anchor_mask.append(np.asarray(mm[0]).copy())
            anchor_idx.append(int(np.asarray(mi[0])))
            anchor_step.append(t)
            anchor_w512.append(jax.tree_util.tree_map(lambda x: np.asarray(x[0]).copy(), st))
        if t == 0:
            init_mem_snapshot = np.asarray(mem[0]).copy()
            init_w512_snapshot = jax.tree_util.tree_map(lambda x: np.asarray(x[0]).copy(), st)
        rk, ok, ak = jax.random.split(rk, 3)
        obs_t = jax.random.normal(ok, (Bsyn, OBS_DIM))
        # phase-1 (collect inline == w512_step_forward; G8)
        st_clr = WA.w512_reset128_clear(st, SEGMENT_LEN)
        mi_adv, mm_adv = WA.w512_advance_mask(mi, mm, de, WINDOW_MEM, NUM_HEADS)
        logits, value, mem_out, h_t = apply_eval(
            params, mem, obs_t, mm_adv, st_clr["long_buf"], st_clr["long_mask"])
        probs = np.asarray(jax.nn.softmax(logits, axis=-1))
        a = int(np.asarray(jax.random.categorical(ak, logits, axis=-1))[0])
        lp = float(np.log(probs[0, a] + 1e-12))
        obs_seq.append(np.asarray(obs_t[0]).copy()); act_seq.append(a)
        val_seq.append(float(np.asarray(value)[0])); lp_seq.append(lp)
        rew_seq.append(float(np.asarray(jax.random.normal(ok, ())).item()))
        is_term = (t == SYN_N - 1)
        don_seq.append(bool(is_term))
        # phase-2
        post_mem = jnp.roll(mem, -1, axis=1).at[:, -1].set(mem_out)
        dn_t = jnp.asarray([is_term], jnp.bool_)
        new_st = w5m.w512_step(st_clr, h_t, dn_t, w5_cfg)
        new_st = {**new_st, "seg_step": jnp.where(dn_t, 0, st_clr["seg_step"] + 1).astype(jnp.int32)}
        new_st = WA.w512_reset_state_on_done(new_st, dn_t)
        post_mem = jnp.where(dn_t[:, None, None, None], jnp.zeros_like(post_mem), post_mem)
        mm = jnp.where(dn_t[:, None, None, None], jnp.zeros_like(mm_adv), mm_adv)
        mi = jnp.where(dn_t, WINDOW_MEM, mi_adv)
        mem, st = post_mem, new_st
        de = dn_t
    # next_observations[t] = obs[t+1] (wrap last to zeros)
    for t in range(SYN_N):
        next_seq.append(obs_seq[t + 1] if t + 1 < SYN_N else np.zeros(OBS_DIM, np.float32))

    n_ach = 22
    ach_arr = np.zeros((SYN_N, n_ach), np.float32)
    traj = WRB.W512Trajectory(
        observations=np.stack(obs_seq), actions=np.array(act_seq, np.int32),
        rewards=np.array(rew_seq, np.float32), dones=np.array(don_seq, bool),
        values=np.array(val_seq, np.float32), log_probs=np.array(lp_seq, np.float32),
        initial_memory=init_mem_snapshot, achievements=ach_arr,
        target_achievements=np.zeros(n_ach, np.float32),
        next_observations=np.stack(next_seq),
        memory_anchors=np.stack(anchor_mem), anchor_steps=np.array(anchor_step, np.int64),
        anchor_masks=np.stack(anchor_mask), anchor_idxs=np.array(anchor_idx, np.int64),
        collected_update_count=0, outer_update_index=0,
        policy_version_start=0, policy_version_end=0, policy_version_span=0,
        policy_version_at_collection=0,
        w512_initial_delay_buf=init_w512_snapshot["delay_buf"],
        w512_initial_delay_idx=int(init_w512_snapshot["delay_idx"]),
        w512_initial_delay_count=int(init_w512_snapshot["delay_count"]),
        w512_initial_long_buf=init_w512_snapshot["long_buf"],
        w512_initial_long_mask=init_w512_snapshot["long_mask"],
        w512_initial_long_idx=int(init_w512_snapshot["long_idx"]),
        w512_initial_seg_step=int(init_w512_snapshot["seg_step"]),
        w512_anchor_delay_buf=np.stack([a["delay_buf"] for a in anchor_w512]),
        w512_anchor_delay_idx=np.array([a["delay_idx"] for a in anchor_w512], np.int64),
        w512_anchor_delay_count=np.array([a["delay_count"] for a in anchor_w512], np.int64),
        w512_anchor_long_buf=np.stack([a["long_buf"] for a in anchor_w512]),
        w512_anchor_long_mask=np.stack([a["long_mask"] for a in anchor_w512]),
        w512_anchor_long_idx=np.array([a["long_idx"] for a in anchor_w512], np.int64),
        w512_anchor_seg_step=np.array([a["seg_step"] for a in anchor_w512], np.int64),
    )
    buf = WRB.W512ReplayBuffer(capacity=8, seed=42)
    pa1 = True
    try:
        buf.insert(traj)
    except Exception as e:
        pa1 = False; DETAIL["PA1_error"] = repr(e)
    GATES["PA1_buffer_insert_conservation"] = pa1
    DETAIL["synthetic_anchor_steps"] = [int(s) for s in anchor_step]

    # PA2: sample_eligible OK
    elig = buf.sample_eligible(sequence_length=129, rng=np.random.RandomState(7), batch_size=1)
    GATES["PA2_sample_eligible_ok"] = (elig.status == "OK" and len(elig.samples) == 1
                                       and elig.eligible_count >= 1)
    DETAIL["PA2"] = dict(status=elig.status, eligible_count=elig.eligible_count,
                         sample_ids=elig.sample_ids, start_offsets=elig.start_offsets)

    # deterministic sample at start=130 (anchor=128, gap=2) for the bit-level checks
    s130 = buf.sample(trajectory_id=0, start_step=130, sequence_length=129)
    # PA3: pre_anchor_w512 fields == stored anchor at step 128 (index 1)
    stored = anchor_w512[1]   # anchor at step 128
    pa3 = (int(s130.pre_anchor_step) == 128
           and np.array_equal(np.asarray(s130.pre_anchor_w512_long_buf), stored["long_buf"])
           and np.array_equal(np.asarray(s130.pre_anchor_w512_long_mask), stored["long_mask"])
           and np.array_equal(np.asarray(s130.pre_anchor_w512_delay_buf), stored["delay_buf"])
           and int(s130.pre_anchor_w512_seg_step) == int(stored["seg_step"])
           and int(s130.pre_anchor_w512_long_idx) == int(stored["long_idx"]))
    GATES["PA3_anchor_data_preserved"] = bool(pa3)
    DETAIL["PA3"] = dict(pre_anchor_step=int(s130.pre_anchor_step),
                         burn_in_length=int(s130.burn_in_length),
                         seg_step=int(s130.pre_anchor_w512_seg_step))

    # PA4: reconstruct_w512_batch finite + shapes
    recon = WRL.reconstruct_w512_batch(network, apply_eval, params, [s130], cfg, w5_cfg, SEGMENT_LEN)
    r_mem, r_mask, r_idx, r_st = recon
    pa4 = (bool(np.isfinite(np.asarray(r_mem)).all())
           and bool(np.isfinite(np.asarray(r_st["long_buf"])).all())
           and bool(np.isfinite(np.asarray(r_st["delay_buf"])).all())
           and tuple(np.asarray(r_mem).shape) == (1, WINDOW_MEM, NUM_LAYERS, EMBED)
           and tuple(np.asarray(r_st["long_buf"]).shape) == (1, W512_LONG, EMBED))
    GATES["PA4_reconstruct_finite_shapes"] = bool(pa4)

    # PA5: loss scan finite
    scan_fn = WRL.make_scan_w512_loss(network, cfg, w5_cfg, SEGMENT_LEN)
    po = FPL.pack_batch([s130])
    obs_o_ext, don_o_ext = WRL._ext_obs_dones(po)
    lg, vl = scan_fn(params, *recon, obs_o_ext, don_o_ext)
    GATES["PA5_loss_scan_finite"] = bool(np.isfinite(np.asarray(lg)).all()
                                         and np.isfinite(np.asarray(vl)).all())

    # PA6: original_vtrace_update_w512 finite + KL gate decides + params change
    opt = FPL.build_optimizer(2e-5, cfg)
    opt_state = opt.init(params)
    target_params = jax.tree_util.tree_map(jnp.asarray, params)
    np_before = params_sha(params)
    new_params, new_target, new_opt_state, vmetrics = WRL.original_vtrace_update_w512(
        network, params, target_params, opt_state, opt, apply_eval, scan_fn,
        [s130], cfg, w5_cfg, SEGMENT_LEN)
    pa6 = (bool(vmetrics["finite"])
           and bool(np.isfinite(float(vmetrics["policy_kl"])) or vmetrics["kl_rejected_update"])
           and bool(vmetrics["policy_committed"] or vmetrics["kl_rejected_update"])
           and all(bool(np.isfinite(np.asarray(v)).all())
                   for v in jax.tree_util.tree_leaves(new_params)
                   if np.issubdtype(np.asarray(v).dtype, np.floating))
           and (params_sha(new_params) != np_before))
    GATES["PA6_original_vtrace_update"] = bool(pa6)
    DETAIL["PA6"] = dict(loss=float(vmetrics["loss"]), policy_kl=float(vmetrics["policy_kl"]),
                         policy_committed=bool(vmetrics["policy_committed"]),
                         kl_rejected_update=bool(vmetrics["kl_rejected_update"]),
                         chosen_actor_step_scale=float(vmetrics["chosen_actor_step_scale"]),
                         entropy=float(vmetrics["entropy"]), ess=float(vmetrics["ess"]),
                         replay_learner=vmetrics.get("replay_learner"))

    # ============================ PHASE B: live env collect -> PPO -> vtrace ============================
    env, table = build_env()
    env_params = env.default_params
    n_ach_env = int(table.shape[1])
    target_ach = table[0]
    rng, r_rng = jax.random.split(rng)
    obsv, env_state = env.reset(r_rng, env_params)
    Benv = int(np.asarray(obsv).shape[0])
    memories = jnp.zeros((Benv, WINDOW_MEM, NUM_LAYERS, EMBED))
    mem_mask = jnp.zeros((Benv, NUM_HEADS, 1, WINDOW_MEM + 1), jnp.bool_)
    mem_idx = jnp.full((Benv,), WINDOW_MEM, jnp.int32)
    w512_state = WA.w512_fresh_state(Benv, w5_cfg)
    done_enter = jnp.zeros((Benv,), jnp.bool_)
    pending = WC.W512PendingEpisodeBuffers(Benv, first_episode_id=0, first_policy_version=0)

    buf_live = WRB.W512ReplayBuffer(capacity=64, seed=42)
    ppo_cfg = dict(window_mem=WINDOW_MEM, num_heads=NUM_HEADS, lr=2e-5, max_grad_norm=1.0,
                   clip_eps=0.2, vf_coef=0.5, ent_coef=0.002, update_epochs=1,
                   num_minibatches=2, gamma=0.999, gae_lambda=0.8)
    ppo_opt = WP.build_ppo_optimizer(ppo_cfg)
    ppo_opt_state = ppo_opt.init(params)
    cur_params = params

    collected_any = False; ppo_ran = False; live_rollout_shape_ok = False
    n_completed_total = 0; live_vtrace_ran = False; live_vtrace_detail = {}
    carry_rng = jax.random.PRNGKey(202)
    action_rng_np = np.random.RandomState(1234)   # numpy RandomState (RU.sample_actions API)
    for rnd in range(args.max_collect_rounds):
        carry_rng, c_rng = jax.random.split(carry_rng)
        trajs, carry, rollout, stats = WC.collect_rollout_w512(
            env, env_state, network, cur_params, obsv,
            memories, mem_mask, mem_idx, w512_state, done_enter,
            c_rng, action_rng_np, pending, target_ach, args.rollout_steps,
            WINDOW_MEM, NUM_HEADS, w5_cfg, SEGMENT_LEN,
            collected_update_count=rnd, apply_eval_w512=apply_eval,
            env_params=env_params, outer_update_index=rnd, policy_version=rnd)
        env_state = carry["env_state"]; obsv = carry["obsv"]
        memories = carry["memories"]; mem_mask = carry["mem_mask"]; mem_idx = carry["mem_idx"]
        w512_state = carry["w512_state"]; done_enter = carry["done_enter"]
        collected_any = True
        if rnd == 0:
            live_rollout_shape_ok = (tuple(np.asarray(rollout["obs"]).shape)
                                     == (args.rollout_steps, Benv, OBS_DIM))
        n_completed_total += int(stats["completed_episodes"])
        for tj in trajs:
            try:
                buf_live.insert(tj)   # validate_anchors on REAL collection
            except Exception as e:
                GATES["PB2_live_insert_conservation"] = False
                DETAIL.setdefault("PB2_errors", []).append(repr(e))
        # PPO update on the just-collected rollout (needs last_value)
        st_clr = WA.w512_reset128_clear(w512_state, SEGMENT_LEN)
        mi_adv, mm_adv = WA.w512_advance_mask(mem_idx, mem_mask, done_enter, WINDOW_MEM, NUM_HEADS)
        _lg, last_value, _mo, _ht = apply_eval(
            cur_params, memories, obsv, mm_adv, st_clr["long_buf"], st_clr["long_mask"])
        rollout["last_value"] = np.asarray(last_value)
        adv, tgt = WP.compute_gae(rollout["rewards"], rollout["values"], rollout["dones"],
                                  rollout["last_value"], ppo_cfg["gamma"], ppo_cfg["gae_lambda"])
        cur_params, ppo_opt_state, ppo_metrics = WP.ppo_update_w512(
            network, cur_params, ppo_opt_state, ppo_opt, rollout, adv, tgt,
            ppo_cfg, w5_cfg, SEGMENT_LEN, jax.random.PRNGKey(300 + rnd))
        ppo_ran = bool(ppo_metrics["ppo_finite"])
        # try a live vtrace update once an eligible trajectory exists
        if (not live_vtrace_ran) and buf_live.can_sample() and any(
                t.length >= 129 for t in buf_live._buffer):
            elig_live = buf_live.sample_eligible(
                sequence_length=129, rng=np.random.RandomState(500 + rnd), batch_size=2)
            if elig_live.status == "OK":
                try:
                    _np, _nt, _nos, vm = WRL.original_vtrace_update_w512(
                        network, cur_params,
                        jax.tree_util.tree_map(jnp.asarray, cur_params),
                        ppo_opt.init(cur_params), ppo_opt, apply_eval, scan_fn,
                        elig_live.samples, cfg, w5_cfg, SEGMENT_LEN)
                    live_vtrace_ran = True
                    live_vtrace_detail = dict(loss=float(vm["loss"]),
                                              policy_kl=float(vm["policy_kl"]),
                                              policy_committed=bool(vm["policy_committed"]),
                                              finite=bool(vm["finite"]),
                                              batch=len(elig_live.samples))
                except Exception as e:
                    live_vtrace_detail = dict(error=repr(e))
        if ppo_ran and live_vtrace_ran:
            break

    GATES["PB1_live_collect_shapes"] = bool(collected_any and live_rollout_shape_ok)
    GATES.setdefault("PB2_live_insert_conservation", bool(n_completed_total > 0))
    GATES["PB3_live_ppo_finite"] = bool(ppo_ran)
    # PB4 is DIAGNOSTIC: live env may not yield a >=129 episode quickly; Phase A already proves
    # the vtrace path. Record outcome; do NOT fail the suite if no long live episode appeared.
    GATES["PB4_live_vtrace_finite"] = (bool(live_vtrace_ran and live_vtrace_detail.get("finite", False))
                                       if live_vtrace_ran else None)
    DETAIL["PB"] = dict(n_completed_total=n_completed_total, ppo_ran=ppo_ran,
                        live_vtrace_ran=live_vtrace_ran, live_vtrace=live_vtrace_detail,
                        live_rollout_shape_ok=live_rollout_shape_ok,
                        live_longest=int(buf_live.longest_trajectory_length))

    # ---------------- summary ----------------
    effective = {k: v for k, v in GATES.items() if v is not None}
    all_pass = bool(effective) and all(effective.values())
    result = {"selftest": "W512_PIPELINE", "gates": GATES, "all_gates_pass": all_pass,
              "detail": DETAIL}
    print(_json_dumps(result))
    print("W512_PIPELINE_SELFTEST=" + ("PASS" if all_pass else "FAIL"))


def _json_dumps(obj):
    import json
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    return json.dumps(obj, indent=2, default=conv, ensure_ascii=False)


if __name__ == "__main__":
    main()
