"""Shared CPU test harness for P2-Full-A Gate 2/3/4 (real Henry network).

Builds the real ActorCriticTransformer + init params, and fabricates a complete
episode (random obs with a trailing goal embedding, an achieved goal for hindsight)
whose sparse memory anchors are the REAL scanned entering-state memories. Inserts it
into the ReplayBuffer and draws K valid loss-window samples + their hindsight relabels.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import sys

HENRY_SRC = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src"
BASE_SRC = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
for p in (HENRY_SRC, BASE_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import jax
import jax.numpy as jnp

import hindsight as H
import memory_anchor as MA
from replay_buffer import ReplayBuffer, Trajectory, anchor_steps_for_length
from full_p2_learner import FullP2Config

CFG = FullP2Config()
N_ACH = 4
ORIG_GOAL = 0
HINDSIGHT_GOAL = 2
HINDSIGHT_ACHIEVE_STEP = 50


def build_net(cfg=CFG):
    from dicode.network import ActorCriticTransformer
    import full_p2_learner as FL
    net = ActorCriticTransformer(
        action_dim=cfg.action_dim, activation="relu", hidden_layers=256,
        encoder_size=cfg.embed, num_heads=cfg.num_heads, qkv_features=256,
        num_layers=cfg.num_layers, gating=True, gating_bias=2.0)
    key = jax.random.PRNGKey(0)
    mem = jnp.zeros((2, cfg.window_mem, cfg.num_layers, cfg.embed))
    obs = jnp.zeros((2, cfg.obs_dim))
    mask = jnp.zeros((2, cfg.num_heads, 1, cfg.window_mem + 1), dtype=jnp.bool_)
    params = net.init(key, mem, obs, mask, method=net.model_forward_eval)["params"]

    # raw non-jitted, non-padded forward (B>=2): traced into the differentiated scan
    def apply_eval_raw(params, memories, obs, mask):
        pi, value, mem_out = net.apply(
            {"params": params}, memories, obs, mask, method=net.model_forward_eval)
        return pi.logits, value, mem_out

    # jitted, batch-1 padded forward (B>=1): reconstruction + episode scan
    apply_eval_recon = jax.jit(MA.make_apply_eval(net))

    # jitted lax.scan region scanner (B>=2): eager target scans + KL probes
    def _scan(params, memories, mem_mask, mem_idx, obs_seq):
        return FL._scan_lax(apply_eval_raw, params, memories, mem_mask, mem_idx,
                            obs_seq, cfg)
    scan_fn = jax.jit(_scan)

    return net, params, apply_eval_recon, apply_eval_raw, scan_fn


def _goal_obs(rng, n, obs_dim, goal_idx):
    obs = rng.standard_normal((n, obs_dim)).astype(np.float32) * 0.1
    obs[:, -67:] = H.goal_embedding(goal_idx, 67)
    return obs


def make_episode(cfg, params, apply_eval, ep_len=300, seed=0):
    """Scan a fabricated episode; return a Trajectory with REAL sparse anchors."""
    rng = np.random.RandomState(seed)
    obs = _goal_obs(rng, ep_len, cfg.obs_dim, ORIG_GOAL)              # [ep_len, obs]
    actions = rng.randint(0, cfg.action_dim, ep_len).astype(np.int32)

    achievements = np.zeros((ep_len, N_ACH), np.float32)
    achievements[HINDSIGHT_ACHIEVE_STEP:, HINDSIGHT_GOAL] = 1.0       # goal 2 reached
    target = np.zeros(N_ACH, np.float32); target[ORIG_GOAL] = 1.0

    obs_seq = jnp.asarray(obs)[:, None, :]                            # [ep_len,1,obs]
    pre_mem, pre_mask, pre_idx, logits, values = MA.scan_memory_eval(
        apply_eval, params, obs_seq, cfg.window_mem, cfg.num_heads,
        num_layers=cfg.num_layers, embed=cfg.embed)
    logits = np.asarray(logits[:, 0])                                 # [ep_len, A]
    values = np.asarray(values[:, 0])                                 # [ep_len]
    lp_all = logits - np.log(np.exp(logits).sum(-1, keepdims=True) + 1e-12)
    log_probs = lp_all[np.arange(ep_len), actions].astype(np.float32)  # behavior mu

    steps = anchor_steps_for_length(ep_len)
    anchors = np.asarray(pre_mem[np.array(steps)][:, 0])              # [N, wm, lay, emb]
    dones = np.zeros(ep_len, dtype=bool); dones[-1] = True

    traj = Trajectory(
        observations=obs, actions=actions,
        rewards=(achievements[:, HINDSIGHT_GOAL] * 0.0).astype(np.float32) + 0.1,
        dones=dones, values=values.astype(np.float32), log_probs=log_probs,
        initial_memory=np.asarray(pre_mem[0][0]),
        achievements=achievements, target_achievements=target,
        next_observations=np.roll(obs, -1, axis=0),
        memory_anchors=anchors, anchor_steps=np.array(steps, np.int64),
        collected_update_count=0,
    )
    return traj


def make_samples(cfg, params, apply_eval, K=2, L_seq=129, ep_len=300, seed=0):
    """Return (samples_orig, samples_rel) of K valid windows + hindsight relabels."""
    traj = make_episode(cfg, params, apply_eval, ep_len=ep_len, seed=seed)
    buf = ReplayBuffer(capacity=8, seed=seed)
    buf.insert(traj)
    samples_orig, samples_rel = [], []
    rng = np.random.RandomState(seed + 1)
    max_start = ep_len - L_seq
    for k in range(K):
        start = int(rng.randint(0, max_start + 1))
        s = buf.sample(sequence_length=L_seq, start_step=start)
        samples_orig.append(s)
        samples_rel.append(H.relabel_sample(s, goal_index=HINDSIGHT_GOAL,
                                            embedding_size=67))
    return samples_orig, samples_rel, traj
