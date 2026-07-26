#!/usr/bin/env python3
"""Verify the fresh Level2 Stage A checkpoint (checkpoints/4096) restores via the
production primitive and that completed(replay)+pending == 4096 (conservation)."""
import os, sys
import numpy as np
import jax
import jax.numpy as jnp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import stage4_continue_launcher as L
from pending_episodes import PendingEpisodeBuffers

cfg = L.Cfg()
ach = jnp.array([L.get_achievement_multi_hot([L.Achievement.DEFEAT_KOBOLD])],
                dtype=jnp.float32)
emb = int(ach.shape[1])
dummy = L.CraftaxAugObsTrain(condition_on_task=True, conditioning_type="embedding",
                             embedding_size=emb, task_embeddings=jnp.zeros((1, emb)))
obs_dim = dummy.observation_space(dummy.default_params).shape[0]
action_dim = dummy.action_space(dummy.default_params).n
net = L.ActorCriticTransformer(
    action_dim=action_dim, activation=cfg.activation,
    hidden_layers=cfg.hidden_layers, encoder_size=cfg.embed_size,
    num_heads=cfg.num_heads, qkv_features=cfg.qkv_features,
    num_layers=cfg.num_layers, gating=cfg.gating, gating_bias=cfg.gating_bias)

r = L.restore_p2_v1_checkpoint(L.CKPT_ROOT, 4096, net, cfg, obs_dim)
pend = PendingEpisodeBuffers.from_state_dict(r["pending_state"])
completed = sum(int(t.length) for t in r["replay_buffer"]._buffer)
pending = pend.total_pending_transitions()
finite = all(bool(jnp.all(jnp.isfinite(jnp.asarray(l))))
             for l in jax.tree_util.tree_leaves(r["train_state"].params))
opt_step = L._optimizer_step_count(r["train_state"])
gs = int(r["global_step"]); uc = int(r["update_count"])
print("global_step=%d update_count=%d opt_step=%d" % (gs, uc, opt_step))
print("completed(replay)=%d pending=%d sum=%d (expect 4096)" % (completed, pending, completed + pending))
print("replay_len=%d longest=%d" % (len(r["replay_buffer"]), r["replay_buffer"].longest_trajectory_length))
print("params_finite=%s collector_present=%s pending_present=%s" % (
    finite, r["collector_state"] is not None, r["pending_state"] is not None))
ok = (completed + pending == 4096) and finite and gs == 4096
print("RESTORE_4096_CONSERVATION_OK=%s" % ok)
sys.exit(0 if ok else 1)
