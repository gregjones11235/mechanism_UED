#!/usr/bin/env python3
"""Fix the replay-aux NONTERMINAL bootstrap batch=1 shape crash in
long_context_learner._replay_aux_loss.

Root cause (diagnosed from the Stage A crash, traceback transformerXL.py:194):
  Henry ``forward_eval`` is written for the rollout batch (num_envs >= 2).  Its
  per-layer ``x = x.squeeze()`` removes EVERY size-1 axis.  With the rollout
  batch=16 this only drops the query-time axis (16,1,E)->(16,E).  But the
  replay-aux off-policy value bootstrap calls ``model_forward_eval`` with a
  SINGLE next-state, i.e. batch=1: after layer 0, x=(1,1,E); squeeze() collapses
  BOTH the batch and the query-time axis -> (E,); on layer 1 (num_layers=2)
  ``jnp.concatenate([memories[:,:,1], x[:,None]])`` then gets (1,128,256) vs
  (256,1) -> TypeError.

This is a P2-v1 CALLER contract violation, NOT a Henry bug; the Henry base is
read-only and correct for its rollout use.  Fix on the P2-v1 side only.

Fix (value-preserving, no algorithm change):
  Batch elements are fully independent in ``forward_eval`` (attention is over
  the seq/memory axis within each element; mask and pos_embed broadcast over the
  batch).  So we TILE the single bootstrap state to batch=2 to keep the batch
  axis alive through the unguarded squeeze, then read element [0].  The returned
  V(next_state) is bit-identical to the intended single-state bootstrap; the new
  CPU test test_replay_aux_bootstrap.py additionally proves batch-independence
  (tile=2 vs tile=3 give the same element [0]).

Each old-string must match EXACTLY ONCE.  Fails loudly otherwise.
"""
import io

PATH = "/home/oseasy/experiments/p2_v1_20260722/src/long_context_learner.py"

with io.open(PATH, encoding="utf-8") as f:
    src = f.read()


def rep(old, new, label):
    n = src.count(old)
    assert n == 1, f"[{label}] anchor matched {n} times (need exactly 1)"
    return src.replace(old, new, 1)


OLD = (
    "            boot_mem = mem_timeline[jnp.clip(eff_L, 0, mem_timeline.shape[0] - 1)][None]\n"
    "            boot_mask = jnp.ones(\n"
    "                (1, self.num_heads, 1, self.window_mem + 1), dtype=jnp.bool_\n"
    "            )\n"
    "            _, boot_val, _ = self.network.apply(\n"
    "                params, boot_mem, next_obs_b, boot_mask,\n"
    "                method=self.network.model_forward_eval,\n"
    "            )\n"
    "            bootstrap = boot_val[0]\n"
)

NEW = (
    "            boot_mem = mem_timeline[jnp.clip(eff_L, 0, mem_timeline.shape[0] - 1)][None]\n"
    "            boot_mask = jnp.ones(\n"
    "                (1, self.num_heads, 1, self.window_mem + 1), dtype=jnp.bool_\n"
    "            )\n"
    "            # Henry forward_eval does an unguarded ``x.squeeze()`` that\n"
    "            # collapses the BATCH axis when batch==1 (it is written for the\n"
    "            # rollout batch = num_envs >= 2).  With this single bootstrap\n"
    "            # state (batch==1) the 2nd transformer layer would receive\n"
    "            # x[:,None] of shape (E,1) instead of (1,1,E) and crash the\n"
    "            # concatenate.  Batch elements are fully independent in\n"
    "            # forward_eval (attention is over the seq axis only; mask and\n"
    "            # pos_embed broadcast over the batch), so we TILE to batch==2 to\n"
    "            # keep the batch axis alive and read element [0] — the returned\n"
    "            # V(next_state) is bit-identical to the intended single-state\n"
    "            # bootstrap.  No change to the Henry base (read-only) and no\n"
    "            # change to the computed value.  Regression-covered by\n"
    "            # tests/test_replay_aux_bootstrap.py.\n"
    "            next_obs_b2 = jnp.tile(next_obs_b, (2, 1))\n"
    "            boot_mem2 = jnp.tile(boot_mem, (2, 1, 1, 1))\n"
    "            _, boot_val, _ = self.network.apply(\n"
    "                params, boot_mem2, next_obs_b2, boot_mask,\n"
    "                method=self.network.model_forward_eval,\n"
    "            )\n"
    "            bootstrap = boot_val[0]\n"
)

src = rep(OLD, NEW, "bootstrap-tile-batch2")

with io.open(PATH, "w", encoding="utf-8") as f:
    f.write(src)

print("PATCH OK: bootstrap batch=1 -> tile-to-2 fix applied")
