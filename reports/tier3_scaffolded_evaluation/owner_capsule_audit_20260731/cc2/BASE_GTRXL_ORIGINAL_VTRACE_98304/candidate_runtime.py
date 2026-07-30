#!/usr/bin/env python
"""CC2 student-pool candidate runtime — BASE_GTRXL_ORIGINAL_VTRACE_98304 (network_family=base_gtrxl).

SELF-CONTAINED §六 ABI. It reuses the FROZEN network + memory modules from the candidate's source
commit (network_rmt16 / rmt16_memory / rmt_memory_anchor) verbatim, so the consumer (CC4 common
evaluator) NEVER implements a Base GTrXL adapter: loading the checkpoint + constructing the network
+ stepping the policy is all done here against the exact frozen code that produced the run.

base_gtrxl operational definition (the ONLY scientific difference vs the persistent/reset128 arms):
the SAME ActorCriticTransformerRMT16 module + SAME ckpt17500, but the RMT16 persistent-token READ
path is SKIPPED — entering_read_tokens(rmt_st, "base_gtrxl") returns None, so model_forward_eval
sees mem_tokens=None and reduces to the pure GTrXL backbone; RMT params get no gradient and stay
frozen at init (rmt_gate zero-init => step0 bit-exact with ckpt17500).

ABI (cc2_student_pool_v1):
    load_candidate(checkpoint_contract) -> Candidate
    Candidate.init_memory(batch_size)                 -> memory_state
    Candidate.policy_step(observation, memory_state, done_mask) -> (action, logits, new_memory_state)
    Candidate.reset_memory(memory_state, reset_mask)  -> new_memory_state
    Candidate.candidate_metadata()                    -> dict

No training, no env, no metric definition lives here. Formal metrics are bound by the CC4 common
contract (formal_eval_binding=WAITING_CC4_COMMON_CONTRACT) — this module only exposes a faithful,
stable, greedy policy + memory semantics.
"""
from __future__ import annotations

import os
import sys
import json
import hashlib
import pickle

ABI_VERSION = "cc2_student_pool_v1"
CANDIDATE_ID = "BASE_GTRXL_ORIGINAL_VTRACE_98304"
NETWORK_FAMILY = "base_gtrxl"
MEMORY_MODE = "none"
REPLAY_MODE = "original_vtrace"
CARRY_MODE = "base_gtrxl"

# ---- frozen architecture constants (train_rmt16_p2replay.py::Cfg + long98304_base_gtrxl.yaml) ----
NET = {
    "activation": "relu",
    "encoder_size": 256,      # embed_size
    "hidden_layers": 256,
    "num_heads": 8,
    "qkv_features": 256,
    "num_layers": 2,
    "gating": True,
    "gating_bias": 2.0,
    "window_mem": 128,
    "rmt_num_tokens": 16,
    "segment_len": 128,       # == num_steps
}
OBS_SHAPE = (8335,)
ACTION_DIM = 43
BASE_CHECKPOINT_PARAMS_SHA256 = "d4e85af58b7f87d689fadea12eec70c852fa098a09f5ea8907448684b3bf60f5"


def _resolve_frozen_module_dir(checkpoint_contract):
    """Locate the frozen modules dir of the candidate's source commit.

    Priority: contract field -> env var -> known server layout. The runtime then VERIFIES the
    loaded module sources against the contract's recorded SHAs (so a wrong/edited tree is caught).
    """
    cand = checkpoint_contract.get("frozen_module_dir")
    if cand and os.path.isdir(cand):
        return cand
    env = os.environ.get("CC2_FROZEN_MODULE_DIR")
    if env and os.path.isdir(env):
        return env
    snap = checkpoint_contract.get("snapshot_root")
    if snap:
        d = os.path.join(snap, "runtime", "frozen_modules")
        if os.path.isdir(d):
            return d
    # known server layout for the base_gtrxl source commit
    d = ("/home/oseasy/cc2_data/cc2_source_2d0cc74/gpu2_rmt16_phase4a_snapshot"
         "/runtime/frozen_modules")
    if os.path.isdir(d):
        return d
    raise FileNotFoundError("cannot locate frozen_modules dir for candidate runtime")


def canonical_params_sha(params):
    """The driver's _params_sha: sha256 over np.ascontiguousarray(asarray(leaf)).tobytes()."""
    import jax
    import numpy as np
    h = hashlib.sha256()
    for v in jax.tree_util.tree_leaves(params):
        h.update(np.ascontiguousarray(np.asarray(v)).tobytes())
    return h.hexdigest()


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Candidate:
    """A loaded base_gtrxl candidate: params + frozen network + greedy step + memory semantics."""

    def __init__(self, params, network, apply_eval_rmt, update_fn, rmt_cfg, metadata):
        self.params = params
        self.network = network
        self.apply_eval_rmt = apply_eval_rmt
        self.update_fn = update_fn
        self.rmt_cfg = rmt_cfg
        self._metadata = metadata
        self.window_mem = NET["window_mem"]
        self.num_heads = NET["num_heads"]
        self.num_layers = NET["num_layers"]
        self.embed_size = NET["encoder_size"]
        self.carry_mode = CARRY_MODE

    # ---- §六 ABI ----

    def init_memory(self, batch_size):
        """Fresh-episode memory state for `batch_size` envs (== the true-done boundary state)."""
        import jax.numpy as jnp
        import rmt16_memory as rmtm
        B = int(batch_size)
        memories = jnp.zeros((B, self.window_mem, self.num_layers, self.embed_size))
        mem_mask = jnp.zeros((B, self.num_heads, 1, self.window_mem + 1), jnp.bool_)
        mem_idx = jnp.full((B,), self.window_mem, dtype=jnp.int32)
        rmt_state = rmtm.rmt16_init(B, self.rmt_cfg)
        return {"memories": memories, "mem_mask": mem_mask,
                "mem_idx": mem_idx, "rmt_state": rmt_state}

    def policy_step(self, observation, memory_state, done_mask):
        """One greedy step through the CANONICAL frozen transition (rmt_step_forward).

        observation : (B, obs_dim) array
        memory_state: dict{memories, mem_mask, mem_idx, rmt_state}
        done_mask   : (B,) bool — envs whose PREVIOUS step terminated (resets their memory first)

        Returns (action, logits, new_memory_state). action = argmax(logits) (greedy).
        base_gtrxl: entering_read_tokens returns None inside rmt_step_forward -> read path skipped.
        """
        import jax.numpy as jnp
        from rmt_memory_anchor import rmt_step_forward
        obs = jnp.asarray(observation)
        done = jnp.asarray(done_mask).astype(jnp.bool_)
        (post_memories, new_mask, new_idx, new_rmt,
         logits, _value, _mem_pre, entering_tokens) = rmt_step_forward(
            self.apply_eval_rmt, self.params,
            memory_state["memories"], memory_state["mem_mask"], memory_state["mem_idx"],
            memory_state["rmt_state"], obs, done,
            self.window_mem, self.num_heads, self.rmt_cfg, self.carry_mode, self.update_fn)
        # base_gtrxl invariant: the read path is skipped -> no entering tokens consumed.
        assert entering_tokens is None, "base_gtrxl must skip the RMT read path (entering=None)"
        action = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        new_state = {"memories": post_memories, "mem_mask": new_mask,
                     "mem_idx": new_idx, "rmt_state": new_rmt}
        return action, logits, new_state

    def reset_memory(self, memory_state, reset_mask):
        """Explicitly reset the memory of the envs flagged in reset_mask (B,) bool."""
        import jax.numpy as jnp
        import rmt16_memory as rmtm
        r = jnp.asarray(reset_mask).astype(jnp.bool_)
        memories = jnp.where(r[:, None, None, None], 0.0, memory_state["memories"])
        mem_mask = jnp.where(r[:, None, None, None], jnp.zeros_like(memory_state["mem_mask"]),
                             memory_state["mem_mask"])
        mem_idx = jnp.where(r, self.window_mem, memory_state["mem_idx"]).astype(jnp.int32)
        rmt_state = rmtm.rmt16_reset_envs(memory_state["rmt_state"], r, self.rmt_cfg)
        return {"memories": memories, "mem_mask": mem_mask,
                "mem_idx": mem_idx, "rmt_state": rmt_state}

    def candidate_metadata(self):
        return dict(self._metadata)


def load_candidate(checkpoint_contract, verify_source_sha=True):
    """Load a base_gtrxl candidate from a checkpoint_contract dict (or path to one).

    Required contract fields: checkpoint_path, params_sha256, snapshot_root (or frozen_module_dir).
    Optional: policy_source_sha256 (network_rmt16.py), memory_anchor_source_sha256,
              rmt16_memory_source_sha256 — verified if present.
    """
    import jax
    import jax.numpy as jnp

    if isinstance(checkpoint_contract, (str, bytes, os.PathLike)):
        with open(checkpoint_contract, "r", encoding="utf-8") as f:
            checkpoint_contract = json.load(f)
    cc = checkpoint_contract

    frozen_dir = _resolve_frozen_module_dir(cc)
    sys.path.insert(0, frozen_dir)
    # also expose experiment_src so any sibling pure module resolves (harmless if absent)
    exp_dir = os.path.join(os.path.dirname(frozen_dir), "experiment_src")
    if os.path.isdir(exp_dir):
        sys.path.insert(0, exp_dir)

    # ---- verify the frozen module sources match the contract (catch a wrong/edited tree) ----
    source_checks = {}
    if verify_source_sha:
        for key, fname in [("policy_source_sha256", "network_rmt16.py"),
                           ("memory_anchor_source_sha256", "rmt_memory_anchor.py"),
                           ("rmt16_memory_source_sha256", "rmt16_memory.py")]:
            expected = cc.get(key)
            fpath = os.path.join(frozen_dir, fname)
            if expected and os.path.isfile(fpath):
                got = _sha256_file(fpath)
                source_checks[fname] = {"expected": expected, "actual": got, "match": got == expected}
                if got != expected:
                    raise ValueError(
                        f"FROZEN_MODULE_SOURCE_MISMATCH {fname}: contract={expected} actual={got}")

    from network_rmt16 import ActorCriticTransformerRMT16
    import rmt16_memory as rmtm
    from rmt_memory_anchor import make_apply_eval_rmt, make_update_fn

    network = ActorCriticTransformerRMT16(
        action_dim=int(cc.get("action_dim", ACTION_DIM)),
        activation=NET["activation"],
        encoder_size=NET["encoder_size"],
        hidden_layers=NET["hidden_layers"],
        num_heads=NET["num_heads"],
        qkv_features=NET["qkv_features"],
        num_layers=NET["num_layers"],
        gating=NET["gating"],
        gating_bias=NET["gating_bias"],
        rmt_num_tokens=NET["rmt_num_tokens"],
    )

    ckpt_path = cc["checkpoint_path"]
    with open(ckpt_path, "rb") as f:
        blob = pickle.load(f)
    params = blob["params"]
    params = jax.tree_util.tree_map(jnp.asarray, params)

    # ---- hard identity gate: recomputed params SHA must equal the contract ----
    recomputed = canonical_params_sha(params)
    expected = cc.get("params_sha256")
    if expected and recomputed != expected:
        raise ValueError(
            f"CHECKPOINT_PARAMS_SHA_MISMATCH: contract={expected} recomputed={recomputed}")

    manifest = blob.get("manifest", {})
    apply_eval_rmt = make_apply_eval_rmt(network)
    update_fn = make_update_fn(network, params)
    rmt_cfg = rmtm.RMT16Config(num_tokens=NET["rmt_num_tokens"],
                               segment_len=NET["segment_len"],
                               encoder_size=NET["encoder_size"])

    metadata = {
        "abi_version": ABI_VERSION,
        "candidate_id": CANDIDATE_ID,
        "network_family": NETWORK_FAMILY,
        "memory_mode": MEMORY_MODE,
        "replay_mode": REPLAY_MODE,
        "carry_mode": CARRY_MODE,
        "observation_shape": list(OBS_SHAPE),
        "action_dim": int(cc.get("action_dim", ACTION_DIM)),
        "window_mem": NET["window_mem"],
        "num_heads": NET["num_heads"],
        "num_layers": NET["num_layers"],
        "embed_size": NET["encoder_size"],
        "rmt_num_tokens": NET["rmt_num_tokens"],
        "segment_len": NET["segment_len"],
        "base_checkpoint_params_sha256": BASE_CHECKPOINT_PARAMS_SHA256,
        "params_sha256": recomputed,
        "manifest_params_sha256": manifest.get("params_sha256"),
        "manifest_step": manifest.get("step"),
        "manifest_arm": manifest.get("arm"),
        "checkpoint_path": ckpt_path,
        "frozen_module_dir": frozen_dir,
        "source_commit_full40": cc.get("source_commit_full40"),
        "read_path_skipped": True,
        "source_checks": source_checks,
    }
    return Candidate(params, network, apply_eval_rmt, update_fn, rmt_cfg, metadata)


if __name__ == "__main__":
    # Minimal self-check (no env): load, init memory, one greedy step on a zero obs batch.
    import argparse
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-contract", required=True)
    ap.add_argument("--batch-size", type=int, default=2)
    args = ap.parse_args()
    cand = load_candidate(args.checkpoint_contract)
    ms = cand.init_memory(args.batch_size)
    obs = np.zeros((args.batch_size, OBS_SHAPE[0]), dtype=np.float32)
    done = np.zeros((args.batch_size,), dtype=bool)
    action, logits, ms2 = cand.policy_step(obs, ms, done)
    md = cand.candidate_metadata()
    print(json.dumps({
        "candidate_id": md["candidate_id"],
        "params_sha256": md["params_sha256"],
        "manifest_params_sha_match": md["params_sha256"] == md["manifest_params_sha256"],
        "action": np.asarray(action).tolist(),
        "logits_shape": list(np.asarray(logits).shape),
        "logits_finite": bool(np.isfinite(np.asarray(logits)).all()),
        "memories_shape": list(np.asarray(ms2["memories"]).shape),
        "read_path_skipped": md["read_path_skipped"],
        "source_checks": md["source_checks"],
        "SELF_CHECK": "PASS",
    }, indent=2))
