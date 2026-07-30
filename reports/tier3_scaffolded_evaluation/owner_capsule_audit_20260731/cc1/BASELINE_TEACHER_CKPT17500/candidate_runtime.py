#!/usr/bin/env python3
"""candidate_runtime.py — CC1 candidate runtime ABI (per-capsule thin binding).
Responsibilities ONLY: checkpoint load, network construction, memory init/reset, policy step,
greedy action, candidate metadata. Does NOT define graph_distance_progress / front_transition_count
/ back_defeat_count / ranking rules (those belong to CC4's common evaluator).

Reads checkpoint_contract.json + candidate_manifest.json from THIS capsule directory and binds the
shared GTRXL128_REFERENCE_RUNTIME (single network implementation, imported from dicode — not copied).
ABI: load_candidate / init_memory / policy_step / reset_memory / candidate_metadata.
"""
import os, sys, json

CAPSULE_DIR = os.path.dirname(os.path.abspath(__file__))

def _shared_runtime_path():
    # Prefer env override; else read environment_lock.json; else default committed worktree path.
    p = os.environ.get("CC1_SHARED_RUNTIME")
    if p: return p
    lock = os.path.join(CAPSULE_DIR, "environment_lock.json")
    if os.path.exists(lock):
        try:
            p = json.load(open(lock)).get("shared_runtime_path")
            if p: return p
        except Exception:
            pass
    return "/home/oseasy/git_work/student_pool_reference_gtrxl/student_pool/shared_runtime"

sys.path.insert(0, _shared_runtime_path())
import gtrxl128_reference_runtime as R  # shared, single network implementation


def _read(name):
    return json.load(open(os.path.join(CAPSULE_DIR, name)))


def load_candidate(checkpoint_contract=None):
    """Load the candidate's params via the loader named in its checkpoint_contract.
    Returns dict(params, network, env_bundle, contract, manifest, obs_dim, action_dim)."""
    manifest = _read("candidate_manifest.json")
    contract = checkpoint_contract or _read("checkpoint_contract.json")
    cfg = {**R.DEFAULT_CFG, **contract.get("cfg_override", {})}
    env_bundle = R.build_stage4_env(batch_size=contract.get("smoke_batch_size", 16),
                                    max_steps=contract.get("max_timesteps", 4096), cfg=cfg)
    network = R.build_network(env_bundle["ACTION_DIM"], cfg=cfg)
    params = R.load_params(contract, network=network, env_bundle=env_bundle, cfg=cfg)
    return {"params": params, "network": network, "env_bundle": env_bundle,
            "contract": contract, "manifest": manifest, "cfg": cfg,
            "obs_dim": env_bundle["OBS_DIM"], "action_dim": env_bundle["ACTION_DIM"]}


def init_memory(batch_size, cfg=None):
    """Stable GTrXL128 window memory; Base/Teacher (no extra memory) use this same stable state."""
    return R.init_memory(batch_size, cfg=cfg)


def policy_step(loaded, observation, memory_state, done_mask, greedy=True, rng=None):
    """Greedy/stochastic forward + memory advance. loaded = load_candidate() output."""
    return R.policy_step(loaded["network"], loaded["params"], observation, memory_state,
                         done_mask, cfg=loaded["cfg"], greedy=greedy, rng=rng)


def reset_memory(memory_state, reset_mask, cfg=None):
    return R.reset_memory(memory_state, reset_mask, cfg=cfg)


def candidate_metadata():
    manifest = _read("candidate_manifest.json")
    contract = _read("checkpoint_contract.json")
    cfg = {**R.DEFAULT_CFG, **contract.get("cfg_override", {})}
    md = R.candidate_metadata(manifest["candidate_id"], cfg=cfg,
                              obs_dim=manifest.get("observation_shape"),
                              action_dim=manifest.get("action_dim"))
    md["candidate_class"] = manifest.get("candidate_class", "STUDENT")
    md["formal_student_ranking_eligible"] = manifest.get("formal_student_ranking_eligible", False)
    return md


if __name__ == "__main__":
    print(json.dumps(candidate_metadata(), indent=2))
