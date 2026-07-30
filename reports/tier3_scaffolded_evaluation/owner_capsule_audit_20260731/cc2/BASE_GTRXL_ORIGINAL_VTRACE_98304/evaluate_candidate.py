#!/usr/bin/env python
"""CC2 student-pool candidate evaluator shim — BASE_GTRXL_ORIGINAL_VTRACE_98304.

§六 unified entrypoint. Accepts the common contract CLI:
    --candidate-manifest  path to this capsule's candidate_manifest.json
    --common-evaluator    path to the CC4 common-evaluator module (optional; may not exist yet)
    --front-bank / --back-bank   evaluation episode banks (defined by the CC4 common contract)
    --profile             evaluation profile name (defined by the CC4 common contract)
    --output-dir          where to write results
    --gpu-uuid            GPU to bind (UUID; this shim verifies the visible device matches)

FORMAL METRICS ARE NOT DEFINED HERE. formal_eval_binding=WAITING_CC4_COMMON_CONTRACT: until CC4
publishes the common evaluator + metric contract, this shim ONLY (a) proves the candidate loads
through the §六 ABI and (b) exposes a greedy-rollout generator the common evaluator can drive. If a
--common-evaluator module is supplied AND importable AND exposes `run_evaluation(candidate, ns)`,
control is delegated to it; otherwise the shim writes a binding-pending result and defines nothing.
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import importlib.util


def _load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_candidate_from_manifest(manifest):
    here = os.path.dirname(os.path.abspath(manifest.get("_manifest_path", "")))
    # candidate_runtime.py lives next to this shim in the capsule
    capsule_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, capsule_dir)
    import candidate_runtime  # noqa: E402
    contract = manifest["checkpoint_contract"]
    if isinstance(contract, str):
        cpath = contract if os.path.isabs(contract) else os.path.join(capsule_dir, contract)
        with open(cpath, "r", encoding="utf-8") as f:
            contract = json.load(f)
    return candidate_runtime, candidate_runtime.load_candidate(contract)


def _bind_gpu(gpu_uuid):
    """Best-effort: confirm the visible CUDA device matches the requested UUID (informational)."""
    info = {"requested_gpu_uuid": gpu_uuid, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
        info["visible_gpus"] = [ln.strip() for ln in out if ln.strip()]
    except Exception:
        info["visible_gpus"] = []
    return info


def greedy_rollout_generator(candidate, env, env_state, obs, env_params, num_steps):
    """Yield per-step greedy (obs, action, logits) by driving candidate.policy_step + env.step.

    This is a CONVENIENCE for the CC4 common evaluator (so it never reimplements the Base GTrXL
    transition). It computes NO metric — scoring/reward-aggregation belongs to the common contract.
    """
    import numpy as np
    import jax
    import jax.numpy as jnp
    ms = candidate.init_memory(int(np.asarray(obs).shape[0]))
    done = jnp.zeros((int(np.asarray(obs).shape[0]),), jnp.bool_)
    rng = jax.random.PRNGKey(0)
    for t in range(int(num_steps)):
        action, logits, ms = candidate.policy_step(np.asarray(obs), ms, np.asarray(done))
        yield t, np.asarray(obs), np.asarray(action), np.asarray(logits)
        rng, s_rng = jax.random.split(rng)
        obs, env_state, _rew, done_j, _info = env.step(s_rng, env_state, np.asarray(action), env_params)
        done = jnp.asarray(done_j).astype(jnp.bool_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-manifest", required=True)
    ap.add_argument("--common-evaluator", default=None)
    ap.add_argument("--front-bank", default=None)
    ap.add_argument("--back-bank", default=None)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--gpu-uuid", default=None)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    manifest = _load_manifest(args.candidate_manifest)
    manifest["_manifest_path"] = args.candidate_manifest
    gpu_info = _bind_gpu(args.gpu_uuid)

    candidate_runtime, candidate = _load_candidate_from_manifest(manifest)
    md = candidate.candidate_metadata()

    result = {
        "candidate_id": md["candidate_id"],
        "abi_version": md["abi_version"],
        "params_sha256": md["params_sha256"],
        "manifest_params_sha_match": md["params_sha256"] == md["manifest_params_sha256"],
        "read_path_skipped": md["read_path_skipped"],
        "candidate_loaded": True,
        "gpu": gpu_info,
        "cli": {
            "candidate_manifest": args.candidate_manifest,
            "common_evaluator": args.common_evaluator,
            "front_bank": args.front_bank,
            "back_bank": args.back_bank,
            "profile": args.profile,
            "gpu_uuid": args.gpu_uuid,
        },
    }

    # ---- delegate to the CC4 common evaluator if (and only if) it is actually available ----
    delegated = False
    if args.common_evaluator and os.path.isfile(args.common_evaluator):
        spec = importlib.util.spec_from_file_location("cc4_common_evaluator", args.common_evaluator)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            if hasattr(mod, "run_evaluation"):
                result["formal_eval_binding"] = "DELEGATED_TO_CC4_COMMON_CONTRACT"
                result["common_evaluator_module"] = args.common_evaluator
                result["formal_metrics"] = mod.run_evaluation(candidate, args)
                delegated = True
        except Exception as e:  # do NOT mask as a metric result
            result["common_evaluator_error"] = repr(e)

    if not delegated:
        # No usable common evaluator yet: bind-pending. Define NO formal metric here.
        result["formal_eval_binding"] = "WAITING_CC4_COMMON_CONTRACT"
        result["formal_metrics"] = None
        result["note"] = ("Formal metrics are defined by the CC4 common contract, not this shim. "
                          "The candidate loads through the cc2_student_pool_v1 ABI and exposes a "
                          "greedy_rollout_generator for the common evaluator to drive.")

    out_path = os.path.join(args.output_dir, "evaluate_candidate_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
