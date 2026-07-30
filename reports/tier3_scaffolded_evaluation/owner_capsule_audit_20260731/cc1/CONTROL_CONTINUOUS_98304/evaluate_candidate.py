#!/usr/bin/env python3
"""evaluate_candidate.py — CC1 capsule evaluation entry point.
Binds this candidate's runtime to CC4's COMMON evaluator + FRONT/BACK state banks. CC1 does NOT
implement scientific metrics (graph_distance_progress / front_transition_count / back_defeat_count)
nor ranking — those are CC4's common-evaluator semantics. Until CC4 delivers the common contract,
this records formal_eval_binding=WAITING_CC4_COMMON_CONTRACT and only proves the runtime binding
(load + greedy forward), writing a stub result. It NEVER fabricates formal metrics.

Required args: --candidate-manifest --common-evaluator --front-bank --back-bank --profile
               --output-dir --gpu-uuid
"""
import os, sys, json, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--candidate-manifest", required=True)
ap.add_argument("--common-evaluator", required=True, help="path to CC4 common evaluator module (or WAITING)")
ap.add_argument("--front-bank", required=True)
ap.add_argument("--back-bank", required=True)
ap.add_argument("--profile", required=True)
ap.add_argument("--output-dir", required=True)
ap.add_argument("--gpu-uuid", required=True)
args = ap.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_deterministic_ops=true")

capsule_dir = os.path.dirname(os.path.abspath(args.candidate_manifest))
sys.path.insert(0, capsule_dir)

manifest = json.load(open(args.candidate_manifest))
result = {
    "candidate_id": manifest.get("candidate_id"),
    "candidate_class": manifest.get("candidate_class", "STUDENT"),
    "formal_student_ranking_eligible": manifest.get("formal_student_ranking_eligible", False),
    "gpu_uuid": args.gpu_uuid,
    "args": {"common_evaluator": args.common_evaluator, "front_bank": args.front_bank,
             "back_bank": args.back_bank, "profile": args.profile},
}

# Binding gate: is CC4's common evaluator actually delivered?
def _delivered(p):
    return p and not str(p).upper().startswith("WAITING") and not str(p).startswith("PENDING") and os.path.exists(p)

common_ready = _delivered(args.common_evaluator) and _delivered(args.front_bank) and _delivered(args.back_bank)

if not common_ready:
    result["formal_eval_binding"] = "WAITING_CC4_COMMON_CONTRACT"
    result["formal_metrics"] = None
    result["note"] = ("CC4 common evaluator / FRONT/BACK state banks not yet delivered. "
                      "CC1 does not implement scientific metrics. Runtime binding check only.")
    # Prove the runtime binding (load + one greedy forward) without computing any formal metric.
    try:
        import candidate_runtime as CR
        loaded = CR.load_candidate()
        import jax.numpy as jnp
        mem = CR.init_memory(loaded["env_bundle"]["eval_env_params"] is not None and 16 or 16)
        obs = jnp.zeros((16, loaded["obs_dim"]))
        done = jnp.zeros((16,), dtype=jnp.bool_)
        step = CR.policy_step(loaded, obs, mem, done, greedy=True)
        result["runtime_binding_check"] = {
            "status": "PASS", "obs_dim": loaded["obs_dim"], "action_dim": loaded["action_dim"],
            "greedy_action0": int(jnp.asarray(step["action"])[0]),
            "logits_finite": bool(jnp.all(jnp.isfinite(step["logits"]))),
        }
    except Exception as e:
        result["runtime_binding_check"] = {"status": "FAIL", "error": repr(e)}
else:
    # A real common evaluator path is present: defer metric computation to CC4's module. CC1 still
    # does NOT redefine metrics; it would import and call CC4's evaluator. Until that contract is
    # frozen and handed off, treat as waiting to avoid any non-canonical metric implementation.
    result["formal_eval_binding"] = "WAITING_CC4_COMMON_CONTRACT"
    result["note"] = "Common evaluator path present but CC4 contract not frozen/handed-off; deferring to CC4."

os.makedirs(args.output_dir, exist_ok=True)
out_path = os.path.join(args.output_dir, "evaluate_result.json")
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
print("WROTE " + out_path)
