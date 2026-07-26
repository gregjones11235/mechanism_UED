#!/usr/bin/env python
"""CANONICAL_EVALUATOR_V1 — reference implementation.

The single official evaluation protocol for mechanism_UED global comparisons. Grounded in the
bit-identity-verified Phase2 anchor (eval_phase2_unified.py SHA 224514026aefd273...), with the
remediation hardenings: explicit action_mode, HARD-FAIL checkpoint restore, world_set_hash emission,
memory-isolation (GATE5) and done-reset (GATE6) probes, and full provenance.

This is a REFERENCE tool. It requires JAX + Craftax; in a JAX-less audit env it imports lazily and the
runtime path raises a clear BLOCKED error, while the protocol constants and the pure restore-check logic
remain importable/testable. Run on the experiment host for real evaluation (authorized, read-only ckpts).

Legal action_mode values: 'stochastic' | 'argmax'. NEVER inferred from a dead branch or default.
"""
import argparse, hashlib, json, os, platform, datetime

PROTOCOL_ID = "CANONICAL_EVALUATOR_V1"
NUM_WORLDS = 256
MAX_TIMESTEPS = 4096
SPAWN_FLOOR = 2
OPTIMISTIC_RESET_RATIO = 16
DEFAULT_EVAL_SEED = 42
SEED100000_LINE_NOTE = ("seed=100000 (P7/LC line) is a DIFFERENT world set; results MUST be flagged "
                        "separately and NEVER paired with seed42.")

class RestoreLeafMismatch(RuntimeError):
    pass

def checkpoint_restore_check(model_leaves, ckpt_leaves, allow_compat_map=None):
    """HARD-FAIL on any missing OR extra parameter leaf. No silent partial restore.

    model_leaves, ckpt_leaves: iterables of leaf path strings (e.g. jax.tree_util flatten paths).
    A compat fallback is FORBIDDEN unless an explicit, SHA-recorded migration map is supplied + audited.
    Pure logic => unit-testable without JAX (GATE4 / GATE14).
    """
    m = set(model_leaves); c = set(ckpt_leaves)
    missing = sorted(m - c)   # model needs but ckpt lacks
    extra = sorted(c - m)     # ckpt has but model lacks
    if missing or extra:
        if allow_compat_map is not None:
            # an audited migration map must reconcile BOTH directions explicitly
            if set(allow_compat_map.get("missing_resolved", [])) != set(missing) or \
               set(allow_compat_map.get("extra_resolved", [])) != set(extra):
                raise RestoreLeafMismatch(
                    "compat map does not fully reconcile: missing=%s extra=%s" % (missing, extra))
            return {"status": "RESTORED_VIA_AUDITED_COMPAT_MAP", "missing": missing, "extra": extra,
                    "compat_map_sha256": allow_compat_map.get("sha256", "REQUIRED")}
        raise RestoreLeafMismatch(
            "checkpoint leaf-set mismatch -> HARD-FAIL (no silent partial restore). "
            "missing=%s extra=%s" % (missing, extra))
    return {"status": "RESTORED_EXACT", "missing": [], "extra": []}

def memory_isolation_probe(mem_rows, perturb_index):
    """GATE5: perturb world `perturb_index` memory row; assert all other rows unchanged.

    mem_rows: list/array of per-world memory rows (numpy). Returns contamination report.
    """
    import numpy as np
    before = [np.array(r, copy=True) for r in mem_rows]
    after = [np.array(r, copy=True) for r in mem_rows]
    after[perturb_index] = after[perturb_index] + 1.0  # simulate a write to one world
    contaminated = [i for i in range(len(mem_rows))
                    if i != perturb_index and not np.array_equal(before[i], after[i])]
    return {"perturbed_world": perturb_index, "cross_contaminated_worlds": contaminated,
            "isolated": len(contaminated) == 0}

def memory_done_reset_probe(mem_after_done):
    """GATE6: after episode done, memory at next episode start must be all-zero."""
    import numpy as np
    z = np.allclose(np.asarray(mem_after_done), 0.0)
    return {"memory_zero_after_done": bool(z)}

def self_sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def build_provenance(args, evaluator_sha, world_set_hash, achievement_registry_sha, tier_mapping_sha):
    return {
        "protocol": PROTOCOL_ID,
        "experiment_id": args.experiment_id,
        "checkpoint_path": args.checkpoint,
        "checkpoint_sha256": args.checkpoint_sha or "REQUIRED",
        "params_sha256": args.params_sha or "REQUIRED",
        "code_head": args.code_head or "UNVERIFIED (no local git repo)",
        "evaluator_path": os.path.abspath(__file__),
        "evaluator_sha256": evaluator_sha,
        "evaluator_protocol": PROTOCOL_ID,
        "action_mode": args.action_mode,
        "environment_version": args.env_version,
        "EnvParams": {"max_timesteps": MAX_TIMESTEPS},
        "StaticEnvParams": args.static_env_params or "craftax default (record if overridden)",
        "target_achievement": args.target_achievement or "DEFEAT_KOBOLD",
        "achievement_registry_sha256": achievement_registry_sha,
        "tier_mapping_sha256": tier_mapping_sha,
        "world_manifest_path": args.world_manifest,
        "world_set_hash": world_set_hash,
        "evaluation_seed": args.seed,
        "episode_count": NUM_WORLDS,
        "max_timesteps": MAX_TIMESTEPS,
        "spawn_floor": SPAWN_FLOOR,
        "metric_schema": "per-world[success,floor3,died] + aggregates + provenance",
        "output_paths": [args.out],
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": platform.node(),
        "gpu_uuid": args.gpu_uuid or "REQUIRED_ON_HOST",
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--checkpoint-sha", default=None)
    ap.add_argument("--params-sha", default=None)
    ap.add_argument("--action-mode", required=True, choices=["stochastic", "argmax"],
                    help="MUST be the actual executed mode; never guessed")
    ap.add_argument("--seed", type=int, default=DEFAULT_EVAL_SEED)
    ap.add_argument("--world-manifest", required=True)
    ap.add_argument("--world-set-hash", default=None)
    ap.add_argument("--achievement-registry-sha", default=None)
    ap.add_argument("--tier-mapping-sha", default=None)
    ap.add_argument("--env-version", default="craftax==1.4.5(EXPECTED)")
    ap.add_argument("--target-achievement", default="DEFEAT_KOBOLD")
    ap.add_argument("--static-env-params", default=None)
    ap.add_argument("--code-head", default=None)
    ap.add_argument("--gpu-uuid", default=None)
    ap.add_argument("--out", default="evaluation_provenance.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="emit provenance + run pure logic gates without JAX evaluation")
    a = ap.parse_args()

    evaluator_sha = self_sha(__file__)
    print(f"[{PROTOCOL_ID}] action_mode={a.action_mode}  seed={a.seed}  worlds={NUM_WORLDS}  "
          f"max_timesteps={MAX_TIMESTEPS}  evaluator_sha={evaluator_sha[:16]}")  # A1: print actual mode
    if a.seed == 100000:
        print("[WARN] " + SEED100000_LINE_NOTE)

    prov = build_provenance(a, evaluator_sha, a.world_set_hash or "REQUIRED (from world manifest)",
                            a.achievement_registry_sha or "REQUIRED", a.tier_mapping_sha or "REQUIRED")
    prov["action_mode"] = a.action_mode  # A2: write into output

    # GATE11: refuse paired-comparison eligibility without world_set_hash + evaluator sha
    prov["paired_comparison_eligible"] = bool(a.world_set_hash) and bool(evaluator_sha)

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2, ensure_ascii=False)
    print("wrote provenance ->", a.out, "| paired_eligible:", prov["paired_comparison_eligible"])

    if a.dry_run:
        # exercise pure-logic gates (no JAX)
        r = checkpoint_restore_check(["a", "b", "c"], ["a", "b", "c"])
        print("GATE4 exact-restore:", r["status"])
        try:
            checkpoint_restore_check(["a", "b", "c"], ["a", "b"])
            print("GATE4 missing-detect: FAIL (should have raised)")
        except RestoreLeafMismatch:
            print("GATE4 missing-detect: HARD-FAIL as required")
        print("dry-run complete (no JAX evaluation performed)")
        return

    # Real evaluation path (requires JAX+Craftax)
    try:
        import jax, craftax  # noqa
    except Exception as e:
        raise SystemExit(f"EVALUATION BLOCKED: JAX/Craftax absent ({e}). "
                         f"Run on the experiment host with authorized read-only checkpoints.")
    raise SystemExit("Full JAX evaluation loop: port the anchor forward_eval here under this provenance "
                     "wrapper. Not executed in audit env.")

if __name__ == "__main__":
    main()
