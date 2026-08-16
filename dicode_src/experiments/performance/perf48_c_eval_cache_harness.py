#!/usr/bin/env python3
"""C: held-out evaluator compile-cache single-variable test (EVAL_CACHE_OFF <-> ON).

Runs the production held-out evaluator (craftax_evaluation.main) twice per arm on
a frozen mid checkpoint, with jax.clear_caches() between the two calls to
simulate the production session boundary (run_dicode.py clears caches each
session). Measures:

  - first/session wall clock (full main() call: lower+compile+execute)
  - second/session wall clock (after jax.clear_caches; cache-on should be a hit)
  - eval_compile / eval_execute tracker spans (cache_hit flag, durations)
  - held-out metrics (mean_return / mean_performance / average_episode_length)
    for off-vs-on semantic equivalence

Usage:
  python perf48_c_eval_cache_harness.py \
    --config <perf48_off.yaml> --checkpoint <2100 dir> --arm off|on \
    --out-dir <dir> [--num-envs 1024] [--eval-steps 8192]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--arm", required=True, choices=("off", "on"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--eval-steps", type=int, default=8192)
    parser.add_argument("--rng-seed", type=int, default=0)
    args = parser.parse_args(argv)

    import jax
    from omegaconf import OmegaConf
    from dicode.runtime_analysis import tracker
    from dicode.setup import _load_agent_state
    from dicode.task_utils import get_achievement_multi_hot
    from dicode.craftax_evaluation import main as heldout_main, clear_compiled_evaluator_cache
    from dicode.wrappers import BatchEnvWrapper
    from minicraftax.envs.craftax import CraftaxAugObsTrain

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # fresh run-scoped evaluator cache + fresh tracker events file
    clear_compiled_evaluator_cache()
    events_path = out / "events.jsonl"
    tracker.configure(enabled=True, output_jsonl=str(events_path), reset=True)
    tracker.set_session("heldout")

    cfg = OmegaConf.load(args.config)
    # force eval budget to frozen values
    cfg.evaluation.num_envs = args.num_envs
    cfg.evaluation.num_steps = args.eval_steps
    if args.arm == "on":
        cfg.performance.eval_compile_cache = True
    else:
        cfg.performance.eval_compile_cache = False

    print(f"[C] arm={args.arm} eval_compile_cache="
          f"{cfg.performance.eval_compile_cache} num_envs={args.num_envs} "
          f"eval_steps={args.eval_steps}")

    train_state = _load_agent_state(cfg, args.checkpoint)
    print("[C] checkpoint loaded")

    # held-out eval embedding (one-hot multi-hot of base Craftax env), tiled
    eval_env = CraftaxAugObsTrain()
    base_emb = get_achievement_multi_hot(eval_env.relevant_achievements)
    eval_embedding = np.tile(base_emb, (args.num_envs, 1)).astype(np.float32)
    print(f"[C] eval_embedding shape={eval_embedding.shape} dtype={eval_embedding.dtype}")

    rng = jax.random.PRNGKey(args.rng_seed)

    def run_session(label):
        t0 = time.monotonic()
        metrics = heldout_main(cfg, rng, train_state=train_state,
                               eval_embedding=eval_embedding, detail=False)
        dt = time.monotonic() - t0
        print(f"[C] {label}: wall_s={dt:.3f} "
              f"mean_return={float(np.asarray(metrics['mean_return'])):.4f} "
              f"mean_performance={float(np.asarray(metrics['mean_performance'])):.4f} "
              f"avg_len={float(np.asarray(metrics['average_episode_length'])):.4f}")
        return {"wall_s": dt,
                "mean_return": float(np.asarray(metrics["mean_return"])),
                "mean_performance": float(np.asarray(metrics["mean_performance"])),
                "average_episode_length": float(np.asarray(metrics["average_episode_length"])),
                "skill_keys": sorted(k for k in metrics if k.startswith("skill_"))}

    first = run_session("first_session(cold)")
    # production session boundary: clear JAX internal caches (run-scoped
    # compiled-evaluator cache survives; see R4 test_clear_caches_survival)
    jax.clear_caches()
    second = run_session("second_session(warm)")

    tracker.derive_reports()

    # parse events for eval_compile / eval_execute spans
    events = []
    if events_path.is_file():
        events = [json.loads(l) for l in events_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    compile_spans = [e for e in events if e.get("phase") == "eval_compile"]
    execute_spans = [e for e in events if e.get("phase") == "eval_execute"]

    summary = {
        "arm": args.arm,
        "eval_compile_cache": bool(cfg.performance.eval_compile_cache),
        "config_path": str(args.config),
        "checkpoint": str(args.checkpoint),
        "num_envs": args.num_envs,
        "eval_steps": args.eval_steps,
        "rng_seed": args.rng_seed,
        "first_session_wall_s": first["wall_s"],
        "second_session_wall_s": second["wall_s"],
        "first_metrics": {k: v for k, v in first.items() if k != "wall_s"},
        "second_metrics": {k: v for k, v in second.items() if k != "wall_s"},
        "first_second_metrics_equal": {k: first[k] == second[k]
                                       for k in ("mean_return", "mean_performance", "average_episode_length")},
        "eval_compile_spans": [{"duration_s": e["duration_s"], "cache_hit": e["cache_hit"]}
                               for e in compile_spans],
        "eval_execute_spans": [{"duration_s": e["duration_s"], "cache_hit": e["cache_hit"]}
                               for e in execute_spans],
        "cache_hit_occurred_on_second": any(e.get("cache_hit") for e in execute_spans),
        "second_session_recompiled": len(compile_spans) >= 2,
        "second_vs_first_wall_ratio": round(second["wall_s"] / first["wall_s"], 4) if first["wall_s"] > 0 else None,
    }
    (out / "RESULT.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
