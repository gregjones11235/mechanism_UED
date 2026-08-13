#!/usr/bin/env python3
"""BC combo pair runner: BC_OFF <-> BC_ON across early/mid/late x repeat0/1.

Reuses perf48_pair_benchmark's run_arm mechanism (per-arm caches, exact GPU
UUID, optional deterministic XLA flag, 2s GPU sampler, fail-closed fatal
detection) and extends it for the two combo arms.

Semantic gate (--deterministic-xla): BC_OFF/BC_ON must agree on every semantic
hash (params/optimizer/RNG before-after, scoring fingerprint, evaluation metrics
fingerprint, checkpoint reload hashes, archive before/after).

Mechanism evidence (recorded per arm, verified per pair):
  * B2: preflight_task_reload_occurred -- true in BC_OFF, explicit-absent in BC_ON
  * C : eval_compile_span_count -- 0 in BC_OFF, 1 in BC_ON;
        eval_execute cache_hit -- first miss, second hit in BC_ON.

Performance evidence is gathered in a SEPARATE default-XLA run; the two evidence
classes (deterministic semantics vs default speed) are never mixed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

# Reuse the proven pair-benchmark mechanisms (pure stdlib module, no jax import).
_PAIR = importlib.util.spec_from_file_location(
    "perf48_pair_benchmark", Path(__file__).with_name("perf48_pair_benchmark.py"))
_pair = importlib.util.module_from_spec(_PAIR)
assert _PAIR.loader
_PAIR.loader.exec_module(_pair)

# Pull the mechanisms we reuse.
append_xla_flag = _pair.append_xla_flag
gpu_snapshot = _pair.gpu_snapshot
gpu_apps = _pair.gpu_apps
assert_gpu_free = _pair.assert_gpu_free
monitor_popen = _pair.monitor_popen
arm_gpu_metrics = _pair.arm_gpu_metrics
fatal_in = _pair.fatal_in
atomic_json = _pair.atomic_json
stop_owned = _pair.stop_owned
FATAL_RE = _pair.FATAL_RE

_MSPEC = importlib.util.spec_from_file_location(
    "perf48_combo_manifest", Path(__file__).with_name("perf48_combo_manifest.py"))
_manifest = importlib.util.module_from_spec(_MSPEC)
assert _MSPEC.loader
_MSPEC.loader.exec_module(_manifest)

ARMS = ("BC_OFF", "BC_ON")
CLASSIFICATION = "PERF48_COMBO_BENCHMARK"

SEMANTIC_FIELDS = (
    "params_sha256_before", "params_sha256_after",
    "optimizer_sha256_before", "optimizer_sha256_after",
    "checkpoint_reloaded_params_sha256", "checkpoint_reloaded_optimizer_sha256",
    "input_rng_sha256", "rng_sha256_before", "heldout_rng_sha256", "preflight_rng_sha256",
    "task_ids", "task_assignment_sha256", "task_code_hashes",
    "embedding_hash", "conditioning_type", "conditioning_shape", "conditioning_dtype",
    "reset_selection_semantics", "global_update_step",
    "score_function", "wrappers_cl_sha256",
    "scoring_fingerprint", "evaluation_metrics_sha256",
    "archive_before_sha256", "archive_after_sha256",
)

PERF_FIELDS = (
    "session_wall_s", "preflight_wall_s", "eval_wall_s", "checkpoint_wall_s",
    "eval_compile_span_count", "eval_cache_hit_count", "eval_first_cache_miss",
    "preflight_task_reload_occurred", "preflight_task_reload_explicit_absent",
    "scoring_wall_s", "preflight_env_steps", "heldout_env_steps", "total_env_steps",
    "preflight_throughput_env_s", "eval_throughput_env_s",
)

MECHANISM_REQUIREMENTS = {
    "BC_OFF": {
        "preflight_task_reload_occurred": True,
        "preflight_task_reload_explicit_absent": False,
        "eval_compile_span_count": 0,
    },
    "BC_ON": {
        "preflight_task_reload_occurred": False,
        "preflight_task_reload_explicit_absent": True,
        "eval_compile_span_count": 1,
        "eval_first_cache_miss": True,
        "eval_cache_hit_count": 1,
    },
}


def _arm_env(args: Any, arm: str, source: str, out: Path) -> dict[str, str]:
    """Isolated harness env: per-arm caches, exact GPU UUID, optional det flag."""
    env = dict(os.environ)
    env.pop("JAX_PLATFORMS", None)
    env.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, WANDB_MODE="offline",
               PYTHONPATH=str(Path(source) / "src"))
    if getattr(args, "deterministic_xla", False):
        env["XLA_FLAGS"] = append_xla_flag(env.get("XLA_FLAGS"))
    for key, subdir in (("TMPDIR", "tmp"), ("TMP", "tmp"), ("TEMP", "tmp"),
                        ("XDG_CACHE_HOME", "cache"), ("WANDB_DIR", "wandb")):
        path = out / subdir
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    return env


def validate_combo_result(doc: Mapping[str, Any], *, manifest_sha256: str, stage: str,
                          repeat: int, arm: str, gpu_uuid: str | None = None,
                          source_commit: str | None = None) -> dict[str, Any]:
    if arm not in ARMS:
        raise RuntimeError("invalid arm")
    checks = {"classification": CLASSIFICATION, "manifest_sha256": manifest_sha256,
              "stage": stage, "repeat": repeat, "arm": arm, "llm_api_calls": 0}
    if source_commit is not None:
        checks["source_commit"] = source_commit
    for key, expected in checks.items():
        if doc.get(key) != expected:
            raise RuntimeError(f"invalid result {key}")
    if gpu_uuid is not None and doc.get("gpu_uuid") != gpu_uuid:
        raise RuntimeError("invalid result gpu_uuid")
    for field in SEMANTIC_FIELDS:
        if field not in doc:
            raise RuntimeError(f"missing semantic field {field}")
    for field in PERF_FIELDS:
        if field not in doc:
            raise RuntimeError(f"missing perf field {field}")
    if not doc.get("checkpoint_loadable"):
        raise RuntimeError("checkpoint_loadable must be true")
    if doc.get("compact_scoring_payload"):
        raise RuntimeError("compact_scoring_payload must be false")
    evidence = doc.get("runtime_source_evidence")
    if not isinstance(evidence, Mapping) or evidence.get("verified") is not True:
        raise RuntimeError("invalid runtime source evidence")
    profiling = doc.get("profiling", {})
    if not profiling.get("enabled") or not profiling.get("event_count") or not profiling.get("events_csv_sha256") or not profiling.get("critical_path_sha256"):
        raise RuntimeError("invalid profiling contract")
    env_ev = doc.get("env_evidence")
    if not isinstance(env_ev, Mapping) or not env_ev.get("jax_version"):
        raise RuntimeError("invalid env evidence")
    return dict(doc)


def compare_combo_pair(off: Mapping[str, Any], on: Mapping[str, Any]) -> str:
    """Semantic gate between BC_OFF and BC_ON (deterministic-XLA evidence)."""
    for field in SEMANTIC_FIELDS:
        if off.get(field) != on.get(field):
            return "REJECTED_SEMANTIC_MISMATCH"
    if (not off.get("checkpoint_loadable")) or (not on.get("checkpoint_loadable")):
        return "REJECTED_RUNTIME_FAILURE"
    if off.get("compact_scoring_payload") or on.get("compact_scoring_payload"):
        return "REJECTED_SEMANTIC_MISMATCH"
    if (off.get("gpu_uuid") != on.get("gpu_uuid")
            or off.get("classification") != on.get("classification")
            or off.get("llm_api_calls") != on.get("llm_api_calls")):
        return "REJECTED_SEMANTIC_MISMATCH"
    return "SEMANTIC_PASS"


def verify_mechanisms(off: Mapping[str, Any], on: Mapping[str, Any]) -> dict[str, Any]:
    """Prove B2 (reload absent in ON) and C (compile once, first miss second hit)."""
    checks: dict[str, Any] = {}
    ok = True
    for arm, reqs in MECHANISM_REQUIREMENTS.items():
        doc = off if arm == "BC_OFF" else on
        for key, expected in reqs.items():
            got = doc.get(key)
            checks[f"{arm}.{key}"] = {"expected": expected, "got": got, "ok": got == expected}
            if got != expected:
                ok = False
    # cross-arm: the ON arm must actually SKIP the second load that OFF performs
    checks["cross_arm_reload_flipped"] = {
        "expected": True, "got": (on.get("preflight_task_reload_occurred") is False
                                  and off.get("preflight_task_reload_occurred") is True),
        "ok": on.get("preflight_task_reload_occurred") is False and off.get("preflight_task_reload_occurred") is True,
    }
    if not checks["cross_arm_reload_flipped"]["ok"]:
        ok = False
    return {"ok": ok, "checks": checks}


def run_combo_arm(args: Any, stage: str, repeat: int, arm: str, out: str | Path) -> dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    result = out / "RESULT.json"
    manifest_sha = args.manifest_sha256
    if result.exists():
        return validate_combo_result(json.loads(result.read_text()), manifest_sha256=manifest_sha,
                                     stage=stage, repeat=repeat, arm=arm, gpu_uuid=args.gpu_uuid,
                                     source_commit=args.source_commit)
    config = args.config_on if arm == "BC_ON" else args.config_off
    env = _arm_env(args, arm, args.source, out)
    cmd = [args.python, args.harness, "--manifest", args.manifest, "--config", config,
           "--out", str(out), "--required-gpu-uuid", args.gpu_uuid, "--source-commit",
           args.source_commit, "--stage", stage, "--repeat", str(repeat), "--arm", arm,
           "--mode", "run"]
    assert_gpu_free(args.gpu_index, args.gpu_uuid)
    so = (out / "trainer.stdout").open("w")
    se = (out / "trainer.stderr").open("w")
    proc: subprocess.Popen[Any] | None = None
    try:
        proc = subprocess.Popen(cmd, env=env, text=True, stdout=so, stderr=se, start_new_session=True)
        stop, thread, violations = monitor_popen(proc, args.gpu_index, args.gpu_uuid, out)
        try:
            while proc.poll() is None:
                fatal = fatal_in([out / "trainer.stdout", out / "trainer.stderr"])
                if violations or fatal:
                    _fail(out, "GPU safety violation" if violations else "fatal trainer output",
                          proc, violations=violations, fatal=fatal)
                    raise RuntimeError("trainer stopped")
                time.sleep(.1)
            if proc.returncode:
                _fail(out, f"{arm} failed rc={proc.returncode}", proc)
                raise RuntimeError(f"{arm} failed rc={proc.returncode}")
        finally:
            stop.set()
            thread.join(timeout=5)
    except Exception:
        if not (out / "failure.json").exists():
            _fail(out, "trainer launch failure", proc)
        raise
    finally:
        so.close()
        se.close()
    if not result.exists():
        _fail(out, "missing RESULT.json")
        raise RuntimeError("missing RESULT.json")
    doc = validate_combo_result(json.loads(result.read_text()), manifest_sha256=manifest_sha,
                                stage=stage, repeat=repeat, arm=arm, gpu_uuid=args.gpu_uuid,
                                source_commit=args.source_commit)
    peak, minimum = arm_gpu_metrics(out / "gpu_memory.csv")
    doc.update(gpu_peak_memory_mib=peak, gpu_min_free_mib=minimum, gpu_uuid=args.gpu_uuid)
    atomic_json(result, doc)
    return doc


def _fail(out: Path, message: str, proc: subprocess.Popen[Any] | None = None, **extra: Any) -> None:
    if proc is not None:
        stop_owned(proc.pid)
    atomic_json(out / "failure.json", {"error": message, **extra})


def _aggregate(pairs: list[dict[str, Any]], *, det: bool, required_pairs: int = 6) -> dict[str, Any]:
    import statistics

    if len(pairs) < 1 or len(pairs) > required_pairs:
        return {"conclusion": "REJECTED_RUNTIME_FAILURE", "pair_count": len(pairs)}

    def arm(p, name):
        return p.get(name, {})

    d0 = [float(arm(p, "off").get("session_wall_s", 0)) for p in pairs]
    d3 = [float(arm(p, "on").get("session_wall_s", 0)) for p in pairs]
    pw0 = [float(arm(p, "off").get("preflight_wall_s", 0)) for p in pairs]
    pw3 = [float(arm(p, "on").get("preflight_wall_s", 0)) for p in pairs]
    ew0 = [float(arm(p, "off").get("eval_wall_s", 0)) for p in pairs]
    ew3 = [float(arm(p, "on").get("eval_wall_s", 0)) for p in pairs]
    t0 = [float(arm(p, "off").get("preflight_throughput_env_s", 0)) for p in pairs]
    t3 = [float(arm(p, "on").get("preflight_throughput_env_s", 0)) for p in pairs]
    et0 = [float(arm(p, "off").get("eval_throughput_env_s", 0)) for p in pairs]
    et3 = [float(arm(p, "on").get("eval_throughput_env_s", 0)) for p in pairs]
    mean_d0, mean_d3 = statistics.mean(d0), statistics.mean(d3)
    duration_imp = (mean_d3 - mean_d0) / mean_d0 if mean_d0 else 0.0
    preflight_imp = (statistics.mean(pw3) - statistics.mean(pw0)) / statistics.mean(pw0) if statistics.mean(pw0) else 0.0
    eval_imp = (statistics.mean(ew3) - statistics.mean(ew0)) / statistics.mean(ew0) if statistics.mean(ew0) else 0.0
    preflight_throughput_imp = (statistics.mean(t3) - statistics.mean(t0)) / statistics.mean(t0) if statistics.mean(t0) else 0.0
    eval_throughput_imp = (statistics.mean(et3) - statistics.mean(et0)) / statistics.mean(et0) if statistics.mean(et0) else 0.0
    regressions = [i for i, (a, b) in enumerate(zip(d0, d3)) if b > a * 1.01]
    peak_deltas = [float(arm(p, "on").get("gpu_peak_memory_mib", 0)) - float(arm(p, "off").get("gpu_peak_memory_mib", 0)) for p in pairs]
    min_free = min(float(arm(p, key).get("gpu_min_free_mib", 0)) for p in pairs for key in ("off", "on"))
    peak_bad = max(peak_deltas, default=0) > 512 or min_free < 4096
    runtime_markers = ("runtime_failure", "fatal_error", "oom", "xid", "checkpoint_error", "gpu_violation")
    runtime_bad = peak_bad or any(any(bool(arm(p, key).get(k)) for k in runtime_markers)
                                  or arm(p, key).get("checkpoint_loadable") is False
                                  for p in pairs for key in ("off", "on"))
    conclusion = "REJECTED_RUNTIME_FAILURE" if runtime_bad else "COMBO_PASS"
    return {
        "conclusion": conclusion,
        "det": det,
        "mean_session_off": mean_d0, "mean_session_on": mean_d3,
        "session_improvement": duration_imp,
        "preflight_improvement": preflight_imp,
        "eval_improvement": eval_imp,
        "preflight_throughput_improvement": preflight_throughput_imp,
        "eval_throughput_improvement": eval_throughput_imp,
        "per_pair_regressions": regressions,
        "max_peak_delta_mib": max(peak_deltas, default=0),
        "min_free_mib": min_free,
        "pair_count": len(pairs),
        "required_pairs": required_pairs,
    }


def _parse_only_pairs(spec: str | None) -> set[tuple[str, int]] | None:
    """Parse a 'stage:repeat[,stage:repeat,...]' filter, or None for all pairs."""
    if spec is None or not spec.strip():
        return None
    out: set[tuple[str, int]] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        stage, _, repeat = token.partition(":")
        if stage not in ("early", "mid", "late") or repeat not in ("0", "1"):
            raise ValueError(f"invalid --only-pairs token {token!r} (want stage:0|1)")
        out.add((stage, int(repeat)))
    return out or None


def main() -> None:
    p = argparse.ArgumentParser()
    for name in ("root", "manifest", "harness", "python", "source", "config_off", "config_on", "source_commit", "gpu_index", "gpu_uuid"):
        p.add_argument("--" + name.replace("_", "-"), required=True)
    p.add_argument("--only-pairs", default=None,
                   help="comma-separated stage:repeat subset, e.g. 'early:0' (default: all six groups)")
    p.add_argument("--deterministic-xla", action="store_true",
                   help="append --xla_gpu_deterministic_ops=true to every harness env (semantic gate only)")
    args = p.parse_args()
    args.gpu_index = int(args.gpu_index)
    only_pairs = _parse_only_pairs(args.only_pairs)
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = _manifest.load_manifest(args.manifest)
    args.manifest_sha256 = manifest["manifest_sha256"]
    pairs: list[dict[str, Any]] = []
    failed_pair: str | None = None
    try:
        for stage in ("early", "mid", "late"):
            for repeat in (0, 1):
                if only_pairs is not None and (stage, repeat) not in only_pairs:
                    continue
                pair_dir = root / "pairs" / f"{stage}_repeat_{repeat}"
                failed_pair = str(pair_dir)
                order = next(item for item in manifest["stages"] if item["name"] == stage)["repeats"][repeat]["order"]
                rows = {}
                for arm in order:
                    rows[arm] = run_combo_arm(args, stage, repeat, arm, pair_dir / arm.lower())
                off, on = rows["BC_OFF"], rows["BC_ON"]
                status = compare_combo_pair(off, on)
                mech = verify_mechanisms(off, on)
                if status != "SEMANTIC_PASS":
                    conclusion = "REJECTED_SEMANTIC_MISMATCH" if status == "REJECTED_SEMANTIC_MISMATCH" else "REJECTED_RUNTIME_FAILURE"
                    atomic_json(root / "COMBINATION_PAIR_RESULT.json",
                                {"conclusion": conclusion, "completed_pair_count": len(pairs),
                                 "evidence": str(pair_dir), "failed_pair": str(pair_dir),
                                 "pairs": pairs, "manifest_sha256": args.manifest_sha256,
                                 "gpu_uuid": args.gpu_uuid})
                    raise RuntimeError(status)
                if not mech["ok"]:
                    atomic_json(root / "COMBINATION_PAIR_RESULT.json",
                                {"conclusion": "REJECTED_MECHANISM", "completed_pair_count": len(pairs),
                                 "evidence": str(pair_dir), "failed_pair": str(pair_dir),
                                 "mechanism": mech, "pairs": pairs,
                                 "manifest_sha256": args.manifest_sha256, "gpu_uuid": args.gpu_uuid})
                    raise RuntimeError("mechanism evidence failed")
                atomic_json(pair_dir / "PAIR.json",
                            {"status": status, "mechanism": mech, "off": off, "on": on})
                pairs.append({"off": off, "on": on, "status": status, "mechanism": mech})
    except Exception as exc:
        existing = root / "COMBINATION_PAIR_RESULT.json"
        if not existing.exists() or json.loads(existing.read_text()).get("conclusion") not in {
                "REJECTED_SEMANTIC_MISMATCH", "REJECTED_RUNTIME_FAILURE", "REJECTED_MECHANISM"}:
            atomic_json(existing, {"conclusion": "REJECTED_RUNTIME_FAILURE",
                                   "completed_pair_count": len(pairs), "evidence": failed_pair,
                                   "failed_pair": failed_pair, "error": str(exc), "pairs": pairs,
                                   "manifest_sha256": args.manifest_sha256, "gpu_uuid": args.gpu_uuid})
        raise
    required_pairs = 6 if only_pairs is None else len(only_pairs)
    agg = _aggregate(pairs, det=args.deterministic_xla, required_pairs=required_pairs)
    final = {**agg, "pairs": pairs, "pair_count": len(pairs), "manifest_sha256": args.manifest_sha256,
             "gpu_uuid": args.gpu_uuid, "deterministic_xla": args.deterministic_xla,
             "only_pairs": args.only_pairs}
    atomic_json(root / "COMBINATION_PAIR_RESULT.json", final)
    if agg.get("conclusion") == "REJECTED_RUNTIME_FAILURE":
        raise RuntimeError("runtime gate failed")


if __name__ == "__main__":
    main()
