#!/usr/bin/env python3
"""Pair runner for B4_SINGLE and FINAL_COMBO frozen replays."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping


def _load_sibling(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


_pair = _load_sibling("perf48_pair_benchmark_fastpath", "perf48_pair_benchmark.py")
_manifest = _load_sibling("perf48_combo_manifest_fastpath", "perf48_combo_manifest.py")
_cfg = _load_sibling("perf48_fastpath_config", "perf48_fastpath_config.py")

append_xla_flag = _pair.append_xla_flag
assert_gpu_free = _pair.assert_gpu_free
monitor_popen = _pair.monitor_popen
arm_gpu_metrics = _pair.arm_gpu_metrics
fatal_in = _pair.fatal_in
atomic_json = _pair.atomic_json
stop_owned = _pair.stop_owned

CLASSIFICATION = "PERF48_FASTPATH_BENCHMARK"
COMPARISONS = _cfg.COMPARISONS
RUNTIME_MARKERS = (
    "runtime_failure", "fatal_error", "oom", "xid", "checkpoint_error", "gpu_violation"
)
SEMANTIC_FIELDS = (
    "params_sha256_before", "params_sha256_after",
    "optimizer_sha256_before", "optimizer_sha256_after",
    "checkpoint_reloaded_params_sha256", "checkpoint_reloaded_optimizer_sha256",
    "input_rng_sha256", "rng_sha256_before", "preflight_rng_sha256", "heldout_rng_sha256",
    "task_ids", "task_assignment_sha256", "task_code_hashes",
    "embedding_hash", "conditioning_type", "conditioning_shape", "conditioning_dtype",
    "reset_selection_semantics", "global_update_step", "score_function",
    "score_projection", "scoring_fingerprint", "accepted_ids", "rejected_ids",
    "archive_before_sha256", "archive_after_sha256", "evaluation_metrics_sha256",
    "evaluation_metrics_second_sha256", "runtime_source_evidence", "env_evidence",
    "preflight_env_steps", "heldout_env_steps", "total_env_steps",
)
PERF_FIELDS = (
    "session_wall_s", "preflight_wall_s", "preflight_build_s",
    "preflight_lower_compile_s", "preflight_execute_s", "preflight_transfer_s",
    "preflight_scoring_cpu_s", "scoring_wall_s", "route_wall_s", "eval_wall_s",
    "checkpoint_wall_s", "preflight_env_steps", "heldout_env_steps", "total_env_steps",
    "preflight_throughput_env_s", "eval_throughput_env_s", "session_throughput_env_s",
    "gpu_peak_memory_mib", "gpu_min_free_mib",
)
THRESHOLDS = {
    "B4_SINGLE": {"preflight_speedup": 0.10, "session_speedup": None,
                  "max_pair_session_regression": 0.01},
    "FINAL_COMBO": {"preflight_speedup": 0.20, "session_speedup": 0.10,
                    "max_pair_session_regression": None},
}


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_config_pair(args: Any, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Exact overlay diff gate plus manifest path/hash binding."""
    left = _cfg.load_overlay(args.config_left)
    right = _cfg.load_overlay(args.config_right)
    gate = _cfg.verify_overlay_pair(left, right, comparison=args.comparison)
    config_entries = manifest.get("source_config", {}).get("config", {})
    for arm, path in zip(COMPARISONS[args.comparison], (args.config_left, args.config_right)):
        resolved = Path(path).resolve()
        entry = next(
            (
                value for value in config_entries.values()
                if Path(value.get("path", "")).resolve() == resolved
            ),
            None,
        )
        if entry is None:
            raise RuntimeError(f"{arm} config is not bound by the frozen manifest")
        if _sha256_file(resolved) != entry.get("sha256"):
            raise RuntimeError(f"{arm} config hash mismatch")
    return gate


def _arm_env(args: Any, source: str, out: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("JAX_PLATFORMS", None)
    env.update(
        CUDA_VISIBLE_DEVICES=args.gpu_uuid,
        WANDB_MODE="offline",
        PYTHONPATH=str(Path(source) / "src"),
    )
    if getattr(args, "deterministic_xla", False):
        env["XLA_FLAGS"] = append_xla_flag(env.get("XLA_FLAGS"))
    for key, subdir in (
        ("TMPDIR", "tmp"), ("TMP", "tmp"), ("TEMP", "tmp"),
        ("XDG_CACHE_HOME", "cache"), ("WANDB_DIR", "wandb"),
    ):
        path = out / subdir
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    return env


def validate_fastpath_result(
    document: Mapping[str, Any], *, manifest_sha256: str, comparison: str,
    stage: str, repeat: int, arm: str, gpu_uuid: str | None = None,
    source_commit: str | None = None, require_gpu_metrics: bool = False,
) -> dict[str, Any]:
    if comparison not in COMPARISONS or arm not in COMPARISONS[comparison]:
        raise RuntimeError("invalid result comparison/arm")
    expected = {
        "classification": CLASSIFICATION,
        "comparison": comparison,
        "manifest_sha256": manifest_sha256,
        "stage": stage,
        "repeat": repeat,
        "arm": arm,
        "llm_api_calls": 0,
        "validation_cache_exercised": False,
        "validation_cache_speedup_claimed": False,
    }
    if source_commit is not None:
        expected["source_commit"] = source_commit
    if gpu_uuid is not None:
        expected["gpu_uuid"] = gpu_uuid
    for key, value in expected.items():
        if document.get(key) != value:
            raise RuntimeError(f"invalid result {key}: expected {value!r}")
    for field in SEMANTIC_FIELDS + PERF_FIELDS:
        if field not in document:
            raise RuntimeError(f"missing result field {field}")
    if require_gpu_metrics and any(document.get(key) is None for key in (
        "gpu_peak_memory_mib", "gpu_min_free_mib"
    )):
        raise RuntimeError("missing sampled GPU memory evidence")
    if not document.get("checkpoint_loadable") or document.get("compact_scoring_payload"):
        raise RuntimeError("invalid checkpoint/compact payload result")
    if document.get("validation_cache_enabled") is not _cfg.EXPECTED_FLAGS[arm]["validation_cache"]:
        raise RuntimeError("validation cache flag evidence mismatch")
    expected_mode = "fused" if _cfg.EXPECTED_FLAGS[arm]["learnability_fused_preflight_summary"] else "legacy"
    if document.get("preflight_summary_mode") != expected_mode:
        raise RuntimeError("preflight summary mode mismatch")
    if any(bool(document.get(marker)) for marker in RUNTIME_MARKERS):
        raise RuntimeError("runtime marker present")
    source = document.get("runtime_source_evidence")
    if not isinstance(source, Mapping) or source.get("verified") is not True:
        raise RuntimeError("invalid runtime source evidence")
    profiling = document.get("profiling")
    if not isinstance(profiling, Mapping) or not profiling.get("enabled"):
        raise RuntimeError("profiling disabled")
    if not profiling.get("event_count") or not profiling.get("events_csv_sha256"):
        raise RuntimeError("profiling event evidence missing")
    if not profiling.get("critical_path_sha256"):
        raise RuntimeError("critical-path evidence missing")
    environment = document.get("env_evidence")
    if not isinstance(environment, Mapping) or not environment.get("jax_version"):
        raise RuntimeError("environment evidence missing")
    return dict(document)


def compare_semantics(left: Mapping[str, Any], right: Mapping[str, Any]) -> str:
    if any(left.get(field) != right.get(field) for field in SEMANTIC_FIELDS):
        return "REJECTED_SEMANTIC_MISMATCH"
    if not left.get("checkpoint_loadable") or not right.get("checkpoint_loadable"):
        return "REJECTED_RUNTIME_FAILURE"
    if left.get("gpu_uuid") != right.get("gpu_uuid"):
        return "REJECTED_SEMANTIC_MISMATCH"
    if any(bool(document.get(marker)) for document in (left, right) for marker in RUNTIME_MARKERS):
        return "REJECTED_RUNTIME_FAILURE"
    return "SEMANTIC_PASS"


def verify_mechanisms(
    left: Mapping[str, Any], right: Mapping[str, Any], *, comparison: str
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    ok = True
    for arm, document in zip(COMPARISONS[comparison], (left, right)):
        flags = _cfg.EXPECTED_FLAGS[arm]
        expected = {
            "preflight_summary_mode": "fused" if flags["learnability_fused_preflight_summary"] else "legacy",
            "preflight_task_reload_occurred": not flags["preflight_reuse_loaded_tasks"],
            "preflight_task_reload_explicit_absent": flags["preflight_reuse_loaded_tasks"],
            "eval_compile_span_count": 1 if flags["eval_compile_cache"] else 0,
            "eval_cache_hit_count": 1 if flags["eval_compile_cache"] else 0,
            "eval_first_cache_miss": flags["eval_compile_cache"],
            "validation_cache_enabled": flags["validation_cache"],
            "validation_cache_exercised": False,
            "validation_cache_speedup_claimed": False,
        }
        for key, wanted in expected.items():
            got = document.get(key)
            passed = got == wanted
            checks[f"{arm}.{key}"] = {"expected": wanted, "got": got, "ok": passed}
            ok = ok and passed
    return {"ok": ok, "checks": checks}


def _fail(out: Path, message: str, process: subprocess.Popen[Any] | None = None, **extra: Any) -> None:
    if process is not None:
        stop_owned(process.pid)
    atomic_json(out / "failure.json", {"error": message, **extra})


def run_fastpath_arm(
    args: Any, stage: str, repeat: int, arm: str, out: str | Path
) -> dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "RESULT.json"
    if result_path.exists():
        return validate_fastpath_result(
            json.loads(result_path.read_text(encoding="utf-8")),
            manifest_sha256=args.manifest_sha256,
            comparison=args.comparison,
            stage=stage,
            repeat=repeat,
            arm=arm,
            gpu_uuid=args.gpu_uuid,
            source_commit=args.source_commit,
            require_gpu_metrics=True,
        )
    left_arm, right_arm = COMPARISONS[args.comparison]
    config = args.config_left if arm == left_arm else args.config_right
    env = _arm_env(args, args.source, out)
    command = [
        args.python, args.harness,
        "--manifest", args.manifest,
        "--config", config,
        "--out", str(out),
        "--required-gpu-uuid", args.gpu_uuid,
        "--source-commit", args.source_commit,
        "--comparison", args.comparison,
        "--stage", stage,
        "--repeat", str(repeat),
        "--arm", arm,
        "--mode", "run",
    ]
    if getattr(args, "perf_mode", False):
        command.append("--perf")
    assert_gpu_free(args.gpu_index, args.gpu_uuid)
    stdout = (out / "trainer.stdout").open("w", encoding="utf-8")
    stderr = (out / "trainer.stderr").open("w", encoding="utf-8")
    process: subprocess.Popen[Any] | None = None
    try:
        process = subprocess.Popen(
            command, env=env, text=True, stdout=stdout, stderr=stderr, start_new_session=True
        )
        stop, monitor, violations = monitor_popen(
            process, args.gpu_index, args.gpu_uuid, out
        )
        try:
            while process.poll() is None:
                fatal = fatal_in([out / "trainer.stdout", out / "trainer.stderr"])
                if violations or fatal:
                    _fail(
                        out,
                        "GPU safety violation" if violations else "fatal trainer output",
                        process,
                        violations=violations,
                        fatal=fatal,
                    )
                    raise RuntimeError("trainer stopped")
                time.sleep(0.1)
            if process.returncode:
                _fail(out, f"{arm} failed rc={process.returncode}", process)
                raise RuntimeError(f"{arm} failed rc={process.returncode}")
        finally:
            stop.set()
            monitor.join(timeout=5)
    except Exception:
        if not (out / "failure.json").exists():
            _fail(out, "trainer launch failure", process)
        raise
    finally:
        stdout.close()
        stderr.close()
    if not result_path.exists():
        _fail(out, "missing RESULT.json")
        raise RuntimeError("missing RESULT.json")
    document = validate_fastpath_result(
        json.loads(result_path.read_text(encoding="utf-8")),
        manifest_sha256=args.manifest_sha256,
        comparison=args.comparison,
        stage=stage,
        repeat=repeat,
        arm=arm,
        gpu_uuid=args.gpu_uuid,
        source_commit=args.source_commit,
    )
    peak, minimum = arm_gpu_metrics(out / "gpu_memory.csv")
    document.update(gpu_peak_memory_mib=peak, gpu_min_free_mib=minimum)
    atomic_json(result_path, document)
    return validate_fastpath_result(
        document,
        manifest_sha256=args.manifest_sha256,
        comparison=args.comparison,
        stage=stage,
        repeat=repeat,
        arm=arm,
        gpu_uuid=args.gpu_uuid,
        source_commit=args.source_commit,
        require_gpu_metrics=True,
    )


def _speedup(before: list[float], after: list[float]) -> float:
    base = statistics.mean(before) if before else 0.0
    return (base - statistics.mean(after)) / base if base else 0.0


def aggregate_pairs(
    pairs: list[Mapping[str, Any]], *, comparison: str, required_pairs: int
) -> dict[str, Any]:
    if len(pairs) != required_pairs or not pairs:
        return {
            "conclusion": "REJECTED_RUNTIME_FAILURE",
            "comparison": comparison,
            "pair_count": len(pairs),
            "required_pairs": required_pairs,
        }
    left_wall = [float(pair["left"]["session_wall_s"]) for pair in pairs]
    right_wall = [float(pair["right"]["session_wall_s"]) for pair in pairs]
    left_preflight = [float(pair["left"]["preflight_wall_s"]) for pair in pairs]
    right_preflight = [float(pair["right"]["preflight_wall_s"]) for pair in pairs]
    session_speedup = _speedup(left_wall, right_wall)
    preflight_speedup = _speedup(left_preflight, right_preflight)
    regressions = [
        index for index, (before, after) in enumerate(zip(left_wall, right_wall))
        if after > before * 1.01
    ]
    peak_deltas = [
        float(pair["right"].get("gpu_peak_memory_mib", 0))
        - float(pair["left"].get("gpu_peak_memory_mib", 0))
        for pair in pairs
    ]
    minimum_free = min(
        float(pair[side].get("gpu_min_free_mib", 0))
        for pair in pairs for side in ("left", "right")
    )
    runtime_bad = (
        max(peak_deltas, default=0.0) > 512
        or minimum_free < 4096
        or any(
            bool(pair[side].get(marker))
            for pair in pairs for side in ("left", "right") for marker in RUNTIME_MARKERS
        )
        or any(
            pair[side].get("checkpoint_loadable") is not True
            for pair in pairs for side in ("left", "right")
        )
        or any(
            pair[side].get("validation_cache_exercised") is not False
            for pair in pairs for side in ("left", "right")
        )
    )
    thresholds = THRESHOLDS[comparison]
    speed_ok = preflight_speedup >= thresholds["preflight_speedup"]
    if thresholds["session_speedup"] is not None:
        speed_ok = speed_ok and session_speedup >= thresholds["session_speedup"]
    if thresholds["max_pair_session_regression"] is not None:
        speed_ok = speed_ok and not regressions
    if runtime_bad:
        conclusion = "REJECTED_RUNTIME_FAILURE"
    elif speed_ok:
        conclusion = "B4_PASS" if comparison == "B4_SINGLE" else "FINAL_COMBO_PASS"
    else:
        conclusion = "NO_SPEEDUP"
    return {
        "conclusion": conclusion,
        "comparison": comparison,
        "pair_count": len(pairs),
        "required_pairs": required_pairs,
        "preflight_speedup": preflight_speedup,
        "session_speedup": session_speedup,
        "preflight_speedup_threshold": thresholds["preflight_speedup"],
        "session_speedup_threshold": thresholds["session_speedup"],
        "per_pair_session_regressions": regressions,
        "max_peak_delta_mib": max(peak_deltas, default=0.0),
        "min_free_mib": minimum_free,
        "validation_cache_exercised": False,
        "validation_cache_speedup_claimed": False,
    }


def should_stop_after_early(pair: Mapping[str, Any], *, comparison: str) -> bool:
    if pair.get("stage") != "early" or pair.get("repeat") != 0:
        return False
    left, right = pair["left"], pair["right"]
    preflight_speedup = _speedup(
        [float(left["preflight_wall_s"])], [float(right["preflight_wall_s"])]
    )
    session_regression = float(right["session_wall_s"]) > float(left["session_wall_s"]) * 1.01
    return preflight_speedup <= 0.0 or session_regression


def parse_pair_filter(
    *, stage: str | None = None, repeat: int | None = None, only_pairs: str | None = None
) -> list[tuple[str, int]]:
    selected = [(name, rep) for name in ("early", "mid", "late") for rep in (0, 1)]
    if stage is not None:
        selected = [item for item in selected if item[0] == stage]
    if repeat is not None:
        selected = [item for item in selected if item[1] == repeat]
    if only_pairs:
        explicit: set[tuple[str, int]] = set()
        for token in only_pairs.split(","):
            token = token.strip()
            if not token:
                continue
            name, separator, value = token.partition(":")
            if separator != ":" or name not in ("early", "mid", "late") or value not in ("0", "1"):
                raise ValueError(f"invalid pair filter token {token!r}")
            explicit.add((name, int(value)))
        selected = [item for item in selected if item in explicit]
    if not selected:
        raise ValueError("pair filter selects no stage/repeat")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "root", "manifest", "harness", "python", "source", "config_left",
        "config_right", "source_commit", "gpu_index", "gpu_uuid",
    ):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    parser.add_argument("--comparison", choices=tuple(COMPARISONS), required=True)
    parser.add_argument("--stage", choices=("early", "mid", "late"))
    parser.add_argument("--repeat", type=int, choices=(0, 1))
    parser.add_argument("--only-pairs")
    parser.add_argument("--deterministic-xla", action="store_true")
    parser.add_argument("--perf-mode", action="store_true")
    parser.add_argument("--early-stoploss", action="store_true")
    args = parser.parse_args()
    args.gpu_index = int(args.gpu_index)
    selected = parse_pair_filter(
        stage=args.stage, repeat=args.repeat, only_pairs=args.only_pairs
    )
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = _manifest.load_manifest(args.manifest)
    args.manifest_sha256 = manifest["manifest_sha256"]
    overlay_gate = verify_config_pair(args, manifest)
    left_arm, right_arm = COMPARISONS[args.comparison]
    pairs: list[dict[str, Any]] = []
    result_path = root / "FASTPATH_BENCHMARK_RESULT.json"
    failed_pair: str | None = None
    try:
        for stage, repeat in selected:
            pair_dir = root / "pairs" / f"{stage}_repeat_{repeat}"
            failed_pair = str(pair_dir)
            order = (left_arm, right_arm) if repeat == 0 else (right_arm, left_arm)
            rows = {
                arm: run_fastpath_arm(args, stage, repeat, arm, pair_dir / arm.lower())
                for arm in order
            }
            left, right = rows[left_arm], rows[right_arm]
            semantic = compare_semantics(left, right)
            if semantic != "SEMANTIC_PASS":
                conclusion = semantic
                atomic_json(result_path, {
                    "conclusion": conclusion,
                    "comparison": args.comparison,
                    "completed_pair_count": len(pairs),
                    "failed_pair": failed_pair,
                    "pairs": pairs,
                    "manifest_sha256": args.manifest_sha256,
                    "validation_cache_exercised": False,
                    "overlay_gate": overlay_gate,
                })
                raise RuntimeError(semantic)
            mechanism = verify_mechanisms(left, right, comparison=args.comparison)
            if not mechanism["ok"]:
                atomic_json(result_path, {
                    "conclusion": "REJECTED_MECHANISM",
                    "comparison": args.comparison,
                    "completed_pair_count": len(pairs),
                    "failed_pair": failed_pair,
                    "mechanism": mechanism,
                    "pairs": pairs,
                    "manifest_sha256": args.manifest_sha256,
                    "validation_cache_exercised": False,
                    "overlay_gate": overlay_gate,
                })
                raise RuntimeError("mechanism evidence failed")
            pair = {
                "stage": stage,
                "repeat": repeat,
                "status": semantic,
                "mechanism": mechanism,
                "left": left,
                "right": right,
            }
            atomic_json(pair_dir / "PAIR.json", pair)
            pairs.append(pair)
            if args.early_stoploss and should_stop_after_early(pair, comparison=args.comparison):
                final = {
                    "conclusion": "NO_SPEEDUP",
                    "reason": "EARLY_REPEAT0_STOPLOSS",
                    "comparison": args.comparison,
                    "pair_count": len(pairs),
                    "required_pairs": len(selected),
                    "pairs": pairs,
                    "manifest_sha256": args.manifest_sha256,
                    "gpu_uuid": args.gpu_uuid,
                    "validation_cache_exercised": False,
                    "validation_cache_speedup_claimed": False,
                    "overlay_gate": overlay_gate,
                }
                atomic_json(result_path, final)
                return
    except Exception as exc:
        if not result_path.exists():
            atomic_json(result_path, {
                "conclusion": "REJECTED_RUNTIME_FAILURE",
                "comparison": args.comparison,
                "completed_pair_count": len(pairs),
                "failed_pair": failed_pair,
                "error": str(exc),
                "pairs": pairs,
                "manifest_sha256": args.manifest_sha256,
                "validation_cache_exercised": False,
                "overlay_gate": overlay_gate,
            })
        raise
    aggregate = aggregate_pairs(
        pairs, comparison=args.comparison, required_pairs=len(selected)
    )
    final = {
        **aggregate,
        "pairs": pairs,
        "manifest_sha256": args.manifest_sha256,
        "gpu_uuid": args.gpu_uuid,
        "source_commit": args.source_commit,
        "deterministic_xla": args.deterministic_xla,
        "perf_mode": args.perf_mode,
        "selected_pairs": selected,
        "overlay_gate": overlay_gate,
    }
    atomic_json(result_path, final)
    if aggregate["conclusion"].startswith("REJECTED_"):
        raise RuntimeError(aggregate["conclusion"])


if __name__ == "__main__":
    main()
